"""Offline AMS availability evidence audit. No network, features or model fits."""
import argparse
import hashlib
import re
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.sources.ams_publications import INDEX, parse_listing
from cottonlens_ml.sprint import freeze_record, read_record

PUBLISHED = 'https://marsapi.ams.usda.gov/services/v3.1/public/listPublishedReports/all'
CORRECTED = 'https://marsapi.ams.usda.gov/services/v1.1/public/listCorrectedReports/all'
LATEST_DOC = 'https://www.ams.usda.gov/mnreports/mp_cn001.txt'
LISTING = INDEX + '?' + urlencode({
    'field_slug_id_value': '3004', 'name': '', 'field_slug_title_value': '',
    'field_published_date_value': '', 'field_report_date_end_value': '',
    'field_api_market_types_target_id': 'All', 'order': '', 'sort': '',
})


def checked_request(root, url):
    key = hashlib.sha256(url.encode()).hexdigest()
    receipt = read_record(root / 'requests' / (key + '.json'))
    checksum = receipt['sha256']
    if receipt['url'] != url or not re.fullmatch('[0-9a-f]{64}', checksum):
        raise ValueError('Evidence request identity mismatch')
    path = root / 'raw' / 'ams' / checksum / 'source.bin'
    if digest(path) != checksum:
        raise ValueError('Evidence bytes corrupted')
    return path


def publication_control(text):
    """The official epoch is authoritative; no historical timezone is inferred."""
    if 'Published Date(msec)' not in text:
        raise ValueError('Changed public index header')
    rows = [line for line in text.splitlines() if re.match(r'\s*3004\s+', line)]
    if len(rows) != 1:
        raise ValueError('Exactly one legacy Cotton index control required')
    match = re.fullmatch(
        r'\s*3004\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+M[DS]T\s+'
        r'(\d{13})\s+(\d{4}-\d{2}-\d{2})\s+(\d{4}-\d{2}-\d{2})\s+'
        r'(?:x\s+)?' + re.escape(LATEST_DOC) + r'\s+\(MP_CN001\).+', rows[0])
    if not match or match[3] != match[4]:
        raise ValueError('Changed legacy Cotton public-index row')
    stamp = datetime.fromtimestamp(int(match[2]) / 1000, UTC)
    return {'report_date': date.fromisoformat(match[3]).isoformat(),
            'published_at_utc': stamp.isoformat(), 'epoch_ms': int(match[2])}


def audit(history, evidence, output):
    history, evidence, output = Path(history), Path(evidence), Path(output)
    upstream = history / 'publication-content-review.json'
    review = read_record(upstream)
    if review['model_eligible'] or review['publication_timezone_verified'] or review['first_version_reviewed']:
        raise ValueError('This diagnostic expects the existing unverified history')
    for name, checksum in review['files'].items():
        path = (history / name).resolve()
        if not safe_member(name) or not path.is_relative_to(history.resolve()) or digest(path) != checksum:
            raise ValueError('Upstream content evidence corrupted')
    rows = review['version_records']
    delayed, suffixes = [], []
    for row in rows:
        if row['published_at'] is not None or row['timezone_verified'] or row['model_eligible']:
            raise ValueError('A historical UTC clock cannot be invented by this diagnostic')
        # Literal displayed dates only: no timezone conversion or availability assignment.
        days = (date.fromisoformat(row['published_local_naive'][:10]) - date.fromisoformat(row['report_date'])).days
        if days < 0:
            raise ValueError('Displayed publication precedes report date')
        if days:
            delayed.append({**row, 'displayed_calendar_day_difference': days,
                            'cause': 'unverified: late publication, correction or archive metadata'})
        if re.search(r'_\d{2}\.txt$', row['document_url']):
            suffixes.append(row['report_date'])
    published = checked_request(evidence, PUBLISHED)
    corrected = checked_request(evidence, CORRECTED)
    control = publication_control(published.read_text(encoding='utf-8'))
    correction_text = corrected.read_text(encoding='utf-8')
    ids = re.findall(r'^\s*(\d{4})\s+', correction_text, re.MULTILINE)
    declared = re.search(r'Total:\s*(\d+) reports', correction_text)
    if not declared or len(ids) != int(declared[1]):
        raise ValueError('Correction index row count changed')
    listing = checked_request(history, LISTING)
    listed, _ = parse_listing(listing.read_bytes(), LISTING)
    same_date = [row for row in listed if row['report_date'] == control['report_date']]
    if len(same_date) != 1:
        raise ValueError('Public epoch control has no unique archived report')
    canonical = checked_request(evidence, LATEST_DOC)
    archived = checked_request(evidence, same_date[0]['document_url'])
    if digest(canonical) != digest(archived):
        raise ValueError('Public-index document differs from archived control version')
    displayed = datetime.fromisoformat(same_date[0]['published_local_naive'])
    epoch_second = datetime.fromisoformat(control['published_at_utc']).replace(tzinfo=None, microsecond=0)
    control.update({'matched_payload_sha256': digest(canonical),
                    'displayed_local_clock': displayed.isoformat(),
                    'observed_display_offset_seconds': int((displayed - epoch_second).total_seconds()),
                    'applies_to_historical_cohort': False,
                    'first_version_verified': False, 'model_eligible': False})
    sources = {url: digest(checked_request(evidence, url))
               for url in (PUBLISHED, CORRECTED, LATEST_DOC, same_date[0]['document_url'])}
    result = {'upstream_review_sha256': digest(upstream), 'review_code_sha256': digest(Path(__file__)),
              'source_sha256': sources, 'reference_rows': review['reference_rows'],
              'matched_content_dates': review['matched_dates'],
              'missing_metadata_dates': review['missing_metadata_dates'],
              'same_displayed_date_count': len(rows) - len(delayed),
              'later_displayed_date_count': len(delayed), 'later_displayed_dates': delayed,
              'delay_calendar_days_distribution': dict(Counter(str(row['displayed_calendar_day_difference']) for row in delayed)),
              'numeric_filename_suffix_count': len(suffixes), 'numeric_filename_suffix_dates': suffixes,
              'suffix_alone_proves_correction': False,
              'correction_index_rows': len(ids), 'correction_index_distinct_slugs': len(set(ids)),
              'cotton_slug_in_correction_index': '3004' in ids,
              'absence_proves_no_corrections': False, 'complete_revision_history_verified': False,
              'latest_legacy_epoch_control': control, 'historical_utc_times_verified': 0,
              'model_eligible': False,
              'remaining_evidence': ['Historical table clock timezone/UTC evidence',
                                     'Publication time tied to each archived file version',
                                     'Missing 2020-01-17 record', 'Source-use review']}
    freeze_record(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.history, args.evidence, args.output)
    print(f"AMS availability: {result['later_displayed_date_count']} later dated records; "
          f"{result['numeric_filename_suffix_count']} suffix flags; historical UTC verified=0; model_eligible=false")


if __name__ == '__main__':
    main()
