"""Pinned public On-Call inventory/acquisition and quarantined Tier-A table.

No guessed weekly URLs, automatic redirects, credentials or model training.
The same immutable archive receipts and single writer used elsewhere apply.
"""
import argparse
import json
import re
import shutil
import time
from contextlib import nullcontext
from html.parser import HTMLParser
from itertools import pairwise
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import numpy as np
import pandas as pd
import requests

from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, read_record, writer
from cottonlens_ml.sources.oncall import audit_receipt
from cottonlens_ml.sources.public import (
    archive_observation,
    validate_url,
    verify_observation,
)

INDEX = 'https://www.cftc.gov/MarketReports/CottonOnCall/HistoricalCottonOn-Call/index.htm'
FIELDS = ('net_share', 'sales_share', 'net_share_change')


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            value = dict(attrs).get('href', '')
            if 'deaoncall' in value.lower():
                self.links.append(urljoin(INDEX, value))


def inventory(receipt, *, first=2010, last=2023):
    if not 2010 <= first <= last <= 2023:
        raise ValueError('Pre-2024 inventory only; warmup no earlier than 2010')
    receipt = Path(receipt).resolve()
    body = verify_observation(receipt)
    if body['kind'] != 'cftc' or body['source_url'] != INDEX:
        raise ValueError('Pinned official On-Call index observation required')
    page = _Links()
    page.feed((receipt.parent.parent / body['source_file']).read_text(encoding='utf-8'))
    items, rejected = {}, []
    for url in page.links:
        match = re.fullmatch(r'deaoncall(\d{6}|\d{8})\.html?', urlsplit(url).path.rsplit('/', 1)[-1], re.IGNORECASE)
        if not match:
            rejected.append({'url': url, 'reason': 'unrecognized_indexed_name'})
            continue
        value = match[1]
        year = int(value[4:]) + (2000 if len(value) == 6 else 0)
        if not first <= year <= last:
            continue
        try:
            validate_url('cftc', url)
            nominal = pd.Timestamp(year=year, month=int(value[:2]), day=int(value[2:4]))
        except ValueError:
            rejected.append({'url': url, 'reason': 'unsafe_host_or_invalid_nominal_date'})
            continue
        items[url] = {'source_url': url, 'nominal_url_date': str(nominal.date())}
    items = sorted(items.values(), key=lambda x: (x['nominal_url_date'], x['source_url']))
    if not items or len(items) > 750:
        raise ValueError('Nonempty bounded historical index required (<=750 links)')
    dates = pd.to_datetime([x['nominal_url_date'] for x in items])
    return {'schema': 'oncall-inventory-v1', 'index_sha256': body['source_sha256'],
            'index_observed_at': body['observed_available_at'], 'index_url': INDEX,
            'scope_years': list(range(first, last+1)), 'files': items, 'rejected_links': rejected,
            'counts_by_nominal_year': {str(y): int(sum(dates.year == y)) for y in range(first, last+1)},
            'nominal_gaps_over_10_days': [
                {'before': str(a.date()), 'after': str(b.date()), 'days': (b-a).days}
                for a, b in pairwise(dates) if (b-a).days > 10],
            'date_policy': 'Filename dates only catalogue links; never inferred as-of/publication/availability',
            'cost_tl': 0, 'model_eligible': False, 'release_allowed': False}


def _adopt(receipt, destination):
    receipt, destination = Path(receipt).resolve(), Path(destination).resolve()
    body = verify_observation(receipt)
    source_root = receipt.parent.parent
    for name in (body['source_file'], str(Path(body['source_file']).parent / 'retrieval.json'),
                 f'observations/{receipt.name}'):
        source, target = source_root/name, destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(source) != digest(target):
                raise ValueError('Conflicting adopted source preserved')
        else:
            shutil.copyfile(source, target)
    target = destination/'observations'/receipt.name
    verify_observation(target)
    return target


def acquire(root, *, reuse=(), spacing=1., session=None, max_reports=None):
    """One request per indexed missing report; durable receipts resume without IO.

    HTTP/format failures are preserved, never silently retried as a new version.
    Five consecutive failed requests pause acquisition to avoid hammering a host.
    """
    if spacing < 1:
        raise ValueError('At least one second between public report requests')
    root = Path(root).resolve()
    catalogue = read_record(root/'inventory.json')
    cached = {}
    for folder in reuse:
        for path in (Path(folder)/'observations').glob('*.json'):
            body = verify_observation(path)
            cached.setdefault(body['source_url'], path)
    counts = {'saved': 0, 'reused': 0, 'failed': 0, 'requested': 0}
    consecutive = 0
    with writer(root), (requests.Session() if session is None else nullcontext(session)) as client:
        for item in catalogue['files']:
            url = item['source_url']
            validate_url('cftc', url)
            marker = root/'items'/(content_id(url)+'.json')
            if marker.exists():
                old = read_record(marker)
                if old['source_url'] != url:
                    raise ValueError('Cached URL identity changed')
                if old['status'] == 'observed':
                    verify_observation(root/old['receipt'])
                    counts['saved'] += 1
                else:
                    counts['failed'] += 1
                continue
            if max_reports is not None and counts['requested'] >= max_reports:
                break
            try:
                if url in cached:
                    receipt = _adopt(cached[url], root/'archive')
                    counts['reused'] += 1
                else:
                    time.sleep(spacing)
                    counts['requested'] += 1
                    receipt = archive_observation('cftc', url, root/'archive', session=client)
                body = {'source_url': url, 'status': 'observed',
                        'receipt': receipt.relative_to(root).as_posix()}
                counts['saved'] += 1
                consecutive = 0
            except Exception as exc:  # noqa: BLE001 - bounded public requests, safe error type only
                body = {'source_url': url, 'status': 'unavailable', 'error_type': type(exc).__name__}
                counts['failed'] += 1
                consecutive += 1
            freeze_record(marker, body)
            if (counts['saved']+counts['failed']) % 25 == 0 or consecutive >= 5:
                print(f"ARCHIVE saved={counts['saved']}/{len(catalogue['files'])} "
                      f"reused={counts['reused']} unavailable={counts['failed']}", flush=True)
            if consecutive >= 5:
                break
    counts['expected'] = len(catalogue['files'])
    counts['remaining'] = counts['expected']-counts['saved']-counts['failed']
    return counts


def derive(reports, *, footer_policy='mask_conflicts'):
    if footer_policy not in ('mask_conflicts', 'latest_date_assumption'):
        raise ValueError('Explicit supported footer policy required')
    rows = []
    for report in reports:
        reasons = []
        if report['printed_release_not_before_utc'] is None:
            reasons.append('release_footer_missing')
        if report['oi_header_matches_report_asof'] is not True:
            reasons.append('oi_date_unverified')
        if report['footer_date_conflict'] and footer_policy == 'mask_conflicts':
            reasons.append('conflicting_footer_dates')
        if report['reporting_threshold_contracts'] != 100:
            reasons.append('reporting_threshold_changed_or_unknown')
        if report['missing_quantity_cells'] or set(report['total_checks'].values()) != {'verified'}:
            reasons.append('incomplete_quantity_or_total')
        total = report['totals']
        if total['open_interest'] is None or total['open_interest'] <= 0:
            reasons.append('nonpositive_total_oi')
        asof = pd.Timestamp(report['as_of'])
        if not pd.Timestamp('2010-01-01') <= asof < pd.Timestamp('2024-01-01'):
            raise ValueError('Audit/legacy as-of cannot enter the historical source table')
        footer = pd.Timestamp(report['printed_release_not_before_utc']) if report['printed_release_not_before_utc'] else None
        assumed = max(asof+pd.Timedelta(days=7), footer.tz_convert('UTC').tz_localize(None).normalize()+pd.Timedelta(days=1)) if footer is not None else asof+pd.Timedelta(days=7)
        if footer_policy == 'latest_date_assumption':
            # Unlabelled dates have no verified meaning. Wait beyond every one;
            # +2 days also passes the end of that Eastern date in UTC. This is
            # an exploratory assumption, never proof of first-vintage availability.
            for value in report.get('footer_additional_dates', []):
                assumed = max(assumed, pd.Timestamp(value)+pd.Timedelta(days=2))
        net = (total['unfixed_sales']-total['unfixed_purchases'])/total['open_interest'] if not reasons else np.nan
        rows.append({'report_date': asof, 'assumed_day': assumed, 'net_share': net,
                     'sales_share': total['unfixed_sales']/total['open_interest'] if not reasons else np.nan,
                     'input_mask_reason': ';'.join(reasons), 'source_sha256': report['source_sha256'],
                     'source_url': report['source_url'], 'revision_note_present': report['revision_note_present']})
    if not rows:
        raise ValueError('No parsed historical reports')
    frame = pd.DataFrame(rows).sort_values(['report_date', 'assumed_day']).reset_index(drop=True)
    if frame.report_date.duplicated().any():
        raise ValueError('Multiple report vintages for an as-of date require explicit review; no last-wins')
    valid = frame.report_date.diff().dt.days.between(5, 9) & frame.input_mask_reason.eq('') & frame.input_mask_reason.shift().eq('')
    frame['net_share_change'] = frame.net_share.diff().where(valid)
    return frame


def compile_table(root, output, *, footer_policy='mask_conflicts'):
    root, output = Path(root).resolve(), Path(output)
    catalogue = read_record(root/'inventory.json')
    reports, failures, source_files = [], [], {}
    for item in catalogue['files']:
        marker = root/'items'/(content_id(item['source_url'])+'.json')
        if not marker.exists():
            raise ValueError('Acquisition incomplete; preserve the partial archive and resume')
        body = read_record(marker)
        if body['source_url'] != item['source_url']:
            raise ValueError('Acquisition marker URL identity changed')
        if body['status'] != 'observed':
            failures.append({**item, 'reason': body['error_type']})
            continue
        # Corrupt or substituted immutable evidence is fatal, not a missing report.
        observed = verify_observation(root/body['receipt'])
        if observed['source_url'] != item['source_url']:
            raise ValueError('Observed report URL identity changed')
        try:
            report = audit_receipt(root/body['receipt'])
            reports.append(report)
            source_files[body['receipt']] = digest(root/body['receipt'])
        except (ValueError, UnicodeError) as exc:
            failures.append({**item, 'reason': str(exc)})
    frame = derive(reports, footer_policy=footer_policy)
    raw = frame.to_csv(index=False, date_format='%Y-%m-%d').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != raw:
            raise ValueError('Conflicting On-Call table preserved; new namespace required')
    else:
        output.write_bytes(raw)
    manifest = {'schema': 'oncall-tier-a-table-v1', 'inventory_sha256': digest(root/'inventory.json'),
                'table_sha256': digest(output), 'rows': len(frame), 'masked_reports': int(frame.input_mask_reason.ne('').sum()),
                'fields': list(FIELDS), 'source_receipts': source_files, 'unavailable_or_unparsed': failures,
                'parsed_versions': [{key: r[key] for key in ('as_of', 'source_url', 'source_sha256',
                    'observed_available_at', 'printed_release_not_before_utc', 'oi_header_matches_report_asof',
                    'footer_date_conflict', 'revision_note_present')} for r in reports],
                'scope': 'CFTC Form 304 unfixed physical call cotton, totals over listed delivery months',
                'source_tier': 'A_exploration_only', 'cost_tl': 0, 'model_eligible': False, 'release_allowed': False,
                'publication_timestamp_verified': False, 'first_version_verified': False,
                'calendar_policy': 'max(asof +7 calendar days, printed release lower-bound UTC day +1), then Cotton lag1/2/6; assumption only',
                'mask_policy': 'ambiguous footer/OI date, quantity/threshold; inputs masked, origins retained',
                'change_policy': 'difference of observed net/OI levels only for 5-9 day consecutive unmasked reports; printed revised changes unused'}
    if footer_policy == 'latest_date_assumption':
        manifest.update({'schema': 'oncall-tier-a-table-v2', 'footer_policy': footer_policy,
            'calendar_policy': 'max(asof +7 days, printed lower-bound UTC day +1, each unlabelled footer date +2 days), then first strictly later Cotton observation and lag1/2/6; assumption only',
            'mask_policy': 'OI date/quantity/threshold/release missing remain masked; footer conflicts retained as unverified delayed assumptions only',
            'conflicting_footer_reports': sum(r['footer_date_conflict'] for r in reports)})
        for version, report in zip(manifest['parsed_versions'], reports, strict=True):
            version['footer_additional_dates'] = report['footer_additional_dates']
    freeze_record(output.with_suffix('.manifest.json'), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('inventory', 'fetch', 'compile'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index-receipt', type=Path)
    parser.add_argument('--reuse-archive', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--footer-policy', choices=('mask_conflicts', 'latest_date_assumption'), default='mask_conflicts')
    args = parser.parse_args()
    if args.stage == 'inventory':
        if args.index_receipt is None:
            parser.error('--index-receipt required')
        result = inventory(args.index_receipt)
        freeze_record(args.root/'inventory.json', result)
        print(json.dumps({'indexed_reports': len(result['files']), 'years': result['counts_by_nominal_year']}))
    elif args.stage == 'fetch':
        print(json.dumps(acquire(args.root, reuse=args.reuse_archive)))
    else:
        if args.output is None:
            parser.error('--output required')
        result = compile_table(args.root, args.output, footer_policy=args.footer_policy)
        print(json.dumps({k: result[k] for k in ('rows', 'masked_reports', 'table_sha256')}))


if __name__ == '__main__':
    main()
