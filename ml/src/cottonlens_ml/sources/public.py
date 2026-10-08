"""Archive public reports and compile explicitly reviewed point-in-time packages.

No publication date is inferred from observation dates, schedules or HTTP dates.
API credentials are intentionally not accepted in URLs or stored in receipts.
"""
import argparse
import hashlib
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests

from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.cohort import content_id
from cottonlens_ml.sprint import freeze_record, read_record

SOURCES = {
    'ams': ('www.ams.usda.gov', 'apps.ams.usda.gov', 'mymarketnews.ams.usda.gov', 'marsapi.ams.usda.gov'),
    'wasde': ('www.usda.gov', 'usda.gov', 'esmis.nal.usda.gov'),
    'export_sales': ('apps.fas.usda.gov', 'www.fas.usda.gov', 'api.fas.usda.gov'),
    'crop_progress': ('www.nass.usda.gov', 'esmis.nal.usda.gov', 'quickstats.nass.usda.gov'),
    'cftc': ('www.cftc.gov', 'cftc.gov'),
    'pink_sheet': ('thedocs.worldbank.org', 'pubdocs.worldbank.org'),
    'power': ('power.larc.nasa.gov',),
}


def validate_url(kind, url):
    parsed = urlsplit(url)
    if (kind not in SOURCES or parsed.scheme != 'https' or parsed.hostname not in SOURCES[kind]
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.fragment or any(word in parsed.query.lower() for word in ('key=', 'token=', 'password='))):
        raise ValueError('Public HTTPS source URL required; credentials must not be stored')


def archive(kind, url, root, *, session=None):
    """One bounded public request. Unreviewed bytes are quarantined, not features."""
    validate_url(kind, url)
    client = session or requests.Session()
    with client.get(url, timeout=(10, 60), stream=True, allow_redirects=False) as response:
        if 300 <= response.status_code < 400:
            raise ValueError('Review redirected source URL explicitly before downloading')
        response.raise_for_status()
        parts, size = [], 0
        for part in response.iter_content(65536):
            size += len(part)
            if size > 25 * 1024 * 1024:
                raise ValueError('Public report exceeds 25 MiB; use a reviewed bulk-data adapter')
            parts.append(part)
        content = b''.join(parts)
        if not content:
            raise ValueError('Empty source response')
        checksum = hashlib.sha256(content).hexdigest()
        folder = Path(root) / kind / checksum
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'source.bin'
        if target.exists():
            if digest(target) != checksum:
                raise ValueError('Corrupt archived public source')
        else:
            with target.open('xb') as handle:
                handle.write(content)
        receipt = folder / 'retrieval.json'
        if not receipt.exists():
            freeze_record(receipt, {'kind': kind, 'source_url': url, 'sha256': checksum,
                'retrieved_at': datetime.now(UTC).isoformat(), 'content_type': response.headers.get('Content-Type'),
                'publication_timestamp_verified': False, 'model_eligible': False,
                'cost_tl': 0, 'policy': 'Raw archive only; review publication, vintage and usage rights before modeling'})
        return folder


def utc_timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Explicit timezone required for publication timestamps')
    return stamp.tz_convert('UTC')


def archive_observation(kind, url, root, *, session=None, clock=None):
    """Observe actual bytes now; never backdate them to a publisher/report date.

    Each call performs a fresh request even if identical bytes are already stored.
    This establishes availability by download completion, not first publication.
    The historical feature compiler still requires its separate reviewed evidence.
    """
    clock = clock or (lambda: datetime.now(UTC))
    root = Path(root).resolve()
    started = utc_timestamp(clock())
    if session is None:
        with requests.Session() as owned_session:
            raw = archive(kind, url, root / 'raw', session=owned_session)
    else:
        raw = archive(kind, url, root / 'raw', session=session)
    completed = utc_timestamp(clock())
    if completed < started:
        raise ValueError('Observation clock moved backwards; no completed observation recorded')
    body = {'schema': 'observed-source-v1', 'kind': kind, 'source_url': url,
            'request_started_at': started.isoformat(),
            'observed_available_at': completed.isoformat(),
            'source_file': (raw / 'source.bin').relative_to(root).as_posix(),
            'source_sha256': digest(raw / 'source.bin'),
            'published_at': None, 'publication_timestamp_verified': False,
            'first_version_verified': False, 'model_eligible': False,
            'availability_basis': 'actual_download_completion_not_first_publication',
            'historical_backdating_allowed': False}
    target = root / 'observations' / (content_id(body) + '.json')
    freeze_record(target, body)
    return target


def verify_observation(path, *, cutoff=None):
    """Verify one observed snapshot; optional cutoff rejects future observations."""
    path = Path(path).resolve()
    body = read_record(path)
    if (body['schema'] != 'observed-source-v1'
            or body['availability_basis'] != 'actual_download_completion_not_first_publication'
            or body['published_at'] is not None
            or any(body[key] for key in ('publication_timestamp_verified', 'first_version_verified',
                                        'model_eligible', 'historical_backdating_allowed'))):
        raise ValueError('Observation cannot certify historical publication or model admission')
    validate_url(body['kind'], body['source_url'])
    started = utc_timestamp(body['request_started_at'])
    completed = utc_timestamp(body['observed_available_at'])
    if completed < started or (cutoff is not None and completed > utc_timestamp(cutoff)):
        raise ValueError('Observation not available at requested cutoff')
    root = path.parent.parent
    name = body['source_file']
    raw = (root / name).resolve()
    if (not safe_member(name) or not raw.is_relative_to(root)
            or digest(raw) != body['source_sha256']):
        raise ValueError('Observed source bytes/path corrupted')
    return body


def reviewed_upper_bound(release, files, root):
    """Validate a human-reviewed, specific-version availability receipt.

    This checks the review contract, not the truth of a publisher's assertion.
    Schedules, embargoes and present-day retrieval dates are not evidence types.
    """
    if release.get('published_at') is not None or release.get('timestamp_verified') is not False:
        raise ValueError('An availability upper bound is not an exact publication timestamp')
    if release.get('availability_verified') is not True:
        raise ValueError('Reviewed availability evidence required')
    name = release['publication_evidence_file']
    if name not in files:
        raise ValueError('Availability receipt must be checksummed')
    evidence = json.loads((Path(root) / name).read_text(encoding='utf-8'))
    stamp = utc_timestamp(release['available_by'])
    if (evidence.get('schema') != 'source-availability-upper-bound-v1'
            or evidence.get('basis') not in ('official_release_record', 'contemporaneous_archive_capture')
            or evidence.get('version_verified') is not True
            or evidence.get('timing_verified') is not True
            or evidence.get('vintage_id') != release['vintage_id']
            or evidence.get('source_sha256') != files[release['source_file']]
            or utc_timestamp(evidence['available_by']) != stamp):
        raise ValueError('Specific-version availability evidence mismatch')
    members = evidence.get('evidence_files')
    if (not isinstance(members, list) or not members
            or any(not isinstance(member, str) or member not in files or member == name for member in members)):
        raise ValueError('Checksummed underlying availability evidence required')
    return stamp


def compile_review(review_file, output):
    """Review rows are complete feature snapshots at an evidenced publication time.

    Review schema contains kind, features, usage, files and releases. Each release
    supplies values, observed_through, published_at, vintage_id, source_url,
    source_file, publication_evidence_file, vintage_evidence_file, timestamp_verified.
    The named evidence files must be included in the SHA256 file map.
    Bounded availability additionally supplies available_by, availability_verified
    and availability_basis=verified_upper_bound, with a reviewed version receipt.
    """
    review_file, output = Path(review_file), Path(output)
    review = json.loads(review_file.read_text(encoding='utf-8'))
    kind, features = review['kind'], review['features']
    if kind not in SOURCES or not features or len(features) != len(set(features)):
        raise ValueError('Supported free source and unique features required')
    if any(not name.startswith(kind + '_') or not name.isidentifier() for name in features):
        raise ValueError('Features must use the source namespace')
    usage = review['usage']
    if usage.get('cost_tl') != 0 or usage.get('research_allowed') is not True or not usage.get('terms_url'):
        raise ValueError('Reviewed zero-cost research usage required')
    root = review_file.parent.resolve()
    for name, checksum in review['files'].items():
        path = (root / name).resolve()
        if not safe_member(name) or not path.is_relative_to(root) or digest(path) != checksum:
            raise ValueError('Review evidence checksum/path mismatch')
    rows = []
    for release in review['releases']:
        validate_url(kind, release['source_url'])
        if release.get('availability_basis', 'exact_publication') not in ('exact_publication', 'verified_upper_bound'):
            raise ValueError('Unsupported availability evidence basis')
        bounded = release.get('availability_basis') == 'verified_upper_bound'
        if not release.get('vintage_id') or (not bounded and release.get('timestamp_verified') is not True):
            raise ValueError('Actual publication and vintage review required')
        stamp = (reviewed_upper_bound(release, review['files'], root) if bounded
                 else utc_timestamp(release['published_at']))
        observed = utc_timestamp(release['observed_through'])
        if observed > stamp:
            raise ValueError('Observed period extends beyond publication')
        if set(release['values']) != set(features) or not np.isfinite(list(release['values'].values())).all():
            raise ValueError('Complete finite feature snapshot required for each release')
        for name in ('source_file', 'publication_evidence_file', 'vintage_evidence_file'):
            if release[name] not in review['files']:
                raise ValueError('Every release needs checksummed publication/vintage evidence')
        if kind == 'wasde':
            from cottonlens_ml.sources.wasde_admission import require_review
            require_review(release, review['files'], root, features, stamp)
        rows.append({**release['values'], 'published_at': pd.NaT if bounded else stamp,
                     'available_at': stamp, 'availability_verified': True,
                     'availability_basis': 'verified_upper_bound' if bounded else 'exact_publication',
                     'observed_through': observed, 'vintage_id': release['vintage_id'],
                     'source_url': release['source_url'], 'source_sha256': review['files'][release['source_file']],
                     'timestamp_verified': not bounded,
                     'source_file': release['source_file'],
                     'publication_evidence_file': release['publication_evidence_file'],
                     'vintage_evidence_file': release['vintage_evidence_file']})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.available_at.duplicated().any():
        raise ValueError('Nonempty unique publication snapshots required')
    has_bounds = frame.availability_basis.eq('verified_upper_bound').any()
    if not has_bounds:
        # Preserve the legacy package schema and join clock for exact reviews.
        frame = frame.drop(columns=['available_at', 'availability_verified', 'availability_basis', 'source_file'])
    if output.exists():
        raise ValueError('Publication packages are immutable; choose a new output')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name('.' + output.name + '-' + uuid.uuid4().hex + '.pending')
    staging.mkdir()
    for name in review['files']:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    rows_name = 'released-features.parquet'
    if rows_name in review['files'] or 'publication-manifest.json' in review['files']:
        raise ValueError('Reserved package member')
    frame.sort_values('available_at' if has_bounds else 'published_at').to_parquet(staging / rows_name, index=False)
    files = {**review['files'], rows_name: digest(staging / rows_name)}
    manifest = {'kind': kind, 'features': features, 'files': files, 'rows_file': rows_name,
                'vintage_policy': 'as_published', 'usage': usage,
                'review_sha256': digest(review_file), 'max_age_days': review['max_age_days']}
    if has_bounds:
        manifest['availability_schema'] = 'verified-availability-v1'
    if not isinstance(manifest['max_age_days'], int) or not 1 <= manifest['max_age_days'] <= 366:
        raise ValueError('Explicit bounded source freshness policy required')
    (staging / 'publication-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    staging.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('archive')
    fetch.add_argument('--kind', choices=tuple(SOURCES), required=True)
    fetch.add_argument('--url', required=True)
    fetch.add_argument('--output', type=Path, required=True)
    observe = commands.add_parser('observe', help='Capture current bytes and actual observation time; not historical admission')
    observe.add_argument('--kind', choices=tuple(SOURCES), required=True)
    observe.add_argument('--url', required=True)
    observe.add_argument('--output', type=Path, required=True)
    observe.add_argument('--browser-transport', action='store_true', help='Use the locked AMS archive transport')
    build = commands.add_parser('compile')
    build.add_argument('--review', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'observe':
        if args.browser_transport:
            if args.kind != 'ams':
                parser.error('Browser transport is limited to the reviewed AMS adapter')
            from curl_cffi import requests as browser_requests

            from cottonlens_ml.sources.ams_publications import PublicationTransport

            def new_client():
                return browser_requests.Session(impersonate='chrome136', verify=True, trust_env=False)

            with PublicationTransport(new_client(), browser_requests.exceptions.RequestException, renew=new_client) as session:
                result = archive_observation(args.kind, args.url, args.output, session=session)
        else:
            result = archive_observation(args.kind, args.url, args.output)
        body = verify_observation(result)
        print(json.dumps({'observation': str(result), 'observed_available_at': body['observed_available_at'],
                          'source_sha256': body['source_sha256'], 'model_eligible': False}))
        return
    print(archive(args.kind, args.url, args.output) if args.command == 'archive'
          else compile_review(args.review, args.output))


if __name__ == '__main__':
    main()
