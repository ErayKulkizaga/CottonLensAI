"""One manual, bounded, journaled recovery of explicitly listed historical daily reports."""

import argparse
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests
from cottonlens_ml.research.ledger import freeze_record, read_record, writer
from cottonlens_ml.sources.nass_archive import (
    ArchiveHTTPError,
    bounded_get,
    release_details,
)

from research.ams_contract_reconciliation import parse_daily, sha

BASE = 'https://esmis.nal.usda.gov/publication/daily-spot-quotations-excerpts/'


def recover_day(day, folder, fetch):
    folder.mkdir(parents=True, exist_ok=False)
    page_url = BASE+day
    page = fetch(page_url); (folder/'release-page.html').write_bytes(page)
    stamp, url = release_details(page.decode('utf-8'), page_url)
    parsed = urlsplit(url)
    if (stamp[:10] != day or parsed.scheme != 'https' or parsed.hostname != 'esmis.nal.usda.gov'
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('Exact dated official daily release required')
    report = fetch(url); (folder/'report.txt').write_bytes(report)
    quotes = parse_daily(report, day)
    return {'date': day, 'status': 'verified_dated_daily_values', 'quotes': quotes,
            'release_page_url': page_url, 'report_url': url, 'page_date_field': stamp,
            'document_sha256': sha(folder/'report.txt'), 'page_sha256': sha(folder/'release-page.html'),
            'historical_availability_verified': False, 'first_vintage_verified': False,
            'training_allowed': False}


def run(plan_path, root, minutes=15):
    plan = json.loads(plan_path.read_text()); days = plan['dates']
    if (days != sorted(set(days)) or any(date.fromisoformat(d).isoformat() != d for d in days)
            or len(days) > 260 or plan['maximum_requests'] != 2*len(days)
            or plan['maximum_report_bytes'] != 2000000 or plan['maximum_total_response_bytes'] != 40000000
            or not 0 < minutes <= 15 or not plan['manual_one_off'] or plan['automatic_scheduler_enabled']):
        raise ValueError('Exact bounded manual source plan required')
    identity = {'plan_sha256': sha(plan_path), 'source_sha256': sha(Path(__file__)),
                'parser_sha256': sha(Path(__file__).with_name('ams_contract_reconciliation.py')),
                'python': sys.version.split()[0], 'requests': requests.__version__, 'new_fits': 0}
    root = Path(root); deadline = time.monotonic()+minutes*60
    with writer(root):
        if (root/'identity.json').exists():
            if read_record(root/'identity.json') != identity:
                raise ValueError('Capture source/plan changed; preserve this namespace')
        else:
            freeze_record(root/'identity.json', identity)
        previous = [read_record(p) for p in (root/'records').glob('*.json')]
        completed = {r['date'] for r in previous}
        if not completed.issubset(days) or len(previous) != len(completed):
            raise ValueError('Capture journal outside exact plan')
        used = sum(r['requests'] for r in previous); size = sum(r['response_bytes'] for r in previous)
        session = requests.Session(); failures = 0

        def fetch(url):
            nonlocal used, size
            if used >= plan['maximum_requests'] or size >= plan['maximum_total_response_bytes']:
                raise ValueError('Frozen total request/byte budget exhausted')
            used += 1
            raw = bounded_get(url, session, stage='manual dated contract recovery')
            size += len(raw)
            if len(raw) > plan['maximum_report_bytes'] or size > plan['maximum_total_response_bytes']:
                raise ValueError('Frozen response byte budget exceeded')
            return raw

        for day in days:
            if day in completed:
                record = read_record(root/'records'/f'{day}.json')
                for name, expected in record['files'].items():
                    if sha(root/name) != expected:
                        raise ValueError('Captured source hash changed; no refetch')
                continue
            if time.monotonic() >= deadline:
                return {'status': 'checkpoint_pause', 'completed_dates': len(completed), 'requests': used, 'bytes': size}
            before, before_bytes = used, size; folder = root/'raw'/day
            try:
                result = recover_day(day, folder, fetch); failures = 0
            except (ValueError, ArchiveHTTPError, requests.RequestException) as error:
                result = {'date': day, 'status': 'not_verified', 'error': f'{type(error).__name__}: {error}'}
                failures += 1
            result.update(requests=used-before, response_bytes=size-before_bytes,
                          retrieved_at=datetime.now(UTC).isoformat(),
                          files={p.relative_to(root).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()})
            freeze_record(root/'records'/f'{day}.json', result); completed.add(day)
            if len(completed) % 10 == 0 or result['status'] != 'verified_dated_daily_values':
                print(json.dumps({'completed_dates': len(completed), 'date': day, 'status': result['status'], 'requests': used, 'bytes': size}), flush=True)
            if failures >= 3:
                return {'status': 'stopped_after_three_source_failures', 'completed_dates': len(completed), 'requests': used, 'bytes': size}
    return {'status': 'completed', 'completed_dates': len(completed), 'requests': used, 'bytes': size, 'new_fits': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--minutes', type=float, default=15)
    a = p.parse_args(); print(json.dumps(run(a.plan, a.root, a.minutes)))
