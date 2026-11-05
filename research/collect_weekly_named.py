"""Resume immutable bounded weekly source capture; never fit or admit features."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from pypdf.errors import PyPdfError
from weekly_named_source import extract_pdf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(root):
    contract = json.loads((root / 'contract.json').read_text())
    if sha(Path('research/weekly_named_source.py')) != contract['parser_sha256']:
        raise ValueError('frozen parser changed')
    urls = contract['urls']
    if len(urls) > contract['max_files'] or len(set(urls)) != len(urls):
        raise ValueError('invalid capture budget')
    sample_root = Path(contract['sample_root'])
    samples = {r['url']: r for r in json.loads((sample_root / 'pdf-receipts.json').read_text())}
    total = 0
    for number, url in enumerate(urls):
        parts = urlsplit(url)
        if parts.scheme != 'https' or parts.netloc != 'apps.ams.usda.gov' or not parts.path.startswith('/Cotton/Historical'):
            raise ValueError('unapproved source')
        dest = root / 'reports' / f'{number:03d}'
        receipt_path = dest / 'receipt.json'
        path = dest / 'report.pdf'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['url'] != url:
                raise ValueError('capture identity mismatch')
            if 'sha256' in receipt and sha(path) != receipt['sha256']:
                raise ValueError('captured bytes changed')
            total += receipt.get('bytes', 0)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError('unreceipted interrupted PDF; preserve and review')
        receipt = {'url': url, 'attempted_at': datetime.now(UTC).isoformat(),
                   'first_vintage_verified': False, 'exact_publication_utc_verified': False,
                   'daily_availability_verified': False, 'training_admitted': False}
        try:
            if url in samples:
                old = samples[url]
                source = sample_root / (old['season'] + '.pdf')
                if sha(source) != old['sha256']:
                    raise ValueError('sample checksum changed')
                data = source.read_bytes()
                receipt.update({'reused_sample': True, 'retrieved_at': old['retrieved_at'],
                                'final_url': old['final_url']})
            else:
                with urllib.request.urlopen(url, timeout=30) as stream:
                    if urlsplit(stream.url).netloc != parts.netloc:
                        raise ValueError('unapproved redirect')
                    data = stream.read(contract['max_file_bytes'] + 1)
                    receipt.update({'retrieved_at': datetime.now(UTC).isoformat(), 'final_url': stream.url})
            if len(data) > contract['max_file_bytes'] or total + len(data) > contract['max_total_bytes']:
                raise ValueError('capture byte budget exceeded')
            if not data.startswith(b'%PDF'):
                raise ValueError('not PDF')
            path.write_bytes(data)
            total += len(data)
            receipt.update({'bytes': len(data), 'sha256': sha(path)})
            try:
                table = extract_pdf(path)
                (dest / 'table.json').write_text(json.dumps(table, indent=2) + '\n')
                receipt.update({'status': 'parsed', 'table_sha256': sha(dest / 'table.json')})
            except (ValueError, RuntimeError, PyPdfError) as exc:
                receipt.update({'status': 'parser_rejected', 'error': str(exc)})
        except (ValueError, RuntimeError, OSError) as exc:
            receipt.update({'status': 'capture_failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        if (number + 1) % 25 == 0:
            print(json.dumps({'terminal_attempts': number + 1, 'captured_bytes': total}), flush=True)
        if receipt.get('error') == 'capture byte budget exceeded':
            raise ValueError('stop at frozen byte budget')
    receipts = [json.loads(p.read_text()) for p in sorted((root / 'reports').glob('*/receipt.json'))]
    status = {key: sum(r['status'] == key for r in receipts)
              for key in ('parsed', 'parser_rejected', 'capture_failed')}
    coverage = {}
    cells = {}
    conflicts = []
    for p in sorted((root / 'reports').glob('*/table.json')):
        for row in json.loads(p.read_text())['rows']:
            day = row['observation_date']
            coverage.setdefault(day[:4], set()).add(day)
            for contract_month, value in row['settlements'].items():
                key = day + '/' + contract_month
                if key in cells and cells[key] != value:
                    conflicts.append(key)
                cells[key] = value
    summary = {'profile': contract['profile'], 'terminal_attempts': len(receipts), 'status': status,
               'captured_bytes': total, 'unique_named_settlement_cells': len(cells),
               'unique_observation_dates': sum(map(len, coverage.values())),
               'dates_by_year': {y: len(v) for y, v in sorted(coverage.items())},
               'conflicting_cells': sorted(set(conflicts)), 'new_fits': 0, 'training_admitted': False}
    summary_path = root / 'coverage.json'
    if summary_path.exists() and json.loads(summary_path.read_text()) != summary:
        raise ValueError('immutable coverage changed')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    shutil.copyfile('research/weekly_named_source.py', root / 'frozen-parser.py')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    capture(parser.parse_args().root)
