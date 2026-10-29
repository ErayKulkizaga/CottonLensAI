"""Separate zero-fit v2 source reconciliation; preserve v1 raw bytes and errors."""
import argparse
import calendar
import hashlib
import json
import re
from datetime import date, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

from research.weekly_named_source import CONTRACT, MONTHS, printed_date

OBS_MONTHS = {**MONTHS, **{m: i for i, m in enumerate(calendar.month_name) if m}}
ROW = re.compile(r'^(' + '|'.join(OBS_MONTHS) + r')[\s-]+(\d{1,2})\s+(.+)$')


def parse_reconciled_table(text, publication):
    if 'icefuturescontractsettlement' not in re.sub(r'\s+', '', text).lower():
        raise ValueError('missing settlement caption')
    matches = list(CONTRACT.finditer(text))
    runs = [matches[i:i + 6] for i in range(len(matches) - 5)
            if all(not text[a.end():b.start()].strip() for a, b in pairwise(matches[i:i + 6]))]
    if len(runs) != 1:
        raise ValueError('ambiguous six-contract header')
    header = runs[0]
    contracts = [f"20{m[2]}-{MONTHS[re.sub(r'\s+', '', m[1])]:02d}" for m in header]
    if len(set(contracts)) != 6 or contracts != sorted(contracts):
        raise ValueError('duplicate or unordered delivery months; no inferred repair')
    rows = []
    started = False
    for line in text[header[-1].end():].splitlines():
        match = ROW.fullmatch(line.strip())
        if not match:
            if started:
                break
            continue
        started = True
        month, day, numbers = match.groups()
        candidates = []
        for year in (publication.year - 1, publication.year):
            try:
                observed = date(year, OBS_MONTHS[month], int(day))
            except ValueError:
                continue
            if timedelta(0) <= publication - observed <= timedelta(days=8):
                candidates.append(observed)
        if len(candidates) != 1:
            raise ValueError('observation outside publication week')
        observed = candidates[0]
        if any(c < observed.strftime('%Y-%m') for c in contracts):
            raise ValueError('expired delivery month')
        row = {'observation_date': observed.isoformat(), 'printed_publication_date': publication.isoformat(),
               'publication_lag_calendar_days': (publication - observed).days}
        tokens = numbers.split()
        if re.search(r'holiday|closed|no trading|no quotation', numbers, re.IGNORECASE):
            row.update({'available': False, 'reason': 'explicit_market_closure_or_no_quotation',
                        'raw_cells': numbers, 'settlements': {}})
        else:
            if not 7 <= len(tokens) <= 9 or any(not re.fullmatch(r'\d+\.\d{2}', t) for t in tokens[:6]):
                raise ValueError('missing or malformed futures cells')
            if any(Decimal(t) <= 0 for t in tokens[:6]):
                raise ValueError('nonpositive futures settlement')
            if any(not re.fullmatch(r'\d+\.\d{2}|\*|\d/', t) for t in tokens[6:]):
                raise ValueError('unrecognized auxiliary index format')
            row.update({'available': True, 'settlements': dict(zip(contracts, tokens[:6])),
                        'ignored_non_futures_cells': tokens[6:]})
        rows.append(row)
    dates = [r['observation_date'] for r in rows]
    if not 3 <= len(rows) <= 5 or dates != sorted(set(dates)):
        raise ValueError('invalid unique ordered daily cohort')
    return rows


def reconcile(root, output):
    from pypdf import PdfReader

    tables, errors = {}, {}
    for receipt_path in sorted((root / 'reports').glob('*/receipt.json')):
        receipt = json.loads(receipt_path.read_text())
        if 'sha256' not in receipt:
            errors[receipt_path.parent.name] = 'no captured PDF'
            continue
        path = receipt_path.parent / 'report.pdf'
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
            raise ValueError('immutable parent PDF changed')
        reader = PdfReader(path)
        pages = [p.extract_text() or '' for p in reader.pages]
        publication = printed_date(pages[0])
        candidates = [(i + 1, s) for i, s in enumerate(pages)
                      if 'icefuturescontractsettlement' in re.sub(r'\s+', '', s).lower()]
        try:
            if len(candidates) != 1:
                raise ValueError('ambiguous settlement page')
            page, text = candidates[0]
            rows = parse_reconciled_table(text, publication)
            tables[receipt_path.parent.name] = {'source_sha256': receipt['sha256'], 'url': receipt['url'],
                                               'page': page, 'rows': rows}
        except ValueError as exc:
            errors[receipt_path.parent.name] = str(exc)
    days = {r['observation_date'] for t in tables.values() for r in t['rows'] if r['available']}
    summary = {'profile': 'weekly-named-settlement-parser-reconciliation-v2', 'parsed_reports': len(tables),
               'rejected_reports': len(errors), 'available_dates': len(days),
               'dates_by_year': {y: sum(d.startswith(y) for d in days) for y in sorted({d[:4] for d in days})},
               'unavailable_rows': sum(not r['available'] for t in tables.values() for r in t['rows']),
               'new_fits': 0, 'training_admitted': False, 'errors': errors}
    for name, body in [('tables.json', tables), ('summary.json', summary)]:
        path = output / name
        if path.exists():
            raise ValueError('immutable reconciliation output exists')
        path.write_text(json.dumps(body, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reconcile(args.root, args.output)
