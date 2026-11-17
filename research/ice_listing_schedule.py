"""Conditional ICE 10.07 schedule diagnostics, never source admission."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

MONTHS = {'Mar': 3, 'May': 5, 'Jul': 7, 'Oct': 10, 'Dec': 12}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_listing_month(contract):
    delivery = date.fromisoformat(contract)
    if delivery.isoformat() != contract or delivery.day != 1 or delivery.month not in MONTHS.values():
        raise ValueError('Canonical Cotton delivery month required')
    year, month = divmod(delivery.year*12+delivery.month-1-35, 12)
    return date(year, month+1, 1)


def classify(cells):
    grouped = defaultdict(list)
    seen = set()
    for row in cells:
        key = (row['source_date'], row['contract'])
        if key in seen:
            raise ValueError('Duplicate unresolved cell')
        seen.add(key)
        grouped[row['contract']].append(date.fromisoformat(row['source_date']))
    results = []
    for contract, days in sorted(grouped.items()):
        month = default_listing_month(contract)
        before = sum(d < month for d in days)
        within = sum(d.year == month.year and d.month == month.month for d in days)
        results.append({'contract': contract, 'default_listing_month_not_verified_date': month.isoformat()[:7],
                        'before_default_month': before, 'within_default_month': within,
                        'after_default_month': len(days)-before-within,
                        'total_unknown_cells': len(days), 'actual_listing_verified': False,
                        'board_exceptions_verified': False, 'source_cells_admitted': 0})
    return results


def run(readiness, rule_folder, expiry_csv, output):
    from pypdf import PdfReader
    if output.exists():
        raise ValueError('Preserve prior conditional schedule diagnostic')
    source = json.loads((rule_folder/'source.json').read_text())
    pdf = rule_folder/'10_Cotton.pdf'
    if source['sha256'] != digest(pdf):
        raise ValueError('Official rule PDF checksum mismatch')
    text = re.sub(r'\s+', ' ', PdfReader(pdf).pages[5].extract_text())
    start = text.index('Rule 10.07.')
    rule = text[start:]
    anchors = ('Unless the Board otherwise directs', 'unless the Board otherwise determines',
               'first (1 st) Business Day', '(35th) month',
               'effective November 16, 2007')
    if not all(a in rule for a in anchors):
        raise ValueError('Expected rule/version/exception anchors not found; no silent rule substitution')
    current = list(csv.DictReader(expiry_csv.open(encoding='utf-8', newline='')))
    checked = 0
    for row in current:
        match = re.fullmatch(r'="(Mar|May|Jul|Oct|Dec)(\d{2})"', row['CONTRACT SYMBOL'])
        if not match:
            raise ValueError('Unexpected official expiry symbol format')
        month, year = match.groups()
        contract = f'20{year}-{MONTHS[month]:02d}-01'
        actual = datetime.strptime(row['FTD'], '%m/%d/%Y').date()  # noqa: DTZ007 - source session day
        expected = default_listing_month(contract)
        if (actual.year, actual.month) != (expected.year, expected.month):
            raise ValueError('Observed current FTD disagrees with default month; review exceptions')
        checked += 1
    if not checked:
        raise ValueError('No current expiry cross-check rows')
    report = json.loads((readiness/'result.json').read_text())
    cells_path = readiness/'unknown-ewma-cells.csv'
    if digest(cells_path) != report['output_hashes']['unknown-ewma-cells.csv']:
        raise ValueError('Unresolved dependency panel changed')
    with cells_path.open(encoding='utf-8', newline='') as f:
        cells = list(csv.DictReader(f))
    if len(cells) != report['unknown_ewma_source_cells_upper_bound']:
        raise ValueError('Unresolved dependency panel count differs')
    rows = classify(cells)
    totals = Counter()
    for row in rows:
        for key in ('before_default_month', 'within_default_month', 'after_default_month'):
            totals[key] += row[key]
    if sum(totals.values()) != len(cells):
        raise ValueError('Conditional partition lost cells')
    output.mkdir(parents=True)
    path = output/'conditional-contract-schedule.csv'
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result = {'schema': 1, 'status': 'conditional_rule_schedule_not_admission',
              'contracts': len(rows), 'total_unknown_cells_unchanged': len(cells),
              'partition': dict(totals), 'current_expiry_month_matches': checked,
              'rule_effective_annotation': '2007-11-16', 'new_fits': 0,
              'source_cells_admitted': 0, 'historical_board_exceptions_verified': False,
              'historical_actual_listing_dates_verified': False,
              'input_sha256': {'rule_pdf': digest(pdf), 'expiry_csv': digest(expiry_csv),
                               'readiness_report': digest(readiness/'result.json'),
                               'unknown_cells': digest(cells_path)},
              'matrix_sha256': digest(path), 'code_sha256': digest(Path(__file__)),
              'limits': ['A month schedule is not the actual first business/trading day.',
                         'The rule contains Board override language; historical exceptions are unchecked.',
                         'Current FTD month agreement is not historical listing or PIT proof.',
                         'No UNKNOWN cell is changed to ABSENT and no forecast or gate is changed.']}
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('readiness', 'rule-folder', 'expiry-csv', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.readiness, args.rule_folder, args.expiry_csv, args.output), indent=2))
