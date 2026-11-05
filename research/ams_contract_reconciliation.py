"""Reconcile annual contract labels against held, dated daily reports; never fit."""

import argparse
import calendar
import csv
import hashlib
import json
import math
import re
import shutil
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
from cottonlens_ml.research.ledger import freeze_record, read_record

PROFILE = 'ams-daily-contract-reconciliation-v1'
MONTHS = {v: k for k, v in enumerate(calendar.month_abbr) if v}
SUSPECT = {'2018-2019 Price file.xlsx', '2019-2020 Price file.xlsx'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cents(value):
    number = Decimal(str(value))
    if not number.is_finite() or number <= 0:
        raise ValueError('Positive finite settlement required')
    rounded = number.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    if abs(number-rounded) > Decimal('1e-8'):
        raise ValueError('Positive two-decimal settlement required')
    return rounded


def parse_daily(data, day):
    """Independent token extraction; spot columns cannot become futures prices."""
    header = data[:600].decode('ascii', errors='strict')
    printed = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-[A-Za-z]{3}-\d{2})\b', header)
    if printed is None:
        raise ValueError('Missing dated daily report header')
    stamp = datetime.strptime(printed[0], '%m/%d/%Y' if '/' in printed[0] else '%d-%b-%y')  # noqa: DTZ007 - printed calendar date only
    if stamp.date().isoformat() != day:
        raise ValueError('Daily header date differs from declared origin')
    if b'FUTURES' not in data or b'COTTON QUOTATIONS' not in data:
        raise ValueError('Missing dated futures table')
    # Only the source header and numerical table are ASCII. Some later prose is CP1252.
    # Do not discard/replace any byte in the relevant table or modify the archived report.
    table = data.split(b'FUTURES', 1)[1].split(b'COTTON QUOTATIONS', 1)[0].decode('ascii', errors='strict')
    values = []
    for line in table.splitlines():
        match = re.search(r'\b(Mar|May|Jul|Oct|Dec)-(\d{2})\s+(\d+(?:\.\d+)?)\s*$', line)
        if match:
            month, year, value = match.groups()
            contract = date(2000+int(year), MONTHS[month], 1).isoformat()
            if contract[:7] < day[:7]:
                raise ValueError('Past delivery month in daily report')
            values.append((contract, str(cents(value))))
    names = [name for name, _ in values]
    if len(values) != 10 or names != sorted(set(names)):
        raise ValueError('Exactly ten unique chronological futures quotes required')
    return dict(values)


def annual_quotes(path):
    days, rejected = {}, []
    for row in csv.DictReader(path.open(encoding='utf-8', newline='')):
        if row['source_date_matches'] != 'True' or row['price_precision_2dp'] != 'True':
            rejected.append(row); continue
        value = str(cents(row['settlement_cents_raw']))
        current = days.setdefault(row['date'], {})
        old = current.get(row['contract'])
        entry = {'price': value, 'container': row['container']}
        if old is not None and old['price'] != value:
            raise ValueError('Conflicting annual date/contract values; no automatic vintage choice')
        current[row['contract']] = entry
    return {d: dict(sorted(v.items())) for d, v in days.items()}, rejected


def reconcile_day(annual, daily):
    """Position equivalence is verified against a dated ten-quote witness, never inferred."""
    old_names, true_names = list(annual), list(daily)
    if len(annual) != 10 or len(daily) != 10:
        return 'incomplete_annual_or_daily', []
    if [cents(v['price']) for v in annual.values()] != [cents(v) for v in daily.values()]:
        return 'price_conflict_no_repair', []
    mapping = [{'annual_contract': a, 'true_contract': b, 'price': daily[b],
                'identity_changed': a != b} for a, b in zip(old_names, true_names, strict=True)]
    return ('verified_relabel' if old_names != true_names else 'verified_equal'), mapping


def labels(calendar_days, original, verified):
    """Retain every original slot; label-only sensitivity, not a forecast test."""
    rows = []
    for item in original:
        day, target = item['date'], item['target_date']
        row = {'date': day, 'target_date': target, 'old_contract': item['contract'],
               'old_risk': item['named_actual'], 'source_pair_verified': False,
               'true_contract': None, 'true_risk': None, 'label_changed': None, 'ordinal_roll_seam': None}
        if calendar_days.get(day) != target:
            raise ValueError('Original target is not next Cotton observation')
        if day in verified and target in verified:
            first, second = list(verified[day]), list(verified[target])
            contract = first[1]
            if contract in verified[target]:
                risk = math.log(float(verified[target][contract])/float(verified[day][contract]))**2
                old = item['named_actual']
                row.update(source_pair_verified=True, true_contract=contract, true_risk=risk,
                           label_changed=bool(old is not None and not math.isclose(old, risk, rel_tol=1e-10, abs_tol=1e-14)),
                           ordinal_roll_seam=contract != second[1])
        rows.append(row)
    return rows


def audit(annual_path, history_path, cohort_path, panel_path, objects, supplement, output):
    if output.exists():
        raise ValueError('Preserve existing audit namespace')
    panel = read_record(panel_path)
    h = pd.read_parquet(history_path)
    cohort = pd.read_csv(cohort_path, keep_default_na=False, na_values=[''], float_precision='round_trip')
    if len(cohort) != 2006 or cohort.date.duplicated().any() or not h.date.is_monotonic_increasing or h.date.duplicated().any():
        raise ValueError('Frozen complete original calendar required')
    annual, rejected = annual_quotes(annual_path)
    output.mkdir(parents=True)
    for name, source in {'annual.csv': annual_path, 'history.parquet': history_path,
                         'cohort.csv': cohort_path, 'daily-panel.json': panel_path}.items():
        destination = output/'inputs'/name; destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    verified, raw_records, day_rows, quote_rows, missing = {}, {}, [], [], []
    for row in panel['rows']:
        day, digest = row['report_date'], row['document_sha256']
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Canonical daily document hash required')
        choices = [objects/digest, supplement/'raw'/digest/'report.txt']
        found = [p for p in choices if p.is_file()]
        if not found:
            missing.append({'date': day, 'sha256': digest}); continue
        if any(sha(p) != digest for p in found):
            raise ValueError('Held daily source differs from frozen panel digest')
        data = found[0].read_bytes(); parsed = parse_daily(data, day)
        expected = {date(2000+int(k[-2:]), MONTHS[k[:3]], 1).isoformat(): str(cents(v)) for k, v in row['quotes'].items()}
        if parsed != expected:
            raise ValueError('Independent raw parser differs from old panel')
        destination = output/'inputs/daily'/f'{digest}.txt'; destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(data)
        raw_records[day] = {'document_sha256': digest, 'source': 'held dated MP_CN001 text',
                            'publication_timestamp_verified': False, 'first_vintage_verified': False}
        status, mapping = reconcile_day(annual.get(day, {}), parsed)
        if status in ('verified_equal', 'verified_relabel'):
            verified[day] = parsed
        day_rows.append({'date': day, 'status': status, 'relabelled_quotes': sum(m['identity_changed'] for m in mapping),
                         'document_sha256': digest,
                         'annual_containers': '|'.join(sorted({r['container'] for r in annual.get(day, {}).values()}))})
        quote_rows.extend({'date': day, 'document_sha256': digest, **m} for m in mapping)
    calendar_days = dict(zip(h.date.dt.strftime('%Y-%m-%d').iloc[:-1], h.date.dt.strftime('%Y-%m-%d').iloc[1:], strict=True))
    originals = [{'date': r.date, 'target_date': r.target_date, 'contract': r.contract if pd.notna(r.contract) else None,
                  'named_actual': r.named_actual if pd.notna(r.named_actual) else None} for r in cohort.itertuples()]
    paired = labels(calendar_days, originals, verified)
    for name, rows in [('day-reconciliation.csv', day_rows), ('quote-reconciliation.csv', quote_rows),
                       ('label-reconciliation.csv', paired)]:
        pd.DataFrame(rows).to_csv(output/name, index=False, lineterminator='\n')
    freeze_record(output/'verified-daily-overlay.json', {'profile': PROFILE, 'quotes': verified,
                  'raw_sources': raw_records, 'model_eligible': False, 'training_allowed': False,
                  'scope': 'Dated raw ten-price matches only; missing/conflicting dates not silently repaired'})
    suspect_days = {day for day, quotes in annual.items() if any(v['container'] in SUSPECT for v in quotes.values())}
    unresolved = sorted(suspect_days-set(verified))
    result = {'profile': PROFILE, 'new_fits': 0, 'new_dataset_downloads': 0,
              'daily_documents_verified': len(raw_records), 'missing_documents': missing,
              'verified_days': len(verified), 'verified_relabel_days': sum(r['status'] == 'verified_relabel' for r in day_rows),
              'verified_relabel_quotes': sum(r['identity_changed'] for r in quote_rows),
              'annual_rejected_rows_preserved': len(rejected), 'original_ordinal_origins': len(paired),
              'independently_reconstructed_target_pairs': sum(r['source_pair_verified'] for r in paired),
              'numeric_target_changes': sum(r['label_changed'] is True for r in paired),
              'verified_ordinal_roll_seams': sum(r['ordinal_roll_seam'] is True for r in paired),
              'suspect_source_days': len(suspect_days), 'unresolved_suspect_days': unresolved,
              'training_allowed': False, 'historical_availability_verified': False, 'first_vintage_verified': False,
              'market_skill': False, 'release_allowed': False,
              'scope_limit': 'Daily witnesses repair value identities, not historical availability. Unknown dates retain unknown status; no full source admission or market conclusion.',
              'source_sha256': sha(Path(__file__)),
              'input_hashes': {p.relative_to(output).as_posix(): sha(p) for p in (output/'inputs').rglob('*') if p.is_file()},
              'output_hashes': {name: sha(output/name) for name in ('verified-daily-overlay.json', 'day-reconciliation.csv',
                               'quote-reconciliation.csv', 'label-reconciliation.csv')}}
    freeze_record(output/'result.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('annual', 'history', 'cohort', 'panel', 'objects', 'supplement', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    result = audit(args.annual, args.history, args.cohort, args.panel, args.objects, args.supplement, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes', 'unresolved_suspect_days')}))
