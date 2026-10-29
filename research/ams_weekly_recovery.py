"""Zero-fit, partial quote recovery from visually transcribed dated USDA tables."""

import argparse
import csv
import json
import re
import shutil
from datetime import date
from pathlib import Path

from cottonlens_ml.research.ledger import freeze_record, read_record

from research.ams_contract_reconciliation import cents, parse_daily, sha
from research.weekly_named_source import printed_date


def parse_review(table, publication):
    """Retain literal blank cells and closures; never infer additional maturities."""
    contracts = table['contracts']
    if (len(contracts) != 6 or contracts != sorted(set(contracts))
            or any(date.fromisoformat(c).day != 1 for c in contracts)):
        raise ValueError('Six literal, ordered contract headers required')
    rows = []
    for line in table['rows']:
        day, *values = line.split()
        lag = (date.fromisoformat(publication) - date.fromisoformat(day)).days
        if not 0 < lag <= 8 or any(c[:7] < day[:7] for c in contracts):
            raise ValueError('Observation/header outside printed report week')
        if values == ['Holiday']:
            quotes, missing, status = {}, contracts, 'explicit_holiday'
        else:
            if len(values) != 6 or any(v != '-' and not re.fullmatch(r'\d+\.\d{2}', v) for v in values):
                raise ValueError('Exactly six literal two-decimal or blank cells required')
            quotes = {c: str(cents(v)) for c, v in zip(contracts, values, strict=True) if v != '-'}
            missing = [c for c, v in zip(contracts, values, strict=True) if v == '-']
            # A leading expired-in-current-month blank is explicit evidence, not a guessed roll.
            if missing and list(quotes) != contracts[len(missing):]:
                raise ValueError('Interior missing quote cannot identify a nearby prefix')
            status = 'visible_quotes'
        rows.append({'date': day, 'quotes': quotes, 'explicit_blank_contracts': missing,
                     'status': status, 'literal_row': line})
    days = [r['date'] for r in rows]
    if not 3 <= len(days) <= 5 or days != sorted(set(days)):
        raise ValueError('Unique chronological report rows required')
    return rows


def annual_prefix_quotes(path):
    """Preserve invalid far cells in position; never compress the visible prefix."""
    days = {}
    with path.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if row['source_date_matches'] != 'True':
                continue  # Parent rejects misdated source files; never use their prices.
            valid = row['price_precision_2dp'] == 'True'
            price = str(cents(row['settlement_cents_raw'])) if valid else row['settlement_cents_raw']
            cell = {'price': price, 'valid': valid}
            current = days.setdefault(row['date'], {})
            if row['contract'] in current and current[row['contract']] != cell:
                raise ValueError('Conflicting annual cells; preserve source ambiguity')
            current[row['contract']] = cell
    return {d: dict(sorted(v.items())) for d, v in days.items()}


def compare(annual, daily, row):
    if not row['quotes']:
        return 'explicit_holiday', []
    q = row['quotes']; conflicts = []
    old = list(annual.items())
    if len(old) != 10:
        return 'incomplete_annual_no_admission', []
    if any(not cell.get('valid', True) for _, cell in old[:len(q)]):
        return 'invalid_annual_prefix_no_admission', []
    for (contract, value), (old_contract, cell) in zip(q.items(), old, strict=False):
        if cents(value) != cents(cell['price']):
            conflicts.append({'kind': 'annual_price', 'contract': contract,
                              'weekly': value, 'annual_contract': old_contract, 'annual': cell['price']})
        if daily is not None and (contract not in daily or cents(daily[contract]) != cents(value)):
            conflicts.append({'kind': 'dated_daily_price', 'contract': contract,
                              'weekly': value, 'daily': daily.get(contract)})
    return ('price_conflict_no_admission' if conflicts else 'verified_partial_prefix'), conflicts


def daily_witness(parent, overlay, day):
    # Rejected/conflicting daily reports remain evidence too. An accepted-only
    # overlay would silently erase the very contradiction we must preserve.
    source = overlay['raw_sources'].get(day)
    if source is None:
        return None
    digest = source['document_sha256']
    raw = parent/'inputs'/'daily'/(digest+'.txt')
    if sha(raw) != digest:
        raise ValueError('Prior raw daily witness changed')
    return parse_daily(raw.read_bytes(), day)


def audit(parent, capture, transcription, output):
    from pypdf import PdfReader

    if output.exists():
        raise ValueError('Preserve existing recovery namespace')
    previous = read_record(parent/'result.json')
    for name, expected in {**previous['input_hashes'], **previous['output_hashes']}.items():
        if sha(parent/name) != expected:
            raise ValueError('Frozen parent evidence changed')
    capture_identity = json.loads((capture/'identity.json').read_text())
    if capture_identity['source_sha256'] != sha(Path(__file__).with_name('ams_weekly_capture.py')):
        raise ValueError('Captured source identity changed')
    reviewed = json.loads(transcription.read_text())
    if reviewed['training_admitted'] is not False or reviewed['historical_availability_verified'] is not False:
        raise ValueError('Value review cannot admit availability or market training')
    overlay = read_record(parent/'verified-daily-overlay.json')
    annual = annual_prefix_quotes(parent/'inputs/annual.csv')
    unknown = set(previous['unresolved_suspect_days'])
    rows, sources, accepted = [], {}, {}
    for publication, table in sorted(reviewed['tables'].items()):
        folder = capture/'reports'/publication
        receipt = json.loads((folder/'receipt.json').read_text())
        if (receipt['status'] != 'candidate_for_value_identity_review'
                or receipt['printed_publication_date'] != publication
                or receipt['historical_availability_verified'] is not False
                or receipt['first_vintage_verified'] is not False or receipt['training_allowed'] is not False):
            raise ValueError('Exact dated, unadmitted source receipt required')
        for name, expected in receipt['files'].items():
            if Path(name).name != name or sha(folder/name) != expected:
                raise ValueError('Captured PDF/image hash changed')
        pdf_date = printed_date(PdfReader(folder/'report.pdf').pages[0].extract_text()).isoformat()
        if pdf_date != publication:
            raise ValueError('Printed PDF date mismatch')
        sources[publication] = {'receipt_sha256': sha(folder/'receipt.json'),
                                'files': receipt['files'], 'url': receipt['url'], 'page': 1,
                                'contracts': table['contracts']}
        for row in parse_review(table, publication):
            day = row['date']
            if any(r['date'] == day for r in rows):
                raise ValueError('Overlapping reviewed dates require explicit reconciliation')
            daily = daily_witness(parent, overlay, day)
            status, conflicts = compare(annual.get(day, {}), daily, row)
            rows.append({**row, 'publication_date': publication, 'status': status,
                         'conflicts': conflicts, 'was_unresolved_suspect': day in unknown,
                         'daily_overlap': daily is not None,
                         'historical_availability_verified': False})
            if status == 'verified_partial_prefix':
                accepted[day] = row['quotes']
    added = sorted(unknown & set(accepted))
    remaining = sorted(unknown - set(accepted))
    output.mkdir(parents=True)
    # Include immutable inputs so replay never relies on a mutable web source.
    shutil.copytree(parent, output/'inputs'/'parent')
    shutil.copytree(capture, output/'inputs'/'capture')
    shutil.copyfile(transcription, output/'inputs'/'transcription.json')
    freeze_record(output/'weekly-overlay.json', {'quotes': accepted, 'rows': rows, 'sources': sources,
                  'training_allowed': False, 'historical_availability_verified': False,
                  'first_vintage_verified': False, 'scope': 'Only literal five/six quote prefixes; no inferred far contracts or daily publication time.'})
    with (output/'rows.csv').open('x', encoding='utf-8', newline='') as f:
        names = ['date', 'publication_date', 'status', 'was_unresolved_suspect', 'daily_overlap',
                 'quote_count', 'conflict_count']
        w = csv.DictWriter(f, fieldnames=names, lineterminator='\n'); w.writeheader()
        w.writerows({**{k: r[k] for k in names[:5]}, 'quote_count': len(r['quotes']),
                     'conflict_count': len(r['conflicts'])} for r in rows)
    result = {'profile': 'ams-weekly-contract-recovery-v1', 'new_fits': 0,
              'reviewed_reports': len(sources), 'reviewed_rows': len(rows),
              'visible_quote_cells': sum(len(r['quotes']) for r in rows),
              'accepted_partial_days': len(accepted), 'newly_resolved_suspect_prefix_days': added,
              'remaining_unresolved_suspect_days': remaining,
              'price_conflict_days': [r['date'] for r in rows if r['conflicts']],
              'explicit_holiday_rows': [r['date'] for r in rows if r['status'] == 'explicit_holiday'],
              'daily_overlap_days': sum(r['daily_overlap'] for r in rows),
              'original_ordinal_origins': previous['original_ordinal_origins'],
              'training_allowed': False, 'source_admission_allowed': False, 'market_skill': False,
              'historical_availability_verified': False, 'first_vintage_verified': False,
              'source_sha256': sha(Path(__file__)), 'capture_identity': capture_identity,
              'input_hashes': {p.relative_to(output).as_posix(): sha(p)
                               for p in sorted((output/'inputs').rglob('*')) if p.is_file()},
              'output_hashes': {n: sha(output/n) for n in ('weekly-overlay.json', 'rows.csv')}}
    freeze_record(output/'result.json', result)
    return {k: v for k, v in result.items() if k not in ('input_hashes', 'capture_identity')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'capture', 'transcription', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(audit(a.parent, a.capture, a.transcription, a.output)))
