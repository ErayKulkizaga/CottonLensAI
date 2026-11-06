"""Trace actual HAR/EWMA dependencies without treating censored quotes as absent."""

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from cottonlens_ml.research.ledger import freeze_record, read_record

from research.ams_contract_reconciliation import SUSPECT, sha
from research.ams_weekly_recovery import annual_prefix_quotes

PRESENT, ABSENT, UNKNOWN = 1, 0, -1


def quote_cell(table, contract, day):
    if contract in table['quotes']:
        return PRESENT, float(table['quotes'][contract])
    # A month comparison or a sorted partial quote list cannot prove absence.
    # Historical labels may themselves be wrong (the annual-source defect).
    # Only an explicitly complete, admitted table establishes a missing cell.
    if table['complete']:
        return ABSENT, None
    return UNKNOWN, None


def trace(cells):
    """Unknown possible updates persist in EWMA; 22 subsequent known pairs heal HAR."""
    tail = deque(maxlen=22); uncertain = set(); last_unknown = None
    known_count = 0; since_unknown = 0; state = None
    for i, (status, value) in enumerate(cells):
        if i:
            previous, old = cells[i-1]
            if ABSENT in (previous, status):
                pass  # No adjacent accepted pair; never bridge the missing Cotton bar.
            elif previous == status == PRESENT:
                if not all(math.isfinite(p) and p > 0 for p in (old, value)):
                    raise ValueError('Finite positive source quotes required')
                square = math.log(value/old)**2
                state = .94*state+.06*square if state is not None else square
                known_count += 1; since_unknown += 1; tail.append(i)
            else:
                uncertain.update(j for j in (i-1, i) if cells[j][0] == UNKNOWN)
                last_unknown = i; since_unknown = 0; tail.clear()
        yield {'known_pair_count': known_count, 'har_ready': len(tail) == 22,
               'ewma_ready': known_count >= 22 and not uncertain,
               'ewma_unknown_cells': len(uncertain), 'last_unknown_pair_index': last_unknown,
               'known_pair_updates_after_last_unknown': since_unknown if uncertain else None,
               'unknown_state_multiplier_after_known_updates': .94**since_unknown if uncertain else 0.,
               'har_first_pair_index': tail[0] if len(tail) == 22 else None}, uncertain


def source_tables(annual_path, daily, merged, conflict_days, full_daily=None):
    annual = annual_prefix_quotes(annual_path); suspect = set()
    with annual_path.open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r['container'] in SUSPECT and r['source_date_matches'] == 'True':
                suspect.add(r['date'])
    tables = {}
    full_daily = full_daily or {}
    for day in sorted(set(annual) | set(merged['quotes']) | set(full_daily)):
        raw = annual.get(day, {})
        unchanged = day not in suspect and day not in conflict_days and bool(raw)
        partial = day in merged['quotes'] and day not in daily['quotes']
        if day in full_daily:
            q = full_daily[day]
            tables[day] = {'quotes': q, 'ranked': sorted(q), 'complete': True, 'kind': 'verified_full_daily_settlement'}
        elif unchanged and partial:
            if any(c not in raw or not raw[c]['valid'] or raw[c]['price'] != p for c, p in merged['quotes'][day].items()):
                raise ValueError('Weekly prefix contradicts an assumed unchanged annual block')
            tables[day] = {'quotes': {c: v['price'] for c, v in raw.items() if v['valid']},
                           'ranked': list(raw), 'complete': True,
                           'kind': 'unchanged_annual_outside_known_bad_blocks_with_prefix_witness'}
        elif day in merged['quotes']:
            q = merged['quotes'][day]
            tables[day] = {'quotes': q, 'ranked': sorted(q), 'complete': day in daily['quotes'],
                           'kind': 'dated_daily' if day in daily['quotes'] else 'verified_weekly_prefix'}
        elif day in suspect or day in conflict_days:
            tables[day] = {'quotes': {}, 'ranked': None, 'complete': False, 'kind': 'unresolved_identity_or_price'}
        else:
            tables[day] = {'quotes': {c: v['price'] for c, v in raw.items() if v['valid']},
                           'ranked': list(raw), 'complete': True, 'kind': 'unchanged_annual_outside_known_bad_blocks'}
    return tables, suspect


def audit(recovery, merged_path, contract_path, output, full_daily_path=None):
    if output.exists():
        raise ValueError('Preserve readiness namespace')
    recovery_result = read_record(recovery/'result.json')
    upstream = {**recovery_result['input_hashes'], **recovery_result['output_hashes']}
    required = ['inputs/parent/result.json', 'inputs/parent/inputs/annual.csv',
                'inputs/parent/inputs/history.parquet', 'inputs/parent/verified-daily-overlay.json',
                'inputs/parent/day-reconciliation.csv', 'weekly-overlay.json']
    normalized_input_hashes = {name: upstream[name] for name in required}
    for name, expected in normalized_input_hashes.items():
        if sha(recovery/name) != expected:
            raise ValueError('Frozen recovery input changed')
    parent = recovery/'inputs'/'parent'
    old = read_record(parent/'result.json'); daily = read_record(parent/'verified-daily-overlay.json')
    merged = read_record(merged_path); weekly = read_record(recovery/'weekly-overlay.json')
    full_quotes, full_evidence = {}, None
    if full_daily_path:
        full = read_record(full_daily_path); full_evidence = read_record(full_daily_path.with_name('result.json'))
        if (full_evidence['status'] != 'independently_verified_full_daily_values'
                or full_evidence['overlay_sha256'] != sha(full_daily_path)
                or full_evidence['verified_days'] != len(full['quotes']) or full['training_allowed'] is not False):
            raise ValueError('Verified, unadmitted full-daily overlay required')
        full_quotes = full['quotes']
    if merged['parent_sha256'] != sha(parent/'verified-daily-overlay.json') or merged['weekly_sha256'] != sha(recovery/'weekly-overlay.json'):
        raise ValueError('Merged source identity differs')
    conflicts = {r['date'] for r in csv.DictReader((parent/'day-reconciliation.csv').open(encoding='utf-8'))
                 if r['status'] == 'price_conflict_no_repair'}
    conflicts.update(r['date'] for r in weekly['rows'] if r['conflicts'])
    tables, suspect = source_tables(parent/'inputs'/'annual.csv', daily, merged, conflicts, full_quotes)
    h = pd.read_parquet(parent/'inputs'/'history.parquet')
    days = h.date.dt.strftime('%Y-%m-%d').tolist(); positions = {d: i for i, d in enumerate(days)}
    design = read_record(contract_path); jobs = [j for f in design['folds'] for j in f['jobs']]
    evaluation = {d for f in design['folds'] for d in f['origins']}
    requested = evaluation | {d for j in jobs for d in j['train_dates']+j['origins']}
    if len(evaluation) != 2006 or len(jobs) != 174 or not requested <= set(days):
        raise ValueError('Original complete cohort/common jobs required')
    empty = {'quotes': {}, 'ranked': [], 'complete': False, 'kind': 'no_accepted_original_source_table'}
    rows, by_contract = {}, defaultdict(dict)
    for day in sorted(requested):
        i = positions[day]; current = tables.get(day, empty); rank = current['ranked']
        contract = rank[1] if rank and len(rank) >= 2 else None
        target = days[i+1] if i+1 < len(days) else None
        origin_known = contract is not None and quote_cell(current, contract, day)[0] == PRESENT
        target_known = bool(origin_known and target and quote_cell(tables.get(target, empty), contract, target)[0] == PRESENT)
        rows[day] = {'date': day, 'target_date': target, 'evaluation_slot': day in evaluation,
                     'origin_table_kind': current['kind'], 'contract': contract,
                     'rank_identified_in_repaired_scope': contract is not None,
                     'origin_quote_ready': origin_known,
                     'target_pair_ready': target_known, 'har_ready': False, 'ewma_ready': False,
                     'ewma_unknown_cells': None, 'last_unknown_pair_date': None,
                     'known_pair_updates_after_last_unknown': None,
                     'unknown_state_multiplier_after_known_updates': None}
        if contract:
            by_contract[contract][i] = day
    needed = defaultdict(lambda: {'using_origins': set(), 'last_using_origin': None})
    for contract, wanted in sorted(by_contract.items()):
        cells = [quote_cell(tables.get(d, empty), contract, d) for d in days]
        for i, (state, unknown_indices) in enumerate(trace(cells)):
            if i not in wanted:
                continue
            day = wanted[i]; row = rows[day]
            row.update({k: state[k] for k in ('har_ready', 'ewma_ready', 'ewma_unknown_cells',
                        'known_pair_updates_after_last_unknown', 'unknown_state_multiplier_after_known_updates')})
            row['har_ready'] = row['har_ready'] and row['origin_quote_ready']
            row['ewma_ready'] = row['ewma_ready'] and row['origin_quote_ready']
            row['last_unknown_pair_date'] = days[state['last_unknown_pair_index']] if state['last_unknown_pair_index'] is not None else None
            if not row['rank_identified_in_repaired_scope']:
                raise ValueError('Rank cannot be inferred from old annual contract names')
            for k in unknown_indices:
                item = needed[(days[k], contract)]; item['using_origins'].add(day); item['last_using_origin'] = day
    records = list(rows.values())
    evaluated = [r for r in records if r['evaluation_slot']]
    for r in records:
        r['all_required_inputs_ready'] = r['target_pair_ready'] and r['har_ready'] and r['ewma_ready']
    dependencies = [{'source_date': d, 'contract': c, 'kind': tables.get(d, empty)['kind'],
                     'using_origin_count': len(v['using_origins']), 'first_using_origin': min(v['using_origins']),
                     'last_using_origin': v['last_using_origin']} for (d, c), v in sorted(needed.items())]
    job_rows = [{'role': j['role'], 'bucket': j['bucket'], 'cutoff': j['cutoff'],
                 'unready_train_origins': sum(not rows[d]['all_required_inputs_ready'] for d in j['train_dates']),
                 'unready_test_origins': sum(not rows[d]['all_required_inputs_ready'] for d in j['origins'])} for j in jobs]
    output.mkdir(parents=True)
    witnesses = {d: dict(v) for d, v in merged['quotes'].items()}
    for day, values in full_quotes.items():
        if day in witnesses and any(witnesses[day].get(c) != p for c, p in values.items()):
            raise ValueError('New full witness conflicts with an accepted prior witness')
        witnesses[day] = values
    freeze_record(output/'witness-overlay.json', {'quotes': dict(sorted(witnesses.items())),
                  'prior_overlay_sha256': sha(merged_path),
                  'full_overlay_sha256': sha(full_daily_path) if full_daily_path else None,
                  'training_allowed': False, 'historical_availability_verified': False,
                  'first_vintage_verified': False,
                  'scope': 'Only independently dated daily/full-daily quotes and weekly literal prefixes; no inferred far cells or clock admission.'})
    for name, values in [('origin-readiness.csv', records), ('unknown-ewma-cells.csv', dependencies), ('job-readiness.csv', job_rows)]:
        with (output/name).open('x', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator='\n'); w.writeheader(); w.writerows(values)
    result = {'profile': 'ams-required-cell-readiness-v1', 'new_fits': 0, 'original_ordinal_origins': len(evaluated),
              'required_unique_origins': len(requested), 'common_jobs': len(jobs),
              'known_bad_annual_source_days': len(suspect),
              'evaluation_rank_unknown': sum(not r['rank_identified_in_repaired_scope'] for r in evaluated),
              'evaluation_target_pairs_ready': sum(r['target_pair_ready'] for r in evaluated),
              'evaluation_har_ready': sum(r['har_ready'] for r in evaluated),
              'evaluation_ewma_ready': sum(r['ewma_ready'] for r in evaluated),
              'evaluation_all_required_ready': sum(r['all_required_inputs_ready'] for r in evaluated),
              'jobs_with_unready_train': sum(r['unready_train_origins'] > 0 for r in job_rows),
              'jobs_with_unready_test': sum(r['unready_test_origins'] > 0 for r in job_rows),
              'unknown_ewma_source_cells_upper_bound': len(dependencies),
              'unknown_ewma_source_dates': sorted({r['source_date'] for r in dependencies}),
              'unknown_source_dates_with_no_origin_rank': sorted(d for d, r in rows.items() if not r['rank_identified_in_repaired_scope']),
              'incomplete_state_multiplier_is_not_an_error_bound': True,
              'scope': 'Conditional use of unchanged annual data outside known bad blocks. Unknown far quote presence is not imputed; dependency count is an upper bound, not proof a far contract was listed. EWMA multiplier needs an initial-error bound before any negligibility claim.',
              'previous_unknown_suspect_days': len(old['unresolved_suspect_days']),
              'remaining_unknown_suspect_days': recovery_result['remaining_unresolved_suspect_days'],
              'remaining_unknown_suspect_days_after_full_daily': sorted(set(recovery_result['remaining_unresolved_suspect_days'])-set(full_quotes)),
              'full_daily_days': len(full_quotes), 'witness_quote_days': len(witnesses),
              'training_allowed': False, 'market_skill': False, 'historical_availability_verified': False,
              'source_sha256': sha(Path(__file__)), 'recovery_report_sha256': sha(recovery/'result.json'),
              'normalized_input_hashes': normalized_input_hashes,
              'full_daily_overlay_sha256': sha(full_daily_path) if full_daily_path else None,
              'full_daily_verification_sha256': sha(full_daily_path.with_name('result.json')) if full_daily_path else None,
              'merged_overlay_sha256': sha(merged_path), 'contract_sha256': sha(contract_path),
              'output_hashes': {n: sha(output/n) for n in ('origin-readiness.csv', 'unknown-ewma-cells.csv', 'job-readiness.csv', 'witness-overlay.json')}}
    freeze_record(output/'result.json', result)
    return {k: v for k, v in result.items() if not isinstance(v, (list, dict))}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recovery', 'merged', 'contract', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--full-daily', type=Path)
    a = p.parse_args(); print(json.dumps(audit(a.recovery, a.merged, a.contract, a.output, a.full_daily)))
