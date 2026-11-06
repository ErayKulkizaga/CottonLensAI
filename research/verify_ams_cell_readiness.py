"""Reference set/bisect calculation of conditional source dependency readiness."""

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import pandas as pd
from cottonlens_ml.research.ledger import freeze_record, read_record


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_cell_status(table, contract):
    """Independent literal-table rule; partial/missing tables prove no absence."""
    quotes, _, complete = table
    if contract in quotes:
        return 1
    return 0 if complete else -1


def verify(root, recovery, merged_path, full_path, contract_path, output):
    report = read_record(root/'result.json')
    for name, expected in report['output_hashes'].items():
        if digest(root/name) != expected:
            raise ValueError('Readiness payload hash')
    for name, expected in report['normalized_input_hashes'].items():
        if digest(recovery/name) != expected:
            raise ValueError('Normalized upstream identity')
    if (digest(merged_path) != report['merged_overlay_sha256']
            or digest(full_path) != report['full_daily_overlay_sha256']
            or digest(contract_path) != report['contract_sha256']):
        raise ValueError('Source/contract identity')
    parent = recovery/'inputs/parent'
    annual, suspicious = defaultdict(dict), set()
    for row in csv.DictReader((parent/'inputs/annual.csv').open(encoding='utf-8')):
        if row['source_date_matches'] != 'True':
            continue
        annual[row['date']][row['contract']] = (str(Decimal(row['settlement_cents_raw']).quantize(Decimal('.01'))), row['price_precision_2dp'] == 'True')
        if row['container'] in ('2018-2019 Price file.xlsx', '2019-2020 Price file.xlsx'):
            suspicious.add(row['date'])
    daily = read_record(parent/'verified-daily-overlay.json')['quotes']
    merged = read_record(merged_path)['quotes']; full = read_record(full_path)['quotes']
    conflicts = {r['date'] for r in csv.DictReader((parent/'day-reconciliation.csv').open(encoding='utf-8')) if r['status'] == 'price_conflict_no_repair'}
    conflicts |= {r['date'] for r in read_record(recovery/'weekly-overlay.json')['rows'] if r['conflicts']}
    pool = {}
    for day in set(annual) | set(merged) | set(full):
        raw = dict(sorted(annual.get(day, {}).items()))
        if day in full:
            q = full[day]; pool[day] = (q, sorted(q), True)
        elif day in daily:
            q = daily[day]; pool[day] = (q, sorted(q), True)
        elif day in merged and (day in suspicious or day in conflicts or not raw):
            q = merged[day]; pool[day] = (q, sorted(q), False)
        elif day in suspicious or day in conflicts:
            pool[day] = ({}, [], False)
        else:
            q = {c: p for c, (p, ok) in raw.items() if ok}
            if day in merged and any(q.get(c) != p for c, p in merged[day].items()):
                raise ValueError('Partial witness contradicts unchanged source')
            pool[day] = (q, list(raw), True)
    h = pd.read_parquet(parent/'inputs/history.parquet'); days = h.date.dt.strftime('%Y-%m-%d').tolist()
    positions = {d: i for i, d in enumerate(days)}
    def cell(day, contract):
        return source_cell_status(pool.get(day, ({}, [], False)), contract)
    design = read_record(contract_path); jobs = [j for f in design['folds'] for j in f['jobs']]
    evaluated = {d for f in design['folds'] for d in f['origins']}
    wanted = evaluated | {d for j in jobs for d in j['train_dates']+j['origins']}
    rows = {r['date']: r for r in csv.DictReader((root/'origin-readiness.csv').open(encoding='utf-8'))}
    if set(rows) != wanted or len(evaluated) != 2006 or len(jobs) != 174:
        raise ValueError('Original full cohort/jobs changed')
    grouped, ready = defaultdict(list), {}
    for day in sorted(wanted):
        q, rank, _ = pool.get(day, ({}, [], True)); c = rank[1] if len(rank) >= 2 else None
        row = rows[day]
        if row['contract'] != (c or '') or row['target_date'] != days[positions[day]+1]:
            raise ValueError('Contract/next Cotton session mismatch')
        if c is None:
            if any(row[k] != 'False' for k in ('har_ready', 'ewma_ready', 'target_pair_ready', 'all_required_inputs_ready')):
                raise ValueError('Unknown origin rank was silently filled')
            ready[day] = False
        else:
            grouped[c].append(day)
    requests = set(); checked = 0
    for contract, dates in grouped.items():
        state = [cell(day, contract) for day in days]
        known = [i for i in range(1, len(days)) if state[i-1] == state[i] == 1]
        uncertain = [i for i in range(1, len(days)) if 0 not in (state[i-1], state[i]) and not state[i-1] == state[i] == 1]
        unresolved = set(); cursor = 0
        for day in dates:
            i = positions[day]; row = rows[day]; n = bisect.bisect_right(known, i); u = bisect.bisect_right(uncertain, i)
            while cursor < u:
                j = uncertain[cursor]; unresolved.update(k for k in (j-1, j) if state[k] == -1); cursor += 1
            last = uncertain[u-1] if u else None
            target = state[i] == 1 and cell(days[i+1], contract) == 1
            har = state[i] == 1 and n >= 22 and (last is None or known[n-22] > last)
            ewma = state[i] == 1 and n >= 22 and not u
            for key, expected in [('target_pair_ready', target), ('har_ready', har), ('ewma_ready', ewma), ('all_required_inputs_ready', target and har and ewma)]:
                if row[key] != str(expected):
                    raise ValueError(f'Independent {key} differs at {day}')
            if int(row['ewma_unknown_cells']) != len(unresolved):
                raise ValueError('Unknown source cells were lost')
            if last is not None:
                count = n-bisect.bisect_right(known, last)
                if int(row['known_pair_updates_after_last_unknown']) != count or row['last_unknown_pair_date'] != days[last]:
                    raise ValueError('EWMA uncertainty memory differs')
                if not math.isclose(float(row['unknown_state_multiplier_after_known_updates']), .94**count, rel_tol=1e-14):
                    raise ValueError('Unknown-state attenuation differs')
            ready[day] = target and har and ewma
            requests.update((days[k], contract) for k in unresolved); checked += 1
    stored_requests = {(r['source_date'], r['contract']) for r in csv.DictReader((root/'unknown-ewma-cells.csv').open(encoding='utf-8'))}
    if stored_requests != requests or len(requests) != report['unknown_ewma_source_cells_upper_bound']:
        raise ValueError('Unknown-dependency upper bound differs')
    jr = list(csv.DictReader((root/'job-readiness.csv').open(encoding='utf-8')))
    for job, r in zip(jobs, jr, strict=True):
        if int(r['unready_train_origins']) != sum(not ready[d] for d in job['train_dates']) or int(r['unready_test_origins']) != sum(not ready[d] for d in job['origins']):
            raise ValueError('Common job readiness differs')
    if sum(ready[d] for d in evaluated) != report['evaluation_all_required_ready']:
        raise ValueError('Full evaluation readiness differs')
    result = {'status': 'independently_verified_conditional_dependencies', 'new_fits': 0,
              'original_ordinal_origins': len(evaluated), 'required_origins_checked': len(rows),
              'known_rank_origins_checked': checked, 'common_jobs_checked': len(jobs),
              'potential_unknown_cells_upper_bound_checked': len(requests),
              'report_sha256': digest(root/'result.json'), 'verifier_sha256': digest(Path(__file__)),
              'training_allowed': False, 'market_skill': False,
              'scope': 'Same explicit conservative source scopes, independent set/bisect dependency calculation; does not prove far contracts were listed or calibrate market error.'}
    freeze_record(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'recovery', 'merged', 'full', 'contract', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(verify(a.root, a.recovery, a.merged, a.full, a.contract, a.output)))
