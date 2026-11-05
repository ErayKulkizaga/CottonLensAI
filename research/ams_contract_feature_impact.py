"""Check actual stored HAR inputs against contiguous, independently identified quotes."""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path

import pandas as pd
from cottonlens_ml.research.ledger import freeze_record, read_record

from research.ams_contract_reconciliation import sha


def reconstruct(history, quotes):
    dates = history.date.dt.strftime('%Y-%m-%d').tolist()
    rows = []
    for i, native in enumerate(history.itertuples()):
        day = dates[i]
        row = {'date': day, 'verified_contiguous_window': False, 'changed': False}
        window = dates[max(0, i-22):i+1]
        if len(window) == 23 and all(d in quotes for d in window):
            contract = sorted(quotes[day])[1]
            if all(contract in quotes[d] for d in window):
                prices = [float(quotes[d][contract]) for d in window]
                if any(not math.isfinite(p) or p <= 0 for p in prices):
                    raise ValueError('Finite positive independently identified prices required')
                squares = [math.log(b/a)**2 for a, b in pairwise(prices)]
                row.update(verified_contiguous_window=True, true_contract=contract,
                           first_input_date=window[0], last_input_date=window[-1])
                for k in (1, 5, 22):
                    true = math.log(max(sum(squares[-k:])/k, 1e-12))
                    old = getattr(native, f'own_har_{k}')
                    row.update({f'true_har_{k}': true, f'stored_har_{k}': old,
                                f'har_{k}_changed': bool(math.isfinite(old) and abs(true-old) > 1e-10)})
                row['changed'] = any(row[f'har_{k}_changed'] for k in (1, 5, 22))
        rows.append(row)
    return pd.DataFrame(rows)


def audit(history_path, overlay_path, contract_path, output):
    if output.exists():
        raise ValueError('Preserve existing feature impact')
    h = pd.read_parquet(history_path); overlay = read_record(overlay_path)
    contract = read_record(contract_path)
    table = reconstruct(h, overlay['quotes'])
    changed = set(table.loc[table.changed, 'date'])
    evaluated = {d for f in contract['folds'] for d in f['origins']}
    jobs = [{'year': f['year'], 'role': j['role'], 'bucket': j['bucket'],
             'verified_changed_train_rows': len(set(j['train_dates']) & changed),
             'verified_changed_test_rows': len(set(j['origins']) & changed)} for f in contract['folds'] for j in f['jobs']]
    output.mkdir(parents=True)
    table.to_csv(output/'feature-impact.csv', index=False, lineterminator='\n')
    result = {'new_fits': 0, 'verified_contiguous_history_windows': int(table.verified_contiguous_window.sum()),
              'verified_changed_history_origins': len(changed),
              'verified_changed_evaluation_origins': len(changed & evaluated),
              'jobs_with_verified_changed_train': sum(j['verified_changed_train_rows'] > 0 for j in jobs),
              'jobs_with_verified_changed_test': sum(j['verified_changed_test_rows'] > 0 for j in jobs),
              'jobs': jobs, 'history_sha256': sha(history_path), 'overlay_sha256': sha(overlay_path),
              'contract_sha256': sha(contract_path), 'source_sha256': sha(Path(__file__)),
              'feature_csv_sha256': sha(output/'feature-impact.csv'), 'market_skill': False,
              'source_admission_allowed': False,
              'scope': 'Lower bound on actual stored feature errors where all23prior Cotton bars are independently identified; unknown windows retained, no gap compression or forecast metric.'}
    freeze_record(output/'result.json', result)
    return {k: v for k, v in result.items() if k != 'jobs'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('history', 'overlay', 'contract', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(audit(a.history, a.overlay, a.contract, a.output)))
