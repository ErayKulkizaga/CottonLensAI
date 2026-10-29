"""Decimal reconstruction of the strict, independently quoted HAR windows; no fits."""

import argparse
import json
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd
from cottonlens_ml.research.ledger import read_record

from research.ams_contract_reconciliation import sha


def verify(root, history_path, overlay_path, contract_path):
    result = read_record(root/'result.json')
    for p, key in [(history_path, 'history_sha256'), (overlay_path, 'overlay_sha256'),
                   (contract_path, 'contract_sha256'), (root/'feature-impact.csv', 'feature_csv_sha256')]:
        if sha(p) != result[key]:
            raise ValueError('Source/feature input differs')
    h = pd.read_parquet(history_path); quotes = read_record(overlay_path)['quotes']
    table = pd.read_csv(root/'feature-impact.csv', float_precision='round_trip')
    assert table.date.tolist() == h.date.dt.strftime('%Y-%m-%d').tolist()
    dates = table.date.tolist(); checked = 0; changed = set()
    for i, row in enumerate(table.itertuples()):
        days = dates[max(0, i-22):i+1]
        allowed = len(days) == 23 and all(d in quotes for d in days)
        contract = sorted(quotes[row.date])[1] if allowed else None
        allowed = allowed and all(contract in quotes[d] for d in days)
        assert row.verified_contiguous_window == allowed
        if not allowed:
            assert not row.changed
            continue
        assert row.last_input_date == row.date and row.first_input_date == days[0]
        with localcontext() as ctx:
            ctx.prec = 50
            prices = [Decimal(quotes[d][contract]) for d in days]
            squares = [(prices[j]/prices[j-1]).ln()**2 for j in range(1, 23)]
            for k in (1, 5, 22):
                value = float(max(sum(squares[-k:])/k, Decimal('1e-12')).ln())
                assert abs(value-getattr(row, f'true_har_{k}')) < 1e-10
                assert getattr(row, f'stored_har_{k}') == h.iloc[i][f'own_har_{k}']
                checked += 1
        if row.changed:
            changed.add(row.date)
    assert len(changed) == result['verified_changed_history_origins']
    contract = read_record(contract_path)
    jobs = [j for f in contract['folds'] for j in f['jobs']]
    assert sum(bool(set(j['train_dates']) & changed) for j in jobs) == result['jobs_with_verified_changed_train']
    assert sum(bool(set(j['origins']) & changed) for j in jobs) == result['jobs_with_verified_changed_test']
    return {'status': 'independently_verified_feature_inputs', 'new_fits': 0,
            'decimal_feature_scalars': checked, 'verified_changed_history_origins': len(changed),
            'native_jobs_with_changed_train': result['jobs_with_verified_changed_train'],
            'result_sha256': sha(root/'result.json'), 'verifier_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'history', 'overlay', 'contract', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); result = verify(a.root, a.history, a.overlay, a.contract)
    if a.output.exists():
        raise ValueError('Preserve independent receipt')
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
