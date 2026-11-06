"""Independent Decimal price-loss and scalar paired-block check; no fitting."""
import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd


def validate_cohort(frame, history, folds, groups=('ridge', 'quantile'), horizon=1):
    """Independently bind every exported origin/target/value to the frozen split.

    Equal row counts across arms do not prove a correct common cohort. This
    verifies frozen dataset consistency, not vendor or named-contract truth.
    """
    dates = pd.DatetimeIndex(history.date)
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Frozen history dates must be unique and chronological')
    expected = [d for fold in folds for d in fold['origins']]
    if not expected or expected != sorted(set(expected)):
        raise ValueError('Frozen fold origins must be unique and chronological')
    keys = dates.strftime('%Y-%m-%d')
    if not set(expected).issubset(keys):
        raise ValueError('Expected origin absent from frozen history')
    target = history.set_index(keys).loc[expected]
    years = [fold['year'] for fold in folds for _ in fold['origins']]
    if set(frame.group) != set(groups):
        raise ValueError('Unexpected or missing prediction arm')
    for group in groups:
        rows = frame.loc[frame.group.eq(group)].reset_index(drop=True)
        if rows.date.tolist() != expected:
            raise ValueError('Prediction origins differ from exact frozen cohort')
        if rows.year.tolist() != years or not rows.horizon.eq(horizon).all():
            raise ValueError('Prediction year/horizon differs from frozen split')
        if rows.target_date.tolist() != target[f'target_date_{horizon}'].dt.strftime('%Y-%m-%d').tolist():
            raise ValueError('Prediction target dates differ from frozen history')
        for field in ('cotton_close', f'target_return_{horizon}'):
            if not np.array_equal(rows[field].to_numpy(), target[field].to_numpy()):
                raise ValueError('Prediction target/price values differ from frozen history')
        numeric = rows[['cotton_close', f'target_return_{horizon}', 'predicted_return', 'raw_predicted_return']]
        if not np.isfinite(numeric).all().all() or rows.cotton_close.le(0).any():
            raise ValueError('Finite predictions/targets and positive prices required')
    return {'origins_per_arm': len(expected), 'arms': list(groups), 'horizon': horizon,
            'exact_frozen_cohort_verified': True, 'vendor_semantics_verified': False}


def decimal_losses(prices, actual_returns, predicted_returns):
    naive, model = [], []
    with localcontext() as ctx:
        ctx.prec = 50
        for p, y, prediction in zip(prices, actual_returns, predicted_returns, strict=True):
            p, y, prediction = (Decimal(str(v)) for v in (p, y, prediction))
            if not all(v.is_finite() for v in (p, y, prediction)) or p <= 0:
                raise ValueError('Finite inputs and positive price required')
            actual = p*y.exp()
            naive.append(abs(actual-p))
            model.append(abs(actual-p*prediction.exp()))
    return naive, model


def independent_interval(parts, block):
    rng = np.random.default_rng(42)
    samples = []
    n = sum(len(p) for p in parts)
    for offset in range(0, 10000, 128):
        count = min(128, 10000-offset)
        sums = np.zeros(count)
        for part in parts:
            length = min(block, len(part))
            starts = rng.integers(0, len(part)-length+1, (count, int(np.ceil(len(part)/length))))
            indexes = (starts[..., None]+np.arange(length)).reshape(count, -1)[:, :len(part)]
            sums += part[indexes].sum(axis=1)
        samples.extend(sums/n)
    values = np.concatenate(parts)
    samples = np.asarray(samples)
    centered = samples-samples.mean()
    return values.mean()-np.quantile(centered, [.975, .025])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.folder/'reports/result.json').read_bytes())
    path = args.folder/'reports/predictions.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != report['predictions_sha256']:
        raise ValueError('Prediction export checksum mismatch')
    frame = pd.read_csv(path, float_precision='round_trip')
    ready_path = args.folder/'ready.json'
    ready = json.loads(ready_path.read_bytes())
    ready_body = {k: v for k, v in ready.items() if k != 'record_id'}
    ready_id = hashlib.sha256(json.dumps(ready_body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if ready_id != ready['record_id']:
        raise ValueError('Frozen readiness identity mismatch')
    history_path = args.folder/'history.parquet'
    if hashlib.sha256(history_path.read_bytes()).hexdigest() != ready['history_sha256']:
        raise ValueError('Frozen history checksum mismatch')
    cohort = validate_cohort(frame, pd.read_parquet(history_path), ready['identity']['split']['folds'])
    arms = {}
    for group in ('ridge', 'quantile'):
        rows = frame.loc[frame.group.eq(group)].copy().reset_index(drop=True)
        if len(rows) != 2006 or rows.date.duplicated().any():
            raise ValueError('Unexpected origin cohort')
        model = {}
        for mode, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
            naive, model[mode] = decimal_losses(rows.cotton_close, rows.target_return_1, rows[field])
        with localcontext() as ctx:
            ctx.prec = 50
            naive_mean = sum(naive)/Decimal(len(rows))
            for mode, losses in model.items():
                mae = float(sum(losses)/Decimal(len(rows)))
                expected = report['arms'][group][mode]
                if abs(mae-expected['mae']) > 1e-12 or abs(float(naive_mean)-expected['naive_mae']) > 1e-12:
                    raise ValueError('Independent Decimal MAE mismatch')
                field = 'predicted_return' if mode == 'selected' else 'raw_predicted_return'
                direction = float(100*np.mean(np.sign(rows[field]) == np.sign(rows.target_return_1)))
                errors = np.asarray([float(v) for v in losses])
                naive_array = np.asarray([float(v) for v in naive])
                wins = sum(errors[rows.year.eq(year)].mean() < naive_array[rows.year.eq(year)].mean()
                           for year in sorted(rows.year.unique()))
                if abs(direction-expected['direction_pct']) > 1e-12 or wins != expected['year_wins']:
                    raise ValueError('Independent direction/year gate mismatch')
                active = rows[field].ne(0)
                active_rate = float(100*active.mean())
                if abs(active_rate-expected['active_rate_pct']) > 1e-12:
                    raise ValueError('Independent active rate mismatch')
                if [r['year'] for r in expected['years']] != sorted(rows.year.unique()):
                    raise ValueError('Independent annual coverage mismatch')
                for annual in expected['years']:
                    mask = rows.year.eq(annual['year'])
                    gain = float(100*(1-errors[mask].mean()/naive_array[mask].mean()))
                    if not mask.any() or abs(gain-annual['gain_pct']) > 1e-10:
                        raise ValueError('Independent annual gain mismatch')
                arms[f'{group}/{mode}'] = {'mae': mae, 'naive_mae': float(naive_mean),
                                         'gain_pct': 100*(1-mae/float(naive_mean)),
                                         'direction_pct': direction, 'year_wins': int(wins),
                                         'active_rate_pct': active_rate,
                                         'active_direction_pct': float(100*(np.sign(rows.loc[active, field])
                                             == np.sign(rows.loc[active, 'target_return_1'])).mean()) if active.any() else None}
        rows['difference'] = np.array([float(a-b) for a, b in zip(naive, model['selected'], strict=True)])
        if group == 'quantile':
            parts = [f.difference.to_numpy() for _, f in rows.groupby('year', sort=True)]
            for block in (20, 60):
                ci = independent_interval(parts, block)
                expected = report['arms']['quantile']['selected']['paired'][str(block)]['difference_ci_95']
                if not np.allclose(ci, expected, rtol=0, atol=1e-11):
                    raise ValueError('Independent block interval mismatch')
    first = frame.loc[frame.group.eq('ridge')].reset_index(drop=True)
    second = frame.loc[frame.group.eq('quantile')].reset_index(drop=True)
    for field in ('date', 'target_date', 'cotton_close', 'target_return_1'):
        if not first[field].equals(second[field]):
            raise ValueError('Cross-arm target mismatch')
    selected = arms['quantile/selected']
    decision = ('CONTINUE' if selected['gain_pct'] >= 5 and selected['direction_pct'] >= 53
                and selected['year_wins'] >= 6 and all(
                    report['arms']['quantile']['selected']['paired'][str(b)]['difference_ci_95'][0] > 0 for b in (20, 60)) else 'STOP')
    if decision != report['decision']:
        raise ValueError('Decision mismatch')
    result = {'status': 'verified', 'new_fits': 0, 'decimal_price_losses': len(frame)*3,
              'paired_primary_intervals': 2, 'decision': decision, 'arms': arms,
              'report_sha256': hashlib.sha256((args.folder/'reports/result.json').read_bytes()).hexdigest(),
              'cohort': cohort, 'history_sha256': ready['history_sha256'],
              'ready_sha256': hashlib.sha256(ready_path.read_bytes()).hexdigest(),
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    encoded = (json.dumps(result, indent=2)+'\n').encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise ValueError('Conflicting independent verification')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
