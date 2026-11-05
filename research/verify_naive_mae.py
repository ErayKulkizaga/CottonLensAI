"""Independent Decimal price-loss and scalar paired-block check; no fitting."""
import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd


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
    arms = {}
    for group in ('ridge', 'quantile'):
        rows = frame.loc[frame.group.eq(group)].copy().reset_index(drop=True)
        if len(rows) != 2006 or rows.date.duplicated().any():
            raise ValueError('Unexpected origin cohort')
        naive, model = [], {'selected': [], 'raw': []}
        with localcontext() as ctx:
            ctx.prec = 50
            for row in rows.itertuples():
                p, y = Decimal(str(row.cotton_close)), Decimal(str(row.target_return_1))
                actual = p*y.exp()
                naive.append(abs(actual-p))
                for mode, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
                    model[mode].append(abs(actual-p*Decimal(str(getattr(row, field))).exp()))
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
                arms[f'{group}/{mode}'] = {'mae': mae, 'naive_mae': float(naive_mean),
                                         'gain_pct': 100*(1-mae/float(naive_mean)),
                                         'direction_pct': direction, 'year_wins': int(wins)}
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
              'report_sha256': hashlib.sha256((args.folder/'reports/result.json').read_bytes()).hexdigest()}
    encoded = (json.dumps(result, indent=2)+'\n').encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise ValueError('Conflicting independent verification')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
