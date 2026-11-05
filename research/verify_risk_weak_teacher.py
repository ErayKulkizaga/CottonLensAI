"""Independent native, past-selection and structural-mean loss replay; no fits."""

import argparse
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    body = {k: v for k, v in value.items() if k != 'record_id'}
    if value.get('record_id') != fingerprint(body):
        raise ValueError('Frozen record checksum differs')
    return body


def read_predictions(path):
    # "null" is a registered experimental condition, not a missing category.
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False, na_values=[''])


def difference_interval(parts, block):
    total = np.concatenate(parts)
    point = float(total[:, 0].sum()/total[:, 1].sum())
    rng = np.random.default_rng(42); draws = []
    for offset in range(0, 10000, 128):
        count = min(128, 10000-offset); numerator = np.zeros(count); denominator = np.zeros(count)
        for part in parts:
            size = min(block, len(part))
            starts = rng.integers(0, len(part)-size+1, (count, math.ceil(len(part)/size)))
            positions = (starts[..., None]+np.arange(size)).reshape(count, -1)[:, :len(part)]
            numerator += part[:, 0][positions].sum(axis=1)
            denominator += part[:, 1][positions].sum(axis=1)
        if (denominator <= 0).any():
            raise ValueError('Bootstrap lost all eligible observations')
        draws.extend(numerator/denominator)
    values = total[total[:, 1] == 1, 0]
    if np.ptp(values) <= 32*np.finfo(float).eps*max(1., np.max(np.abs(values))):
        return [point, point]
    draws = np.asarray(draws)
    return (point-np.quantile(draws-draws.mean(), [.975, .025])).tolist()


def verify(root):
    ready = read(root/'ready.json'); design = read(root/'decision-contract.json')
    calibration = read(root/'calibration.json'); result = read(root/'reports/result.json')
    if result['market_skill'] or result['native_synthetic_fits'] != 1044 or result['new_market_fits']:
        raise ValueError('Wrong synthetic evidence scope')
    for name, checksum in ready['frozen_hashes'].items():
        if sha(root/name) != checksum:
            raise ValueError('Frozen input hash differs')
    h = pd.read_parquet(root/'inputs/history.parquet')
    past = h.set_index(h.date.dt.strftime('%Y-%m-%d')).loc[calibration['dates']]
    if len(past) != 619 or not (past.target_date_5 < pd.Timestamp(design['calibration_cutoff'])).all():
        raise ValueError('Calibration maturity differs')
    noise = (past.named_variance/past.named_ewma).to_numpy()
    np.testing.assert_allclose(noise/noise.mean(), calibration['innovation_pool'], rtol=1e-14)
    past_x = past.own_har_5.to_numpy()-np.log(past.named_ewma.to_numpy())
    np.testing.assert_allclose([past_x.mean(), past_x.std()],
                               [calibration['feature_mean'], calibration['feature_scale']], rtol=1e-13)
    normalizer = np.log(np.exp(.15*(past_x-past_x.mean())/past_x.std()).mean())
    if not math.isclose(normalizer, calibration['weak_log_normalizer'], abs_tol=1e-13):
        raise ValueError('Past teacher normalization differs')
    cases = {}
    for condition in ('null', 'weak'):
        for seed in (42, 1701, 2703):
            f = pd.read_parquet(root/'cases'/f'{condition}-{seed}.parquet')
            pd.testing.assert_frame_equal(f.drop(columns=['teacher_mean', 'teacher_innovation', 'named_variance']),
                                          h.drop(columns=['named_variance']), check_exact=True)
            selected = f.own_feature_ready & f.named_ewma_eligible
            x = (f.own_har_5.to_numpy()-np.log(f.named_ewma.to_numpy())-calibration['feature_mean'])/calibration['feature_scale']
            mean = (f.named_ewma.to_numpy()*np.exp(.15*x-calibration['weak_log_normalizer'])
                    if condition == 'weak' else f.named_ewma.to_numpy())
            np.testing.assert_allclose(f.loc[selected, 'teacher_mean'], mean[selected], rtol=1e-13)
            innovation = np.random.default_rng(seed).choice(np.asarray(calibration['innovation_pool']), len(f))
            np.testing.assert_array_equal(innovation, f.teacher_innovation)
            valid = selected & f.named_eligible
            np.testing.assert_allclose(f.loc[valid, 'named_variance'], mean[valid]*innovation[valid], rtol=1e-13)
            cases[(condition, seed)] = f.set_index(f.date.dt.strftime('%Y-%m-%d'))
    native = {}
    for path in (root/'ledger/completed').glob('*.json'):
        r = read(path); s = r['specification']; recipe = s['recipe']
        if ('synthetic' not in s['role'] or s['kind'] != 'synthetic_risk_weak_teacher'
                or r['identity'] != ready['identity']
                or fingerprint({'identity': r['identity'], 'specification': s}) != r['experiment_id']):
            raise ValueError('Native synthetic identity differs')
        for name, expected in r['files'].items():
            if sha(root/'ledger'/name) != expected:
                raise ValueError('Native payload checksum differs')
        key = (recipe['synthetic_condition'], recipe['synthetic_seed'])
        case = cases[key]; test = case.loc[s['test_dates']]
        model = json.loads((root/'ledger'/r['payload_root']/'variance-model.json').read_text())
        x = test[['own_har_1', 'own_har_5', 'own_har_22']].to_numpy()-np.log(test.named_ewma.to_numpy())[:, None]
        prediction = test.named_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(test)), x])@model['coef'])
        np.testing.assert_allclose(prediction, r['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
        native[key+(s['role'], tuple(s['train_dates']), tuple(s['test_dates']))] = r
    if len(native) != 1044:
        raise ValueError('Native coverage differs')
    report = read_predictions(root/'reports/paired-gains.csv')
    if len(report) != 6*2006:
        raise ValueError('Ordinal output coverage differs')
    scalars, intervals, selections = 0, 0, 0
    for item in result['cases']:
        key = item['condition'], item['seed']; case = cases[key]
        part = report.loc[(report.condition == key[0]) & (report.seed == key[1])].copy()
        if (len(part) != 2006 or part.date.duplicated().any() or int(part.eligible.sum()) != 1989
                or not np.array_equal(part.target_date.to_numpy(),
                                      case.loc[part.date].target_date_1.dt.strftime('%Y-%m-%d').to_numpy())):
            raise ValueError('Origin/target/cohort mismatch')
        for fold in design['folds']:
            predictions = {}
            for job in fold['jobs']:
                if not (case.loc[job['train_dates']].target_date_5 < pd.Timestamp(job['cutoff'])).all():
                    raise ValueError('Training label is not mature')
                role = f'synthetic-{key[0]}-{key[1]}-{job["role"]}-{job["bucket"]}'
                r = native[key+(role, tuple(job['train_dates']), tuple(job['origins']))]
                predictions.update(zip(job['origins'], r['result']['variance_predictions'], strict=True))
            scores = []
            for weight in (0., .25, .5, .75, 1.):
                losses = []
                for dates in fold['inner']:
                    rows = case.loc[dates]; p = weight*np.array([predictions[d] for d in dates])+(1-weight)*rows.named_ewma.to_numpy()
                    losses.append(float(np.mean(np.log(p)+rows.named_variance.to_numpy()/p)))
                scores.append((float(np.mean(losses)), weight))
            chosen = min(scores)[1]
            if chosen != item['yearly_weights'][str(fold['year'])]:
                raise ValueError('Past-only mixture selection differs')
            rows = part.loc[part.year == fold['year']]
            np.testing.assert_allclose(rows.selected, chosen*rows.raw+(1-chosen)*rows.baseline, rtol=1e-13)
            selections += 1
        valid = part.eligible
        for arm in ('raw', 'selected', 'teacher_mean'):
            expected, realized = [], []
            with localcontext() as context:
                context.prec = 34
                for row in part.loc[valid].itertuples():
                    baseline, prediction, mean, actual = (Decimal(str(v)) for v in
                                                         [row.baseline, getattr(row, arm), row.teacher_mean, row.named_actual])
                    log_ratio = (baseline/prediction).ln()
                    expected.append(float(log_ratio+mean/baseline-mean/prediction))
                    realized.append(float(log_ratio+actual/baseline-actual/prediction))
            for kind, values in [('expected', expected), ('realized', realized)]:
                column = arm+'_'+kind+'_gain'
                np.testing.assert_allclose(part.loc[valid, column], values, rtol=1e-9, atol=1e-12)
                metric = item['metrics'][arm]
                if not math.isclose(float(np.mean(values)), metric[kind+'_gain'], abs_tol=1e-11):
                    raise ValueError('Mean gain differs')
                d = part[['year']].copy(); d['gain'] = 0.; d.loc[valid, 'gain'] = values; d['weight'] = valid.astype(int)
                blocks = [g[['gain', 'weight']].to_numpy() for _, g in d.groupby('year')]
                for block in (20, 60):
                    np.testing.assert_allclose(difference_interval(blocks, block), metric['gain_ci'][kind][str(block)], rtol=0, atol=1e-10)
                    intervals += 1
                scalars += len(values)
    weak = [item for item in result['cases'] if item['condition'] == 'weak']
    passed = (all(item['metrics']['raw']['expected_gain'] > 0 for item in weak)
              and np.mean([item['metrics']['raw']['expected_gain']/item['metrics']['teacher_mean']['expected_gain']
                           for item in weak]) >= .5)
    if bool(passed) != result['recovery_passed']:
        raise ValueError('Recovery decision differs')
    return {'status': 'verified', 'new_fits': 0, 'native_synthetic_models': len(native),
            'independent_loss_scalars': scalars, 'independent_intervals': intervals,
            'past_selections': selections, 'recovery_passed': bool(passed), 'market_skill': False,
            'result_sha256': sha(root/'reports/result.json'), 'csv_sha256': sha(root/'reports/paired-gains.csv'),
            'verifier_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); receipt = verify(args.root)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
