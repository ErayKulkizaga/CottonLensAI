"""Independent no-fit replay of native five-parameter forecasts, clocks, losses and selection."""

import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

from research.verify_risk_weak_teacher import difference_interval, read, sha


def verify(root):
    ready, design, report = (read(root/p) for p in ('ready.json', 'decision-contract.json', 'reports/result.json'))
    for name, expected in ready['frozen_hashes'].items():
        if sha(root/name) != expected:
            raise ValueError('Changed frozen input')
    h = pd.read_parquet(root/'reference/inputs/history.parquet').set_index('date')
    raw = pd.read_parquet(root/'inputs/market.parquet')
    quotes = raw.loc[raw.series.eq('dxy')].sort_values('date').set_index('date').close
    # Independent raw-calendar return computation and merge-asof alignment.
    logp = np.log(quotes.where(np.isfinite(quotes) & quotes.gt(0)))
    variance = logp.diff().pow(2).rolling(5, min_periods=5).mean()
    src = pd.DataFrame({'source': variance.index, 'rv': variance.to_numpy()})
    src['clock'] = pd.to_datetime(src.source, utc=True)+pd.Timedelta(days=1)
    decisions = pd.DataFrame({'date': h.index, 'clock': pd.to_datetime(h.index, utc=True)+pd.Timedelta(days=1, minutes=15)})
    aligned = pd.merge_asof(decisions, src, on='clock', direction='backward').set_index('date')
    states = pd.read_parquet(root/'inputs/dxy-state.parquet').set_index('date')
    pd.testing.assert_series_equal(states.dxy_state_source_date, aligned.source, check_names=False)
    np.testing.assert_allclose(states.dxy_rv5, aligned.rv, rtol=1e-10, atol=1e-17, equal_nan=True)
    models, losses, selection_checks, ci_checks, scalar_predictions = 0, 0, 0, 0, 0
    forecasts = {}
    native_by_key = {p.stem: read(p) for p in (root/'ledger/completed').glob('*.json')}
    if len(native_by_key) != 348 or list((root/'ledger/attempts').glob('*.json')):
        raise ValueError('Incomplete or failed market fit set')
    for rec in native_by_key.values():
        arm = rec['specification']['recipe']['arm']
        tr = h.loc[pd.to_datetime(rec['specification']['train_dates'])]
        te = h.loc[pd.to_datetime(rec['specification']['test_dates'])]
        job = next(j for f in design['folds'] for j in f['jobs']
                   if j['train_dates'] == rec['specification']['train_dates']
                   and j['origins'] == rec['specification']['test_dates'])
        if (not (tr.target_date_5 < pd.Timestamp(job['cutoff'])).all()
                or rec['identity'] != ready['identity']):
            raise ValueError('Maturity/identity violation')
        for name, checksum in rec['files'].items():
            if sha(root/'ledger'/name) != checksum:
                raise ValueError('Native payload mismatch')
        m = read(root/'ledger'/rec['payload_root']/'variance-model.json')
        if m['effective_parameters'] != 5 or m['recipe'] != design['recipes'][arm]:
            raise ValueError('Native recipe mismatch')

        def x(rows, arm=arm):
            logb = np.log(rows.named_ewma.to_numpy())
            common = rows[['own_har_1', 'own_har_5', 'own_har_22']].to_numpy()-logb[:, None]
            extra = logb if arm == 'own_level' else np.log(np.maximum(aligned.loc[rows.index].rv.to_numpy(), 1e-12))
            return np.column_stack([common, extra])

        train_x = x(tr); o = m['optimization']
        np.testing.assert_allclose(train_x.mean(axis=0), o['training_feature_mean'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(train_x.std(axis=0), o['training_feature_scale'], rtol=0, atol=1e-12)
        eta = m['coef'][0]+train_x@np.array(m['coef'][1:])
        ratio = (tr.named_variance/tr.named_ewma).to_numpy()*np.exp(-eta)
        z = (train_x-o['training_feature_mean'])/o['training_feature_scale']
        gradient = np.column_stack([np.ones(len(tr)), z]).T@(1-ratio)/len(tr)
        if not o['success'] or np.max(np.abs(gradient)) > 1.001e-6:
            raise ValueError('Uncertified optimizer')
        assert math.isclose(float(np.mean(eta+ratio)), o['final_training_qlike'], abs_tol=1e-10)
        predicted = te.named_ewma.to_numpy()*np.exp(m['coef'][0]+x(te)@np.array(m['coef'][1:]))
        np.testing.assert_allclose(predicted, rec['result']['variance_predictions'], rtol=1e-10, atol=1e-14)
        if rec['result']['origins'] != rec['specification']['test_dates']:
            raise ValueError('Prediction origin mismatch')
        for day, value in zip(rec['result']['origins'], predicted, strict=True):
            if (arm, job['role'], day) in forecasts:
                raise ValueError('Duplicate origin in role')
            forecasts[arm, job['role'], day] = value
        scalar_predictions += len(predicted); models += 1
    frame = pd.read_csv(root/'reports/paired-losses.csv', keep_default_na=False, na_values=[''], float_precision='round_trip')
    oldframe = pd.read_csv(root/'reference/reports/paired-losses.csv', keep_default_na=False, na_values=[''], float_precision='round_trip')
    for name in ('date', 'target_date', 'contract', 'eligible', 'named_actual', 'baseline', 'raw', 'selected',
                 'baseline_loss', 'raw_loss', 'selected_loss'):
        pd.testing.assert_series_equal(frame[name], oldframe[name], check_names=False, check_exact=True)
    if len(frame) != 2006 or frame.date.duplicated().any() or int(frame.eligible.sum()) != 1989:
        raise ValueError('Cohort incomplete')
    ordered = [day for f in design['folds'] for day in f['origins']]
    if frame.date.tolist() != ordered:
        raise ValueError('Ordinal origin mismatch')
    for fold in design['folds']:
        for arm in ('own_level', 'dxy_state'):
            scores = []
            for weight in (0., .25, .5, .75, 1.):
                parts = []
                for i, days in enumerate(fold['inner']):
                    rows = h.loc[pd.to_datetime(days)]
                    p = np.array([forecasts[arm, f'inner_{i}', day] for day in days])
                    mix = weight*p+(1-weight)*rows.named_ewma.to_numpy()
                    parts.append(float(np.mean(np.log(mix)+rows.named_variance.to_numpy()/mix)))
                scores.append((float(np.mean(parts)), weight))
            loss, weight = min(scores)
            saved = report['selections'][arm][str(fold['year'])]
            if saved['weight'] != weight or not math.isclose(saved['mean_inner_qlike'], loss, abs_tol=1e-10):
                raise ValueError('Past-only selection differs')
            subset = frame.loc[frame.year.eq(fold['year']) & frame.forecast_available]
            expected = np.array([forecasts[arm, 'outer', d] for d in subset.date])
            np.testing.assert_allclose(expected, subset[f'{arm}_raw'], rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(weight*expected+(1-weight)*subset.baseline, subset[f'{arm}_selected'], rtol=1e-10, atol=1e-14)
            selection_checks += 1
    for row in frame.itertuples():
        origin = h.loc[pd.Timestamp(row.date)]
        if row.target_date != origin.target_date_1.strftime('%Y-%m-%d') or row.eligible != bool(origin.named_eligible and origin.named_ewma_eligible):
            raise ValueError('Target-date/eligibility mismatch')
        if row.eligible:
            assert row.named_actual == origin.named_variance and row.baseline == origin.named_ewma
            for name in ('baseline', 'raw', 'selected', 'own_level_raw', 'own_level_selected', 'dxy_state_raw', 'dxy_state_selected'):
                with localcontext() as c:
                    c.prec = 50
                    actual, p = Decimal(str(row.named_actual)), Decimal(str(getattr(row, name)))
                    expected = float(p.ln()+actual/p)
                assert math.isclose(expected, getattr(row, name+'_loss'), abs_tol=1e-11)
                losses += 1
    for key, metric in report['comparisons'].items():
        if key.startswith('dxy_raw_vs'):
            a, b = 'own_level_raw_loss', 'dxy_state_raw_loss'
        elif key.startswith('dxy_selected_vs'):
            a, b = 'own_level_selected_loss', 'dxy_state_selected_loss'
        else:
            model, control = key.split('_vs_'); a, b = control+'_loss', model+'_loss'
        clean = frame.loc[frame.eligible]; delta = frame[a]-frame[b]
        assert math.isclose(float(delta[frame.eligible].mean()), metric['gain'], abs_tol=1e-11)
        blocks = [np.column_stack([np.where(g.eligible, delta.loc[g.index].fillna(0), 0), g.eligible.astype(int)])
                  for _, g in frame.groupby('year')]
        for k in (20, 60):
            np.testing.assert_allclose(difference_interval(blocks, k), metric['gain_ci'][str(k)], rtol=0, atol=1e-10)
            ci_checks += 1
        trim = clean.drop(clean.nlargest(math.ceil(len(clean)*.01), 'baseline_loss').index)
        assert math.isclose(float((trim[a]-trim[b]).mean()), metric['trim_top1pct_gain'], abs_tol=1e-11)
        yearly = {str(y): float((g[a]-g[b]).mean()) for y, g in clean.groupby('year')}
        np.testing.assert_allclose(list(yearly.values()), list(metric['yearly_gain'].values()), atol=1e-11, rtol=0)
    smoke_paths = list((root/'learning/ledger/completed').glob('*.json'))
    if len(smoke_paths) != 1 or list((root/'learning/ledger/attempts').glob('*.json')):
        raise ValueError('Invalid synthetic control completion')
    smoke = read(smoke_paths[0])
    for name, checksum in smoke['files'].items():
        if sha(root/'learning/ledger'/name) != checksum:
            raise ValueError('Changed smoke payload')
    sx = np.random.default_rng(42).uniform(-2, 2, (1200, 4))
    sb = np.exp(-8+sx[:, 3]); sx[:, 3] = np.log(sb)
    expected = sb*np.exp(.1+sx@np.array([.2, .1, -.1, .08]))
    np.testing.assert_allclose(smoke['result']['variance_predictions'], expected[1000:], rtol=1e-5)
    return {'status': 'independently_verified', 'new_fits': 0, 'new_native_models': models,
            'synthetic_control_native_checks': 1, 'unchanged_reference_forecast_rows': len(frame),
            'prediction_scalar_checks': scalar_predictions, 'decimal_loss_scalar_checks': losses,
            'past_selection_checks': selection_checks, 'independent_intervals': ci_checks,
            'report_sha256': sha(root/'reports/result.json'), 'csv_sha256': sha(root/'reports/paired-losses.csv'),
            'verifier_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.root)
    if a.output.exists():
        raise ValueError('Preserve old independent receipt')
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
