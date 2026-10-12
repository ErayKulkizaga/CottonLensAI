"""Read-only audit of frozen variance forecasts; never selects or fits models."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.full_year import chunks, volatility_recipe
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import full_year_manifest, mature, rows_at
from cottonlens_ml.research.statistics import paired_bootstrap
from cottonlens_ml.research.volatility import FLOOR, features, qlike

YEARS = tuple(range(2016, 2024))
BLOCKS = (20, 60)


def assert_close(actual, expected, name, *, atol=1e-12, rtol=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
        raise ValueError(f'{name} mismatch')


def reconstruct(history):
    """Prices are the source, not stored target columns or future outcomes."""
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique ordered history required')
    if history.date.max() >= pd.Timestamp('2024-01-01'):
        raise ValueError('Reviewed 2024+ cannot enter this audit')
    prices = history.cotton_close
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Positive finite prices required')
    result = features(history)
    for name in ('daily_variance_proxy', 'har_1', 'har_5', 'har_22',
                 'variance_target_1', 'variance_target_5'):
        assert_close(history[name], result[name], name)
    for h in (1, 5):
        expected_date = history.date.shift(-h)
        in_snapshot = expected_date.notna()
        if not history.loc[in_snapshot, f'target_date_{h}'].equals(expected_date.loc[in_snapshot]):
            raise ValueError('Target date mismatch')
        tail_dates = history.loc[~in_snapshot, f'target_date_{h}'].dropna()
        if (tail_dates <= history.date.max()).any():
            raise ValueError('Out-of-snapshot target date mismatch')
        target_return = np.log(prices.shift(-h) / prices)
        assert_close(history.loc[in_snapshot, f'target_return_{h}'],
                     target_return.loc[in_snapshot], 'Target return')
        if f'gk_target_{h}' in history:
            assert_close(history[f'gk_target_{h}'], result[f'gk_target_{h}'], 'OHLC target')
    # Explicit recursion independent of pandas EWM and existing ewma().
    observed = result.daily_variance_proxy.to_numpy()
    variance = np.full(len(history), np.nan)
    last = None
    for i, value in enumerate(observed):
        if np.isfinite(value):
            last = value if last is None else .94 * last + .06 * value
        if last is not None:
            variance[i] = max(last, FLOOR)
    result['audit_ewma_1'] = variance
    return result


def align_output(history, original, corrected, fold, h):
    """Identity first: no intersection, imputation or changed evaluated population."""
    for record in (original, corrected):
        if record['year'] != fold['year'] or record['horizon'] != h:
            raise ValueError('Year/horizon mismatch')
        if [r['date'] for r in record['records']] != fold['origins']:
            raise ValueError('Origin mismatch')
    preserved = ('date', 'cotton_close', f'target_return_{h}', f'variance_target_{h}',
                 'predicted_return', 'variance', 'ewma_variance')
    for a, b in zip(original['records'], corrected['records'], strict=True):
        if any(a[k] != b[k] for k in preserved):
            raise ValueError('Original price/variance predictions or targets changed')
    frame = pd.DataFrame(corrected['records'])
    reference = rows_at(history, fold['origins'])
    for name in ('cotton_close', f'target_return_{h}', f'variance_target_{h}'):
        assert_close(frame[name], reference[name], name)
    assert_close(frame.ewma_variance, np.maximum(reference.audit_ewma_1 * h, FLOOR), 'EWMA')
    frame['target_date'] = reference[f'target_date_{h}'].dt.strftime('%Y-%m-%d')
    frame['year'], frame['horizon'] = fold['year'], h
    return frame


def verify_fit(root, receipt, history, identity):
    """Check canonical receipt, training identity, maturity and native bytes."""
    spec = receipt['specification']
    if receipt['identity'] != identity or receipt['experiment_id'] != content_id({
            'identity': identity, 'specification': spec}):
        raise ValueError('Fit identity mismatch')
    recipe = spec['recipe']
    if recipe != volatility_recipe(recipe['family'], recipe['horizon']):
        raise ValueError('Variance recipe mismatch')
    test = rows_at(history, spec['test_dates'])
    if len(test) == 0 or spec['validation_dates']:
        raise ValueError('Invalid variance fit test/validation dates')
    train = mature(history, test.date.min(), recipe.get('years'))
    if identity['split'].get('coverage_start'):
        train = train.loc[train.date >= pd.Timestamp(identity['split']['coverage_start'])]
    train = train.loc[train.date >= history.date.iloc[119]]
    if train.date.dt.strftime('%Y-%m-%d').tolist() != spec['train_dates']:
        raise ValueError('Training dates/maturity mismatch')
    # Frame identity uses the exact original snapshot, not added audit columns.
    if frame_identity(train, list(train.columns)) != spec['train_identity']:
        raise ValueError('Training frame identity mismatch')
    if receipt['result']['origins'] != spec['test_dates']:
        raise ValueError('Native prediction origins mismatch')
    if not receipt.get('files'):
        raise ValueError('Missing native payload')
    for name, checksum in receipt['files'].items():
        if not safe_member(name) or digest(root / 'ledger' / name) != checksum:
            raise ValueError('Native payload hash mismatch')
    models = [name for name in receipt['files'] if name.endswith('/variance-model.json')]
    if len(models) != 1:
        raise ValueError('Exactly one variance model required')
    model = json.loads((root / 'ledger' / models[0]).read_bytes())
    prediction = replay_native(model, recipe, history, train, test)
    assert_close(receipt['result']['variance_predictions'], prediction, 'Native variance',
                 atol=1e-12, rtol=1e-8)
    return test


def replay_native(model, recipe, history, train, test):
    """Inference from JSON only; no optimizer, fit function or pickle loading."""
    if recipe['family'] == 'har':
        if model['kind'] != 'log_har' or model['features'] != recipe['features']:
            raise ValueError('HAR model identity mismatch')
        x = np.column_stack([np.ones(len(test)), test[model['features']]])
        return np.maximum(np.exp(x @ np.asarray(model['coef'])) * model['smearing'], FLOOR)
    if model['kind'] != 'garch':
        raise ValueError('GARCH model identity mismatch')
    omega, alpha, beta = (model[k] for k in ('omega', 'alpha', 'beta'))
    if (not np.isfinite([omega, alpha, beta]).all() or omega <= 0
            or min(alpha, beta) < 0 or alpha + beta >= 1):
        raise ValueError('Invalid GARCH state')
    if model['last_training_origin'] != train.date.max().isoformat():
        raise ValueError('GARCH training endpoint mismatch')
    assert_close(model['last_observed_return'], train.cotton_ret_1.iloc[-1], 'GARCH endpoint return')
    variance = (omega + alpha * model['last_observed_return'] ** 2
                + beta * model['conditional_variance_at_last_origin'])
    desired = set(test.date)
    predictions = []
    for row in history.loc[(history.date > train.date.max()) & (history.date <= test.date.max())].itertuples():
        variance = omega + alpha * row.cotton_ret_1 ** 2 + beta * variance
        if row.date in desired:
            total, step = 0., variance
            for _ in range(recipe['horizon']):
                total += step
                step = omega + (alpha + beta) * step
            predictions.append(max(total, FLOOR))
    return np.asarray(predictions)


def predictions(receipts, history, recipe, origins, role):
    output = []
    ids = []
    for i, chunk in enumerate(chunks(history, origins)):
        expected = chunk.date.dt.strftime('%Y-%m-%d').tolist()
        candidates = [r for r in receipts if r['specification']['recipe'] == recipe
                      and r['specification']['role'] == f'{role}-{i}'
                      and r['specification']['test_dates'] == expected
                      and r['specification'].get('repeat_reason') is None]
        if len(candidates) != 1:
            raise ValueError('Missing/ambiguous canonical variance fit')
        output.extend(candidates[0]['result']['variance_predictions'])
        ids.append(candidates[0]['experiment_id'])
    return np.asarray(output), ids


def decision(summary):
    intervals = [summary['bootstrap'][str(b)]['difference_ci_95'] for b in BLOCKS]
    if (summary['mean_difference'] > 0 and summary['year_wins'] >= 6
            and all(ci[0] > 0 for ci in intervals)):
        return 'HISTORICAL_FOLLOWUP_CANDIDATE'
    if all(ci[1] <= 0 for ci in intervals):
        return 'FIXED_RECIPE_SUPERIORITY_NOT_SUPPORTED'
    return 'INCONCLUSIVE'


def loss_summary(frame, *, repetitions=10000):
    actual = frame['actual_variance'].to_numpy()
    baseline = qlike(actual, frame.ewma_variance)
    candidate = qlike(actual, frame.variance)
    # Independent scalar identity; avoids ratios of possibly negative QLIKE.
    delta = (np.log(frame.ewma_variance / frame.variance)
             + actual * (1 / frame.ewma_variance - 1 / frame.variance)).to_numpy()
    assert_close(delta, baseline - candidate, 'Paired QLIKE algebra')
    frame = frame.copy()
    frame['baseline_loss'], frame['candidate_loss'], frame['difference'] = baseline, candidate, delta
    folds = [g[['baseline_loss', 'candidate_loss']].to_numpy() for _, g in frame.groupby('year')]
    yearly = [{'year': int(year), 'count': len(g), 'mean_difference': float(g.difference.mean())}
              for year, g in frame.groupby('year')]
    result = {'count': len(frame), 'mean_difference': float(delta.mean()),
              'year_wins': sum(y['mean_difference'] > 0 for y in yearly), 'years': yearly,
              'bootstrap': {str(b): paired_bootstrap(folds, block=b, repetitions=repetitions, seed=42)
                            for b in BLOCKS}}
    # Stable top1% tie-break by chronological original row; never changes main result.
    count = int(np.ceil(len(frame) * .01))
    extremes = np.argsort(-baseline, kind='stable')[:count]
    retained = frame.drop(frame.index[extremes])
    result['top1pct_sensitivity'] = {'removed_count': count, 'retained_count': len(retained),
        'mean_difference': float(retained.difference.mean()),
        'removed_origins': frame.iloc[extremes].date.tolist(),
        'removed_contribution_sum': float(delta[extremes].sum()),
        'role': 'posthoc diagnostic, not primary population or selection'}
    result['decision'] = decision(result)
    return result, frame


def audit(experiment_root, corrected_root, output, *, repetitions=10000):
    root, corrected, output = map(Path, (experiment_root, corrected_root, output))
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) for p in (root, corrected)):
        raise ValueError('New output namespace outside immutable inputs required')
    started = time.monotonic()
    bound = {}

    def record(path, prefix):
        value = read_record(path)
        bound[f'{prefix}/{path.relative_to(root if prefix == "original" else corrected).as_posix()}'] = digest(path)
        return value

    ready = record(root / 'ready.json', 'original')
    corrected_ready = record(corrected / 'ready.json', 'corrected')
    if digest(root / 'history.parquet') != ready['history_sha256']:
        raise ValueError('Frozen history hash mismatch')
    bound['original/history.parquet'] = ready['history_sha256']
    history = pd.read_parquet(root / 'history.parquet')
    reconstructed = reconstruct(history)
    identity = ready['identity']
    if (identity['profile'] != 'full-year-v1'
            or identity['research_data_id'] != frame_identity(history, list(history.columns))
            or identity['split'] != full_year_manifest(history)):
        raise ValueError('Frozen source/data/split identity mismatch')
    analysis = corrected_ready['derived_analysis']
    if (analysis['version'] != 'causal-ewma-calibration-v1'
            or analysis['parent_ready_sha256'] != digest(root / 'ready.json')
            or corrected_ready['identity'] != identity):
        raise ValueError('Correction parent/policy identity mismatch')
    reports = list((corrected / 'reports').glob('pilot-decision-*.json'))
    if len(reports) != 1:
        raise ValueError('One explicitly frozen corrected report required')
    old_report = record(reports[0], 'corrected')
    if old_report['ready_sha256'] != digest(corrected / 'ready.json'):
        raise ValueError('Corrected report ready hash mismatch')
    receipts = []
    for path in sorted((root / 'ledger/completed').glob('*.json')):
        item = json.loads(path.read_bytes())
        spec = item['specification']
        if (spec['recipe'].get('task') != 'variance'
                or not spec['role'].startswith(('vol-inner-', 'vol-outer-'))
                or spec.get('repeat_reason') is not None):
            continue
        receipt = record(path, 'original')
        verify_fit(root, receipt, history, identity)
        for name, sha in receipt['files'].items():
            bound[f'original/ledger/{name}'] = sha
        receipts.append(receipt)
    result, frames, selection = {}, [], []
    for h in (1, 5):
        horizon_frames = []
        for fold in identity['split']['folds']:
            year = fold['year']
            name = f't{h}-year{year}.json'
            original = record(root / 'pilot-full-year' / name, 'original')
            derived = record(corrected / 'pilot-full-year' / name, 'corrected')
            if (analysis['parent_outputs_sha256'][name] != digest(root / 'pilot-full-year' / name)
                    or old_report['input_outputs_sha256'][name] != digest(corrected / 'pilot-full-year' / name)
                    or derived['parent_output_sha256'] != digest(root / 'pilot-full-year' / name)):
                raise ValueError('Output parent hash mismatch')
            frame = align_output(reconstructed, original, derived, fold, h)
            chosen = record(root / 'decisions/full-year' / name, 'original')
            if original['decision_id'] != content_id(chosen):
                raise ValueError('Selection decision identity mismatch')
            options = chosen['all_variance_candidates']
            if not options or len({v['recipe']['family'] for v in options}) != len(options):
                raise ValueError('Invalid candidate coverage')
            for option in options:
                scores = []
                for i, inner in enumerate(fold['inner']):
                    pred, ids = predictions(receipts, history, option['recipe'], inner['origins'],
                                            f'vol-inner-{year}-{i}')
                    actual = rows_at(reconstructed, inner['origins'])[f'variance_target_{h}']
                    scores.append(float(qlike(actual, pred).mean()))
                    if max(rows_at(history, inner['origins'])[f'target_date_{h}']) >= pd.Timestamp(fold['origins'][0]):
                        raise ValueError('Inner labels not mature before outer selection')
                    selection.append({'year': year, 'horizon': h, 'family': option['recipe']['family'],
                                      'inner_block': i, 'fit_ids': ids})
                assert_close(option['inner_score'], np.mean(scores), 'Past-only selection score')
            if chosen['variance'] != min(options, key=lambda x: x['inner_score']):
                raise ValueError('Family selection differs from historical validation')
            missing = {'har', 'garch'} - {v['recipe']['family'] for v in options}
            if missing != {v['family'] for v in chosen.get('unselectable', [])}:
                raise ValueError('Unexplained missing variance family')
            recipe = chosen['variance']['recipe']
            pred, ids = predictions(receipts, history, recipe, fold['origins'], f'vol-outer-{year}')
            assert_close(frame.variance, pred, 'Outer receipt predictions')
            if original['volatility_family'] != recipe['family']:
                raise ValueError('Outer family mismatch')
            frame['family'] = recipe['family']
            frame['actual_variance'] = frame[f'variance_target_{h}']
            horizon_frames.append(frame)
        frame = pd.concat(horizon_frames, ignore_index=True)
        if len(frame) != 2006 or tuple(sorted(frame.year.unique())) != YEARS:
            raise ValueError('Frozen eight-year/2006-origin coverage mismatch')
        summary, scored = loss_summary(frame, repetitions=repetitions)
        assert_close(summary['mean_difference'], old_report['horizons'][str(h)]['variance_mean_loss_difference'],
                     'Previous report scalar')
        gk_name = f'gk_target_{h}'
        if gk_name in frame:
            valid = frame[gk_name].notna() & frame[gk_name].ge(0)
            sub = frame.loc[valid].copy()
            assert_close(sub[gk_name], rows_at(reconstructed, sub.date)[gk_name], 'OHLC sensitivity identity')
            sub['actual_variance'] = sub[gk_name]
            sensitivity, _ = loss_summary(sub, repetitions=repetitions)
            sensitivity['role'] = 'Common-origin OHLC proxy sensitivity; not primary or intraday variance'
            sensitivity['primary_target_on_same_origins_mean_difference'] = float(scored.loc[valid, 'difference'].mean())
            summary['ohlc_sensitivity'] = sensitivity
        result[str(h)] = summary
        frames.append(scored[['date', 'target_date', 'year', 'horizon', 'family', 'actual_variance',
                              'variance', 'ewma_variance', 'baseline_loss', 'candidate_loss', 'difference']])
    # No successful publication until all checks and input immutability checks pass.
    for name, sha in bound.items():
        prefix, relative = name.split('/', 1)
        if digest((root if prefix == 'original' else corrected) / relative) != sha:
            raise ValueError('Inputs changed during audit')
    if time.monotonic() - started >= 1800:
        raise ValueError('Thirty-minute analysis budget exceeded; no completed report')
    output.mkdir(parents=True)
    csv_path = output / 'paired-losses.csv'
    pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False, lineterminator='\n')
    body = {'profile': 'risk-evidence-audit-v1', 'status': 'complete', 'new_fits': 0,
            'primary_horizon': 1, 'secondary_horizon': 5, 'horizons': result,
            'source_id': identity['source_id'], 'data_id': identity['research_data_id'],
            'split_id': identity['split']['split_id'], 'input_hashes': bound,
            'verified_variance_receipts': len(receipts), 'selection_checks': selection,
            'paired_losses_sha256': digest(csv_path), 'code_sha256': digest(Path(__file__)),
            'statistics_code_sha256': digest(Path(__import__(paired_bootstrap.__module__, fromlist=['']).__file__)),
            'decision': result['1']['decision'], 'release_allowed': False,
            'limits': ['T+1 chosen after historical inspection; reviewed history, not independent holdout',
                       'Daily squared return is a noisy proxy, not measured intraday variance',
                       'CT=F contract transitions and historical availability/vintages remain unverified',
                       'No economic, trading or production skill established'],
            'next_evidence_if_inconclusive': 'Independent prospective paired forecasts of the locked existing recipe versus EWMA; no historical retuning'}
    freeze_record(output / 'report.json', body)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment-root', 'corrected-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.experiment_root, args.corrected_root, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'new_fits', 'decision', 'verified_variance_receipts')}))


if __name__ == '__main__':
    main()
