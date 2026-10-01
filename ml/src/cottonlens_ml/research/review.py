"""Read saved predictions and source evidence without fitting or changing experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import mature, rows_at, split_manifest

VERSION = 'research-review-v1'


def verify_history(root):
    root = Path(root)
    ready = read_record(root / 'ready.json')
    if digest(root / 'history.parquet') != ready['history_sha256']:
        raise ValueError('Historical snapshot checksum mismatch')
    history = pd.read_parquet(root / 'history.parquet')
    if (history.date >= pd.Timestamp('2024-01-01')).any():
        raise ValueError('Seen audit must not enter the research review')
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique sorted history required')
    return ready, history


def training_rows(history, cutoff, recipe, coverage_start):
    rows = mature(history, cutoff, recipe.get('years'))
    if coverage_start:
        rows = rows.loc[rows.date >= pd.Timestamp(coverage_start)]
    # Match Experiment.train_rows, including the common sequence warmup.
    return rows.loc[rows.date >= history.date.iloc[119]].copy()


def prediction_rows(history, record, fold, coverage_start=None):
    if record['origins'] != fold['origins'] or record['fold'] != fold['fold']:
        raise ValueError('Saved predictions do not match the frozen fold origins')
    recipe = record['recipe']
    h = recipe['horizon']
    rows = rows_at(history, record['origins']).copy()
    predicted = np.asarray(record['predictions'], dtype=float)
    if predicted.shape != (len(rows),) or not np.isfinite(predicted).all():
        raise ValueError('Nonfinite or misaligned saved predictions')
    if (rows[f'target_date_{h}'] >= pd.Timestamp('2024-01-01')).any():
        raise ValueError('Audit labels cannot enter research metrics')
    rows['predicted_return'] = predicted
    rows['median_return'] = np.nan
    rows['past_majority_sign'] = np.nan
    rows['volatility_group'] = 'unknown'
    rows['price_group'] = 'unknown'
    rows['movement_group'] = 'unknown'
    cadence = recipe.get('cadence', 126)
    if not isinstance(cadence, int) or cadence < 1:
        raise ValueError('Positive recorded-observation cadence required')
    start = int(rows.cotton_session_index.iloc[0])
    buckets = (rows.cotton_session_index - start) // cadence
    dates = history.set_index('cotton_session_index').date
    for bucket in sorted(buckets.unique()):
        selected = buckets.eq(bucket)
        cutoff = rows.date.min() if cadence == 126 else dates.loc[start + int(bucket) * cadence]
        train = training_rows(history, cutoff, recipe, coverage_start)
        if train.empty:
            raise ValueError('No mature baseline training labels')
        target = train[f'target_return_{h}'].to_numpy()
        signs, counts = np.unique(np.sign(target), return_counts=True)
        rows.loc[selected, 'median_return'] = float(np.median(target))
        # Deterministic ties: np.unique orders down, flat, up. Never use test frequency.
        rows.loc[selected, 'past_majority_sign'] = float(signs[np.argmax(counts)])
        for column, name in [('cotton_volatility_20', 'volatility_group'),
                             ('cotton_close', 'price_group')]:
            cuts = train[column].dropna().quantile([1 / 3, 2 / 3]).to_numpy()
            values = rows.loc[selected, column]
            if np.isfinite(cuts).all():
                labels = np.asarray(['low', 'middle', 'high'])[np.searchsorted(cuts, values)]
                rows.loc[selected, name] = np.where(values.notna(), labels, 'unknown')
        threshold = float(np.quantile(np.abs(target), .9))
        rows.loc[selected, 'movement_group'] = np.where(
            rows.loc[selected, f'target_return_{h}'].abs() > threshold, 'large', 'ordinary')
    actual = rows[f'target_return_{h}'].to_numpy()
    prices = rows.cotton_close.to_numpy()
    evaluate(prices, actual, predicted)  # Validate exponential domain before aggregating.
    rows['actual_return'] = actual
    for key, values in [('model', predicted), ('naive', np.zeros(len(rows))),
                        ('median', rows.median_return.to_numpy())]:
        rows[f'{key}_error'] = prices * np.abs(np.exp(actual) - np.exp(values))
        rows[f'{key}_direction_hit'] = np.sign(actual) == np.sign(values)
    rows['majority_direction_hit'] = np.sign(actual) == rows.past_majority_sign
    rows['signed_price_error'] = prices * (np.exp(predicted) - np.exp(actual))
    rows['missing_features'] = rows[recipe['features']].isna().any(axis=1).map(
        {True: 'missing', False: 'complete'})
    rows['year'] = rows.date.dt.year.astype(str)
    rows['fold'] = fold['fold']
    return rows


def summary(rows):
    metrics = evaluate(rows.cotton_close, rows.actual_return, rows.predicted_return)
    naive = float(rows.naive_error.mean())
    result = {'count': len(rows), 'model_mae': float(rows.model_error.mean()),
              'naive_mae': naive, 'median_return_mae': float(rows.median_error.mean()),
              'mae_gain_pct': float(100 * (1 - rows.model_error.mean() / naive)) if naive else None,
              'direction_pct': metrics['directional_accuracy'],
              'median_direction_pct': float(rows.median_direction_hit.mean() * 100),
              'past_majority_direction_pct': float(rows.majority_direction_hit.mean() * 100),
              'balanced_accuracy_pct': metrics['balanced_accuracy'],
              'predicted_up_pct': metrics['predicted_up_pct'],
              'predicted_flat_pct': metrics['predicted_flat_pct'],
              'predicted_down_pct': float((rows.predicted_return < 0).mean() * 100),
              'actual_up_pct': metrics['actual_up_pct'],
              'actual_flat_pct': float((rows.actual_return == 0).mean() * 100),
              'predicted_log_return_std': float(rows.predicted_return.std(ddof=0)),
              'actual_log_return_std': float(rows.actual_return.std(ddof=0)),
              'signed_price_error_mean': float(rows.signed_price_error.mean()),
              'unique_predictions': int(rows.predicted_return.nunique()),
              'unique_predictions_rounded_1e6': int(rows.predicted_return.round(6).nunique())}
    result['prediction_actual_correlation'] = (
        float(rows.predicted_return.corr(rows.actual_return))
        if result['predicted_log_return_std'] > 0 and result['actual_log_return_std'] > 0 else None)
    return result


def review_predictions(root, ready, history):
    """Read outer records only. No global ledger/payload scan or winner selection."""
    candidates, hashes = [], {}
    folds = {f['fold']: f for f in ready['identity']['split']['folds']}
    coverage = ready['identity']['split'].get('coverage_start')
    for namespace in sorted((Path(root) / 'outer').iterdir()):
        if not namespace.is_dir():
            continue
        collected = {}
        for path in sorted(namespace.glob('*.json')):
            record = read_record(path)
            recipe = record['recipe']
            if recipe.get('task', 'price') != 'price':
                continue
            fold = folds.get(record['fold'])
            if fold is None:
                raise ValueError('Unexpected frozen fold')
            key = recipe['family'], recipe['horizon']
            rows = prediction_rows(history, record, fold, coverage)
            bucket = collected.setdefault(key, {})
            if record['fold'] in bucket:
                raise ValueError('Duplicate fold for a candidate')
            bucket[record['fold']] = rows
            hashes[path.relative_to(root).as_posix()] = digest(path)
        for (family, horizon), blocks in sorted(collected.items()):
            rows = pd.concat([blocks[k] for k in sorted(blocks)], ignore_index=True)
            strata = {name: {str(label): summary(part) for label, part in rows.groupby(name)}
                      for name in ('year', 'volatility_group', 'price_group', 'movement_group', 'missing_features')}
            candidates.append({'namespace': namespace.name, 'family': family, 'horizon': horizon,
                'complete_frozen_folds': set(blocks) == set(folds), 'fold_count': len(blocks),
                'fold_wins': int(sum(r.model_error.mean() < r.naive_error.mean() for r in blocks.values())),
                'overall': summary(rows), 'strata': strata,
                'origins_id': content_id(rows.date.dt.strftime('%Y-%m-%d').tolist())})
    if not candidates:
        raise ValueError('No saved price predictions to review')
    return {'candidates': candidates, 'verified_outer_files': hashes,
            'all_model_payloads_verified': False,
            'training_curves': 'not loaded; outer records do not contain train/validation curves',
            'regime_thresholds': 'past mature training labels/features at the applicable refit cutoff',
            'movement_strata': 'realized-move diagnostics only; never available predictor or origin filter',
            'baseline_policy': 'train median and train majority at each model refit; no test-frequency baseline',
            'winner_selected': False}


def source_inventory(repo, readiness, history):
    """Bind the readiness audit to its real evidence; calendar coverage is not admission."""
    repo = Path(repo)
    body = read_record(Path(readiness))
    evidence = {}
    for name, item in body['evidence'].items():
        if not safe_member(item['path']) or digest(repo / item['path']) != item['sha256']:
            raise ValueError(f'Source evidence checksum mismatch: {name}')
        evidence[name] = json.loads((repo / item['path']).read_text(encoding='utf-8'))
    inventory = []
    for source in body['sources']:
        item = dict(source)
        item['eligible_fold_years'] = []
        item['eligible_origin_count'] = 0
        item['blocker'] = None if source.get('model_eligible') else source['status']
        item['content_calendar_potential'] = None
        if source['source'] == 'AMS Cotton Spot Quotations':
            manifest = evidence['ams_content']
            start, end = manifest['first_report_date'], manifest['last_report_date']
            hypothetical = history.loc[(history.date >= start) & (history.date <= end)]
            try:
                split = split_manifest(hypothetical, earliest=start)
                years = [fold['year'] for fold in split['folds']]
            except ValueError:
                years = []
            item['content_calendar_potential'] = {'first': start, 'last': end,
                'fold_years': years, 'outer_origins': 126 * len(years),
                'assumption': 'content-calendar bound only; publication coverage is unverified',
                'model_eligible': False}
        elif source['source'] == 'FAS Export Sales':
            dates = sorted({r['pdf_report_period_end'] for r in evidence['fas_recovery']['responses']
                            if r.get('numeric_reconciliation_allowed')})
            item['reconciled_historical_periods'] = dates
            item['reconciliation_is_full_vintage_coverage'] = False
        inventory.append(item)
    # Existing research is allowed by its own frozen protocol. It is not a newly admitted source.
    return {'audit_sha256': digest(Path(readiness)), 'sources': inventory,
            'newly_admitted_external_groups': body['newly_admitted_external_groups'],
            'pilot_status': 'blocked_pending_verified_publication_package',
            'pilot_blocker': 'AMS/FAS require per-version historical availability evidence and a frozen common cohort',
            'published_calendar_is_publication_proof': False}


def sample_learning_curve(root, ready, history):
    """One predeclared past-validation sample; direct identity lookup, never a ledger scan."""
    root = Path(root)
    fold = ready['identity']['split']['folds'][0]
    namespace = 'ablate-expanded_availability'
    decision_path = root / 'decisions' / namespace / f'xgboost-t5-fold{fold["fold"]}.json'
    decision = read_record(decision_path)
    spec = next(seed['recipe'] for seed in decision['chosen']['seeds'] if seed['recipe']['seed'] == 42)
    validation = rows_at(history, fold['inner'][0]['origins'])
    coverage = ready['identity']['split'].get('coverage_start')
    train = training_rows(history, validation.date.min(), spec, coverage)
    early_validation = train.tail(63)
    early_train = training_rows(history, early_validation.date.min(), spec, coverage)
    dates = early_validation.date.dt.strftime('%Y-%m-%d').tolist()
    # Reconstruct the frozen historical Experiment.fit identity, not a new recipe.
    specification = {'recipe': spec, 'role': f'early-stop-{fold["fold"]}-0', 'iterations': None,
        'train_dates': early_train.date.dt.strftime('%Y-%m-%d').tolist(),
        'validation_dates': dates, 'test_dates': dates,
        'train_identity': frame_identity(early_train, list(early_train.columns)), 'repeat_reason': None}
    key = content_id({'identity': ready['identity'], 'specification': specification})
    path = root / 'ledger/completed' / (key + '.json')
    fit = read_record(path)
    if fit['specification'] != specification or fit['identity'] != ready['identity']:
        raise ValueError('Learning-curve sample identity mismatch')
    names = [name for name in fit['files'] if name.endswith('/curves.json')]
    if len(names) != 1 or not safe_member(names[0]):
        raise ValueError('One checksummed learning-curve payload required')
    curves_path = root / 'ledger' / names[0]
    if digest(curves_path) != fit['files'][names[0]]:
        raise ValueError('Learning-curve checksum mismatch')
    return {'scope': 'one predeclared sample; not all candidates or evidence of generalization',
        'namespace': namespace, 'fold': fold['fold'], 'seed': 42, 'inner_block': 1,
        'receipt_sha256': digest(path), 'curve_sha256': digest(curves_path),
        'iterations_selected': fit['result']['iterations'],
        'details': fit['result']['details'], 'training_metrics': fit['result']['training_metrics'],
        'validation_metrics': fit['result']['metrics'],
        'curves': json.loads(curves_path.read_text(encoding='utf-8')),
        'validation_used_for_selection': True, 'model_payload_verified': False}


def build_review(root, repo, readiness, *, with_curve_sample=False):
    ready, history = verify_history(root)
    return {'version': VERSION, 'experiment': Path(root).name,
        'evidence': 'seen_historical_research_diagnostic_not_independent_holdout',
        'source_id': ready['identity']['source_id'],
        'ready_sha256': digest(Path(root) / 'ready.json'),
        'history_sha256': ready['history_sha256'], 'reviewer_sha256': digest(Path(__file__)),
        'research_rows': len(history), 'audit_used': False, 'new_fits': 0,
        'predictions': review_predictions(root, ready, history),
        'learning_curve_sample': sample_learning_curve(root, ready, history) if with_curve_sample else None,
        'source_inventory': source_inventory(repo, readiness, history),
        'large_source_return_dates': history.loc[history.cotton_ret_1.abs() > .2, 'date'].dt.strftime('%Y-%m-%d').tolist(),
        'contract_roll_interpretation': 'unknown; flagged dates retained in every score',
        'next_action': 'Resolve a bounded AMS/FAS publication-version package; do not rerun completed ablation'}


def write_review(body, output):
    output = Path(output)
    ident = content_id(body)
    folder = output / ('research-review-' + ident[:16])
    freeze_record(folder / 'review.json', body)
    lines = ['CottonLens saved-prediction review', f'Evidence: {body["evidence"]}',
             'New fits: 0. No winner selected. Model payloads were not globally verified.', '',
             'Group | model | horizon | MAE gain % | direction % | past majority % | fold wins']
    for candidate in body['predictions']['candidates']:
        s = candidate['overall']
        gain_text = 'n/a' if s['mae_gain_pct'] is None else f'{s["mae_gain_pct"]:+.3f}'
        lines.append(f'{candidate["namespace"]} | {candidate["family"]} | T+{candidate["horizon"]} | '
            f'{gain_text} | {s["direction_pct"]:.2f} | '
            f'{s["past_majority_direction_pct"]:.2f} | {candidate["fold_wins"]}/{candidate["fold_count"]}')
    lines.extend(['', 'Source pilot: ' + body['source_inventory']['pilot_status'],
                  body['source_inventory']['pilot_blocker'],
                  'Term: MAE is the average absolute error in the original price unit.'])
    text = '\n'.join(lines) + '\n'
    path = folder / 'review.txt'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError('Conflicting frozen review text')
    if not path.exists():
        path.write_text(text, encoding='utf-8')
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--evidence-root', '--repo', dest='repo', type=Path, required=True)
    parser.add_argument('--readiness', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-curve-sample', action='store_true',
                        help='Load one selected fit receipt and curve; may hydrate Drive files')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.experiment_root.resolve()):
        raise ValueError('Review output must be outside the preserved experiment')
    folder = write_review(build_review(args.experiment_root, args.repo, args.readiness,
                                      with_curve_sample=args.with_curve_sample), args.output)
    print((folder / 'review.txt').read_text(encoding='utf-8'))
    print('Detailed report:', folder / 'review.json')


if __name__ == '__main__':
    main()
