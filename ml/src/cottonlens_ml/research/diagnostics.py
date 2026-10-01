"""Learning controls and hypothesis tests, explicitly separate from performance evidence."""
import json

import numpy as np
import pandas as pd

from cottonlens_ml.cohort import content_id
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.models import fit_predict
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.search import diagnostic_recipes


def synthetic_control(kind):
    rng = np.random.default_rng(703)
    frame = pd.DataFrame(rng.normal(size=(700, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    frame['date'] = pd.bdate_range('2010-01-01', periods=len(frame))
    frame['cotton_close'] = 80.
    frame['cotton_session_index'] = np.arange(len(frame))
    for h in (1, 5):
        frame[f'target_return_{h}'] = .02 * frame[FEATURE_NAMES[0]]
        frame[f'target_date_{h}'] = frame.date.shift(-h)
    train, validation, test = frame.iloc[:480].copy(), frame.iloc[490:580].copy(), frame.iloc[590:680].copy()
    if kind == 'negative_block_shift':
        for h in (1, 5):
            train[f'target_return_{h}'] = np.roll(train[f'target_return_{h}'].to_numpy(), 137)
    elif kind == 'small_subset_overfit':
        train = train.head(128).copy()
        for h in (1, 5):
            train[f'target_return_{h}'] = rng.normal(0, .02, len(train))
        test, validation = train.copy(), None
    elif kind != 'known_signal':
        raise ValueError('Unknown diagnostic control')
    return frame, train, validation, test


def diagnose(experiment):
    path = experiment.root / 'diagnosis.json'
    if path.exists():
        if read_record(path)['status'] != 'passed':
            raise ValueError('Failed diagnosis requires a documented bug fix and new source/experiment identity')
        return
    controls = {}
    free_profile = experiment.identity.get('profile') == 'free-data-v1'
    families = ('xgboost', 'catboost') if free_profile else ('xgboost', 'catboost', 'mlp', 'lstm', 'tcn')
    for family in families:
        # GPU compatibility smoke is intentionally small, including custom price metrics.
        for kind in (('known_signal', 'negative_block_shift', 'small_subset_overfit') if family == 'xgboost' else ('known_signal',)):
            frame, train, validation, test = synthetic_control(kind)
            spec = diagnostic_recipes(1, FEATURE_NAMES)[-1]
            spec.update(family=family, hypothesis=f'synthetic_{kind}', window=1)
            spec['params'] = {'max_depth': 8, 'eta': .15, 'min_child_weight': 1, 'alpha': 0, 'lambda': .01}
            if family == 'catboost':
                spec['params'] = {'depth': 6, 'learning_rate': .1, 'l2_leaf_reg': 1}
            elif family in ('mlp', 'lstm', 'tcn'):
                spec['params'] = {'width': 32, 'dropout': 0., 'batch_size': 64, 'learning_rate': .003}
                spec['window'] = 20 if family != 'mlp' else 1
                train = train.iloc[19:]
            def operation(workspace, frame=frame, train=train, validation=validation, test=test, spec=spec, family=family, kind=kind):
                return fit_predict(frame, train, validation, test, spec, workspace,
                                   iterations=40 if family in ('mlp', 'lstm', 'tcn') else (700 if kind == 'small_subset_overfit' else 250))
            record = experiment.ledger.run({'role': 'synthetic_control_not_market_evidence', 'kind': kind,
                'family': family, 'spec': spec, 'synthetic_seed': 703}, operation)
            naive = evaluate(test.cotton_close.to_numpy(), test.target_return_1.to_numpy(), np.zeros(len(test)))['mae']
            controls[f'{family}/{kind}'] = {'relative_mae_gain': 1 - record['result']['metrics']['mae'] / naive,
                                          'record': record['experiment_id'], 'device': record['result']['details']['effective_device']}
    # Ensure each tree objective and multiclass implementation actually runs on GPU.
    for family in ('xgboost', 'catboost'):
        for task in ('absolute_price', 'direction'):
            frame, train, validation, test = synthetic_control('known_signal')
            spec = diagnostic_recipes(1, FEATURE_NAMES)[-1]
            spec.update(family=family, task='direction' if task == 'direction' else 'price', loss='reg:absoluteerror')
            if family == 'catboost':
                spec['params'] = {'depth': 4, 'learning_rate': .1, 'l2_leaf_reg': 1}
            record = experiment.ledger.run({'role': 'gpu_objective_smoke', 'family': family, 'task': task},
                lambda workspace, f=frame, tr=train, va=validation, te=test, s=spec:
                    fit_predict(f, tr, va, te, s, workspace, iterations=12))
            controls[f'{family}/{task}'] = {'record': record['experiment_id'], 'status': 'passed'}
    failures = []
    if controls['xgboost/known_signal']['relative_mae_gain'] < .5:
        failures.append('known synthetic signal was not learned')
    if controls['xgboost/small_subset_overfit']['relative_mae_gain'] < .9:
        failures.append('small-subset learning sanity failed')
    if controls['xgboost/negative_block_shift']['relative_mae_gain'] > .2:
        failures.append('negative control unexpectedly predicts held-out synthetic targets')
    if failures:
        freeze_record(path, {'status': 'failed', 'failures': failures, 'controls': controls})
        raise ValueError('; '.join(failures))
    comparisons = []
    # The old scale/regularization hypothesis was already measured. Do not repeat
    # all historical fits merely to enter the new information-ablation round.
    for h in (() if free_profile else (1, 5)):
        for spec in diagnostic_recipes(h, experiment.identity['groups']['base']):
            scores = []
            for fold in experiment.identity['split']['folds']:
                scores.append(experiment.inner(spec, fold))
            comparisons.append({'hypothesis': spec['hypothesis'], 'horizon': h,
                                'mean_inner_relative_mae': float(np.mean([s['score'] for s in scores])),
                                'folds': scores})
    quality = {'rows': len(experiment.history), 'missing_features': experiment.history[FEATURE_NAMES].isna().sum().to_dict(),
               'large_absolute_log_returns_over_20pct': experiment.history.loc[experiment.history.cotton_ret_1.abs() > .2, 'date'].dt.strftime('%Y-%m-%d').tolist(),
               'contract_roll_metadata': 'unavailable; price jumps are not assumed to be contract rolls',
               'volume_zero_source_cause': 'unverified', 'audit_used': False}
    freeze_record(experiment.root / 'quality.json', quality)
    freeze_record(path, {'status': 'passed', 'controls': controls, 'comparisons': comparisons,
                        'conclusion': ('Tabular GPU controls passed; use ablate for information comparisons. '
                                       'Prior scale/regularization comparisons were not repeated.' if free_profile else
                                       'GPU and learning controls passed; regularization effects are measured, not presumed')})
    print('Diagnostic gate passed. Controlled comparisons saved:', path, flush=True)


def residual_report(history, origins, predicted, horizon, train):
    rows = rows_at(history, origins)
    p = rows.cotton_close.to_numpy()
    error = p * np.abs(np.exp(rows[f'target_return_{horizon}']) - np.exp(predicted))
    result = {}
    vol_cut = train.cotton_volatility_20.quantile([1 / 3, 2 / 3]).to_numpy()
    price_cut = train.cotton_close.quantile([1 / 3, 2 / 3]).to_numpy()
    groups = {'year': rows.date.dt.year.astype(str),
              'volatility': np.searchsorted(vol_cut, rows.cotton_volatility_20),
              'price': np.searchsorted(price_cut, rows.cotton_close),
              'missing_features': rows[FEATURE_NAMES].isna().any(axis=1).astype(int)}
    for name, group in groups.items():
        table = pd.DataFrame({'group': group, 'error': error}).groupby('group').error.agg(['mean', 'count'])
        result[name] = {str(index): {'mae': float(row['mean']), 'count': int(row['count'])} for index, row in table.iterrows()}
    result['contract_roll'] = 'unknown, not inferred from returns'
    return result


def write_lesson_packet(experiment, comparison):
    """Drive packet is reviewed before concise Obsidian capture; no invented lessons."""
    body = {'identity': content_id(experiment.identity), 'evidence': comparison,
            'distinction': 'Historical research; prospective performance remains unverified'}
    path = experiment.root / 'reports' / f'lessons-{content_id(body)[:16]}.json'
    path.write_text(json.dumps(body, indent=2), encoding='utf-8')
