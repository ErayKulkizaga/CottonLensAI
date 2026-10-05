"""Synthetic/no-fit temporal, resumption and inference contracts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import engine
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record
from cottonlens_ml.research.models import (
    PriceMetric,
    class_metrics,
    fit_predict,
    temperature,
)
from cottonlens_ml.research.protocol import (
    Preprocessor,
    Target,
    attach_releases,
    feature_history,
    inputs,
    mature,
    rows_at,
    split_manifest,
)
from cottonlens_ml.research.search import (
    BUDGETS,
    historical_gate,
    next_round,
    recipes,
    simplex_weights,
)
from test_sprint import history


def test_split_eight_nested_folds_and_no_audit_dependence():
    frame = history()
    manifest = split_manifest(frame)
    assert len(manifest['folds']) == 8
    assert len({d for f in manifest['folds'] for d in f['origins']}) == 1008
    for fold in manifest['folds']:
        outer = rows_at(frame, fold['origins'])
        assert outer.target_date_5.max() < pd.Timestamp('2024-01-01')
        assert len(fold['inner']) == 3
        for block in fold['inner']:
            val = rows_at(frame, block['origins'])
            assert len(val) == 63 and val.target_date_5.max() < outer.date.min()
            assert mature(frame, val.date.min()).target_date_5.max() < val.date.min()
    frame.loc[frame.date >= '2024-01-01', 'cotton_close'] *= 99
    assert split_manifest(frame) == manifest


def test_missingness_is_train_fitted_and_sequences_preserve_observations():
    frame = history().head(150)
    names = ['cotton_ret_1', 'cotton_ret_5']
    frame.loc[30:50, names[0]] = np.nan
    fit = Preprocessor.fit(frame.iloc[:100], names)
    original = fit.transform(frame.iloc[:100])
    frame.loc[100:, names] = 1e12
    np.testing.assert_array_equal(original, Preprocessor.fit(frame.iloc[:100], names).transform(frame.iloc[:100]))
    matrix = inputs(frame, frame.iloc[[59]], fit, 60)
    assert matrix.shape == (1, 60, 4)
    assert matrix[0, 30:51, 2].sum() == 21


@pytest.mark.parametrize('kind', ['raw_log', 'scaled_log', 'price_delta'])
def test_target_roundtrip_uses_training_only(kind):
    frame = history().head(300)
    transform = Target.fit(frame.head(150), 5, kind)
    original = frame.iloc[150:250]
    reconstructed = transform.inverse(transform.forward(original, 5), original.cotton_close.to_numpy())
    np.testing.assert_allclose(reconstructed, original.target_return_5, atol=1e-8)
    frame.loc[150:, 'target_return_5'] = 100
    assert Target.fit(frame.head(150), 5, kind) == transform


def test_causal_feature_extension_is_unchanged_by_future():
    frame = history().head(350)
    earlier = feature_history(frame).iloc[:200]
    frame.loc[200:, 'cotton_close'] *= 5
    pd.testing.assert_frame_equal(earlier, feature_history(frame).iloc[:200])


def test_publication_timestamp_and_vintage_fail_closed():
    frame = history().head(5)
    release = pd.DataFrame({'published_at': ['2010-01-05T12:00:00Z'], 'vintage_id': ['original'],
        'source_url': ['https://example.org/report'], 'source_sha256': ['a' * 64], 'timestamp_verified': [True], 'wasde_stocks': [8.]})
    values = attach_releases(frame, release, ['wasde_stocks'])
    assert values.loc[values.date < '2010-01-05', 'wasde_stocks'].isna().all()
    assert values.loc[values.date >= '2010-01-05', 'wasde_stocks'].eq(8).all()
    release.timestamp_verified = False
    with pytest.raises(ValueError, match='verified'):
        attach_releases(frame, release, ['wasde_stocks'])


def test_ledger_reuses_completed_models_rejects_corruption_and_requires_repeat_reason(tmp_path):
    ledger = Ledger(tmp_path, {'source': 'one'})
    calls = []
    def run(workspace):
        calls.append(1)
        (workspace / 'model.json').write_text('{}')
        return {'prediction': 1}
    one = ledger.run({'seed': 42}, run)
    assert ledger.run({'seed': 42}, run) == one and len(calls) == 1
    ledger.run({'seed': 42}, run, repeat='reproduction')
    assert len(calls) == 2
    with pytest.raises(ValueError, match='repetition'):
        ledger.run({}, run, repeat='try_again_until_win')
    payload = next(iter(one['files']))
    (tmp_path / payload).write_text('corrupted')
    with pytest.raises(ValueError, match='Corrupt'):
        ledger.run({'seed': 42}, run)


def test_interruption_does_not_publish_completed_record(tmp_path):
    ledger = Ledger(tmp_path, {})
    def interrupted(workspace):
        (workspace / 'model.json').write_text('{}')
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        ledger.run({}, interrupted)
    assert not list((tmp_path / 'completed').glob('*.json'))
    assert read_record(next((tmp_path / 'attempts').glob('*.json')))['state'] == 'interrupted'


def test_all_budgeted_recipes_are_unique_reproducible_and_bounded():
    for round_name, families in BUDGETS.items():
        for family, count in families.items():
            rows = recipes(family, 1, ['x'], count, task='direction' if round_name == 'B' else 'price')
            assert len(rows) == count
            assert rows == recipes(family, 1, ['x'], count, task='direction' if round_name == 'B' else 'price')
            assert len({json.dumps(row, sort_keys=True) for row in rows}) == count
    assert next_round([.001, .002]) == 'new_information_or_diagnosis'
    assert not historical_gate({'relative_mae_improvement_pct': 5., 'directional_accuracy': 54.}, 8, 5)['passes']


def test_gpu_training_cannot_run_locally(tmp_path, monkeypatch):
    monkeypatch.delenv('COLAB_RELEASE_TAG', raising=False)
    monkeypatch.delenv('COLAB_BACKEND_VERSION', raising=False)
    with pytest.raises(RuntimeError, match='Colab'):
        fit_predict(None, None, None, None, {}, tmp_path)


def test_direction_flat_class_calibration_and_metrics():
    probability = np.array([[.9, .05, .05], [.1, .8, .1], [.2, .1, .7]])
    score = class_metrics(np.array([0, 1, 2]), probability)
    assert score['directional_accuracy'] == 100 and score['flat_targets'] == 1
    np.testing.assert_allclose(temperature(probability, 1), probability)


def test_catboost_custom_metric_matches_price_metric():
    frame = history().head(120)
    for kind in ('raw_log', 'scaled_log', 'price_delta'):
        target = Target.fit(frame, 5, kind)
        truth = target.forward(frame, 5)
        pred = truth + .0001
        metric = PriceMetric(kind, target.mean, target.scale)
        error, weight = metric.evaluate([pred], truth, frame.cotton_close.to_numpy())
        actual_return = target.inverse(truth, frame.cotton_close.to_numpy())
        predicted_return = target.inverse(pred, frame.cotton_close.to_numpy())
        expected = np.mean(frame.cotton_close.to_numpy() * abs(np.exp(actual_return) - np.exp(predicted_return)))
        assert metric.get_final_error(error, weight) == pytest.approx(expected, abs=1e-9)


def test_ensemble_weights_are_nonnegative_and_sum_to_one():
    actual = np.linspace(-.02, .02, 100)
    weights = simplex_weights(np.array([actual, -actual]).T, np.full(100, 80.), actual)
    assert min(weights) >= 0 and sum(weights) == pytest.approx(1)
    assert weights[0] > .99


def test_nested_tuning_uses_earlier_stopping_and_cadence_mature_labels(tmp_path, monkeypatch):
    frame = history()
    frame = frame.loc[frame.date < '2024-01-01'].copy()
    frame.to_parquet(tmp_path / 'history.parquet', index=False)
    identity = {'split': split_manifest(frame)}
    freeze_record(tmp_path / 'ready.json', {'identity': identity, 'history_sha256': digest(tmp_path / 'history.parquet')})
    experiment = engine.Experiment(tmp_path)
    calls = []
    def fake(history, train, validation, test, spec, workspace, *, iterations=None):
        calls.append((train, validation, test, iterations))
        assert train.target_date_5.max() < test.date.min()
        (workspace / 'model.json').write_text('{}')
        return {'predictions': [0.] * len(test), 'origins': test.date.dt.strftime('%Y-%m-%d').tolist(),
                'metrics': {'mae': 1}, 'iterations': iterations or 7}
    monkeypatch.setattr(engine, 'fit_predict', fake)
    spec = {**recipes('xgboost', 1, ['cotton_ret_1'], 1)[0], 'cadence': 21}
    result = experiment.inner(spec, identity['split']['folds'][0])
    assert len(result['predictions']) == 189
    assert result['score'] == pytest.approx(1)
    assert sum(v is not None for _, v, _, _ in calls) == 3
    assert all(count == 7 for _, v, _, count in calls if v is None)
    count = len(calls)
    assert experiment.inner(spec, identity['split']['folds'][0]) == result
    assert len(calls) == count


def test_notebook_does_not_start_release_or_remote_actions():
    path = Path(__file__).resolve().parents[1] / 'notebooks/archive/cottonlens_research_colab.ipynb'
    if not path.exists():
        pytest.skip('Notebook delivery is generated after engine tests')
    import ast
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'research=True' in source and "ROUND = 'A'" in source
    assert 'git push' not in source


def test_prospective_emissions_are_immutable_and_not_backdated(tmp_path):
    from cottonlens_ml.research.prospective import record, score
    frame = pd.DataFrame({'date': pd.to_datetime(['2026-09-24']), 'cotton_close': [80.]})
    locked = {'prospective_start': '2026-09-23T12:00:00+00:00', 'horizons': {'1': {}, '5': {}}}
    now = '2026-09-25T02:00:00+00:00'
    saved = record(tmp_path, locked, 'release-1', frame, lambda h, x: .01, now=now)
    assert saved['predictions']['1']['price'] == pytest.approx(80 * np.exp(.01))
    assert record(tmp_path, locked, 'release-1', frame, lambda h, x: .99, now=now) == saved
    assert score(tmp_path, pd.DataFrame())['status'] == 'pending'
    with pytest.raises(ValueError, match='no backdating'):
        record(tmp_path, locked, 'release-1', frame, lambda h, x: .01, now='2026-09-27T00:00:00+00:00')
    with pytest.raises(ValueError):
        record(tmp_path, locked, 'release-2', frame, lambda h, x: .01, now=now)


def test_lock_resume_preserves_prospective_start(tmp_path, monkeypatch):
    from types import SimpleNamespace
    identity = {'protocol': 'test'}
    locked = {'identity': content_id(identity), 'prospective_start': '2026-09-25T00:00:00+00:00'}
    freeze_record(tmp_path / 'locked.json', locked)
    monkeypatch.setattr(engine, 'compare', lambda _: pytest.fail('Already locked; must not select again'))
    assert engine.lock(SimpleNamespace(root=tmp_path, identity=identity)) == locked


def test_keras_data_adapter_and_save_do_not_inherit_forced_gpu_scope():
    """RangeDataset has no GPU kernel: Keras orchestration must be unscoped."""
    import ast
    import inspect

    from cottonlens_ml.research.models import fit_predict
    tree = ast.parse(inspect.getsource(fit_predict))
    fits = []
    class ScopeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.gpu_depth = 0
        def visit_With(self, node):
            forced = any(isinstance(item.context_expr, ast.Call)
                and any(isinstance(arg, ast.Constant) and arg.value == '/GPU:0'
                        for arg in item.context_expr.args) for item in node.items)
            self.gpu_depth += int(forced)
            self.generic_visit(node)
            self.gpu_depth -= int(forced)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr in ('fit', 'save'):
                assert self.gpu_depth == 0, 'CPU-only orchestration forced onto GPU'
                if node.func.attr == 'fit' and any(k.arg == 'callbacks' for k in node.keywords):
                    fits.append(node)
            self.generic_visit(node)
    ScopeVisitor().visit(tree)
    assert len(fits) == 1


def test_tensorflow_gpu_variable_guard_rejects_cpu_and_unknown_devices():
    from types import SimpleNamespace

    from cottonlens_ml.research.models import require_gpu_variables
    def model(devices):
        return SimpleNamespace(trainable_variables=[SimpleNamespace(value=SimpleNamespace(device=d)) for d in devices])
    gpu = '/job:localhost/replica:0/task:0/device:GPU:0'
    assert require_gpu_variables(model([gpu, gpu])) == [gpu]
    for devices in ([], ['/device:CPU:0'], [''], [gpu, '/device:CPU:0']):
        with pytest.raises(RuntimeError, match='must reside on GPU'):
            require_gpu_variables(model(devices))
