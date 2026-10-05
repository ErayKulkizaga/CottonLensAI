"""No-fit regression tests for the bounded sprint protocol."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml import sprint
from cottonlens_ml.config import FEATURE_NAMES


def history():
    dates = pd.bdate_range('2010-01-01', '2026-09-22')
    frame = pd.DataFrame({'date': dates, 'cotton_close': 80 + np.arange(len(dates)) * .001,
                          'cotton_session_index': np.arange(len(dates))})
    for i, name in enumerate(FEATURE_NAMES):
        frame[name] = np.sin(np.arange(len(dates)) / (10 + i))
    for h in (1, 5):
        frame[f'target_date_{h}'] = frame.date.shift(-h)
        frame[f'target_return_{h}'] = np.log(frame.cotton_close.shift(-h) / frame.cotton_close)
    return frame


def test_development_has_504_origins_no_2022_targets_and_stable_against_audit():
    frame = history()
    folds = sprint.development_cohort(frame)
    dates = [d for f in folds for d in f['origins']]
    assert len(set(dates)) == 504
    chosen = frame.set_index('date').loc[pd.to_datetime(dates)]
    assert chosen.target_date_5.max() < pd.Timestamp('2022-01-01')
    frame.loc[frame.date >= '2022-01-01', FEATURE_NAMES] = np.nan
    assert sprint.development_cohort(frame) == folds


def test_history_window_purge_and_weights_use_only_fit_rows():
    frame = history()
    recipe = {**sprint.recipes()[1], 'years': 5}
    cutoff = pd.Timestamp('2021-07-01')
    train, validation = sprint.partition(frame, recipe, cutoff)
    assert len(validation) == 126
    assert train.target_date_5.max() < validation.date.min()
    assert validation.target_date_5.max() < cutoff
    assert train.date.min() >= cutoff - pd.DateOffset(years=5)
    weights = sprint.price_weights(train)
    assert weights.mean() == pytest.approx(1)
    frame.loc[frame.date >= cutoff, 'cotton_close'] *= 1000
    later_train, _ = sprint.partition(frame, recipe, cutoff)
    np.testing.assert_array_equal(weights, sprint.price_weights(later_train))


def test_native_full_schema_adapter_retains_order_and_zeros_only_unused_features():
    rows = history().tail(2)
    recipe = {**sprint.recipes()[0], 'features': 'cotton'}
    rows['cotton_volume_change'] = np.nan
    values = sprint.model_inputs(rows, recipe)
    assert list(values) == FEATURE_NAMES
    np.testing.assert_array_equal(values[sprint.COTTON], rows[sprint.COTTON])
    inactive = list(set(FEATURE_NAMES) - set(sprint.COTTON))
    assert values[inactive].eq(0).all().all()
    assert rows.cotton_volume_change.isna().all()


def test_cadence_counts_source_observations_not_retained_origins():
    frame = history().iloc[:200]
    test = frame.iloc[60:190].drop(index=[81, 82, 83, 84])
    schedule = sprint.refit_schedule(frame, test, 21)
    assert schedule[1][0] == frame.iloc[81].date
    assert schedule[1][1].date.min() == frame.iloc[85].date
    assert sum(len(chunk) for _, chunk in schedule) == 126
    assert len(sprint.refit_schedule(frame, test, 126)) == 1
    assert all(sprint.fit_rows(history(), sprint.recipes()[0], c).target_date_5.max() < c
               for c, _ in sprint.refit_schedule(history(), history().iloc[400:526], 21))


def result(mae=1., gain=1., wins=3, direction=55., horizon=1):
    return {'recipe': sprint.recipes()[0], 'horizon': horizon, 'cadence': 126,
            'aggregate': {'mae': mae, 'relative_mae_improvement_pct': gain, 'directional_accuracy': direction},
            'fold_wins': wins, 'folds': [{'metrics': {'mae': mae}} for _ in range(4)]}


def test_refinement_cadence_and_final_gates_are_distinct():
    old = result()
    assert sprint.refinement_allowed(old)
    assert not sprint.gate(old)['passes']
    assert not sprint.refinement_allowed(result(gain=0))
    assert not sprint.refinement_allowed(result(wins=2))
    assert sprint.cadence_wins(old, result(mae=.99))
    assert not sprint.cadence_wins(old, result(mae=.99, direction=54.99))
    assert not sprint.cadence_wins(old, result(mae=.995))
    assert sprint.gate(result(gain=5, direction=53))['passes']
    assert not sprint.gate(result(gain=5, direction=53, horizon=5))['passes']
    assert sprint.gate(result(gain=5, direction=55, horizon=5))['passes']


def test_frozen_records_reject_overwrite_and_corruption(tmp_path):
    path = tmp_path / 'complete.json'
    sprint.freeze_record(path, {'value': 1})
    sprint.freeze_record(path, {'value': 1})
    with pytest.raises(ValueError, match='changed'):
        sprint.freeze_record(path, {'value': 2})
    body = json.loads(path.read_text())
    body['value'] = 2
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='Corrupt'):
        sprint.read_record(path)


def test_deadline_and_lock_close_search(tmp_path):
    ready = {'created_at': (datetime.now(UTC) - timedelta(hours=49)).isoformat()}
    with pytest.raises(ValueError, match='deadline'):
        sprint.check_search_open(tmp_path, ready)
    (tmp_path / 'locked.json').touch()
    with pytest.raises(ValueError, match='already locked'):
        sprint.check_search_open(tmp_path, {'created_at': datetime.now(UTC).isoformat()})


def test_writer_excludes_concurrent_sessions(tmp_path):
    with sprint.writer(tmp_path), pytest.raises(RuntimeError, match='writer'), sprint.writer(tmp_path):
        pass
    assert not (tmp_path / '.writer-lock').exists()


def test_equal_score_prefers_shallower_then_smaller_features():
    a, b, c = result(), result(), result()
    a['recipe'] = {**a['recipe'], 'depth': 2}
    c['recipe'] = {**c['recipe'], 'features': 'cotton'}
    assert sprint.best_result([a, b, c]) is c


def test_score_run_locks_trees_and_refits_only_mature_labels(tmp_path, monkeypatch):
    frame = history()
    folds = sprint.development_cohort(frame)
    calls = []

    def fake_fit(train, validation, recipe, horizon, folder, *, trees=None, fresh=False):
        calls.append((train.copy(), validation, trees, fresh))
        return SimpleNamespace(best_iteration=6, predict=lambda x: np.zeros(len(x)))

    monkeypatch.setattr(sprint, 'fit_model', fake_fit)
    scored = sprint.score_run(frame, folds, sprint.recipes()[0], 1, 21, tmp_path, fresh=True)
    assert scored['aggregate']['sample_count'] == 504
    assert sum(validation is not None for _, validation, _, _ in calls) == 4
    assert all(trees == 7 for _, validation, trees, _ in calls if validation is None)
    assert all(fresh for _, _, _, fresh in calls)
    for fold in scored['folds']:
        for refit in fold['refits']:
            assert refit['last_label'] < refit['cutoff']
    for record in scored['aggregate']['origin_records']:
        assert record['fit_cutoff'] <= record['origin']


def test_local_real_fit_is_blocked(tmp_path, monkeypatch):
    from cottonlens_ml import runtime_guard
    monkeypatch.delenv('COLAB_RELEASE_TAG', raising=False)
    monkeypatch.delenv('COLAB_BACKEND_VERSION', raising=False)
    assert runtime_guard.require_colab_training is sprint.require_colab_training
    with pytest.raises(RuntimeError):
        sprint.fit_model(history(), None, sprint.recipes()[0], 1, tmp_path, trees=1)


def test_benchmark_requires_lock_without_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(sprint, 'require_colab_training', lambda: None)
    monkeypatch.setattr(sprint, 'load_ready', lambda *_: (tmp_path, {}, history()))
    with pytest.raises(ValueError, match='Lock before'):
        sprint.run_stage(tmp_path, tmp_path, 'test', 'benchmark')


def test_report_accepts_tabular_release_without_fabricating_lstm():
    from cottonlens_ml.report import build_report
    from test_report import _evidence

    evidence = _evidence()
    evidence['metrics'] = [r for r in evidence['metrics'] if r['model'] in ('Naive', 'XGBoost')]
    walk = evidence['manifest']['walkforward_report']
    walk['aggregate'] = {k: v for k, v in walk['aggregate'].items() if k.startswith(('Naive-', 'XGBoost-'))}
    for fold in walk['folds']:
        fold['metrics'] = {k: v for k, v in fold['metrics'].items() if k.startswith(('Naive-', 'XGBoost-'))}
    for decision in evidence['manifest']['selection_audit'].values():
        decision['historical_audit'].pop('LSTM')
    assert 'LSTM not evaluated in this experiment' in build_report(evidence)


def test_stages_resume_lock_then_benchmark_once_and_reproduce_fresh(tmp_path, monkeypatch):
    frame = history()
    folds = sprint.development_cohort(frame)
    ready = {'created_at': datetime.now(UTC).isoformat(), 'development_folds': folds,
             'benchmark_cohort': {'folds': folds}}
    monkeypatch.setattr(sprint, 'require_colab_training', lambda: None)
    monkeypatch.setattr(sprint, 'load_ready', lambda *_: (tmp_path, ready, frame))
    monkeypatch.setattr(sprint, 'baseline_runs', lambda *_: {'Naive': {'mae': 1, 'role': 'reference_only'}})
    calls = []

    def fake_score(history, folds, recipe, horizon, cadence, folder, *, fresh=False):
        calls.append((recipe, horizon, cadence, fresh))
        scored = result(gain=-1, horizon=horizon)
        scored.update(recipe=recipe, cadence=cadence, elapsed_seconds=1.)
        for number, fold in enumerate(scored['folds'], 1):
            fold.update(fold=number, trees=7, naive={'mae': 1.})
            fold['metrics']['directional_accuracy'] = 50.
            fold['predictions'] = [0.] * 126
        return scored

    monkeypatch.setattr(sprint, 'score_run', fake_score)
    for stage in ('search', 'search', 'refine', 'cadence', 'lock', 'benchmark', 'benchmark', 'reproduce', 'reproduce'):
        sprint.run_stage(tmp_path, tmp_path, 'synthetic', stage)
    assert len(calls) == 18  # 12 initial + 2 cadence + 2 benchmark + 2 fresh reproduction
    assert sum(fresh for _, _, _, fresh in calls) == 2
    assert (tmp_path / 'refinement-skipped-t1.json').exists()
    assert sprint.read_record(tmp_path / 'selection.json')['audit_used'] is False
    assert sprint.read_record(tmp_path / 'reproduction-t1.json')['fresh_fits'] is True
    with pytest.raises(ValueError, match='closed'):
        sprint.run_stage(tmp_path, tmp_path, 'synthetic', 'search')


def test_sprint_notebook_preserves_r2_and_stages_do_not_run_all_implicitly():
    import ast
    from pathlib import Path
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/archive/cottonlens_sprint_colab.ipynb').read_text())
    cells = {c['id']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
    for source in cells.values():
        ast.parse(source)
    assert "STAGE = 'search'" in cells['stage']
    assert 'cottonlens_ml.prepare' not in '\n'.join(cells.values())
    assert 'SPRINT_ROOT' in cells['validate']
    assert "'--drive-root', SPRINT_ROOT" in cells['validate']
