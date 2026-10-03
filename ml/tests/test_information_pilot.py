"""Synthetic orchestration only: budgets, resume and source admission. No training."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import pilot
from cottonlens_ml.research.ledger import FitBudgetReached, Ledger, freeze_record


def frozen():
    dates = pd.bdate_range('2010-01-04', periods=900)
    history = pd.DataFrame({'date': dates, 'cotton_session_index': np.arange(len(dates))})
    inner = [{'origins': dates[300 + i * 63:363 + i * 63].strftime('%Y-%m-%d').tolist()} for i in range(3)]
    return {'profile': 'free-data-v1', 'groups': {'base': ['one'], 'source_ams': ['one', 'ams_spot']},
            'publication_sources': [{'manifest': {'kind': 'ams', 'vintage_policy': 'as_published',
                'usage': {'cost_tl': 0, 'research_allowed': True}}}],
            'split': {'audit_used': False, 'folds': [{'fold': 1, 'year': 2012,
                'origins': dates[700:826].strftime('%Y-%m-%d').tolist(), 'inner': inner}]}}, history


def test_pilot_exact_fit_budget_and_matched_recipes():
    identity, history = frozen()
    plan = pilot.pilot_plan(identity, history)
    assert plan['horizons'] == [5]
    assert len(plan['jobs']) == 16
    assert plan['core_fit_budget'] == 96
    assert plan['outer_fit_budget'] == 4
    assert plan['automatic_seed_confirmation'] is False
    jobs = plan['jobs']
    for family in plan['families']:
        a = [j['recipe'] for j in jobs if j['group'] == 'base' and j['family'] == family]
        b = [j['recipe'] for j in jobs if j['group'] == 'source_ams' and j['family'] == family]
        assert [r['params'] for r in a] == [r['params'] for r in b]


def test_ordinal_gaps_are_counted_in_fit_budget():
    identity, history = frozen()
    history.cotton_session_index *= 4
    plan = pilot.pilot_plan(identity, history)
    assert plan['core_fit_budget'] == 16 * 9  # Each inner block now needs two refits.


def test_unadmitted_sources_and_seen_audit_cannot_train():
    identity, history = frozen()
    identity['publication_sources'] = []
    assert pilot.pilot_plan(identity, history)['status'] == 'blocked'
    identity, history = frozen()
    identity['publication_sources'][0]['manifest']['vintage_policy'] = 'current_snapshot'
    with pytest.raises(ValueError, match='as-published'):
        pilot.pilot_plan(identity, history)
    identity, history = frozen()
    identity['split']['folds'][0]['year'] = 2024
    with pytest.raises(ValueError, match='audit'):
        pilot.pilot_plan(identity, history)


def test_budget_pause_is_not_failed_fit_and_cached_fit_does_not_compute(tmp_path):
    ledger = Ledger(tmp_path, {'synthetic': True})
    called = []
    def pause():
        raise FitBudgetReached('planned stop')
    with pytest.raises(FitBudgetReached):
        ledger.run({}, lambda w: called.append(1), before_compute=pause)
    assert not called
    assert ledger.summary()['failed_or_interrupted_attempts'] == 0
    result = ledger.run({}, lambda w: {'predictions': [.1]})
    assert ledger.run({}, lambda w: pytest.fail('Repeated fit'), before_compute=pause) == result


def test_budget_hook_does_not_bypass_cached_payload_verification(tmp_path):
    ledger = Ledger(tmp_path, {})
    def operation(workspace):
        (workspace / 'model.json').write_text('{}')
        return {}
    record = ledger.run({}, operation)
    (tmp_path / next(iter(record['files']))).write_text('corrupt')
    with pytest.raises(ValueError, match='Corrupt'):
        ledger.run({}, operation, before_compute=lambda: None)


def test_pilot_resume_does_not_repeat_completed_candidates(tmp_path, monkeypatch):
    identity, history = frozen()
    freeze_record(tmp_path / 'diagnosis.json', {'status': 'passed'})
    calls, outers = [], []
    def inner(recipe, fold):
        calls.append(recipe)
        experiment.before_compute()
        return {'recipe': recipe, 'score': .9, 'iterations': 10,
                'predictions': [0.] * 189, 'origins': [d for b in fold['inner'] for d in b['origins']]}
    experiment = SimpleNamespace(root=tmp_path, identity=identity, history=history, inner=inner,
        outer=lambda *a: outers.append(a))
    monkeypatch.setattr(pilot, 'require_colab_training', lambda: None)
    result = pilot.run_pilot(experiment, max_minutes=1)
    assert result['status'] == 'complete'
    assert len(calls) == 16 and len(outers) == 4
    assert result['gate_pass_claimed'] is False
    pilot.run_pilot(experiment, max_minutes=1)
    assert len(calls) == 16
    assert experiment.before_compute is None


def test_expired_budget_pauses_without_candidate_completion(tmp_path, monkeypatch):
    identity, history = frozen()
    freeze_record(tmp_path / 'diagnosis.json', {'status': 'passed'})
    experiment = SimpleNamespace(root=tmp_path, identity=identity, history=history)
    def inner(*a):
        experiment.before_compute()
        pytest.fail('Expired fit started')
    experiment.inner = inner
    monkeypatch.setattr(pilot, 'require_colab_training', lambda: None)
    times = iter([0., 61.])
    result = pilot.run_pilot(experiment, max_minutes=1, clock=lambda: next(times))
    assert result['status'] == 'planned_pause'
    assert not list((tmp_path / 'pilot/candidates').glob('*.json'))
    assert experiment.before_compute is None


def test_pause_mid_candidate_resumes_saved_subfits_without_recomputation(tmp_path, monkeypatch):
    identity, history = frozen()
    freeze_record(tmp_path / 'diagnosis.json', {'status': 'passed'})
    ledger = Ledger(tmp_path / 'ledger', identity)
    calls = []
    experiment = SimpleNamespace(root=tmp_path, identity=identity, history=history,
                                 outer=lambda *a: None)
    def inner(recipe, fold):
        for i in range(6):
            def synthetic_operation(workspace):
                calls.append(1)
                return {'predictions': [0.]}
            ledger.run({'recipe': recipe, 'step': i}, synthetic_operation,
                       before_compute=experiment.before_compute)
        return {'recipe': recipe, 'score': .99, 'iterations': 1, 'predictions': [0.], 'origins': []}
    experiment.inner = inner
    monkeypatch.setattr(pilot, 'require_colab_training', lambda: None)
    times = iter([0., 1., 2., 61.])
    result = pilot.run_pilot(experiment, max_minutes=1, clock=lambda: next(times))
    assert result['status'] == 'planned_pause'
    assert len(calls) == ledger.summary()['durably_saved'] == 2
    assert ledger.summary()['failed_or_interrupted_attempts'] == 0
    result = pilot.run_pilot(experiment, max_minutes=1, clock=lambda: 0.)
    assert result['status'] == 'complete'
    assert len(calls) == ledger.summary()['durably_saved'] == 96
