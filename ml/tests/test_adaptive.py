"""Optuna journal replay with fake objective values; never train an estimator."""
from types import SimpleNamespace

import pytest

pytest.importorskip('optuna')
from cottonlens_ml.research.adaptive import tune


def fake(root, calls, interrupt_at=None):
    def inner(spec, fold, repeat=None, progress=None):
        calls.append(spec)
        if interrupt_at is not None and len(calls) == interrupt_at:
            raise KeyboardInterrupt()
        score = 1 + spec['params'].get('eta', spec['params'].get('learning_rate'))
        if progress:
            for step in range(3):
                progress(step, score)
        return {'recipe': spec, 'score': score, 'iterations': 10, 'records': [],
                'predictions': [0.], 'origins': ['2020-01-01']}
    return SimpleNamespace(root=root, inner=inner)


def test_adaptive_replays_proposals_and_completed_candidates_after_interrupt(tmp_path):
    calls = []
    fold = {'fold': 1}
    with pytest.raises(KeyboardInterrupt):
        tune(fake(tmp_path, calls, interrupt_at=3), 'xgboost', 1, fold, ['x'], count=4)
    assert len(list((tmp_path / 'adaptive-trials').rglob('*-result.json'))) == 2
    resumed_calls = []
    resumed = tune(fake(tmp_path, resumed_calls), 'xgboost', 1, fold, ['x'], count=4)
    uninterrupted = tune(fake(tmp_path / 'reference', []), 'xgboost', 1, fold, ['x'], count=4)
    assert resumed == uninterrupted
    # Two completed seed-42 candidates are replayed, not re-evaluated.
    assert len([s for s in resumed_calls if s['seed'] == 42]) == 2
    def fail(*args, **kwargs):
        pytest.fail('Completed decision was retrained')
    assert tune(SimpleNamespace(root=tmp_path, inner=fail), 'xgboost', 1, fold, ['x'], count=4) == resumed


def test_extension_reuses_initial_candidate_journal(tmp_path):
    fold = {'fold': 1}
    tune(fake(tmp_path, []), 'xgboost', 1, fold, ['x'], count=2)
    calls = []
    extended = tune(fake(tmp_path, calls), 'xgboost', 1, fold, ['x'],
                    count=4, namespace='adaptive-A-extension-1')
    assert len([spec for spec in calls if spec['seed'] == 42]) == 2
    assert len(extended['trials']) == 4
