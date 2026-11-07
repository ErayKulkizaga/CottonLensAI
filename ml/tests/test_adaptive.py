"""Optuna journal replay with fake objective values; never train an estimator."""
from types import SimpleNamespace

import pytest

pytest.importorskip('optuna')
from cottonlens_ml.research.adaptive import tune

from ml.tests.test_selection_boundary import fixture as split_fixture


def fold_fixture():
    return {**split_fixture()[1], 'fold': 1}


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
    history, _ = split_fixture()
    history['x'] = 1.
    return SimpleNamespace(root=root, inner=inner, history=history)


def test_adaptive_replays_proposals_and_completed_candidates_after_interrupt(tmp_path):
    calls = []
    fold = fold_fixture()
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
    cached = fake(tmp_path, [])
    cached.inner = fail
    assert tune(cached, 'xgboost', 1, fold, ['x'], count=4) == resumed


def test_extension_reuses_initial_candidate_journal(tmp_path):
    fold = fold_fixture()
    tune(fake(tmp_path, []), 'xgboost', 1, fold, ['x'], count=2)
    calls = []
    extended = tune(fake(tmp_path, calls), 'xgboost', 1, fold, ['x'],
                    count=4, namespace='adaptive-A-extension-1')
    assert len([spec for spec in calls if spec['seed'] == 42]) == 2
    assert len(extended['trials']) == 4


@pytest.mark.parametrize('stage', ['complete', 'interrupted'])
@pytest.mark.parametrize('change', ['history', 'features', 'count', 'fold', 'optuna_version'])
def test_changed_adaptive_context_never_replays_or_fits(tmp_path, monkeypatch, stage, change):
    fold = fold_fixture()
    if stage == 'interrupted':
        with pytest.raises(KeyboardInterrupt):
            tune(fake(tmp_path, [], interrupt_at=2), 'xgboost', 1, fold, ['x'], count=2)
    else:
        tune(fake(tmp_path, []), 'xgboost', 1, fold, ['x'], count=2)
    calls = []
    experiment = fake(tmp_path, calls)
    features, count = ['x'], 2
    if change == 'history':
        experiment.history.loc[0, 'x'] += 1
    elif change == 'features':
        features = ['other']
    elif change == 'count':
        count = 3
    elif change == 'optuna_version':
        import optuna
        monkeypatch.setattr(optuna, '__version__', 'changed-for-test')
    else:
        fold['year'] = 2099
    with pytest.raises(ValueError, match='binding'):
        tune(experiment, 'xgboost', 1, fold, features, count=count)
    assert not calls


def test_unbound_journal_is_not_adopted(tmp_path):
    journal = tmp_path/'adaptive-trials/xgboost-t1-fold1'
    journal.mkdir(parents=True)
    old = journal/'0000-result.json'
    old.write_text('preserved old partial evidence')
    calls = []
    with pytest.raises(ValueError, match='Unbound adaptive journal'):
        tune(fake(tmp_path, calls), 'xgboost', 1, fold_fixture(), ['x'], count=2)
    assert not calls and old.read_text() == 'preserved old partial evidence'


@pytest.mark.parametrize('field,value', [('state', 'UNKNOWN'), ('proposal_id', 'wrong'),
                                        ('execution_binding', 'wrong')])
def test_rehashed_trial_cannot_bypass_binding(tmp_path, field, value):
    import json

    from cottonlens_ml.cohort import content_id
    from cottonlens_ml.research.ledger import read_record
    with pytest.raises(KeyboardInterrupt):
        tune(fake(tmp_path, [], interrupt_at=2), 'xgboost', 1, fold_fixture(), ['x'], count=2)
    path = next((tmp_path/'adaptive-trials').rglob('*-result.json'))
    body = {**read_record(path), field: value}
    path.write_text(json.dumps({**body, 'record_id': content_id(body)}))
    calls = []
    with pytest.raises(ValueError, match='trial binding or state'):
        tune(fake(tmp_path, calls), 'xgboost', 1, fold_fixture(), ['x'], count=2)
    assert not calls


def test_stopped_extensions_copy_only_verified_records_without_fits(tmp_path, monkeypatch):
    from cottonlens_ml.research import adaptive
    from cottonlens_ml.research.decision_cache import execution_binding, read_selection
    from cottonlens_ml.research.ledger import freeze_record, read_record
    experiment = fake(tmp_path, [])
    fold = fold_fixture()
    experiment.identity = {'publication_sources': ['generated'],
                          'split': {'folds': [fold]}, 'groups': {'base': ['x']}}
    freeze_record(tmp_path/'diagnosis.json', {'status': 'passed'})
    monkeypatch.setattr(adaptive, 'require_colab_training', lambda: None)
    monkeypatch.setattr('cottonlens_ml.research.ablation.read_summary',
                        lambda e: {'comparisons': {'generated': {'base': {'admitted': True}}}})
    for family in ('xgboost', 'catboost'):
        for h in (1, 5):
            decision = tune(experiment, family, h, fold, ['x'], count=2)
            request = {'operation': 'outer', 'chosen': decision['chosen'],
                       'namespace': 'adaptive-A', 'repeat': None}
            freeze_record(tmp_path/'outer/adaptive-A'/f'{family}-t{h}-fold1.json',
                          {'predictions': [0.], 'execution_binding': execution_binding(
                              experiment, fold, request, selection=False)})
    def forbidden(*args, **kwargs):
        pytest.fail('Stopped extension attempted computation')
    experiment.inner = forbidden
    monkeypatch.setattr(adaptive, 'tune', forbidden)
    for extension in (1, 2):
        adaptive.search_tabular(experiment, extension)
        namespace = f'adaptive-A-extension-{extension}'
        decision = read_selection(experiment,
            tmp_path/'decisions'/namespace/'xgboost-t1-fold1.json',
            fold, namespace, 'xgboost', 1)
        assert decision['extension_skipped']
        assert read_record(tmp_path/'outer'/namespace/'xgboost-t1-fold1.json')['predictions'] == [0.]
