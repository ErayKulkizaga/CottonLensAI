"""Synthetic dependency graphs; no market data or training."""
from copy import deepcopy
from hashlib import sha256

import pytest
from cottonlens_ml.research.admission import (
    AdmissionError,
    ExecutionAdmission,
    identity,
    inspect,
    model_inputs_id,
    require,
    require_backend_inputs,
    require_model_inputs,
)
from cottonlens_ml.research.ledger import Ledger


@pytest.fixture
def graph(tmp_path):
    (tmp_path/'source.txt').write_text('synthetic source witness')
    base = {'provider': 'fixture', 'symbol': 'CT', 'contract': '2020-03', 'unit': 'cent/lb',
            'field': 'settlement', 'session': '2020-01-02', 'vintage': 'first', 'status': 'VERIFIED',
            'evidence': ['source.txt'], 'parents': [], 'available_at': '2020-01-02T20:00:00Z',
            'mature_at': '2020-01-02T20:00:00Z', 'target_policy': 'first'}
    value = {'schema': 1, 'bindings': dict.fromkeys(
        ('source', 'parser', 'calendar', 'features', 'target', 'split', 'recipe', 'code'), 'fixture'),
        'files': {'source.txt': sha256((tmp_path/'source.txt').read_bytes()).hexdigest()},
        'target_policy': 'first', 'open_critical_defects': [],
        'cells': [{**base, 'id': 'feature'}, {**base, 'id': 'label'},
                  {**base, 'id': 'future', 'available_at': '2020-01-06T20:00:00Z'}],
        'jobs': [{'id': 'job', 'model_inputs_id': 'synthetic_model_stub', 'fit_at': '2020-01-03T20:00:00Z',
                  'decision_at': '2020-01-03T21:00:00Z',
                  'inputs': {'feature': ['feature'], 'training_feature': ['feature'], 'training_label': ['label'],
                             'evaluation_target': ['future']}}]}
    return value, tmp_path


def test_valid_graph_allows_future_evaluation_target_and_repeat_validation(graph):
    value, root = graph
    receipt = inspect(value, root)
    assert receipt['accepted']
    assert require(value, receipt, root, value['bindings']) == receipt


@pytest.mark.parametrize(('field', 'replacement', 'reason'), [
    ('available_at', None, 'unknown_availability'),
    ('available_at', '2020-01-04T00:00:00Z', 'not_available_before_cutoff'),
    ('status', 'UNKNOWN', 'unverified_source'),
    ('contract', None, 'missing_contract'),
    ('vintage', None, 'missing_vintage'),
    ('evidence', [], 'unbound_source_evidence'),
])
def test_one_fault_has_specific_reason_and_repair_passes(graph, field, replacement, reason):
    value, root = graph
    original = deepcopy(value)
    value['cells'][0][field] = replacement
    result = inspect(value, root)
    assert not result['accepted']
    assert reason in [e['reason'] for e in result['errors']]
    assert inspect(original, root)['accepted']


def test_training_label_strict_maturity_boundary(graph):
    value, root = graph
    value['cells'][1]['mature_at'] = value['jobs'][0]['fit_at']
    assert 'label_not_mature_before_fit' in [e['reason'] for e in inspect(value, root)['errors']]


def test_feature_available_between_fit_and_decision_is_inference_only(graph):
    value, root = graph
    value['cells'].append({**value['cells'][0], 'id': 'late',
                           'available_at': '2020-01-03T20:30:00Z'})
    value['jobs'][0]['inputs']['feature'] = ['late']
    assert inspect(value, root)['accepted']
    value['jobs'][0]['inputs']['training_feature'] = ['late']
    result = inspect(value, root)
    assert not result['accepted']
    assert any(e['role'] == 'training_feature' and e['reason'] == 'not_available_before_cutoff'
               for e in result['errors'])


def test_unknown_ancestor_taints_derived_feature(graph):
    value, root = graph
    value['cells'].append({**value['cells'][0], 'id': 'ancestor', 'status': 'UNKNOWN'})
    value['cells'][0]['parents'] = ['ancestor']
    assert not inspect(value, root)['accepted']
    value['cells'][-1]['status'] = 'VERIFIED'
    assert inspect(value, root)['accepted']


def test_mutation_after_admission_and_recipe_change_rejected(graph):
    value, root = graph
    receipt = inspect(value, root)
    with pytest.raises(AdmissionError, match='bindings'):
        require(value, receipt, root, {**value['bindings'], 'recipe': 'new'})
    receipt['accepted'] = False
    with pytest.raises(AdmissionError, match='changed or rejected'):
        require(value, receipt, root, value['bindings'])
    (root/'source.txt').write_text('revised source')
    with pytest.raises(AdmissionError, match='hash changed'):
        inspect(value, root)


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'cycle', 'naive_time', 'path'])
def test_malformed_closure_fails_closed(graph, fault):
    value, root = graph
    if fault == 'duplicate':
        value['cells'].append(value['cells'][0])
    elif fault == 'missing':
        value['cells'][0]['parents'] = ['missing']
    elif fault == 'cycle':
        value['cells'][0]['parents'] = ['feature']
    elif fault == 'naive_time':
        value['jobs'][0]['decision_at'] = '2020-01-03T21:00:00'
    else:
        value['files'] = {'../outside.txt': '0'*64}
    with pytest.raises(AdmissionError):
        inspect(value, root)


def test_open_defect_and_dropped_role_cannot_be_accepted(graph):
    value, root = graph
    value['open_critical_defects'] = ['unresolved_contract']
    assert not inspect(value, root)['accepted']
    value['jobs'][0]['inputs']['feature'] = []
    with pytest.raises(AdmissionError, match='coverage'):
        inspect(value, root)


def test_ledger_admission_checks_exact_job_and_cache_before_return(graph):
    value, root = graph
    ledger_identity = {'profile': 'synthetic-admission-fixture', 'admission_policy': 1}
    spec = {'role': 'synthetic-test', 'recipe': {'family': 'fixture'}}
    value['ledger_identity'] = ledger_identity
    value['jobs'][0]['specification_id'] = identity(spec)
    admission = ExecutionAdmission(value, inspect(value, root), root)
    ledger = Ledger(root/'ledger', ledger_identity)
    called = []

    def compute(workspace):
        called.append(True)
        (workspace/'fixture.txt').write_text('result')
        return {'synthetic': True}

    with pytest.raises(ValueError, match='admission is missing'):
        ledger.run(spec, compute)
    first = ledger.run(spec, compute, admission=admission)
    second = ledger.run(spec, compute, admission=admission)
    assert first == second and len(called) == 1
    with pytest.raises(AdmissionError, match='specification'):
        ledger.run({**spec, 'recipe': {'family': 'other'}}, compute, admission=admission)
    (root/'source.txt').write_text('changed after successful fit')
    with pytest.raises(AdmissionError, match='hash changed'):
        ledger.run(spec, compute, admission=admission)
    assert len(called) == 1


def test_change_in_budget_hook_is_detected_before_compute(graph):
    value, root = graph
    ledger_identity = {'profile': 'synthetic-admission-fixture', 'admission_policy': 1}
    spec = {'role': 'synthetic-test'}
    value['ledger_identity'] = ledger_identity
    value['jobs'][0]['specification_id'] = identity(spec)
    admission = ExecutionAdmission(value, inspect(value, root), root)

    def forbidden(_):
        pytest.fail('Must not fit after source mutation')

    def mutate():
        (root/'source.txt').write_text('changed by staging')

    with pytest.raises(AdmissionError, match='hash changed'):
        Ledger(root/'ledger', ledger_identity).run(
            spec, forbidden, admission=admission, before_compute=mutate)


@pytest.mark.parametrize('ledger_identity,spec', [
    ({'profile': 'historical-market-profile'}, {'role': 'outer'}),
    ({'source_id': 'historical-source'}, {'role': 'outer'}),
    ({}, {'recipe': {'family': 'ridge'}}),
])
def test_legacy_namespace_cannot_start_an_unadmitted_fit(tmp_path, ledger_identity, spec):
    def forbidden(_):
        pytest.fail('Legacy missing admission must not compute')
    with pytest.raises(ValueError, match='New estimator fit requires'):
        Ledger(tmp_path/'ledger', ledger_identity).run(spec, forbidden)


@pytest.mark.parametrize('changed', ['history', 'train', 'validation', 'test', 'recipe', 'iterations', None])
def test_training_scope_binds_actual_arrays_and_resets_on_exit(graph, changed):
    import pandas as pd

    value, root = graph
    frames = {name: pd.DataFrame({'value': [1., 2.]})
              for name in ('history', 'train', 'validation', 'test')}
    recipe, iterations = {'family': 'synthetic-test'}, 1
    ledger_identity = {'profile': 'synthetic-test', 'admission_policy': 1}
    specification = {'recipe': recipe}
    value['ledger_identity'] = ledger_identity
    value['jobs'][0].update(specification_id=identity(specification),
                           model_inputs_id=model_inputs_id(**frames, recipe=recipe, iterations=iterations))
    admission = ExecutionAdmission(value, inspect(value, root), root)
    with pytest.raises(AdmissionError, match='Direct training'):
        require_model_inputs(**frames, recipe=recipe, iterations=iterations)

    with admission.training_scope(ledger_identity, specification):
        require_model_inputs(**frames, recipe=recipe, iterations=iterations)
        if changed in frames:
            frames[changed].iloc[0, 0] += 1
        elif changed == 'recipe':
            recipe = {'family': 'other'}
        elif changed == 'iterations':
            iterations = 2
        if changed:
            with pytest.raises(AdmissionError, match='inputs changed'):
                require_model_inputs(**frames, recipe=recipe, iterations=iterations)
    with pytest.raises(AdmissionError, match='Direct training'):
        require_model_inputs(**frames, recipe=recipe, iterations=iterations)


def test_backend_dispatch_preserves_validated_inputs_and_drops_authority(graph):
    import pandas as pd

    value, root = graph
    frame = pd.DataFrame({'value': [1., 2.]})
    recipe = {'family': 'synthetic-har'}
    ledger_identity, spec = {'profile': 'synthetic-backend'}, {'recipe': recipe}
    value['ledger_identity'] = ledger_identity
    value['jobs'][0].update(specification_id=identity(spec),
                           model_inputs_id=model_inputs_id(frame, frame, None, frame, recipe, 1))
    admission = ExecutionAdmission(value, inspect(value, root), root)
    with admission.training_scope(ledger_identity, spec):
        require_model_inputs(frame, frame, None, frame, recipe, 1)
        require_backend_inputs(frame, frame, frame, recipe)
        with pytest.raises(AdmissionError, match='Backend inputs changed'):
            require_backend_inputs(frame*2, frame, frame, recipe)
    with pytest.raises(AdmissionError, match='Direct backend'):
        require_backend_inputs(frame, frame, frame, recipe)
