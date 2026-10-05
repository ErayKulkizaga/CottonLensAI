"""History checks cannot turn partial metadata or synthetic fits into market evidence."""
import json

import pytest
from cottonlens_ml.research.history import check, fingerprint, main, validate
from cottonlens_ml.research.history_index import index


def receipt(root, *, corrupt=False, role='outer-2016'):
    identity = {'source_id': 'source-v1', 'split': {'origins': ['2016-01-04']}, 'profile': 'pilot-v1'}
    recipe = {'family': 'ridge', 'horizon': 1, 'params': {'alpha': 1}, 'features': ['dxy_ret_1']}
    spec = {'recipe': recipe, 'role': role}
    value = {'identity': identity, 'specification': spec,
             'experiment_id': fingerprint({'identity': identity, 'specification': spec}), 'result': {}}
    value['record_id'] = fingerprint(value) if not corrupt else 'wrong'
    folder = root / 'experiment/ledger/completed'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'fit.json').write_text(json.dumps(value))
    return recipe


def test_complete_frozen_recipe_matches_but_changed_source_does_not(tmp_path):
    recipe = receipt(tmp_path)
    trials = index(tmp_path)
    scope_id = trials['records'][0]['scope_id']
    registry = {'records': []}
    assert check(registry, trials, proposal={'recipe': recipe, 'scope_id': scope_id})['status'] == 'REPEAT_FIT_RECIPE'
    for proposal in ({'recipe': recipe}, {'recipe': recipe, 'scope_id': 'different-code-or-data'}):
        assert check(registry, trials, proposal=proposal)['status'] == 'RELATED_EVIDENCE'
    changed = {**recipe, 'params': {'alpha': 10}}
    assert check(registry, trials, proposal={'recipe': changed, 'scope_id': scope_id})['status'] == 'RELATED_EVIDENCE'


def test_incomplete_market_and_synthetic_are_visible_without_negative_inference(tmp_path):
    receipt(tmp_path, role='diagnostic-known-signal')
    trials = index(tmp_path)
    assert trials['records'][0]['kind'] == 'synthetic_control:unspecified'
    partial = {'horizon': 5, 'classification': 'INCONCLUSIVE', 'coverage': 'partial', 'recipes': []}
    result = check({'records': [partial]}, trials, horizon=5)
    assert result['experiments'] == [partial]
    assert result['exact_fit_recipes'] == []
    assert 'not proof of novelty' in check({'records': []}, {'records': []})['limits']


def test_corrupt_receipt_and_attempt_are_not_indexed(tmp_path):
    receipt(tmp_path, corrupt=True)
    attempts = tmp_path / 'experiment/ledger/attempts'
    attempts.mkdir(); (attempts/'pending.json').write_text('{}')
    result = index(tmp_path)
    assert result['records'] == []
    assert len(result['rejected_receipts']) == 1


def test_duplicate_restored_receipts_count_once(tmp_path):
    receipt(tmp_path / 'first')
    receipt(tmp_path / 'restored')
    assert sum(r['completed_fits'] for r in index(tmp_path)['records']) == 1


def test_invalid_registry_fails_closed_and_command_is_read_only(tmp_path):
    (tmp_path/'registry.json').write_text(json.dumps({'schema': 1, 'records': [{'id': 'duplicate'}, {'id': 'duplicate'}]}))
    (tmp_path/'trials.json').write_text(json.dumps({'schema': 1, 'records': []}))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert main(['--registry-root', str(tmp_path), 'validate']) == 2
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='Missing or duplicate'):
        validate(tmp_path)


def test_corrupt_evidence_and_path_traversal_fail_validation(tmp_path):
    for path, digest in [('evidence.json', 'incorrect'), ('../outside', 'incorrect')]:
        (tmp_path/'evidence.json').write_text('{}')
        (tmp_path/'registry.json').write_text(json.dumps({'schema': 1, 'records': [],
            'local_evidence': [{'path': path, 'sha256': digest}]}))
        (tmp_path/'trials.json').write_text(json.dumps({'schema': 1, 'records': []}))
        assert main(['--registry-root', str(tmp_path), 'validate']) == 2
