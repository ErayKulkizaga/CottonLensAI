"""Reproduction evidence checks use synthetic fixtures and never fit a model."""

import json

import numpy as np
import pytest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import engine
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record
from cottonlens_ml.research.protocol import rows_at, split_manifest
from cottonlens_ml.research.reproduction import (
    receipt_path,
    resolve_fit_identity,
    verify_for_export,
)
from test_sprint import history


@pytest.fixture
def native_repeats(admit_generated_jobs, tmp_path, request):
    frame = history()
    fold = split_manifest(frame)['folds'][0]
    fold = {**fold, 'origins': fold['origins'][:3]}
    identity = {'split': {'folds': [fold]}}
    experiment = object.__new__(engine.Experiment)
    experiment.root, experiment.identity, experiment.history = tmp_path, identity, frame
    experiment.ledger = Ledger(tmp_path/'ledger', identity)
    lock = {'identity': content_id(identity), 'horizons': {}}
    files = {}

    def record(h, family, value):
        namespace = 'fixed'
        recipe = {'family': family, 'horizon': h}
        chosen = {'recipe': recipe, 'seeds': [{'recipe': recipe, 'iterations': 1}]}
        rows = rows_at(frame, fold['origins'])
        cutoff = rows.date.min()
        spec = experiment.fit_specification(recipe, experiment.train_rows(cutoff, recipe),
            None, rows, f'outer-{namespace}-{fold["fold"]}-{cutoff}', 1,
            **({} if getattr(request, 'param', None) == 'legacy' else {'history': frame}))
        prediction = [value] * len(rows)

        def fake_native(work):
            (work/'adapter.json').write_text(json.dumps({'spec': recipe, 'model_file': 'model.json'}))
            (work/'model.json').write_text('{}')
            return {'predictions': prediction, 'origins': fold['origins']}

        fit = experiment.ledger.run(spec, fake_native, repeat='reproduction')
        name = f'{family}-t{h}-fold1.json'
        fresh = tmp_path/'reproduction'/namespace/name
        body = {'recipe': recipe, 'fold': 1, 'origins': fold['origins'], 'predictions': prediction}
        freeze_record(tmp_path/'outer'/namespace/name, body)
        freeze_record(fresh, {**body, 'records': [fit['experiment_id']]})
        freeze_record(tmp_path/'decisions'/namespace/name, {'chosen': chosen, 'selection_used_outer': False})
        files[h, family] = fresh, experiment.ledger.root/fit['payload_root']/'model.json'
        return {'namespace': namespace, 'decision': chosen, 'family': family}

    for h in (1, 5):
        member = record(h, 'ridge', .01)
        lock['horizons'][str(h)] = {'family': 'ridge', 'namespace': member['namespace']}
        if h == 5:
            alternative = record(h, 'elasticnet', .03)
            chosen = {'recipe': {'family': 'ensemble', 'horizon': h},
                      'members': [member, alternative], 'weights': [.25, .75]}
            prediction = (np.asarray([[.01]*3, [.03]*3]).T @ [.25, .75]).tolist()
            body = {'recipe': chosen['recipe'], 'fold': 1, 'origins': fold['origins'], 'predictions': prediction}
            for directory in ('outer', 'reproduction'):
                freeze_record(tmp_path/directory/'combined/ensemble-t5-fold1.json', body)
            freeze_record(tmp_path/'decisions/combined/ensemble-t5-fold1.json',
                          {'chosen': chosen, 'selection_used_outer': False})
            lock['horizons']['5'] = {'family': 'ensemble', 'namespace': 'combined'}
        freeze_record(tmp_path/f'reproduction-t{h}.json', {'status': 'passed', 'fresh_fits': True})

    freeze_record(tmp_path/'locked.json', lock)
    experiment.outer = lambda *a, **k: pytest.fail('Existing repetition must never start a fit during rebinding')
    return experiment, lock, files


def test_reproduction_binds_legacy_outputs_without_new_fits(native_repeats):
    experiment, lock, _ = native_repeats
    engine.reproduce(experiment)
    receipts = verify_for_export(experiment, lock)
    assert receipts['1']['fresh_fits'] is True
    assert receipts['1']['lock_identity'] == content_id(lock)
    assert len(receipts['1']['fit_records']) == 1
    assert len(receipts['5']['fit_records']) == 2
    assert receipts['5']['max_abs_log_return_difference'] == 0.
    assert read_record(experiment.root/'reproduction-t1.json') == {'status': 'passed', 'fresh_fits': True}
    engine.reproduce(experiment)
    assert read_record(receipt_path(experiment, 1)) == receipts['1']


def test_legacy_receipt_with_missing_native_model_never_refits(native_repeats):
    experiment, _, files = native_repeats
    files[1, 'ridge'][1].unlink()
    with pytest.raises(ValueError, match='native payload checksum'):
        engine.reproduce(experiment)
    assert not receipt_path(experiment, 1).exists()
    assert read_record(experiment.root/'reproduction-t1.json')['status'] == 'passed'


@pytest.mark.parametrize('fault', [None, 'missing_policy', 'wrong_weight'])
@pytest.mark.parametrize('policy', ['fold-local-refinement-v2', 'fold-local-search-extension-v2'])
def test_fold_local_policy_replays_exact_native_leaf_without_fitting(native_repeats, fault, policy):
    experiment, original_lock, _ = native_repeats
    root = experiment.root
    leaf = read_record(root/'decisions/fixed/ridge-t1-fold1.json')['chosen']
    chosen = {'recipe': {'family': 'ensemble', 'horizon': 1},
              'selection_policy': policy,
              'weights': [1.], 'members': [{'namespace': 'fixed', 'decision': leaf}]}
    if fault == 'missing_policy':
        del chosen['selection_policy']
    elif fault == 'wrong_weight':
        chosen['weights'] = [.5]
    body = read_record(root/'outer/fixed/ridge-t1-fold1.json')
    body['recipe'] = chosen['recipe']
    for directory in ('outer', 'reproduction'):
        freeze_record(root/directory/'policy/ensemble-t1-fold1.json', body)
    freeze_record(root/'decisions/policy/ensemble-t1-fold1.json',
                  {'chosen': chosen, 'selection_used_outer': False})
    lock = {**original_lock, 'horizons': {**original_lock['horizons'],
                                        '1': {'family': 'ensemble', 'namespace': 'policy'}}}
    (root/'locked.json').unlink()  # Generated fixture only.
    freeze_record(root/'locked.json', lock)
    if fault:
        with pytest.raises(ValueError, match='ensemble weights'):
            engine.reproduce(experiment)
    else:
        engine.reproduce(experiment)
        receipt = verify_for_export(experiment, lock)['1']
        assert len(receipt['fit_records']) == 1
        assert receipt['max_abs_log_return_difference'] == 0.


@pytest.mark.parametrize('fault', [None, 'source', 'receipt', 'archive', 'gate'])
def test_cached_export_rechecks_current_evidence_before_return(native_repeats, monkeypatch, fault):
    from cottonlens_ml.code_identity import digest
    from cottonlens_ml.research import release
    experiment, lock, _ = native_repeats
    freeze_record(experiment.root/'diagnosis.json', {'status': 'passed'})
    engine.reproduce(experiment)
    receipts = verify_for_export(experiment, lock)
    archive = experiment.root/'generated-release.zip'
    archive.write_bytes(b'generated cache-control fixture; not a model package')
    prior = {'path': str(archive), 'sha256': digest(archive), 'reproduction': receipts,
             'runtime_parity': 'passed', 'prospective_verified': False}
    if fault == 'receipt':
        prior['reproduction'] = {}
    freeze_record(experiment.root/'release.json', prior)
    monkeypatch.setattr(release, 'require_colab_training', lambda: None)
    checked = []
    def gate(*args):
        checked.append(True)
        if fault == 'gate':
            raise ValueError('Injected paired gate failure')
        return {}
    monkeypatch.setattr(release, '_checked_price_lock', gate)
    monkeypatch.setattr(release, 'load_frozen_data',
                        lambda *a: pytest.fail('Cached export attempted deployment computation'))
    if fault == 'source':
        experiment.identity = {**experiment.identity, 'source_id': 'changed'}
    elif fault == 'archive':
        archive.write_bytes(b'changed')
    if fault:
        with pytest.raises(ValueError):
            release.export(experiment, experiment.root)
    else:
        release.export(experiment, experiment.root)
        assert checked == [True]
    assert read_record(experiment.root/'release.json') == prior


@pytest.mark.parametrize('native_repeats', ['legacy'], indirect=True)
def test_legacy_native_receipts_remain_readable_without_minting_full_input_proof(native_repeats):
    experiment, lock, _ = native_repeats
    engine.reproduce(experiment)
    record = read_record(receipt_path(experiment, 1))
    assert record['input_binding_scope'] == 'legacy_missing_full_input_identity'
    assert record['max_abs_log_return_difference'] == 0.
    with pytest.raises(ValueError, match='historical inspection only'):
        verify_for_export(experiment, lock)
    assert read_record(experiment.root/'reproduction-t1.json') == {'status': 'passed', 'fresh_fits': True}


def test_v1_receipts_and_native_outputs_rebind_without_refitting(native_repeats):
    experiment, lock, _ = native_repeats
    originals = {}
    for h in (1, 5):
        path = experiment.root/f'reproduction-t{h}-bound-v1.json'
        freeze_record(path, {'version': 'bound-reproduction-v1', 'historical_fixture': True})
        originals[path] = path.read_bytes()
        (experiment.root/f'reproduction-t{h}.json').unlink()
    engine.reproduce(experiment)
    assert verify_for_export(experiment, lock)['1']['input_binding_scope'] == 'full_model_inputs_v2'
    assert all(p.read_bytes() == raw for p, raw in originals.items())


@pytest.mark.parametrize('lost', ['one_marker', 'all_markers'])
def test_partial_native_reproduction_is_not_silently_retrained(native_repeats, lost):
    experiment, _, files = native_repeats
    for h in (1, 5):
        (experiment.root/f'reproduction-t{h}.json').unlink()
    files[1, 'ridge'][0].unlink()  # Generated incomplete repetition only.
    if lost == 'all_markers':
        for path in (experiment.root/'reproduction').rglob('*.json'):
            path.unlink()
    # A remaining T+5 native file still proves this is an interrupted
    # repetition, not permission to start the missing work automatically.
    with pytest.raises(ValueError):
        engine.reproduce(experiment)


@pytest.mark.parametrize('formats', [(), ('legacy',), ('full',), ('legacy', 'full')])
def test_native_identity_resolution_never_guesses_input_schema(tmp_path, formats):
    ledger = Ledger(tmp_path/'ledger', {'generated': True})
    (ledger.root/'completed').mkdir(parents=True)
    legacy, full = {'role': 'generated'}, {'role': 'generated', 'model_inputs_schema': 2}
    specs = {'legacy': legacy, 'full': full}
    for name in formats:
        (ledger.root/'completed'/f'{ledger.key(specs[name])}.json').touch()
    for recorded_ids in (None, [ledger.key(specs[name]) for name in formats]):
        if len(formats) != 1:
            with pytest.raises(ValueError, match='missing.*or.*ambiguous'):
                resolve_fit_identity(ledger, legacy, full, recorded_ids)
        else:
            spec, key, scope = resolve_fit_identity(ledger, legacy, full, recorded_ids)
            assert spec == specs[formats[0]] and key == ledger.key(spec)
            assert scope == ('full_model_inputs_v2' if formats[0] == 'full'
                             else 'legacy_missing_full_input_identity')


@pytest.mark.parametrize('corruption', ['native_bytes', 'predictions', 'fit_key', 'decision', 'lock'])
def test_export_requires_exact_reproduction_evidence(native_repeats, corruption):
    experiment, lock, files = native_repeats
    engine.reproduce(experiment)
    marker, model = files[1, 'ridge']
    if corruption == 'native_bytes':
        model.write_text('changed')
        match = 'native payload checksum'
    elif corruption == 'decision':
        decision = experiment.root/'decisions/fixed/ridge-t1-fold1.json'
        body = read_record(decision)
        body['selection_used_outer'] = True
        decision.unlink()
        freeze_record(decision, body)
        match = 'frozen past-only decision'
    elif corruption == 'lock':
        lock = {**lock, 'prospective_start': 'changed'}
        match = 'does not prove fresh parity'
    else:
        body = read_record(marker)
        if corruption == 'predictions':
            body['predictions'][0] += .02
            match = 'disagrees with its native fit predictions'
        else:
            body['records'][0] = '0'*64
            match = 'missing an expected fresh fit record'
        marker.unlink()
        freeze_record(marker, body)
    with pytest.raises(ValueError, match=match):
        verify_for_export(experiment, lock)
