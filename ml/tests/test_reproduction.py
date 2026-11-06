"""Reproduction evidence checks use synthetic fixtures and never fit a model."""

import json

import numpy as np
import pytest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import engine
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record
from cottonlens_ml.research.protocol import rows_at, split_manifest
from cottonlens_ml.research.reproduction import receipt_path, verify_for_export
from test_sprint import history


@pytest.fixture
def native_repeats(admit_generated_jobs, tmp_path):
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
            None, rows, f'outer-{namespace}-{fold["fold"]}-{cutoff}', 1)
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
