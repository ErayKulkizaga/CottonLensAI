"""Bind historical reproduction to the locked choice and its native checkpoints."""

import json
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np

from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.research.protocol import frozen_outer_folds, verified_outer

VERSION = 'bound-reproduction-v1'


class FoldPlan(NamedTuple):
    fold: dict
    namespace: str
    chosen: dict
    original_prediction: np.ndarray
    original_path: Path
    original_sha256: str
    decision_path: Path
    decision_sha256: str


def _record_with_sha(path):
    before = digest(path)
    value = read_record(path)
    if digest(path) != before:
        raise ValueError('Reproduction evidence changed during verification')
    return value, before


def receipt_path(experiment, horizon):
    return experiment.root / f'reproduction-t{horizon}-bound-v1.json'


def _marker(experiment, directory, namespace, family, horizon, fold):
    if any(not safe_member(name) or PurePosixPath(name).parts != (name,) for name in (namespace, family)):
        raise ValueError('Unsafe reproduction path identity')
    return experiment.root / directory / namespace / f'{family}-t{horizon}-fold{fold["fold"]}.json'


def plan(experiment, locked):
    if locked['identity'] != content_id(experiment.identity):
        raise ValueError('Locked research identity changed')
    folds = frozen_outer_folds(experiment.identity['split']['folds'])
    plans = {}
    for h, candidate in locked['horizons'].items():
        family, namespace = candidate['family'], candidate['namespace']
        plans[h] = []
        for fold in folds.values():
            original = _marker(experiment, 'outer', namespace, family, h, fold)
            original_record, original_sha = _record_with_sha(original)
            _, prediction = verified_outer(experiment.history, fold, original_record, family, int(h))
            decision = _marker(experiment, 'decisions', namespace, family, h, fold)
            selection, decision_sha = _record_with_sha(decision)
            if selection.get('selection_used_outer') is not False or selection['chosen']['recipe'] != original_record['recipe']:
                raise ValueError('Original prediction disagrees with the frozen past-only decision')
            plans[h].append(FoldPlan(fold, namespace, selection['chosen'], prediction,
                                     original, original_sha, decision, decision_sha))
    return plans


def verified_native_predictions(ledger, fit_id, expected):
    path = ledger.root / 'completed' / f'{fit_id}.json'
    fit, fit_sha = _record_with_sha(path)
    if (fit['identity'] != ledger.identity or fit['experiment_id'] != fit_id
            or fit['specification'] != expected or ledger.key(expected) != fit_id):
        raise ValueError('Reproduction fit identity mismatch')
    payload = fit['payload_root']
    files = fit['files']
    if (not safe_member(payload) or PurePosixPath(payload).parts[:2] != ('payloads', fit_id)
            or len(PurePosixPath(payload).parts) != 3 or not files):
        raise ValueError('Reproduction native payload is missing or mismatched')
    payload_path = ledger.root.resolve() / payload
    for name, checksum in files.items():
        member = ledger.root / name
        if (not safe_member(name) or not name.startswith(payload + '/')
                or not member.resolve().is_relative_to(payload_path)
                or not member.is_file() or digest(member) != checksum):
            raise ValueError('Reproduction native payload checksum mismatch')
    recipe = expected['recipe']
    if recipe.get('task') == 'variance':
        model_path = f'{payload}/variance-model.json'
        if model_path not in files:
            raise ValueError('Reproduction native variance model is missing')
        model = json.loads((ledger.root / model_path).read_bytes())
        expected_kind = {'har': 'log_har', 'garch': 'garch'}[recipe['family']]
        if model['kind'] != expected_kind:
            raise ValueError('Reproduction native variance model identity mismatch')
        field = 'variance_predictions'
    else:
        adapter_path = f'{payload}/adapter.json'
        if adapter_path not in files:
            raise ValueError('Reproduction native adapter is missing')
        adapter = json.loads((ledger.root / adapter_path).read_bytes())
        model_file = adapter['model_file']
        if (adapter['spec'] != recipe or Path(model_file).name != model_file
                or model_file == 'adapter.json' or f'{payload}/{model_file}' not in files):
            raise ValueError('Reproduction native model or adapter identity mismatch')
        field = 'predictions'
    result = fit['result']
    prediction = np.asarray(result[field], dtype=float)
    if (result['origins'] != expected['test_dates'] or prediction.shape != (len(expected['test_dates']),)
            or not np.isfinite(prediction).all()):
        raise ValueError('Reproduction native predictions are misaligned')
    return prediction, fit_sha


def _repeat_predictions(experiment, fold, namespace, chosen, receipts):
    recipe = chosen['recipe']
    family, h = recipe['family'], recipe['horizon']
    path = _marker(experiment, 'reproduction', namespace, family, h, fold)
    fresh, fresh_sha = _record_with_sha(path)
    rows, prediction = verified_outer(experiment.history, fold, fresh, family, h)
    if fresh['recipe'] != recipe:
        raise ValueError('Reproduction changed the chosen recipe')
    if family == 'ensemble':
        weights = np.asarray(chosen['weights'], dtype=float)
        if (len(chosen['members']) < 2 or weights.shape != (len(chosen['members']),)
                or not np.isfinite(weights).all() or (weights < 0).any() or abs(weights.sum()-1) > 1e-8):
            raise ValueError('Invalid reproduction ensemble weights')
        members = [_repeat_predictions(experiment, fold, member['namespace'], member['decision'], receipts)[0]
                   for member in chosen['members']]
        expected = np.asarray(members).T @ weights
    else:
        if not chosen['seeds']:
            raise ValueError('Reproduction requires complete native fits')
        expected_ids, predictions = [], []
        for seed in chosen['seeds']:
            chunks = []
            for cutoff, test in experiment.outer_chunks(rows, seed['recipe']):
                train = experiment.train_rows(cutoff, seed['recipe'])
                spec = experiment.fit_specification(seed['recipe'], train, None, test,
                    f'outer-{namespace}-{fold["fold"]}-{cutoff}', seed['iterations'])
                spec['repeat_reason'] = 'reproduction'
                fit_id = experiment.ledger.key(spec)
                expected_ids.append(fit_id)
                if fit_id not in fresh.get('records', ()):
                    raise ValueError('Reproduction is missing an expected fresh fit record')
                chunk, receipts[fit_id] = verified_native_predictions(experiment.ledger, fit_id, spec)
                chunks.append(chunk)
            predictions.append(np.concatenate(chunks))
        if fresh.get('records') != expected_ids or len(set(expected_ids)) != len(expected_ids):
            raise ValueError('Reproduction fit record order or coverage changed')
        expected = np.mean(np.asarray(predictions), axis=0)
    if not np.allclose(prediction, expected, rtol=0, atol=1e-13):
        raise ValueError('Reproduction output disagrees with its native fit predictions')
    return prediction, fresh_sha


def verified_receipt(experiment, locked, horizon, folds):
    fit_records, evidence = {}, []
    for item in folds:
        if digest(item.original_path) != item.original_sha256 or digest(item.decision_path) != item.decision_sha256:
            raise ValueError('Locked reproduction inputs changed since planning')
        fresh, fresh_sha = _repeat_predictions(experiment, item.fold, item.namespace, item.chosen, fit_records)
        evidence.append({'fold': item.fold['fold'], 'original_sha256': item.original_sha256,
                         'decision_sha256': item.decision_sha256, 'repeated_sha256': fresh_sha,
                         'max_abs_difference': float(np.max(np.abs(fresh - item.original_prediction)))})
    if not fit_records:
        raise ValueError('Reproduction requires complete native fits')
    difference = max(item['max_abs_difference'] for item in evidence)
    return {'version': VERSION, 'lock_identity': content_id(locked), 'horizon': int(horizon),
            'folds': evidence, 'fit_records': fit_records, 'max_abs_log_return_difference': difference,
            'fresh_fits': True, 'status': 'passed' if difference <= 1e-6 else 'failed'}


def verify_for_export(experiment, locked):
    paths = {h: receipt_path(experiment, h) for h in ('1', '5')}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError('Bound reproduction receipt required before export')
    if set(locked['horizons']) != set(paths):
        raise ValueError('Bound reproduction requires both locked horizons')
    plans = plan(experiment, locked)
    receipts = {h: read_record(path) for h, path in paths.items()}
    for h, receipt in receipts.items():
        if receipt != verified_receipt(experiment, locked, h, plans[h]) or receipt['status'] != 'passed':
            raise ValueError('Bound reproduction receipt does not prove fresh parity for the locked choice')
    return receipts
