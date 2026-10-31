"""Analytic fixtures only: no fitted models, provider calls or real forecast scores."""

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from cottonlens_ml.research.ledger import freeze_record, read_record

from research import verify_ct_first_publication as module
from research.tests.test_ct_forward_report import analytic_study


def replace_record(path, body):
    body = {k: v for k, v in body.items() if k != 'record_id'}
    path.write_text(json.dumps({**body, 'record_id': module.content_id(body)}))


def fixture(root):
    identity = analytic_study(root)
    path = root/'forecasts/2026-10-30/forecast.json'; f = read_record(path)
    f.update(ordinal_slot=0, decision_at='2026-10-31T00:15:00+00:00', market_skill=False)
    replace_record(path, f)
    relink(root)
    return identity


def relink(root):
    path = root/'forecasts/2026-10-30/publication-receipt.json'; pub = read_record(path)
    pub['forecast_sha256'] = module.digest(path.parent/'forecast.json'); replace_record(path, pub)
    event_path = root/'events/000.json'; event = read_record(event_path)
    event.update(forecast_sha256=pub['forecast_sha256'], publication_receipt_sha256=module.digest(path))
    replace_record(event_path, event)


@contextmanager
def boundary(identity):
    # Fixture intentionally cannot pass the real frozen-code/source boundary.
    with (patch.object(module.core, 'verify', return_value=({'identity': identity}, {'no_partial_score': True, 'required_origins': 252})),
          patch.object(module.core, 'now', return_value=module.core.utc('2028-01-01T12:00Z'))):
        yield


def test_analytic_first_publication_is_replayed_without_target_or_loss(tmp_path):
    identity = fixture(tmp_path)
    with boundary(identity), patch.object(module.report, 'score_pairs', side_effect=AssertionError('No forward scoring')):
        result = module.replay(tmp_path, tmp_path)
    assert result['origin'] == '2026-10-30' and result['target_date'] == '2026-11-02'
    assert result['completed_market_fits'] == 10
    assert result['performance_scored'] is result['actual_forward_target_or_loss_reported'] is result['market_skill'] is False
    assert result['new_fits'] == result['new_forecasts'] == 0
    assert not {'actual_variance', 'loss', 'gain', 'direction_accuracy'} & set(result)


@pytest.mark.parametrize('defect', ['target', 'horizon', 'early_publication', 'wrong_prediction', 'native_gradient'])
def test_self_consistent_payloads_cannot_hide_scientific_errors(tmp_path, defect):
    identity = fixture(tmp_path)
    p = tmp_path/'forecasts/2026-10-30/forecast.json'; forecast = read_record(p)
    if defect == 'native_gradient':
        path = tmp_path/'ledger/completed'/(forecast['fit_id']+'.json'); record = read_record(path)
        model_path = tmp_path/'ledger'/record['payload_root']/'variance-model.json'
        model = json.loads(model_path.read_bytes()); model['optimization']['success'] = False
        model_path.write_text(json.dumps(model))
        record['files'][model_path.relative_to(tmp_path/'ledger').as_posix()] = module.digest(model_path)
        replace_record(path, record)
    else:
        forecast.update({'target': {'target_date': '2026-11-03'}, 'horizon': {'horizon': 5},
            'early_publication': {'generated_at': '2026-10-31T00:14Z'},
            'wrong_prediction': {'raw_variance': 1.}}[defect])
        replace_record(p, forecast); relink(tmp_path)
    with boundary(identity), pytest.raises((ValueError, AssertionError)):
        module.replay(tmp_path, tmp_path)


def test_actual_runner_links_require_both_phases_and_matching_execution(tmp_path):
    root = tmp_path/'study'; root.mkdir(); identity = fixture(root)
    with boundary(identity):
        result = module.replay(root, root)
    runtime = tmp_path/'runtime'; runtime.mkdir(); runner = runtime/'ct_forward_daily.py'; runner.write_bytes(b'analytic fixture')
    body = {'files': {runner.name: module.digest(runner)}, 'core_source_id': identity['source_id']}
    eid = module.content_id(body)
    (runtime/'execution-manifest.json').write_text(json.dumps({**body, 'execution_source_id': eid}))
    for phase, started, ended in [('capture', '00:04Z', '00:06Z'), ('publish', '00:20Z', '00:21Z')]:
        item = {'phase': phase, 'origin': result['origin']}
        if phase == 'capture':
            item.update(status='CAPTURED_AND_STATE_READY', observation_id=result['observation_id'],
                        state={'fit_id': result['fit_id'], 'market_fits': result['completed_market_fits'], 'weight': result['weight']})
        else:
            item.update(status='TIMELY', new_fits=0, forward_record_id=result['forecast_id'],
                        forecast_sha256=result['forecast_sha256'], publication_receipt_sha256=module.digest(root/'forecasts/2026-10-30/publication-receipt.json'))
        freeze_record(root/'daily-runs'/phase/'receipt.json', {'run_id': phase, 'result': item,
            'started_at': '2026-10-31T'+started, 'completed_at': '2026-10-31T'+ended,
            'execution_source_id': eid, 'core_source_id': identity['source_id'], 'phase_requested': 'auto',
            'exit_code': 0, 'error': None, 'runner_sha256': module.digest(runner), 'console_requested': False})
    assert set(module.run_receipts(root, runtime, eid, result)['runs']) == {'capture', 'publish'}
    path = root/'daily-runs/publish/receipt.json'; bad = read_record(path); bad['result']['forward_record_id'] = 'wrong'
    replace_record(path, bad)
    with pytest.raises(ValueError, match='Publication execution'):
        module.run_receipts(root, runtime, eid, result)


def test_no_output_overwrite_or_inspection_of_active_writer(tmp_path):
    study = tmp_path/'study'; study.mkdir()
    with pytest.raises(ValueError, match='new output'):
        module.audit(study, study, study, 'unused', study/'child')
    (study/'.writer-lock').mkdir()
    with pytest.raises(RuntimeError, match='writer'):
        module.audit(study, study, study, 'unused', tmp_path/'new-output')
