"""Read-only first-publication provenance and numeric replay; never score forward losses."""

import argparse
import json
import math
import shutil
import socket
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record

from research import ct_forward_report as report
from research import ct_manual_forward as core
from research.ct_forward_daily import verify_execution

HELPERS = ('research/verify_ct_first_publication.py', 'research/tests/test_ct_first_publication.py',
           'research/ct_forward_report.py', 'research/ct_forward_daily.py',
           'research/verify_ct_signed_risk_score.py', 'research/verify_ct_prospective_readiness.py')


def replay(root, source):
    ready, design = core.verify(root, source)
    core.status(root, source)  # Existing event/payload/clock/target checks; no score.
    lock = read_record(root/'cohort-lock.json'); day = lock['first_origin']
    if (lock['identity'] != ready['identity'] or design['no_partial_score'] is not True or lock['required_origins'] != 252
            or day != core.first_origin(core.calendar(root), core.utc(lock['locked_at']))):
        raise ValueError('Different cohort identity or scoring policy')
    _, _, histories, observations = report.observation_history(root)
    forecast_path = root/'forecasts'/day/'forecast.json'
    forecast = read_record(forecast_path)
    pub = read_record(forecast_path.parent/'publication-receipt.json')
    if (forecast['origin'] != day or forecast['target_date'] != core.target_date(core.calendar(root), day)
            or forecast['horizon'] != 1 or forecast['ordinal_slot'] != 0
            or forecast['decision_at'] != core.decision(day).isoformat()
            or forecast['cohort_lock_id'] != content_id(lock) or forecast['market_skill'] is not False
            or pub['forecast_sha256'] != digest(forecast_path) or pub['status'] != 'TIMELY'):
        raise ValueError('Wrong first origin, target, horizon, cohort or publication receipt')
    oid = forecast['observation_id']; observation = observations[oid]; prices = histories[oid]
    stamp = core.decision(day)
    if (core.utc(observation['observed_available_at']) > stamp or observation['latest_completed_date'] != day
            or forecast['history_sha256'] != frame_identity(prices, list(prices))
            or not stamp <= core.utc(forecast['generated_at']) <= core.utc(pub['completed_at']) < stamp+pd.Timedelta(minutes=15)):
        raise ValueError('Wrong source availability, history or actual publication window')
    h = core.history(prices); checked = report.independent_history(h)
    ledger = Ledger(root/'ledger', {**ready['identity'], 'cohort_lock_id': content_id(lock)})
    consumed, keys = fit_consumption(ledger.root)
    completed = [read_record(p) for p in (ledger.root/'completed').glob('*.json')]
    if consumed != len(keys) or len(keys) != len(completed) or consumed > 31:
        raise ValueError('Excess, failed or incomplete fit state')
    record = read_record(ledger.root/'completed'/(forecast['fit_id']+'.json'))
    record = core.native_record(ledger, {k: v for k, v in record['specification'].items() if k != 'repeat_reason'})
    dates = h.date.dt.strftime('%Y-%m-%d').tolist()
    mature = [dates[i] for i in range(119, len(dates)-5) if dates[i+5] < day]
    if core.mature(h, day) != mature:
        raise ValueError('Independent five-session maturity differs')
    job = {'year': int(day[:4]), 'role': 'outer', 'bucket': 0, 'cutoff': day, 'origins': [], 'train_dates': mature}
    spec = core.specification(job, core.projection(h, mature), record['specification']['observation_id'])
    if record['specification'] != {**spec, 'repeat_reason': None} or core.utc(record['completed_at']) > stamp:
        raise ValueError('Different first training population or model completed after decision')
    training_oid = spec['observation_id']
    available = (json.loads((root/'inputs/initial-receipt.json').read_bytes())['observed_available_at']
                 if training_oid == 'initial_seed' else observations[training_oid]['observed_available_at'])
    if core.utc(available) > core.utc(record['completed_at']):
        raise ValueError('Training source arrived after model completion')
    selection = core.verify_selection(root, ledger, h, int(day[:4]), day)
    if (core.utc(selection['created_at']) > stamp or content_id(selection) != forecast['selection_id']
            or selection['weight'] != forecast['weight']):
        raise ValueError('Future or changed past-only selection')
    model_path = ledger.root/record['payload_root']/'variance-model.json'
    model = json.loads(model_path.read_bytes())
    if model['features'] != core.NAMES or model['recipe'] != spec['recipe'] or model['effective_parameters'] != 5 or len(model['coef']) != 5:
        raise ValueError('Different native representation or learner')
    train = core.learner.relative(core.projection(h, mature), True, core.learner.STATE_PROFILE)
    x_train = train[core.NAMES].to_numpy(); eta = np.column_stack([np.ones(len(train)), x_train])@model['coef']
    ratio = train.variance_target_1.to_numpy()*np.exp(-eta)
    opt = model['optimization']; z = (x_train-opt['training_feature_mean'])/opt['training_feature_scale']
    gradient = np.column_stack([np.ones(len(train)), z]).T@(1-ratio)/len(train)
    if (not opt['success'] or np.max(np.abs(gradient)) > 1.001e-6
            or not math.isclose(float(np.mean(eta+ratio)), opt['final_training_qlike'], abs_tol=1e-10)):
        raise ValueError('Uncertified native training loss or gradient')
    baseline = float(checked.loc[day, 'checked_ewma'])
    relative = [float(checked.loc[day, 'checked_'+n])-math.log(baseline) for n in ('har_1', 'har_5', 'har_22')]
    raw = baseline*math.exp(np.dot([1., *relative, math.log(baseline)], model['coef']))
    selected = selection['weight']*raw+(1-selection['weight'])*baseline
    np.testing.assert_allclose([baseline, raw, selected],
        [forecast['baseline_variance'], forecast['raw_variance'], forecast['selected_variance']], rtol=1e-10, atol=1e-14)
    return {'status': 'first_forward_publication_replayed_without_performance_score',
            'source_id': ready['identity']['source_id'], 'design_id': ready['identity']['design_id'],
            'cohort_lock_id': content_id(lock), 'origin': day, 'target_date': forecast['target_date'], 'horizon': 1,
            'source_observed_at': observation['observed_available_at'], 'decision_at': forecast['decision_at'],
            'generated_at': forecast['generated_at'], 'publication_completed_at': pub['completed_at'],
            'observation_id': oid, 'forecast_id': content_id(forecast), 'forecast_sha256': digest(forecast_path),
            'fit_id': record['experiment_id'], 'model_sha256': digest(model_path), 'model_completed_at': record['completed_at'],
            'selection_created_at': selection['created_at'], 'training_source_available_at': available,
            'mature_training_rows': len(mature), 'history_rows': len(prices), 'completed_market_fits': consumed,
            'baseline_variance': baseline, 'raw_variance': raw, 'selected_variance': selected, 'weight': selection['weight'],
            'new_fits': 0, 'new_forecasts': 0, 'performance_scored': False, 'market_skill': False,
            'actual_forward_target_or_loss_reported': False, 'first_historical_vintage_verified': False}


def run_receipts(root, runtime, expected_execution, result):
    execution = verify_execution(runtime, expected_execution)
    if execution['core_source_id'] != result['source_id']:
        raise ValueError('Daily execution core differs from the replayed model source')
    matched = {}
    for path in sorted((root/'daily-runs').glob('*/receipt.json')):
        run = read_record(path); item = run.get('result') or {}; phase = item.get('phase')
        if item.get('origin') != result['origin'] or phase not in ('capture', 'publish'):
            continue
        if phase in matched or run['run_id'] != path.parent.name:
            raise ValueError('Ambiguous actual first-origin execution')
        if (run['exit_code'] != 0 or run['error'] is not None or run['phase_requested'] != 'auto'
                or run['execution_source_id'] != expected_execution or run['core_source_id'] != result['source_id']
                or run['runner_sha256'] != digest(runtime/'ct_forward_daily.py') or run['console_requested'] is not False):
            raise ValueError('Failed or changed actual consoleless execution')
        if phase == 'capture':
            if (item['status'] != 'CAPTURED_AND_STATE_READY' or item['observation_id'] != result['observation_id']
                    or item['state']['fit_id'] != result['fit_id'] or item['state']['market_fits'] != result['completed_market_fits']
                    or item['state']['weight'] != result['weight']
                    or not core.utc(run['started_at']) <= core.utc(result['source_observed_at']) <= core.utc(run['completed_at']) <= core.utc(result['decision_at'])):
                raise ValueError('Capture execution/source/cutoff link differs')
        elif (item['status'] != 'TIMELY' or item['new_fits'] != 0 or item['forward_record_id'] != result['forecast_id']
                or item['forecast_sha256'] != result['forecast_sha256']
                or item['publication_receipt_sha256'] != digest(root/'forecasts'/result['origin']/'publication-receipt.json')
                or not core.utc(run['started_at']) <= core.utc(result['generated_at']) <= core.utc(result['publication_completed_at']) <= core.utc(run['completed_at'])):
            raise ValueError('Publication execution/payload link differs')
        matched[phase] = {'run_id': run['run_id'], 'receipt_sha256': digest(path)}
    if set(matched) != {'capture', 'publish'}:
        raise ValueError('Both real scheduled capture and publication executions are required')
    return {'execution_source_id': expected_execution, 'runs': matched}


def audit(root, source, runtime, expected_execution, output):
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Use a new output outside the preserved study')
    if (root/'.writer-lock').exists() or (root/'operations/.writer-lock').exists():
        raise RuntimeError('A real writer is active or interrupted; do not audit partial state')
    before = report.study_hashes(root)
    denial = AssertionError('Publication audit cannot train, optimize, collect, publish or score forward losses')
    with ExitStack() as stack:
        for target, name in ((Ledger, 'run'), (report, 'score_pairs'), (report, 'extract'),
                             (core, 'capture'), (core, 'fit_for_origin'), (core, 'publish'), (socket.socket, 'connect')):
            stack.enter_context(patch.object(target, name, side_effect=denial))
        stack.enter_context(patch('scipy.optimize.minimize', side_effect=denial))
        result = replay(root, source)
        result.update(run_receipts(root, runtime, expected_execution, result))
    if before != report.study_hashes(root):
        raise ValueError('Preserved inputs changed during read-only publication audit')
    analysis_repo = Path(__file__).resolve().parents[1]
    analysis = {'core_source': core.source(source), 'execution_source_id': expected_execution,
                'helpers': {name: digest(analysis_repo/name) for name in HELPERS}}
    result['analysis_source_id'] = content_id(analysis)
    output.mkdir(parents=True)
    freeze_record(output/'input-files.json', {'files': before})
    freeze_record(output/'analysis-manifest.json', analysis)
    for name in HELPERS:
        destination = output/'source-snapshot'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(analysis_repo/name, destination)
    freeze_record(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'runtime', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-execution', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.source, args.runtime, args.expected_execution, args.output), indent=2))
