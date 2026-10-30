"""Read-only verification of the first9past selection+1in-sample state fits."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.ledger import Ledger, read_record

from research import ct_manual_forward as core
from research.verify_ct_signed_risk_score import independent_history


def verify(root, source):
    ready, _ = core.verify(root, source); lock = read_record(root/'cohort-lock.json')
    if lock['identity'] != ready['identity']:
        raise ValueError('Initial state cohort identity changed')
    ledger = Ledger(root/'ledger', {**ready['identity'], 'cohort_lock_id': content_id(lock)})
    count, keys = fit_consumption(ledger.root)
    if count != 10 or len(keys) != 10 or list((root/'forecasts').glob('*/forecast.json')):
        raise ValueError('This verifies exactly10initial fits and no future forecast')
    paths = [p for directory in ('ledger', 'fits', 'fit-plans', 'selections') for p in (root/directory).rglob('*') if p.is_file()]
    before = {p.relative_to(root).as_posix(): digest(p) for p in paths}
    seed = pd.read_parquet(root/'inputs/seed.parquet'); h = core.history(seed)
    independent_history(h)  # Decimal return/target/HAR/EWMA check, prefix causal.
    if seed.date.iloc[-1] >= pd.Timestamp(lock['first_origin']):
        raise ValueError('Initial state must not contain the first forecast origin bar')
    jobs = core.annual_jobs(h, lock['first_origin'])
    jobs += [{'year': int(lock['first_origin'][:4]), 'role': 'outer', 'bucket': 0,
              'cutoff': lock['first_origin'], 'origins': [], 'train_dates': core.mature(h, lock['first_origin'])}]
    dates = h.date.dt.strftime('%Y-%m-%d').tolist(); rows = []; state_checks = 0; past_predictions = 0
    for job in jobs:
        independently_mature = [dates[i] for i in range(119, len(dates)-5) if dates[i+5] < job['cutoff']]
        if job['train_dates'] != independently_mature:
            raise ValueError('Actual training violates independent5session maturity')
        role = f'{job["year"]}-{job["role"]}-{job["bucket"]}'
        plan = read_record(root/'fit-plans'/(role+'.json'))
        train = core.projection(h, job['train_dates'])
        expected = core.specification(job, train, 'initial_seed')
        if plan != expected:
            raise ValueError('Initial role/data identity changed')
        record = core.native_record(ledger, plan)
        pointer = read_record(root/'fits'/(role+'.json'))
        if pointer != {'fit_id': record['experiment_id'], 'specification_id': content_id(plan)}:
            raise ValueError('Checkpoint pointer changed')
        model_path = ledger.root/record['payload_root']/'variance-model.json'
        model = json.loads(model_path.read_bytes())
        if model['features'] != core.NAMES or model['recipe'] != plan['recipe'] or len(model['coef']) != 5:
            raise ValueError('Native model representation changed')
        tr = core.learner.relative(train, True, core.learner.STATE_PROFILE); x = tr[core.NAMES].to_numpy()
        eta = np.column_stack([np.ones(len(tr)), x])@model['coef']
        ratio = tr.variance_target_1.to_numpy()*np.exp(-eta)
        opt = model['optimization']; z = (x-opt['training_feature_mean'])/opt['training_feature_scale']
        gradient = np.column_stack([np.ones(len(tr)), z]).T@(1-ratio)/len(tr)
        norm = float(np.max(np.abs(gradient)))
        if not opt['success'] or norm > 1.001e-6 or not math.isclose(float(np.mean(eta+ratio)), opt['final_training_qlike'], abs_tol=1e-10):
            raise ValueError('Native optimizer/training loss not certified')
        test = train.tail(1).copy() if job['role'] == 'outer' else core.projection(h, job['origins'])
        test['variance_target_1'] = np.nan
        predictions = core.predict_model(ledger, record, test)
        np.testing.assert_allclose(predictions, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
        if record['result']['origins'] != test.date.dt.strftime('%Y-%m-%d').tolist() or record['result']['prediction_scope'] != plan['prediction_scope']:
            raise ValueError('In-sample state check must never be labeled future OOS')
        state_checks += int(job['role'] == 'outer'); past_predictions += len(test) if job['role'] != 'outer' else 0
        rows.append({'role': role, 'fit_id': record['experiment_id'], 'train_rows': len(train),
                     'model_sha256': digest(model_path), 'gradient_inf_norm': norm,
                     'checkpoint_predictions_checked': len(test), 'prediction_scope': plan['prediction_scope']})
    selection = core.verify_selection(root, ledger, h, int(lock['first_origin'][:4]), lock['first_origin'])
    if before != {p.relative_to(root).as_posix(): digest(p) for p in paths}:
        raise ValueError('Native state changed during read-only verification')
    return {'status': 'initial10_native_states_verified_no_forward_result', 'source_id': ready['identity']['source_id'],
            'design_id': ready['identity']['design_id'], 'cohort_lock_id': content_id(lock),
            'first_origin': lock['first_origin'], 'target_date': core.target_date(core.calendar(root), lock['first_origin']),
            'completed_market_fits': count, 'new_verification_fits': 0, 'new_synthetic_fits': 0,
            'past_validation_predictions_checked': past_predictions, 'in_sample_state_checks': state_checks,
            'new_forward_forecasts': 0, 'market_skill': False, 'weight': selection['weight'],
            'minimum_inner_train': min(r['train_rows'] for r in rows if r['prediction_scope'] == 'PAST_SELECTION_ONLY'),
            'outer_train': rows[-1]['train_rows'], 'native_files_unchanged': before, 'models': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.source), indent=2))
