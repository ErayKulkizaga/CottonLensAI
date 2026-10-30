"""Independent ordinal maturity and prospective preflight; no fits or future scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.verify_risk_weak_teacher import fingerprint, read, sha


def verify(root):
    ready = read(root/'ready.json'); design = read(root/'decision-contract.json')
    manifest = read(root/'source-manifest.json'); identity = ready['identity']
    body = {k: v for k, v in manifest.items() if k != 'source_id'}
    if fingerprint(body) != identity['source_id'] or fingerprint(design) != identity['design_id']:
        raise ValueError('Frozen prospective identity changed')
    for name, checksum in {**manifest['ml']['files'], **manifest['helpers']}.items():
        if sha(root/'source-snapshot'/name) != checksum:
            raise ValueError('Frozen prospective source changed')
    for name, checksum in ready['frozen_hashes'].items():
        if sha(root/name) != checksum:
            raise ValueError('Prospective input/native payload changed')
    proof = read(root/'inputs/source-proof.json')
    seed = pd.read_parquet(root/'inputs/seed.parquet')
    if len(seed) != 4232 or sha(root/'inputs/seed.parquet') != proof['proposed_seed_sha256']:
        raise ValueError('Wrong initial history vintage')
    if (design['required_origins'] != 252 or design['fit_budget']['market'] != 31
            or design['fit_budget']['synthetic'] != 0 or design['automatic_tasks']
            or design['decision_utc'] != '00:15' or design['publication_end_utc'] != '00:30'):
        raise ValueError('Prospective capacity, clock or sample budget changed')
    if design['recipe']['parameters'] != 5 or design['recipe']['arm'] != 'own_level':
        raise ValueError('Different candidate')
    # Integer positions establish the original five-observation purge independently.
    dates = seed.date.dt.strftime('%Y-%m-%d').tolist(); first = '2026-10-30'
    eligible = [dates[i] for i in range(119, len(dates)-5) if dates[i+5] < first]
    past = eligible[-189:]; jobs = []; counts = []
    for start in range(0, 189, 21):
        cutoff = past[start]
        train = [dates[i] for i in range(119, len(dates)-5) if dates[i+5] < cutoff]
        if len(train) < 500:
            raise ValueError('Insufficient fixed mature train')
        jobs.append({'cutoff': cutoff, 'train_dates': train, 'origins': past[start:start+21]})
        counts.append(len(train))
    fixture = read(root/'learning-control.json')
    if fixture['held_predictions_verified'] != 200 or fixture['new_synthetic_fits'] != 0:
        raise ValueError('Missing exact native learning reuse')
    returns = np.log(seed.cotton_close.to_numpy()[1:]/seed.cotton_close.to_numpy()[:-1])
    # This only checks retained source values, never an outcome or forecast comparison.
    if not np.isfinite(returns).all():
        raise ValueError('Invalid source return')
    return {'status': 'independent_zero_fit_manual_protocol_preflight', 'source_id': identity['source_id'],
            'design_id': identity['design_id'], 'new_fits': 0, 'forward_forecasts': 0, 'cohort_locked': False,
            'seed_rows': len(seed), 'warmup_date': dates[119], 'illustrative_first_origin': first,
            'actual_first_origin_chosen_only_at_real_lock': True,
            'initial_annual_validation_origins': past, 'initial_inner_jobs': jobs,
            'initial_outer_train_dates': eligible, 'minimum_inner_train': min(counts),
            'reused_learning_predictions': 200, 'power_guaranteed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
