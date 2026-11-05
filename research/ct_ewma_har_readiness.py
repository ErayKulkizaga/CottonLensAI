"""Source-bound CT full-year temporal readiness, no fit or training admission."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.research import named_risk_control as planner
from cottonlens_ml.research import protocol


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inspect(root, history, predictions):
    root.mkdir(parents=True, exist_ok=False)
    contract = {'profile': 'ct-ewma-har-full-year-readiness-v1', 'new_fits': 0,
                'history_sha256': sha(history), 'predictions_sha256': sha(predictions),
                'analysis_sha256': sha(Path(__file__)), 'planner_sha256': sha(Path(planner.__file__)),
                'protocol_sha256': sha(Path(protocol.__file__)),
                'candidate': 'fixed4parameter QLIKE HAR relative to unchanged origin EWMA0.94',
                'feature_formula': 'har_k-log(origin_ct_ewma)', 'target_formula': 'variance_target_1/origin_ct_ewma',
                'minimum_train': 500, 'warmup_row_index': 119, 'maturity': 'date<cut and target_date_5<cut',
                'validation': 'recent189 mature origins/3x63', 'refit': 'existing21Cotton-observation buckets',
                'years': list(range(2016, 2024)), 'decision': 'both20/60candidate-minusEWMA QLIKE CIuppers<0;>=6/8negative yearly delta;top1pctbaseline-loss trimdelta<0',
                'scope': 'CT continuous price squared-return proxy; different population from named2023 controls',
                'training_allowed': False, 'source_clock_verified': False, 'independent_holdout': False,
                'new_synthetic_or_market_fits': 0, 'pure_loss_contribution_claim_allowed': False}
    (root/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    original = pd.read_parquet(history); h = original.copy()
    reference = pd.read_csv(predictions).query('horizon==1').reset_index(drop=True)
    split = protocol.full_year_manifest(h)
    outer = [day for fold in split['folds'] for day in fold['origins']]
    if outer != reference.date.tolist():
        raise ValueError('Exact original2006-slot risk cohort required')
    baseline = np.maximum(h.daily_variance_proxy.ewm(alpha=.06, adjust=False).mean(), 1e-12)
    np.testing.assert_allclose(baseline.loc[h.date.dt.strftime('%Y-%m-%d').isin(outer)], reference.ewma_variance,
                               rtol=1e-12, atol=1e-14)
    # Planner compatibility only. No alias is written to data or sent to a
    # learner; neither original CT labels nor current quotes become named.
    h = h.assign(named_eligible=True, named_variance=h.variance_target_1,
                 named_label_available_at=pd.to_datetime(h.target_date_1, utc=True)+pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(h[original.columns], original, check_exact=True)
    folds, all_jobs = [], []
    for parent in split['folds']:
        past = planner.mature_dates(h, parent['origins'][0])[-189:]
        if len(past) != 189:
            raise ValueError('Three past63-observation validation blocks required')
        jobs = [job for i in range(3) for job in planner.control_jobs(h, past[i*63:(i+1)*63], f'inner_{i}')]
        jobs += planner.control_jobs(h, parent['origins'], 'outer')
        for job in jobs:
            train = protocol.rows_at(h, job['train_dates'])
            if len(train) < 500 or not (train.target_date_5 < pd.Timestamp(job['cutoff'])).all():
                raise ValueError('Past-only mature training boundary failed')
            if not np.isfinite(train[['har_1', 'har_5', 'har_22', 'variance_target_1']]).all().all():
                raise ValueError('Finite historical risk fields required')
        all_jobs.extend({'year': parent['year'], **job} for job in jobs)
        folds.append({'year': parent['year'], 'ordinal_origins': len(parent['origins']), 'market_jobs': len(jobs),
                      'minimum_train': min(len(j['train_dates']) for j in jobs), 'validation_origins': len(past)})
    result = {'profile': contract['profile'], 'status': 'TEMPORAL_DATA_FEASIBLE_NOT_TRAINING_AUTHORIZATION',
              'new_fits': 0, 'ordinal_origins': len(outer), 'folds': folds,
              'market_jobs_if_preregistered': len(all_jobs), 'minimum_train': min(f['minimum_train'] for f in folds),
              'training_allowed': False, 'candidate_predictions_computed': False,
              'contract_sha256': sha(root/'contract.json')}
    for name, value in [('split.json', split), ('jobs.json', all_jobs), ('result.json', result)]:
        (root/name).write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True); parser.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args(); inspect(args.root, args.history, args.predictions)
