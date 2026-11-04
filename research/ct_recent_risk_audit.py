"""Fixed walk-forward audit on the existing post-2023 Cotton history."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.volatility import qlike

from research import ct_manual_forward as forward
from research import ct_shock_control as learner
from research.verify_ct_prospective_readiness import decimal_difference
from research.verify_ct_signed_risk_score import independent_history
from research.verify_risk_weak_teacher import difference_interval

PROFILE = 'ct-own-level-recent-history-audit-v1'
SEED = 'b4f16662a5a717bf999fdcec511f4c3b7bb916749c415915a46e7dbf7464fa90'
HELPERS = ('research/ct_recent_risk_audit.py', 'research/verify_ct_signed_risk_score.py',
           'research/verify_ct_prospective_readiness.py')


def plan(history):
    folds, jobs = [], []
    for year in (2024, 2025, 2026):
        origins = history.loc[history.date.dt.year.eq(year) & history.target_date_1.notna(),
                              'date'].dt.strftime('%Y-%m-%d').tolist()
        if not origins:
            raise ValueError('All three calendar years must be represented')
        folds.append({'year': year, 'origins': origins})
        jobs.extend(forward.annual_jobs(history, origins[0]))
        for start in range(0, len(origins), 21):
            cutoff = origins[start]
            jobs.append({'year': year, 'role': 'outer', 'bucket': start//21, 'cutoff': cutoff,
                         'train_dates': forward.mature(history, cutoff), 'origins': origins[start:start+21]})
    return {'split': {'folds': folds}, 'jobs': jobs}


def prepare(repo, seed, root):
    if root.exists() or digest(seed) != SEED:
        raise ValueError('Use the original verified pre-cohort seed and a new namespace')
    h = forward.history(pd.read_parquet(seed))
    if str(h.date.iloc[-1].date()) != '2026-10-29':
        raise ValueError('Unexpected final seed date')
    independent_history(h)
    design = plan(h)
    core = forward.source(repo)
    names = set(core['ml']['files']) | set(core['helpers']) | set(HELPERS)
    sources = {name: digest(repo/name) for name in sorted(names)}
    contract = {'profile': PROFILE, 'candidate': learner.recipe('own_level', learner.STATE_PROFILE),
                'baseline': 'EWMA 0.94', 'target': 'next observed CT=F squared log return',
                'scope': 'All observed origins in 2024 through 2026-10-28 with targets through 2026-10-29',
                'fit_budget': len(design['jobs']), 'threads': 2, 'session_minutes': 30,
                'split': 'Expanding history, original warmup119, fifth target date strictly before cutoff',
                'selection': 'Fixed annual 3x63 past-only selection, weights 0/.25/.5/.75/1, smallest tie',
                'refit': '21 observed origins per outer fit, reset annually',
                'decision': 'Support in this historical audit requires mean paired gain and both block20 and block60 95% lower bounds above zero.',
                'sensitivity': 'Yearly gains and omission of top 1% of EWMA losses. Report regardless of outcome.',
                'independent_holdout': False, 'historical_vintage_verified': False,
                'forward_publication': False, 'release_allowed': False, 'seed_sha256': SEED}
    with writer(root):
        shutil.copyfile(seed, root/'seed.parquet')
        for name in sources:
            out = root/'source-snapshot'/name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/name, out)
        freeze_record(root/'source-manifest.json', sources)
        freeze_record(root/'design.json', design)
        freeze_record(root/'decision-contract.json', contract)
        freeze_record(root/'ready.json', {'profile': PROFILE, 'design_id': content_id(design),
                      'contract_id': content_id(contract), 'source_id': content_id(sources), **forward.environment()})
    return {'market_fits_planned': len(design['jobs']),
            'origins': sum(len(f['origins']) for f in design['split']['folds'])}


def inputs(root):
    source = root/'source-snapshot'
    if Path(__file__).resolve() != (source/'research/ct_recent_risk_audit.py').resolve():
        raise ValueError('Execute the frozen source snapshot')
    manifest = read_record(root/'source-manifest.json')
    design = read_record(root/'design.json')
    contract = read_record(root/'decision-contract.json')
    identity = read_record(root/'ready.json')
    if (identity['source_id'] != content_id(manifest) or identity['design_id'] != content_id(design)
            or identity['contract_id'] != content_id(contract) or digest(root/'seed.parquet') != SEED
            or any(digest(source/p) != sha for p, sha in manifest.items())
            or any(identity[k] != v for k, v in forward.environment().items())):
        raise ValueError('Audit identity changed')
    h = forward.history(pd.read_parquet(root/'seed.parquet'))
    if plan(h) != design or contract['fit_budget'] != len(design['jobs']):
        raise ValueError('The planned origin/training population differs')
    return identity, design, h


def run(root, fit):
    identity, design, h = inputs(root)
    ledger = Ledger(root/'ledger', identity)
    deadline = time.monotonic()+30*60
    arm, profile = 'own_level', learner.STATE_PROFILE

    def prediction(job):
        train = forward.projection(h, job['train_dates'])
        test = forward.projection(h, job['origins'])
        test['variance_target_1'] = np.nan
        spec = learner.specification(job, train, arm, profile)
        if fit:
            if time.monotonic() >= deadline:
                raise TimeoutError('30-minute audit limit reached')
            ledger.run(spec, lambda path: learner.fit(train, test, path, arm, profile))
        return learner.native(ledger, job, train, test, arm, profile)

    with writer(root):
        forecasts, weights = learner.folded_output(h, design, prediction)
        if ledger.summary()['durably_saved'] != len(design['jobs']):
            raise ValueError('Require exact native fit coverage')
        check = independent_history(h)
        days = [day for fold in design['split']['folds'] for day in fold['origins']]
        if list(forecasts) != days:
            raise ValueError('Forecast origin coverage differs')
        rows = check.loc[days]
        actual = rows.checked_actual.to_numpy()
        baseline = rows.checked_ewma.to_numpy()
        model = np.array([forecasts[day][1] for day in days])
        gains = [decimal_difference(y, b, m) for y, b, m in zip(actual, baseline, model, strict=True)]
        np.testing.assert_allclose(gains, qlike(actual, baseline)-qlike(actual, model), atol=1e-12)
        frame = pd.DataFrame({'origin': days, 'target_date': rows.target_date_1.dt.strftime('%Y-%m-%d').to_list(),
                              'year': rows.date.dt.year.to_numpy(), 'actual': actual,
                              'ewma': baseline, 'candidate': model, 'gain': gains,
                              'ewma_loss': qlike(actual, baseline)})
        by_year = frame.groupby('year')
        parts = [np.column_stack([part.gain, np.ones(len(part))]) for _, part in by_year]
        intervals = {str(block): difference_interval(parts, block) for block in (20, 60)}
        trimmed = frame.drop(frame.nlargest(math.ceil(len(frame)*.01), 'ewma_loss').index)
        result = {'origins': len(frame), 'fits': len(design['jobs']), 'mean_gain': float(frame.gain.mean()),
                  'intervals': intervals, 'years': {str(k): {'origins': len(g), 'gain': float(g.gain.mean())}
                  for k, g in by_year}, 'weights': weights, 'trimmed_gain': float(trimmed.gain.mean()),
                  'historical_support': bool(frame.gain.mean() > 0 and all(ci[0] > 0 for ci in intervals.values())),
                  'independent_holdout': False, 'forward_publication': False, 'market_skill': False}
        path = root/'paired-losses.csv'
        if path.exists():
            saved = pd.read_csv(path, float_precision='round_trip', dtype={'year': frame.year.dtype})
            pd.testing.assert_frame_equal(saved, frame, check_exact=True)
            if read_record(root/'result.json') != result:
                raise ValueError('Saved audit differs on replay')
        else:
            frame.to_csv(path, index=False, lineterminator='\n')
            freeze_record(root/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run', 'replay'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=Path)
    args = parser.parse_args()
    result = prepare(Path.cwd(), args.seed, args.root) if args.phase == 'prepare' else run(args.root, args.phase == 'run')
    print(json.dumps(result, indent=2))
