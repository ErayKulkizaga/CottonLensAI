"""Matched T+5 label control; historical clocks/vintages remain assumptions."""
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.contract_curve import groups
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.full_year import SHRINKAGE, chunks
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.recency import copy_immutable

PROFILE = 'named-label-control-t5-v1'
ARMS = ('ct', 'named')
RECIPE = {'family': 'ridge', 'horizon': 5, 'device': 'cpu', 'seed': 42,
          'task': 'price', 'target': 'scaled_log', 'loss': 'reg:squarederror',
          'params': {'alpha': 1.}, 'window': 1, 'cadence': 21, 'years': None,
          'features': groups()['numeric_D0']}


def attach(history, labels):
    """Metadata only: keep all parent rows/features/targets unchanged."""
    h = history.reset_index(drop=True).copy()
    lab = labels.loc[labels['rank'].eq('second') & labels.horizon.eq(5)].reset_index(drop=True)
    dates = h.date.dt.strftime('%Y-%m-%d').tolist()
    if (len(lab) != len(h) or lab.date.tolist() != dates
            or not pd.to_datetime(lab.target_date).equals(h.target_date_5)):
        raise ValueError('Exact original origin/target order required; no silent intersection')
    if any(str(c).startswith('named_') for c in h):
        raise ValueError('Named metadata already present')
    columns = {'selected_contract': 'named_contract', 'origin_reference_price': 'named_price',
               'target_price': 'named_target_price', 'target_quote_available_at': 'named_label_available_at',
               'origin_quote_available_at': 'named_origin_available_at',
               'contemporaneous_log_return': 'named_return',
               'contemporaneous_assumption_eligible': 'named_eligible',
               'contemporaneous_reason': 'named_reason'}
    for original, name in columns.items():
        h[name] = lab[original].to_numpy()
    for name in ('named_origin_available_at', 'named_label_available_at'):
        h[name] = pd.to_datetime(h[name], utc=True)
    if not h.named_eligible.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError('Eligibility must be explicit boolean, not string truthiness')
    valid = h.named_eligible
    decision = h.date.dt.tz_localize('UTC') + pd.Timedelta(days=1, minutes=15)
    if (not np.isfinite(h.loc[valid, ['named_price', 'named_target_price', 'named_return']]).all().all()
            or not h.loc[valid, ['named_price', 'named_target_price']].gt(0).all().all()
            or h.loc[valid, 'named_contract'].isna().any()
            or not h.loc[valid, 'named_origin_available_at'].le(decision.loc[valid]).all()):
        raise ValueError('Eligible named label has invalid price/name/origin clock')
    np.testing.assert_allclose(h.loc[valid, 'named_return'],
                               np.log(h.loc[valid, 'named_target_price'] / h.loc[valid, 'named_price']),
                               rtol=0, atol=1e-14)
    pd.testing.assert_frame_equal(h[list(history)], history.reset_index(drop=True), check_exact=True)
    return h


def mature_dates(history, cutoff):
    """Common CT/named population; target quote must have arrived by refit clock."""
    cutoff = pd.Timestamp(cutoff)
    decision = cutoff.tz_localize('UTC') + pd.Timedelta(days=1, minutes=15)
    ok = (history.named_eligible & history.date.lt(cutoff) & history.target_date_5.lt(cutoff)
          & history.date.ge(history.date.iloc[119]) & history.named_label_available_at.le(decision)
          & np.isfinite(history.target_return_5) & np.isfinite(history.cotton_close) & history.cotton_close.gt(0))
    return history.loc[ok, 'date'].dt.strftime('%Y-%m-%d').tolist()


def jobs(history, origins, role):
    start = int(rows_at(history, origins).cotton_session_index.iloc[0])
    calendar = history.set_index('cotton_session_index').date
    result = []
    for chunk in chunks(history, origins):
        bucket = (int(chunk.cotton_session_index.iloc[0]) - start) // 21
        cutoff = calendar.loc[start + bucket * 21].strftime('%Y-%m-%d')
        result.append({'role': role, 'bucket': bucket, 'cutoff': cutoff,
                       'origins': chunk.date.dt.strftime('%Y-%m-%d').tolist(),
                       'train_dates': mature_dates(history, cutoff)})
    return result


def plan(history, parent_split):
    readiness, folds = [], []
    for original in parent_split['folds']:
        past = mature_dates(history, original['origins'][0])[-189:]
        inner = [past[i:i+63] for i in range(0, len(past), 63)]
        all_jobs = [j for i, block in enumerate(inner) for j in jobs(history, block, f'inner_{i}')]
        all_jobs += jobs(history, original['origins'], 'outer')
        supported = (len(past) == 189 and len(inner) == 3
                     and all(len(b) == 63 for b in inner) and all(len(j['train_dates']) >= 500 for j in all_jobs))
        readiness.append({'year': original['year'], 'supported': supported,
                          'minimum_train_rows': min(len(j['train_dates']) for j in all_jobs)})
        if supported:
            folds.append({'year': original['year'], 'origins': original['origins'],
                          'inner': inner, 'jobs': all_jobs})
    return {'profile': PROFILE, 'recipe': RECIPE, 'arms': list(ARMS), 'folds': folds, 'readiness': readiness,
            'fit_budget': {'market': 2 * sum(len(f['jobs']) for f in folds), 'synthetic': 1,
                           'threads': 2, 'processes': 1, 'session_minutes': 30},
            'primary': 'raw named vs CT training label, paired named-contract price-MAE',
            'secondary': 'past-only selected shrinkage; named-contract Naive; direction; block20/60',
            'selection': 'Latest 189 past source-mature labels, 3x63; equal mean relative price-MAE; smallest-weight tie',
            'clock': 'Cotton date+1 day00:15UTC; archive publication calendar+1 day00:00UTC assumed; final vintage assumed',
            'gate_evaluation_allowed': False, 'release_allowed': False, 'historical_source_admitted': False,
            'source_timestamp_verified': False, 'first_vintage_verified': False,
            'deliberate_repeat_reason': 'Frozen CT-return transport failed; isolate supervised label at common sample/model/features'}


def projection(history, dates, arm, *, test=False):
    if arm not in ARMS:
        raise ValueError('Unknown label arm')
    rows = rows_at(history, dates).copy()
    if test:
        # Target is identical in both arms. Missing actuals are never imputed.
        rows['cotton_close'] = rows.named_price
        rows['target_return_5'] = rows.named_return.where(rows.named_eligible)
    elif arm == 'named':
        rows['target_return_5'] = rows.named_return
    if not test and (not rows.named_eligible.all() or not np.isfinite(rows.target_return_5).all()):
        raise ValueError('Training requires common, finite eligible labels')
    return rows


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder / 'ready.json')
    identity = ready['identity']
    if identity['profile'] != PROFILE or identity['source_id'] != research_source_identity(repo)['source_id']:
        raise ValueError('Source/profile changed; preserve cache and choose new namespace')
    actual = {p: importlib.metadata.version(p) for p in identity['versions']}
    if actual != identity['versions'] or sys.version.split()[0] != identity['python']:
        raise ValueError('Frozen environment changed')
    if digest(repo / 'ml/uv.lock') != identity['lock_sha256']:
        raise ValueError('Dependency lock changed')
    for name, checksum in identity['inputs_sha256'].items():
        if digest(folder / 'inputs' / name) != checksum:
            raise ValueError('Frozen input changed')
    h = attach(pd.read_parquet(folder / 'inputs/history.parquet'), pd.read_csv(folder / 'inputs/labels.csv'))
    split = read_record(folder / 'inputs/parent-ready.json')['identity']['split']
    design = plan(h, split)
    if (design != identity['design'] or content_id(design) != identity['design_id']
            or not design['folds'] or digest(folder / 'history.parquet') != ready['history_sha256']
            or frame_identity(h, list(h)) != identity['research_data_id']):
        raise ValueError('Frozen cohort/data/design changed')
    pd.testing.assert_frame_equal(pd.read_parquet(folder / 'history.parquet'), h, check_exact=True)
    return ready, h


def prepare(repo, folder, registration_root, decision_contract):
    repo, folder, registration_root, decision_contract = map(Path, (repo, folder, registration_root, decision_contract))
    ready, _ = verify(registration_root, repo)
    public = read_record(repo / 'research/evidence/named-label-control-preregistration-20261010.json')
    if (public['ready_sha256'] != digest(registration_root / 'ready.json')
            or public['decision_contract_sha256'] != digest(decision_contract)
            or read_record(decision_contract) != ready['identity']['design'] or public['new_fits'] != 0):
        raise ValueError('Public preregistration/decision contract mismatch')
    with writer(folder):
        if (folder / 'ready.json').exists():
            old, _ = verify(folder, repo)
            if old != ready:
                raise ValueError('Existing execution identity differs; no overwrite')
            return old
        if any(p.name != '.writer-lock' for p in folder.iterdir()):
            raise ValueError('Preserve partial preparation; choose new namespace')
        for path in registration_root.rglob('*'):
            if path.is_file():
                copy_immutable(path, folder / path.relative_to(registration_root))
        copy_immutable(decision_contract, folder / 'decision-contract.json')
    verify(folder, repo)
    return ready


def make_output(history, fold, arm, design, records):
    predictions, fit_ids = {}, {}
    for job, record in zip(fold['jobs'], records, strict=True):
        values = record['result']['predictions']
        if (record['result']['origins'] != job['origins'] or len(values) != len(job['origins'])
                or not np.isfinite(values).all() or set(predictions).intersection(job['origins'])):
            raise ValueError('Fit predictions must be finite, exact, disjoint original origins')
        predictions.update(zip(job['origins'], values, strict=True))
        fit_ids[f'{job["role"]}-{job["bucket"]}'] = record['experiment_id']
    scores = []
    for weight in SHRINKAGE:
        relative = []
        for block in fold['inner']:
            test = rows_at(history, block)
            actual, price = test.named_target_price.to_numpy(), test.named_price.to_numpy()
            baseline = np.abs(actual - price).mean()
            if not np.isfinite(baseline) or baseline <= 0:
                raise ValueError('Finite nonzero named validation baseline required')
            pred = np.array([predictions[d] for d in block])
            relative.append(float(np.abs(actual - price * np.exp(weight * pred)).mean() / baseline))
        scores.append(float(np.mean(relative)))
    weight = SHRINKAGE[int(np.argmin(scores))]
    return {'year': fold['year'], 'arm': arm, 'design_id': content_id(design),
            'weight': weight, 'scores': scores, 'fit_ids': fit_ids,
            'inner_predictions': {d: predictions[d] for b in fold['inner'] for d in b},
            'origins': fold['origins'], 'raw_predictions': [predictions[d] for d in fold['origins']],
            'selected_predictions': [weight * predictions[d] for d in fold['origins']]}


def verify_output(experiment, history, fold, arm):
    records = []
    for job in fold['jobs']:
        train = projection(history, job['train_dates'], arm)
        expected = {'recipe': RECIPE, 'role': f'{arm}-{fold["year"]}-{job["role"]}-{job["bucket"]}',
                    'iterations': 1, 'train_dates': job['train_dates'], 'validation_dates': [],
                    'test_dates': job['origins'], 'train_identity': frame_identity(train, list(train)),
                    'repeat_reason': None}
        key = experiment.ledger.key(expected)
        record = read_record(experiment.ledger.root / 'completed' / f'{key}.json')
        if record['specification'] != expected or record['identity'] != experiment.identity or record['experiment_id'] != key:
            raise ValueError('Fit checkpoint identity differs from frozen label control')
        if not record['files']:
            raise ValueError('Completed fit without native payload is invalid')
        for name, checksum in record['files'].items():
            if not safe_member(name) or digest(experiment.ledger.root / name) != checksum:
                raise ValueError('Fit payload missing or corrupted')
        records.append(record)
    expected = make_output(history, fold, arm, experiment.identity['design'], records)
    path = experiment.root / 'label-outputs' / f'{arm}-{fold["year"]}.json'
    if read_record(path) != expected:
        raise ValueError('Stored output/selection differs from exact frozen fit receipts')
    return expected


def run(experiment, repo, max_minutes):
    from cottonlens_ml.research import wasde_regional_execution as shared
    if not 0 < max_minutes <= 30:
        raise ValueError('One CPU session must be <=30 minutes')
    ready, history = verify(experiment.root, repo)
    design = ready['identity']['design']
    learning = experiment.root / 'learning-control-ledger'
    used, reusable = fit_consumption(learning)
    if used > 1 or (used and not reusable):
        raise ValueError('Synthetic attempt consumed; no automatic extra fit')
    shared.learning_control(experiment, horizon=5, recipe_fn=lambda *_: RECIPE)
    shared.verify_learning_evidence(experiment.root, ready['identity'], horizon=5)
    deadline = time.monotonic() + max_minutes * 60
    used, _ = fit_consumption(experiment.root / 'ledger')
    count = [used]
    def before():
        if time.monotonic() >= deadline or count[0] >= design['fit_budget']['market']:
            raise FitBudgetReached('Frozen time/fit budget; checkpoint retained')
        count[0] += 1
    experiment.before_compute = before
    try:
        for fold in design['folds']:
            for arm in ARMS:
                output = experiment.root / 'label-outputs' / f'{arm}-{fold["year"]}.json'
                if output.exists():
                    verify_output(experiment, history, fold, arm)
                    continue
                records = []
                for job in fold['jobs']:
                    train = projection(history, job['train_dates'], arm)
                    test = projection(history, job['origins'], arm, test=True)
                    record = experiment.fit(RECIPE, train, None, test,
                                             f'{arm}-{fold["year"]}-{job["role"]}-{job["bucket"]}', iterations=1)
                    records.append(record)
                freeze_record(output, make_output(history, fold, arm, design, records))
                verify_output(experiment, history, fold, arm)
    except FitBudgetReached:
        return {'status': 'planned_pause', 'completed_market_fits': len(experiment.ledger.results())}
    if len(experiment.ledger.results()) != design['fit_budget']['market']:
        raise ValueError('Exact budget/receipt count required before completion')
    return {'status': 'complete', 'completed_market_fits': len(experiment.ledger.results()),
            'synthetic_fits': 1, 'market_skill_claimed': False}


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    folder = root_path(args.drive_root, args.experiment)
    if args.mirror_root:
        raise ValueError('No remote mirror for this CPU control')
    if args.stage == 'prepare':
        if not args.registration_root or not args.decision_contract:
            raise ValueError('Explicit preregistration and contract required')
        ready = prepare(args.repo, folder, args.registration_root, args.decision_contract)
        result = {'status': 'prepared', 'fit_budget': ready['identity']['design']['fit_budget']}
    elif args.stage in ('status', 'pilot-plan'):
        ready, _ = verify(folder, args.repo)
        result = {'status': 'ready', 'fit_budget': ready['identity']['design']['fit_budget'],
                  'completed_fits': len(list((folder / 'ledger/completed').glob('*.json')))}
    elif args.stage == 'pilot':
        with writer(folder):
            result = run(Experiment(folder, repo=args.repo), args.repo, args.max_minutes)
    else:
        raise ValueError('Only prepare/status/pilot-plan/pilot authorized; independent result verification required')
    print(json.dumps(result, indent=2))
