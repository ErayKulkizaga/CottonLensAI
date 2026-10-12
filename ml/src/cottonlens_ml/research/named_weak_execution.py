"""Execute the registered weak-teacher control through Experiment, never a grid."""

import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import named_history_control as parent
from cottonlens_ml.research import named_label_control as labels
from cottonlens_ml.research import named_weak_signal as weak
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.named_capacity_control import CORE
from cottonlens_ml.research.protocol import Preprocessor, Target, inputs, rows_at
from cottonlens_ml.research.recency import copy_immutable

PROFILE = weak.PROFILE
RECIPE = parent.RECIPE
VERSIONS = ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'xgboost', 'threadpoolctl')
REPORT_POLICY = 'named-weak-expected-raw-v1'


def scenarios():
    return [(condition, seed) for condition in weak.CONDITIONS for seed in weak.SEEDS]


def classify(metrics):
    """Apply the pre-existing three-seed boundary; null failure takes precedence."""
    required = {f'{condition}-{seed}' for condition, seed in scenarios()}
    if set(metrics) != required:
        raise ValueError('All six frozen scenarios required; no seed selection')
    gains = [metrics[f'injected-{seed}']['raw']['gain_pct'] for seed in weak.SEEDS]
    nulls = [metrics[f'null-{seed}']['raw']['gain_pct'] for seed in weak.SEEDS]
    if not np.isfinite([*gains, *nulls]).all():
        raise ValueError('Finite expected gains required')
    if any(gain > 0 for gain in nulls):
        return 'INCONCLUSIVE_NULL_CONTROL_FAILED'
    if all(gain >= 5 for gain in gains):
        return 'SIMPLE_WEAK_TEACHER_RECOVERY_SUPPORTED'
    if all(gain < 5 for gain in gains):
        return 'FIXED_LEARNER_WEAK_TEACHER_RECOVERY_INADEQUATE'
    return 'INCONCLUSIVE_MIXED_SEEDS'


def verify_readiness(folder, repo):
    """Bind inputs to the published readiness, not to a mutable local summary."""
    folder, repo = Path(folder), Path(repo)
    proof = read_record(repo / 'research/evidence/named-weak-signal-readiness-20261011.json')
    delivery = read_record(repo / 'research/evidence/named-weak-signal-readiness-delivery-20261011.json')
    prepared = folder / 'prepared'
    ready = read_record(prepared / 'ready.json')
    if (digest(prepared / 'ready.json') != proof['ready_sha256']
            or digest(prepared / 'history.parquet') != proof['history_sha256']
            or digest(prepared / 'parent-ready.json') != proof['parent_ready_sha256']
            or research_source_identity(prepared / 'source-snapshot')['source_id'] != proof['source_id']
            or ready['source_id'] != proof['source_id']
            or ready['design_id'] != content_id(ready['design'])
            or ready['new_fits'] != 0
            or digest(folder / 'verification.json') != proof['verification_sha256']
            or digest(folder / 'controls.json') != proof['controls_sha256']
            or delivery['status'] != 'fresh_github_weak_signal_readiness_verified'
            or not delivery['fresh_github_restore']['exact_ready_history_source_scenarios']):
        raise ValueError('Published readiness/data/source/delivery mismatch')
    for name in (*CORE, 'ml/src/cottonlens_ml/research/named_weak_signal.py',
                 'ml/src/cottonlens_ml/research/named_history_control.py'):
        if digest(repo / name) != digest(prepared / 'source-snapshot' / name):
            raise ValueError('Fixed learner/target/feature/teacher code changed')
    history = pd.read_parquet(prepared / 'history.parquet')
    design = ready['design']
    if (design['profile'] != PROFILE or design['recipe'] != RECIPE
            or design['iterations'] != parent.TREES
            or design['conditions'] != list(weak.CONDITIONS)
            or design['seeds'] != list(weak.SEEDS)
            or design['fit_budget']['total'] != 132 or design['fit_budget']['market'] != 0
            or any(name not in history for name in RECIPE['features'])):
        raise ValueError('Frozen 32-field teacher design required')
    first = min((job for fold in design['folds'] for job in fold['jobs']),
                key=lambda job: job['cutoff'])
    if weak.calibrate(history, first['train_dates']) != design['calibration']:
        raise ValueError('Early calibration changed')
    if design['folds'] != read_record(prepared / 'parent-ready.json')['identity']['design']['folds']:
        raise ValueError('Parent cohorts/jobs changed')
    for fold in design['folds']:
        for job in fold['jobs']:
            validate_job(history, job)
    for condition, seed in scenarios():
        h, mean, noise = weak.build(history, design['calibration'], condition, seed)
        fields = ['date', 'named_return', 'named_target_price', 'target_return_5', 'named_eligible']
        frame = h[fields].copy().assign(oracle_mean=mean, unit_overlap_noise=noise)
        frame.date = frame.date.dt.strftime('%Y-%m-%d')
        name = f'scenarios/{condition}-{seed}.csv'
        import hashlib
        checksum = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
        if not safe_member(name) or digest(prepared / name) != ready['files'][name] or checksum != ready['files'][name]:
            raise ValueError('Frozen matched scenario changed')
    return ready, history


def validate_job(history, job):
    """Recheck the original admission population and strict label maturity."""
    train = rows_at(history, job['train_dates'])
    cutoff = pd.Timestamp(job['cutoff'])
    decision = cutoff.tz_localize('UTC') + pd.Timedelta(days=1, minutes=15)
    if (job['train_dates'] != labels.mature_dates(history, cutoff)
            or not train.date.lt(cutoff).all()
            or not train.target_date_5.lt(cutoff).all()
            or not train.named_label_available_at.le(decision).all()
            or not train.named_eligible.all()
            or rows_at(history, job['origins']).date.min() < cutoff):
        raise ValueError('Frozen job label maturity/admission changed')


def design(readiness):
    original = readiness['design']
    expected_fits = len(scenarios()) * sum(len(fold['jobs']) for fold in original['folds'])
    if expected_fits != 132:
        raise ValueError('Exact 132 synthetic fits required')
    return {**original, 'no_fitting_in_readiness': False,
            'readiness_design_id': readiness['design_id'], 'report_policy': REPORT_POLICY,
            'execution': 'Existing Experiment/ledger; no learning-control extra fits or grid',
            'deliberate_repeat_reason': 'Readiness inputs only; same named XGB weak/noisy teacher has no completed fit evidence'}


def register(repo, folder, readiness_root):
    """Prepare a new immutable registration; publishing/restore precedes prepare."""
    repo, folder, readiness_root = map(Path, (repo, folder, readiness_root))
    old, history = verify_readiness(readiness_root, repo)
    fixed = design(old)
    identity = {'profile': PROFILE, 'source_id': research_source_identity(repo)['source_id'],
                'python': sys.version.split()[0],
                'versions': {name: importlib.metadata.version(name) for name in VERSIONS},
                'lock_sha256': digest(repo / 'ml/uv.lock'), 'design': fixed,
                'design_id': content_id(fixed), 'research_data_id': frame_identity(history, list(history))}
    with writer(folder):
        if any(path.name != '.writer-lock' for path in folder.iterdir()):
            raise ValueError('New registration namespace required; preserve partial files')
        for path in readiness_root.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and 'negative-controls' not in path.parts and 'replayed' not in path.parts:
                copy_immutable(path, folder / 'readiness' / path.relative_to(readiness_root))
        identity['input_hashes'] = {path.relative_to(folder).as_posix(): digest(path)
                                    for path in sorted((folder / 'readiness').rglob('*')) if path.is_file()}
        copy_immutable(readiness_root / 'prepared/history.parquet', folder / 'history.parquet')
        freeze_record(folder / 'ready.json', {'identity': identity, 'history_sha256': digest(folder / 'history.parquet')})
        freeze_record(folder / 'decision-contract.json', fixed)
    verify(folder, repo)
    return read_record(folder / 'ready.json')


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder / 'ready.json')
    identity = ready['identity']
    if (identity['profile'] != PROFILE or research_source_identity(repo)['source_id'] != identity['source_id']
            or identity['python'] != sys.version.split()[0]
            or identity['versions'] != {name: importlib.metadata.version(name) for name in VERSIONS}
            or identity['lock_sha256'] != digest(repo / 'ml/uv.lock')):
        raise ValueError('Frozen execution source/environment changed; new namespace required')
    for name, checksum in identity['input_hashes'].items():
        if not safe_member(name) or digest(folder / name) != checksum:
            raise ValueError('Frozen readiness input changed')
    original, history = verify_readiness(folder / 'readiness', repo)
    fixed = design(original)
    if (identity['design'] != fixed or identity['design_id'] != content_id(fixed)
            or identity['research_data_id'] != frame_identity(history, list(history))
            or digest(folder / 'history.parquet') != ready['history_sha256']
            or read_record(folder / 'decision-contract.json') != fixed):
        raise ValueError('Frozen design/history/contract changed')
    pd.testing.assert_frame_equal(pd.read_parquet(folder / 'history.parquet'), history, check_exact=True)
    return ready, history


def prepare(repo, folder, registration_root, decision_contract):
    repo, folder, registration_root, decision_contract = map(Path, (repo, folder, registration_root, decision_contract))
    ready, _ = verify(registration_root, repo)
    proof = read_record(repo / 'research/evidence/named-weak-execution-preregistration-20261012.json')
    delivery = read_record(repo / 'research/evidence/named-weak-execution-preregistration-delivery-20261012.json')
    if (proof['ready_sha256'] != digest(registration_root / 'ready.json')
            or proof['decision_contract_sha256'] != digest(decision_contract)
            or read_record(decision_contract) != ready['identity']['design']
            or proof['new_fits'] != 0 or delivery['status'] != 'fresh_github_preregistration_verified'
            or delivery['source_id'] != ready['identity']['source_id']
            or delivery['ready_sha256'] != proof['ready_sha256']):
        raise ValueError('Fresh published execution preregistration required')
    with writer(folder):
        if (folder / 'ready.json').exists():
            current, _ = verify(folder, repo)
            if current != ready:
                raise ValueError('Preserve existing execution identity')
            return current
        if any(path.name != '.writer-lock' for path in folder.iterdir()):
            raise ValueError('Preserve partial preparation')
        for path in registration_root.rglob('*'):
            if path.is_file():
                copy_immutable(path, folder / path.relative_to(registration_root))
    verify(folder, repo)
    return ready


def specification(h, job, fold, scenario):
    train = labels.projection(h, job['train_dates'], 'named')
    return {'recipe': RECIPE, 'role': f'{scenario}-{fold["year"]}-{job["role"]}-{job["bucket"]}',
            'iterations': parent.TREES, 'train_dates': job['train_dates'], 'validation_dates': [],
            'test_dates': job['origins'], 'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}


def records_for(experiment, h, fold, scenario):
    records = []
    for job in fold['jobs']:
        spec = specification(h, job, fold, scenario)
        key = experiment.ledger.key(spec)
        record = read_record(experiment.ledger.root / 'completed' / f'{key}.json')
        if (record['identity'] != experiment.identity or record['specification'] != spec
                or record['experiment_id'] != key or record['result']['iterations'] != parent.TREES
                or not record['files']):
            raise ValueError('Synthetic fit identity/count/payload mismatch')
        for name, checksum in record['files'].items():
            if not safe_member(name) or digest(experiment.ledger.root / name) != checksum:
                raise ValueError('Native payload changed')
        records.append(record)
    return records


def verify_output(experiment, h, fold, scenario):
    result = labels.make_output(h, fold, scenario, experiment.identity['design'], records_for(experiment, h, fold, scenario))
    if read_record(experiment.root / 'weak-outputs' / f'{scenario}-{fold["year"]}.json') != result:
        raise ValueError('Synthetic output/shrinkage changed')
    return result


def run(experiment, repo, max_minutes):
    if not 0 < max_minutes <= 30:
        raise ValueError('One CPU session must be positive and <=30 minutes')
    ready, history = verify(experiment.root, repo)
    deadline = time.monotonic() + max_minutes * 60
    used, _ = fit_consumption(experiment.root / 'ledger')
    budget = ready['identity']['design']['fit_budget']['total']
    if used > budget:
        raise ValueError('Synthetic attempt budget exceeded')
    count = [used]

    def before():
        if time.monotonic() >= deadline or count[0] >= budget:
            raise FitBudgetReached('Frozen synthetic fit/time budget; checkpoints retained')
        if research_source_identity(repo)['source_id'] != experiment.identity['source_id']:
            raise ValueError('Source changed during execution')
        count[0] += 1

    experiment.before_compute = before
    original = experiment.history
    try:
        for condition, seed in scenarios():
            h, _, _ = weak.build(history, ready['identity']['design']['calibration'], condition, seed)
            experiment.history = h
            scenario = f'{condition}-{seed}'
            for fold in ready['identity']['design']['folds']:
                path = experiment.root / 'weak-outputs' / f'{scenario}-{fold["year"]}.json'
                if path.exists():
                    verify_output(experiment, h, fold, scenario)
                    continue
                records = []
                for job in fold['jobs']:
                    validate_job(history, job)
                    train = labels.projection(h, job['train_dates'], 'named')
                    test = labels.projection(h, job['origins'], 'named', test=True)
                    role = specification(h, job, fold, scenario)['role']
                    records.append(experiment.fit(RECIPE, train, None, test, role, iterations=parent.TREES))
                freeze_record(path, labels.make_output(h, fold, scenario, ready['identity']['design'], records))
                verify_output(experiment, h, fold, scenario)
    except FitBudgetReached:
        return {'status': 'planned_pause', 'completed_synthetic_fits': len(experiment.ledger.results()), 'market_fits': 0}
    finally:
        experiment.history = original
    if len(experiment.ledger.results()) != budget:
        raise ValueError('Exact 132 completed synthetic fits required')
    verify(experiment.root, repo)
    return {'status': 'complete', 'synthetic_fits': budget, 'market_fits': 0, 'market_skill_claimed': False}


def replay(experiment, h, fold, scenario):
    """No-fit native replay; independently refit preprocessing/scaling on past rows."""
    import xgboost as xgb
    count = 0
    for job, record in zip(fold['jobs'], records_for(experiment, h, fold, scenario), strict=True):
        train = labels.projection(h, job['train_dates'], 'named')
        test = labels.projection(h, job['origins'], 'named', test=True)
        root = experiment.ledger.root / record['payload_root']
        adapter = json.loads((root / 'adapter.json').read_bytes())
        processor = Preprocessor.fit(train, RECIPE['features'])
        target = Target.fit(train, 5, RECIPE['target'])
        if (adapter['spec'] != RECIPE or adapter['processor'] != processor.as_dict()
                or adapter['target'] != target.as_dict() or adapter['iterations'] != parent.TREES
                or not safe_member(adapter['model_file'])
                or adapter['last_fit_label'] != train.target_date_5.max().isoformat()):
            raise ValueError('Native adapter/past-only preprocessing/scaling mismatch')
        model = xgb.Booster(params={'nthread': 2})
        model.load_model(root / adapter['model_file'])
        model.set_param({'nthread': 2})
        if model.num_boosted_rounds() != parent.TREES:
            raise ValueError('Native tree count changed')
        raw = target.inverse(model.predict(xgb.DMatrix(inputs(h, test, processor))), test.cotton_close.to_numpy())
        np.testing.assert_array_equal(raw, record['result']['predictions'])
        count += 1
    return count


def report(experiment, repo):
    ready, history = verify(experiment.root, repo)
    if len(experiment.ledger.results()) != 132 or fit_consumption(experiment.root / 'ledger')[0] != 132:
        raise ValueError('Incomplete or over-budget run cannot be reported completed')
    metrics, rows, replays = {}, [], 0
    fixed = ready['identity']['design']
    if len(fixed['folds']) != 1:
        raise ValueError('Frozen one-year synthetic diagnosis required')
    fold = fixed['folds'][0]
    for condition, seed in scenarios():
        scenario = f'{condition}-{seed}'
        h, mean, noise = weak.build(history, fixed['calibration'], condition, seed)
        output = verify_output(experiment, h, fold, scenario)
        replays += replay(experiment, h, fold, scenario)
        outer = rows_at(h, fold['origins'])
        indices = pd.Index(h.date).get_indexer(outer.date)
        actual, price = outer.named_target_price.to_numpy(), outer.named_price.to_numpy()
        eligible = outer.named_eligible.to_numpy(dtype=bool)
        raw, selected = np.array(output['raw_predictions']), np.array(output['selected_predictions'])
        scales, centers = {}, {}
        for job, record in zip(fold['jobs'], records_for(experiment, h, fold, scenario), strict=True):
            adapter = json.loads((experiment.ledger.root / record['payload_root'] / 'adapter.json').read_bytes())
            for date in job['origins']:
                scales[date] = adapter['target']['scale']
                centers[date] = adapter['target']['mean']
        scale = np.array([scales[date] for date in fold['origins']])[eligible]
        actual_return = outer.named_return.to_numpy()[eligible]
        realized = {}
        for name, pred in (('raw', raw), ('selected', selected)):
            baseline = float(np.abs(actual[eligible] - price[eligible]).mean())
            loss = float(np.abs(actual[eligible] - price[eligible] * np.exp(pred[eligible])).mean())
            realized[name] = {'naive': baseline, 'model': loss, 'gain_pct': 100 * (1 - loss / baseline),
                              'direction_pct': float(100 * np.mean(np.sign(pred[eligible]) == np.sign(outer.named_return.to_numpy()[eligible]))),
                              'active_pct': float(100 * np.mean(pred[eligible] != 0)),
                              'log_mse_gain_pct': float(100 * (1 - np.mean((actual_return - pred[eligible])**2) / np.mean(actual_return**2))),
                              'standardized_mse_gain_pct': float(100 * (1 - np.mean(((actual_return - pred[eligible]) / scale)**2) / np.mean((actual_return / scale)**2)))}
        metrics[scenario] = {'raw': weak.expected(outer, raw, mean[indices], fixed['calibration']['sigma']),
                             'selected': weak.expected(outer, selected, mean[indices], fixed['calibration']['sigma']),
                             'oracle': weak.expected(outer, mean[indices], mean[indices], fixed['calibration']['sigma']),
                             'weight': output['weight'], 'realized': realized}
        for index, row in enumerate(outer.itertuples()):
            rows.append({'scenario': scenario, 'condition': condition, 'noise_seed': seed,
                         'origin': row.date.strftime('%Y-%m-%d'), 'target_date': row.target_date_5.strftime('%Y-%m-%d'),
                         'eligible': bool(row.named_eligible), 'origin_price': row.named_price,
                         'actual_price': row.named_target_price if row.named_eligible else None,
                         'actual_log_return': row.named_return if row.named_eligible else None,
                         'raw_log_return': raw[index], 'selected_log_return': selected[index],
                         'fit_target_mean': centers[fold['origins'][index]],
                         'fit_target_scale': scales[fold['origins'][index]],
                         'weight': output['weight'], 'oracle_mean': mean[indices[index]],
                         'unit_overlap_noise': noise[indices[index]]})
    frame = pd.DataFrame(rows)
    path = experiment.root / 'reports/outer-predictions.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.to_csv(index=False, lineterminator='\n').encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Preserve prior report; prediction bytes changed')
    if not path.exists():
        path.write_bytes(data)
    result = {'profile': PROFILE, 'report_policy': REPORT_POLICY, 'source_id': experiment.identity['source_id'],
              'design_id': experiment.identity['design_id'], 'synthetic_fits': 132, 'market_fits': 0,
              'metrics': metrics, 'decision': classify(metrics), 'native_replays': replays,
              'prediction_rows': len(rows), 'scored_rows': int(frame.eligible.sum()), 'unknown_rows': int((~frame.eligible).sum()),
              'predictions_sha256': digest(path), 'market_skill_claimed': False,
              'independent_market_holdout': False, 'statistical_power_estimated': False, 'release_allowed': False}
    freeze_record(experiment.root / 'reports/result.json', result)
    return read_record(experiment.root / 'reports/result.json')


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    folder = root_path(args.drive_root, args.experiment)
    if args.mirror_root:
        raise ValueError('No Drive mirror or model promotion')
    if args.stage == 'prepare':
        if not args.registration_root or not args.decision_contract:
            raise ValueError('Explicit preregistration/decision contract required')
        ready = prepare(args.repo, folder, args.registration_root, args.decision_contract)
        result = {'status': 'prepared', 'fit_budget': ready['identity']['design']['fit_budget']}
    elif args.stage in ('status', 'pilot-plan'):
        ready, _ = verify(folder, args.repo)
        result = {'status': 'ready', 'fit_budget': ready['identity']['design']['fit_budget'],
                  'completed_fits': len(list((folder / 'ledger/completed').glob('*.json')))}
    elif args.stage in ('pilot', 'report'):
        with writer(folder):
            experiment = Experiment(folder, repo=args.repo)
            result = run(experiment, args.repo, args.max_minutes) if args.stage == 'pilot' else report(experiment, args.repo)
    else:
        raise ValueError('Only prepare/status/pilot-plan/pilot/report; no search or promotion')
    print(json.dumps(result, indent=2))
