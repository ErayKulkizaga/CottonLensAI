"""Bounded, semi-synthetic small-signal calibration on the frozen Texas cohort."""
import importlib.metadata
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import availability_clock as clock
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research.full_year import SHRINKAGE, chunks
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.nass_regional_execution import fit_consumption
from cottonlens_ml.research.nass_regional_pilot import groups as reference_groups
from cottonlens_ml.research.protocol import full_year_manifest, rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import verify_history
from cottonlens_ml.research.statistics import paired_bootstrap

PROFILE = 'weak-signal-control-v1'
GROUP = 'numeric_D0'
NAMESPACE = 'weak-signal'
SEEDS = tuple(range(1201, 1211))
CONDITIONS = ('null', 'injected')
SOURCE = 'numeric_D0_texas_minus_national_ge'
REGISTRATION = 'weak-signal-registration.json'
RECIPE = {'family': 'ridge', 'params': {'alpha': 1.}, 'horizon': 1, 'task': 'price',
          'device': 'cpu', 'seed': 42, 'target': 'scaled_log', 'years': None, 'cadence': 21, 'window': 1}
CAL_START, CAL_END = '2010-01-01', '2015-01-01'
VERSIONS = ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'threadpoolctl')


def recipe(design, group, h):
    if group != GROUP or h != 1:
        raise ValueError('Only frozen numeric_D0 T+1 Ridge is permitted')
    return {**RECIPE, 'features': design['groups'][GROUP]}


def reference(repo, root):
    repo, root = Path(repo), Path(root)
    evidence = json.loads((repo / 'research/evidence/nass-regional-result-20261009.json').read_bytes())
    ready, frame = verify_history(root)
    if (digest(root / 'ready.json') != evidence['ready_sha256']
            or ready['identity']['source_id'] != evidence['ml_source_id']
            or ready['identity']['registration_id'] != evidence['registration_id']
            or digest(root / 'reports/outer-predictions.csv') != evidence['predictions_sha256']
            or ready['identity']['design']['groups'][GROUP] != reference_groups()[GROUP]
            or frame.date.max() >= pd.Timestamp('2024-01-01')):
        raise ValueError('Completed frozen Texas reference required')
    return ready, frame, evidence


def expected_absolute_log_price(close, mu, sigma, prediction):
    """E[|C exp(mu + sigma Z) - C exp(prediction)|], Z standard normal."""
    c, m, s, p = (np.asarray(x, dtype=float) for x in (close, mu, sigma, prediction))
    if np.any(c <= 0) or np.any(s <= 0) or not all(np.isfinite(x).all() for x in (c, m, s, p)):
        raise ValueError('Finite positive prices and normal scales required')
    mean = np.exp(m + s * s / 2)
    below = norm.cdf((p - m) / s)
    truncated = mean * norm.cdf((p - m - s * s) / s)
    return c * np.maximum(mean - 2 * truncated + np.exp(p) * (2 * below - 1), 0)


def calibrate(frame):
    cal = frame.loc[(frame.date >= CAL_START) & (frame.date < CAL_END) &
                    (frame.cotton_session_index >= 119)].copy()
    known = cal[SOURCE].to_numpy(dtype=float)
    known = known[np.isfinite(known)]
    if len(known) < 35:
        raise ValueError('Insufficient early observed source values to calibrate')
    center = float(np.median(known))
    scale = float(np.std(known))
    if not scale > 1e-6:
        raise ValueError('No early source variation')
    raw = cal.cotton_volatility_20.to_numpy(dtype=float)
    finite = raw[np.isfinite(raw) & (raw > 0)]
    if len(finite) < 100:
        raise ValueError('Insufficient early causal volatility history')
    median_vol = float(np.median(finite))
    cal_z = standardized_source(cal, center, scale)
    cal_sigma = noise_scale(cal, median_vol)
    close = cal.cotton_close.to_numpy(dtype=float)

    def gain(beta):
        mu = beta * cal_z
        naive = expected_absolute_log_price(close, mu, cal_sigma, 0).mean()
        oracle = expected_absolute_log_price(close, mu, cal_sigma, mu).mean()
        return float(100 * (1 - oracle / naive))

    if gain(0.05) <= 5:
        raise ValueError('Five-percent calibration bracket invalid; do not adjust after evaluation')
    beta = float(brentq(lambda x: gain(x) - 5., 0., 0.05, xtol=1e-15))
    return {'period_start': CAL_START, 'period_end_exclusive': CAL_END, 'observations': len(cal),
            'known_source_observations': len(known), 'source': SOURCE, 'source_median': center,
            'source_scale': scale, 'volatility_fallback_median': median_vol,
            'noise_volatility_clip': [0.005, 0.05], 'beta': beta,
            'expected_calibration_oracle_gain_pct': gain(beta),
            'calibrated_without_2015_or_later_outcomes': True}


def standardized_source(frame, center, scale):
    value = frame[SOURCE].to_numpy(dtype=float)
    return (np.where(np.isfinite(value), value, center) - center) / scale


def noise_scale(frame, fallback):
    raw = frame.cotton_volatility_20.to_numpy(dtype=float)
    return np.clip(np.where(np.isfinite(raw) & (raw > 0), raw, fallback), .005, .05)


def synthetic_frame(parent, calibration, condition, seed):
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError('Unregistered synthetic condition/seed')
    result = parent.copy()
    z = standardized_source(result, calibration['source_median'], calibration['source_scale'])
    sigma = noise_scale(result, calibration['volatility_fallback_median'])
    mean = calibration['beta'] * z if condition == 'injected' else np.zeros(len(result))
    rng = np.random.default_rng(seed)
    result['target_return_1'] = mean + sigma * rng.standard_normal(len(result))
    result['oracle_return_1'] = mean
    # The real close/features and five-observation maturity dates are reference metadata.
    # Synthetic T+1 labels are never asserted to form a coherent traded price path.
    return result


def scenario_name(condition, seed):
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError('Unregistered scenario')
    return f'{condition}-seed{seed}'


def study(root):
    record = read_record(Path(root) / REGISTRATION)
    if (record['profile'] != PROFILE or record['completed_experiment'] is not False
            or record['market_fits'] != 0 or record['release_allowed'] is not False
            or record['seeds'] != list(SEEDS) or record['conditions'] != list(CONDITIONS)
            or record['source_feature'] != SOURCE or record['fit_budget'] != 3380):
        raise ValueError('Frozen small-signal registration required')
    return record


def prepare(repo, root, reference_root):
    repo, root, reference_root = map(Path, (repo, root, reference_root))
    validate(repo / 'research')
    if root.exists() and any(root.iterdir()):
        # Registry additions after preregistration cannot change the frozen study.
        return verify_all(repo, root, reference_root)
    parent_ready, parent, evidence = reference(repo, reference_root)
    source = research_source_identity(repo)
    calibration = calibrate(parent)
    groups = {GROUP: parent_ready['identity']['design']['groups'][GROUP]}
    scenarios = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            frame = synthetic_frame(parent, calibration, condition, seed)
            split = full_year_manifest(frame)
            if ([fold['origins'] for fold in split['folds']] !=
                    [fold['origins'] for fold in parent_ready['identity']['split']['folds']]
                    or [[block['origins'] for block in fold['inner']] for fold in split['folds']] !=
                    [[block['origins'] for block in fold['inner']] for fold in parent_ready['identity']['split']['folds']]):
                raise ValueError('Synthetic label altered a frozen origin')
            inner = sum(len(chunks(frame, b['origins'])) for fold in split['folds'] for b in fold['inner'])
            outer = sum(len(chunks(frame, fold['origins'])) for fold in split['folds'])
            if (inner, outer) != (72, 97):
                raise ValueError('Changed 169-fit scenario budget')
            scenarios[scenario_name(condition, seed)] = (frame, split)
    registration = {'profile': PROFILE, 'completed_experiment': False, 'market_fits': 0,
                    'source_id': source['source_id'], 'reference_ready_sha256': digest(reference_root / 'ready.json'),
                    'reference_history_sha256': digest(reference_root / 'history.parquet'),
                    'reference_predictions_sha256': evidence['predictions_sha256'],
                    'source_feature': SOURCE, 'calibration': calibration,
                    'conditions': list(CONDITIONS), 'seeds': list(SEEDS), 'model_seed': 42,
                    'model_recipe': RECIPE, 'groups': groups,
                    'hypothesis': 'Can past-only Ridge and shrinkage recover a representable approximately five-percent expected synthetic price-MAE effect with real missingness and refit schedule?',
                    'primary': 'Separate raw and selected price-MAE gain divided by same-seed observed oracle gain',
                    'negative_control': 'Same noise seed, zero injected signal; do not report market skill',
                    'decision_rules': {'valid_oracle_mean_gain_pct_at_least': 3.,
                        'minimum_positive_seeds': 8, 'median_fraction_of_oracle_at_least': .5,
                        'repeated_null_practical_false_positives_at_least': 2,
                        'classification': ['INVALID_CALIBRATION', 'NULL_CONTROL_FAILS',
                                           'RAW_AND_SELECTED_RECOVER', 'RAW_ONLY_RECOVERS',
                                           'FIXED_RECIPE_DOES_NOT_RECOVER', 'INCONCLUSIVE']},
                    'bootstrap': {'within_year_blocks': [20, 60], 'replicates': 10000, 'seed': 42},
                    'price_gate_unchanged': {'mae_gain_pct': 5., 'direction_pct': 53., 'years_won': 6},
                    'fit_budget': 3380, 'per_scenario_fit_budget': {'inner': 72, 'outer': 97, 'total': 169},
                    'annual_outputs': 160, 'prediction_rows': 40120,
                    'execution': {'processes': 1, 'threads': 2, 'session_minutes': 30},
                    'experimental_status': 'synthetic_only_not_market_predictive_evidence',
                    'independent_holdout': False, 'release_allowed': False,
                    'deliberate_repeat_reason': 'Prior noiseless known-signal controls near 100% MAE gain did not test a weak effect under frozen Texas missingness and past shrinkage.'}
    candidates, trials = load_registry(repo / 'research')
    related = check(candidates, trials, query='synthetic', family='ridge', horizon=1)
    registration['related_existing_experiments'] = [x.get('experiment') for x in related['experiments']]
    registration['related_exact_fit_recipes'] = len(related['exact_fit_recipes'])
    with writer(root):
        copy_immutable(reference_root / 'ready.json', root / 'reference/ready.json')
        copy_immutable(reference_root / 'history.parquet', root / 'reference/history.parquet')
        copy_immutable(reference_root / 'reports/outer-predictions.csv', root / 'reference/outer-predictions.csv')
        freeze_record(root / 'source-manifest.json', source)
        for name, expected in source['files'].items():
            copy_immutable(repo / name, root / 'source-snapshot' / name)
            if digest(root / 'source-snapshot' / name) != expected:
                raise ValueError('Source changed during freeze')
        for name, (frame, split) in scenarios.items():
            directory = root / 'scenarios' / name
            directory.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(directory / 'history.parquet', index=False)
            design = {'profile': PROFILE, 'condition': name.split('-')[0], 'seed': int(name.split('seed')[1]),
                      'recipe': RECIPE, 'groups': groups, 'split': split,
                      'shrinkage_weights': list(SHRINKAGE), 'fit_budget': {'inner': 72, 'outer': 97, 'total': 169}}
            identity = {'profile': PROFILE, 'source_id': source['source_id'],
                        'design': design, 'design_id': content_id(design), 'split': split,
                        'registration_id': content_id(registration),
                        'research_data_id': frame_identity(frame, list(frame.columns)),
                        'python': sys.version.split()[0],
                        'versions': {package: importlib.metadata.version(package) for package in VERSIONS},
                        'lock_sha256': digest(repo / 'ml/uv.lock'),
                        'execution': {'device': 'cpu', **registration['execution']},
                        'synthetic_only': True, 'market_evidence': False, 'release_allowed': False}
            freeze_record(directory / 'ready.json', {'identity': identity,
                          'history_sha256': digest(directory / 'history.parquet'),
                          'created_at': datetime.now(UTC).isoformat()})
        freeze_record(root / REGISTRATION, registration)
        freeze_record(root / 'complete.json', {'registration_id': content_id(registration),
                      'source_id': source['source_id'], 'scenario_sha256': {
                          name: {'ready': digest(root / 'scenarios' / name / 'ready.json'),
                                 'history': digest(root / 'scenarios' / name / 'history.parquet')}
                          for name in scenarios}, 'reference_ready_sha256': digest(root / 'reference/ready.json'),
                      'reference_history_sha256': digest(root / 'reference/history.parquet'),
                      'reference_predictions_sha256': digest(root / 'reference/outer-predictions.csv'),
                      'market_fits': 0, 'synthetic_fits': 0, 'registration_only': True})
    verify_all(repo, root, reference_root)
    return registration


def verify_all(repo, root, reference_root):
    root, repo, reference_root = map(Path, (root, repo, reference_root))
    registration = study(root)
    reference(repo, reference_root)
    complete = read_record(root / 'complete.json')
    if (complete['registration_id'] != content_id(registration)
            or complete['source_id'] != registration['source_id']
            or research_source_identity(repo)['source_id'] != registration['source_id']
            or complete['market_fits'] != 0 or complete['synthetic_fits'] != 0
            or complete['registration_only'] is not True
            or digest(root / 'reference/ready.json') != complete['reference_ready_sha256']
            or digest(root / 'reference/history.parquet') != complete['reference_history_sha256']
            or digest(root / 'reference/outer-predictions.csv') != complete['reference_predictions_sha256']
            or digest(reference_root / 'ready.json') != complete['reference_ready_sha256']
            or digest(reference_root / 'history.parquet') != complete['reference_history_sha256']):
        raise ValueError('Frozen weak-signal identity/source/reference changed')
    manifest = read_record(root / 'source-manifest.json')
    if manifest != research_source_identity(repo):
        raise ValueError('Source manifest differs from active code')
    for name, expected in manifest['files'].items():
        if digest(root / 'source-snapshot' / name) != expected:
            raise ValueError('Source snapshot changed')
    _, parent = verify_history(root / 'reference')
    if registration['calibration'] != calibrate(parent):
        raise ValueError('Early-only calibration changed')
    required = {scenario_name(c, s) for s in SEEDS for c in CONDITIONS}
    if set(complete['scenario_sha256']) != required:
        raise ValueError('Registered scenarios changed')
    for name, hashes in complete['scenario_sha256'].items():
        directory = root / 'scenarios' / name
        if digest(directory / 'ready.json') != hashes['ready'] or digest(directory / 'history.parquet') != hashes['history']:
            raise ValueError('Frozen synthetic scenario changed')
        frame = pd.read_parquet(directory / 'history.parquet')
        condition, seed = name.split('-seed')
        expected = synthetic_frame(parent, registration['calibration'], condition, int(seed))
        pd.testing.assert_frame_equal(frame, expected, check_exact=True)
        ready = read_record(directory / 'ready.json')
        if (ready['identity']['registration_id'] != content_id(registration)
                or ready['identity']['source_id'] != registration['source_id']
                or ready['identity']['python'] != sys.version.split()[0]
                or ready['identity']['versions'] != {p: importlib.metadata.version(p) for p in VERSIONS}
                or ready['identity']['lock_sha256'] != digest(repo / 'ml/uv.lock')
                or ready['identity']['design']['split'] != full_year_manifest(frame)
                or ready['identity']['research_data_id'] != frame_identity(frame, list(frame.columns))):
            raise ValueError('Synthetic origin/identity changed')
        validate_design({'design_id': ready['identity']['design_id'], 'design': ready['identity']['design']})
    return registration


def validate_design(record):
    design = record['design']
    if (record['design_id'] != content_id(design) or design['profile'] != PROFILE
            or design['groups'] != {GROUP: reference_groups()[GROUP]} or design['recipe'] != RECIPE
            or design['condition'] not in CONDITIONS or design['seed'] not in SEEDS
            or design['shrinkage_weights'] != list(SHRINKAGE)
            or design['fit_budget'] != {'inner': 72, 'outer': 97, 'total': 169}
            or design['split']['purge_observations'] != 5 or design['split']['refit_cadence'] != 21
            or [f['year'] for f in design['split']['folds']] != list(range(2016, 2024))):
        raise ValueError('Frozen weak-signal design changed')
    return design


def record_frame(test, payload, group, h, chosen):
    payload['horizon'] = h
    payload['actual_return'] = test.target_return_1
    payload['oracle_return'] = test.oracle_return_1
    payload['target_date'] = pd.to_datetime(test.target_date_1).dt.strftime('%Y-%m-%d')
    payload['selected_weight'] = chosen['weight']
    payload['decision_time'] = pd.to_datetime(test.nass_decision_time, utc=True).map(lambda x: x.isoformat())
    payload['source_known'] = test[SOURCE].notna().astype(int)
    return payload


def output(folder, name, fold, group, h, design, history, *, namespace=NAMESPACE, recipe_fn=recipe):
    saved, frame = path_pilot.output(folder, name, fold, group, h, design, history,
                                     namespace=namespace, recipe_fn=recipe_fn)
    test = rows_at(history, fold['origins'])
    if (not np.array_equal(frame.oracle_return, test.oracle_return_1)
            or not np.array_equal(frame.source_known, test[SOURCE].notna().astype(int))
            or frame.target_date.tolist() != test.target_date_1.dt.strftime('%Y-%m-%d').tolist()):
        raise ValueError('Synthetic truth/provenance changed')
    return saved, frame


def run(repo, root, reference_root, minutes):
    if not 0 < minutes <= 30:
        raise ValueError('CPU session must be at most 30 minutes')
    deadline = time.monotonic() + minutes * 60
    registration = verify_all(repo, root, reference_root)
    proof = verify_registered_evidence(repo, root)
    from cottonlens_ml.research.engine import Experiment

    scenario_roots = [Path(root) / 'scenarios' / scenario_name(c, s) for s in SEEDS for c in CONDITIONS]
    for directory in scenario_roots:
        _, reusable = fit_consumption(directory / 'ledger')
        for attempt in (directory / 'ledger/attempts').glob('*.json'):
            if read_record(attempt)['experiment_id'] not in reusable:
                raise ValueError('Failed synthetic computation has no verified checkpoint; no automatic refit')
    consumed = sum(fit_consumption(p / 'ledger')[0] for p in scenario_roots)
    counter = [consumed]

    def before_fit():
        if time.monotonic() >= deadline:
            raise FitBudgetReached('Global weak-signal session deadline reached')
        if counter[0] >= registration['fit_budget']:
            raise FitBudgetReached('Frozen 3380-fit synthetic budget exhausted')
        counter[0] += 1

    for directory in scenario_roots:
        remaining = (deadline - time.monotonic()) / 60
        if remaining <= 0:
            return {'status': 'planned_pause', 'consumed': counter[0], 'scenario': directory.name}
        with writer(directory):
            experiment = Experiment(directory, repo=repo)
            if experiment.identity['registration_id'] != content_id(registration):
                raise ValueError('Scenario registration changed')
            result = path_pilot.run(experiment, remaining, group_names=(GROUP,), namespace=NAMESPACE,
                                    horizons=(1,), recipe_fn=recipe,
                                    validate_fn=validate_design,
                                    record_frame_fn=record_frame, output_fn=output, before_fit=before_fit)
            if result['status'] != 'complete':
                return {'status': 'planned_pause', 'scenario': directory.name,
                        'consumed': counter[0], 'saved_outputs': result['saved_outputs']}
    return {'status': 'complete', 'consumed': counter[0], 'scenarios': len(scenario_roots),
            'saved_outputs': 8 * len(scenario_roots), 'registered_evidence_sha256': proof}


def verify_registered_evidence(repo, root):
    registration = study(root)
    research = Path(repo) / 'research'
    validate(research)
    registry = json.loads((research / 'registry.json').read_bytes())
    candidates = []
    for item in registry['local_evidence']:
        path = research / item['path']
        if path.name.startswith('weak-signal-preregistration-'):
            body = json.loads(path.read_bytes())
            if body.get('completed_experiment') is False and body.get('registration_id') == content_id(registration):
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError('Exactly one registered zero-fit proof for this frozen identity required')
    path = candidates[0]
    proof = read_record(path)
    if (proof['completed_experiment'] is not False or proof['synthetic_fits'] != 0
            or proof['market_fits'] != 0 or proof['registration_id'] != content_id(registration)
            or proof['registration_sha256'] != digest(Path(root) / REGISTRATION)
            or proof['complete_sha256'] != digest(Path(root) / 'complete.json')
            or proof['source_id'] != registration['source_id']):
        raise ValueError('Published zero-fit preregistration differs from study')
    return digest(path)


def classify(scenarios, rules):
    injected = [v for k, v in scenarios.items() if k.startswith('injected-')]
    nulls = [v for k, v in scenarios.items() if k.startswith('null-')]
    if len(injected) != len(SEEDS) or len(nulls) != len(SEEDS):
        raise ValueError('Every registered seed required for classification')
    oracle = [v['modes']['oracle']['naive_gain_pct'] for v in injected]
    summary = {'oracle_mean_gain_pct': float(np.mean(oracle)), 'null_practical_false_positives':
               sum(v['practical_false_positive'] for v in nulls), 'recovery': {}}
    for mode in ('raw', 'selected'):
        gains = [v['modes'][mode]['naive_gain_pct'] for v in injected]
        fraction = [g / o if o > 0 else None for g, o in zip(gains, oracle, strict=True)]
        median = float(np.median(fraction)) if all(f is not None for f in fraction) else None
        positive = sum(g > 0 for g in gains)
        summary['recovery'][mode] = {'mean_gain_pct': float(np.mean(gains)), 'median_gain_pct': float(np.median(gains)),
                                    'positive_seeds': positive, 'fractions_of_observed_oracle': fraction,
                                    'median_fraction_of_oracle': median,
                                    'recovers': bool(median is not None and median >= rules['median_fraction_of_oracle_at_least']
                                                     and positive >= rules['minimum_positive_seeds'])}
    if summary['oracle_mean_gain_pct'] < rules['valid_oracle_mean_gain_pct_at_least'] or any(o <= 0 for o in oracle):
        decision = 'INVALID_CALIBRATION'
    elif summary['null_practical_false_positives'] >= rules['repeated_null_practical_false_positives_at_least']:
        decision = 'NULL_CONTROL_FAILS'
    elif all(s['recovers'] for s in summary['recovery'].values()):
        decision = 'RAW_AND_SELECTED_RECOVER'
    elif summary['recovery']['raw']['recovers']:
        decision = 'RAW_ONLY_RECOVERS'
    elif not summary['recovery']['selected']['recovers']:
        decision = 'FIXED_RECIPE_DOES_NOT_RECOVER'
    else:
        decision = 'INCONCLUSIVE'
    return decision, summary


def compare(repo, root, reference_root, repetitions=10000):
    from cottonlens_ml.research.weak_signal_review import replay

    root = Path(root)
    registration = verify_all(repo, root, reference_root)
    proof = verify_registered_evidence(repo, root)
    directories = [root / 'scenarios' / scenario_name(c, s) for s in SEEDS for c in CONDITIONS]
    expected = {f'{GROUP}-t1-year{y}.json' for y in range(2016, 2024)}
    complete = 0
    for directory in directories:
        names = {p.name for p in (directory / f'{NAMESPACE}-outputs').glob('*.json')}
        if names - expected:
            raise ValueError('Unregistered annual output')
        complete += len(names)
    if complete != registration['annual_outputs']:
        return {'status': 'pending', 'complete_outputs': complete, 'required_outputs': registration['annual_outputs']}
    scenarios, rows, replays, output_hashes = {}, [], {}, {}
    for directory in directories:
        replays[directory.name] = replay(directory)
        ready, history = verify_history(directory)
        design = validate_design({'design_id': ready['identity']['design_id'], 'design': ready['identity']['design']})
        annual = []
        weights = []
        for fold in design['split']['folds']:
            name = f'{GROUP}-t1-year{fold["year"]}.json'
            saved, frame = output(directory, name, fold, GROUP, 1, design, history)
            frame['year'], frame['condition'], frame['seed'] = fold['year'], design['condition'], design['seed']
            annual.append(frame)
            weights.append({'year': fold['year'], 'weight': saved['weight']})
            output_hashes[f'{directory.name}/{name}'] = digest(directory / f'{NAMESPACE}-outputs' / name)
        combined = pd.concat(annual, ignore_index=True)
        if len(combined) != 2006:
            raise ValueError('All 2006 common origins required')
        modes = {}
        baseline = clock._metrics(combined, 'predicted_return')[0]
        keep = np.ones(len(combined), bool)
        keep[np.argsort(-baseline, kind='stable')[:int(np.ceil(.01 * len(baseline)))]] = False
        for mode, field in [('raw', 'raw_predicted_return'), ('selected', 'predicted_return'), ('oracle', 'oracle_return')]:
            _, _, score = clock._metrics(combined, field)
            parts = [np.column_stack(clock._metrics(f, field)[:2]) for f in annual]
            intervals = {}
            for block in (20, 60):
                ci = paired_bootstrap(parts, block=block, repetitions=repetitions, seed=42,
                                      normalization=float(baseline.mean()))
                ci['gain_ci_pct'] = (100 * np.asarray(ci['difference_ci_95']) / baseline.mean()).tolist()
                intervals[str(block)] = ci
            years = {str(f.year.iloc[0]): clock._metrics(f, field)[2] for f in annual}
            score.update(years=years, year_wins=sum(s['naive_gain_pct'] > 0 for s in years.values()),
                         versus_naive=intervals, without_top_1pct_naive_errors=clock._metrics(combined.loc[keep], field)[2])
            modes[mode] = score
        selected = modes['selected']
        false_positive = bool(selected['naive_gain_pct'] >= 5 and selected['direction_pct'] >= 53
                              and selected['year_wins'] >= 6
                              and all(selected['versus_naive'][str(b)]['difference_ci_95'][0] > 0 for b in (20, 60)))
        scenarios[directory.name] = {'modes': modes, 'weights': weights,
                                     'practical_false_positive': false_positive,
                                     'source_known_origins': int(combined.source_known.sum())}
        rows.append(combined)
    for seed in SEEDS:
        a, b = [rows[directories.index(root / 'scenarios' / scenario_name(c, seed))] for c in CONDITIONS]
        for field in ('date', 'target_date', 'cotton_close', 'horizon', 'source_known', 'decision_time'):
            np.testing.assert_array_equal(a[field], b[field])
        np.testing.assert_allclose(b.actual_return - b.oracle_return, a.actual_return, rtol=0, atol=1e-16)
    decision, summary = classify(scenarios, registration['decision_rules'])
    report_root = root / 'reports'
    report_root.mkdir(exist_ok=True)
    export = report_root / 'outer-predictions.csv'
    encoded = pd.concat(rows, ignore_index=True).to_csv(index=False, lineterminator='\n').encode()
    if export.exists() and export.read_bytes() != encoded:
        raise ValueError('Frozen synthetic prediction export changed')
    if not export.exists():
        export.write_bytes(encoded)
    body = {'status': 'complete', 'profile': PROFILE, 'registration_id': content_id(registration),
            'registered_evidence_sha256': proof, 'market_fits': 0, 'synthetic_fits': 3380,
            'origin_count': 2006, 'prediction_rows': 40120, 'annual_outputs': 160,
            'decision': decision, 'summary': summary, 'scenarios': scenarios, 'independent_replay': replays,
            'output_sha256': output_hashes, 'predictions_sha256': digest(export),
            'bootstrap_repetitions': repetitions, 'synthetic_only': True, 'market_skill_demonstrated': False,
            'independent_holdout': False, 'release_allowed': False}
    freeze_record(report_root / f'{NAMESPACE}-{content_id(body)[:16]}.json', body)
    return body


def dispatch(args):
    from cottonlens_ml.research.engine import root_path

    if args.mirror_root:
        raise ValueError('Small-signal study has no remote mirror')
    root = root_path(args.drive_root, args.experiment)
    if not args.reference_root:
        raise ValueError('Explicit completed Texas reference required')
    if args.stage == 'prepare':
        result = prepare(args.repo, root, args.reference_root)
        print(json.dumps({'status': 'prepared', 'registration_id': content_id(result),
                          'scenarios': len(SEEDS) * len(CONDITIONS), 'fit_budget': result['fit_budget']}, indent=2))
    elif args.stage in ('status', 'pilot-plan'):
        result = verify_all(args.repo, root, args.reference_root)
        consumed = sum(fit_consumption(root / 'scenarios' / scenario_name(c, s) / 'ledger')[0]
                       for s in SEEDS for c in CONDITIONS)
        print(json.dumps({'status': 'prepared', 'registration_id': content_id(result),
                          'consumed': consumed, 'fit_budget': result['fit_budget']}, indent=2))
    elif args.stage == 'pilot':
        with writer(root):
            print(json.dumps(run(args.repo, root, args.reference_root, args.max_minutes), indent=2))
    elif args.stage in ('compare', 'report'):
        with writer(root):
            result = compare(args.repo, root, args.reference_root)
            if args.stage == 'report' and result['status'] == 'complete':
                from cottonlens_ml.research.weak_signal_review import verify_report
                freeze_record(root / 'reports/independent-verification.json', verify_report(root))
        print(json.dumps({'status': result['status'], 'decision': result.get('decision'),
                          'summary': result.get('summary'), 'prediction_rows': result.get('prediction_rows')}, indent=2))
    else:
        raise ValueError('Small-signal profile supports prepare/pilot-plan/pilot/status/compare/report')
