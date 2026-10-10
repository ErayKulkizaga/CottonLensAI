"""Nested chronological research; all fitted state is past-only and checksummed."""
import argparse
import importlib.metadata
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import (
    digest,
    research_source_identity,
    source_identity,
)
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.preflight import load_frozen_data, verify_target_contract
from cottonlens_ml.research import VERSION
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.models import (
    calibrate,
    class_labels,
    class_metrics,
    fit_predict,
    gpu_sample,
    temperature,
)
from cottonlens_ml.research.protocol import (
    feature_groups,
    feature_history,
    mature,
    rows_at,
    split_manifest,
)
from cottonlens_ml.research.search import (
    BUDGETS,
    SEEDS,
    historical_gate,
    recipes,
)
from cottonlens_ml.runtime_guard import require_colab_training
from cottonlens_ml.sprint import R2, R2_READINESS
from cottonlens_ml.walkforward import summarize


def root_path(root, name):
    import re
    if not re.fullmatch(r'research-[a-z0-9-]+', name):
        raise ValueError('Research experiment names must start research- and contain lowercase letters/digits/hyphens')
    return Path(root) / 'experiments' / name


def prepare(repo, drive, name, publications=(), profile='legacy'):
    require_colab_training()
    code = source_identity(repo)
    old = json.loads((drive / 'experiments' / R2 / 'ready.json').read_text())
    if old['readiness_id'] != R2_READINESS or content_id({k: v for k, v in old.items() if k != 'readiness_id'}) != R2_READINESS:
        raise ValueError('Frozen R2 identity changed')
    market, history = load_frozen_data(old['data_identity'])
    verify_target_contract(history)
    # No 2024+ rows enter feature engineering, missingness fitting, or diagnostics.
    history = history.loc[history.date < '2024-01-01'].copy()
    market = market.loc[market.date < '2024-01-01'].copy()
    history = feature_history(history, market)
    groups = feature_groups(history)
    if profile == 'free-data-v1':
        from cottonlens_ml.research.ablation import groups as information_groups
        groups = information_groups(history, groups)
    sources, earliest = [], None
    for package in publications:
        from cottonlens_ml.research.publications import (
            attach_package,
            availability_features,
            load_package,
        )
        manifest, released, features = load_package(package)
        if profile == 'free-data-v1':
            usage = manifest.get('usage', {})
            if usage.get('cost_tl') != 0 or usage.get('research_allowed') is not True:
                raise ValueError('Free-data profile requires reviewed zero-cost usage for every source')
            if any(s['manifest']['kind'] == manifest['kind'] for s in sources):
                raise ValueError('Combine same-source releases into one immutable package')
        history = attach_package(history, manifest, released, features)
        available = history.loc[history[features].notna().all(axis=1), 'date']
        if available.empty:
            raise ValueError('No pre-2024 information coverage')
        earliest = max(earliest or available.min(), available.min())
        if profile == 'free-data-v1':
            groups[f'source_{manifest["kind"]}'] = [*groups['base'], *features, *availability_features(manifest)]
            groups[f'availability_{manifest["kind"]}'] = [*groups['base'], *availability_features(manifest)]
        else:
            groups['expanded'] += features
        sources.append({'path': str(package), 'manifest': manifest})
    if profile == 'free-data-v1' and sources:
        combined = list(dict.fromkeys([*groups['expanded_availability'],
            *(name for source in sources for name in [*source['manifest']['features'],
                                                     *availability_features(source['manifest'])])]))
        groups['all_information'] = combined
        for source in sources:
            excluded = {*source['manifest']['features'], *availability_features(source['manifest'])}
            groups[f'without_{source["manifest"]["kind"]}'] = [c for c in combined if c not in excluded]
    folds = split_manifest(history, earliest=earliest)
    folder = root_path(drive, name)
    versions = {package: importlib.metadata.version(package) for package in
                ('numpy', 'pandas', 'xgboost', 'catboost', 'tensorflow', 'scikit-learn')}
    identity = {'version': VERSION, 'source_id': code['source_id'], 'r2_readiness_id': R2_READINESS,
                'source_data': old['data_identity'], 'research_data_id': frame_identity(history, list(history.columns)),
                'split': folds, 'groups': groups, 'publication_sources': sources, 'versions': versions,
                'python': sys.version.split()[0], 'budgets': BUDGETS, 'lock_sha256': digest(repo / 'ml/uv.lock')}
    if profile != 'legacy':
        identity['profile'] = profile
        identity['versions']['optuna'] = importlib.metadata.version('optuna')
    with writer(folder):
        if (folder / 'ready.json').exists():
            prior = read_record(folder / 'ready.json')
            if prior['identity'] != identity or digest(folder / 'history.parquet') != prior['history_sha256']:
                raise ValueError('Frozen research source/data/environment changed; use a new research namespace')
            return
        target = folder / 'history.parquet'
        if target.exists():
            raise ValueError('Incomplete prepare exists; preserve it and use a new namespace')
        history.to_parquet(target, index=False)
        freeze_record(folder / 'ready.json', {'identity': identity, 'history_sha256': digest(target),
                      'created_at': datetime.now(UTC).isoformat(), 'gpu': gpu_sample()})
        prior_file = repo / 'ml/research_history.json'
        shutil.copyfile(prior_file, folder / 'prior-evidence.json')
    print('Research frozen:', folder, 'outer origins:', len(folds['folds']) * 126, flush=True)


class Experiment:
    def __init__(self, folder, *, repo=None):
        self.root = Path(folder)
        self.ready = read_record(self.root / 'ready.json')
        self.identity = self.ready['identity']
        identity_fn = research_source_identity if self.identity.get('profile') in ('full-year-v1', 'ams-exploration-v1', 'fas-exploration-v1', 'nass-exploration-v1', 'wasde-exploration-v1', 'cftc-exploration-v1', 'fx-exploration-v1', 'crop-exploration-v1', 'weather-exploration-v1', 'oncall-exploration-v1', 'oncall-exploration-v2', 'recency-pilot-v1', 'availability-clock-pilot-v1', 'return-path-pilot-v1', 'agri-transfer-pilot-v1', 'agri-nonlinear-pilot-v1', 'statistical-pilot-v1', 'nonlinear-path-pilot-v1', 'wasde-text-pilot-v1', 'fundamental-joint-pilot-v1', 'wasde-regional-t1-pilot-v1', 'nass-regional-t1-pilot-v1', 'weak-signal-control-v1', 'paired-price-loss-control-v1', 'contract-curve-t1-pilot-v1', 'contract-curve-t5-pilot-v1', 'named-label-control-t5-v1', 'named-capacity-control-t5-v1') else source_identity
        if repo is not None and identity_fn(repo)['source_id'] != self.identity['source_id']:
            raise ValueError('Research source identity changed')
        if repo is not None:
            versions = {name: importlib.metadata.version(name) for name in self.identity['versions']}
            if versions != self.identity['versions'] or sys.version.split()[0] != self.identity['python']:
                raise ValueError('Research environment changed; use a new namespace')
        if digest(self.root / 'history.parquet') != self.ready['history_sha256']:
            raise ValueError('Research snapshot corrupted')
        # Local Colab cache; never repeatedly stream model inputs from Drive.
        source = self.root / 'history.parquet'
        if Path('/content').is_dir():
            cached = Path('/content/cottonlens-research-data') / (self.ready['history_sha256'] + '.parquet')
            cached.parent.mkdir(parents=True, exist_ok=True)
            if not cached.exists() or digest(cached) != self.ready['history_sha256']:
                shutil.copyfile(source, cached)
            source = cached
        self.history = pd.read_parquet(source)
        if self.history.date.max() >= pd.Timestamp('2024-01-01'):
            raise ValueError('Seen audit cannot enter research')
        self.ledger = Ledger(self.root / 'ledger', self.identity)

    def train_rows(self, cutoff, spec):
        rows = mature(self.history, cutoff, spec.get('years'))
        if self.identity['split'].get('coverage_start'):
            rows = rows.loc[rows.date >= pd.Timestamp(self.identity['split']['coverage_start'])]
        # Common 120-observation warmup across all families. Never remove feature-missing rows.
        return rows.loc[rows.date >= self.history.date.iloc[119]].copy()

    def fit(self, spec, train, validation, test, role, *, iterations=None, repeat=None):
        specification = {'recipe': spec, 'role': role, 'iterations': iterations,
                         'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(),
                         'validation_dates': [] if validation is None else validation.date.dt.strftime('%Y-%m-%d').tolist(),
                         'test_dates': test.date.dt.strftime('%Y-%m-%d').tolist(),
                         'train_identity': frame_identity(train, list(train.columns))}
        record = self.ledger.run(specification, lambda workspace: fit_predict(
            self.history, train, validation, test, spec, workspace, iterations=iterations), repeat=repeat,
            before_compute=getattr(self, 'before_compute', None))
        if getattr(self, 'after_fit', None):
            self.after_fit(record)
        return record

    def inner(self, spec, fold, repeat=None, progress=None):
        records, scores = [], []
        for index, block in enumerate(fold['inner']):
            print(f'STAGE inner fold={fold["fold"]} block={index + 1}/3 '
                  f'family={spec["family"]} T+{spec["horizon"]} seed={spec["seed"]}', flush=True)
            validation = rows_at(self.history, block['origins'])
            train = self.train_rows(validation.date.min(), spec)
            early_validation = train.tail(63)
            early_train = self.train_rows(early_validation.date.min(), spec)
            tuned = self.fit(spec, early_train, early_validation, early_validation,
                             f'early-stop-{fold["fold"]}-{index}', repeat=repeat)
            count = tuned['result']['iterations']
            predicted, chunk_ids = [], []
            cadence = spec.get('cadence', 126)
            start = int(validation.cotton_session_index.iloc[0])
            buckets = (validation.cotton_session_index - start) // cadence
            source_dates = self.history.set_index('cotton_session_index').date
            for bucket in sorted(buckets.unique()):
                chunk = validation.loc[buckets.eq(bucket)]
                cutoff = source_dates.loc[start + int(bucket) * cadence]
                refit = self.train_rows(cutoff, spec)
                record = self.fit(spec, refit, None, chunk, f'inner-score-{fold["fold"]}-{index}-{bucket}',
                                  iterations=count, repeat=repeat)
                predicted.extend(record['result']['predictions'])
                chunk_ids.append(record['experiment_id'])
            metrics = (class_metrics(class_labels(validation, spec['horizon']), predicted)
                       if spec.get('task') == 'direction' else evaluate(validation.cotton_close.to_numpy(),
                           validation[f'target_return_{spec["horizon"]}'].to_numpy(), predicted))
            result = {'metrics': metrics, 'iterations': count, 'predictions': predicted,
                      'origins': validation.date.dt.strftime('%Y-%m-%d').tolist()}
            if spec.get('task') == 'direction':
                scores.append(result['metrics']['log_loss'])
            else:
                naive = evaluate(validation.cotton_close.to_numpy(), validation[f'target_return_{spec["horizon"]}'].to_numpy(),
                                 np.zeros(len(validation)))['mae']
                if naive <= 0:
                    raise ValueError('Relative MAE requires nonzero Naive MAE')
                scores.append(result['metrics']['mae'] / naive)
            records.append({'experiment_id': chunk_ids, 'result': result, 'tuning_record': tuned['experiment_id']})
            if progress is not None:
                progress(index, float(np.mean(scores)))
        return {'recipe': spec, 'score': float(np.mean(scores)), 'records': [r['experiment_id'] for r in records],
                'iterations': max(1, int(np.median([r['result']['iterations'] for r in records]))),
                'predictions': [p for r in records for p in r['result']['predictions']],
                'origins': [d for r in records for d in r['result']['origins']]}

    def tune(self, family, horizon, fold, candidates, namespace):
        marker = self.root / 'decisions' / namespace / f'{family}-t{horizon}-fold{fold["fold"]}.json'
        if marker.exists():
            return read_record(marker)
        evaluated = []
        for i, spec in enumerate(candidates):
            result = self.inner(spec, fold)
            evaluated.append(result)
            print(f'{namespace} {family} T+{horizon} fold={fold["fold"]} candidate={i + 1}/{len(candidates)} inner={result["score"]:.6f}', flush=True)
        shortlist = sorted(evaluated, key=lambda r: (r['score'], complexity(r['recipe'])))[:
            3 if family in ('mlp', 'lstm', 'tcn') else 5]
        confirmed = []
        for candidate in shortlist:
            seeds = []
            for seed in SEEDS:
                spec = {**candidate['recipe'], 'seed': seed}
                seeds.append(candidate if seed == 42 else self.inner(spec, fold, repeat='new_seed'))
            confirmed.append({'score': float(np.mean([r['score'] for r in seeds])), 'seeds': seeds,
                              'recipe': candidate['recipe']})
        chosen = min(confirmed, key=lambda r: (r['score'], complexity(r['recipe'])))
        result = {'chosen': chosen, 'trials': [{'recipe': r['recipe'], 'score': r['score'], 'records': r['records']}
                                              for r in evaluated], 'fold': fold['fold'], 'selection_used_outer': False}
        freeze_record(marker, result)
        return result

    def outer(self, chosen, fold, namespace, *, repeat=None):
        marker = self.root / ('reproduction' if repeat else 'outer') / namespace / f'{chosen["recipe"]["family"]}-t{chosen["recipe"]["horizon"]}-fold{fold["fold"]}.json'
        if marker.exists():
            return read_record(marker)
        if chosen['recipe']['family'] == 'ensemble':
            members = [self.outer(member['decision'], fold, member['namespace'], repeat=repeat)
                       for member in chosen['members']]
            predicted = np.asarray([m['predictions'] for m in members]).T @ chosen['weights']
            test = rows_at(self.history, fold['origins'])
            h = chosen['recipe']['horizon']
            result = {'recipe': chosen['recipe'], 'predictions': predicted.tolist(), 'fold': fold['fold'],
                      'origins': fold['origins'], 'inner_score': chosen['score'],
                      'metrics': evaluate(test.cotton_close.to_numpy(), test[f'target_return_{h}'].to_numpy(), predicted)}
            if self.identity.get('profile') == 'free-data-v1':
                from cottonlens_ml.research.uncertainty import for_candidate
                result['uncertainty'] = for_candidate(self.history, chosen, fold['origins'], predicted)
            freeze_record(marker, result)
            return result
        test = rows_at(self.history, fold['origins'])
        members, records = [], []
        for seed in chosen['seeds']:
            spec = seed['recipe']
            cadence = spec.get('cadence', 126)
            if cadence == 126:
                chunks = [(test.date.min(), test)]
            else:
                start = int(test.cotton_session_index.iloc[0])
                buckets = (test.cotton_session_index - start) // cadence
                dates = self.history.set_index('cotton_session_index').date
                chunks = [(dates.loc[start + int(b) * cadence], test.loc[buckets.eq(b)]) for b in sorted(buckets.unique())]
            predicted = []
            for cutoff, chunk in chunks:
                train = self.train_rows(cutoff, spec)
                record = self.fit(spec, train, None, chunk, f'outer-{namespace}-{fold["fold"]}-{cutoff}',
                                  iterations=seed['iterations'], repeat=repeat)
                predicted.extend(record['result']['predictions'])
                records.append(record['experiment_id'])
            members.append(predicted)
        predicted = np.mean(np.asarray(members), axis=0)
        spec = chosen['recipe']
        h = spec['horizon']
        calibration = None
        if spec.get('task') == 'direction':
            inner_probability = np.mean([r['predictions'] for r in chosen['seeds']], axis=0)
            inner_rows = rows_at(self.history, chosen['seeds'][0]['origins'])
            calibration = calibrate(inner_probability, class_labels(inner_rows, h))
            predicted = temperature(predicted, calibration)
            metrics = class_metrics(class_labels(test, h), predicted)
        else:
            metrics = evaluate(test.cotton_close.to_numpy(), test[f'target_return_{h}'].to_numpy(), predicted)
        from cottonlens_ml.research.diagnostics import residual_report
        residuals = (residual_report(self.history, fold['origins'], predicted, h, self.train_rows(test.date.min(), spec))
                     if spec.get('task', 'price') == 'price' else None)
        result = {'recipe': spec, 'predictions': predicted.tolist(), 'metrics': metrics,
                  'origins': fold['origins'], 'records': records, 'fold': fold['fold'],
                  'calibration_temperature': calibration, 'inner_score': chosen['score'], 'residuals': residuals}
        if self.identity.get('profile') == 'free-data-v1' and spec.get('task', 'price') == 'price':
            from cottonlens_ml.research.uncertainty import for_candidate
            result['uncertainty'] = for_candidate(self.history, chosen, fold['origins'], predicted)
        freeze_record(marker, result)
        return result


def complexity(spec):
    parameters = spec['params']
    return (parameters.get('max_depth', parameters.get('depth', parameters.get('width', 0))),
            len(spec['features']), spec.get('window', 1), content_id(spec))


def search(experiment, round_name, extension=0):
    require_colab_training()
    if experiment.identity.get('profile') == 'free-data-v1':
        if round_name != 'A':
            if extension:
                raise ValueError('Only the adaptive tabular A round accepts --extension')
            from cottonlens_ml.research.free_rounds import run
            return run(experiment, round_name)
        from cottonlens_ml.research.adaptive import search_tabular
        return search_tabular(experiment, extension)
    if (experiment.root / 'locked.json').exists():
        raise ValueError('Experiment is locked; a new hypothesis requires a new experiment')
    if read_record(experiment.root / 'diagnosis.json')['status'] != 'passed':
        raise ValueError('Diagnosis gate must pass')
    if round_name not in BUDGETS:
        return refinement_search(experiment, round_name)
    budgets = BUDGETS[round_name]
    if extension < 0 or (extension and round_name == 'B'):
        raise ValueError('Extensions apply to price search rounds with a positive extension number')
    extension_path = experiment.root / 'extensions' / f'{round_name}-{extension}.json'
    baseline = None
    if extension:
        previous = compare(experiment)
        from cottonlens_ml.research.search import next_round
        base = [r for r in previous['candidates'] if r['task'] == 'price' and r['namespace'] == round_name]
        if {r['horizon'] for r in base} != {1, 5}:
            raise ValueError('Complete the initial round before extending it')
        histories = [read_record(experiment.root / 'extensions' / f'{round_name}-{i}.json')
                     for i in range(1, extension)]
        initial_score = float(np.mean([min(r['inner_score'] for r in base if r['horizon'] == h) for h in (1, 5)]))
        gains = [1 - initial_score, *[r['relative_improvement'] for r in histories]]
        if next_round(gains) != 'extend_top_two_families_64_each':
            raise ValueError('Plateau: new information or diagnosis required')
        allowed = {round_name, *[f'{round_name}-extension-{i}' for i in range(1, extension)]}
        best = [r for r in previous['candidates'] if r['task'] == 'price' and r['namespace'] in allowed]
        baseline = float(np.mean([min(r['inner_score'] for r in best if r['horizon'] == h) for h in (1, 5)]))
        family_scores = {f: float(np.mean([min(r['inner_score'] for r in best if r['family'] == f and r['horizon'] == h)
                                           for h in (1, 5)])) for f in {r['family'] for r in best}}
        families = sorted(family_scores, key=lambda f: (family_scores[f], f))[:2]
        budgets = {family: 64 for family in families}
    namespace = round_name if not extension else f'{round_name}-extension-{extension}'
    for family, count in budgets.items():
        for h in (1, 5):
            candidates = recipes(family, h, experiment.identity['groups']['base'], count,
                                 task='direction' if round_name == 'B' else 'price',
                                 offset=0 if not extension else BUDGETS[round_name][family] + (extension - 1) * 64)
            for fold in experiment.identity['split']['folds']:
                decision = experiment.tune(family, h, fold, candidates, namespace)
                experiment.outer(decision['chosen'], fold, namespace)
    if extension:
        report = compare(experiment)
        completed = [r for r in report['candidates'] if r['namespace'] == namespace]
        current = float(np.mean([min(r['inner_score'] for r in completed if r['horizon'] == h) for h in (1, 5)]))
        freeze_record(extension_path, {'baseline_inner_score': baseline, 'new_inner_score': current,
                      'relative_improvement': max(0., (baseline - current) / baseline),
                      'families': sorted(budgets), 'selection_used_outer': False})


def reference_report(experiment):
    result = {}
    for h in (1, 5):
        predictions, labels, frequencies, prices, actual = [], [], [], [], []
        for fold in experiment.identity['split']['folds']:
            test = rows_at(experiment.history, fold['origins'])
            train = experiment.train_rows(test.date.min(), {})
            past = train[f'target_return_{h}'].to_numpy()
            frequency = np.bincount(class_labels(train, h), minlength=3).astype(float) + 1
            frequency /= frequency.sum()
            frequencies.extend([frequency.tolist()] * len(test))
            predictions.extend([float(np.median(past))] * len(test))
            labels.extend(class_labels(test, h))
            prices.extend(test.cotton_close)
            actual.extend(test[f'target_return_{h}'])
        p, y = np.asarray(prices), np.asarray(actual)
        result[str(h)] = {'naive': evaluate(p, y, np.zeros(len(p))),
                          'train_median_return': evaluate(p, y, np.asarray(predictions)),
                          'past_class_frequency_and_majority': class_metrics(np.asarray(labels), frequencies),
                          'policy': 'Train labels available before each outer block; Laplace class frequencies; same origins'}
    return result


def compare(experiment):
    print('STAGE reporting: reading outer predictions and compact timing receipts; no fits', flush=True)
    candidates = []
    for namespace in sorted((experiment.root / 'outer').glob('*')):
        grouped = {}
        for file in namespace.glob('*.json'):
            row = read_record(file)
            if 'recipe' not in row:
                continue
            key = (row['recipe']['family'], row['recipe']['horizon'], row['recipe'].get('task', 'price'))
            grouped.setdefault(key, []).append(row)
        for (family, h, task), folds in grouped.items():
            if len(folds) != len(experiment.identity['split']['folds']):
                continue
            folds.sort(key=lambda f: f['fold'])
            if task == 'price':
                paired, wins = [], 0
                for fold in folds:
                    rows = rows_at(experiment.history, fold['origins'])
                    naive = evaluate(rows.cotton_close.to_numpy(), rows[f'target_return_{h}'].to_numpy(), np.zeros(len(rows)))
                    wins += fold['metrics']['mae'] < naive['mae']
                    paired.append((rows, SimpleNamespace(name=family, horizon=h, predictions=np.asarray(fold['predictions']), parameters={})))
                metrics = summarize(paired, bootstrap_replicates=10000
                                    if experiment.identity.get('profile') == 'free-data-v1' else 1000)
                gate = historical_gate(metrics, wins, h)
                if len(folds) != 8:
                    gate.update(passes=False, reason='shorter_source_cohort_research_only', total_folds=len(folds))
            else:
                rows = rows_at(experiment.history, [d for fold in folds for d in fold['origins']])
                metrics = class_metrics(class_labels(rows, h), np.concatenate([fold['predictions'] for fold in folds]))
                gate = {'role': 'separate_direction_model_not_price_gate_substitute'}
            candidates.append({'family': family, 'horizon': h, 'task': task, 'namespace': namespace.name,
                               'inner_score': float(np.mean([f['inner_score'] for f in folds])),
                               'metrics': metrics, 'gate': gate,
                               'interval_evaluation_by_fold': {str(f['fold']): f['uncertainty']['evaluation']
                                   for f in folds if 'uncertainty' in f}})
    report = {'evidence_role': 'seen_historical_research', 'audit_used': False, 'candidates': candidates,
              'references': reference_report(experiment),
              'completed_fits': len(list((experiment.ledger.root / 'completed').glob('*.json')))}
    report['ledger_status'] = experiment.ledger.summary()
    report['timing'] = report['ledger_status']['timing']
    path = experiment.root / 'reports' / f'comparison-{content_id(report)[:16]}.json'
    freeze_record(path, report)
    lines = ['CottonLens continuous research â€” seen historical evidence, not an independent holdout.',
             f'Completed fits: {report["completed_fits"]}', '2024+ audit was not used for research or selection.']
    for row in candidates:
        metrics = row['metrics']
        lines.append(f"{row['namespace']} {row['family']} T+{row['horizon']} {row['task']} inner={row['inner_score']:.6f} "
                     f"MAE={metrics.get('mae', 'n/a')} gain={metrics.get('relative_mae_improvement_pct', 'n/a')} "
                     f"direction={metrics['directional_accuracy']:.3f}% gate={row['gate']}")
    path.with_suffix('.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print('REPORT:', path.with_suffix('.txt'), flush=True)
    from cottonlens_ml.research.diagnostics import write_lesson_packet
    write_lesson_packet(experiment, report)
    return report


def lock(experiment):
    if (experiment.root / 'locked.json').exists():
        existing = read_record(experiment.root / 'locked.json')
        if existing['identity'] != content_id(experiment.identity):
            raise ValueError('Locked research identity changed')
        return existing
    report = compare(experiment)
    chosen = {}
    for h in (1, 5):
        rows = [r for r in report['candidates'] if r['task'] == 'price' and r['horizon'] == h]
        if not rows:
            raise ValueError('Both horizons need complete eight-fold evidence')
        best = min(rows, key=lambda r: (r['inner_score'], r['family'], r['namespace']))
        chosen[str(h)] = {**best, 'primary_forecast': best['family'] if best['gate']['passes'] else 'Naive'}
    freeze_record(experiment.root / 'locked.json', {'identity': content_id(experiment.identity), 'horizons': chosen,
                  'prospective_start': datetime.now(UTC).isoformat(), 'prospective_origins': 126})


def reproduce(experiment):
    locked = read_record(experiment.root / 'locked.json')
    for h, candidate in locked['horizons'].items():
        differences = []
        for fold in experiment.identity['split']['folds']:
            path = experiment.root / 'decisions' / candidate['namespace'] / f'{candidate["family"]}-t{h}-fold{fold["fold"]}.json'
            decision = read_record(path)
            fresh = experiment.outer(decision['chosen'], fold, candidate['namespace'], repeat='reproduction')
            original = read_record(experiment.root / 'outer' / candidate['namespace'] / path.name)
            differences.append(float(np.max(np.abs(np.asarray(fresh['predictions']) - original['predictions']))))
        receipt = {'max_abs_log_return_difference': max(differences), 'fresh_fits': True,
                   'status': 'passed' if max(differences) <= 1e-6 else 'failed'}
        freeze_record(experiment.root / f'reproduction-t{h}.json', receipt)
        if receipt['status'] != 'passed':
            raise ValueError('Fresh GPU reproduction exceeds 1e-6; release verification blocked')


def refinement_search(experiment, round_name):
    from cottonlens_ml.research.refinements import run_refinement
    run_refinement(experiment, round_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--drive-root', type=Path, required=True)
    parser.add_argument('--experiment', default='research-v1')
    parser.add_argument('--stage', choices=('status', 'report', 'track', 'prepare', 'diagnose', 'ablate', 'search', 'compare', 'lock', 'reproduce', 'export', 'round', 'pilot-plan', 'pilot'), required=True)
    parser.add_argument('--round', choices=('A', 'B', 'C', 'D', 'E'), default='A')
    parser.add_argument('--extension', type=int, default=0)
    parser.add_argument('--publication-package', type=Path, action='append', default=[])
    parser.add_argument('--profile', choices=('legacy', 'free-data-v1', 'full-year-v1', 'ams-exploration-v1', 'fas-exploration-v1', 'nass-exploration-v1', 'wasde-exploration-v1', 'cftc-exploration-v1', 'fx-exploration-v1', 'crop-exploration-v1', 'weather-exploration-v1', 'oncall-exploration-v1', 'oncall-exploration-v2', 'recency-pilot-v1', 'availability-clock-pilot-v1', 'return-path-pilot-v1', 'agri-transfer-pilot-v1', 'agri-nonlinear-pilot-v1', 'statistical-pilot-v1', 'nonlinear-path-pilot-v1', 'wasde-text-pilot-v1', 'fundamental-joint-pilot-v1', 'wasde-regional-t1-pilot-v1', 'nass-regional-t1-pilot-v1', 'weak-signal-control-v1', 'paired-price-loss-control-v1', 'contract-curve-t1-pilot-v1', 'contract-curve-t5-pilot-v1', 'named-label-control-t5-v1', 'named-capacity-control-t5-v1'), default='legacy')
    parser.add_argument('--registration-root', type=Path)
    parser.add_argument('--decision-contract', type=Path)
    parser.add_argument('--market-file', type=Path)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--mirror-root', type=Path, help='Separate Drive root for local-first CPU copies')
    parser.add_argument('--ams-table', type=Path)
    parser.add_argument('--ams-publications', type=Path)
    parser.add_argument('--fas-table', type=Path)
    parser.add_argument('--nass-table', type=Path)
    parser.add_argument('--nass-audit-root', type=Path)
    parser.add_argument('--wasde-table', type=Path)
    parser.add_argument('--wasde-corpus', type=Path)
    parser.add_argument('--information-inputs', type=Path)
    parser.add_argument('--cftc-table', type=Path)
    parser.add_argument('--fx-table', type=Path)
    parser.add_argument('--crop-table', type=Path)
    parser.add_argument('--weather-table', type=Path)
    parser.add_argument('--oncall-table', type=Path)
    parser.add_argument('--max-minutes', type=float, default=60.)
    args = parser.parse_args()
    if args.profile == 'named-capacity-control-t5-v1':
        from cottonlens_ml.research.named_capacity_control import dispatch
        dispatch(args)
        return
    if args.profile == 'named-label-control-t5-v1':
        from cottonlens_ml.research.named_label_control import dispatch
        dispatch(args)
        return
    if args.profile in ('contract-curve-t1-pilot-v1', 'contract-curve-t5-pilot-v1'):
        from cottonlens_ml.research.contract_curve_execution import dispatch
        dispatch(args)
        return
    if args.profile == 'paired-price-loss-control-v1':
        from cottonlens_ml.research.paired_price_loss import dispatch
        dispatch(args)
        return
    if args.profile == 'weak-signal-control-v1':
        from cottonlens_ml.research.weak_signal import dispatch
        dispatch(args)
        return
    if args.profile == 'nass-regional-t1-pilot-v1':
        from cottonlens_ml.research.nass_regional_execution import dispatch
        dispatch(args)
        return
    if args.profile == 'wasde-regional-t1-pilot-v1':
        from cottonlens_ml.research.wasde_regional_execution import dispatch
        dispatch(args)
        return
    if args.profile == 'availability-clock-pilot-v1':
        from cottonlens_ml.research.availability_clock import dispatch
        dispatch(args)
        return
    if args.profile in ('oncall-exploration-v1', 'oncall-exploration-v2'):
        from cottonlens_ml.research.oncall_exploration import dispatch
        dispatch(args)
        return
    if args.profile == 'weather-exploration-v1':
        from cottonlens_ml.research.weather_exploration import dispatch
        dispatch(args)
        return
    if args.profile == 'crop-exploration-v1':
        from cottonlens_ml.research.crop_exploration import dispatch
        dispatch(args)
        return
    if args.profile == 'fx-exploration-v1':
        from cottonlens_ml.research.fx_exploration import dispatch
        dispatch(args)
        return
    if args.profile == 'cftc-exploration-v1':
        from cottonlens_ml.research.cftc_exploration import dispatch
        dispatch(args)
        return
    if args.profile == 'fundamental-joint-pilot-v1':
        from cottonlens_ml.research.fundamental_joint import dispatch
        dispatch(args)
        return
    if args.profile == 'wasde-text-pilot-v1':
        from cottonlens_ml.research.wasde_narrative import dispatch
        dispatch(args)
        return
    if args.profile == 'wasde-exploration-v1':
        from cottonlens_ml.research.wasde_exploration import dispatch
        dispatch(args)
        return
    if args.profile == 'nass-exploration-v1':
        from cottonlens_ml.research.nass_exploration import dispatch
        dispatch(args)
        return
    if args.profile == 'fas-exploration-v1':
        from cottonlens_ml.research.fas_exploration import dispatch
        dispatch(args)
        return
    if args.profile == 'ams-exploration-v1':
        from cottonlens_ml.research.ams_exploration import dispatch
        dispatch(args)
        return
    if args.profile == 'nonlinear-path-pilot-v1':
        from cottonlens_ml.research.nonlinear_path import dispatch
        dispatch(args)
        return
    if args.profile == 'statistical-pilot-v1':
        from cottonlens_ml.research.statistical_pilot import dispatch
        dispatch(args)
        return
    if args.profile == 'agri-nonlinear-pilot-v1':
        from cottonlens_ml.research.nonlinear_transfer import dispatch
        dispatch(args)
        return
    if args.profile == 'agri-transfer-pilot-v1':
        from cottonlens_ml.research.transfer import dispatch
        dispatch(args)
        return
    if args.profile == 'return-path-pilot-v1':
        from cottonlens_ml.research.path_pilot import dispatch
        dispatch(args)
        return
    if args.profile == 'recency-pilot-v1':
        from cottonlens_ml.research.recency import dispatch
        dispatch(args)
        return
    if args.profile == 'full-year-v1':
        from cottonlens_ml.research.full_year import dispatch
        dispatch(args)
        return
    if args.stage == 'status':
        from cottonlens_ml.research.pilot import saved_pilot_status
        folder = root_path(args.drive_root, args.experiment)
        ready = read_record(folder / 'ready.json') if (folder / 'ready.json').exists() else None
        print(json.dumps({'prepared': ready is not None,
              'writer_lock_present': (folder / '.writer-lock').exists(),
              'ledger': Ledger(folder / 'ledger', {}).summary(),
              'pilot': saved_pilot_status(folder)}, indent=2))
        return
    if args.stage == 'prepare':
        prepare(args.repo, args.drive_root, args.experiment, args.publication_package, args.profile)
        return
    if args.stage not in ('report', 'compare', 'track', 'pilot-plan'):
        require_colab_training()
    experiment = Experiment(root_path(args.drive_root, args.experiment), repo=args.repo)
    if args.stage == 'pilot-plan':
        from cottonlens_ml.research.pilot import pilot_plan
        plan = pilot_plan(experiment.identity, experiment.history)
        print(json.dumps({**{k: v for k, v in plan.items() if k != 'jobs'},
                          'candidate_count': len(plan['jobs'])}, indent=2))
        return
    with writer(experiment.root):
        if args.stage == 'pilot':
            from cottonlens_ml.research.pilot import run_pilot
            print(json.dumps(run_pilot(experiment, max_minutes=args.max_minutes), indent=2))
            return
        if args.stage == 'track':
            from cottonlens_ml.research.tracking import project
            project(experiment)
            return
        if args.stage == 'ablate':
            from cottonlens_ml.research.ablation import run_ablation
            run_ablation(experiment)
        if args.stage in ('diagnose', 'round'):
            from cottonlens_ml.research.diagnostics import diagnose
            diagnose(experiment)
        if args.stage in ('search', 'round'):
            search(experiment, args.round, args.extension)
        elif args.stage == 'lock':
            lock(experiment)
        elif args.stage == 'reproduce':
            reproduce(experiment)
        elif args.stage == 'export':
            from cottonlens_ml.research.release import export
            export(experiment, args.repo)
        compare(experiment)


if __name__ == '__main__':
    main()
