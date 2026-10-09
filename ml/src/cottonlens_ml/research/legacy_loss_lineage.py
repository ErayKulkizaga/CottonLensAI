"""No-fit audit of the existing legacy XGBoost target/loss search; not a new model."""
import argparse
import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.statistics import paired_bootstrap

PROFILE = 'legacy-loss-lineage-v1'
REFERENCE_SOURCE_ID = '73a92265cbdeb2ecae26f589193086fbfac1367673d8985ae02a97a12741b6a1'


def validate_recipe(recorded, generated):
    """Keep exact recorded floats; permit only tiny generator arithmetic drift."""
    if recorded.get('seed') not in (17, 42, 101):
        raise ValueError('Only the original three seeds admitted')
    normalized = {**recorded, 'seed': 42}
    if ({k: v for k, v in normalized.items() if k != 'params'} !=
            {k: v for k, v in generated.items() if k != 'params'} or
            normalized['params'].keys() != generated['params'].keys()):
        raise ValueError('Recorded recipe structure differs from frozen generator')
    differences = {}
    for name, value in normalized['params'].items():
        if not np.isfinite([value, generated['params'][name]]).all():
            raise ValueError('Finite recipe parameters required')
        np.testing.assert_array_max_ulp(np.asarray(value, float), np.asarray(generated['params'][name], float), maxulp=8)
        if value != generated['params'][name]:
            differences[name] = {'recorded': value, 'regenerated_here': generated['params'][name]}
    return differences


def generated_candidates(features):
    """Decode the authenticated v2 XGBoost generator without executing its archive."""
    rng = np.random.default_rng(int(content_id({'family': 'xgboost', 'horizon': 1, 'task': 'price'})[:8], 16))

    def log(low, high):
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))

    result = []
    for index in range(128):
        params = {'max_depth': int(rng.integers(1, 9)), 'eta': log(.005, .15),
                  'min_child_weight': log(1, 100), 'alpha': 0. if index % 3 == 0 else log(1e-6, 10),
                  'lambda': log(1e-3, 100), 'subsample': float(rng.uniform(.6, 1)),
                  'colsample_bytree': float(rng.uniform(.6, 1))}
        target = ('scaled_log', 'price_delta')[index % 2]
        loss = ('reg:squarederror', 'reg:absoluteerror')[(index // 2) % 2]
        result.append({'family': 'xgboost', 'horizon': 1, 'task': 'price', 'params': params, 'seed': 42,
                       'target': target, 'loss': loss, 'window': 1, 'features': list(features), 'years': None,
                       'cadence': 126, 'hypothesis': f'xgboost price: candidate {index}; target={target}; loss={loss}'})
    return result


def run(input_root, archive_manifest, output):
    if not __debug__:
        raise RuntimeError('Scientific integrity assertions must remain enabled')
    base, primary = Path(output).resolve(), Path(input_root).resolve()
    assert not base.is_relative_to(primary), 'Output must not modify original inputs'
    base.mkdir(parents=True, exist_ok=True)
    root = primary / 'output/claude-audit/drive/experiments/research-v2-tf-placement'
    source_zip = primary / 'output/claude-audit/drive/sources/cottonlens-research-source-v2.zip'
    archive_manifest = Path(archive_manifest)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(archive_manifest) == '8f5d46a65f5345195df7b5e8c02e08b87acb294c0cd0b977e0dba450d81144f9'
    with gzip.open(archive_manifest, 'rt', encoding='utf-8') as f:
        archive = json.load(f)
    public = {x['path']: x for x in archive['files']}
    inputs = {}

    def checked(p):
        name = p.relative_to(primary).as_posix()
        expected = public[name]
        assert not expected['sanitized'] and sha(p) == expected['sha256']
        inputs[name] = expected['sha256']
        return p

    ready = read_record(checked(root / 'ready.json'))
    history = pd.read_parquet(checked(root / 'history.parquet'))
    assert sha(root / 'history.parquet') == ready['history_sha256']
    indexed = history.set_index('date', drop=False)
    with zipfile.ZipFile(checked(source_zip)) as z:
        source = json.loads(z.read('.source-manifest.json'))
        source_body = {k: v for k, v in source.items() if k != 'source_id'}
        assert source['source_id'] == ready['identity']['source_id'] == hashlib.sha256(json.dumps(source_body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert source['source_id'] == REFERENCE_SOURCE_ID
        assert all(safe_member(n) and hashlib.sha256(z.read(n)).hexdigest() == h for n, h in source['files'].items())
        candidates = generated_candidates(ready['identity']['groups']['base'])

    def candidate_id(p):
        return content_id({**p, 'seed': 42})

    candidate_ids = {candidate_id(p) for p in candidates}
    assert len(candidate_ids) == 128
    analysis_source = research_source_identity(Path(__file__).resolve().parents[4])
    settings = {'profile': PROFILE, 'analysis_source_id': analysis_source['source_id'], 'reference_source_id': source['source_id'],
                'archive_manifest_sha256': sha(archive_manifest), 'source_archive_sha256': sha(source_zip),
                'new_fits': 0, 'candidate_count': 128, 'shortlist_count': 5, 'seeds': [17, 42, 101],
                'bootstrap_blocks': [20, 60], 'replicates': 10000, 'seed': 42, 'maximum_generator_ulp_difference': 8,
                'independent_holdout': False, 'release_allowed': False}
    freeze_record(base / 'source-manifest.json', analysis_source)
    freeze_record(base / 'settings.json', settings)
    records, recorded_recipes, numeric_differences, train_cache = {}, {}, {}, {}
    paths = sorted((root / 'ledger/completed').glob('*.json'))
    for i, p in enumerate(paths):
        r = read_record(p)
        spec = r['specification']
        if 'recipe' not in spec:
            assert not spec.get('role', '').startswith(('inner-score-', 'early-stop-', 'outer-A-'))
            continue  # Synthetic learning-control specifications are not market recipes.
        recipe = spec['recipe']
        match = re.fullmatch(r'xgboost price: candidate (\d+); target=(\w+); loss=([\w:]+)', recipe.get('hypothesis', ''))
        if recipe.get('horizon') != 1 or not match or int(match[1]) >= 128:
            continue
        role = spec['role']
        if not role.startswith(('inner-score-', 'early-stop-', 'outer-A-')):
            continue
        index = int(match[1])
        normalized = {**recipe, 'seed': 42}
        expected = candidates[index]
        differences = validate_recipe(recipe, expected)
        numeric_differences.update({f'{index}/{name}': value for name, value in differences.items()})
        if index in recorded_recipes:
            assert recorded_recipes[index] == normalized
        else:
            recorded_recipes[index] = normalized
        checked(p)
        assert r['identity'] == ready['identity']
        assert r['experiment_id'] == content_id({'identity': ready['identity'], 'specification': spec})
        cutoff = pd.Timestamp(min(spec['test_dates']))
        if cutoff not in train_cache:
            train = history.loc[(history.cotton_close > 0) & np.isfinite(history[['cotton_close', 'target_return_5']]).all(axis=1)
                                & history.target_date_5.notna() & (history.date < cutoff) & (history.target_date_5 < cutoff)
                                & (history.date >= history.date.iloc[119])].copy()
            coverage = ready['identity']['split'].get('coverage_start')
            if coverage:
                train = train.loc[train.date >= pd.Timestamp(coverage)].copy()
            train_cache[cutoff] = (train.date.dt.strftime('%Y-%m-%d').tolist(), frame_identity(train, list(train)))
        dates, training_identity = train_cache[cutoff]
        assert spec['train_dates'] == dates and spec['train_identity'] == training_identity

        key = (candidate_id(spec['recipe']), spec['recipe']['seed'], role)
        assert key not in records
        records[key] = {'id': r['experiment_id'], 'prediction': r['result']['predictions'], 'iterations': r['result']['iterations'],
                        'spec_iterations': spec['iterations'], 'test_dates': spec['test_dates'], 'origins': r['result']['origins'],
                        'train_last': max(spec['train_dates']), 'validation_dates': spec['validation_dates'],
                        'metrics': r['result']['metrics'], 'payload_files': r['files']}
        if i % 1000 == 0:
            print('READ', i, 'of', len(paths), 'selected-scope receipts', len(records), flush=True)
    assert set(recorded_recipes) == set(range(128))
    candidates = [recorded_recipes[i] for i in range(128)]  # Preserve exact historical floats; never round a fit identity.

    def complexity(p):
        return (p['params'].get('max_depth', 0), len(p['features']), p.get('window', 1), content_id(p))

    def inner(p, fold, seed):
        scores, iterations, used = [], [], []
        for i, block in enumerate(fold['inner']):
            raw = records[(candidate_id(p), seed, f'inner-score-{fold["fold"]}-{i}-0')]
            early = records[(candidate_id(p), seed, f'early-stop-{fold["fold"]}-{i}')]
            assert raw['test_dates'] == raw['origins'] == block['origins']
            assert raw['spec_iterations'] == early['iterations']
            h = indexed.loc[pd.to_datetime(block['origins'])]
            assert h.target_date_5.max() < pd.Timestamp(fold['origins'][0])
            assert pd.Timestamp(raw['train_last']) < h.date.min()
            actual, price = h.target_return_1.to_numpy(), h.cotton_close.to_numpy()
            a = price * np.exp(actual)
            prediction = price * np.exp(np.asarray(raw['prediction'], float))
            loss, naive = np.abs(a - prediction).mean(), np.abs(a - price).mean()
            np.testing.assert_allclose(loss, raw['metrics']['mae'], rtol=1e-12, atol=1e-12)
            scores.append(float(loss / naive))
            iterations.append(raw['iterations'])
            used.extend([raw['id'], early['id']])
        return {'recipe': {**p, 'seed': seed}, 'score': float(np.mean(scores)), 'iterations': max(1, int(np.median(iterations))), 'records': used}

    results, frames = [], []
    for fold in ready['identity']['split']['folds']:
        evaluated = [inner(p, fold, 42) for p in candidates]
        shortlist = sorted(evaluated, key=lambda r: (r['score'], complexity(r['recipe'])))[:5]
        confirmed = []
        for p in shortlist:
            seeds = [inner(p['recipe'], fold, seed) for seed in (17, 42, 101)]
            confirmed.append({'recipe': p['recipe'], 'score': float(np.mean([x['score'] for x in seeds])), 'seeds': seeds})
        chosen = min(confirmed, key=lambda r: (r['score'], complexity(r['recipe'])))
        h = indexed.loc[pd.to_datetime(fold['origins'])]
        members, ids = [], []
        for seed in chosen['seeds']:
            raw = records[(candidate_id(chosen['recipe']), seed['recipe']['seed'], f'outer-A-{fold["fold"]}-{h.date.min()}')]
            assert raw['spec_iterations'] == seed['iterations'] and raw['test_dates'] == raw['origins'] == fold['origins']
            assert pd.Timestamp(raw['train_last']) < h.date.min()
            members.append(raw['prediction']); ids.append(raw['id'])
        forecast = np.mean(members, axis=0)
        actual_price = h.cotton_close.to_numpy() * np.exp(h.target_return_1.to_numpy())
        predicted_price = h.cotton_close.to_numpy() * np.exp(forecast)
        frame = pd.DataFrame({'date': fold['origins'], 'target_date': h.target_date_1.dt.strftime('%Y-%m-%d').to_numpy(),
                              'year': fold['year'], 'cotton_close': h.cotton_close.to_numpy(),
                              'actual_return': h.target_return_1.to_numpy(), 'predicted_return': forecast,
                              'naive_error': np.abs(actual_price - h.cotton_close.to_numpy()), 'model_error': np.abs(actual_price - predicted_price)})
        frames.append(frame)
        results.append({'year': fold['year'], 'chosen': chosen, 'outer_fit_ids': ids,
                        'target': chosen['recipe']['target'], 'loss': chosen['recipe']['loss'],
                        'price_mae': float(frame.model_error.mean()), 'naive_mae': float(frame.naive_error.mean()),
                        'gain_pct': float(100 * (1 - frame.model_error.mean() / frame.naive_error.mean()))})
        print('RECONSTRUCTED', fold['year'], results[-1]['target'], results[-1]['loss'], 'gain', results[-1]['gain_pct'], flush=True)
    combined = pd.concat(frames, ignore_index=True)
    assert len(combined) == 1008 and not combined.date.duplicated().any()
    paired = {str(b): paired_bootstrap([f[['naive_error', 'model_error']].to_numpy() for f in frames], block=b, repetitions=10000, seed=42,
                                      normalization=float(combined.naive_error.mean())) for b in (20, 60)}
    result = {'status': 'reconstructed_legacy_selection_from_saved_predictions', 'source_id': source['source_id'],
              'new_fits': 0, 'input_hashes': inputs, 'unique_origins': len(combined), 'model_receipts': len(records),
              'mean_gain_pct': float(100 * (1 - combined.model_error.mean() / combined.naive_error.mean())),
              'direction_pct': float(100 * np.mean(np.sign(combined.actual_return) == np.sign(combined.predicted_return))),
              'year_wins': sum(x['gain_pct'] > 1e-10 for x in results), 'years': results, 'paired_bootstrap': paired,
              'loss_effect_isolated': False, 'serialized_model_inference_verified': False, 'independent_holdout': False}
    result['generator_float_differences'] = numeric_differences
    result['settings'] = settings
    result['price_strategy_classification'] = 'DECISIVE_NEGATIVE'
    result['loss_ablation_classification'] = 'INCONCLUSIVE'
    result['serialized_payloads_available'] = sum((root / 'ledger' / n).is_file() for r in records.values() for n in r['payload_files'])
    result['historical_source_admitted'] = False
    result['train_cohorts_verified'] = len(train_cache)
    encoded = combined.to_csv(index=False, lineterminator='\n').encode()
    export = base / 'outer-predictions.csv'
    if export.exists() and export.read_bytes() != encoded:
        raise ValueError('Frozen prediction export changed')
    if not export.exists():
        export.write_bytes(encoded)
    result['predictions_sha256'] = digest(export)
    freeze_record(base / 'lineage-report.json', result)
    print('GAIN', result['mean_gain_pct'], 'yearwins', result['year_wins'], 'direction', result['direction_pct'], flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_root, args.archive_manifest, args.output)
