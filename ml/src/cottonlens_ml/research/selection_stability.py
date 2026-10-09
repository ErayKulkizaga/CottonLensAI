"""Read-only uncertainty diagnostic from frozen, past-only inner predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.review import training_rows

PROFILE = 'weak-signal-selection-stability-v1'
WEIGHTS = np.array([0., .25, .5, .75, 1.])


def scores(parts):
    return np.mean([p.mean(axis=0) / p[:, 0].mean() for p in parts], axis=0)


def bootstrap(parts, selected, *, block, repetitions=10000, seed=42):
    """Pair all weight losses; keep each of the three historical blocks intact."""
    parts = [np.asarray(p, float) for p in parts]
    if (len(parts) != 3 or block < 1 or repetitions < 20 or selected not in range(5)
            or any(p.ndim != 2 or p.shape[1] != 5 or not len(p) or
                   not np.isfinite(p).all() or (p < 0).any() or p[:, 0].mean() <= 0 for p in parts)):
        raise ValueError('Three finite paired loss blocks and valid settings required')
    original = scores(parts)
    if selected != int(np.argmin(original)):
        raise ValueError('Recorded selection must match historical scores')
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        batch = np.zeros((count, 5))
        for p in parts:
            width = min(block, len(p))
            starts = rng.integers(0, len(p) - width + 1, (count, int(np.ceil(len(p) / width))))
            indices = (starts[..., None] + np.arange(width)).reshape(count, -1)[:, :len(p)]
            means = p[indices].mean(axis=1)
            if (means[:, 0] <= 0).any():
                raise ValueError('Resampled baseline must be positive')
            batch += means / means[:, :1] / 3
        draws.append(batch)
    draws = np.concatenate(draws)
    choices = np.argmin(draws, axis=1)
    frequency = [float(np.mean(choices == i)) for i in range(5)]
    intervals = {}
    for name, reference in [('versus_zero', 0), ('versus_raw', 4)]:
        gain = original[reference] - original[selected]
        distribution = draws[:, reference] - draws[:, selected]
        noise = distribution - distribution.mean()
        intervals[name] = (100 * (gain - np.quantile(noise, [.975, .025]))).tolist()
    return {'block': block, 'repetitions': repetitions, 'seed': seed,
            'selection_frequencies': frequency, 'recorded_choice_frequency': frequency[selected],
            'score_gain_ci_pp': intervals,
            'scope': 'Conditional descriptive stability of saved inner losses, not market skill probability; intervals are not adjusted for selecting the winner.'}


def analyze(study, release_manifest, output, *, repetitions=10000):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output == study or output.is_relative_to(study):
        raise ValueError('Analysis output must be outside immutable study')
    manifest = json.loads(Path(release_manifest).read_bytes())
    inventory_path = study.parent / manifest['inventory_member']
    if digest(inventory_path) != manifest['inventory_sha256']:
        raise ValueError('Published input inventory changed')
    inventory = json.loads(inventory_path.read_bytes())
    hashes = {}

    def checked(relative):
        name = 'study/' + relative
        if not safe_member(relative) or name not in inventory:
            raise ValueError('Unregistered input path')
        path = study / relative
        expected = inventory[name]['sha256']
        if digest(path) != expected:
            raise ValueError('Published input changed: ' + relative)
        hashes[relative] = expected
        return path

    settings = {'profile': PROFILE, 'repetitions': repetitions, 'blocks': [20, 60], 'seed': 42,
                'weights': WEIGHTS.tolist(), 'study_archive_sha256': manifest['archive']['sha256'],
                'analysis_code_sha256': digest(Path(__file__)), 'new_fits': 0,
                'outer_outcomes_used_for_selection': False}
    output.mkdir(parents=True, exist_ok=True)
    source = research_source_identity(Path(__file__).resolve().parents[4])
    settings['analysis_source_id'] = source['source_id']
    freeze_record(output / 'source-manifest.json', source)
    freeze_record(output / 'settings.json', settings)
    results = []
    for condition in ('null', 'injected'):
        for seed in range(1201, 1211):
            directory = f'scenarios/{condition}-seed{seed}'
            ready = read_record(checked(directory + '/ready.json'))
            identity = ready['identity']
            history_path = checked(directory + '/history.parquet')
            if digest(history_path) != ready['history_sha256']:
                raise ValueError('History and ready disagree')
            history = pd.read_parquet(history_path)
            design = identity['design']
            if (design['condition'] != condition or design['seed'] != seed or
                    design['shrinkage_weights'] != WEIGHTS.tolist() or
                    [f['year'] for f in design['split']['folds']] != list(range(2016, 2024))):
                raise ValueError('Frozen scenario design changed')
            recipe = {**design['recipe'], 'features': design['groups']['numeric_D0']}
            indexed = history.set_index('date', drop=False)
            for fold in design['split']['folds']:
                parts = []
                for i, validation in enumerate(fold['inner']):
                    test = indexed.loc[pd.to_datetime(validation['origins'])]
                    if len(test) != 63 or test.target_date_5.max() >= pd.Timestamp(fold['origins'][0]):
                        raise ValueError('Inner labels must mature before the outer decision')
                    buckets = (test.cotton_session_index - test.cotton_session_index.iloc[0]) // 21
                    predictions = []
                    for j, (_, chunk) in enumerate(test.groupby(buckets, sort=True)):
                        train = training_rows(history, chunk.date.min(), recipe, identity['split'].get('coverage_start'))
                        spec = {'recipe': recipe, 'role': f'inner-{fold["year"]}-{i}-{j}', 'iterations': 1,
                                'repeat_reason': None,
                                'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(), 'validation_dates': [],
                                'test_dates': chunk.date.dt.strftime('%Y-%m-%d').tolist(),
                                'train_identity': frame_identity(train, list(train))}
                        key = content_id({'identity': identity, 'specification': spec})
                        record = read_record(checked(directory + f'/ledger/completed/{key}.json'))
                        if record['identity'] != identity or record['specification'] != spec or train.target_date_5.max() >= chunk.date.min():
                            raise ValueError('Inner receipt or maturity mismatch')
                        for name in record['files']:
                            checked(directory + '/ledger/' + name)
                        prediction = np.asarray(record['result']['predictions'], float)
                        if len(prediction) != len(chunk) or not np.isfinite(prediction).all():
                            raise ValueError('Finite aligned inner predictions required')
                        predictions.extend(prediction)
                    actual, price = test.target_return_1.to_numpy(), test.cotton_close.to_numpy()
                    losses = price[:, None] * np.abs(np.exp(actual[:, None]) - np.exp(np.array(predictions)[:, None] * WEIGHTS))
                    parts.append(losses)
                observed = scores(parts)
                decision = read_record(checked(directory + f'/weak-signal-decisions/numeric_D0-t1-year{fold["year"]}.json'))
                selected = int(np.argmin(observed))
                if decision['selection_used_outer'] or decision['selected']['weight'] != WEIGHTS[selected]:
                    raise ValueError('Recorded past-only selection mismatch')
                np.testing.assert_allclose(decision['selected']['inner_score'], observed[selected], rtol=1e-12, atol=1e-12)
                row = {'condition': condition, 'noise_seed': seed, 'year': fold['year'],
                       'selected_weight': float(WEIGHTS[selected]), 'scores': observed.tolist(),
                       'winner_runner_up_margin_pp': float(100 * (np.sort(observed)[1] - observed[selected])),
                       'single_block_weights': [float(WEIGHTS[np.argmin(scores([p]))]) for p in parts],
                       'leave_one_block_out_weights': [float(WEIGHTS[np.argmin(scores([p for j, p in enumerate(parts) if j != i]))]) for i in range(3)],
                       'bootstrap': {str(b): bootstrap(parts, selected, block=b, repetitions=repetitions) for b in (20, 60)}}
                results.append(row)
            print(f'VERIFIED {condition} seed{seed}: 8 historical selections', flush=True)
    summary = {}
    for condition in ('null', 'injected'):
        rows = [r for r in results if r['condition'] == condition]
        summary[condition] = {'year_selections': len(rows),
                              'median_winner_runner_up_margin_pp': float(np.median([r['winner_runner_up_margin_pp'] for r in rows])),
                              'leave_one_block_out_always_same': sum(all(w == r['selected_weight'] for w in r['leave_one_block_out_weights']) for r in rows),
                              'median_recorded_choice_frequency': {str(b): float(np.median([r['bootstrap'][str(b)]['recorded_choice_frequency'] for r in rows])) for b in (20, 60)}}
    body = {'settings': settings, 'input_hashes': hashes, 'inner_fit_receipts': 1440,
            'selection_count': len(results), 'new_fits': 0, 'summary': summary, 'results': results,
            'limitations': ['Historical reuse; no independent holdout.', 'Resampling saved losses does not refit models or measure full selection uncertainty.',
                            'Block60 inside a63-row block has few possible starts; apparent stability is not evidence of power.',
                            'No outer predictions or outcomes used to choose a new weight; original selections and market gates remain unchanged.']}
    freeze_record(output / 'selection-stability.json', body)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--release-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.study, args.release_manifest, args.output), indent=2))
