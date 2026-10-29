"""One fixed zero-fit front-curve versus named EWMA-error moment screen."""

import argparse
import csv
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from research.ams_upland_futures import sha

PROFILE = 'ams-front-slope-named-ewma-error-t1-v1'
MOTIVATION = 'https://onlinelibrary.wiley.com/doi/10.1002/fut.22250'


def source_quotes(path):
    ranked = defaultdict(set)
    values = defaultdict(dict)
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['source_date_matches'] != 'True':
                continue
            day, contract = row['date'], row['contract']
            ranked[day].add(contract)
            if row['price_precision_2dp'] == 'True':
                price = Decimal(row['settlement_cents_raw'])
                if price <= 0:
                    raise ValueError('Nonpositive settlement')
                prior = values[day].get(contract)
                if prior is not None and abs(prior-price) > Decimal('1e-8'):
                    raise ValueError('Conflicting named settlement')
                values[day][contract] = price
    return ranked, values


def weighted_covariance(part):
    x, y, weight = part.T
    count = weight.sum()
    if count < 3:
        raise ValueError('Too few eligible observations')
    xbar = (x*weight).sum()/count
    ybar = (y*weight).sum()/count
    return float((weight*(x-xbar)*(y-ybar)).sum()/count)


def pooled(parts):
    counts = [part[:, 2].sum() for part in parts]
    return sum(n*weighted_covariance(part) for n, part in zip(counts, parts, strict=True))/sum(counts)


def interval(parts, block, repetitions=10000):
    point = pooled(parts)
    rng = np.random.default_rng(42)
    draws = []
    for start in range(0, repetitions, 128):
        batch = min(128, repetitions-start)
        total = np.zeros(batch)
        count_total = np.zeros(batch)
        for part in parts:
            size = min(block, len(part))
            begins = rng.integers(0, len(part)-size+1, (batch, math.ceil(len(part)/size)))
            index = (begins[..., None]+np.arange(size)).reshape(batch, -1)[:, :len(part)]
            sampled = part[index]
            x, y, weight = sampled.transpose(2, 0, 1)
            count = weight.sum(axis=1)
            if (count < 3).any():
                raise ValueError('Block draw lost source coverage')
            xbar = (x*weight).sum(axis=1)/count
            ybar = (y*weight).sum(axis=1)/count
            total += (weight*(x-xbar[:, None])*(y-ybar[:, None])).sum(axis=1)
            count_total += count
        draws.extend(total/count_total)
    noise = np.asarray(draws)-np.mean(draws)
    return {'covariance': float(point),
            'ci_95': (point-np.quantile(noise, [.975, .025])).tolist(),
            'block': block, 'replicates': repetitions}


def prepare(root, report, observations):
    root.mkdir(parents=True, exist_ok=False)
    contract = {'profile': PROFILE, 'new_fits': 0,
                'report_path': str(report.resolve()), 'report_sha256': sha(report),
                'observations_path': str(observations.resolve()),
                'observations_sha256': sha(observations), 'analysis_sha256': sha(__file__),
                'source': 'USDA AMS historical upland Futures settlement archives',
                'source_clock': 'date+1day00UTC assumed; first vintage unverified',
                'feature': 'origin log(F1/F2), first two dated listed futures contracts',
                'error': 'next same-second-contract squared log return / origin named EWMA0.94 - 1',
                'statistic': 'eligible-origin-weighted within-year covariance(feature,error)',
                'positive_hypothesis': 'front backwardation indicates tighter storage and higher next-day risk than EWMA',
                'motivation': MOTIVATION,
                'screen': 'both 20/60 full-ordinal weighted block CI lower>0; >=6/8 positive years; positive top1pct EWMA-loss trim',
                'bootstrap': '10000 draws/seed42, resample within ordinal year and recenter eligible rows',
                'scope': 'seen 2016-2023 named T+1/2006 ordinal origins; only one fixed slope sign',
                'model_skill_established': False, 'training_allowed': False,
                'independent_holdout': False, 'source_clock_verified': False,
                'multiplicity_adjusted': False, 'history_reused': True,
                '2024_plus_used_for_selection': False}
    (root/'contract.json').write_text(json.dumps(contract, indent=2)+'\n', encoding='utf-8')
    return {'profile': PROFILE, 'new_fits': 0, 'status': 'prepared'}


def run(root):
    contract = json.loads((root/'contract.json').read_text(encoding='utf-8'))
    report, observations = map(Path, (contract['report_path'], contract['observations_path']))
    if (sha(report) != contract['report_sha256']
            or sha(observations) != contract['observations_sha256']
            or sha(__file__) != contract['analysis_sha256']):
        raise ValueError('Frozen source, report or analysis changed')
    frame = pd.read_csv(report, float_precision='round_trip')
    if (len(frame) != 2006 or sorted(frame.year.unique()) != list(range(2016, 2024))
            or frame.date.duplicated().any() or int(frame.eligible.sum()) != 1989):
        raise ValueError('Frozen eight-year ordinal named cohort required')
    ranked, quotes = source_quotes(observations)
    feature, error, weight = [], [], []
    for row in frame.itertuples():
        contracts = sorted(ranked[row.date])
        second = contracts[1] if len(contracts) >= 2 else None
        if row.eligible and second != row.contract:
            raise ValueError('Source second contract differs from frozen target')
        if (row.eligible and len(contracts) >= 2
                and contracts[0] in quotes[row.date]
                and second in quotes[row.date]):
            first_price = float(quotes[row.date][contracts[0]])
            second_price = float(quotes[row.date][second])
            feature.append(math.log(first_price/second_price))
            error.append(row.named_actual/row.baseline-1)
            weight.append(1)
        else:
            feature.append(0.)
            error.append(0.)
            weight.append(0)
    frame['front_slope'] = feature
    frame['relative_error'] = error
    frame['moment_weight'] = weight
    if sum(weight) < 1900:
        raise ValueError('Insufficient front-slope coverage for fixed screen')
    parts = [g[['front_slope', 'relative_error', 'moment_weight']].to_numpy(float)
             for _, g in frame.groupby('year', sort=True)]
    yearly = [{'year': int(year), 'ordinal': len(g), 'eligible': int(g.moment_weight.sum()),
               'covariance': weighted_covariance(part)}
              for (year, g), part in zip(frame.groupby('year', sort=True), parts, strict=True)]
    bootstrap = {str(block): interval(parts, block) for block in (20, 60)}
    eligible = frame.loc[frame.moment_weight.eq(1)]
    trim_ids = eligible.nlargest(math.ceil(len(eligible)*.01), 'baseline_loss').index
    trimmed = frame.copy()
    trimmed.loc[trim_ids, 'moment_weight'] = 0
    trim_parts = [g[['front_slope', 'relative_error', 'moment_weight']].to_numpy(float)
                  for _, g in trimmed.groupby('year', sort=True)]
    trim_cov = pooled(trim_parts)
    passed = (all(value['ci_95'][0] > 0 for value in bootstrap.values())
              and sum(value['covariance'] > 0 for value in yearly) >= 6 and trim_cov > 0)
    result = {'profile': PROFILE, 'new_fits': 0, 'retained_origins': len(frame),
              'eligible_origins': sum(weight), 'yearly': yearly,
              'bootstrap': bootstrap, 'top1pct_trim_covariance': float(trim_cov),
              'screen_passed': bool(passed), 'model_skill_established': False,
              'training_allowed': False, 'independent_holdout': False,
              'source_clock_verified': False, 'multiplicity_adjusted': False,
              'contract_sha256': sha(root/'contract.json')}
    if (root/'result.json').exists():
        raise ValueError('Preserve immutable result')
    frame.to_csv(root/'moments.csv', index=False, lineterminator='\n')
    result['moments_sha256'] = sha(root/'moments.csv')
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--observations', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.report, args.observations)
                     if args.stage == 'prepare' else run(args.root)))
