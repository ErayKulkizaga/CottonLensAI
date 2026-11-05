"""Independent source, target and full-calendar bootstrap replay; never fits."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covariance(part):
    x, y, weight = part.T
    count = weight.sum()
    if count < 3 or not np.isfinite(part).all() or not np.isin(weight, [0, 1]).all():
        raise ValueError('Invalid weighted calendar')
    # Sufficient statistics, independent of the analysis centered-product code.
    return ((weight*x*y).sum() - (weight*x).sum()*(weight*y).sum()/count)/count


def bootstrap(parts, block, repetitions=10000):
    if block < 1 or repetitions < 2:
        raise ValueError('Invalid bootstrap configuration')
    count = sum(part[:, 2].sum() for part in parts)
    point = sum(covariance(part)*part[:, 2].sum() for part in parts)/count
    rng = np.random.default_rng(42)
    draws = []
    for offset in range(0, repetitions, 128):
        batch = min(128, repetitions-offset)
        total = np.zeros(batch)
        samples = np.zeros(batch)
        for part in parts:
            size = min(block, len(part))
            starts = rng.integers(0, len(part)-size+1,
                                  (batch, math.ceil(len(part)/size)))
            positions = (starts[..., None]+np.arange(size)).reshape(batch, -1)[:, :len(part)]
            drawn = part[positions]
            x, y, weight = drawn[..., 0], drawn[..., 1], drawn[..., 2]
            n = weight.sum(axis=1)
            if (n < 3).any():
                raise ValueError('Insufficient sampled source coverage')
            total += ((weight*x*y).sum(axis=1)
                      - (weight*x).sum(axis=1)*(weight*y).sum(axis=1)/n)
            samples += n
        draws.extend(total/samples)
    draws = np.asarray(draws)
    centered = draws-draws.mean()
    return float(point), (point-np.quantile(centered, [.975, .025])).tolist()


def verify(root, report, observations):
    root, report, observations = map(Path, (root, report, observations))
    contract = json.loads((root/'contract.json').read_text(encoding='utf-8'))
    result = json.loads((root/'result.json').read_text(encoding='utf-8'))
    for path, expected in [(report, contract['report_sha256']),
                           (observations, contract['observations_sha256']),
                           (root/'contract.json', result['contract_sha256']),
                           (root/'moments.csv', result['moments_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Frozen input hash mismatch')
    if any(result[name] for name in ['new_fits', 'model_skill_established',
                                    'training_allowed', 'source_clock_verified']):
        raise ValueError('Unexpected training or skill claim')
    ranked, prices = defaultdict(set), defaultdict(dict)
    with observations.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['source_date_matches'] != 'True':
                continue
            day, name = row['date'], row['contract']
            ranked[day].add(name)
            if row['price_precision_2dp'] != 'True':
                continue
            value = Decimal(row['settlement_cents_raw'])
            if value <= 0 or (name in prices[day] and prices[day][name] != value):
                raise ValueError('Invalid or conflicting source price')
            prices[day][name] = value
    frame = pd.read_csv(root/'moments.csv', float_precision='round_trip')
    source = pd.read_csv(report, float_precision='round_trip')
    keys = ['date', 'target_date', 'contract', 'eligible', 'named_actual', 'baseline', 'year']
    if (len(frame) != 2006 or not frame[keys].equals(source[keys])
            or frame.date.duplicated().any()
            or sorted(frame.year.unique()) != list(range(2016, 2024))):
        raise ValueError('Origin, target or cohort mismatch')
    scalars = 0
    for row in frame.itertuples():
        names = sorted(ranked[row.date])
        usable = (row.eligible and len(names) >= 2
                  and names[0] in prices[row.date] and names[1] in prices[row.date])
        if usable != bool(row.moment_weight):
            raise ValueError('Source coverage differs')
        if not usable:
            if row.front_slope != 0 or row.relative_error != 0:
                raise ValueError('Missing slot contributes to moment')
            continue
        if row.contract != names[1] or row.target_date <= row.date:
            raise ValueError('Wrong target contract or date')
        with localcontext() as context:
            context.prec = 40
            origin_price = prices[row.date][row.contract]
            target_price = prices[row.target_date].get(row.contract)
            if target_price is None:
                raise ValueError('Missing target source')
            actual = float((target_price/origin_price).ln()**2)
            feature = float((prices[row.date][names[0]]/origin_price).ln())
        error = actual/row.baseline-1
        for expected, observed in [(actual, row.named_actual), (feature, row.front_slope),
                                   (error, row.relative_error)]:
            if not math.isclose(expected, observed, rel_tol=1e-11, abs_tol=1e-13):
                raise ValueError('Independent source scalar differs')
            scalars += 1
    groups = list(frame.groupby('year', sort=True))
    parts = [group[['front_slope', 'relative_error', 'moment_weight']].to_numpy()
             for _, group in groups]
    positive = 0
    for (_, group), part, annual in zip(groups, parts, result['yearly'], strict=True):
        if (annual['year'] != int(group.year.iloc[0]) or annual['ordinal'] != len(group)
                or annual['eligible'] != int(group.moment_weight.sum())
                or not math.isclose(covariance(part), annual['covariance'], abs_tol=1e-12)):
            raise ValueError('Annual moment differs')
        positive += covariance(part) > 0
    for block in (20, 60):
        point, limits = bootstrap(parts, block)
        stored = result['bootstrap'][str(block)]
        if (not math.isclose(point, stored['covariance'], abs_tol=1e-12)
                or not np.allclose(limits, stored['ci_95'], rtol=0, atol=1e-12)):
            raise ValueError('Independent full-calendar interval differs')
    trimmed = frame.copy()
    eligible = trimmed.loc[trimmed.moment_weight.eq(1)]
    excluded = eligible.nlargest(math.ceil(len(eligible)*.01), 'baseline_loss').index
    trimmed.loc[excluded, 'moment_weight'] = 0
    trim_parts = [g[['front_slope', 'relative_error', 'moment_weight']].to_numpy()
                  for _, g in trimmed.groupby('year', sort=True)]
    count = sum(part[:, 2].sum() for part in trim_parts)
    trim = sum(covariance(part)*part[:, 2].sum() for part in trim_parts)/count
    passed = (all(result['bootstrap'][str(b)]['ci_95'][0] > 0 for b in (20, 60))
              and positive >= 6 and trim > 0)
    if (not math.isclose(trim, result['top1pct_trim_covariance'], abs_tol=1e-12)
            or bool(passed) != result['screen_passed']
            or len(eligible) != result['eligible_origins']):
        raise ValueError('Sensitivity, screen or population differs')
    return {'schema': 1, 'status': 'verified', 'new_fits': 0,
            'retained_origins': len(frame), 'eligible_origins': len(eligible),
            'independent_decimal_scalars': scalars, 'independent_intervals': 2,
            'positive_years': int(positive), 'trimmed_origin_count': len(excluded),
            'screen_passed': bool(passed), 'model_skill_established': False,
            'source_clock_verified': False, 'result_sha256': sha(root/'result.json'),
            'moments_sha256': sha(root/'moments.csv'), 'report_sha256': sha(report),
            'observations_sha256': sha(observations), 'verifier_sha256': sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'report', 'observations', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.root, args.report, args.observations)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
