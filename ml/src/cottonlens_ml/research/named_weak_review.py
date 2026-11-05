"""Independent scalar/quadrature review of a completed synthetic control, zero fits."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research import named_weak_execution as pilot
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.research.protocol import rows_at


def expected_loss(price, mean, sigma, predicted):
    """Integrate the normal density instead of using the production CDF formula."""
    if not all(math.isfinite(x) for x in (price, mean, sigma, predicted)) or price <= 0 or sigma <= 0:
        raise ValueError('Finite price/mean/prediction and positive noise required')
    split = max(-10., min(10., (predicted - mean) / sigma))

    def integrand(z):
        return abs(math.exp(mean + sigma * z) - math.exp(predicted)) * math.exp(-z*z/2) / math.sqrt(2*math.pi)

    return price * (quad(integrand, -10, split, epsabs=1e-12)[0] + quad(integrand, split, 10, epsabs=1e-12)[0])


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready, history = pilot.verify(folder, repo)
    result = read_record(folder / 'reports/result.json')
    prediction_path = folder / 'reports/outer-predictions.csv'
    if result['predictions_sha256'] != digest(prediction_path):
        raise ValueError('Prediction checksum changed')
    predictions = pd.read_csv(prediction_path, float_precision='round_trip', keep_default_na=False, na_values=[''])
    fixed = ready['identity']['design']
    c, fold = fixed['calibration'], fixed['folds'][0]
    verified = 0
    expected_gains = {}
    for condition, seed in pilot.scenarios():
        name = f'{condition}-{seed}'
        rows = predictions.loc[predictions.scenario.eq(name)]
        if (rows.origin.tolist() != fold['origins'] or not rows.condition.eq(condition).all()
                or not rows.noise_seed.eq(seed).all()):
            raise ValueError('Missing, duplicated, selected or reordered synthetic origins')
        original = rows_at(history, fold['origins'])
        output = read_record(folder / 'weak-outputs' / f'{name}-{fold["year"]}.json')
        if (rows.target_date.tolist() != original.target_date_5.dt.strftime('%Y-%m-%d').tolist()
                or rows.eligible.tolist() != original.named_eligible.tolist()
                or rows.raw_log_return.tolist() != output['raw_predictions']
                or rows.selected_log_return.tolist() != output['selected_predictions']
                or not rows.weight.eq(output['weight']).all()):
            raise ValueError('Output/origin/target/price/eligibility mismatch')
        np.testing.assert_array_equal(rows.origin_price.to_numpy(), original.named_price.to_numpy())
        # Independent past-only shrinkage reconstruction from all three blocks.
        scores = []
        for weight in (0., .25, .5, .75, 1.):
            relative = []
            for dates in fold['inner']:
                block = rows_at(history, dates)
                noises = np.random.default_rng(seed).standard_normal(len(history) + 5)
                indices = pd.Index(history.date).get_indexer(block.date)
                baseline = losses = 0.
                for index, row in zip(indices, block.itertuples(), strict=True):
                    x = getattr(row, c['source'])
                    mean = 0. if condition == 'null' or not math.isfinite(x) else c['beta'] * (1. if x >= c['threshold'] else -1.)
                    noise = math.fsum(noises[index+1:index+6]) / math.sqrt(5.)
                    actual = row.named_price * math.exp(mean + c['sigma'] * noise)
                    baseline += abs(actual - row.named_price)
                    pred = output['inner_predictions'][row.date.strftime('%Y-%m-%d')]
                    losses += abs(actual - row.named_price * math.exp(weight * pred))
                relative.append(losses / baseline)
            scores.append(math.fsum(relative) / len(relative))
        np.testing.assert_allclose(scores, output['scores'], rtol=0, atol=1e-12)
        if output['weight'] != (0., .25, .5, .75, 1.)[min(range(5), key=lambda index: scores[index])]:
            raise ValueError('Past-only weight selection mismatch')
        innovations = np.random.default_rng(seed).standard_normal(len(history) + 5)
        indices = pd.Index(history.date).get_indexer(original.date)
        baseline, losses = [], {name: [] for name in ('raw', 'selected', 'oracle')}
        for index, row, source in zip(indices, rows.itertuples(), original.itertuples(), strict=True):
            x = getattr(source, c['source'])
            mean = 0. if condition == 'null' or not math.isfinite(x) else c['beta'] * (1. if x >= c['threshold'] else -1.)
            noise = math.fsum(innovations[index+1:index+6]) / math.sqrt(5.)
            if abs(row.oracle_mean - mean) > 1e-14 or abs(row.unit_overlap_noise - noise) > 1e-14:
                raise ValueError('Teacher/noise mismatch')
            if not row.eligible:
                if not math.isnan(row.actual_price) or not math.isnan(row.actual_log_return):
                    raise ValueError('Unknown actuals must remain unknown')
                continue
            actual = row.origin_price * math.exp(mean + c['sigma'] * noise)
            if abs(row.actual_price - actual) > 1e-10 or abs(row.actual_log_return - mean - c['sigma'] * noise) > 1e-14:
                raise ValueError('Synthetic actual price/return mismatch')
            baseline.append(expected_loss(row.origin_price, mean, c['sigma'], 0.))
            for label, pred in [('raw', row.raw_log_return), ('selected', row.selected_log_return), ('oracle', mean)]:
                losses[label].append(expected_loss(row.origin_price, mean, c['sigma'], pred))
            verified += 1
        naive = math.fsum(baseline) / len(baseline)
        for label, values in losses.items():
            measured = result['metrics'][name][label]
            loss = math.fsum(values) / len(values)
            gain = 100 * (1 - loss / naive)
            np.testing.assert_allclose([naive, loss, gain], [measured['naive'], measured['model'], measured['gain_pct']], rtol=0, atol=1e-8)
            if label == 'raw':
                expected_gains[name] = gain
    if (len(predictions) != 1476 or verified != 1452 or result['unknown_rows'] != 24
            or result['synthetic_fits'] != 132 or result['market_fits'] != 0 or result['native_replays'] != 132
            or result['source_id'] != ready['identity']['source_id'] or result['market_skill_claimed']):
        raise ValueError('Exact completed synthetic scope required')
    scalar_decision = pilot.classify({name: {'raw': {'gain_pct': gain}} for name, gain in expected_gains.items()})
    if result['decision'] != scalar_decision:
        raise ValueError('Frozen decision differs from independently verified expected gains')
    return {'status': 'verified', 'scalar_scored_rows': verified,
            'quadrature_expected_gains': expected_gains, 'new_fits': 0, 'market_skill_claimed': False}
