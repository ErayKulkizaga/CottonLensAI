"""Independent zero-fit replay of the declared synthetic inference experiment."""

import argparse
import math
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

import numpy as np
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record
from scipy.signal import lfilter


def independent_weights(sizes, block):
    """Interval endpoint increments, not the analysis's sampled-index bincounts."""
    rng = np.random.default_rng(42)
    result = np.zeros((10000, sum(sizes)))
    for offset in range(0, 10000, 128):
        count = min(128, 10000-offset)
        column = 0
        for n in sizes:
            width = min(n, block)
            starts = rng.integers(n-width+1, size=(count, math.ceil(n/width)))
            changes = np.zeros((count, n+1))
            rows = np.arange(count)
            for j in range(starts.shape[1]):
                take = min(width, n-j*width)
                np.add.at(changes, (rows, starts[:, j]), 1)
                np.add.at(changes, (rows, starts[:, j]+take), -1)
            result[offset:offset+count, column:column+n] = np.cumsum(changes, axis=1)[:, :n]/sum(sizes)
            column += n
    return result


def exact_iid_ratio(sizes, block):
    trace = Fraction()
    for n in sizes:
        width = min(n, block)
        starts = n-width+1
        for j in range(math.ceil(n/width)):
            take = min(width, n-j*width)
            for i in range(n):
                cover = max(0, min(i, starts-1)-max(0, i-take+1)+1)
                probability = Fraction(cover, starts)
                trace += probability*(1-probability)
    return float(trace/sum(sizes))


def binomial_interval(count, n):
    z = NormalDist().inv_cdf(.975)
    center = count/n+z*z/(2*n)
    radius = z*math.sqrt(count/n*(1-count/n)/n+z*z/(4*n*n))
    return np.array([center-radius, center+radius])/(1+z*z/n)


def verify(root, frozen_repo, output):
    if output.exists():
        raise ValueError('Preserve the previous verification receipt')
    result = read_record(root/'result.json')
    inputs = read_record(root/'inputs.json')
    for name, checksum in inputs['files'].items():
        if digest(frozen_repo/name) != checksum:
            raise ValueError('Frozen source/input mismatch')
    assert result['new_fits'] == 0 and not result['market_skill']
    assert not result['market_payloads_read'] and not result['forward_performance_scored']
    assert not result['active_protocol_changed'] and result['native_interval_checks'] == 24
    expected = {(partition, phi, effect) for partition in ('active_calendar', 'pooled_sensitivity')
                for phi in (0., .8) for effect in (0., 0.01770104344442714)}
    keys = {(r['partition'], r['phi'], r['true_mean_qlike_gain']) for r in result['cases']}
    assert keys == expected and len(result['cases']) == 8
    scalars = 0
    for label, sizes in [('active_calendar', (43, 209)), ('pooled_sensitivity', (252,))]:
        weights = {block: independent_weights(sizes, block) for block in (20, 60)}
        for block in (20, 60):
            exact = exact_iid_ratio(sizes, block)
            np.testing.assert_allclose(exact, result['analytic_iid_variance_ratios'][label][str(block)]['ratio'],
                                       rtol=0., atol=1e-14)
        for row in [r for r in result['cases'] if r['partition'] == label]:
            phi, effect = row['phi'], row['true_mean_qlike_gain']
            innovations = np.random.default_rng(20261031).normal(size=(1024, 252))
            innovations[:, 1:] *= math.sqrt(1-phi*phi)
            z = lfilter([1.], [1., -phi], innovations, axis=1)
            realized = (.25+effect)/(1-math.exp(-.25))*z*z
            delta = math.log(1/math.exp(.25))+realized*(1-1/math.exp(.25))
            path = root/f'{label}-phi{phi}-effect{effect}.npz'
            with np.load(path, allow_pickle=False) as saved:
                assert set(saved.files) == {'differences', 'ci20', 'ci60', 'joint_passed'}
                np.testing.assert_allclose(saved['differences'], delta, rtol=0., atol=2e-14)
                passes = delta.mean(axis=1) > 0
                for block in (20, 60):
                    all_bounds = []
                    for start in range(0, len(delta), 128):
                        batch = delta[start:start+128]
                        draws = weights[block]@batch.T
                        noise = draws-draws.mean(axis=0)
                        all_bounds.append((batch.mean(axis=1)-np.quantile(noise, [.975, .025], axis=0)).T)
                    bounds = np.concatenate(all_bounds)
                    np.testing.assert_allclose(bounds, saved['ci'+str(block)], rtol=0., atol=3e-14)
                    assert int(np.sum(bounds[:, 0] > 0)) == row['single_interval_positive_counts'][str(block)]
                    passes &= bounds[:, 0] > 0
                    scalars += bounds.size
                np.testing.assert_array_equal(passes, saved['joint_passed'])
                scalars += delta.size
            count = int(passes.sum())
            assert count == row['joint_positive_count'] and count/1024 == row['joint_positive_rate']
            ci = binomial_interval(count, 1024)
            np.testing.assert_allclose(ci, row['monte_carlo_wilson_95'], rtol=0., atol=1e-14)
            assert row['null_oversize_demonstrated'] == bool(effect == 0. and ci[0] > .025)
    receipt = {'status': 'independent_synthetic_inference_verification_passed', 'cases': 8,
               'synthetic_scalars_checked': scalars, 'analytic_variance_ratios_checked': 4,
               'independent_generator': 'scipy.signal.lfilter stationary Gaussian square',
               'independent_resampler': 'interval-endpoint increments and prefix sums',
               'new_fits': 0, 'market_skill': False, 'forward_performance_scored': False,
               'actual_market_power_estimated': False, 'result_sha256': digest(root/'result.json')}
    freeze_record(output, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'frozen-repo', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.root, args.frozen_repo, args.output))
