"""Zero-fit synthetic calibration of the unchanged end-report decision rule.

No market payload, forward outcome, optimizer, collector or scheduler is used.
The fast path is a linear representation of the existing frozen resampler,
not a replacement inference method for the active study.
"""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, writer

from research.verify_risk_weak_teacher import difference_interval

PROFILE = 'ct-forward-inference-calibration-v1'
SIZES = (43, 209)
BLOCKS = (20, 60)
REPLICATES = 10000
PATHS = 1024
SEED = 20261031
EFFECT = 0.01770104344442714
LOG_RATIO = .25
FILES = ('research/ct_inference_calibration.py',
         'research/tests/test_ct_inference_calibration.py',
         'research/verify_risk_weak_teacher.py',
         'research/ct_forward_report.py',
         'research/evidence/ct-calendar-2027-extension-20261031.json',
         'research/evidence/ct-forward-inference-calibration-prereg-20261031.json')


def bootstrap_weights(sizes, block, repetitions=REPLICATES):
    """Same RNG order, within-year starts, truncation and seed as the verifier."""
    if (not sizes or any(n < 1 or int(n) != n for n in sizes)
            or block < 1 or int(block) != block or repetitions < 20):
        raise ValueError('Positive integer strata/block and at least20 replicates required')
    n = sum(sizes)
    weights = np.zeros((repetitions, n))
    rng = np.random.default_rng(42)
    for offset in range(0, repetitions, 128):
        count = min(128, repetitions-offset)
        column = 0
        for length in sizes:
            width = min(block, length)
            starts = rng.integers(0, length-width+1, (count, math.ceil(length/width)))
            positions = (starts[..., None]+np.arange(width)).reshape(count, -1)[:, :length]
            for row, indexes in enumerate(positions, start=offset):
                weights[row, column:column+length] = np.bincount(indexes, minlength=length)/n
            column += length
    return weights


def iid_variance_ratio(sizes, block):
    """Exact E[conditional bootstrap variance] / Var(sample mean), iid unit noise.

    Counts from independent block starts add their covariance traces. A final
    truncated block retains the original full-width allowable start positions.
    This is the infinite-bootstrap ratio, not a simulated coverage estimate.
    """
    if not sizes or any(n < 1 for n in sizes) or block < 1:
        raise ValueError('Positive strata and block required')
    traces = []
    for length in sizes:
        width = min(length, block)
        starts = length-width+1
        trace = 0.
        remaining = length
        while remaining:
            take = min(width, remaining)
            incidence = np.zeros((starts, length))
            for start in range(starts):
                incidence[start, start:start+take] = 1.
            probabilities = incidence.mean(axis=0)
            trace += take-float(probabilities@probabilities)
            remaining -= take
        traces.append({'stratum_size': length, 'expected_count_covariance_trace': trace,
                       'stratum_variance_ratio': trace/length})
    return {'ratio': sum(r['expected_count_covariance_trace'] for r in traces)/sum(sizes),
            'strata': traces}


def intervals(weights, differences):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or weights.shape[1] != values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Finite aligned synthetic differences required')
    means = values.mean(axis=1)
    bounds = []
    for offset in range(0, len(values), 128):
        batch = values[offset:offset+128]
        draws = weights@batch.T
        noise = draws-draws.mean(axis=0)
        bounds.append((batch.mean(axis=1)-np.quantile(noise, [.975, .025], axis=0)).T)
    return means, np.concatenate(bounds)


def synthetic_differences(phi, effect):
    """Valid QLIKE pairs with known unconditional mean, no fitted predictor.

    b=1, m=exp(a), y=(a+effect)/(1-exp(-a))*Z^2, Z stationary N(0,1).
    QLIKE(b)-QLIKE(m)=(a+effect)*Z^2-a, expectation=effect.
    AR dependence is continuous across the calendar boundary, with an exactly
    stationary initial draw. These are declared DGPs, not fitted market noise.
    """
    if phi not in (0., .8) or effect not in (0., EFFECT):
        raise ValueError('Only the two preregistered dependence/effect cases allowed')
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(PATHS, sum(SIZES)))
    for t in range(1, z.shape[1]):
        z[:, t] = phi*z[:, t-1]+math.sqrt(1-phi*phi)*z[:, t]
    y = (LOG_RATIO+effect)/(-math.expm1(-LOG_RATIO))*z*z
    differences = (LOG_RATIO+effect)*z*z-LOG_RATIO
    np.testing.assert_allclose(differences, -LOG_RATIO+y*(-math.expm1(-LOG_RATIO)),
                               rtol=1e-12, atol=1e-14)
    return differences


def wilson(successes, total):
    z = 1.959963984540054
    p = successes/total
    denominator = 1+z*z/total
    midpoint = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [midpoint-radius, midpoint+radius]


def check_native(weights, values, sizes, block):
    """Verify the acceleration against the actual existing helper, not itself."""
    _, fast = intervals(weights, values)
    boundaries = np.cumsum((0, *sizes))
    for i, row in enumerate(values):
        parts = [np.column_stack([row[a:b], np.ones(b-a)])
                 for a, b in pairwise(boundaries)]
        np.testing.assert_allclose(fast[i], difference_interval(parts, block), rtol=0., atol=1e-12)
    return len(values)


def audit(repo, output):
    if output.exists():
        raise ValueError('Preserve completed or partial analysis; use a new namespace')
    prereg = repo/FILES[-1]
    contract = json.loads(prereg.read_bytes())
    expected = {'profile': PROFILE, 'paths': PATHS, 'bootstrap_repetitions': REPLICATES,
                'stratum_sizes': list(SIZES), 'pooled_sensitivity_sizes': [sum(SIZES)],
                'seed': SEED, 'effect': EFFECT, 'log_forecast_ratio': LOG_RATIO,
                'blocks': list(BLOCKS), 'bootstrap_seed': 42, 'phi_cases': [0., .8],
                'effect_cases': [0., EFFECT], 'new_fits': 0, 'market_payloads_read': False,
                'forward_performance_scored': False, 'active_protocol_changed': False}
    if any(contract.get(k) != value for k, value in expected.items()):
        raise ValueError('Frozen calibration contract differs')
    files = {name: digest(repo/name) for name in FILES}
    expected_helper = contract['frozen_interval_helper_sha256']
    if files[FILES[2]] != expected_helper:
        raise ValueError('Existing interval helper differs from the frozen active source')
    if files[FILES[-2]] != contract['calendar_evidence_sha256']:
        raise ValueError('Calendar source evidence differs')
    calendar = json.loads((repo/FILES[-2]).read_bytes())
    if tuple(calendar['year_slots'][y] for y in sorted(calendar['year_slots'])) != SIZES:
        raise ValueError('Actual calendar strata differ from the fixed control')
    cases = []
    native_checks = 0
    analytic = {}
    with writer(output):
        freeze_record(output/'inputs.json', {'profile': PROFILE, 'files': files,
                      'numpy_version': np.__version__, 'market_payloads_read': False})
        for name in FILES:
            dest = output/'source-snapshot'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((repo/name).read_bytes())
        for label, sizes in [('active_calendar', SIZES), ('pooled_sensitivity', (sum(SIZES),))]:
            weights = {b: bootstrap_weights(sizes, b) for b in BLOCKS}
            analytic[label] = {str(b): iid_variance_ratio(sizes, b) for b in BLOCKS}
            for phi in (0., .8):
                for effect in (0., EFFECT):
                    values = synthetic_differences(phi, effect)
                    estimates = {b: intervals(weights[b], values) for b in BLOCKS}
                    if effect == 0.:
                        for b in BLOCKS:
                            native_checks += check_native(weights[b], values[:3], sizes, b)
                    mean = estimates[20][0]
                    passed = (mean > 0) & (estimates[20][1][:, 0] > 0) & (estimates[60][1][:, 0] > 0)
                    count = int(passed.sum())
                    interval = wilson(count, len(passed))
                    row = {'partition': label, 'phi': phi, 'true_mean_qlike_gain': effect,
                           'paths': len(passed), 'joint_positive_count': count,
                           'joint_positive_rate': count/len(passed), 'monte_carlo_wilson_95': interval,
                           'single_interval_positive_counts': {str(b): int(np.sum(estimates[b][1][:, 0] > 0)) for b in BLOCKS},
                           'null_oversize_demonstrated': bool(effect == 0. and interval[0] > .025)}
                    cases.append(row)
                    prefix = f'{label}-phi{phi}-effect{effect}'
                    np.savez(output/(prefix+'.npz'), differences=values, ci20=estimates[20][1],
                             ci60=estimates[60][1], joint_passed=passed)
        if files != {name: digest(repo/name) for name in FILES}:
            raise ValueError('Analysis source changed during execution')
        result = {'profile': PROFILE, 'new_fits': 0, 'market_payloads_read': False,
                  'new_data_downloads': 0, 'forward_performance_scored': False,
                  'active_protocol_changed': False, 'market_skill': False,
                  'native_interval_checks': native_checks, 'analytic_iid_variance_ratios': analytic,
                  'cases': cases, 'limits': 'Calibration only under two declared stationary DGPs; '
                  'not actual market power, source correctness, guaranteed sample size, or a new success gate.'}
        freeze_record(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.repo, args.output), indent=2))
