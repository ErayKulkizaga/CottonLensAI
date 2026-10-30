"""Independent block-sum covariance, Decimal paired losses and planning checks."""

import argparse
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

from research.verify_risk_weak_teacher import fingerprint, read, sha


def decimal_difference(actual, baseline, model):
    with localcontext() as context:
        context.prec = 40
        y, b, m = (Decimal(str(v)) for v in (actual, baseline, model))
        # Algebraic paired form preserves exactly zero for identical forecasts.
        return float((b/m).ln()+y*(1/b-1/m))


def verify(root):
    ready = read(root/'ready.json'); result = read(root/'result.json')
    manifest = ready['source']; source_body = {k: v for k, v in manifest.items() if k != 'source_id'}
    if fingerprint(source_body) != result['source_id'] or result['source_id'] != manifest['source_id']:
        raise ValueError('Source identity changed')
    for name, checksum in {**manifest['ml']['files'], **manifest['helpers']}.items():
        if sha(root/'source-snapshot'/name) != checksum:
            raise ValueError('Frozen source payload changed')
    for name, checksum in ready['inputs_sha256'].items():
        if sha(root/'inputs'/name) != checksum:
            raise ValueError('Immutable input changed')
    if any(result[n] for n in ('new_fits', 'new_data_downloads', 'new_forward_forecasts', 'forward_preregistration_locked', 'market_skill')):
        raise ValueError('This is conditional readiness, not forward skill or a new fit')
    proof = read(root/'inputs/parent-proof.json'); parent = read(root/'inputs/parent-result.json')
    if (sha(root/'inputs/paired-losses.csv') != proof['paired_losses_sha256']
            or sha(root/'inputs/parent-result.json') != proof['result_sha256']
            or proof['source_id'] != '5846f9cbb0f360731e84f251ed311f5f665510590594645204daa95c6438ab26'
            or sha(root/'inputs/market.parquet') != '23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c'
            or parent['primary_gate_passed']
            or not parent['comparisons']['own_level_selected_vs_EWMA']['fixed_risk_gate_passed']):
        raise ValueError('Wrong primary/secondary parent or source')
    if sha(root/'planning-losses.csv') != result['losses_sha256'] or sha(root/'seed-history.parquet') != result['seed_sha256']:
        raise ValueError('Derived evidence changed')
    f = pd.read_csv(root/'inputs/paired-losses.csv', float_precision='round_trip')
    losses = pd.read_csv(root/'planning-losses.csv', float_precision='round_trip')
    if len(f) != 2006 or f.horizon.ne(1).any():
        raise ValueError('Wrong cohort')
    pd.testing.assert_frame_equal(f[['date', 'target_date', 'year']], losses[['date', 'target_date', 'year']], check_exact=True)
    delta = np.array([decimal_difference(y, b, m) for y, b, m in zip(
        f.actual_variance, f.ewma_variance, f.own_level_selected, strict=True)])
    np.testing.assert_allclose(delta, losses.delta, rtol=1e-10, atol=1e-12)
    mean = float(np.mean(delta)); report = result['planning']
    if report['calibrated_power'] or report['joint_20_60_gate_power'] or report['selection_adjusted']:
        raise ValueError('Do not present an approximation as calibrated power')
    np.testing.assert_allclose(mean, report['mean_paired_gain'], rtol=0, atol=1e-12)
    lower = min(v[0] for v in parent['comparisons']['own_level_selected_vs_EWMA']['gain_ci'].values())
    checked_scenarios = 0; variances = {}
    for block in (20, 60):
        squared_sum = 0.
        for year in sorted(set(f.year)):
            x = delta[f.year.to_numpy() == year]-mean
            sums = np.convolve(x, np.ones(block), mode='full')
            squared_sum += float(sums@sums)
        variance = squared_sum/(len(delta)*block); variances[str(block)] = variance
        claimed = report['normal_hac_scenarios'][str(block)]
        np.testing.assert_allclose(variance, claimed['long_run_variance'], rtol=1e-10, atol=1e-12)
        for name, effect in (('historical_point', mean), ('historical_min_lci', lower)):
            for n in (126, 252, 504, 1008, 2006):
                z = effect*math.sqrt(n/variance)-1.959963984540054
                probability = .5*math.erfc(-z/math.sqrt(2))
                needed = math.ceil((1.959963984540054+.8416212335729143)**2*variance/effect**2)
                scenario = claimed['scenarios'][name][str(n)]
                np.testing.assert_allclose(probability, scenario['single_interval_positive_lci_probability'], rtol=0, atol=1e-11)
                if needed != scenario['origins_for_80pct_single_interval_probability']:
                    raise ValueError('Independent normal planning count differs')
                checked_scenarios += 1
    windows = 0
    for size, claim in report['descriptive_contiguous_windows'].items():
        n = int(size); means = np.array([math.fsum(delta[i:i+n])/n for i in range(len(delta)-n+1)])
        np.testing.assert_allclose(np.quantile(means, [0, .5, 1]), claim['mean_gain_min_median_max'], rtol=0, atol=1e-12)
        if claim['independent_replications'] or claim['power_estimate'] or len(means) != claim['overlapping_windows']:
            raise ValueError('Overlapping historical windows are not independent experiments')
        np.testing.assert_allclose(np.mean(means > 0), claim['positive_mean_fraction'], rtol=0, atol=1e-12)
        windows += len(means)
    seed = pd.read_parquet(root/'seed-history.parquet')
    raw = pd.read_parquet(root/'inputs/market.parquet')
    expected = raw.loc[raw.series.eq('cotton'), ['date', 'close']].rename(columns={'close': 'cotton_close'}).reset_index(drop=True)
    pd.testing.assert_frame_equal(seed, expected, check_exact=True)
    old = pd.read_parquet(root/'inputs/parent-history.parquet')
    pd.testing.assert_frame_equal(seed.iloc[:len(old)].reset_index(drop=True), old[['date', 'cotton_close']], check_exact=True)
    if len(seed) != result['seed_rows'] or result['original_warmup_date'] != str(old.date.iloc[119].date()):
        raise ValueError('Expanding history or original warmup changed')
    return {'status': 'independently_verified_conditional_readiness', 'new_fits': 0,
            'forward_skill': False, 'calibrated_power': False, 'source_id': result['source_id'],
            'decimal_loss_differences': len(delta), 'independent_covariance_method': 'sum of squared zero-padded block sums per year',
            'long_run_variances': variances, 'normal_scenarios_checked': checked_scenarios,
            'overlapping_descriptive_windows_checked': windows, 'seed_rows': len(seed),
            'historical_price_prefix_rows': len(old)}


if __name__ == '__main__':
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve independent verification receipt')
    checked = verify(args.root)
    args.output.write_text(json.dumps(checked, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(checked))
