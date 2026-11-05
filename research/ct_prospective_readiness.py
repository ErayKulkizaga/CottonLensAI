"""Zero-fit sample-size and immutable-history readiness for the fixed CT candidate."""

import argparse
import importlib.metadata
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, read_record, writer
from cottonlens_ml.research.volatility import FLOOR, features, qlike

PROFILE = 'ct-own-level-prospective-readiness-v1'
PARENT_PROFILE = 'ct-risk-level-dxy-control-t1-v1'
PARENT_SOURCE = '5846f9cbb0f360731e84f251ed311f5f665510590594645204daa95c6438ab26'
RAW_SHA = '23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c'
BLOCKS = (20, 60)
SIZES = (126, 252, 504, 1008, 2006)
HELPERS = ('research/ct_prospective_readiness.py', 'research/verify_ct_prospective_readiness.py',
           'research/tests/test_ct_prospective_readiness.py', 'research/verify_risk_weak_teacher.py')
VERSIONS = ('numpy', 'pandas', 'pyarrow')


def source(repo):
    body = {'ml': research_source_identity(repo), 'helpers': {n: digest(repo/n) for n in HELPERS}}
    return {**body, 'source_id': content_id(body)}


def prices(frame):
    rows = frame[['date', 'cotton_close']].copy().reset_index(drop=True)
    dates = pd.DatetimeIndex(rows.date)
    if (rows.empty or dates.tz is not None or dates.hasnans or not dates.is_unique
            or not dates.is_monotonic_increasing or not dates.equals(dates.normalize())
            or not np.isfinite(rows.cotton_close).all() or not rows.cotton_close.gt(0).all()):
        raise ValueError('Positive finite prices and unique chronological date-only sessions required')
    return rows


def append_snapshot(seed, snapshot, *, retain_first_observed=False):
    """No guessed sessions, overlapping revisions, or hidden history truncation."""
    old, new = prices(seed), prices(snapshot)
    last = old.date.iloc[-1]
    if last not in set(new.date):
        raise ValueError('Snapshot must overlap the final seed session; no unverified gap bridge')
    a = old.loc[old.date.ge(new.date.iloc[0])].reset_index(drop=True)
    b = new.loc[new.date.le(last)].reset_index(drop=True)
    if not a.date.equals(b.date):
        raise ValueError('Overlapping source calendar changed; do not compress or insert past sessions')
    if not retain_first_observed and not np.array_equal(a.cotton_close, b.cotton_close):
        raise ValueError('Overlapping price revision; preserve both sources and review, never overwrite seed')
    return pd.concat([old, new.loc[new.date.gt(last)]], ignore_index=True)


def overlap_revisions(history, snapshot):
    matched = history.merge(snapshot, on='date', suffixes=('_first', '_later'), validate='one_to_one')
    changed = matched[matched.cotton_close_first.ne(matched.cotton_close_later)]
    return [{'date': str(row.date.date()), 'first_observed': row.cotton_close_first,
             'later_quote': row.cotton_close_later} for row in changed.itertuples(index=False)]


def candidate_history(frame):
    # Reuse the actual variance feature builder; no second feature/training engine.
    h = features(prices(frame))
    h['ct_ewma'] = h.daily_variance_proxy.ewm(alpha=.06, adjust=False).mean().clip(lower=FLOOR)
    h['origin_state'] = np.log(h.ct_ewma)
    return h


def long_run_variance(frame, block):
    """Bartlett HAC on pooled residuals, with products confined to each year."""
    if block not in BLOCKS or len(frame) < block or not np.isfinite(frame.delta).all():
        raise ValueError('Finite paired differences and declared planning block required')
    centered = frame.assign(residual=frame.delta-float(frame.delta.mean()))
    parts = [g.residual.to_numpy() for _, g in centered.groupby('year', sort=True)]
    variance = sum(float(x@x) for x in parts)/len(frame)
    for lag in range(1, block):
        covariance = sum(float(x[lag:]@x[:-lag]) for x in parts if len(x) > lag)/len(frame)
        variance += 2*(1-lag/block)*covariance
    if not np.isfinite(variance) or variance < 0:
        raise ValueError('Invalid planning covariance; no silent clipping')
    return variance


def normal_scenario(effect, variance, size):
    if not effect > 0 or not variance > 0 or size < 1:
        raise ValueError('Strictly positive scenario effect/covariance and sample size required')
    normal = NormalDist(); critical = normal.inv_cdf(.975)
    return {'effect': effect, 'origins': size,
            'single_interval_positive_lci_probability': normal.cdf(effect*math.sqrt(size/variance)-critical),
            'origins_for_80pct_single_interval_probability': math.ceil((critical+normal.inv_cdf(.8))**2*variance/effect**2)}


def planning(frame, lower_effect):
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing or not lower_effect > 0:
        raise ValueError('Original paired ordered cohort and positive conditional scenario required')
    delta = frame.delta.to_numpy(); mean = float(delta.mean())
    models = {}
    for block in BLOCKS:
        variance = long_run_variance(frame, block)
        models[str(block)] = {'long_run_variance': variance, 'scenarios': {
            name: {str(n): normal_scenario(effect, variance, n) for n in SIZES}
            for name, effect in (('historical_point', mean), ('historical_min_lci', lower_effect))}}
    prefix = np.r_[0., np.cumsum(delta)]; windows = {}
    for n in SIZES:
        if len(frame) < n:
            continue
        means = (prefix[n:]-prefix[:-n])/n
        windows[str(n)] = {'overlapping_windows': len(means),
            'positive_mean_fraction': float(np.mean(means > 0)),
            'mean_gain_min_median_max': np.quantile(means, [0, .5, 1]).tolist(),
            'independent_replications': False, 'power_estimate': False}
    return {'mean_paired_gain': mean, 'normal_hac_scenarios': models, 'descriptive_contiguous_windows': windows,
            'calibrated_power': False, 'joint_20_60_gate_power': False, 'selection_adjusted': False,
            'zero_true_effect_has_no_finite_confirmatory_sample_guarantee': True,
            'assumptions': 'Historical covariance/effect stationarity and a normal mean approximation; conditional planning only. Neither tail/regime change nor research selection is corrected. No promised duration, joint-gate power, or future skill.'}


def audit(repo, reference, market, proof_path, output):
    if output.exists():
        raise ValueError('Preserve completed or partial readiness output')
    registry = json.loads((repo/'research/registry.json').read_bytes())
    registered = {r['path']: r['sha256'] for r in registry['local_evidence']}
    proof_name = proof_path.resolve().relative_to((repo/'research').resolve()).as_posix()
    if registered.get(proof_name) != digest(proof_path):
        raise ValueError('Parent proof is not the registered immutable evidence')
    proof = read_record(proof_path)
    paths = {'paired-losses.csv': reference/'reports/paired-losses.csv',
             'parent-result.json': reference/'reports/result.json',
             'parent-history.parquet': reference/'reference/history.parquet', 'market.parquet': market,
             'parent-proof.json': proof_path}
    hashes = {n: digest(p) for n, p in paths.items()}
    if (proof['profile'] != PARENT_PROFILE or proof['source_id'] != PARENT_SOURCE
            or hashes['paired-losses.csv'] != proof['paired_losses_sha256']
            or hashes['parent-result.json'] != proof['result_sha256'] or hashes['market.parquet'] != RAW_SHA):
        raise ValueError('Parent forecast or held raw source identity changed')
    parent = read_record(paths['parent-result.json'])
    if parent['primary_gate_passed'] or not parent['comparisons']['own_level_selected_vs_EWMA']['fixed_risk_gate_passed']:
        raise ValueError('Do not silently reinterpret a different primary/secondary result')
    old = pd.read_parquet(paths['parent-history.parquet'])
    raw = pd.read_parquet(market)
    seed = prices(raw.loc[raw.series.eq('cotton'), ['date', 'close']].rename(columns={'close': 'cotton_close'}))
    common = seed.set_index('date').loc[old.date]
    np.testing.assert_array_equal(common.cotton_close.to_numpy(), old.cotton_close.to_numpy())
    rebuilt = candidate_history(seed.iloc[:len(old)])
    np.testing.assert_array_equal(rebuilt.date, old.date)
    for name in ('daily_variance_proxy', 'har_1', 'har_5', 'har_22', 'ct_ewma'):
        np.testing.assert_allclose(rebuilt[name], old[name], rtol=1e-12, atol=1e-14, equal_nan=True)
    frame = pd.read_csv(paths['paired-losses.csv'], float_precision='round_trip')
    if len(frame) != 2006 or frame.horizon.ne(1).any():
        raise ValueError('Exact original T1 cohort required')
    origins = rebuilt.set_index('date').loc[pd.to_datetime(frame.date)]
    target_dates = old.set_index('date').loc[pd.to_datetime(frame.date), 'target_date_1'].dt.strftime('%Y-%m-%d')
    if frame.target_date.tolist() != target_dates.tolist():
        raise ValueError('Target date mismatch; no cohort intersection')
    np.testing.assert_allclose(frame.actual_variance, origins.variance_target_1, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(frame.ewma_variance, origins.ct_ewma, rtol=1e-10, atol=1e-14)
    for name, predictions in (('baseline_loss', 'ewma_variance'), ('own_level_selected_loss', 'own_level_selected')):
        np.testing.assert_allclose(frame[name], qlike(frame.actual_variance, frame[predictions]), rtol=1e-10, atol=1e-12)
    losses = frame[['date', 'target_date', 'year']].copy()
    losses['delta'] = frame.baseline_loss-frame.own_level_selected_loss
    lower = min(v[0] for v in parent['comparisons']['own_level_selected_vs_EWMA']['gain_ci'].values())
    summary = planning(losses, lower)
    ready = {'profile': PROFILE, 'source': source(repo), 'inputs_sha256': hashes,
             'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in VERSIONS}}
    with writer(output):
        for name, path in paths.items():
            (output/'inputs').mkdir(exist_ok=True)
            shutil.copyfile(path, output/'inputs'/name)
        for name in {**ready['source']['ml']['files'], **ready['source']['helpers']}:
            dest = output/'source-snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/name, dest)
        freeze_record(output/'ready.json', ready)
        seed.to_parquet(output/'seed-history.parquet', index=False)
        losses.to_csv(output/'planning-losses.csv', index=False, lineterminator='\n')
        result = {'profile': PROFILE, 'source_id': ready['source']['source_id'], 'parent_fitted_source_id': PARENT_SOURCE,
                  'new_fits': 0, 'new_data_downloads': 0, 'new_forward_forecasts': 0,
                  'forward_preregistration_locked': False, 'market_skill': False,
                  'parent_primary_dxy_inconclusive': True, 'parent_secondary_own_level_historical_candidate': True,
                  'planning': summary, 'verified_price_rows': len(old), 'seed_rows': len(seed),
                  'seed_first_date': str(seed.date.iloc[0].date()), 'seed_last_date': str(seed.date.iloc[-1].date()),
                  'original_warmup_origin_index': 119, 'original_warmup_date': str(old.date.iloc[119].date()),
                  'history_policy': 'Use full2010 seed with exact overlapping prices/calendar; no rolling2y replacement, inferred gap bridge or revision overwrite. Existing capture alone does not preserve expanding history.',
                  'seed_sha256': digest(output/'seed-history.parquet'), 'losses_sha256': digest(output/'planning-losses.csv')}
        if {n: digest(p) for n, p in paths.items()} != hashes:
            raise ValueError('Original evidence changed during readiness audit')
        freeze_record(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'reference', 'market', 'proof', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repo, args.reference, args.market, args.proof, args.output)
    print(json.dumps({'status': 'zero_fit_readiness_completed', 'source_id': report['source_id'],
                      'new_fits': 0, 'seed_rows': report['seed_rows'], 'forward_preregistration_locked': False}))
