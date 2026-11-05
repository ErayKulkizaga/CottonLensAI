"""Reconstruct held DXY inputs from raw quotes; no target fit or effect screen."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.verify_risk_weak_teacher import read, sha


def reconstruct(market, dates):
    dates = pd.DatetimeIndex(dates)
    if not dates.is_unique or not dates.is_monotonic_increasing or dates.hasnans:
        raise ValueError('Unique chronological Cotton calendar required')
    quotes = market.loc[market.series.eq('dxy'), ['date', 'close']].copy().sort_values('date')
    if quotes.date.duplicated().any() or not len(quotes):
        raise ValueError('Unique raw DXY source dates required')
    prices = quotes.set_index('date').close
    prices = prices.where(prices > 0).dropna()
    if not len(prices) or not np.isfinite(prices).all():
        raise ValueError('Finite positive source price history required')
    previous = pd.Series(dates).shift()
    position = prices.index.searchsorted(pd.DatetimeIndex(previous), side='right')-1
    usable = previous.notna().to_numpy() & (position >= 0)
    chosen = np.clip(position, 0, len(prices)-1)
    source = pd.Series(prices.index.take(chosen)).where(usable)
    age = pd.Series(np.arange(len(dates))-(dates.searchsorted(pd.DatetimeIndex(source), side='right')-1)).where(usable)
    aligned = pd.Series(np.where(usable, prices.to_numpy()[chosen], np.nan)).where(age <= 3)
    result = pd.DataFrame({'date': dates, 'source_date': source, 'age': age,
                           'available_at': pd.to_datetime(source, utc=True)+pd.Timedelta(days=1)})
    for k in (1, 5, 20):
        result[f'ret_{k}'] = np.log(aligned/aligned.shift(k))
    result['variance_5'] = result.ret_1.pow(2).rolling(5, min_periods=5).mean()
    return result


def audit(market_path, history_path, cohort_path, parent_ready_path):
    market = pd.read_parquet(market_path); h = pd.read_parquet(history_path)
    cohort = pd.read_csv(cohort_path, keep_default_na=False, na_values=[''], float_precision='round_trip')
    ready = read(parent_ready_path)
    if sha(history_path) != ready['identity']['inputs_sha256']['history.parquet']:
        raise ValueError('History differs from registered parent')
    if (len(cohort) != 2006 or cohort.date.duplicated().any()
            or sorted(cohort.year.unique()) != list(range(2016, 2024))
            or not np.array_equal(h.target_date_1.iloc[:-1], h.date.iloc[1:])):
        raise ValueError('Original calendar/cohort required')
    result = reconstruct(market, h.date)
    pd.testing.assert_series_equal(result.source_date, pd.to_datetime(h.dxy_source_date).reset_index(drop=True), check_names=False)
    pd.testing.assert_series_equal(result.available_at, pd.to_datetime(h.dxy_available_at, utc=True).reset_index(drop=True), check_names=False)
    np.testing.assert_allclose(result.age, h.dxy_age_sessions, rtol=0, atol=0, equal_nan=True)
    finite_slots, max_error = 0, 0.
    for k in (1, 5, 20):
        a, b = result[f'ret_{k}'].to_numpy(), h[f'dxy_ret_{k}'].to_numpy()
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13, equal_nan=True)
        finite_slots += int(np.isfinite(a).sum())
        max_error = max(max_error, float(np.nanmax(np.abs(a-b))))
    cotton = market.loc[market.series.eq('cotton')].set_index('date').close
    if not cotton.index.is_unique:
        raise ValueError('Duplicate raw Cotton date')
    np.testing.assert_allclose(cotton.reindex(h.date), h.cotton_close, rtol=0, atol=0)
    days = pd.to_datetime(cohort.date)
    indexed = result.set_index('date')
    evaluation = indexed.loc[days]
    train_days = sorted({d for f in ready['identity']['design']['folds'] for j in f['jobs'] for d in j['train_dates']})
    training = indexed.loc[pd.to_datetime(train_days)]
    decision = pd.to_datetime(days, utc=True)+pd.Timedelta(days=1, minutes=15)
    if (evaluation.variance_5.isna().any() or training.variance_5.isna().any()
            or (evaluation.variance_5 <= 0).any() or (training.variance_5 <= 0).any()
            or evaluation.available_at.isna().any()
            or (evaluation.available_at.to_numpy() > decision.to_numpy()).any()):
        raise ValueError('Candidate source not available on unchanged origin/train cohort')
    return result, {'schema': 1, 'status': 'verified_conditional_source_readiness',
                    'new_forecast_model_fits': 0, 'effect_screen_performed': False,
                    'cotton_price_checks': len(h), 'dxy_finite_return_scalar_checks': finite_slots,
                    'dxy_metadata_rows_checked': len(h), 'max_return_difference': max_error,
                    'eval_origins': len(evaluation), 'unique_training_origins': len(training),
                    'maximum_source_age_eval': float(evaluation.age.max()),
                    'feature': 'mean last5 Cotton-observation mapped DXY squared log returns',
                    'alignment': 'positive source no later than preceding Cotton observation; maxage3',
                    'clock': 'source date+1day00UTC assumed, decision origin+1day00:15UTC',
                    'historical_availability_verified': False, 'first_vintage_verified': False,
                    'market_skill': False, 'model_admission_allowed': False,
                    'raw_market_sha256': sha(market_path), 'history_sha256': sha(history_path),
                    'cohort_sha256': sha(cohort_path), 'parent_ready_sha256': sha(parent_ready_path),
                    'audit_source_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('market', 'history', 'cohort', 'parent-ready', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing source audit')
    table, receipt = audit(args.market, args.history, args.cohort, args.parent_ready)
    args.output.mkdir(parents=True)
    table.to_csv(args.output/'reconstructed.csv', index=False, lineterminator='\n')
    receipt['reconstructed_sha256'] = sha(args.output/'reconstructed.csv')
    (args.output/'result.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))
