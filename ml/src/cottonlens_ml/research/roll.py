"""Flag data/roll risks without adjusting prices or filtering the primary cohort."""
import numpy as np
import pandas as pd


def audit(history, *, verified_roll_dates=()):
    rows = history.copy()
    price = rows.cotton_close.astype(float)
    returns = np.log(price / price.shift())
    # Earlier data only: not a full-sample robust threshold.
    center = returns.shift().rolling(126, min_periods=63).median()
    mad = returns.shift().rolling(126, min_periods=63).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    rows['return_jump_flag'] = (returns.abs() > .20) | ((returns - center).abs() > 8 * 1.4826 * mad.clip(lower=1e-4))
    rows['duplicate_date_flag'] = rows.date.duplicated(keep=False)
    rows['invalid_price_flag'] = ~np.isfinite(price) | (price <= 0)
    rows['zero_volume_flag'] = rows.get('cotton_volume', pd.Series(np.nan, index=rows.index)).eq(0)
    if 'cotton_volume' not in rows and 'volume_zero' in rows:
        rows['zero_volume_flag'] = rows.volume_zero.eq(1)
    rows['listed_contract_month'] = rows.date.dt.month.isin([3, 5, 7, 10, 12])
    names = ['cotton_open', 'cotton_high', 'cotton_low', 'cotton_close']
    rows['ohlc_unavailable_flag'] = not set(names).issubset(rows.columns)
    rows['invalid_ohlc_flag'] = False
    if set(names).issubset(rows.columns):
        o, hi, lo, c = (rows[n] for n in names)
        rows['ohlc_unavailable_flag'] = rows[names].isna().any(axis=1)
        rows['invalid_ohlc_flag'] = (hi < np.maximum(o, c)) | (lo > np.minimum(o, c)) | (hi < lo) | (lo <= 0)
    rows['verified_roll_flag'] = rows.date.isin(pd.to_datetime(list(verified_roll_dates)))
    rows['suspected_roll_flag'] = rows.return_jump_flag | rows.zero_volume_flag
    for h in (1, 5):
        # Outcome-window annotation is explicitly ex-post diagnostic, never a predictor.
        flags = rows.verified_roll_flag | rows.suspected_roll_flag
        rows[f'roll_window_{h}'] = pd.concat([flags.shift(-i) for i in range(1, h + 1)], axis=1).any(axis=1)
    return rows
