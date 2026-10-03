"""Causal inputs for one Cotton/corn/soybean transfer-learning feasibility check.

Historical crop publication and roll metadata are unverified. These inputs are
Tier-A exploration only; preparing them never authorizes training or release.
"""
import numpy as np
import pandas as pd

SERIES = ('cotton', 'corn', 'soybean')
FEATURES = ['own_return_1', 'own_return_5', 'own_return_20',
    'own_volatility_5', 'own_volatility_20', 'own_sma_ratio_5_20']


def asset_rows(bars, asset):
    """Keep recorded rows, lag features one own observation, never fill labels."""
    if asset not in SERIES or bars.empty:
        raise ValueError('Only nonempty pinned Cotton/corn/soybean bars are allowed')
    frame = bars[['date', 'close']].copy()
    frame['date'] = pd.to_datetime(frame.date)
    if (frame.date.isna().any() or frame.date.duplicated().any()
            or not frame.date.is_monotonic_increasing or frame.date.dt.tz is not None
            or frame.date.ne(frame.date.dt.normalize()).any()
            or frame.date.ge(pd.Timestamp('2024-01-01')).any()):
        raise ValueError('Unique chronological pre-audit provider dates required')
    values = frame.close.to_numpy(dtype=float)
    if np.isinf(values).any() or frame.close.dropna().le(0).any():
        raise ValueError('Missing closes may remain missing; observed closes must be positive and finite')
    log = np.log(frame.close)
    ret = log.diff()
    for lag in (1, 5, 20):
        frame[f'own_return_{lag}'] = log.diff(lag).shift(1)
    for window in (5, 20):
        frame[f'own_volatility_{window}'] = ret.rolling(window, min_periods=window).std(ddof=0).shift(1)
    frame['own_sma_ratio_5_20'] = (frame.close.rolling(5).mean()/frame.close.rolling(20).mean()-1).shift(1)
    frame['feature_asof'] = frame.date.shift(1)
    frame['asset'], frame['source_observation_index'] = asset, np.arange(len(frame))
    for horizon in (1, 5):
        frame[f'target_return_{horizon}'] = log.shift(-horizon)-log
        frame[f'target_date_{horizon}'] = frame.date.shift(-horizon)
    return frame


def mature_auxiliary(rows, cutoff, purge_cutoff):
    """Origins precede the purge boundary; both labels must be available.

The next UTC midnight plus 15 minutes is a declared Tier-A target-bar access
assumption, not verified publication evidence. A caller must derive purge_cutoff
from the frozen Cotton observation calendar (five observations), never weekdays.
"""
    cutoff, purge_cutoff = pd.Timestamp(cutoff), pd.Timestamp(purge_cutoff)
    if cutoff.tzinfo is None or str(cutoff.tzinfo) != 'UTC' or purge_cutoff.tzinfo is not None:
        raise ValueError('UTC decision cutoff and naive provider-date purge boundary required')
    if rows.date.duplicated().any() or rows.asset.nunique()!=1:
        raise ValueError('Maturity must be applied to one asset at a time')
    available = pd.to_datetime(rows.target_date_5, utc=True)+pd.Timedelta(days=1, minutes=15)
    finite = np.isfinite(rows[['target_return_1','target_return_5']].to_numpy(dtype=float)).all(axis=1)
    mask = rows.date.lt(purge_cutoff) & available.lt(cutoff) & finite
    return rows.loc[mask].copy()
