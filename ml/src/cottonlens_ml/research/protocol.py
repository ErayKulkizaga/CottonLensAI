"""Pure, testable temporal contracts. No training libraries are imported here."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import VERSION


def eligible(history, horizon=5):
    names = ['cotton_close', f'target_return_{horizon}']
    return history.loc[(history.cotton_close > 0) & np.isfinite(history[names]).all(axis=1)
                       & history[f'target_date_{horizon}'].notna()].copy()


def mature(history, cutoff, years=None):
    cutoff = pd.Timestamp(cutoff)
    rows = eligible(history)
    rows = rows.loc[(rows.date < cutoff) & (rows.target_date_5 < cutoff)]
    if years is not None:
        rows = rows.loc[rows.date >= cutoff - pd.DateOffset(years=years)]
    if len(rows) < 126:
        raise ValueError('Insufficient mature history')
    return rows.copy()


def split_manifest(history, *, earliest=None):
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique sorted Cotton observations required')
    rows = eligible(history)
    rows = rows.loc[(rows.date < '2024-01-01') & (rows.target_date_5 < '2024-01-01')]
    if earliest is not None:
        rows = rows.loc[rows.date >= pd.Timestamp(earliest)]
    folds = []
    for year in range(2016, 2024):
        outer = rows.loc[rows.date <= f'{year}-12-31'].tail(126)
        if len(outer) != 126 or outer.date.min().year != year:
            if earliest is not None:
                continue
            raise ValueError(f'Insufficient outer origins in {year}')
        if len(rows.loc[(rows.date < outer.date.min()) & (rows.target_date_5 < outer.date.min())]) < 500:
            if earliest is not None:
                continue
            raise ValueError('Insufficient nested history')
        preceding = mature(rows, outer.date.min()).tail(189)
        if len(preceding) != 189:
            raise ValueError('Three 63-origin validation blocks required')
        inner = []
        for i in range(3):
            validation = preceding.iloc[i * 63:(i + 1) * 63]
            train = mature(rows, validation.date.min())
            inner.append({'origins': validation.date.dt.strftime('%Y-%m-%d').tolist(),
                          'cutoff': validation.date.min().isoformat(), 'mature_train_rows': len(train)})
        folds.append({'fold': len(folds) + 1, 'year': year,
                      'origins': outer.date.dt.strftime('%Y-%m-%d').tolist(), 'inner': inner})
    if not folds:
        raise ValueError('No complete historical blocks for this information coverage')
    payload = {'version': VERSION, 'role': 'seen_historical_research', 'audit_used': False,
               'coverage_start': None if earliest is None else str(earliest),
               'folds': folds, 'label_identity': frame_identity(rows, ['date', 'cotton_close',
                   'target_return_1', 'target_return_5', 'target_date_1', 'target_date_5']),
               'decision_time': '00:00 UTC following the completed source bar date'}
    return {**payload, 'split_id': content_id(payload)}


def rows_at(history, dates):
    return history.set_index('date', drop=False).loc[pd.to_datetime(dates)].reset_index(drop=True)


def full_year_manifest(history):
    """New identity only: all eligible pre-2024 dates, never alter legacy splits."""
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique sorted Cotton observations required')
    rows = eligible(history)
    rows = rows.loc[(rows.date >= '2016-01-01') & (rows.date < '2024-01-01')
                    & (rows.target_date_5 < '2024-01-01')]
    folds = []
    for year in range(2016, 2024):
        outer = rows.loc[rows.date.dt.year == year]
        if outer.empty:
            raise ValueError(f'No complete-year cohort for {year}')
        preceding = mature(history, outer.date.min()).tail(189)
        if len(preceding) != 189:
            raise ValueError('Three past 63-origin inner blocks required')
        inner = []
        for i in range(3):
            validation = preceding.iloc[63 * i:63 * (i + 1)]
            train = mature(history, validation.date.min())
            if len(train) < 500:
                raise ValueError('At least 500 past mature training rows required')
            inner.append({'origins': validation.date.dt.strftime('%Y-%m-%d').tolist(),
                          'cutoff': validation.date.min().isoformat()})
        folds.append({'fold': len(folds) + 1, 'year': year,
                      'origins': outer.date.dt.strftime('%Y-%m-%d').tolist(), 'inner': inner})
    payload = {'version': 'full-year-v1', 'role': 'seen_historical_research', 'audit_used': False,
               'folds': folds, 'purge_observations': 5, 'refit_cadence': 21,
               'decision_time': '00:15 UTC following the completed source bar date',
               'label_identity': frame_identity(rows, ['date', 'cotton_close', 'target_return_1',
                                                     'target_return_5', 'target_date_1', 'target_date_5'])}
    return {**payload, 'split_id': content_id(payload)}


def feature_history(history, market=None):
    """Add only causal features; preserve rows and ordinal time, including missingness."""
    result = history.copy()
    price = result.cotton_close
    for lag in (2, 3, 4, 7, 10):
        result[f'return_lag_{lag}'] = result.cotton_ret_1.shift(lag)
    for length in (10, 40, 120):
        result[f'trend_{length}'] = price / price.rolling(length).mean() - 1
        result[f'volatility_{length}'] = result.cotton_ret_1.rolling(length).std()
    if market is not None:
        cotton = market.loc[market.series.eq('cotton')].set_index('date').reindex(result.date)
        volume = pd.Series(cotton.volume.to_numpy(), index=result.index)
        result['volume_zero'] = volume.eq(0).astype(float)
        result['volume_log1p'] = np.log1p(volume.where(volume >= 0))
        result['volume_log_change'] = result.volume_log1p.diff()
        result['close_open_return'] = np.log(price / cotton.open.to_numpy())
    result['days_since_previous_observation'] = result.date.diff().dt.days.astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


def feature_groups(history):
    extensions = [c for c in history if c.startswith(('return_lag_', 'trend_', 'volatility_',
                   'volume_', 'close_open_', 'days_since_'))]
    return {'base': list(FEATURE_NAMES), 'expanded': [*FEATURE_NAMES, *extensions]}


@dataclass
class Preprocessor:
    names: list
    median: list
    mean: list
    scale: list

    @classmethod
    def fit(cls, train, names):
        matrix = train[names].to_numpy(dtype=float)
        finite = np.isfinite(matrix)
        medians = [float(np.median(matrix[finite[:, i], i])) if finite[:, i].any() else 0.
                   for i in range(len(names))]
        filled = np.where(finite, matrix, medians)
        scale = filled.std(axis=0)
        return cls(list(names), medians, filled.mean(axis=0).tolist(), np.where(scale > 1e-12, scale, 1.).tolist())

    def transform(self, rows):
        matrix = rows[self.names].to_numpy(dtype=float)
        finite = np.isfinite(matrix)
        values = (np.where(finite, matrix, self.median) - self.mean) / self.scale
        return np.concatenate([values, (~finite).astype(float)], axis=1).astype(np.float32)

    def as_dict(self):
        return asdict(self)


@dataclass
class Target:
    kind: str
    mean: float = 0.
    scale: float = 1.

    @classmethod
    def fit(cls, train, horizon, kind):
        raw = train[f'target_return_{horizon}'].to_numpy(dtype=float)
        if kind == 'price_delta':
            raw = train.cotton_close.to_numpy() * np.expm1(raw)
        if kind == 'raw_log':
            return cls(kind)
        if kind not in ('scaled_log', 'price_delta'):
            raise ValueError('Unknown target transformation')
        return cls(kind, float(raw.mean()), max(float(raw.std()), 1e-8))

    def forward(self, rows, horizon):
        raw = rows[f'target_return_{horizon}'].to_numpy(dtype=float)
        if self.kind == 'price_delta':
            raw = rows.cotton_close.to_numpy() * np.expm1(raw)
        return ((raw - self.mean) / self.scale).astype(np.float32)

    def inverse(self, predicted, prices):
        raw = np.asarray(predicted).reshape(-1) * self.scale + self.mean
        if self.kind == 'price_delta':
            ratio = 1 + raw / np.asarray(prices)
            if (ratio <= 0).any():
                raise ValueError('Predicted price outside positive domain')
            raw = np.log(ratio)
        if not np.isfinite(raw).all():
            raise ValueError('Non-finite inverse target')
        return raw

    def as_dict(self):
        return asdict(self)


def inputs(history, origins, processor, window=1):
    indexes = history.set_index('date').index.get_indexer(pd.DatetimeIndex(origins.date))
    if (indexes < window - 1).any():
        raise ValueError('Insufficient recorded-observation sequence context')
    transformed = processor.transform(history)
    if window == 1:
        return transformed[indexes]
    return np.stack([transformed[i - window + 1:i + 1] for i in indexes])


DECISION_CLOCKS = {
    'legacy-midnight-v1': (pd.Timedelta(days=1), False),
    'cotton-next-day-0015-v1': (pd.Timedelta(days=1, minutes=15), True),
}


def attach_releases(history, records, features, *, decision_clock='legacy-midnight-v1'):
    """Point-in-time integration requires explicit publication and vintage evidence."""
    if decision_clock not in DECISION_CLOCKS:
        raise ValueError('Unknown decision clock; an explicit versioned policy is required')
    offset, exact_matches = DECISION_CLOCKS[decision_clock]
    required = {'published_at', 'vintage_id', 'source_url', 'source_sha256', 'timestamp_verified', *features}
    if not required.issubset(records.columns):
        raise ValueError('Actual verified publication/vintage evidence required')
    clock = 'published_at'
    if 'available_at' in records:
        availability = {'availability_verified', 'availability_basis'}
        if (not availability.issubset(records.columns)
                or not records.availability_verified.eq(True).all()
                or not records.availability_basis.isin(['exact_publication', 'verified_upper_bound']).all()):
            raise ValueError('Verified availability clock required')
        exact = records.availability_basis.eq('exact_publication')
        if (not records.loc[exact, 'timestamp_verified'].eq(True).all()
                or not records.loc[~exact, 'timestamp_verified'].eq(False).all()
                or not records.loc[~exact, 'published_at'].isna().all()):
            raise ValueError('Upper bounds must not claim exact publication')
        if not (pd.to_datetime(records.loc[exact, 'published_at'], utc=True)
                == pd.to_datetime(records.loc[exact, 'available_at'], utc=True)).all():
            raise ValueError('Exact publication and availability clocks disagree')
        clock = 'available_at'
    elif not records.timestamp_verified.eq(True).all():
        raise ValueError('Actual verified publication/vintage evidence required')
    if records[['vintage_id', 'source_url', 'source_sha256']].isna().any().any():
        raise ValueError('Missing source evidence')
    release = records.copy()
    release['published_at'] = pd.to_datetime(release.published_at, utc=True)
    release[clock] = pd.to_datetime(release[clock], utc=True)
    if release[clock].isna().any() or release[clock].duplicated().any():
        raise ValueError('Unique publication timestamps required')
    left = history.copy()
    left['decision_at'] = pd.to_datetime(left.date, utc=True) + offset
    return pd.merge_asof(left.sort_values('decision_at'), release.sort_values(clock),
                         left_on='decision_at', right_on=clock, direction='backward', allow_exact_matches=exact_matches)
