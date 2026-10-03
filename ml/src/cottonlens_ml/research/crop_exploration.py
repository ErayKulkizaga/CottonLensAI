"""Bounded crop-price information control on the existing CPU ledger runner."""
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.sources.crop_prices import END, START, TICKERS

PROFILE = 'crop-exploration-v1'
LAGS = information.LAGS
FIELDS = ('return_1', 'return_5', 'return_21', 'volatility_21')


def validate_prices(frame):
    if (set(frame.series) != set(TICKERS) or frame.date.isna().any()
            or frame.duplicated(['date', 'series']).any()
            or frame.date.min() < pd.Timestamp(START) or frame.date.max() >= pd.Timestamp(END)):
        raise ValueError('Unique pre-audit corn/soybean prices required')
    for _, rows in frame.groupby('series'):
        if not rows.date.is_monotonic_increasing or np.isinf(rows.close.to_numpy(dtype=float)).any():
            raise ValueError('Chronological finite/missing source closes required')


def read_prices(table):
    table = Path(table)
    source = read_record(table.with_suffix('.manifest.json'))
    if (digest(table) != source['table_sha256'] or source['model_eligible'] or source['release_allowed']
            or source['publication_timestamp_verified'] or source['first_version_verified']):
        raise ValueError('Pinned Tier-A crop table required')
    frame = pd.read_csv(table, parse_dates=['date'])
    validate_prices(frame)
    if len(frame) != source['rows']:
        raise ValueError('Frozen crop coverage changed')
    return frame, source


def add_prices(history, prices):
    validate_prices(prices)
    if (history.date.isna().any() or history.date.duplicated().any() or not history.date.is_monotonic_increasing):
        raise ValueError('Unique sorted Cotton observations required')
    result = history.copy()
    calendar = history.date.to_numpy(dtype='datetime64[ns]')
    for series, rows in prices.groupby('series'):
        starts = np.searchsorted(calendar, rows.date.to_numpy(dtype='datetime64[ns]'), side='right')
        raw = rows.close.to_numpy(dtype=float)
        for lag in LAGS:
            level, age = np.full(len(history), np.nan), np.full(len(history), np.nan)
            available = starts + lag - 1
            cursor, selected = 0, None
            for index in range(len(history)):
                while cursor < len(available) and available[cursor] <= index:
                    # Empty/nonpositive source quotes cannot refresh the valid quote age.
                    if np.isfinite(raw[cursor]) and raw[cursor] > 0:
                        selected = cursor
                    cursor += 1
                if selected is not None:
                    age[index] = index - starts[selected]
                    if index - available[selected] <= 3:
                        level[index] = raw[selected]
            log = pd.Series(np.log(level), index=history.index)
            returns = log.diff()
            values = {'return_1': returns, 'return_5': log.diff(5), 'return_21': log.diff(21),
                'volatility_21': returns.rolling(21, min_periods=21).std(ddof=0)}
            for field, value in values.items():
                result[f'crop_{series}_{field}_L{lag}'] = value
            result[f'crop_{series}_age_L{lag}'] = age
            result[f'crop_{series}_missing_L{lag}'] = (~np.isfinite(np.column_stack(list(values.values())))).any(axis=1).astype(float)
    return result


def group_names():
    groups = {'base': list(FEATURE_NAMES)}
    for lag in LAGS:
        control = [f'crop_{s}_{f}_L{lag}' for s in TICKERS for f in ('age', 'missing')]
        groups[f'missing_L{lag}'] = [*FEATURE_NAMES, *control]
        groups[f'prices_L{lag}'] = [*FEATURE_NAMES, *control,
            *[f'crop_{s}_{f}_L{lag}' for s in TICKERS for f in FIELDS]]
    return groups


def prepare(repo, folder, reference, table):
    rows, source = read_prices(table)
    information.prepare_information(repo, folder, Path(reference), profile=PROFILE, source=source,
        add_features=lambda history: add_prices(history, rows), first_source_day=rows.date.min(),
        groups=group_names(), controls={f'prices_L{l}': f'missing_L{l}' for l in LAGS}, prefix='crop-prices',
        hypothesis='Corn/soybean returns and risk context add T+5 information beyond age/missingness',
        policy={'assumed_day': 'provider-declared local Yahoo bar day, not certified historical availability',
            'cotton_lags': list(LAGS), 'max_extra_age': 3,
            'windows': '1/5/21 retained Cotton observations; no dropped dates',
            'publication_clock_verified': False, 'first_version_verified': False, 'roll_verified': False})


def dispatch(args):
    def prepare_source(folder):
        if not all((args.reference_root, args.crop_table)):
            raise ValueError('Pinned parent and crop table required')
        prepare(args.repo, folder, args.reference_root, args.crop_table)
    information.dispatch_information(args, prepare_source)
