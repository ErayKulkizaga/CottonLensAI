"""Causal dense return path for one bounded, matched Ridge comparison."""
import numpy as np

from cottonlens_ml.config import FEATURE_NAMES

PATH_WINDOW = 60
PATH_COLUMNS = [f'cotton_path_ret_lag_{lag}' for lag in range(1, PATH_WINDOW)]
PATH_FEATURES = [*FEATURE_NAMES, *PATH_COLUMNS]


def add_return_path(history):
    """Current return is already in base; add its 59 preceding observations.

    Reject a compressed ordinal history. Missing feature values stay missing;
    the existing train-only Preprocessor handles them without removing rows.
    """
    if history.empty:
        raise ValueError('Return path requires nonempty observation history')
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Return path requires unique sorted dates')
    ordinals = history.cotton_session_index.to_numpy(dtype=float)
    if (not np.isfinite(ordinals).all() or (ordinals != np.floor(ordinals)).any()
            or (len(ordinals) > 1 and not np.equal(np.diff(ordinals), 1).all())):
        raise ValueError('Return path requires consecutive source Cotton observation ordinals')
    if set(PATH_COLUMNS).intersection(history.columns):
        raise ValueError('Return path already present; preserve frozen input')
    result = history.copy()
    for lag, name in enumerate(PATH_COLUMNS, 1):
        result[name] = history.cotton_ret_1.shift(lag)
    return result
