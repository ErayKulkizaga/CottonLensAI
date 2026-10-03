import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.return_path import (
    PATH_COLUMNS,
    PATH_FEATURES,
    add_return_path,
)


def history():
    return pd.DataFrame({'date': pd.date_range('2010-01-01', periods=160),
        'cotton_session_index': np.arange(160), 'cotton_ret_1': np.arange(160) / 10000,
        'missing_feature': np.nan, 'target_return_5': np.arange(160) / 1000})


def test_future_changes_do_not_change_past_path():
    frame = history()
    changed = frame.copy()
    changed.loc[100:, ['cotton_ret_1', 'target_return_5']] = 999
    pd.testing.assert_frame_equal(add_return_path(frame).iloc[:100], add_return_path(changed).iloc[:100])


def test_path_keeps_missing_rows_and_original_frame():
    frame = history()
    original = frame.copy()
    frame.loc[80, 'cotton_ret_1'] = np.nan
    result = add_return_path(frame)
    assert len(result) == 160 and result.index.equals(frame.index)
    assert np.isnan(result.loc[81, PATH_COLUMNS[0]])
    assert result.loc[120, PATH_COLUMNS[-1]] == frame.loc[61, 'cotton_ret_1']
    assert list(result.columns) == [*frame.columns, *PATH_COLUMNS]
    assert len(PATH_FEATURES) == 83 and len(set(PATH_FEATURES)) == 83
    pd.testing.assert_frame_equal(original.drop(columns='cotton_ret_1'), frame.drop(columns='cotton_ret_1'))
    pd.testing.assert_frame_equal(result[frame.columns], frame)


def test_compressed_observations_are_rejected():
    with pytest.raises(ValueError, match='consecutive'):
        add_return_path(history().drop(index=80))
    frame = history()
    frame['cotton_session_index'] = frame.cotton_session_index.astype(float)
    frame.loc[80, 'cotton_session_index'] = 80.5
    with pytest.raises(ValueError, match='consecutive'):
        add_return_path(frame)


def test_duplicate_or_repeated_path_is_rejected():
    frame = history()
    frame.loc[80, 'date'] = frame.loc[79, 'date']
    with pytest.raises(ValueError, match='unique sorted'):
        add_return_path(frame)
    with pytest.raises(ValueError, match='already present'):
        add_return_path(add_return_path(history()))
