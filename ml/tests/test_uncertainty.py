import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.uncertainty import evaluate_intervals, residual_intervals
from test_sprint import history


def test_intervals_ignore_unmatured_future_labels_and_nest():
    frame = history().head(500)
    past = frame.date.iloc[130:300].dt.strftime('%Y-%m-%d').tolist()
    test = frame.date.iloc[310:350].dt.strftime('%Y-%m-%d').tolist()
    intervals = residual_intervals(frame, past, np.zeros(len(past)), test, np.zeros(len(test)), 5)
    modified = frame.copy()
    origin = pd.Timestamp(test[0])
    modified.loc[modified.target_date_5 >= origin, 'target_return_5'] = 100.
    altered = residual_intervals(modified, past, np.zeros(len(past)), test[:1], [0.], 5)
    assert intervals['rows'][0] == altered['rows'][0]
    for row in intervals['rows']:
        assert row['mature_calibration_count'] == 126
        assert row['90']['lower'] <= row['80']['lower'] <= row['80']['upper'] <= row['90']['upper']
    report = evaluate_intervals(frame, intervals, 5)
    assert report['count'] == len(test)
    assert 0 <= report['levels']['90']['coverage'] <= 1


def test_insufficient_calibration_is_explicit_and_overlapping_origins_rejected():
    frame = history().head(300)
    past = frame.date.iloc[130:140].dt.strftime('%Y-%m-%d').tolist()
    test = frame.date.iloc[150:151].dt.strftime('%Y-%m-%d').tolist()
    result = residual_intervals(frame, past, [0.] * 10, test, [0.], 1)
    assert result['rows'][0]['status'] == 'insufficient_mature_calibration'
    with pytest.raises(ValueError, match='disjoint'):
        residual_intervals(frame, past, [0.] * 10, past[:1], [0.], 1)
