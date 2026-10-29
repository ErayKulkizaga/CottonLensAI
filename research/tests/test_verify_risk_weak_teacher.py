import numpy as np
import pytest

from research.verify_risk_weak_teacher import (
    difference_interval,
    fingerprint,
    read,
    read_predictions,
)


def test_independent_interval_retains_calendar_gaps_for_constant_gain():
    part = np.column_stack([np.full(25, .125), np.ones(25)])
    part[::4] = 0
    for block in (20, 60):
        assert difference_interval([part], block) == pytest.approx([.125, .125])


def test_independent_reader_rejects_modified_receipt(tmp_path):
    import json
    body = {'new_fits': 0}
    path = tmp_path/'record.json'
    path.write_text(json.dumps({**body, 'record_id': fingerprint(body)}), encoding='utf-8')
    assert read(path) == body
    path.write_text(json.dumps({'new_fits': 1, 'record_id': fingerprint(body)}), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        read(path)


def test_null_condition_survives_csv_while_empty_numeric_is_missing(tmp_path):
    path = tmp_path/'predictions.csv'
    path.write_text('condition,seed,gain\nnull,42,\nweak,1701,0.01\n', encoding='utf-8')
    rows = read_predictions(path)
    assert rows.condition.tolist() == ['null', 'weak']
    assert np.isnan(rows.gain.iloc[0])
