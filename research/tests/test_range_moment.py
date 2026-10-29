import importlib.util
from pathlib import Path

import numpy as np
import pytest

loader = importlib.util.spec_from_file_location('range_moment', Path(__file__).parents[1]/'range_moment.py')
moment = importlib.util.module_from_spec(loader)
loader.loader.exec_module(moment)


def test_covariance_ignores_ordinal_missing_slots():
    a = np.array([[1, 2, 1], [999, 999, 0], [3, 6, 1]], float)
    assert moment.covariance(a) == 2


def test_global_offsets_do_not_manufacture_covariance():
    a = np.array([[1, 1, 1], [2, 3, 1], [3, 2, 1]], float)
    b = a.copy(); b[:, :2] += [1000, 3000]
    assert moment.covariance(a) == pytest.approx(moment.covariance(b), abs=1e-9)


def test_known_relation_positive_and_constant_error_zero():
    z = np.arange(40, dtype=float)
    positive = np.column_stack([z, z*2, np.ones(40)])
    out = moment.intervals([positive], 10, 256)
    assert out['ci_95'][0] > 0
    positive[:, 1] = 1
    assert moment.intervals([positive], 10, 256)['ci_95'] == [0, 0]


def test_empty_range_cohort_cannot_be_a_negative_result():
    with pytest.raises(ValueError, match='No eligible'):
        moment.covariance(np.zeros((10, 3)))
