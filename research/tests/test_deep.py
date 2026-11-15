"""Checks for the T35 histogram extraction (no network)."""
import numpy as np
import pytest

from research.deep import county_hist as H


def test_histograms_count_per_county_band_and_skip_invalid():
    v = np.full((6, 5), 0.3, np.float32)
    v[0] = [-0.2, 0.99, 1.5, np.nan, 0.4]  # NDVI: low edge, high edge, clipped above, invalid, middle
    idx = np.array([0, 0, 1, 1, 1])
    h = H.histograms(v, idx, 2)
    assert h.shape == (2, 6, H.BINS)
    assert h[0, 0, 0] == 1 and h[0, 0, -1] == 1 and h[1, 0, -1] == 1 and h[:, 0].sum() == 4
    assert h[:, 2].sum() == 5  # reflectance band, all valid


def test_hist_refuses_sealed_years(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['x'])
    with pytest.raises(SystemExit):
        H.main(str(tmp_path / 'p.json'), str(tmp_path), 2009, 2022)


def test_slot_respects_cutoff_lag_and_start():
    from research.deep import build_tensor as B
    assert B.n_slots('oct1') == 22
    assert B.slot_of('2019-04-07', 'oct1') == 0
    assert B.slot_of('2019-09-22', 'oct1') == 21  # 22 Sep + 8 days = 30 Sep, before 1 Oct
    assert B.slot_of('2019-09-24', 'oct1') is None  # 24 Sep + 8 days = 2 Oct, after the cutoff
