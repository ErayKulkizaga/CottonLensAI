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
