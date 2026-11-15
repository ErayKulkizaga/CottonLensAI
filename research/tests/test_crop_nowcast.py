"""Guards that keep the crop-nowcast test years sealed and the API key out of files (no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from research.crop_nowcast import (
    county_ndvi,
    county_up,
    landsat_fields,
    nass_pull,
    power_pull,
)


def test_nass_years_are_capped_before_seal():
    assert nass_pull.MAX_YEAR < nass_pull.SEALED_FROM == 2022


def test_nass_refuses_responses_with_other_years(tmp_path, monkeypatch):
    monkeypatch.setenv('NASS_API_KEY', 'SECRET-KEY')
    monkeypatch.setattr(nass_pull, 'QUERIES', {'q': {}})
    monkeypatch.setattr(nass_pull, 'fetch', lambda p, k, y: json.dumps({'data': [{'year': '2023'}]}).encode())
    with pytest.raises(SystemExit):
        nass_pull.main(tmp_path)
    assert not list(tmp_path.iterdir())


def test_nass_receipt_never_contains_key(tmp_path, monkeypatch):
    monkeypatch.setenv('NASS_API_KEY', 'SECRET-KEY')
    monkeypatch.setattr(nass_pull, 'QUERIES', {'q': {}})
    monkeypatch.setattr(nass_pull, 'fetch', lambda p, k, y: json.dumps({'data': [{'year': str(y), 'Value': '1'}]}).encode())
    nass_pull.main(tmp_path)
    for f in tmp_path.iterdir():
        assert 'SECRET-KEY' not in f.read_text()
    assert json.loads((tmp_path / 'receipt.json').read_text())['queries']['q']['rows'] == 14


@pytest.mark.parametrize('mod', [power_pull, county_ndvi, landsat_fields])
def test_sealed_years_need_explicit_unseal(mod, tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['x'])
    with pytest.raises(SystemExit):
        mod.main(str(tmp_path / 'panel.json'), str(tmp_path), 2009, 2022)


def test_landsat_features_respect_lag_and_count_failed_fields():
    # 40 points in one county: half green (0.7) on 1 July, half never green (0.2);
    # a late green scene on 25 July for every point must not count for aug1 (25 Jul + 16 d > 1 Aug)
    n = 40
    idx = np.r_[np.arange(n), np.arange(n)]
    day = np.array(['2019-07-01'] * n + ['2019-07-25'] * n, dtype='datetime64[D]')
    nd = np.r_[np.where(np.arange(n) < 20, 0.7, 0.2), np.full(n, 0.9)]
    rows = landsat_fields.county_features(idx, day, nd, np.zeros(n, int), ['48001'], 2019)
    aug = next(r for r in rows if r['cutoff'] == 'aug1')
    sep = next(r for r in rows if r['cutoff'] == 'sep1')
    assert aug['l_fail25'] == 0.5 and aug['l_p90'] == 0.7 and aug['l_nobs'] == 1
    assert sep['l_fail25'] == 0 and sep['l_last'] == 0.9 and sep['l_nobs'] == 2  # usable by 1 Sep


def test_landsat_merge_anomaly_uses_only_earlier_years(tmp_path):
    for y, v in ((2010, 0.4), (2011, 0.2), (2012, 0.6)):
        row = dict.fromkeys(county_up.LANDSAT, v) | {'fips': '48303', 'year': y, 'cutoff': 'aug1'}
        pd.DataFrame([row, row | {'cutoff': 'oct1', 'l_p50': 9.9}]).to_csv(tmp_path / f'landsat_TX_{y}.csv', index=False)
    c = pd.DataFrame({'fips': ['48303'] * 3, 'year': [2010, 2011, 2012]})
    m = county_up.add_landsat(c, tmp_path, 'aug1').set_index('year')
    assert m.l_p50.tolist() == [0.4, 0.2, 0.6]  # aug1 rows only
    assert np.isnan(m.loc[2010, 'l_p50_a'])
    assert np.isclose(m.loc[2011, 'l_p50_a'], -0.2) and np.isclose(m.loc[2012, 'l_p50_a'], 0.3)


def test_search_items_retries_transient_stac_errors(monkeypatch):
    APIError = pytest.importorskip('pystac_client.exceptions').APIError

    class Fake:
        calls = 0

        def search(self, **kw):
            Fake.calls += 1
            if Fake.calls < 3:
                raise APIError('upstream connect error')
            return type('S', (), {'item_collection': lambda self: ['item']})()

    monkeypatch.setattr(landsat_fields.time, 'sleep', lambda s: None)
    assert landsat_fields.search_items(Fake(), collections=['x']) == ['item']
    assert Fake.calls == 3
