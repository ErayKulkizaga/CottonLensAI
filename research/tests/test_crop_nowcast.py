"""Guards that keep the crop-nowcast test years sealed and the API key out of files (no network)."""
import json

import pytest

from research.crop_nowcast import county_ndvi, nass_pull, power_pull


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


@pytest.mark.parametrize('mod', [power_pull, county_ndvi])
def test_sealed_years_need_explicit_unseal(mod, tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['x'])
    with pytest.raises(SystemExit):
        mod.main(str(tmp_path / 'panel.json'), str(tmp_path), 2009, 2022)
