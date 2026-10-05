import json

import pytest
from cottonlens_ml.sources.ecb_fx import archive_ecb_year, parse_ecb_csv

HEADER = 'KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n'
ROWS = ''.join(f'EXR.D.{currency}.EUR.SP00.A,D,{currency},EUR,SP00,A,2020-04-01,5.25,A\n'
               for currency in ('BRL', 'CNY', 'INR', 'USD'))


class Response:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, _):
        yield self.body


class Session:
    def get(self, url, **kwargs):
        assert url.endswith('/D.BRL+CNY+INR+USD.EUR.SP00.A')
        assert kwargs['params']['includeHistory'] == 'true'
        assert kwargs['allow_redirects'] is False
        return Response((HEADER + ROWS).encode())


def test_archive_ecb_rates_keeps_original_unit_and_quarantine(tmp_path):
    folder, receipt = archive_ecb_year(2020, tmp_path, session=Session())
    assert receipt['rows'] == 4 and receipt['series_rows']['INR'] == 1
    assert receipt['unit'].startswith('currency units per 1 EUR')
    assert not receipt['publication_clock_verified'] and not receipt['model_eligible']
    assert (folder / 'source.csv').read_bytes() == (HEADER + ROWS).encode()
    same, _ = archive_ecb_year(2020, tmp_path, session=Session())
    assert same == folder
    (folder / 'source.csv').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='corrupted'):
        archive_ecb_year(2020, tmp_path, session=Session())


def test_ecb_validation_rejects_wrong_series_and_preserves_revisions():
    summary = parse_ecb_csv((HEADER + ROWS + ROWS.splitlines()[0] + '\n').encode(), 2020)
    assert summary['repeated_currency_dates'] == 1
    assert not summary['vintage_verified']
    with pytest.raises(ValueError, match='series identity'):
        parse_ecb_csv((HEADER + ROWS.replace('EXR.D.INR', 'EXR.D.PKR')).encode(), 2020)
    with pytest.raises(ValueError, match='positive'):
        parse_ecb_csv((HEADER + ROWS.replace('5.25,A', '-1,A', 1)).encode(), 2020)
    holiday = ROWS.replace('2020-04-01,5.25,A', '2020-04-01,,H', 1)
    summary = parse_ecb_csv((HEADER + holiday).encode(), 2020)
    assert summary['missing_value_rows'] == 1
    assert summary['missing_value_statuses'] == {'H': 1}


def test_ecb_receipt_contains_no_model_ready_claim(tmp_path):
    folder, _ = archive_ecb_year(2020, tmp_path, session=Session())
    saved = json.loads((folder / 'retrieval.json').read_text(encoding='utf-8'))
    assert saved['model_eligible'] is False
    assert saved['vintage_verified'] is False
