import hashlib

import pandas as pd
import pytest
from cottonlens_ml.research.price_reconciliation import parse_quotes, reconcile


def document(date, first, second):
    return f'MEMPHIS TN {date}\nFUTURES TODAY\n Mar-20 {first:.2f}\n May-20 {second:.2f}\nCOTTON QUOTATIONS'.encode()


def test_quote_correspondence_and_contract_spread_are_not_price_repairs(tmp_path):
    entries = []
    for date, data in [('2020-03-09', document('3/9/2020', 60, 61)),
                       ('2020-03-10', b'MEMPHIS TN 3/10/2020\nFUTURES TODAY\n May-20 62.0\nCOTTON QUOTATIONS')]:
        sha = hashlib.sha256(data).hexdigest()
        folder = tmp_path / 'raw' / 'ams' / sha
        folder.mkdir(parents=True)
        (folder / 'source.bin').write_bytes(data)
        entries.append({'report_date': date, 'status': 'Final', 'document_sha256': sha, 'document_url': 'https://example.com'})
    market = pd.DataFrame({'series': 'cotton', 'date': pd.to_datetime(['2020-03-09', '2020-03-10']),
        'close': [60., 62.], 'low': [61., 61.], 'high': [63., 63.]})
    original = market.copy(deep=True)
    summary, _ = reconcile(market, {'version_records': entries[::-1]}, tmp_path)
    pd.testing.assert_frame_equal(original, market)
    transition = summary['quoted_contract_transitions'][0]
    assert transition['observed_change_cents'] == 2.
    assert transition['same_contract_change_cents'] == 1.
    assert transition['previous_day_contract_spread_cents'] == 1.
    assert summary['first_quote_matches'] == 2 and summary['outside_range_first_quote_matches'] == 1
    assert not summary['forecast_labels_repaired'] and not summary['model_eligible']
    with pytest.raises(ValueError, match='Unique'):
        reconcile(market, {'version_records': entries + entries}, tmp_path)
    with pytest.raises(ValueError, match='absent'):
        reconcile(market.iloc[:1], {'version_records': entries}, tmp_path)
    folder.joinpath('source.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        reconcile(market, {'version_records': entries}, tmp_path)


@pytest.mark.parametrize('data', [document('3/9/2020', 60, 61), b'MEMPHIS 3/10/2020\nFUTURES TODAY\nCOTTON QUOTATIONS',
    b'MEMPHIS 3/10/2020\nFUTURES TODAY\n Mar-20 60.0\n Mar-20 61.0\nCOTTON QUOTATIONS'])
def test_wrong_reference_or_ambiguous_empty_table_rejected(data):
    with pytest.raises(ValueError):
        parse_quotes(data, '2020-03-10')


def test_historical_date_format_and_unreadable_header_are_distinct(tmp_path):
    assert parse_quotes(document('24-Nov-21', 60., 61.), '2021-11-24')['Mar-20'] == 60.
    data = document('#########', 60., 61.)
    sha = hashlib.sha256(data).hexdigest()
    folder = tmp_path / 'raw' / 'ams' / sha
    folder.mkdir(parents=True)
    (folder / 'source.bin').write_bytes(data)
    market = pd.DataFrame({'series': ['cotton'], 'date': pd.to_datetime(['2021-10-20']), 'close': [60.], 'low': [59.], 'high': [61.]})
    entry = {'report_date': '2021-10-20', 'status': 'Final', 'document_sha256': sha, 'document_url': 'https://example.com'}
    # No verified dated table remains, so an empty successful reconciliation is forbidden.
    with pytest.raises(ValueError, match='Existing'):
        reconcile(market, {'version_records': [entry]}, tmp_path)


def test_missing_history_cannot_infer_consecutive_roll(tmp_path):
    entries = []
    for date, data in [('2020-03-09', document('3/9/2020', 60.0, 61.0)),
                       ('2020-03-11', b'MEMPHIS 3/11/2020\nFUTURES TODAY\n May-20 62.0\nCOTTON QUOTATIONS')]:
        sha = hashlib.sha256(data).hexdigest()
        folder = tmp_path / 'raw' / 'ams' / sha
        folder.mkdir(parents=True)
        (folder / 'source.bin').write_bytes(data)
        entries.append({'report_date': date, 'status': 'Final', 'document_sha256': sha, 'document_url': 'https://example.com'})
    market = pd.DataFrame({'series': 'cotton', 'date': pd.to_datetime(['2020-03-09', '2020-03-10', '2020-03-11']),
        'close': [60., 61., 62.], 'low': 59., 'high': 63.})
    summary, _ = reconcile(market, {'version_records': entries}, tmp_path)
    assert not summary['quoted_contract_transitions'][0]['price_decomposition_supported']
