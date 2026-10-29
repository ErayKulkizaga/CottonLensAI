import csv
import hashlib

import pytest
from cottonlens_ml.research.ledger import freeze_record, read_record

from research.ams_weekly_merge import merge
from research.ams_weekly_recovery import (
    annual_prefix_quotes,
    compare,
    daily_witness,
    parse_review,
)


def table():
    return {'contracts': ['2018-10-01', '2018-12-01', '2019-03-01', '2019-05-01', '2019-07-01', '2019-10-01'],
            'rows': ['2018-10-05 76.49 76.10 77.14 77.95 78.71 76.14',
                     '2018-10-08 77.94 77.55 78.49 79.21 79.84 77.27',
                     '2018-10-09 77.40 77.01 78.07 78.89 79.54 77.05',
                     '2018-10-10 - 76.80 78.04 78.86 79.52 76.93']}


def test_literal_leading_blank_is_retained_without_far_contract_invention():
    row = parse_review(table(), '2018-10-12')[-1]
    assert row['explicit_blank_contracts'] == ['2018-10-01']
    assert len(row['quotes']) == 5
    assert list(row['quotes'])[:2] == ['2018-12-01', '2019-03-01']
    assert '2019-12-01' not in row['quotes']


@pytest.mark.parametrize('bad', ['2018-10-12', '2018-09-01'])
def test_no_future_or_out_of_week_observation(bad):
    t = table(); t['rows'][0] = t['rows'][0].replace('2018-10-05', bad)
    with pytest.raises(ValueError, match='outside printed'):
        parse_review(t, '2018-10-12')


@pytest.mark.parametrize('bad', ['0.00', '-2.00', 'nan', '77.555', '-'])
def test_bad_or_interior_blank_cells_rejected(bad):
    t = table(); t['rows'][0] = t['rows'][0].replace('76.10', bad)
    with pytest.raises(ValueError):
        parse_review(t, '2018-10-12')


def test_price_disagreement_cannot_be_repaired_by_changing_transcription():
    row = parse_review(table(), '2018-10-12')[0]
    annual = {str(i): {'price': p} for i, p in enumerate([*row['quotes'].values(), '75.00', '76.00', '77.00', '78.00'])}
    status, conflicts = compare(annual, None, row)
    assert status == 'verified_partial_prefix' and not conflicts
    daily = dict(row['quotes']); daily['2018-12-01'] = '76.11'
    status, conflicts = compare(annual, daily, row)
    assert status == 'price_conflict_no_admission'
    assert conflicts[0]['kind'] == 'dated_daily_price'
    annual['0']['price'] = '75.49'
    assert compare(annual, None, row)[0] == 'price_conflict_no_admission'


def test_rejected_daily_witness_still_prevents_weekly_conflict_admission(tmp_path):
    names = ['Oct-18', 'Dec-18', 'Mar-19', 'May-19', 'Jul-19', 'Oct-19', 'Dec-19', 'Mar-20', 'May-20', 'Jul-20']
    data = ('MP_CN001 10/05/2018 FUTURES\n'
            + '\n'.join(n+' 75.00' for n in names)+'\nCOTTON QUOTATIONS').encode('ascii')
    digest = hashlib.sha256(data).hexdigest()
    raw = tmp_path/'inputs'/'daily'/(digest+'.txt'); raw.parent.mkdir(parents=True); raw.write_bytes(data)
    overlay = {'quotes': {}, 'raw_sources': {'2018-10-05': {'document_sha256': digest}}}
    daily = daily_witness(tmp_path, overlay, '2018-10-05')
    assert daily['2018-12-01'] == '75.00'
    row = parse_review(table(), '2018-10-12')[0]
    annual = {str(i): {'price': p} for i, p in enumerate([*row['quotes'].values(), '75.00', '76.00', '77.00', '78.00'])}
    assert compare(annual, daily, row)[0] == 'price_conflict_no_admission'
    raw.write_bytes(data+b'changed')
    with pytest.raises(ValueError, match='witness changed'):
        daily_witness(tmp_path, overlay, '2018-10-05')


def test_merge_keeps_existing_quotes_and_rejects_conflict(tmp_path):
    p, w = tmp_path/'parent.json', tmp_path/'weekly.json'
    freeze_record(p, {'training_allowed': False, 'quotes': {'2018-10-05': {'A': '75.00', 'B': '76.00'}}})
    freeze_record(w, {'training_allowed': False, 'quotes': {'2018-10-05': {'A': '75.00'}}})
    merge(p, w, tmp_path/'out.json')
    assert read_record(tmp_path/'out.json')['quotes']['2018-10-05']['B'] == '76.00'
    broken = tmp_path/'broken.json'
    freeze_record(broken, {'training_allowed': False, 'quotes': {'2018-10-05': {'A': '74.00'}}})
    with pytest.raises(ValueError, match='prices differ'):
        merge(p, broken, tmp_path/'rejected.json')
    assert not (tmp_path/'rejected.json').exists()


def test_invalid_far_cell_preserves_ordinal_positions_and_valid_prefix(tmp_path):
    path = tmp_path/'annual.csv'; row = parse_review(table(), '2018-10-12')[0]
    names = ['2019-05-01', '2019-07-01', '2019-10-01', '2019-12-01', '2020-03-01',
             '2020-05-01', '2020-07-01', '2020-10-01', '2020-12-01', '2021-03-01']
    values = [*row['quotes'].values(), '76.843', '76.84', '77.54', '77.71']
    def save(bad_index):
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'contract', 'settlement_cents_raw', 'source_date_matches', 'price_precision_2dp'])
            writer.writeheader()
            writer.writerow({'date': row['date'], 'contract': names[0], 'settlement_cents_raw': '99.99',
                             'source_date_matches': 'False', 'price_precision_2dp': 'True'})
            writer.writerows({'date': row['date'], 'contract': c, 'settlement_cents_raw': p,
                              'source_date_matches': 'True', 'price_precision_2dp': str(i != bad_index)}
                             for i, (c, p) in enumerate(zip(names, values, strict=True)))
    save(6)
    annual = annual_prefix_quotes(path)[row['date']]
    assert len(annual) == 10 and annual[names[6]]['valid'] is False
    assert compare(annual, None, row)[0] == 'verified_partial_prefix'
    # Change only the validity flag of an actual prefix cell, not its position.
    annual[names[1]]['valid'] = False
    assert compare(annual, None, row)[0] == 'invalid_annual_prefix_no_admission'
