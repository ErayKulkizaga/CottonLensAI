"""Named rank and origin state stay fixed when a source price is unusable."""

import csv

import pandas as pd

from research.ams_named_readiness import attach, quotes_from


def test_invalid_second_price_does_not_promote_third(tmp_path):
    path = tmp_path/'observations.csv'
    fields = ['date', 'contract', 'settlement_cents_raw', 'source_date_matches',
              'price_precision_2dp']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for contract, price, valid in [('2023-03-01', '80.00', True),
                                       ('2023-05-01', '81.001', False),
                                       ('2023-07-01', '82.00', True)]:
            writer.writerow({'date': '2023-01-03', 'contract': contract,
                             'settlement_cents_raw': price, 'source_date_matches': True,
                             'price_precision_2dp': valid})
    quote = quotes_from(path)['2023-01-03']
    assert quote['ranked'][1] == '2023-05-01'
    assert '2023-05-01' not in quote['prices']


def test_origin_ewma_uses_only_observed_past_pairs():
    days = pd.bdate_range('2023-01-03', periods=27)
    first, second = '2023-03-01', '2023-05-01'
    quotes = {}
    for index, day in enumerate(days):
        prices = {first: 100., second: 101.+index}
        if index == 23:
            del prices[second]
        quotes[day.strftime('%Y-%m-%d')] = {'ranked': [first, second], 'prices': prices}
    rows = attach(pd.DataFrame({'date': days}), quotes)
    assert pd.isna(rows.iloc[0].named_ewma)
    assert rows.iloc[0].named_eligible
    assert rows.iloc[22].named_ewma_eligible
    assert rows.iloc[22].own_feature_ready
    assert not rows.iloc[23].named_eligible
    assert not rows.iloc[23].own_feature_ready
    assert rows.iloc[24].own_latest_pair_age == 2
    assert not rows.iloc[-1].named_eligible
