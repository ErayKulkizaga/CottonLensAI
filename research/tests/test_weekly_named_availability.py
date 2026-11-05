from decimal import Decimal

from research.weekly_named_availability import collect_additions


def table(value, publication='2014-06-06'):
    return {'rows': [{'observation_date': '2014-06-04', 'printed_publication_date': publication,
                     'available': True, 'settlements': {'2014-10': value}}]}


def test_original_and_correction_conflict_excludes_date_in_both_orders():
    original, correction = table('79.96'), table('76.96')
    for tables in ({'original': original, 'Correction': correction},
                   {'Correction': correction, 'original': original}):
        additions, conflicts = collect_additions(tables)
        assert additions == {}
        assert conflicts == ['2014-06-04']


def test_identical_republication_keeps_earliest_printed_date():
    additions, conflicts = collect_additions({'later': table('76.96', '2014-06-13'),
                                             'earlier': table('76.96')})
    assert conflicts == []
    assert additions['2014-06-04']['printed'] == '2014-06-06'
    assert additions['2014-06-04']['prices']['2014-10'] == Decimal('76.96')
