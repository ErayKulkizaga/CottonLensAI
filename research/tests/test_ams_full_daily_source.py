import pytest

from research.ams_full_daily_source import pdf_release, settlements


def fixture():
    names = ['Dec-20', 'Mar-21', 'May-21', 'Jul-21', 'Oct-21', 'Dec-21', 'Mar-22', 'May-22', 'Jul-22', 'Oct-22']
    return ('Month\nToday Previous Change\n'+'\n'.join(n+' 71.00 70.00 1.00' for n in names)
            +'\n1/  Settlement prices provided by ICE Futures US\nICE FUTURES US - CONTRACT NO. 2\nICE Futures\n04-Dec-20\n')


def test_roles_dates_and_no_previous_date_inference():
    result = settlements(fixture(), '2020-12-04')
    assert result['quotes']['2021-03-01'] == '71.00'
    assert result['previous_column_date'] is None
    assert result['previous_column_admitted_as_dated_quote'] is False
    assert result['historical_availability_verified'] is False


def test_older_header_layout_is_dated_by_publisher_header_not_season_high():
    text = fixture().replace('ICE Futures\n04-Dec-20', 'Cotton and Tobacco Market News 04-Dec-20')
    assert settlements(text, '2020-12-04')['observation_date'] == '2020-12-04'
    text += '\nCotton and Tobacco Market News 03-Dec-20'
    with pytest.raises(ValueError, match='header date'):
        settlements(text, '2020-12-04')


@pytest.mark.parametrize('old,new', [('04-Dec-20', '03-Dec-20'), (' 1.00', ' 2.00'), ('Mar-21', 'Dec-20'),
                                    ('Settlement prices provided by ICE Futures US', 'Estimated prices')])
def test_mismatch_fails_closed(old, new):
    with pytest.raises(ValueError):
        settlements(fixture().replace(old, new), '2020-12-04')


def test_pdf_link_parser_is_separate_from_txt_parser_and_does_not_admit_clock():
    html = '<time datetime="2020-12-04T12:00:00Z"></time><a href="/sites/default/release-files/a/b/CNDDSQ.PDF">PDF</a>'
    stamp, url = pdf_release(html, 'https://esmis.nal.usda.gov/publication/daily-spot-quotations/2020-12-04', '2020-12-04')
    assert stamp == '2020-12-04T12:00:00Z' and url.endswith('/CNDDSQ.PDF')
    with pytest.raises(ValueError, match='date differs'):
        pdf_release(html, 'https://esmis.nal.usda.gov', '2018-11-26')
