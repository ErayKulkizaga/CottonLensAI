import pytest
from cottonlens_ml.sprint import freeze_record
from review_wasde_availability import inventory


def test_joined_archive_is_not_admitted_as_available(tmp_path):
    urls = [f'https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/'
            f'{year}-{month:02d}-10'
            for year in range(2016, 2024) for month in range(1, 13)
            if (year, month) != (2019, 1)]
    urls.extend([urls[35] + '-0', urls[45] + '-0', urls[45] + '-1'])
    earlier = [url for url in urls if int(url.rsplit('/', 1)[-1][:4]) < 2020]
    later = [url for url in urls if int(url.rsplit('/', 1)[-1][:4]) >= 2020]
    ambiguous = {month: [url for url in earlier
                         if url.rsplit('/', 1)[-1].startswith(month)]
                 for month in ('2018-12', '2019-11')}
    first, second = tmp_path / 'earlier.json', tmp_path / 'later.json'
    freeze_record(first, {'release_count': len(earlier),
                          'missing_months': ['2019-01'],
                          'ambiguous_months': ambiguous,
                          'releases': [{'url': url, 'numeric_status': 'passed'}
                                       for url in earlier]})
    freeze_record(second, {'release_count': len(later),
                           'releases': [{'url': url, 'numeric_status': 'passed'}
                                        for url in later]})
    result = inventory(first, second)
    assert result['published_months'] == 95
    assert result['release_page_links'] == 98
    assert result['publication_clock_verified_months'] == 0
    assert result['model_eligible'] is False


def test_numeric_mismatch_cannot_enter_inventory(tmp_path):
    first, second = tmp_path / 'earlier.json', tmp_path / 'later.json'
    freeze_record(first, {'release_count': 1, 'releases': [
        {'url': 'https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/2016-01-10',
         'numeric_status': 'mismatch'}]})
    freeze_record(second, {'release_count': 0, 'releases': []})
    with pytest.raises(ValueError, match='numeric review incomplete'):
        inventory(first, second)
