import hashlib
import json

import pytest

from cottonlens_ml.sources.nass_archive import (
    HistoricalDelayNotice,
    archive_release,
    audit_condition_year,
    compare_condition_to_quickstats,
    cotton_condition_summary,
    release_details,
    verified_cached_release,
)

PAGE = 'https://esmis.nal.usda.gov/publication/crop-progress/2020-06-01'
HTML = '''<time datetime="2020-06-01T12:00:00Z">Jun 01 2020</time>
<a href="/sites/default/release-files/a/b/prog2320.txt">TXT</a>'''


class Response:
    def __init__(self, raw, status_code=200):
        self.raw = raw
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, _):
        yield self.raw


class Session:
    def __init__(self):
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        assert kwargs['allow_redirects'] is False
        return Response(HTML.encode() if len(self.urls) == 1 else b'Crop Progress\nReleased June 1, 2020')


def test_archive_uses_report_link_and_stays_ineligible(tmp_path):
    session = Session()
    folder, meta = archive_release(PAGE, tmp_path, session=session)
    assert session.urls[1].endswith('/prog2320.txt')
    assert not meta['model_eligible'] and not meta['publication_clock_verified']
    assert (folder / 'report.txt').read_bytes().startswith(b'Crop Progress')


def test_cached_release_is_verified_before_reuse(tmp_path):
    folder, metadata = archive_release(PAGE, tmp_path, session=Session())
    cached_folder, cached_metadata = verified_cached_release(PAGE, tmp_path)
    assert cached_folder == folder and cached_metadata['report_sha256'] == metadata['report_sha256']
    (folder / 'report.txt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verified_cached_release(PAGE, tmp_path)


def test_release_page_must_have_one_official_txt_link():
    with pytest.raises(ValueError, match='one dated release'):
        release_details(HTML.replace('esmis', 'other').replace('/sites/default/release-files/',
                        'https://evil.example/release-files/'), PAGE)
    with pytest.raises(ValueError, match='one dated release'):
        release_details(HTML + HTML, PAGE)


def test_two_official_links_are_accepted_only_when_report_bytes_agree(tmp_path):
    html = (HTML + '\n<a href="/sites/default/release-files/a/c/prog2320.txt">TXT</a>')

    class MultiSession:
        def __init__(self, differing=False):
            self.differing = differing

        def get(self, url, **_):
            if '/publication/' in url:
                return Response(html.encode())
            return Response(b'Crop Progress\nReleased June 1, 2020'
                            + (b' changed' if self.differing and '/a/c/' in url else b''))

    folder, receipt = archive_release(PAGE, tmp_path, session=MultiSession())
    assert len(receipt['candidate_report_urls']) == 2
    assert verified_cached_release(PAGE, tmp_path)[0] == folder
    with pytest.raises(ValueError, match='different bytes'):
        archive_release(PAGE, tmp_path / 'different', session=MultiSession(differing=True))


@pytest.mark.parametrize(('day', 'notice'), [
    ('2012-10-29', b'NASS Delays October 29, 2012 Reports'),
    ('2018-09-10', b'USDA September 10 Crop Progress Report Delayed'),
])
def test_delay_notice_is_preserved_and_not_parsed_as_report(tmp_path, day, notice):
    page = 'https://esmis.nal.usda.gov/publication/crop-progress/' + day

    class NoticeSession:
        def get(self, url, **_):
            if '/publication/' in url:
                return Response((f'<time datetime="{day}T12:00:00Z"></time>'
                                 '<a href="/sites/default/release-files/a/b/notice.txt">TXT</a>').encode())
            return Response(notice + b'\nIssued by USDA')

    with pytest.raises(HistoricalDelayNotice):
        archive_release(page, tmp_path, session=NoticeSession())
    receipts = list((tmp_path / 'nass_crop_progress_notices').rglob('retrieval.json'))
    assert len(receipts) == 1
    assert not json.loads(receipts[0].read_text())['model_eligible']


def test_rejects_credential_url_before_network(tmp_path):
    with pytest.raises(ValueError, match='Explicit official'):
        archive_release(PAGE + '?key=secret', tmp_path, session=Session())


def test_condition_parser_rejects_changed_table_shape():
    report = ('Cotton Condition - Selected States: Week Ending June 7, 2015\n'
              'State : Very poor : Poor : Fair : Good : Excellent\n'
              '15 States .......: - 7 43 44 6\n')
    assert cotton_condition_summary(report)['national_condition']['VERY POOR'] == 0
    with pytest.raises(ValueError, match='sum to 100'):
        cotton_condition_summary(report.replace(' 44 ', ' 45 '))


def test_legacy_condition_table_has_explicit_category_and_week_checks():
    report = ('  Cotton:  Crop Condition by Percent, \n'
              '           Selected States,           \n'
              '       Week Ending May 30, 2010       \n'
              '  State : VP  :  P  :  F  :  G  : EX  \n'
              '15 Sts  :  0     4    33    51    12  \n')
    result = cotton_condition_summary(report)
    assert result['week_ending'] == '2010-05-30'
    assert result['national_condition']['GOOD'] == 51
    with pytest.raises(ValueError, match='category order'):
        cotton_condition_summary(report.replace('VP  :  P', 'P  :  VP'))


def test_quickstats_comparison_requires_five_unique_national_rows():
    summary = {'week_ending': '2015-06-07', 'national_condition':
               {'VERY POOR': 0, 'POOR': 7, 'FAIR': 43, 'GOOD': 44, 'EXCELLENT': 6}}
    rows = [{'week_ending': '2015-06-07', 'agg_level_desc': 'NATIONAL',
             'class_desc': 'UPLAND', 'short_desc':
             f'COTTON, UPLAND - CONDITION, MEASURED IN PCT {key}', 'Value': str(value)}
            for key, value in summary['national_condition'].items()]
    checked = compare_condition_to_quickstats(summary, rows)
    assert checked['all_values_match'] and not checked['model_eligible']
    partial = compare_condition_to_quickstats(summary, rows[1:])
    assert not partial['all_values_match']
    assert partial['missing_zero_categories'] == ['VERY POOR']
    with pytest.raises(ValueError, match='Missing'):
        compare_condition_to_quickstats(summary, rows[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        compare_condition_to_quickstats(summary, rows + rows[:1])
    with pytest.raises(ValueError, match='suppressed'):
        compare_condition_to_quickstats(summary, [{**rows[0], 'Value': '(D)'}, *rows[1:]])


def test_full_year_audit_checks_all_weeks_and_holiday_release(tmp_path):
    weeks = ('2023-07-02', '2023-09-03')
    values = {'VERY POOR': 13, 'POOR': 18, 'FAIR': 28, 'GOOD': 35, 'EXCELLENT': 6}
    rows = [{'week_ending': week, 'agg_level_desc': 'NATIONAL', 'class_desc': 'UPLAND',
             'short_desc': f'COTTON, UPLAND - CONDITION, MEASURED IN PCT {key}',
             'Value': str(value)} for week in weeks for key, value in values.items()]
    source = tmp_path / 'source.json'
    raw = json.dumps({'data': rows}).encode()
    source.write_bytes(raw)
    (tmp_path / 'retrieval.json').write_text(json.dumps({
        'provider': 'nass', 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'parameters': {'commodity_desc': 'COTTON', 'source_desc': 'SURVEY',
                       'freq_desc': 'WEEKLY', 'domain_desc': 'TOTAL',
                       'agg_level_desc': 'NATIONAL', 'year': 2023},
    }), encoding='utf-8')

    class AuditSession:
        def get(self, url, **_):
            if url.endswith('2023-09-04'):
                return Response(b'', 404)
            if '/publication/' in url:
                day = url.rsplit('/', 1)[1]
                return Response((f'<time datetime="{day}T12:00:00Z"></time>'
                                 '<a href="/sites/default/release-files/a/b/report.txt">TXT</a>').encode())
            week = 'July 2, 2023' if not hasattr(self, 'report_seen') else 'September 3, 2023'
            self.report_seen = True
            report = ('Crop Progress\nReleased July 3, 2023\n'
                      f'Cotton Condition - Selected States: Week Ending {week}\n'
                      'State : Very poor : Poor : Fair : Good : Excellent\n'
                      '15 States .......: 13 18 28 35 6\n')
            return Response(report.encode())

    result = audit_condition_year(source, tmp_path / 'archive', session=AuditSession())
    assert result['counts'] == {'numeric_match': 2}
    assert result['results'][1]['attempted_release_dates'] == ['2023-09-04', '2023-09-05']
    assert not result['model_eligible'] and not result['publication_clock_verified']


def test_audit_rejects_tampered_source(tmp_path):
    (tmp_path / 'source.json').write_text('{"data": []}', encoding='utf-8')
    (tmp_path / 'retrieval.json').write_text(json.dumps({
        'provider': 'nass', 'source_sha256': 'wrong', 'parameters': {},
    }), encoding='utf-8')
    with pytest.raises(ValueError, match='Verified NASS'):
        audit_condition_year(tmp_path / 'source.json', tmp_path / 'archive')
