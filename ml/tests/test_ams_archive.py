import json

import pytest

from cottonlens_ml.sources.ams_archive import (
    archive_spot_excerpt,
    archive_spot_month_listing,
    audit_spot_inventory,
    decode_ams_text,
    parse_spot_excerpt,
    rebuild_month_inventory,
    verified_cached_excerpt,
)

PAGE = ('<time datetime="2020-04-01T12:00:00Z"></time>'
        '<a href="/sites/default/release-files/a/b/MP_CN001.TXT">TXT</a>')
REPORT = ('MP_CN001                            0\n'
          'MEMPHIS, TN                  4/1/2020        USDA Cotton and Tobacco Program, MND\n'
          '       MARKET        41-4/34 31-3/35  BALES    FUTURES     TODAY\n'
          'AVERAGE               43.23   46.72     73      Mar-22     55.52\n')


class Response:
    status_code = 200

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, _):
        yield self.data


class Session:
    def __init__(self):
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        assert kwargs['allow_redirects'] is False
        return Response(PAGE.encode() if len(self.urls) == 1 else REPORT.encode())


def test_archive_spot_from_linked_uppercase_txt_and_keep_ineligible(tmp_path):
    session = Session()
    folder, parsed = archive_spot_excerpt('2020-04-01', tmp_path, session=session)
    assert session.urls[1].endswith('MP_CN001.TXT')
    assert parsed['seven_market_average_41_4_34_cents_per_lb'] == 43.23
    assert parsed['reported_spot_bales'] == 73
    assert parsed['published_at'] is None and not parsed['model_eligible']
    assert (folder / 'report.txt').read_bytes() == REPORT.encode()
    same, _ = archive_spot_excerpt('2020-04-01', tmp_path, session=Session())
    assert same == folder


def test_report_date_and_quality_must_match():
    alternate = REPORT.replace('4/1/2020', '1-Apr-20')
    assert parse_spot_excerpt(alternate, '2020-04-01')['reported_spot_bales'] == 73
    with pytest.raises(ValueError, match='date disagrees'):
        parse_spot_excerpt(REPORT, '2020-04-02')
    with pytest.raises(ValueError, match='quality/header'):
        parse_spot_excerpt(REPORT.replace('41-4/34', 'unknown'), '2020-04-01')


def test_legacy_cp1252_punctuation_does_not_change_ascii_quote_values():
    raw = (REPORT + "Farmer's cotton\n").replace("'", '\u2019').encode('cp1252')
    decoded, encoding = decode_ams_text(raw)
    assert encoding == 'cp1252'
    assert parse_spot_excerpt(decoded, '2020-04-01')['reported_spot_bales'] == 73
    with pytest.raises(UnicodeDecodeError):
        decode_ams_text(b'\x81')


def test_masked_date_needs_matching_independent_official_report(tmp_path):
    masked = REPORT.replace('4/1/2020', '#########')
    independent = ('MP_CN002    Memphis, TN Cotton and Tobacco Progr  1-Apr-20\n'
                   'Average       -318    Dec-21      43.23    46.72       73\n')
    with pytest.raises(ValueError, match='Expected one report date'):
        parse_spot_excerpt(masked, '2020-04-01')
    assert parse_spot_excerpt(masked, '2020-04-01', independent_text=independent)[
        'seven_market_average_41_4_34_cents_per_lb'] == 43.23
    with pytest.raises(ValueError, match='Independent AMS date or average'):
        parse_spot_excerpt(masked, '2020-04-01',
                           independent_text=independent.replace('43.23', '44.23'))

    independent_url = ('https://esmis.nal.usda.gov/publication/'
                       'daily-spot-quotations-pg-1-front-page/2020-04-01')
    independent_page = ('<time datetime="2020-04-01T12:00:00Z"></time>'
                        '<a href="/sites/default/release-files/a/b/MP_CN002.TXT">TXT</a>')

    class FourSources:
        def __init__(self):
            self.data = iter((PAGE, masked, independent_page, independent))

        def get(self, *_args, **_kwargs):
            return Response(next(self.data).encode())

    folder, receipt = archive_spot_excerpt('2020-04-01', tmp_path, session=FourSources(),
                                           independent_page_url=independent_url)
    assert receipt['independent_page_url'] == independent_url
    assert verified_cached_excerpt('2020-04-01', receipt['release_page_url'], tmp_path)
    (folder / 'independent-report.txt').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='independent AMS evidence checksum'):
        verified_cached_excerpt('2020-04-01', receipt['release_page_url'], tmp_path)


def test_month_listing_preserves_exact_duplicate_suffix_link(tmp_path):
    first = ('<tr><td><time datetime="2020-04-01T12:00:00Z">April 1</time></td>'
             '<td><a href="/publication/daily-spot-quotations-excerpts/2020-04-01-1">View</a></td>'
             '</tr><a href="?date=2020-04&amp;page=1" aria-label="Next page" '
             'class="usa-pagination__next-page">Next</a>'
             '<a href="?date=2020-04&amp;page=8" aria-label="Last page" '
             'class="usa-pagination__next-page">Last</a>')
    second = ('<tr><td><time datetime="2020-04-02T12:00:00Z">April 2</time></td>'
              '<td><a href="/publication/daily-spot-quotations-excerpts/2020-04-02">View</a></td>'
              '</tr>')

    class ListingSession:
        def get(self, url, **_):
            return Response((second if 'page=1' in url else first).encode())

    inventory = archive_spot_month_listing('2020-04', tmp_path, session=ListingSession())
    assert inventory['release_count'] == inventory['distinct_dates'] == 2
    assert inventory['entries'][0]['release_page_url'].endswith('2020-04-01-1')
    assert len(inventory['listing_pages']) == 2
    assert not inventory['publication_clock_verified'] and not inventory['model_eligible']


def test_explicit_release_url_must_match_date_before_network(tmp_path):
    with pytest.raises(ValueError, match='matching report date'):
        archive_spot_excerpt('2020-04-01', tmp_path, session=Session(),
                             release_page_url='https://evil.example/2020-04-01')
    with pytest.raises(ValueError, match='matching report date'):
        archive_spot_excerpt('2020-04-01', tmp_path, session=Session(),
                             release_page_url='https://esmis.nal.usda.gov/publication/'
                                              'daily-spot-quotations-excerpts/2020-04-02')


def test_rebuild_month_inventory_only_from_verified_archived_pages(tmp_path):
    row = ('<tr><td><time datetime="2020-04-01T12:00:00Z">April 1</time></td>'
           '<td><a href="/publication/daily-spot-quotations-excerpts/2020-04-01">View</a></td></tr>')

    class ListingSession:
        def get(self, *_args, **_kwargs):
            return Response(row.encode())

    inventory = archive_spot_month_listing('2020-04', tmp_path, session=ListingSession())
    year = tmp_path / 'year.json'
    year.write_text(json.dumps({'year': 2020, 'months': [inventory]}), encoding='utf-8')
    rebuilt = rebuild_month_inventory(year, '2020-04')
    assert rebuilt['entries'] == inventory['entries']
    assert rebuilt['release_count'] == 1 and not rebuilt['model_eligible']
    with pytest.raises(ValueError, match='disagree'):
        rebuild_month_inventory(year, '2021-04')
    page = next(tmp_path.glob('ams_month_listings/2020-04/0/*/listing.html'))
    page.write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        rebuild_month_inventory(year, '2020-04')


def test_month_audit_reuses_only_verified_completed_reports(tmp_path):
    row = ('<tr><td><time datetime="2020-04-01T12:00:00Z">April 1</time></td>'
           '<td><a href="/publication/daily-spot-quotations-excerpts/2020-04-01">View</a></td></tr>')

    class ListingSession:
        def get(self, *_args, **_kwargs):
            return Response(row.encode())

    inventory = archive_spot_month_listing('2020-04', tmp_path, session=ListingSession())
    file = tmp_path / 'inventory.json'
    file.write_text(json.dumps(inventory), encoding='utf-8')
    url = inventory['entries'][0]['release_page_url']
    folder, _ = archive_spot_excerpt('2020-04-01', tmp_path, session=Session(), release_page_url=url)
    audit = audit_spot_inventory(file)
    assert audit['release_count'] == 1 and audit['results'][0]['reported_spot_bales'] == 73
    assert not audit['model_eligible']
    (folder / 'report.txt').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verified_cached_excerpt('2020-04-01', url, tmp_path)
    with pytest.raises(ValueError, match='checksum mismatch'):
        audit_spot_inventory(file)
