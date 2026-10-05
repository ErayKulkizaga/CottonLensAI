import hashlib
from decimal import InvalidOperation

import pytest
from cottonlens_ml.sources.wasde import (
    ATTRIBUTES,
    acquire_manifest,
    archive_release,
    audit_archives,
    compare_pdf_pages,
    compare_text,
    cotton_rows,
    inspect_archive,
)
from cottonlens_ml.sprint import freeze_record, read_record


def xml(value='89.93', month='Jan'):
    return f'''<Report><Report Report_Month="January 2023"
    sub_report_title="World Cotton Supply and Use  1/"
    sub_report_subtitle="(Million 480-Pound Bales)">
    <matrix2 region_header2="2022/23 Proj."><region region2="World">
    <month forecast_month2="{month}"><attribute attribute2="Ending Stocks">
    <Cell cell_value2="{value}"/><Cell filler="filler"/>
    </attribute></month></region></matrix2></Report></Report>'''.encode()


def test_preserves_report_season_and_forecast_dimensions():
    row, = cotton_rows(xml())
    assert row['marketing_year'] == '2022/23 Proj.'
    assert row['report_month'] == 'January 2023'
    assert row['forecast_month'] == 'Jan'
    assert row['value'] == '89.93'
    assert 'published_at' not in row


@pytest.mark.parametrize('raw', [xml('UNKNOWN'), xml('NaN'), xml().replace(
    b'Million 480-Pound Bales', b'Metric Tons'), b'<!DOCTYPE x><Report/>',
    xml().replace(b'<Cell filler="filler"/>', b'<Cell cell_value2="89.93"/>')])
def test_rejects_unknown_units_values_and_duplicates(raw):
    with pytest.raises((ValueError, InvalidOperation)):
        cotton_rows(raw)


def test_archive_integrity_and_no_automatic_eligibility(tmp_path):
    raw = xml()
    (tmp_path / 'source.bin').write_bytes(raw)
    freeze_record(tmp_path / 'retrieval.json', {'kind': 'wasde',
        'sha256': hashlib.sha256(raw).hexdigest(), 'source_url': 'fixture'})
    result = inspect_archive(tmp_path)
    assert result['row_count'] == 1 and result['model_eligible'] is False
    (tmp_path / 'source.bin').write_bytes(xml('90'))
    with pytest.raises(ValueError, match='checksum'):
        inspect_archive(tmp_path)


def test_small_value_footnote_is_not_silently_zero():
    with pytest.raises(ValueError, match='numeric marker'):
        cotton_rows(xml('3/'))
    raw = xml('3/').replace(b'<matrix2', b'<matrix2 sub_report_footer="3/ Less than 5,000 bales."')
    # Footnotes must belong to the report, not an arbitrary child.
    with pytest.raises(ValueError, match='numeric marker'):
        cotton_rows(raw)
    raw = xml('3/').replace(b'Report_Month=', b'sub_report_footer="3/ Less than 5,000 bales." Report_Month=')
    row, = cotton_rows(raw)
    assert row['value'] is None and row['upper_bound_million_bales'] == '0.005'


def test_text_comparison_detects_values_dates_and_missing_rows():
    rows = [{'marketing_year': '2022/23 Proj.', 'region': 'World', 'forecast_month': 'Jan',
             'report_month': 'January 2023', 'attribute': name, 'value': '1.00'} for name in ATTRIBUTES]
    text = '''WASDE - 632 - 27 January 2023
World Cotton Supply and Use
(Million 480-Pound Bales)
Beginning Produc-
Stocks    tion Imports Domestic Exports
2022/23 Proj.
World
Jan 1.00 1.00 1.00 1.00 1.00 1.00 1.00
'''
    assert compare_text(rows, text)['compared_cells'] == 7
    changed = compare_text(rows, text.replace('Jan 1.00', 'Jan 2.00'))
    assert not changed['all_values_match'] and len(changed['mismatches']) == 1
    with pytest.raises(ValueError, match='month mismatch'):
        compare_text(rows, text.replace('January 2023', 'February 2023'))
    with pytest.raises(ValueError, match='coverage mismatch'):
        compare_text(rows, text.replace('World\n', 'China\n'))
    missing = [{**r, 'value': None, 'qualifier': 'not_available'} for r in rows]
    assert compare_text(missing, text.replace('1.00', 'NA'))['all_values_match']
    assert not compare_text(missing, text.replace('1.00', '3/'))['all_values_match']


def test_new_season_unavailable_forecast_is_not_zero():
    row, = cotton_rows(xml('NA', 'Apr'))
    assert row['value'] is None and row['qualifier'] == 'not_available'


def test_batch_audit_blocks_missing_pairs_and_rejects_corruption(tmp_path):
    raw = b'<time datetime="2023-01-12T12:00:00Z"></time><a href="/report.xml">XML</a>'
    checksum = hashlib.sha256(raw).hexdigest()
    folder = tmp_path / 'wasde' / checksum
    folder.mkdir(parents=True)
    (folder / 'source.bin').write_bytes(raw)
    freeze_record(folder / 'retrieval.json', {'kind': 'wasde', 'sha256': checksum,
        'source_url': 'https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/2023-01-12'})
    report = audit_archives(tmp_path)
    assert report['release_count'] == 1 and report['numeric_passed'] == 0
    assert not report['model_eligible']
    assert report['releases'][0]['publication_clock_status'] == 'unverified'
    assert 'Missing or ambiguous' in report['releases'][0]['reason']
    (folder / 'source.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        audit_archives(tmp_path)


def test_release_acquisition_resumes_verified_files(tmp_path, monkeypatch):
    from cottonlens_ml.sources import wasde

    base = 'https://esmis.nal.usda.gov'
    page = base + '/publication/world-agricultural-supply-and-demand-estimates/2023-01-12'
    calls = []

    def fetch(kind, url, root):
        calls.append(url)
        raw = (b'<a href="/report.xml">XML</a><a href="/report.txt">TXT</a>'
               if url == page else b'fixture report')
        # Different payloads ensure independent receipts, as real XML/TXT do.
        raw += url.encode()
        checksum = hashlib.sha256(raw).hexdigest()
        folder = root / 'wasde' / checksum
        folder.mkdir(parents=True)
        (folder / 'source.bin').write_bytes(raw)
        freeze_record(folder / 'retrieval.json', {'kind': kind, 'sha256': checksum, 'source_url': url})
        return folder

    monkeypatch.setattr(wasde, 'archive', fetch)
    first = archive_release(page, tmp_path)
    assert len(calls) == 3
    assert archive_release(page, tmp_path) == first and len(calls) == 3
    with pytest.raises(ValueError, match='official WASDE'):
        archive_release('https://example.com/report', tmp_path)


def test_manifest_records_failure_and_continues(tmp_path, monkeypatch):
    from cottonlens_ml.sources import wasde

    manifest = tmp_path / 'manifest.json'
    freeze_record(manifest, {'release_urls': ['first', 'second']})
    seen = []

    def fail(url, root, **kwargs):
        seen.append(url)
        raise ValueError('unsupported fixture')

    monkeypatch.setattr(wasde, 'archive_release', fail)
    result = acquire_manifest(manifest, tmp_path)
    assert seen == ['first', 'second']
    assert all(row['status'] == 'blocked' for row in result['results'])
    assert not result['model_eligible']


def test_older_release_archives_pdf_when_txt_was_not_published(tmp_path, monkeypatch):
    from cottonlens_ml.sources import wasde

    page = 'https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/2016-01-12'
    calls = []

    def fetch(kind, url, root):
        calls.append(url)
        raw = (b'<a href="/report.xml">XML</a><a href="/report.pdf">PDF</a>'
               if url == page else url.encode())
        checksum = hashlib.sha256(raw).hexdigest()
        folder = root / kind / checksum
        folder.mkdir(parents=True)
        (folder / 'source.bin').write_bytes(raw)
        freeze_record(folder / 'retrieval.json', {'kind': kind, 'sha256': checksum,
                                                'source_url': url})
        return folder

    monkeypatch.setattr(wasde, 'archive', fetch)
    files = archive_release(page, tmp_path, permit_missing_text=True, include_pdf=True)
    assert set(files) == {'page', '.xml', '.pdf'} and len(calls) == 3
    with pytest.raises(ValueError, match='official .txt'):
        archive_release(page, tmp_path)


def test_identical_release_bytes_preserve_distinct_official_urls(tmp_path, monkeypatch):
    from cottonlens_ml.sources import wasde

    page = 'https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/2019-11-08'
    calls = []

    def fetch(kind, url, root):
        calls.append(url)
        raw = (b'<a href="/first.xml">XML</a><a href="/second.txt">TXT</a>'
               if url == page else b'identical report bytes')
        checksum = hashlib.sha256(raw).hexdigest()
        folder = root / kind / checksum
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'source.bin').write_bytes(raw)
        freeze_record(folder / 'retrieval.json', {'kind': kind, 'sha256': checksum,
                                                'source_url': calls[1] if len(calls) > 1 else url})
        return folder

    monkeypatch.setattr(wasde, 'archive', fetch)
    files = archive_release(page, tmp_path)
    assert files['.xml'] == files['.txt']
    alias_url = 'https://esmis.nal.usda.gov/second.txt'
    alias = tmp_path / 'wasde_url_aliases' / hashlib.sha256(alias_url.encode()).hexdigest() / 'alias.json'
    assert read_record(alias)['source_url'] == alias_url
    assert archive_release(page, tmp_path) == files and len(calls) == 3


def test_pdf_comparison_checks_headers_and_inline_forecast_rows():
    rows = [{'marketing_year': '2019/20 Proj.', 'region': 'World', 'forecast_month': 'Jan',
             'report_month': 'January 2020', 'attribute': name, 'value': '1.00'} for name in ATTRIBUTES]
    page = '''January 2020
WASDE - 596 - 27
World Cotton Supply and Use 1/
(Million 480-Pound Bales)
2019/20 Proj. Beginning Production Imports Domestic Exports Loss Ending
Stocks Use /2 Stocks
World Jan 1.00 1.00 1.00 1.00 1.00 1.00 1.00
'''
    assert compare_pdf_pages(rows, [page])['all_values_match']
    repaired = compare_pdf_pages(rows, [page.replace('Jan 1.00', 'Jan 1.00filler')])
    assert repaired['all_values_match'] and repaired['pdf_filler_tokens_removed'] == 1
    assert not compare_pdf_pages(rows, [page.replace('Jan 1.00', 'Jan 2.00')])['all_values_match']
    with pytest.raises((ValueError, InvalidOperation)):
        compare_pdf_pages(rows, [page.replace('Jan 1.00', 'Jan 1.00bogus')])
    with pytest.raises(ValueError, match='PDF column order'):
        compare_pdf_pages(rows, [page.replace('Imports Domestic', 'Domestic Imports')])
    with pytest.raises(ValueError, match='coverage mismatch'):
        compare_pdf_pages(rows, [page.replace('World Jan', 'China Jan')])


def test_fixed_column_pdf_layout_rejects_missing_and_changed_cells():
    from cottonlens_ml.sources.wasde import compare_pdf_layout_rows

    rows = [{'marketing_year': '2021/22 Proj.', 'region': 'China', 'forecast_month': 'Aug',
             'report_month': 'September 2021', 'attribute': name, 'value': '1.00'}
            for name in ATTRIBUTES]
    page = '''Septem ber 2021
World Cotton Supply and Use 1/
(Million 480-Pound Bales)
2021/22 P roj. Beginning Production Imports Domestic Exports Loss Ending
Stocks Use /2 Stocks
Chi na Aug 1.00 1.00 1.00 1.00 1.00 1.00 1.00
'''
    assert compare_pdf_layout_rows(rows, [page])['all_values_match']
    assert not compare_pdf_layout_rows(rows, [page.replace('Aug 1.00', 'Aug 2.00')])['all_values_match']
    with pytest.raises(ValueError, match='PDF cotton row'):
        compare_pdf_layout_rows(rows, [page.replace('1.00 1.00 1.00\n', '1.00 1.00\n')])
    with pytest.raises(ValueError, match='column order'):
        compare_pdf_layout_rows(rows, [page.replace('Imports Domestic', 'Domestic Imports')])


def test_listing_manifest_requires_every_month_and_resumes_archive(tmp_path, monkeypatch):
    from cottonlens_ml.sources import wasde

    links = ''.join(
        f'<a href="/publication/world-agricultural-supply-and-demand-estimates/2016-{m:02d}-10" '
        'aria-label="View World Agricultural Supply and Demand Estimates">View</a>'
        for m in range(1, 13))
    calls = []

    def fetch(kind, url, root):
        calls.append(url)
        raw = links.encode()
        checksum = hashlib.sha256(raw).hexdigest()
        folder = root / kind / checksum
        folder.mkdir(parents=True)
        (folder / 'source.bin').write_bytes(raw)
        freeze_record(folder / 'retrieval.json', {'kind': kind, 'sha256': checksum,
                                                'source_url': url})
        return folder

    monkeypatch.setattr(wasde, 'archive', fetch)
    report = wasde.discover_release_manifest(2016, 2016, [8], tmp_path)
    assert len(report['release_urls']) == 12 and not report['model_eligible']
    assert wasde.discover_release_manifest(2016, 2016, [8], tmp_path) == report
    assert len(calls) == 1
    incomplete = wasde.discover_release_manifest(2016, 2017, [8], tmp_path)
    assert incomplete['missing_months'] == [f'2017-{m:02d}' for m in range(1, 13)]
    links += ('<a href="/publication/world-agricultural-supply-and-demand-estimates/2016-11-10-1" '
              'aria-label="View World Agricultural Supply and Demand Estimates">View</a>')
    ambiguous = wasde.discover_release_manifest(2016, 2016, [8], tmp_path / 'other')
    assert len(ambiguous['release_urls']) == 13
    assert len(ambiguous['ambiguous_months']['2016-11']) == 2
