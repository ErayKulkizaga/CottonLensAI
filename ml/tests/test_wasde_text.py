"""Offline narrative extraction; fake PDFs/pages only, no model fitting."""
import hashlib

import pandas as pd
import pytest
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources.wasde_text import (
    align_corpus,
    compile_corpus,
    cotton_narrative,
)

DAY = '2020-01-10'
BASE = 'https://esmis.nal.usda.gov'
RELEASE = BASE+'/publication/world-agricultural-supply-and-demand-estimates/'
WORDS = 'Production declined in Texas while domestic consumption and exports remained unchanged. '


def pages(day='January 10, 2020'):
    return [f'WASDE - 596 {day}\nWHEAT: other crop',
        'WASDE-596-4\nCOTTON: '+WORDS * 3,
        'WASDE-596-5\n'+WORDS * 3+'\nApproved by the Secretary of Agriculture\nTABLES: 1234']


def archive(tmp_path, url, raw):
    sha = hashlib.sha256(raw).hexdigest()
    folder = tmp_path/'wasde'/sha
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'source.bin').write_bytes(raw)
    freeze_record(folder/'retrieval.json', {'kind': 'wasde', 'source_url': url,
        'sha256': sha, 'retrieved_at': '2026-10-03T00:00:00+00:00', 'model_eligible': False})
    return folder


def release(tmp_path, day=DAY, suffix='', pdf='report.pdf'):
    url = RELEASE+day+suffix
    pdf_url = BASE+'/sites/default/release-files/'+pdf
    archive(tmp_path, url, f'<a href="{pdf_url}">PDF</a>'.encode())
    return archive(tmp_path, pdf_url, b'%PDF-fake-'+pdf.encode())


def compile_fake(tmp_path, extractor=lambda _: pages(), name='corpus.json'):
    return compile_corpus(tmp_path, tmp_path/name, extract_pages=extractor, extractor_identity='fake-v1')


def test_extracts_multpage_cotton_only():
    body = cotton_narrative(pages(), DAY)
    assert body.count('Production') == 6
    assert all(word not in body for word in ('WHEAT', 'WASDE', 'TABLES', 'Secretary'))


@pytest.mark.parametrize('page_text', [pages('January 11, 2020'),
    ['January 10, 2020\nCOTTON: table numbers 1 2 3'],
    [p.replace('COTTON:', 'Cotton table') for p in pages()],
    [pages()[0], pages()[1]+'\nCOTTON: duplicate', pages()[2]]])
def test_date_heading_and_bounds_are_required(page_text):
    with pytest.raises(ValueError):
        cotton_narrative(page_text, DAY)


def test_corpus_is_checksumming_only_never_model_admission(tmp_path):
    release(tmp_path)
    body = compile_fake(tmp_path)
    assert body['unique_report_count'] == body['unique_narrative_count'] == 1
    assert not body['model_eligible'] and not body['publication_timestamp_verified']
    assert body['records'][0]['retrieved_at'].startswith('2026')
    assert body['records'][0]['report_day'] == DAY
    assert 'available_at' not in body['records'][0]
    assert read_record(tmp_path/'corpus.json')['records'] == body['records']
    assert compile_fake(tmp_path) == body


def test_full_url_matching_not_shared_latest_basename(tmp_path):
    release(tmp_path, pdf='one/latest.pdf')
    archive(tmp_path, BASE+'/sites/default/release-files/two/latest.pdf', b'other')
    selected = []
    def extract(path):
        selected.append(path.read_bytes())
        return pages()
    assert compile_fake(tmp_path, extract)['unique_report_count'] == 1
    assert selected == [b'%PDF-fake-one/latest.pdf']


def test_verified_url_alias_reuses_same_bytes_without_losing_provenance(tmp_path):
    folder = release(tmp_path, pdf='canonical.pdf')
    alias_url = BASE+'/sites/default/release-files/alias.pdf'
    canonical_url = BASE+'/sites/default/release-files/canonical.pdf'
    alias_path = tmp_path/'wasde_url_aliases'/hashlib.sha256(alias_url.encode()).hexdigest()/'alias.json'
    freeze_record(alias_path, {'source_url': alias_url, 'canonical_source_url': canonical_url,
                              'sha256': folder.name})
    # A second dated page points to an identical PDF hosted at a different URL.
    archive(tmp_path, RELEASE+DAY+'-1', f'<a href="{alias_url}">PDF</a>'.encode())
    body = compile_fake(tmp_path)
    assert body['unique_report_count'] == 1
    assert not body['exclusions']
    alias_path.write_text('corrupt')
    with pytest.raises(ValueError):
        compile_fake(tmp_path, name='corrupt.json')


def test_corrupt_archived_pdf_is_fatal(tmp_path):
    folder = release(tmp_path)
    (folder/'source.bin').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='checksum'):
        compile_fake(tmp_path)


def test_seen_audit_is_not_read_or_admitted(tmp_path):
    release(tmp_path)
    bad = release(tmp_path, day='2024-01-10', pdf='future.pdf')
    (bad/'source.bin').write_bytes(b'future changed')
    assert compile_fake(tmp_path)['unique_report_count'] == 1


def test_missing_pdf_and_host_mismatch_are_explicit_exclusions(tmp_path):
    archive(tmp_path, RELEASE+DAY, b'<a href="https://untrusted.example/r.pdf">PDF</a>')
    body = compile_fake(tmp_path)
    assert body['unique_report_count'] == 0
    assert 'Public HTTPS' in body['exclusions'][0]['reason']


def test_conflicting_same_day_revisions_excluded(tmp_path):
    release(tmp_path)
    release(tmp_path, suffix='-1', pdf='corrected.pdf')
    def extract(path):
        text = pages()
        if b'corrected' in path.read_bytes():
            text[1] = text[1].replace('declined', 'increased')
        return text
    body = compile_fake(tmp_path, extract)
    assert body['unique_report_count'] == 0
    assert len(body['exclusions']) == 2


def test_lag_uses_recorded_origins_and_keeps_missing_rows(tmp_path):
    release(tmp_path)
    compile_fake(tmp_path)
    history = pd.DataFrame({'date': pd.bdate_range('2020-01-09', periods=10),
                            'cotton_close': 60.})
    nominal = align_corpus(history, tmp_path/'corpus.json', lag=1, max_age=6)
    stress = align_corpus(history, tmp_path/'corpus.json', lag=6, max_age=6)
    assert len(nominal) == len(stress) == len(history)
    assert nominal.wasde_text_missing.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    assert stress.wasde_text_missing.tolist() == [1]*7+[0, 1, 1]
    assert nominal.wasde_text_age.iloc[2] == 1
    assert stress.wasde_text_age.iloc[7] == 6
    pd.testing.assert_series_equal(nominal.cotton_close, history.cotton_close)


def test_future_report_and_prices_do_not_change_past_alignment(tmp_path):
    release(tmp_path)
    compile_fake(tmp_path)
    history = pd.DataFrame({'date': pd.bdate_range('2020-01-09', periods=80),
                            'cotton_close': 60.})
    before = align_corpus(history, tmp_path/'corpus.json')
    release(tmp_path, day='2020-03-10', pdf='march.pdf')
    def extract(path):
        return pages('March 10, 2020') if b'march' in path.read_bytes() else pages()
    compile_fake(tmp_path, extract, name='later.json')
    changed = history.copy()
    changed.loc[changed.date.ge('2020-03-10'), 'cotton_close'] = 999.
    after = align_corpus(changed, tmp_path/'later.json')
    pd.testing.assert_frame_equal(before.loc[before.date.lt('2020-03-10')],
                                  after.loc[after.date.lt('2020-03-10')])


def test_alignment_rejects_unsorted_calendar(tmp_path):
    release(tmp_path)
    compile_fake(tmp_path)
    history = pd.DataFrame({'date': pd.to_datetime(['2020-01-10', '2020-01-09'])})
    with pytest.raises(ValueError, match='chronological'):
        align_corpus(history, tmp_path/'corpus.json')
