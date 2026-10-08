import hashlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources import wasde_regional as regional
from cottonlens_ml.sources import wasde_regional_validation as audit


def values():
    return {'crop_year': 2019, **{name: 1. for name in regional.LEVELS}}


def test_values_missingness_revision_and_future_mutation_are_preserved():
    records = [{'report_date': date, 'values': values()} for date in ['2020-01-10', '2020-02-11', '2020-05-12']]
    records[-1]['values']['crop_year'] = 2020
    frame = regional.add_revisions(pd.DataFrame([{'report_date': pd.Timestamp(r['report_date']), **r['values']} for r in records]))
    expected = audit.verify_candidate_values(frame, records)
    assert expected.world_production_revision.iloc[[0, 2]].isna().all()
    records[-1]['values']['china_consumption'] = 999.
    frame.loc[2, 'china_consumption'] = 999.
    pd.testing.assert_frame_equal(expected.head(2), audit.verify_candidate_values(frame, records).head(2))
    frame.loc[0, 'us_exports'] = 0.
    with pytest.raises(ValueError, match='values'):
        audit.verify_candidate_values(frame, records)
    with pytest.raises(ValueError, match='chronological'):
        audit.verify_candidate_values(frame.iloc[::-1], records)


def test_gap_or_different_crops_cannot_fabricate_revision():
    records = [{'report_date': date, 'values': values()} for date in ['2020-01-10', '2020-05-12']]
    frame = regional.add_revisions(pd.DataFrame([{'report_date': pd.Timestamp(r['report_date']), **r['values']} for r in records]))
    frame['world_production_revision'] = 0.
    with pytest.raises(ValueError, match='revisions'):
        audit.verify_candidate_values(frame, records)


@pytest.mark.parametrize('text', ['World   Cotton    Supply and Use', 'unrecognized layout'])
@pytest.mark.parametrize('matching', [True, False])
def test_pdf_fallback_never_uses_an_empty_page_selection(tmp_path, monkeypatch, text, matching):
    path = tmp_path / 'source.pdf'
    path.write_bytes(b'synthetic PDF; no market evidence')
    page = SimpleNamespace(extract_text=lambda **kwargs: text)
    reader = SimpleNamespace(pages=[page], metadata={})
    monkeypatch.setitem(sys.modules, 'pypdf', SimpleNamespace(PdfReader=lambda raw: reader))
    class PDF:
        pages = (page,)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setitem(sys.modules, 'pdfplumber', SimpleNamespace(open=lambda raw: PDF()))
    def blocked(*args):
        raise ValueError('Unreviewed first extractor layout')
    monkeypatch.setattr(audit, 'compare_pdf_layout_rows', blocked)
    calls = []
    def compare(rows, pages):
        assert pages == [text]
        calls.append(pages)
        return {'all_values_match': matching, 'compared_cells': 196, 'mismatches': [] if matching else ['changed cell']}
    monkeypatch.setattr(audit, 'compare_pdf_pages', compare)
    if not matching:
        with pytest.raises(ValueError, match='PDF/XML mismatch'):
            audit.verify_pdf([], path)
        assert len(calls) == 1  # No alternate setting may erase a numeric mismatch.
        return
    result = audit.verify_pdf([], path)
    assert result['pdf_page_indexes'] == [0]
    assert result['metadata_is_publication_evidence'] is False


def test_archived_alias_requires_exact_canonical_source_identity(tmp_path):
    canonical, alias = 'https://www.usda.gov/a.pdf', 'https://www.usda.gov/b.pdf'
    raw = b'one synthetic PDF'
    sha = hashlib.sha256(raw).hexdigest()
    folder = tmp_path / 'wasde' / sha
    folder.mkdir(parents=True)
    (folder / 'source.bin').write_bytes(raw)
    freeze_record(folder / 'retrieval.json', {'kind': 'wasde', 'source_url': canonical,
        'sha256': sha, 'retrieved_at': '2026-09-29T00:00:00Z'})
    path = tmp_path / 'wasde_url_aliases' / hashlib.sha256(alias.encode()).hexdigest() / 'alias.json'
    proof = {'source_url': alias, 'canonical_source_url': canonical, 'sha256': sha}
    freeze_record(path, proof)
    index, hashes = audit.archive_index(tmp_path)
    assert index[alias] == index[canonical] == {sha}
    assert hashes[path.relative_to(tmp_path).as_posix()] == digest(path)
    path.unlink()  # Only synthetic test evidence, deliberately rebuild a bad receipt.
    freeze_record(path, {**proof, 'canonical_source_url': 'https://www.usda.gov/another.pdf'})
    with pytest.raises(ValueError, match='alias source'):
        audit.archive_index(tmp_path)


def test_shifted_second_cotton_page_is_not_silently_omitted(tmp_path, monkeypatch):
    path = tmp_path / 'source.pdf'
    path.write_bytes(b'synthetic shifted pages')
    texts = ['other commodity'] * 31
    texts[28], texts[29] = 'World Cotton Supply and Use older crop', 'World Cotton Supply and Use newer crop'
    pages = [SimpleNamespace(extract_text=lambda text=t, **kwargs: text) for t in texts]
    monkeypatch.setitem(sys.modules, 'pypdf', SimpleNamespace(PdfReader=lambda raw: SimpleNamespace(pages=pages, metadata={})))
    def compare(rows, actual):
        assert texts[28] in actual and texts[29] in actual
        return {'all_values_match': True, 'compared_cells': 196}
    monkeypatch.setattr(audit, 'compare_pdf_layout_rows', compare)
    assert audit.verify_pdf([], path)['pdf_page_indexes'] == [28, 29]


def test_signed_candidate_fresh_checkpoints_and_no_historical_promotion(tmp_path, monkeypatch):
    root = tmp_path / 'raw'
    release_url = 'https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/2020-01-10'
    source = {}
    for field, data in [('xml', b'synthetic XML'), ('pdf', b'synthetic PDF'),
                        ('page', b'<a href="https://www.usda.gov/a.pdf">PDF</a><a href="https://www.usda.gov/xml">XML.xml</a><a href="https://www.usda.gov/a.xml">XML</a>')]:
        sha = hashlib.sha256(data).hexdigest()
        folder = root / 'wasde' / sha
        folder.mkdir(parents=True)
        (folder / 'source.bin').write_bytes(data)
        freeze_record(folder / 'retrieval.json', {'kind': 'wasde', 'sha256': sha,
            'source_url': {'pdf': 'https://www.usda.gov/a.pdf', 'xml': 'https://www.usda.gov/a.xml', 'page': release_url}[field],
            'retrieved_at': '2026-09-29T00:00:00Z'})
        source[field + '_sha256'] = sha
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    frame = regional.add_revisions(pd.DataFrame([{'report_date': pd.Timestamp('2020-01-10'), **values()}]))
    frame.to_csv(candidate / 'regional.csv', index=False)
    freeze_record(candidate / 'candidate-manifest.json', {'version': regional.VERSION,
        'model_eligible': False, 'publication_timestamp_verified': False, 'first_version_verified': False,
        'source_versions': [{**source, 'release_url': release_url, 'report_date': '2020-01-10'}],
        'files': {'regional.csv': digest(candidate / 'regional.csv')}})
    freeze_record(candidate / 'complete.json', {'version': regional.VERSION, 'completed': True, 'fits': 0,
        'files': {name: digest(candidate / name) for name in ['regional.csv', 'candidate-manifest.json']}})
    monkeypatch.setitem(sys.modules, 'pypdf', SimpleNamespace(__version__='synthetic-test'))
    monkeypatch.setitem(sys.modules, 'pdfplumber', SimpleNamespace(__version__='synthetic-test'))
    monkeypatch.setattr(audit, 'cotton_rows', lambda raw: [])
    monkeypatch.setattr(audit, 'regional_values', lambda rows, date: (values(), {}))
    monkeypatch.setattr(audit, 'verify_pdf', lambda rows, path: {'pdf_sha256': digest(path), 'compared_cells': 1})
    original = digest(candidate / 'regional.csv')
    output = tmp_path / 'audit'
    report = audit.analyze(candidate, root, output)
    assert report['numeric_verified_rows'] == 1 and report['historical_model_eligible_rows'] == 0
    assert not report['model_eligible'] and read_record(output / 'complete.json')['fits'] == 0
    assert digest(candidate / 'regional.csv') == original
    coverage = pd.read_csv(output / 'coverage.csv')
    assert coverage.available_at.isna().all() and not coverage.historical_model_eligible.any()
    # Deliberately remove only the new synthetic completion/output to simulate interruption.
    for name in ['complete.json', 'report.json', 'coverage.csv']:
        (output / name).unlink()
    monkeypatch.setattr(audit, 'verify_pdf', lambda *args: pytest.fail('Verified checkpoint was needlessly re-extracted'))
    audit.analyze(candidate, root, output)
    with pytest.raises(ValueError, match='immutable'):
        audit.analyze(candidate, root, output)
    for name in ['complete.json', 'report.json', 'coverage.csv']:
        (output / name).unlink()
    receipt_path = root / 'wasde' / source['xml_sha256'] / 'retrieval.json'
    receipt = read_record(receipt_path)
    receipt_path.unlink()  # A changed ingestion receipt must invalidate the old checkpoint.
    freeze_record(receipt_path, {**receipt, 'retrieved_at': '2026-09-30T00:00:00Z'})
    calls = []
    def fresh(rows, path):
        calls.append(path)
        return {'pdf_sha256': digest(path), 'compared_cells': 1}
    monkeypatch.setattr(audit, 'verify_pdf', fresh)
    changed = audit.analyze(candidate, root, output)
    assert len(calls) == 1 and changed['records'][0]['xml_retrieved_at'] == '2026-09-30T00:00:00Z'
    assert len(list((output / 'checkpoints').glob('*/*.json'))) == 2
    (root / 'wasde' / source['xml_sha256'] / 'source.bin').write_bytes(b'corrupt source')
    with pytest.raises(ValueError, match='checksum'):
        audit.analyze(candidate, root, tmp_path / 'corrupt-source-audit')
    assert not (tmp_path / 'corrupt-source-audit' / 'complete.json').exists()
    (candidate / 'regional.csv').write_text('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        audit.analyze(candidate, root, tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists()
