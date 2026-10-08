import csv
import hashlib
import io
import zipfile

import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources import wasde_as_reported as audit
from cottonlens_ml.sources import wasde_regional as regional


def rows(day='2021-01-12', crop=2020, scale=1):
    # Distinct values exercise region/attribute selection and the US denominator.
    levels = [120, 100, 80, 20, 5, 10, 3, 40, 30, 12]
    result = []
    stamp = pd.Timestamp(day)
    for (name, (region, attribute)), value in zip(regional.FIELDS.items(), levels, strict=True):
        result.append(dict(zip(audit.COLUMNS, ['607', stamp.strftime('%B %Y'), audit.TITLE,
            attribute, '', 'Cotton', region, f'{crop}/{(crop + 1) % 100:02d}', 'Proj.',
            'Annual', str(value * scale), audit.UNIT, day, '12:00:00', str(stamp.year), str(stamp.month)], strict=True)))
    return result


def encode(records):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=audit.COLUMNS)
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode()


def candidate_frame(records):
    frame = pd.DataFrame([{'report_date': pd.Timestamp(r['report_date']), 'crop_year': r['crop_year'], **r['values']} for r in records])
    return regional.add_revisions(frame)


def test_two_date_formats_latest_season_right_table_and_report_clock():
    current = rows()
    for row in current:
        row['ReleaseDate'] = '01/12/2021'
    other_table = [{**r, 'ReportTitle': 'Reliability of January Projections', 'Value': '9999'} for r in current]
    older = rows(crop=2019, scale=9)
    loss = {**current[0], 'Attribute': 'Loss', 'Value': '-0.3'}
    parsed = audit.parse_export(encode([*older, *other_table, *current, loss]), 'synthetic.csv')
    assert parsed['rows_read'] == 31
    record = parsed['records'][0]
    assert record['crop_year'] == 2020
    assert record['values']['world_stock_use'] == .8
    assert record['values']['us_stock_use'] == .2
    assert record['values']['brazil_production'] == 12
    assert record['declared_report_local_time'] == '12:00:00'
    assert record['csv_available_at'] is None and record['historical_model_eligible'] is False


@pytest.mark.parametrize('change', [
    {'Unit': 'Thousand Bales'}, {'ForecastYear': '2022'}, {'ForecastMonth': '2'},
    {'ReportDate': 'February 2021'}, {'ReleaseDate': '12/01/21'}, {'MarketYear': '2020/22'},
    {'Value': 'NA'}, {'Value': 'NaN'}, {'Value': '-1'}, {'ReleaseTime': '25:00:00'},
    {'WasdeNumber': 'next'}, {'ReliabilityProjection': 'above_final'}, {'AnnualQuarterFlag': 'Quarter'},
])
def test_unreviewed_metadata_values_and_qualifiers_stop(change):
    data = rows()
    data[0].update(change)
    with pytest.raises(ValueError):
        audit.parse_export(encode(data), 'synthetic.csv')


@pytest.mark.parametrize('duplicate', [True, False])
def test_duplicate_or_missing_target_cell_cannot_silently_pass(duplicate):
    data = rows()
    data = [*data, data[0]] if duplicate else data[1:]
    with pytest.raises(ValueError, match='Exactly one'):
        audit.parse_export(encode(data), 'synthetic.csv')


@pytest.mark.parametrize('members', [ {'../outside.csv': b'x'}, {'a.csv': b'x', 'b.csv': b'y'}, {'a.exe': b'x'} ])
def test_zip_does_not_extract_unsafe_or_multiple_members(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match='safe CSV'):
        audit.csv_payload(stream.getvalue(), 'bulk.zip')


def test_exact_dates_crop_and_revision_missingness_future_invariance():
    records = audit.parse_export(encode(rows() + rows('2021-02-09', scale=2) + rows('2021-05-12', crop=2021, scale=3)), 'synthetic.csv')['records']
    frame = candidate_frame(records)
    cells = audit.compare(frame, records)
    assert cells.matches.all() and cells.as_reported_value.isna().sum() == 26
    assert frame.us_stock_use_revision.iloc[1] == 0
    with pytest.raises(ValueError, match='Exact report dates'):
        audit.compare(frame.iloc[:-1], records)
    with pytest.raises(ValueError, match='Exact report dates'):
        audit.compare(frame, records + [records[0]])
    bad = frame.copy()
    bad.loc[0, 'crop_year'] = 2019
    with pytest.raises(ValueError, match='Crop-year'):
        audit.compare(bad, records)
    bad = frame.copy()
    bad.loc[0, 'world_production_revision'] = 0
    assert not audit.compare(bad, records).matches.all()
    records[-1]['values']['china_consumption'] = 999
    frame.loc[2, 'china_consumption'] = 999
    pd.testing.assert_frame_equal(cells[cells.report_date < '2021-05-12'].reset_index(drop=True),
        audit.compare(frame, records).query("report_date < '2021-05-12'").reset_index(drop=True))


def package(tmp_path):
    root = tmp_path / 'exports'
    root.mkdir()
    url = 'https://www.usda.gov/sites/default/files/documents/synthetic.csv'
    review = root / 'review.json'
    freeze_record(review, {'links': [url]})
    raw = encode(rows())
    sha = hashlib.sha256(raw).hexdigest()
    source = root / 'raw' / sha / 'source.bin'
    source.parent.mkdir(parents=True)
    source.write_bytes(raw)
    receipt = source.with_name('retrieval.json')
    freeze_record(receipt, {'source_url': url, 'sha256': sha, 'retrieved_at': '2026-10-08T10:00:00Z', 'model_eligible': False})
    freeze_record(root / 'sources.json', {'version': 'wasde-as-reported-exports-v1', 'model_eligible': False,
        'link_review_file': 'review.json', 'link_review_sha256': digest(review),
        'sources': [{'name': 'synthetic.csv', 'url': url, 'file': source.relative_to(root).as_posix(), 'sha256': sha,
            'retrieval_file': receipt.relative_to(root).as_posix(), 'retrieval_sha256': digest(receipt)}]})
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    candidate_frame(audit.parse_export(raw, 'synthetic.csv')['records']).to_csv(candidate / 'regional.csv', index=False)
    freeze_record(candidate / 'candidate-manifest.json', {'version': regional.VERSION, 'model_eligible': False,
        'first_version_verified': False, 'publication_timestamp_verified': False, 'rows': 1,
        'files': {'regional.csv': digest(candidate / 'regional.csv')}})
    freeze_record(candidate / 'complete.json', {'version': regional.VERSION, 'completed': True,
        'files': {n: digest(candidate / n) for n in ['regional.csv', 'candidate-manifest.json']}})
    return root, candidate, source


def test_full_no_fit_audit_rechecks_inputs_and_never_admits(tmp_path):
    exports, candidate, source = package(tmp_path)
    original = {p: digest(p) for base in [exports, candidate] for p in base.rglob('*') if p.is_file()}
    output = tmp_path / 'audit'
    report = audit.analyze(candidate, exports, output)
    assert report['numeric_verified'] and report['base_cells_compared'] == 10
    assert report['missing_cells_preserved'] == 13
    assert report['historical_model_eligible_rows'] == 0 and not report['model_eligible']
    assert read_record(output / 'complete.json')['fits'] == 0
    coverage = pd.read_csv(output / 'coverage.csv')
    assert coverage.csv_available_at.isna().all() and not coverage.historical_model_eligible.any()
    assert {p: digest(p) for p in original} == original
    with pytest.raises(ValueError, match='Immutable'):
        audit.analyze(candidate, exports, output)
    source.write_bytes(b'corrupt CSV')
    with pytest.raises(ValueError, match='checksum'):
        audit.analyze(candidate, exports, tmp_path / 'failed')
    assert not (tmp_path / 'failed' / 'complete.json').exists()
