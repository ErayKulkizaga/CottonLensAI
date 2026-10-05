import hashlib

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.publications import load_package
from cottonlens_ml.sources import wasde_regional as regional


def cells(day='2020-05-12', season='2020/21 Proj.'):
    day = pd.Timestamp(day)
    values = {'world_production': 90., 'world_consumption': 100., 'world_ending_stocks': 40.,
        'us_production': 20., 'us_consumption': 5., 'us_exports': 15., 'us_ending_stocks': 10.,
        'china_consumption': 30., 'india_production': 25., 'brazil_production': 10.}
    return [{'report_month': day.strftime('%B %Y'), 'marketing_year': season,
        'forecast_month': day.strftime('%b'), 'region': region, 'attribute': field,
        'units': regional.UNIT, 'value': str(values[name])}
        for name, (region, field) in regional.FIELDS.items()]


def test_latest_crop_year_current_forecast_and_country_ratio_definition():
    rows = cells() + cells(season='2019/20 Est.')
    rows[-1]['value'] = '9999'
    values, _ = regional.regional_values(rows, '2020-05-12')
    assert values['crop_year'] == 2020 and values['brazil_production'] == 10.
    assert values['world_stock_use'] == .4 and values['us_stock_use'] == .5
    assert values['world_production_use'] == .9
    with pytest.raises(ValueError, match='month'):
        regional.regional_values(rows, '2020-06-12')
    with pytest.raises(ValueError, match='One correctly dimensioned'):
        regional.regional_values(rows + [rows[0]], '2020-05-12')


def test_qualified_missing_is_not_zero_and_wrong_unit_denominator_rejected():
    rows = cells()
    rows[0].update(value=None, qualifier='not_available')
    values, qualifiers = regional.regional_values(rows, '2020-05-12')
    assert np.isnan(values['world_production']) and np.isnan(values['world_production_use'])
    assert qualifiers['world_production'] == 'not_available'
    rows[0]['qualifier'] = 'suppressed-unknown'
    with pytest.raises(ValueError, match='Unknown missing'):
        regional.regional_values(rows, '2020-05-12')
    rows = cells()
    rows[0]['units'] = 'running bales'
    with pytest.raises(ValueError, match='dimensioned'):
        regional.regional_values(rows, '2020-05-12')
    rows = cells()
    rows[1]['value'] = '0'
    with pytest.raises(ValueError, match='denominator'):
        regional.regional_values(rows, '2020-05-12')


def test_revision_does_not_cross_crop_year_or_long_gap_and_future_is_causal():
    records = []
    for day, season in [('2020-03-10', '2019/20 Est.'), ('2020-04-09', '2019/20 Est.'),
                        ('2020-05-12', '2020/21 Proj.'), ('2020-08-12', '2020/21 Proj.')]:
        values, _ = regional.regional_values(cells(day, season), day)
        records.append({'report_date': pd.Timestamp(day), **values})
    frame = pd.DataFrame(records)
    frame.loc[1, 'china_consumption'] += 1
    result = regional.add_revisions(frame)
    assert result.china_consumption_revision.iloc[1] == 1
    assert result.china_consumption_revision.iloc[[0, 2, 3]].isna().all()
    frame.loc[3, 'china_consumption'] = 999
    pd.testing.assert_frame_equal(result.head(3), regional.add_revisions(frame).head(3))


def source_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'raw'
    sources, parsed, proof = [], {}, []
    for index, day in enumerate(['2020-01-10', '2020-01-20', '2020-01-22']):
        xml = f'synthetic-xml-{index}'.encode()
        page = f'synthetic-page-{index}'.encode()
        shas = []
        for raw in [xml, page]:
            sha = hashlib.sha256(raw).hexdigest()
            folder = root / 'wasde' / sha
            folder.mkdir(parents=True)
            (folder / 'source.bin').write_bytes(raw)
            shas.append(sha)
        values = cells(day, '2019/20 Est.')
        if index:
            values[7]['value'] = '31'  # World unchanged, China changed.
        parsed[xml] = values
        url = 'https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/' + day
        sources.append({'release_url': url, 'xml_sha256': shas[0], 'page_sha256': shas[1],
                        'numeric_evidence_file': 'numeric.json'})
        proof.append({'url': url, 'xml_sha256': shas[0], 'numeric_status': 'passed'})
    freeze_record(root / 'numeric.json', {'releases': proof})
    table = tmp_path / 'world.csv'
    table.write_text('synthetic parent table, not measured data\n', encoding='utf-8')
    freeze_record(table.with_suffix('.manifest.json'), {'table_sha256': digest(table),
        'model_eligible': False, 'publication_timestamp_verified': False, 'first_version_verified': False,
        'numeric_evidence_sha256': {'numeric.json': digest(root / 'numeric.json')}, 'source_versions': sources})
    monkeypatch.setattr(regional, 'cotton_rows', lambda raw: parsed[raw])
    return root, table, sources


def test_full_regional_duplicate_policy_immutable_candidate_and_no_admission(tmp_path, monkeypatch):
    root, table, _ = source_fixture(tmp_path, monkeypatch)
    before = digest(table)
    output = tmp_path / 'candidate'
    manifest = regional.compile_candidate(root, table, output)
    result = pd.read_csv(output / 'regional.csv')
    assert result.report_date.tolist() == ['2020-01-10', '2020-01-20']
    assert result.china_consumption_revision.iloc[1] == 1.
    assert manifest['identical_later_same_month_copies'] == ['2020-01-22']
    assert not manifest['model_eligible'] and not manifest['release_allowed']
    assert not (output / 'publication-manifest.json').exists()
    assert read_record(output / 'complete.json')['fits'] == 0 and digest(table) == before
    with pytest.raises(ValueError, match='overwrite'):
        regional.compile_candidate(root, table, output)
    with pytest.raises(FileNotFoundError):
        load_package(output)


def test_corrupt_source_fails_before_creating_candidate(tmp_path, monkeypatch):
    root, table, sources = source_fixture(tmp_path, monkeypatch)
    (root / 'wasde' / sources[0]['xml_sha256'] / 'source.bin').write_bytes(b'bad')
    output = tmp_path / 'new'
    with pytest.raises(ValueError, match='checksum mismatch'):
        regional.compile_candidate(root, table, output)
    assert not output.exists()
