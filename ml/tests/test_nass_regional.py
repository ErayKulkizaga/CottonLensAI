import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from cottonlens_ml.code_identity import digest, manifest_id
from cottonlens_ml.sources import nass_regional as regional


def report(*, legacy=False, current='50', previous='35', average='54'):
    if legacy:
        progress = f'''       Cotton:  Percent Planted,
          Selected States 1/
--------------------------------------
      :      Week Ending      :
      :-----------------------: 2018-
 State:June 4,:May 28,:June 4,: 2022
      : 2023  : 2023  : 2022  : Avg.
--------------------------------------
TX    : {current} {previous} 58 {average}
15 Sts: 60 45 66 62
--------------------------------------'''
        condition = '''  Cotton:  Crop Condition by Percent,
           Selected States,
       Week Ending June 4, 2023
--------------------------------------
  State : VP  :  P  :  F  :  G  : EX
--------------------------------------
TX      : 10 20 30 30 10
15 Sts  : 0 10 40 40 10'''
    else:
        progress = f'''Cotton Planted - Selected States
[These 15 States planted 99% of the 2022 cotton acreage]
-----------------------------------------------------------------
                 :            Week ending            :
                 :-----------------------------------:
      State      : June 4,  :  May 28,  :  June 4,  : 2018-2022
                 :   2022    :   2023    :   2023    :  Average
-----------------------------------------------------------------
Texas ...........: 58 {previous} {current} {average}
15 States .......: 66 45 60 62
-----------------------------------------------------------------'''
        condition = '''Cotton Condition - Selected States: Week Ending June 4, 2023
----------------------------------------------------------------------------
      State     : Very poor : Poor : Fair : Good : Excellent
----------------------------------------------------------------------------
Texas ..........: 10 20 30 30 10
15 States ......: - 10 40 40 10'''
    return ('Crop Progress\nReleased June 5, 2023, by the National Agricultural Statistics Service\n'
            + progress + '\n' + condition
            + '\n- Represents zero.\n(NA) Not available.\n* Revised.\n'
            + 'Corn Planted - Selected States\nTexas ..........: 99 99 99 99\n')


@pytest.mark.parametrize('legacy', [False, True])
def test_header_dates_determine_current_column_and_published_average(legacy):
    result = regional.parse_report(report(legacy=legacy), release_day='2023-06-05')
    planted = result['progress']['Planted']
    assert planted['cells']['current']['value'] == 50
    assert planted['cells']['previous_week']['value'] == 35
    assert planted['cells']['previous_year']['value'] == 58
    assert planted['published_average_years'] == [2018, 2022]
    assert planted['gap_to_average_pp'] == -4
    assert result['texas_condition']['GOOD']['value'] == 30
    assert result['national_condition']['GOOD'] == 40
    assert result['progress']['Harvested']['section_present'] is False
    assert result['progress']['Harvested']['cells'] is None


@pytest.mark.parametrize('token,value,qualifier', [('0', 0, None), ('-', 0, '-'),
    ('(NA)', None, '(NA)'), ('NA', None, 'NA'), ('(D)', None, '(D)'), ('*83', 83, '*')])
def test_qualifiers_do_not_become_unpublished_zero(token, value, qualifier):
    result = regional.parse_report(report(current=token), release_day='2023-06-05')
    assert result['progress']['Planted']['cells']['current'] == {'value': value, 'qualifier': qualifier}
    if value is None:
        assert result['progress']['Planted']['gap_to_average_pp'] is None


@pytest.mark.parametrize('token', ['101', '-1', '1.5', 'True', '(Z)', '*-'])
def test_unknown_or_invalid_percentage_is_rejected(token):
    with pytest.raises(ValueError, match='percentage/qualifier'):
        regional.parse_report(report(current=token), release_day='2023-06-05')


@pytest.mark.parametrize('token,note', [('-', '- Represents zero.'), ('*83', '* Revised.')])
def test_numeric_symbol_requires_published_definition(token, note):
    with pytest.raises(ValueError, match='percentage/qualifier'):
        regional.parse_report(report(current=token).replace(note, ''), release_day='2023-06-05')


@pytest.mark.parametrize('old,new,match', [
    ('2018-2022', '2019-2023', 'preceding five years'),
    ('June 4,  : 2018', 'June 5,  : 2018', 'column dates'),
    ('May 28,  :', 'May 29,  :', 'column dates'),
    ('Texas ...........:', 'Arizona .........:', 'Texas row'),
    ('10 20 30 30 10', '10 20 30 30 11', 'sum to 100'),
    ('15 States .......:', '14 States .......:', 'national row'),
])
def test_changed_semantics_fail_closed(old, new, match):
    with pytest.raises(ValueError, match=match):
        regional.parse_report(report().replace(old, new), release_day='2023-06-05')


def test_duplicate_section_and_wrong_release_date_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        regional.parse_report(report() + '\nCotton Planted - Selected States', release_day='2023-06-05')
    with pytest.raises(ValueError, match='release date'):
        regional.parse_report(report(), release_day='2023-06-06')


def freeze(path, body):
    path.write_text(json.dumps({**body, 'record_id': manifest_id(body)}), encoding='utf-8')


def fixture(tmp_path):
    audits = tmp_path / 'audits'
    audits.mkdir()
    text = report()
    import hashlib
    sha = hashlib.sha256(text.encode()).hexdigest()
    folder = audits / 'nass_crop_progress/2023-06-05' / sha
    folder.mkdir(parents=True)
    (folder / 'report.txt').write_bytes(text.encode())
    page_url = 'https://esmis.nal.usda.gov/publication/crop-progress/2023-06-05'
    report_url = 'https://esmis.nal.usda.gov/sites/default/release-files/a/b/report.txt'
    page = '<time datetime="2023-06-05T12:00:00Z"></time><a href="' + report_url + '">TXT</a>'
    (folder / 'release-page.html').write_text(page, encoding='utf-8')
    receipt = {'release_page_url': page_url, 'report_url': report_url,
               'release_page_sha256': digest(folder / 'release-page.html'), 'report_sha256': sha,
               'release_page_date_field': '2023-06-05T12:00:00Z', 'retrieved_at': '2026-10-01T10:00:00+00:00',
               'model_eligible': False, 'publication_clock_verified': False}
    (folder / 'retrieval.json').write_text(json.dumps(receipt), encoding='utf-8')
    item = {'week_ending': '2023-06-04', 'status': 'numeric_match',
            'report_sha256': sha, 'release_page': page_url}
    annual = {'year': 2023, 'weeks': 1, 'results': [item]}
    freeze(audits / 'year-audit-2023.json', annual)
    row = {'week_ending': '2023-06-04', 'release_page_date': '2023-06-05',
           'very_poor_pct': 0, 'poor_pct': 10, 'fair_pct': 40, 'good_pct': 40, 'excellent_pct': 10,
           'quickstats_check': 'numeric_match', 'release_page_url': page_url, 'report_sha256': sha}
    table = tmp_path / 'condition.csv'
    with table.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    manifest = {'row_count': 1, 'table_sha256': digest(table), 'model_eligible': False,
                'publication_clock_verified': False,
                'source_year_audit_sha256': {'2023': digest(audits / 'year-audit-2023.json')}}
    freeze(table.with_suffix('.manifest.json'), manifest)
    return table, audits, folder


def test_audit_reads_only_pinned_inputs_with_no_network_or_admission(tmp_path, monkeypatch):
    table, audits, _ = fixture(tmp_path)
    before = {p: digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail('offline audit attempted a provider request')
    monkeypatch.setattr(requests.Session, 'get', forbidden)
    result = regional.audit(table, audits)
    assert regional.audit(table, audits) == result
    assert result['reports'] == 1 and len(result['input_sha256']) == 6
    assert result['coverage']['Planted']['present_reports'] == 1
    assert result['coverage']['Harvested']['absent_reports'] == 1
    assert result['market_fits'] == 0 and result['model_eligible'] is False
    assert result['release_allowed'] is False and result['historical_publication_verified'] is False
    assert result['first_version_verified'] is False and result['availability_policy'] == 'UNSET'
    assert result['panel'][0]['available_at'] is None
    # Noon in the release page and 2026 retrieval are not historical availability.
    assert '2026' in result['panel'][0]['retrieved_at']
    assert all(not key.startswith(str(tmp_path)) for key in result['input_sha256'])
    assert before == {p: digest(p) for p in before}
    (audits / 'unselected-future.txt').write_text('future', encoding='utf-8')
    assert regional.audit(table, audits) == result


@pytest.mark.parametrize('name', ['report.txt', 'retrieval.json', 'release-page.html', 'annual', 'table', 'manifest'])
def test_corrupt_or_missing_input_is_not_accepted(tmp_path, name):
    table, audits, folder = fixture(tmp_path)
    path = {'annual': audits / 'year-audit-2023.json', 'table': table,
            'manifest': table.with_suffix('.manifest.json')}.get(name, folder / name)
    path.write_text('corrupt', encoding='utf-8')
    with pytest.raises((ValueError, KeyError)):
        regional.audit(table, audits)


def test_changed_source_during_audit_is_rejected(tmp_path, monkeypatch):
    table, audits, _ = fixture(tmp_path)
    original = regional.digest
    count = 0
    def changing(path):
        nonlocal count
        if path == table:
            count += 1
            if count > 1:
                return '0' * 64
        return original(path)
    monkeypatch.setattr(regional, 'digest', changing)
    with pytest.raises(ValueError, match='changed during audit'):
        regional.audit(table, audits)


def test_later_previous_week_revision_never_overwrites_original_report(tmp_path):
    table, audits, folder = fixture(tmp_path)
    before = regional.audit(table, audits)['panel'][0]
    text = (report(current='60', previous='*52').replace('June 4', 'June 11')
            .replace('May 28', 'June 4').replace('June 5, 2023', 'June 12, 2023'))
    import hashlib
    sha = hashlib.sha256(text.encode()).hexdigest()
    next_folder = audits / 'nass_crop_progress/2023-06-12' / sha
    next_folder.mkdir(parents=True)
    (next_folder / 'report.txt').write_bytes(text.encode())
    page = (folder / 'release-page.html').read_text().replace('2023-06-05', '2023-06-12')
    (next_folder / 'release-page.html').write_text(page, encoding='utf-8')
    receipt = json.loads((folder / 'retrieval.json').read_bytes())
    receipt.update(release_page_url=receipt['release_page_url'].replace('2023-06-05', '2023-06-12'),
                   release_page_date_field='2023-06-12T12:00:00Z', report_sha256=sha,
                   release_page_sha256=digest(next_folder / 'release-page.html'))
    (next_folder / 'retrieval.json').write_text(json.dumps(receipt), encoding='utf-8')
    annual_path = audits / 'year-audit-2023.json'
    annual = regional.read_frozen(annual_path)
    annual['weeks'] = 2
    annual['results'].append({**annual['results'][0], 'week_ending': '2023-06-11',
                              'release_page': receipt['release_page_url'], 'report_sha256': sha})
    freeze(annual_path, annual)
    with table.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    rows.append({**rows[0], 'week_ending': '2023-06-11', 'release_page_date': '2023-06-12',
                 'release_page_url': receipt['release_page_url'], 'report_sha256': sha})
    with table.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = table.with_suffix('.manifest.json')
    manifest = regional.read_frozen(manifest_path)
    manifest.update(row_count=2, table_sha256=digest(table),
                    source_year_audit_sha256={'2023': digest(annual_path)})
    freeze(manifest_path, manifest)
    result = regional.audit(table, audits)
    assert result['panel'][0] == before
    assert result['previous_week_revisions'] == [{'week_ending': '2023-06-11', 'stage': 'Planted',
                                                 'original_previous': 50, 'later_previous': 52}]
    assert result['panel'][1]['progress']['Planted']['cells']['previous_week']['qualifier'] == '*'


def test_cli_failure_creates_no_output_files(tmp_path):
    table, audits, folder = fixture(tmp_path)
    (folder / 'report.txt').unlink()
    before = set(tmp_path.rglob('*'))
    script = Path(__file__).resolve().parents[1] / 'review_nass_regions.py'
    run = subprocess.run([sys.executable, str(script), '--table', str(table), '--audits', str(audits)],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 2 and not run.stdout
    assert set(tmp_path.rglob('*')) == before
