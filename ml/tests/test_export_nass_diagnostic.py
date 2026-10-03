import pytest

import export_nass_diagnostic as diagnostic
from cottonlens_ml.sprint import freeze_record


def test_export_uses_checked_report_and_freezes_table(tmp_path, monkeypatch):
    root, output = tmp_path / 'raw', tmp_path / 'nass.csv'
    folder = root / 'report'
    folder.mkdir(parents=True)
    (folder / 'report.txt').write_text(
        'Cotton Condition - Selected States: Week Ending June 7, 2015\n'
        'State : Very poor : Poor : Fair : Good : Excellent\n'
        '15 States .......: 0 7 43 44 6\n', encoding='utf-8')
    url = 'https://esmis.nal.usda.gov/publication/crop-progress/2015-06-08'
    audit = {'year': 2015, 'weeks': 1, 'results': [
        {'week_ending': '2015-06-07', 'status': 'reference_missing_zero',
         'release_page': url, 'report_sha256': 'abc'}]}
    freeze_record(root / 'year-audit-2015.json', audit)
    monkeypatch.setattr(diagnostic, 'verified_cached_release',
                        lambda *args: (folder, {'report_sha256': 'abc'}))
    result = diagnostic.export(root, output, years=[2015])
    assert result['row_count'] == 1 and result['model_eligible'] is False
    assert 'reference_missing_zero' in output.read_text(encoding='utf-8')
    assert diagnostic.export(root, output, years=[2015]) == result
    (folder / 'report.txt').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Cotton Condition'):
        diagnostic.export(root, output, years=[2015])


def test_export_rejects_failed_week(tmp_path):
    freeze_record(tmp_path / 'year-audit-2015.json', {'year': 2015, 'weeks': 1,
                  'results': [{'status': 'numeric_mismatch'}]})
    with pytest.raises(ValueError, match='incomplete'):
        diagnostic.export(tmp_path, tmp_path / 'nass.csv', years=[2015])
