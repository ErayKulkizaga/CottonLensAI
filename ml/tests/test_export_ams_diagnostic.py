import json

import pytest

import export_ams_diagnostic as diagnostic
from cottonlens_ml.code_identity import digest


def test_identical_url_aliases_collapse_but_revisions_fail():
    row = {'report_date': '2019-01-24', 'release_page_url': 'https://source/first',
           'report_sha256': 'same', 'spot_41_4_34_cents_per_lb': 68.89,
           'spot_31_3_35_cents_per_lb': 72.0, 'reported_spot_bales': 10223}
    kept, aliases = diagnostic.collapse_identical_report_aliases([
        {**row, 'release_page_url': 'https://source/second'}, row])
    assert kept == [row] and aliases == 1
    with pytest.raises(ValueError, match='Conflicting same-date'):
        diagnostic.collapse_identical_report_aliases([
            row, {**row, 'report_sha256': 'changed'}])


def test_export_checks_archived_inputs_and_preserves_frozen_table(tmp_path, monkeypatch):
    root, output = tmp_path / 'archive', tmp_path / 'spot.csv'
    root.mkdir()
    months = []
    for number in range(1, 13):
        month = f'2021-{number:02d}'
        day = month + '-10'
        url = 'https://esmis.nal.usda.gov/publication/daily-spot-quotations-excerpts/' + day
        listing_path = root / f'ams-month-listing-{month}.json'
        listing_path.write_text(json.dumps({'month': month, 'entries': [
            {'date': day, 'release_page_url': url}]}), encoding='utf-8')
        audit_path = root / f'ams-month-audit-{month}.json'
        audit_path.write_text(json.dumps({'month': month, 'inventory_sha256': digest(listing_path),
            'release_count': 1, 'results': [{'date': day, 'release_page_url': url,
                'evidence_path': str(root / 'cached'), 'report_sha256': 'abc',
                'reported_spot_bales': 0, 'spot_41_4_34_cents_per_lb': 60.0}]}),
            encoding='utf-8')
        months.append({'month': month, 'audit_sha256': digest(audit_path)})
    (root / 'year-content-audit-2021.json').write_text(
        json.dumps({'year': 2021, 'months': months}), encoding='utf-8')
    monkeypatch.setattr(diagnostic, 'verified_cached_excerpt', lambda *args: (
        root / 'cached', {'report_sha256': 'abc', 'reported_spot_bales': 0,
                          'seven_market_average_41_4_34_cents_per_lb': 60.0,
                          'seven_market_average_31_3_35_cents_per_lb': 55.0}))

    result = diagnostic.export([root], output)
    assert result['row_count'] == 12
    assert result['model_eligible'] is False
    assert diagnostic.export([root], output) == result

    audit = root / 'ams-month-audit-2021-01.json'
    audit.write_text(audit.read_text(encoding='utf-8').replace('60.0', '70.0'), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum mismatch'):
        diagnostic.export([root], output)
