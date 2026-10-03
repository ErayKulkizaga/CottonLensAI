import pytest

import prepare_ams_history as history


def test_year_resume_uses_completed_months_and_rejects_missing_report(tmp_path, monkeypatch):
    calls = {'listings': 0, 'audits': 0}

    def listing(month, root):
        calls['listings'] += 1
        return {'month': month, 'release_count': 1,
                'entries': [{'date': month + '-10', 'release_page_url': 'https://example.test/report'}]}

    def audit(path, **_):
        calls['audits'] += 1
        return {'month': path.stem[-7:], 'release_count': 1,
                'results': [{'reported_spot_bales': 0}]}

    monkeypatch.setattr(history, 'archive_spot_month_listing', listing)
    monkeypatch.setattr(history, 'audit_spot_inventory', audit)
    monkeypatch.setattr(history, 'verified_cached_excerpt', lambda *args: object())
    first = history.prepare_year(2021, tmp_path)
    assert first['report_count'] == 12
    assert first['zero_reported_spot_bales'] == 12
    assert calls == {'listings': 12, 'audits': 12}
    assert history.prepare_year(2021, tmp_path) == first
    assert calls == {'listings': 12, 'audits': 24}

    monkeypatch.setattr(history, 'verified_cached_excerpt', lambda *args: None)
    with pytest.raises(ValueError, match='lost an archived report'):
        history.prepare_year(2021, tmp_path)
