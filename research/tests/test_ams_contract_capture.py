from unittest.mock import patch

import pytest

from research.ams_contract_capture import recover_day
from research.tests.test_ams_contract_reconciliation import report


def test_wrong_release_day_is_rejected_before_fetching_report(tmp_path):
    fetched = []

    def fetch(url):
        fetched.append(url); return b'<html>date</html>'

    with patch('research.ams_contract_capture.release_details', return_value=(
            '2026-10-29T12:00:00Z', 'https://esmis.nal.usda.gov/wrong.txt')), pytest.raises(ValueError, match='Exact dated'):
        recover_day('2020-02-26', tmp_path/'day', fetch)
    assert len(fetched) == 1
    assert (tmp_path/'day/release-page.html').read_bytes() == b'<html>date</html>'


def test_source_bytes_survive_bad_report_and_no_model_is_admitted(tmp_path):
    with patch('research.ams_contract_capture.release_details', return_value=(
            '2020-02-26T12:00:00Z', 'https://esmis.nal.usda.gov/daily.txt')):
        result = recover_day('2020-02-26', tmp_path/'valid', lambda url: report() if url.endswith('.txt') else b'page')
        assert not result['training_allowed'] and not result['historical_availability_verified']
        with pytest.raises(ValueError, match='date differs'):
            recover_day('2020-02-26', tmp_path/'wrong', lambda url: report().replace(b'2/26/2020', b'2/27/2020')
                        if url.endswith('.txt') else b'page')
    assert (tmp_path/'wrong/report.txt').is_file()
