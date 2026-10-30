import json
import os
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from research import ct_forward_daily as runner
from research import ct_manual_forward as forward


def fake_core(stamp='2026-10-31T00:05Z'):
    core = Mock(); identity = {'source_id': 'fixed'}
    core.now.return_value = forward.utc(stamp); core.pd = pd
    core.verify.return_value = ({'identity': identity}, {'fit_budget': {'market': 31}})
    def read(path):
        if path.name == 'authorization.json':
            return {'human_authorization_sha256': 'consent-hash'}
        if path.name == 'human-authorization.json':
            return {'human_user_answer': 'Evet, yalnız günlük ve konsolsuz görev'}
        if path.name == 'cohort-lock.json':
            return {'identity': identity, 'first_origin': '2026-10-30'}
        return {'fixture': True}
    core.read_record.side_effect = read
    core.digest.side_effect = lambda p: 'consent-hash' if p.name == 'human-authorization.json' else 'file-hash'
    core.content_id.side_effect = runner.fingerprint
    core.calendar.return_value = forward.CottonCalendar(json.loads(Path('research/calendars/ice-cotton-2026.json').read_bytes()))
    core.slots.side_effect = forward.slots; core.target_date.side_effect = forward.target_date
    core.decision.side_effect = forward.decision
    core.capture.return_value = {'latest_completed_date': '2026-10-30', 'observation_id': 'capture'}
    core.fit_for_origin.return_value = {'status': 'state_ready_no_forward_forecast', 'market_fits': 10}
    core.publish.return_value = {'origin': '2026-10-30', 'status': 'TIMELY'}
    return core


@pytest.mark.parametrize(('stamp', 'phase'), [('00:00', 'capture'), ('00:14:59', 'capture'),
    ('00:15', 'publish'), ('00:29:59', 'publish'), ('00:30', 'outside_window'), ('14:00', 'outside_window')])
def test_exact_utc_windows(stamp, phase):
    assert runner.phase_at(forward.utc('2026-10-31T'+stamp+'Z')) == phase


def test_daytime_on_demand_task_test_performs_no_capture_fit_or_publication(tmp_path):
    core = fake_core('2026-10-30T14:00Z')
    result = runner.operate(core, tmp_path, tmp_path, tmp_path/'authorization.json')
    assert result['status'] == 'SKIPPED_OUTSIDE_WINDOW'
    core.capture.assert_not_called(); core.fit_for_origin.assert_not_called(); core.publish.assert_not_called()
    core.authorize.assert_called_once()


def test_capture_and_refit_are_bounded_and_permission_does_not_leak(tmp_path, monkeypatch):
    monkeypatch.delenv('COTTONLENS_ALLOW_LOCAL_CPU_TABULAR', raising=False)
    core = fake_core(); result = runner.operate(core, tmp_path, tmp_path, tmp_path/'authorization.json')
    assert result['status'] == 'CAPTURED_AND_STATE_READY' and result['new_forecasts'] == 0
    core.capture.assert_called_once()
    assert core.fit_for_origin.call_args.kwargs['minutes'] == 10.
    core.publish.assert_not_called()
    assert 'COTTONLENS_ALLOW_LOCAL_CPU_TABULAR' not in os.environ


def test_publish_has_no_fit_and_returns_actual_record_link(tmp_path):
    core = fake_core('2026-10-31T00:20Z')
    result = runner.operate(core, tmp_path, tmp_path, tmp_path/'authorization.json')
    assert result['status'] == 'TIMELY' and result['forward_record_id']
    assert result['new_fits'] == 0
    core.capture.assert_not_called(); core.fit_for_origin.assert_not_called(); core.publish.assert_called_once()


def test_closed_cotton_session_is_skipped_without_http_or_fit(tmp_path):
    core = fake_core('2026-11-27T00:05Z')  # Thanksgiving origin is closed.
    result = runner.operate(core, tmp_path, tmp_path, tmp_path/'authorization.json')
    assert result['status'] == 'SKIPPED_NON_COHORT_SESSION'
    core.capture.assert_not_called(); core.fit_for_origin.assert_not_called()


@pytest.mark.parametrize('stamp', ['2027-01-01T00:05Z', '2027-01-05T00:05Z'])
def test_unknown_target_or_calendar_fails_before_http_or_fit(tmp_path, stamp):
    core = fake_core(stamp)
    with pytest.raises(ValueError, match='calendar|Calendar'):
        runner.operate(core, tmp_path, tmp_path, tmp_path/'authorization.json')
    core.capture.assert_not_called(); core.fit_for_origin.assert_not_called()


def test_capture_that_crosses_cutoff_cannot_start_a_late_fit(tmp_path):
    core = fake_core(); core.now.side_effect = [forward.utc('2026-10-31T00:05Z'), forward.utc('2026-10-31T00:16Z')]
    with pytest.raises(ValueError, match='Capture missed'):
        runner.operate(core, tmp_path, tmp_path, tmp_path/'authorization.json')
    core.fit_for_origin.assert_not_called(); core.publish.assert_not_called()


def test_frozen_execution_manifest_rejects_modified_runner(tmp_path):
    p = tmp_path/'ct_forward_daily.py'; p.write_bytes(b'synthetic runner')
    body = {'files': {p.name: runner.sha(p)}, 'core_source_id': 'core'}
    expected = runner.fingerprint(body)
    (tmp_path/'execution-manifest.json').write_text(json.dumps({**body, 'execution_source_id': expected}))
    assert runner.verify_execution(tmp_path, expected)['core_source_id'] == 'core'
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='runner changed'):
        runner.verify_execution(tmp_path, expected)


def test_installer_uses_pythonw_two_daily_triggers_and_no_hourly_or_logon_trigger():
    text = Path('research/scripts/install-risk-daily.ps1').read_text()
    assert text.count('<CalendarTrigger id=') == 2
    assert '<DaysInterval>1</DaysInterval>' in text
    assert '<Repetition>' not in text and '<LogonTrigger>' not in text
    assert '<StartWhenAvailable>false' in text and '<WakeToRun>false' in text
    assert '<Hidden>false</Hidden>' in text  # Inspectable in Task Scheduler; pythonw suppresses consoles.
    assert "-ne 'pythonw.exe'" in text
    assert 'powershell.exe' not in text
    assert "'CottonLens-Observed-Archive-v2'" in text


@pytest.mark.parametrize('name', ['injected.py', 'injected.DLL', 'injected.cmd'])
def test_runtime_rejects_additional_executable_code(tmp_path, name):
    p = tmp_path/'ct_forward_daily.py'; p.write_bytes(b'synthetic runner')
    body = {'files': {p.name: runner.sha(p)}, 'core_source_id': 'core'}
    expected = runner.fingerprint(body)
    (tmp_path/'execution-manifest.json').write_text(json.dumps({**body, 'execution_source_id': expected}))
    (tmp_path/name).write_bytes(b'not executable: fixture')
    with pytest.raises(ValueError, match='Unmanifested'):
        runner.verify_execution(tmp_path, expected)


@pytest.mark.parametrize('failure', ['changed_code', 'extra_code', 'manifest'])
def test_core_is_verified_before_import(tmp_path, failure):
    study = tmp_path/'study'; study.mkdir(); source = study/'source-snapshot'; source.mkdir()
    p = source/'model.py'; p.write_bytes(b'not imported: fixture')
    body = {'ml': {'files': {p.name: runner.sha(p)}}, 'helpers': {}}
    identity = runner.fingerprint(body); body['source_id'] = identity
    manifest = {**body, 'record_id': runner.fingerprint(body)}
    (study/'source-manifest.json').write_text(json.dumps(manifest))
    runner.verify_core_before_import(study, source, identity)
    if failure == 'changed_code':
        p.write_bytes(b'changed')
    elif failure == 'extra_code':
        (source/'injected.DLL').write_bytes(b'not executable: fixture')
    else:
        manifest['source_id'] = 'changed'
        (study/'source-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Core|core'):
        runner.verify_core_before_import(study, source, identity)
