import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record

from research import ct_manual_forward as study


def seed(n=1300):
    dates = pd.bdate_range('2020-01-02', periods=n)
    rng = np.random.default_rng(42)
    return pd.DataFrame({'date': dates, 'cotton_close': 100*np.exp(np.cumsum(rng.normal(0, .01, n)))})


def calendar():
    return study.CottonCalendar(json.loads(Path('research/calendars/ice-cotton-2026.json').read_bytes()))


def test_maturity_is_strict_five_cotton_observations_and_original_warmup():
    h = study.history(seed()); cut = h.date.iloc[1100]
    dates = study.mature(h, cut)
    assert dates[0] == str(h.date.iloc[119].date())
    assert dates[-1] == str(h.date.iloc[1094].date())
    assert str(h.date.iloc[1095].date()) not in dates
    assert len(dates) == 1100-6-119+1


def test_annual_three_blocks_and_twenty_one_observation_refits_use_only_past():
    h = study.history(seed()); origin = str(h.date.iloc[-1].date())
    jobs = study.annual_jobs(h, origin)
    assert len(jobs) == 9 and all(len(j['origins']) == 21 for j in jobs)
    assert [j['role'] for j in jobs] == ['inner_0']*3+['inner_1']*3+['inner_2']*3
    assert len({d for j in jobs for d in j['origins']}) == 189
    for job in jobs:
        selected = h.set_index(h.date.dt.strftime('%Y-%m-%d')).loc[job['train_dates']]
        assert selected.target_date_5.lt(pd.Timestamp(job['cutoff'])).all()
        assert job['train_dates'] == study.mature(h, job['cutoff'])


def test_pretraining_before_origin_close_has_identical_mature_training():
    p = seed(); full = study.history(p); partial = study.history(p.iloc[:-1])
    origin = str(p.date.iloc[-1].date())
    full_dates, partial_dates = study.mature(full, origin), study.mature(partial, origin)
    assert full_dates == partial_dates
    pd.testing.assert_frame_equal(study.projection(full, full_dates), study.projection(partial, partial_dates), check_exact=True)
    assert study.annual_jobs(full, origin) == study.annual_jobs(partial, origin)


def test_future_prices_and_targets_cannot_change_past_training():
    p = seed(); altered = p.copy(); altered.loc[1100:, 'cotton_close'] *= 100
    a, b = study.history(p), study.history(altered); cut = str(p.date.iloc[1100].date())
    dates = study.mature(a, cut)
    pd.testing.assert_frame_equal(study.projection(a, dates), study.projection(b, dates), check_exact=True)
    rows = study.projection(a, [cut]); rows['variance_target_1'] = np.nan
    assert rows[study.NAMES].notna().all().all()


def test_first_origin_uses_real_lock_not_already_started_publication():
    cal = calendar()
    assert study.first_origin(cal, study.utc('2026-10-30T10:00Z')) == '2026-10-30'
    assert study.first_origin(cal, study.utc('2026-10-31T00:14:59Z')) == '2026-10-30'
    assert study.first_origin(cal, study.utc('2026-10-31T00:15:00Z')) == '2026-11-02'
    assert study.decision('2026-11-02').isoformat() == '2026-11-03T00:15:00+00:00'


def test_missing_slots_are_not_removed_from_twenty_one_bucket_counter():
    dates = study.slots(calendar(), '2026-10-30')
    recorded = [d for i, d in enumerate(dates) if i not in (2, 3, 4, 5)]
    origin = dates[21]
    assert dates.index(origin)//21 == 1
    assert recorded.index(origin)//21 == 0
    assert len(dates) < 252  # Explicit calendar extension is still required.


def test_incomplete_source_capture_is_not_silently_ignored(tmp_path):
    (tmp_path/'observations/uncompleted').mkdir(parents=True)
    with pytest.raises(ValueError, match='Incomplete observation'):
        study.current_history(tmp_path)


def test_incomplete_calendar_extension_is_not_silently_ignored(tmp_path):
    (tmp_path/'inputs').mkdir()
    (tmp_path/'inputs/calendar.json').write_bytes(Path('research/calendars/ice-cotton-2026.json').read_bytes())
    (tmp_path/'calendar-extensions/2028-01-01').mkdir(parents=True)
    with pytest.raises(ValueError, match='Incomplete calendar extension'):
        study.calendar_body(tmp_path)


def test_native_checkpoint_predictions_ignore_target_columns(tmp_path):
    h = study.history(seed()); rows = study.projection(h, h.date.iloc[-10:].dt.strftime('%Y-%m-%d').tolist())
    m = {'features': study.NAMES, 'recipe': study.learner.recipe('own_level', study.learner.STATE_PROFILE),
         'coef': [.1, .2, .1, -.1, -.15]}
    (tmp_path/'payload').mkdir(); (tmp_path/'payload/variance-model.json').write_text(json.dumps(m))
    ledger = Ledger(tmp_path, {}); record = {'payload_root': 'payload'}
    a = study.predict_model(ledger, record, rows)
    rows['variance_target_1'] = float('inf')
    np.testing.assert_array_equal(a, study.predict_model(ledger, record, rows))


def test_auth_rejects_new_data_design_source_or_unrestored_evidence(tmp_path):
    freeze_record(tmp_path/'ready.json', {'identity': {'source_id': 'source', 'design_id': 'design'}})
    ready = {'identity': {'source_id': 'source', 'design_id': 'design'}}; design = {'fit_budget': {'market': 31}}
    for changed in ('ready_sha256', 'source_id', 'design_id', 'fresh_restore_verified', 'fit_budget'):
        body = {'ready_sha256': study.digest(tmp_path/'ready.json'), 'source_id': 'source', 'design_id': 'design',
                'fresh_restore_verified': True, 'fit_budget': design['fit_budget']}
        body[changed] = None
        p = tmp_path/(changed+'.json'); freeze_record(p, body)
        with pytest.raises(ValueError, match='authorization'):
            study.authorize(tmp_path, ready, design, p)


@pytest.mark.parametrize('stamp', ['2026-10-31T00:14:59Z', '2026-10-31T00:30Z', '2026-11-01T00:16Z'])
def test_publish_rejects_early_late_and_retroactive_windows_before_model_access(tmp_path, stamp):
    lock = {'first_origin': '2026-10-30', 'locked_at': '2026-10-30T10:00Z'}
    freeze_record(tmp_path/'cohort-lock.json', lock)
    observation = {'latest_completed_date': '2026-10-30', 'observed_available_at': '2026-10-31T00:05Z'}
    with (patch.object(study, 'verify', return_value=({'identity': {}}, {})),
          patch.object(study, 'current_history', return_value=(seed(), observation)),
          patch.object(study, 'now', return_value=study.utc(stamp)),
          pytest.raises(ValueError, match='No late source')):
        study.publish(tmp_path, tmp_path)


def test_publish_rejects_postcutoff_source_at_valid_publication_time(tmp_path):
    freeze_record(tmp_path/'cohort-lock.json', {'first_origin': '2026-10-30', 'locked_at': '2026-10-30T10:00Z'})
    observation = {'latest_completed_date': '2026-10-30', 'observed_available_at': '2026-10-31T00:15:01Z'}
    with (patch.object(study, 'verify', return_value=({'identity': {}}, {})),
          patch.object(study, 'current_history', return_value=(seed(), observation)),
          patch.object(study, 'now', return_value=study.utc('2026-10-31T00:20Z')),
          pytest.raises(ValueError, match='No late source')):
        study.publish(tmp_path, tmp_path)


@pytest.mark.parametrize('change', ['past_holiday', 'coverage_start', 'close_time', 'timezone', 'late_review'])
def test_calendar_cannot_change_past_slots_or_be_rewritten_after_new_period(change):
    old = json.loads(Path('research/calendars/ice-cotton-2026.json').read_bytes())
    new = copy.deepcopy(old); new['coverage_end'] = '2028-01-01'
    reviewed = study.utc('2026-12-01T12:00Z')
    study.check_calendar_extension(old, new, reviewed)
    if change == 'past_holiday':
        new['cotton_closed_dates'].append('2026-11-11')
    elif change == 'coverage_start':
        new['coverage_start'] = '2025-01-01'
    elif change == 'close_time':
        new['completion_time_local'] = '14:00'
    elif change == 'timezone':
        new['timezone'] = 'UTC'
    else:
        reviewed = study.utc('2027-01-02T00:00Z')
    with pytest.raises(ValueError):
        study.check_calendar_extension(old, new, reviewed)


def test_two_year_annual_selection_budget_is31_not21():
    import math
    bounds = [18+math.ceil(n/21)+math.ceil((252-n)/21) for n in range(1, 252)]
    assert max(bounds) == 31


def publishing_fixture(root, first='2026-10-30'):
    p = seed(); p['date'] = pd.bdate_range(end='2026-10-30', periods=len(p))
    h = study.history(p); origin = '2026-10-30'
    identity = {'source_id': 'synthetic', 'design_id': 'synthetic'}
    locked = {'first_origin': first, 'locked_at': '2026-10-29T10:00Z', 'identity': identity}
    freeze_record(root/'cohort-lock.json', locked)
    (root/'inputs').mkdir()
    (root/'inputs/calendar.json').write_bytes(Path('research/calendars/ice-cotton-2026.json').read_bytes())
    job = {'year': 2026, 'role': 'outer', 'bucket': 0, 'cutoff': first, 'origins': [],
           'train_dates': study.mature(h, first)}
    spec = study.specification(job, study.projection(h, job['train_dates']), 'capture')
    record = {'specification': {**spec, 'repeat_reason': None}, 'experiment_id': 'fixture',
              'completed_at': '2026-10-30T12:00Z', 'payload_root': 'payload'}
    freeze_record(root/'ledger/completed/fixture.json', record)
    freeze_record(root/'fits/2026-outer-0.json', {'fit_id': 'fixture', 'specification_id': study.content_id(spec)})
    (root/'ledger/payload').mkdir()
    (root/'ledger/payload/variance-model.json').write_text(json.dumps({
        'features': study.NAMES, 'recipe': study.learner.recipe('own_level', study.learner.STATE_PROFILE),
        'coef': [.1, .2, .1, -.1, -.15]}))
    observation = {'latest_completed_date': origin, 'observed_available_at': '2026-10-31T00:05Z', 'observation_id': 'capture'}
    selection = {'weight': .5, 'created_at': '2026-10-30T12:00Z'}
    return p, identity, record, observation, selection


@pytest.mark.parametrize('first', ['2026-10-29', '2026-10-30'])
def test_manual_publication_records_real_links_and_preserves_missing_slots(tmp_path, first):
    p, identity, record, observation, selection = publishing_fixture(tmp_path, first)
    with (patch.object(study, 'verify', return_value=({'identity': identity}, {'required_origins': 252})),
          patch.object(study, 'current_history', return_value=(p, observation)),
          patch.object(study, 'native_record', return_value=record),
          patch.object(study, 'verify_selection', return_value=selection),
          patch.object(study, 'now', return_value=study.utc('2026-10-31T00:20Z'))):
        result = study.publish(tmp_path, tmp_path)
        assert result['status'] == 'TIMELY'
        report = study.status(tmp_path, tmp_path)
        assert report['timely_forecasts'] == 1 and not report['performance_scored']
        expected_missing = ['2026-10-29'] if first == '2026-10-29' else []
        assert report['elapsed_missing_origins'] == expected_missing
        with pytest.raises(ValueError, match='Preserve existing'):
            study.publish(tmp_path, tmp_path)
    event = read_record(tmp_path/'events/000.json')
    assert event['kind'] == ('MISSING' if first == '2026-10-29' else 'FORECAST')
    if first == '2026-10-29':
        assert read_record(tmp_path/'events/001.json')['previous_event_id'] == study.content_id(event)
    forecast = read_record(tmp_path/'forecasts/2026-10-30/forecast.json')
    assert forecast['observation_id'] == 'capture' and forecast['fit_id'] == 'fixture'
    assert forecast['selected_variance'] == .5*(forecast['raw_variance']+forecast['baseline_variance'])


def test_serialization_that_crosses_publication_deadline_is_late(tmp_path):
    p, identity, record, observation, selection = publishing_fixture(tmp_path)
    times = [study.utc(v) for v in ['2026-10-31T00:29:59Z', '2026-10-31T00:30Z', '2026-10-31T00:30Z']]
    with (patch.object(study, 'verify', return_value=({'identity': identity}, {'required_origins': 252})),
          patch.object(study, 'current_history', return_value=(p, observation)),
          patch.object(study, 'native_record', return_value=record),
          patch.object(study, 'verify_selection', return_value=selection),
          patch.object(study, 'now', side_effect=times)):
        assert study.publish(tmp_path, tmp_path)['status'] == 'LATE'


def test_published_payload_tamper_is_rejected(tmp_path):
    (tmp_path/'inputs').mkdir()
    (tmp_path/'inputs/calendar.json').write_bytes(Path('research/calendars/ice-cotton-2026.json').read_bytes())
    freeze_record(tmp_path/'cohort-lock.json', {'first_origin': '2026-10-30', 'locked_at': '2026-10-29T10:00Z'})
    freeze_record(tmp_path/'events/000.json', {'ordinal_slot': 0, 'origin': '2026-10-30',
        'previous_event_id': None, 'kind': 'FORECAST', 'forecast_sha256': 'wrong', 'publication_receipt_sha256': 'wrong'})
    freeze_record(tmp_path/'forecasts/2026-10-30/forecast.json', {'origin': '2026-10-30'})
    freeze_record(tmp_path/'forecasts/2026-10-30/publication-receipt.json', {'status': 'TIMELY'})
    with (patch.object(study, 'verify', return_value=({}, {'required_origins': 252})),
          pytest.raises(ValueError, match='payload changed')):
        study.status(tmp_path, tmp_path)


def test_active_writer_and_unsealed_publication_never_count_as_complete(tmp_path):
    (tmp_path/'.writer-lock').mkdir()
    with pytest.raises(RuntimeError, match='writer is active'):
        study.status(tmp_path, tmp_path)
    (tmp_path/'.writer-lock').rmdir()
    (tmp_path/'inputs').mkdir()
    (tmp_path/'inputs/calendar.json').write_bytes(Path('research/calendars/ice-cotton-2026.json').read_bytes())
    freeze_record(tmp_path/'cohort-lock.json', {'first_origin': '2026-10-30'})
    freeze_record(tmp_path/'forecasts/2026-10-30/publication-receipt.json', {'status': 'TIMELY'})
    with (patch.object(study, 'verify', return_value=({}, {'required_origins': 252})),
          pytest.raises(ValueError, match='Partial publication')):
        study.status(tmp_path, tmp_path)


def test_manual_capture_cannot_poll_outside_cutoff_window(tmp_path):
    freeze_record(tmp_path/'cohort-lock.json', {'first_origin': '2026-10-30'})
    with (patch.object(study, 'verify', return_value=({}, {})),
          patch.object(study, 'now', return_value=study.utc('2026-10-30T12:00Z')),
          patch('requests.get', side_effect=AssertionError('No network')),
          pytest.raises(ValueError, match='no background polling')):
        study.capture(tmp_path, tmp_path)
