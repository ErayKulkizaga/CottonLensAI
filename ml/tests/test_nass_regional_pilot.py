"""Synthetic only: assumptions stay explicit and no future report is backfilled."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import nass_regional_pilot as pilot
from cottonlens_ml.research.protocol import Preprocessor


def panel():
    items = []
    for week, release, current in [('2023-05-28', '2023-05-30', 50),
                                   ('2023-06-04', '2023-06-05', 60),
                                   ('2023-06-18', '2023-06-19', 80)]:
        progress = {stage: {'section_present': False, 'cells': None, 'gap_to_average_pp': None,
                            'published_average_years': None} for stage in pilot.STAGES}
        progress['Planted'] = {'section_present': True,
                               'cells': {key: {'value': value, 'qualifier': None} for key, value in
                                         [('current', current), ('published_average', 54),
                                          ('previous_week', 999), ('previous_year', 70)]},
                               'published_average_years': [2018, 2022], 'gap_to_average_pp': current - 54}
        items.append({'week_ending': week, 'release_day': release, 'report_sha256': 'a' * 64,
                      'available_at': None, 'first_version_verified': False,
                      'national_condition': dict(zip(pilot.CATEGORIES, (1, 12, 39, 41, 7), strict=True)),
                      'texas_condition': {c: {'value': v, 'qualifier': None} for c, v in
                                          zip(pilot.CATEGORIES, (2, 19, 51, 24, 4), strict=True)},
                      'progress': progress})
    return {'panel': items, 'reports': 3, 'model_eligible': False, 'release_allowed': False,
            'availability_policy': 'UNSET'}


def history(dates=None):
    if dates is None:
        dates = pd.bdate_range('2023-05-26', '2023-07-10')
    f = pd.DataFrame({'date': pd.to_datetime(dates)})
    for i, name in enumerate(pilot.FEATURE_NAMES):
        f[name] = np.arange(len(f), dtype=float) + i
    f['cotton_close'] = 70.
    return f


def aligned(frame=None, source=None):
    return pilot.align_assumed(history() if frame is None else frame,
                               pilot.report_features(panel() if source is None else source))


def test_first_decision_inclusive_and_one_observation_delay():
    r = aligned(history(['2023-05-29', '2023-05-30', '2023-05-31', '2023-06-01']))
    name = 'texas_planted_gap'
    assert pd.isna(r[f'numeric_D0_{name}'].iloc[0])
    assert r[f'numeric_D0_{name}'].iloc[1] == -.04
    assert pd.isna(r[f'numeric_D1_{name}'].iloc[1])
    assert r[f'numeric_D1_{name}'].iloc[2] == -.04
    for d in pilot.DELAYS:
        a = pd.to_datetime(r[f'nass_D{d}_assumed_available_at'], utc=True)
        b = pd.to_datetime(r.nass_decision_time, utc=True)
        assert (a.dropna() <= b.loc[a.notna()]).all()
    assert pilot.CLOCK['publication_verified'] is False


def test_weekend_and_missing_cotton_date_use_recorded_observations():
    p = panel()
    p['panel'][0]['release_day'] = '2023-06-03'
    r = aligned(history(['2023-06-02', '2023-06-05', '2023-06-07']), p)
    assert pd.isna(r.numeric_D0_texas_planted_gap.iloc[0])
    assert r.numeric_D0_texas_planted_gap.iloc[1] == .06  # Jun5 report is newest at Jun6 cutoff.
    assert r.numeric_D1_texas_planted_gap.iloc[2] == .06


def test_common_expiry_and_no_winter_carry():
    p = panel(); p['panel'] = p['panel'][:1]; p['reports'] = 1
    f = history(pd.bdate_range('2023-05-30', periods=13))
    r = aligned(f, p)
    for d in pilot.DELAYS:
        assert r[f'numeric_D{d}_texas_planted_gap'].iloc[10] == -.04
        assert pd.isna(r[f'numeric_D{d}_texas_planted_gap'].iloc[11])
    p['panel'][0]['week_ending'] = '2023-12-24'
    p['panel'][0]['release_day'] = '2023-12-26'
    r = aligned(history(['2023-12-26', '2023-12-27', '2024-01-02']), p)
    assert r.numeric_D0_texas_planted_gap.iloc[0] == -.04
    assert pd.isna(r.numeric_D0_texas_planted_gap.iloc[2])


def test_latest_absent_stage_does_not_carry_old_stage():
    p = panel()
    p['panel'][1]['progress']['Planted'] = p['panel'][1]['progress']['Squaring'].copy()
    r = aligned(source=p)
    assert pd.isna(r.loc[r.date.eq('2023-06-05'), 'numeric_D0_texas_planted_gap'].iloc[0])


def test_later_previous_week_revision_is_not_used():
    a = pilot.report_features(panel())
    p = panel(); p['panel'][1]['progress']['Planted']['cells']['previous_week']['value'] = 0
    pd.testing.assert_frame_equal(a, pilot.report_features(p))
    assert a.national_ge_change.iloc[1] == 0
    assert pd.isna(a.national_ge_change.iloc[2])  # Nonconsecutive report: unknown weekly delta.


def test_future_source_change_and_future_cotton_rows_do_not_change_past():
    f, p = history(), panel()
    a = aligned(f, p)
    changed = deepcopy(p)
    cell = changed['panel'][2]['progress']['Planted']
    cell['cells']['current']['value'] = 90; cell['gap_to_average_pp'] = 36
    b = aligned(f, changed)
    pd.testing.assert_frame_equal(a.loc[a.date < '2023-06-19'], b.loc[b.date < '2023-06-19'])
    pd.testing.assert_frame_equal(a.iloc[:10], aligned(f.iloc[:10], p))
    pd.testing.assert_frame_equal(f, history())
    assert p == panel()


@pytest.mark.parametrize('delay', pilot.DELAYS)
def test_control_shares_national_values_and_all_automatic_masks(delay):
    f = aligned()
    a, b = pilot.groups()[f'mask_D{delay}'], pilot.groups()[f'numeric_D{delay}']
    left = Preprocessor.fit(f.iloc[:10], a).transform(f)
    right = Preprocessor.fit(f.iloc[:10], b).transform(f)
    assert left.shape == right.shape
    np.testing.assert_array_equal(left[:, len(a):], right[:, len(b):])
    for name in pilot.REGIONAL:
        c, v = f[f'mask_D{delay}_{name}'], f[f'numeric_D{delay}_{name}']
        assert c.isna().equals(v.isna()) and c.dropna().eq(0).all()
    assert a[:29] == b[:29]  # core24 + timing2 + national3
    future = f.copy(); future.loc[10:, b[-5]] = 1e6
    original = Preprocessor.fit(f.iloc[:10], b).transform(f.iloc[:10])
    altered = Preprocessor.fit(future.iloc[:10], b).transform(future.iloc[:10])
    np.testing.assert_array_equal(original, altered)


@pytest.mark.parametrize('defect', ['admitted', 'duplicate', 'future', 'intraday', 'sum', 'gap', 'average', 'absent', 'clock', 'order'])
def test_source_defects_fail_closed(defect):
    p = panel()
    if defect == 'admitted': p['model_eligible'] = True
    elif defect == 'duplicate': p['panel'][1]['release_day'] = p['panel'][0]['release_day']
    elif defect == 'future': p['panel'][2]['release_day'] = '2024-01-01'
    elif defect == 'intraday': p['panel'][0]['release_day'] += 'T12:00:00'
    elif defect == 'sum': p['panel'][0]['national_condition']['GOOD'] += 1
    elif defect == 'gap': p['panel'][0]['progress']['Planted']['gap_to_average_pp'] = 0
    elif defect == 'average': p['panel'][0]['progress']['Planted']['published_average_years'] = [2019, 2023]
    elif defect == 'absent': p['panel'][0]['progress']['Squaring']['gap_to_average_pp'] = 0
    elif defect == 'clock': p['panel'][0]['available_at'] = '2023-05-30T20:00:00Z'
    elif defect == 'order': p['panel'] = list(reversed(p['panel']))
    with pytest.raises(ValueError): pilot.report_features(p)


def test_duplicate_history_and_overwrite_rejected():
    f = history(); f.loc[1, 'date'] = f.date.iloc[0]
    with pytest.raises(ValueError): aligned(f)
    with pytest.raises(ValueError): aligned(aligned())


def test_unknown_current_or_average_remains_missing_and_targets_are_preserved():
    p = panel()
    progress = p['panel'][0]['progress']['Planted']
    progress['cells']['current'] = {'value': None, 'qualifier': '(NA)'}
    progress['gap_to_average_pp'] = None
    f = history()
    f['target_date_1'] = f.date.shift(-1)
    f['target_return_1'] = np.linspace(-.1, .1, len(f))
    r = aligned(f, p)
    pd.testing.assert_frame_equal(r[f.columns], f)
    assert pd.isna(r.loc[r.date.eq('2023-05-30'), 'numeric_D0_texas_planted_gap'].iloc[0])
    assert pd.isna(r.loc[r.date.eq('2023-05-30'), 'mask_D0_texas_planted_gap'].iloc[0])


def test_source_presence_is_not_publication_approval():
    assert pilot.CLOCK['first_version_verified'] is False
    assert pilot.RULES['source_admission'] is False
    assert pilot.RULES['automatic_release'] is False
    assert pilot.RULES['independent_holdout'] is False
