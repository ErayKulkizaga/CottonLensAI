from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ams_exploration import (
    LAGS,
    add_quotes,
    dispatch,
    group_names,
    read_quotes,
    short_split,
    specs,
)
from cottonlens_ml.research.ledger import freeze_record


def cotton():
    dates = pd.bdate_range('2010-01-01','2024-01-31')
    frame = pd.DataFrame({'date':dates,'cotton_close':100.,'cotton_session_index':np.arange(len(dates))})
    for h in (1,5):
        frame[f'target_date_{h}'] = frame.date.shift(-h)
        frame[f'target_return_{h}'] = .01
    return frame


def test_lags_never_backfill_and_late_older_report_does_not_replace_newer():
    history = cotton()
    quotes = pd.DataFrame({'report_date':pd.to_datetime(['2020-01-02','2020-01-03']),
        'assumed_day':pd.to_datetime(['2020-01-08','2020-01-03']),
        'spot_41_4_34_cents_per_lb':[80.,90.]})
    out = add_quotes(history,quotes).set_index('date')
    assert np.isnan(out.loc['2020-01-03','ams_spread_L1'])
    assert out.loc['2020-01-06','ams_spread_L1'] == pytest.approx(-.1)
    assert np.isnan(out.loc['2020-01-06','ams_spread_L2'])
    assert out.loc['2020-01-09','ams_spread_L1'] == pytest.approx(-.1)
    # The late old report cannot revive a stale older quotation.
    assert np.isnan(out.loc['2020-01-10','ams_spread_L1'])
    assert len(out) == len(history)


def test_future_quotation_and_cotton_changes_do_not_change_past_features():
    history = cotton()
    quotes = pd.DataFrame({'report_date':pd.to_datetime(['2020-01-02','2020-01-20']),
        'assumed_day':pd.to_datetime(['2020-01-02','2020-01-20']),
        'spot_41_4_34_cents_per_lb':[80.,90.]})
    expected = add_quotes(history,quotes)
    future = quotes.copy(); future.loc[1,'spot_41_4_34_cents_per_lb'] = 1000
    changed = history.copy(); changed.loc[changed.date>'2020-01-15','cotton_close'] = 200
    actual = add_quotes(changed,future)
    past = expected.date <= '2020-01-15'
    pd.testing.assert_frame_equal(expected.loc[past],actual.loc[past])


def test_short_cohort_keeps_missing_features_and_full_year_past_purge():
    frame = cotton()
    for lag in LAGS:frame[f'ams_spread_L{lag}'] = np.nan
    split = short_split(frame,pd.Timestamp('2020-01-10'))
    assert [f['year'] for f in split['folds']] == [2023]
    assert len(split['folds'][0]['origins']) > 240
    assert split['gate_evaluation_allowed'] is False
    for b in split['folds'][0]['inner']:
        assert len(b['origins']) == 63
        train = frame.loc[(frame.date>='2020-01-10') & (frame.target_date_5<pd.Timestamp(b['cutoff']))]
        assert len(train) >= 500


def test_equal_small_model_budget_and_matching_missingness_controls():
    groups = group_names()
    assert len(groups)==7
    for lag in LAGS:
        assert groups[f'quote_L{lag}'][:-1] == groups[f'missing_L{lag}']
    for names in groups.values():
        recipes = specs(names)
        assert len(recipes)==2 and recipes[0]['family']=='ridge'
        assert recipes[0]['params']['alpha']==1
        assert recipes[1]['max_iterations']==600
        assert all(r['device']=='cpu' and r['seed']==42 and r['horizon']==5 for r in recipes)


def test_tier_a_export_is_rejected_before_read_or_fit(tmp_path):
    with pytest.raises(ValueError,match='cannot lock/export'):
        dispatch(SimpleNamespace(stage='export',drive_root=tmp_path,experiment='research-fixture'))
    assert not list(tmp_path.iterdir())


def test_input_checksums_and_exact_quotation_bytes_are_required(tmp_path):
    table = tmp_path/'spot.csv'
    pd.DataFrame({'report_date':['2020-01-02'],'spot_41_4_34_cents_per_lb':[80.],
                  'report_sha256':['a'*64]}).to_csv(table,index=False)
    sha = digest(table)
    freeze_record(table.with_suffix('.manifest.json'),{'table_sha256':sha,'row_count':1,'model_eligible':False})
    review = tmp_path/'review.json'
    freeze_record(review,{'table_sha256':sha,'model_eligible':False,'version_records':[
        {'report_date':'2020-01-02','matches_esmis_bytes':True,'document_sha256':'a'*64,
         'published_local_naive':'2020-01-03T15:00:00'}]})
    rows, evidence = read_quotes(table,review)
    assert rows.assumed_day.iloc[0]==pd.Timestamp('2020-01-03')
    assert evidence['release_allowed'] is False
    table.write_text('corrupt')
    with pytest.raises(ValueError,match='identity mismatch'):
        read_quotes(table,review)


def test_planned_pause_preserves_choice_and_completed_output_is_reused(tmp_path, monkeypatch):
    from cottonlens_ml.research import ams_exploration as module
    calls, stamp = [], [0.]
    monkeypatch.setattr(module.time,'monotonic',lambda:stamp[0])
    frame=cotton().loc[lambda f:f.date.between('2023-01-03','2023-01-04')].copy()
    fold={'year':2023,'origins':frame.date.dt.strftime('%Y-%m-%d').tolist()}
    experiment=SimpleNamespace(root=tmp_path,history=frame,
        identity={'split':{'folds':[fold]},'groups':{'base':['cotton_close']}})
    def inner(exp, recipe, block):
        calls.append('inner')
        return {'recipe':recipe,'weight':.5,'inner_score':.99,'iterations':1}
    def outer(exp, recipe, origins, role, iterations):
        exp.before_compute()
        calls.append('outer')
        return np.zeros(len(origins))
    monkeypatch.setattr(module,'inner_price',inner)
    monkeypatch.setattr(module,'predict_chunks',outer)
    def paused_outer(*args):
        stamp[0]=100.
        return outer(*args)
    monkeypatch.setattr(module,'predict_chunks',paused_outer)
    assert module.run(experiment,1)['status']=='planned_pause'
    assert (tmp_path/'information-decisions/base-2023.json').exists()
    assert not list((tmp_path/'information-outputs').glob('*.json'))
    stamp[0]=0.
    monkeypatch.setattr(module,'predict_chunks',outer)
    assert module.run(experiment,1)['status']=='complete'
    assert module.run(experiment,1)['status']=='complete'
    assert calls==['inner','inner','outer']
