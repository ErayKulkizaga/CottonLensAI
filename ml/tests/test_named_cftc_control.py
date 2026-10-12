"""No real-market fits; fixed source contrast, clock and ledger regressions."""
import importlib.metadata
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import named_cftc_control as pilot
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record
from cottonlens_ml.research.protocol import Preprocessor, inputs


def example():
    h = pd.DataFrame({'date': pd.bdate_range('2023-01-02', periods=5),
                      'cotton_session_index': np.arange(5), 'cotton_ret_1': np.arange(5.)})
    h['cftc_age_L1'], h['cftc_missing_L1'] = [np.nan, 8., 9., 10., 11.], [1., 0., 0., 1., 0.]
    for i, name in enumerate(pilot.POSITION):
        h[name] = [np.nan, .1+i, .2+i, np.nan, .3+i]
    return h


def test_same_37_fields_mask_only_three_finite_positions_and_keep_processor_missing_bits():
    h = example()
    num, mask = pilot.arm_history(h, 'numeric'), pilot.arm_history(h, 'mask')
    pd.testing.assert_frame_equal(num, h, check_exact=True)
    pd.testing.assert_frame_equal(num.drop(columns=list(pilot.POSITION)), mask.drop(columns=list(pilot.POSITION)), check_exact=True)
    np.testing.assert_array_equal(num.isna(), mask.isna())
    assert not mask[list(pilot.POSITION)].fillna(0).to_numpy().any()
    pn, pm = Preprocessor.fit(num, list(pilot.FIELDS)), Preprocessor.fit(mask, list(pilot.FIELDS))
    xn, xm = inputs(num, num, pn), inputs(mask, mask, pm)
    np.testing.assert_array_equal(xn[:, len(pilot.FIELDS):], xm[:, len(pilot.FIELDS):])
    np.testing.assert_array_equal(xn[:, :2], xm[:, :2])
    assert np.any(xn[:, 2:5] != xm[:, 2:5])
    assert len(pilot.RECIPE['features']) == 37
    assert pilot.RECIPE['features'][:-5] == pilot.shared.RECIPE['features']
    assert {k:v for k,v in pilot.RECIPE.items() if k != 'features'} == {
        k:v for k,v in pilot.shared.RECIPE.items() if k != 'features'}
    assert not {'cftc_managed_money_net', 'target_return_5', 'named_return', 'named_target_price'}.intersection(pilot.RECIPE['features'])
    pd.testing.assert_frame_equal(h, example())


@pytest.mark.parametrize('arm', ['ct', 'position_L2', 'wrong'])
def test_no_extra_arm_or_lag_grid(arm):
    with pytest.raises(ValueError, match='Exact CFTC arm'):
        pilot.arm_history(example(), arm)


def test_source_prefix_invariant_to_future_report_or_label_changes_and_no_origin_drop():
    h = pd.DataFrame({'date': pd.bdate_range('2023-03-20', periods=40),
                      'target_return_5': np.arange(40.)})
    raw = pd.DataFrame({'report_date': pd.to_datetime(['2023-03-21', '2023-04-04']),
        'open_interest': [100, 100], 'managed_long': [50, 60], 'managed_short': [20, 10],
        'producer_long': [10, 20], 'producer_short': [40, 60]})
    rows = pilot.source.derive(raw)
    original = pilot.attach(h, rows)
    assert len(original) == len(h)
    assert original.loc[h.date.le('2023-03-28'), pilot.POSITION[0]].isna().all()
    changed = rows.copy()
    changed.loc[1, 'managed_net_share'] = -.9
    future = pilot.attach(h, changed)
    prefix = h.date.le('2023-04-10')
    pd.testing.assert_frame_equal(original.loc[prefix], future.loc[prefix], check_exact=True)
    labels = h.copy()
    labels.loc[h.date.gt('2023-04-10'), 'target_return_5'] = -999.
    pd.testing.assert_frame_equal(original[list(pilot.FIELDS)], pilot.attach(labels, rows)[list(pilot.FIELDS)], check_exact=True)
    with pytest.raises(ValueError, match='already attached'):
        pilot.attach(original, rows)


def test_controls_use_new_position_signal_and_existing_four_fit_budget():
    linear, nonlinear, negative = [pilot.synthetic(k) for k in pilot.KINDS[:3]]
    for frame, train, test, recipe, trees in (linear, nonlinear, negative):
        assert recipe['features'] == pilot.RECIPE['features'] and trees == 100
        assert train.date.max() < test.date.min()
        for name in pilot.FIELDS:
            np.testing.assert_array_equal(train[name], frame.loc[train.index, name])
            np.testing.assert_array_equal(test[name], frame.loc[test.index, name])
    np.testing.assert_array_equal(linear[0].target_return_5, .02 * linear[0][pilot.POSITION[0]])
    np.testing.assert_array_equal(nonlinear[0].target_return_5, .02 * (nonlinear[0][pilot.POSITION[0]]**2-1))
    np.testing.assert_array_equal(negative[1].target_return_5, np.roll(linear[1].target_return_5, 137))
    _, train, test, recipe, trees = pilot.synthetic('small_subset_overfit')
    pd.testing.assert_frame_equal(train, test)
    assert trees == 400 and len(train) == 128 and recipe['features'] == pilot.RECIPE['features']


def fixture(tmp_path, monkeypatch):
    """Small locally generated evidence binding; no real data or model fit."""
    repo, folder = tmp_path/'repo', tmp_path/'registration'
    code = (*pilot.capacity.CORE, 'ml/src/cottonlens_ml/research/cftc_exploration.py')
    for name in (*code, 'ml/uv.lock'):
        p = repo/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('synthetic fixture', encoding='utf-8')
    h = pd.DataFrame({'date': pd.bdate_range('2019-01-02', '2023-12-20')})
    h['cotton_session_index'] = np.arange(len(h))
    h['cotton_close'], h['named_price'] = 80., 80.
    h['target_date_5'] = h.date.shift(-5)
    h['target_return_5'], h['named_return'], h['named_target_price'] = .01, .01, 80.*np.exp(.01)
    h['named_eligible'] = h.target_date_5.notna()
    # One admitted-label gap makes a 63-label inner block span four calendar
    # buckets, matching the frozen 22-job design without compressing time.
    h.loc[h.date.eq('2022-12-01'), 'named_eligible'] = False
    h['named_label_available_at'] = h.target_date_5.dt.tz_localize('UTC') + pd.Timedelta(days=1)
    for name in pilot.shared.RECIPE['features']:
        h[name] = .1
    origins = h.loc[h.date.dt.year.eq(2023), 'date'].iloc[:246].dt.strftime('%Y-%m-%d').tolist()
    planned = pilot.labels.plan(h, {'folds': [{'year': 2023, 'origins': origins}]})
    parent_design = {'profile': pilot.weak.PROFILE, 'recipe': pilot.shared.RECIPE, 'folds': planned['folds']}
    (folder/'inputs/named').mkdir(parents=True)
    h.to_parquet(folder/'inputs/named/history.parquet', index=False)
    parent = {'history_sha256': digest(folder/'inputs/named/history.parquet'), 'identity': {
        'design': parent_design, 'design_id': content_id(parent_design), 'research_data_id': frame_identity(h, list(h))}}
    freeze_record(folder/'inputs/named/ready.json', parent)
    raw = pd.DataFrame({'report_date': pd.date_range('2018-12-18', '2023-12-12', freq='7D')})
    raw['market_code'], raw['report_type'] = pilot.source.MARKET, 'FutOnly'
    for name, value in [('open_interest',100), ('managed_long',50), ('managed_short',20), ('producer_long',10), ('producer_short',40)]:
        raw[name] = value
    rows = pilot.source.derive(raw)
    table = folder/'inputs/inputs/positions-2010-2023.csv'
    table.parent.mkdir(parents=True); rows.to_csv(table, index=False, date_format='%Y-%m-%d')
    freeze_record(table.with_suffix('.manifest.json'), {'table_sha256': digest(table), 'rows': len(rows),
        'model_eligible': False, 'release_allowed': False, 'publication_timestamp_verified': False, 'first_version_verified': False})
    (folder/'inputs/result').mkdir()
    pilot.source.add_positions(h, rows).to_parquet(folder/'inputs/result/candidate-history.parquet', index=False)
    freeze_record(folder/'inputs/result/candidate.json', {'status': 'exploratory_alignment_only_not_admitted_or_preregistered',
        'parent_history_sha256': parent['history_sha256'], 'release_allowed': False, 'new_fits': 0})
    freeze_record(folder/'inputs/result/audit.json', {'named_target_source_effect_tested_by_old_experiment': False})
    freeze_record(folder/'inputs/controls.json', {'status': 'synthetic_fixture'})
    inventory = {name: {'sha256': digest(folder/'inputs'/name), 'bytes': (folder/'inputs'/name).stat().st_size} for name in pilot.INPUTS}
    inventory.update({'source-snapshot/'+name: {'sha256': digest(repo/name)} for name in code})
    (folder/'inputs-inventory.json').write_text(json.dumps(inventory), encoding='utf-8')
    freeze_record(repo/pilot.RELEASE_PROOF, {'inventory_sha256': digest(folder/'inputs-inventory.json'), 'archive': {'sha256': 'fixture'}})
    freeze_record(repo/pilot.AUDIT_PROOF, {'audit_sha256': digest(folder/'inputs/result/audit.json'),
        'candidate_sha256': digest(folder/'inputs/result/candidate-history.parquet'), 'controls_sha256': digest(folder/'inputs/controls.json')})
    freeze_record(repo/pilot.DELIVERY_PROOF, {'status': 'fresh_github_cftc_audit_native_and_scalar_recovery_verified',
        'fresh_restore': {'archive_sha256': 'fixture', 'exact_audit_candidate_controls': True},
        'historical_source_admitted': False, 'first_vintage_verified': False})
    enhanced = pilot.attach(h, pilot.source.read_positions(table)[0])
    enhanced.to_parquet(folder/'history.parquet', index=False)
    d = pilot.design(parent)
    freeze_record(folder/'decision-contract.json', d)
    identity = {'profile': pilot.PROFILE, 'source_id': 'fixture', 'python': sys.version.split()[0],
        'versions': {name: importlib.metadata.version(name) for name in pilot.VERSIONS}, 'lock_sha256': digest(repo/'ml/uv.lock'),
        'design': d, 'design_id': content_id(d), 'research_data_id': frame_identity(enhanced, list(enhanced)),
        'input_hashes': {'inputs-inventory.json': digest(folder/'inputs-inventory.json'), **{'inputs/'+name:digest(folder/'inputs'/name) for name in pilot.INPUTS}},
        'evidence_hashes': {name:digest(repo/name) for name in [pilot.AUDIT_PROOF,pilot.RELEASE_PROOF,pilot.DELIVERY_PROOF]},
        'historical_source_admitted': False, 'release_allowed': False}
    freeze_record(folder/'ready.json', {'identity': identity, 'history_sha256': digest(folder/'history.parquet')})
    monkeypatch.setattr(pilot, 'research_source_identity', lambda _: {'source_id':'fixture'})
    return repo, folder


def test_exact_frozen_input_recipe_and_mature_jobs_rebuild(tmp_path, monkeypatch):
    repo, folder = fixture(tmp_path, monkeypatch)
    ready, h = pilot.verify(folder, repo)
    assert ready['identity']['design']['fit_budget']['market'] == 44
    assert ready['identity']['design']['fit_budget']['synthetic'] == 4
    assert len(h) > 1000 and set(pilot.FIELDS).issubset(h)
    assert ready['identity']['design']['release_allowed'] is False


@pytest.mark.parametrize('change', ['source', 'environment', 'input', 'publication_evidence', 'core', 'history', 'design'])
def test_corruption_or_reidentity_rejected_before_fits(tmp_path, monkeypatch, change):
    repo, folder = fixture(tmp_path, monkeypatch)
    if change == 'source':
        monkeypatch.setattr(pilot, 'research_source_identity', lambda _: {'source_id': 'changed'})
    elif change == 'environment':
        monkeypatch.setattr(pilot.sys, 'version', '0.0.0 synthetic')
    else:
        path = {'input': folder/'inputs/inputs/positions-2010-2023.csv', 'publication_evidence': repo/pilot.DELIVERY_PROOF,
                'core': repo/pilot.capacity.CORE[0], 'history': folder/'history.parquet', 'design': folder/'decision-contract.json'}[change]
        path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises((ValueError, AssertionError, json.JSONDecodeError)):
        pilot.verify(folder, repo)
    assert not (folder/'ledger').exists()


@pytest.mark.parametrize('interrupt', [False, True])
def test_shared_engine_uses_37_field_policy_and_restores_history_on_pause(tmp_path, monkeypatch, interrupt):
    h = example()
    ds = h.date.dt.strftime('%Y-%m-%d').tolist()
    fold = {'year': 2023, 'jobs': [{'role':'outer','bucket':0,'train_dates':ds[:3],'origins':ds[3:]}]}
    ex = SimpleNamespace(root=tmp_path, history=h.copy(), identity={'source_id':'fixture'})
    original = ex.history
    calls = []
    ex.ledger = SimpleNamespace(results=lambda: calls)
    monkeypatch.setattr(pilot, 'verify', lambda *_: ({'identity': {'design': {'folds':[fold], 'fit_budget': {'market':2}}}}, h))
    monkeypatch.setattr(pilot, 'publication_check', lambda *_: None)
    monkeypatch.setattr(pilot.capacity, 'learning', lambda *a,**kw: None)
    monkeypatch.setattr(pilot.shared, 'fit_consumption', lambda _: (0,set()))
    monkeypatch.setattr(pilot.shared, 'research_source_identity', lambda _: {'source_id':'fixture'})
    monkeypatch.setattr(pilot.labels, 'projection', lambda history,dates,*a,**kw: history.loc[history.date.dt.strftime('%Y-%m-%d').isin(dates)].copy())
    monkeypatch.setattr(pilot.labels, 'make_output', lambda *_: {'synthetic_test':True})
    monkeypatch.setattr(pilot, 'verify_output', lambda *_: None)
    def fit(recipe, train, validation, test, role, iterations):
        ex.before_compute()
        assert recipe == pilot.RECIPE and iterations == 100 and validation is None
        pd.testing.assert_frame_equal(ex.history, pilot.arm_history(h, role.split('-')[0]), check_exact=True)
        if role.startswith('mask') and interrupt:
            raise FitBudgetReached('test planned pause')
        calls.append(role)
        return {'synthetic_test':True}
    ex.fit = fit
    result = pilot.run(ex, tmp_path, 30)
    assert result['status'] == ('planned_pause' if interrupt else 'complete')
    assert ex.history is original and len(calls) == (1 if interrupt else 2)


def test_failed_learning_control_cannot_start_market(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'verify', lambda *_: ({}, example()))
    monkeypatch.setattr(pilot, 'publication_check', lambda *_: None)
    def fail(*a, **kw):
        assert kw['profile'] == pilot.PROFILE
        raise ValueError('Learning control failed')
    monkeypatch.setattr(pilot.capacity, 'learning', fail)
    with pytest.raises(ValueError, match='Learning control failed'):
        pilot.run(SimpleNamespace(root=tmp_path), tmp_path, 30)
    assert not (tmp_path/'ledger').exists()


def test_unknown_or_incomplete_output_cannot_be_completed(tmp_path):
    class Ledger:
        root = tmp_path/'ledger'
        def key(self, expected):
            assert expected['recipe'] == pilot.RECIPE
            return 'missing'
    h = example()
    ds = h.date.dt.strftime('%Y-%m-%d').tolist()
    ex = SimpleNamespace(ledger=Ledger(), identity={})
    fold = {'year':2023,'jobs':[{'role':'outer','bucket':0,'train_dates':ds[:3],'origins':ds[3:]}]}
    with pytest.raises(FileNotFoundError):
        pilot.verify_output(ex, h.assign(named_return=.01,named_eligible=True), fold, 'mask')


def test_partial_prepare_preserved(tmp_path, monkeypatch):
    repo, registration = fixture(tmp_path/'fixture', monkeypatch)
    proof = {'ready_sha256':digest(registration/'ready.json'),
             'decision_contract_sha256':digest(registration/'decision-contract.json'), 'new_fits':0,
             'source_id':'fixture', 'design_id':read_record(registration/'ready.json')['identity']['design_id']}
    freeze_record(repo/pilot.PREREGISTRATION, proof)
    with pytest.raises(FileNotFoundError):
        pilot.publication_check(registration, repo)
    freeze_record(repo/pilot.PREREGISTRATION_DELIVERY, {'status':'fresh_github_named_cftc_preregistration_verified',
        'preregistration_sha256':digest(repo/pilot.PREREGISTRATION), 'ready_sha256':proof['ready_sha256'],
        'source_id':'fixture','exact_members':True,'new_fits':0})
    target = tmp_path/'partial'; target.mkdir(); (target/'precious').write_bytes(b'keep')
    with pytest.raises(ValueError, match='partial preparation'):
        pilot.prepare(repo, target, registration, registration/'decision-contract.json')
    assert (target/'precious').read_bytes() == b'keep'
    target = tmp_path/'new'
    pilot.prepare(repo, target, registration, registration/'decision-contract.json')
    first = (target/'ready.json').read_bytes()
    pilot.prepare(repo, target, registration, registration/'decision-contract.json')
    assert (target/'ready.json').read_bytes() == first


def test_cftc_decision_keeps_practical_and_contribution_questions_separate():
    intervals = {f'mask_minus_numeric_raw_{b}': {'difference_ci_95':[-.1,.1]} for b in [20,60]}
    intervals.update({f'numeric_raw_vs_naive_{b}':{'gain_ci_95_pct':[-3,4.99]} for b in [20,60]})
    r = pilot.classify({'numeric_raw':{'gain_pct':1,'direction_pct':56}}, intervals)
    assert r['decision'] == 'FIXED_NAMED_CFTC_BELOW_PRACTICAL_GOAL'
    assert r['source_contribution_supported'] is False
    assert r['numeric_practical_effect_excluded'] is True
