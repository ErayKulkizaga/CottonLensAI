"""Synthetic contracts only; real market fitting is a separately authorized pilot."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.features import (
    AVAILABILITY_ALIGNMENT,
    _external_alignment,
    build_feature_history,
)
from cottonlens_ml.research import availability_clock as clock
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research.diagnostics import (
    VALIDATION_POLICY,
    control_failures,
    diagnose,
)
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record
from cottonlens_ml.research.protocol import full_year_manifest
from cottonlens_ml.sequences import sequences


def test_clock_boundary_weekend_missing_nonpositive_and_stale():
    dates = pd.to_datetime(['2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12'])
    market = pd.DataFrame({'date': pd.to_datetime(['2023-01-06', '2023-01-09', '2023-01-10']),
                           'series': 'wti', 'close': [80., 0., -2.]})
    times = pd.to_datetime(dates, utc=True) + pd.Timedelta(days=1)
    at = _external_alignment(market, dates, 'wti', policy=AVAILABILITY_ALIGNMENT, decision_times=times)
    assert at.close.iloc[0] == 80  # equality is available; weekend age counts Cotton rows
    assert at.close.iloc[3] == 80 and at.age_sessions.iloc[3] == 3
    assert np.isnan(at.close.iloc[4]) and at.stale.iloc[4]
    early = _external_alignment(market, dates, 'wti', policy=AVAILABILITY_ALIGNMENT,
                                decision_times=times - pd.Timedelta(seconds=1))
    assert np.isnan(early.close.iloc[0])
    with pytest.raises(ValueError, match='policy'):
        _external_alignment(market, dates, 'wti', policy='unknown')


@pytest.fixture
def packet(tmp_path):
    dates = pd.bdate_range('2010-01-01', '2023-12-29')
    market = pd.concat([pd.DataFrame({'date': dates, 'series': name,
        'close': base + np.sin(np.arange(len(dates)) / period),
        'open': base, 'high': base + 2, 'low': base - 2, 'volume': 1000.})
        for name, base, period in [('cotton', 80, 10), ('dxy', 100, 9), ('wti', 60, 13)]], ignore_index=True)
    market_file = tmp_path / 'market.parquet'
    market.to_parquet(market_file)
    history = build_feature_history(market, pd.DataFrame(columns=['available_date', 'cftc_managed_money_net']))
    history.attrs = {}
    split = full_year_manifest(history)
    for fold in split['folds']:
        count = 251 if fold['year'] == 2016 else (255 if fold['year'] == 2023 else 250)
        fold['origins'] = fold['origins'][:count]
    reference = tmp_path / 'reference'
    reference.mkdir()
    history.to_parquet(reference / 'history.parquet')
    freeze_record(reference / 'ready.json', {'identity': {'split': split,
        'raw_audit': {'market_sha256': digest(market_file)}}, 'history_sha256': digest(reference / 'history.parquet')})
    folder = tmp_path / 'pilot'
    clock.prepare(Path(__file__).parents[2], folder, reference, market_file)
    return folder, reference, market_file, history, market


def test_control_matches_and_future_data_cannot_change_past(packet):
    folder, _, _, history, market = packet
    expanded = pd.read_parquet(folder / 'history.parquet')
    pd.testing.assert_frame_equal(expanded[history.columns], history)
    changed = market.copy()
    changed.loc[changed.date > '2020-01-01', 'close'] *= 1.2
    future = build_feature_history(changed, pd.DataFrame(columns=['available_date', 'cftc_managed_money_net']),
                                   alignment_policy=AVAILABILITY_ALIGNMENT)
    mask = history.date < '2020-01-01'
    np.testing.assert_allclose(future.loc[mask, clock.CROSS],
        expanded.loc[mask, ['available_' + n for n in clock.CROSS]], equal_nan=True)
    bad = history.copy()
    bad.loc[400, FEATURE_NAMES[0]] += 1e-8
    with pytest.raises(ValueError, match='Control feature'):
        clock.expanded_history(bad, market)
    bad.loc[400, FEATURE_NAMES[0]] = np.nan
    with pytest.raises(ValueError, match='Control feature'):
        clock.expanded_history(bad, market)


@pytest.mark.parametrize('field', ['date', 'target_date', 'horizon', 'cotton_close', 'actual_return'])
def test_comparison_rejects_mismatched_targets(field):
    frame = pd.DataFrame({'date': ['2020-01-01'], 'target_date': ['2020-01-02'],
                          'horizon': [1], 'cotton_close': [80.], 'actual_return': [.01]})
    changed = frame.copy()
    changed.loc[0, field] = '2020-01-03' if 'date' in field else 5
    with pytest.raises(ValueError, match='Unmatched'):
        clock.require_same_targets(frame, changed)


def test_all_learning_families_and_old_pass_cache(tmp_path):
    controls = {'xgboost/known_signal': {'relative_mae_gain': .95},
        'xgboost/small_subset_overfit': {'relative_mae_gain': .99},
        'xgboost/negative_block_shift': {'relative_mae_gain': 0},
        **{f'{family}/known_signal': {'relative_mae_gain': .8} for family in ('catboost', 'mlp', 'lstm')},
        'tcn/known_signal': {'relative_mae_gain': -.235931}}
    assert 'tcn' in control_failures(controls, ('xgboost', 'tcn'))[0]
    path = tmp_path / 'diagnosis.json'
    freeze_record(path, {'status': 'passed', 'controls': controls})
    before = digest(path)
    with pytest.raises(ValueError, match='Failed diagnosis'):
        diagnose(SimpleNamespace(root=tmp_path, identity={}))
    assert digest(path) == before
    controls['tcn/known_signal']['relative_mae_gain'] = .8
    assert not control_failures(controls, ('xgboost', 'catboost', 'mlp', 'lstm', 'tcn'))
    for bad in (None, float('nan'), .499):
        controls['tcn/known_signal']['relative_mae_gain'] = bad
        assert control_failures(controls, ('tcn',))
    assert control_failures({}, ('ridge',))
    assert VALIDATION_POLICY == 'all-executed-families-v2'


def test_prepare_rejects_identity_and_policy_changes(packet, monkeypatch):
    folder, reference, market, _, _ = packet
    repo = Path(__file__).parents[2]
    before = digest(folder / 'ready.json')
    clock.prepare(repo, folder, reference, market)
    assert digest(folder / 'ready.json') == before
    registration = read_record(folder / 'preregistered.json')
    changed = copy.deepcopy(registration)
    changed['design']['availability_assumption'] = 'verified'
    changed['design_id'] = content_id(changed['design'])
    with pytest.raises(ValueError, match='design changed'):
        clock.validate_design(changed)
    monkeypatch.setattr(clock, 'research_source_identity', lambda _: {'source_id': 'changed'})
    with pytest.raises(ValueError, match='namespace'):
        clock.prepare(repo, folder, reference, market)
    market.write_bytes(market.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum'):
        clock.prepare(repo, folder, reference, market)


def test_resume_payload_validation_report_and_independent_price_calculation(packet, monkeypatch):
    folder, _, _, _, _ = packet
    exp = Experiment(folder)
    calls = []

    def inner(experiment, recipe, fold):
        calls.append((recipe['horizon'], fold['year']))
        if len(calls) == 2:
            raise FitBudgetReached('synthetic interruption')
        return {'recipe': recipe, 'iterations': 1, 'weight': .5, 'inner_score': 1.}

    monkeypatch.setattr(path_pilot, 'inner_price', inner)
    monkeypatch.setattr(path_pilot, 'predict_chunks', lambda exp, spec, origins, role, iterations: np.full(len(origins), .001))
    assert clock.run(exp, 30)['status'] == 'planned_pause'
    assert clock.compare(folder, 20)['status'] == 'pending'
    assert clock.run(exp, 30)['saved_outputs'] == 32
    before = {p: digest(p) for p in (folder / 'clock-outputs').glob('*.json')}
    result = clock.compare(folder, 20)
    assert len(pd.read_csv(folder / 'reports/outer-predictions.csv')) == 8024
    assert clock.compare(folder, 20) == result
    assert all(digest(p) == sha for p, sha in before.items())
    rows = pd.read_csv(folder / 'reports/outer-predictions.csv')
    for h in (1, 5):
        for group in clock.GROUPS:
            part = rows.loc[rows.horizon.eq(h) & rows.group.eq(group)]
            actual_price = part.cotton_close * np.exp(part.actual_return)
            expected = (actual_price - part.cotton_close * np.exp(part.predicted_return)).abs().mean()
            assert result['horizons'][str(h)]['arms'][group]['selected']['price_mae'] == pytest.approx(expected)
    assert len(calls) == 33  # first complete selection is reused after the pause
    assert clock.run(exp, 30)['status'] == 'complete'
    assert len(calls) == 33
    path = next(iter(before))
    record = read_record(path)
    record['records'][0]['target_date'] = '2020-01-01'
    path.unlink()  # synthetic fixture only: valid checksum but invalid semantic target
    freeze_record(path, record)
    with pytest.raises(ValueError, match='Unmatched'):
        clock.run(exp, 30)


def test_sequence_preserves_all_498_evaluation_origins_with_history():
    history = pd.DataFrame(np.ones((600, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    history['date'] = pd.bdate_range('2010-01-01', periods=600)
    history['target_return_1'], history['target_return_5'] = .01, .02
    inputs, targets = sequences(history.tail(498), None, history=history)
    assert inputs.shape == (498, 60, len(FEATURE_NAMES)) and len(targets) == 498


def test_budget_cannot_expand_and_decision_does_not_use_t5(tmp_path, monkeypatch):
    completed = tmp_path / 'ledger/completed'
    completed.mkdir(parents=True)
    for i in range(676):
        (completed / f'{i}.json').touch()
    def shared_run(*args, **kwargs):
        kwargs['before_fit']()
    monkeypatch.setattr(path_pilot, 'run', shared_run)
    with pytest.raises(FitBudgetReached, match='676-fit'):
        clock.run(SimpleNamespace(root=tmp_path), 30)
    with pytest.raises(ValueError, match='30 minutes'):
        clock.run(SimpleNamespace(root=tmp_path), 31)
    h = {'arms': {'available': {'selected': {'naive_gain_pct': 5., 'direction_pct': 53.,
        'year_wins': 6, 'versus_naive': {str(b): {'gain_ci_pct': [1, 6]} for b in (20, 60)}}}},
        'selected_vs_control': {str(b): {'difference_ci_95': [.01, .1]} for b in (20, 60)}}
    assert clock.decision(h).startswith('strong_timing')
    h['arms']['available']['selected']['naive_gain_pct'] = 1.
    assert clock.decision(h).startswith('timing_contribution')
    h['selected_vs_control']['60']['difference_ci_95'][0] = -.1
    assert clock.decision(h).startswith('inconclusive')
    for b in (20, 60):
        h['arms']['available']['selected']['versus_naive'][str(b)]['gain_ci_pct'][1] = 4.9
    assert clock.decision(h).startswith('timing_does_not_rescue')
