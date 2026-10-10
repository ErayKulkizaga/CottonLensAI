import copy
import sys

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import named_label_control as pilot
from cottonlens_ml.research.protocol import Preprocessor


@pytest.fixture
def data():
    dates = pd.bdate_range('2020-01-01', periods=1000)
    h = pd.DataFrame({'date': dates, 'cotton_session_index': np.arange(1000),
                      'cotton_close': 80., 'target_return_5': .01,
                      'target_date_5': pd.Series(dates).shift(-5)})
    for name in pilot.RECIPE['features']:
        h[name] = np.sin(np.arange(1000) / 19)
    lab = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'),
                        'target_date': h.target_date_5.dt.strftime('%Y-%m-%d'),
                        'rank': 'second', 'horizon': 5, 'selected_contract': 'Dec 24',
                        'origin_reference_price': 70., 'target_price': 72.,
                        'target_quote_available_at': h.target_date_5.dt.tz_localize('UTC') + pd.Timedelta(days=1),
                        'origin_quote_available_at': dates.tz_localize('UTC') + pd.Timedelta(days=1),
                        'contemporaneous_log_return': np.log(72/70),
                        'contemporaneous_assumption_eligible': h.target_date_5.notna(),
                        'contemporaneous_reason': 'eligible_under_assumption'})
    return h, lab


def test_nonzero_filtered_csv_indexes_cannot_misalign(data):
    h, lab = data
    lab.index = np.arange(len(lab)) * 4 + 3
    expanded = pilot.attach(h, lab)
    assert len(pilot.mature_dates(expanded, h.date.iloc[800])) == 800 - 5 - 119


@pytest.mark.parametrize('field', ['date', 'target_date'])
def test_target_origin_mismatch_stops(data, field):
    h, lab = data
    lab.loc[10, field] = '2030-01-01'
    with pytest.raises(ValueError, match='Exact original'):
        pilot.attach(h, lab)


def test_source_clock_equality_allowed_later_excluded(data):
    h, lab = data
    decision = h.date.iloc[500].tz_localize('UTC') + pd.Timedelta(days=1, minutes=15)
    lab.loc[500, 'origin_quote_available_at'] = decision
    pilot.attach(h, lab)
    lab.loc[500, 'origin_quote_available_at'] = decision + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match='origin clock'):
        pilot.attach(h, lab)


def test_label_maturity_requires_source_and_five_session_purge(data):
    h, lab = data
    expanded = pilot.attach(h, lab)
    cutoff = h.date.iloc[800]
    d = h.date.iloc[700].strftime('%Y-%m-%d')
    clock = cutoff.tz_localize('UTC') + pd.Timedelta(days=1, minutes=15)
    expanded.loc[700, 'named_label_available_at'] = clock
    assert d in pilot.mature_dates(expanded, cutoff)
    expanded.loc[700, 'named_label_available_at'] = clock + pd.Timedelta(seconds=1)
    assert d not in pilot.mature_dates(expanded, cutoff)
    assert h.date.iloc[795].strftime('%Y-%m-%d') not in pilot.mature_dates(expanded, cutoff)


def test_label_only_projection_preserves_features_and_preprocessing(data):
    h, lab = data
    expanded = pilot.attach(h, lab)
    dates = pilot.mature_dates(expanded, h.date.iloc[800])
    ct, named = [pilot.projection(expanded, dates, arm) for arm in pilot.ARMS]
    pd.testing.assert_frame_equal(ct.drop(columns='target_return_5'), named.drop(columns='target_return_5'))
    assert not np.array_equal(ct.target_return_5, named.target_return_5)
    assert Preprocessor.fit(ct, pilot.RECIPE['features']).as_dict() == Preprocessor.fit(named, pilot.RECIPE['features']).as_dict()


def test_missing_target_receipt_kept_without_imputation(data):
    h, lab = data
    lab.loc[900, 'contemporaneous_assumption_eligible'] = False
    lab.loc[900, 'target_price'] = np.nan
    lab.loc[900, 'contemporaneous_log_return'] = np.nan
    expanded = pilot.attach(h, lab)
    dates = h.date.iloc[899:902].dt.strftime('%Y-%m-%d').tolist()
    a = pilot.projection(expanded, dates, 'ct', test=True)
    b = pilot.projection(expanded, dates, 'named', test=True)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 3 and np.isnan(a.target_return_5.iloc[1])


def test_future_label_change_does_not_change_past_training(data):
    h, lab = data
    original = pilot.attach(h, lab)
    lab.loc[900:, 'target_price'] = 90.
    lab.loc[900:, 'contemporaneous_log_return'] = np.log(90/70)
    revised = pilot.attach(h, lab)
    dates = pilot.mature_dates(original, h.date.iloc[800])
    assert dates == pilot.mature_dates(revised, h.date.iloc[800])
    pd.testing.assert_frame_equal(pilot.projection(original, dates, 'named'), pilot.projection(revised, dates, 'named'))


def test_sparse_inner_cadence_uses_cotton_boundary_not_next_valid_row(data):
    h, _ = data
    origins = h.date.iloc[600:665].dt.strftime('%Y-%m-%d').tolist()
    origins.pop(21)
    origins.pop(21)
    # Supply maturity metadata; no fitting happens in this test.
    _, lab = data
    expanded = pilot.attach(h, lab)
    jobs = pilot.jobs(expanded, origins, 'inner_0')
    assert len(jobs) == 4
    assert jobs[1]['cutoff'] == h.date.iloc[621].strftime('%Y-%m-%d')
    assert jobs[1]['origins'][0] == h.date.iloc[623].strftime('%Y-%m-%d')


def test_year_selection_never_uses_outer_price_values(data):
    h, lab = data
    expanded = pilot.attach(h, lab)
    origins = h.date.iloc[850:975].dt.strftime('%Y-%m-%d').tolist()
    split = {'folds': [{'year': 2023, 'origins': origins}]}
    first = pilot.plan(expanded, split)
    changed = copy.deepcopy(expanded)
    changed.loc[850:, 'named_target_price'] *= 10
    changed.loc[850:, 'named_return'] += np.log(10)
    assert pilot.plan(changed, split) == first
    assert len(first['folds']) == 1 and first['gate_evaluation_allowed'] is False


@pytest.mark.parametrize('field,value', [('origin_reference_price', 0.), ('target_price', -1.),
                                       ('contemporaneous_assumption_eligible', 'False')])
def test_bad_prices_or_boolean_encoding_rejected(data, field, value):
    h, lab = data
    lab[field] = lab[field].astype(object)
    lab.loc[500, field] = value
    with pytest.raises((ValueError, TypeError)):
        pilot.attach(h, lab)


@pytest.fixture
def frozen(data, tmp_path, monkeypatch):
    h, lab = data
    repo, folder = tmp_path / 'repo', tmp_path / 'registration'
    (repo / 'ml').mkdir(parents=True)
    (repo / 'ml/uv.lock').write_text('test-lock')
    (folder / 'inputs').mkdir(parents=True)
    h.to_parquet(folder / 'inputs/history.parquet', index=False)
    lab.to_csv(folder / 'inputs/labels.csv', index=False)
    split = {'folds': [{'year': 2023, 'origins': h.date.iloc[850:975].dt.strftime('%Y-%m-%d').tolist()}]}
    pilot.freeze_record(folder / 'inputs/parent-ready.json', {'identity': {'split': split}})
    expanded = pilot.attach(pd.read_parquet(folder / 'inputs/history.parquet'), pd.read_csv(folder / 'inputs/labels.csv'))
    expanded.to_parquet(folder / 'history.parquet', index=False)
    design = pilot.plan(expanded, split)
    identity = {'profile': pilot.PROFILE, 'source_id': 'test-source', 'python': sys.version.split()[0],
                'versions': {}, 'lock_sha256': pilot.digest(repo / 'ml/uv.lock'),
                'inputs_sha256': {p.name: pilot.digest(p) for p in (folder / 'inputs').iterdir()},
                'design': design, 'design_id': pilot.content_id(design),
                'research_data_id': pilot.frame_identity(expanded, list(expanded))}
    pilot.freeze_record(folder / 'ready.json', {'identity': identity, 'history_sha256': pilot.digest(folder / 'history.parquet')})
    monkeypatch.setattr(pilot, 'research_source_identity', lambda _: {'source_id': 'test-source'})
    return repo, folder


@pytest.mark.parametrize('file', ['inputs/labels.csv', 'history.parquet'])
def test_corrupt_input_or_cache_is_rejected(frozen, file):
    repo, folder = frozen
    pilot.verify(folder, repo)
    with (folder / file).open('ab') as f:
        f.write(b'altered')
    with pytest.raises(ValueError, match='changed'):
        pilot.verify(folder, repo)


def test_code_identity_change_rejects_cache(frozen, monkeypatch):
    repo, folder = frozen
    monkeypatch.setattr(pilot, 'research_source_identity', lambda _: {'source_id': 'new-source'})
    with pytest.raises(ValueError, match='Source/profile'):
        pilot.verify(folder, repo)


def test_partial_preparation_is_not_completed_or_overwritten(frozen):
    repo, registration = frozen
    folder = registration.parent / 'partial'
    folder.mkdir()
    (folder / 'old.pending').write_text('keep')
    decision = registration.parent / 'contract.json'
    ready = pilot.read_record(registration / 'ready.json')
    pilot.freeze_record(decision, ready['identity']['design'])
    (repo / 'research/evidence').mkdir(parents=True)
    pilot.freeze_record(repo / 'research/evidence/named-label-control-preregistration-20261010.json',
                        {'ready_sha256': pilot.digest(registration / 'ready.json'),
                         'decision_contract_sha256': pilot.digest(decision), 'new_fits': 0})
    with pytest.raises(ValueError, match='Preserve partial'):
        pilot.prepare(repo, folder, registration, decision)
    assert (folder / 'old.pending').read_text() == 'keep'
    assert not (folder / 'ready.json').exists()
