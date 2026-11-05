"""Synthetic causality and execution guards; no market fitting in CI."""
import copy
import hashlib
import importlib.metadata
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import contract_curve as curve
from cottonlens_ml.research import contract_curve_execution as execute
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record
from cottonlens_ml.research.protocol import Preprocessor, rows_at


def report(date='2023-01-05', displayed=None, price=71.):
    return {'report_date': date, 'publication_local_naive': displayed or date + 'T13:00:00',
            'document_sha256': hashlib.sha256(date.encode()).hexdigest(), 'quotes': {'Mar-23': 70., 'May-23': price},
            'first_contract': 'Mar-23', 'second_contract': 'May-23', 'front_second_gap_months': 2,
            'published_at': None, 'timezone_verified': False, 'first_vintage_verified': False,
            'model_eligible': False}


def test_cutoff_weekend_and_expiry():
    h = pd.DataFrame({'date': pd.to_datetime(['2023-01-05', '2023-01-06', '2023-01-09',
                                             '2023-01-10', '2023-01-11'])})
    a = curve.align(h, [report()])
    assert a.numeric_D0_spread.notna().tolist() == [True, True, True, True, False]
    assert a.numeric_D1_spread.notna().tolist() == [False, True, True, True, False]
    late = curve.align(h, [report(displayed='2023-01-09T23:59:00')])
    assert late.numeric_D0_spread.notna().tolist() == [False, False, True, True, True]
    for d in (0, 1):
        assert (a[f'curve_D{d}_assumed_available_at'].dropna() <= a.curve_decision_time.loc[
            a[f'curve_D{d}_assumed_available_at'].notna()]).all()
        np.testing.assert_array_equal(a[f'mask_D{d}_spread'].isna(), a[f'numeric_D{d}_spread'].isna())


def test_late_older_report_never_overwrites_newer():
    h = pd.DataFrame({'date': pd.bdate_range('2023-01-05', periods=6)})
    old = report(displayed='2023-01-10T13:00:00', price=74.)
    new = report('2023-01-06', price=72.)
    new['document_sha256'] = 'b' * 64
    a = curve.align(h, [old, new])
    assert a.curve_D0_document_sha256.iloc[3] == 'b' * 64
    assert a.numeric_D0_spread.iloc[3] == np.log(72 / 70)


@pytest.mark.parametrize('field,value', [('published_at', '2023-01-06T00:00:00Z'),
                                       ('timezone_verified', True), ('first_vintage_verified', True),
                                       ('model_eligible', True), ('front_second_gap_months', 1)])
def test_false_source_admission_or_contract_gap_rejected(field, value):
    r = report()
    r[field] = value
    with pytest.raises(ValueError):
        curve.align(pd.DataFrame({'date': pd.bdate_range('2023-01-05', periods=5)}), [r])


@pytest.mark.parametrize('price', [0, -1, np.nan, np.inf])
def test_bad_quote_rejected(price):
    with pytest.raises(ValueError):
        curve.align(pd.DataFrame({'date': pd.bdate_range('2023-01-05', periods=5)}), [report(price=price)])


def test_no_future_quote_or_label_feature_leak():
    h = pd.DataFrame({'date': pd.bdate_range('2023-01-05', periods=9), 'cotton_close': 70.,
                      'target_return_1': .01})
    r = [report(), report('2023-01-11')]
    a = curve.align(h, r)
    mutated = copy.deepcopy(r)
    mutated[1]['quotes']['May-23'] = 79.
    b = curve.align(h, mutated)
    pd.testing.assert_frame_equal(a.loc[h.date < '2023-01-11'], b.loc[h.date < '2023-01-11'], check_exact=True)
    assert not a.numeric_D0_spread.equals(b.numeric_D0_spread)
    h['target_return_1'] = 999.
    h['cotton_close'] = 999.
    pd.testing.assert_frame_equal(a.drop(columns=['cotton_close', 'target_return_1']),
                                 curve.align(h, r).drop(columns=['cotton_close', 'target_return_1']), check_exact=True)
    with pytest.raises(ValueError):
        curve.align(h.iloc[::-1], r)
    with pytest.raises(ValueError):
        curve.align(h, r + [r[0]])


def test_missing_provenance_never_removes_prediction():
    h = pd.DataFrame({'date': pd.to_datetime(['2023-01-05']), 'target_date_1': pd.to_datetime(['2023-01-06']),
                      'target_return_1': [.01], 'cotton_close': [70.]})
    test = curve.align(h, [report('2023-01-06')])
    payload = pd.DataFrame({'date': ['2023-01-05'], 'cotton_close': [70.],
                            'predicted_return': [.001], 'raw_predicted_return': [.002]})
    payload = execute.record_frame(test, payload, 'numeric_D0', 1, {'weight': .5})
    records = path_pilot.json_predictions(payload)
    assert len(records) == 1 and records[0]['predicted_return'] == .001
    assert records[0]['gap_months'] is None and records[0]['document_sha256'] is None


def test_source_decision_does_not_replace_primary_with_delay():
    arms = {'numeric_D0': {'selected': {'versus_naive': {str(b): {'gain_ci_pct': [-1., 4.9]} for b in (20, 60)}}}}
    paired = {f'selected_D{d}': {str(b): {'difference_ci_95': [.01, .1]} for b in (20, 60)} for d in (0, 1)}
    assert execute.decision(arms, paired) == 'CONDITIONAL_SOURCE_CONTRIBUTION_GATE_UNTESTED'
    paired['selected_D0']['60']['difference_ci_95'][0] = -.01
    assert execute.decision(arms, paired) == 'FIXED_RECIPE_BELOW_PRACTICAL_GOAL_SOURCE_INCONCLUSIVE'


@pytest.fixture
def design():
    counts = {2016: 250, 2017: 251, 2018: 251, 2019: 252, 2020: 253, 2021: 252, 2022: 251, 2023: 251}
    dates = pd.DatetimeIndex(np.concatenate([pd.bdate_range(f'{y}-01-01', f'{y}-12-31')[:counts.get(y, 250)]
                                            for y in range(2010, 2024)]))
    i = np.arange(len(dates))
    h = pd.DataFrame({'date': dates, 'cotton_session_index': i, 'cotton_close': 80. + np.sin(i / 10)})
    for j, name in enumerate(FEATURE_NAMES):
        h[name] = np.sin(i / (j + 2))
    for horizon in (1, 5):
        h[f'target_return_{horizon}'] = np.log(h.cotton_close.shift(-horizon) / h.cotton_close)
        h[f'target_date_{horizon}'] = h.date.shift(-horizon)
    records = [report(str(d.date()), price=71. + np.sin(j))
               for j, d in enumerate(dates[dates >= '2020-01-01'])]
    expanded, spec, coverage = curve.make_design(h, records)
    assert [f['year'] for f in spec['split']['folds']] == [2021, 2022, 2023]
    assert spec['rules']['practical_goal']['year_wins'] == '6/8'
    assert not spec['split']['gate_evaluation_allowed']
    assert spec['fit_budget']['total'] == 252 and spec['fit_budget']['prediction_rows'] == 2996
    for year in range(2016, 2020):
        assert all(r['train_reports'] == 0 for r in coverage if r['year'] == year)
    for d in (0, 1):
        empty = expanded.loc[expanded.date < '2020-01-01']
        a = Preprocessor.fit(empty, curve.names('mask', d)).transform(empty)
        b = Preprocessor.fit(empty, curve.names('numeric', d)).transform(empty)
        np.testing.assert_array_equal(a, b)
    pd.testing.assert_frame_equal(expanded[list(h)], h, check_exact=True)
    return expanded, spec


def test_outer_labels_do_not_choose_years(design):
    h, expected = design
    parent = h[[c for c in h if not c.startswith(('curve_', 'mask_', 'numeric_'))]].copy()
    parent.loc[parent.date >= '2021-01-01', 'target_return_1'] = 99.
    records = [report(str(d.date()), price=71. + np.sin(j))
               for j, d in enumerate(parent.date[parent.date >= '2020-01-01'])]
    _, changed, _ = curve.make_design(parent, records)
    assert [f['origins'] for f in changed['split']['folds']] == [f['origins'] for f in expected['split']['folds']]


def test_resume_matching_metadata_and_t1_only(design, tmp_path, monkeypatch):
    history, spec = design
    calls = []

    def select(exp, recipe, fold):
        calls.append((recipe['horizon'], fold['year']))
        if len(calls) == 2:
            raise FitBudgetReached('Synthetic pause before compute')
        return {'recipe': recipe, 'weight': .5, 'iterations': 1, 'inner_score': 1.}

    monkeypatch.setattr(execute, 'verify_execution', lambda folder: (None, history))
    monkeypatch.setattr(execute, 'learning_control', lambda exp: None)
    monkeypatch.setattr(path_pilot, 'inner_price', select)
    monkeypatch.setattr(path_pilot, 'predict_chunks', lambda exp, recipe, dates, role, count: np.full(len(dates), .001))
    exp = SimpleNamespace(root=tmp_path, history=history,
                          identity={'design': spec, 'design_id': content_id(spec), 'split': spec['split']},
                          train_rows=lambda cutoff, recipe: history.loc[history.target_date_5 < cutoff])
    assert execute.run(exp, 30)['status'] == 'planned_pause'
    assert execute.run(exp, 30)['saved_outputs'] == 12
    assert len(calls) == 13 and all(h == 1 for h, _ in calls)
    execute.run(exp, 30)
    assert len(calls) == 13
    fold = spec['split']['folds'][0]
    _, frame = execute.output(tmp_path, f'mask_D0-t1-year{fold["year"]}.json', fold, 'mask_D0', 1, spec, history)
    assert frame.selected_weight.eq(.5).all()
    assert frame.known_spread.eq(1).all()
    target = rows_at(history, fold['origins']).copy()
    bad = execute.record_frame(target, frame.copy(), 'mask_D0', 1, {'weight': .5})
    bad.loc[0, 'target_date'] = '1900-01-01'
    with pytest.raises(ValueError, match='Unmatched'):
        execute.clock.require_same_targets(frame, bad)
    with pytest.raises(ValueError):
        execute.recipe(spec, 'numeric_D0', 5)
    with pytest.raises(ValueError):
        execute.run(exp, 31)


def test_budget_and_failed_synthetic_cache_block(design, tmp_path, monkeypatch):
    history, spec = design
    monkeypatch.setattr(execute, 'verify_execution', lambda folder: (None, history))
    monkeypatch.setattr(execute, 'learning_control', lambda exp: None)
    monkeypatch.setattr(execute, 'fit_consumption', lambda root: (spec['fit_budget']['total'], set()))
    exp = SimpleNamespace(root=tmp_path, identity={'design': spec, 'design_id': content_id(spec)}, history=history)

    def attempt(*args, **kwargs):
        kwargs['before_fit']()

    monkeypatch.setattr(path_pilot, 'run', attempt)
    with pytest.raises(FitBudgetReached):
        execute.run(exp, 30)
    freeze_record(tmp_path / 'learning-control-ledger/attempts/failed.json', {'experiment_id': 'failed'})
    monkeypatch.undo()
    with pytest.raises(ValueError, match='attempt already consumed'):
        execute.learning_control(exp)


def test_registered_inputs_code_environment_and_incomplete_cache(design, tmp_path, monkeypatch):
    expanded, spec = design
    repo, registration, folder = [tmp_path / n for n in ('repo', 'registration', 'execution')]
    (repo / 'ml').mkdir(parents=True)
    (repo / 'ml/uv.lock').write_text('Synthetic lock')
    (repo / 'research/evidence').mkdir(parents=True)
    (registration / 'inputs').mkdir(parents=True)
    parent = expanded[[c for c in expanded if not c.startswith(('curve_', 'mask_', 'numeric_'))]]
    parent.to_parquet(registration / 'inputs/reference.parquet', index=False)
    records = [report(str(d.date()), price=71. + np.sin(j))
               for j, d in enumerate(parent.date[parent.date >= '2020-01-01'])]
    (registration / 'inputs/quotes.json').write_text(json.dumps({'model_eligible': False, 'rows': records}))
    proof = {'private_quote_payload_sha256': digest(registration / 'inputs/quotes.json'),
             'reports': len(records), 'model_eligible_rows': 0}
    (registration / 'inputs/source-proof.json').write_text(json.dumps(proof))
    expanded.to_parquet(registration / 'history.parquet', index=False)
    source = {'source_id': content_id({}), 'files': {}}
    freeze_record(registration / 'source-manifest.json', source)
    identity = {'design': spec, 'source_id': source['source_id'], 'python': sys.version.split()[0],
                'versions': {n: importlib.metadata.version(n) for n in curve.VERSIONS},
                'lock_sha256': digest(repo / 'ml/uv.lock'),
                'input_sha256': {p.name: digest(p) for p in (registration / 'inputs').iterdir()}}
    reg = {'profile': curve.PROFILE, 'identity': identity, 'registration_id': content_id(identity),
           'completed_experiment': False, 'fits': 0, 'availability_verified': False, 'model_eligible': False}
    freeze_record(registration / 'preregistered.json', reg)
    contract = registration / 'decision-contract.json'
    freeze_record(contract, {'registration_id': reg['registration_id'], 'rules': spec['rules'],
                            'completed_experiment': False, 'fits': 0, 'automatic_release': False})
    freeze_record(registration / 'complete.json', {'completed': True, 'fits': 0, 'model_eligible': False,
                  'registration_id': reg['registration_id'],
                  'files': {p.relative_to(registration).as_posix(): digest(p) for p in registration.rglob('*') if p.is_file()}})
    evidence = {'preregistered_sha256': digest(registration / 'preregistered.json'),
                'complete_sha256': digest(registration / 'complete.json'),
                'decision_contract_sha256': digest(contract), 'registration_id': reg['registration_id'],
                'source_inventory_sha256': digest(registration / 'inputs/source-proof.json')}
    (repo / 'research/evidence' / execute.EVIDENCE).write_text(json.dumps(evidence))
    monkeypatch.setattr(execute, 'validate', lambda root: None)
    monkeypatch.setattr(execute, 'research_source_identity', lambda root: source)
    monkeypatch.setattr(execute, 'load_registry', lambda root: ({}, {}))
    monkeypatch.setattr(execute, 'check', lambda *a, **kw: {'exact_fit_recipes': [], 'fit_recipes': [], 'status': 'synthetic'})
    ready = execute.prepare(repo, folder, registration, contract)
    assert execute.prepare(repo, folder, registration, contract) == ready
    monkeypatch.setattr(execute, 'research_source_identity', lambda root: {'source_id': 'changed', 'files': {}})
    with pytest.raises(ValueError, match='Code changed'):
        execute.prepare(repo, folder, registration, contract)
    monkeypatch.setattr(execute, 'research_source_identity', lambda root: source)
    monkeypatch.setattr(execute.importlib.metadata, 'version', lambda name: 'changed-version')
    with pytest.raises(ValueError, match='environment changed'):
        execute.prepare(repo, folder, registration, contract)
    monkeypatch.undo()
    (folder / 'preregistration/inputs/quotes.json').write_text('{}')
    with pytest.raises(ValueError, match='payload changed'):
        execute.verify_execution(folder)
    with pytest.raises(FileNotFoundError):
        execute.verify_execution(tmp_path / 'incomplete')
    bad = {**evidence, 'decision_contract_sha256': '0' * 64}
    with pytest.raises(ValueError, match='differs from registered'):
        execute.verify_registration(registration, contract, bad)


def test_registration_transient_writer_never_enters_complete_manifest(design, tmp_path, monkeypatch):
    history, _ = design
    parent = history[[c for c in history if not c.startswith(('curve_', 'mask_', 'numeric_'))]]
    repo, output = tmp_path / 'repo', tmp_path / 'registration'
    (repo / 'ml').mkdir(parents=True)
    (repo / 'ml/uv.lock').write_text('Synthetic lock')
    (repo / 'research/evidence').mkdir(parents=True)
    history_path = tmp_path / 'history.parquet'
    parent.to_parquet(history_path, index=False)
    quotes = tmp_path / 'quotes.json'
    records = [report(str(d.date()), price=71. + np.sin(j))
               for j, d in enumerate(parent.date[parent.date >= '2020-01-01'])]
    quotes.write_text(json.dumps({'model_eligible': False, 'rows': records}))
    proof = repo / 'research/evidence/information-admission-20261010.json'
    proof.write_text(json.dumps({'private_quote_payload_sha256': digest(quotes),
                                 'model_eligible_rows': 0, 'reports': len(records)}))
    for name in ('registry', 'trials'):
        (repo / f'research/{name}.json').write_text('{"schema":1,"records":[]}')
    monkeypatch.setattr(curve, 'REFERENCE_HISTORY_SHA256', digest(history_path))
    reg = curve.register(repo, output, history_path, quotes, proof, curve.MODE)
    complete = execute.completed_files(output)
    assert complete['registration_id'] == reg['registration_id']
    assert not any('.writer-lock' in name for name in complete['files'])
    assert not (output / '.writer-lock').exists()
    with pytest.raises(ValueError, match='Preserve previous'):
        curve.register(repo, output, history_path, quotes, proof, curve.MODE)
