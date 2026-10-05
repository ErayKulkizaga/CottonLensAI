"""Synthetic contracts; no market training, GPU, network or backend database."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import recency
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.full_year import chunks, price_recipes
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record
from cottonlens_ml.research.protocol import full_year_manifest, rows_at


@pytest.fixture
def packet(tmp_path):
    reference = tmp_path / 'packet/reference'
    reference.mkdir(parents=True)
    dates = pd.bdate_range('2010-01-01', '2023-12-29')
    history = pd.DataFrame({'date': dates, 'cotton_close': 80 * np.exp(np.sin(np.arange(len(dates)) / 40) / 10),
        'cotton_session_index': np.arange(len(dates))})
    for feature in FEATURE_NAMES:
        history[feature] = np.sin(np.arange(len(dates)) / 20)
    for h in (1, 5):
        history[f'target_date_{h}'] = history.date.shift(-h)
        history[f'target_return_{h}'] = np.log(history.cotton_close.shift(-h) / history.cotton_close)
    split = full_year_manifest(history)
    history.to_parquet(reference / 'history.parquet', index=False)
    freeze_record(reference / 'ready.json', {'identity': {'split': split}, 'history_sha256': digest(reference / 'history.parquet')})
    controls = {}
    for fold in split['folds']:
        for h in (1, 5):
            name = f't{h}-year{fold["year"]}.json'
            selected = {'recipe': price_recipes(h)[0], 'weight': 0., 'inner_score': 1., 'iterations': 1}
            decision = {'price': selected}
            freeze_record(reference / 'decisions/full-year' / name, decision)
            frame = rows_at(history, fold['origins']).copy()
            frame['predicted_return'] = 0.
            frame['date'] = frame.date.dt.strftime('%Y-%m-%d')
            freeze_record(reference / 'pilot-full-year' / name, {'year': fold['year'], 'horizon': h,
                'decision_id': content_id(decision), 'inner_score': 1.,
                'records': frame[['date', 'cotton_close', f'target_return_{h}', 'predicted_return']].to_dict('records')})
            controls[name] = digest(reference / 'pilot-full-year' / name)
    budget = 2 * (6 * sum(len(chunks(history, b['origins'])) for f in split['folds'] for b in f['inner'])
        + sum(len(f['inner']) for f in split['folds']) + sum(len(chunks(history, f['origins'])) for f in split['folds']))
    design = {'profile': recency.PROFILE, 'history_calendar_years': 3, 'horizons': [1, 5], 'split': split,
        'recipes': {str(h): [{**r, 'years': 3} for r in price_recipes(h)] for h in (1, 5)},
        'control_retraining_allowed': False, 'control_ready_sha256': digest(reference / 'ready.json'),
        'control_outputs_sha256': controls, 'history_sha256': digest(reference / 'history.parquet'),
        'fit_budget': {'maximum_total_new_fits': budget}, 'evidence': 'synthetic contract fixture'}
    design['design_id'] = content_id(design)
    (reference.parent / 'recency-preregistration.json').write_text(json.dumps(design))
    folder = tmp_path / 'experiment'
    recency.prepare(Path(__file__).parents[2], folder, reference)
    return folder, reference, design


def test_preparation_is_idempotent_and_checks_preserved_control(packet):
    folder, reference, design = packet
    before = digest(folder / 'ready.json')
    recency.prepare(Path(__file__).parents[2], folder, reference)
    assert digest(folder / 'ready.json') == before
    recency.verify_controls(folder / 'controls', design)
    path = next((reference / 'pilot-full-year').glob('*.json'))
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='checksum'):
        recency.prepare(Path(__file__).parents[2], folder, reference)


def test_three_calendar_year_cutoff_purge_and_future_invariance(packet):
    folder, _, design = packet
    experiment = Experiment(folder)
    spec = design['recipes']['5'][0]
    cutoff = pd.Timestamp('2019-06-01')
    before = experiment.train_rows(cutoff, spec)
    assert before.date.min() >= cutoff - pd.DateOffset(years=3)
    assert (before.target_date_5 < cutoff).all()
    changed = experiment.history.copy()
    changed.loc[changed.date >= cutoff, ['cotton_close', 'target_return_5']] = 1e6
    experiment.history = changed
    pd.testing.assert_frame_equal(before, experiment.train_rows(cutoff, spec))


def test_design_rejects_an_extra_history_or_recipe_axis(packet):
    _, _, design = packet
    changed = copy.deepcopy(design)
    changed['recipes']['5'][0]['years'] = 5
    changed.pop('design_id')
    changed['design_id'] = content_id(changed)
    with pytest.raises(ValueError, match='three-year recipes'):
        recency.validate_design(changed)


def test_interruption_resume_report_and_corrupt_output(packet, monkeypatch):
    folder, _, _ = packet
    experiment = Experiment(folder)
    computed, pause = [], [True]

    def inner(exp, spec, fold):
        assert spec['years'] == 3
        if pause[0] and len(computed) == 1:
            raise FitBudgetReached('Synthetic interruption before another operation')

        def operation(work):
            computed.append((fold['year'], spec['horizon'], spec['family'], spec['params']))
            (work / 'model.txt').write_text('synthetic fixture')
            return {'iterations': 1}

        exp.ledger.run({'recipe': spec, 'year': fold['year'], 'synthetic': True}, operation)
        return {'recipe': spec, 'weight': 0., 'inner_score': 1., 'iterations': 1}

    monkeypatch.setattr(recency, 'inner_price', inner)
    monkeypatch.setattr(recency, 'predict_chunks', lambda exp, spec, origins, role, iterations: np.zeros(len(origins)))
    assert recency.run(experiment, 60)['status'] == 'planned_pause'
    assert len(computed) == 1 and not list((folder / 'recency-outputs').glob('*.json'))
    pause[0] = False
    assert recency.run(experiment, 60)['saved_outputs'] == 16
    assert len(computed) == 96  # Six recipes, eight years, two horizons. First operation was reused.
    count = len(computed)
    assert recency.run(experiment, .00001)['control_fits'] == 0
    assert len(computed) == count
    report = recency.compare(folder, repetitions=100)
    assert report['status'] == 'complete' and report['control_fits'] == 0
    assert all(v['naive_mae_gain_pct'] == 0 and not v['price_thresholds_passed'] for v in report['horizons'].values())
    path = next((folder / 'recency-outputs').glob('*.json'))
    path.write_bytes(path.read_bytes() + b'broken')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        recency.run(experiment, 60)
    assert len(computed) == count


def test_completed_markers_cannot_rewrite_a_frozen_decision(packet, monkeypatch):
    folder, _, _ = packet
    experiment = Experiment(folder)
    monkeypatch.setattr(recency, 'inner_price', lambda exp, spec, fold:
        {'recipe': spec, 'weight': 0., 'inner_score': 1., 'iterations': 1})
    monkeypatch.setattr(recency, 'predict_chunks', lambda exp, spec, origins, role, iterations: np.zeros(len(origins)))
    recency.run(experiment, 60)
    path = next((folder / 'recency-decisions').glob('*.json'))
    decision = read_record(path)
    decision['selected']['weight'] = .5
    path.unlink()  # Temporary synthetic fixture only.
    freeze_record(path, decision)
    with pytest.raises(ValueError, match='frozen decision'):
        recency.run(experiment, 60)
