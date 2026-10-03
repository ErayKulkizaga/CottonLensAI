"""Synthetic matched-path and metadata-resume contracts; no market fits."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.full_year import chunks
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record
from cottonlens_ml.research.mirror import Mirror
from cottonlens_ml.research.protocol import full_year_manifest
from cottonlens_ml.research.return_path import (
    PATH_COLUMNS,
    PATH_FEATURES,
    add_return_path,
)


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
    expanded = add_return_path(history)
    budget = 4 * (sum(len(chunks(history, b['origins'])) for f in split['folds'] for b in f['inner'])
        + sum(len(chunks(history, f['origins'])) for f in split['folds']))
    design = {'profile': path_pilot.PROFILE, 'reference_ready_sha256': digest(reference / 'ready.json'),
        'history_sha256': digest(reference / 'history.parquet'), 'feature_data_id': frame_identity(expanded, list(expanded.columns)),
        'feature_helper_sha256': digest(Path(__file__).parents[1] / 'src/cottonlens_ml/research/return_path.py'),
        'groups': {'base': list(FEATURE_NAMES), 'base_plus_return_path': PATH_FEATURES}, 'new_columns': PATH_COLUMNS,
        'recipes': {'family': 'ridge', 'alpha': 10., 'target': 'scaled_log', 'years': None, 'seed': 42, 'cadence': 21, 'window': 1},
        'split': split, 'shrinkage_weights': [0., .25, .5, .75, 1.],
        'fit_budget': {'total': budget, 'annual_outputs': 32},
        'price_gate': {'gain_pct': 5., 'direction_pct': {'1': 53., '5': 55.}, 'year_wins': 6},
        'automatic_release': False, 'audit_2024_used': False}
    registration = {'design_id': content_id(design), 'design': design}
    freeze_record(reference.parent / 'return-path-preregistration.json', registration)
    folder = tmp_path / 'experiment'
    path_pilot.prepare(Path(__file__).parents[2], folder, reference)
    return folder, reference, registration


def test_prepare_binds_reference_features_and_environment(packet):
    folder, reference, _ = packet
    before = digest(folder / 'ready.json')
    path_pilot.prepare(Path(__file__).parents[2], folder, reference)
    assert digest(folder / 'ready.json') == before
    path = reference / 'history.parquet'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum'):
        path_pilot.prepare(Path(__file__).parents[2], folder, reference)


def test_design_cannot_expand_recipe_search(packet):
    _, _, registration = packet
    changed = copy.deepcopy(registration)
    changed['design']['recipes']['alpha'] = 1.
    changed['design_id'] = content_id(changed['design'])
    with pytest.raises(ValueError, match='design changed'):
        path_pilot.validate_design(changed)


def test_interruption_cache_complete_comparison_and_corrupt_marker(packet, monkeypatch):
    folder, _, _ = packet
    experiment = Experiment(folder)
    computed, pause = [], [True]

    def inner(exp, spec, fold):
        if pause[0] and len(computed) == 1:
            raise FitBudgetReached('Synthetic pause')

        def operation(work):
            computed.append((fold['year'], spec['horizon'], len(spec['features'])))
            (work / 'model.txt').write_text('synthetic fixture')
            return {'iterations': 1}

        exp.ledger.run({'recipe': spec, 'year': fold['year'], 'synthetic': True}, operation)
        return {'recipe': spec, 'weight': 0., 'inner_score': 1., 'iterations': 1}

    monkeypatch.setattr(path_pilot, 'inner_price', inner)
    monkeypatch.setattr(path_pilot, 'predict_chunks', lambda exp, spec, origins, role, iterations: np.zeros(len(origins)))
    assert path_pilot.run(experiment, 60)['status'] == 'planned_pause'
    assert len(computed) == 1
    pause[0] = False
    assert path_pilot.run(experiment, 60)['saved_outputs'] == 32
    assert len(computed) == 32
    path_pilot.run(experiment, .00001)
    assert len(computed) == 32
    result = path_pilot.compare(folder, repetitions=100)
    assert result['status'] == 'complete' and result['release_allowed'] is False
    for horizon, h in result['horizons'].items():
        assert h['research_priority_signal'] is False
        assert all(v['naive_mae_gain_pct'] == 0 and not v['price_thresholds_passed'] for v in h['arms'].values())
        for group, arm in h['arms'].items():
            records = [row for marker in (folder / 'path-outputs').glob(f'{group}-t{horizon}-year*.json')
                for row in read_record(marker)['records']]
            expected = 100 * sum(np.sign(row[f'target_return_{horizon}']) == row['past_majority_sign']
                for row in records) / len(records)
            assert arm['majority_direction_pct'] == pytest.approx(expected)
            assert 0 <= arm['majority_direction_pct'] <= 100
    path = next((folder / 'path-outputs').glob('*.json'))
    path.write_bytes(path.read_bytes() + b'bad')
    with pytest.raises(ValueError):
        path_pilot.run(experiment, 60)
    assert len(computed) == 32


def test_catalogue_restores_metadata_without_loading_model_batches(tmp_path):
    local, target, restored = tmp_path / 'local', tmp_path / 'drive', tmp_path / 'restored'
    mirror = Mirror(local, target)
    freeze_record(local / 'reports/result.json', {'status': 'complete'})
    mirror.publish_metadata(['reports/result.json'])
    (local / 'ledger/payloads').mkdir(parents=True)
    (local / 'ledger/payloads/model.bin').write_bytes(b'model bytes')
    mirror.enqueue(['ledger/payloads/model.bin'])
    mirror.flush()
    assert Mirror(restored, target).hydrate_metadata()['restored_packages'] == 1
    assert read_record(restored / 'reports/result.json')['status'] == 'complete'
    assert not (restored / 'ledger').exists()
    assert len(list((restored / 'transfer-downloads').glob('*.zip'))) == 1
    with pytest.raises(ValueError, match='cannot publish'):
        mirror.publish_metadata(['ledger/payloads/model.bin'])
    with pytest.raises(ValueError, match='Unsafe'):
        mirror.hydrate(package_names=['../escape.zip'])


def test_catalogue_rejects_corrupt_or_unsafe_package(tmp_path):
    local, target = tmp_path / 'local', tmp_path / 'drive'
    mirror = Mirror(local, target)
    freeze_record(local / 'ready.json', {'ok': True})
    mirror.publish_metadata(['ready.json'])
    package = next((target / 'packages').glob('*.zip'))
    package.write_bytes(b'broken package')
    with pytest.raises(ValueError, match='Corrupt'):
        Mirror(tmp_path / 'new', target).hydrate_metadata()
    freeze_record(target / 'metadata-catalog/unsafe.json', {'package_sha256': '../escape'})
    with pytest.raises(ValueError, match='Invalid'):
        Mirror(tmp_path / 'other', target).hydrate_metadata()
