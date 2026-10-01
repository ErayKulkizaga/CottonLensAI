"""Read-only historical diagnosis: past baselines, frozen origins and source evidence."""
import json

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.review import (
    prediction_rows,
    review_predictions,
    sample_learning_curve,
    source_inventory,
    summary,
    verify_history,
    write_review,
)


def frame():
    dates = pd.bdate_range('2010-01-04', periods=900)
    result = pd.DataFrame({'date': dates, 'cotton_session_index': np.arange(len(dates)),
        'cotton_close': 80., 'cotton_volatility_20': .01, 'cotton_ret_1': .001, 'signal': 1.})
    for h in (1, 5):
        result[f'target_return_{h}'] = .001
        result[f'target_date_{h}'] = result.date.shift(-h)
    return result


def saved(history):
    origins = history.date.iloc[700:826].dt.strftime('%Y-%m-%d').tolist()
    fold = {'fold': 1, 'year': 2012, 'origins': origins}
    record = {'recipe': {'family': 'xgboost', 'horizon': 5, 'features': ['signal'],
                         'years': None, 'cadence': 126},
              'origins': origins, 'fold': 1, 'predictions': [.001] * 126}
    return fold, record


def test_baselines_are_past_only_and_do_not_use_evaluation_class_frequency():
    history = frame()
    fold, record = saved(history)
    before = prediction_rows(history, record, fold)
    changed = history.copy()
    changed.loc[changed.date >= fold['origins'][0], 'target_return_5'] = -.2
    after = prediction_rows(changed, record, fold)
    np.testing.assert_array_equal(before.median_return, after.median_return)
    np.testing.assert_array_equal(before.past_majority_sign, after.past_majority_sign)
    assert summary(after)['past_majority_direction_pct'] == 0


def test_unmatured_labels_are_excluded_from_baseline():
    history = frame()
    fold, record = saved(history)
    history.loc[history.index.isin(range(694, 700)), 'target_return_5'] = 1000
    rows = prediction_rows(history, record, fold)
    assert rows.median_return.eq(.001).all()


@pytest.mark.parametrize('corruption', ['origins', 'nonfinite', 'shape'])
def test_saved_predictions_must_match_frozen_origins(corruption):
    history = frame()
    fold, record = saved(history)
    if corruption == 'origins':
        record['origins'] = record['origins'][::-1]
    elif corruption == 'nonfinite':
        record['predictions'][0] = float('nan')
    else:
        record['predictions'].pop()
    with pytest.raises(ValueError):
        prediction_rows(history, record, fold)


def test_review_preserves_experiment_and_never_reads_model_payload(tmp_path, monkeypatch):
    history = frame()
    fold, record = saved(history)
    path = tmp_path / 'outer/base/xgboost-t5-fold1.json'
    freeze_record(path, record)
    before = path.read_bytes()
    ready = {'identity': {'split': {'folds': [fold]}}}
    original = pd.read_parquet
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: pytest.fail('Unexpected payload/history read'))
    result = review_predictions(tmp_path, ready, history)
    assert path.read_bytes() == before
    assert result['winner_selected'] is False
    assert result['all_model_payloads_verified'] is False
    assert result['candidates'][0]['overall']['model_mae'] == 0
    monkeypatch.setattr(pd, 'read_parquet', original)
    altered = json.loads(path.read_text())
    altered['predictions'][0] = .99
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='Corrupt'):
        review_predictions(tmp_path, ready, history)


def test_history_checksum_and_audit_boundary(tmp_path):
    history = frame()
    history.to_parquet(tmp_path / 'history.parquet')
    freeze_record(tmp_path / 'ready.json', {'identity': {}, 'history_sha256': digest(tmp_path / 'history.parquet')})
    verify_history(tmp_path)
    (tmp_path / 'history.parquet').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        verify_history(tmp_path)


def test_content_calendar_does_not_admit_a_source(tmp_path):
    evidence = tmp_path / 'ams.json'
    evidence.write_text(json.dumps({'first_report_date': '2020-01-02', 'last_report_date': '2023-12-29'}))
    audit = tmp_path / 'readiness.json'
    freeze_record(audit, {'evidence': {'ams_content': {'path': 'ams.json', 'sha256': digest(evidence)}},
        'sources': [{'source': 'AMS Cotton Spot Quotations', 'model_eligible': False,
                     'status': 'blocked_publication'}], 'newly_admitted_external_groups': 0})
    history = frame()
    result = source_inventory(tmp_path, audit, history)
    assert result['pilot_status'].startswith('blocked')
    assert result['sources'][0]['eligible_origin_count'] == 0
    assert result['sources'][0]['content_calendar_potential']['model_eligible'] is False
    evidence.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        source_inventory(tmp_path, audit, history)


def test_zero_naive_error_is_not_infinite_gain_and_review_is_idempotent(tmp_path):
    history = frame()
    history.target_return_5 = 0.
    fold, record = saved(history)
    record['predictions'] = [0.] * 126
    assert summary(prediction_rows(history, record, fold))['mae_gain_pct'] is None
    body = {'evidence': 'synthetic', 'predictions': {'candidates': []},
            'source_inventory': {'pilot_status': 'blocked', 'pilot_blocker': 'evidence'}}
    a = write_review(body, tmp_path)
    assert write_review(body, tmp_path) == a


def test_learning_curve_lookup_matches_engine_identity_and_checks_payload(tmp_path):
    history = frame()
    fold, record = saved(history)
    fold['inner'] = [{'origins': history.date.iloc[500:563].dt.strftime('%Y-%m-%d').tolist()}]
    spec = {**record['recipe'], 'seed': 42}
    ready = {'identity': {'split': {'folds': [fold]}}}
    freeze_record(tmp_path / 'decisions/ablate-expanded_availability/xgboost-t5-fold1.json',
                  {'chosen': {'seeds': [{'recipe': spec}]}})
    curve = tmp_path / 'ledger/payloads/sample/curves.json'
    curve.parent.mkdir(parents=True)
    curve.write_text(json.dumps({'train': {'price_mae': [1., .8]},
                                 'validation': {'price_mae': [1.1, .9]}}))

    class CaptureFit:
        def run(self, specification, operation, *, repeat=None, before_compute=None):
            # Capture the real engine's specification without calling its training lambda.
            specification = {**specification, 'repeat_reason': repeat}
            key = content_id({'identity': ready['identity'], 'specification': specification})
            freeze_record(tmp_path / 'ledger/completed' / (key + '.json'),
                {'identity': ready['identity'], 'specification': specification,
                 'files': {'payloads/sample/curves.json': digest(curve)},
                 'result': {'iterations': 2, 'details': {'splits': 1},
                            'training_metrics': {'mae': .8}, 'metrics': {'mae': .9}}})
    experiment = object.__new__(Experiment)
    experiment.history, experiment.identity, experiment.ledger = history, ready['identity'], CaptureFit()
    validation = rows_at(history, fold['inner'][0]['origins'])
    train = experiment.train_rows(validation.date.min(), spec)
    early_validation = train.tail(63)
    early_train = experiment.train_rows(early_validation.date.min(), spec)
    experiment.fit(spec, early_train, early_validation, early_validation, 'early-stop-1-0')
    result = sample_learning_curve(tmp_path, ready, history)
    assert result['iterations_selected'] == 2
    assert result['curves']['validation']['price_mae'] == [1.1, .9]
    curve.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        sample_learning_curve(tmp_path, ready, history)
