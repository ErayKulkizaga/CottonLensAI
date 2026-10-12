"""No-market-fit checks of the execution boundary, cache and decision rules."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import named_label_control as labels
from cottonlens_ml.research import named_weak_execution as pilot
from cottonlens_ml.research import named_weak_signal as weak
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
)
from cottonlens_ml.research.named_weak_review import expected_loss
from cottonlens_ml.research.protocol import Preprocessor, inputs


def metrics(injected=(6, 6, 6), null=(-1, -1, -1)):
    return {f'{condition}-{seed}': {'raw': {'gain_pct': gain}}
            for condition, values in [('injected', injected), ('null', null)]
            for seed, gain in zip(weak.SEEDS, values, strict=True)}


@pytest.mark.parametrize('injected,null,decision', [
    ((5, 6, 7), (0, -1, -2), 'SIMPLE_WEAK_TEACHER_RECOVERY_SUPPORTED'),
    ((4, 3, -1), (0, -1, -2), 'FIXED_LEARNER_WEAK_TEACHER_RECOVERY_INADEQUATE'),
    ((4, 5, 6), (0, -1, -2), 'INCONCLUSIVE_MIXED_SEEDS'),
    ((8, 8, 8), (0, 0.1, -1), 'INCONCLUSIVE_NULL_CONTROL_FAILED'),
])
def test_primary_raw_rules_and_null_failure_precedence(injected, null, decision):
    result = metrics(injected, null)
    for value in result.values():
        value['selected'] = {'gain_pct': 99}
    assert pilot.classify(result) == decision


def test_missing_extra_and_nonfinite_seeds_cannot_be_reported_successful():
    result = metrics()
    del result['null-703']
    with pytest.raises(ValueError, match='six'):
        pilot.classify(result)
    result = metrics()
    result['injected-703']['raw']['gain_pct'] = np.nan
    with pytest.raises(ValueError, match='Finite'):
        pilot.classify(result)


def test_quadrature_reference_discriminates_oracle_and_wrong_direction():
    price, mean, sigma = 80., .017, .035
    baseline = expected_loss(price, mean, sigma, 0.)
    oracle = expected_loss(price, mean, sigma, mean)
    wrong = expected_loss(price, mean, sigma, -mean)
    assert oracle < baseline < wrong
    assert expected_loss(price, 0., sigma, 0.) < expected_loss(price, 0., sigma, mean)


def sample():
    n = 190
    h = pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=n),
                      'cotton_session_index': np.arange(n), 'cotton_close': 80.,
                      'target_date_5': pd.bdate_range('2020-01-08', periods=n),
                      'named_price': 80., 'named_target_price': 81.,
                      'named_return': np.resize([-.04, .02], n),
                      'target_return_5': .01, 'named_eligible': True,
                      'named_label_available_at': pd.bdate_range('2020-01-08', periods=n).tz_localize('UTC') + pd.Timedelta(days=1)})
    for index, name in enumerate(pilot.RECIPE['features']):
        h[name] = np.sin(np.arange(n) * .17 + index) * .03
    h.loc[125, weak.SOURCE] = np.nan
    dates = h.date.dt.strftime('%Y-%m-%d').tolist()
    jobs = []
    for role, pos in [('inner_0', 150), ('outer', 160)]:
        jobs.append({'role': role, 'bucket': 0, 'cutoff': dates[pos],
                     'train_dates': labels.mature_dates(h, dates[pos]), 'origins': dates[pos:pos+2]})
    fold = {'year': 2020, 'inner': [dates[150:152]], 'origins': dates[160:162], 'jobs': jobs}
    calibration = {'source': weak.SOURCE, 'threshold': 0., 'beta': .017, 'sigma': .035}
    return h, fold, calibration


def test_label_clock_rejects_unmatured_or_not_yet_available_label():
    h, fold, _ = sample()
    job = fold['jobs'][0]
    pilot.validate_job(h, job)
    bad = {**job, 'train_dates': [*job['train_dates'], job['origins'][0]]}
    with pytest.raises(ValueError, match='maturity'):
        pilot.validate_job(h, bad)
    changed = h.copy()
    changed.loc[changed.date.eq(pd.Timestamp(job['train_dates'][0])), 'named_label_available_at'] = pd.Timestamp('2029-01-01', tz='UTC')
    with pytest.raises(ValueError, match='maturity'):
        pilot.validate_job(changed, job)


def test_future_labels_and_oracle_noise_never_enter_actual_feature_stream():
    h, fold, calibration = sample()
    original = h.copy(deep=True)
    injected, mu, noise = weak.build(h, calibration, 'injected', 703)
    null, _, _ = weak.build(h, calibration, 'null', 703)
    dates = fold['jobs'][0]['train_dates']
    train = labels.projection(injected, dates, 'named')
    processor = Preprocessor.fit(train, pilot.RECIPE['features'])
    test = labels.projection(injected, fold['origins'], 'named', test=True)
    expected = inputs(h, test, processor)
    np.testing.assert_array_equal(inputs(null, test, processor), expected)
    changed = injected.assign(oracle_mean=mu, unit_overlap_noise=noise)
    changed.loc[changed.date.ge(pd.Timestamp(fold['jobs'][0]['cutoff'])), ['named_return', 'target_return_5']] = 1e6
    np.testing.assert_array_equal(inputs(changed, test, processor), expected)
    pd.testing.assert_frame_equal(labels.projection(changed.drop(columns=['oracle_mean', 'unit_overlap_noise']), dates, 'named'), train)
    assert not {'oracle_mean', 'unit_overlap_noise', 'named_return', 'target_return_5'}.intersection(pilot.RECIPE['features'])
    pd.testing.assert_frame_equal(h, original, check_exact=True)


def test_engine_run_resume_and_corrupt_native_cache_fails_without_recompute(tmp_path, monkeypatch):
    h, fold, calibration = sample()
    fixed = {'folds': [fold], 'calibration': calibration, 'fit_budget': {'total': 12}}
    ready = {'identity': {'design': fixed}}
    monkeypatch.setattr(pilot, 'verify', lambda folder, repo: (ready, h))
    monkeypatch.setattr(pilot, 'research_source_identity', lambda repo: {'source_id': 'frozen'})
    identity = {'source_id': 'frozen', 'design': fixed, 'profile': pilot.PROFILE}
    experiment = SimpleNamespace(root=tmp_path, identity=identity, history=h,
                                 ledger=Ledger(tmp_path / 'ledger', identity))
    operations = []
    paused = [False]

    def fit(recipe, train, validation, test, role, iterations):
        spec = {'recipe': recipe, 'role': role, 'iterations': iterations,
                'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(), 'validation_dates': [],
                'test_dates': test.date.dt.strftime('%Y-%m-%d').tolist(),
                'train_identity': pilot.frame_identity(train, list(train))}

        def operation(workspace):
            operations.append(role)
            # A stub payload exercises persistence without fitting any learner.
            (workspace / 'native-stub.txt').write_text(role)
            return {'origins': spec['test_dates'], 'predictions': [.001] * len(test), 'iterations': iterations}

        def before():
            if len(operations) == 3 and not paused[0]:
                paused[0] = True
                raise FitBudgetReached('Simulated session pause')
            experiment.before_compute()

        return experiment.ledger.run(spec, operation, before_compute=before)

    experiment.fit = fit
    first = pilot.run(experiment, tmp_path, 30)
    assert first['status'] == 'planned_pause' and len(operations) == 3
    assert experiment.history is h
    assert pilot.run(experiment, tmp_path, 30)['status'] == 'complete'
    assert len(operations) == 12 and experiment.history is h
    assert pilot.run(experiment, tmp_path, 30)['status'] == 'complete'
    assert len(operations) == 12
    record = experiment.ledger.results()[0]
    (experiment.ledger.root / next(iter(record['files']))).write_text('tampered')
    with pytest.raises(ValueError, match='Native payload'):
        pilot.run(experiment, tmp_path, 30)
    assert len(operations) == 12 and experiment.history is h


def test_incomplete_payload_cannot_be_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'verify', lambda folder, repo: ({}, None))
    exp = SimpleNamespace(root=tmp_path, ledger=Ledger(tmp_path / 'ledger', {}))
    freeze_record(tmp_path / 'ledger/attempts/failed.json', {'experiment_id': 'failed'})
    assert read_record(tmp_path / 'ledger/attempts/failed.json')['experiment_id'] == 'failed'
    with pytest.raises(ValueError, match='Incomplete'):
        pilot.report(exp, tmp_path)
