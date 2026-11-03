import json
import math
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record

from research import ct_forward_report as report


def synthetic_slots():
    dates = pd.bdate_range('2026-10-30', periods=253)
    return pd.DataFrame({'origin': dates[:-1].strftime('%Y-%m-%d'),
        'target_date': dates[1:].strftime('%Y-%m-%d'), 'horizon': 1, 'published': True,
        'actual_variance': .0002, 'baseline_variance': .0004, 'selected_variance': .0002})


def test_complete_synthetic_positive_is_not_a_market_fit():
    result, pairs = report.score_pairs(synthetic_slots())
    assert result['classification'] == 'FORWARD_STATISTICAL_SUPERIORITY'
    assert result['paired_slots'] == 252 and result['confirmatory_gate_passed']
    assert all(v[0] > 0 for v in result['gain_ci'].values())
    assert result['trim_is_gate'] is False and len(pairs) == 252
    assert not result['economic_or_price_direction_success']


def test_one_missing_slot_blocks_confirmation_without_becoming_a_model_negative():
    f = synthetic_slots(); f.loc[101, 'published'] = False
    result, pairs = report.score_pairs(f)
    assert result['classification'] == 'INCONCLUSIVE_INCOMPLETE_PUBLICATION'
    assert result['required_slots'] == 252 and result['paired_slots'] == 251
    assert result['mean_paired_qlike_gain'] > 0
    assert not result['confirmatory_gate_passed'] and result['gain_ci'] is None
    assert not result['incomplete_result_is_no_signal_evidence']
    assert 101 not in pairs.index  # Retain original slots; do not compress time for inference.


def test_zero_publications_has_no_zero_width_skill_interval_or_zero_signal_claim():
    f = synthetic_slots(); f['published'] = False
    f['baseline_variance'] = f['selected_variance'] = np.nan
    result, pairs = report.score_pairs(f)
    assert result['classification'] == 'INCONCLUSIVE_INCOMPLETE_PUBLICATION'
    assert result['mean_paired_qlike_gain'] is None and result['gain_ci'] is None
    assert pairs.empty and result['missing_or_late_slots'] == 252


def test_equal_forecasts_have_exact_zero_difference_and_no_false_positive():
    f = synthetic_slots(); f['selected_variance'] = f.baseline_variance
    result, pairs = report.score_pairs(f)
    assert result['mean_paired_qlike_gain'] == 0.
    assert pairs.delta.eq(0.).all()
    assert result['gain_ci'] == {'20': [0., 0.], '60': [0., 0.]}
    assert not result['confirmatory_gate_passed']


def test_negative_recipe_and_zero_actual_are_valid_distinct_cases():
    f = synthetic_slots(); f['baseline_variance'], f['selected_variance'] = .0002, .0004
    negative, _ = report.score_pairs(f)
    assert negative['classification'] == 'FIXED_RECIPE_SUPERIORITY_NOT_SUPPORTED'
    f['actual_variance'] = 0.; f['selected_variance'] = .0001
    zero, _ = report.score_pairs(f)
    assert zero['mean_paired_qlike_gain'] > 0


@pytest.mark.parametrize('change', ['drop', 'duplicate', 'reorder', 'horizon', 'target', 'actual', 'variance', 'flag'])
def test_malformed_comparisons_cannot_silently_intersect_or_fill(change):
    f = synthetic_slots()
    if change == 'drop':
        f = f.iloc[1:]
    elif change == 'duplicate':
        f.loc[1, 'origin'] = f.origin.iloc[0]
    elif change == 'reorder':
        f = f.iloc[::-1]
    elif change == 'horizon':
        f.loc[2, 'horizon'] = 5
    elif change == 'target':
        f.loc[2, 'target_date'] = f.origin.iloc[2]
    elif change == 'actual':
        f.loc[2, 'actual_variance'] = -1.
    elif change == 'variance':
        f.loc[2, 'selected_variance'] = -1.
    else:
        f['published'] = 'False'
    with pytest.raises(ValueError):
        report.score_pairs(f)


def test_completeness_probabilities_are_assumptions_not_uptime_or_power():
    values = report.completeness_scenarios()
    assert values['assumptions_only_not_estimated_uptime']
    assert not values['forecast_skill_probability']
    assert values['all252_is_new_operational_gate_not_original_human_risk_gate']
    for row in values['scenarios']:
        lower, upper = row['arbitrary_dependence_complete_probability_bounds']
        assert lower <= row['iid_probability_all252_present'] <= upper
    assert values['scenarios'][2]['iid_probability_all252_present'] < .08


def test_unlocked_study_fails_before_any_market_prediction_or_loss_is_read(tmp_path):
    with (patch.object(report.forward, 'verify', return_value=({}, {})),
          patch.object(report, 'extract', side_effect=AssertionError('No predictions')),
          patch.object(report, 'score_pairs', side_effect=AssertionError('No score')),
          pytest.raises(ValueError, match='No real cohort lock')):
        report.audit(tmp_path/'study', tmp_path/'repo', tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('state', ['calendar', 'future_target'])
def test_calendar_or_future_target_blocks_intermediate_score(tmp_path, state):
    identity = {'source_id': 'fixture'}
    lock = {'identity': identity, 'required_origins': 252, 'first_origin': '2026-10-30', 'locked_at': '2026-10-30T10:00Z'}
    freeze_record(tmp_path/'cohort-lock.json', lock)
    dates = synthetic_slots().origin.tolist(); cal = Mock()
    cal.completion.return_value = report.forward.utc('2027-11-03T19:20Z')
    with (patch.object(report.forward, 'verify', return_value=({'identity': identity}, {'no_partial_score': True})),
          patch.object(report.forward, 'calendar', return_value=cal),
          patch.object(report.forward, 'first_origin', return_value='2026-10-30'),
          patch.object(report.forward, 'slots', return_value=dates[:30] if state == 'calendar' else dates),
          patch.object(report.forward, 'target_date', return_value='2027-11-03'),
          patch.object(report.forward, 'now', return_value=report.forward.utc('2026-10-30T12:00Z')),
          patch.object(report.forward, 'status', side_effect=AssertionError('No publication inspection')),
          pytest.raises(ValueError, match='no partial score|no intermediate score')):
        report.require_end(tmp_path, tmp_path)


def test_output_cannot_overwrite_study_or_existing_evidence(tmp_path):
    existing = tmp_path/'existing'; existing.mkdir()
    with pytest.raises(ValueError, match='new output outside'):
        report.audit(tmp_path/'study', tmp_path, existing)
    with pytest.raises(ValueError, match='new output outside'):
        report.audit(tmp_path/'study', tmp_path, tmp_path/'study/new-output')


def test_study_hashes_expose_input_mutations(tmp_path):
    p = tmp_path/'forecast.json'; p.write_bytes(b'first')
    before = report.study_hashes(tmp_path)
    p.write_bytes(b'changed')
    assert report.study_hashes(tmp_path) != before


def analytic_study(root):
    freeze_record(root/'decision-contract.json', {'no_partial_score': True})
    """Analytic mock records, NOT fitted models, Yahoo data or a real cohort.

    The real source-verifier is deliberately not satisfiable by this fixture;
    tests patch that boundary. No optimizer, Ledger.run or network is called.
    """
    f = report.forward
    past = pd.bdate_range(end='2026-10-29', periods=1000)
    dates = past.append(pd.bdate_range('2026-10-30', periods=253))
    theta = np.array([-1.8, .1, .2, .1, -.2])
    rng = np.random.default_rng(42); close = [100.]; squares = []; ewma = None
    for i in range(len(dates)-1):
        if i < 119:
            r = float(rng.normal(0., .01))
        else:
            x = [math.log(np.mean(squares[-k:]))-math.log(ewma) for k in (1, 5, 22)]
            variance = ewma*math.exp(np.dot([1., *x, math.log(ewma)], theta))
            r = math.sqrt(variance)*(-1 if i % 2 else 1)
        close.append(close[-1]*math.exp(r))
        actual = math.log(close[-1]/close[-2])**2; squares.append(actual)
        ewma = actual if ewma is None else .94*ewma+.06*actual
    prices = pd.DataFrame({'date': dates, 'cotton_close': close})
    initial, origin_prices = prices.iloc[:1000], prices.iloc[:1001]
    body = {'market': 'ICE Futures US Cotton No. 2', 'timezone': 'America/New_York',
            'coverage_start': str(dates[0].date()), 'coverage_end': '2028-12-31',
            'completion_time_local': '14:20', 'cotton_closed_dates': [],
            'review': 'SYNTHETIC_WEEKDAY_CALENDAR_FIXTURE_NOT_OFFICIAL'}
    (root/'inputs').mkdir(parents=True)
    (root/'inputs/calendar.json').write_text(json.dumps(body))
    initial.to_parquet(root/'inputs/seed.parquet', index=False)
    (root/'inputs/initial-receipt.json').write_text(json.dumps({'observed_available_at': '2026-10-30T09:00Z'}))
    identity = {'source_id': 'SYNTHETIC_ANALYTIC_FIXTURE', 'design_id': 'NOT_A_REAL_PREREGISTRATION'}
    lock = {'identity': identity, 'required_origins': 252, 'first_origin': '2026-10-30',
            'locked_at': '2026-10-30T10:00Z'}
    freeze_record(root/'cohort-lock.json', lock)

    def observation(frame, previous):
        last = str(frame.date.iloc[-1].date()); observed = f.decision(last)-pd.Timedelta(minutes=10)
        chart = {'chart': {'error': None, 'result': [{'meta': {'symbol': 'CT=F', 'dataGranularity': '1d',
            'exchangeTimezoneName': 'America/New_York', 'regularMarketTime': int(observed.timestamp())},
            'timestamp': [int(pd.Timestamp(d, tz='America/New_York').timestamp()) for d in frame.date],
            'indicators': {'quote': [{n: frame.cotton_close.tolist() for n in ('open', 'high', 'low', 'close', 'volume')}]}}]}}
        data = json.dumps(chart).encode()
        http = {'http_status': 200, 'request_started_at': (observed-pd.Timedelta(seconds=1)).isoformat(),
                'observed_available_at': observed.isoformat(), 'response_sha256': report.forward.hashlib.sha256(data).hexdigest(),
                'response_bytes': len(data),
                'source_url': 'https://query1.finance.yahoo.com/v8/finance/chart/CT%3DF?range=2y&interval=1d&includePrePost=false'}
        http_bytes = json.dumps(http).encode()
        oid = f.content_id({'source_sha256': http['response_sha256'], 'receipt_sha256': f.hashlib.sha256(http_bytes).hexdigest()})
        directory = root/'observations'/oid; directory.mkdir(parents=True)
        (directory/'chart.json').write_bytes(data); (directory/'http-receipt.json').write_bytes(http_bytes)
        frame.to_parquet(directory/'history.parquet', index=False)
        freeze_record(directory/'receipt.json', {'observation_id': oid, 'previous_observation_id': previous,
            'observed_available_at': observed.isoformat(), 'latest_completed_date': last, 'calendar_id': f.content_id(body),
            'files': {p.name: f.digest(p) for p in directory.iterdir()}})
        return oid

    first_oid = observation(origin_prices, None); observation(prices, first_oid)
    h = f.history(origin_prices); ledger = Ledger(root/'ledger', {**identity, 'cohort_lock_id': f.content_id(lock)})

    def model_record(job):
        train = f.projection(h, job['train_dates']); spec = f.specification(job, train, first_oid)
        spec = {**spec, 'repeat_reason': None}; key = ledger.key(spec)
        directory = root/'ledger/payloads'/key; directory.mkdir(parents=True)
        tr = f.learner.relative(train, True, f.learner.STATE_PROFILE); x = tr[f.NAMES].to_numpy()
        eta = np.column_stack([np.ones(len(tr)), x])@theta
        ratio = tr.variance_target_1.to_numpy()*np.exp(-eta)
        scale = x.std(axis=0); scale = np.where(scale > 1e-12, scale, 1.)
        model = {'features': f.NAMES, 'effective_parameters': 5, 'coef': theta.tolist(),
            'recipe': spec['recipe'], 'fixture_only_not_fitted': True,
            'optimization': {'success': True, 'training_feature_mean': x.mean(axis=0).tolist(),
                             'training_feature_scale': scale.tolist(), 'final_training_qlike': float(np.mean(eta+ratio))}}
        p = directory/'variance-model.json'; p.write_text(json.dumps(model))
        payload = directory.relative_to(ledger.root).as_posix()
        prediction = []
        if job['origins']:
            te = f.projection(h, job['origins']); rel = f.learner.relative(te, False, f.learner.STATE_PROFILE)
            prediction = (te.ct_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(te)), rel[f.NAMES]])@theta)).tolist()
        record = {'experiment_id': key, 'identity': ledger.identity, 'specification': spec,
                  'completed_at': '2026-10-31T00:10Z', 'payload_root': payload,
                  'files': {p.relative_to(ledger.root).as_posix(): f.digest(p)},
                  'result': {'variance_predictions': prediction, 'origins': job['origins'], 'fixture_only_not_fitted': True}}
        freeze_record(ledger.root/'completed'/(key+'.json'), record)
        return record

    jobs = f.annual_jobs(h, '2026-10-30'); records = [model_record(j) for j in jobs]
    origins = [d for j in jobs for d in j['origins']]; rows = f.projection(h, origins)
    predicted = np.array([v for r in records for v in r['result']['variance_predictions']])
    scores = [(float(report.qlike(rows.variance_target_1, w*predicted+(1-w)*rows.ct_ewma).mean()), w) for w in f.WEIGHTS]
    selection = {'year': 2026, 'cutoff': '2026-10-30', 'validation_origins': origins,
                 'fit_ids': [r['experiment_id'] for r in records], 'scores': scores,
                 'weight': min(scores)[1], 'created_at': '2026-10-31T00:12Z', 'future_results_used': False}
    freeze_record(root/'selections/2026.json', selection)
    outer = {'year': 2026, 'role': 'outer', 'bucket': 0, 'cutoff': '2026-10-30',
             'origins': [], 'train_dates': f.mature(h, '2026-10-30')}
    model = model_record(outer); test = f.projection(h, ['2026-10-30'])
    raw = float(f.predict_model(ledger, model, test)[0]); baseline = float(test.ct_ewma.iloc[0]); w = selection['weight']
    forecast = {'origin': '2026-10-30', 'target_date': '2026-11-02', 'horizon': 1,
                'generated_at': '2026-10-31T00:20Z', 'cohort_lock_id': f.content_id(lock),
                'observation_id': first_oid, 'history_sha256': f.frame_identity(origin_prices, list(origin_prices)),
                'fit_id': model['experiment_id'], 'selection_id': f.content_id(selection), 'weight': w,
                'baseline_variance': baseline, 'raw_variance': raw, 'selected_variance': w*raw+(1-w)*baseline}
    directory = root/'forecasts/2026-10-30'; freeze_record(directory/'forecast.json', forecast)
    freeze_record(directory/'publication-receipt.json', {'status': 'TIMELY', 'completed_at': '2026-10-31T00:20:01Z',
                  'forecast_sha256': f.digest(directory/'forecast.json')})
    freeze_record(root/'events/000.json', {'ordinal_slot': 0, 'origin': '2026-10-30', 'previous_event_id': None,
        'kind': 'FORECAST', 'forecast_sha256': f.digest(directory/'forecast.json'),
        'publication_receipt_sha256': f.digest(directory/'publication-receipt.json')})
    return identity


def test_full_synthetic_provenance_path_uses_native_state_and_independent_targets(tmp_path):
    identity = analytic_study(tmp_path)
    with (patch.object(report.forward, 'verify', return_value=({'identity': identity}, {'no_partial_score': True, 'required_origins': 252})),
          patch.object(report.forward, 'now', return_value=report.forward.utc('2028-01-01T12:00Z'))):
        frame, final = report.extract(tmp_path, tmp_path)
    assert len(frame) == 252 and frame.published.sum() == 1
    assert frame.target_date.iloc[0] == '2026-11-02'
    result, pairs = report.score_pairs(frame)
    assert result['classification'] == 'INCONCLUSIVE_INCOMPLETE_PUBLICATION' and len(pairs) == 1
    assert final['latest_completed_date'] >= frame.target_date.iloc[-1]


@pytest.mark.parametrize('defect', ['future_training_source', 'uncertified_gradient'])
def test_self_consistent_metadata_cannot_hide_future_input_or_invalid_training(tmp_path, defect):
    identity = analytic_study(tmp_path)
    for p in (tmp_path/'ledger/completed').glob('*.json'):
        record = read_record(p)
        if record['specification']['role'] != '2026-outer-0':
            continue
        if defect == 'future_training_source':
            record['completed_at'] = '2026-10-31T00:04Z'  # Source was observed00:05.
        else:
            model_path = tmp_path/'ledger'/record['payload_root']/'variance-model.json'
            model = json.loads(model_path.read_bytes()); model['optimization']['success'] = False
            model_path.write_text(json.dumps(model))
            record['files'][model_path.relative_to(tmp_path/'ledger').as_posix()] = report.digest(model_path)
        p.write_text(json.dumps({**record, 'record_id': report.content_id(record)}))
    with (patch.object(report.forward, 'verify', return_value=({'identity': identity}, {'no_partial_score': True, 'required_origins': 252})),
          patch.object(report.forward, 'now', return_value=report.forward.utc('2028-01-01T12:00Z')),
          pytest.raises(ValueError, match='Training source arrived|Uncertified native')):
        report.extract(tmp_path, tmp_path)
