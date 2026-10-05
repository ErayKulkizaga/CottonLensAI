import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research import trading_diagnostic as diagnostic
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.research.trading_diagnostic import (
    match_market,
    pnl_metrics,
    summarize,
)


def fixture():
    market = pd.DataFrame({'date': pd.to_datetime(['2020-06-12', '2020-06-15', '2020-06-16']),
        'series': 'cotton', 'open': [60., 61., 62.], 'close': [60., 62., 61.]})
    predictions = pd.DataFrame({'date': ['2020-06-12', '2020-06-15'],
        'target_date': ['2020-06-15', '2020-06-16'], 'horizon': 1, 'cotton_close': [60., 62.],
        'actual_return': np.log([62 / 60, 61 / 62]), 'decision_time':
        ['2020-06-13T00:15:00Z', '2020-06-16T00:15:00Z'],
        'predicted_return': [0.01, -0.01], 'raw_predicted_return': [0.02, -0.02]})
    return predictions, market


def test_weekend_next_recorded_observation_and_separate_unattainable_component():
    predictions, market = fixture()
    joined = match_market(predictions, market)
    assert joined.proxy_entry_open.tolist() == [61., 62.]
    assert joined.open_to_close_cents.tolist() == [1., -1.]
    assert joined.close_to_close_cents.tolist() == [2., -1.]
    report, positions, pnl = summarize({'control': joined, 'available': joined}, repetitions=20)
    selected = report['strategies']['available_selected']
    assert pnl['available_selected'].tolist() == [500., 500.]
    assert selected['gross_total_usd_equivalent'] == 1000.
    assert selected['break_even_round_trip_cost_usd_equivalent'] == 500.
    assert selected['unattainable_prior_close_entry_gross_usd_equivalent'] == 1500.
    assert selected['prior_close_to_open_component_usd_equivalent'] == 500.
    assert selected['net_pnl_usd'] is None and not report['execution_verified']
    assert positions['flat'].tolist() == [0., 0.]


@pytest.mark.parametrize('change', ['origin', 'target', 'value', 'horizon', 'clock', 'missing_open', 'zero_open', 'duplicate_market'])
def test_bad_comparison_or_price_cannot_silently_shrink_sample(change):
    predictions, market = fixture()
    if change == 'origin':
        predictions.loc[0, 'date'] = '2020-06-11'
    elif change == 'target':
        predictions.loc[0, 'target_date'] = '2020-06-16'
    elif change == 'value':
        predictions.loc[0, 'actual_return'] = 0
    elif change == 'horizon':
        predictions.loc[0, 'horizon'] = 5
    elif change == 'clock':
        predictions.loc[0, 'decision_time'] = '2020-06-13T00:00:00Z'
    elif change in ['missing_open', 'zero_open']:
        market.loc[1, 'open'] = np.nan if change == 'missing_open' else 0.
    else:
        market = pd.concat([market, market.head(1)])
    with pytest.raises(ValueError):
        match_market(predictions, market)


def test_flat_cost_and_initial_equity_drawdown_and_active_reference():
    metrics = pnl_metrics([-100., 0., 50.], [-1., 0., 1.])
    assert metrics['trades'] == 2 and metrics['turnover_one_way_contract_units'] == 4
    assert metrics['break_even_round_trip_cost_usd_equivalent'] == -25.
    assert metrics['max_gross_drawdown_usd_equivalent'] == 100.
    assert pnl_metrics([0., 0.], [0., 0.])['break_even_round_trip_cost_usd_equivalent'] is None
    predictions, market = fixture()
    predictions.loc[0, 'predicted_return'] = 0
    joined = match_market(predictions, market)
    report, positions, _ = summarize({'control': joined, 'available': joined}, repetitions=20)
    assert positions['available_active_long'].tolist() == [0., 1.]
    assert report['strategies']['available_selected']['trades'] == 1
    bad = joined.copy()
    bad.loc[0, 'proxy_entry_open'] = 999.
    with pytest.raises(ValueError, match='Unmatched execution'):
        summarize({'control': joined, 'available': bad}, repetitions=20)


@pytest.mark.parametrize('mutate_input', [False, True])
def test_analyze_uses_clock_namespace_and_checks_inputs_before_completion(tmp_path, monkeypatch, mutate_input):
    predictions, market = fixture()
    predictions['selected_weight'] = 1.
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'ready.json').write_text('{}')
    (source / 'history.parquet').write_bytes(b'frozen history')
    market_path = tmp_path / 'market.parquet'
    market.to_parquet(market_path, index=False)
    design = {'market_sha256': digest(market_path), 'split': {'folds': [{'year': 2020}]}}
    monkeypatch.setattr(diagnostic, 'verify_history', lambda _: ({'identity': {'design_id': 'test', 'design': design}}, None))
    monkeypatch.setattr(diagnostic.clock, 'validate_design', lambda _: design)
    for arm in diagnostic.clock.GROUPS:
        for folder in ['clock-outputs', 'clock-decisions']:
            (source / folder).mkdir(exist_ok=True)
            (source / folder / f'{arm}-t1-year2020.json').write_text('{}')
    calls = []

    def output(folder, name, fold, arm, h, design, history, *, namespace):
        assert namespace == 'clock' and h == 1
        calls.append(arm)
        return {}, predictions.copy()

    monkeypatch.setattr(diagnostic.clock, 'output', output)
    if mutate_input:
        original = diagnostic.summarize

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            (source / 'ready.json').write_text('{"changed":true}')
            return result

        monkeypatch.setattr(diagnostic, 'summarize', changed)
    destination = tmp_path / 'result'
    if mutate_input:
        with pytest.raises(ValueError, match='input changed'):
            diagnostic.analyze(source, market_path, destination, repetitions=20)
        assert not destination.exists()
    else:
        diagnostic.analyze(source, market_path, destination, repetitions=20)
        complete = read_record(destination / 'complete.json')
        assert complete['completed'] and complete['fits'] == 0
        assert complete['files']['decisions.csv'] == digest(destination / 'decisions.csv')
        with pytest.raises(ValueError, match='Never overwrite'):
            diagnostic.analyze(source, market_path, destination, repetitions=20)
    assert calls == ['control', 'available']
