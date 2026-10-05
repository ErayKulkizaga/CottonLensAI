"""No-fit T+1 open/close proxy diagnostic; never certifies execution or net skill."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research import availability_clock as clock
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.review import verify_history
from cottonlens_ml.research.statistics import paired_bootstrap

VERSION = 'clock-t1-open-close-proxy-v1'
MULTIPLIER = 500.  # USD per cent/lb for a 50,000 lb contract equivalent.


def match_market(predictions, market):
    """Match every origin and next recorded Cotton observation; no intersection/fill."""
    if not {'date', 'series', 'open', 'close'}.issubset(market):
        raise ValueError('Cotton OHLC market schema required')
    cotton = market.loc[market.series.eq('cotton')].copy().sort_values('date')
    cotton['date'] = pd.to_datetime(cotton.date)
    if cotton.empty or cotton.date.isna().any() or cotton.date.duplicated().any():
        raise ValueError('Unique Cotton market dates required')
    frame = predictions.copy().reset_index(drop=True)
    clock.require_same_targets(frame, frame)
    if not frame.horizon.eq(1).all():
        raise ValueError('This diagnostic is T+1 only')
    frame['date'] = pd.to_datetime(frame.date)
    frame['target_date'] = pd.to_datetime(frame.target_date)
    indexes = pd.Index(cotton.date).get_indexer(frame.date)
    if (indexes < 0).any() or (indexes + 1 >= len(cotton)).any():
        raise ValueError('Every origin needs its next recorded Cotton observation')
    origin = cotton.iloc[indexes].reset_index(drop=True)
    target = cotton.iloc[indexes + 1].reset_index(drop=True)
    if (not np.array_equal(target.date, frame.target_date)
            or not np.array_equal(origin.close, frame.cotton_close)):
        raise ValueError('Frozen origin/target date or price differs from raw market')
    values = target[['open', 'close']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Every matched entry/exit requires a positive finite price; no filling')
    actual = np.log(target.close.to_numpy() / frame.cotton_close.to_numpy())
    if not np.allclose(actual, frame.actual_return, atol=1e-12, rtol=0):
        raise ValueError('Frozen realized target differs from raw market')
    decision = pd.to_datetime(frame.decision_time, utc=True)
    expected = pd.to_datetime(frame.date, utc=True) + pd.Timedelta(days=1, minutes=15)
    if not decision.equals(expected):
        raise ValueError('Frozen decision clock differs from the 00:15 contract')
    frame['proxy_entry_open'] = target.open.to_numpy()
    frame['proxy_exit_close'] = target.close.to_numpy()
    frame['close_to_close_cents'] = target.close.to_numpy() - frame.cotton_close.to_numpy()
    frame['open_to_close_cents'] = target.close.to_numpy() - target.open.to_numpy()
    frame['prior_close_to_open_cents'] = target.open.to_numpy() - frame.cotton_close.to_numpy()
    frame['execution_verified'] = False
    return frame


def pnl_metrics(pnl, positions):
    pnl, positions = np.asarray(pnl, float), np.asarray(positions, float)
    if (pnl.shape != positions.shape or pnl.ndim != 1 or not len(pnl)
            or not np.isfinite([pnl, positions]).all() or not np.isin(positions, [-1, 0, 1]).all()):
        raise ValueError('Finite aligned PnL and fixed-unit positions required')
    trades = int(np.count_nonzero(positions))
    equity = np.r_[0., np.cumsum(pnl)]
    return {'decision_days': len(pnl), 'trades': trades, 'active_rate_pct': 100 * trades / len(pnl),
        'gross_total_usd_equivalent': float(pnl.sum()), 'gross_mean_per_decision_usd_equivalent': float(pnl.mean()),
        'gross_mean_per_trade_usd_equivalent': float(pnl.sum() / trades) if trades else None,
        'break_even_round_trip_cost_usd_equivalent': float(pnl.sum() / trades) if trades else None,
        'max_gross_drawdown_usd_equivalent': float(np.max(np.maximum.accumulate(equity) - equity)),
        'turnover_one_way_contract_units': 2 * trades,
        'net_pnl_usd': None, 'capital_return_pct': None}


def summarize(frames, *, repetitions=10000):
    control, available = (frames[name].reset_index(drop=True) for name in clock.GROUPS)
    clock.require_same_targets(control, available)
    for field in ['proxy_entry_open', 'proxy_exit_close', 'decision_time']:
        if not np.array_equal(control[field], available[field]):
            raise ValueError('Unmatched execution proxy or decision time')
    strategies = {'flat': np.zeros(len(control)), 'always_long': np.ones(len(control)),
                  'always_short': -np.ones(len(control))}
    for arm, frame in [('control', control), ('available', available)]:
        for kind, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
            if not np.isfinite(frame[field]).all():
                raise ValueError('Finite stored predictions required')
            strategies[f'{arm}_{kind}'] = np.sign(frame[field].to_numpy())
        active = np.abs(strategies[f'{arm}_selected'])
        strategies[f'{arm}_active_long'] = active
        strategies[f'{arm}_active_short'] = -active
    movement = control.open_to_close_cents.to_numpy() * MULTIPLIER
    pnl = {name: position * movement for name, position in strategies.items()}
    years = pd.to_datetime(control.date).dt.year.to_numpy()
    result = {'version': VERSION, 'horizon': 1, 'multiplier_usd_per_cent_lb': MULTIPLIER,
        'rule': 'sign(stored selected predicted return); raw sign is diagnostic only',
        'execution_verified': False, 'independent_holdout': False, 'automatic_release': False,
        'entry_time_verified': False,
        'entry_time_limitation': 'Daily open may precede the decision clock; no executable entry is established',
        'costs_known': False, 'scope': 'CT=F daily OHLC contract-equivalent gross sensitivity only',
        'strategies': {}, 'paired_gross_differences': {}}
    for name, position in strategies.items():
        item = pnl_metrics(pnl[name], position)
        item['years'] = {str(year): pnl_metrics(pnl[name][years == year], position[years == year])
                         for year in sorted(set(years))}
        item['positive_gross_years'] = sum(v['gross_total_usd_equivalent'] > 0 for v in item['years'].values())
        if name.endswith(('_selected', '_raw')):
            prior_movement = control.close_to_close_cents.to_numpy() * MULTIPLIER
            item['unattainable_prior_close_entry_gross_usd_equivalent'] = float((position * prior_movement).sum())
            item['prior_close_to_open_component_usd_equivalent'] = float(
                (position * control.prior_close_to_open_cents.to_numpy() * MULTIPLIER).sum())
        result['strategies'][name] = item
    comparisons = [('available_selected', 'control_selected')]
    for arm in clock.GROUPS:
        candidate = f'{arm}_selected'
        comparisons.extend((candidate, ref) for ref in ['flat', 'always_long', 'always_short',
            f'{arm}_active_long', f'{arm}_active_short'])
    for candidate, reference in comparisons:
        parts = [np.column_stack([pnl[candidate][years == year], pnl[reference][years == year]])
                 for year in sorted(set(years))]
        intervals = {}
        for block in (20, 60):
            bootstrap = paired_bootstrap(parts, block=block, repetitions=repetitions, seed=42)
            # Relative-loss power diagnostics do not apply to contract-equivalent PnL.
            intervals[str(block)] = {k: v for k, v in bootstrap.items() if k in (
                'count', 'block_length', 'replicates', 'paired_difference', 'difference_ci_95',
                'difference_ci_method', 'standard_error', 'centered_bootstrap_p_two_sided')}
        result['paired_gross_differences'][f'{candidate}_minus_{reference}'] = intervals
    result['decision'] = 'NO_EXECUTION_OR_NET_SKILL_CLAIM'
    return result, strategies, pnl


def analyze(experiment, market_file, output, *, repetitions=10000):
    experiment, market_file, output = map(Path, (experiment, market_file, output))
    if output.exists():
        raise ValueError('Never overwrite a diagnostic or old experiment')
    ready, history = verify_history(experiment)
    design = clock.validate_design({'design_id': ready['identity']['design_id'], 'design': ready['identity']['design']})
    if digest(market_file) != design['market_sha256']:
        raise ValueError('Market checksum differs from the frozen experiment')
    inputs = {'market.parquet': digest(market_file), 'ready.json': digest(experiment / 'ready.json'),
              'history.parquet': digest(experiment / 'history.parquet')}
    frames = {}
    market = pd.read_parquet(market_file)
    for arm in clock.GROUPS:
        parts = []
        for fold in design['split']['folds']:
            name = f'{arm}-t1-year{fold["year"]}.json'
            _, frame = clock.output(experiment, name, fold, arm, 1, design, history, namespace='clock')
            parts.append(frame)
            for sub in ['clock-outputs', 'clock-decisions']:
                inputs[f'{sub}/{name}'] = digest(experiment / sub / name)
        frames[arm] = match_market(pd.concat(parts, ignore_index=True), market)
    result, positions, pnl = summarize(frames, repetitions=repetitions)
    result['inputs'] = inputs
    rows = frames['control'][['date', 'target_date', 'decision_time', 'cotton_close', 'actual_return',
        'proxy_entry_open', 'proxy_exit_close', 'close_to_close_cents', 'open_to_close_cents',
        'prior_close_to_open_cents', 'execution_verified']].copy()
    for name, position in positions.items():
        rows[name + '_position'] = position
        rows[name + '_gross_usd_equivalent'] = pnl[name]
    for arm in clock.GROUPS:
        for field in ['predicted_return', 'raw_predicted_return', 'selected_weight']:
            rows[f'{arm}_{field}'] = frames[arm][field].to_numpy()
    for name, sha in inputs.items():
        path = market_file if name == 'market.parquet' else experiment / name
        if digest(path) != sha:
            raise ValueError('Frozen input changed during the diagnostic')
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output / 'decisions.csv', index=False, date_format='%Y-%m-%d')
    freeze_record(output / 'report.json', result)
    freeze_record(output / 'complete.json', {'version': VERSION, 'fits': 0, 'inputs': inputs,
        'files': {name: digest(output / name) for name in ['decisions.csv', 'report.json']},
        'execution_verified': False, 'completed': True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--market', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.experiment, args.market, args.output)
    print(result['decision'])


if __name__ == '__main__':
    main()
