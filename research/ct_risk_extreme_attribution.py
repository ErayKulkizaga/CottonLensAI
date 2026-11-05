"""Descriptive attribution of the largest baseline-loss CT origin; no fits."""

import argparse
import json
import math
import shutil
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record

from research.ams_contract_reconciliation import parse_daily


def attribute(frame, history, first, second):
    if len(frame) != 2006 or frame.date.duplicated().any() or set(frame.year) != set(range(2016, 2024)):
        raise ValueError('Preserve exact original2006/eightyear cohort')
    h = history.set_index(history.date.dt.strftime('%Y-%m-%d'))
    dates = frame.date.tolist()
    if frame.target_date.tolist() != h.loc[dates].target_date_1.dt.strftime('%Y-%m-%d').tolist():
        raise ValueError('Target endpoints differ')
    at = frame.loc[frame.baseline_loss.idxmax()]
    quotes = [parse_daily(first, at.date), parse_daily(second, at.target_date)]
    contracts = [next(iter(q)) for q in quotes]
    if contracts[0] != contracts[1]:
        raise ValueError('Witnesses do not label the same front contract')
    with localcontext() as context:
        context.prec = 40
        actual = np.array([float((Decimal(str(h.loc[d, 'cotton_close']))/
                                  Decimal(str(h.loc[o, 'cotton_close']))).ln()**2)
                           for o, d in zip(dates, frame.target_date, strict=True)])
        losses = {}
        for name in ('ewma_variance', 'raw_variance'):
            losses[name] = np.array([float(Decimal(str(p)).ln()+Decimal(str(y))/Decimal(str(p)))
                                     for p, y in zip(frame[name], frame.actual_variance, strict=True)])
        front = [Decimal(q[c]) for q, c in zip(quotes, contracts, strict=True)]
        second_contract = list(quotes[0])[1]
        if list(quotes[1])[1] != second_contract:
            raise ValueError('Second-contract rank differs; do not infer a mapping')
        back = [Decimal(q[second_contract]) for q in quotes]
        front_return, second_return = float((front[1]/front[0]).ln()), float((back[1]/back[0]).ln())
    np.testing.assert_allclose(actual, frame.actual_variance, rtol=1e-10, atol=1e-14)
    for p, day in zip(front, (at.date, at.target_date), strict=True):
        if not math.isclose(float(p), h.loc[day, 'cotton_close'], rel_tol=0, abs_tol=5e-5):
            raise ValueError('Printed front quote does not match held CT close')
    np.testing.assert_allclose(losses['ewma_variance'], frame.baseline_loss, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(losses['raw_variance'], frame.raw_loss, rtol=1e-10, atol=1e-12)
    delta = losses['ewma_variance']-losses['raw_variance']
    position = frame.index.get_loc(at.name)
    if abs(delta.sum()) <= 1e-12 or abs(frame.score.sum()) <= 1e-12:
        raise ValueError('Net contribution share undefined')
    return {'profile': 'ct-largest-risk-loss-attribution-v1', 'new_forecast_fits': 0,
            'original_origins': len(frame), 'focus_rule': 'largest stored EWMA QLIKE loss, descriptive post-result audit',
            'origin': at.date, 'target_date': at.target_date,
            'front_contract_on_both_prints': contracts[0], 'front_prices': [str(v) for v in front],
            'front_log_return': front_return, 'second_contract_on_both_prints': second_contract,
            'second_prices': [str(v) for v in back], 'second_log_return': second_return,
            'held_CT_volumes': [float(h.loc[d, 'cotton_volume']) for d in (at.date, at.target_date)],
            'total_raw_QLIKE_net_gain_sum': float(delta.sum()), 'focus_raw_QLIKE_gain': float(delta[position]),
            'focus_fraction_of_net_raw_gain': float(delta[position]/delta.sum()),
            'focus_fraction_of_net_signed_score': float(at.score/frame.score.sum()),
            'without_focus_raw_mean_gain_descriptive': float(np.delete(delta, position).mean()),
            'decimal_target_rows_checked': len(actual), 'decimal_loss_scalars_checked': 2*len(actual),
            'continuous_contract_switch_established': False, 'source_is_same_publisher': True,
            'exchange_or_first_vintage_verified': False, 'market_skill': False,
            'interpretation': 'Both dated AMS prints label July22. An automatic continuous-roll explanation is not established for this event. The source also does not prove the economic cause or rule out shared vendor revision.',
            'not_done': 'No target deletion, cohort change, retraining, strategy or substitution of second-contract labels'}


def run(analysis, daily_first, daily_second, output):
    if output.exists():
        raise ValueError('Preserve prior attribution')
    frame = pd.read_csv(analysis/'scores.csv', float_precision='round_trip')
    history = pd.read_parquet(analysis/'reference/history.parquet')
    result = attribute(frame, history, daily_first.read_bytes(), daily_second.read_bytes())
    output.mkdir(parents=True)
    for source, name in ((daily_first, 'origin-report.txt'), (daily_second, 'target-report.txt'),
                         (Path(__file__), 'source.py'),
                         (Path(__file__).with_name('ams_contract_reconciliation.py'), 'daily-parser.py')):
        shutil.copyfile(source, output/name)
    result['inputs_sha256'] = {
        'scores.csv': digest(analysis/'scores.csv'), 'history.parquet': digest(analysis/'reference/history.parquet'),
        **{p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    freeze_record(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('analysis', 'daily-first', 'daily-second', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.analysis, args.daily_first, args.daily_second, args.output)))
