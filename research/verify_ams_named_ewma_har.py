"""Independent zero-fit source/label/QLIKE/interval audit of the AMS control."""

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.research.ledger import read_record

from research.ams_upland_futures import sha


def observed(path):
    ranked = defaultdict(set)
    values = defaultdict(dict)
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['source_date_matches'] != 'True':
                continue
            day, contract = row['date'], row['contract']
            ranked[day].add(contract)
            if row['price_precision_2dp'] == 'True':
                price = Decimal(row['settlement_cents_raw'])
                old = values[day].get(contract)
                if old is not None and abs(old-price) > Decimal('1e-8'):
                    raise ValueError('Conflicting valid source prices')
                values[day][contract] = price
    return ranked, values


def decimal_qlike(actual, forecast):
    with localcontext() as ctx:
        ctx.prec = 34
        a, f = Decimal(str(actual)), Decimal(str(forecast))
        if a < 0 or f <= 0:
            raise ValueError('Invalid QLIKE inputs')
        return f.ln()+a/f


def run(root, output):
    root = Path(root)
    if output.exists():
        raise ValueError('Preserve prior independent verification')
    ready = read_record(root/'ready.json')['identity']
    report = read_record(root/'reports/result.json')
    frame = pd.read_csv(root/'reports/paired-losses.csv', float_precision='round_trip')
    if (report['paired_losses_sha256'] != sha(root/'reports/paired-losses.csv')
            or len(frame) != 2006 or int(frame.eligible.sum()) != 1989
            or report['native_market_fits'] != 174 or report['new_fits_during_report'] != 0
            or ready['design']['2024_plus_used_for_selection']
            or report['source_clock_verified'] or report['independent_holdout']):
        raise ValueError('Frozen report/cohort contract differs')
    original = pd.read_parquet(root/'inputs/history.parquet').set_index('date')
    ranked, values = observed(root/'inputs/observations.csv')
    if sha(root/'inputs/observations.csv') != ready['inputs_sha256']['observations.csv']:
        raise ValueError('Source observations changed')
    label_count = 0
    loss_count = 0
    for row in frame.itertuples():
        stored = original.loc[pd.Timestamp(row.date)]
        if (row.target_date != stored.target_date_1.strftime('%Y-%m-%d')
                or bool(row.eligible) != bool(stored.named_eligible and stored.named_ewma_eligible)):
            raise ValueError(f'Original ordinal/eligibility mismatch: {row.date}')
        if not row.eligible:
            continue
        contract = sorted(ranked[row.date])[1]
        if (contract != row.contract or contract not in values[row.date]
                or contract not in values[row.target_date]):
            raise ValueError('Origin second contract or target quote differs')
        origin = values[row.date][contract]
        target = values[row.target_date][contract]
        actual = math.log(float(target/origin)) ** 2
        if not math.isclose(actual, row.named_actual, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Independent named variance differs')
        if not math.isclose(row.baseline, stored.named_ewma, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Independent reference state differs')
        selected = row.selected_weight*row.raw+(1-row.selected_weight)*row.baseline
        if not math.isclose(selected, row.selected, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Past-selected mixture differs')
        label_count += 1
        for arm in ('baseline', 'raw', 'selected'):
            expected = decimal_qlike(row.named_actual, getattr(row, arm))
            if not math.isclose(float(expected), getattr(row, arm+'_loss'),
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Independent Decimal QLIKE differs: {row.date} {arm}')
            loss_count += 1
    bootstrap_path = root/'inputs/bootstrap.py'
    loader = importlib.util.spec_from_file_location('frozen_bootstrap_audit', bootstrap_path)
    bootstrap = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(bootstrap)
    valid = frame.loc[frame.eligible]
    for arm in ('raw', 'selected'):
        a, b = 'baseline_loss', arm+'_loss'
        saved = report['comparisons'][arm]
        offset = max(0., 1-min(valid[a].min(), valid[b].min()))
        parts = []
        for _, year in frame.groupby('year'):
            weight = year.eligible.to_numpy(int)
            parts.append(np.column_stack([np.where(weight, year[a].fillna(0)+offset, 0),
                                          np.where(weight, year[b].fillna(0)+offset, 0), weight]))
        for block in (20, 60):
            ci = bootstrap.weighted_bootstrap(parts, block=block)['difference_ci_95']
            np.testing.assert_allclose(ci, saved['gain_ci'][str(block)], rtol=0, atol=1e-12)
        gain = float((valid[a]-valid[b]).mean())
        yearly = {str(int(year)): float((group[a]-group[b]).mean())
                  for year, group in valid.groupby('year')}
        trim = valid.drop(valid.nlargest(math.ceil(len(valid)*.01), a).index)
        if (not math.isclose(gain, saved['baseline_minus_candidate_gain'], abs_tol=1e-12)
                or yearly != saved['yearly_gain']
                or not math.isclose(float((trim[a]-trim[b]).mean()),
                                    saved['trim_top1pct_gain'], abs_tol=1e-12)):
            raise ValueError('Independent aggregation or trim differs')
    result = {'profile': 'ams-named-ewma-har-independent-v1', 'status': 'verified',
              'new_fits': 0, 'source_label_pairs': label_count,
              'decimal_qlike_losses': loss_count, 'bootstrap_intervals': 4,
              'native_fit_copies_from_report': 174,
              'source_clock_verified': False, 'independent_holdout': False,
              'report_sha256': sha(root/'reports/result.json'),
              'paired_losses_sha256': sha(root/'reports/paired-losses.csv'),
              'analysis_sha256': sha(__file__)}
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
