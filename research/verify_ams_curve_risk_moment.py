"""Independent zero-fit source and moment audit for the fixed front slope."""

import argparse
import csv
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from research.ams_upland_futures import sha


def run(root, output):
    if output.exists():
        raise ValueError('Preserve prior verifier result')
    contract = json.loads((root/'contract.json').read_text(encoding='utf-8'))
    result = json.loads((root/'result.json').read_text(encoding='utf-8'))
    report = Path(contract['report_path'])
    observations = Path(contract['observations_path'])
    if (sha(report) != contract['report_sha256']
            or sha(observations) != contract['observations_sha256']
            or result['moments_sha256'] != sha(root/'moments.csv')
            or result['contract_sha256'] != sha(root/'contract.json')
            or result['screen_passed'] or result['new_fits'] != 0):
        raise ValueError('Frozen zero-fit input/result differs')
    ranked = defaultdict(set)
    values = defaultdict(dict)
    with observations.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['source_date_matches'] == 'True':
                ranked[row['date']].add(row['contract'])
                if row['price_precision_2dp'] == 'True':
                    values[row['date']][row['contract']] = Decimal(row['settlement_cents_raw'])
    source = pd.read_csv(report, float_precision='round_trip')
    moment = pd.read_csv(root/'moments.csv', float_precision='round_trip')
    if (len(source) != len(moment) or len(moment) != 2006
            or not source.date.equals(moment.date)
            or int(moment.moment_weight.sum()) != result['eligible_origins']):
        raise ValueError('Ordinal calendar or source coverage differs')
    checked = 0
    for source_row, row in zip(source.itertuples(), moment.itertuples(), strict=True):
        names = sorted(ranked[row.date])
        usable = (source_row.eligible and len(names) >= 2
                  and names[0] in values[row.date] and names[1] in values[row.date])
        if bool(row.moment_weight) != bool(usable):
            raise ValueError('Front curve availability differs')
        if not usable:
            if row.front_slope != 0 or row.relative_error != 0:
                raise ValueError('Missing source has nonzero contribution')
            continue
        if source_row.contract != names[1]:
            raise ValueError('Origin second contract changed')
        x = math.log(float(values[row.date][names[0]]/values[row.date][names[1]]))
        y = source_row.named_actual/source_row.baseline-1
        if (not math.isclose(x, row.front_slope, rel_tol=1e-12, abs_tol=1e-14)
                or not math.isclose(y, row.relative_error, rel_tol=1e-12, abs_tol=1e-14)):
            raise ValueError('Independent source or EWMA error differs')
        checked += 1
    eligible = moment.loc[moment.moment_weight.eq(1)]
    annual = {}
    for year, group in eligible.groupby('year'):
        x = group.front_slope.to_numpy()
        y = group.relative_error.to_numpy()
        annual[str(int(year))] = float(np.mean(x*y)-np.mean(x)*np.mean(y))
    for row in result['yearly']:
        if not math.isclose(annual[str(row['year'])], row['covariance'], abs_tol=1e-12):
            raise ValueError('Independent within-year covariance differs')
    pooled = float(np.average(list(annual.values()), weights=[r['eligible'] for r in result['yearly']]))
    for block in (20, 60):
        if not math.isclose(pooled, result['bootstrap'][str(block)]['covariance'], abs_tol=1e-12):
            raise ValueError('Pooled point statistic differs')
    out = {'profile': 'ams-front-slope-independent-v1', 'status': 'verified', 'new_fits': 0,
           'source_scalar_pairs': checked, 'yearly_covariances': len(annual),
           'pooled_covariance': pooled, 'report_sha256': sha(root/'result.json'),
           'moments_sha256': sha(root/'moments.csv'),
           'analysis_sha256': sha(__file__),
           'bootstrap_interval_scope': 'frozen algorithm, source/point independently checked',
           'model_skill_established': False, 'source_clock_verified': False}
    output.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
