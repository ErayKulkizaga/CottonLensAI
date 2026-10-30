"""Independent raw-row, seed-prefix and scalar EWMA checks; no learner imports."""

import argparse
import json
import math
from datetime import datetime, time
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.verify_risk_weak_teacher import fingerprint, read, sha


def scalar_ewma(close):
    values = [math.log(b/a)**2 for a, b in pairwise(close)]
    value = values[0]
    for observation in values[1:]:
        value = .94*value+.06*observation
    return max(value, 1e-12)


def verify(root):
    ready, report = read(root/'ready.json'), read(root/'result.json')
    identity = ready['source']; body = {k: v for k, v in identity.items() if k != 'source_id'}
    if fingerprint(body) != identity['source_id'] or report['source_id'] != identity['source_id']:
        raise ValueError('Source identity changed')
    for name, checksum in {**identity['ml']['files'], **identity['helpers']}.items():
        if sha(root/'source-snapshot'/name) != checksum:
            raise ValueError('Frozen source changed')
    for name, checksum in ready['inputs_sha256'].items():
        if sha(root/'inputs'/name) != checksum:
            raise ValueError('Source input changed')
    for name, checksum in report['outputs_sha256'].items():
        if sha(root/name) != checksum:
            raise ValueError('Source output changed')
    if any(report[k] for k in ('new_fits', 'new_forecasts', 'market_skill', 'cohort_locked', 'forward_authorized')):
        raise ValueError('A source proposal must not claim fitting, skill or forward authorization')
    receipt = json.loads((root/'inputs/chart-receipt.json').read_bytes())
    if receipt['response_sha256'] != sha(root/'inputs/chart.json') or receipt['http_status'] != 200:
        raise ValueError('Wrong capture receipt')
    observed = datetime.fromisoformat(receipt['observed_available_at'])
    calendar = json.loads((root/'inputs/calendar.json').read_bytes())
    ny = ZoneInfo(calendar['timezone']); start = datetime.fromisoformat(calendar['coverage_start']).date()
    end = datetime.fromisoformat(calendar['coverage_end']).date()
    closed = {datetime.fromisoformat(v).date() for v in calendar['cotton_closed_dates']}
    for evidence in calendar['evidence']:
        if sha(root/'inputs'/evidence['file']) != evidence['sha256']:
            raise ValueError('Calendar evidence changed')
    raw = json.loads((root/'inputs/chart.json').read_bytes())['chart']['result'][0]
    rows = pd.read_csv(root/'classified-bars.csv', keep_default_na=False, na_values=[''], float_precision='round_trip')
    completed = pd.read_parquet(root/'completed-quotes.parquet')
    expected_quotes = []
    if len(rows) != len(raw['timestamp']):
        raise ValueError('A raw timestamp was silently removed')
    for i, (stamp, row) in enumerate(zip(raw['timestamp'], rows.to_dict('records'), strict=True)):
        timestamp = datetime.fromtimestamp(stamp, ny); day = timestamp.date()
        q = {k: raw['indicators']['quote'][0][k][i] for k in ('open', 'high', 'low', 'close', 'volume')}
        bound = datetime.combine(day, time(14, 20), ny)
        expected = ('empty_placeholder' if all(v is None for v in q.values()) else
                    'completed_quote' if bound <= observed else 'incomplete_quote')
        if row['date'] != str(day) or row['classification'] != expected:
            raise ValueError('Incorrect raw date or completion classification')
        if q['close'] is not None and row['cotton_close'] != q['close']:
            raise ValueError('Close altered during source classification')
        if expected == 'completed_quote':
            expected_quotes.append((pd.Timestamp(day), q['close']))
    expected_frame = pd.DataFrame(expected_quotes, columns=['date', 'cotton_close'])
    pd.testing.assert_frame_equal(completed, expected_frame, check_exact=True)
    expected_dates = [d for d in pd.date_range(max(start, expected_frame.date.iloc[0].date()), observed.astimezone(ny).date())
                      if d.weekday() < 5 and d.date() not in closed
                      and datetime.combine(d.date(), time(14, 20), ny) <= observed]
    observed_dates = completed.loc[completed.date.ge(pd.Timestamp(start)), 'date'].tolist()
    if observed.astimezone(ny).date() > end or expected_dates != observed_dates:
        raise ValueError('Independent expected Cotton session list differs')
    counts = rows.classification.value_counts().sort_index().to_dict()
    if counts != report['classification_counts']:
        raise ValueError('Row classification count differs')
    old = pd.read_parquet(root/'inputs/seed.parquet')
    common = completed.loc[completed.date.le(old.date.iloc[-1])].reset_index(drop=True)
    old_common = old.loc[old.date.ge(common.date.iloc[0])].reset_index(drop=True)
    pd.testing.assert_series_equal(common.date, old_common.date, check_exact=True)
    changed = common.loc[common.cotton_close.ne(old_common.cotton_close), 'date'].dt.strftime('%Y-%m-%d').tolist()
    checked = 0
    if (root/'proposed-seed.parquet').exists():
        proposed = pd.read_parquet(root/'proposed-seed.parquet')
        if changed:
            decision = json.loads((root/'inputs/initial-version-decision.json').read_bytes())
            if (changed != [str(old.date.iloc[-1].date())] or decision['changed_date'] != changed[0]
                    or decision['original_seed_sha256'] != sha(root/'inputs/seed.parquet')
                    or decision['snapshot_sha256'] != sha(root/'inputs/chart.json')
                    or decision['old_close'] != float(old.cotton_close.iloc[-1])
                    or decision['new_close'] != float(common.cotton_close.iloc[-1])):
                raise ValueError('Initial source version differs from explicit decision')
            expected = pd.concat([old.iloc[:-1], completed.loc[completed.date.ge(old.date.iloc[-1])]], ignore_index=True)
        else:
            expected = pd.concat([old, completed.loc[completed.date.gt(old.date.iloc[-1])]], ignore_index=True)
        pd.testing.assert_frame_equal(proposed, expected, check_exact=True)
        pd.testing.assert_frame_equal(proposed.iloc[:3520], old.iloc[:3520], check_exact=True)
        if report['proposed_seed_sha256'] != sha(root/'proposed-seed.parquet'):
            raise ValueError('Proposed seed hash differs')
        if changed:
            unsafe = pd.concat([old, completed.loc[completed.date.gt(old.date.iloc[-1])]], ignore_index=True)
            mixed, current = scalar_ewma(unsafe.cotton_close.tolist()), scalar_ewma(proposed.cotton_close.tolist())
            diag = report['diagnostic_only_ewma']
            np.testing.assert_allclose([mixed, current, 100*(mixed/current-1)],
                                       [diag['unsafe_mixed_vintage'], diag['proposed_vintage'], diag['relative_difference_pct']],
                                       rtol=1e-10, atol=1e-12)
            checked = len(proposed)-1
    elif not changed or report['status'] != 'REJECTED_REVISION':
        raise ValueError('A strict revision failure was not retained')
    return {'status': 'independently_verified_manual_source_proposal', 'source_id': report['source_id'],
            'new_fits': 0, 'new_forecasts': 0, 'forward_authorized': False,
            'raw_rows_checked': len(rows), 'completed_quotes_checked': len(completed),
            'calendar_sessions_checked': len(expected_dates), 'overlapping_quotes_checked': len(common),
            'changed_close_dates': changed, 'scalar_ewma_updates_checked': checked,
            'historical_prefix_unchanged': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.execution), indent=2))
