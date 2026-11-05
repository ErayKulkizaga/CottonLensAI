"""Offline source admission for manual CT research; never fit or publish forecasts."""

import argparse
import importlib.metadata
import json
import math
import shutil
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, writer

from research.ct_prospective_readiness import append_snapshot, candidate_history, prices

PROFILE = 'ct-manual-source-admission-v1'
NY = ZoneInfo('America/New_York')
HELPERS = ('research/ct_manual_source.py', 'research/verify_ct_manual_source.py',
           'research/tests/test_ct_manual_source.py', 'research/ct_prospective_readiness.py',
           'research/calendars/ice-cotton-2026.json', 'research/verify_risk_weak_teacher.py')
OLD_SEED_SHA = 'b3d6f01b9d8a7383d9e64b1d44bbad764fd65b5d54f67ed4aee23b91debe1fef'
QUALIFICATION = ('Yahoo CT=F provider proxy; named ICE contract, settlement interpretation, '
                 'first historical publication and first vintage are unverified. '
                 'A current snapshot is not a historical availability certificate.')


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tz is None or pd.isna(stamp):
        raise ValueError('An explicit timezone is required for source availability')
    return stamp.tz_convert('UTC')


class CottonCalendar:
    def __init__(self, body):
        self.start = pd.Timestamp(body['coverage_start']).date()
        self.end = pd.Timestamp(body['coverage_end']).date()
        self.closed = {pd.Timestamp(d).date() for d in body['cotton_closed_dates']}
        if (body['market'] != 'ICE Futures US Cotton No. 2'
                or body['timezone'] != 'America/New_York'
                or body['completion_time_local'] != '14:20'
                or self.start > self.end or any(d < self.start or d > self.end for d in self.closed)):
            raise ValueError('Reviewed IFUS Cotton calendar and conservative completion bound required')

    def session(self, day):
        if not self.start <= day <= self.end:
            raise ValueError('Cotton calendar coverage unavailable; do not infer future holidays')
        return day.weekday() < 5 and day not in self.closed

    def completion(self, day):
        return pd.Timestamp(datetime.combine(day, time(14, 20), NY)).tz_convert('UTC')

    def completed(self, first, observed):
        last = observed.tz_convert(NY).date()
        if last > self.end or last < self.start:
            raise ValueError('Receipt outside reviewed Cotton calendar coverage')
        return [d.date() for d in pd.date_range(max(first, self.start), last)
                if self.session(d.date()) and self.completion(d.date()) <= observed]


def verified_response(path, receipt_path):
    receipt = json.loads(receipt_path.read_bytes())
    started, observed = utc(receipt['request_started_at']), utc(receipt['observed_available_at'])
    if (receipt['http_status'] != 200 or receipt['response_sha256'] != digest(path)
            or receipt['response_bytes'] != path.stat().st_size or started > observed):
        raise ValueError('Source HTTP receipt, availability order or payload hash mismatch')
    return receipt


def classify_chart(body, receipt, calendar):
    chart = body['chart']
    if chart.get('error') is not None or len(chart.get('result') or []) != 1:
        raise ValueError('Exactly one successful Yahoo daily series required')
    result = chart['result'][0]; meta = result['meta']; observed = utc(receipt['observed_available_at'])
    if (meta.get('symbol') != 'CT=F' or meta.get('dataGranularity') != '1d'
            or meta.get('exchangeTimezoneName') != 'America/New_York'
            or pd.Timestamp(meta['regularMarketTime'], unit='s', tz='UTC') > observed):
        raise ValueError('Wrong CT proxy, bar frequency, timezone or future market timestamp')
    stamps = result['timestamp']; quotes = result['indicators']['quote']
    if (not stamps or len(quotes) != 1 or any(len(quotes[0][k]) != len(stamps)
            for k in ('open', 'high', 'low', 'close', 'volume'))):
        raise ValueError('Daily OHLCV arrays and timestamps must have identical lengths')
    timestamps = pd.to_datetime(stamps, unit='s', utc=True)
    dates = timestamps.tz_convert(NY).tz_localize(None).normalize()
    if (not timestamps.is_unique or not timestamps.is_monotonic_increasing or timestamps.hasnans
            or not dates.is_unique):
        raise ValueError('Unique chronological daily source timestamps required')
    rows = []
    for i, (stamp, date) in enumerate(zip(timestamps, dates, strict=True)):
        q = {k: quotes[0][k][i] for k in ('open', 'high', 'low', 'close', 'volume')}
        flags = []
        if all(v is None for v in q.values()):
            status = 'empty_placeholder'
        else:
            if (q['close'] is None or isinstance(q['close'], bool)
                    or not math.isfinite(q['close']) or q['close'] <= 0):
                raise ValueError('A nonempty bar requires a positive finite close; no filling or dropping')
            for field, value in q.items():
                if value is not None and (isinstance(value, bool) or not math.isfinite(value)
                        or (value < 0 if field == 'volume' else value <= 0)):
                    raise ValueError('Invalid OHLCV numeric field; preserve and investigate source')
            if stamp > observed:
                raise ValueError('Future priced source timestamp')
            if date.weekday() >= 5 or (calendar.start <= date.date() <= calendar.end
                                      and not calendar.session(date.date())):
                raise ValueError('Priced row on a non-Cotton session')
            if date.date() > calendar.end:
                raise ValueError('Priced row outside reviewed calendar coverage')
            # A close outside traded high/low can be a settlement convention; flag, do not repair.
            if q['low'] is not None and q['high'] is not None and not q['low'] <= q['close'] <= q['high']:
                flags.append('close_outside_traded_range')
            if any(q[k] is None for k in ('open', 'high', 'low', 'volume')):
                flags.append('partial_ohlv')
            status = 'completed_quote' if calendar.completion(date.date()) <= observed else 'incomplete_quote'
        rows.append({'source_timestamp_utc': stamp.isoformat(), 'date': date,
                     'classification': status, 'cotton_close': q['close'], 'flags': '|'.join(flags)})
    frame = pd.DataFrame(rows)
    completed = frame.loc[frame.classification.eq('completed_quote'), ['date', 'cotton_close']]
    expected = calendar.completed(dates[0].date(), observed)
    actual = completed.loc[completed.date.ge(pd.Timestamp(calendar.start)), 'date'].dt.date.tolist()
    if actual != expected:
        raise ValueError('Completed Cotton calendar gap or extra priced date; no silent intersection')
    return frame, prices(completed)


def initial_seed(seed, snapshot, decision, seed_sha, snapshot_sha):
    """A hash-bound proposal for a new pre-cohort vintage; never a general revision bypass."""
    old, new = prices(seed), prices(snapshot)
    a = old.loc[old.date.ge(new.date.iloc[0])].reset_index(drop=True)
    b = new.loc[new.date.le(old.date.iloc[-1])].reset_index(drop=True)
    if not a.date.equals(b.date):
        raise ValueError('Initial seed cannot change overlapping session identities')
    changes = a.loc[a.cotton_close.ne(b.cotton_close)]
    if (decision.get('purpose') != 'PROPOSED_PRE_COHORT_SEED_VERSION'
            or decision.get('original_seed_sha256') != seed_sha
            or decision.get('snapshot_sha256') != snapshot_sha
            or decision.get('cohort_locked') is not False or decision.get('forward_forecasts') != 0
            or len(changes) != 1 or changes.date.iloc[0] != old.date.iloc[-1]
            or decision.get('changed_date') != str(old.date.iloc[-1].date())
            or decision.get('old_close') != float(old.cotton_close.iloc[-1])
            or decision.get('new_close') != float(b.cotton_close.iloc[-1])):
        raise ValueError('Only the explicit sole terminal revision may enter a proposed initial vintage')
    return pd.concat([old.iloc[:-1], new.loc[new.date.ge(old.date.iloc[-1])]], ignore_index=True)


def audit(repo, seed_path, source_dir, decision_path, output):
    if output.exists():
        raise ValueError('Preserve existing completed or partial source evidence; use a new namespace')
    calendar_path = repo/'research/calendars/ice-cotton-2026.json'
    calendar_body = json.loads(calendar_path.read_bytes()); calendar = CottonCalendar(calendar_body)
    paths = {'seed.parquet': seed_path, 'calendar.json': calendar_path,
             'chart.json': source_dir/'yahoo-ct-chart.json',
             'chart-receipt.json': source_dir/'yahoo-ct-chart.json.receipt.json'}
    if decision_path:
        paths['initial-version-decision.json'] = decision_path
    for source in calendar_body['evidence']:
        path = source_dir/source['file']; receipt_path = path.with_name(path.name+'.receipt.json')
        if digest(path) != source['sha256']:
            raise ValueError('Reviewed calendar or hours document changed')
        verified_response(path, receipt_path)
        paths[source['file']], paths[source['file']+'.receipt.json'] = path, receipt_path
    hashes = {n: digest(p) for n, p in paths.items()}
    if hashes['seed.parquet'] != OLD_SEED_SHA:
        raise ValueError('Original full-history seed identity changed')
    receipt = verified_response(paths['chart.json'], paths['chart-receipt.json'])
    classified, completed = classify_chart(json.loads(paths['chart.json'].read_bytes()), receipt, calendar)
    seed = prices(pd.read_parquet(seed_path)); rejection = None; proposed = None
    try:
        proposed = append_snapshot(seed, completed)
    except ValueError as exc:
        rejection = str(exc)
        if decision_path:
            proposed = initial_seed(seed, completed, json.loads(decision_path.read_bytes()),
                                    hashes['seed.parquet'], hashes['chart.json'])
    if decision_path and rejection is None:
        raise ValueError('A revision decision cannot be applied to a different, unrevised source')
    helpers = {n: digest(repo/n) for n in HELPERS}
    identity = {'ml': research_source_identity(repo), 'helpers': helpers}
    source_id = content_id(identity)
    with writer(output):
        for name, path in paths.items():
            dest = output/'inputs'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        for name in {**identity['ml']['files'], **helpers}:
            dest = output/'source-snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/name, dest)
        classified.to_csv(output/'classified-bars.csv', index=False, lineterminator='\n')
        completed.to_parquet(output/'completed-quotes.parquet', index=False)
        report = {'profile': PROFILE, 'source_id': source_id, 'new_fits': 0, 'new_forecasts': 0,
                  'market_skill': False, 'cohort_locked': False, 'forward_authorized': False,
                  'source_qualification': QUALIFICATION, 'revision_cause_verified': False,
                  'observed_available_at': receipt['observed_available_at'],
                  'classification_counts': classified.classification.value_counts().sort_index().to_dict(),
                  'latest_completed_date': str(completed.date.iloc[-1].date()),
                  'strict_append_rejection': rejection, 'original_seed_rows': len(seed),
                  'proposed_seed_rows': None, 'status': 'REJECTED_REVISION' if proposed is None
                  else 'PROPOSED_SEED_NOT_FORWARD_AUTHORIZATION'}
        if proposed is not None:
            proposed.to_parquet(output/'proposed-seed.parquet', index=False)
            report.update(proposed_seed_rows=len(proposed), proposed_seed_sha256=digest(output/'proposed-seed.parquet'))
            if rejection:
                unsafe = pd.concat([seed, completed.loc[completed.date.gt(seed.date.iloc[-1])]], ignore_index=True)
                mixed = float(candidate_history(unsafe).ct_ewma.iloc[-1])
                current = float(candidate_history(proposed).ct_ewma.iloc[-1])
                report['diagnostic_only_ewma'] = {'unsafe_mixed_vintage': mixed, 'proposed_vintage': current,
                                                'relative_difference_pct': 100*(mixed/current-1)}
        report['outputs_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
        if {n: digest(p) for n, p in paths.items()} != hashes:
            raise ValueError('Original input changed during source review')
        freeze_record(output/'ready.json', {'profile': PROFILE, 'source': {**identity, 'source_id': source_id},
                                          'inputs_sha256': hashes, 'python': sys.version.split()[0],
                                          'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'pyarrow')}})
        freeze_record(output/'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'seed', 'source-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--initial-version-decision', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.repo, args.seed, args.source_dir, args.initial_version_decision, args.output), indent=2))
