"""Local source observations + locked forward baselines; no training or backfill."""
import argparse
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, read_record, writer
from cottonlens_ml.research.mirror import Mirror
from cottonlens_ml.research.prospective import (
    baseline_lock,
    baseline_status,
    record_baselines,
    score_baselines,
    verify_baseline_chain,
)
from cottonlens_ml.sources.public import archive_observation
from cottonlens_ml.sources.usda import KEYS, USDAClient


def contract_symbols(now):
    """ICE listed months; candidate symbols are smoke probes, not proven roll dates."""
    contracts = [(y, m, code) for y in (now.year, now.year+1)
                 for m, code in ((3, 'H'), (5, 'K'), (7, 'N'), (10, 'V'), (12, 'Z'))
                 if (y, m) >= (now.year, now.month)]
    return [f'CT{code}{y%100:02d}.NYB' for y, _, code in contracts[:3]]


def capture_market(root, symbol='CT=F', *, download=None, now=None):
    import yfinance as yf
    root = Path(root)
    if symbol != 'CT=F' and not re.fullmatch(r'CT[HKNVZ]\d{2}\.NYB', symbol):
        raise ValueError('Only Cotton proxy and contract probes allowed')
    started = datetime.now(UTC)
    raw = (download or yf.download)(symbol, period='2y', auto_adjust=False, progress=False, threads=False, timeout=15)
    completed = pd.Timestamp(now or datetime.now(UTC))
    if raw.empty:
        raise ValueError('No contract/proxy quotes available; not evidence of a roll')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    frame = raw.rename(columns=str.lower).reset_index()
    frame = frame.rename(columns={frame.columns[0]: 'date'})
    frame['date'] = pd.to_datetime(frame.date).dt.tz_localize(None)
    # Provider dates at/after the current UTC day are not completed daily bars.
    frame = frame.loc[frame.date < completed.tz_convert('UTC').tz_localize(None).normalize()].sort_values('date').reset_index(drop=True)
    if frame.empty or frame.date.duplicated().any() or not np.isfinite(frame.close).all() or (frame.close<=0).any():
        raise ValueError('Invalid completed quote snapshot')
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / (uuid.uuid4().hex + '.pending')
    frame.to_parquet(temporary, index=False)
    checksum = digest(temporary)
    source = root / 'objects' / (checksum + '.parquet')
    source.parent.mkdir(exist_ok=True)
    if source.exists():
        if digest(source) != checksum:
            raise ValueError('Quote archive corruption')
        temporary.unlink()
    else:
        temporary.rename(source)
    body = {'symbol': symbol, 'request_started_at': started.isoformat(), 'observed_available_at': completed.isoformat(),
        'source_file': source.relative_to(root).as_posix(), 'source_sha256': checksum,
        'representation': 'provider dataframe snapshot; not original HTTP response',
        'published_at': None, 'first_publication_verified': False, 'redistribution_reviewed': False,
        'model_eligible_historical': False, 'last_bar_date': str(frame.date.max().date())}
    receipt = root / 'observations' / (content_id(body) + '.json')
    freeze_record(receipt, body)
    return receipt


def market_snapshots(root):
    items = []
    for path in sorted((Path(root)/'observations').glob('*.json'), key=lambda p:p.stat().st_mtime)[-72:]:
        body = read_record(path)
        if body['symbol'] != 'CT=F':
            continue
        source = Path(root)/body['source_file']
        if digest(source) != body['source_sha256']:
            raise ValueError('Observed market snapshot checksum mismatch')
        frame = pd.read_parquet(source).rename(columns={'close': 'cotton_close'})
        items.append((frame, body))
    return items


def collect(root, *, mirror_root=None):
    root = Path(root)
    now = datetime.now(UTC)
    tasks, status = [], {'execution_source_id': research_source_identity(Path(__file__).resolve().parents[4])['source_id']}
    # Publication uses existing pre-cutoff receipts before any slow network work.
    baseline_lock(root/'forward', now=now)
    snapshots = market_snapshots(root/'market')
    if snapshots:
        status['forward'] = record_baselines(root/'forward', snapshots)
    if now.hour == 0 and 15 <= now.minute < 30:
        # Use pre-cutoff receipts only; unrelated source requests/mirroring cannot
        # occupy this task's publication invocation or expose partial scores.
        _, records = verify_baseline_chain(root/'forward')
        status.update({'phase': 'publication_only', 'observed_at': now.isoformat(),
                       'tasks': [], 'training': False, 'forward_record_ids': [content_id(r) for r in records]})
        freeze_record(root/'runs'/(content_id(status)+'.json'), status)
        return status
    def task(name, operation):
        try:
            result = operation()
            tasks.append({'source': name, 'status': 'observed', 'receipt_or_archive': str(result)})
        except (OSError, ValueError, RuntimeError) as exc:
            # Never include response bodies, prepared URLs or credential-bearing errors.
            tasks.append({'source': name, 'status': 'unavailable', 'error_type': type(exc).__name__})
        except Exception as exc:  # noqa: BLE001 - provider failures must not hide credential-safe source status
            tasks.append({'source': name, 'status': 'failed', 'error_type': type(exc).__name__})
    task('CT=F', lambda: capture_market(root/'market'))
    # Raw information refresh once per UTC day; hourly proxy capture remains cheap.
    daily = root/'daily'/(now.strftime('%Y-%m-%d')+'.json')
    if not daily.exists():
        for symbol in contract_symbols(now):
            task(symbol, lambda symbol=symbol: capture_market(root/'contracts', symbol))
        urls = [('ams', 'https://www.ams.usda.gov/mnreports/ams_3804.pdf'),
                ('cftc', f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{now.year}.zip'),
                ('export_sales', 'https://apps.fas.usda.gov/export-sales/cotton.htm'),
                ('crop_progress', f'https://www.nass.usda.gov/Publications/Todays_Reports/reports/prog{now.strftime("%U%y")}.txt'),
                ('wasde', f'https://www.usda.gov/oce/commodity/wasde/wasde{now.strftime("%m%y")}.pdf'),
                ('wasde', f'https://www.usda.gov/oce/commodity/wasde/wasde{(pd.Timestamp(now)-pd.DateOffset(months=1)).strftime("%m%y")}.pdf')]
        for kind, url in urls:
            task(kind, lambda kind=kind,url=url: archive_observation(kind, url, root/'public'))
        freeze_record(daily, {'observed_at': now.isoformat(), 'tasks': list(tasks),
            'model_admission': False, 'failure_retry': 'next UTC day; hourly market retries continue'})
    # API credentials may become available after a public-source run on another host.
    # An earlier credential-missing receipt must not block the same day's API fetch.
    for provider in ('ams', 'fas', 'nass'):
        api_daily = root/'api-daily'/(now.strftime('%Y-%m-%d')+'-'+provider+'.json')
        if api_daily.exists():
            read_record(api_daily)
            continue
        if not os.environ.get(KEYS[provider]):
            tasks.append({'source': provider+'-api', 'status': 'credential_not_available_locally', 'secret_name': KEYS[provider]})
            continue
        client = USDAClient(provider)
        if provider == 'ams':
            operation = lambda client=client: client.report(3804, (pd.Timestamp(now)-pd.Timedelta(days=7)).date().isoformat(), now.date().isoformat(), root/'usda')
        elif provider == 'fas':
            operation = lambda client=client: client.exports(1404, now.year, root/'usda')
        else:
            operation = lambda client=client: client.cotton_by_year(now.year, root/'usda')
        task(provider+'-api', lambda operation=operation: operation()[0])
        freeze_record(api_daily, {'observed_at': now.isoformat(), 'task': tasks[-1],
            'model_admission': False, 'failure_retry': 'next UTC day'})
    snapshots = market_snapshots(root/'market')
    if snapshots:
        status['forward'] = record_baselines(root/'forward', snapshots)
        status['outcome'] = score_baselines(root/'forward', snapshots[-1][0])
    _, records = verify_baseline_chain(root/'forward')
    status.update({'observed_at': now.isoformat(), 'tasks': tasks, 'training': False,
                   'phase': 'collection', 'forward_record_ids': [content_id(r) for r in records]})
    report = root/'runs'/(content_id(status)+'.json')
    freeze_record(report, status)
    if mirror_root:
        mirror = Mirror(root, Path(mirror_root)/'live-baselines-v2')
        queued = {name for p in (root/'transfer-queue').glob('*.json') for name in read_record(p)['files']}
        names = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
                 and not set(p.parts).intersection({'transfer-queue','transfer-receipts','transfer-packages','transfer-downloads','hydrated-packages'})
                 and p.suffix not in ('.log','.pending') and p.relative_to(root).as_posix() not in queued]
        if names:
            mirror.enqueue(names)
        status['mirror'] = mirror.flush()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--mirror-root', type=Path)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        try:
            result = baseline_status(args.store/'forward', args.store/'market')
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            parser.exit(2, f'Forward evidence audit failed ({type(exc).__name__}); no collection or training performed.\n')
        print(json.dumps(result, indent=2))
        return
    with writer(args.store):
        result = collect(args.store, mirror_root=args.mirror_root)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
