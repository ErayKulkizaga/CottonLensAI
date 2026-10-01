"""Append-only forward predictions. No partial performance feedback before 126 mature origins."""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.research.ledger import freeze_record, read_record, writer


def record(root, locked, release, history, predict, *, now=None):
    now = pd.Timestamp(now or datetime.now(UTC))
    if now.tzinfo is None:
        raise ValueError('UTC-aware emission time required')
    locked_at = pd.Timestamp(locked['prospective_start'])
    latest = history.iloc[-1]
    if not np.isfinite(latest.cotton_close) or latest.cotton_close <= 0:
        raise ValueError('Positive finite origin price required')
    origin = pd.Timestamp(latest.date).tz_localize('UTC')
    decision = origin + pd.Timedelta(days=1)
    if decision <= locked_at or decision > now or now >= decision + pd.Timedelta(days=1):
        raise ValueError('Record the newest completed bar after lock, within its decision day; no backdating')
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique chronological input snapshot required')
    if any(str(h) not in locked['horizons'] for h in (1, 5)):
        raise ValueError('Both locked horizons required')
    root = Path(root)
    with writer(root):
        freeze_record(root / 'identity.json', {'lock_id': content_id(locked), 'release': release, 'required_origins': 126})
        key = origin.strftime('%Y-%m-%d')
        path = root / 'predictions' / f'{key}.json'
        if path.exists():
            return read_record(path)
        if len(list((root / 'predictions').glob('*.json'))) >= 126:
            raise ValueError('First 126 origins already frozen; evaluate without extending the cohort')
        outputs = {}
        for h in (1, 5):
            prediction = float(predict(h, history))
            if not np.isfinite(prediction):
                raise ValueError('Nonfinite forward prediction')
            outputs[str(h)] = {'log_return': prediction, 'price': float(latest.cotton_close * np.exp(prediction))}
        snapshot = root / 'inputs' / f'{key}.parquet'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            raise ValueError('Incomplete prospective input exists; inspect it without replacing evidence')
        history.to_parquet(snapshot, index=False)
        body = {'origin': key, 'emitted_at': now.isoformat(), 'available_at': decision.isoformat(),
                'current_price': float(latest.cotton_close), 'predictions': outputs,
                'release': release, 'input_sha256': digest(snapshot), 'lock_id': content_id(locked)}
        freeze_record(path, body)
        return body


def score(root, cotton, *, now=None):
    root = Path(root)
    identity = read_record(root / 'identity.json')
    records = [read_record(p) for p in sorted((root / 'predictions').glob('*.json'))]
    if len(records) < identity['required_origins']:
        return {'status': 'pending', 'recorded_origins': len(records), 'required_origins': 126}
    if len(records) != 126:
        raise ValueError('Prospective cohort must contain exactly the first 126 recorded origins')
    source = cotton.sort_values('date').reset_index(drop=True)
    if source.date.duplicated().any() or not np.isfinite(source.close).all() or (source.close <= 0).any():
        raise ValueError('Invalid outcome source')
    cutoff = pd.Timestamp(now or datetime.now(UTC))
    dates = pd.DatetimeIndex(source.date)
    indexes = dates.get_indexer(pd.to_datetime([r['origin'] for r in records]))
    if (indexes < 0).any() or indexes[-1] + 5 >= len(source):
        return {'status': 'pending', 'reason': '126 origins not all mature at T+5'}
    maturity = pd.Timestamp(source.date.iloc[indexes[-1] + 5]).tz_localize('UTC') + pd.Timedelta(days=1)
    if maturity > cutoff:
        return {'status': 'pending', 'reason': 'last target bar not yet available'}
    for record in records:
        if record['lock_id'] != identity['lock_id'] or record['release'] != identity['release']:
            raise ValueError('Mixed prospective model identities')
        snapshot = root / 'inputs' / f'{record["origin"]}.parquet'
        if digest(snapshot) != record['input_sha256']:
            raise ValueError('Prospective input checksum mismatch')
    prices = np.asarray([r['current_price'] for r in records])
    results = {}
    for h in (1, 5):
        actual = np.log(source.close.to_numpy()[indexes + h] / prices)
        predicted = np.asarray([r['predictions'][str(h)]['log_return'] for r in records])
        metrics = evaluate(prices, actual, predicted)
        naive = evaluate(prices, actual, np.zeros(len(records)))
        metrics['relative_mae_improvement_pct'] = 100 * (1 - metrics['mae'] / naive['mae']) if naive['mae'] else None
        results[str(h)] = {'model': metrics, 'naive': naive}
    # Missed observation days are visible and prevent a complete-coverage claim.
    gaps = sorted(set(dates[indexes[0]:indexes[-1] + 1].strftime('%Y-%m-%d')) - {r['origin'] for r in records})
    report = {'status': 'complete', 'evidence': 'prospective_recorded_before_targets', 'horizons': results,
              'missed_source_origins': gaps, 'complete_origin_coverage': not gaps, 'identity': identity}
    with writer(root):
        freeze_record(root / 'outcome.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('record', 'score'))
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Frozen feature history for record; Cotton date/close for score')
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--repo', type=Path)
    args = parser.parse_args()
    history = pd.read_parquet(args.input)
    if args.action == 'score':
        value = score(args.store, history)
    else:
        if not all((args.lock, args.artifact, args.repo)):
            parser.error('record requires --lock, --artifact and --repo')
        sys.path.insert(0, str(args.repo / 'backend'))
        from app.runtime import ArtifactRuntime
        runtime = ArtifactRuntime(str(args.artifact))
        runtime.load()
        locked = read_record(args.lock)
        if runtime.manifest.get('protocol_identity', {}).get('locked') != locked:
            raise ValueError('Artifact does not belong to the prospective candidate lock')
        def predict(horizon, rows):
            return runtime.predict_return(horizon, rows.tail(runtime.required_window(horizon)).to_dict('records'))
        value = record(args.store, locked, runtime.manifest['artifact_version'], history, predict)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
