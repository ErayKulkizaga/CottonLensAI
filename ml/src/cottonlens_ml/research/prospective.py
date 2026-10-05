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


def baseline_lock(root, *, now=None):
    """New version, separate from legacy candidate records. No trained artifact."""
    root = Path(root)
    path = root / 'baseline-lock.json'
    if path.exists():
        return read_record(path)
    stamp = pd.Timestamp(now or datetime.now(UTC))
    if stamp.tzinfo is None:
        raise ValueError('Timezone-aware lock time required')
    body = {'schema': 'prospective-baselines-v2', 'locked_at': stamp.isoformat(),
        'required_origins': 126, 'cohort_rule': 'first 126 valid recorded Cotton bar origins with decision after lock; missing never replaced',
        'decision_utc': '00:15', 'publication_end_utc': '00:30',
        'recipes': {'price': {'family': 'naive', 'log_return': 0.}, 'variance': {'family': 'ewma', 'lambda': .94}},
        'immutability': 'local hash chain detects changes; not external attestation'}
    freeze_record(path, body)
    return read_record(path)


def verify_baseline_chain(root):
    root = Path(root)
    lock = read_record(root / 'baseline-lock.json')
    records, previous = [], None
    for path in sorted((root / 'baseline-origins').glob('*.json')):
        body = read_record(path)
        if body['previous_record_id'] != previous or body['lock_id'] != content_id(lock):
            raise ValueError('Forward chain or lock identity mismatch')
        if digest(root / body['input_file']) != body['input_sha256']:
            raise ValueError('Forward input changed')
        records.append(body)
        previous = content_id(body)
    if len(records) > 126:
        raise ValueError('Fixed forward cohort exceeded')
    return lock, records


def record_baselines(root, snapshots, *, now=None):
    """snapshots = [(frame, receipt)]; input availability uses actual capture time.

    A new origin is only finalized after its publication window. During the
    window it is published only if a suitable pre-cutoff snapshot exists.
    Late/outage origins become missing, never retrospective forecasts.
    """
    root = Path(root)
    stamp = pd.Timestamp(now or datetime.now(UTC))
    if stamp.tzinfo is None:
        raise ValueError('Timezone-aware emission required')
    with writer(root):
        baseline_lock(root, now=stamp)
        locked, existing = verify_baseline_chain(root)
        seen = {r['origin'] for r in existing}
        candidates = {}
        for frame, receipt in snapshots:
            if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
                raise ValueError('Unique chronological Cotton bars required')
            observed = pd.Timestamp(receipt['observed_available_at'])
            if observed.tzinfo is None or observed > stamp:
                raise ValueError('Invalid input observation time')
            for row in frame.itertuples():
                if not np.isfinite(row.cotton_close) or row.cotton_close <= 0:
                    continue
                origin = pd.Timestamp(row.date).tz_localize('UTC')
                decision = origin + pd.Timedelta(days=1, minutes=15)
                if decision <= pd.Timestamp(locked['locked_at']) or decision > stamp:
                    continue
                key = origin.strftime('%Y-%m-%d')
                candidates.setdefault(key, []).append((frame, receipt, observed, decision))
        added = []
        for key in sorted(candidates):
            if key in seen or len(existing) >= 126:
                continue
            if existing and key < existing[-1]['origin']:
                raise ValueError('Late source-origin insertion; cohort review required, never silently replace origins')
            options = candidates[key]
            decision = options[0][3]
            ontime = [x for x in options if x[2] <= decision]
            end = decision + pd.Timedelta(minutes=15)
            if not ontime and stamp < end:
                continue  # No claim yet; later within-window execution may obtain a receipt.
            published = bool(ontime and decision <= stamp < end)
            frame, receipt, _, _ = max(ontime or options, key=lambda x: x[2])
            context = frame.loc[frame.date <= key].copy()
            returns = np.log(context.cotton_close / context.cotton_close.shift())
            variance = float((returns**2).ewm(alpha=.06, adjust=False).mean().iloc[-1])
            if not np.isfinite(variance) or variance <= 0:
                published = False
            input_file = 'baseline-inputs/' + key + '.parquet'
            target = root / input_file
            target.parent.mkdir(parents=True, exist_ok=True)
            # A previous crash may have persisted the same input before the record.
            if target.exists():
                pd.testing.assert_frame_equal(pd.read_parquet(target), context.reset_index(drop=True))
            else:
                temporary = target.with_suffix('.pending')
                context.reset_index(drop=True).to_parquet(temporary, index=False)
                temporary.rename(target)
            body = {'origin': key, 'decision_at': decision.isoformat(), 'recorded_at': stamp.isoformat(),
                'state': 'published' if published else 'missing', 'lock_id': content_id(locked),
                'previous_record_id': content_id(existing[-1]) if existing else None,
                'current_price': float(context.cotton_close.iloc[-1]),
                'source_receipt_id': content_id(receipt), 'input_file': input_file, 'input_sha256': digest(target),
                'predictions': {str(h): {'price_model': 'naive', 'log_return': 0.,
                    'variance_model': 'ewma-lambda-0.94', 'variance': h*variance}
                    for h in (1, 5)} if published else {},
                'missing_reason': None if published else 'outside_window_or_no_valid_pre_cutoff_input'}
            path = root / 'baseline-origins' / (key + '.json')
            freeze_record(path, body)
            record = read_record(path)
            existing.append(record)
            added.append(record)
        return {'registered_origins': len(existing), 'published_origins': sum(r['state'] == 'published' for r in existing),
                'required_origins': 126, 'added_origins': len(added), 'performance_feedback': 'closed_until_all_targets_mature'}


def score_baselines(root, cotton, *, now=None):
    """Outcome snapshots are separately hashed; partial performance stays closed."""
    root = Path(root)
    _, records = verify_baseline_chain(root)
    if len(records) != 126:
        return {'status': 'pending', 'registered_origins': len(records), 'required_origins': 126}
    stamp = pd.Timestamp(now or datetime.now(UTC))
    dates = pd.DatetimeIndex(cotton.date)
    if dates.has_duplicates or not dates.is_monotonic_increasing or not np.isfinite(cotton.cotton_close).all() or (cotton.cotton_close<=0).any():
        raise ValueError('Unique chronological outcomes required')
    indexes = dates.get_indexer(pd.to_datetime([r['origin'] for r in records]))
    if (indexes < 0).any() or indexes.max()+5 >= len(cotton):
        return {'status': 'pending', 'reason': 'targets_not_all_mature'}
    if pd.Timestamp(dates[indexes.max()+5]).tz_localize('UTC') + pd.Timedelta(days=1, minutes=15) > stamp:
        return {'status': 'pending', 'reason': 'target_bar_not_completed'}
    use = np.array([r['state'] == 'published' for r in records])
    metrics = {}
    for h in (1, 5):
        prices = np.array([r['current_price'] for r in records])[use]
        actual = np.log(cotton.cotton_close.to_numpy()[indexes[use]+h] / prices)
        metrics[str(h)] = evaluate(prices, actual, np.zeros(len(prices))) if use.any() else None
    output = {'status': 'complete', 'cohort_count': 126, 'published_count': int(use.sum()),
              'coverage': float(use.mean()), 'horizons': metrics, 'missing_origins': [r['origin'] for r in records if r['state']=='missing']}
    with writer(root):
        outcome_file = root / 'baseline-outcomes.parquet'
        if outcome_file.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(outcome_file), cotton.reset_index(drop=True))
        else:
            cotton.reset_index(drop=True).to_parquet(outcome_file, index=False)
        output['outcome_sha256'] = digest(outcome_file)
        freeze_record(root / 'baseline-performance.json', output)
    return output


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
