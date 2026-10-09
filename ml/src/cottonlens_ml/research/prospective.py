"""Append-only forward predictions. No partial performance feedback before 126 mature origins."""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, safe_member
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
        if not safe_member(body['input_file']) or body['input_file'] != 'baseline-inputs/' + path.stem + '.parquet':
            raise ValueError('Invalid forward input path')
        if digest(root / body['input_file']) != body['input_sha256']:
            raise ValueError('Forward input changed')
        records.append(body)
        previous = content_id(body)
    if len(records) > 126:
        raise ValueError('Fixed forward cohort exceeded')
    return lock, records


def baseline_status(root, market_root, *, now=None):
    """Read-only receipt/input/clock audit; never score, collect or backfill.

    Absence of a pre-cutoff archived snapshot proves absence of local evidence,
    not that the provider could not have supplied the bar at that time.
    """
    root, market_root = Path(root), Path(market_root)
    stamp = pd.Timestamp(now or datetime.now(UTC))
    if stamp.tzinfo is None:
        raise ValueError('Timezone-aware audit time required')
    if (root / '.writer-lock').exists() or (root.parent / '.writer-lock').exists():
        raise RuntimeError('Forward writer active or interrupted; audit a stable copy')
    if not (root / 'baseline-lock.json').exists():
        if any((root / 'baseline-origins').glob('*.json')):
            raise ValueError('Forward records without lock')
        return {'status': 'not_locked', 'baseline_lock_present': False}
    locked, records = verify_baseline_chain(root)
    expected = {'schema': 'prospective-baselines-v2', 'required_origins': 126,
                'decision_utc': '00:15', 'publication_end_utc': '00:30',
                'recipes': {'price': {'family': 'naive', 'log_return': 0.},
                            'variance': {'family': 'ewma', 'lambda': .94}}}
    if any(locked.get(k) != v for k, v in expected.items()):
        raise ValueError('Unsupported forward lock policy; do not reinterpret existing evidence')
    locked_at = pd.Timestamp(locked['locked_at'])
    if locked_at.tzinfo is None or locked_at > stamp:
        raise ValueError('Invalid forward lock time')

    def frame_dates(frame):
        dates = pd.DatetimeIndex(frame.date)
        if (frame.empty or dates.tz is not None or dates.hasnans or dates.has_duplicates
                or not dates.is_monotonic_increasing or not dates.equals(dates.normalize())
                or not np.isfinite(frame.cotton_close).all() or (frame.cotton_close <= 0).any()):
            raise ValueError('Invalid forward daily-bar snapshot')
        return dates

    receipts, eligible = {}, set()
    for path in sorted((market_root / 'observations').glob('*.json')):
        receipt = read_record(path)
        if receipt['symbol'] != 'CT=F':
            continue
        observed = pd.Timestamp(receipt['observed_available_at'])
        if observed.tzinfo is None or observed > stamp:
            raise ValueError('Invalid forward receipt observation time')
        name = receipt['source_file']
        if not safe_member(name) or not name.startswith('objects/'):
            raise ValueError('Invalid market snapshot path')
        source = market_root / name
        if digest(source) != receipt['source_sha256']:
            raise ValueError('Observed market snapshot checksum mismatch')
        frame = pd.read_parquet(source).rename(columns={'close': 'cotton_close'})
        dates = frame_dates(frame)
        if dates.max() >= observed.tz_convert('UTC').tz_localize(None).normalize():
            raise ValueError('Receipt contains an uncompleted daily bar')
        receipts[content_id(receipt)] = (frame, dates, observed)
        for date in dates:
            decision = date.tz_localize('UTC') + pd.Timedelta(days=1, minutes=15)
            if locked_at < decision <= stamp:
                eligible.add(date.strftime('%Y-%m-%d'))

    details, seen = [], set()
    for path, record in zip(sorted((root / 'baseline-origins').glob('*.json')), records, strict=True):
        origin = pd.Timestamp(record['origin'])
        decision = origin.tz_localize('UTC') + pd.Timedelta(days=1, minutes=15)
        emitted = pd.Timestamp(record['recorded_at'])
        if (record['origin'] != path.stem or origin != origin.normalize()
                or record['origin'] in seen or (seen and record['origin'] <= max(seen))
                or pd.Timestamp(record['decision_at']) != decision
                or not locked_at < decision <= stamp or emitted.tzinfo is None
                or not decision <= emitted <= stamp or record['state'] not in ('published', 'missing')):
            raise ValueError('Invalid forward origin, state or publication clock')
        seen.add(record['origin'])
        if record['source_receipt_id'] not in receipts:
            raise ValueError('Missing selected forward source receipt')
        frame, _, observed = receipts[record['source_receipt_id']]
        context = frame.loc[frame.date <= origin].reset_index(drop=True)
        frozen = pd.read_parquet(root / record['input_file'])
        frame_dates(frozen)
        try:
            pd.testing.assert_frame_equal(frozen, context)
        except AssertionError as exc:
            raise ValueError('Forward input does not match selected source receipt') from exc
        if (frozen.date.iloc[-1] != origin or record['current_price'] != float(frozen.cotton_close.iloc[-1])
                or observed > emitted):
            raise ValueError('Forward origin price or selected receipt time mismatch')
        ontime = sum(observed_at <= decision and origin in dates
                     for _, dates, observed_at in receipts.values())
        if record['state'] == 'published':
            if observed > decision or not decision <= emitted < decision + pd.Timedelta(minutes=15):
                raise ValueError('Published forecast missed input cutoff or publication window')
            returns = np.log(frozen.cotton_close / frozen.cotton_close.shift())
            variance = float((returns**2).ewm(alpha=.06, adjust=False).mean().iloc[-1])
            predictions = record['predictions']
            if set(predictions) != {'1', '5'} or record['missing_reason'] is not None:
                raise ValueError('Incomplete published baseline predictions')
            for h in (1, 5):
                prediction = predictions[str(h)]
                if (prediction['price_model'] != 'naive' or prediction['log_return'] != 0.
                        or prediction['variance_model'] != 'ewma-lambda-0.94'
                        or not np.isfinite(variance) or variance <= 0
                        or not np.isclose(prediction['variance'], h * variance, rtol=1e-12, atol=0)):
                    raise ValueError('Forward prediction differs from locked baseline recipe')
            diagnosis = 'published_with_pre_cutoff_input'
        else:
            if record['predictions'] or not record['missing_reason']:
                raise ValueError('Missing origin must not contain a forecast')
            diagnosis = ('no_archived_pre_cutoff_input' if not ontime else
                         'publication_window_missed' if emitted >= decision + pd.Timedelta(minutes=15)
                         else 'recorded_missing_requires_input_review')
        first = min(t for _, dates, t in receipts.values() if origin in dates)
        details.append({'origin': record['origin'], 'state': record['state'], 'diagnosis': diagnosis,
                        'decision_at': decision.isoformat(), 'recorded_at': emitted.isoformat(),
                        'pre_cutoff_snapshots': int(ontime), 'first_archived_input_at': first.isoformat(),
                        'source_receipt_id': record['source_receipt_id'], 'input_sha256': record['input_sha256']})
    published = sum(r['state'] == 'published' for r in records)
    unregistered = sorted(eligible - seen)
    overdue = [d for d in unregistered if pd.Timestamp(d).tz_localize('UTC') +
               pd.Timedelta(days=1, minutes=30) <= stamp]
    return {'status': 'collecting' if published else 'awaiting_forward_predictions',
            'audit_schema': 'prospective-baseline-status-v1', 'audited_at': stamp.isoformat(),
            'baseline_lock_present': True, 'lock_id': content_id(locked),
            'registered_origins': len(records), 'published_origins': published,
            'missing_origins': len(records) - published, 'required_origins': 126,
            'remaining_origin_slots': 126 - len(records),
            'recorded_coverage': published / len(records) if records else None,
            'unregistered_origins': overdue,
            'pending_publication_origins': sorted(set(unregistered) - set(overdue)), 'origins': details,
            'performance_feedback': 'not_computed; separate score requires all 126 origins mature',
            'limits': 'Local archive evidence only; no external attestation, provider latency proof or independent model skill.'}


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
            # Real execution must not label a slow write using its earlier start
            # time. Explicit now remains a deterministic synthetic-test clock.
            emitted = stamp if now is not None else pd.Timestamp(datetime.now(UTC))
            published = published and decision <= emitted < end
            body = {'origin': key, 'decision_at': decision.isoformat(), 'recorded_at': emitted.isoformat(),
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
