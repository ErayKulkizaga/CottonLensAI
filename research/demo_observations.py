"""Manual immutable demo observations; no browser, scheduler or training authority."""
import argparse
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

from cottonlens_ml.research.admission import identity, utc
from cottonlens_ml.sprint import freeze_record, read_record, writer

FIELDS = {'provider', 'page', 'observed_at', 'target_date', 'horizon', 'forecast',
          'unit', 'contract', 'price_field', 'displayed_issue_time', 'origin_price'}


def validate(value):
    if set(value) != FIELDS:
        raise ValueError('Explicit observation schema required')
    for field in ('provider', 'page', 'unit'):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f'Explicit {field} required')
    stamp = utc(value['observed_at'])
    if date.fromisoformat(value['target_date']) < stamp.date():
        raise ValueError('Past target cannot become a forward observation')
    if type(value['horizon']) is not int or value['horizon'] < 1:
        raise ValueError('Explicit positive horizon required')
    for name in ('forecast', 'origin_price'):
        number = value[name]
        if name == 'origin_price' and number == 'UNKNOWN':
            continue
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError('Finite positive price required')
    for name in ('contract', 'price_field', 'displayed_issue_time'):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError('Use explicit UNKNOWN for missing metadata')
    return stamp


def records(root):
    result, previous = [], None
    for index, path in enumerate(sorted((Path(root)/'events').glob('*.json'))):
        record = read_record(path)
        if path.name != f'{index:06d}.json' or record['previous'] != previous:
            raise ValueError('Incomplete or reordered observation chain')
        validate(record['observation'])
        result.append(record)
        previous = identity(record)
    return result


def append(root, observation):
    stamp = validate(observation)
    observation = {**observation, 'observed_at': stamp.astimezone(UTC).isoformat().replace('+00:00', 'Z')}
    now = datetime.now(UTC)
    if stamp > now:
        raise ValueError('Future observation timestamp rejected')
    root = Path(root)
    with writer(root):
        prior = records(root)
        if prior and prior[-1]['observation'] == observation:
            return prior[-1]  # Idempotent repeat, never modify first observation.
        if prior and stamp < utc(prior[-1]['observation']['observed_at']):
            raise ValueError('Backdated append rejected; preserve chronological evidence')
        key = {k: observation[k] for k in ('provider', 'target_date', 'horizon', 'unit', 'contract', 'price_field')}
        revisions = [r for r in prior if r['series_key'] == identity(key)]
        record = {'schema': 1, 'observation': observation, 'recorded_at': now.isoformat(),
                  'previous': identity(prior[-1]) if prior else None,
                  'series_key': identity(key), 'revision': len(revisions),
                  'first_observed_at': revisions[0]['first_observed_at'] if revisions else observation['observed_at'],
                  'publication_verified': False, 'training_admitted': False}
        freeze_record(root/'events'/f'{len(prior):06d}.json', record)
        return record


def readiness(root):
    events = records(root)
    missing = sorted({k for r in events for k in ('contract', 'price_field', 'origin_price', 'displayed_issue_time')
                      if r['observation'][k] == 'UNKNOWN'})
    return {'events': len(events), 'revisions': sum(r['revision'] > 0 for r in events),
            'unknown_fields': missing, 'primary_evaluation_allowed': False,
            'reason': 'Independent same-contract outcomes and source/availability acceptance are required.',
            'market_skill': False, 'new_fits': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    if args.input:
        append(args.root, json.loads(args.input.read_text(encoding='utf-8')))
    print(json.dumps(readiness(args.root), indent=2))
