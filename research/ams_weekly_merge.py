"""Merge verified partial prefixes without altering complete daily witnesses."""

import argparse
from pathlib import Path

from cottonlens_ml.research.ledger import freeze_record, read_record

from research.ams_contract_reconciliation import sha


def merge(parent_path, weekly_path, output):
    parent, weekly = read_record(parent_path), read_record(weekly_path)
    if parent['training_allowed'] is not False or weekly['training_allowed'] is not False:
        raise ValueError('Only unadmitted value overlays may be combined')
    quotes = {d: dict(v) for d, v in parent['quotes'].items()}; new = []
    for day, values in weekly['quotes'].items():
        if day in quotes:
            if any(quotes[day].get(c) != p for c, p in values.items()):
                raise ValueError('Overlapping immutable source prices differ')
        else:
            quotes[day] = values; new.append(day)
    return freeze_record(output, {'quotes': dict(sorted(quotes.items())), 'parent_sha256': sha(parent_path),
                         'weekly_sha256': sha(weekly_path), 'new_partial_dates': sorted(new),
                         'training_allowed': False, 'historical_availability_verified': False,
                         'first_vintage_verified': False,
                         'scope': 'Exact existing ten-quote daily witnesses plus literal five/six-quote weekly prefixes; no inferred far quotes, missing days or daily availability.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'weekly', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); merge(a.parent, a.weekly, a.output)
