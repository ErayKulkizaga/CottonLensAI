"""Count immutable daily source coverage under separate explicit clock assumptions."""
import argparse
import calendar
import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path


def collect_additions(tables):
    additions, conflicts = {}, set()
    for table in tables.values():
        for row in table['rows']:
            if not row['available']:
                continue
            day = row['observation_date']
            prices = {k: Decimal(v) for k, v in row['settlements'].items()}
            q = {'prices': prices, 'printed': row['printed_publication_date'], 'kind': 'new_weekly'}
            if day in additions:
                a = additions[day]
                if a['prices'] != prices:
                    conflicts.add(day)
                if q['printed'] < a['printed']:
                    additions[day] = q
            else:
                additions[day] = q
    # No first-vintage clock can choose between the original and its correction.
    return {d: q for d, q in additions.items() if d not in conflicts}, sorted(conflicts)


def inspect(root):
    import pandas as pd

    contract = json.loads((root / 'availability-contract.json').read_text())
    history_path = Path(contract['canonical_calendar'])
    calendar_dates = pd.read_parquet(history_path, columns=['date']).date.dt.strftime('%Y-%m-%d').tolist()
    if calendar_dates != sorted(set(calendar_dates)):
        raise ValueError('invalid canonical Cotton calendar')
    previous = dict(zip(calendar_dates[1:], calendar_dates[:-1]))
    old_path = Path(contract['old_quotes'])
    old = json.loads(old_path.read_text())
    quotes = {}
    for row in old['rows']:
        converted = {}
        for name, value in row['quotes'].items():
            month, year = name.split('-')
            converted[f'20{year}-{list(calendar.month_abbr).index(month):02d}'] = Decimal(str(value))
        quotes[row['report_date']] = {'prices': converted, 'printed': row['report_date'], 'kind': 'existing_daily'}
    tables_path = root / 'tables.json'
    tables = json.loads(tables_path.read_text())
    additions, conflicts = collect_additions(tables)
    for day, q in additions.items():
        if day in quotes:
            common = quotes[day]['prices'].keys() & q['prices'].keys()
            if any(quotes[day]['prices'][k] != q['prices'][k] for k in common):
                raise ValueError('existing source price conflict')
        else:
            quotes[day] = q
    output = []
    for day in calendar_dates:
        q = quotes.get(day)
        row = {'date': day, 'quote_observed': q is not None, 'new_weekly': False,
               'report_clock_current_quote_available': False, 'exact_prior_pair_observed': False}
        if day in conflicts:
            if q is not None:
                raise ValueError('conflicting vintage date already in existing source')
            row['unknown_quote_reason'] = 'conflicting_archive_vintages'
        if q:
            row['new_weekly'] = q['kind'] == 'new_weekly'
            row['publication_lag_calendar_days'] = (date.fromisoformat(q['printed']) - date.fromisoformat(day)).days
            row['report_clock_current_quote_available'] = q['printed'] <= day
            names = sorted(q['prices'])
            if len(names) < 2:
                raise ValueError('missing second contract')
            second = names[1]
            prior = quotes.get(previous.get(day))
            row['exact_prior_pair_observed'] = prior is not None and second in prior['prices']
        output.append(row)
    grouped = {}
    for year in range(2010, 2024):
        rows = [r for r in output if r['date'].startswith(str(year))]
        grouped[str(year)] = {key: sum(r[key] for r in rows) for key in (
            'quote_observed', 'new_weekly', 'report_clock_current_quote_available', 'exact_prior_pair_observed')}
        grouped[str(year)]['retained_calendar_slots'] = len(rows)
    summary = {'profile': contract['profile'], 'new_weekly_calendar_dates': sum(r['new_weekly'] for r in output),
               'conflicting_vintage_dates_retained_unknown': conflicts,
               'weekly_same_printed_day_observations': sum(r['new_weekly'] and r['report_clock_current_quote_available'] for r in output),
               'coverage': grouped, 'new_fits': 0, 'training_admitted': False,
               'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (history_path, old_path, tables_path)}}
    for name, body in [('availability-rows.json', output), ('availability-summary.json', summary)]:
        p = root / name
        if p.exists():
            raise ValueError('immutable output exists')
        p.write_text(json.dumps(body, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    inspect(parser.parse_args().root)
