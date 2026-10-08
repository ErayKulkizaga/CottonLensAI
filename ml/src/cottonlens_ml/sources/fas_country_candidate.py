"""Prepare a quarantined country panel; observation dates are not availability."""
import hashlib
import json
from collections import defaultdict
from datetime import date, timedelta
from decimal import Context, Decimal, localcontext
from pathlib import Path

from cottonlens_ml.code_identity import digest, manifest_id
from cottonlens_ml.sources.fas_country_audit import audit, number

COUNTRY_CODES = (4890, 5350, 5520, 5700)
FIELDS = ('currentMYNetSales', 'weeklyExports', 'outstandingSales')
FLOW_FIELDS = FIELDS[:2]


def _market_year(day):
    return day.year + int(day.month >= 8)


def _panel(observations, countries):
    """Use reported calendar weeks, all reported countries, and no future rows."""
    by_week, totals, share_defined = defaultdict(dict), {}, {}
    for day, country, values in observations:
        if country in by_week[day]:
            raise ValueError('Duplicate country/week')
        if values['weeklyExports'] < 0:
            raise ValueError('Negative shipment volume needs separate review')
        by_week[day][country] = values
    rows = []
    # Decimal sums and division must not inherit a caller's rounding policy.
    with localcontext(Context(prec=28)):
        for day, reported in by_week.items():
            totals[day] = {field: sum((v[field] for v in reported.values()), Decimal(0))
                           for field in FIELDS}
            share_defined[day] = {field: totals[day][field] > 0
                                  and all(v[field] >= 0 for v in reported.values())
                                  for field in FIELDS[1:]}
        for day in sorted(by_week):
            for country in countries:
                values = by_week[day].get(country)
                history = [day - timedelta(weeks=i) for i in range(4)]
                complete = all(_market_year(d) == _market_year(day)
                               and country in by_week.get(d, {}) for d in history)
                shares = {field: (str(values[field] / totals[day][field])
                                  if values is not None and share_defined[day][field] else None)
                          for field in FIELDS[1:]}
                rows.append({
                    'week_ending': day.isoformat(), 'marketing_year': _market_year(day),
                    'country_code': country, 'reported_row_present': values is not None,
                    'values': {f: str(values[f]) if values is not None else None for f in FIELDS},
                    'reported_national_totals': {f: str(totals[day][f]) for f in FIELDS},
                    'share_denominator_defined': dict(share_defined[day]),
                    'shares_of_reported_total': shares,
                    'trailing_four_calendar_weeks': {
                        f: str(sum((by_week[d][country][f] for d in history), Decimal(0)))
                        if complete else None for f in FLOW_FIELDS},
                })
    return rows


def prepare(raw_root, manifest_path, table_path, countries=COUNTRY_CODES):
    """Read pinned snapshots only; never write, align to Cotton, or grant admission."""
    countries = tuple(countries)
    if (not countries or any(type(c) is not int or c < 0 for c in countries)
            or len(set(countries)) != len(countries)):
        raise ValueError('Unique nonnegative integer country codes required')
    countries = tuple(sorted(countries))
    checked = audit(raw_root, manifest_path, table_path)
    raw_root, manifest_path, table_path = map(Path, (raw_root, manifest_path, table_path))
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != checked['manifest_sha256']:
        raise ValueError('Source changed after audit')
    manifest = json.loads(manifest_bytes)
    paths = defaultdict(list)
    for path in raw_root.glob('*/*/source.json'):
        paths[path.parent.name].append(path)
    observations = []
    for year, checksum in manifest['annual_sources'].items():
        candidates = paths[checksum]
        if len(candidates) != 1:
            raise ValueError('Pinned annual source missing or ambiguous')
        path = candidates[0]
        if not path.resolve().is_relative_to(raw_root.resolve()) or digest(path) != checksum:
            raise ValueError('Pinned annual source escaped or changed')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError('Source changed during read')
        for row in json.loads(raw):
            day = date.fromisoformat(row['weekEndingDate'][:10])
            if _market_year(day) != int(year) or day >= date(2024, 1, 1):
                continue
            observations.append((day, row['countryCode'],
                                 {f: number(row[f]) for f in FIELDS}))
    rows = _panel(observations, countries)
    if audit(raw_root, manifest_path, table_path) != checked:
        raise ValueError('Inputs changed during preparation')
    payload = {
        'schema': 1, 'profile': 'fas-country-quarantine-v1', 'market_fits': 0,
        'model_eligible': False, 'release_allowed': False,
        'historical_publication_verified': False, 'first_version_verified': False,
        'country_names_verified_for_full_history': False,
        'availability_policy': 'UNSET; week_ending is an observation period only',
        'unit': 'running_bales', 'country_codes': list(countries),
        'weeks': checked['weeks'], 'row_count': len(rows),
        'input_audit_id': manifest_id(checked),
        'manifest_sha256': checked['manifest_sha256'], 'table_sha256': checked['table_sha256'],
        'annual_sources': manifest['annual_sources'], 'catalog_sha256': checked['catalog_sha256'],
        'source_sha256': digest(Path(__file__)), 'rows': rows,
        'negative_outstanding_observations': [
            {'week_ending': day.isoformat(), 'country_code': code,
             'reported_value': str(values['outstandingSales'])}
            for day, code, values in sorted(observations) if values['outstandingSales'] < 0],
        'limits': [
            'Quarantined latest snapshots, not a historically admitted model input.',
            'Country codes are identifiers; the full historical name mapping remains unverified.',
            'Shares use all reported rows, whose full report coverage is not certified.',
            'Missing country/week or nonpositive denominator yields null, never zero.',
            'Signed outstanding values are preserved; any negative component disables that week share.',
            'Four-week totals require four consecutive calendar weeks in one marketing year.',
            'No accumulatedExports/commitment field, Cotton alignment, imputation or fitting.',
            'Past-only arithmetic cannot remove revision/vintage leakage in source snapshots.',
        ],
    }
    return {**payload, 'candidate_id': manifest_id(payload)}
