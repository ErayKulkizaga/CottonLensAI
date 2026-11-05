"""Offline Texas cotton report inventory; neither a clock nor training admission."""
import csv
import json
import re
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

from cottonlens_ml.code_identity import digest, manifest_id
from cottonlens_ml.sources.nass_archive import (
    cotton_condition_summary,
    verified_cached_release,
)

STAGES = ('Planted', 'Squaring', 'Setting Bolls', 'Bolls Opening', 'Harvested')
CATEGORIES = ('VERY POOR', 'POOR', 'FAIR', 'GOOD', 'EXCELLENT')


def percentage(token, *, dash_is_zero, revised_is_defined=False):
    """Keep the published qualifier; an absent/suppressed cell is never zero."""
    if token in ('NA', '(NA)', '(D)'):
        return {'value': None, 'qualifier': token}
    if token == '-' and dash_is_zero:
        return {'value': 0, 'qualifier': '-'}
    revised = token.startswith('*') and revised_is_defined
    if revised:
        token = token[1:]
    if not re.fullmatch(r'\d{1,3}', token) or not 0 <= int(token) <= 100:
        raise ValueError('Unknown or out-of-range percentage/qualifier')
    return {'value': int(token), 'qualifier': '*' if revised else None}


def table_block(lines, start, *, legacy):
    """Stop at the national cotton row: do not accidentally read the next crop."""
    national = r'^\s*15 Sts\s*:' if legacy else r'^15 States\s*\.+:'
    for end in range(start + 1, min(start + 65, len(lines))):
        if re.match(national, lines[end]):
            return lines[start:end + 1]
        if (re.match(r'^[A-Z].* - Selected States', lines[end])
                or re.match(r'^\s*[A-Za-z]+:\s+(?:Percent|Crop Condition)', lines[end])):
            break
    raise ValueError('Cotton table has no bounded national row')


def state_tokens(block, *, legacy):
    pattern = r'^\s*TX\s*:' if legacy else r'^Texas\s*\.+:'
    rows = [line for line in block if re.match(pattern, line)]
    if len(rows) != 1:
        raise ValueError('Expected one Texas row in each present Cotton table')
    return rows[0].split(':', 1)[1].split()


def progress_columns(block, week, *, legacy):
    """Verify each column date and the published, strictly historical average."""
    starts = [i for i, line in enumerate(block) if re.match(r'^\s*State\s*:', line)]
    if len(starts) != 1:
        raise ValueError('Unique progress date header required')
    i = starts[0]
    dates, years = block[i].split(':'), block[i + 1].split(':')
    if len(dates) != 5 or len(years) != 5:
        raise ValueError('Four explicitly dated progress columns required')
    parsed = []
    for day, year in zip(dates[1:4], years[1:4], strict=True):
        if not re.fullmatch(r'\s*\d{4}\s*', year):
            raise ValueError('Progress column year changed')
        parsed.append(datetime.strptime(day.strip().rstrip(',') + ' ' + year.strip(),
                                        '%B %d %Y').replace(tzinfo=UTC).date())
    previous_week = week - timedelta(days=7)
    previous_year = week.replace(year=week.year - 1)
    expected = [week, previous_week, previous_year] if legacy else [previous_year, previous_week, week]
    if parsed != expected:
        raise ValueError('Progress column dates disagree with observation week')
    average_prefix = block[i - 1].split(':')[-1] if legacy else ''
    average = re.sub(r'\s+', '', average_prefix + dates[4] + years[4])
    match = re.fullmatch(r'(\d{4})-(\d{4})(?:Average|Avg\.)', average)
    if not match or tuple(map(int, match.groups())) != (week.year - 5, week.year - 1):
        raise ValueError('Published average must cover the preceding five years')
    return ((0, 1, 2, 3) if legacy else (2, 1, 0, 3)), match.groups()


def parse_report(text, *, release_day):
    lines = text.splitlines()
    summary = cotton_condition_summary(text)
    week = date.fromisoformat(summary['week_ending'])
    if week.weekday() != 6:
        raise ValueError('Sunday Cotton observation week required')
    released = re.findall(r'^Released ([A-Za-z]+ \d{1,2}, \d{4}),', text[:1500], re.MULTILINE)
    if (len(released) != 1
            or datetime.strptime(released[0], '%B %d, %Y').replace(tzinfo=UTC).date().isoformat() != release_day
            or release_day < week.isoformat()):
        raise ValueError('Report release date disagrees with cached page/observation')
    dash_is_zero = bool(re.search(r'^\s*-\s+Represents zero\.', text, re.MULTILINE))
    revised_is_defined = bool(re.search(r'^\s*\*\s+Revised\.', text, re.MULTILINE))
    condition = [(i, line.startswith('Cotton Condition')) for i, line in enumerate(lines)
                 if line.startswith('Cotton Condition - Selected States: Week Ending ')
                 or re.match(r'^\s*Cotton:\s+Crop Condition by Percent,\s*$', line)]
    if len(condition) != 1:
        raise ValueError('One Cotton condition table required')
    start, modern = condition[0]
    cells = [percentage(t, dash_is_zero=dash_is_zero, revised_is_defined=revised_is_defined)
             for t in state_tokens(table_block(lines, start, legacy=not modern), legacy=not modern)]
    if len(cells) != 5:
        raise ValueError('Five Texas condition categories required')
    if all(c['value'] is not None for c in cells) and sum(c['value'] for c in cells) != 100:
        raise ValueError('Texas condition percentages do not sum to 100')
    result = {'week_ending': week.isoformat(), 'release_day': release_day,
              'national_condition': summary['national_condition'],
              'texas_condition': dict(zip(CATEGORIES, cells, strict=True)), 'progress': {}}
    for stage in STAGES:
        pattern = r'^\s*Cotton:\s+Percent ' + re.escape(stage) + r',\s*$'
        found = [(i, bool(re.match(pattern, line))) for i, line in enumerate(lines)
                 if line == f'Cotton {stage} - Selected States' or re.match(pattern, line)]
        if len(found) > 1:
            raise ValueError('Duplicate Cotton progress section')
        if not found:
            result['progress'][stage] = {'section_present': False, 'cells': None,
                                         'published_average_years': None, 'gap_to_average_pp': None}
            continue
        start, legacy = found[0]
        block = table_block(lines, start, legacy=legacy)
        order, average_years = progress_columns(block, week, legacy=legacy)
        tokens = state_tokens(block, legacy=legacy)
        if len(tokens) != 4:
            raise ValueError('Four Texas progress percentages required')
        cells = [percentage(tokens[index], dash_is_zero=dash_is_zero,
                            revised_is_defined=revised_is_defined) for index in order]
        named = dict(zip(('current', 'previous_week', 'previous_year', 'published_average'), cells, strict=True))
        current, average = named['current']['value'], named['published_average']['value']
        result['progress'][stage] = {'section_present': True, 'cells': named,
            'published_average_years': list(map(int, average_years)),
            'gap_to_average_pp': None if current is None or average is None else current - average}
    return result


def read_frozen(path):
    record = json.loads(path.read_bytes())
    body = {k: v for k, v in record.items() if k != 'record_id'}
    if record.get('record_id') != manifest_id(body):
        raise ValueError('Frozen NASS record identity mismatch')
    return body


def audit(table, audits):
    """Recheck old inputs and derive a new quarantine panel without any writes."""
    table, audits = Path(table).resolve(), Path(audits).resolve()
    manifest_path = table.with_suffix('.manifest.json')
    manifest = read_frozen(manifest_path)
    if manifest.get('model_eligible') is not False or manifest.get('publication_clock_verified') is not False:
        raise ValueError('Only the old ineligible diagnostic table is supported')
    inputs = {table: digest(table), manifest_path: digest(manifest_path)}
    if inputs[table] != manifest['table_sha256']:
        raise ValueError('Diagnostic table checksum mismatch')
    with table.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row['week_ending']: row for row in rows}
    if len(rows) != manifest['row_count'] or len(indexed) != len(rows) or not rows:
        raise ValueError('Diagnostic count/unique weeks required')
    panel, seen = [], set()
    for year, checksum in sorted(manifest['source_year_audit_sha256'].items()):
        annual_path = audits / f'year-audit-{year}.json'
        if digest(annual_path) != checksum:
            raise ValueError('Annual audit checksum mismatch')
        inputs[annual_path] = checksum
        annual = read_frozen(annual_path)
        if annual['year'] != int(year) or annual['weeks'] != len(annual['results']):
            raise ValueError('Annual audit year/count mismatch')
        for item in annual['results']:
            week = item['week_ending']
            if week in seen or week not in indexed or week[:4] != year:
                raise ValueError('Unreviewed/duplicate/incorrect year source week')
            seen.add(week)
            row = indexed[week]
            if (item['status'] not in ('numeric_match', 'reference_missing_zero')
                    or row['quickstats_check'] != item['status']
                    or row['report_sha256'] != item['report_sha256']
                    or row['release_page_url'] != item['release_page']):
                raise ValueError('Diagnostic row and annual audit disagree')
            cached = verified_cached_release(item['release_page'], audits)
            if cached is None:
                raise ValueError('Pinned report is missing')
            folder, receipt = cached
            if not folder.resolve().is_relative_to(audits):
                raise ValueError('Cached report escapes input root')
            if receipt['report_sha256'] != item['report_sha256']:
                raise ValueError('Cached report differs from audited version')
            for name in ('report.txt', 'retrieval.json', 'release-page.html'):
                inputs[folder / name] = digest(folder / name)
            release_day = item['release_page'].rsplit('/', 1)[-1]
            parsed = parse_report((folder / 'report.txt').read_text(encoding='utf-8'), release_day=release_day)
            if parsed['week_ending'] != week or row['release_page_date'] != release_day:
                raise ValueError('Pinned report and diagnostic dates disagree')
            for name, category in zip(('very_poor', 'poor', 'fair', 'good', 'excellent'), CATEGORIES, strict=True):
                if row[name + '_pct'] != str(parsed['national_condition'][category]):
                    raise ValueError('National published condition and diagnostic values disagree')
            panel.append({**parsed, 'report_sha256': item['report_sha256'],
                          'release_page_url': item['release_page'],
                          'retrieved_at': receipt['retrieved_at'],
                          'available_at': None, 'availability_basis': 'UNSET',
                          'first_version_verified': False})
    if seen != set(indexed):
        raise ValueError('Audit/table week coverage differs')
    panel.sort(key=lambda row: row['week_ending'])
    coverage = {}
    for stage in STAGES:
        present = [row for row in panel if row['progress'][stage]['section_present']]
        coverage[stage] = {'present_reports': len(present), 'absent_reports': len(panel) - len(present),
                          'by_year': dict(sorted(Counter(row['week_ending'][:4] for row in present).items()))}
    # Compare only exactly adjacent reports in the same crop year; never fill gaps.
    backwards, previous_revisions = [], []
    for previous, current in pairwise(panel):
        if (current['week_ending'][:4] != previous['week_ending'][:4]
                or date.fromisoformat(current['week_ending']) - date.fromisoformat(previous['week_ending']) != timedelta(days=7)):
            continue
        for stage in STAGES:
            old, new = previous['progress'][stage], current['progress'][stage]
            if not old['section_present'] or not new['section_present']:
                continue
            a, b, repeated = old['cells']['current']['value'], new['cells']['current']['value'], new['cells']['previous_week']['value']
            if a is not None and b is not None and b < a:
                backwards.append({'week_ending': current['week_ending'], 'stage': stage, 'previous': a, 'current': b})
            if a is not None and repeated is not None and repeated != a:
                previous_revisions.append({'week_ending': current['week_ending'], 'stage': stage,
                                           'original_previous': a, 'later_previous': repeated})
    if any(digest(path) != checksum for path, checksum in inputs.items()):
        raise ValueError('NASS inputs changed during audit')
    safe_inputs = {('audits/' + p.relative_to(audits).as_posix() if p.is_relative_to(audits)
                    else 'table/' + p.name): checksum for p, checksum in inputs.items()}
    return {'schema': 'nass-texas-report-audit-v1', 'reports': len(panel),
            'first_week': panel[0]['week_ending'], 'last_week': panel[-1]['week_ending'],
            'coverage': coverage, 'backwards_progress': backwards,
            'previous_week_revisions': previous_revisions, 'input_sha256': safe_inputs,
            'inputs_unchanged': True, 'market_fits': 0,
            'model_eligible': False, 'release_allowed': False,
            'historical_publication_verified': False, 'first_version_verified': False,
            'availability_policy': 'UNSET', 'full_crop_season_coverage_verified': False,
            'panel': panel}
