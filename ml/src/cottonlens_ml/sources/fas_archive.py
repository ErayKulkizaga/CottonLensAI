"""Validate ESRQS archive identity before any numerical reconciliation.

Archive labels and sentinel createdTime fields are not publication evidence.
These checks never grant model eligibility or reinterpret current files as history.
"""
import calendar
import json
import re
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID


def parse_index(raw, year):
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty public ESRQS archive index required')
    records, ids, days = [], set(), set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str):
            raise TypeError('Archive row must be an object with a string identifier')
        try:
            identity = str(UUID(row['id']))
            stamp = datetime.fromisoformat(row['weekEndingDate'])
            day = stamp.date()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid archive identifier or observation date') from exc
        if (row['id'] != identity or stamp.tzinfo is not None or stamp.time() != datetime.min.time()
                or day.year != year or row.get('fullFileName') != day.strftime('%Y/%m/%d')
                or row.get('fileType') != 'ESRWeeklyReport' or row.get('fileExtension') != 'pdf'):
            raise ValueError('Archive row date/year/report identity mismatch')
        if identity in ids or day in days:
            raise ValueError('Duplicate archive identifier or observation date')
        ids.add(identity)
        days.add(day)
        records.append({'archive_id': identity, 'index_week_ending': day.isoformat(),
                        'created_time_literal': row.get('createdTime'),
                        'publication_timestamp_verified': False, 'model_eligible': False})
    return records


def report_period_end(text):
    normalized = ' '.join(text.split())
    # Legacy CAM PDFs start with the period sentence, without the modern title.
    periods = re.findall(r'This summary is based on reports from exporters for the period ([^.]{1,100})\.', normalized)
    if len(periods) != 1:
        raise ValueError('One unambiguous report-period declaration required')
    match = re.fullmatch(
        r'([A-Za-z]+)\s+\d{1,2}(?:,\s*\d{4})?\s*[-\u2013]\s*'
        r'(?:([A-Za-z]+)\s+)?(\d{1,2}),\s*(\d{4})', periods[0])
    if not match:
        raise ValueError('Unsupported report-period format; no guessed observation date')
    months = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}
    month = months.get((match[2] or match[1]).lower())
    if month is None:
        raise ValueError('Unknown report-period month')
    return date(int(match[4]), month, int(match[3]))


def embargo_declaration(text):
    """An official report's embargo is planned timing, not observed publication."""
    normalized = ' '.join(text.split())
    clocks = re.findall(r'EMBARGOED UNTIL (\d{1,2}):(\d{2}) (AM|PM)', normalized)
    # Multi-column cover extraction may interleave the agency address between
    # the clock and date. Require one printed uppercase date; never pick among
    # several dates or reinterpret the title-case observation date as release.
    dates = re.findall(r'\b([A-Z]+) (\d{1,2}),? (\d{4})\b', normalized)
    if len(clocks) != 1 or len(dates) != 1:
        raise ValueError('One explicit embargo clock and cover date required')
    hour, minute, meridiem = clocks[0]
    month, day, year = dates[0]
    if not 1 <= int(hour) <= 12 or not 0 <= int(minute) <= 59:
        raise ValueError('Invalid embargo clock')
    months = {name.upper(): number for number, name in enumerate(calendar.month_name) if name}
    if month not in months:
        raise ValueError('Unknown embargo month')
    return {'scheduled_release_date': date(int(year), months[month], int(day)).isoformat(),
            'local_time_literal': f'{hour}:{minute} {meridiem}', 'timezone': None,
            'basis': 'per_report_printed_embargo_not_actual_publication',
            'publication_timestamp_verified': False, 'first_version_verified': False,
            'model_eligible': False}


def identity_check(records, archive_id, report_text):
    """Quarantine future-period responses; do not infer archive date semantics.

    An index date may label a release rather than its reporting period. Matching
    it is insufficient evidence for numerical reconciliation or model admission.
    """
    selected = [row for row in records if row['archive_id'] == archive_id]
    if len(selected) != 1:
        raise ValueError('Exactly one selected archive row required')
    row = selected[0]
    actual = report_period_end(report_text).isoformat()
    future = actual > row['index_week_ending']
    return {'archive_id': archive_id, 'requested_index_week_ending': row['index_week_ending'],
            'pdf_report_period_end': actual,
            'content_identity_status': ('future_period_for_historical_row' if future
                                        else 'historical_period_unverified'),
            'index_date_equals_pdf_period_end': row['index_week_ending'] == actual,
            'index_date_semantics_verified': False,
            'numeric_reconciliation_allowed': False,
            'created_time_literal': row['created_time_literal'],
            'publication_timestamp_verified': False, 'vintage_verified': False,
            'model_eligible': False}


def upland_stock_totals(text):
    """Read only the historical ALL UPLAND grand total, not Pima or subtotals.

    Decimal tenths of 1,000 running bales imply a 50-bale rounding half-width.
    This is content reconciliation only, never release/version certification.
    """
    heading = r'ALL UPLAND COTTON\s+MARKETING YEAR\s+08/01\s*-\s*07/31'
    matches = list(re.finditer(heading, text))
    if len(matches) != 1:
        raise ValueError('One ALL UPLAND COTTON table required')
    section = text[matches[0].end():]
    following = re.search(r'\n[^\n]+\s+MARKETING YEAR\s+\d{2}/\d{2}\s*-\s*\d{2}/\d{2}', section)
    if following:
        section = section[:following.start()]
    unit = re.search(r'1000 RUNNING BALES\s+AS OF\s+([A-Z]+)\s+(\d{1,2}),\s*(\d{4})', section)
    if not unit:
        raise ValueError('Explicit running-bale unit and as-of date required')
    months = {name.upper(): number for number, name in enumerate(calendar.month_name) if name}
    if unit[1] not in months:
        raise ValueError('Unknown table observation month')
    day = date(int(unit[3]), months[unit[1]], int(unit[2])).isoformat()
    totals = re.findall(r'TOTAL KNOWN & UNKNOWN\s*:\s*([^\n]+)', section)
    if len(totals) != 1:
        raise ValueError('One ALL UPLAND grand-total row required')
    values = totals[0].split()
    if len(values) != 6 or any(not re.fullmatch(r'\d+\.\d', v) for v in values):
        raise ValueError('Six explicit one-decimal totals required; suppressed values are not zero')
    return {'observation_date': day, 'unit': 'running_bales', 'rounding_half_width': 50,
            'totals': {key: int(Decimal(values[column]) * 1000) for key, column in
                       [('outstandingSales', 0), ('accumulatedExports', 2), ('nextMYOutstandingSales', 4)]},
            'scope': 'TOTAL KNOWN & UNKNOWN; own-account exports excluded'}


def reconcile_upland(payload, catalog, report_text):
    """Compare a pinned API snapshot to the selected report at printed precision."""
    table = upland_stock_totals(report_text)
    if report_period_end(report_text).isoformat() != table['observation_date']:
        raise ValueError('Report summary and cotton table periods differ')
    commodity = [r for r in catalog if r.get('commodityCode') == 1404]
    if (len(commodity) != 1 or commodity[0].get('commodityName') != 'All Upland Cotton'
            or commodity[0].get('unitId') != 2):
        raise ValueError('Pinned ALL UPLAND commodity/unit catalog required')
    selected = [r for r in payload if r.get('weekEndingDate') == table['observation_date'] + 'T00:00:00']
    if not selected or any(r.get('commodityCode') != 1404 or r.get('unitId') != commodity[0]['unitId']
                           for r in selected):
        raise ValueError('Wrong commodity/unit or absent API week')
    countries = [r.get('countryCode') for r in selected]
    if any(not isinstance(c, int) for c in countries) or len(set(countries)) != len(countries):
        raise ValueError('Unique country records required; do not double-count')
    comparisons = {}
    for key, printed in table['totals'].items():
        values = [r.get(key) for r in selected]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError('Nonnegative integer stock totals required')
        actual = sum(values)
        comparisons[key] = {'api_total': actual, 'pdf_printed_total': printed,
                            'difference_bales': actual - printed,
                            'within_printed_precision': abs(actual - printed) <= table['rounding_half_width']}
    return {**table, 'country_rows': len(selected), 'api_commodity_code': 1404,
            'api_unit_id': commodity[0]['unitId'], 'comparisons': comparisons,
            'unit_basis': 'pinned 1404 commodity catalog and same-commodity PDF running-bale table',
            'content_matches_at_printed_precision': all(c['within_printed_precision'] for c in comparisons.values()),
            'publication_timestamp_verified': False, 'vintage_verified': False, 'model_eligible': False}
