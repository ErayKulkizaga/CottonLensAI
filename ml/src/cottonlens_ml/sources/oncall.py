"""Bounded Cotton On-Call format/timing audit; never historical model admission.

Use public.archive_observation for immutable bytes and actual retrieval receipts.
The publisher's 'release after' footer is a lower bound, not an exact timestamp
or evidence that today's payload is the unrevised first vintage.
"""
import argparse
import json
import re
from datetime import date, datetime, time
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources.public import verify_observation

MONTHS = {name: i for i, name in enumerate(
    ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
     'September', 'October', 'November', 'December'), 1)}
FIELDS = ('unfixed_sales', 'sales_change', 'unfixed_purchases', 'purchases_change',
          'open_interest', 'oi_change')
MONTH_PATTERN = '|'.join(MONTHS)
DATE_MONTHS = {**MONTHS, **{name[:3]: value for name, value in MONTHS.items()}, 'Sept': 9}
DATE_MONTH_PATTERN = '|'.join(sorted(DATE_MONTHS, key=len, reverse=True))
DELIVERY = re.compile(rf'^({MONTH_PATTERN})\s+[\'’]?(\d{{2}}|\d{{4}})$', re.IGNORECASE)


class _ReportHTML(HTMLParser):
    """Extract table cells and legacy preformatted text without executing HTML."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text, self.rows, self.pre = [], [], []
        self.row, self.cell, self.in_pre, self.hidden = None, None, False, 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if tag == 'pre':
            self.in_pre = True
        if tag == 'tr':
            self.row = []
        if tag in ('td', 'th'):
            self.cell = []
        if tag in ('br', 'p', 'tr'):
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)
        if tag == 'pre':
            self.in_pre = False
        if tag in ('td', 'th') and self.cell is not None:
            if self.row is not None:
                self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        if tag in ('p', 'td', 'th', 'tr'):
            self.text.append('\n')

    def handle_data(self, value):
        if self.hidden:
            return
        self.text.append(value)
        if self.cell is not None:
            self.cell.append(value)
        if self.in_pre:
            self.pre.append(value)


def _date(value):
    value = value.strip().rstrip('.')
    if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{2,4}', value):
        month, day, year = map(int, value.split('/'))
        year = year + (2000 if year < 70 else 1900) if year < 100 else year
    else:
        match = re.fullmatch(rf'({DATE_MONTH_PATTERN})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})', value, re.IGNORECASE)
        if not match:
            raise ValueError('Unrecognized printed date')
        month, day, year = DATE_MONTHS[match[1].title()], int(match[2]), int(match[3])
    return date(year, month, day)


def _number(value):
    clean = value.strip().rstrip('*').strip()
    if clean in ('', '-', '—', 'N/A', 'NA'):
        return None
    if not re.fullmatch(r'-?(?:\d{1,3}(?:,\d{3})+|\d+)', clean):
        raise ValueError('Contract quantity must be an integer; no silent filling')
    return int(clean.replace(',', ''))


def parse_report(content):
    page = _ReportHTML()
    page.feed(content.decode('utf-8', errors='strict') if isinstance(content, bytes) else content)
    text = ''.join(page.text)
    if not re.search(r'\(In\s+Contracts\)', text, re.IGNORECASE):
        raise ValueError('Cotton On-Call contract units required')
    asof = re.search(rf'as\s+of\s+(\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|'
                    rf'(?:{DATE_MONTH_PATTERN})\.?\s+\d{{1,2}},?\s+\d{{4}})', text, re.IGNORECASE)
    if not asof:
        raise ValueError('Reported as-of date required')
    observed = _date(asof[1])
    rows = [row for row in page.rows if row and (DELIVERY.fullmatch(row[0]) or
            re.fullmatch(r'Totals:?', row[0], re.IGNORECASE))]
    header_asof, header_matches = None, None
    if rows:
        headers = [r for r in page.rows if r and 'Unfixed' in r[0]]
        patterns = (r'Unfixed\s+Call\s+Sales', r'Change', r'Unfixed\s+Call\s+Purchases',
                    r'Change', r'At\s+Close', r'Change')
        if len(headers) != 1 or len(headers[0]) != 6 or not all(
                re.search(pattern, value, re.IGNORECASE) for pattern, value in zip(patterns, headers[0], strict=True)):
            raise ValueError('Column semantics/order not verified')
        header_asof = re.sub(r'^At\s+Close\s*', '', headers[0][4], flags=re.IGNORECASE)
        try:
            header_matches = _date(header_asof) == observed
        except ValueError:
            header_matches = False
    if not rows:
        # Old CFTC reports use a fixed-width pre block, including revision stars.
        for line in ''.join(page.pre).splitlines():
            line = line.strip().lstrip('|').strip()
            match = re.match(rf'((?:{MONTH_PATTERN})\s+[\'’]?\d{{2,4}}|Totals:?)\s+(.+)$', line, re.IGNORECASE)
            if match:
                rows.append([match[1], *match[2].split()])
    contracts, totals = [], None
    for row in rows:
        if len(row) != 7:
            raise ValueError('Expected six contract quantity columns')
        values = dict(zip(FIELDS, map(_number, row[1:]), strict=True))
        if any(values[name] is not None and values[name] < 0 for name in
               ('unfixed_sales', 'unfixed_purchases', 'open_interest')):
            raise ValueError('Negative contract level')
        revised = [name for name, value in zip(FIELDS, row[1:], strict=True) if '*' in value]
        if re.fullmatch(r'Totals:?', row[0], re.IGNORECASE):
            if totals is not None:
                raise ValueError('Duplicate totals')
            totals = values
            continue
        match = DELIVERY.fullmatch(row[0])
        month, year = MONTHS[match[1].title()], int(match[2])
        year = year + (2000 if year < 70 else 1900) if year < 100 else year
        if month not in (3, 5, 7, 10, 12):
            raise ValueError('Unrecognized Cotton delivery month')
        contracts.append({'delivery_month': f'{year:04d}-{month:02d}', **values,
                          'revised_change_fields': revised})
    if not contracts or totals is None or len({r['delivery_month'] for r in contracts}) != len(contracts):
        raise ValueError('Unique contract rows and one totals row required')
    checks = {}
    for field in FIELDS:
        complete = totals[field] is not None and all(r[field] is not None for r in contracts)
        if complete and sum(r[field] for r in contracts) != totals[field]:
            raise ValueError(f'Printed {field} total does not match contract rows')
        checks[field] = 'verified' if complete else 'unknown_missing_quantity'
    release = re.search(
        rf'Release[d]?\s+after\s+(\d{{1,2}}):(\d{{2}})\s*p\.?m\.?\s*Eastern\s+time[,\s|]*'
        rf'((?:{DATE_MONTH_PATTERN})\.?\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})', text, re.IGNORECASE)
    bound = None
    footer_dates, footer_conflict = [], False
    if release:
        day = _date(release[3])
        hour, minute = int(release[1]), int(release[2])
        if not 1 <= hour <= 12 or not 0 <= minute < 60:
            raise ValueError('Invalid printed Eastern release time')
        local = datetime.combine(day, time(hour % 12 + 12, minute), tzinfo=ZoneInfo('America/New_York'))
        if day < observed:
            raise ValueError('Release footer precedes the report as-of date')
        bound = local.astimezone(ZoneInfo('UTC')).isoformat()
        footer_dates = sorted({_date(v).isoformat() for v in re.findall(
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text[release.end():release.end()+160])})
        footer_conflict = any(value != day.isoformat() for value in footer_dates)
    revision_note = bool(re.search(r'revised\s+data|revis(?:ion|ed)', text, re.IGNORECASE))
    threshold = re.search(r'futures\s+positions\s+of\s+(\d+)\s+or\s+more\s+contracts', text, re.IGNORECASE)
    return {'schema': 'cotton-oncall-format-audit-v1', 'as_of': observed.isoformat(),
            'units': 'contracts', 'contracts': contracts, 'totals': totals, 'total_checks': checks,
            'reporting_threshold_contracts': int(threshold[1]) if threshold else None,
            'revision_note_present': revision_note,
            'oi_header_asof_text': header_asof, 'oi_header_matches_report_asof': header_matches,
            'footer_additional_dates': footer_dates, 'footer_date_conflict': footer_conflict,
            'zero_oi_rows': sum(r['open_interest'] == 0 for r in contracts),
            'missing_quantity_cells': sum(r[k] is None for r in contracts for k in FIELDS),
            'printed_release_not_before_utc': bound, 'release_bound_exclusive': bound is not None,
            'published_at': None, 'available_by_historical': None,
            'historical_vintage_verified': False, 'model_eligible': False,
            'availability_policy': 'Footer is a lower bound only; retrieval proves availability now, not historical vintage'}


def audit_receipt(receipt):
    receipt = Path(receipt).resolve()
    body = verify_observation(receipt)
    if body['kind'] != 'cftc' or 'deaoncall' not in body['source_url'].lower():
        raise ValueError('CFTC Cotton On-Call observation required')
    raw = receipt.parent.parent / body['source_file']
    return {**parse_report(raw.read_bytes()), 'source_url': body['source_url'],
            'source_sha256': body['source_sha256'], 'observed_available_at': body['observed_available_at']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_receipt(args.receipt)
    freeze_record(args.output, result)
    assert read_record(args.output)['source_sha256'] == result['source_sha256']
    print(json.dumps({k: result[k] for k in ('as_of', 'source_sha256', 'model_eligible')}, indent=2))


if __name__ == '__main__':
    main()
