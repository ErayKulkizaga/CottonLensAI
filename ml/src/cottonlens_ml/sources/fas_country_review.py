"""Four labelled country-stock checks, never a historical admission certificate."""
import calendar
import re
from datetime import date
from decimal import Decimal
from html.parser import HTMLParser

from cottonlens_ml.sources.fas_archive import reconcile_upland, report_period_end

# Named scope fixed before inspecting country API values. Pakistan's report
# abbreviation is reviewed against the printed row, not inferred from values.
COUNTRIES = (('China', 'CHINA', 'CN'), ('Vietnam', 'VIETNAM', 'VN'),
             ('Turkey', 'TURKEY', 'TR'), ('Pakistan', 'PAKISTN', 'PK'))
STOCK_COLUMNS = {'outstandingSales': 0, 'accumulatedExports': 2, 'nextMYOutstandingSales': 4}
SUBTYPES = {
    1401: 'COTTON - UPLAND RAW, 1 1/16 INCHES AND OVER',
    1402: 'COTTON - UPLAND RAW, 1 INCH UP TO 1 1/16 INCHES',
    1403: 'COTTON - UPLAND - RAW, UNDER 1 INCH',
}


class ReferenceRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag == 'td' and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == 'td' and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def country_reference(html):
    parser = ReferenceRows()
    parser.feed(html)
    result, codes = {}, set()
    for name, label, iso in COUNTRIES:
        matches = [r for r in parser.rows if len(r) == 3 and r[0] == name]
        if len(matches) != 1 or not re.fullmatch(r'\d{4}', matches[0][1]) or matches[0][2] != iso:
            raise ValueError('One exact country/code/ISO reference row required')
        code = int(matches[0][1])
        if code in codes:
            raise ValueError('Duplicate reference country code')
        codes.add(code)
        result[label] = {'name': name, 'code': code, 'iso': iso}
    return result


def checked_table_metadata(section, report_day):
    unit = re.search(r'1000 RUNNING BALES\s+AS OF\s+([A-Z]+)\s+(\d{1,2}),?\s*(\d{4})', section, re.IGNORECASE)
    if not unit:
        raise ValueError('Explicit running-bale unit and as-of date required')
    months = {name.upper(): number for number, name in enumerate(calendar.month_name) if name}
    day = date(int(unit[3]), months[unit[1].upper()], int(unit[2]))
    if day != report_day:
        raise ValueError('Report summary and country table periods differ')
    header = (r'DESTINATION\s*:\s*THIS WEEK\s*:\s*YR AGO\s*:\s*THIS WEEK\s*:'
              r'\s*YR AGO\s*:\s*SECOND YR\s*:\s*THIRD YR')
    if not re.search(header, section):
        raise ValueError('Explicit six-column stock header required')
    return unit, day


def country_section(text):
    headings = list(re.finditer(r'ALL UPLAND COTTON\s+MARKETING YEAR\s+08/01\s*-\s*07/31', text))
    if len(headings) != 1:
        raise ValueError('One ALL UPLAND COTTON table required')
    section = text[headings[0].end():]
    following = re.search(r'\n[^\n]+\s+MARKETING YEAR\s+\d{2}/\d{2}\s*-\s*\d{2}/\d{2}', section)
    if following:
        section = section[:following.start()]
    unit, day = checked_table_metadata(section, report_period_end(text))
    # Canonicalize only the spelling/punctuation of the cover's date for the
    # existing total verifier. Source PDF bytes and all numeric tokens stay intact.
    canonical = f'1000 RUNNING BALES AS OF {day.strftime("%B").upper()} {day.day:02d}, {day.year}'
    canonical_text = text[:headings[0].end()] + section.replace(unit[0], canonical, 1)
    return section, day.isoformat(), canonical_text


def reconcile_countries(payload, commodity_catalog, report_text, reference_html):
    reference = country_reference(reference_html)
    section, day, canonical_text = country_section(report_text)
    national = reconcile_upland(payload, commodity_catalog, canonical_text)
    if not national['content_matches_at_printed_precision']:
        raise ValueError('National stock totals must reconcile before country review')
    selected = {r['countryCode']: r for r in payload if r.get('weekEndingDate') == day + 'T00:00:00'}
    countries = []
    for label, record in reference.items():
        matches = re.findall(r'^\s*' + re.escape(label) + r'\s*:\s*([^\n]+)$', section, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError('Country row absent/ambiguous; no region or zero substitution')
        tokens = matches[0].split()
        if len(tokens) != 6 or any(not re.fullmatch(r'\d+\.\d', token) for token in tokens):
            raise ValueError('Six explicit stock values required; suppression is unknown')
        if record['code'] not in selected:
            raise ValueError('Country API row absent; never impute zero')
        api = selected[record['code']]
        comparisons = {}
        for field, column in STOCK_COLUMNS.items():
            printed = int(Decimal(tokens[column]) * 1000)
            value = api.get(field)
            if type(value) is not int or value < 0:
                raise ValueError('Nonnegative integer country stock required')
            comparisons[field] = {'api_value': value, 'report_printed_value': printed,
                                  'difference_bales': value - printed, 'rounding_half_width': 50,
                                  'within_printed_precision': abs(value - printed) <= 50}
        commitment = api.get('currentMYTotalCommitment')
        if type(commitment) is not int or commitment != api['outstandingSales'] + api['accumulatedExports']:
            raise ValueError('Country commitment accounting identity mismatch')
        countries.append({**record, 'report_label': label, 'comparisons': comparisons,
                          'commitment_api_value': commitment,
                          'commitment_is_derived_not_separately_printed': True,
                          'all_stock_values_match': all(c['within_printed_precision'] for c in comparisons.values())})
    return {'observation_date': day, 'unit': 'running_bales', 'countries': countries,
            'direct_stock_checks': len(countries) * len(STOCK_COLUMNS),
            'all_stock_values_match': all(c['all_stock_values_match'] for c in countries),
            'national_stock_values_match': True,
            'reference_basis': 'four current Census Schedule C code/name/ISO pairs; not full historical FAS catalog',
            'weekly_exports_verified': False, 'weekly_net_sales_verified': False,
            'historical_code_validity_verified': False, 'publication_timestamp_verified': False,
            'first_version_verified': False, 'model_eligible': False, 'release_allowed': False}


def subtype_precision(payload, commodity_catalog, report_text, reference_html):
    """Test a rounding explanation without changing the direct stock gate.

    Printed grade values are not exact API grade values. Compatibility of
    their summed rounding intervals cannot establish the publisher's algorithm,
    identical vintages or historical availability.
    """
    direct = reconcile_countries(payload, commodity_catalog, report_text, reference_html)
    sections = {}
    for code, heading in SUBTYPES.items():
        catalog = [r for r in commodity_catalog if r.get('commodityCode') == code]
        if len(catalog) != 1 or catalog[0].get('unitId') != 2:
            raise ValueError('Pinned running-bale subtype catalog required')
        matches = list(re.finditer(re.escape(heading) + r'\s+MARKETING YEAR\s+08/01\s*-\s*07/31', report_text))
        if len(matches) != 1:
            raise ValueError('One exact subtype heading required')
        section = report_text[matches[0].end():]
        following = re.search(r'\n[^\n]+\s+MARKETING YEAR\s+\d{2}/\d{2}\s*-\s*\d{2}/\d{2}', section)
        if following:
            section = section[:following.start()]
        # Reuse exact unit/date/header checks; no guessed page positions,
        # substituted commodity headings or numeric matching of Pima rows.
        checked_table_metadata(section, date.fromisoformat(direct['observation_date']))
        sections[code] = section
    countries = []
    # This scope addresses the two already observed mismatches, not a search
    # for the subset of countries that passes a looser bound.
    for record in direct['countries']:
        if record['report_label'] not in ('CHINA', 'PAKISTN'):
            continue
        parts = []
        for code, section in sections.items():
            rows = re.findall(r'^\s*' + re.escape(record['report_label']) + r'\s*:\s*([^\n]+)$', section, re.MULTILINE)
            if len(rows) != 1:
                raise ValueError('Subtype country row absent/ambiguous; never impute zero')
            tokens = rows[0].split()
            if len(tokens) != 6 or any(not re.fullmatch(r'\d+\.\d', token) for token in tokens):
                raise ValueError('Six explicit subtype values required; suppression is unknown')
            printed = int(Decimal(tokens[2]) * 1000)
            parts.append({'commodity_code': code, 'printed_accumulated_bales': printed,
                          'assumed_rounding_min_bales': max(0, printed - 50),
                          'assumed_rounding_max_bales': printed + 50})
        comparison = record['comparisons']['accumulatedExports']
        low = sum(p['assumed_rounding_min_bales'] for p in parts)
        high = sum(p['assumed_rounding_max_bales'] for p in parts)
        countries.append({'name': record['name'], 'country_code': record['code'],
                          'parts': parts, 'printed_subtype_sum_bales': sum(p['printed_accumulated_bales'] for p in parts),
                          'printed_all_upland_bales': comparison['report_printed_value'],
                          'api_all_upland_bales': comparison['api_value'],
                          'direct_difference_bales': comparison['difference_bales'],
                          'direct_within_50_bales': comparison['within_printed_precision'],
                          'assumed_sum_min_bales': low, 'assumed_sum_max_bales': high,
                          'api_compatible_with_subtype_rounding': low <= comparison['api_value'] <= high})
    return {'scope': 'China/Pakistan accumulatedExports only; explanatory diagnostic',
            'observation_date': direct['observation_date'], 'countries': countries,
            'assumption': 'Each printed grade rounds independently to nearest 100 running bales; endpoints inclusive, stocks nonnegative',
            'publisher_rounding_method_verified': False, 'exact_api_grade_values_present': False,
            'rounding_only_cause_verified': False, 'vintage_verified': False,
            'direct_stock_gate_changed': False, 'publication_timestamp_verified': False,
            'first_version_verified': False, 'model_eligible': False, 'release_allowed': False}
