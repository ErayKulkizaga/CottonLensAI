"""Independently check full-daily PDF values and preserve source disagreements."""

import argparse
import calendar
import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from cottonlens_ml.research.ledger import freeze_record, read_record

from research.ams_contract_reconciliation import cents, parse_daily, sha
from research.ams_weekly_recovery import annual_prefix_quotes


def layout_quotes(text, expected):
    compact = re.sub(r'\s+', '', text).lower()
    if ('settlementpricesprovidedbyicefuturesus' not in compact
            or 'icefuturesus-contractno.2' not in compact):
        raise ValueError('Independent ICE field identification failed')
    dates = {datetime.strptime(s, '%d-%b-%y').date().isoformat()  # noqa: DTZ007
             for s in re.findall(r'\b\d{2}-[A-Za-z]{3}-\d{2}\b', text)}
    if dates != {expected}:
        raise ValueError('Independent printed date mismatch')
    matches = re.findall(r'^\s*(Mar|May|Jul|Oct|Dec)-(\d{2})\s+(\d+\.\d{2})\s+(\d+\.\d{2})\s+(-?\d+\.\d{2})(?:\s|$)', text, re.MULTILINE)
    if len(matches) != 10:
        raise ValueError('Independent ten-row extraction failed')
    quotes, rows = {}, []
    months = {m: i for i, m in enumerate(calendar.month_abbr) if m}
    for month, year, today, previous, change in matches:
        contract = f'20{year}-{months[month]:02d}-01'
        if Decimal(today)-Decimal(previous) != Decimal(change) or min(Decimal(today), Decimal(previous)) <= 0:
            raise ValueError('Independent price arithmetic mismatch')
        quotes[contract] = today
        rows.append({'contract': contract, 'today': today, 'previous': previous, 'change': change})
    if list(quotes) != sorted(quotes) or len(quotes) != 10:
        raise ValueError('Independent maturity order mismatch')
    return quotes, rows


def verify(capture, parent, output):
    import pdfplumber

    if output.exists():
        raise ValueError('Preserve independent source verification')
    plan = json.loads((capture/'plan.json').read_text())
    if plan['parser_sha256'] != sha(Path(__file__).with_name('ams_full_daily_source.py')):
        raise ValueError('Capture parser changed')
    summary = json.loads((capture/'summary.json').read_text())
    if summary['receipts'] != len(plan['dates']) or summary['new_fits'] != 0:
        raise ValueError('Incomplete source capture')
    annual = annual_prefix_quotes(parent/'inputs/annual.csv')
    daily = read_record(parent/'verified-daily-overlay.json')
    output.mkdir(parents=True); (output/'layout').mkdir()
    quotes, comparisons, sources = {}, [], {}
    for day in plan['dates']:
        folder = capture/day; receipt = json.loads((folder/'receipt.json').read_text())
        for name, digest in receipt['files'].items():
            if Path(name).name != name or sha(folder/name) != digest:
                raise ValueError('Captured file changed')
        if receipt['status'] != 'literal_settlements_parsed_not_admitted':
            continue
        page = (folder/'page.html').read_text(encoding='utf-8')
        fields = set(re.findall(r'<time\s+datetime="([^"]+)"', page))
        if fields != {receipt['page_date_field']} or next(iter(fields))[:10] != day:
            raise ValueError('Independent release page date mismatch')
        with pdfplumber.open(folder/'report.pdf') as pdf:
            text_bytes = (pdf.pages[0].extract_text()+'\n').encode('utf-8')
        (output/'layout'/(day+'.txt')).write_bytes(text_bytes)
        values, rows = layout_quotes(text_bytes.decode('utf-8'), day)
        parsed = json.loads((folder/'settlements.json').read_text())
        if parsed['quotes'] != values or parsed['rows'] != rows or parsed['observation_date'] != day:
            raise ValueError('Independent extractor disagreement')
        quotes[day] = values
        old = annual.get(day, {})
        daily_values = None
        if day in daily['raw_sources']:
            digest = daily['raw_sources'][day]['document_sha256']
            raw = parent/'inputs/daily'/(digest+'.txt')
            if sha(raw) != digest:
                raise ValueError('Old daily witness changed')
            daily_values = parse_daily(raw.read_bytes(), day)
        comparisons.append({'date': day, 'annual_contracts': list(old), 'full_daily_contracts': list(values),
                            'annual_second': list(old)[1] if len(old) >= 2 else None,
                            'full_daily_second': list(values)[1],
                            'annual_missing_contracts': sorted(set(values)-set(old)),
                            'annual_extra_contracts': sorted(set(old)-set(values)),
                            'annual_common_value_differences': [{'contract': c, 'annual': old[c]['price'], 'daily': p}
                              for c, p in values.items() if c in old and (not old[c]['valid'] or cents(old[c]['price']) != Decimal(p))],
                            'excerpt_present': daily_values is not None,
                            'excerpt_same_ten_values_and_names': daily_values == values if daily_values else None})
        sources[day] = {'url': receipt['url'], 'pdf_sha256': sha(folder/'report.pdf'),
                        'page_sha256': sha(folder/'page.html'), 'receipt_sha256': sha(folder/'receipt.json'),
                        'independent_layout_sha256': hashlib.sha256(text_bytes).hexdigest(),
                        'printed_date': day, 'page': 1, 'source_field': 'ICE settlement Today',
                        'page_date_field_is_not_verified_publication_clock': True}
    if len(quotes) != summary['parsed']:
        raise ValueError('Independent source scope differs')
    overlay = {'quotes': quotes, 'sources': sources, 'training_allowed': False,
               'historical_availability_verified': False, 'first_vintage_verified': False,
               'scope': 'Literal current full-daily official report values. Not first vintage, daily UTC proof or a market-model result. Previous column retained only as arithmetic control; not dated/imputed.'}
    freeze_record(output/'verified-full-daily-overlay.json', overlay)
    result = {'status': 'independently_verified_full_daily_values', 'new_fits': 0, 'verified_days': len(quotes),
              'decimal_price_triplets': 10*len(quotes), 'decimal_price_scalars': 30*len(quotes),
              'comparisons': comparisons, 'capture_summary_sha256': sha(capture/'summary.json'),
              'capture_plan_sha256': sha(capture/'plan.json'), 'verifier_sha256': sha(Path(__file__)),
              'independent_extractor': 'pdfplumber', 'independent_extractor_version': pdfplumber.__version__,
              'overlay_sha256': sha(output/'verified-full-daily-overlay.json'),
              'training_allowed': False, 'historical_availability_verified': False, 'market_skill': False}
    freeze_record(output/'result.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'parent', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(verify(a.capture, a.parent, a.output)))
