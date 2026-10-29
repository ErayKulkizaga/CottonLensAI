"""Independent decimal/raw-source verification of a zero-fit weekly recovery."""

import argparse
import csv
import hashlib
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from cottonlens_ml.research.ledger import freeze_record, read_record


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, plan_path, output):
    from pypdf import PdfReader

    report = read_record(root/'result.json')
    for name, expected in {**report['input_hashes'], **report['output_hashes']}.items():
        if checksum(root/name) != expected:
            raise ValueError('Recovery payload hash mismatch')
    if checksum(Path(__file__).with_name('ams_weekly_recovery.py')) != report['source_sha256']:
        raise ValueError('Recovery source mismatch')
    parent = root/'inputs'/'parent'
    before = read_record(parent/'result.json')
    source = read_record(parent/'verified-daily-overlay.json')
    result = read_record(root/'weekly-overlay.json')
    capture = root/'inputs'/'capture'
    identity = json.loads((capture/'identity.json').read_text())
    if (checksum(plan_path) != identity['plan_sha256']
            or checksum(Path(__file__).with_name('ams_weekly_capture.py')) != identity['source_sha256']):
        raise ValueError('Fixed capture source/plan mismatch')
    plan = json.loads(plan_path.read_text())
    journals = {p.parent.name: json.loads(p.read_text()) for p in (capture/'reports').glob('*/receipt.json')}
    if sorted(journals) != plan['candidate_publication_dates']:
        raise ValueError('Incomplete capture scope')
    total_bytes = 0
    for pub, r in journals.items():
        for name, digest in r['files'].items():
            if checksum(capture/'reports'/pub/name) != digest:
                raise ValueError('Captured source mismatch')
        total_bytes += r.get('bytes', 0)
        if r.get('bytes', 0) > plan['max_file_bytes']:
            raise ValueError('Per-file source budget')
    if total_bytes > plan['max_total_bytes']:
        raise ValueError('Total source budget')
    annual = {}
    with (parent/'inputs'/'annual.csv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r['source_date_matches'] != 'True':
                continue
            valid = r['price_precision_2dp'] == 'True'
            value = Decimal(r['settlement_cents_raw'])
            canonical = value.quantize(Decimal('.01'))
            if valid and (not value.is_finite() or value <= 0 or abs(value-canonical) > Decimal('1e-8')):
                raise ValueError('Annual cell is not an encoded two-decimal price')
            annual.setdefault(r['date'], {})[r['contract']] = (canonical if valid else value, valid)
    manual = json.loads((root/'inputs'/'transcription.json').read_text())['tables']
    accepted, conflicts, rows, scalar_count = {}, [], [], 0
    for pub, t in sorted(manual.items()):
        r = journals[pub]
        folder = capture/'reports'/pub
        if not (r['status'] == 'candidate_for_value_identity_review' and r['printed_publication_date'] == pub):
            raise ValueError('No exact printed publication')
        # Independently inspect the PDF header date without the production parser.
        header = PdfReader(folder/'report.pdf').pages[0].extract_text()
        named = date.fromisoformat(pub)
        stamp = re.search(r'\b'+named.strftime('%B')+r'\s+'+str(named.day)+r',?\s+'+str(named.year)+r'\b', header)
        if stamp is None:
            raise ValueError('Publication absent from raw PDF header')
        for line in t['rows']:
            day, *cells = line.split(); quotes = {}
            if cells != ['Holiday']:
                quotes = {c: str(Decimal(v)) for c, v in zip(t['contracts'], cells, strict=True) if v != '-'}
            scalar_count += len(quotes)
            daily = None
            if day in source['raw_sources']:
                digest = source['raw_sources'][day]['document_sha256']
                raw = (parent/'inputs'/'daily'/(digest+'.txt')).read_bytes()
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError('Raw daily digest')
                table_bytes = raw.split(b'FUTURES', 1)[1].split(b'COTTON QUOTATIONS', 1)[0]
                names = {'Mar': '03', 'May': '05', 'Jul': '07', 'Oct': '10', 'Dec': '12'}
                pairs = re.findall(rb'(Mar|May|Jul|Oct|Dec)-(\d\d)\s+(\d+\.\d+)\s*(?:\r?\n|$)', table_bytes)
                daily = {'20'+y.decode()+'-'+names[m.decode()]+'-01': Decimal(v.decode()) for m, y, v in pairs}
                if len(daily) != 10:
                    raise ValueError('Independent ten-quote daily extraction')
            cells = [v for _, v in sorted(annual.get(day, {}).items())]
            values = [v for v, _ in cells]
            invalid_prefix = any(not valid for _, valid in cells[:len(quotes)])
            mismatch = bool(quotes) and len(values) == 10 and not invalid_prefix and (
                any(Decimal(v) != values[i] for i, v in enumerate(quotes.values()))
                or (daily is not None and any(daily.get(c) != Decimal(v) for c, v in quotes.items())))
            status = ('explicit_holiday' if not quotes else 'incomplete_annual_no_admission' if len(values) != 10
                      else 'invalid_annual_prefix_no_admission' if invalid_prefix
                      else 'price_conflict_no_admission' if mismatch else 'verified_partial_prefix')
            found = [r for r in result['rows'] if r['date'] == day]
            if len(found) != 1 or found[0]['status'] != status or found[0]['quotes'] != quotes:
                raise ValueError(f'Independent row status/cells disagree at {day}: expected={status}, stored={[r["status"] for r in found]}')
            if mismatch:
                conflicts.append(day)
            if status == 'verified_partial_prefix':
                accepted[day] = quotes
            rows.append(day)
    if result['quotes'] != accepted or len(rows) != len(set(rows)):
        raise ValueError('Recovered subset differs')
    old_unknown = set(before['unresolved_suspect_days'])
    added = sorted(old_unknown & set(accepted)); remaining = sorted(old_unknown - set(accepted))
    if (report['newly_resolved_suspect_prefix_days'] != added
            or report['remaining_unresolved_suspect_days'] != remaining
            or report['price_conflict_days'] != conflicts or report['visible_quote_cells'] != scalar_count
            or report['accepted_partial_days'] != len(accepted) or report['original_ordinal_origins'] != 2006):
        raise ValueError('Independent totals differ')
    with (parent/'inputs'/'cohort.csv').open(encoding='utf-8', newline='') as f:
        cohort = list(csv.DictReader(f))
    if len(cohort) != 2006 or len({r['date'] for r in cohort}) != 2006:
        raise ValueError('Original full cohort lost')
    for body in (result, report):
        for k in ('training_allowed', 'historical_availability_verified', 'first_vintage_verified'):
            if body[k] is not False:
                raise ValueError('Partial value evidence overclaimed')
    receipt = {'status': 'verified_partial_value_evidence', 'new_fits': 0, 'reports': len(manual),
               'reviewed_rows': len(rows), 'decimal_price_cells': scalar_count, 'capture_bytes': total_bytes,
               'accepted_prefix_days': len(accepted), 'new_suspect_prefix_days': len(added),
               'remaining_unknown_days': len(remaining), 'conflicts': conflicts,
               'original_ordinal_origins': len(cohort), 'report_sha256': checksum(root/'result.json'),
               'source_sha256': checksum(Path(__file__)), 'training_allowed': False,
               'historical_availability_verified': False, 'market_skill': False}
    freeze_record(output, receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'plan', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(verify(a.root, a.plan, a.output)))
