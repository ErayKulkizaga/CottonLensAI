"""Independent raw-token/Decimal checks for the derived source overlay; no training."""

import argparse
import calendar
import csv
import json
import re
from datetime import date, datetime
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd
from cottonlens_ml.research.ledger import read_record

from research.ams_contract_reconciliation import sha


def verify(root):
    result = read_record(root/'result.json'); overlay = read_record(root/'verified-daily-overlay.json')
    for name, expected in {**result['input_hashes'], **result['output_hashes']}.items():
        if sha(root/name) != expected:
            raise ValueError('Frozen source audit byte mismatch')
    panel = read_record(root/'inputs/daily-panel.json')
    annual = {}
    for r in csv.DictReader((root/'inputs/annual.csv').open(encoding='utf-8', newline='')):
        if r['source_date_matches'] == 'True' and r['price_precision_2dp'] == 'True':
            annual.setdefault(r['date'], {})[r['contract']] = Decimal(r['settlement_cents_raw']).quantize(Decimal('.01'))
    raw, checked = {}, 0
    for row in panel['rows']:
        p = root/'inputs/daily'/f'{row["document_sha256"]}.txt'
        if not p.exists():
            continue
        if sha(p) != row['document_sha256']:
            raise ValueError('Wrong dated raw report')
        blob = p.read_bytes(); first = re.search(rb'\b(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-[A-Za-z]{3}-\d{2})\b', blob[:600])
        assert first is not None
        text = first[0].decode('ascii')
        day = datetime.strptime(text, '%m/%d/%Y' if '/' in text else '%d-%b-%y').date().isoformat()  # noqa: DTZ007 - printed calendar date
        assert day == row['report_date']
        table = blob[blob.index(b'FUTURES'):blob.index(b'COTTON QUOTATIONS')]
        quotes = {}
        for line in table.splitlines():
            parts = line.split()
            if len(parts) < 2 or re.fullmatch(rb'(Mar|May|Jul|Oct|Dec)-\d{2}', parts[-2]) is None:
                continue
            name = parts[-2].decode('ascii'); month = list(calendar.month_abbr).index(name[:3])
            contract = date(2000+int(name[4:]), month, 1).isoformat()
            assert contract not in quotes
            quotes[contract] = Decimal(parts[-1].decode('ascii'))
        assert len(quotes) == 10 and list(quotes) == sorted(quotes)
        original = dict(sorted(annual.get(day, {}).items()))
        equivalent = list(original.values()) == list(quotes.values())
        if equivalent:
            assert day in overlay['quotes']
            assert quotes == {k: Decimal(v) for k, v in overlay['quotes'][day].items()}
            raw[day] = quotes
        else:
            assert day not in overlay['quotes']
        checked += 10
    rows = pd.read_csv(root/'label-reconciliation.csv', keep_default_na=False, na_values=[''], float_precision='round_trip')
    source = pd.read_csv(root/'inputs/cohort.csv', keep_default_na=False, na_values=[''], float_precision='round_trip')
    assert len(rows) == len(source) == 2006
    for name in ('date', 'target_date'):
        pd.testing.assert_series_equal(rows[name], source[name], check_exact=True)
    target_checks, changed = 0, 0
    for row in rows.itertuples():
        available = row.date in raw and row.target_date in raw
        if not available:
            assert not row.source_pair_verified
            continue
        contract = list(raw[row.date])[1]
        if contract not in raw[row.target_date]:
            assert not row.source_pair_verified
            continue
        with localcontext() as ctx:
            ctx.prec = 50
            square = (raw[row.target_date][contract]/raw[row.date][contract]).ln()**2
        assert row.source_pair_verified and row.true_contract == contract
        assert abs(float(square)-row.true_risk) < 1e-14
        target_checks += 1
        if pd.notna(row.old_risk) and abs(float(square)-row.old_risk) > 1e-14+abs(row.old_risk)*1e-10:
            changed += 1
    assert target_checks == result['independently_reconstructed_target_pairs']
    assert changed == result['numeric_target_changes']
    return {'status': 'independently_verified_source_values', 'new_fits': 0,
            'daily_quote_scalar_checks': checked, 'decimal_target_checks': target_checks,
            'numeric_target_changes': changed, 'unchanged_ordinal_rows': len(rows),
            'result_sha256': sha(root/'result.json'), 'verifier_sha256': sha(Path(__file__)),
            'training_allowed': False, 'historical_availability_verified': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = verify(a.root)
    if a.output.exists():
        raise ValueError('Preserve existing verification receipt')
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
