"""Read only the named Futures sheet of official AMS upland price archives."""
import argparse
import calendar
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
PRICE_STEP = Decimal('0.01')
EXCEL_EPOCH = date(1899, 12, 30)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def excel_date(value):
    if value is None:
        raise ValueError('Missing Excel date')
    if re.fullmatch(r'\d+(?:\.\d+)?', value):
        serial = Decimal(value)
        if serial != int(serial):
            raise ValueError('Non-day Excel date')
        return EXCEL_EPOCH+timedelta(days=int(serial))
    return date.fromisoformat(value)


def contract_month(value):
    if re.fullmatch(r'\d+(?:\.\d+)?', value):
        day = excel_date(value)
        return day.replace(day=1)
    match = re.fullmatch(r'([A-Z][a-z]{2})-(\d{2})', value)
    if not match or match.group(1) not in calendar.month_abbr:
        raise ValueError(f'Unexpected contract month: {value}')
    return date(2000+int(match.group(2)), list(calendar.month_abbr).index(match.group(1)), 1)


def source_date(filename):
    match = re.match(r'(20\d{6})', Path(filename).name)
    if not match:
        return None
    text = match.group(1)
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def workbook_rows(blob, source_name):
    with zipfile.ZipFile(io.BytesIO(blob)) as book:
        workbook = ET.fromstring(book.read('xl/workbook.xml'))
        futures = [sheet for sheet in workbook.findall('.//x:sheet', NS)
                   if sheet.attrib['name'] == 'Futures']
        if len(futures) != 1:
            raise ValueError(f'One Futures sheet required: {source_name}')
        relation = futures[0].attrib[REL]
        rels = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        targets = [row.attrib['Target'] for row in rels
                   if row.attrib.get('Id') == relation]
        if len(targets) != 1 or not re.fullmatch(r'worksheets/sheet\d+\.xml', targets[0]):
            raise ValueError('Unsafe Futures worksheet target')
        strings = []
        if 'xl/sharedStrings.xml' in book.namelist():
            shared = ET.fromstring(book.read('xl/sharedStrings.xml'))
            strings = [''.join(node.text or '' for node in entry.findall('.//x:t', NS))
                       for entry in shared.findall('x:si', NS)]
        sheet = ET.fromstring(book.read('xl/'+targets[0]))
        rows = sheet.findall('.//x:sheetData/x:row', NS)
        if not rows:
            raise ValueError('Empty Futures sheet')

        def cells(row):
            values = {}
            for cell in row.findall('x:c', NS):
                value = cell.find('x:v', NS)
                if value is None:
                    continue
                text = strings[int(value.text)] if cell.attrib.get('t') == 's' else value.text
                values[re.match(r'[A-Z]+', cell.attrib['r']).group()] = text
            return values

        headers = {label: col for col, label in cells(rows[0]).items()}
        needed = {'Date', 'Settlement Price'}
        if not needed.issubset(headers):
            raise ValueError(f'Futures headers differ: {source_name}: {headers}')
        contract_col = headers.get('Contract Month', headers.get('Contract'))
        if contract_col is None:
            raise ValueError(f'No named contract month: {source_name}')
        has_source_col = 'Source.Name' in headers
        records = []
        for row in rows[1:]:
            values = cells(row)
            if not values:
                continue
            filename = values[headers['Source.Name']] if has_source_col else source_name
            trading_day = excel_date(values[headers['Date']])
            if 'Contract Month' in headers and values.get(headers.get('Contract')) != 'US':
                raise ValueError(f'Unexpected exchange code: {filename}')
            contract = contract_month(values[contract_col])
            raw = Decimal(values[headers['Settlement Price']])
            cents = raw.quantize(PRICE_STEP, rounding=ROUND_HALF_UP)
            if raw <= 0:
                raise ValueError(f'Nonpositive settlement: {filename}')
            records.append({'date': trading_day.isoformat(), 'contract': contract.isoformat(),
                            'settlement_cents_raw': str(raw), 'source_name': filename,
                            'source_date_matches': trading_day == source_date(filename),
                            'price_precision_2dp': abs(raw-cents) <= Decimal('1e-8')})
        return records


def archive_rows(path):
    if path.suffix.lower() == '.xlsx':
        return workbook_rows(path.read_bytes(), path.name)
    if path.suffix.lower() != '.zip':
        raise ValueError('Expected an AMS .xlsx or daily .zip')
    records = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not names:
            raise ValueError('Duplicate or empty daily archive')
        for name in names:
            if (Path(name).name != name or not name.lower().endswith('.xlsx')
                    or not re.fullmatch(r'20\d{6}.*\.xlsx', name)):
                raise ValueError(f'Unexpected daily archive member: {name}')
            records.extend(workbook_rows(archive.read(name), name))
    return records


def scan(paths, paired):
    observations = []
    sources = []
    for path in paths:
        rows = archive_rows(path)
        sources.append({'name': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size,
                        'rows': len(rows), 'dates': len({row['date'] for row in rows}),
                        'mismatched_rows': sum(not row['source_date_matches'] for row in rows),
                        'subcent_rows': sum(not row['price_precision_2dp'] for row in rows)})
        for row in rows:
            observations.append({**row, 'container': path.name})
    keys = defaultdict(list)
    ranked = defaultdict(set)
    for row in observations:
        if row['source_date_matches']:
            ranked[row['date']].add(row['contract'])
            if row['price_precision_2dp']:
                keys[(row['date'], row['contract'])].append(row)
    conflicts = [{'date': key[0], 'contract': key[1], 'values': sorted({v['settlement_cents_raw'] for v in rows})}
                 for key, rows in keys.items() if len({v['settlement_cents_raw'] for v in rows}) > 1]
    if conflicts:
        raise ValueError(f'Conflicting date/contract settlements: {conflicts[:3]}')
    available = defaultdict(set)
    for trading_day, contract in keys:
        available[trading_day].add(contract)
    with paired.open(newline='', encoding='utf-8') as stream:
        cohort = list(csv.DictReader(stream))
    if (len(cohort) != 2006 or len({row['date'] for row in cohort}) != 2006
            or any(row['horizon'] != '1' or not 2016 <= int(row['year']) <= 2023 for row in cohort)):
        raise ValueError('Original eight-year CT T+1 cohort required')
    yearly = []
    second_pairs = []
    for year in range(2016, 2024):
        dates = [row for row in cohort if int(row['year']) == year]
        current = sum(row['date'] in available for row in dates)
        target = sum(row['target_date'] in available for row in dates)
        pairs = sum(bool(available[row['date']] & available[row['target_date']]) for row in dates)
        second = 0
        for row in dates:
            # An invalid first/second price cannot silently promote the next
            # contract into that rank.
            origin_contracts = sorted(ranked[row['date']])
            contract = origin_contracts[1] if len(origin_contracts) >= 2 else None
            eligible = (contract is not None and contract in available[row['date']]
                        and contract in available[row['target_date']])
            if eligible:
                second += 1
                second_pairs.append({'date': row['date'], 'target_date': row['target_date'],
                                     'contract': contract})
        yearly.append({'year': year, 'origins': len(dates), 'current': current,
                       'target': target, 'same_contract_pair': pairs,
                       'origin_second_pair': second})
    return observations, {'profile': 'ams-upland-futures-source-preflight-v1', 'new_fits': 0,
                          'sources': sources, 'source_rows': len(observations),
                          'unique_dates': len(available),
                          'source_date_mismatches': [row for row in observations
                                                     if not row['source_date_matches']],
                          'subcent_rows': [row for row in observations
                                           if not row['price_precision_2dp']],
                          'duplicate_equal_keys': sum(len(rows)-1 for rows in keys.values()),
                          'yearly': yearly, 'cohort_pairs': sum(y['same_contract_pair'] for y in yearly),
                          'origin_second_pairs': second_pairs,
                          'source_clock_verified': False, 'training_allowed': False,
                          '2024_plus_used_for_selection': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted([*args.source_dir.glob('*.xlsx'), *args.source_dir.glob('*.zip')])
    if args.output.exists():
        raise ValueError('Immutable output namespace already exists')
    observations, result = scan(paths, args.paired)
    args.output.mkdir(parents=True)
    with (args.output/'observations.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader(); writer.writerows(observations)
    result['observations_sha256'] = sha(args.output/'observations.csv')
    result['paired_sha256'] = sha(args.paired)
    result['analysis_sha256'] = sha(__file__)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: value for k, value in result.items()
                      if k in ('source_rows', 'unique_dates', 'cohort_pairs', 'yearly', 'source_clock_verified')}))
