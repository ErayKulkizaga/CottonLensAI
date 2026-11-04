"""Bounded USDA source acquisition for the fixed pre-2010 risk replication."""

import io
import json
import re
import struct
import zipfile
import zlib
from datetime import date, datetime
from decimal import Decimal

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record

from research.ams_full_daily_source import MONTHS

BASE = 'https://apps.ams.usda.gov/Cotton/HistoricalUplandPrices/'
ARCHIVES = ('2008-2009', '2009-2010')
MAX_CAPTURE_BYTES = 40_000_000
EXCHANGE = r'(?:NEW YORK FUTURES|ICE FUTURES US)\s*-\s*CONTRACT\s+NO\.\s*2'
PROVIDER = ('Settlement prices provided by New York Cotton Exchange',
            'Settlement prices provided by ICE Futures US')


def delivery(day):
    year, month = map(int, day[:7].split('-'))
    return next((f'{year}-{m:02}-01' for m in MONTHS.values() if m > month),
                f'{year+1}-03-01')


def source_plan(metadata, dates):
    remaining = set(dates)
    output = []
    for season in ARCHIVES:
        receipt = json.loads((metadata/f'{season}.json').read_bytes())
        url = BASE+season+'%20DSQ.zip'
        tail = metadata/f'{season}-tail.bin'
        if receipt['url'] != url or digest(tail) != receipt['tail_sha256']:
            raise ValueError('USDA archive directory identity differs')
        start, _, total = map(int, re.findall(r'\d+', receipt['content_range']))
        with zipfile.ZipFile(io.BytesIO(tail.read_bytes())) as archive:
            entries = sorted(archive.infolist(), key=lambda m: m.header_offset)
        members = {}
        stop = 0
        for index, member in enumerate(entries):
            match = re.fullmatch(r'CN(\d{4})(\d\d)(\d\d)DDSQ\.(pdf|xls)', member.filename, re.IGNORECASE)
            if not match:
                continue
            year, month, day, suffix = match.groups()
            date = f'{year}-{month}-{day}'
            if date not in dates:
                continue
            if suffix.lower() in members.setdefault(date, {}):
                raise ValueError('Duplicate dated report of the same format')
            offset = member.header_offset+start
            end = entries[index+1].header_offset+start if index+1 < len(entries) else total
            if not (0 <= offset < end <= total and 0 < member.file_size <= 2_000_000):
                raise ValueError('Unsafe member bounds')
            members[date][suffix.lower()] = {
                'name': member.filename, 'offset': offset, 'end': end,
                'crc32': member.CRC, 'bytes': member.file_size,
                'compressed_bytes': member.compress_size, 'method': member.compress_type,
            }
            stop = max(stop, end)
        if not members or not set(members) <= remaining:
            raise ValueError('Missing or duplicated seasonal source')
        remaining.difference_update(members)
        begin = min(m['offset'] for row in members.values() for m in row.values())
        output.append({'season': season, 'url': url, 'start': begin, 'stop': stop,
                       'total': total, 'members': members, 'tail_sha256': digest(tail)})
    if remaining or sum(s['stop']-s['start'] for s in output) > MAX_CAPTURE_BYTES:
        raise ValueError('Incomplete planned sessions or archive byte budget exceeded')
    return output


def archive_member(data, source, member):
    offset = member['offset']-source['start']
    header = struct.unpack_from('<4s5H3I2H', data, offset)
    if header[0] != b'PK\x03\x04' or header[2] & 1 or header[3] != member['method']:
        raise ValueError('ZIP member header differs')
    end_name = offset+30+header[-2]
    name = data[offset+30:end_name].decode('utf-8' if header[2] & 2048 else 'cp437')
    begin = end_name+header[-1]
    end = begin+member['compressed_bytes']
    if name != member['name'] or end > member['end']-source['start']:
        raise ValueError('ZIP member boundary differs')
    if member['method'] == 8:
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(data[begin:end], member['bytes']+1)
        if not decoder.eof:
            raise ValueError('Incomplete bounded ZIP member')
    elif member['method'] == 0:
        raw = data[begin:end]
    else:
        raise ValueError('Unexpected ZIP method')
    if len(raw) != member['bytes'] or zlib.crc32(raw) != member['crc32']:
        raise ValueError('ZIP checksum mismatch')
    return raw


def pdf_quotes(raw, day, *, previous=False):
    from pypdf import PdfReader

    page = PdfReader(io.BytesIO(raw)).pages[0].extract_text()
    printed = {datetime.strptime(s, '%d-%b-%y').date().isoformat()  # noqa: DTZ007 - printed date
               for s in re.findall(r'\b\d{1,2}-[A-Z][a-z]{2}-\d{2}\b', page)}
    compact = re.sub(r'\s+', ' ', page)
    if (printed != {day} or not re.search(EXCHANGE, compact)
            or not any(p in compact for p in PROVIDER)
            or 'Month Today Previous Change' not in compact):
        raise ValueError('Printed daily date or settlement semantics differ')
    return validated_quotes([
        (f'20{y}-{MONTHS[m]:02}-01', Decimal(p), Decimal(b), Decimal(delta))
        for m, y, p, b, delta in re.findall(
            r'^(Mar|May|Jul|Oct|Dec)-(\d{2})\s+(\d+\.\d{2})\s+(\d+\.\d{2})\s+(-?\d+\.\d{2})(?:\s|$)',
            page, re.MULTILINE)], day, previous=previous)


def xls_quotes(raw, day):
    import xlrd

    try:
        book = xlrd.open_workbook(file_contents=raw, on_demand=True, formatting_info=True)
        sheet = book.sheet_by_name('page1')
    except xlrd.XLRDError as exc:
        raise ValueError('Unreadable BIFF settlement workbook') from exc
    dates = {xlrd.xldate_as_datetime(cell.value, book.datemode).date().isoformat()
             for row in range(12) for cell in sheet.row(row) if cell.ctype == xlrd.XL_CELL_DATE}
    text = ' '.join(cell.value for row in range(sheet.nrows) for cell in sheet.row(row)
                    if cell.ctype == xlrd.XL_CELL_TEXT)
    starts = [i for i in range(sheet.nrows)
              if sheet.row_values(i)[1:4] == ['Today', 'Previous', 'Change']]
    if (dates != {day} or len(starts) != 1 or not re.search(EXCHANGE, text)
            or not any(p in text for p in PROVIDER)):
        raise ValueError('Workbook printed date or settlement semantics differ')
    rows = []
    for row in range(starts[0]+1, starts[0]+11):
        contract, *values = sheet.row(row)[:4]
        if contract.ctype != xlrd.XL_CELL_DATE:
            break
        if book.format_map[book.xf_list[contract.xf_index].format_key].format_str != 'mmm-yy':
            raise ValueError('Workbook contract is not displayed as delivery month')
        month = xlrd.xldate_as_datetime(contract.value, book.datemode).replace(day=1).date()
        if month.month not in MONTHS.values() or any(v.ctype != xlrd.XL_CELL_NUMBER for v in values):
            continue
        exact = [Decimal(str(v.value)) for v in values]
        cents = [v.quantize(Decimal('.01')) for v in exact]
        if any(abs(a-b) > Decimal('1e-10') for a, b in zip(exact, cents, strict=True)):
            continue
        rows.append((month.isoformat(), *cents))
    book.release_resources()
    return validated_quotes(rows, day)


def validated_quotes(rows, day, *, previous=False):
    months = [m for m, *_ in rows]
    if months != sorted(set(months)):
        raise ValueError('Duplicate or unsorted contract months')
    return {m: str(b if previous else p) for m, p, b, delta in rows
            if m[:7] >= day[:7] and min(p, b) > 0 and p-b == delta}


def normalize(source, data, required, late):
    if len(data) != source['stop']-source['start']:
        raise ValueError('Archive span byte count differs')
    rows = {}
    for day, members in source['members'].items():
        if day in late:
            continue
        errors = {}
        for suffix, parser in (('pdf', pdf_quotes), ('xls', xls_quotes)):
            if suffix not in members:
                continue
            member = members[suffix]
            raw = archive_member(data, source, member)
            try:
                quotes = parser(raw, day)
                if not set(required[day]) <= set(quotes):
                    raise ValueError('One or more required named Today quotes missing')
            except (KeyError, ValueError) as exc:
                errors[suffix] = str(exc)
                continue
            rows[day] = {'quotes': {m: quotes[m] for m in required[day]},
                         'member': member['name'], 'preferred_report_errors': errors}
            break
        if day not in rows:
            raise ValueError(f'{day}: no admissible scheduled source: {errors}')
    return rows


def capture(root, sources):
    import requests

    receipts = []
    for source in sources:
        path = root/(source['season']+'.bin')
        receipt_path = root/(source['season']+'-receipt.json')
        if not path.exists():
            begin, stop = source['start'], source['stop']
            with requests.get(source['url'], stream=True, allow_redirects=False, timeout=(15, 35),
                              headers={'Range': f'bytes={begin}-{stop-1}', 'Accept-Encoding': 'identity'}) as response:
                if (response.status_code != 206 or response.headers.get('Content-Range')
                        != f"bytes {begin}-{stop-1}/{source['total']}"):
                    raise ValueError('The official server did not honor the frozen span')
                data = response.raw.read(stop-begin+1)
            if len(data) != stop-begin:
                raise ValueError('Incomplete or oversized source response')
            with path.open('xb') as f:
                f.write(data)
            freeze_record(receipt_path, {'source': source['url'], 'sha256': digest(path),
                                         'range': [begin, stop], 'bytes': len(data)})
        if not receipt_path.exists():
            raise ValueError('An existing partial archive lacks its capture receipt')
        receipts.append(read_record(receipt_path))
    return receipts


def weekly_quotes(raw, observation, publication, required):
    from pypdf import PdfReader

    observed = date.fromisoformat(observation)
    published = date.fromisoformat(publication)
    if not 0 < (published-observed).days <= 8:
        raise ValueError('Weekly dated price is outside the printed publication week')
    pages = [page.extract_text() for page in PdfReader(io.BytesIO(raw)).pages]
    if not re.search(rf'{published:%B}\s+{published.day},\s*{published.year}', pages[0]):
        raise ValueError('Weekly printed publication date differs')
    matches = []
    for page in pages:
        if not re.search(r'Futures\s+(?:Contract\s+)?Settlement',page,re.IGNORECASE):
            continue
        header = re.findall(
            r'((?:(?:Mar|May|Jul|Oct|Dec)[ -]+\d\s*\d\s+){5,6})(?:Average|7 Market)',
            page,re.IGNORECASE)
        values = re.findall(rf'^{observed:%b}[\s-]+0?{observed.day}\s+(.+)$',page,re.MULTILINE)
        if len(header)!=1 or len(values)!=1:
            continue
        months = [f"20{y.replace(' ', '')}-{MONTHS[m.title()]:02}-01"
                  for m,y in re.findall(r'(Mar|May|Jul|Oct|Dec)[ -]+(\d\s*\d)',
                                        header[0],re.IGNORECASE)]
        compact = re.sub(r'(?<![\d.])(\d)\s+(\d\.\d{2})(?=\s|$)',r'\1\2',values[0])
        tokens = compact.split()[:len(months)]
        if len(tokens)!=len(months) or months!=sorted(set(months)):
            continue
        quotes = {m:p for m,p in zip(months,tokens,strict=True)
                  if re.fullmatch(r'\d+\.\d{2}',p) and Decimal(p)>0}
        if set(required)<=set(quotes):
            matches.append({m:quotes[m] for m in required})
    if len(matches)!=1:
        raise ValueError(f'One dated named weekly price row required: {observation}')
    return matches[0]


def weekly_payload(path, url, acquire):
    if acquire and not path.exists():
        import requests

        with requests.get(url,stream=True,timeout=15,allow_redirects=False) as response:
            if response.status_code!=200:
                raise ValueError('Exact official weekly source is unavailable')
            raw = response.raw.read(250_001)
        if not raw.startswith(b'%PDF-') or len(raw)>250_000:
            raise ValueError('Unexpected bounded weekly document')
        with path.open('xb') as file:
            file.write(raw)
    return path.read_bytes()
