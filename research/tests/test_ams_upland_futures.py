"""The official archive's two Futures layouts must not change contract identity."""

import io
import zipfile

import pytest

from research.ams_upland_futures import workbook_rows


def book(headers, values):
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    # Encode text as shared strings, as in the official workbooks.
    strings = list(dict.fromkeys(str(item) for row in (headers, values) for item in row
                                if not str(item).replace('.', '', 1).isdigit()))
    rows = []
    for number, cells in enumerate((headers, values), 1):
        data = ''.join(f'<c r="{chr(65+i)}{number}" t="s"><v>{strings.index(str(value))}</v></c>'
                       if str(value) in strings else
                       f'<c r="{chr(65+i)}{number}"><v>{value}</v></c>'
                       for i, value in enumerate(cells))
        rows.append(f'<row r="{number}">{data}</row>')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" '
                         'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                         '<sheets><sheet name="Futures" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels',
                         '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
                         '</Relationships>')
        archive.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}">'
                         + ''.join(f'<si><t>{value}</t></si>' for value in strings) + '</sst>')
        archive.writestr('xl/worksheets/sheet1.xml',
                         f'<worksheet xmlns="{ns}"><sheetData>{"".join(rows)}</sheetData></worksheet>')
    return stream.getvalue()


def test_older_contract_month_not_exchange_code():
    data = book(['Date', 'Contract Month', 'Contract', 'Settlement Price'],
                [41718, 41760, 'US', 92.18])
    row, = workbook_rows(data, '20140320DSQ data file.xlsx')
    assert (row['date'], row['contract'], row['settlement_cents_raw']) == (
        '2014-03-20', '2014-05-01', '92.18')
    assert row['source_date_matches'] and row['price_precision_2dp']


def test_consolidated_bad_date_and_subcent_stay_flagged():
    data = book(['Source.Name', 'Date', 'Contract', 'Settlement Price'],
                ['20180518Futures.xlsx', 42712, 42795, 71.421])
    row, = workbook_rows(data, '2017-2018 Price file.xlsx')
    assert not row['source_date_matches']
    assert not row['price_precision_2dp']
    assert row['settlement_cents_raw'] == '71.421'


def test_older_other_exchange_fails_closed():
    data = book(['Date', 'Contract Month', 'Contract', 'Settlement Price'],
                [41718, 41760, 'Other', 92.18])
    with pytest.raises(ValueError, match='exchange code'):
        workbook_rows(data, '20140320DSQ data file.xlsx')
