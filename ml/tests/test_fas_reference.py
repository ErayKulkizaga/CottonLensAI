"""Historical reference witnesses cannot certify source numbers or availability."""
import base64
import hashlib
import json

import pytest
from cottonlens_ml.code_identity import digest
from review_fas_reference import COUNTRIES, ORIGINAL, review, selected_pairs


def html():
    options = ''.join(f'<option value="9:{code}">{label}</option>' for label, code in COUNTRIES.items())
    return ('<select name="ctl00$MainContent$lbCountry">' + options + '</select>'
            '<select name="ctl00$MainContent$lbCommodity"><option value="1404">All Upland Cotton</option></select>')


def fixture(tmp_path):
    rows, receipts = [], []
    for stamp in ['20200417213154', '20200701005521']:
        row = {'urlkey': 'gov,usda,fas,apps)/esrquery/esrq.aspx', 'timestamp': stamp,
               'original': ORIGINAL, 'mimetype': 'text/html', 'statuscode': '200',
               'digest': base64.b32encode(hashlib.sha1(html().encode(), usedforsecurity=False).digest()).decode().rstrip('='),
               'length': '123'}
        rows.append(row)
        source = tmp_path / f'{stamp}.html'
        source.write_text(html(), encoding='utf-8')
        receipt = tmp_path / f'{stamp}.json'
        url = f'https://web.archive.org/web/{stamp}id_/{ORIGINAL}'
        receipt.write_text(json.dumps({'archive_index_row': row, 'status': 200, 'url': url,
                                      'final_url': url, 'credentials_sent': False,
                                      'file': source.name, 'sha256': digest(source)}), encoding='utf-8')
        receipts.append(receipt)
    index = tmp_path / 'index.json'
    index.write_text(json.dumps([list(rows[0]), *[list(r.values()) for r in rows]]), encoding='utf-8')
    return index, receipts


def test_reference_scope_never_admits_numerical_cell_vintage_or_model(tmp_path):
    index, receipts = fixture(tmp_path)
    body = review(index, receipts, tmp_path / 'review.json')
    assert body['four_code_name_pairs_observed']
    assert len(body['snapshots']) == 2
    for key in ['continuous_validity_between_snapshots_verified', 'full_historical_catalog_verified',
                'unit_verified', 'exact_accumulated_exports_cell_verified', 'publication_timestamp_verified',
                'first_version_verified', 'model_eligible']:
        assert body[key] is False
    assert body['market_fits'] == 0


@pytest.mark.parametrize('mutation', ['missing', 'wrong', 'duplicate', 'shadow', 'wrong_select', 'commodity'])
def test_wrong_or_ambiguous_country_identity_rejected(mutation):
    text = html()
    if mutation == 'missing':
        text = text.replace('PAKISTAN', 'OTHER')
    elif mutation == 'wrong':
        text = text.replace('9:5700', '9:9999')
    elif mutation == 'duplicate':
        text = text.replace('</select>', '<option value="9:5700">CHINA, PEOPLES REPUBLIC OF</option></select>', 1)
    elif mutation == 'shadow':
        text = text.replace('</select>', '<option value="7:5700">OTHER</option></select>', 1)
    elif mutation == 'wrong_select':
        text = text.replace('lbCountry', 'lbCommodity')
    else:
        text = text.replace('All Upland Cotton', 'Pima')
    with pytest.raises(ValueError):
        selected_pairs(text)


@pytest.mark.parametrize('mutation', ['bytes', 'receipt', 'index', 'digest', 'duplicate_snapshot', 'path'])
def test_invalid_provenance_never_persists_result(tmp_path, mutation):
    index, receipts = fixture(tmp_path)
    if mutation == 'bytes':
        (tmp_path / '20200417213154.html').write_text('changed')
    elif mutation == 'duplicate_snapshot':
        receipts = [receipts[0], receipts[0]]
    elif mutation == 'index':
        rows = json.loads(index.read_text())
        rows[1][2] = 'https://untrusted.example/'
        index.write_text(json.dumps(rows))
    elif mutation == 'digest':
        rows = json.loads(index.read_text())
        rows[1][5] = 'WRONG'
        index.write_text(json.dumps(rows))
        body = json.loads(receipts[0].read_text())
        body['archive_index_row']['digest'] = 'WRONG'
        receipts[0].write_text(json.dumps(body))
    else:
        body = json.loads(receipts[0].read_text())
        body['final_url' if mutation == 'receipt' else 'file'] = 'https://untrusted.example/' if mutation == 'receipt' else '../outside.html'
        receipts[0].write_text(json.dumps(body))
    with pytest.raises((ValueError, OSError)):
        review(index, receipts, tmp_path / 'never.json')
    assert not (tmp_path / 'never.json').exists()
