"""Reconcile reviewed report numbers with a checksummed API snapshot.

This diagnostic never grants model eligibility. Matching values do not prove an
original publication timestamp, revision history, or completeness of other weeks.
"""
import argparse
import json
import math
from pathlib import Path

from cottonlens_ml.code_identity import digest, safe_member


def reconcile(review_file):
    review_file = Path(review_file)
    review = json.loads(review_file.read_text(encoding='utf-8'))
    root = review_file.parent.resolve()
    for name, expected in review['files'].items():
        path = (root / name).resolve()
        if not safe_member(name) or not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('Evidence checksum/path mismatch')
    provider = review['provider']
    if provider not in ('nass', 'fas'):
        raise ValueError('Unsupported reconciliation provider')
    if review['api_file'] not in review['files'] or review['report_file'] not in review['files']:
        raise ValueError('API and report evidence must be checksummed')
    payload = json.loads((root / review['api_file']).read_text(encoding='utf-8'))
    rows = payload['data'] if provider == 'nass' else payload
    if not isinstance(rows, list) or not rows or not review['observations']:
        raise ValueError('Nonempty API and reviewed observations required')
    results = []
    identities = set()
    for item in review['observations']:
        identity = json.dumps([item['selector'], item['field']], sort_keys=True)
        if identity in identities:
            raise ValueError('Duplicate reviewed observation')
        identities.add(identity)
        matches = [row for row in rows if all(row.get(k) == v for k, v in item['selector'].items())]
        if not item.get('report_page') or not item.get('report_label'):
            raise ValueError('Report page and label required for reviewer traceability')
        if provider == 'nass':
            if len(matches) != 1:
                raise ValueError('NASS selector must match exactly one record')
        else:
            # Aggregate countries within one commodity/week, never across MYs.
            if not {'commodityCode', 'weekEndingDate'} <= set(item['selector']) or not matches:
                raise ValueError('FAS requires a nonempty commodity/week selection')
            keys = [(r['commodityCode'], r['countryCode'], r['weekEndingDate']) for r in matches]
            if len(set(keys)) != len(keys) or len({r['unitId'] for r in matches}) != 1:
                raise ValueError('FAS duplicate countries or mixed units')
            if item.get('unit_id') != matches[0]['unitId']:
                raise ValueError('FAS report/API unit identity required')
        try:
            values = [float(str(r[item['field']]).replace(',', '')) for r in matches]
            expected = float(item['report_value'])
            quantum = float(item['report_rounding_quantum'])
        except (ValueError, KeyError, TypeError):
            raise ValueError('Finite numeric report/API values required; no suppression imputation') from None
        if not all(math.isfinite(v) for v in [*values, expected, quantum]) or quantum < 0:
            raise ValueError('Invalid numeric value or rounding quantum')
        actual = sum(values)
        delta = actual - expected
        results.append({'report_label': item['report_label'], 'report_page': item['report_page'],
                        'api_value': actual, 'report_value': expected, 'difference': delta,
                        'rounding_tolerance': quantum / 2, 'matched_rows': len(matches),
                        'matches_report_precision': abs(delta) <= quantum / 2})
    return {'provider': provider, 'review_sha256': digest(review_file),
            'results': results, 'all_values_match': all(r['matches_report_precision'] for r in results),
            'model_eligible': False, 'publication_timestamp_verified': False,
            'scope': 'Only the selected observations; not a full-history vintage certification',
            'next_gate': 'Independently review original release timestamp and revision provenance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = reconcile(args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
