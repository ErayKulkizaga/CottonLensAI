"""Bound source requests to actual unresolved dependency cells; never admit data."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from cottonlens_ml.research.ledger import read_record


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cells, origins):
    groups = defaultdict(list)
    seen = set()
    for cell in cells:
        key = (cell['source_date'], cell['contract'])
        if key in seen:
            raise ValueError('Duplicate unresolved source cell')
        seen.add(key)
        for field in ('source_date', 'contract', 'first_using_origin', 'last_using_origin'):
            if date.fromisoformat(cell[field]).isoformat() != cell[field]:
                raise ValueError('Canonical source/request dates required')
        if (int(cell['using_origin_count']) <= 0
                or not cell['source_date'] <= cell['first_using_origin'] <= cell['last_using_origin']):
            raise ValueError('Invalid unresolved dependency interval/count')
        groups[cell['contract']].append(cell)
    edges = sum(int(c['using_origin_count']) for c in cells)
    if edges != sum(int(r['ewma_unknown_cells']) for r in origins if r['ewma_unknown_cells']):
        raise ValueError('Cell and origin dependency-edge totals differ')
    rows = []
    for contract, values in sorted(groups.items()):
        kinds = Counter(v['kind'] for v in values)
        rows.append({'contract': contract, 'unknown_cells_upper_bound': len(values),
                     'first_source_date': min(v['source_date'] for v in values),
                     'last_source_date': max(v['source_date'] for v in values),
                     'first_using_origin': min(v['first_using_origin'] for v in values),
                     'last_using_origin': max(v['last_using_origin'] for v in values),
                     'cell_origin_dependency_edges': sum(int(v['using_origin_count']) for v in values),
                     'no_accepted_table': kinds['no_accepted_original_source_table'],
                     'partial_prefix': kinds['verified_weekly_prefix'],
                     'identity_or_price_conflict': kinds['unresolved_identity_or_price'],
                     'report_without_listing_coverage': sum(kinds[k] for k in (
                         'unchanged_annual_outside_known_bad_blocks',
                         'unchanged_annual_outside_known_bad_blocks_with_prefix_witness',
                         'dated_daily', 'verified_full_daily_settlement')),
                     'required_evidence': 'official_listed_on,last_trade_on,source_document_hash; '
                                          'dated_settlements_or_documented_absence_for_listed_sessions; '
                                          'publication_and_vintage_proof_separately'})
        if sum(kinds.values()) != sum(rows[-1][k] for k in
                                     ('no_accepted_table', 'partial_prefix', 'identity_or_price_conflict',
                                      'report_without_listing_coverage')):
            raise ValueError('Unrecognized unresolved source kind')
    return rows, edges


def build(readiness, output):
    if output.exists():
        raise ValueError('Preserve previous request matrix')
    report = read_record(readiness/'result.json')
    names = ('unknown-ewma-cells.csv', 'origin-readiness.csv')
    for name in names:
        if digest(readiness/name) != report['output_hashes'][name]:
            raise ValueError('Readiness dependency input changed')
    tables = []
    for name in names:
        with (readiness/name).open(encoding='utf-8', newline='') as f:
            tables.append(list(csv.DictReader(f)))
    cells, origins = tables
    if len(cells) != report['unknown_ewma_source_cells_upper_bound']:
        raise ValueError('Unresolved source cell count differs')
    rows, edges = summarize(cells, origins)
    if not rows:
        raise ValueError('No unresolved cells to request; this does not imply admission')
    output.mkdir(parents=True)
    matrix = output/'contract-evidence-request.csv'
    with matrix.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result = {'schema': 1, 'status': 'source_requests_bounded_not_data_admitted',
              'contracts': len(rows), 'unknown_cells_upper_bound': len(cells),
              'unique_source_dates': len({r['source_date'] for r in cells}),
              'source_kind_counts': dict(sorted(Counter(r['kind'] for r in cells).items())),
              'cell_origin_dependency_edges': edges,
              'input_hashes': {n: digest(readiness/n) for n in (*names, 'result.json')},
              'matrix_sha256': digest(matrix), 'generator_sha256': digest(Path(__file__)),
              'new_fits': 0, 'source_admission_granted': False,
              'limits': ['Upper bounds are not proven price errors or proof a contract was listed.',
                         'Edges reuse origins and cells; they are not unique fits, trials or affected dates.',
                         'Lifecycle evidence may establish absence; no listing date is inferred here.',
                         'This request scope does not resolve conditional annual data or historical availability.']}
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readiness', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.readiness, args.output), indent=2))
