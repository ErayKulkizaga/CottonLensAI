"""Zero-fit audit of frozen inner selection dates in a restored experiment."""
import argparse
import json
from pathlib import Path

import pandas as pd
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import frozen_outer_folds, validate_inner_fold


def audit(folder):
    ready = read_record(folder/'ready.json')
    history_path = folder/'history.parquet'
    if digest(history_path) != ready['history_sha256']:
        raise ValueError('Frozen selection history checksum differs')
    identity = ready['identity']
    split = identity['split']
    if identity.get('design', {}).get('split', split) != split:
        raise ValueError('Frozen design and execution selection splits differ')
    folds = frozen_outer_folds(split['folds'])
    history = pd.read_parquet(history_path)
    summary = []
    for fold in folds.values():
        validate_inner_fold(history, fold)
        summary.append({'fold': fold['fold'], 'year': fold.get('year'),
                        'outer_start': min(fold['origins']), 'outer_origins': len(fold['origins']),
                        'inner_origin_counts': [len(b['origins']) for b in fold['inner']]})
    return {'schema': 1, 'status': 'recorded_selection_boundaries_verified',
            'profile': identity.get('profile'), 'source_id': identity.get('source_id'),
            'design_id': identity.get('design_id'), 'split_id': split.get('split_id'),
            'ready_sha256': digest(folder/'ready.json'), 'history_sha256': digest(history_path),
            'validator_sha256': digest(Path(__file__).resolve().parents[1]/
                                       'ml/src/cottonlens_ml/research/protocol.py'),
            'folds': summary, 'new_fits': 0, 'market_skill': False,
            'historical_source_availability_proven': False,
            'all_selection_procedures_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing selection audit; use a new output path')
    result = audit(args.folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    freeze_record(args.output, result)
    print(json.dumps(result, indent=2))
