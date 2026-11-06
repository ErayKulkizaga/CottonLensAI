"""Zero-fit causality audit at every recorded fixed-median market fit cutoff."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.research.naive_mae import FEATURES, features
from cottonlens_ml.research.temporal_audit import audit_features


def audit(folder):
    path = folder/'reference/history.parquet'
    history = pd.read_parquet(path)
    positions = {d: i for i, d in enumerate(history.date.dt.strftime('%Y-%m-%d'))}
    cutoffs, fits = set(), []
    for p in (folder/'ledger/completed').glob('*.json'):
        record = read_record(p)
        spec = record['specification']
        if spec.get('role') and spec.get('test_dates'):
            if spec['test_dates'][0] not in positions:
                raise ValueError('Recorded market origin absent from frozen history')
            cutoffs.add(positions[spec['test_dates'][0]]+1)
            fits.append(record['experiment_id'])
    if len(fits) != 338 or len(set(fits)) != 338:
        raise ValueError('Complete fixed-median338market records required')
    result = audit_features(history, features, FEATURES, sorted(cutoffs))
    result.update(completed_market_records_inspected=338,
                  new_market_fits=0,
                  market_fit_ids_sha256=hashlib.sha256(json.dumps(sorted(fits)).encode()).hexdigest(),
                  historical_data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  builder_sha256=hashlib.sha256(Path(features.__code__.co_filename).read_bytes()).hexdigest(),
                  audit_sha256=hashlib.sha256(Path(audit_features.__code__.co_filename).read_bytes()).hexdigest())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous audit output')
    result = audit(args.folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': result['status'], 'unique_cutoffs': len(result['cutoffs']), 'new_fits': 0}))
