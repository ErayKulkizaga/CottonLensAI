"""Inference-only release validation; importing training frameworks is prohibited."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    args = parser.parse_args()
    for name in ('tensorflow', 'keras', 'catboost', 'mlflow'):
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f'Training package leaked into inference environment: {name}')
    sys.path.insert(0, str(args.repo / 'backend'))
    from app.research_runtime import ResearchModel
    from app.runtime import ArtifactRuntime
    runtime = ArtifactRuntime(str(args.artifact))
    runtime.load()
    snapshots = pq.read_table(args.artifact / 'feature_snapshots.parquet').to_pylist()
    parity = json.loads((args.artifact / 'parity.json').read_text())
    for row in parity:
        h = row['horizon']
        value = runtime.predict_return(h, snapshots[-runtime.required_window(h):])
        if not np.isfinite(value) or abs(value - row['primary_return']) > 1e-6:
            raise ValueError('Primary backend parity mismatch')
        entry = next(m for m in runtime.manifest['production_models'] if m['horizon'] == h)
        if entry.get('experimental_members'):
            model = ResearchModel(args.artifact, {**entry, 'members': entry['experimental_members']})
            actual = model.predict(snapshots[-model.window:])
            if abs(actual - row['expected_learned_return']) > 1e-6:
                raise ValueError('Experimental learned model CPU parity mismatch')
    print('Research release: checksum, CPU inference and primary/experimental parity passed; no training stack')


if __name__ == '__main__':
    main()
