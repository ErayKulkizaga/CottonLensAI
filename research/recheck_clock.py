"""Zero-fit recheck of the completed timing pilot's native states and predictions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ml/src'))
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.availability_clock import output, validate_design
from cottonlens_ml.research.ledger import Ledger, freeze_record
from cottonlens_ml.research.protocol import Preprocessor, Target, inputs, rows_at
from cottonlens_ml.research.reproduction import verified_native_predictions
from cottonlens_ml.research.review import verify_history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    root = args.experiment_root
    ready, history = verify_history(root)
    design = validate_design({'design_id': ready['identity']['design_id'], 'design': ready['identity']['design']})
    ledger = Ledger(root/'ledger', ready['identity'])
    maximum, count = 0., 0
    for path in sorted((root/'ledger/completed').glob('*.json')):
        rec = json.loads(path.read_bytes())
        spec = rec['specification']
        recorded, _ = verified_native_predictions(ledger, rec['experiment_id'], spec)
        train, test = rows_at(history, spec['train_dates']), rows_at(history, spec['test_dates'])
        if not train.target_date_5.lt(test.date.min()).all():
            raise ValueError('Immature training labels')
        state = root/'ledger'/rec['payload_root']
        adapter = json.loads((state/'adapter.json').read_bytes())
        processor = Preprocessor.fit(train, spec['recipe']['features'])
        h = spec['recipe']['horizon']
        target = Target.fit(train, h, spec['recipe']['target'])
        if processor.as_dict() != adapter['processor'] or target.as_dict() != adapter['target']:
            raise ValueError('Training-only native state mismatch')
        model = json.loads((state/adapter['model_file']).read_bytes())
        raw = inputs(history, test, processor, 1) @ np.asarray(model['coef'], dtype=np.float32) + np.float32(model['intercept'])
        replay = target.inverse(raw, test.cotton_close.to_numpy())
        maximum = max(maximum, float(np.max(np.abs(replay-recorded))))
        count += 1
    if count != 676 or maximum > 1e-7:
        raise ValueError('Incomplete or divergent native timing evidence')
    metrics, hashes = [], {}
    csv = root/'reports/outer-predictions.csv'
    exported = pd.read_csv(csv, float_precision='round_trip')
    for h in (1, 5):
        for group in ('control', 'available'):
            frames = []
            for fold in design['split']['folds']:
                name = f'{group}-t{h}-year{fold["year"]}.json'
                _, frame = output(root, name, fold, group, h, design, history, namespace='clock')
                frames.append(frame)
                hashes[name] = digest(root/'clock-outputs'/name)
            frame = pd.concat(frames, ignore_index=True)
            other = exported.loc[exported.horizon.eq(h) & exported.group.eq(group)]
            for field in ('cotton_close', f'target_return_{h}', 'predicted_return', 'raw_predicted_return'):
                if not np.array_equal(frame[field], other[field]):
                    raise ValueError('Export differs from verified annual predictions')
            if frame.date.tolist() != other.date.tolist():
                raise ValueError('Export origin mismatch')
            price = frame.cotton_close.to_numpy()
            actual = frame[f'target_return_{h}'].to_numpy()
            naive = float(np.mean(price*np.abs(np.expm1(actual))))
            for field in ('predicted_return', 'raw_predicted_return'):
                mae = float(np.mean(price*np.abs(np.expm1(actual)-np.expm1(frame[field].to_numpy()))))
                metrics.append({'horizon': h, 'group': group, 'field': field, 'count': len(frame),
                                'naive_mae': naive, 'mae': mae, 'gain_pct': 100*(1-mae/naive)})
    freeze_record(args.output, {'status': 'verified', 'new_fits': 0, 'native_fits': count,
                              'max_native_log_return_difference': maximum,
                              'ready_sha256': digest(root/'ready.json'), 'csv_sha256': digest(csv),
                              'annual_outputs': hashes, 'metrics': metrics,
                              'scope': 'Reviewed historical timing pilot, not first-vintage or independent skill.'})
    print('Verified', count, 'native fits, 32 annual outputs, 8024 exported rows; zero new fits')


if __name__ == '__main__':
    main()
