"""Versioned research inference adapters: NumPy, XGBoost and ONNX Runtime only."""
import json
from pathlib import Path

import numpy as np


class ResearchModel:
    def __init__(self, root, entry):
        self.root, self.entry = Path(root), entry
        self.members = []
        for member in entry.get('members', []):
            adapter = json.loads(self.checked_path(member['adapter']).read_text(encoding='utf-8'))
            processor, target = adapter['processor'], adapter['target']
            names = processor['names']
            if not names or len(set(names)) != len(names) or not isinstance(adapter['window'], int) or adapter['window'] < 1:
                raise ValueError('Invalid research feature/window contract')
            for key in ('median', 'mean', 'scale'):
                values = np.asarray(processor[key], dtype=float)
                if values.shape != (len(names),) or not np.isfinite(values).all():
                    raise ValueError('Invalid research preprocessing state')
            if (np.asarray(processor['scale']) <= 0).any():
                raise ValueError('Invalid research preprocessing scale')
            if (target['kind'] not in ('raw_log', 'scaled_log', 'price_delta')
                    or not np.isfinite([target['mean'], target['scale']]).all() or target['scale'] <= 0):
                raise ValueError('Invalid research target transform')
            path = self.checked_path(member['path'])
            kind = member['format']
            if kind == 'xgboost_json':
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(path)
                model.set_param({'device': 'cpu', 'nthread': 2})
            elif kind == 'onnx':
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                model = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            elif kind == 'linear_json':
                model = json.loads(path.read_text(encoding='utf-8'))
            else:
                raise ValueError('Unsupported research inference format')
            weight = float(member['weight'])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError('Invalid model weight')
            self.members.append((member, adapter, model))
        if self.members and abs(sum(float(m['weight']) for m, _, _ in self.members) - 1) > 1e-8:
            raise ValueError('Ensemble weights must sum to one')
        if not self.members and entry.get('name') != 'Naive':
            raise ValueError('Learned research model requires members')

    def checked_path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise ValueError('Unsafe or missing research model payload')
        return path

    @property
    def window(self):
        return max([int(a['window']) for _, a, _ in self.members] or [1])

    def predict(self, features):
        if not self.members:
            return 0.
        rows = features if isinstance(features, list) else [features]
        if len(rows) < self.window:
            raise ValueError(f'Research inference requires {self.window} ordered snapshots')
        result = 0.
        for member, adapter, model in self.members:
            processor, target, window = adapter['processor'], adapter['target'], int(adapter['window'])
            raw = np.asarray([[row.get(n, np.nan) if row.get(n) is not None else np.nan for n in processor['names']]
                              for row in rows[-window:]], dtype=float)
            finite = np.isfinite(raw)
            scaled = (np.where(finite, raw, processor['median']) - processor['mean']) / processor['scale']
            value = np.concatenate([scaled, (~finite).astype(float)], axis=1).astype(np.float32)
            value = value[-1:] if window == 1 else value[None, ...]
            if member['format'] == 'xgboost_json':
                import xgboost as xgb
                predicted = np.asarray(model.predict(xgb.DMatrix(value))).reshape(-1)
            elif member['format'] == 'onnx':
                predicted = np.asarray(model.run(None, {model.get_inputs()[0].name: value})[0]).reshape(-1)
            else:
                predicted = np.asarray(value @ np.asarray(model['coef']) + model['intercept']).reshape(-1)
            prediction = float(predicted[0]) * float(target['scale']) + float(target['mean'])
            if target['kind'] == 'price_delta':
                current = float(rows[-1]['cotton_close'])
                ratio = 1 + prediction / current
                if current <= 0 or ratio <= 0:
                    raise ValueError('Predicted price outside positive domain')
                prediction = float(np.log(ratio))
            if not np.isfinite(prediction):
                raise ValueError('Nonfinite research prediction')
            result += float(member['weight']) * prediction
        return result
