"""Zero-fit native prediction causality under fixed recorded model/selection state."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.research.ledger import Ledger, read_record
from cottonlens_ml.research.naive_mae import FEATURES, features
from cottonlens_ml.research.protocol import Preprocessor, Target
from cottonlens_ml.research.reproduction import verified_native_predictions
from cottonlens_ml.research.temporal_audit import audit_predictions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(history, context):
    """Saved JSON state only: no scaler/target/model fitting or selection."""
    native = context['native']
    processor = Preprocessor(**native['processor'])
    target = Target(**native['target'])
    model = native['model']
    if processor.names != FEATURES or model.get('dtype', 'float32') not in ('float32', 'float64'):
        raise ValueError('Fixed price-only native feature/dtype contract required')
    built = features(history)
    dtype = model.get('dtype', 'float32')
    coef = np.asarray(model['coef'], dtype=dtype)
    intercept = np.asarray(model['intercept'], dtype=dtype)
    if coef.shape != (10,) or intercept.size != 1 or not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ValueError('Finite native linear state required')
    # Fixed per-row reduction avoids BLAS batch-size dependent roundoff when
    # comparing a prefix against the full frame. Native equivalence is checked.
    raw = np.sum(processor.transform(built)*coef, axis=1)+intercept.reshape(-1)[0]
    log = target.inverse(raw, built.cotton_close.to_numpy())
    result = pd.DataFrame({'date': built.date, 'raw_prediction': log,
                           'selected_prediction': context['selection']['weight']*log})
    outside = ~result.date.dt.strftime('%Y-%m-%d').isin(context['test_dates'])
    result.loc[outside, ['raw_prediction', 'selected_prediction']] = np.nan
    return result


def audit(folder):
    ready = read_record(folder/'ready.json')
    reference = read_record(folder/'reference/ready.json')
    path = folder/'reference/history.parquet'
    if digest(path) != reference['history_sha256']:
        raise ValueError('Frozen raw price history changed')
    history = pd.read_parquet(path)[['date', 'cotton_close', 'cotton_session_index']].copy()
    positions = {d: i for i, d in enumerate(history.date.dt.strftime('%Y-%m-%d'))}
    ledger = Ledger(folder/'ledger', ready['identity'])
    results = []
    maximum = 0.
    for filename in sorted((folder/'ledger/completed').glob('*.json')):
        record = read_record(filename)
        spec = record['specification']
        if not spec.get('role'):
            continue  # Existing generated learning controls are a separate scope.
        recipe = spec['recipe']
        if (recipe['family'] not in ('ridge', 'quantile') or recipe['horizon'] != 1
                or recipe['features'] != FEATURES or recipe['research_profile'] != 'price-only-median-t1-v1'):
            raise ValueError('Unexpected market fit recipe')
        saved, _ = verified_native_predictions(ledger, record['experiment_id'], spec)
        root = ledger.root/record['payload_root']
        adapter = json.loads((root/'adapter.json').read_bytes())
        model = json.loads((root/adapter['model_file']).read_bytes())
        selection = {'kind': 'raw_inner_validation', 'weight': 1.}
        match = re.fullmatch(r'path-outer-(ridge|quantile)-(\d{4})-\d+', spec['role'])
        output = None
        if match:
            family, year = match.groups()
            output_path = folder/f'median-outputs/{family}-t1-year{year}.json'
            output = read_record(output_path)
            if output['recipe'] != recipe or output['weight'] not in (0., .25, .5, .75, 1.):
                raise ValueError('Frozen outer selection changed')
            selection = {'kind': 'frozen_outer', 'weight': output['weight'],
                         'decision_sha256': output['decision_sha256'], 'output_sha256': digest(output_path)}
        elif not re.fullmatch(r'inner-\d{4}-\d+-\d+', spec['role']):
            raise ValueError('Unexpected fit role; no inferred selection state')
        context = {'recipe': recipe, 'selection': selection, 'test_dates': spec['test_dates'],
                   'native': {'processor': adapter['processor'], 'target': adapter['target'], 'model': model}}
        baseline = predict(history, context).set_index('date').loc[pd.to_datetime(spec['test_dates'])]
        error = float(np.max(np.abs(baseline.raw_prediction.to_numpy()-saved)))
        if error > 1e-7:
            raise ValueError('Audit predictor disagrees with saved native predictions')
        maximum = max(maximum, error)
        if output is not None:
            expected = {r['date']: r['predicted_return'] for r in output['records']}
            np.testing.assert_allclose(baseline.selected_prediction,
                                       [expected[d] for d in spec['test_dates']], rtol=0, atol=1e-7)
        cutoffs = sorted({positions[spec['test_dates'][0]]+1, positions[spec['test_dates'][-1]]+1})
        result = audit_predictions(history, predict, context, ['raw_prediction', 'selected_prediction'], cutoffs)
        results.append({'fit_id': record['experiment_id'], 'role': spec['role'],
                        'cutoffs': cutoffs, 'fixed_state_id': result['fixed_state_id'],
                        'prediction_frame_id': result['prediction_frame_id']})
    if len(results) != 338 or len({r['fit_id'] for r in results}) != 338:
        raise ValueError('Complete338market fit scope required')
    return {'schema': 1, 'status': 'fixed_native_prediction_causality_verified',
            'market_records_checked': 338, 'checks_at_first_and_last_test_origins': results,
            'maximum_saved_native_return_difference': maximum, 'native_equivalence_tolerance': 1e-7,
            'raw_history_sha256': digest(path), 'ready_sha256': digest(folder/'ready.json'),
            'pipeline_sha256': digest(Path(__file__)),
            'builder_sha256': digest(Path(features.__code__.co_filename)),
            'audit_sha256': digest(Path(audit_predictions.__code__.co_filename)),
            'new_fits': 0, 'historical_availability_verified': False,
            'all_selection_procedures_verified': False, 'market_skill': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous prediction audit')
    result = audit(args.folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks_at_first_and_last_test_origins'}, indent=2))
