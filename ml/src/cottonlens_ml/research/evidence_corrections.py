"""Read-only reconstruction of legacy claims; write derived evidence to a new directory."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.diagnostics import VALIDATION_POLICY, control_failures
from cottonlens_ml.research.ledger import freeze_record, read_record


def legacy_correction(repo):
    root = repo / 'runtime/artifacts/current'
    manifest = json.loads((root / 'manifest.json').read_text())
    metrics = json.loads((root / 'metrics.json').read_text())
    source = subprocess.check_output(['git', 'show', manifest['git_sha'] + ':ml/src/cottonlens_ml/training.py'], cwd=repo)
    if b'aligned = test.iloc[59:]' not in source or b'test_x, _ = _sequences(test, scaler)' not in source:
        raise ValueError('Legacy sequence alignment cannot be established from artifact code')
    forecasts = pd.read_parquet(root / 'forecasts.parquet')
    result = []
    for h in (1, 5):
        rows = forecasts.loc[forecasts.origin_type.eq('backtest') & forecasts.horizon.eq(h)].sort_values('as_of_date')
        if len(rows) != 498 or rows.as_of_date.duplicated().any():
            raise ValueError('Legacy 498-origin reconstruction changed')
        naive = float((rows.actual_price - rows.current_price).abs().mean())
        stored = next(m['mae'] for m in metrics if m['model'] == 'Naive' and m['horizon'] == h)
        if not np.isclose(naive, stored, rtol=0, atol=1e-12):
            raise ValueError('Stored Naive loss cannot be reconstructed')
        matched = rows.iloc[59:]
        matched_naive = float((matched.actual_price - matched.current_price).abs().mean())
        lstm = next(m['mae'] for m in metrics if m['model'] == 'LSTM' and m['horizon'] == h)
        result.append({'horizon': h, 'naive_origins': 498, 'lstm_origins': len(matched),
            'lstm_mae_stored_aggregate': lstm, 'naive_full_mae': naive, 'naive_matched_mae': matched_naive,
            'unmatched_gain_pct_invalid': 100 * (1 - lstm / naive),
            'matched_aggregate_gain_pct': 100 * (1 - lstm / matched_naive),
            'matched_start': str(matched.as_of_date.iloc[0]), 'matched_end': str(matched.as_of_date.iloc[-1]),
            'paired_ci': None, 'lstm_prediction_payload_available': False,
            'classification': 'METHODOLOGICALLY COMPROMISED',
            'scope': 'Stored LSTM aggregate compared to reconstructed same-window Naive; no LSTM row-level verification'})
    return {'artifact_version': manifest['artifact_version'], 'git_sha': manifest['git_sha'],
        'training_source_sha256': hashlib.sha256(source).hexdigest(), 'horizons': result,
        'sources': {str(p.relative_to(repo)): digest(p) for p in
                    (root / 'manifest.json', root / 'metrics.json', root / 'forecasts.parquet')}}


def learning_correction(repo):
    root = repo / 'output/claude-audit/drive/experiments/research-v2-tf-placement'
    path = root / 'diagnosis.json'
    diagnosis = read_record(path)
    controls = diagnosis['controls']
    families = ('xgboost', 'catboost', 'mlp', 'lstm', 'tcn')
    record = root / 'ledger/completed' / (controls['tcn/known_signal']['record'] + '.json')
    trial = read_record(record)
    return {'experiment': root.name, 'old_status': diagnosis['status'], 'validation_policy': VALIDATION_POLICY,
        'corrected_status': 'failed' if control_failures(controls, families) else 'passed',
        'failures': control_failures(controls, families), 'tcn_control': controls['tcn/known_signal'],
        'trial_identity': trial['identity'], 'trial_metrics': trial['result']['metrics'],
        'classification': 'MISINTERPRETED', 'scope': 'synthetic learning control, not market evidence',
        'sources': {str(p.relative_to(repo)): digest(p) for p in (path, record)}}


def summarize(rows, h):
    a, c = rows[f'target_return_{h}'].to_numpy(), rows.cotton_close.to_numpy()
    naive = c * np.abs(np.expm1(a))
    summary = {'count': len(rows), 'start': str(rows.date.min()), 'end': str(rows.date.max()),
               'naive_mae': float(naive.mean())}
    for mode, field in (('selected', 'predicted_return'), ('raw', 'raw_predicted_return')):
        if field not in rows:
            summary[mode] = None
            continue
        p = rows[field].to_numpy()
        error = c * np.abs(np.exp(a) - np.exp(p))
        summary[mode] = {'mae': float(error.mean()), 'gain_pct': float(100 * (1 - error.mean() / naive.mean())),
            'all_origin_direction_pct': float(100 * np.mean(np.sign(a) == np.sign(p))),
            'active_rate_pct': float(100 * np.mean(p != 0))}
    return summary


def inventory(repo):
    """Each entry binds the actually available payload; missing outputs remain inconclusive."""
    roots = list((repo / 'output/full-year/experiments').glob('*/ready.json'))
    roots += [repo / p for p in (
        'output/claude-review-20261004/current-experiment/ready.json',
        'output/reports/wasde-text-colab-review-20261004/restored/ready.json',
        'output/reports/recency-colab-review-20261003/evidence/ready.json')]
    entries = []
    for ready_path in sorted(set(roots)):
        ready = read_record(ready_path)
        identity, root = ready['identity'], ready_path.parent
        if identity.get('profile') == 'availability-clock-pilot-v1':
            continue  # This file corrects historical evidence, not the new experiment.
        base = {'experiment': root.name, 'profile': identity.get('profile'),
            'code_id': identity.get('source_id'), 'data_id': ready.get('history_sha256'),
            'ready': str(ready_path.relative_to(repo)), 'ready_sha256': digest(ready_path),
            'independent_holdout': False}
        outputs = sorted(root.glob('*outputs/*.json')) + sorted(root.glob('pilot-full-year/*.json'))
        groups = {}
        for path in outputs:
            item = read_record(path)
            if not item.get('records') or 'predicted_return' not in item['records'][0]:
                continue
            h = item.get('horizon', item.get('recipe', {}).get('horizon', 5))
            key = (item.get('group', 'selected'), h)
            groups.setdefault(key, []).append((path, item))
        if not groups:
            entries.append({**base, 'classification': 'INCONCLUSIVE', 'reason': 'No completed prediction payload at this location',
                'prediction_payload_exists': False, 'negative_performance_evidence': False})
        for (group, h), items in groups.items():
            rows = pd.concat([pd.DataFrame(v['records']).assign(year=v['year']) for _, v in items], ignore_index=True)
            if rows.date.duplicated().any():
                raise ValueError(f'Duplicate saved origins in {root}/{group}/T{h}')
            recipes = [v.get('recipe', {'family': v.get('price_family', 'unavailable')}) for _, v in items]
            available_years = sorted(rows.year.unique().tolist())
            expected_years = [f['year'] for f in identity.get('split', {}).get('folds', [])]
            complete = available_years == expected_years and bool(expected_years)
            entries.append({**base, 'group': group, 'horizon': h, 'metrics': summarize(rows, h),
                'years': available_years, 'expected_years': expected_years,
                'model_families': sorted({r['family'] for r in recipes}),
                'recipes': list({json.dumps(r, sort_keys=True): r for r in recipes}.values()),
                'selection': 'See checksum-bound recipe, decisions and ready; no selection inferred from report prose',
                'prediction_payload_exists': True, 'output_years_complete': complete,
                'classification': 'INCONCLUSIVE',
                'reason': ('Reviewed historical selected recipe; not source-wide no-signal proof' if complete
                           else 'Only partial annual payload; not a completed negative experiment'),
                'payloads': {str(p.relative_to(repo)): digest(p) for p, _ in items}})
    # Additional restored reviews have prediction-level CSVs but not a full local engine folder.
    for name, filename in [('nonlinear-path', 'saved-predictions.csv'), ('agri-nonlinear', 'saved-predictions.csv'),
                           ('statistical', 'saved-predictions.csv')]:
        path = repo / f'output/reports/{name}-colab-review-20261003' / filename
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        for (group, h), part in rows.groupby(['group', 'horizon']):
            entries.append({'experiment': name, 'group': group, 'horizon': int(h), 'metrics': summarize(part, h),
                'prediction_payload_exists': True, 'classification': 'INCONCLUSIVE',
                'code_id': None, 'data_id': None, 'selection': 'Unverified from CSV alone; consult restored manifest',
                'reason': 'Prediction-level review only; code identity not re-established by this CSV',
                'independent_holdout': False, 'payloads': {str(path.relative_to(repo)): digest(path)}})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # freeze_record also prevents changes; explicit derived-only output location.
    if (args.output.exists() and any(args.output.iterdir())
            and not (args.output / 'legacy-lstm-correction.json').exists()):
        raise ValueError('Use an empty derived-report directory')
    freeze_record(args.output / 'legacy-lstm-correction.json', legacy_correction(args.repo))
    freeze_record(args.output / 'learning-control-correction.json', learning_correction(args.repo))
    freeze_record(args.output / 'experiment-inventory.json', {'entries': inventory(args.repo),
        'scope': 'Local payload inventory; completeness and inference limits are explicit',
        'rules': ['Synthetic learning is not market evidence', 'Incomplete is not negative',
            'T+5 sources do not answer T+1', 'Raw direction and selected active rate are distinct',
            'Degenerate zero-weight CI does not establish no source signal',
            '2016-2023 and previously viewed 2024+ are development/reviewed history']})
    print('Derived corrections and inventory saved; source evidence was read only:', args.output)


if __name__ == '__main__':
    main()
