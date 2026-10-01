"""Schema-3 portable releases. Every transform is explicit; no training package in runtime."""
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.preflight import load_frozen_data
from cottonlens_ml.research import VERSION
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import feature_history, rows_at
from cottonlens_ml.runtime_guard import require_colab_training
from cottonlens_ml.walkforward import summarize


def convert_member(source, destination, adapter):
    family = adapter['spec']['family']
    if family == 'xgboost':
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(source)
        model.set_param({'device': 'cpu'})
        target = destination.with_suffix('.json')
        model.save_model(target)
        return target, 'xgboost_json'
    if family in ('ridge', 'elasticnet'):
        target = destination.with_suffix('.json')
        shutil.copyfile(source, target)
        return target, 'linear_json'
    target = destination.with_suffix('.onnx')
    if family == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(source))
        model.save_model(str(target), format='onnx')
    else:
        import tensorflow as tf
        fitted = tf.keras.models.load_model(source, compile=False)
        def clone(layer):
            config = layer.get_config()
            if isinstance(layer, tf.keras.layers.LSTM):
                config['use_cudnn'] = False
            return layer.__class__.from_config(config)
        with tf.device('/CPU:0'):
            model = tf.keras.models.clone_model(fitted, clone_function=clone)
            model.set_weights(fitted.get_weights())
            shape = (1, *model.input_shape[1:])
            model.export(str(target), format='onnx', verbose=False,
                         input_signature=[tf.TensorSpec(shape, tf.float32, name='features')])
    return target, 'onnx'


def component_seeds(chosen, weight=1.):
    if chosen['recipe']['family'] == 'ensemble':
        return [pair for member, value in zip(chosen['members'], chosen['weights'], strict=True)
                for pair in component_seeds(member['decision'], weight * value)]
    return [(seed, weight / len(chosen['seeds'])) for seed in chosen['seeds']]


def export(experiment, repo):
    require_colab_training()
    locked = read_record(experiment.root / 'locked.json')
    if read_record(experiment.root / 'diagnosis.json')['status'] != 'passed':
        raise ValueError('Diagnosis must pass before export')
    receipts = {str(h): read_record(experiment.root / f'reproduction-t{h}.json') for h in (1, 5)}
    if any(r['status'] != 'passed' or r['fresh_fits'] is not True for r in receipts.values()):
        raise ValueError('Fresh reproduction must pass before export')
    pending = experiment.root / 'release.json'
    if pending.exists():
        prior = read_record(pending)
        if digest(Path(prior['path'])) != prior['sha256']:
            raise ValueError('Frozen release changed')
        print('Validated release:', prior['path'], flush=True)
        return
    market, full = load_frozen_data(experiment.identity['source_data'])
    full = feature_history(full, market)
    if experiment.identity.get('profile') == 'free-data-v1':
        from cottonlens_ml.research.ablation import groups
        from cottonlens_ml.research.protocol import feature_groups
        groups(full, feature_groups(full))  # same causal age channels as prepare
    for package in experiment.identity.get('publication_sources', []):
        from cottonlens_ml.research.publications import attach_package, load_package
        manifest, records, names = load_package(Path(package['path']))
        if manifest != package['manifest']:
            raise ValueError('Publication package changed after research lock')
        full = attach_package(full, manifest, records, names)
    original_history = experiment.history
    now = datetime.now(UTC).isoformat()
    version = 'research-' + datetime.now(UTC).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    sys.path.insert(0, str(repo / 'backend'))
    from app.artifacts import verify_directory
    from app.research_runtime import ResearchModel
    metrics, forecasts, explanations, production, evaluation, parity = [], [], [], [], [], []
    fold_reports = {f['fold']: {'fold': f['fold'], 'test_start': f['origins'][0],
                    'test_end': f['origins'][-1], 'sample_count': len(f['origins']), 'metrics': {}}
                    for f in experiment.identity['split']['folds']}
    all_names = list(dict.fromkeys([*(name for names in experiment.identity['groups'].values() for name in names),
                                    'cotton_close']))
    schema_hash = content_id({'features': all_names, 'missing': 'train_fitted_adapter', 'version': 3})
    with tempfile.TemporaryDirectory(prefix='cotton-research-export-') as temporary:
        root = Path(temporary)
        (root / 'model').mkdir()
        for h in (1, 5):
            candidate = locked['horizons'][str(h)]
            namespace, family = candidate['namespace'], candidate['family']
            selected = candidate['primary_forecast']
            last_fold = len(experiment.identity['split']['folds'])
            decision = read_record(experiment.root / 'decisions' / namespace / f'{family}-t{h}-fold{last_fold}.json')['chosen']
            entry = {'horizon': h, 'name': selected, 'format': 'research_adapter_v1', 'path': f'model/entry-t{h}.json',
                     'members': [], 'model_role': 'deployment_live', 'fit_cutoff': full.date.max().isoformat(),
                     'recipe_identity': content_id(decision), 'model_identity': content_id({'chosen': decision, 'role': 'live', 'date': str(full.date.max())})}
            last_labels, expected_members = [], []
            experiment.history = full
            for i, (seed, weight) in enumerate(component_seeds(decision)):
                train = experiment.train_rows(full.date.max(), seed['recipe'])
                test = full.tail(1)
                record = experiment.fit(seed['recipe'], train, None, test, 'deployment_live', iterations=seed['iterations'])
                source_root = experiment.ledger.root / record['payload_root']
                adapter = json.loads((source_root / 'adapter.json').read_text())
                target, kind = convert_member(source_root / adapter['model_file'], root / 'model' / f't{h}-{i}', adapter)
                adapter_path = root / 'model' / f't{h}-{i}-adapter.json'
                adapter_path.write_text(json.dumps(adapter, indent=2), encoding='utf-8')
                member = {'path': target.relative_to(root).as_posix(), 'format': kind,
                          'adapter': adapter_path.relative_to(root).as_posix(), 'weight': weight}
                entry['members'].append(member)
                expected_members.append(weight * record['result']['predictions'][0])
                last_labels.append(train.target_date_5.max())
            experiment.history = original_history
            entry['fit_label_cutoff'] = max(last_labels).isoformat()
            research_model = ResearchModel(root, entry)
            context = full[all_names].tail(research_model.window).to_dict('records')
            expected = float(sum(expected_members))
            actual = research_model.predict(context)
            difference = abs(expected - actual)
            if difference > 1e-6:
                raise ValueError(f'Portable runtime parity failed T+{h}: {difference}')
            # Keep experimental model payloads, but primary inference remains Naive when gates failed.
            if selected == 'Naive':
                entry['experimental_members'] = entry['members']
                entry['members'] = []
                predicted = 0.
            else:
                predicted = actual
            parity.append({'horizon': h, 'expected_learned_return': expected, 'actual_learned_return': actual,
                           'max_abs_difference': difference, 'primary_return': predicted})
            (root / entry['path']).write_text(json.dumps(entry, indent=2), encoding='utf-8')
            production.append(entry)
            evaluation.append({'horizon': h, 'name': selected, 'model_role': 'evaluation_backtest',
                'model_identity': content_id({'candidate': candidate, 'role': 'historical_nested'}),
                'recipe_identity': content_id(decision), 'fit_cutoff': experiment.identity['split']['folds'][-1]['origins'][0],
                'fit_label_cutoff': str(experiment.train_rows(pd.Timestamp(experiment.identity['split']['folds'][-1]['origins'][0]), decision['recipe'] if family != 'ensemble' else component_seeds(decision)[0][0]['recipe']).target_date_5.max())})
            pairs = []
            for fold in experiment.identity['split']['folds']:
                data = rows_at(original_history, fold['origins'])
                result = read_record(experiment.root / 'outer' / namespace / f'{family}-t{h}-fold{fold["fold"]}.json')
                from cottonlens_ml.evaluation import evaluate
                fold_reports[fold['fold']]['metrics'][f'{family}-T+{h}'] = result['metrics']
                fold_reports[fold['fold']]['metrics'][f'Naive-T+{h}'] = evaluate(
                    data.cotton_close.to_numpy(), data[f'target_return_{h}'].to_numpy(), np.zeros(len(data)))
                from types import SimpleNamespace
                pairs.append((data, SimpleNamespace(name='Naive', horizon=h, predictions=np.zeros(len(data)), parameters={})))
                for row, value in zip(data.itertuples(), result['predictions'], strict=True):
                    log_return = 0. if selected == 'Naive' else value
                    forecasts.append(forecast_row(version, h, row, log_return, 'backtest'))
            naive = summarize(pairs)
            for name, evidence in (('Naive', naive), (family, candidate['metrics'])):
                metrics.append({'model': name, 'horizon': h, 'selected': selected == name,
                    **{k: evidence[k] for k in ('mae', 'rmse', 'mape', 'directional_accuracy')},
                    'walkforward': evidence, 'parameters': {'role': 'seen_historical_research', 'namespace': namespace}})
            live = forecast_row(version, h, next(full.tail(1).itertuples()), predicted, 'live')
            forecasts.append(live)
            if selected == 'Naive':
                explanations.append({'forecast_id': live['id'], 'base_value': 0., 'explainer': 'persistence_no_learned_effect',
                                     'contributions': '[]'})
            else:
                # Exact telescoping last-observation feature replacement, clearly not SHAP/causality.
                base = [dict(row) for row in context]
                for name in all_names:
                    if name != 'cotton_close':
                        values = [a['processor']['median'][a['processor']['names'].index(name)] for _, a, _ in research_model.members
                                  if name in a['processor']['names']]
                        if values:
                            base[-1][name] = float(np.mean(values))
                initial = current = research_model.predict(base)
                contributions = []
                for name in all_names:
                    if name == 'cotton_close':
                        continue
                    value = context[-1].get(name)
                    base[-1][name] = value
                    updated = research_model.predict(base)
                    contributions.append({'feature': name, 'display_name': name.replace('_', ' '),
                                          'feature_value': float(value) if value is not None and np.isfinite(value) else 0.,
                                          'contribution_pct': (updated - current) * 100})
                    current = updated
                explanations.append({'forecast_id': live['id'], 'base_value': initial * 100,
                    'explainer': 'ordered_last_observation_replacement_not_causal_not_shap',
                    'contributions': json.dumps(contributions)})
        evidence = {'code_identity': {'source_id': experiment.identity['source_id']},
                    'protocol_identity': {'version': VERSION, 'locked': locked},
                    'data_identity': experiment.identity['source_data'], 'cohort_identity': experiment.identity['split'],
                    'data_quality': read_record(experiment.root / 'quality.json'),
                    'environment_smoke': read_record(experiment.root / 'diagnosis.json')}
        # Backend schema contract uses the established passed status for smoke evidence.
        manifest = {'artifact_schema_version': 3, 'artifact_version': version, 'generated_at': now,
                    'research_protocol': VERSION, 'data_quality': 'historical_audit',
                    'feature_schema_hash': schema_hash, 'production_models': production,
                    'evaluation_models': evaluation, 'reproduction': receipts, 'evidence_path': 'evidence.json',
                    **{k: evidence[k] for k in ('code_identity', 'protocol_identity', 'data_identity', 'cohort_identity')},
                    'walkforward_report': {'protocol': VERSION, 'fold_count': len(experiment.identity['split']['folds']), 'aggregate': {
                        f'{m["model"]}-T+{m["horizon"]}': m['walkforward'] for m in metrics},
                        'folds': list(fold_reports.values()), 'feature_ablation': [],
                        'cftc_candidate': 'Only verified as-published packages; see frozen research identity'},
                    'selection_audit': locked['horizons'], 'prospective_verified': False}
        for name, value in [('manifest.json', manifest), ('feature_schema.json', {'features': all_names, 'schema_hash': schema_hash}),
                            ('metrics.json', metrics), ('evidence.json', evidence), ('parity.json', parity)]:
            (root / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        market[['date', 'series', 'close']].rename(columns={'date': 'observed_on'}).to_parquet(root / 'market_history.parquet', index=False)
        full[['date', *all_names]].tail(500).to_parquet(root / 'feature_snapshots.parquet', index=False)
        pd.DataFrame(forecasts).to_parquet(root / 'forecasts.parquet', index=False)
        pd.DataFrame(explanations).to_parquet(root / 'explanations.parquet', index=False)
        (root / 'checksums.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n'
             for p in sorted(root.rglob('*')) if p.is_file()), encoding='utf-8')
        verify_directory(root)
        inference = Path('/content/cottonlens-inference/bin/python')
        if not inference.is_file():
            raise ValueError('Separate inference-only environment required')
        subprocess.run([str(inference), str(repo / 'ml/research_probe.py'), '--repo', str(repo), '--artifact', str(root)], check=True)
        output = experiment.root / 'artifacts/releases' / f'cottonlens-model-{version}.zip'
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
            for payload in sorted(root.rglob('*')):
                if payload.is_file():
                    archive.write(payload, payload.relative_to(root).as_posix())
        output.with_suffix('.zip.sha256').write_text(digest(output) + '  ' + output.name + '\n', encoding='utf-8')
        freeze_record(pending, {'path': str(output), 'sha256': digest(output), 'runtime_parity': 'passed',
                               'reproduction': receipts, 'prospective_verified': False})
        print('Validated release:', output, flush=True)


def forecast_row(version, horizon, row, predicted, role):
    return {'id': content_id({'version': version, 'horizon': horizon, 'date': str(row.date), 'role': role})[:32],
            'as_of_date': row.date, 'target_date': None if role == 'live' else getattr(row, f'target_date_{horizon}'),
            'horizon': horizon, 'current_price': float(row.cotton_close),
            'predicted_price': float(row.cotton_close * np.exp(predicted)), 'predicted_return_pct': float(predicted * 100),
            'actual_price': None if role == 'live' else float(row.cotton_close * np.exp(getattr(row, f'target_return_{horizon}'))),
            'origin_type': role}
