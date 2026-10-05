"""Rebuildable MLflow projection of immutable research results; never the authority."""
import json
import math
import shutil
import tempfile
import uuid
from pathlib import Path

from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.tracking import copy_database


def project_record(client, experiment_id, record):
    """Idempotent even if projection was interrupted after creating the MLflow run."""
    key = record['experiment_id']
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Checksum trial key required')
    found = client.search_runs([experiment_id], filter_string=f"tags.ledger_id = '{key}'", max_results=2)
    if len(found) > 1:
        raise ValueError('Duplicate MLflow ledger projection; inspect before proceeding')
    run = found[0] if found else client.create_run(experiment_id, tags={
        'ledger_id': key, 'evidence_role': 'seen_historical_research',
        'authority': 'checksummed_research_ledger', 'identity': content_id(record['identity'])})
    run_id = run.info.run_id
    if run.data.tags.get('projection_complete') == 'true':
        return run_id
    specification = record['specification']
    recipe = specification.get('recipe', specification.get('spec', {}))
    for name, value in {'role': specification.get('role'), 'family': recipe.get('family'),
                        'horizon': recipe.get('horizon'), 'seed': recipe.get('seed'),
                        'target': recipe.get('target'), 'recipe_id': content_id(recipe)}.items():
        if value is not None:
            client.log_param(run_id, name, str(value))
    metrics = {**record['result'].get('metrics', {}), 'compute_seconds': record.get('compute_seconds', 0),
               'checkpoint_copy_seconds': record.get('checkpoint_copy_seconds', 0)}
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            client.log_metric(run_id, name, float(value), step=0)
    client.set_tag(run_id, 'payload_root', record['payload_root'])
    client.set_terminated(run_id, status='FINISHED')
    client.set_tag(run_id, 'projection_complete', 'true')
    return run_id


def project(experiment):
    """Explicit offline-capable stage; do not add Drive I/O to every training step."""
    from mlflow.tracking import MlflowClient
    root = experiment.root / 'tracking-projection'
    root.mkdir(parents=True, exist_ok=True)
    latest = root / 'latest.json'
    with tempfile.TemporaryDirectory(prefix='cottonlens-research-mlflow-') as directory:
        local = Path(directory)
        database = local / 'mlflow.sqlite'
        if latest.exists():
            prior = read_record(latest)
            name = prior['snapshot']
            if Path(name).name != name or digest(root / name) != prior['sha256']:
                raise ValueError('Corrupt MLflow projection snapshot')
            copy_database(root / name, database)
        client = MlflowClient(tracking_uri='sqlite:///' + database.as_posix())
        name = 'cottonlens-' + content_id(experiment.identity)[:16]
        existing = client.get_experiment_by_name(name)
        experiment_id = existing.experiment_id if existing else client.create_experiment(
            name, artifact_location=(root / 'artifacts').resolve().as_uri())
        count = 0
        for path in sorted((experiment.ledger.root / 'completed').glob('*.json')):
            project_record(client, experiment_id, read_record(path))
            count += 1
            if count % 100 == 0:
                print(f'STAGE tracking projected {count} completed fits; no training', flush=True)
        snapshot = local / 'snapshot.sqlite'
        copy_database(database, snapshot)
        snapshot_name = 'mlflow-' + digest(snapshot) + '.sqlite'
        target = root / snapshot_name
        if not target.exists():
            pending = root / (uuid.uuid4().hex + '.pending')
            shutil.copyfile(snapshot, pending)
            if digest(pending) != digest(snapshot):
                raise ValueError('MLflow projection transfer checksum mismatch')
            pending.rename(target)
        elif digest(target) != digest(snapshot):
            raise ValueError('Existing MLflow snapshot corrupted')
        receipt = {'snapshot': snapshot_name, 'sha256': digest(target), 'projected_fits': count,
                   'experiment_name': name, 'authority': 'ledger', 'models_copied': False}
        pointer = root / (uuid.uuid4().hex + '.pending')
        freeze_record(pointer, receipt)
        pointer.replace(latest)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt
