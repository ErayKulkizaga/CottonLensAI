"""Index existing completed ledger receipts, never execute or overwrite an experiment."""
import argparse
import json
import os
from pathlib import Path

from cottonlens_ml.research.history import fingerprint

SKIP = {'.git', '.venv', 'node_modules', 'site-packages', '__pycache__', 'payloads',
        'tracking', 'mlruns', 'tools'}


def index(root):
    groups, errors, seen, controls = {}, [], set(), {}
    for folder, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.endswith('-deps')]
        directory = Path(folder)
        if directory.name != 'completed' or directory.parent.name != 'ledger':
            continue
        for name in sorted(names):
            if not name.endswith('.json'):
                continue
            path = directory / name
            try:
                value = json.loads(path.read_text(encoding='utf-8'))
                body = {k: v for k, v in value.items() if k != 'record_id'}
                if value.get('record_id') != fingerprint(body):
                    raise ValueError('Invalid completed receipt checksum')
                identity, specification = value['identity'], value['specification']
                if value['experiment_id'] != fingerprint({'identity': identity, 'specification': specification}):
                    raise ValueError('Invalid frozen experiment identity')
                role = specification.get('role', '')
                recipe = specification.get('recipe')
                if recipe is None and role == 'synthetic_control_not_market_evidence':
                    recipe = specification.get('spec')
                if recipe is None and role == 'gpu_objective_smoke':
                    controls[value['record_id']] = {'id': value['record_id'],
                        'model_families': [specification['family']], 'task': specification['task'],
                        'kind': 'gpu_objective_smoke_not_market_evidence',
                        'receipt': path.relative_to(root).as_posix()}
                    continue
                if not isinstance(recipe, dict) or not recipe.get('family'):
                    raise ValueError('Missing executed recipe')
                scope = {'identity_id': fingerprint(identity), 'source_id': identity.get('source_id'),
                         'data_id': identity.get('research_data_id') or identity.get('source_data', {}).get('data_id'),
                         'split_id': fingerprint(identity.get('split', {})),
                         'profile': identity.get('profile', identity.get('version'))}
                scope_id = fingerprint(scope)
                kind = ('synthetic_control:' + specification.get('kind', 'unspecified')
                        if any(w in role.lower() for w in ('synthetic', 'diagnostic', 'known-signal', 'overfit', 'negative-control'))
                        else 'market_fit_receipt')
                rid = fingerprint({'scope_id': scope_id, 'recipe': recipe, 'kind': kind})
                # Restored duplicate copies must not inflate evidence counts.
                fit_id = (scope_id, value['experiment_id'], value['record_id'])
                if fit_id in seen:
                    continue
                seen.add(fit_id)
                record = groups.setdefault(rid, {'id': rid, 'scope': scope, 'scope_id': scope_id,
                    'profile': scope['profile'], 'horizon': recipe.get('horizon'),
                    'recipes': [recipe], 'model_families': [recipe.get('family')], 'kind': kind,
                    'completed_fits': 0, 'sample_receipts': [], 'roles': [],
                    'coverage': 'completed fit receipts; not experiment completion',
                    'payload_hashes_revalidated_by_index': False})
                record['completed_fits'] += 1
                if len(record['sample_receipts']) < 3:
                    record['sample_receipts'].append({'path': path.relative_to(root).as_posix(),
                        'record_id': value['record_id'], 'experiment_id': value['experiment_id']})
                # Stage category only; every origin-specific role stays in the archived receipt.
                stage = role.split('-')[0]
                if stage not in record['roles']:
                    record['roles'].append(stage)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({'path': path.relative_to(root).as_posix(), 'error': str(exc)})
    return {'schema': 1, 'scope': 'Available checksum-verified ledger receipts; no unrun recipes inferred',
            'records': sorted(groups.values(), key=lambda r: r['id']), 'rejected_receipts': errors,
            'non_fit_controls': sorted(controls.values(), key=lambda r: r['id'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = index(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'fit_recipes': len(result['records']), 'rejected': len(result['rejected_receipts']),
                      'unique_completed_fits': sum(r['completed_fits'] for r in result['records'])}))
    return 2 if result['rejected_receipts'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
