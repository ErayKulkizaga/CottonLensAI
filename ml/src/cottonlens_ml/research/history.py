"""Read-only, standard-library lookup. A completed fit is not a completed experiment."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def load_registry(root):
    registry = json.loads((root / 'registry.json').read_text(encoding='utf-8'))
    trials = json.loads((root / 'trials.json').read_text(encoding='utf-8'))
    if registry.get('schema') != 1 or trials.get('schema') != 1:
        raise ValueError('Unsupported research registry schema')
    return registry, trials


def matches(record, *, query='', family=None, horizon=None, feature=None, profile=None):
    if horizon is not None and record.get('horizon') != horizon:
        return False
    if profile and record.get('profile') != profile:
        return False
    recipes = record.get('recipes', [])
    if family and not (family in record.get('model_families', []) or
                       record.get('model', '').lower() == family or
                       any(r.get('family') == family for r in recipes)):
        return False
    if feature and not any(any(feature.lower() in f.lower() for f in r.get('features', []))
                           for r in recipes):
        return False
    return all(word in json.dumps(record, sort_keys=True).lower() for word in query.lower().split())


def check(registry, trials, *, proposal=None, **filters):
    if proposal is not None:
        if not isinstance(proposal, dict) or not isinstance(proposal.get('recipe'), dict):
            raise ValueError('Proposal requires a recipe object')
        recipe = proposal['recipe']
        filters['family'] = recipe.get('family')
        filters['horizon'] = recipe.get('horizon')
    records = [r for r in registry['records'] if matches(r, **filters)]
    fits = [r for r in trials['records'] if matches(r, **filters)]
    # Full frozen scope + full recipe required. Partial metadata never implies equivalence.
    exact = [r for r in fits if proposal is not None and proposal.get('scope_id') == r['scope_id']
             and proposal['recipe'] == r['recipes'][0]]
    status = 'REPEAT_FIT_RECIPE' if exact else ('RELATED_EVIDENCE' if records or fits else 'NO_REGISTERED_MATCH')
    return {'status': status, 'exact_fit_recipes': exact, 'experiments': records, 'fit_recipes': fits,
            'limits': 'No match is not proof of novelty. Exact fits are not OOS success. Partial, synthetic '
                      'and compromised evidence cannot establish a market negative. Inspect frozen scope '
                      'and prediction coverage before repeating; deliberate repeats need a recorded reason.'}


def validate(root):
    registry, trials = load_registry(root)
    ids = set()
    for document in (registry, trials):
        for record in document['records']:
            rid = record.get('id')
            if not rid or rid in ids:
                raise ValueError('Missing or duplicate registry id')
            ids.add(rid)
            if document is trials:
                if record['scope_id'] != fingerprint(record['scope']):
                    raise ValueError('Frozen scope fingerprint mismatch')
                if record['id'] != fingerprint({'scope_id': record['scope_id'],
                                                'recipe': record['recipes'][0], 'kind': record['kind']}):
                    raise ValueError('Fit recipe fingerprint mismatch')
                if record['completed_fits'] < 1:
                    raise ValueError('Empty completed-fit record')
    for item in registry.get('local_evidence', []):
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Unsafe evidence path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Evidence checksum mismatch: {item["path"]}')
    return {'status': 'valid', 'experiments': len(registry['records']),
            'fit_recipes': len(trials['records']),
            'completed_fit_copies': sum(r['completed_fits'] for r in trials['records'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry-root', type=Path, default=ROOT / 'research')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('validate')
    commands.add_parser('list')
    lookup = commands.add_parser('check')
    lookup.add_argument('--query', default='')
    lookup.add_argument('--family')
    lookup.add_argument('--horizon', type=int, choices=(1, 5))
    lookup.add_argument('--feature', help='Feature-name substring, e.g. nass or return_path')
    lookup.add_argument('--profile')
    lookup.add_argument('--proposal', type=Path, help='Full recipe + scope_id copied from trials.json')
    lookup.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            print(json.dumps(validate(args.registry_root), indent=2))
            return 0
        registry, trials = load_registry(args.registry_root)
        if args.command == 'list':
            result = {'experiments': registry['records'], 'fit_recipes': trials['records']}
        else:
            proposal = json.loads(args.proposal.read_text(encoding='utf-8')) if args.proposal else None
            result = check(registry, trials, proposal=proposal, query=args.query, family=args.family,
                           horizon=args.horizon, feature=args.feature, profile=args.profile)
        if args.command == 'list' or args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result['status'])
            for r in result['experiments']:
                metric = r.get('metrics', {}).get('selected', {}) or {}
                print(f"{r['id'][:12]} | {r.get('profile') or r.get('experiment')} | "
                      f"{r.get('group', '')} | T+{r.get('horizon', '?')} | "
                      f"{r.get('coverage', 'unknown')} | {r.get('classification')} | "
                      f"MAE gain={metric.get('gain_pct', 'unknown')}")
            print(f"Related completed-fit recipes: {len(result['fit_recipes'])}; "
                  f"exact frozen matches: {len(result['exact_fit_recipes'])}")
            print(result['limits'])
        return 3 if result.get('status') == 'REPEAT_FIT_RECIPE' else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Research history invalid/unavailable: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
