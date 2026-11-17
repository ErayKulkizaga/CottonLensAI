"""Read-only evidence index; does not import training or change scientific records."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    registry = ROOT / 'research/registry.json'
    trials = ROOT / 'research/trials.json'
    records = json.loads(registry.read_text(encoding='utf-8'))['records']
    fits = json.loads(trials.read_text(encoding='utf-8'))['records']
    corrections = [r for r in records if r.get('profile') == 'ams-contract-lifecycle-correction-v1']
    affected = {p for r in corrections for p in r['affected_profiles']}
    out = ROOT / 'research/evidence/root-cause-inventory-20261106.csv'
    fields = ['record_id', 'profile', 'experiment', 'horizon', 'families',
              'original_classification', 'source_qualification', 'kind',
              'scope', 'evidence', 'release', 'metrics', 'explicit_fit_counts']
    with out.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for r in records:
            profile = r.get('profile', '')
            writer.writerow(dict(zip(fields, [r['id'], profile, r.get('experiment', ''),
                r.get('horizon', ''), '|'.join(r.get('model_families', [])), r['classification'],
                'SOURCE_COMPROMISED_BY_LIFECYCLE_CORRECTION' if profile in affected else 'NO_BLANKET_CLEAN_SOURCE_CLAIM',
                r.get('kind', ''), r.get('classification_scope', r.get('reason', '')),
                r.get('evidence', ''), r.get('payload_release', r.get('release', '')),
                json.dumps(r.get('metrics', {}), ensure_ascii=False, sort_keys=True),
                json.dumps({k: v for k, v in r.items() if 'fit' in k and isinstance(v, (int, float))}, sort_keys=True)])))
    receipt = {
        'schema': 1, 'date': '2026-11-06', 'new_market_fits': 0, 'new_synthetic_fits': 0,
        'registry_records': len(records), 'trial_records': len(fits),
        'profiles': len({r.get('profile') for r in records if r.get('profile')}),
        'trial_kinds': dict(Counter(r.get('kind', 'unspecified') for r in fits)),
        'affected_profiles': sorted(affected),
        'limits': ['Rows are evidence records, not independent hypotheses.',
                   'Fit copies and recipe rows must not be summed as unique scientific trials.',
                   'Absence from affected_profiles does not certify clean data.',
                   'Explicit fit counts may overlap reused records; missing counts are unknown.',
                   'Original classifications preserved; later source qualification is separate.',
                   'No old release archive or training was rerun for this inventory.'],
        'sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [registry, trials, out]},
    }
    destination = out.with_suffix('.json')
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
