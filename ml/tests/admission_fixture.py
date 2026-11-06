"""Explicit admission for generated unit-test fixtures, never production data.

Protocol tests use model stubs or generated labels. This adapter exercises the
real graph validator while authorizing each generated job on demand. It is not
installed by default and is not shipped in the training package.
"""
import hashlib
import json

from cottonlens_ml.research.admission import ExecutionAdmission, identity, inspect


def synthetic_admitter(root, ledger_identity, *, model_inputs=None, legacy_fit_ids=()):
    root.mkdir(parents=True, exist_ok=True)

    class FixtureAdmission(ExecutionAdmission):
        def __init__(self):
            pass

        def __call__(self, actual_identity, specification):
            name = identity(specification) + '.json'
            witness = root/name
            witness.write_text(json.dumps({'synthetic_unit_test': specification}, sort_keys=True))
            cell = {'id': 'fixture', 'provider': 'synthetic-test', 'symbol': 'fixture',
                    'contract': 'fixture', 'unit': 'fixture', 'field': 'fixture',
                    'session': '2000-01-01', 'vintage': 'generated', 'status': 'VERIFIED',
                    'evidence': [name], 'parents': [], 'available_at': '2000-01-01T00:00:00Z',
                    'mature_at': '2000-01-01T00:00:00Z', 'target_policy': 'generated'}
            manifest = {'schema': 1, 'purpose': 'synthetic', 'bindings': dict.fromkeys(
                ('source', 'parser', 'calendar', 'features', 'target', 'split', 'recipe', 'code'), 'synthetic'),
                'files': {name: hashlib.sha256(witness.read_bytes()).hexdigest()},
                'ledger_identity': ledger_identity, 'target_policy': 'generated', 'cells': [cell],
                'jobs': [{'id': name, 'specification_id': identity(specification),
                          'legacy_fit_ids': list(legacy_fit_ids),
                          'model_inputs_id': model_inputs or specification.get('model_inputs_id', 'generated_model_stub'),
                          'fit_at': '2000-01-02T00:00:00Z', 'decision_at': '2000-01-02T01:00:00Z',
                          'inputs': {r: ['fixture'] for r in
                                     ('feature', 'training_feature', 'training_label', 'evaluation_target')}}]}
            ExecutionAdmission.__init__(self, manifest, inspect(manifest, root), root)
            return super().__call__(actual_identity, specification)

    return FixtureAdmission()
