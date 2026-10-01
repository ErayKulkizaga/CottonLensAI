"""Offline orchestration contracts; no training or Drive calls."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).parents[1] / 'notebooks'


def code(name):
    notebook = json.loads((ROOT / name).read_text(encoding='utf-8'))
    cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for source in cells:
        ast.parse(source)
    return cells


def test_research_run_all_defaults_to_read_only_status():
    cells = code('cottonlens_research_workbench.ipynb')
    scope = {}
    exec(compile(cells[0], '<workflow>', 'exec'), scope)  # noqa: S102 - trusted local config cell
    assert scope['STAGES'] == [('status', 'A')]
    assert scope['RUN_TRAINING'] is False
    assert scope['MAX_MINUTES'] == 60
    assert all(stage not in ('lock', 'reproduce', 'export', 'search') for stage, _ in scope['STAGES'])
    assert scope['WORKFLOWS']['pilot'] == [('diagnose', 'A'), ('pilot', 'A')]
    assert 'feature_comparison' not in scope['WORKFLOWS']
    source = '\n'.join(cells)
    assert 'EXPECTED_ZIP_SHA256' in source
    assert "REFERENCE_EXPERIMENT = 'research-market-v1'" in source
    assert "EXPERIMENT = 'research-information-pilot-v1'" in source
    assert "reference_ready['identity']['publication_sources'] == []" in source
    assert 'MAX_MINUTES' in source and "'--max-minutes'" in source
    assert 'assert RUN_TRAINING' in source
    assert "'--stage', 'prepare', '--profile', 'free-data-v1'" in source
    assert "'b7aecc5c96a240069a3251e3e88b5f1c0c448fac5c8eb2f02782a16b4a0c03ca'" in source
    assert 'experiment.inner(' not in source
    assert 'experiment.tune(' not in source
    assert "'--stage', stage" in source


def test_data_inventory_has_no_training_and_no_secret_values():
    cells = code('cottonlens_data_workbench.ipynb')
    scope = {}
    exec(compile(cells[0], '<data-config>', 'exec'), scope)  # noqa: S102 - trusted local config cell
    assert scope['SOURCE_ACTION'] == 'readiness'
    source = '\n'.join(cells)
    assert 'cottonlens_ml.research.engine' not in source
    assert 'userdata.get(name)' in source
    assert "provider == 'ams' and year < 2025" in source


def test_ams_publication_workflow_uses_existing_frozen_table_without_training():
    cells = code('cottonlens_data_workbench.ipynb')
    source = '\n'.join(cells)
    assert "SOURCE_ACTION == 'ams_publication_review'" in source
    assert "'cottonlens_ml.sources.ams_publications'" in source
    assert '7cc587d313e8c39e5f8bb77f9e4359d9b6275951c54e5d0f26c59e555e832e94' in source
    assert 'cottonlens_ml.research.engine' not in source
    assert "'first_version_reviewed', 'model_eligible'" in source
    assert 'setup(research=True, data=True)' in source


def test_observation_workflow_fetches_public_current_sources_without_admission():
    cells = code('cottonlens_data_workbench.ipynb')
    source = '\n'.join(cells)
    assert "SOURCE_ACTION == 'observe_ams'" in source
    assert "'cottonlens_ml.sources.public', 'observe'" in source
    assert "'--browser-transport'" in source
    assert 'observed-public-sources-v1' in source
    assert "'--stage'" not in source
    assert 'ams_3804.pdf' in source


def test_pilot_verifier_resolves_payload_under_ledger(tmp_path):
    cells = code('cottonlens_free_research_pilot.ipynb')
    tree = ast.parse(cells[-1])
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'pilot_program' for t in n.targets))
    program = ast.parse(ast.literal_eval(assignment.value))
    call = next(n for n in ast.walk(program) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'digest')
    payload = tmp_path / 'ledger' / 'payloads' / 'model.json'
    payload.parent.mkdir(parents=True)
    payload.write_text('saved model')
    expression = ast.Expression(call.args[0])
    resolved = eval(compile(expression, '<payload-path>', 'eval'),
                    {'root': tmp_path, 'name': 'payloads/model.json'})
    assert resolved.read_text() == 'saved model'
