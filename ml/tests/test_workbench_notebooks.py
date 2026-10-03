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
    assert scope['WORKFLOWS']['pilot'] == [('pilot', 'A')]
    assert 'feature_comparison' not in scope['WORKFLOWS']
    source = '\n'.join(cells)
    assert 'EXPECTED_ZIP_SHA256' in source
    assert "REFERENCE_EXPERIMENT = 'research-market-v1'" in source
    assert "EXPERIMENT = 'research-full-year-v1-r2'" in source
    assert 'setup_cpu()' in source
    assert "'--mirror-root',DRIVE_ROOT" in source
    assert 'MAX_MINUTES' in source and "'--max-minutes'" in source
    assert 'assert RUN_TRAINING' in source
    assert "'--profile','full-year-v1'" in source
    assert "'--reference-root',reference" in source
    assert 'experiment.inner(' not in source
    assert 'experiment.tune(' not in source
    assert "'--stage',stage" in source


def test_data_inventory_has_no_training_and_no_secret_values():
    cells = code('cottonlens_data_workbench.ipynb')
    scope = {}
    exec(compile(cells[0], '<data-config>', 'exec'), scope)  # noqa: S102 - trusted local config cell
    assert scope['SOURCE_ACTION'] == 'readiness'
    source = '\n'.join(cells)
    assert 'cottonlens_ml.research.engine' not in source
    assert 'userdata.get(name)' in source
    assert "provider == 'ams' and year < 2025" in source


def test_resume_selects_fit_source_and_reporting_selects_current_source():
    configuration = code('cottonlens_research_workbench.ipynb')[0]
    for profile, fit_bundle in [('full-year-v1','source-full-year-v1-r2-20261001.zip'),
                                ('ams-exploration-v1','source-ams-exploration-v1-20261002.zip'),
                                ('fas-exploration-v1','source-fas-exploration-v1-20261002.zip'),
                                ('nass-exploration-v1','source-nass-exploration-v1-20261002.zip'),
                                ('wasde-exploration-v1','source-wasde-exploration-v1-20261002.zip'),
                                ('cftc-exploration-v1','source-cftc-exploration-v1-20261002.zip'),
                                ('fx-exploration-v1','source-fx-exploration-v1-20261002.zip'),
                                ('crop-exploration-v1','source-crop-exploration-v1-20261002.zip'),
                                ('weather-exploration-v1','source-weather-exploration-v1-20261002.zip'),
                                ('oncall-exploration-v1','source-oncall-exploration-v1-20261002.zip'),
                                ('oncall-exploration-v2','source-oncall-exploration-v3-20261003.zip'),
                                ('recency-pilot-v1','source-recency-pilot-v1-20261003.zip'),
                                ('return-path-pilot-v1','source-return-path-pilot-v1-20261003.zip'),
                                ('agri-transfer-pilot-v1','source-agri-transfer-pilot-v1-20261003.zip')]:
        for workflow in ('status','prepare','pilot_plan','pilot','compare'):
            source = configuration.replace("WORKFLOW = 'status'", f'WORKFLOW = {workflow!r}', 1)
            source = source.replace("PROFILE = 'full-year-v1'", f'PROFILE = {profile!r}', 1)
            scope = {}
            exec(compile(source, '<workflow>', 'exec'), scope)  # noqa: S102 - trusted local config
            expected = fit_bundle if workflow in ('prepare','pilot_plan','pilot') else 'source-agri-transfer-pilot-v1-20261003.zip'
            assert scope['SOURCE_BUNDLE'] == expected
            assert scope['RUN_TRAINING'] is False
    source = '\n'.join(code('cottonlens_research_workbench.ipynb'))
    assert "'information-outputs' if PROFILE in ('ams-exploration-v1','fas-exploration-v1','nass-exploration-v1','wasde-exploration-v1','cftc-exploration-v1','fx-exploration-v1','crop-exploration-v1','weather-exploration-v1','oncall-exploration-v1','oncall-exploration-v2')" in source
    assert "else 56 if PROFILE in ('fas-exploration-v1','nass-exploration-v1','cftc-exploration-v1','fx-exploration-v1','crop-exploration-v1','weather-exploration-v1','oncall-exploration-v1','oncall-exploration-v2')" in source
    assert "BASE += ['--fas-table',local/'weekly-sales-2010-2023.csv']" in source
    assert "BASE += ['--nass-table',local/'upland-condition-2010-2023.csv','--nass-audit-root',local]" in source
    assert "BASE += ['--wasde-table',local/'world-balance-2016-2023.csv']" in source
    assert "BASE += ['--cftc-table',local/'positions-2010-2023.csv']" in source
    assert "BASE += ['--weather-table',local/'weather-2010-2023.csv']" in source
    assert "BASE += ['--oncall-table',local/'oncall-2010-2023.csv']" in source
    assert "BASE += ['--crop-table',local/'crop-prices-2010-2023.csv']" in source
    assert "BASE += ['--fx-table',local/'currency-crosses-2010-2023.csv']" in source
    assert "else 35 if PROFILE=='wasde-exploration-v1'" in source


def test_oncall_data_action_is_pinned_read_only_and_cpu_only():
    source = '\n'.join(code('cottonlens_data_workbench.ipynb'))
    assert "SOURCE_ACTION == 'oncall_status'" in source
    assert "if SOURCE_ACTION == 'oncall_status':" in source
    assert "colab_setup.UV_VERSION = '0.12.14'" in source
    assert 'PYTHON = setup_cpu()' in source
    assert 'oncall-exploration-v2-inputs-20261002' in source
    assert "receipt['files']" in source
    assert 'cottonlens_ml.research.engine' not in source


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


def test_recency_selects_new_source_and_preserves_read_only_defaults():
    cells = code('cottonlens_research_workbench.ipynb')
    configuration = cells[0].replace("PROFILE = 'full-year-v1'", "PROFILE = 'recency-pilot-v1'", 1)
    for workflow in ('status', 'prepare', 'pilot', 'compare'):
        scope = {}
        exec(compile(configuration.replace("WORKFLOW = 'status'", f'WORKFLOW = {workflow!r}', 1), '<recency-workflow>', 'exec'), scope)  # noqa: S102
        assert scope['EXPERIMENT'] == 'research-recency-pilot-v1'
        assert scope['SOURCE_BUNDLE'] == ('source-recency-pilot-v1-20261003.zip' if workflow in ('prepare','pilot') else 'source-agri-transfer-pilot-v1-20261003.zip')
        assert scope['RUN_TRAINING'] is False
    source = '\n'.join(cells)
    assert "'recency-outputs' if PROFILE == 'recency-pilot-v1'" in source
    assert 'recency-pilot-v1-inputs-20261003' in source
    assert "run([*BASE,'--stage','prepare','--reference-root',reference]" in source
    assert "run([*BASE,'--stage','compare']" in source


def test_return_path_uses_same_workbench_and_explicit_training_gate():
    cells = code('cottonlens_research_workbench.ipynb')
    configuration = cells[0].replace("PROFILE = 'full-year-v1'", "PROFILE = 'return-path-pilot-v1'", 1)
    scope = {}
    exec(compile(configuration, '<path-workflow>', 'exec'), scope)  # noqa: S102
    assert scope['EXPERIMENT'] == 'research-return-path-pilot-v1'
    assert scope['SOURCE_BUNDLE'] == 'source-agri-transfer-pilot-v1-20261003.zip'
    assert scope['RUN_TRAINING'] is False and scope['STAGES'] == [('status','A')]
    source = '\n'.join(cells)
    assert 'return-path-pilot-v1-inputs-20261003' in source
    assert "'path-outputs' if PROFILE == 'return-path-pilot-v1'" in source
    assert "total=32 if PROFILE in ('return-path-pilot-v1','agri-transfer-pilot-v1','agri-nonlinear-pilot-v1')" in source


def test_agricultural_transfer_uses_frozen_packet_and_existing_workbench():
    cells = code('cottonlens_research_workbench.ipynb')
    configuration = cells[0].replace("PROFILE = 'full-year-v1'", "PROFILE = 'agri-transfer-pilot-v1'",1)
    for workflow in ('status','prepare','pilot','compare'):
        scope = {}
        exec(compile(configuration.replace("WORKFLOW = 'status'",f'WORKFLOW = {workflow!r}',1),  # noqa: S102 - trusted local configuration
            '<transfer-workflow>','exec'),scope)
        assert scope['EXPERIMENT']=='research-agri-transfer-pilot-v1'
        assert scope['SOURCE_BUNDLE']=='source-agri-transfer-pilot-v1-20261003.zip'
        assert scope['RUN_TRAINING'] is False
    source = '\n'.join(cells)
    assert "'data/agri-transfer-pilot-v1-inputs-20261003'" in source
    assert "reference = local if PROFILE in ('agri-transfer-pilot-v1','agri-nonlinear-pilot-v1')" in source
    assert "'transfer-outputs' if PROFILE == 'agri-transfer-pilot-v1'" in source


def test_nonlinear_transfer_uses_gpu_only_for_prepare_and_training():
    cells = code('cottonlens_research_workbench.ipynb')
    configuration = cells[0].replace("PROFILE = 'full-year-v1'", "PROFILE = 'agri-nonlinear-pilot-v1'",1)
    for workflow in ('status','prepare','pilot_plan','pilot','compare'):
        scope = {}
        exec(compile(configuration.replace("WORKFLOW = 'status'",f'WORKFLOW = {workflow!r}',1),  # noqa: S102 - trusted local configuration
            '<nonlinear-workflow>','exec'),scope)
        assert scope['EXPERIMENT']=='research-agri-nonlinear-pilot-v1'
        assert scope['SOURCE_BUNDLE']=='source-agri-nonlinear-pilot-v1-20261003.zip'
        assert scope['RUN_TRAINING'] is False
    source = '\n'.join(cells)
    assert "GPU_FIT = PROFILE == 'agri-nonlinear-pilot-v1' and WORKFLOW in ('prepare','pilot')" in source
    assert 'setup(research=True) if GPU_FIT else setup_cpu()' in source
    assert "'data/agri-nonlinear-pilot-v1-inputs-20261003'" in source
    assert "'nonlinear-outputs' if PROFILE == 'agri-nonlinear-pilot-v1'" in source
