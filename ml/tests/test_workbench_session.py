"""Exercise notebook session control with fake stages; no model or Drive access."""
import ast
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def functions():
    notebook = Path(__file__).parents[1] / 'notebooks' / 'cottonlens_research_workbench.ipynb'
    cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
    source = ''.join(cells[4]['source'])
    module = ast.parse(source)
    nodes = [node for node in module.body if
        isinstance(node, ast.FunctionDef) and node.name in ('pilot_session', 'terminal_stage_result')
        or isinstance(node, (ast.Import, ast.ImportFrom)) and all(
            alias.name in ('json', 'math', 'time', 'Path') for alias in node.names)]
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook-session>', 'exec'), scope)  # noqa: S102
    return scope


def test_automatically_continues_planned_pause_and_stops_on_complete(functions):
    state = {'seconds': 0, 'saved': 10}
    budgets = []
    def stage(minutes):
        budgets.append(minutes)
        state['seconds'] += 60
        state['saved'] += 1
        return {'status': 'planned_pause' if len(budgets) == 1 else 'complete'}
    result = functions['pilot_session'](stage, lambda: state['saved'], minutes=120,
        segment_minutes=60, gpu=True, update=lambda _: None, clock=lambda: state['seconds'])
    assert result['status'] == 'complete'
    assert budgets == [60, 60]
    assert len(result['segments']) == 2


def test_session_deadline_preserves_pause(functions):
    state = {'seconds': 0, 'saved': 0}
    def stage(minutes):
        assert minutes == 1
        state.update(seconds=61, saved=1)
        return {'status': 'planned_pause'}
    result = functions['pilot_session'](stage, lambda: state['saved'], minutes=1,
        segment_minutes=60, gpu=False, clock=lambda: state['seconds'], update=lambda _: None)
    assert result['status'] == 'planned_pause'
    assert len(result['segments']) == 1


def test_completed_cache_does_not_need_new_fits(functions):
    result = functions['pilot_session'](lambda _: {'status': 'complete'}, lambda: 32,
        minutes=60, segment_minutes=60, gpu=False, update=lambda _: None)
    assert result['status'] == 'complete'


def test_frozen_supervisor_receipt_continues_without_a_new_runner(functions, tmp_path):
    import colab_progress
    state = {'calls': 0}
    def stage(_):
        state['calls'] += 1
        status = 'planned_pause' if state['calls'] == 1 else 'complete'
        command = [sys.executable, '-c',
            f"import json; print(json.dumps({{'status': '{status}'}}, indent=2))"]
        receipt = colab_progress.supervise(command, cwd=tmp_path, env=os.environ.copy(),
            drive_logs=tmp_path/'drive', decisions=tmp_path/'decisions',
            scratch=tmp_path, interval=.01, sync_interval=.01, update=lambda _: None)
        # Frozen supervisors expose log paths without a status field.
        return {**functions['terminal_stage_result'](receipt['local_log']), **receipt}
    result = functions['pilot_session'](stage, lambda: state['calls'], minutes=1,
        segment_minutes=1, gpu=False, update=lambda _: None)
    assert state['calls'] == 2 and result['status'] == 'complete'
    assert all(Path(item['drive_log']).exists() for item in result['segments'])


def test_pause_without_saved_progress_is_not_retried(functions):
    with pytest.raises(RuntimeError, match='no saved progress'):
        functions['pilot_session'](lambda _: {'status': 'planned_pause'}, lambda: 0,
            minutes=60, segment_minutes=60, gpu=True, update=lambda _: None)


@pytest.mark.parametrize('failure', [RuntimeError('writer lock'), KeyboardInterrupt()])
def test_failure_or_interrupt_is_not_retried(functions, failure):
    calls = []
    def stage(_):
        calls.append(1)
        raise failure
    with pytest.raises(type(failure)):
        functions['pilot_session'](stage, lambda: 0, minutes=60,
            segment_minutes=60, gpu=True, update=lambda _: None)
    assert len(calls) == 1


@pytest.mark.parametrize(('minutes', 'segment', 'gpu'), [
    (241, 60, False), (721, 60, True), (60, 61, True), (0, 60, False),
    (float('nan'), 60, True), (60, float('inf'), True)])
def test_invalid_budget_cannot_start_a_stage(functions, minutes, segment, gpu):
    def stage(_):
        pytest.fail('Invalid budget started a stage')
    with pytest.raises(ValueError):
        functions['pilot_session'](stage, lambda: 0, minutes=minutes,
            segment_minutes=segment, gpu=gpu)


@pytest.mark.parametrize('status', ['complete', 'planned_pause'])
def test_terminal_result_reads_final_json_after_large_log(functions, tmp_path, status):
    log = tmp_path / 'local.log'
    log.write_text('x' * 100000 + '\n{"status": "complete"}\nnoise\n'
        + json.dumps({'status': status, 'saved_outputs': 32}, indent=2) + '\n', encoding='utf-8')
    assert functions['terminal_stage_result'](log)['status'] == status


@pytest.mark.parametrize('text', [
    '{"status": "complete"}\nTraceback: failure',
    'PILOT planned_pause: not a terminal result', '{"status": "failed"}',
    '{"status": "complete",'])
def test_missing_or_invalid_terminal_result_stops(functions, tmp_path, text):
    log = tmp_path / 'local.log'
    log.write_text(text, encoding='utf-8')
    with pytest.raises(RuntimeError, match='terminal pilot result'):
        functions['terminal_stage_result'](log)
