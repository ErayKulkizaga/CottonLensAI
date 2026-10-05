import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/presentation.py'
    spec = importlib.util.spec_from_file_location('presentation', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('name', ['../backend', '.', 'C:/temp', 'nested/demo', '', 'UPPER'])
def test_demo_paths_reject_ambiguous_targets(name):
    with pytest.raises(ValueError):
        module().demo_paths(name)


def test_demo_uses_only_explicit_isolated_locations():
    presentation = module()
    root, db, artifact = presentation.demo_paths('r2-demo')
    assert root.is_relative_to(presentation.REPO / 'output/presentation')
    assert db != presentation.REPO / 'backend/cottonlens.db'
    assert artifact != presentation.REPO / 'runtime/artifacts/current'
    assert db.parent == artifact.parent == root
