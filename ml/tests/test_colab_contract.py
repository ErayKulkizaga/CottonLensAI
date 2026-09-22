import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_notebook_is_single_valid_python_orchestration():
    notebook = json.loads((ROOT / "ml/notebooks/cottonlens_colab.ipynb").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [cell["id"] for cell in code] == ["drive", "repo", "environment", "smoke", "data", "train", "artifact"]
    for cell in code:
        ast.parse("".join(cell["source"]))


def test_headless_backend_overrides_inherited_colab_value_before_import():
    env = dict(os.environ, MPLBACKEND="module://matplotlib_inline.backend_inline")
    result = subprocess.run([sys.executable, "-c", "import cottonlens_ml, os; print(os.environ['MPLBACKEND'])"],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "Agg"


def test_streaming_runner_displays_original_error_and_stops(capsys):
    spec = importlib.util.spec_from_file_location("colab_setup", ROOT / "ml/colab_setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    import pytest
    with pytest.raises(RuntimeError, match="Stage failed"):
        setup.run([sys.executable, "-c", "raise ValueError('ORIGINAL DIAGNOSTIC')"], cwd=ROOT)
    assert "ValueError: ORIGINAL DIAGNOSTIC" in capsys.readouterr().out


def test_all_direct_pins_match_lock_and_constraints():
    project = tomllib.loads((ROOT / "ml/pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "ml/uv.lock").read_text())
    pins = {item.split("==")[0]: item.split("==")[1] for item in project["project"]["dependencies"]}
    locked = {item["name"]: item["version"] for item in lock["package"]}
    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert all(locked[name] == version for name, version in pins.items())
    constraints = (ROOT / "ml/constraints/colab-py312.txt").read_text().splitlines()
    assert all(f"{name}=={version}" in constraints for name, version in pins.items())


def test_setup_uses_managed_python_lock_and_only_targeted_bootstrap(monkeypatch):
    spec = importlib.util.spec_from_file_location("colab_setup", ROOT / "ml/colab_setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    commands = []
    monkeypatch.setattr(setup.sys, "platform", "linux")
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(setup, "run", lambda command, **kwargs: commands.append([str(x) for x in command]))
    setup.setup()
    bootstrap = commands[0]
    assert "--target" in bootstrap and "--no-deps" in bootstrap
    sync = next(command for command in commands if "sync" in command)
    assert all(option in sync for option in ("--locked", "--managed-python", "3.12.11", "cuda"))
    assert not any("--system-site-packages" in command or "--system" in command for command in commands)
