"""The one-click Colab entry point must stay scoped to the prepared diagnosis."""

import ast
import json
from pathlib import Path


def test_diagnose_notebook_is_pinned_and_does_not_launch_search():
    notebook = Path(__file__).parents[1] / "notebooks/archive/cottonlens_free_research_diagnose.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    for cell in code:
        ast.parse(cell)
    combined = "\n".join(code)
    assert "EXPERIMENT = 'research-free-data-v1'" in combined
    assert "EXPECTED_SOURCE_ID = 'c669ca9a9d4db2647b80993881efda2ba65b050ea26c6fd16a06a5c1e5dd1b93'" in combined
    assert "EXPECTED_ZIP_SHA256 = 'ff4eec811ef9549e7311a467475cca2b7826044566aecf5704b792668b06c6b1'" in combined
    assert "'publication_sources'" in combined
    assert "'--stage', 'diagnose'" in combined
    assert "'--stage', 'ablate'" not in combined
    assert "'--stage', 'search'" not in combined
    assert "total=8" in combined
    assert "'.writer-lock'" in combined


def test_pilot_notebook_runs_one_resume_safe_real_data_candidate():
    notebook = Path(__file__).parents[1] / "notebooks/archive/cottonlens_free_research_pilot.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    for cell in code:
        ast.parse(cell)
    final = ast.parse(code[-1])
    program = next(
        ast.literal_eval(node.value)
        for node in final.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pilot_program" for target in node.targets)
    )
    ast.parse(program)
    assert "experiment.inner(spec, fold)" in program
    assert "recipes('xgboost', 1, experiment.identity['groups']['base'], 16)[0]" in program
    assert "len(pilot_ids) == 6" in program
    assert "digest(root / 'ledger' / name) == checksum" in program
    assert "digest(root / name) == checksum" not in program
    assert "freeze_record(marker, result)" in program
    assert "total=14" in code[-1]
    assert "'--stage', 'ablate'" not in code[-1]
    assert "'--stage', 'search'" not in code[-1]
