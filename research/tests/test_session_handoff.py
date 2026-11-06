import ast
import hashlib
import json
import zipfile

import pytest

from research import build_session_handoff as builder


def test_handoff_repeatability_and_changed_source_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    (tmp_path/'ml/src').mkdir(parents=True)
    for name in ('gpu_session_probe.py', 'gpu_restart_probe.py', 'a100_preflight.py', 'colab_setup.py', 'pyproject.toml', 'uv.lock'):
        (tmp_path/'ml'/name).write_text('generated fixture')
    (tmp_path/'research/evidence').mkdir(parents=True)
    for name in ('l4-technical-verification-20261106.json', 'l4-restart-verification-20261106.json', 'l4-oom-verification-20261107.json'):
        (tmp_path/'research/evidence'/name).write_text('{"generated_fixture":true}')
    first = builder.build()
    assert builder.build() == first
    root = tmp_path/'output/l4-session-handoff-20261107-v2'
    notebook = json.loads((root/'CottonLens_L4_Remaining_Checks_Drive_v2.ipynb').read_text(encoding='utf-8'))
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    ast.parse(code)
    from colab_setup import PROJECT
    assert f'root = Path({PROJECT.as_posix()!r})' in code
    assert 'files.upload' not in code and 'files.download' not in code
    with zipfile.ZipFile(root/'cottonlens-l4-session-source.zip') as z:
        assert len([n for n in z.namelist() if n.startswith('prior-evidence/')]) == 3
        assert hashlib.sha256((root/'cottonlens-l4-session-source.zip').read_bytes()).hexdigest() == first['source_sha256']
    (tmp_path/'ml/gpu_session_probe.py').write_text('changed fixture')
    with pytest.raises(ValueError, match='Preserve previous handoff'):
        builder.build()
