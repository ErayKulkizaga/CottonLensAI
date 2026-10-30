import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from research import ct_forward_backup as module


class FakeGitHub:
    def __init__(self, public=False, corrupt=False):
        self.assets = {}; self.release = False; self.public = public; self.corrupt = corrupt; self.calls = []

    def __call__(self, execution, *args, check=True):
        self.calls.append(args)
        if args[0] == 'api':
            return SimpleNamespace(returncode=0, stdout='false' if self.public else 'true', stderr='')
        if args[1] == 'view':
            return SimpleNamespace(returncode=0 if self.release else 1, stdout='', stderr='' if self.release else 'release not found')
        if args[1] == 'create':
            self.release = True
        elif args[1] == 'upload':
            for name in args[3:5]:
                path = Path(name)
                assert path.name not in self.assets
                self.assets[path.name] = path.read_bytes()
        elif args[1] == 'download':
            target = Path(args[args.index('--dir')+1]); target.mkdir(parents=True, exist_ok=False)
            for i, value in enumerate(args):
                if value == '--pattern':
                    name = args[i+1]; data = self.assets[name]
                    (target/name).write_bytes(b'CORRUPTED' if self.corrupt and name.endswith('.zip') else data)
        return SimpleNamespace(returncode=0, stdout='', stderr='')


def environment(tmp_path):
    study = tmp_path/'study'; study.mkdir(); (study/'model.json').write_text('{"fixture":1}')
    gh = tmp_path/'gh.exe'; gh.write_bytes(b'not executable: mocked CLI')
    runtime = tmp_path/'runtime'; runtime.mkdir(); (runtime/'ct_forward_daily.py').write_bytes(b'fixture')
    body = {'backup_repository': 'ErayKulkizaga/CottonLensAI', 'gh_path': str(gh), 'gh_sha256': module.file_sha(gh),
            'core_source_id': 'fixture', 'files': {'ct_forward_daily.py': module.file_sha(runtime/'ct_forward_daily.py')},
            'repository_commit': '0'*40}
    body['execution_source_id'] = module.fingerprint(body)
    (runtime/'execution-manifest.json').write_text(json.dumps(body))
    return study, {**body, 'runtime_path': str(runtime)}


def test_private_delta_is_freshly_downloaded_and_preserves_all_prior_objects(tmp_path):
    study, execution = environment(tmp_path); github = FakeGitHub()
    a = module.backup(study, execution, '0'*32, 'manual', gh_call=github)
    old_assets = dict(github.assets)
    (study/'new-origin.json').write_text('{"fixture":2}')
    b = module.backup(study, execution, '1'*32, 'publish', gh_call=github)
    assert a['status'] == b['status'] == 'private_delta_fresh_download_verified'
    assert b['parent_backup_id'] == a['backup_id'] and b['delta_scientific_files'] == 1
    assert a['source_files_deleted'] == b['source_files_deleted'] == 0
    assert all(github.assets[k] == v for k, v in old_assets.items())
    assert (study/'model.json').read_text() == '{"fixture":1}'
    assert module.read(study/'backups/head.json')['backup_id'] == b['backup_id']
    assert sum(c[1] == 'download' for c in github.calls if c[0] == 'release') == 2
    assert not any('--clobber' in c or 'delete' in c for c in github.calls)


@pytest.mark.parametrize('failure', ['public', 'corrupt', 'wrong_binary', 'wrong_runtime'])
def test_unverified_destination_or_payload_cannot_advance_backup_head(tmp_path, failure):
    study, execution = environment(tmp_path)
    if failure == 'wrong_binary':
        execution['gh_sha256'] = 'wrong'
    elif failure == 'wrong_runtime':
        (Path(execution['runtime_path'])/'ct_forward_daily.py').write_bytes(b'tampered')
    github = FakeGitHub(public=failure == 'public', corrupt=failure == 'corrupt')
    with pytest.raises(ValueError):
        module.backup(study, execution, '2'*32, 'capture', gh_call=github)
    assert not (study/'backups/head.json').exists()
    assert (study/'model.json').read_text() == '{"fixture":1}'


def test_disappeared_original_is_not_mirrored_as_a_cloud_delete(tmp_path):
    study, execution = environment(tmp_path); github = FakeGitHub()
    module.backup(study, execution, '3'*32, 'manual', gh_call=github)
    before = dict(github.assets); (study/'model.json').unlink()
    with pytest.raises(ValueError, match='disappeared'):
        module.backup(study, execution, '4'*32, 'manual', gh_call=github)
    assert github.assets == before


def test_github_cli_has_no_console_stdin_or_unbounded_wait():
    with patch.object(module.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
        module.gh({'gh_path': 'gh.exe'}, 'version')
    assert run.call_args.kwargs['stdin'] == subprocess.DEVNULL
    assert run.call_args.kwargs['timeout'] == 90
    expected = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    assert run.call_args.kwargs['creationflags'] == expected
