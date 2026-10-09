"""Exercise task construction without registering a task or collecting data."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cottonlens_ml.code_identity import research_source_identity


@pytest.mark.skipif(not shutil.which('powershell.exe'), reason='Windows task construction')
def test_logon_trigger_is_scoped_to_current_user(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/install-live-task.ps1'
    # Stubs validate the constructed task, including Windows' non-admin boundary.
    harness = tmp_path / 'task-contract.ps1'
    harness.write_text(r'''
param($Installer, $Repo)
$ErrorActionPreference = 'Stop'
$global:TaskTestOwner = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
function Get-ScheduledTask { param($TaskName, $ErrorAction) }
function New-ScheduledTaskAction { param($Execute, $Argument, $WorkingDirectory) }
function New-ScheduledTaskTrigger {
    param([switch]$AtLogOn, $User, [switch]$Once, $At, $RepetitionInterval, [switch]$Daily)
    if ($AtLogOn -and $User -ne $global:TaskTestOwner) { throw 'Unscoped logon trigger requires admin' }
    @{kind= $(if ($AtLogOn) {'logon'} elseif ($Daily) {'daily'} else {'hourly'}); user=$User}
}
function New-ScheduledTaskSettingsSet {
    param($MultipleInstances, $ExecutionTimeLimit, [switch]$AllowStartIfOnBatteries, [switch]$DontStopIfGoingOnBatteries, [switch]$WakeToRun)
    if ($MultipleInstances -ne 'IgnoreNew') { throw 'Concurrent writers enabled' }
}
function New-ScheduledTaskPrincipal {
    param($UserId, $LogonType, $RunLevel)
    if ($UserId -ne $global:TaskTestOwner -or $LogonType -ne 'Interactive' -or $RunLevel -ne 'Limited') { throw 'Privilege contract' }
}
function Register-ScheduledTask {
    param($TaskName, $Action, $Trigger, $Settings, $Principal, $Description)
    if ($Trigger.Count -ne 3) { throw 'Missing trigger' }
    [pscustomobject]@{TaskName=$TaskName; State='ValidatedWithoutRegistration'}
}
& $Installer -PythonPath (Get-Process -Id $PID).Path -RepoPath $Repo -StorePath (Join-Path $Repo 'output/synthetic-only')
''', encoding='utf-8')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File', str(harness),
                             str(script), str(script.parents[2])], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'ValidatedWithoutRegistration' in result.stdout


@pytest.mark.skipif(not shutil.which('powershell.exe'), reason='Windows native stderr')
@pytest.mark.parametrize('code', [0, 7])
def test_collector_stderr_does_not_abort_native_process(tmp_path, code):
    script = Path(__file__).resolve().parents[1] / 'scripts/live-task.ps1'
    package = tmp_path / 'ml/src/cottonlens_ml/research'
    package.mkdir(parents=True)
    (package / 'live.py').write_text(
        f"import sys\nprint('provider warning', file=sys.stderr, flush=True)\n"
        f"print('collector completed', flush=True)\nsys.exit({code})\n", encoding='utf-8')
    store = tmp_path / 'archive'
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File', str(script),
                             '-PythonPath', sys.executable, '-RepoPath', str(tmp_path),
                             '-StorePath', str(store)], capture_output=True, check=False)
    assert result.returncode == code
    log = (store / 'task-last-run.log').read_text(encoding='utf-16')
    assert 'provider warning' in log and 'collector completed' in log


@pytest.mark.skipif(not shutil.which('powershell.exe'), reason='Windows pinned source launcher')
@pytest.mark.parametrize('corrupt', [False, True])
def test_pinned_launcher_checks_source_before_collection(tmp_path, corrupt):
    script = Path(__file__).resolve().parents[1] / 'scripts/live-task.ps1'
    package = tmp_path / 'ml/src/cottonlens_ml'
    (package / 'research').mkdir(parents=True)
    original = script.parents[1] / 'src/cottonlens_ml/code_identity.py'
    shutil.copyfile(original, package / 'code_identity.py')
    module = package / 'research/live.py'
    module.write_text("print('synthetic collector completed')\n", encoding='utf-8')
    identity = research_source_identity(tmp_path)
    (tmp_path / '.source-manifest.json').write_text(json.dumps(identity), encoding='utf-8')
    if corrupt:
        module.write_text("raise RuntimeError('must not execute')\n", encoding='utf-8')
    store = tmp_path / 'archive'
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File', str(script),
                             '-PythonPath', sys.executable, '-RepoPath', str(tmp_path),
                             '-StorePath', str(store), '-ExpectedSourceId', identity['source_id']],
                            capture_output=True, check=False)
    if corrupt:
        assert result.returncode != 0 and not store.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert 'synthetic collector completed' in (store / 'task-last-run.log').read_text(encoding='utf-16')


@pytest.mark.skipif(not shutil.which('powershell.exe'), reason='Windows task update contract')
@pytest.mark.parametrize('mode', ['success', 'running', 'persist_failure', 'success_sid', 'foreign'])
def test_task_update_preserves_triggers_and_rolls_back_failed_persistence(tmp_path, mode):
    script = Path(__file__).resolve().parents[1] / 'scripts/install-live-task.ps1'
    package = tmp_path / 'ml/src/cottonlens_ml'
    package.mkdir(parents=True)
    shutil.copyfile(script.parents[1] / 'src/cottonlens_ml/code_identity.py', package / 'code_identity.py')
    (tmp_path / 'ml/scripts').mkdir()
    shutil.copyfile(script, tmp_path / 'ml/scripts/live-task.ps1')
    identity = research_source_identity(tmp_path)
    (tmp_path / '.source-manifest.json').write_text(json.dumps(identity), encoding='utf-8')
    harness = tmp_path / 'update-contract.ps1'
    harness.write_text(r'''
param($Installer,$Python,$Repo,$SourceId,$Mode)
$ErrorActionPreference='Stop'
$global:Updated=$false
$global:RolledBack=$false
$global:Owner=if ($Mode -eq 'success_sid') {[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value} elseif ($Mode -eq 'foreign') {'S-1-5-18'} else {[System.Security.Principal.WindowsIdentity]::GetCurrent().Name}
function Get-ScheduledTask {
    param($TaskName,$ErrorAction)
    $argsText=if ($global:Updated) {$global:NewAction.Arguments} else {'old action'}
    if ($global:Updated -and $Mode -eq 'persist_failure') {$argsText='not persisted'}
    [pscustomobject]@{TaskName=$TaskName;State=$(if ($Mode -eq 'running') {'Running'} else {'Ready'});
      Principal=[pscustomobject]@{UserId=$global:Owner;LogonType='Interactive';RunLevel='Limited'};
      Actions=@([pscustomobject]@{Arguments=$argsText});Settings=[pscustomobject]@{WakeToRun=$global:Updated}}
}
function New-ScheduledTaskAction {param($Execute,$Argument,$WorkingDirectory)
    [pscustomobject]@{Arguments=$Argument;WorkingDirectory=$WorkingDirectory}}
function Export-ScheduledTask {param($TaskName);'<Task>synthetic</Task>'}
function Set-ScheduledTask {param($TaskName,$Action,$Settings)
    if (-not $Settings.WakeToRun) {throw 'Wake flag missing'}
    $global:NewAction=$Action;$global:Updated=$true}
function Register-ScheduledTask {param($TaskName,$Xml,[switch]$Force)
    if ($Xml -ne '<Task>synthetic</Task>' -or -not $Force) {throw 'Wrong rollback'}
    $global:RolledBack=$true}
function New-ScheduledTaskTrigger {throw 'Existing triggers must not be rebuilt'}
function New-ScheduledTaskPrincipal {throw 'Existing principal must not be rebuilt'}
$backup=Join-Path $Repo 'private-before.xml'
try {
    & $Installer -PythonPath $Python -RepoPath $Repo -StorePath (Join-Path $Repo 'archive') `
      -ExpectedSourceId $SourceId -UpdateExisting -WakeToRun -BackupPath $backup
    if ($Mode -notin @('success','success_sid')) {throw 'Expected update rejection'}
} catch {
    if ($Mode -in @('running','foreign') -and -not $global:Updated -and -not (Test-Path $backup)) {'REFUSED_ACTIVE';exit 0}
    if ($Mode -eq 'persist_failure' -and $global:RolledBack) {'ROLLBACK_CONFIRMED';exit 0}
    throw
}
if (-not $global:Updated -or $global:RolledBack -or [IO.File]::ReadAllText($backup) -ne '<Task>synthetic</Task>') {throw 'Bad update/backup'}
'UPDATE_CONFIRMED'
''', encoding='utf-8')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File', str(harness),
                             str(script), sys.executable, str(tmp_path), identity['source_id'], mode],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert {'success': 'UPDATE_CONFIRMED', 'success_sid': 'UPDATE_CONFIRMED',
            'running': 'REFUSED_ACTIVE', 'foreign': 'REFUSED_ACTIVE',
            'persist_failure': 'ROLLBACK_CONFIRMED'}[mode] in result.stdout

