"""Exercise task construction without registering a task or collecting data."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


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
    param($MultipleInstances, $ExecutionTimeLimit, [switch]$AllowStartIfOnBatteries, [switch]$DontStopIfGoingOnBatteries)
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

