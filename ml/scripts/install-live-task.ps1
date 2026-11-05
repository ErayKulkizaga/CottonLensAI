param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$RepoPath,
      [Parameter(Mandatory=$true)][string]$StorePath,
      [string]$DrivePath,
      [string]$SecretsPath,
      [string]$TaskName = 'CottonLens-Observed-Archive-v2',
      [string]$ExpectedSourceId,
      [switch]$UpdateExisting,
      [switch]$WakeToRun,
      [string]$BackupPath)
$ErrorActionPreference = 'Stop'
$taskScript = Join-Path $RepoPath 'ml/scripts/live-task.ps1'
foreach ($path in @($PythonPath,$taskScript)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing task dependency: $path" }
}
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and -not $UpdateExisting) {
    throw 'Task already exists; inspect its action instead of silently replacing it'
}
if ($UpdateExisting -and (-not $existing -or -not $ExpectedSourceId -or -not $BackupPath)) {
    throw 'Update requires existing task, frozen identity and a new private XML backup path'
}
if ($ExpectedSourceId) {
    if ($ExpectedSourceId -notmatch '^[a-f0-9]{64}$') { throw 'Invalid frozen source identity' }
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $RepoPath 'ml/src'
        $verification = "import json,sys; from pathlib import Path; from cottonlens_ml.code_identity import research_source_identity; r=Path(sys.argv[1]); actual=research_source_identity(r); frozen=json.loads((r/'.source-manifest.json').read_text(encoding='utf-8')); assert actual == frozen and actual['source_id'] == sys.argv[2], 'Frozen collector source changed'"
        & $PythonPath -c $verification $RepoPath $ExpectedSourceId
        if ($LASTEXITCODE -ne 0) { throw 'Frozen source validation failed' }
    } finally { $env:PYTHONPATH = $previousPythonPath }
}
foreach ($value in @($PythonPath,$RepoPath,$StorePath,$DrivePath,$SecretsPath)) {
    if ($value.Contains('"')) { throw 'Task paths cannot contain quotation marks' }
}
$argsText = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $taskScript + '" -PythonPath "' + $PythonPath + '" -RepoPath "' + $RepoPath + '" -StorePath "' + $StorePath + '"'
if ($DrivePath) { $argsText += ' -DrivePath "' + $DrivePath + '"' }
if ($SecretsPath) { $argsText += ' -SecretsPath "' + $SecretsPath + '"' }
if ($ExpectedSourceId) { $argsText += ' -ExpectedSourceId "' + $ExpectedSourceId + '"' }
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argsText -WorkingDirectory $RepoPath
if ($UpdateExisting) {
    # Windows may persist the account as its SID rather than DOMAIN\name.
    # Compare resolved security identities, never weaken the ownership check.
    $taskOwner = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $taskAccount = [string]$existing.Principal.UserId
    if ($taskAccount -match '^S-1-') {
        $existingOwner = [System.Security.Principal.SecurityIdentifier]::new($taskAccount).Value
    } else {
        $existingOwner = [System.Security.Principal.NTAccount]::new($taskAccount).Translate([System.Security.Principal.SecurityIdentifier]).Value
    }
    if ($existing.State -ne 'Ready' -or $existingOwner -ne $taskOwner -or
        $existing.Principal.LogonType -ne 'Interactive' -or $existing.Principal.RunLevel -ne 'Limited' -or
        (Test-Path -LiteralPath (Join-Path $StorePath '.writer-lock'))) {
        throw 'Update requires idle task owned by current limited interactive user and no active writer'
    }
    $backupFull = [System.IO.Path]::GetFullPath($BackupPath)
    $storeFull = [System.IO.Path]::GetFullPath($StorePath).TrimEnd('\')+'\'
    if ($backupFull.StartsWith($storeFull,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Private task XML must be outside the mirrored data store'
    }
    $xml = Export-ScheduledTask -TaskName $TaskName
    $stream = [System.IO.File]::Open($backupFull, [System.IO.FileMode]::CreateNew)
    try { $bytes=[System.Text.Encoding]::UTF8.GetBytes($xml); $stream.Write($bytes,0,$bytes.Length) }
    finally { $stream.Dispose() }
    $settings = $existing.Settings
    if ($WakeToRun) { $settings.WakeToRun = $true }
    try {
        Set-ScheduledTask -TaskName $TaskName -Action $action -Settings $settings | Out-Null
        $updated = Get-ScheduledTask -TaskName $TaskName
        if ($updated.Actions.Count -ne 1 -or $updated.Actions[0].Arguments -ne $argsText -or
            ($WakeToRun -and -not $updated.Settings.WakeToRun)) { throw 'Task update did not persist' }
    } catch {
        Register-ScheduledTask -TaskName $TaskName -Xml $xml -Force | Out-Null
        throw
    }
    $updated | Select-Object TaskName,State
    return
}
$next = [DateTime]::UtcNow.Date.AddMinutes(5)
while ($next -le [DateTime]::UtcNow) { $next = $next.AddHours(1) }
$hourly = New-ScheduledTaskTrigger -Once -At $next.ToLocalTime() -RepetitionInterval ([TimeSpan]::FromHours(1))
$publish = New-ScheduledTaskTrigger -Daily -At ([DateTime]::UtcNow.Date.AddMinutes(20).ToLocalTime())
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
# An unscoped logon trigger means every user and requires administrator rights.
# Scope it to the same interactive account that owns the limited task.
$triggers = @((New-ScheduledTaskTrigger -AtLogOn -User $user),$hourly,$publish)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::FromMinutes(15)) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun:$WakeToRun
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings -Principal $principal -Description 'Local-first Cotton sources and Naive/EWMA forward records; no training and no missed-forecast backfill' | Select-Object TaskName,State
