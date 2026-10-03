param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$RepoPath,
      [Parameter(Mandatory=$true)][string]$StorePath,
      [string]$DrivePath,
      [string]$SecretsPath,
      [string]$TaskName = 'CottonLens-Observed-Archive-v2')
$ErrorActionPreference = 'Stop'
$taskScript = Join-Path $RepoPath 'ml/scripts/live-task.ps1'
foreach ($path in @($PythonPath,$taskScript)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing task dependency: $path" }
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw 'Task already exists; inspect its action instead of silently replacing it'
}
foreach ($value in @($PythonPath,$RepoPath,$StorePath,$DrivePath,$SecretsPath)) {
    if ($value.Contains('"')) { throw 'Task paths cannot contain quotation marks' }
}
$argsText = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $taskScript + '" -PythonPath "' + $PythonPath + '" -RepoPath "' + $RepoPath + '" -StorePath "' + $StorePath + '"'
if ($DrivePath) { $argsText += ' -DrivePath "' + $DrivePath + '"' }
if ($SecretsPath) { $argsText += ' -SecretsPath "' + $SecretsPath + '"' }
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argsText -WorkingDirectory $RepoPath
$next = [DateTime]::UtcNow.Date.AddMinutes(5)
while ($next -le [DateTime]::UtcNow) { $next = $next.AddHours(1) }
$hourly = New-ScheduledTaskTrigger -Once -At $next.ToLocalTime() -RepetitionInterval ([TimeSpan]::FromHours(1))
$publish = New-ScheduledTaskTrigger -Daily -At ([DateTime]::UtcNow.Date.AddMinutes(20).ToLocalTime())
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
# An unscoped logon trigger means every user and requires administrator rights.
# Scope it to the same interactive account that owns the limited task.
$triggers = @((New-ScheduledTaskTrigger -AtLogOn -User $user),$hourly,$publish)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::FromMinutes(15)) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings -Principal $principal -Description 'Local-first Cotton sources and Naive/EWMA forward records; no training and no missed-forecast backfill' | Select-Object TaskName,State
