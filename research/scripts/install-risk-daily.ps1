param([Parameter(Mandatory=$true)][string]$PythonwPath,
      [Parameter(Mandatory=$true)][string]$RunnerPath,
      [Parameter(Mandatory=$true)][string]$StudyPath,
      [Parameter(Mandatory=$true)][string]$SourcePath,
      [Parameter(Mandatory=$true)][string]$AuthorizationPath,
      [Parameter(Mandatory=$true)][string]$ExpectedExecutionSource,
      [Parameter(Mandatory=$true)][string]$ExpectedCoreSource,
      [string]$TaskName='CottonLens-Risk-Daily-v1')
$ErrorActionPreference='Stop'
if ($TaskName -eq 'CottonLens-Observed-Archive-v2') { throw 'The old hourly task must remain disabled' }
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) { throw 'Task exists; inspect it instead of silently replacing it' }
if ([IO.Path]::GetFileName($PythonwPath) -ne 'pythonw.exe') { throw 'Use the GUI-subsystem pythonw executable, not a console shell' }
foreach($p in @($PythonwPath,$RunnerPath,$AuthorizationPath)) {
    if(-not(Test-Path -LiteralPath $p -PathType Leaf)){throw "Missing daily runtime dependency: $p"}
}
$pythonImage=[IO.File]::ReadAllBytes($PythonwPath)
$peOffset=[BitConverter]::ToInt32($pythonImage,60)
if([BitConverter]::ToUInt16($pythonImage,$peOffset+24+68) -ne 2){throw 'The scheduled executable must use the Windows GUI subsystem; a renamed console executable is not accepted'}
foreach($p in @($PythonwPath,$RunnerPath,$StudyPath,$SourcePath,$AuthorizationPath)) {
    if($p.Contains('"')){throw 'Runtime paths cannot contain quotation marks'}
}
if($ExpectedExecutionSource -notmatch '^[a-f0-9]{64}$' -or $ExpectedCoreSource -notmatch '^[a-f0-9]{64}$') {throw 'Expected immutable identities required'}
$taskArguments='"'+$RunnerPath+'" --study "'+$StudyPath+'" --source "'+$SourcePath+'" --authorization "'+$AuthorizationPath+'" --expected-execution-source '+$ExpectedExecutionSource+' --expected-core-source '+$ExpectedCoreSource+' --phase auto'
$escape={param($v) [System.Security.SecurityElement]::Escape($v)}
$first=[DateTime]::UtcNow.Date.AddMinutes(5)
if($first -le [DateTime]::UtcNow){$first=$first.AddDays(1)}
$firstCapture=$first.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")
$firstPublish=$first.Date.AddMinutes(20).ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")
$sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$xml=@"
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
<RegistrationInfo><Description>Explicitly approved consoleless daily Cotton risk capture and research publication; old hourly task stays disabled</Description></RegistrationInfo>
<Triggers>
<CalendarTrigger id="DailyCapture"><StartBoundary>$firstCapture</StartBoundary><EndBoundary>2028-01-02T00:30:00Z</EndBoundary><Enabled>true</Enabled><ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>
<CalendarTrigger id="DailyPublish"><StartBoundary>$firstPublish</StartBoundary><EndBoundary>2028-01-02T00:30:00Z</EndBoundary><Enabled>true</Enabled><ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>
</Triggers>
<Principals><Principal id="Owner"><UserId>$sid</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
<Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><AllowHardTerminate>true</AllowHardTerminate><StartWhenAvailable>false</StartWhenAvailable><RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable><IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings><AllowStartOnDemand>true</AllowStartOnDemand><Enabled>true</Enabled><Hidden>false</Hidden><RunOnlyIfIdle>false</RunOnlyIfIdle><WakeToRun>false</WakeToRun><ExecutionTimeLimit>PT10M</ExecutionTimeLimit><Priority>7</Priority></Settings>
<Actions Context="Owner"><Exec><Command>$(& $escape $PythonwPath)</Command><Arguments>$(& $escape $taskArguments)</Arguments><WorkingDirectory>$(& $escape ([IO.Path]::GetDirectoryName($RunnerPath)))</WorkingDirectory></Exec></Actions>
</Task>
"@
$consent=Get-Content -LiteralPath (Join-Path $StudyPath 'human-authorization.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$approvedAnswer='"Evet, yaln\u0131z g\u00fcnl\u00fck ve konsolsuz g\u00f6rev"' | ConvertFrom-Json
if($consent.human_user_answer -ne $approvedAnswer){throw 'Explicit daily-consoleless consent required'}
$snapshotPath=Join-Path $StudyPath 'daily-task-registration.xml'
$stream=[IO.File]::Open($snapshotPath,[IO.FileMode]::CreateNew)
try{$bytes=[Text.Encoding]::UTF8.GetBytes($xml);$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
Register-ScheduledTask -TaskName $TaskName -Xml $xml | Out-Null
$task=Get-ScheduledTask -TaskName $TaskName
if($task.Actions.Count -ne 1 -or $task.Actions[0].Execute -ne $PythonwPath -or $task.Triggers.Count -ne 2){throw 'Persisted daily task differs from reviewed configuration'}
$task | Select-Object TaskName,@{n='State';e={[string]$_.State}},@{n='Executor';e={$_.Actions[0].Execute}}
