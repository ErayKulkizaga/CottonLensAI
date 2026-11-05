param([Parameter(Mandatory=$true)][string]$PythonPath,
      [Parameter(Mandatory=$true)][string]$RepoPath,
      [Parameter(Mandatory=$true)][string]$StorePath,
      [string]$DrivePath,
      [string]$SecretsPath,
      [string]$ExpectedSourceId)
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = Join-Path $RepoPath 'ml/src'
$env:PYTHONNOUSERSITE = '1'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
# This task only collects observations. It never enables local training.
Remove-Item Env:COTTONLENS_ALLOW_LOCAL_CPU_TABULAR -ErrorAction SilentlyContinue
if ($ExpectedSourceId) {
    if ($ExpectedSourceId -notmatch '^[a-f0-9]{64}$') { throw 'Invalid frozen source identity' }
    $verification = "import json,sys; from pathlib import Path; from cottonlens_ml.code_identity import research_source_identity; r=Path(sys.argv[1]); actual=research_source_identity(r); frozen=json.loads((r/'.source-manifest.json').read_text(encoding='utf-8')); assert actual == frozen and actual['source_id'] == sys.argv[2], 'Frozen collector source changed'"
    & $PythonPath -c $verification $RepoPath $ExpectedSourceId
    if ($LASTEXITCODE -ne 0) { throw 'Frozen collector validation failed; no collection performed' }
}
New-Item -ItemType Directory -Path $StorePath -Force | Out-Null
$arguments = @('-m','cottonlens_ml.research.live','--store',$StorePath)
if ($DrivePath) { $arguments += @('--mirror-root',$DrivePath) }
if (-not $SecretsPath) { $SecretsPath = Join-Path $RepoPath 'output/live-secrets.clixml' }
# DPAPI store is outside StorePath, so archive/mirror never includes credentials.
$storeResolved = [System.IO.Path]::GetFullPath($StorePath).TrimEnd('\')+'\'
$secretResolved = [System.IO.Path]::GetFullPath($SecretsPath)
if ($secretResolved.StartsWith($storeResolved,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Credential store must be outside the archive directory'
}
$previous = @{}
try {
    if (Test-Path -LiteralPath $SecretsPath) {
        $secrets = Import-Clixml -LiteralPath $SecretsPath
        foreach ($name in @('USDA_AMS_API_KEY','USDA_FAS_API_KEY','USDA_NASS_API_KEY')) {
            if ($secrets[$name] -isnot [System.Security.SecureString]) { throw 'Invalid encrypted credential store' }
            $previous[$name] = [Environment]::GetEnvironmentVariable($name,'Process')
            $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secrets[$name])
            try {
                [Environment]::SetEnvironmentVariable($name,[Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer),'Process')
            } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
        }
    }
    # Windows PowerShell wraps native stderr as ErrorRecords. A harmless provider
    # warning must not abort Python (or strand its writer lock); use its exit code.
    $ErrorActionPreference = 'Continue'
    try {
        & $PythonPath @arguments *> (Join-Path $StorePath 'task-last-run.log')
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = 'Stop' }
} finally {
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
}
exit $code
