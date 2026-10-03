param([Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputPath) { throw 'Secret store already exists; it will not be overwritten' }
$parentPath = Split-Path -Parent ([System.IO.Path]::GetFullPath($OutputPath))
New-Item -ItemType Directory -Path $parentPath -Force | Out-Null
$secrets = @{}
foreach ($name in @('USDA_AMS_API_KEY','USDA_FAS_API_KEY','USDA_NASS_API_KEY')) {
    $secrets[$name] = Read-Host "$name (input hidden)" -AsSecureString
    if ($secrets[$name].Length -eq 0) { throw 'All three credentials are required; nothing saved' }
}
$pendingPath = Join-Path $parentPath ([Guid]::NewGuid().ToString('N')+'.pending')
try {
    # Windows DPAPI binds SecureString values to this account on this computer.
    $secrets | Export-Clixml -LiteralPath $pendingPath
    $user = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    $acl = Get-Acl -LiteralPath $pendingPath
    $acl.SetAccessRuleProtection($true,$false)
    $acl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new($user,'FullControl','Allow'))
    Set-Acl -LiteralPath $pendingPath -AclObject $acl
    Move-Item -LiteralPath $pendingPath -Destination $OutputPath
} finally {
    if (Test-Path -LiteralPath $pendingPath) { Remove-Item -LiteralPath $pendingPath }
}
Write-Output 'Encrypted credentials saved locally for the current Windows account. Keep this file outside the archive and Drive.'
