[CmdletBinding()]
param(
    [string]$DedicatedParent = 'C:\ProgramData\AxonLab',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')]
    [string]$RootDirectoryName = 'loop171-20260715-1049',
    [string]$ReceiptPath = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ReceiptPath)) {
    $ReceiptPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'reports\roadmap_9997\loop171\protected_root_creation_20260715.json'
}

$initializer = Join-Path $PSScriptRoot 'Initialize-Loop171ProtectedRoot.ps1'
if (-not (Test-Path -LiteralPath $initializer -PathType Leaf)) {
    throw 'Loop171 protected-root initializer is missing.'
}

$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $initializer,
    '-DedicatedParent', $DedicatedParent,
    '-RootDirectoryName', $RootDirectoryName,
    '-ReceiptPath', $ReceiptPath
)
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -Verb RunAs -Wait -PassThru
if ($process.ExitCode -ne 0) {
    exit $process.ExitCode
}
