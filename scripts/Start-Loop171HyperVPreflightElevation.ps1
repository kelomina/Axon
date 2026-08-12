[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$BaseImageArchive,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$LinuxCapaArchive,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$DedicatedRoot,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ReceiptPath,
    [string]$PlannedVmName = 'AxonLoop171CapaIsolated'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$preflight = Join-Path $PSScriptRoot 'Invoke-Loop171HyperVPreflight.ps1'
if (-not (Test-Path -LiteralPath $preflight -PathType Leaf)) {
    throw 'Loop171 Hyper-V preflight script is missing.'
}

$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $preflight,
    '-BaseImageArchive', $BaseImageArchive,
    '-LinuxCapaArchive', $LinuxCapaArchive,
    '-DedicatedRoot', $DedicatedRoot,
    '-PlannedVmName', $PlannedVmName,
    '-ReceiptPath', $ReceiptPath
)
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -Verb RunAs -Wait -PassThru
if ($process.ExitCode -ne 0) {
    exit $process.ExitCode
}
