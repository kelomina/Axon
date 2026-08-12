[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$BaseImageArchive,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$LinuxCapaArchive,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$DedicatedRoot,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$PlannedVmName,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$HarmlessFixtureIso,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$HarmlessFixtureIsoSha256,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$PreflightReceiptPath,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ReceiptPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$executor = Join-Path $PSScriptRoot 'Invoke-Loop171HarmlessFixtureAcceptance.ps1'
if (-not (Test-Path -LiteralPath $executor -PathType Leaf)) { throw 'Loop171 harmless-fixture acceptance executor is missing.' }
$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $executor, '-BaseImageArchive', $BaseImageArchive, '-LinuxCapaArchive', $LinuxCapaArchive, '-DedicatedRoot', $DedicatedRoot, '-PlannedVmName', $PlannedVmName, '-HarmlessFixtureIso', $HarmlessFixtureIso, '-HarmlessFixtureIsoSha256', $HarmlessFixtureIsoSha256, '-PreflightReceiptPath', $PreflightReceiptPath, '-ReceiptPath', $ReceiptPath)
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -Verb RunAs -Wait -PassThru
if ($process.ExitCode -ne 0) { exit $process.ExitCode }
