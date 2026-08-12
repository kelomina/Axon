[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$BaseImageArchive,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$LinuxCapaArchive,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$DedicatedRoot,
    [Parameter(Mandatory = $true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')][string]$PlannedVmName,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$HarmlessFixtureIso,
    [Parameter(Mandatory = $true)][ValidatePattern('^[a-f0-9]{64}$')][string]$HarmlessFixtureIsoSha256,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$PreflightReceiptPath,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ReceiptPath,
    [UInt64]$MinimumAvailableMemoryBytes = 13958643712,
    [ValidateRange(30, 900)][int]$BootTimeoutSeconds = 300,
    [ValidateRange(4096, 65536)][int]$MaxReceiptBytes = 65536
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$LoopId = 'loop171_hyperv_isolation'
$ReceiptSchema = 'axon_loop171_harmless_fixture_acceptance_v1'
$FixtureSchema = 'axon_loop171_fixture_v1'
$createdVm = $false
$createdChildVhd = $false
$baseVhdPath = $null
$childVhdPath = $null
$vmPath = $null
$pipe = $null
$errors = New-Object System.Collections.Generic.List[string]
$details = [ordered]@{}
$gates = [ordered]@{}
$decision = 'harmless_fixture_acceptance_blocked_fail_closed'

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $volumeRoot = [System.IO.Path]::GetPathRoot($fullPath)
    if ($fullPath.Equals($volumeRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Filesystem-root paths are forbidden.' }
    return $fullPath.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
}

function Assert-NoReparsePoint {
    param([Parameter(Mandatory = $true)][string]$Path)
    $current = Get-Item -LiteralPath $Path -Force
    while ($true) {
        if (($current.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse-point paths are forbidden: $($current.FullName)" }
        $parent = if ($current -is [System.IO.DirectoryInfo]) { $current.Parent } else { $current.Directory }
        if ($null -eq $parent -or $parent.FullName -eq $current.FullName) { return }
        $current = $parent
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-ElevatedAdministrator {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'An elevated Windows administrator token is required.' }
}

function Assert-EmptyDedicatedRoot {
    param([Parameter(Mandatory = $true)][string]$Root)
    Assert-NoReparsePoint -Path $Root
    if (@(Get-ChildItem -LiteralPath $Root -Force).Count -ne 0) { throw 'Dedicated root must be empty before acceptance materializes disposable assets.' }
}

function Set-Gate {
    param([Parameter(Mandatory = $true)][string]$Name, [Parameter(Mandatory = $true)][scriptblock]$Check)
    try { $details[$Name] = & $Check; $gates[$Name] = $true }
    catch { $gates[$Name] = $false; $errors.Add("${Name}: $($_.Exception.Message)") }
}

function Write-Receipt {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)]$Payload)
    if (Test-Path -LiteralPath $Path) { throw 'Acceptance receipt overwrite is forbidden.' }
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) { throw 'Acceptance receipt parent must already exist.' }
    $encoded = [System.Text.Encoding]::UTF8.GetBytes(($Payload | ConvertTo-Json -Depth 10))
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try { $stream.Write($encoded, 0, $encoded.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}

function Get-ProcessIdsByName {
    param([Parameter(Mandatory = $true)][string]$Name)
    return @((Get-Process -Name $Name -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id) | Sort-Object -Unique)
}

function Remove-RegisteredAssets {
    # 仅处理本次固定登记的 VM、差分盘和根内目录，绝不递归清理根目录外的内容。
    $cleanupErrors = New-Object System.Collections.Generic.List[string]
    if ($createdVm -and (Get-VM -Name $PlannedVmName -ErrorAction SilentlyContinue)) {
        try { Stop-VM -Name $PlannedVmName -TurnOff -Force -ErrorAction SilentlyContinue | Out-Null } catch { $cleanupErrors.Add("stop_vm: $($_.Exception.Message)") }
        try { Remove-VM -Name $PlannedVmName -Force -ErrorAction Stop } catch { $cleanupErrors.Add("remove_vm: $($_.Exception.Message)") }
    }
    foreach ($path in @($childVhdPath, $baseVhdPath, $vmPath)) {
        if ([string]::IsNullOrWhiteSpace($path) -or -not (Test-Path -LiteralPath $path)) { continue }
        try {
            Assert-NoReparsePoint -Path $path
            $item = Get-Item -LiteralPath $path -Force
            if ($item.PSIsContainer) { Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop }
            else { Remove-Item -LiteralPath $path -Force -ErrorAction Stop }
        }
        catch { $cleanupErrors.Add("cleanup ${path}: $($_.Exception.Message)") }
    }
    return @($cleanupErrors)
}

$root = Get-FullPath -Path $DedicatedRoot
$fixture = Get-FullPath -Path $HarmlessFixtureIso
$preflightReceipt = Get-FullPath -Path $PreflightReceiptPath
$receipt = Get-FullPath -Path $ReceiptPath
$baselineVmwpIds = Get-ProcessIdsByName -Name 'vmwp'

try {
    Assert-ElevatedAdministrator
    Assert-EmptyDedicatedRoot -Root $root
    if (Test-Path -LiteralPath $preflightReceipt) { throw 'Preflight receipt path must be new for the same-process preflight.' }
    if (Test-Path -LiteralPath $receipt) { throw 'Acceptance receipt path must be new.' }
    if (-not (Test-Path -LiteralPath $fixture -PathType Leaf)) { throw 'Harmless fixture ISO is missing.' }
    Assert-NoReparsePoint -Path $fixture
    if ((Get-Sha256 -Path $fixture) -ne $HarmlessFixtureIsoSha256.ToLowerInvariant()) { throw 'Harmless fixture ISO hash mismatches before boot.' }

    # 在同一提升权限进程中重新预检，旧收据不能单独授权任何 VM 操作。
    $preflight = Join-Path $PSScriptRoot 'Invoke-Loop171HyperVPreflight.ps1'
    if (-not (Test-Path -LiteralPath $preflight -PathType Leaf)) { throw 'Loop171 preflight script is missing.' }
    & $preflight -BaseImageArchive $BaseImageArchive -LinuxCapaArchive $LinuxCapaArchive -DedicatedRoot $root -PlannedVmName $PlannedVmName -ReceiptPath $preflightReceipt -MinimumAvailableMemoryBytes $MinimumAvailableMemoryBytes
    if ($LASTEXITCODE -ne 0) { throw 'Same-process preflight failed.' }
    $preflightHash = Get-Sha256 -Path $preflightReceipt
    $preflightPayload = Get-Content -LiteralPath $preflightReceipt -Raw | ConvertFrom-Json
    if ($preflightPayload.decision -ne 'preflight_pass_no_vm_or_sample_action_authorized') { throw 'Same-process preflight did not pass.' }
    $details['preflight'] = [ordered]@{ receipt_sha256 = $preflightHash; decision = $preflightPayload.decision }

    Set-Gate -Name 'fixture_hash_before_boot' -Check { [PSCustomObject]@{ sha256 = Get-Sha256 -Path $fixture; bytes = (Get-Item -LiteralPath $fixture).Length } }
    if (-not $gates['fixture_hash_before_boot']) { throw 'Fixture hash gate failed.' }

    $baseDir = Join-Path $root 'base'
    $vmPath = Join-Path $root 'vm'
    $childVhdPath = Join-Path $root 'loop171-child.vhdx'
    New-Item -ItemType Directory -Path $baseDir -ErrorAction Stop | Out-Null
    New-Item -ItemType Directory -Path $vmPath -ErrorAction Stop | Out-Null
    $tar = Get-Command -Name tar.exe -CommandType Application -ErrorAction Stop
    & $tar.Source '-xzf' $BaseImageArchive '-C' $baseDir
    if ($LASTEXITCODE -ne 0) { throw 'Verified Ubuntu archive extraction failed.' }
    $baseVhds = @(Get-ChildItem -LiteralPath $baseDir -File -Filter '*.vhd' -Recurse)
    if ($baseVhds.Count -ne 1) { throw 'Verified Ubuntu archive must expand to exactly one VHD.' }
    $baseVhdPath = $baseVhds[0].FullName
    Assert-NoReparsePoint -Path $baseVhdPath
    $baseVhdHashBefore = Get-Sha256 -Path $baseVhdPath
    New-VHD -Path $childVhdPath -ParentPath $baseVhdPath -Differencing -ErrorAction Stop | Out-Null
    $createdChildVhd = $true

    New-VM -Name $PlannedVmName -Generation 2 -Path $vmPath -VHDPath $childVhdPath -MemoryStartupBytes 4GB -ErrorAction Stop | Out-Null
    $createdVm = $true
    Set-VM -Name $PlannedVmName -CheckpointType Disabled -AutomaticStartAction Nothing -AutomaticStopAction TurnOff -ErrorAction Stop | Out-Null
    $networkAdapters = @(Get-VMNetworkAdapter -VMName $PlannedVmName -ErrorAction Stop)
    foreach ($adapter in $networkAdapters) { Remove-VMNetworkAdapter -VMName $PlannedVmName -Name $adapter.Name -ErrorAction Stop }
    if (@(Get-VMNetworkAdapter -VMName $PlannedVmName -ErrorAction Stop).Count -ne 0) { throw 'VM retains a network adapter.' }
    $integrationServices = @(Get-VMIntegrationService -VMName $PlannedVmName -ErrorAction Stop)
    foreach ($service in $integrationServices) { Disable-VMIntegrationService -VMName $PlannedVmName -Name $service.Name -ErrorAction Stop }
    if (@(Get-VMIntegrationService -VMName $PlannedVmName -ErrorAction Stop | Where-Object { $_.Enabled }).Count -ne 0) { throw 'VM retains an enabled integration service.' }
    Add-VMDvdDrive -VMName $PlannedVmName -Path $fixture -ErrorAction Stop | Out-Null
    if (@(Get-VMDvdDrive -VMName $PlannedVmName -ErrorAction Stop | Where-Object { $_.Path -eq $fixture }).Count -ne 1) { throw 'Fixture ISO attachment is missing or ambiguous.' }

    $pipeName = "AxonLoop171-$([Guid]::NewGuid().ToString('N'))"
    $pipe = [System.IO.Pipes.NamedPipeServerStream]::new($pipeName, [System.IO.Pipes.PipeDirection]::In, 1, [System.IO.Pipes.PipeTransmissionMode]::Byte, [System.IO.Pipes.PipeOptions]::Asynchronous)
    Set-VMComPort -VMName $PlannedVmName -Number 1 -Path "\\.\pipe\$pipeName" -ErrorAction Stop
    Start-VM -Name $PlannedVmName -ErrorAction Stop | Out-Null
    $connection = $pipe.WaitForConnectionAsync()
    if (-not $connection.Wait($BootTimeoutSeconds * 1000)) { throw 'Guest did not connect to the one-way serial receipt pipe before timeout.' }
    $buffer = New-Object byte[] ($MaxReceiptBytes + 1)
    $read = $pipe.ReadAsync($buffer, 0, $buffer.Length)
    if (-not $read.Wait($BootTimeoutSeconds * 1000)) { throw 'Guest serial receipt did not finish before timeout.' }
    $receivedBytes = $read.Result
    if ($receivedBytes -le 0 -or $receivedBytes -gt $MaxReceiptBytes) { throw 'Guest serial receipt violates the aggregate byte limit.' }
    $serialText = [System.Text.Encoding]::UTF8.GetString($buffer, 0, $receivedBytes).Trim()
    $guest = $serialText | ConvertFrom-Json
    if ($guest.schema -ne $FixtureSchema -or $guest.readonly_media -ne $true -or $guest.write_attempt_blocked -ne $true -or $guest.no_default_route -ne $true -or $guest.only_loopback -ne $true) { throw 'Guest harmless-fixture receipt does not prove the required isolation conditions.' }
    $details['guest_receipt'] = [ordered]@{ schema = $guest.schema; bytes = $receivedBytes; readonly_media = $guest.readonly_media; write_attempt_blocked = $guest.write_attempt_blocked; no_default_route = $guest.no_default_route; only_loopback = $guest.only_loopback }
    $details['vm_before_teardown'] = [ordered]@{ network_adapter_count = @(Get-VMNetworkAdapter -VMName $PlannedVmName).Count; enabled_integration_service_count = @(Get-VMIntegrationService -VMName $PlannedVmName | Where-Object { $_.Enabled }).Count }
    if ((Get-Sha256 -Path $fixture) -ne $HarmlessFixtureIsoSha256.ToLowerInvariant()) { throw 'Harmless fixture ISO changed during boot.' }
    if ((Get-Sha256 -Path $baseVhdPath) -ne $baseVhdHashBefore) { throw 'Base VHD changed during boot.' }
    $decision = 'harmless_fixture_acceptance_pass_no_sample_parser_training_heldout_or_f1_action_authorized'
}
catch {
    $errors.Add($_.Exception.Message)
}
finally {
    if ($null -ne $pipe) { $pipe.Dispose() }
    $cleanupErrors = Remove-RegisteredAssets
    foreach ($cleanupError in $cleanupErrors) { $errors.Add($cleanupError) }
    $survivingVm = Get-VM -Name $PlannedVmName -ErrorAction SilentlyContinue
    $newVmwpIds = @(Get-ProcessIdsByName -Name 'vmwp' | Where-Object { $baselineVmwpIds -notcontains $_ })
    $details['teardown'] = [ordered]@{ vm_survives = ($null -ne $survivingVm); new_vmwp_process_ids = $newVmwpIds; dedicated_root_empty = (@(Get-ChildItem -LiteralPath $root -Force -ErrorAction SilentlyContinue).Count -eq 0) }
    if ($null -ne $survivingVm -or $newVmwpIds.Count -ne 0 -or -not $details['teardown'].dedicated_root_empty -or $cleanupErrors.Count -ne 0) { $decision = 'harmless_fixture_acceptance_blocked_fail_closed' }
    $payload = [ordered]@{ schema = $ReceiptSchema; loop_id = $LoopId; created_at_utc = [DateTime]::UtcNow.ToString('o'); claim_scope = 'harmless_fixture_vm_isolation_only_not_sample_parser_training_heldout_or_f1_evidence'; preflight_receipt_sha256 = $(if (Test-Path -LiteralPath $preflightReceipt) { Get-Sha256 -Path $preflightReceipt } else { $null }); fixture_iso_sha256 = $HarmlessFixtureIsoSha256.ToLowerInvariant(); gates = $gates; details = $details; errors = @($errors); hard_boundaries = [ordered]@{ sample_access_allowed = $false; parser_execution_allowed = $false; training_allowed = $false; heldout_allowed = $false; f1_claim_allowed = $false }; decision = $decision }
    Write-Receipt -Path $receipt -Payload $payload
}

$payload | ConvertTo-Json -Depth 10
if ($decision -ne 'harmless_fixture_acceptance_pass_no_sample_parser_training_heldout_or_f1_action_authorized') { exit 2 }
