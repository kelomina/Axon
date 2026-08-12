[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$BaseImageArchive,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$LinuxCapaArchive,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$DedicatedRoot,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')]
    [string]$PlannedVmName,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ReceiptPath,

    [string[]]$AllowedRootPrincipalSid = @(),

    [UInt64]$MinimumAvailableMemoryBytes = 13958643712
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$LoopId = 'loop171_hyperv_isolation'
$ReceiptSchema = 'axon_loop171_hyperv_preflight_v1'
$OfficialArchiveName = 'ubuntu-24.04-server-cloudimg-amd64-azure.vhd.tar.gz'
$OfficialArchiveSha256 = '05b7b5bb6172e5b0dd1248d5598c1bc27927c4625ba4c09c0442d4751725c43f'
$LinuxCapaArchiveName = 'capa-v9.4.0-linux.zip'
$LinuxCapaArchiveSha256 = '07800a1d20a21eb18fc98716e2ae81b668e0c9a04defd588c8aa17ea3d3281e4'

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $volumeRoot = [System.IO.Path]::GetPathRoot($fullPath)
    if ($fullPath.Equals($volumeRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Filesystem-root paths are forbidden for Loop171 isolation.'
    }
    return $fullPath.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}

function Test-PathIntersection {
    param(
        [Parameter(Mandatory = $true)][string]$Left,
        [Parameter(Mandatory = $true)][string]$Right
    )

    $leftFull = Get-FullPath -Path $Left
    $rightFull = Get-FullPath -Path $Right
    if ($leftFull.Equals($rightFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $separator = [System.IO.Path]::DirectorySeparatorChar
    return $leftFull.StartsWith("$rightFull$separator", [System.StringComparison]::OrdinalIgnoreCase) -or
        $rightFull.StartsWith("$leftFull$separator", [System.StringComparison]::OrdinalIgnoreCase)
}

function Assert-NoReparsePoint {
    param([Parameter(Mandatory = $true)][string]$Path)

    $current = Get-Item -LiteralPath $Path -Force
    while ($true) {
        if (($current.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse-point paths are forbidden: $($current.FullName)"
        }
        $parent = if ($current -is [System.IO.DirectoryInfo]) { $current.Parent } else { $current.Directory }
        if ($null -eq $parent -or $parent.FullName -eq $current.FullName) {
            return
        }
        $current = $parent
    }
}

function Convert-IdentityToSid {
    param([Parameter(Mandatory = $true)]$Identity)

    try {
        if ($Identity -is [string]) {
            if ($Identity -match '^S-1-') {
                return [System.Security.Principal.SecurityIdentifier]::new($Identity).Value
            }
            $Identity = [System.Security.Principal.NTAccount]::new($Identity)
        }
        return $Identity.Translate([System.Security.Principal.SecurityIdentifier]).Value
    }
    catch {
        throw "Unable to resolve ACL identity to a SID: $Identity"
    }
}

function Get-CurrentIdentity {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
    return [PSCustomObject]@{
        elevated = $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
        sid = $identity.User.Value
    }
}

function Test-ProtectedDedicatedRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string[]]$AllowedSids
    )

    # The elevated setup creates this root; preflight only audits it.
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        throw 'Dedicated root does not exist as a directory; this preflight never creates it.'
    }
    Assert-NoReparsePoint -Path $Root
    $children = @(Get-ChildItem -LiteralPath $Root -Force)
    if ($children.Count -ne 0) {
        throw 'Dedicated root is not empty, so it cannot prove zero reuse.'
    }

    $acl = Get-Acl -LiteralPath $Root
    if (-not $acl.AreAccessRulesProtected) {
        throw 'Dedicated root ACL inherits parent permissions.'
    }
    $ownerSid = Convert-IdentityToSid -Identity $acl.Owner
    if ($AllowedSids -notcontains $ownerSid) {
        throw 'Dedicated root owner is outside the explicit protected-root allowlist.'
    }

    $allowRules = @($acl.Access | Where-Object { $_.AccessControlType -eq [System.Security.AccessControl.AccessControlType]::Allow })
    if ($allowRules.Count -eq 0) {
        throw 'Dedicated root has no explicit allow rule.'
    }
    foreach ($rule in $acl.Access) {
        if ($rule.IsInherited) {
            throw 'Dedicated root contains an inherited ACL rule.'
        }
        if ($rule.AccessControlType -eq [System.Security.AccessControl.AccessControlType]::Deny) {
            throw 'Dedicated root contains a deny rule; use an allowlist-only ACL for auditable isolation.'
        }
        $sid = Convert-IdentityToSid -Identity $rule.IdentityReference
        if ($AllowedSids -notcontains $sid) {
            throw "Dedicated root grants access outside the protected-root allowlist: $sid"
        }
    }
    return [PSCustomObject]@{
        owner_sid = $ownerSid
        allow_rule_count = $allowRules.Count
        acl_protected = $acl.AreAccessRulesProtected
    }
}

function Get-ExistingHyperVAssetPaths {
    # Inventory existing assets for zero-reuse comparison without selecting a switch.
    $getVm = Get-Command -Name Get-VM -ErrorAction Stop
    $getVmDisk = Get-Command -Name Get-VMHardDiskDrive -ErrorAction Stop
    $getVmHost = Get-Command -Name Get-VMHost -ErrorAction Stop
    $getVmSwitch = Get-Command -Name Get-VMSwitch -ErrorAction Stop
    if ($null -eq $getVm -or $null -eq $getVmDisk -or $null -eq $getVmHost -or $null -eq $getVmSwitch) {
        throw 'Required Hyper-V read-only cmdlets are unavailable.'
    }

    $vms = @(Get-VM -ErrorAction Stop)
    $paths = New-Object System.Collections.Generic.List[string]
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($vm in $vms) {
        $names.Add([string]$vm.Name)
        foreach ($property in @('Path', 'SnapshotFileLocation', 'SmartPagingFilePath')) {
            $value = $vm.$property
            if (-not [string]::IsNullOrWhiteSpace([string]$value)) {
                $paths.Add((Get-FullPath -Path ([string]$value)))
            }
        }
        foreach ($disk in @(Get-VMHardDiskDrive -VMName $vm.Name -ErrorAction Stop)) {
            if (-not [string]::IsNullOrWhiteSpace([string]$disk.Path)) {
                $paths.Add((Get-FullPath -Path ([string]$disk.Path)))
            }
        }
    }
    $vmHost = Get-VMHost -ErrorAction Stop
    foreach ($property in @('VirtualMachinePath', 'VirtualHardDiskPath')) {
        $value = $vmHost.$property
        if (-not [string]::IsNullOrWhiteSpace([string]$value)) {
            $paths.Add((Get-FullPath -Path ([string]$value)))
        }
    }
    $switches = @(Get-VMSwitch -ErrorAction Stop)
    $pathSet = @($paths | Sort-Object -Unique)
    $nameSet = @($names | Sort-Object -Unique)
    $commitmentInput = @($pathSet + $nameSet + @($switches.Name | Sort-Object)) -join "`n"
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $commitment = ([System.BitConverter]::ToString($sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($commitmentInput))) -replace '-', '').ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
    return [PSCustomObject]@{
        vm_names = $nameSet
        asset_paths = $pathSet
        vm_count = $vms.Count
        switch_count = $switches.Count
        inventory_sha256 = $commitment
    }
}

function Write-Receipt {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Payload
    )

    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        throw 'Receipt parent directory must already exist; this preflight never creates directories.'
    }
    [System.IO.File]::WriteAllText(
        $Path,
        ($Payload | ConvertTo-Json -Depth 8),
        [System.Text.UTF8Encoding]::new($false)
    )
}

$gates = [ordered]@{}
$errors = New-Object System.Collections.Generic.List[string]
$details = [ordered]@{}
function Set-Gate {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Check
    )

    try {
        $details[$Name] = & $Check
        $gates[$Name] = $true
    }
    catch {
        $gates[$Name] = $false
        $errors.Add("${Name}: $($_.Exception.Message)")
    }
}

$receiptFullPath = Get-FullPath -Path $ReceiptPath
$receiptParent = Split-Path -Parent $receiptFullPath
if (-not (Test-Path -LiteralPath $receiptParent -PathType Container)) {
    throw 'Receipt parent directory must exist before preflight starts.'
}

$identity = Get-CurrentIdentity
$allowedSids = @('S-1-5-18', 'S-1-5-32-544', $identity.sid) + $AllowedRootPrincipalSid
foreach ($sid in @($allowedSids | Sort-Object -Unique)) {
    try {
        [void][System.Security.Principal.SecurityIdentifier]::new($sid)
    }
    catch {
        throw "AllowedRootPrincipalSid contains an invalid SID: $sid"
    }
}
$allowedSids = @($allowedSids | Sort-Object -Unique)

Set-Gate -Name 'elevated_token' -Check {
    if (-not $identity.elevated) {
        throw 'An elevated Windows administrator token is required.'
    }
    return [PSCustomObject]@{ current_sid = $identity.sid }
}

Set-Gate -Name 'available_memory_minimum' -Check {
    $operatingSystem = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
    $availableBytes = [UInt64]$operatingSystem.FreePhysicalMemory * 1KB
    if ($availableBytes -lt $MinimumAvailableMemoryBytes) {
        throw "Available physical memory is below the required $MinimumAvailableMemoryBytes bytes."
    }
    return [PSCustomObject]@{ available_bytes = $availableBytes; minimum_bytes = $MinimumAvailableMemoryBytes }
}

Set-Gate -Name 'official_base_archive' -Check {
    if (-not (Test-Path -LiteralPath $BaseImageArchive -PathType Leaf)) {
        throw 'Official base-image archive does not exist as a regular file.'
    }
    $archive = Get-Item -LiteralPath $BaseImageArchive -Force
    Assert-NoReparsePoint -Path $archive.FullName
    if ($archive.Name -ne $OfficialArchiveName) {
        throw "Unexpected base-image archive name: $($archive.Name)"
    }
    $actualSha256 = (Get-FileHash -LiteralPath $archive.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $OfficialArchiveSha256) {
        throw 'Official base-image SHA-256 does not match the signed release checksum.'
    }
    return [PSCustomObject]@{ archive_name = $archive.Name; sha256 = $actualSha256; bytes = $archive.Length }
}

Set-Gate -Name 'official_linux_capa_archive' -Check {
    # An Ubuntu guest requires this Linux asset instead of the closed host Windows route.
    if (-not (Test-Path -LiteralPath $LinuxCapaArchive -PathType Leaf)) {
        throw 'Linux capa archive does not exist as a regular file.'
    }
    $archive = Get-Item -LiteralPath $LinuxCapaArchive -Force
    Assert-NoReparsePoint -Path $archive.FullName
    if ($archive.Name -ne $LinuxCapaArchiveName) {
        throw "Unexpected Linux capa archive name: $($archive.Name)"
    }
    $actualSha256 = (Get-FileHash -LiteralPath $archive.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $LinuxCapaArchiveSha256) {
        throw 'Linux capa archive SHA-256 does not match the official v9.4.0 release digest.'
    }
    return [PSCustomObject]@{ archive_name = $archive.Name; sha256 = $actualSha256; bytes = $archive.Length }
}

$rootFullPath = $null
Set-Gate -Name 'protected_dedicated_root' -Check {
    $rootFullPath = Get-FullPath -Path $DedicatedRoot
    return Test-ProtectedDedicatedRoot -Root $rootFullPath -AllowedSids $allowedSids
}

Set-Gate -Name 'root_separation' -Check {
    $root = Get-FullPath -Path $DedicatedRoot
    $archive = Get-FullPath -Path $BaseImageArchive
    if (Test-PathIntersection -Left $root -Right $archive) {
        throw 'Dedicated root intersects the staged base-image archive.'
    }
    if (Test-PathIntersection -Left $root -Right $receiptFullPath) {
        throw 'Dedicated root intersects the receipt path; keep evidence outside the disposable root.'
    }
    $linuxCapa = Get-FullPath -Path $LinuxCapaArchive
    if (Test-PathIntersection -Left $root -Right $linuxCapa) {
        throw 'Dedicated root intersects the staged Linux capa archive.'
    }
    return [PSCustomObject]@{ separate_from_base_archive = $true; separate_from_linux_capa = $true; separate_from_receipt = $true }
}

$hyperVInventory = $null
Set-Gate -Name 'hyperv_zero_reuse' -Check {
    # The planned name and root must be disjoint from existing and default storage.
    $hyperVInventory = Get-ExistingHyperVAssetPaths
    if ($hyperVInventory.vm_names -contains $PlannedVmName) {
        throw 'Planned VM name already exists and therefore cannot be reused.'
    }
    $root = Get-FullPath -Path $DedicatedRoot
    foreach ($assetPath in $hyperVInventory.asset_paths) {
        if (Test-PathIntersection -Left $root -Right $assetPath) {
            throw 'Dedicated root intersects a pre-existing Hyper-V asset or default Hyper-V storage location.'
        }
    }
    return [PSCustomObject]@{
        planned_vm_name_unused = $true
        existing_vm_count = $hyperVInventory.vm_count
        existing_switch_count = $hyperVInventory.switch_count
        existing_inventory_sha256 = $hyperVInventory.inventory_sha256
        selected_switch = $null
        selected_network_adapter_count = 0
    }
}

$allPassed = @($gates.Values) -notcontains $false -and $gates.Count -eq 6
$payload = [ordered]@{
    schema = $ReceiptSchema
    loop_id = $LoopId
    claim_scope = 'read_only_hyperv_environment_preflight_not_vm_sample_parser_training_or_f1_evidence'
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    official_base_image = [ordered]@{
        archive_name = $OfficialArchiveName
        sha256 = $OfficialArchiveSha256
        source = 'https://cloud-images.ubuntu.com/releases/noble/release/ubuntu-24.04-server-cloudimg-amd64-azure.vhd.tar.gz'
    }
    official_linux_capa = [ordered]@{
        archive_name = $LinuxCapaArchiveName
        sha256 = $LinuxCapaArchiveSha256
        source = 'https://github.com/mandiant/capa/releases/download/v9.4.0/capa-v9.4.0-linux.zip'
        execution_environment = 'future_disposable_ubuntu_guest_only'
    }
    planned_vm = [ordered]@{
        name = $PlannedVmName
        network_adapter_count_required = 0
        switch_reuse_forbidden = $true
    }
    hard_boundaries = [ordered]@{
        creates_or_mutates_hyperv_resources = $false
        creates_or_mounts_disks = $false
        accesses_samples = $false
        executes_parsers = $false
        sample_access_allowed = $false
        training_allowed = $false
        heldout_allowed = $false
        f1_claim_allowed = $false
    }
    gates = $gates
    details = $details
    errors = @($errors)
    decision = $(if ($allPassed) { 'preflight_pass_no_vm_or_sample_action_authorized' } else { 'preflight_blocked_fail_closed' })
}

Write-Receipt -Path $receiptFullPath -Payload $payload
$payload | ConvertTo-Json -Depth 8
if (-not $allPassed) {
    exit 2
}
