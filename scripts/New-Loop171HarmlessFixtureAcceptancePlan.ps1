[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$PreflightReceipt,
    [Parameter(Mandatory = $true)][ValidatePattern('^[a-f0-9]{64}$')][string]$PreflightReceiptSha256,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$DedicatedRoot,
    [Parameter(Mandatory = $true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')][string]$PlannedVmName,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$HarmlessFixtureIso,
    [Parameter(Mandatory = $true)][ValidatePattern('^[a-f0-9]{64}$')][string]$HarmlessFixtureIsoSha256,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ReceiptPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$LoopId = 'loop171_hyperv_isolation'
$PlanSchema = 'axon_loop171_harmless_fixture_acceptance_plan_v1'
$PreflightSchema = 'axon_loop171_hyperv_preflight_v1'
$PreflightDecision = 'preflight_pass_no_vm_or_sample_action_authorized'
$RequiredGates = @(
    'elevated_token', 'available_memory_minimum', 'official_base_archive',
    'official_linux_capa_archive', 'protected_dedicated_root', 'root_separation', 'hyperv_zero_reuse'
)

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $volumeRoot = [System.IO.Path]::GetPathRoot($fullPath)
    if ($fullPath.Equals($volumeRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Filesystem-root paths are forbidden for Loop171 acceptance planning.'
    }
    return $fullPath.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
}

function Assert-NoReparsePoint {
    param([Parameter(Mandatory = $true)][string]$Path)
    $current = Get-Item -LiteralPath $Path -Force
    while ($true) {
        if (($current.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse-point paths are forbidden: $($current.FullName)"
        }
        $parent = if ($current -is [System.IO.DirectoryInfo]) { $current.Parent } else { $current.Directory }
        if ($null -eq $parent -or $parent.FullName -eq $current.FullName) { return }
        $current = $parent
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Receipt {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)]$Payload)
    if (Test-Path -LiteralPath $Path) { throw 'Acceptance-plan receipt overwrite is forbidden.' }
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) { throw 'Receipt parent must already exist.' }
    [System.IO.File]::WriteAllText($Path, ($Payload | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
}

$gates = [ordered]@{}
$errors = New-Object System.Collections.Generic.List[string]
function Set-Gate {
    param([Parameter(Mandatory = $true)][string]$Name, [Parameter(Mandatory = $true)][scriptblock]$Check)
    try { & $Check; $gates[$Name] = $true }
    catch { $gates[$Name] = $false; $errors.Add("${Name}: $($_.Exception.Message)") }
}

$preflightPath = Get-FullPath -Path $PreflightReceipt
$rootPath = Get-FullPath -Path $DedicatedRoot
$fixturePath = Get-FullPath -Path $HarmlessFixtureIso
$receiptFullPath = Get-FullPath -Path $ReceiptPath

Set-Gate -Name 'preflight_receipt_sha256_bound' -Check {
    Assert-NoReparsePoint -Path $preflightPath
    if ((Get-Sha256 -Path $preflightPath) -ne $PreflightReceiptSha256.ToLowerInvariant()) { throw 'Preflight receipt SHA-256 binding mismatches.' }
}
Set-Gate -Name 'preflight_receipt_passes_all_required_gates' -Check {
    $preflight = Get-Content -LiteralPath $preflightPath -Raw | ConvertFrom-Json
    if ($preflight.schema -ne $PreflightSchema -or $preflight.loop_id -ne $LoopId -or $preflight.decision -ne $PreflightDecision) { throw 'Preflight receipt is not an exact passing Loop171 receipt.' }
    if ($preflight.planned_vm.name -ne $PlannedVmName -or $preflight.planned_vm.network_adapter_count_required -ne 0) { throw 'Preflight planned VM binding mismatches.' }
    foreach ($name in $RequiredGates) { if ($preflight.gates.$name -ne $true) { throw "Preflight gate is not true: $name" } }
    if ($preflight.hard_boundaries.sample_access_allowed -ne $false -or $preflight.hard_boundaries.training_allowed -ne $false) { throw 'Preflight receipt improperly grants sample or training authority.' }
}
Set-Gate -Name 'new_empty_protected_root' -Check {
    if (-not (Test-Path -LiteralPath $rootPath -PathType Container)) { throw 'Dedicated root does not exist.' }
    Assert-NoReparsePoint -Path $rootPath
    if (@(Get-ChildItem -LiteralPath $rootPath -Force).Count -ne 0) { throw 'Dedicated root must remain empty until a future acceptance executor materializes only new assets.' }
}
Set-Gate -Name 'harmless_fixture_is_sha256_bound' -Check {
    if (-not (Test-Path -LiteralPath $fixturePath -PathType Leaf) -or [System.IO.Path]::GetExtension($fixturePath) -ne '.iso') { throw 'A regular harmless fixture ISO is required.' }
    Assert-NoReparsePoint -Path $fixturePath
    if ((Get-Sha256 -Path $fixturePath) -ne $HarmlessFixtureIsoSha256.ToLowerInvariant()) { throw 'Harmless fixture ISO SHA-256 binding mismatches.' }
}
Set-Gate -Name 'receipt_is_external_and_new' -Check {
    if (Test-Path -LiteralPath $receiptFullPath) { throw 'Acceptance-plan receipt overwrite is forbidden.' }
    if ($receiptFullPath.StartsWith($rootPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Receipt must be outside the disposable root.' }
}

$passed = @($gates.Values) -notcontains $false -and $gates.Count -eq 5
$payload = [ordered]@{
    schema = $PlanSchema
    loop_id = $LoopId
    claim_scope = 'static_future_harmless_vm_acceptance_plan_not_vm_sample_parser_training_heldout_or_f1_evidence'
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    activation = [ordered]@{ preflight_receipt_sha256 = $PreflightReceiptSha256; gates = $gates; errors = @($errors) }
    future_execution = [ordered]@{
        new_root_only = $true; vm_generation = 2; network_adapter_count = 0; switch_reuse_forbidden = $true
        all_integration_channels_disabled = $true; enhanced_session_disabled = $true
        fixture_input_attachment = 'immutable_read_only_no_sample_iso'; fixture_hash_must_match_before_and_after = $true
        output_policy = [ordered]@{ aggregate_only = $true; max_bytes = 65536; raw_sample_or_parser_output_persisted = $false }
        termination_policy = [ordered]@{ forced_guest_termination = $true; no_new_vm_or_vmwp_survivor = $true; teardown_receipt_required = $true }
    }
    hard_boundaries = [ordered]@{ creates_or_starts_vm = $false; creates_or_mounts_disks = $false; accesses_samples = $false; executes_parsers = $false; sample_access_allowed = $false; parser_execution_allowed = $false; training_allowed = $false; heldout_allowed = $false; f1_claim_allowed = $false }
    decision = $(if ($passed) { 'acceptance_plan_ready_no_vm_or_sample_action_authorized' } else { 'acceptance_plan_blocked_fail_closed' })
}
Write-Receipt -Path $receiptFullPath -Payload $payload
$payload | ConvertTo-Json -Depth 8
if (-not $passed) { exit 2 }
