[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$DedicatedParent,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')]
    [string]$RootDirectoryName,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ReceiptPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$LoopId = 'loop171_hyperv_isolation'
$ReceiptSchema = 'axon_loop171_protected_root_creation_v1'
$SystemSid = 'S-1-5-18'
$AdministratorsSid = 'S-1-5-32-544'

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $volumeRoot = [System.IO.Path]::GetPathRoot($fullPath)
    if ($fullPath.Equals($volumeRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Filesystem-root paths are forbidden for Loop171 protected-root creation.'
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

function Get-CurrentElevatedIdentity {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'An elevated Windows administrator token is required before any directory is created.'
    }
    return [PSCustomObject]@{
        sid = $identity.User.Value
        elevated = $true
    }
}

function Get-MissingDirectoryChain {
    param([Parameter(Mandatory = $true)][string]$Target)

    $missing = New-Object System.Collections.Generic.List[string]
    $current = $Target
    while (-not (Test-Path -LiteralPath $current)) {
        $missing.Add($current)
        $parent = Split-Path -Parent $current
        if ([string]::IsNullOrWhiteSpace($parent) -or
            $parent.Equals($current, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Unable to locate an existing non-root ancestor for the dedicated parent.'
        }
        $current = $parent
    }

    $anchor = Get-Item -LiteralPath $current -Force
    if ($anchor -isnot [System.IO.DirectoryInfo]) {
        throw 'The first existing ancestor is not a directory.'
    }
    Assert-NoReparsePoint -Path $anchor.FullName
    $ordered = @($missing.ToArray())
    [array]::Reverse($ordered)
    return [PSCustomObject]@{
        anchor = $anchor.FullName
        directories_to_create = $ordered
    }
}

function Set-AllowlistDirectoryAcl {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$AllowedSids,
        [Parameter(Mandatory = $true)][string]$OwnerSid
    )

    $acl = [System.Security.AccessControl.DirectorySecurity]::new()
    $acl.SetAccessRuleProtection($true, $false)
    $owner = [System.Security.Principal.SecurityIdentifier]::new($OwnerSid)
    $acl.SetOwner($owner)
    $inheritance = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
        [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
    $propagation = [System.Security.AccessControl.PropagationFlags]::None
    $rights = [System.Security.AccessControl.FileSystemRights]::FullControl
    $allow = [System.Security.AccessControl.AccessControlType]::Allow
    foreach ($sidValue in $AllowedSids) {
        $sid = [System.Security.Principal.SecurityIdentifier]::new($sidValue)
        $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $sid,
            $rights,
            $inheritance,
            $propagation,
            $allow
        )
        [void]$acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
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

function Test-AllowlistDirectoryAcl {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$AllowedSids
    )

    $acl = Get-Acl -LiteralPath $Path
    if (-not $acl.AreAccessRulesProtected) {
        throw "ACL inheritance remains enabled: $Path"
    }
    $ownerSid = Convert-IdentityToSid -Identity $acl.Owner
    if ($AllowedSids -notcontains $ownerSid) {
        throw "Directory owner is outside the allowlist: $ownerSid"
    }
    $rules = @($acl.Access)
    if ($rules.Count -ne $AllowedSids.Count) {
        throw "Unexpected ACL rule count for allowlist-only directory: $Path"
    }
    $observedSids = New-Object System.Collections.Generic.List[string]
    foreach ($rule in $rules) {
        if ($rule.IsInherited) {
            throw "Inherited ACL rule remains on protected directory: $Path"
        }
        if ($rule.AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) {
            throw "Non-allow ACL rule is forbidden on protected directory: $Path"
        }
        if ($rule.FileSystemRights -ne [System.Security.AccessControl.FileSystemRights]::FullControl) {
            throw "Allowlist ACL rule is not FullControl: $Path"
        }
        $sid = Convert-IdentityToSid -Identity $rule.IdentityReference
        if ($AllowedSids -notcontains $sid) {
            throw "Directory grants access outside the allowlist: $sid"
        }
        $observedSids.Add($sid)
    }
    if ((@($observedSids | Sort-Object -Unique) -join ',') -ne (@($AllowedSids | Sort-Object) -join ',')) {
        throw "Directory ACL does not exactly match the required allowlist: $Path"
    }
    return [PSCustomObject]@{
        owner_sid = $ownerSid
        allow_rule_count = $rules.Count
        acl_protected = $acl.AreAccessRulesProtected
        allowlist_sids = @($AllowedSids | Sort-Object)
    }
}

function Write-Receipt {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Payload
    )

    [System.IO.File]::WriteAllText(
        $Path,
        ($Payload | ConvertTo-Json -Depth 8),
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Remove-CreatedDirectories {
    param([Parameter(Mandatory = $true)][string[]]$Paths)

    $cleanupErrors = New-Object System.Collections.Generic.List[string]
    $reversePaths = @($Paths)
    [array]::Reverse($reversePaths)
    foreach ($directoryPath in $reversePaths) {
        try {
            if (-not (Test-Path -LiteralPath $directoryPath)) {
                continue
            }
            $item = Get-Item -LiteralPath $directoryPath -Force
            if ($item -isnot [System.IO.DirectoryInfo]) {
                throw 'Created path is no longer a directory.'
            }
            if (-not $item.FullName.Equals($directoryPath, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'Created directory no longer resolves to its recorded path.'
            }
            Assert-NoReparsePoint -Path $item.FullName
            $children = @(Get-ChildItem -LiteralPath $item.FullName -Force)
            if ($children.Count -ne 0) {
                throw 'Created directory is not empty; recursive deletion is forbidden.'
            }
            Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop
            if (Test-Path -LiteralPath $directoryPath) {
                throw 'Created directory still exists after non-recursive cleanup.'
            }
        }
        catch {
            $cleanupErrors.Add("${directoryPath}: $($_.Exception.Message)")
        }
    }
    return $cleanupErrors.ToArray()
}

$gates = [ordered]@{}
$errors = New-Object System.Collections.Generic.List[string]
$details = [ordered]@{}
$createdPaths = New-Object System.Collections.Generic.List[string]
$identity = $null
$dedicatedParentFullPath = $null
$dedicatedRootFullPath = $null
$receiptFullPath = $null
$receiptWritable = $false
$decision = 'protected_root_blocked_fail_closed'

try {
    $dedicatedParentFullPath = Get-FullPath -Path $DedicatedParent
    $dedicatedRootFullPath = Join-Path -Path $dedicatedParentFullPath -ChildPath $RootDirectoryName
    $receiptFullPath = Get-FullPath -Path $ReceiptPath
    $receiptParent = Split-Path -Parent $receiptFullPath
    if (-not (Test-Path -LiteralPath $receiptParent -PathType Container)) {
        throw 'Receipt parent directory must already exist; protected-root setup never creates report paths.'
    }
    Assert-NoReparsePoint -Path $receiptParent
    if (Test-Path -LiteralPath $receiptFullPath) {
        throw 'Receipt path already exists; protected-root setup refuses to overwrite evidence.'
    }
    if (Test-PathIntersection -Left $dedicatedParentFullPath -Right $receiptFullPath) {
        throw 'Receipt path must be outside the new dedicated parent and root.'
    }
    $receiptWritable = $true
    $gates['receipt_path_safe'] = $true
    $details['receipt_path_safe'] = [PSCustomObject]@{ path = $receiptFullPath }

    $identity = Get-CurrentElevatedIdentity
    $gates['elevated_token'] = $true
    $details['elevated_token'] = [PSCustomObject]@{ current_sid = $identity.sid }

    if (Test-Path -LiteralPath $dedicatedParentFullPath) {
        throw 'Dedicated parent already exists; zero-reuse requires a new dedicated parent path.'
    }
    if (Test-Path -LiteralPath $dedicatedRootFullPath) {
        throw 'Dedicated root already exists; zero-reuse requires a new dedicated root path.'
    }
    $gates['target_paths_absent'] = $true
    $details['target_paths_absent'] = [PSCustomObject]@{
        dedicated_parent = $dedicatedParentFullPath
        dedicated_root = $dedicatedRootFullPath
    }

    $chain = Get-MissingDirectoryChain -Target $dedicatedParentFullPath
    $gates['existing_ancestor_no_reparse'] = $true
    $details['existing_ancestor_no_reparse'] = [PSCustomObject]@{ anchor = $chain.anchor }

    $allowedSids = @($SystemSid, $AdministratorsSid, $identity.sid | Sort-Object -Unique)
    foreach ($directoryPath in @($chain.directories_to_create) + @($dedicatedRootFullPath)) {
        if (Test-Path -LiteralPath $directoryPath) {
            throw "Refusing path created or reused during setup: $directoryPath"
        }
        [void](New-Item -ItemType Directory -Path $directoryPath -ErrorAction Stop)
        [void]$createdPaths.Add($directoryPath)
        Assert-NoReparsePoint -Path $directoryPath
        Set-AllowlistDirectoryAcl -Path $directoryPath -AllowedSids $allowedSids -OwnerSid $identity.sid
    }
    $gates['new_directories_created'] = $true
    $details['new_directories_created'] = [PSCustomObject]@{ paths = @($createdPaths) }

    $aclAudits = [ordered]@{}
    foreach ($directoryPath in @($createdPaths)) {
        $aclAudits[$directoryPath] = Test-AllowlistDirectoryAcl -Path $directoryPath -AllowedSids $allowedSids
    }
    $gates['allowlist_acl_applied'] = $true
    $details['allowlist_acl_applied'] = $aclAudits
    $decision = 'protected_root_created_no_vm_or_sample_action_authorized'
}
catch {
    $errors.Add($_.Exception.Message)
}

$cleanup = [ordered]@{
    attempted = $false
    completed = $true
    removed_paths = @()
    errors = @()
}
if ($decision -ne 'protected_root_created_no_vm_or_sample_action_authorized' -and $createdPaths.Count -gt 0) {
    $cleanup.attempted = $true
    $cleanupErrors = Remove-CreatedDirectories -Paths @($createdPaths)
    $cleanup.errors = @($cleanupErrors)
    $cleanup.completed = $cleanupErrors.Count -eq 0
    if ($cleanup.completed) {
        $cleanup.removed_paths = @($createdPaths | Sort-Object -Descending)
    }
    else {
        $errors.Add("Compensating cleanup failed; manual remediation is required before any Loop171 action: $($cleanupErrors -join ' | ')")
        $decision = 'protected_root_cleanup_incomplete_hard_error'
    }
}

$payload = [ordered]@{
    schema = $ReceiptSchema
    loop_id = $LoopId
    claim_scope = 'protected_directory_creation_only_not_vm_vhd_switch_mount_guest_sample_parser_training_or_f1_evidence'
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    dedicated_parent = $dedicatedParentFullPath
    dedicated_root = $dedicatedRootFullPath
    created_paths = @($createdPaths)
    required_allowlist_sids = $(if ($null -eq $identity) { @($SystemSid, $AdministratorsSid) } else { @($SystemSid, $AdministratorsSid, $identity.sid | Sort-Object -Unique) })
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
    cleanup = $cleanup
    errors = @($errors)
    decision = $decision
}

if ($receiptWritable) {
    Write-Receipt -Path $receiptFullPath -Payload $payload
}
$payload | ConvertTo-Json -Depth 8
if ($decision -ne 'protected_root_created_no_vm_or_sample_action_authorized') {
    exit 2
}
