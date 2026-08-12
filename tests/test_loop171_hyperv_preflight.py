from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Invoke-Loop171HyperVPreflight.ps1"
PROVISION_SCRIPT = ROOT / "scripts" / "Initialize-Loop171ProtectedRoot.ps1"
ELEVATION_LAUNCHER = ROOT / "scripts" / "Start-Loop171ProtectedRootElevation.ps1"
PREFLIGHT_ELEVATION_LAUNCHER = ROOT / "scripts" / "Start-Loop171HyperVPreflightElevation.ps1"
DOC = ROOT / "docs" / "loop171_hyperv_preflight.md"


def test_preflight_binds_the_frozen_ubuntu_and_linux_capa_digests() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "05b7b5bb6172e5b0dd1248d5598c1bc27927c4625ba4c09c0442d4751725c43f" in source
    assert "07800a1d20a21eb18fc98716e2ae81b668e0c9a04defd588c8aa17ea3d3281e4" in source
    assert "capa-v9.4.0-linux.zip" in source
    assert "MinimumAvailableMemoryBytes = 13958643712" in source


def test_preflight_has_only_read_only_hyperv_inventory_commands() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "New-VM",
        "Set-VM",
        "Start-VM",
        "Stop-VM",
        "Remove-VM",
        "New-VHD",
        "Mount-VHD",
        "Dismount-VHD",
        "New-VMSwitch",
        "Remove-VMSwitch",
        "Add-VMNetworkAdapter",
        "Set-VMNetworkAdapter",
        "Copy-VMFile",
        "Invoke-Command",
        "Enter-PSSession",
    )

    for command in forbidden:
        assert command not in source
    assert "Get-VM" in source
    assert "Get-VMHardDiskDrive" in source
    assert "Get-VMSwitch" in source
    assert "$vmHost = Get-VMHost" in source
    assert "$host = Get-VMHost" not in source
    assert "$current -is [System.IO.DirectoryInfo]" in source
    assert "$current.Directory" in source


def test_preflight_is_fail_closed_and_keeps_sample_authority_false() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "preflight_blocked_fail_closed" in source
    assert "sample_access_allowed = $false" in source
    assert "training_allowed = $false" in source
    assert "heldout_allowed = $false" in source
    assert "f1_claim_allowed = $false" in source
    assert "Dedicated root is not empty" in source
    assert "Dedicated root ACL inherits parent permissions" in source


def test_documentation_keeps_linux_capa_guest_only_and_defers_sample_access() -> None:
    text = DOC.read_text(encoding="utf-8")

    assert "Ubuntu guest cannot use the existing Windows" in text
    assert "does not authorize sample access" in text
    assert "preflight_blocked_fail_closed" in text


@pytest.mark.parametrize("script", (SCRIPT, PROVISION_SCRIPT, ELEVATION_LAUNCHER, PREFLIGHT_ELEVATION_LAUNCHER))
def test_powershell_parser_accepts_the_script_when_windows_powershell_is_available(script: Path) -> None:
    executable = Path(os.environ.get("SystemRoot", r"C:\\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not executable.exists():
        pytest.skip("Windows PowerShell is unavailable in this environment")

    command = (
        "$tokens=$null; $errors=$null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{script}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -ne 0) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    result = subprocess.run(
        [str(executable), "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_protected_root_provisioner_requires_elevation_and_new_paths() -> None:
    source = PROVISION_SCRIPT.read_text(encoding="utf-8")

    assert "An elevated Windows administrator token is required before any directory is created." in source
    assert "Dedicated parent already exists; zero-reuse requires a new dedicated parent path." in source
    assert "Dedicated root already exists; zero-reuse requires a new dedicated root path." in source
    assert "Refusing path created or reused during setup" in source
    assert "Reparse-point paths are forbidden" in source
    assert "Receipt path already exists; protected-root setup refuses to overwrite evidence." in source


def test_protected_root_provisioner_compensates_only_its_new_empty_directories() -> None:
    source = PROVISION_SCRIPT.read_text(encoding="utf-8")

    assert "function Remove-CreatedDirectories" in source
    assert "[array]::Reverse($reversePaths)" in source
    assert "Created directory is not empty; recursive deletion is forbidden." in source
    assert "Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop" in source
    assert "-Recurse" not in source
    assert "protected_root_cleanup_incomplete_hard_error" in source
    assert "Compensating cleanup failed; manual remediation is required" in source
    assert "[void]$createdPaths.Add($directoryPath)" in source


def test_protected_root_provisioner_applies_exact_system_admin_caller_allowlist() -> None:
    source = PROVISION_SCRIPT.read_text(encoding="utf-8")

    assert "$SystemSid = 'S-1-5-18'" in source
    assert "$AdministratorsSid = 'S-1-5-32-544'" in source
    assert "$allowedSids = @($SystemSid, $AdministratorsSid, $identity.sid | Sort-Object -Unique)" in source
    assert "$acl.SetAccessRuleProtection($true, $false)" in source
    assert "$acl.SetOwner($owner)" in source
    assert "Directory ACL does not exactly match the required allowlist" in source
    assert "allowlist_acl_applied" in source
    assert "New-Item -ItemType Directory -Path $directoryPath" in source
    assert "New-Item -ItemType Directory -LiteralPath" not in source


def test_protected_root_provisioner_never_operates_vm_or_samples() -> None:
    source = PROVISION_SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "New-VM",
        "Set-VM",
        "Start-VM",
        "Stop-VM",
        "Remove-VM",
        "New-VHD",
        "Mount-VHD",
        "Dismount-VHD",
        "New-VMSwitch",
        "Remove-VMSwitch",
        "Get-VM",
        "Get-VMSwitch",
        "Copy-VMFile",
        "Invoke-Command",
        "Enter-PSSession",
        "capa",
    )

    for command in forbidden:
        assert command not in source
    assert "protected_root_created_no_vm_or_sample_action_authorized" in source
    assert "protected_root_blocked_fail_closed" in source
    assert "sample_access_allowed = $false" in source
    assert "training_allowed = $false" in source
    assert "heldout_allowed = $false" in source
    assert "f1_claim_allowed = $false" in source


def test_documentation_covers_protected_root_provisioning() -> None:
    text = DOC.read_text(encoding="utf-8")

    assert "Initialize-Loop171ProtectedRoot.ps1" in text
    assert "requires the requested dedicated parent and its root child to be new" in text
    assert "protected_root_created_no_vm_or_sample_action_authorized" in text


def test_elevation_launcher_can_only_invoke_the_protected_root_initializer() -> None:
    source = ELEVATION_LAUNCHER.read_text(encoding="utf-8")

    assert "Initialize-Loop171ProtectedRoot.ps1" in source
    assert "-Verb RunAs" in source
    assert "if ([string]::IsNullOrWhiteSpace($ReceiptPath))" in source
    assert "New-VM" not in source
    assert "New-VHD" not in source
    assert "Start-VM" not in source


def test_preflight_elevation_launcher_can_only_invoke_the_read_only_preflight() -> None:
    source = PREFLIGHT_ELEVATION_LAUNCHER.read_text(encoding="utf-8")

    assert "Invoke-Loop171HyperVPreflight.ps1" in source
    assert "-Verb RunAs" in source
    assert "New-VM" not in source
    assert "New-VHD" not in source
    assert "Start-VM" not in source
