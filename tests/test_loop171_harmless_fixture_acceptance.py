from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "New-Loop171HarmlessFixtureAcceptancePlan.ps1"
CONTRACT = ROOT / "manifests/roadmap_9997/loop171_hyperv_isolation/harmless_fixture_acceptance_contract.json"


def test_acceptance_contract_preserves_all_isolation_requirements() -> None:
    contract = CONTRACT.read_text(encoding="utf-8")

    for requirement in (
        '"generation": 2',
        '"network_adapter_count": 0',
        '"fixture_input_is_immutable_read_only": true',
        '"output_max_bytes": 65536',
        '"forced_guest_termination_required": true',
        '"no_new_vm_or_vmwp_survivor_required": true',
        '"sample_access_allowed": false',
    ):
        assert requirement in contract


def test_plan_is_gated_by_a_sha_bound_passing_preflight_receipt() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "PreflightReceiptSha256" in source
    assert "preflight_pass_no_vm_or_sample_action_authorized" in source
    assert "Preflight gate is not true" in source
    assert "preflight_receipt_sha256_bound" in source
    assert "preflight_receipt_passes_all_required_gates" in source
    assert "Dedicated root must remain empty" in source


def test_plan_defines_future_acceptance_without_vm_or_sample_operations() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "New-VM", "Set-VM", "Start-VM", "Stop-VM", "Remove-VM", "New-VHD",
        "Mount-VHD", "Dismount-VHD", "New-VMSwitch", "Remove-VMSwitch",
        "Add-VMNetworkAdapter", "Copy-VMFile", "Invoke-Command", "Enter-PSSession",
    )

    for command in forbidden:
        assert command not in source
    assert "immutable_read_only_no_sample_iso" in source
    assert "no_new_vm_or_vmwp_survivor" in source
    assert "acceptance_plan_blocked_fail_closed" in source
    assert "sample_access_allowed = $false" in source


def test_powershell_parser_accepts_the_static_plan_when_available() -> None:
    executable = Path(os.environ.get("SystemRoot", r"C:\\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not executable.exists():
        pytest.skip("Windows PowerShell is unavailable in this environment")

    command = (
        "$tokens=$null; $errors=$null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -ne 0) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    result = subprocess.run(
        [str(executable), "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
