from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = ROOT / "scripts" / "Invoke-Loop171HarmlessFixtureAcceptance.ps1"
LAUNCHER = ROOT / "scripts" / "Start-Loop171HarmlessFixtureAcceptanceElevation.ps1"


def test_executor_binds_same_process_preflight_and_fixture_hashes() -> None:
    source = EXECUTOR.read_text(encoding="utf-8")

    assert "& $preflight" in source
    assert "Same-process preflight failed" in source
    assert "Harmless fixture ISO hash mismatches before boot" in source
    assert "Harmless fixture ISO changed during boot" in source
    assert "Base VHD changed during boot" in source
    assert "Acceptance receipt overwrite is forbidden" in source


def test_executor_enforces_no_network_no_integration_and_bounded_serial_receipt() -> None:
    source = EXECUTOR.read_text(encoding="utf-8")

    for token in (
        "Remove-VMNetworkAdapter",
        "Get-VMNetworkAdapter",
        "Disable-VMIntegrationService",
        "Get-VMIntegrationService",
        "NamedPipeServerStream",
        "PipeDirection]::In",
        "Set-VMComPort",
        "Guest serial receipt violates the aggregate byte limit",
        "write_attempt_blocked",
    ):
        assert token in source
    assert "Copy-VMFile" not in source
    assert "Invoke-Command" not in source
    assert "Enter-PSSession" not in source


def test_executor_uses_disposable_differencing_storage_and_proves_teardown() -> None:
    source = EXECUTOR.read_text(encoding="utf-8")

    assert "New-VHD -Path $childVhdPath -ParentPath $baseVhdPath -Differencing" in source
    assert "CheckpointType Disabled" in source
    assert "AutomaticStartAction Nothing" in source
    assert "Remove-RegisteredAssets" in source
    assert "new_vmwp_process_ids" in source
    assert "dedicated_root_empty" in source
    assert "sample_access_allowed = $false" in source
    assert "f1_claim_allowed = $false" in source


@pytest.mark.parametrize("script", (EXECUTOR, LAUNCHER))
def test_powershell_parser_accepts_executor_scripts_when_available(script: Path) -> None:
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
