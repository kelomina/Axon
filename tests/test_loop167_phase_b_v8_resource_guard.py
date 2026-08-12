from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.loop167_phase_b.resource_guard_v8 as resource_guard_v8
from src.loop167_phase_b.contracts import PhaseBContractError, canonical_json_bytes, sha256_file
from src.loop167_phase_b.execution_contract_v8 import (
    EXECUTION_CONTRACT_RELATIVE_PATH,
    LOOP_ID,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
)
from src.loop167_phase_b.windows_job_v8 import WindowsJobAssignmentProbeV8

NOW = datetime(2026, 7, 14, 0, 0, tzinfo=UTC)
GIBIBYTE = 1024**3


def _write_json(root: Path, relative_path: str, payload: dict[str, object]) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return path


def _binding(root: Path, relative_path: str) -> dict[str, str]:
    return {"path": relative_path, "sha256": sha256_file(root / relative_path)}


def _ready_probe() -> WindowsJobAssignmentProbeV8:
    return WindowsJobAssignmentProbeV8(
        ready=True,
        operation=None,
        win32_error_code=None,
        detail=None,
        assignment={
            "current_process_assigned": True,
            "kill_on_job_close": False,
            "job_limit_flags": 0x100,
        },
    )


def _resource_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, dict[str, str]]]:
    root = tmp_path / "project"
    root.mkdir()

    _write_json(root, EXECUTION_CONTRACT_RELATIVE_PATH, {"schema": "synthetic_contract"})
    _write_json(
        root,
        SOURCE_CLOSURE_RELATIVE_PATH,
        {"schema": "axon_loop167_phase_b_source_closure_v8", "loop_id": LOOP_ID},
    )
    _write_json(
        root,
        RUNTIME_LOCK_RELATIVE_PATH,
        {"schema": "axon_loop167_phase_b_runtime_lock_v8", "loop_id": LOOP_ID},
    )
    bindings = {
        "execution_contract": _binding(root, EXECUTION_CONTRACT_RELATIVE_PATH),
        "source_closure": _binding(root, SOURCE_CLOSURE_RELATIVE_PATH),
        "runtime_lock": _binding(root, RUNTIME_LOCK_RELATIVE_PATH),
    }
    resource_contract = {
        "maximum_training_peak_rss_bytes": 8 * GIBIBYTE,
        "maximum_extraction_peak_rss_bytes": 4 * GIBIBYTE,
        "worker_count": 1,
    }

    def verify_contract(_root: Path, binding: object) -> SimpleNamespace:
        assert binding == bindings["execution_contract"]
        return SimpleNamespace(
            resource_contract=resource_contract,
            contract_sha256=bindings["execution_contract"]["sha256"],
        )

    monkeypatch.setattr(resource_guard_v8, "verify_execution_contract_v8", verify_contract)
    return root, bindings


def _build_guard(
    root: Path,
    bindings: dict[str, dict[str, str]],
    *,
    available_memory_bytes: int,
) -> dict[str, object]:
    return resource_guard_v8.build_resource_guard_payload_v8(
        root,
        execution_contract_binding=bindings["execution_contract"],
        source_closure_binding=bindings["source_closure"],
        runtime_lock_binding=bindings["runtime_lock"],
        snapshot=resource_guard_v8.SystemResourceSnapshotV8(
            total_memory_bytes=32 * GIBIBYTE,
            available_memory_bytes=available_memory_bytes,
            cpu_count=1,
        ),
        created_at_utc="2026-07-14T00:00:00Z",
        probe=_ready_probe(),
    )


def _validate_guard(
    root: Path,
    bindings: dict[str, dict[str, str]],
    payload: dict[str, object],
    *,
    now_utc: datetime,
) -> None:
    resource_guard_v8.validate_resource_guard_payload_v8(
        root,
        payload,
        expected_execution_contract_binding=bindings["execution_contract"],
        expected_source_closure_binding=bindings["source_closure"],
        expected_runtime_lock_binding=bindings["runtime_lock"],
        now_utc=now_utc,
    )


def test_v7_resource_guard_fails_closed_below_twelve_gib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, bindings = _resource_fixture(tmp_path, monkeypatch)
    payload = _build_guard(root, bindings, available_memory_bytes=12 * GIBIBYTE - 1)

    assert payload["minimum_available_memory_bytes"] == 12 * GIBIBYTE
    assert payload["guard_ready"] is False
    assert payload["decision"] == "fail_closed"
    assert "available_memory_below_sealed_launch_floor" in payload["failures"]
    assert payload["raw_open_attempts"] == 0
    with pytest.raises(PhaseBContractError, match="resource guard is not ready"):
        _validate_guard(root, bindings, payload, now_utc=NOW)


def test_v7_resource_guard_accepts_fresh_ready_twelve_gib_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, bindings = _resource_fixture(tmp_path, monkeypatch)
    payload = _build_guard(root, bindings, available_memory_bytes=12 * GIBIBYTE)
    _write_json(root, RESOURCE_GUARD_RELATIVE_PATH, payload)
    guard_binding = _binding(root, RESOURCE_GUARD_RELATIVE_PATH)

    verified = resource_guard_v8.verify_resource_guard_v8(
        root,
        guard_binding,
        expected_execution_contract_binding=bindings["execution_contract"],
        expected_source_closure_binding=bindings["source_closure"],
        expected_runtime_lock_binding=bindings["runtime_lock"],
        now_utc=NOW,
    )

    assert payload["guard_ready"] is True
    assert payload["available_memory_margin_bytes"] == 0
    assert verified.guard_sha256 == guard_binding["sha256"]


def test_v7_resource_guard_rejects_a_stale_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, bindings = _resource_fixture(tmp_path, monkeypatch)
    payload = _build_guard(root, bindings, available_memory_bytes=12 * GIBIBYTE)
    _write_json(root, RESOURCE_GUARD_RELATIVE_PATH, payload)
    guard_binding = _binding(root, RESOURCE_GUARD_RELATIVE_PATH)

    with pytest.raises(PhaseBContractError, match="stale"):
        resource_guard_v8.verify_resource_guard_v8(
            root,
            guard_binding,
            expected_execution_contract_binding=bindings["execution_contract"],
            expected_source_closure_binding=bindings["source_closure"],
            expected_runtime_lock_binding=bindings["runtime_lock"],
            now_utc=NOW + timedelta(seconds=301),
        )


def test_v7_resource_guard_rejects_drifted_static_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, bindings = _resource_fixture(tmp_path, monkeypatch)
    payload = _build_guard(root, bindings, available_memory_bytes=12 * GIBIBYTE)
    _write_json(root, RESOURCE_GUARD_RELATIVE_PATH, payload)
    guard_binding = _binding(root, RESOURCE_GUARD_RELATIVE_PATH)
    _write_json(
        root,
        SOURCE_CLOSURE_RELATIVE_PATH,
        {
            "schema": "axon_loop167_phase_b_source_closure_v8",
            "loop_id": LOOP_ID,
            "drift": True,
        },
    )

    with pytest.raises(PhaseBContractError, match="source_closure hash mismatch"):
        resource_guard_v8.verify_resource_guard_v8(
            root,
            guard_binding,
            expected_execution_contract_binding=bindings["execution_contract"],
            expected_source_closure_binding=bindings["source_closure"],
            expected_runtime_lock_binding=bindings["runtime_lock"],
            now_utc=NOW,
        )
