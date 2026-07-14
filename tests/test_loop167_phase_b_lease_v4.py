from __future__ import annotations

from pathlib import Path

import pytest
from test_loop167_phase_b_v4_execution_authorization import NOW, _synthetic_v4_root

from src.loop167_phase_b.contracts import PhaseBContractError
from src.loop167_phase_b.execution_authorization_v4 import validate_execution_authorization_v4
from src.loop167_phase_b.lease_v4 import (
    ExecutionLeaseError,
    _ensure_safe_parent_directory,
    _write_marker_exclusive,
    consume_execution_lease_v4,
    verify_consumed_execution_lease_v4,
)


def test_v4_lease_is_exclusive_durable_and_verifiable(tmp_path: Path) -> None:
    root, context = _synthetic_v4_root(tmp_path)
    authorization = validate_execution_authorization_v4(root, context["authorization_path"], now_utc=NOW)

    consumed = consume_execution_lease_v4(root, context["authorization_path"], now_utc=NOW)

    assert consumed.marker_path == authorization.lease_marker_path
    assert consumed.marker_path.is_file()
    assert verify_consumed_execution_lease_v4(root, authorization).marker_sha256 == consumed.marker_sha256
    with pytest.raises(PhaseBContractError, match="lease marker already exists"):
        consume_execution_lease_v4(root, context["authorization_path"], now_utc=NOW)


def test_v4_lease_retains_the_marker_when_parent_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "marker.json"
    monkeypatch.setattr(
        "src.loop167_phase_b.lease_v4._fsync_parent_directory",
        lambda _parent: (_ for _ in ()).throw(OSError("synthetic sync failure")),
    )

    with pytest.raises(ExecutionLeaseError, match="remains consumed"):
        _write_marker_exclusive(marker, {"run_authorization": {"sha256": "a" * 64}})

    assert marker.is_file()
    with pytest.raises(ExecutionLeaseError, match="already"):
        _write_marker_exclusive(marker, {"run_authorization": {"sha256": "a" * 64}})


def test_v4_lease_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "reports").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")

    with pytest.raises(ExecutionLeaseError, match="symlink"):
        _ensure_safe_parent_directory(root, "reports/roadmap_9997/loop167/marker.json")


def test_v4_lease_rechecks_static_bindings_before_raw_access(tmp_path: Path) -> None:
    root, context = _synthetic_v4_root(tmp_path)
    authorization = validate_execution_authorization_v4(root, context["authorization_path"], now_utc=NOW)
    consume_execution_lease_v4(root, context["authorization_path"], now_utc=NOW)
    controller = root / authorization.controller_binding["path"]
    controller.write_text("print('drifted controller')\n", encoding="ascii")

    with pytest.raises(ExecutionLeaseError, match="controller changed"):
        verify_consumed_execution_lease_v4(root, authorization)
