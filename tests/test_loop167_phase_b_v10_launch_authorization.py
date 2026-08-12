from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.loop167_phase_b.launch_authorization_v10 as launch_authorization_v10


def test_v9_fresh_authorization_writes_a_low_memory_guard_without_a_prelaunch_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    observed: dict[str, object] = {}

    def output_path(_root: Path, relative_path: str) -> Path:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def build_guard(_root: Path, **kwargs: object) -> dict[str, object]:
        snapshot = kwargs["snapshot"]
        observed["available_memory_bytes"] = snapshot.available_memory_bytes
        return {"guard_ready": True, "raw_open_attempts": 0}

    monkeypatch.setattr(launch_authorization_v10, "assert_output_catalog_is_fresh_v10", lambda _root: {})
    monkeypatch.setattr(launch_authorization_v10, "ensure_v10_static_artifact_parent", output_path)
    monkeypatch.setattr(launch_authorization_v10, "_binding", lambda _root, path: {"path": path, "sha256": "a" * 64})
    monkeypatch.setattr(
        launch_authorization_v10,
        "current_system_snapshot_v10",
        lambda: SimpleNamespace(total_memory_bytes=32, available_memory_bytes=0, cpu_count=1),
    )
    monkeypatch.setattr(launch_authorization_v10, "build_resource_guard_payload_v10", build_guard)
    monkeypatch.setattr(
        launch_authorization_v10,
        "build_execution_authorization_payload_v10",
        lambda *_args, **kwargs: {"resource_guard": kwargs["resource_guard_binding"]},
    )
    monkeypatch.setattr(
        launch_authorization_v10,
        "validate_execution_authorization_v10",
        lambda *_args, **_kwargs: SimpleNamespace(authorized=True),
    )

    result = launch_authorization_v10.create_fresh_launch_authorization_v10(
        root,
        now_utc=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert observed["available_memory_bytes"] == 0
    assert result.resource_guard_sha256
    assert (root / launch_authorization_v10.RESOURCE_GUARD_RELATIVE_PATH).is_file()
    assert (root / launch_authorization_v10.RUN_AUTHORIZATION_RELATIVE_PATH).is_file()


def test_v9_fresh_authorization_refuses_precreated_dynamic_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "project"
    existing = root / launch_authorization_v10.RESOURCE_GUARD_RELATIVE_PATH
    existing.parent.mkdir(parents=True)
    existing.write_text("{}", encoding="ascii")
    monkeypatch.setattr(launch_authorization_v10, "assert_output_catalog_is_fresh_v10", lambda _root: {})

    with pytest.raises(launch_authorization_v10.FreshAuthorizationV9Error, match="pre-created"):
        launch_authorization_v10.create_fresh_launch_authorization_v10(
            root,
            now_utc=datetime(2026, 7, 14, tzinfo=UTC),
        )
