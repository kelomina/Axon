from __future__ import annotations

import hashlib

from src.loop167_phase_b.runtime_lock import (
    REQUIRED_ENVIRONMENT,
    build_runtime_lock_payload,
    validate_runtime_lock,
)


def test_runtime_lock_is_reproducible_and_requires_pinned_environment(tmp_path, monkeypatch) -> None:
    controller = tmp_path / "controller.py"
    controller.write_text("print('synthetic')\n", encoding="ascii")
    binding = {"path": "controller.py", "sha256": hashlib.sha256(controller.read_bytes()).hexdigest()}
    argv = ("vnev/Scripts/python.exe", "-I", "controller.py", "--preflight")
    for name, value in REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    payload = build_runtime_lock_payload(tmp_path, controller_binding=binding, canonical_argv=argv)
    validate_runtime_lock(tmp_path, payload, controller_binding=binding, canonical_argv=argv)

    monkeypatch.delenv("OMP_NUM_THREADS")
    try:
        validate_runtime_lock(tmp_path, payload, controller_binding=binding, canonical_argv=argv)
    except ValueError as exc:
        assert "environment" in str(exc)
    else:
        raise AssertionError("Runtime lock accepted an unpinned environment")
