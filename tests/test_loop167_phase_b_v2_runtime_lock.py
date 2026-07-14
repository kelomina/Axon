from __future__ import annotations

import hashlib

from src.loop167_phase_b.runtime_lock_v2 import (
    REQUIRED_ENVIRONMENT,
    build_runtime_lock_payload,
    validate_runtime_lock,
)


def test_runtime_lock_v2_is_compatible_with_isolated_python_hash_policy(tmp_path, monkeypatch) -> None:
    controller = tmp_path / "controller.py"
    addendum = tmp_path / "addendum.json"
    controller.write_text("print('synthetic')\n", encoding="ascii")
    addendum.write_text("{}\n", encoding="ascii")
    controller_binding = {"path": "controller.py", "sha256": hashlib.sha256(controller.read_bytes()).hexdigest()}
    addendum_binding = {"path": "addendum.json", "sha256": hashlib.sha256(addendum.read_bytes()).hexdigest()}
    argv = ("vnev/Scripts/python.exe", "-I", "controller.py", "--preflight")
    for name, value in REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    payload = build_runtime_lock_payload(
        tmp_path,
        controller_binding=controller_binding,
        isolation_addendum_binding=addendum_binding,
        canonical_argv=argv,
    )
    validate_runtime_lock(
        tmp_path,
        payload,
        controller_binding=controller_binding,
        isolation_addendum_binding=addendum_binding,
        canonical_argv=argv,
    )
