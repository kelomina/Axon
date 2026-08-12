from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

import src.loop167_phase_b.lease_v7 as lease_v7
from src.loop167_phase_b.contracts import PhaseBContractError, canonical_json_bytes
from src.loop167_phase_b.execution_authorization_v7 import VerifiedExecutionAuthorizationV7
from src.loop167_phase_b.execution_contract_v7 import (
    EXPECTED_LEASE,
    FIXED_OUTPUT_CATALOG,
    RUN_AUTHORIZATION_RELATIVE_PATH,
)
from src.loop167_phase_b.lease_v7 import (
    ExecutionLeaseV7Error,
    build_execution_lease_payload_v7,
    consume_execution_lease_v7,
    verify_consumed_execution_lease_v7,
)
from src.loop167_phase_b.supervisor_v7 import ValidatedLaunchReceiptV7

NOW = datetime(2026, 7, 14, 0, 0, tzinfo=UTC)
LAUNCH_ID = "b" * 64


def _output_path(root: Path, name: str) -> Path:
    return root / next(entry["path"] for entry in FIXED_OUTPUT_CATALOG if entry["name"] == name)


def _authorization(root: Path, *, digest: str = "a" * 64) -> VerifiedExecutionAuthorizationV7:
    authorization_path = root / RUN_AUTHORIZATION_RELATIVE_PATH
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    authorization_path.write_bytes(b"synthetic authorization")
    launch_path = _output_path(root, "supervisor_launch_receipt")
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    launch_path.write_bytes(b"synthetic launch")
    child_path = _output_path(root, "child_job_attestation")
    child_path.write_bytes(b"synthetic child attestation")
    bindings = {
        "path": "manifests/static.json",
        "sha256": "c" * 64,
    }
    return VerifiedExecutionAuthorizationV7(
        project_root=root,
        authorization_path=authorization_path,
        authorization_sha256=digest,
        execution_contract_binding=MappingProxyType(dict(bindings)),
        protocol_binding=MappingProxyType({"path": "manifests/protocol.json", "sha256": "d" * 64}),
        source_closure_binding=MappingProxyType({"path": "manifests/source.json", "sha256": "e" * 64}),
        runtime_lock_binding=MappingProxyType({"path": "manifests/runtime.json", "sha256": "f" * 64}),
        controller_binding=MappingProxyType({"path": "scripts/controller.py", "sha256": "1" * 64}),
        supervisor_binding=MappingProxyType({"path": "scripts/supervisor.py", "sha256": "2" * 64}),
        loop166_windows_job_binding=MappingProxyType(
            {"path": "src/loop166/windows_job.py", "sha256": "4" * 64}
        ),
        loop166_windows_process_lineage_binding=MappingProxyType(
            {"path": "src/loop166/windows_process_lineage.py", "sha256": "5" * 64}
        ),
        resource_guard_binding=MappingProxyType({"path": "manifests/guard.json", "sha256": "3" * 64}),
        output_paths=MappingProxyType(
            {
                "supervisor_launch_receipt": launch_path,
                "child_job_attestation": child_path,
            }
        ),
        lease_marker_path=root / EXPECTED_LEASE["marker_path"],
    )


def _install_verification_mocks(
    monkeypatch: pytest.MonkeyPatch,
    authorization: VerifiedExecutionAuthorizationV7,
    *,
    child_digest: str = "8" * 64,
) -> None:
    launch_path = authorization.output_paths["supervisor_launch_receipt"]
    child_path = authorization.output_paths["child_job_attestation"]
    expected_bindings = {
        "source_closure": dict(authorization.source_closure_binding),
        "execution_contract": dict(authorization.execution_contract_binding),
        "runtime_lock": dict(authorization.runtime_lock_binding),
        "controller": dict(authorization.controller_binding),
        "supervisor": dict(authorization.supervisor_binding),
        "loop166_windows_job": dict(authorization.loop166_windows_job_binding),
        "loop166_windows_process_lineage": dict(authorization.loop166_windows_process_lineage_binding),
    }
    launch_digest = "7" * 64
    monkeypatch.setattr(
        lease_v7,
        "validate_launch_receipt_v7",
        lambda *_args, **_kwargs: ValidatedLaunchReceiptV7(
            receipt_path=launch_path,
            payload={"launch_id": LAUNCH_ID},
            canonical_sha256=launch_digest,
        ),
    )
    monkeypatch.setattr(
        lease_v7,
        "verify_child_job_attestation_v7",
        lambda *_args, **_kwargs: (
            child_path,
            child_digest,
            {
                "launch_id": LAUNCH_ID,
                "launch_receipt": {
                    "path": launch_path.relative_to(authorization.project_root).as_posix(),
                    "sha256": launch_digest,
                },
                "static_bindings": expected_bindings,
            },
        ),
    )
    monkeypatch.setattr(lease_v7, "validate_execution_authorization_v7", lambda *_args, **_kwargs: authorization)
    real_sha256_file = lease_v7.sha256_file
    monkeypatch.setattr(
        lease_v7,
        "sha256_file",
        lambda path: (
            authorization.authorization_sha256
            if Path(path) == authorization.authorization_path
            else real_sha256_file(path)
        ),
    )


def test_v7_lease_is_one_shot_and_binds_the_validated_launch_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    authorization = _authorization(root)
    _install_verification_mocks(monkeypatch, authorization)

    consumed = consume_execution_lease_v7(root, authorization.authorization_path, now_utc=NOW, launch_id=LAUNCH_ID)
    verified = verify_consumed_execution_lease_v7(
        root,
        authorization,
        launch_id=LAUNCH_ID,
        now_utc=NOW,
    )

    assert consumed.marker_path == authorization.lease_marker_path
    assert consumed.payload["pre_resume_launch_receipt"]["sha256"] == "7" * 64
    assert verified.marker_sha256 == consumed.marker_sha256
    with pytest.raises(ExecutionLeaseV7Error, match="already exists"):
        consume_execution_lease_v7(root, authorization.authorization_path, now_utc=NOW, launch_id=LAUNCH_ID)


def test_v7_lease_rejects_a_child_attestation_with_a_different_exact_launch_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    authorization = _authorization(root)
    _install_verification_mocks(monkeypatch, authorization)
    child_path = authorization.output_paths["child_job_attestation"]
    monkeypatch.setattr(
        lease_v7,
        "verify_child_job_attestation_v7",
        lambda *_args, **_kwargs: (
            child_path,
            "8" * 64,
            {
                "launch_id": LAUNCH_ID,
                "launch_receipt": {
                    "path": authorization.output_paths["supervisor_launch_receipt"].relative_to(root).as_posix(),
                    "sha256": "9" * 64,
                },
            },
        ),
    )

    with pytest.raises(ExecutionLeaseV7Error, match="launch receipt digest drifted"):
        build_execution_lease_payload_v7(authorization, launch_id=LAUNCH_ID, consumed_at_utc=NOW)


def test_v7_lease_revalidates_authorization_after_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    authorization = _authorization(root)
    _install_verification_mocks(monkeypatch, authorization)
    consume_execution_lease_v7(root, authorization.authorization_path, now_utc=NOW, launch_id=LAUNCH_ID)
    drifted = _authorization(root, digest="b" * 64)
    monkeypatch.setattr(lease_v7, "validate_execution_authorization_v7", lambda *_args, **_kwargs: drifted)

    with pytest.raises(ExecutionLeaseV7Error, match="authorization drifted after lease"):
        verify_consumed_execution_lease_v7(root, authorization, launch_id=LAUNCH_ID, now_utc=NOW)


def test_v7_lease_propagates_post_lease_source_or_guard_revalidation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    authorization = _authorization(root)
    _install_verification_mocks(monkeypatch, authorization)
    consume_execution_lease_v7(root, authorization.authorization_path, now_utc=NOW, launch_id=LAUNCH_ID)
    monkeypatch.setattr(
        lease_v7,
        "validate_execution_authorization_v7",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PhaseBContractError("source closure hash mismatch")),
    )

    with pytest.raises(PhaseBContractError, match="source closure hash mismatch"):
        verify_consumed_execution_lease_v7(root, authorization, launch_id=LAUNCH_ID, now_utc=NOW)


def test_v7_lease_rejects_a_marker_replaced_after_its_canonical_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    authorization = _authorization(root)
    _install_verification_mocks(monkeypatch, authorization)
    consume_execution_lease_v7(root, authorization.authorization_path, now_utc=NOW, launch_id=LAUNCH_ID)
    original_builder = lease_v7.build_execution_lease_payload_v7

    def replace_marker_after_expected_payload(*args, **kwargs):
        payload = original_builder(*args, **kwargs)
        authorization.lease_marker_path.write_bytes(canonical_json_bytes({"replaced": True}))
        return payload

    monkeypatch.setattr(lease_v7, "build_execution_lease_payload_v7", replace_marker_after_expected_payload)

    with pytest.raises(ExecutionLeaseV7Error, match="marker changed during verification"):
        verify_consumed_execution_lease_v7(root, authorization, launch_id=LAUNCH_ID, now_utc=NOW)
