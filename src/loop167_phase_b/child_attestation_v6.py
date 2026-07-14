"""Child-side containment attestation written before the v6 authorization lease."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Any, Mapping

from .contracts import PhaseBContractError, canonical_json_bytes
from .execution_contract_v6 import (
    FIXED_OUTPUT_CATALOG,
    LOOP_ID,
    assert_contained_child_prelease_surface_v6,
)
from .loop166_v6_bridge import (
    JobMembershipV6Error,
    ProcessLineageV6Error,
    audit_current_process_job_membership,
    audit_process_job_membership,
    validate_spawn_lineage,
)
from .path_safety_v4 import safe_project_path, safe_project_relative_path, safe_project_root
from .supervisor_v6 import _write_new_json, validate_launch_receipt_v6

CHILD_ATTESTATION_SCHEMA = "axon_loop167_phase_b_v6_child_job_attestation"


def _verify_live_process_identity(pid: int, expected_creation_time_filetime: int) -> dict[str, int | bool | str]:
    if os.name != "nt" or pid <= 0 or expected_creation_time_filetime <= 0:
        raise PhaseBContractError("v6 supervisor identity input is invalid")

    class FileTime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        raise PhaseBContractError(f"v6 supervisor identity cannot open PID {pid}: Win32 {ctypes.get_last_error()}")
    try:
        if kernel32.WaitForSingleObject(handle, 0) != 0x00000102:
            raise PhaseBContractError("v6 supervisor is no longer active")
        creation = FileTime()
        exit_time = FileTime()
        kernel_time = FileTime()
        user_time = FileTime()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            raise PhaseBContractError(
                f"v6 supervisor identity cannot read creation time: Win32 {ctypes.get_last_error()}"
            )
        observed_creation = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        if observed_creation != expected_creation_time_filetime:
            raise PhaseBContractError("v6 supervisor PID creation time drifted")
        return {
            "pid": pid,
            "creation_time_filetime": observed_creation,
            "active": True,
            "verification_scope": "supervisor_liveness_and_creation_time",
        }
    finally:
        kernel32.CloseHandle(handle)


def _output_path(root: Path) -> Path:
    entries = {entry["name"]: entry["path"] for entry in FIXED_OUTPUT_CATALOG}
    return safe_project_path(root, entries["child_job_attestation"], require_exists=False)


def _binding(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PhaseBContractError(f"{label} binding is invalid")
    path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise PhaseBContractError(f"{label} binding is invalid")
    return {"path": path, "sha256": digest}


def build_child_job_attestation_payload_v6(
    root: Path | str,
    *,
    launch_receipt_path: Path | str,
    launch_id: str,
    expected_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    root_path = safe_project_root(root)
    receipt_path = Path(launch_receipt_path)
    validated_launch = validate_launch_receipt_v6(
        root_path,
        receipt_path,
        mode="execute",
        expected_bindings=expected_bindings,
        expected_launch_id=launch_id,
        expected_pid=os.getpid(),
        expected_supervisor_pid=int(os.environ.get("AXON_LOOP167_V6_SUPERVISOR_PID", "0") or "0"),
    )
    receipt = validated_launch.payload
    job_proof_binding = _binding(
        expected_bindings.get("loop166_windows_job"),
        label="loop166_windows_job",
    )
    lineage_proof_binding = _binding(
        expected_bindings.get("loop166_windows_process_lineage"),
        label="loop166_windows_process_lineage",
    )
    if job_proof_binding["path"] != "src/loop166/windows_job.py":
        raise PhaseBContractError("v6 Loop166 Job proof binding path drifted")
    if lineage_proof_binding["path"] != "src/loop166/windows_process_lineage.py":
        raise PhaseBContractError("v6 Loop166 lineage proof binding path drifted")
    assignment = receipt["pre_resume_assignment"]
    try:
        membership = audit_current_process_job_membership(
            root_path,
            job_proof_binding,
            int(assignment["process_creation_time_filetime"]),
            expected_pid=os.getpid(),
        )
    except JobMembershipV6Error as error:
        raise PhaseBContractError("v6 child cannot prove current Job membership") from error
    supervisor = receipt["supervisor_identity"]
    try:
        launcher_membership = audit_process_job_membership(
            root_path,
            job_proof_binding,
            int(assignment["process_pid"]),
            int(assignment["process_creation_time_filetime"]),
        )
        supervisor_identity = _verify_live_process_identity(
            int(supervisor["pid"]),
            int(supervisor["creation_time_filetime"]),
        )
        lineage = validate_spawn_lineage(
            root_path,
            lineage_proof_binding,
            int(supervisor["pid"]),
            launcher_executable=receipt["command"][0],
            base_executable=sys._base_executable,
        )
    except (
        JobMembershipV6Error,
        ProcessLineageV6Error,
        KeyError,
        TypeError,
        ValueError,
        PhaseBContractError,
    ) as error:
        raise PhaseBContractError("v6 child cannot prove launcher lineage and liveness") from error
    expected_launcher_pid = os.getpid() if lineage["mode"] == "direct_parent" else lineage["redirector_pid"]
    if int(assignment["process_pid"]) != expected_launcher_pid:
        raise PhaseBContractError("v6 pre-resume receipt launcher identity differs from bounded lineage")
    receipt_relative = safe_project_relative_path(
        root_path,
        receipt_path,
        require_exists=True,
        require_regular_file=True,
    )
    return {
        "schema": CHILD_ATTESTATION_SCHEMA,
        "loop_id": LOOP_ID,
        "status": "child_identity_and_job_membership_verified_before_static_preflight",
        "launch_id": launch_id,
        "launch_receipt": {
            "path": receipt_relative,
            "sha256": validated_launch.canonical_sha256,
        },
        "pre_resume_assignment": dict(assignment),
        "child_membership": membership,
        "launcher_membership": launcher_membership,
        "supervisor_identity": supervisor_identity,
        "lineage": lineage,
        "static_bindings": {name: _binding(binding, label=name) for name, binding in expected_bindings.items()},
        "raw_open_attempts": 0,
    }


def write_child_job_attestation_v6(root: Path | str, payload: Mapping[str, Any]) -> tuple[Path, str]:
    root_path = safe_project_root(root)
    assert_contained_child_prelease_surface_v6(root_path)
    output_path = _output_path(root_path)
    digest = _write_new_json(root_path, output_path, payload)
    return output_path, digest


def verify_child_job_attestation_v6(
    root: Path | str,
    *,
    expected_launch_receipt_path: Path | str,
    expected_launch_id: str,
    expected_bindings: Mapping[str, Mapping[str, str]],
) -> tuple[Path, str, Mapping[str, Any]]:
    root_path = safe_project_root(root)
    path = _output_path(root_path)
    if not path.is_file() or path.is_symlink():
        raise PhaseBContractError("v6 child Job attestation is unavailable")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseBContractError("v6 child Job attestation is unavailable") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise PhaseBContractError("v6 child Job attestation is not canonical")
    expected = build_child_job_attestation_payload_v6(
        root_path,
        launch_receipt_path=expected_launch_receipt_path,
        launch_id=expected_launch_id,
        expected_bindings=expected_bindings,
    )
    if payload != expected:
        raise PhaseBContractError("v6 child Job attestation drifted")
    return path, hashlib.sha256(raw).hexdigest(), payload


__all__ = [
    "CHILD_ATTESTATION_SCHEMA",
    "build_child_job_attestation_payload_v6",
    "verify_child_job_attestation_v6",
    "write_child_job_attestation_v6",
]
