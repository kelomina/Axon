#!/usr/bin/env python3
"""Launch the frozen Loop166 recovery-v2 parent outside the interactive session."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop166.windows_job import (  # noqa: E402
    WindowsJobTimeoutError,
    WindowsKillOnCloseJob,
    run_subprocess_in_job,
)

FOUNDATION_DIR = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop166_code_section_foundation"
REPORT_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop166"
MODEL_DIR = PROJECT_ROOT / "models" / "roadmap_9997" / "loop166"

PYTHON_EXECUTABLE = PROJECT_ROOT / "vnev" / "Scripts" / "python.exe"
CONTROLLER = PROJECT_ROOT / "scripts" / "run_loop166_phase_b1_step4096_recovery_v2.py"
POWERSHELL_LAUNCHER = (
    PROJECT_ROOT / "scripts" / "run_loop166_phase_b1_step4096_recovery_v2_detached.ps1"
)
WINDOWS_JOB_SOURCE = SRC_DIR / "loop166" / "windows_job.py"
WINDOWS_LINEAGE_SOURCE = SRC_DIR / "loop166" / "windows_process_lineage.py"
RAW_LEDGER_SOURCE = SRC_DIR / "loop166" / "raw_progress_ledger.py"
CONTROLLER_TESTS = PROJECT_ROOT / "tests" / "test_loop166_phase_b1_step4096_recovery_v2.py"
WINDOWS_JOB_TESTS = PROJECT_ROOT / "tests" / "test_loop166_windows_job.py"
SUPERVISOR_TESTS = PROJECT_ROOT / "tests" / "test_loop166_recovery_v2_supervisor.py"
CONTRACT = FOUNDATION_DIR / "phase_b1_step4096_recovery_v2.json"
AUTHORIZATION = FOUNDATION_DIR / "phase_b1_step4096_recovery_v2_authorization.json"
FOLDS = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_train_diagnostic_folds.jsonl"
FOLDS_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop164"
    / "local_train_diagnostic_folds_summary.json"
)
DATA_ROOT = PROJECT_ROOT / "data" / "random_20w_worktree"
SOURCE_TOKENIZER = REPORT_DIR / "phase_b1_tokenizer.json"
SOURCE_CHECKPOINT = MODEL_DIR / "phase_b1_tiny_mlm.pt"
CHECKPOINT_OUTPUT = MODEL_DIR / "phase_b1_step4096_recovery_v2_tiny_mlm.pt"
REPORT_OUTPUT = REPORT_DIR / "phase_b1_step4096_recovery_v2_report.json"
LAUNCH_RECEIPT = REPORT_DIR / "phase_b1_step4096_recovery_v2_launch_receipt.json"
EXIT_RECEIPT = REPORT_DIR / "phase_b1_step4096_recovery_v2_exit_receipt.json"
STDOUT_LOG = REPORT_DIR / "phase_b1_step4096_recovery_v2_stdout.log"
STDERR_LOG = REPORT_DIR / "phase_b1_step4096_recovery_v2_stderr.log"

LAUNCH_SCHEMA = "axon_loop166_phase_b1_step4096_recovery_v2_supervisor_launch_v1"
EXIT_SCHEMA = "axon_loop166_phase_b1_step4096_recovery_v2_supervisor_exit_v1"
MAXIMUM_SUPERVISOR_SECONDS = 25_044.0
MAXIMUM_COMBINED_LOG_BYTES = 64 * 1024 * 1024
SENSITIVE_ENVIRONMENT_PREFIX = "AXON_B1_RECOVERY_V2_"


class SupervisorError(RuntimeError):
    """Raised when detached recovery supervision cannot remain fail closed."""


@dataclass(frozen=True)
class SupervisorConfig:
    project_root: Path
    command: tuple[str, ...]
    launch_receipt: Path
    exit_receipt: Path
    stdout_log: Path
    stderr_log: Path
    source_bindings: tuple[tuple[str, Path], ...]
    timeout_seconds: float
    maximum_combined_log_bytes: int


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _canonical_json_sha256(payload: object) -> str:
    return _sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        kernel32.FlushFileBuffers.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateFileW(
            str(path.parent.resolve(strict=True)),
            0x80000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x02000000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in {None, 0, invalid}:
            return
        try:
            kernel32.FlushFileBuffers(handle)
        finally:
            kernel32.CloseHandle(handle)
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_sensitive_receipt_value(value: object, *, context: str = "receipt") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            folded = str(key).casefold()
            if "nonce" in folded or folded in {"environment", "env"}:
                raise SupervisorError(f"{context} contains a sensitive field")
            _reject_sensitive_receipt_value(item, context=context)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive_receipt_value(item, context=context)
    elif isinstance(value, str):
        folded = value.casefold()
        if "axon_b1_recovery_v2_nonce" in folded or "--nonce" in folded:
            raise SupervisorError(f"{context} contains a sensitive value")


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    _reject_sensitive_receipt_value(payload)
    raw = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent(path)
    except FileExistsError as exc:
        raise SupervisorError(f"Supervisor output already exists: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return _sha256(raw)


def _load_json_strict(path: Path) -> tuple[dict[str, Any], bytes]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise SupervisorError(f"Duplicate JSON key in {path}: {key}")
            payload[key] = value
        return payload

    def reject_constant(value: str) -> None:
        raise SupervisorError(f"Non-finite JSON value in {path}: {value}")

    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"Unable to read supervisor receipt: {path}") from exc
    if not isinstance(payload, dict):
        raise SupervisorError(f"Supervisor receipt is not a JSON object: {path}")
    _reject_sensitive_receipt_value(payload)
    return payload, raw


def _canonical_existing_file(path: Path, project_root: Path) -> Path:
    try:
        root = project_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SupervisorError(f"Supervisor source is outside the project: {path}") from exc
    if not resolved.is_file() or path.is_symlink():
        raise SupervisorError(f"Supervisor source is not a regular file: {path}")
    return resolved


def _canonical_output(path: Path, project_root: Path) -> Path:
    try:
        root = project_root.resolve(strict=True)
        parent = path.parent.resolve(strict=True)
        parent.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SupervisorError(f"Supervisor output is outside the project: {path}") from exc
    if path.exists() or path.is_symlink():
        raise SupervisorError(f"Supervisor output already exists: {path}")
    return parent / path.name


def _command_sha256(command: Sequence[str]) -> str:
    return _canonical_json_sha256(list(command))


def _exit_binding_payload(
    config: SupervisorConfig,
    *,
    launch_id: str,
    command_sha256: str,
) -> dict[str, Any]:
    return {
        "exit_schema": EXIT_SCHEMA,
        "launch_id": launch_id,
        "launch_receipt_path": str(config.launch_receipt),
        "exit_receipt_path": str(config.exit_receipt),
        "command_sha256": command_sha256,
    }


def _validate_config(config: SupervisorConfig) -> tuple[Path, dict[str, str]]:
    try:
        project_root = config.project_root.resolve(strict=True)
    except OSError as exc:
        raise SupervisorError("Supervisor project root is unavailable") from exc
    if Path.cwd().resolve(strict=True) != project_root:
        raise SupervisorError("Supervisor working directory is not canonical")
    if (
        not config.command
        or any(not isinstance(item, str) or not item for item in config.command)
        or not math.isfinite(config.timeout_seconds)
        or not 0 < config.timeout_seconds <= MAXIMUM_SUPERVISOR_SECONDS
        or not isinstance(config.maximum_combined_log_bytes, int)
        or isinstance(config.maximum_combined_log_bytes, bool)
        or not 0 < config.maximum_combined_log_bytes <= MAXIMUM_COMBINED_LOG_BYTES
    ):
        raise SupervisorError("Supervisor invocation or timeout is invalid")
    _reject_sensitive_receipt_value(list(config.command), context="command")
    executable = _canonical_existing_file(Path(config.command[0]), project_root)
    if str(executable) != str(Path(config.command[0]).resolve(strict=True)):
        raise SupervisorError("Supervisor executable path drifted")
    source_hashes: dict[str, str] = {}
    names: set[str] = set()
    for name, path in config.source_bindings:
        if not name or name in names or "nonce" in name.casefold():
            raise SupervisorError("Supervisor source binding name is invalid")
        names.add(name)
        source_hashes[name] = _artifact_sha(_canonical_existing_file(path, project_root))
    outputs = {
        _canonical_output(config.launch_receipt, project_root),
        _canonical_output(config.exit_receipt, project_root),
        _canonical_output(config.stdout_log, project_root),
        _canonical_output(config.stderr_log, project_root),
    }
    if len(outputs) != 4:
        raise SupervisorError("Supervisor output paths overlap")
    return project_root, source_hashes


def _sanitized_environment(
    source: Mapping[str, str],
    *,
    launch_id: str,
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in source.items()
        if not key.upper().startswith(SENSITIVE_ENVIRONMENT_PREFIX)
    }
    environment["PYTHONUNBUFFERED"] = "1"
    environment["AXON_B1_RECOVERY_V2_SUPERVISOR_LAUNCH_ID"] = launch_id
    environment["AXON_B1_RECOVERY_V2_SUPERVISOR_PID"] = str(os.getpid())
    return environment


def validate_receipt_pair(
    launch_path: Path,
    exit_path: Path,
    *,
    expected_command: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    launch, launch_raw = _load_json_strict(launch_path)
    exit_payload, _exit_raw = _load_json_strict(exit_path)
    command_sha256 = launch.get("command_sha256")
    expected_binding = {
        "exit_schema": EXIT_SCHEMA,
        "launch_id": launch.get("launch_id"),
        "launch_receipt_path": str(launch_path),
        "exit_receipt_path": str(exit_path),
        "command_sha256": command_sha256,
    }
    if (
        launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("status") != "supervisor_launch_frozen_before_controller_start"
        or exit_payload.get("schema") != EXIT_SCHEMA
        or exit_payload.get("launch_id") != launch.get("launch_id")
        or exit_payload.get("launch_receipt_sha256") != _sha256(launch_raw)
        or launch.get("exit_binding_sha256") != _canonical_json_sha256(expected_binding)
        or exit_payload.get("exit_binding_sha256") != launch.get("exit_binding_sha256")
        or exit_payload.get("command_sha256") != command_sha256
    ):
        raise SupervisorError("Supervisor launch and exit receipts are not mutually bound")
    if expected_command is not None and command_sha256 != _command_sha256(expected_command):
        raise SupervisorError("Supervisor receipt command commitment drifted")
    return launch, exit_payload


def run_supervised(
    config: SupervisorConfig,
    *,
    environment: Mapping[str, str] | None = None,
    job_factory: Callable[[], Any] = WindowsKillOnCloseJob,
) -> dict[str, Any]:
    project_root, source_hashes_before = _validate_config(config)
    launch_id = secrets.token_hex(32)
    command_sha256 = _command_sha256(config.command)
    exit_binding = _canonical_json_sha256(
        _exit_binding_payload(
            config,
            launch_id=launch_id,
            command_sha256=command_sha256,
        )
    )
    launch = {
        "schema": LAUNCH_SCHEMA,
        "loop_id": "loop166_code_section_foundation",
        "status": "supervisor_launch_frozen_before_controller_start",
        "launch_id": launch_id,
        "started_at_utc": _utc_now(),
        "supervisor_pid": os.getpid(),
        "project_root": str(project_root),
        "python_executable": config.command[0],
        "controller_path": config.command[2] if len(config.command) > 2 else "",
        "command_sha256": command_sha256,
        "command_argument_count": len(config.command),
        "timeout_seconds": config.timeout_seconds,
        "maximum_combined_log_bytes": config.maximum_combined_log_bytes,
        "python_unbuffered": True,
        "stdout_log": str(config.stdout_log),
        "stderr_log": str(config.stderr_log),
        "source_bindings": source_hashes_before,
        "expected_exit_receipt": {
            "path": str(config.exit_receipt),
            "schema": EXIT_SCHEMA,
        },
        "exit_binding_sha256": exit_binding,
        "job_object_policy": "windows_kill_on_job_close",
        "raw_access_performed_by_supervisor": False,
    }
    stdout_descriptor = os.open(
        config.stdout_log,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    stderr_descriptor: int | None = None
    launch_sha256: str | None = None
    launch_persisted = False
    controller_returncode: int | None = None
    controller_started = False
    assignment_audit: dict[str, Any] | None = None
    controller_error: str | None = None
    timeout_termination: dict[str, Any] | None = None
    job_audit: dict[str, Any] | None = None
    try:
        stderr_descriptor = os.open(
            config.stderr_log,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        child_environment = _sanitized_environment(
            dict(os.environ) if environment is None else environment,
            launch_id=launch_id,
        )

        def persist_launch_before_resume(
            process: Any,
            observed_assignment: Mapping[str, Any],
        ) -> None:
            nonlocal assignment_audit, controller_started, launch_persisted
            nonlocal launch_sha256
            if (
                observed_assignment.get("assigned_before_resume") is not True
                or observed_assignment.get("process_resumed") is not False
                or observed_assignment.get("exact_limit_flags") != 0x00002000
                or observed_assignment.get("breakaway_allowed") is not False
                or observed_assignment.get("process_pid") != process.pid
            ):
                raise SupervisorError("Job assignment audit is not fail closed")
            assignment_audit = dict(observed_assignment)
            launch_with_assignment = {
                **launch,
                "controller_launcher_pid": process.pid,
                "controller_launcher_creation_time_filetime": observed_assignment.get(
                    "process_creation_time_filetime"
                ),
                "controller_launcher_executable": config.command[0],
                "controller_launcher_semantics": (
                    "windows_venv_redirector_launcher_not_runtime_base_python"
                ),
                "pre_resume_assignment_audit": assignment_audit,
            }
            launch_sha256 = _write_exclusive_json(
                config.launch_receipt,
                launch_with_assignment,
            )
            controller_started = True
            launch_persisted = True

        def enforce_log_cap() -> None:
            combined = (
                os.fstat(stdout_handle.fileno()).st_size + os.fstat(stderr_handle.fileno()).st_size
            )
            if combined > config.maximum_combined_log_bytes:
                raise SupervisorError("Supervisor combined log cap was exceeded")

        with (
            os.fdopen(stdout_descriptor, "wb", buffering=0, closefd=True) as stdout_handle,
            os.fdopen(stderr_descriptor, "wb", buffering=0, closefd=True) as stderr_handle,
        ):
            stdout_descriptor = -1
            stderr_descriptor = -1
            try:
                result = run_subprocess_in_job(
                    config.command,
                    cwd=project_root,
                    env=child_environment,
                    timeout_seconds=config.timeout_seconds,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    job_factory=job_factory,
                    before_resume=persist_launch_before_resume,
                    monitor_callback=enforce_log_cap,
                )
                controller_returncode = result.returncode
                job_audit = result.job_audit
            except WindowsJobTimeoutError as exc:
                timeout_termination = dict(exc.termination)
                controller_error = "WindowsJobTimeoutError: supervised controller timed out"
            except Exception as exc:
                controller_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
            stdout_handle.flush()
            stderr_handle.flush()
            os.fsync(stdout_handle.fileno())
            os.fsync(stderr_handle.fileno())
    finally:
        if stdout_descriptor not in {-1, None}:
            os.close(stdout_descriptor)
        if stderr_descriptor not in {-1, None}:
            os.close(stderr_descriptor)

    if not launch_persisted or launch_sha256 is None:
        raise SupervisorError("Supervisor launch receipt was not persisted before resume")
    observed_launch_sha = _artifact_sha(config.launch_receipt)
    source_hashes_after = {
        name: _artifact_sha(path.resolve(strict=True)) for name, path in config.source_bindings
    }
    source_closure_unchanged = source_hashes_after == source_hashes_before
    launch_receipt_unchanged = observed_launch_sha == launch_sha256
    stdout_sha256 = _artifact_sha(config.stdout_log)
    stderr_sha256 = _artifact_sha(config.stderr_log)
    combined_log_bytes = config.stdout_log.stat().st_size + config.stderr_log.stat().st_size
    logs_within_cap = combined_log_bytes <= config.maximum_combined_log_bytes
    if not launch_receipt_unchanged:
        decision = "supervisor_launch_receipt_drift_fail_closed"
    elif not source_closure_unchanged:
        decision = "supervisor_source_closure_drift_fail_closed"
    elif not logs_within_cap:
        decision = "supervisor_log_cap_exceeded_fail_closed"
    elif timeout_termination is not None:
        decision = "supervisor_timeout_killed_job_tree_fail_closed"
    elif controller_error is not None:
        decision = "supervisor_controller_launch_or_job_failure_fail_closed"
    elif controller_returncode != 0:
        decision = "supervisor_controller_nonzero_exit_fail_closed"
    else:
        decision = "supervisor_controller_zero_exit_and_closure_verified"
    exit_payload = {
        "schema": EXIT_SCHEMA,
        "loop_id": "loop166_code_section_foundation",
        "status": "supervisor_exit_recorded",
        "decision": decision,
        "launch_id": launch_id,
        "finished_at_utc": _utc_now(),
        "supervisor_pid": os.getpid(),
        "controller_started": controller_started,
        "controller_returncode": controller_returncode,
        "controller_error": controller_error,
        "command_sha256": command_sha256,
        "launch_receipt_path": str(config.launch_receipt),
        "launch_receipt_sha256": launch_sha256,
        "observed_launch_receipt_sha256": observed_launch_sha,
        "launch_receipt_unchanged": launch_receipt_unchanged,
        "exit_binding_sha256": exit_binding,
        "source_bindings_before": source_hashes_before,
        "source_bindings_after": source_hashes_after,
        "source_closure_unchanged": source_closure_unchanged,
        "stdout_log": {
            "path": str(config.stdout_log),
            "sha256": stdout_sha256,
            "size_bytes": config.stdout_log.stat().st_size,
        },
        "stderr_log": {
            "path": str(config.stderr_log),
            "sha256": stderr_sha256,
            "size_bytes": config.stderr_log.stat().st_size,
        },
        "combined_log_bytes": combined_log_bytes,
        "maximum_combined_log_bytes": config.maximum_combined_log_bytes,
        "logs_within_cap": logs_within_cap,
        "job_audit": job_audit,
        "pre_resume_assignment_audit": assignment_audit,
        "timeout_termination": timeout_termination,
        "python_unbuffered": True,
        "sensitive_environment_persisted": False,
        "raw_access_performed_by_supervisor": False,
    }
    _write_exclusive_json(config.exit_receipt, exit_payload)
    validate_receipt_pair(
        config.launch_receipt,
        config.exit_receipt,
        expected_command=config.command,
    )
    return exit_payload


def production_config() -> SupervisorConfig:
    command = (
        str(PYTHON_EXECUTABLE.resolve(strict=True)),
        "-u",
        str(CONTROLLER.resolve(strict=True)),
        "--contract",
        str(CONTRACT.resolve(strict=True)),
        "--authorization",
        str(AUTHORIZATION.resolve(strict=True)),
        "--folds",
        str(FOLDS.resolve(strict=True)),
        "--folds-summary",
        str(FOLDS_SUMMARY.resolve(strict=True)),
        "--data-root",
        str(DATA_ROOT.resolve(strict=True)),
        "--source-tokenizer",
        str(SOURCE_TOKENIZER.resolve(strict=True)),
        "--source-checkpoint",
        str(SOURCE_CHECKPOINT.resolve(strict=True)),
        "--checkpoint-output",
        str(CHECKPOINT_OUTPUT.absolute()),
        "--report-output",
        str(REPORT_OUTPUT.absolute()),
    )
    return SupervisorConfig(
        project_root=PROJECT_ROOT,
        command=command,
        launch_receipt=LAUNCH_RECEIPT,
        exit_receipt=EXIT_RECEIPT,
        stdout_log=STDOUT_LOG,
        stderr_log=STDERR_LOG,
        source_bindings=(
            ("contract", CONTRACT),
            ("authorization", AUTHORIZATION),
            ("controller", CONTROLLER),
            ("supervisor", Path(__file__)),
            ("windows_job", WINDOWS_JOB_SOURCE),
            ("windows_process_lineage", WINDOWS_LINEAGE_SOURCE),
            ("raw_progress_ledger", RAW_LEDGER_SOURCE),
            ("controller_tests", CONTROLLER_TESTS),
            ("windows_job_tests", WINDOWS_JOB_TESTS),
            ("supervisor_tests", SUPERVISOR_TESTS),
            ("powershell_launcher", POWERSHELL_LAUNCHER),
        ),
        timeout_seconds=MAXIMUM_SUPERVISOR_SECONDS,
        maximum_combined_log_bytes=MAXIMUM_COMBINED_LOG_BYTES,
    )


def main(argv: Sequence[str] | None = None) -> int:
    if list(sys.argv[1:] if argv is None else argv):
        raise SupervisorError("Detached recovery supervisor accepts no arguments")
    if os.name != "nt":
        raise SupervisorError("Detached recovery supervisor requires Windows")
    result = run_supervised(production_config())
    return 0 if result["decision"] == "supervisor_controller_zero_exit_and_closure_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
