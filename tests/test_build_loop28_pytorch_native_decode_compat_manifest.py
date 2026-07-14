from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_loop28_pytorch_native_decode_compat_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("loop28_decode_compat_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_successor_base_chain_binds_negative_parent() -> None:
    module = _load_module()
    root = Path(__file__).resolve().parents[1]
    chain = module._verify_base_chain(root)
    assert chain["proposal"]["parent_closure"]["sha256"] == (module.EXPECTED_PARENT_POST_SHA256)
    assert chain["proposal"]["parent_failure_manifest"]["sha256"] == (
        module.EXPECTED_PARENT_FAILURE_SHA256
    )
    assert chain["proposal"]["claim_boundary"]["package_load_allowed"] is False
    assert chain["proposal"]["stage_budgets"] == {
        "preflight": module.PREFLIGHT_BUDGET,
        "package": module.PACKAGE_BUDGET,
    }
    assert (
        chain["authorization"]["terminal_evidence_contract"]
        == module.TERMINAL_EVIDENCE_CONTRACT
    )


def test_official_v213_source_contract_is_pinned() -> None:
    module = _load_module()
    root = Path(__file__).resolve().parents[1]
    payload = module.build_official_research_manifest(
        root, generated_at_utc="2026-07-12T00:20:00Z"
    )

    assert payload["official_source"]["tag"] == "v2.13.0"
    assert payload["official_source"]["tag_commit"] == (
        "cf30153c4c131c8164ee7798e5022d810682e2cb"
    )
    assert payload["official_source"]["source_payload_sha256"] == (
        "cb369b351ac1021ecd6127e536ef35c1a6d57de6ae3689e95d097c9ab3ebad02"
    )
    assert payload["contract"]["protected_stage_network_requests"] == 0
    assert payload["contract"]["actual_v213_runtime_qualified"] is False
    assert module.verify_official_research_manifest(root)["official_source"] == payload[
        "official_source"
    ]


def test_reused_cpp_binaries_are_artifact_only() -> None:
    module = _load_module()
    root = Path(__file__).resolve().parents[1]
    payload = module.build_reused_binaries_manifest(
        root, generated_at_utc="2026-07-12T00:20:00Z"
    )

    assert [row["sha256"] for row in payload["binaries"]] == [
        module.EXPECTED_ATEN_HOST_SHA256,
        module.EXPECTED_AOTI_HOST_SHA256,
    ]
    assert payload["authority_boundary"]["artifact_hash_reuse_allowed"] is True
    assert payload["authority_boundary"]["parent_authorization_reuse_allowed"] is False
    assert payload["authority_boundary"]["parent_ready_or_consumed_lease_reuse_allowed"] is False
    assert module.verify_reused_binaries_manifest(root)["binaries"] == payload["binaries"]


def test_timestamp_validation_fails_closed() -> None:
    module = _load_module()
    for value in ("", "2026-07-12T00:00:00", "not-a-timeZ"):
        with pytest.raises(module.DecodeCompatManifestError):
            module._validate_timestamp(value)


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "duplicate.json"
    path.write_text('{"decision": "one", "decision": "two"}', encoding="utf-8")
    with pytest.raises(module.DecodeCompatManifestError, match="Duplicate JSON key"):
        module.load_json_strict(path)


def test_strict_json_rejects_nonfinite_numbers(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "nonfinite.json"
    path.write_text('{"wall_clock_seconds": NaN}', encoding="utf-8")
    with pytest.raises(module.DecodeCompatManifestError, match="Non-finite JSON"):
        module.load_json_strict(path)


def test_preflight_builder_waits_for_lease_gated_evidence() -> None:
    module = _load_module()
    root = Path(__file__).resolve().parents[1]
    if (root / module.PREFLIGHT_EVIDENCE).is_file():
        pytest.skip("Preflight already executed; verification is covered by the live manifest")
    with pytest.raises((module.DecodeCompatManifestError, FileNotFoundError)):
        module.build_preflight_manifest(root, generated_at_utc="2026-07-12T00:20:00Z")


def test_source_binding_requires_auth_evidence_and_current_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    names = (
        "runner.py",
        "runner_test.py",
        "builder.py",
        "builder_test.py",
        "base.py",
        "safety.py",
        "aten.exe",
        "aoti.exe",
        "official.json",
        "reuse.json",
    )
    paths = [Path(name) for name in names]
    for index, path in enumerate(paths):
        (tmp_path / path).write_bytes(f"artifact-{index}".encode())
    for attribute, path in zip(
        (
            "RUNNER",
            "RUNNER_TEST",
            "BUILDER",
            "BUILDER_TEST",
            "BASE_MODEL_SOURCE",
            "PARENT_SAFETY_SOURCE",
            "ATEN_HOST",
            "AOTI_HOST",
            "OFFICIAL_RESEARCH_MANIFEST",
            "REUSED_BINARIES_MANIFEST",
        ),
        paths,
        strict=True,
    ):
        monkeypatch.setattr(module, attribute, path)
    records = [module._artifact_record_by_path(tmp_path, path) for path in paths]
    authorization = {"source_artifacts": records}

    bound = module._verify_source_binding(
        tmp_path,
        authorization,
        embedded_authorization=authorization,
    )
    assert len(bound) == len(paths)

    with pytest.raises(module.DecodeCompatManifestError, match="contain only"):
        module._index_source_records([dict(records[0], name="unexpected")])

    (tmp_path / paths[0]).write_bytes(b"drift")
    with pytest.raises(module.DecodeCompatManifestError, match="source binding drifted"):
        module._verify_source_binding(tmp_path, authorization)


def test_stage_chain_binds_attempt_command_roots_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    artifact_root = Path("artifacts/tiny_v2/package_attempt_001")
    work_root = Path("reports/work/decode_probe_attempt_001")
    authorization_path = Path("manifests/preflight_authorization.json")
    lease_path = Path("manifests/preflight_lease.final.json")
    outputs = (Path("reports/success.json"), Path("reports/failure.json"))
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    budget = dict(module.PREFLIGHT_BUDGET)
    command = [
        "E:/Project/python/Axon_v2.6Exp/vnev/Scripts/python.exe",
        module.RUNNER.as_posix(),
        "preflight",
    ]
    authorization = {
        "schema": module.PREFLIGHT_AUTHORIZATION_SCHEMA,
        "loop_id": module.LOOP_ID,
        "attempt_id": module.PREFLIGHT_ATTEMPT_ID,
        "decision": module.PREFLIGHT_AUTHORIZATION_DECISION,
        "canonical_command": command,
        "artifact_root": artifact_root.as_posix(),
        "work_root": work_root.as_posix(),
        "terminal_outputs": [path.as_posix() for path in outputs],
        "budget": budget,
    }
    authorization_file = tmp_path / authorization_path
    authorization_file.parent.mkdir(parents=True)
    authorization_file.write_text(json.dumps(authorization), encoding="utf-8")
    lease = {
        "schema": module.PREFLIGHT_LEASE_SCHEMA,
        "loop_id": module.LOOP_ID,
        "lease_id": module.PREFLIGHT_ATTEMPT_ID,
        "attempt_id": module.PREFLIGHT_ATTEMPT_ID,
        "status": "consumed_before_execution",
        "single_use": True,
        "authorization_path": authorization_path.as_posix(),
        "authorization_sha256": module.sha256_file(authorization_file),
        "canonical_command": command,
        "artifact_root": artifact_root.as_posix(),
        "work_root": work_root.as_posix(),
        "terminal_outputs": [path.as_posix() for path in outputs],
        "budget_sha256": module._canonical_sha256(budget),
        "consumed_path": lease_path.as_posix(),
        "original_lease_sha256": "a" * 64,
        "consumed_at_utc": "2026-07-12T00:21:00Z",
    }
    lease_file = tmp_path / lease_path
    lease_file.write_text(json.dumps(lease), encoding="utf-8")

    checked_authorization, checked_lease = module._validate_stage_chain(
        tmp_path,
        authorization_path=authorization_path,
        final_lease_path=lease_path,
        authorization_schema=module.PREFLIGHT_AUTHORIZATION_SCHEMA,
        authorization_decision=module.PREFLIGHT_AUTHORIZATION_DECISION,
        lease_schema=module.PREFLIGHT_LEASE_SCHEMA,
        attempt_id=module.PREFLIGHT_ATTEMPT_ID,
        mode="preflight",
        work_root=work_root,
        output_paths=outputs,
    )
    assert checked_authorization == authorization
    assert checked_lease == lease

    lease["budget_sha256"] = "b" * 64
    lease_file.write_text(json.dumps(lease), encoding="utf-8")
    with pytest.raises(module.DecodeCompatManifestError, match="budget digest"):
        module._validate_stage_chain(
            tmp_path,
            authorization_path=authorization_path,
            final_lease_path=lease_path,
            authorization_schema=module.PREFLIGHT_AUTHORIZATION_SCHEMA,
            authorization_decision=module.PREFLIGHT_AUTHORIZATION_DECISION,
            lease_schema=module.PREFLIGHT_LEASE_SCHEMA,
            attempt_id=module.PREFLIGHT_ATTEMPT_ID,
            mode="preflight",
            work_root=work_root,
            output_paths=outputs,
        )


def _valid_launch(*, process_started: bool = True) -> dict[str, object]:
    return {
        "process_started": process_started,
        "returncode": 0 if process_started else None,
        "timed_out": False,
        "launch_error": None if process_started else "worker launch failed",
        "worker_parse_error": None,
        "process_tree_termination": None,
        "windows_job_object": {
            "created": process_started,
            "assigned": process_started,
            "gate_released": process_started,
            "active_processes_after": 0 if process_started else None,
            "closed": process_started,
        },
        "temp_cleanup": {"removed": True},
        "console_code_pages": {
            "before": {"input": 936, "output": 936},
            "configured": {"input": 65001, "output": 65001},
            "restored": {"input": 936, "output": 936},
            "restore_verified": True,
        },
    }


def _valid_worker(module, *, torch_imports: int = 1) -> dict[str, object]:
    counters = {name: 0 for name in module.WORKER_COUNTER_NAMES}
    counters.update(
        {
            "torch_imports": torch_imports,
            "total_subprocesses": 2 if torch_imports else 0,
            "compiler_processes": 2 if torch_imports else 0,
            "compiler_help_processes": 2 if torch_imports else 0,
        }
    )
    worker: dict[str, object] = {
        "counters": counters,
        "process_telemetry": {
            "total_subprocesses": 2 if torch_imports else 0,
            "compiler_processes": counters["compiler_processes"],
            "compiler_help_processes": counters["compiler_help_processes"],
            "dumpbin_processes": 0,
        },
    }
    if torch_imports:
        worker.update(
            {
                "environment": {
                    "python_executable_sha256": module.EXPECTED_PYTHON_SHA256,
                    "utf8_mode": 0,
                    "preferred_encoding": "cp936",
                    "console_code_pages": {"input": 65001, "output": 65001},
                    "torchinductor_compile_threads_env": 1,
                    "torchinductor_autotune_in_subproc_env": "0",
                },
                "torch": {
                    "version": module.EXPECTED_TORCH_VERSION,
                    "cuda_initialized_before": False,
                    "cuda_initialized_after": False,
                    "cpp_builder_sha256_after": module.EXPECTED_CPP_BUILDER_SHA256,
                    "cpu_vec_isa_sha256_after": module.EXPECTED_CPU_VEC_ISA_SHA256,
                },
                "shim": {
                    "process_local": True,
                    "installed_file_modified": False,
                    "installed_file_sha256_before": module.EXPECTED_CPP_BUILDER_SHA256,
                    "after_args": ["cp936", "replace"],
                    "watched_modules_before": {"torch": False},
                    "compiler_modules_after_torch_import": {"cpp_builder": False},
                    "cache_sizes_before": {"is_msvc_cl": 0},
                },
            }
        )
    return worker


def _journal_record(
    module,
    *,
    stage: str,
    sequence: int,
    event: str,
    previous_sha256: str | None,
    counters: dict[str, int],
) -> dict[str, object]:
    body = {
        "schema": "axon_loop28_pytorch_native_decode_worker_event_v1",
        "loop_id": module.LOOP_ID,
        "stage": stage,
        "sequence": sequence,
        "event": event,
        "previous_record_sha256": previous_sha256,
        "counters": counters,
    }
    return {**body, "record_sha256": module._canonical_sha256(body)}


def test_terminal_integrity_requires_cleanup_restore_and_exact_snapshots() -> None:
    module = _load_module()
    source_records = [{"path": "runner.py", "size_bytes": 4, "sha256": "a" * 64}]
    terminal = {
        "launch": _valid_launch(),
        "work_root_cleanup": {"removed": True},
        "work_root_cleanup_error": None,
        "source_artifacts_before": source_records,
        "source_artifacts_after": source_records,
    }

    module._validate_launch_integrity(terminal, success=True)
    module._validate_source_snapshots(terminal, source_records, success=True)

    terminal["source_artifacts_after"] = []
    with pytest.raises(module.DecodeCompatManifestError, match="source-after"):
        module._validate_source_snapshots(terminal, source_records, success=False)

    terminal["source_artifacts_after"] = source_records
    terminal["launch"]["console_code_pages"]["restore_verified"] = False
    with pytest.raises(module.DecodeCompatManifestError, match="not restored"):
        module._validate_launch_integrity(terminal, success=True)

    terminal["launch"]["console_code_pages"]["restore_verified"] = True
    terminal["launch"].update(
        {
            "returncode": 1,
            "timed_out": True,
            "process_tree_termination": {
                "method": "windows_job_object_terminate",
                "job_assigned": True,
                "tree_termination_requested": True,
                "tree_termination_confirmed": False,
                "active_processes_after": 0,
            },
        }
    )
    with pytest.raises(module.DecodeCompatManifestError, match="termination proof"):
        module._validate_launch_integrity(terminal, success=False)

    terminal["launch"]["process_tree_termination"]["tree_termination_confirmed"] = True
    module._validate_launch_integrity(terminal, success=False)


def test_worker_runtime_and_budget_reconcile_with_frozen_evidence() -> None:
    module = _load_module()
    worker = _valid_worker(module)
    terminal = {
        "worker": worker,
        "launch": _valid_launch(),
        "budget_actual": {
            "worker_processes": 1,
            "vcvars_activations": 1,
            "compiler_help_processes": 2,
            "dumpbin_processes": 0,
            "wall_clock_seconds": 3.5,
            "retained_output_bytes": 0,
        },
    }

    module._validate_worker_runtime(worker, success=True)
    actual = module._validate_preflight_budget_actual(
        terminal, module.PREFLIGHT_BUDGET, success=True
    )
    assert actual["compiler_help_processes"] == 2

    terminal["budget_actual"]["compiler_help_processes"] = 1
    with pytest.raises(module.DecodeCompatManifestError, match="worker counters"):
        module._validate_preflight_budget_actual(
            terminal, module.PREFLIGHT_BUDGET, success=True
        )


def test_administrative_failure_requires_structured_zero_worker_receipt() -> None:
    module = _load_module()
    worker = _valid_worker(module, torch_imports=0)
    launch = _valid_launch(process_started=False)
    launch.pop("windows_job_object")
    launch["console_code_pages"]["configured"] = None
    launch["durable_worker_journal"] = {
        "record_count": 0,
        "records": [],
        "last_counters": worker["counters"],
    }
    terminal = {
        "worker": worker,
        "launch": launch,
        "work_root_cleanup": {"removed": True},
        "work_root_cleanup_error": None,
    }

    module._validate_launch_integrity(terminal, success=False)
    module._validate_worker_runtime(worker, success=False)

    terminal["worker"] = {}
    with pytest.raises(module.DecodeCompatManifestError, match="omitted counters"):
        module._validate_worker_runtime(terminal["worker"], success=False)


def test_taskkill_fallback_is_pre_gate_and_zero_journal_only() -> None:
    module = _load_module()
    worker = _valid_worker(module, torch_imports=0)
    launch = _valid_launch()
    launch.update(
        {
            "returncode": 1,
            "process_tree_termination": {
                "method": "taskkill_before_job_gate_release",
                "job_assigned": False,
                "tree_termination_requested": True,
                "tree_termination_confirmed": True,
            },
            "durable_worker_journal": {
                "record_count": 0,
                "records": [],
                "last_counters": worker["counters"],
            },
            "windows_job_object": {
                "created": True,
                "assigned": False,
                "gate_released": False,
                "active_processes_after": None,
                "closed": True,
            },
        }
    )
    terminal = {
        "worker": worker,
        "launch": launch,
        "work_root_cleanup": {"removed": True},
        "work_root_cleanup_error": None,
    }

    module._validate_launch_integrity(terminal, success=False)

    launch["windows_job_object"].update(
        {"assigned": True, "gate_released": True, "active_processes_after": 0}
    )
    with pytest.raises(module.DecodeCompatManifestError, match="taskkill fallback"):
        module._validate_launch_integrity(terminal, success=False)


def test_reconstructed_worker_receipt_binds_durable_journal() -> None:
    module = _load_module()
    initial = {name: 0 for name in module.WORKER_COUNTER_NAMES}
    imported = dict(initial, torch_imports=1)
    first = _journal_record(
        module,
        stage="package",
        sequence=1,
        event="worker_started",
        previous_sha256=None,
        counters=initial,
    )
    second = _journal_record(
        module,
        stage="package",
        sequence=2,
        event="torch_import_about_to_start",
        previous_sha256=first["record_sha256"],
        counters=imported,
    )
    worker = _valid_worker(module, torch_imports=0)
    worker.update(
        {
            "status": "failed",
            "counters": imported,
            "receipt_reconstructed_from_journal": True,
            "durable_journal_last_record_sha256": second["record_sha256"],
        }
    )
    launch = _valid_launch()
    launch.update(
        {
            "returncode": 1,
            "durable_worker_journal": {
                "record_count": 2,
                "records": [first, second],
                "last_event": second["event"],
                "last_record_sha256": second["record_sha256"],
                "last_counters": imported,
            },
        }
    )
    terminal = {"launch": launch}

    reconstructed = module._validate_durable_worker_journal(
        terminal,
        worker,
        stage="package",
        success=False,
    )
    assert reconstructed is True
    module._validate_worker_runtime(worker, success=False, reconstructed=True)

    second["record_sha256"] = "f" * 64
    with pytest.raises(module.DecodeCompatManifestError, match="hash chain"):
        module._validate_durable_worker_journal(
            terminal,
            worker,
            stage="package",
            success=False,
        )


def test_journal_corruption_cannot_fall_back_to_zero_counters() -> None:
    module = _load_module()
    worker = _valid_worker(module, torch_imports=0)
    worker.update(
        {
            "status": "failed",
            "receipt_reconstructed_from_journal": False,
            "durable_journal_last_record_sha256": None,
        }
    )
    launch = _valid_launch(process_started=False)
    launch["durable_worker_journal"] = {
        "record_count": 0,
        "records": [],
        "last_counters": worker["counters"],
        "integrity_error": "journal record hash mismatch",
    }
    with pytest.raises(module.DecodeCompatManifestError, match="integrity failed"):
        module._validate_durable_worker_journal(
            {"launch": launch},
            worker,
            stage="preflight",
            success=False,
        )


@pytest.mark.parametrize(
    ("status", "updates", "error", "expected"),
    [
        ("failed", {}, "worker launch failed", "administrative"),
        ("failed", {"torch_imports": 1}, "export setup failed", "pre_export"),
        (
            "failed",
            {"torch_imports": 1, "model_constructions": 1, "torch_export_calls": 1,
             "aoti_compile_and_package_calls": 1},
            "compiler failed",
            "protected_call",
        ),
        (
            "passed",
            {key: value for key, value in _load_module().PACKAGE_COUNTER_LIMITS.items()},
            "dependency closure has forbidden hits",
            "dependency",
        ),
        (
            "passed",
            {key: value for key, value in _load_module().PACKAGE_COUNTER_LIMITS.items()},
            "archive collision",
            "static_audit",
        ),
        (
            "passed",
            {key: value for key, value in _load_module().PACKAGE_COUNTER_LIMITS.items()},
            "Dependency audit exhausted its dumpbin process budget",
            "budget",
        ),
        (
            "failed",
            {"aoti_compile_and_package_calls": 1},
            "TimeoutExpired: protected worker",
            "budget",
        ),
    ],
)
def test_package_failure_classification(
    status: str, updates: dict[str, int], error: str, expected: str
) -> None:
    module = _load_module()
    counters = {key: 0 for key in module.PACKAGE_COUNTER_LIMITS}
    counters.update(updates)
    failure = {"worker": {"status": status, "counters": counters}, "error": error}
    assert module._classify_package_failure(failure) == expected


def test_package_timeout_launch_has_budget_precedence() -> None:
    module = _load_module()
    counters = {key: 0 for key in module.PACKAGE_COUNTER_LIMITS}
    failure = {
        "worker": {"status": "failed", "counters": counters},
        "launch": {"timed_out": True},
        "error": "worker terminated",
    }
    assert module._classify_package_failure(failure) == "budget"


def test_partial_artifacts_cannot_escape_or_reuse_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    artifact_root = Path("artifacts/successor/tiny_v2/package_attempt_001")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    artifact = tmp_path / artifact_root / "input.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fresh")
    record = {
        "path": artifact_root.joinpath("input.bin").as_posix(),
        "sha256": module.sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
    }
    assert module._validate_partial_artifacts(tmp_path, [record]) == [
        ("partial_0", artifact_root / "input.bin")
    ]

    escaped = dict(record, path="artifacts/parent/tiny_v1/input.bin")
    with pytest.raises(module.DecodeCompatManifestError, match="escapes"):
        module._validate_partial_artifacts(tmp_path, [escaped])

    with pytest.raises(module.DecodeCompatManifestError, match="incomplete or unsafe"):
        module._validate_failure_partials(
            tmp_path,
            {
                "partial_artifacts": [],
                "partial_inventory_error": "reparse point prevented inventory",
            },
        )
