from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.loop167_phase_b.contracts import PhaseBContractError, canonical_json_bytes, sha256_file
from src.loop167_phase_b.execution_authorization import validate_execution_authorization


def _write_canonical(path, payload) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def test_execution_authorization_requires_fresh_guard_and_new_outputs(tmp_path) -> None:
    closure = tmp_path / "closure.json"
    runtime = tmp_path / "runtime.json"
    controller = tmp_path / "controller.py"
    protocol = tmp_path / "protocol.json"
    controller.write_text("print('synthetic')\n", encoding="ascii")
    _write_canonical(closure, {"schema": "closure"})
    _write_canonical(runtime, {"schema": "runtime"})
    _write_canonical(protocol, {"schema": "protocol"})
    source_binding = {"path": "closure.json", "sha256": sha256_file(closure)}
    runtime_binding = {"path": "runtime.json", "sha256": sha256_file(runtime)}
    controller_binding = {"path": "controller.py", "sha256": sha256_file(controller)}
    argv = ("python", "-I", "controller.py", "--preflight")
    guard = tmp_path / "guard.json"
    _write_canonical(
        guard,
        {
            "schema": "axon_loop167_phase_b_resource_guard_v1",
            "loop_id": "loop167_ember_v3_novel_delta",
            "source_closure": source_binding,
            "phase_b_protocol": {"path": "protocol.json", "sha256": sha256_file(protocol)},
            "runtime_lock": runtime_binding,
            "controller": controller_binding,
            "canonical_argv": list(argv),
            "canonical_argv_sha256": "e0fc7eb3c9a15b20ff2f383c7972d2586b9be77e2d0d37f4a33e80bdced02f09",
            "resource_contract": {
                "maximum_raw_open_attempts": 20000,
                "maximum_raw_bytes": 26843545600,
                "maximum_feature_cache_bytes": 1073741824,
                "maximum_extraction_peak_rss_bytes": 4294967296,
                "maximum_training_peak_rss_bytes": 8589934592,
                "maximum_extraction_wall_seconds": 6000,
                "maximum_training_wall_seconds": 18000,
                "reserved_seal_evaluation_wall_seconds": 4800,
                "maximum_total_wall_seconds": 28800,
                "worker_count": 1,
                "thread_count": 1,
                "maximum_gpu_allocated_bytes": 0,
                "kill_conditions": ["oom"],
            },
            "created_at_utc": "2026-07-13T00:00:00Z",
            "maximum_age_seconds": 300,
            "snapshot": {"total_memory_bytes": 1, "available_memory_bytes": 1, "cpu_count": 1},
            "minimum_available_memory_bytes": 1,
            "guard_ready": True,
            "failures": [],
            "decision": "pass",
            "raw_open_attempts": 0,
        },
    )
    authorization = tmp_path / "authorization.json"
    _write_canonical(
        authorization,
        {
            "schema": "axon_loop167_phase_b_run_authorization_v1",
            "loop_id": "loop167_ember_v3_novel_delta",
            "claim_scope": "single_train_only_raw_pass_then_fixed_oof_not_promotion_or_heldout_evaluation",
            "status": "authorized_pending_one_shot_lease",
            "execution_authorization_granted": True,
            "source_closure": source_binding,
            "runtime_lock": runtime_binding,
            "controller": controller_binding,
            "canonical_argv": list(argv),
            "canonical_argv_sha256": "e0fc7eb3c9a15b20ff2f383c7972d2586b9be77e2d0d37f4a33e80bdced02f09",
            "resource_guard": {"path": "guard.json", "sha256": sha256_file(guard)},
            "lease": {
                "lease_id": "loop167-phase-b-train-oof-v1",
                "marker_path": "lease.json",
                "consume_before_first_raw_open": True,
                "retry_allowed": False,
            },
            "outputs": ["features.npz", "raw.jsonl", "fit.jsonl", "receipt.json"],
            "ready_for": {"raw_access": True, "fit": True, "val": False, "test10k": False, "legacy_full_test": False, "promotion": False},
            "forbidden": ["val_test10k_legacy_full_sentinel_or_sealed_window_access"],
        },
    )

    with pytest.raises(PhaseBContractError, match="argv hash"):
        validate_execution_authorization(
            tmp_path,
            authorization,
            expected_source_closure=source_binding,
            expected_runtime_lock=runtime_binding,
            expected_controller=controller_binding,
            canonical_argv=argv,
            now_utc=datetime(2026, 7, 13, 0, 1, tzinfo=UTC),
        )
