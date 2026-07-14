from __future__ import annotations

from src.loop167_phase_b.resource_guard import (
    SystemResourceSnapshot,
    build_resource_guard_payload,
    evaluate_resource_guard,
    minimum_available_memory_bytes,
)

RESOURCE_CONTRACT = {
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
}


def test_resource_guard_requires_closed_budget_and_memory_headroom() -> None:
    minimum = minimum_available_memory_bytes(RESOURCE_CONTRACT)
    passing = evaluate_resource_guard(
        SystemResourceSnapshot(total_memory_bytes=32 * 1024**3, available_memory_bytes=minimum, cpu_count=4),
        RESOURCE_CONTRACT,
    )
    failing = evaluate_resource_guard(
        SystemResourceSnapshot(total_memory_bytes=32 * 1024**3, available_memory_bytes=minimum - 1, cpu_count=4),
        RESOURCE_CONTRACT,
    )

    assert passing.ready is True
    assert failing.ready is False
    assert failing.failures == ("available_memory_below_launch_floor",)


def test_resource_guard_binds_static_inputs_without_raw_access() -> None:
    payload = build_resource_guard_payload(
        source_closure_binding={"path": "closure.json", "sha256": "a" * 64},
        protocol_binding={"path": "protocol.json", "sha256": "b" * 64},
        runtime_lock_binding={"path": "runtime.json", "sha256": "c" * 64},
        controller_binding={"path": "controller.py", "sha256": "d" * 64},
        canonical_argv=("python", "-I", "controller.py", "--preflight"),
        resource_contract=RESOURCE_CONTRACT,
        snapshot=SystemResourceSnapshot(total_memory_bytes=32 * 1024**3, available_memory_bytes=16 * 1024**3, cpu_count=4),
        created_at_utc="2026-07-13T00:00:00Z",
    )

    assert payload["guard_ready"] is True
    assert payload["raw_open_attempts"] == 0
    assert payload["maximum_age_seconds"] == 300
