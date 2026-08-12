#!/usr/bin/env python3
"""Authorize and resume the Loop175 full-Train identity-free region cache."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.phase_b_cache_builder import (  # noqa: E402
    build_region_cache,
    load_region_sources,
    source_plan_commitment,
)
from src.loop175.phase_b_contract import (  # noqa: E402
    load_json_object,
    load_phase_b_protocol,
    sha256_file,
    validate_bound_evidence,
    write_exclusive_json,
)
from src.loop175.phase_b_source_closure import (  # noqa: E402
    validate_phase_b_source_closure,
)


def _available_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        return 0


def _gpu_free_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            return int(free_bytes)
    except ImportError:
        pass
    return 0


def build_preflight(output_root: Path) -> dict[str, object]:
    memory_available = _available_memory_bytes()
    disk_free = int(shutil.disk_usage(output_root).free)
    gpu_free = _gpu_free_bytes()
    blockers = []
    if memory_available and memory_available < 2 * 1024**3:
        blockers.append("available_memory_below_2_GiB")
    if disk_free < 4 * 1024**3:
        blockers.append("output_disk_free_below_4_GiB")
    return {
        "schema": "axon_loop175_region_cache_preflight_v1",
        "claim_scope": "runtime_resource_preflight_before_train_raw_open",
        "available_memory_bytes": memory_available,
        "output_disk_free_bytes": disk_free,
        "gpu_free_bytes": gpu_free,
        "minimum_available_memory_bytes": 2 * 1024**3,
        "minimum_output_disk_free_bytes": 4 * 1024**3,
        "blockers": blockers,
        "decision": "cache_run_authorized" if not blockers else "cache_run_blocked",
    }


def _load_or_create_exact(path: Path, payload: Mapping[str, object], *, label: str) -> None:
    if path.exists():
        observed = load_json_object(path)
        if observed != dict(payload):
            raise RuntimeError(f"Loop175 {label} does not match this resume scope")
        return
    write_exclusive_json(path, payload)


def run_cache(
    *,
    source_closure_path: Path,
    fold_manifest: Path,
    staging_directory: Path,
    output_cache: Path,
    output_receipt: Path,
    authorization_path: Path,
    lease_path: Path,
    block_rows: int,
) -> dict[str, object]:
    protocol = load_phase_b_protocol(PROJECT_ROOT)
    validate_bound_evidence(PROJECT_ROOT, protocol)
    closure = load_json_object(source_closure_path)
    validate_phase_b_source_closure(PROJECT_ROOT, closure)
    closure_sha256 = sha256_file(source_closure_path)
    if output_receipt.exists():
        raise RuntimeError("Loop175 region-cache receipt already exists")

    preflight = build_preflight(output_cache.parent)
    if preflight["blockers"]:
        raise RuntimeError(f"Loop175 cache preflight failed: {preflight['blockers']}")
    authorization = {
        "schema": "axon_loop175_region_cache_run_authorization_v1",
        "loop_id": "Loop175",
        "protocol_sha256": protocol.sha256,
        "source_closure_sha256": closure_sha256,
        "preflight": preflight,
        "raw_scope": "exact_20000_row_train_fold_authority_only",
        "val_test_or_full_access_allowed": False,
        "decision": "authorize_resumable_full_train_region_cache",
    }
    _load_or_create_exact(authorization_path, authorization, label="cache authorization")

    sources = load_region_sources(fold_manifest)
    plan_commitment = source_plan_commitment(sources)
    lease = {
        "schema": "axon_loop175_region_cache_lease_v1",
        "loop_id": "Loop175",
        "protocol_sha256": protocol.sha256,
        "source_closure_sha256": closure_sha256,
        "source_plan_commitment": plan_commitment,
        "rows": len(sources),
        "resume_allowed": True,
        "scope_change_allowed": False,
        "val_test_or_full_access_allowed": False,
        "status": "consumed_before_first_raw_open",
    }
    _load_or_create_exact(lease_path, lease, label="cache lease")

    result = build_region_cache(
        sources,
        output_cache=output_cache,
        staging_directory=staging_directory,
        block_rows=block_rows,
    )
    payload: dict[str, object] = {
        "schema": "axon_loop175_full_train_region_cache_receipt_v1",
        "loop_id": "Loop175",
        "claim_scope": "full_train_region_cache_coverage_and_resource_evidence_not_model_quality",
        "decision": result.decision,
        "inputs": {
            "protocol_sha256": protocol.sha256,
            "source_closure_sha256": closure_sha256,
            "authorization_sha256": sha256_file(authorization_path),
            "lease_sha256": sha256_file(lease_path),
            "source_plan_commitment": plan_commitment,
            "fold_manifest_sha256": closure["input_bindings"]["fold_manifest"]["sha256"],
            "rows": len(sources),
            "split_role": "train",
        },
        "cache": {
            "path": str(result.cache.path),
            "sha256": result.cache.sha256,
            "bytes": result.cache.size_bytes,
            "rows": result.cache.row_count,
            "regions": result.cache.region_count,
            "tokens": result.cache.token_count,
            "identity_fields_persisted": False,
        },
        "progress": {
            "ledger_path": str(result.ledger_path),
            "ledger_sha256": result.ledger_sha256,
            "final_record_sha256": result.final_record_sha256,
        },
        "coverage": {
            "attempted": result.attempted,
            "supported": result.supported,
            "coverage": result.supported / result.attempted,
            "class_coverage": dict(result.class_coverage),
            "class_coverage_gap": result.class_coverage_gap,
            "silent_drops": result.silent_drops,
            "status_counts": dict(result.status_counts),
        },
        "resources": {
            "source_bytes_verified": result.source_bytes_verified,
            "maximum_rss_bytes": result.maximum_rss_bytes,
            "maximum_new_disk_bytes": result.maximum_new_disk_bytes,
        },
        "blockers": list(result.blockers),
        "raw_train_rows_opened": result.attempted,
        "val_rows_opened": 0,
        "test10k_rows_opened": 0,
        "full_test_rows_opened": 0,
        "val_test_or_full_rows_opened": 0,
        "training_runs": 0,
    }
    write_exclusive_json(output_receipt, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-closure", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--staging-directory", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, default=64)
    arguments = parser.parse_args()
    payload = run_cache(
        source_closure_path=arguments.source_closure.resolve(),
        fold_manifest=arguments.fold_manifest.resolve(),
        staging_directory=arguments.staging_directory.resolve(),
        output_cache=arguments.output_cache.resolve(),
        output_receipt=arguments.output_receipt.resolve(),
        authorization_path=arguments.authorization.resolve(),
        lease_path=arguments.lease.resolve(),
        block_rows=arguments.block_rows,
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if not payload["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

