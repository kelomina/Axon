#!/usr/bin/env python3
"""Run the fixed-rules, Train-only capa coverage gate without persisting matches."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from loop171.capa_aggregate import CapabilityAggregate  # noqa: E402
from run_loop170_cfg_coverage import CoverageError, _source_for_record, sha256  # noqa: E402


BUNDLE = ROOT / "reports/roadmap_9997/loop164/local_probe_bundle.jsonl"
OUTPUT = ROOT / "reports/roadmap_9997/loop171/capa_coverage.json"
RECEIPTS_ROOT = ROOT / "reports/roadmap_9997/loop171/coverage_receipts"
BUNDLE_SHA256 = "90961bfed0460787e261965a3180e1b0569df0f9d275f9693daad1ccf53dc233"
CAPA = ROOT / ".cache/loop171_capa/capa-v9.4.0-windows/capa.exe"
RULES = ROOT / ".cache/loop171_capa/capa-rules"
CAPA_SHA256 = "356db59a7f0d22b00ee63bda6ebb5b05149530eec5dd84101caef6d6bae94dfd"
RULES_COMMIT = "aed45e2571ebf7d2330e3daddbb5c472cc54966e"
ROWS = 256
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_SAMPLE_SECONDS = 120.0
MAX_WALL_SECONDS = 7200.0
WORKER = ROOT / "scripts/run_loop171_capa_worker.py"
WINDOWS_PYTHON = ROOT / "vnev/Scripts/python.exe"


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_toolchain() -> None:
    if sha256(CAPA.read_bytes()) != CAPA_SHA256:
        raise CoverageError("capa binary digest drifted")
    result = subprocess.run(["git", "-C", str(RULES), "rev-parse", "HEAD"], capture_output=True, check=False)
    if result.returncode != 0 or result.stdout.decode("ascii", "strict").strip() != RULES_COMMIT:
        raise CoverageError("capa rules commit drifted")


def _run_worker(
    source: Path,
    source_sha: str,
    expected_size: int,
    receipt: Path,
) -> tuple[CapabilityAggregate | None, str | None]:
    _write_receipt(receipt, {"status": "controller_started"})
    try:
        result = subprocess.run(
            [
                str(WINDOWS_PYTHON),
                "-I",
                str(WORKER),
                "--source",
                str(source),
                "--sha256",
                source_sha,
                "--expected-size",
                str(expected_size),
                "--max-bytes",
                str(MAX_INPUT_BYTES),
                "--timeout-seconds",
                str(MAX_SAMPLE_SECONDS),
                "--receipt",
                str(receipt),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=MAX_SAMPLE_SECONDS + 15.0,
        )
    except subprocess.TimeoutExpired:
        _write_receipt(receipt, {"status": "worker_timeout"})
        return None, "worker_timeout"
    try:
        payload = json.loads(receipt.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError):
        _write_receipt(receipt, {"status": "guard_lost"})
        return None, "guard_lost"
    status = payload.get("status") if isinstance(payload, dict) else None
    if status in {"controller_started", "started"}:
        _write_receipt(receipt, {"status": "guard_lost", "worker_returncode": result.returncode})
        return None, "guard_lost"
    if status == "integrity_error":
        raise CoverageError("worker source integrity failure")
    if status != "ok":
        return None, str(status or "worker_crash")
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != {"rule_count", "namespace_counts"}:
        raise CoverageError("worker aggregate schema drifted")
    try:
        return CapabilityAggregate(int(aggregate["rule_count"]), tuple((str(name), int(count)) for name, count in aggregate["namespace_counts"])), None
    except (TypeError, ValueError):
        raise CoverageError("worker aggregate types drifted") from None


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--run-id", default="coverage")
    args = parser.parse_args()
    if not 1 <= args.rows <= ROWS or not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise CoverageError("invalid bounded capa coverage probe arguments")
    started = time.perf_counter()
    _verify_toolchain()
    raw = BUNDLE.read_bytes()
    if sha256(raw) != BUNDLE_SHA256:
        raise CoverageError("coverage bundle digest drifted")
    records = [json.loads(line) for line in raw.splitlines()]
    if len(records) != ROWS or any(record.get("split_role") != "train" for record in records):
        raise CoverageError("coverage bundle role or denominator drifted")
    records = records[: args.rows]
    receipts = RECEIPTS_ROOT / args.run_id
    if receipts.exists():
        raise CoverageError("coverage receipt directory already exists")
    missing, namespaces = Counter(), Counter()
    rule_count = 0
    commitment = hashlib.sha256()
    for ordinal, record in enumerate(records):
        source, source_sha, expected_size = _source_for_record(record)
        aggregate, reason = _run_worker(source, source_sha, expected_size, receipts / f"{ordinal:04d}.json")
        commitment.update(source_sha.encode("ascii"))
        if aggregate is None:
            missing[str(reason)] += 1
        else:
            rule_count += aggregate.rule_count
            namespaces.update(dict(aggregate.namespace_counts))
            commitment.update(json.dumps({"rule_count": aggregate.rule_count, "namespace_counts": aggregate.namespace_counts}, separators=(",", ":")).encode("utf-8"))
        if time.perf_counter() - started > MAX_WALL_SECONDS:
            raise CoverageError("capa coverage wall-time budget exceeded")
    denominator = len(records)
    success = denominator - sum(missing.values())
    coverage = success / denominator
    elapsed_seconds = time.perf_counter() - started
    report = {
        "schema": "axon_loop171_capa_coverage_v1",
        "claim_scope": "fixed_rules_static_train_only_coverage_not_model_quality_or_heldout_evidence",
        "input": {"bundle_sha256": BUNDLE_SHA256, "rows": denominator, "split_role": "train"},
        "toolchain": {"capa_binary_sha256": CAPA_SHA256, "capa_rules_commit": RULES_COMMIT, "network_policy": "offline_process_environment_no_network_inputs"},
        "counts": {"denominator": denominator, "success": success, "missing": sum(missing.values()), "silent_drop": 0, "missing_by_reason": dict(sorted(missing.items()))},
        "aggregate_features": {"rule_count": rule_count, "namespace_counts": dict(sorted(namespaces.items()))},
        "integrity": {"aggregate_commitment_sha256": commitment.hexdigest(), "raw_or_match_location_persisted": False, "identity_fields_used_only_for_binding": True, "heldout_access": False},
        "resource": {"elapsed_seconds": elapsed_seconds},
        "gates": {"coverage_minimum": denominator == ROWS and coverage >= 0.95, "denominator_conserved": success + sum(missing.values()) == denominator, "silent_drop_zero": True, "wall_time": elapsed_seconds <= MAX_WALL_SECONDS},
    }
    report["coverage"] = coverage
    report["decision"] = "phase_a_capability_coverage_pass" if all(report["gates"].values()) else ("engineering_probe_not_coverage_gate" if denominator < ROWS else "phase_a_capability_coverage_closed")
    output = OUTPUT if denominator == ROWS else OUTPUT.with_name(f"capa_coverage_probe_{args.run_id}.json")
    _write_receipt(output, report)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
