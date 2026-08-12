#!/usr/bin/env python3
"""Run Loop170's fixed Train-only CFG semantic extraction coverage gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loop170.cfg_semantics import CFGSemanticFeatures, MISSING_REASONS  # noqa: E402


BUNDLE = ROOT / "reports/roadmap_9997/loop164/local_probe_bundle.jsonl"
DATA_ROOT = ROOT / "data/random_20w_worktree"
OUTPUT = ROOT / "reports/roadmap_9997/loop170/cfg_coverage.json"
PROPOSAL = ROOT / "manifests/roadmap_9997/loop170_cfg_semantic_expert/proposal.json"
BUNDLE_SHA256 = "90961bfed0460787e261965a3180e1b0569df0f9d275f9693daad1ccf53dc233"
ROWS = 256
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_WALL_SECONDS = 180.0
MAX_WORKER_SECONDS = 15.0
WORKER = ROOT / "scripts/run_loop170_cfg_worker.py"


class CoverageError(ValueError):
    """Raised when identity binding or the fail-closed coverage contract drifts."""


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _relative_source(value: str) -> Path:
    source = Path(value)
    root = DATA_ROOT.resolve(strict=True)
    try:
        relative = source.absolute().relative_to(root)
    except ValueError:
        source_parts = tuple(part.casefold() for part in source.absolute().parts)
        root_parts = tuple(part.casefold() for part in root.parts)
        if source_parts[: len(root_parts)] != root_parts:
            raise CoverageError("source escapes Train root") from None
        relative = Path(*source.absolute().parts[len(root.parts) :])
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise CoverageError("invalid source relative path")
    return relative


def _source_for_record(payload: dict[str, Any]) -> tuple[Path, str, int]:
    source_sha = str(payload.get("source_sha256") or "").casefold()
    expected_size = payload.get("source_size_bytes")
    if len(source_sha) != 64 or not isinstance(expected_size, int) or not 0 < expected_size <= MAX_INPUT_BYTES:
        raise CoverageError("record identity binding is invalid")
    source = DATA_ROOT / _relative_source(str(payload.get("source_path") or ""))
    if source.is_symlink() or not source.is_file():
        raise CoverageError("source is not a regular Train file")
    before = os.stat(source, follow_symlinks=False)
    if before.st_size != expected_size:
        raise CoverageError("source size drifted")
    return source, source_sha, expected_size


def _worker_feature(source: Path, source_sha: str, expected_size: int) -> tuple[CFGSemanticFeatures, int]:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                f"import runpy, sys; sys.argv={[str(WORKER), '--source', str(source), '--sha256', source_sha, '--expected-size', str(expected_size), '--max-bytes', str(MAX_INPUT_BYTES)]!r}; runpy.run_path({str(WORKER)!r}, run_name='__main__')",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_WORKER_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return CFGSemanticFeatures("unknown", 0, 0, 0, 0, 0, 0, 0, 0, (), "worker_timeout"), 0
    try:
        payload = json.loads(result.stdout)
        if result.returncode == 2 and payload.get("status") == "integrity_error":
            raise CoverageError(f"worker source integrity failure: {payload.get('detail')}")
        if result.returncode != 0:
            return CFGSemanticFeatures("unknown", 0, 0, 0, 0, 0, 0, 0, 0, (), "worker_crash"), 0
        if set(payload) != {"status", "feature", "peak_rss_bytes"} or payload["status"] != "ok":
            raise ValueError("worker payload schema drifted")
        payload = payload["feature"]
        allowed = {field.name for field in fields(CFGSemanticFeatures)}
        if set(payload) != allowed:
            raise ValueError("worker feature schema drifted")
        payload["category_counts"] = tuple((str(name), int(value)) for name, value in payload["category_counts"])
        return CFGSemanticFeatures(**payload), int(json.loads(result.stdout)["peak_rss_bytes"])
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise CoverageError("CFG worker emitted an invalid aggregate record") from error


def main() -> None:
    started = time.perf_counter()
    bundle_raw = BUNDLE.read_bytes()
    if sha256(bundle_raw) != BUNDLE_SHA256:
        raise CoverageError("fixed Train coverage bundle drifted")
    records = [json.loads(line) for line in bundle_raw.splitlines()]
    if len(records) != ROWS or any(row.get("split_role") != "train" for row in records):
        raise CoverageError("coverage bundle denominator or split role drifted")
    missing, architectures, categories = Counter(), Counter(), Counter()
    totals = Counter()
    peak_worker_rss = 0
    commitment = hashlib.sha256()
    for row in records:
        source, source_sha, expected_size = _source_for_record(row)
        feature, worker_rss = _worker_feature(source, source_sha, expected_size)
        peak_worker_rss = max(peak_worker_rss, worker_rss)
        commitment.update(str(row["source_sha256"]).encode("ascii"))
        commitment.update(json.dumps(feature.__dict__, sort_keys=True).encode("utf-8"))
        if feature.available:
            architectures[feature.architecture] += 1
            categories.update(dict(feature.category_counts))
            for key in ("instruction_count", "decoded_byte_count", "estimated_block_count", "call_count", "direct_branch_count", "indirect_control_count", "return_count", "interrupt_count"):
                totals[key] += int(getattr(feature, key))
        else:
            missing[str(feature.missing_reason)] += 1
        if time.perf_counter() - started > MAX_WALL_SECONDS:
            raise CoverageError("coverage extraction exceeded fixed wall-time budget")
    success = ROWS - sum(missing.values())
    coverage = success / ROWS
    report = {
        "schema": "axon_loop170_cfg_coverage_v1",
        "claim_scope": "train_only_static_extraction_reliability_not_model_quality_or_heldout_evidence",
        "input": {"bundle_sha256": BUNDLE_SHA256, "rows": ROWS, "split_role": "train"},
        "counts": {"denominator": ROWS, "success": success, "missing": sum(missing.values()), "silent_drop": 0, "missing_by_reason": {reason: missing[reason] for reason in MISSING_REASONS}},
        "coverage": coverage,
        "aggregate_features": {"architectures": dict(sorted(architectures.items())), "opcode_categories": dict(sorted(categories.items())), "totals": dict(sorted(totals.items()))},
        "integrity": {"aggregate_commitment_sha256": commitment.hexdigest(), "raw_code_persisted": False, "identity_fields_used_only_for_loading_and_binding": True, "heldout_access": False},
        "runtime": {"capstone_version": __import__("capstone").__version__, "worker_isolation": True},
        "resource": {"elapsed_seconds": time.perf_counter() - started, "peak_worker_rss_bytes": peak_worker_rss},
        "gates": {"coverage_minimum": coverage >= 0.95, "denominator_conserved": success + sum(missing.values()) == ROWS, "silent_drop_zero": True, "raw_code_output_zero": True, "wall_time": time.perf_counter() - started <= MAX_WALL_SECONDS},
    }
    report["decision"] = "phase_a_cfg_coverage_pass" if all(report["gates"].values()) else "phase_a_cfg_coverage_closed"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
