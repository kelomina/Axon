#!/usr/bin/env python3
"""Run the preregistered aggregate-only Loop175 Train coverage probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.protocol import CoverageAccounting, assert_coverage_gate  # noqa: E402
from src.loop175.region_extractor import (  # noqa: E402
    RegionExtractionConfig,
    extract_regions_from_path,
    region_kind_counts,
)
from src.loop175.resource_guard import process_rss_bytes  # noqa: E402

PROPOSAL = Path("manifests/roadmap_9997/loop175_section_region_moe/proposal.json")
PHASE0_SCHEMA = "axon_loop175_phase0_receipt_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_phase0_receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != PHASE0_SCHEMA:
        raise RuntimeError("unexpected Loop175 Phase-0 receipt schema")
    if receipt.get("decision") != "phase0_pass_phase_a_256_train_coverage_probe_ready":
        raise RuntimeError("Loop175 Phase-0 receipt does not authorize the coverage probe")
    for source in receipt.get("sources", []):
        source_path = PROJECT_ROOT / str(source["path"])
        if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
            raise RuntimeError(f"Loop175 source drift: {source_path}")
    return receipt


def load_train_rows(split_csv: Path) -> list[dict[str, str]]:
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if str(row.get("split", "")).casefold() == "train"]
    if not rows:
        raise RuntimeError("Loop175 split contains no Train rows")
    required = {"source_path", "source_sha256", "label", "split"}
    if not required.issubset(rows[0]):
        raise RuntimeError("Loop175 split is missing required columns")
    return rows


def select_balanced_rows(
    rows: Sequence[dict[str, str]], *, count: int, seed: int
) -> list[dict[str, str]]:
    if count <= 0 or count % 2:
        raise ValueError("coverage probe row count must be positive and even")
    by_label = {
        label: [row for row in rows if int(row["label"]) == label]
        for label in (0, 1)
    }
    per_label = count // 2
    if any(len(label_rows) < per_label for label_rows in by_label.values()):
        raise RuntimeError("insufficient rows for the balanced coverage probe")
    randomizer = random.Random(seed)
    selected = []
    for label in (0, 1):
        selected.extend(randomizer.sample(by_label[label], per_label))
    return sorted(selected, key=lambda row: row["source_sha256"].casefold())


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def run_probe(
    *,
    split_csv: Path,
    phase0_receipt: Path,
    output: Path,
    row_count: int = 256,
    seed: int = 175,
) -> dict[str, object]:
    phase0 = validate_phase0_receipt(phase0_receipt)
    rows = load_train_rows(split_csv)
    selected = select_balanced_rows(rows, count=row_count, seed=seed)
    config = RegionExtractionConfig()
    accounting = CoverageAccounting()
    timings: list[float] = []
    kind_counts: dict[str, int] = {}
    missing_reason_counts: dict[str, int] = {}
    maximum_rss = process_rss_bytes()
    started = time.monotonic()
    for row in selected:
        item_started = time.monotonic()
        result = extract_regions_from_path(row["source_path"], config)
        timings.append(time.monotonic() - item_started)
        label = int(row["label"])
        accounting.observe(result, label=label)
        for name, count in region_kind_counts(result.regions).items():
            kind_counts[name] = kind_counts.get(name, 0) + count
        for region in result.regions:
            if region.missing_reason:
                missing_reason_counts[region.missing_reason] = (
                    missing_reason_counts.get(region.missing_reason, 0) + 1
                )
        maximum_rss = max(maximum_rss, process_rss_bytes())
    wall_seconds = time.monotonic() - started
    summary = accounting.summary()
    average_model_bytes = summary["model_region_bytes"] / max(summary["attempted"], 1)
    estimated_train_cache_bytes = math.ceil(average_model_bytes * len(rows))
    p95 = percentile(timings, 0.95)
    blockers: list[str] = []
    try:
        assert_coverage_gate(accounting)
    except RuntimeError as error:
        blockers.append(str(error))
    if p95 > 2.0:
        blockers.append("p95 extraction wall exceeds 2 seconds")
    if maximum_rss > 11 * 1024**3:
        blockers.append("peak RSS exceeds 11 GiB")
    if estimated_train_cache_bytes > 30 * 1024**3:
        blockers.append("estimated Train cache exceeds 30 GiB")

    payload: dict[str, object] = {
        "schema": "axon_loop175_region_coverage_probe_v1",
        "loop_id": "Loop175",
        "claim_scope": "aggregate_train_only_coverage_and_resource_evidence_not_model_quality",
        "inputs": {
            "split_csv": str(split_csv),
            "split_csv_sha256": sha256(split_csv),
            "proposal_sha256": sha256(PROJECT_ROOT / PROPOSAL),
            "phase0_receipt_sha256": sha256(phase0_receipt),
            "phase0_source_count": len(phase0.get("sources", [])),
            "train_pool_rows": len(rows),
        },
        "sampling": {
            "seed": seed,
            "rows": row_count,
            "label_counts": {"0": row_count // 2, "1": row_count // 2},
            "row_identity_persisted": False,
            "labels_visible_to_extractor": False,
        },
        "coverage": summary,
        "aggregate_region_kind_counts": dict(sorted(kind_counts.items())),
        "aggregate_missing_reason_counts": dict(sorted(missing_reason_counts.items())),
        "timing": {
            "wall_seconds": wall_seconds,
            "mean_seconds_per_file": sum(timings) / max(len(timings), 1),
            "p50_seconds_per_file": percentile(timings, 0.50),
            "p95_seconds_per_file": p95,
            "maximum_seconds_per_file": max(timings, default=0.0),
        },
        "resources": {
            "maximum_rss_bytes": maximum_rss,
            "estimated_full_train_region_cache_bytes": estimated_train_cache_bytes,
            "gpu_used": False,
        },
        "blockers": blockers,
        "decision": (
            "phase_a_pass_seed41_implementation_may_begin"
            if not blockers
            else "close_loop175_current_region_extraction_recipe"
        ),
        "training_runs": 0,
        "val_test_or_full_rows_opened": 0,
    }
    write_exclusive_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--phase0-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=175)
    arguments = parser.parse_args()
    payload = run_probe(
        split_csv=arguments.split_csv.resolve(),
        phase0_receipt=arguments.phase0_receipt.resolve(),
        output=arguments.output.resolve(),
        row_count=arguments.rows,
        seed=arguments.seed,
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if not payload["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
