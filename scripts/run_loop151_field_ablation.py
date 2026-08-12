#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop151_runtime.raw_runtime import (
    FieldAblationPrediction,
    FieldAblationResult,
    Loop151FieldAblationRuntime,
    Loop151RuntimeError,
)


ARMS = ("A", "B", "C", "D", "E")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    resolved = Path(path).resolve()
    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"Cannot read manifest: {resolved}") from exc
    if not rows:
        raise ValueError("Manifest is empty")
    required = {"sample_id", "path", "sha256", "label"}
    if set(rows[0]) != required:
        raise ValueError("Manifest columns must be sample_id,path,sha256,label")
    output: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    digests: set[str] = set()
    for row in rows:
        sample_id = str(row["sample_id"] or "").strip()
        expected_sha = str(row["sha256"] or "").strip().casefold()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"Duplicate or empty sample_id: {sample_id}")
        if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
            raise ValueError(f"Invalid SHA-256 for sample {sample_id}")
        if expected_sha in digests:
            raise ValueError(f"Duplicate SHA-256 identity: {expected_sha}")
        file_path = Path(str(row["path"] or "")).resolve()
        if not file_path.is_file():
            raise ValueError(f"Sample path does not exist: {file_path}")
        actual_sha = _sha256(file_path)
        if actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch for sample {sample_id}")
        try:
            label = int(str(row["label"]).strip())
        except ValueError as exc:
            raise ValueError(f"Invalid label for sample {sample_id}") from exc
        if label not in (0, 1):
            raise ValueError(f"Invalid label for sample {sample_id}")
        sample_ids.add(sample_id)
        digests.add(expected_sha)
        output.append(
            {
                "sample_id": sample_id,
                "path": str(file_path),
                "sha256": expected_sha,
                "label": label,
            }
        )
    return output


def _arm_metrics(records: list[dict[str, Any]], arm: str) -> dict[str, float | int]:
    tp = sum(int(record["label"] == 1 and record["arms"][arm] == 1) for record in records)
    fn = sum(int(record["label"] == 1 and record["arms"][arm] == 0) for record in records)
    fp = sum(int(record["label"] == 0 and record["arms"][arm] == 1) for record in records)
    tn = sum(int(record["label"] == 0 and record["arms"][arm] == 0) for record in records)
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(records) if records else 0.0,
    }


def _transition(records: list[dict[str, Any]], left: str, right: str) -> dict[str, int]:
    repairs = 0
    breaks = 0
    unchanged = 0
    for record in records:
        label = int(record["label"])
        left_correct = int(record["arms"][left]) == label
        right_correct = int(record["arms"][right]) == label
        if not left_correct and right_correct:
            repairs += 1
        elif left_correct and not right_correct:
            breaks += 1
        else:
            unchanged += 1
    return {"repairs": repairs, "breaks": breaks, "unchanged": unchanged}


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    all_records = list(records)
    successful = [record for record in all_records if record.get("ok") is True]
    for record in successful:
        arms = record.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise ValueError("Successful records must contain A/B/C/D/E arms")
        if record.get("label") not in (0, 1):
            raise ValueError("Successful records must contain a binary label")
        if not math.isfinite(float(record.get("latency_ms", math.nan))):
            raise ValueError("Successful records must contain finite latency_ms")
    transitions = {
        f"{left}->{right}": _transition(successful, left, right)
        for left, right in zip(ARMS, ARMS[1:])
    }
    return {
        "rows": len(all_records),
        "successful_rows": len(successful),
        "failed_rows": len(all_records) - len(successful),
        "arms": {arm: _arm_metrics(successful, arm) for arm in ARMS},
        "transitions": transitions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loop151 实战决策链消融")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV manifest: sample_id,path,sha256,label")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON receipt file")
    parser.add_argument("--device", default="cpu", help="Inference device (default: cpu)")
    args = parser.parse_args(argv)
    project_root = PROJECT_ROOT.resolve()

    output_path = Path(args.output).resolve()
    try:
        output_path.relative_to(project_root)
    except ValueError:
        print(f"receipt escapes project root: {output_path}", file=sys.stderr)
        return 1

    try:
        manifest = read_manifest(args.manifest)
    except ValueError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 1

    try:
        runtime = Loop151FieldAblationRuntime(device=args.device)
    except (Loop151RuntimeError, ValueError) as exc:
        print(f"runtime init error: {exc}", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    for row in manifest:
        start = time.perf_counter()
        sample_id = row["sample_id"]
        file_path = Path(row["path"])
        label = int(row["label"])
        try:
            result: FieldAblationResult = runtime.predict_one(file_path)
            elapsed = time.perf_counter() - start
            pred = FieldAblationPrediction.from_result(result)
            records.append({
                "ok": True,
                "sample_id": sample_id,
                "label": label,
                "arms": dict(pred.arm_predictions),
                "latency_ms": round(elapsed * 1000.0, 2),
                "arms_detail": {
                    "A": {"score": round(pred.loop28_probability, 6), "prediction": pred.loop28_prediction},
                    "B": {"score": round(pred.primary_probability, 6), "prediction": pred.primary_prediction},
                    "C": {"prediction": pred.loop130_prediction},
                    "D": {"prediction": pred.loop136_prediction},
                    "E": {"prediction": pred.final_prediction},
                },
                "scores": {
                    "loop28": round(pred.loop28_probability, 6),
                    "primary": round(pred.primary_probability, 6),
                    "conservative": round(pred.conservative_probability, 6),
                    "content_cross": round(pred.content_cross_probability, 6),
                    "noise": round(pred.noise_probability, 6),
                    "selector": round(pred.selector_score, 6) if pred.selector_score is not None else None,
                },
            })
        except (Loop151RuntimeError, ValueError) as exc:
            elapsed = time.perf_counter() - start
            records.append({
                "ok": False, "sample_id": sample_id, "label": label,
                "latency_ms": round(elapsed * 1000.0, 2),
                "error": str(exc),
            })

    summary = summarize_records(records)
    receipt = {
        "schema": "axon_loop151_field_ablation_v1",
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "records": records,
        "summary": summary,
    }
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written {len(records)} records to {output_path}, {summary['successful_rows']} ok / {summary['failed_rows']} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
