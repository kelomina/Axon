#!/usr/bin/env python3
"""Compare the raw Loop151 runtime with frozen Val-stage prediction records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop151_runtime.raw_runtime import Loop151Runtime, Loop151RuntimeError


STAGE_FILES = {
    "primary": "reports/phase3_loop127/oof_fixed_v2_all_valonly_with_logreg/stage2_oof_stacker_val_predictions.csv",
    "conservative": "reports/phase3_loop127/oof_fixed_v2_all_valonly_no_logreg/stage2_oof_stacker_val_predictions.csv",
    "content_cross": "reports/phase3_loop127/phase1_content_cross_hgb_local_valonly/loop43_content_cross_val_predictions.csv",
    "loop130": "reports/phase3_loop128/loop130_content_string_guard_val_predictions.csv",
    "loop134": "reports/phase3_loop134/oof_fixed_v2_string_noise_valonly/stage2_oof_stacker_val_predictions.csv",
    "loop136": "reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_valonly/loop135_pairwise_selector_val_predictions.csv",
    "loop151": "reports/phase3_loop151/loop151_trusted_signer_guard_val_predictions.csv",
}


def _rows_by_sha(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {str(row["source_sha256"]).casefold(): row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"No rows in frozen parity file: {path}")
    return rows


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance


def _check_sample(
    runtime: Loop151Runtime,
    sha: str,
    rows: dict[str, dict[str, dict[str, str]]],
    tolerance: float,
) -> dict:
    primary_row = rows["primary"][sha]
    prediction = runtime.predict_path(primary_row["source_path"])
    stages = {
        "primary": (
            prediction.primary_probability,
            int(primary_row["prediction"]),
            int(prediction.primary_probability >= 0.31),
        ),
        "conservative": (
            prediction.conservative_probability,
            int(rows["conservative"][sha]["prediction"]),
            int(prediction.conservative_probability >= 0.415),
        ),
        "content_cross": (
            prediction.content_cross_probability,
            int(rows["content_cross"][sha]["prediction"]),
            int(prediction.content_cross_probability >= 0.4),
        ),
        "loop130": (
            prediction.primary_probability,
            int(rows["loop130"][sha]["prediction"]),
            prediction.loop130_prediction,
        ),
        "loop134": (
            prediction.loop134_probability,
            int(rows["loop134"][sha]["prediction"]),
            int(prediction.loop134_probability >= 0.39),
        ),
        "loop136": (
            prediction.probability,
            int(rows["loop136"][sha]["prediction"]),
            prediction.loop136_prediction,
        ),
        "loop151": (
            prediction.probability,
            int(rows["loop151"][sha]["trusted_signer_guard_prediction"]),
            prediction.prediction,
        ),
    }
    stage_records = {}
    passed = True
    for name, (actual_score, expected_prediction, actual_prediction) in stages.items():
        row = rows[name][sha]
        expected_score = float(row.get("stage2_prob_malicious") or primary_row["stage2_prob_malicious"])
        score_required = name in {"primary", "conservative", "content_cross", "loop130", "loop134"}
        score_ok = not score_required or _close(actual_score, expected_score, tolerance)
        prediction_ok = actual_prediction == expected_prediction
        stage_records[name] = {
            "expected_score": expected_score,
            "actual_score": actual_score,
            "score_checked": score_required,
            "score_ok": score_ok,
            "expected_prediction": expected_prediction,
            "actual_prediction": actual_prediction,
            "prediction_ok": prediction_ok,
        }
        passed = passed and score_ok and prediction_ok
    return {
        "source_sha256": sha,
        "source_path": primary_row["source_path"],
        "label": int(primary_row["label"]),
        "passed": passed,
        "stages": stage_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify raw Loop151 parity on accessible frozen Val samples.")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/roadmap_9997/loop151_raw_parity_receipt.json",
    )
    args = parser.parse_args()
    if args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")
    if args.tolerance < 0:
        raise SystemExit("--tolerance must be non-negative")
    rows = {name: _rows_by_sha(PROJECT_ROOT / relative) for name, relative in STAGE_FILES.items()}
    common = set.intersection(*(set(stage_rows) for stage_rows in rows.values()))
    selected = []
    for sha in sorted(common):
        source_path = Path(rows["primary"][sha]["source_path"])
        if source_path.is_file():
            selected.append(sha)
        if len(selected) >= args.max_samples:
            break
    if not selected:
        raise SystemExit("No accessible frozen Val samples are available for raw parity")
    try:
        runtime = Loop151Runtime(device=args.device)
        samples = [_check_sample(runtime, sha, rows, args.tolerance) for sha in selected]
    except Loop151RuntimeError as exc:
        raise SystemExit(f"Loop151 runtime failed: {exc}") from exc
    passed = all(sample["passed"] for sample in samples)
    receipt = {
        "schema": "axon_loop151_raw_parity_receipt_v1",
        "loop_id": "Loop151",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "accessible frozen Val samples only; no fitting, threshold selection, or data mutation",
        "tolerance": args.tolerance,
        "sample_count": len(samples),
        "passed": passed,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
