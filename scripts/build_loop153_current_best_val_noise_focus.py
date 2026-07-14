#!/usr/bin/env python3
"""Build a blinded Val noise focus package for the current strict best.

Loop151 changes the strict-best predictions by applying the trusted-signer
guard on top of Loop136. This script rebuilds the Val noise review package from
the Loop151 remaining Val errors only, while reusing the existing Loop136 Val
neighbor/content evidence as read-only context.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_loop145_loop136_blinded_noise_focus import build_loop145_focus  # noqa: E402


IDENTITY_FEATURE_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/split/row order/model score fields are loading, alignment, "
    "cache-audit, duplicate-review, and manual-review fields only; they are not verdict evidence, model evidence, "
    "feature-selection evidence, threshold evidence, replacement-sampling evidence, or production inference inputs"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _int(value: object) -> int:
    return int(float(str(value).strip()))


def _sha(row: dict[str, str]) -> str:
    return str(row.get("source_sha256") or "").strip().casefold()


def current_wrong_sha_set(
    *,
    predictions_csv: Path,
    prediction_column: str,
) -> tuple[set[str], dict[str, object]]:
    rows, fieldnames = read_rows(predictions_csv)
    if prediction_column not in fieldnames:
        raise ValueError(f"prediction column not found: {prediction_column}")
    wrong: set[str] = set()
    error_counts: Counter[str] = Counter()
    missing_sha_rows = 0
    for row in rows:
        label = _int(row.get("label"))
        prediction = _int(row.get(prediction_column))
        if label == prediction:
            continue
        sha = _sha(row)
        if not sha:
            missing_sha_rows += 1
            continue
        wrong.add(sha)
        error_counts["fn" if label == 1 and prediction == 0 else "fp"] += 1
    return wrong, {
        "prediction_rows": len(rows),
        "prediction_column": prediction_column,
        "current_error_rows": len(wrong),
        "current_error_counts": dict(sorted(error_counts.items())),
        "missing_sha_error_rows": missing_sha_rows,
    }


def filter_csv_by_sha(
    *,
    input_csv: Path,
    output_csv: Path,
    keep_sha: set[str],
) -> tuple[int, int]:
    rows, fieldnames = read_rows(input_csv)
    filtered = [row for row in rows if _sha(row) in keep_sha]
    write_rows(output_csv, filtered, fieldnames)
    return len(rows), len(filtered)


def build_loop153_focus(
    *,
    predictions_csv: Path,
    prediction_column: str,
    neighbor_csv: Path,
    content_review_csv: Path,
    filtered_neighbor_csv: Path,
    filtered_content_review_csv: Path,
    output_focus_csv: Path,
    output_private_map_csv: Path,
    output_json: Path,
    review_prefix: str = "loop153_val_focus",
    support_bucket: str = "neighbors_support_model_prediction",
    max_priority: int = 90,
    max_rows: Optional[int] = None,
) -> dict[str, object]:
    wrong_sha, prediction_summary = current_wrong_sha_set(
        predictions_csv=predictions_csv,
        prediction_column=prediction_column,
    )
    input_neighbor_rows, filtered_neighbor_rows = filter_csv_by_sha(
        input_csv=neighbor_csv,
        output_csv=filtered_neighbor_csv,
        keep_sha=wrong_sha,
    )
    input_content_rows, filtered_content_rows = filter_csv_by_sha(
        input_csv=content_review_csv,
        output_csv=filtered_content_review_csv,
        keep_sha=wrong_sha,
    )
    payload = build_loop145_focus(
        neighbor_csv=filtered_neighbor_csv,
        content_review_csv=filtered_content_review_csv,
        output_focus_csv=output_focus_csv,
        output_private_map_csv=output_private_map_csv,
        output_json=output_json,
        review_prefix=review_prefix,
        max_rows=max_rows,
        support_bucket=support_bucket,
        max_priority=max_priority,
    )
    payload.update(
        {
            "schema": "axon_loop153_current_best_val_noise_focus_v1",
            "protocol": (
                "Build a Loop151 current-strict-best Val-only high-conflict blinded review package. "
                "Public rows exclude identity, model score, probability, threshold, prediction, neighbor labels, "
                "and similarity fields."
            ),
            "identity_feature_policy": IDENTITY_FEATURE_POLICY,
            "verdict_policy": (
                "No automatic relabeling or replacement is performed. A row can enter redraw only after independent "
                "content or external evidence confirms label_wrong, feature_broken, or out_of_scope."
            ),
            "current_best": {
                "name": "Loop151 trusted signer guard",
                "predictions_csv": str(resolve_path(predictions_csv)),
                "prediction_column": prediction_column,
            },
            "prediction_summary": prediction_summary,
            "source_filter_summary": {
                "input_neighbor_rows": input_neighbor_rows,
                "filtered_neighbor_rows": filtered_neighbor_rows,
                "input_content_rows": input_content_rows,
                "filtered_content_rows": filtered_content_rows,
                "missing_neighbor_rows_for_current_errors": max(0, int(prediction_summary["current_error_rows"]) - filtered_neighbor_rows),
                "missing_content_rows_for_current_errors": max(0, int(prediction_summary["current_error_rows"]) - filtered_content_rows),
            },
            "filtered_neighbor_csv": str(resolve_path(filtered_neighbor_csv)),
            "filtered_content_review_csv": str(resolve_path(filtered_content_review_csv)),
        }
    )
    resolve_path(output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop153 Loop151-current Val noise focus package.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--prediction-column", default="trusted_signer_guard_prediction")
    parser.add_argument("--neighbor-csv", type=Path, required=True)
    parser.add_argument("--content-review-csv", type=Path, required=True)
    parser.add_argument("--filtered-neighbor-csv", type=Path, required=True)
    parser.add_argument("--filtered-content-review-csv", type=Path, required=True)
    parser.add_argument("--output-focus-csv", type=Path, required=True)
    parser.add_argument("--output-private-map-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--review-prefix", default="loop153_val_focus")
    parser.add_argument("--support-bucket", default="neighbors_support_model_prediction")
    parser.add_argument("--max-priority", type=int, default=90)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop153_focus(
        predictions_csv=args.predictions_csv,
        prediction_column=args.prediction_column,
        neighbor_csv=args.neighbor_csv,
        content_review_csv=args.content_review_csv,
        filtered_neighbor_csv=args.filtered_neighbor_csv,
        filtered_content_review_csv=args.filtered_content_review_csv,
        output_focus_csv=args.output_focus_csv,
        output_private_map_csv=args.output_private_map_csv,
        output_json=args.output_json,
        review_prefix=args.review_prefix,
        support_bucket=args.support_bucket,
        max_priority=args.max_priority,
        max_rows=args.max_rows,
    )
    print(
        json.dumps(
            {
                "current_error_rows": payload["prediction_summary"]["current_error_rows"],
                "focus_rows": payload["focus_rows"],
                "error_counts": payload["error_counts"],
                "output_json": str(resolve_path(args.output_json)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
