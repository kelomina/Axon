#!/usr/bin/env python3
"""Build a blinded review package for all current-best Val errors.

Loop153 intentionally focused on the highest-conflict subset of Loop151 Val
errors. Loop156 keeps the same identity policy but exports every remaining
Loop151 Val error so independent reviewers can cover the full current-best
validation error surface before any redraw or training is authorized.
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

from build_loop145_loop136_blinded_noise_focus import (  # noqa: E402
    PRIVATE_FIELDNAMES,
    PUBLIC_CONTENT_COLUMNS,
    PUBLIC_FIELDNAMES,
    _assert_public_schema_safe,
    _content_tags,
    _float,
    _int,
    _priority_band,
    _review_lane,
    _source_sha,
    read_rows,
    resolve_path,
    write_rows,
)


IDENTITY_FEATURE_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/split/row order/model score fields are loading, alignment, "
    "cache-audit, duplicate-review, and manual-review fields only; they are not verdict evidence, model evidence, "
    "feature-selection evidence, threshold evidence, replacement-sampling evidence, or production inference inputs"
)


def build_loop156_review(
    *,
    neighbor_csv: Path,
    content_review_csv: Path,
    output_review_csv: Path,
    output_private_map_csv: Path,
    output_json: Path,
    review_prefix: str = "loop156_val_error",
) -> dict[str, object]:
    neighbor_rows = read_rows(neighbor_csv)
    content_rows = read_rows(content_review_csv)
    content_by_sha = {_source_sha(row): row for row in content_rows if _source_sha(row)}
    if len(content_by_sha) != len([row for row in content_rows if _source_sha(row)]):
        raise ValueError("content_review_csv contains duplicate source_sha256 rows")

    ranked: list[dict[str, object]] = []
    skipped: Counter[str] = Counter()
    for row in neighbor_rows:
        content = content_by_sha.get(_source_sha(row))
        if content is None:
            skipped["missing_content_row"] += 1
            continue
        opposite_ratio = _float(row, "opposite_label_ratio")
        nearest_similarity = _float(row, "nearest_similarity")
        priority = _int(row, "priority", 999)
        support_bucket = str(row.get("support_bucket", "")).strip()
        ranked.append(
            {
                "neighbor": row,
                "content": content,
                "priority": priority,
                "opposite_ratio": opposite_ratio,
                "nearest_similarity": nearest_similarity,
                "support_bucket": support_bucket,
                "tags": _content_tags(content),
            }
        )

    ranked.sort(
        key=lambda item: (
            0 if item["support_bucket"] == "neighbors_support_model_prediction" else 1,
            int(item["priority"]),
            -float(item["opposite_ratio"]),
            -float(item["nearest_similarity"]),
            str(item["neighbor"].get("error_type", "")),
            str(item["neighbor"].get("source_sha256", "")),
        )
    )

    _assert_public_schema_safe(PUBLIC_FIELDNAMES)
    public_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    lane_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    band_counts: Counter[str] = Counter()
    support_counts: Counter[str] = Counter()

    for rank, item in enumerate(ranked, start=1):
        neighbor = item["neighbor"]
        content = item["content"]
        tags = list(item["tags"])
        error_type = str(neighbor.get("error_type", "")).strip().lower()
        priority = int(item["priority"])
        opposite_ratio = float(item["opposite_ratio"])
        nearest_similarity = float(item["nearest_similarity"])
        focus_id = f"{review_prefix}_{rank:06d}"
        priority_band = _priority_band(priority, opposite_ratio, nearest_similarity)
        lane = _review_lane(error_type, tags)
        support_bucket = str(item["support_bucket"])

        lane_counts[lane] += 1
        error_counts[error_type] += 1
        band_counts[priority_band] += 1
        support_counts[support_bucket] += 1
        tag_counts.update(tags)

        public_row: dict[str, object] = {
            "review_focus_id": focus_id,
            "focus_rank": rank,
            "priority_band": priority_band,
            "current_label": neighbor.get("label", ""),
            "error_type": error_type,
            "review_lane": lane,
            "content_signal_count": len(tags),
            "content_tags": "|".join(tags),
            "recommended_review_action": "review_content_or_external_evidence_without_identity_fields",
            "manual_label_verdict": "",
            "manual_verdict_note": "",
            "recommended_action": "",
        }
        for column in PUBLIC_CONTENT_COLUMNS:
            public_row[column] = content.get(column, "")
        public_rows.append(public_row)

        private_rows.append(
            {
                "review_focus_id": focus_id,
                "focus_rank": rank,
                "source_path": content.get("source_path", neighbor.get("source_path", "")),
                "source_sha256": neighbor.get("source_sha256", ""),
                "cache_path": content.get("cache_path", ""),
                "priority": priority,
                "support_bucket": support_bucket,
                "error_type": error_type,
                "label": neighbor.get("label", ""),
                "prediction": neighbor.get("prediction", ""),
                "prob_malicious": neighbor.get("prob_malicious", ""),
                "opposite_label_ratio": neighbor.get("opposite_label_ratio", ""),
                "nearest_similarity": neighbor.get("nearest_similarity", ""),
                "reason": neighbor.get("reason", ""),
            }
        )

    write_rows(output_review_csv, public_rows, PUBLIC_FIELDNAMES)
    write_rows(output_private_map_csv, private_rows, PRIVATE_FIELDNAMES)
    payload: dict[str, object] = {
        "schema": "axon_loop156_current_best_val_full_error_review_v1",
        "protocol": (
            "Build a blinded review package for all Loop151 current-strict-best Val errors. Public rows exclude "
            "identity, model score, probability, threshold, prediction, neighbor labels, and similarity fields."
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "verdict_policy": (
            "No automatic relabeling or replacement is performed. A row can enter redraw only after independent "
            "content or external evidence confirms label_wrong, feature_broken, or out_of_scope."
        ),
        "neighbor_csv": str(resolve_path(neighbor_csv)),
        "content_review_csv": str(resolve_path(content_review_csv)),
        "output_review_csv": str(resolve_path(output_review_csv)),
        "output_private_map_csv": str(resolve_path(output_private_map_csv)),
        "input_neighbor_rows": len(neighbor_rows),
        "input_content_rows": len(content_rows),
        "review_rows": len(public_rows),
        "skipped_counts": dict(sorted(skipped.items())),
        "support_bucket_counts": dict(sorted(support_counts.items())),
        "priority_band_counts": dict(sorted(band_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "review_lane_counts": dict(sorted(lane_counts.items())),
        "content_tag_counts": dict(sorted(tag_counts.items())),
        "public_columns": PUBLIC_FIELDNAMES,
        "private_columns": PRIVATE_FIELDNAMES,
    }
    resolved_json = resolve_path(output_json)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop156 all-current-Val-error blinded review package.")
    parser.add_argument("--neighbor-csv", type=Path, required=True)
    parser.add_argument("--content-review-csv", type=Path, required=True)
    parser.add_argument("--output-review-csv", type=Path, required=True)
    parser.add_argument("--output-private-map-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--review-prefix", default="loop156_val_error")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop156_review(
        neighbor_csv=args.neighbor_csv,
        content_review_csv=args.content_review_csv,
        output_review_csv=args.output_review_csv,
        output_private_map_csv=args.output_private_map_csv,
        output_json=args.output_json,
        review_prefix=args.review_prefix,
    )
    print(
        json.dumps(
            {
                "review_rows": payload["review_rows"],
                "error_counts": payload["error_counts"],
                "support_bucket_counts": payload["support_bucket_counts"],
                "review_lane_counts": payload["review_lane_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
