#!/usr/bin/env python3
"""Build blinded review focus rows from strict content evidence packages."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOOKUP_COLUMNS = [
    "source_path",
    "source_sha256",
    "cache_path",
    "sample_index",
]

PUBLIC_CONTENT_COLUMNS = [
    "pe_schema_version",
    "pe_file_size",
    "pe_log_size",
    "sections_count",
    "section_entropy_max",
    "section_entropy_avg",
    "section_high_entropy_ratio",
    "section_raw_size_cv",
    "long_sections_ratio",
    "short_sections_ratio",
    "executable_sections_ratio",
    "writable_sections_ratio",
    "readable_sections_ratio",
    "rwx_sections_ratio",
    "has_signature",
    "api_network_ratio",
    "api_process_ratio",
    "api_filesystem_ratio",
    "api_registry_ratio",
    "api_crypto_ratio",
    "api_injection_ratio",
    "packer_keyword_hits_count",
    "packer_keyword_hits_ratio",
    "stat_mean_byte",
    "stat_std_byte",
    "stat_count_0x00",
    "stat_count_0xff",
    "stat_count_ascii_printable",
    "stat_byte_entropy",
]

PUBLIC_FIELDNAMES = [
    "review_focus_id",
    "focus_rank",
    "priority_band",
    "split",
    "current_label",
    "error_type",
    "error_transition",
    "triage_confidence_bucket",
    "review_lane",
    "content_signal_count",
    "content_tags",
    "recommended_review_action",
    *PUBLIC_CONTENT_COLUMNS,
]

PRIVATE_FIELDNAMES = [
    "review_focus_id",
    "focus_rank",
    "priority_value",
    "source_path",
    "source_sha256",
    "cache_path",
    "split",
    "sample_index",
    "label",
    "score",
    "prediction",
    "error_type",
    "error_transition",
    "severity_score",
]

FORBIDDEN_PUBLIC_FIELD_TOKENS = [
    "source",
    "sha",
    "hash",
    "cache",
    "path",
    "sample_index",
    "filename",
    "directory",
    "extension",
    "score",
    "prob",
    "threshold",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(row: dict[str, str], name: str, default: int = 0) -> int:
    return int(round(_float(row, name, float(default))))


def _confidence(row: dict[str, str]) -> float:
    score = _float(row, "score")
    prediction = _int(row, "prediction")
    return score if prediction == 1 else 1.0 - score


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.95:
        return "very_high"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "near_boundary"


def _content_tags(row: dict[str, str]) -> list[str]:
    tags: list[str] = []
    if _float(row, "section_entropy_max") >= 0.80 or _float(row, "section_high_entropy_ratio") >= 0.25:
        tags.append("high_section_entropy")
    if _float(row, "rwx_sections_ratio") > 0.0:
        tags.append("rwx_section_present")
    if _float(row, "executable_sections_ratio") > 0.0 and _float(row, "writable_sections_ratio") > 0.0:
        tags.append("mixed_executable_writable_sections")
    if _float(row, "packer_keyword_hits_count") > 0.0 or _float(row, "packer_keyword_hits_ratio") > 0.0:
        tags.append("packer_section_name_hit")
    if _float(row, "section_raw_size_cv") >= 1.5:
        tags.append("large_section_size_skew")
    api_sum = sum(
        _float(row, name)
        for name in [
            "api_network_ratio",
            "api_process_ratio",
            "api_filesystem_ratio",
            "api_registry_ratio",
            "api_crypto_ratio",
            "api_injection_ratio",
        ]
    )
    if api_sum <= 0.0 and _float(row, "sections_count") > 0.0:
        tags.append("sparse_api_category_surface")
    stat_entropy = _float(row, "stat_byte_entropy", default=-1.0)
    if 0.0 <= stat_entropy < 0.25:
        tags.append("low_byte_entropy")
    if stat_entropy >= 0.80:
        tags.append("high_byte_entropy")
    file_size = _float(row, "pe_file_size")
    if 0.0 < file_size < 4096.0:
        tags.append("tiny_pe_file")
    if row.get("has_signature", "") != "" and _float(row, "has_signature") <= 0.0:
        tags.append("unsigned_pe")
    return tags


def _review_lane(row: dict[str, str], tags: Sequence[str]) -> str:
    error_transition = row.get("error_transition", "")
    error_type = row.get("error_type", "")
    suspicious = {
        "high_section_entropy",
        "rwx_section_present",
        "mixed_executable_writable_sections",
        "packer_section_name_hit",
        "large_section_size_skew",
        "high_byte_entropy",
    }
    weak_surface = {"sparse_api_category_surface", "low_byte_entropy", "tiny_pe_file"}
    tag_set = set(tags)
    if error_transition == "broken_by_calibrator":
        return "calibration_regression_review"
    if error_type == "FN" and tag_set & suspicious:
        return "model_blindspot_review"
    if error_type == "FN" and tag_set & weak_surface:
        return "feature_or_label_quality_review"
    if error_type == "FP" and tag_set & suspicious:
        return "feature_or_label_quality_review"
    return "boundary_model_review"


def _priority_value(row: dict[str, str], tags: Sequence[str]) -> float:
    confidence = _confidence(row)
    priority = _float(row, "severity_score") * 10.0
    if row.get("error_transition") == "persistent_error":
        priority += 100.0
    elif row.get("error_transition") == "broken_by_calibrator":
        priority += 35.0
    if confidence >= 0.95:
        priority += 20.0
    elif confidence >= 0.85:
        priority += 12.0
    elif confidence >= 0.70:
        priority += 5.0

    tag_weights = {
        "high_section_entropy": 8.0,
        "rwx_section_present": 8.0,
        "mixed_executable_writable_sections": 5.0,
        "packer_section_name_hit": 7.0,
        "large_section_size_skew": 4.0,
        "sparse_api_category_surface": 3.0,
        "low_byte_entropy": 3.0,
        "high_byte_entropy": 5.0,
        "tiny_pe_file": 3.0,
        "unsigned_pe": 1.0,
    }
    priority += sum(tag_weights.get(tag, 0.0) for tag in tags)
    return priority


def _priority_band(priority_value: float) -> str:
    if priority_value >= 140.0:
        return "critical"
    if priority_value >= 120.0:
        return "high"
    if priority_value >= 80.0:
        return "medium"
    return "standard"


def _assert_public_schema_safe(fieldnames: Sequence[str]) -> None:
    unsafe = []
    for fieldname in fieldnames:
        folded = fieldname.casefold()
        if any(token in folded for token in FORBIDDEN_PUBLIC_FIELD_TOKENS):
            unsafe.append(fieldname)
    if unsafe:
        raise ValueError(f"Public focus schema contains forbidden identity/model fields: {unsafe}")


def _read_rows(evidence_csv: Path) -> list[dict[str, str]]:
    with resolve_path(evidence_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_noise_audit_focus(
    *,
    evidence_csv: Path,
    output_focus_csv: Path,
    output_private_map_csv: Path,
    output_json: Path,
    review_prefix: str = "loop126_focus",
    max_rows: Optional[int] = None,
) -> dict:
    rows = _read_rows(evidence_csv)
    ranked_rows = []
    for row_index, row in enumerate(rows):
        tags = _content_tags(row)
        priority_value = _priority_value(row, tags)
        ranked_rows.append(
            {
                "row_index": row_index,
                "row": row,
                "tags": tags,
                "priority_value": priority_value,
                "confidence": _confidence(row),
                "review_lane": _review_lane(row, tags),
            }
        )

    ranked_rows.sort(
        key=lambda item: (
            -float(item["priority_value"]),
            str(item["row"].get("error_type", "")),
            str(item["row"].get("error_transition", "")),
            int(item["row_index"]),
        )
    )
    if max_rows is not None:
        ranked_rows = ranked_rows[:max_rows]

    _assert_public_schema_safe(PUBLIC_FIELDNAMES)
    focus_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    lane_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    error_counts: Counter = Counter()
    transition_counts: Counter = Counter()

    for rank, item in enumerate(ranked_rows, start=1):
        row = item["row"]
        tags = list(item["tags"])
        review_focus_id = f"{review_prefix}_{rank:06d}"
        priority_value = float(item["priority_value"])
        lane = str(item["review_lane"])
        confidence_bucket = _confidence_bucket(float(item["confidence"]))
        lane_counts[lane] += 1
        error_counts[row.get("error_type", "")] += 1
        transition_counts[row.get("error_transition", "")] += 1
        tag_counts.update(tags)

        focus_row = {
            "review_focus_id": review_focus_id,
            "focus_rank": rank,
            "priority_band": _priority_band(priority_value),
            "split": row.get("split", ""),
            "current_label": row.get("label", ""),
            "error_type": row.get("error_type", ""),
            "error_transition": row.get("error_transition", ""),
            "triage_confidence_bucket": confidence_bucket,
            "review_lane": lane,
            "content_signal_count": len(tags),
            "content_tags": "|".join(tags),
            "recommended_review_action": "review_content_evidence_without_identity_fields",
        }
        for column in PUBLIC_CONTENT_COLUMNS:
            focus_row[column] = row.get(column, "")
        focus_rows.append(focus_row)

        private_rows.append(
            {
                "review_focus_id": review_focus_id,
                "focus_rank": rank,
                "priority_value": f"{priority_value:.6f}",
                "source_path": row.get("source_path", ""),
                "source_sha256": row.get("source_sha256", ""),
                "cache_path": row.get("cache_path", ""),
                "split": row.get("split", ""),
                "sample_index": row.get("sample_index", ""),
                "label": row.get("label", ""),
                "score": row.get("score", ""),
                "prediction": row.get("prediction", ""),
                "error_type": row.get("error_type", ""),
                "error_transition": row.get("error_transition", ""),
                "severity_score": row.get("severity_score", ""),
            }
        )

    resolved_focus_csv = resolve_path(output_focus_csv)
    resolved_focus_csv.parent.mkdir(parents=True, exist_ok=True)
    with resolved_focus_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(focus_rows)

    resolved_private_csv = resolve_path(output_private_map_csv)
    resolved_private_csv.parent.mkdir(parents=True, exist_ok=True)
    with resolved_private_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIVATE_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(private_rows)

    payload = {
        "schema": "axon_strict_noise_audit_focus_v1",
        "identity_feature_policy": (
            "The public focus CSV is blinded: source_path/source_sha256/cache_path/sample_index and "
            "filename/directory/extension-derived fields are excluded. The private map is for lookup only."
        ),
        "verdict_policy": "No row is relabeled automatically; outputs are review priorities based on model-error state plus PE/stat content evidence.",
        "evidence_csv": str(resolve_path(evidence_csv)),
        "output_focus_csv": str(resolved_focus_csv),
        "output_private_map_csv": str(resolved_private_csv),
        "input_rows": len(rows),
        "focus_rows": len(focus_rows),
        "max_rows": max_rows,
        "review_prefix": review_prefix,
        "public_forbidden_field_tokens": FORBIDDEN_PUBLIC_FIELD_TOKENS,
        "public_columns": PUBLIC_FIELDNAMES,
        "lookup_columns_private_only": LOOKUP_COLUMNS,
        "error_counts": dict(sorted(error_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "review_lane_counts": dict(sorted(lane_counts.items())),
        "content_tag_counts": dict(sorted(tag_counts.items())),
    }
    resolved_output_json = resolve_path(output_json)
    resolved_output_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build blinded strict noise-audit focus rows.")
    parser.add_argument("--evidence-csv", type=Path, required=True)
    parser.add_argument("--output-focus-csv", type=Path, required=True)
    parser.add_argument("--output-private-map-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--review-prefix", default="loop126_focus")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_noise_audit_focus(
        evidence_csv=args.evidence_csv,
        output_focus_csv=args.output_focus_csv,
        output_private_map_csv=args.output_private_map_csv,
        output_json=args.output_json,
        review_prefix=args.review_prefix,
        max_rows=args.max_rows,
    )
    print(
        json.dumps(
            {
                "focus_rows": payload["focus_rows"],
                "error_counts": payload["error_counts"],
                "transition_counts": payload["transition_counts"],
                "review_lane_counts": payload["review_lane_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
