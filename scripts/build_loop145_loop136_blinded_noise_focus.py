#!/usr/bin/env python3
"""Build a blinded review focus package for Loop136 noise governance.

The public focus file contains only labels, error type, and content-derived
numeric evidence. Identity fields and model/neighbor scores are kept in a
private map for lookup and audit only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_CONTENT_COLUMNS = [
    "content_is_dll",
    "content_export_count_log",
    "content_dir_export_log_size",
    "content_dir_security_log_size",
    "content_overlay_log_size",
    "content_resource_entry_count_log",
    "content_resource_type_count_log",
    "content_dir_resource_size_ratio",
    "content_dir_resource_log_size",
    "content_overlay_entropy",
    "content_import_api_count_log",
    "content_avg_imports_per_dll",
    "content_image_base_log",
    "v2_resource_data_entry_count_log",
    "v2_resource_type_icon_count_log",
    "v2_resource_type_version_count_log",
    "v2_resource_type_manifest_count_log",
    "v2_resource_type_dialog_count_log",
    "v2_last_section_entropy",
    "v2_section_max_virtual_raw_ratio_log",
    "v2_api_file_mutation_ratio",
    "v2_import_dll_version_api_ratio",
    "string_benign_vendor_count_log",
    "string_version_resource_count_log",
    "string_script_exec_count_log",
    "string_script_exec_present",
]

MANUAL_FIELDS = ["manual_label_verdict", "manual_verdict_note", "recommended_action"]

PUBLIC_FIELDNAMES = [
    "review_focus_id",
    "focus_rank",
    "priority_band",
    "current_label",
    "error_type",
    "review_lane",
    "content_signal_count",
    "content_tags",
    "recommended_review_action",
    *PUBLIC_CONTENT_COLUMNS,
    *MANUAL_FIELDS,
]

PRIVATE_FIELDNAMES = [
    "review_focus_id",
    "focus_rank",
    "source_path",
    "source_sha256",
    "cache_path",
    "priority",
    "support_bucket",
    "error_type",
    "label",
    "prediction",
    "prob_malicious",
    "opposite_label_ratio",
    "nearest_similarity",
    "reason",
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
    "prediction",
    "neighbor",
    "similarity",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def _source_sha(row: dict[str, str]) -> str:
    return str(row.get("source_sha256") or "").strip().casefold()


def _priority_band(priority: int, opposite_ratio: float, nearest_similarity: float) -> str:
    if priority <= 0 or (opposite_ratio >= 0.9 and nearest_similarity >= 0.95):
        return "critical"
    if priority <= 10 or opposite_ratio >= 0.8:
        return "high"
    if priority <= 30:
        return "medium"
    return "standard"


def _content_tags(row: dict[str, str]) -> list[str]:
    tags: list[str] = []
    if _float(row, "content_dir_security_log_size") > 0.0:
        tags.append("security_directory_present")
    if _float(row, "content_overlay_log_size") > 0.0:
        tags.append("overlay_present")
    if _float(row, "content_overlay_entropy") >= 0.80:
        tags.append("high_overlay_entropy")
    if _float(row, "content_resource_entry_count_log") >= 3.0:
        tags.append("resource_rich")
    if _float(row, "v2_resource_type_version_count_log") > 0.0:
        tags.append("version_resource_present")
    if _float(row, "v2_section_max_virtual_raw_ratio_log") >= 3.5:
        tags.append("large_virtual_raw_gap")
    if _float(row, "v2_api_file_mutation_ratio") >= 0.04:
        tags.append("file_mutation_api_ratio_high")
    if _float(row, "v2_import_dll_version_api_ratio") >= 0.02:
        tags.append("version_api_ratio_high")
    if _float(row, "string_benign_vendor_count_log") > 0.0:
        tags.append("benign_vendor_string_present")
    if _float(row, "string_script_exec_present") > 0.0:
        tags.append("script_exec_string_present")
    return tags


def _review_lane(error_type: str, tags: Sequence[str]) -> str:
    suspicious_tags = {
        "overlay_present",
        "high_overlay_entropy",
        "large_virtual_raw_gap",
        "file_mutation_api_ratio_high",
        "script_exec_string_present",
    }
    benignish_tags = {"benign_vendor_string_present", "version_resource_present", "security_directory_present"}
    tag_set = set(tags)
    if error_type == "fn" and tag_set & suspicious_tags:
        return "malware_blindspot_or_label_quality_review"
    if error_type == "fp" and tag_set & benignish_tags:
        return "benign_trust_or_label_quality_review"
    return "content_evidence_review"


def _field_has_forbidden_token(fieldname: str, token: str) -> bool:
    folded = fieldname.casefold()
    parts = [part for part in folded.replace("-", "_").split("_") if part]
    return token in parts


def _assert_public_schema_safe(fieldnames: Sequence[str]) -> None:
    unsafe = []
    for fieldname in fieldnames:
        if any(_field_has_forbidden_token(fieldname, token) for token in FORBIDDEN_PUBLIC_FIELD_TOKENS):
            unsafe.append(fieldname)
    if unsafe:
        raise ValueError(f"Public focus schema contains forbidden identity/model fields: {unsafe}")


def build_loop145_focus(
    *,
    neighbor_csv: Path,
    content_review_csv: Path,
    output_focus_csv: Path,
    output_private_map_csv: Path,
    output_json: Path,
    review_prefix: str = "loop145_focus",
    max_rows: Optional[int] = 300,
    support_bucket: str = "neighbors_support_model_prediction",
    max_priority: int = 90,
) -> dict[str, object]:
    neighbor_rows = read_rows(neighbor_csv)
    content_rows = read_rows(content_review_csv)
    content_by_sha = {_source_sha(row): row for row in content_rows if _source_sha(row)}
    if len(content_by_sha) != len([row for row in content_rows if _source_sha(row)]):
        raise ValueError("content_review_csv contains duplicate source_sha256 rows")

    ranked: list[dict[str, object]] = []
    skipped = Counter()
    for row in neighbor_rows:
        if str(row.get("support_bucket", "")).strip() != support_bucket:
            skipped["support_bucket"] += 1
            continue
        if _int(row, "priority", 999) > max_priority:
            skipped["priority"] += 1
            continue
        content_row = content_by_sha.get(_source_sha(row))
        if content_row is None:
            skipped["missing_content_row"] += 1
            continue
        opposite_ratio = _float(row, "opposite_label_ratio")
        nearest_similarity = _float(row, "nearest_similarity")
        priority = _int(row, "priority", 999)
        tags = _content_tags(content_row)
        ranked.append(
            {
                "neighbor": row,
                "content": content_row,
                "priority": priority,
                "opposite_ratio": opposite_ratio,
                "nearest_similarity": nearest_similarity,
                "tags": tags,
            }
        )

    ranked.sort(
        key=lambda item: (
            int(item["priority"]),
            -float(item["opposite_ratio"]),
            -float(item["nearest_similarity"]),
            str(item["neighbor"].get("error_type", "")),
            str(item["neighbor"].get("source_sha256", "")),
        )
    )
    if max_rows is not None:
        ranked = ranked[: max(0, int(max_rows))]

    _assert_public_schema_safe(PUBLIC_FIELDNAMES)
    focus_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    lane_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    band_counts: Counter[str] = Counter()

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
        lane_counts[lane] += 1
        error_counts[error_type] += 1
        band_counts[priority_band] += 1
        tag_counts.update(tags)

        focus_row: dict[str, object] = {
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
            focus_row[column] = content.get(column, "")
        focus_rows.append(focus_row)

        private_rows.append(
            {
                "review_focus_id": focus_id,
                "focus_rank": rank,
                "source_path": content.get("source_path", neighbor.get("source_path", "")),
                "source_sha256": neighbor.get("source_sha256", ""),
                "cache_path": content.get("cache_path", ""),
                "priority": priority,
                "support_bucket": neighbor.get("support_bucket", ""),
                "error_type": error_type,
                "label": neighbor.get("label", ""),
                "prediction": neighbor.get("prediction", ""),
                "prob_malicious": neighbor.get("prob_malicious", ""),
                "opposite_label_ratio": neighbor.get("opposite_label_ratio", ""),
                "nearest_similarity": neighbor.get("nearest_similarity", ""),
                "reason": neighbor.get("reason", ""),
            }
        )

    write_rows(output_focus_csv, focus_rows, PUBLIC_FIELDNAMES)
    write_rows(output_private_map_csv, private_rows, PRIVATE_FIELDNAMES)
    payload: dict[str, object] = {
        "schema": "axon_loop145_loop136_blinded_noise_focus_v1",
        "protocol": (
            "Build a Loop136 high-conflict blinded review package. Public rows exclude source_path, cache_path, "
            "source_sha256, sample_index, filename, directory, extension, model score, probability, threshold, "
            "prediction, neighbor labels, and similarity fields."
        ),
        "identity_feature_policy": (
            "Private identity and neighbor/model fields are lookup and audit fields only. They are not verdict evidence, "
            "model features, threshold inputs, feature-mask inputs, or replacement sampling signals."
        ),
        "verdict_policy": (
            "No automatic relabeling or replacement is performed. A row can enter redraw only after independent "
            "content or external evidence confirms label_wrong, feature_broken, or out_of_scope."
        ),
        "neighbor_csv": str(resolve_path(neighbor_csv)),
        "content_review_csv": str(resolve_path(content_review_csv)),
        "output_focus_csv": str(resolve_path(output_focus_csv)),
        "output_private_map_csv": str(resolve_path(output_private_map_csv)),
        "support_bucket": support_bucket,
        "max_priority": int(max_priority),
        "max_rows": max_rows,
        "input_neighbor_rows": len(neighbor_rows),
        "input_content_rows": len(content_rows),
        "focus_rows": len(focus_rows),
        "skipped_counts": dict(sorted(skipped.items())),
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
    parser = argparse.ArgumentParser(description="Build Loop145 blinded Loop136 noise focus rows.")
    parser.add_argument("--neighbor-csv", type=Path, required=True)
    parser.add_argument("--content-review-csv", type=Path, required=True)
    parser.add_argument("--output-focus-csv", type=Path, required=True)
    parser.add_argument("--output-private-map-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--review-prefix", default="loop145_focus")
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--support-bucket", default="neighbors_support_model_prediction")
    parser.add_argument("--max-priority", type=int, default=90)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop145_focus(
        neighbor_csv=args.neighbor_csv,
        content_review_csv=args.content_review_csv,
        output_focus_csv=args.output_focus_csv,
        output_private_map_csv=args.output_private_map_csv,
        output_json=args.output_json,
        review_prefix=args.review_prefix,
        max_rows=args.max_rows,
        support_bucket=args.support_bucket,
        max_priority=args.max_priority,
    )
    print(
        json.dumps(
            {
                "focus_rows": payload["focus_rows"],
                "error_counts": payload["error_counts"],
                "review_lane_counts": payload["review_lane_counts"],
                "content_tag_counts": payload["content_tag_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
