#!/usr/bin/env python3
"""Build and unblind a reviewer-facing full-queue evidence package."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


PROTOCOL = (
    "read-only blinded review package builder; no model fitting, no threshold selection, no automatic relabeling, "
    "no replacement sampling, no split/cache mutation"
)
IDENTITY_POLICY = (
    "The blinded reviewer CSV omits filenames, paths, extensions, directories, source hashes, cache paths, "
    "sample indices, split, row order, wave/rank fields, and model scores. These remain private alignment fields only."
)
MANUAL_FIELDS = ["manual_label_verdict", "manual_verdict_note", "recommended_action"]
BLIND_EXTRA_FIELDS = ["blind_review_id", "current_label"]
PRIVATE_EXTRA_FIELDS = ["blind_review_id", "loop96_shuffled_position"]
UNBLIND_EXTRA_FIELDS = ["loop96_blind_review_id"]

PRIVATE_ALIGNMENT_COLUMNS = {
    "source_path",
    "cache_path",
    "source_sha256",
    "source_sha256_actual",
    "sample_index",
    "split",
    "review_batch_rank",
    "review_priority_rank",
    "loop95_wave_id",
    "loop95_wave_row_number",
    "loop95_intake_row_number",
    "manifest_duplicate_group_id",
    "manifest_duplicate_group_focus_rows",
}
MODEL_CONTEXT_COLUMNS = {
    "review_category",
    "loop57_error_type",
    "loop57_final_prob",
    "loop57_base_prob",
    "loop57_candidate_prob",
    "loop57_gate_prob",
    "loop39_corrected_by_any_compared_model",
    "model_score_columns_are_not_verdict_evidence",
}
PRIVATE_ONLY_COLUMNS = PRIVATE_ALIGNMENT_COLUMNS | MODEL_CONTEXT_COLUMNS | {"label"}
BLINDED_CONTENT_COLUMNS = [
    "review_tags",
    "content_evidence_fields",
    "source_exists",
    "source_size_bytes",
    "file_entropy",
    "file_entropy_bytes",
    "file_entropy_truncated",
    "mz_signature",
    "pe_parse_status",
    "pe_machine",
    "pe_timestamp",
    "pe_characteristics",
    "pe_optional_magic",
    "pe_subsystem",
    "pe_dll_characteristics",
    "pe_number_of_sections",
    "pe_section_names",
    "pe_section_raw_size_total",
    "pe_section_raw_end_max",
    "pe_has_export_directory",
    "pe_export_directory_size",
    "pe_has_import_directory",
    "pe_import_directory_size",
    "pe_has_resource_directory",
    "pe_resource_directory_size",
    "pe_has_security_directory",
    "pe_security_directory_file_offset",
    "pe_security_directory_size",
    "overlay_size",
    "overlay_entropy",
    "overlay_entropy_bytes",
    "overlay_entropy_truncated",
    "overlay_after_security_size",
    "overlay_after_security_entropy",
    "overlay_after_security_entropy_bytes",
    "overlay_after_security_entropy_truncated",
    "cache_exists",
    "cache_size_bytes",
    "duplicate_manifest_sha_group",
    "manifest_duplicate_group_size",
    "objective_issue_count",
    "objective_issue_flags",
    "identity_columns_are_not_evidence",
]
BLINDED_CONTENT_EVIDENCE_DENYLIST = {
    "source_sha256_match",
    "source_sha256_actual",
    "source_sha256",
    "cache_path",
    "source_path",
    "sample_index",
}


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def _manual_blank_row() -> dict[str, str]:
    return {field: "" for field in MANUAL_FIELDS}


def _fieldnames_with(base: Sequence[str], extras: Sequence[str]) -> list[str]:
    output = list(base)
    seen = set(output)
    for field in extras:
        if field not in seen:
            output.append(field)
            seen.add(field)
    return output


def _private_fieldnames(source_fieldnames: Sequence[str]) -> list[str]:
    return _fieldnames_with(PRIVATE_EXTRA_FIELDS, source_fieldnames)


def _blinded_fieldnames(source_fieldnames: Sequence[str]) -> list[str]:
    present_content = [field for field in BLINDED_CONTENT_COLUMNS if field in source_fieldnames]
    return [*BLIND_EXTRA_FIELDS, *present_content, *MANUAL_FIELDS]


def _forbidden_blinded_columns(fieldnames: Sequence[str]) -> list[str]:
    lowered = {field.lower(): field for field in fieldnames}
    forbidden = []
    for field in PRIVATE_ONLY_COLUMNS:
        if field.lower() in lowered:
            forbidden.append(lowered[field.lower()])
    for field in fieldnames:
        name = field.lower()
        if any(token in name for token in ["source_sha", "sample_index", "source_path", "cache_path"]):
            forbidden.append(field)
        if name.startswith("loop57_") or name.startswith("loop39_") or "prob" in name or "score" in name:
            forbidden.append(field)
    return sorted(set(forbidden))


def _sanitize_blinded_value(field: str, value: object) -> str:
    text = normalize_text(value)
    if field != "content_evidence_fields" or not text:
        return text
    kept = [
        item
        for item in text.split("|")
        if item and item.strip().lower() not in BLINDED_CONTENT_EVIDENCE_DENYLIST
    ]
    return "|".join(kept)


def build_blinded_package(
    *,
    input_csv: Path,
    blinded_csv: Path,
    private_map_csv: Path,
    output_json: Path,
    expected_rows: Optional[int] = None,
    seed: int = 9601,
) -> dict[str, Any]:
    rows, source_fieldnames = read_csv_rows(input_csv)
    blockers: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        blockers.append("input_row_count_mismatch_expected")
    missing_manual = [field for field in MANUAL_FIELDS if field not in source_fieldnames]
    if missing_manual:
        blockers.append("input_missing_manual_fields")
    if "label" not in source_fieldnames:
        blockers.append("input_missing_label")

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    blinded_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    blinded_fieldnames = _blinded_fieldnames(source_fieldnames)
    private_fieldnames = _private_fieldnames(source_fieldnames)
    label_counts: Counter[str] = Counter()

    for position, original_index in enumerate(order, start=1):
        source_row = rows[original_index]
        blind_id = f"blind-{position:05d}"
        label = normalize_text(source_row.get("label"))
        label_counts[label or "blank"] += 1

        blinded_row: dict[str, Any] = {
            "blind_review_id": blind_id,
            "current_label": label,
            **{
                field: _sanitize_blinded_value(field, source_row.get(field, ""))
                for field in BLINDED_CONTENT_COLUMNS
                if field in source_fieldnames
            },
            **_manual_blank_row(),
        }
        private_row: dict[str, Any] = {
            "blind_review_id": blind_id,
            "loop96_shuffled_position": str(position),
            **source_row,
        }
        blinded_rows.append(blinded_row)
        private_rows.append(private_row)

    forbidden_blinded_columns = _forbidden_blinded_columns(blinded_fieldnames)
    if forbidden_blinded_columns:
        blockers.append("blinded_csv_contains_private_or_model_columns")

    write_csv_rows(blinded_csv, blinded_rows, blinded_fieldnames)
    write_csv_rows(private_map_csv, private_rows, private_fieldnames)
    summary = {
        "schema": "axon_loop96_blinded_review_package_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "mode": "build",
        "input_csv": str(input_csv),
        "rows": len(rows),
        "expected_rows": expected_rows,
        "seed": seed,
        "blockers": blockers,
        "label_counts": dict(sorted(label_counts.items())),
        "blinded_field_count": len(blinded_fieldnames),
        "private_map_field_count": len(private_fieldnames),
        "forbidden_blinded_columns": forbidden_blinded_columns,
        "decisions": {
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "ready_for_blinded_review": not blockers,
            "ready_for_loop87_import": False,
        },
        "outputs": {
            "blinded_csv": str(blinded_csv),
            "private_map_csv": str(private_map_csv),
            "summary_json": str(output_json),
        },
        "notes": [
            "Reviewers should annotate only the blinded CSV.",
            "The private map is for trusted pipeline alignment only and must not be used as verdict evidence.",
            "After annotation, run this script in unblind mode and then run Loop87 on the unblinded CSV.",
        ],
    }
    write_json(output_json, summary)
    return summary


def _rows_by_blind_id(rows: Sequence[dict[str, str]], *, source_name: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    by_id: dict[str, dict[str, str]] = {}
    issues: list[str] = []
    counts: Counter[str] = Counter()
    for row in rows:
        blind_id = normalize_text(row.get("blind_review_id"))
        if not blind_id:
            issues.append(f"{source_name}_missing_blind_review_id")
            continue
        counts[blind_id] += 1
        by_id[blind_id] = row
    for blind_id, count in sorted(counts.items()):
        if count > 1:
            issues.append(f"{source_name}_duplicate_blind_review_id:{blind_id}")
    return by_id, issues


def unblind_verdicts(
    *,
    annotated_blinded_csv: Path,
    private_map_csv: Path,
    output_csv: Path,
    output_json: Path,
    expected_rows: Optional[int] = None,
) -> dict[str, Any]:
    annotated_rows, _annotated_fieldnames = read_csv_rows(annotated_blinded_csv)
    private_rows, private_fieldnames = read_csv_rows(private_map_csv)
    blockers: list[str] = []
    if expected_rows is not None and len(annotated_rows) != expected_rows:
        blockers.append("annotated_row_count_mismatch_expected")
    if len(annotated_rows) != len(private_rows):
        blockers.append("annotated_private_row_count_mismatch")

    annotated_by_id, annotated_issues = _rows_by_blind_id(annotated_rows, source_name="annotated")
    private_by_id, private_issues = _rows_by_blind_id(private_rows, source_name="private")
    blockers.extend(annotated_issues)
    blockers.extend(private_issues)
    missing_private = sorted(set(annotated_by_id) - set(private_by_id))
    missing_annotated = sorted(set(private_by_id) - set(annotated_by_id))
    if missing_private:
        blockers.append("annotated_ids_missing_from_private_map")
    if missing_annotated:
        blockers.append("private_ids_missing_from_annotated_csv")

    output_rows: list[dict[str, Any]] = []
    verdict_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for private_row in private_rows:
        blind_id = normalize_text(private_row.get("blind_review_id"))
        annotated = annotated_by_id.get(blind_id, {})
        item = {field: private_row.get(field, "") for field in private_fieldnames if field not in PRIVATE_EXTRA_FIELDS}
        item["loop96_blind_review_id"] = blind_id
        for field in MANUAL_FIELDS:
            item[field] = annotated.get(field, "")
        verdict_counts[normalize_text(item.get("manual_label_verdict")) or "blank"] += 1
        action_counts[normalize_text(item.get("recommended_action")) or "blank"] += 1
        output_rows.append(item)

    output_fieldnames = _fieldnames_with(
        [field for field in private_fieldnames if field not in PRIVATE_EXTRA_FIELDS],
        UNBLIND_EXTRA_FIELDS,
    )
    write_csv_rows(output_csv, output_rows, output_fieldnames)
    summary = {
        "schema": "axon_loop96_unblinded_verdicts_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "mode": "unblind",
        "annotated_blinded_csv": str(annotated_blinded_csv),
        "private_map_csv": str(private_map_csv),
        "rows": len(output_rows),
        "expected_rows": expected_rows,
        "blockers": sorted(set(blockers)),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "decisions": {
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "ready_for_loop87_import": not blockers,
        },
        "outputs": {
            "loop87_ready_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "notes": [
            "Unblinding only restores alignment fields and manual verdicts. Loop87 remains the strict verdict quality gate.",
            "Identity fields restored here are for pipeline alignment only, not verdict evidence.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or unblind Loop96 review packages.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a blinded reviewer CSV and private map.")
    build.add_argument("--input-csv", type=Path, required=True)
    build.add_argument("--blinded-csv", type=Path, required=True)
    build.add_argument("--private-map-csv", type=Path, required=True)
    build.add_argument("--output-json", type=Path, required=True)
    build.add_argument("--expected-rows", type=int, default=None)
    build.add_argument("--seed", type=int, default=9601)

    unblind = subparsers.add_parser("unblind", help="Merge annotated blinded verdicts back into Loop87-ready CSV.")
    unblind.add_argument("--annotated-blinded-csv", type=Path, required=True)
    unblind.add_argument("--private-map-csv", type=Path, required=True)
    unblind.add_argument("--output-csv", type=Path, required=True)
    unblind.add_argument("--output-json", type=Path, required=True)
    unblind.add_argument("--expected-rows", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        summary = build_blinded_package(
            input_csv=args.input_csv,
            blinded_csv=args.blinded_csv,
            private_map_csv=args.private_map_csv,
            output_json=args.output_json,
            expected_rows=args.expected_rows,
            seed=args.seed,
        )
    else:
        summary = unblind_verdicts(
            annotated_blinded_csv=args.annotated_blinded_csv,
            private_map_csv=args.private_map_csv,
            output_csv=args.output_csv,
            output_json=args.output_json,
            expected_rows=args.expected_rows,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
