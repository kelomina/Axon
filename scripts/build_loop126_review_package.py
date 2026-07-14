#!/usr/bin/env python3
"""Build a blinded Loop126 review annotation template."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANUAL_FIELDS = ["manual_label_verdict", "manual_verdict_note", "recommended_action"]
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
]
ALLOWED_MANUAL_LABEL_VERDICTS = [
    "label_correct",
    "label_wrong",
    "feature_broken",
    "invalid_pe",
    "out_of_scope",
    "uncertain",
    "needs_more_evidence",
]
ALLOWED_RECOMMENDED_ACTIONS = [
    "keep_label",
    "model_blindspot",
    "replace_with_fresh_same_label_candidate",
    "quarantine_for_more_evidence",
    "needs_more_evidence",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _forbidden_public_columns(fieldnames: Sequence[str]) -> list[str]:
    allowed = {
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
        "pe_schema_version",
    }
    found: list[str] = []
    for fieldname in fieldnames:
        if fieldname in allowed or fieldname in MANUAL_FIELDS:
            continue
        if any(_field_has_forbidden_token(fieldname, token) for token in FORBIDDEN_PUBLIC_FIELD_TOKENS):
            found.append(fieldname)
    return sorted(set(found))


def _field_has_forbidden_token(fieldname: str, token: str) -> bool:
    folded = fieldname.casefold()
    parts = [part for part in folded.replace("-", "_").split("_") if part]
    return token in parts


def build_loop126_review_template(
    *,
    focus_blinded_csv: Path,
    output_annotations_csv: Path,
    output_json: Path,
    expected_rows: Optional[int] = None,
) -> dict:
    rows, fieldnames = read_csv_rows(focus_blinded_csv)
    blockers: list[str] = []
    if "review_focus_id" not in fieldnames:
        blockers.append("focus_missing_review_focus_id")
    forbidden_columns = _forbidden_public_columns(fieldnames)
    if forbidden_columns:
        blockers.append("focus_contains_identity_or_model_columns")
    if expected_rows is not None and len(rows) != expected_rows:
        blockers.append("focus_row_count_mismatch_expected")

    output_fieldnames = list(fieldnames)
    for field in MANUAL_FIELDS:
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    output_rows = []
    for row in rows:
        item = dict(row)
        for field in MANUAL_FIELDS:
            item[field] = ""
        output_rows.append(item)

    write_csv_rows(output_annotations_csv, output_rows, output_fieldnames)
    payload = {
        "schema": "axon_loop126_review_template_v1",
        "protocol": "build blinded Loop126 review annotation template; no private-map read, no unblind, no split/cache mutation",
        "identity_policy": (
            "The template is public/blinded. It excludes source_path/source_sha256/cache_path/sample_index and "
            "filename/path/directory/extension-derived fields. Manual verdicts must cite content or external evidence."
        ),
        "focus_blinded_csv": str(resolve_path(focus_blinded_csv)),
        "output_annotations_csv": str(resolve_path(output_annotations_csv)),
        "expected_rows": expected_rows,
        "rows": len(rows),
        "blockers": blockers,
        "template_ready": not blockers,
        "forbidden_columns": forbidden_columns,
        "manual_fields": MANUAL_FIELDS,
        "allowed_manual_label_verdicts": ALLOWED_MANUAL_LABEL_VERDICTS,
        "allowed_recommended_actions": ALLOWED_RECOMMENDED_ACTIONS,
        "automatic_relabel_allowed": False,
        "automatic_replacement_allowed": False,
    }
    resolved_json = resolve_path(output_json)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop126 blinded review annotation template.")
    parser.add_argument("--focus-blinded-csv", type=Path, required=True)
    parser.add_argument("--output-annotations-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop126_review_template(
        focus_blinded_csv=args.focus_blinded_csv,
        output_annotations_csv=args.output_annotations_csv,
        output_json=args.output_json,
        expected_rows=args.expected_rows,
    )
    print(json.dumps({"template_ready": payload["template_ready"], "rows": payload["rows"], "blockers": payload["blockers"]}, indent=2, ensure_ascii=False))
    return 0 if payload["template_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
