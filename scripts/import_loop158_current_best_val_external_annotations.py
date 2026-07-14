#!/usr/bin/env python3
"""Import Loop157 external annotations through a strict Loop158 guard.

The external return file is deliberately tiny: only the review key and three
manual fields are accepted. Current labels are restored from the already-safe
Loop157 context file after that field audit passes, then Loop126 and Loop152
handle verdict quality and same-original-label redraw readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from preflight_loop126_review_annotations import preflight_loop126_review_annotations  # noqa: E402
from run_loop152_loop150_val_focus_redraw_readiness import (  # noqa: E402
    run_loop150_val_focus_redraw_readiness,
)


ANNOTATION_FIELDS = ["review_focus_id", "manual_label_verdict", "manual_verdict_note", "recommended_action"]
PREFLIGHT_INPUT_FIELDS = ["review_focus_id", "current_label", *ANNOTATION_FIELDS[1:]]
FORBIDDEN_HEADER_TOKENS = [
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
    "split",
    "rank",
    "neighbor",
    "similarity",
]
FORBIDDEN_NOTE_TERMS = [
    "filename",
    "file name",
    "path",
    "directory",
    "folder",
    "extension",
    "source_path",
    "cache_path",
    "source_sha256",
    "sha256",
    "hash",
    "sample_index",
    "split",
    "row order",
    "rank",
    "model score",
    "probability",
    "prob_malicious",
    "prediction",
    "threshold",
    "confidence",
]
PROTOCOL = (
    "Loop158 strict import guard for Loop157 returned annotations; exact four-column external return, "
    "Loop126 verdict preflight, then Loop152/Loop76 same-original-label redraw readiness only"
)
IDENTITY_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/filename/directory/extension/split/row order/model score/"
    "probability/prediction/threshold fields are forbidden in returned annotations and are never verdict, model, "
    "feature-mask, threshold, replacement-sampling, or production inference evidence"
)
TRAINING_DECISIONS = {
    "training_allowed": False,
    "test10k_allowed": False,
    "full_test_allowed": False,
}


def resolve_path(path: Optional[Path | str]) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_csv_rows(path: Path | str) -> tuple[list[dict[str, str]], list[str]]:
    resolved = resolve_path(path)
    assert resolved is not None
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(path: Path | str, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize(value: object) -> str:
    return str(value or "").strip()


def _tokenized(field: str) -> set[str]:
    return {part for part in field.casefold().replace("-", "_").split("_") if part}


def forbidden_columns(fieldnames: Sequence[str]) -> list[str]:
    found: list[str] = []
    for field in fieldnames:
        parts = _tokenized(field)
        if any(token in parts for token in FORBIDDEN_HEADER_TOKENS):
            found.append(field)
    return sorted(set(found))


def note_identity_violations(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        note = normalize(row.get("manual_verdict_note")).casefold()
        if not note:
            continue
        hits = [term for term in FORBIDDEN_NOTE_TERMS if term in note]
        if hits:
            violations.append(
                {
                    "row_number": row_number,
                    "review_focus_id": normalize(row.get("review_focus_id")),
                    "terms": sorted(set(hits)),
                }
            )
    return violations


def context_by_review_id(context_csv: Path | str) -> tuple[dict[str, dict[str, str]], list[str], dict[str, Any]]:
    rows, fieldnames = read_csv_rows(context_csv)
    missing_required = [field for field in ["review_focus_id", "current_label"] if field not in fieldnames]
    mapping: dict[str, dict[str, str]] = {}
    duplicate_ids: list[str] = []
    missing_id_rows = 0
    label_counts: Counter[str] = Counter()
    for row in rows:
        review_id = normalize(row.get("review_focus_id"))
        label = normalize(row.get("current_label"))
        if label:
            label_counts[label] += 1
        if not review_id:
            missing_id_rows += 1
            continue
        if review_id in mapping:
            duplicate_ids.append(review_id)
            continue
        mapping[review_id] = row
    summary = {
        "rows": len(rows),
        "field_count": len(fieldnames),
        "missing_required_columns": missing_required,
        "missing_review_focus_id_rows": missing_id_rows,
        "duplicate_review_focus_id_rows": len(duplicate_ids),
        "label_counts": dict(sorted(label_counts.items())),
    }
    return mapping, fieldnames, summary


def audit_external_annotations(
    *,
    annotation_rows: Sequence[dict[str, str]],
    annotation_fieldnames: Sequence[str],
    context_ids: Sequence[str],
    expected_rows: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    exact_columns = list(annotation_fieldnames) == ANNOTATION_FIELDS
    found_forbidden_columns = forbidden_columns(annotation_fieldnames)
    if not exact_columns:
        blockers.append("external_annotations_must_contain_exact_four_columns")
    if found_forbidden_columns:
        blockers.append("external_annotations_contain_identity_or_model_columns")
    if len(annotation_rows) > expected_rows:
        blockers.append("external_annotation_row_count_exceeds_context")
    note_violations = note_identity_violations(annotation_rows)
    if note_violations:
        blockers.append("external_annotation_notes_reference_identity_or_model_terms")

    expected_set = set(context_ids)
    seen: set[str] = set()
    missing_id_rows = 0
    duplicate_ids: list[str] = []
    unknown_ids: list[str] = []
    for row in annotation_rows:
        review_id = normalize(row.get("review_focus_id"))
        if not review_id:
            missing_id_rows += 1
            continue
        if review_id in seen:
            duplicate_ids.append(review_id)
        seen.add(review_id)
        if review_id not in expected_set:
            unknown_ids.append(review_id)

    unreturned_expected_ids = sorted(expected_set - seen)
    if missing_id_rows:
        blockers.append("external_annotations_missing_review_focus_id")
    if duplicate_ids:
        blockers.append("external_annotations_duplicate_review_focus_id")
    if unknown_ids:
        blockers.append("external_annotations_unknown_review_focus_id")

    annotated_rows = sum(
        1
        for row in annotation_rows
        if any(normalize(row.get(field)) for field in ANNOTATION_FIELDS[1:])
    )
    blank_manual_rows = max(0, len(annotation_rows) - annotated_rows)
    if blank_manual_rows:
        blockers.append("external_annotations_blank_manual_rows")
    return {
        "exact_columns": exact_columns,
        "required_columns": ANNOTATION_FIELDS,
        "input_columns": list(annotation_fieldnames),
        "forbidden_columns": found_forbidden_columns,
        "note_identity_or_model_term_rows": len(note_violations),
        "note_identity_or_model_term_examples": note_violations[:10],
        "rows": len(annotation_rows),
        "context_rows": expected_rows,
        "annotated_rows": annotated_rows,
        "blank_manual_rows": blank_manual_rows,
        "missing_review_focus_id_rows": missing_id_rows,
        "duplicate_review_focus_id_rows": len(duplicate_ids),
        "unknown_review_focus_id_rows": len(unknown_ids),
        "unreturned_context_review_focus_id_rows": len(unreturned_expected_ids),
        "duplicate_review_focus_id_examples": sorted(set(duplicate_ids))[:10],
        "unknown_review_focus_id_examples": sorted(set(unknown_ids))[:10],
        "unreturned_context_review_focus_id_examples": unreturned_expected_ids[:10],
        "blockers": sorted(set(blockers)),
    }


def build_preflight_input_rows(
    *,
    annotation_rows: Sequence[dict[str, str]],
    context_rows_by_id: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for annotation in annotation_rows:
        review_id = normalize(annotation.get("review_focus_id"))
        context_row = context_rows_by_id[review_id]
        output_rows.append(
            {
                "review_focus_id": review_id,
                "current_label": normalize(context_row.get("current_label")),
                "manual_label_verdict": normalize(annotation.get("manual_label_verdict")),
                "manual_verdict_note": normalize(annotation.get("manual_verdict_note")),
                "recommended_action": normalize(annotation.get("recommended_action")),
            }
        )
    return output_rows


def _summary_base(
    *,
    returned_annotations_csv: Path,
    context_csv: Path,
    output_dir: Path,
    output_json: Path,
    expected_rows: int,
    external_audit: dict[str, Any],
    context_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "axon_loop158_external_annotation_import_guard_v1",
        "protocol": PROTOCOL,
        "identity_feature_policy": IDENTITY_POLICY,
        "inputs": {
            "returned_annotations_csv": str(resolve_path(returned_annotations_csv)),
            "context_csv": str(resolve_path(context_csv)),
        },
        "output_dir": str(resolve_path(output_dir)),
        "output_json": str(resolve_path(output_json)),
        "expected_rows": expected_rows,
        "context_summary": context_summary,
        "external_annotation_audit": external_audit,
        "private_join_performed": False,
        "decisions": dict(TRAINING_DECISIONS),
        "ready_for": {
            "fresh_redraw": False,
            "cache_recovery": False,
            "train_val_only": False,
            "test10k": False,
            "full_test": False,
        },
        "outputs": {},
        "notes": [
            "Returned annotations are accepted only as review governance input.",
            "No automatic relabeling, replacement, training, Test-10k, or full-test evaluation is authorized here.",
        ],
    }


def write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Loop158 External Annotation Import Guard",
        "",
        f"- Decision: `{payload.get('decision')}`",
        f"- Blockers: `{payload.get('blockers', [])}`",
        f"- Rows: `{payload.get('external_annotation_audit', {}).get('rows')}`",
        f"- Annotated rows: `{payload.get('external_annotation_audit', {}).get('annotated_rows')}`",
        f"- Private join performed: `{payload.get('private_join_performed', False)}`",
        f"- Fresh redraw allowed: `{payload.get('ready_for', {}).get('fresh_redraw', False)}`",
        f"- Train/Val allowed: `{payload.get('ready_for', {}).get('train_val_only', False)}`",
        f"- Test-10k allowed: `{payload.get('ready_for', {}).get('test10k', False)}`",
        f"- Full test allowed: `{payload.get('ready_for', {}).get('full_test', False)}`",
        "",
        "## Policy",
        "",
        IDENTITY_POLICY,
        "",
    ]
    resolved.write_text("\n".join(lines), encoding="utf-8")


def import_loop158_external_annotations(
    *,
    returned_annotations_csv: Path,
    context_csv: Path,
    private_map_csv: Path,
    split_csv: Path,
    output_dir: Path,
    output_json: Path,
    output_md: Optional[Path] = None,
    expected_rows: Optional[int] = 162,
    manifest_json: Optional[Path] = None,
    data_dir: Path = Path("data"),
    output_prefix: Optional[Path] = None,
    candidate_pool_json: Optional[Path] = None,
    corrected_split_json: Optional[Path] = None,
    replacement_audit_json: Optional[Path] = None,
    cache_ready_json: Optional[Path] = None,
    split_metadata_json: Optional[Path] = None,
    candidate_csv: Optional[Path] = None,
    corrected_split_csv: Optional[Path] = None,
    enforce_label_balance: bool = True,
) -> dict[str, Any]:
    resolved_output_dir = resolve_path(output_dir)
    assert resolved_output_dir is not None
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    context_rows, _context_fields, context_summary = context_by_review_id(context_csv)
    annotation_rows, annotation_fields = read_csv_rows(returned_annotations_csv)
    expected = int(expected_rows if expected_rows is not None else len(context_rows))
    context_blockers: list[str] = []
    if context_summary["missing_required_columns"]:
        context_blockers.append("context_missing_required_columns")
    if context_summary["missing_review_focus_id_rows"]:
        context_blockers.append("context_missing_review_focus_id")
    if context_summary["duplicate_review_focus_id_rows"]:
        context_blockers.append("context_duplicate_review_focus_id")
    if len(context_rows) != expected:
        context_blockers.append("context_row_count_mismatch")

    external_audit = audit_external_annotations(
        annotation_rows=annotation_rows,
        annotation_fieldnames=annotation_fields,
        context_ids=list(context_rows.keys()),
        expected_rows=expected,
    )
    summary = _summary_base(
        returned_annotations_csv=returned_annotations_csv,
        context_csv=context_csv,
        output_dir=output_dir,
        output_json=output_json,
        expected_rows=expected,
        external_audit=external_audit,
        context_summary=context_summary,
    )

    early_blockers = sorted(set(context_blockers + list(external_audit["blockers"])))
    if early_blockers:
        summary.update(
            {
                "decision": "blocked_external_annotation_import",
                "blockers": early_blockers,
                "next_action": "return_a_complete_exact_four_column_annotation_file_before_private_mapping",
            }
        )
        write_json(output_json, summary)
        if output_md is not None:
            write_summary_markdown(output_md, summary)
        return summary

    if int(external_audit["rows"]) == 0:
        summary.update(
            {
                "decision": "ready_noop_no_external_annotations",
                "blockers": [],
                "next_action": "await_external_content_or_manual_verdict_rows",
            }
        )
        write_json(output_json, summary)
        if output_md is not None:
            write_summary_markdown(output_md, summary)
        return summary

    preflight_input_csv = resolved_output_dir / "loop158_external_return_preflight_input.csv"
    preflight_csv = resolved_output_dir / "loop158_external_return_preflight.csv"
    preflight_json = resolved_output_dir / "loop158_external_return_preflight.json"
    redraw_summary_json = resolved_output_dir / "loop158_redraw_readiness_summary.json"
    redraw_summary_md = resolved_output_dir / "loop158_redraw_readiness_summary.md"
    preflight_rows = build_preflight_input_rows(
        annotation_rows=annotation_rows,
        context_rows_by_id=context_rows,
    )
    write_csv_rows(preflight_input_csv, preflight_rows, PREFLIGHT_INPUT_FIELDS)
    preflight = preflight_loop126_review_annotations(
        annotations_csv=preflight_input_csv,
        output_csv=preflight_csv,
        output_json=preflight_json,
        expected_rows=int(external_audit["rows"]),
    )
    summary["outputs"].update(
        {
            "preflight_input_csv": str(preflight_input_csv),
            "preflight_csv": str(preflight_csv),
            "preflight_json": str(preflight_json),
        }
    )
    summary["preflight"] = {
        "rows": preflight.get("rows"),
        "annotated_rows": preflight.get("annotated_rows"),
        "actionable_rows": preflight.get("actionable_rows"),
        "replacement_required_rows": preflight.get("replacement_required_rows"),
        "invalid_rows": preflight.get("invalid_rows", 0),
        "blockers": preflight.get("blockers", []),
        "status_counts": preflight.get("status_counts", {}),
        "issue_counts": preflight.get("issue_counts", {}),
        "manual_quality": preflight.get("manual_quality", {}),
    }
    preflight_blockers = list(preflight.get("blockers", []))
    if int(preflight.get("invalid_rows", 0) or 0) > 0:
        preflight_blockers.append("preflight_invalid_rows_present")
    if preflight_blockers:
        summary.update(
            {
                "decision": "blocked_preflight_failed_no_private_join",
                "blockers": sorted(set(preflight_blockers)),
                "next_action": "fix_returned_annotation_verdicts_or_evidence_notes_before_private_mapping",
            }
        )
        write_json(output_json, summary)
        if output_md is not None:
            write_summary_markdown(output_md, summary)
        return summary

    prefix = output_prefix if output_prefix is not None else resolved_output_dir / "loop158"
    redraw = run_loop150_val_focus_redraw_readiness(
        preflight_csv=preflight_csv,
        preflight_json=preflight_json,
        private_map_csv=private_map_csv,
        split_csv=split_csv,
        output_dir=resolved_output_dir,
        output_json=redraw_summary_json,
        output_md=redraw_summary_md,
        candidate_pool_json=candidate_pool_json,
        corrected_split_json=corrected_split_json,
        replacement_audit_json=replacement_audit_json,
        cache_ready_json=cache_ready_json,
        split_metadata_json=split_metadata_json,
        manifest_json=manifest_json,
        data_dir=data_dir,
        output_prefix=prefix,
        candidate_csv=candidate_csv,
        corrected_split_csv=corrected_split_csv,
        enforce_label_balance=enforce_label_balance,
    )
    summary.update(
        {
            "decision": redraw.get("decision"),
            "blockers": list(redraw.get("blockers", [])),
            "private_join_performed": True,
            "redraw_readiness": {
                "decision": redraw.get("decision"),
                "counts": redraw.get("counts", {}),
                "ready_for": redraw.get("ready_for", {}),
                "decisions": redraw.get("decisions", {}),
                "loop76_readiness": redraw.get("loop76_readiness", {}),
            },
            "ready_for": redraw.get("ready_for", summary["ready_for"]),
            "decisions": {
                **TRAINING_DECISIONS,
                "fresh_redraw_allowed": bool(redraw.get("ready_for", {}).get("fresh_redraw")),
                "next_allowed_step": redraw.get("decisions", {}).get("next_allowed_step"),
            },
            "next_action": redraw.get("decisions", {}).get("next_allowed_step")
            or redraw.get("loop76_readiness", {}).get("next_step")
            or "await_external_verdicts",
        }
    )
    summary["outputs"].update(
        {
            "redraw_readiness_json": str(redraw_summary_json),
            "redraw_readiness_md": str(redraw_summary_md),
            **{
                f"redraw_{key}": value
                for key, value in dict(redraw.get("outputs", {})).items()
            },
        }
    )
    write_json(output_json, summary)
    if output_md is not None:
        write_summary_markdown(output_md, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly import Loop157 returned external annotations.")
    parser.add_argument("--returned-annotations-csv", type=Path, required=True)
    parser.add_argument("--context-csv", type=Path, required=True)
    parser.add_argument("--private-map-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=162)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--candidate-pool-json", type=Path, default=None)
    parser.add_argument("--corrected-split-json", type=Path, default=None)
    parser.add_argument("--replacement-audit-json", type=Path, default=None)
    parser.add_argument("--cache-ready-json", type=Path, default=None)
    parser.add_argument("--split-metadata-json", type=Path, default=None)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--corrected-split-csv", type=Path, default=None)
    parser.add_argument("--no-enforce-label-balance", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = import_loop158_external_annotations(
        returned_annotations_csv=args.returned_annotations_csv,
        context_csv=args.context_csv,
        private_map_csv=args.private_map_csv,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        output_json=args.output_json,
        output_md=args.output_md,
        expected_rows=args.expected_rows,
        manifest_json=args.manifest_json,
        data_dir=args.data_dir,
        output_prefix=args.output_prefix,
        candidate_pool_json=args.candidate_pool_json,
        corrected_split_json=args.corrected_split_json,
        replacement_audit_json=args.replacement_audit_json,
        cache_ready_json=args.cache_ready_json,
        split_metadata_json=args.split_metadata_json,
        candidate_csv=args.candidate_csv,
        corrected_split_csv=args.corrected_split_csv,
        enforce_label_balance=not bool(args.no_enforce_label_balance),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2 if str(summary.get("decision", "")).startswith("blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
