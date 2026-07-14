#!/usr/bin/env python3
"""Bridge Loop150 Val focus verdicts into strict redraw readiness.

This command is intentionally read-only. It converts already-preflighted
Loop150/Loop126 focus verdicts into a non-destructive fresh-redraw plan, then
asks Loop76 what the next allowed step is. It never trains, evaluates, samples
replacement files, mutates the split/cache, or treats identity fields as
verdict/model evidence.
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

from build_loop76_redraw_readiness import build_readiness, write_markdown as write_loop76_markdown  # noqa: E402


PROTOCOL = (
    "Loop152 read-only bridge from Loop150 Val high-conflict focus verdicts to Loop76 redraw readiness; "
    "confirmed label_wrong/feature_broken/out_of_scope rows become quarantine plus fresh same-original-label redraw requests only"
)
IDENTITY_FEATURE_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/split/row order/review_focus_id/model score fields are logistics, "
    "alignment, duplicate-check, and audit fields only; they are never verdict evidence, model evidence, "
    "feature-selection evidence, threshold evidence, replacement-sampling evidence, or production inference inputs"
)
PLAN_FIELDNAMES = [
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "original_label",
    "planned_label",
    "plan_action",
    "reason",
    "manual_label_verdict",
    "recommended_action",
    "manual_verdict_note",
    "replacement_required",
    "replacement_label",
    "usable_for_training_policy",
]
REQUIRED_PREFLIGHT_COLUMNS = {
    "review_focus_id",
    "current_label",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
    "loop126_status",
    "loop126_issue_flags",
    "loop126_replacement_required",
}
TRAINING_ALLOWED = {
    "training_allowed": False,
    "test10k_allowed": False,
    "full_test_allowed": False,
}


def resolve_path(path: Optional[Path | str]) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_json(path: Path | str) -> dict[str, Any]:
    resolved = resolve_path(path)
    assert resolved is not None
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def normalize(value: object) -> str:
    return "" if value is None else str(value).strip()


def normalize_bool(value: object) -> bool:
    return normalize(value).casefold() in {"1", "true", "yes", "y"}


def normalize_label(value: object) -> str:
    text = normalize(value).casefold()
    if text in {"0", "benign", "white", "clean"}:
        return "0"
    if text in {"1", "malicious", "black", "malware"}:
        return "1"
    return ""


def split_summary_and_index(split_csv: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    rows, _fieldnames = read_csv_rows(split_csv)
    split_counts: Counter[str] = Counter()
    label_split_counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter(), "test": Counter()}
    by_sample_index: dict[str, dict[str, str]] = {}
    by_source_sha256: dict[str, dict[str, str]] = {}

    for row in rows:
        split = normalize(row.get("split"))
        label = normalize(row.get("label"))
        split_counts[split] += 1
        if split in label_split_counts:
            label_split_counts[split][label] += 1
        sample_index = normalize(row.get("sample_index"))
        source_sha256 = normalize(row.get("source_sha256")).casefold()
        if sample_index:
            by_sample_index.setdefault(sample_index, row)
        if source_sha256:
            by_source_sha256.setdefault(source_sha256, row)

    return (
        {
            "rows": len(rows),
            "split_counts": dict(sorted(split_counts.items())),
            "label_split_counts": {split: dict(sorted(counts.items())) for split, counts in label_split_counts.items()},
        },
        by_sample_index,
        by_source_sha256,
    )


def private_map_by_focus_id(private_map_csv: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    rows, _fieldnames = read_csv_rows(private_map_csv)
    mapping: dict[str, dict[str, str]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        focus_id = normalize(row.get("review_focus_id"))
        if not focus_id:
            continue
        if focus_id in mapping:
            duplicate_ids.append(focus_id)
            continue
        mapping[focus_id] = row
    return mapping, duplicate_ids


def find_split_row(
    *,
    private_row: dict[str, str],
    by_sample_index: dict[str, dict[str, str]],
    by_source_sha256: dict[str, dict[str, str]],
) -> Optional[dict[str, str]]:
    sample_index = normalize(private_row.get("sample_index"))
    if sample_index and sample_index in by_sample_index:
        return by_sample_index[sample_index]
    source_sha256 = normalize(private_row.get("source_sha256")).casefold()
    if source_sha256 and source_sha256 in by_source_sha256:
        return by_source_sha256[source_sha256]
    return None


def preflight_is_ready(preflight: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers = list(preflight.get("blockers", []))
    if blockers:
        return False, [f"preflight:{item}" for item in blockers]
    if int(preflight.get("invalid_rows", 0) or 0) > 0:
        return False, ["preflight:invalid_rows_present"]
    manual_quality = preflight.get("manual_quality", {})
    if int(manual_quality.get("evidence_note_identity_or_score_only_rows", 0) or 0) > 0:
        return False, ["preflight:identity_or_model_score_only_evidence"]
    if int(manual_quality.get("evidence_note_missing_content_or_external_rows", 0) or 0) > 0:
        return False, ["preflight:missing_content_or_external_evidence"]
    return True, []


def build_strict_import_payload(
    *,
    preflight_json: Path,
    preflight: dict[str, Any],
    split_summary: dict[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    rows = int(preflight.get("rows", 0) or 0)
    expected_rows = preflight.get("expected_rows")
    import_ready = not blockers
    return {
        "schema": "axon_loop152_loop150_val_focus_import_v1",
        "protocol": PROTOCOL,
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "source_preflight_json": str(resolve_path(preflight_json)),
        "decision": "ready_for_redraw_plan_review_only"
        if import_ready and int(preflight.get("replacement_required_rows", 0) or 0) > 0
        else "ready_noop_no_actionable_verdicts"
        if import_ready
        else "blocked_invalid_verdicts",
        "import_ready": import_ready,
        "rows": rows,
        "review_rows": rows,
        "expected_rows": expected_rows,
        "invalid_rows": int(preflight.get("invalid_rows", 0) or 0),
        "training_policy_rows": 0,
        "blocking_issues": list(blockers),
        "duplicate_sample_index_rows": 0,
        "status_counts": dict(preflight.get("status_counts", {})),
        "manual_quality": preflight.get(
            "manual_quality",
            {
                "blank_verdict_rows": int(preflight.get("status_counts", {}).get("no_decision", 0) or 0),
                "actionable_verdict_missing_note_rows": 0,
                "evidence_note_missing_content_or_external_rows": 0,
                "evidence_note_identity_or_score_only_rows": 0,
            },
        ),
        "split_summary": split_summary,
        "notes": [
            "Loop150 focus rows are Val-only governance inputs; they do not authorize Test-10k or full-test evaluation.",
            "Only preflighted replacement_required rows become fresh same-original-label redraw requests.",
        ],
    }


def build_strict_adjustment_plan(
    *,
    preflight_csv: Path,
    private_map_csv: Path,
    split_csv: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    rows, fieldnames = read_csv_rows(preflight_csv)
    private_by_id, duplicate_private_ids = private_map_by_focus_id(private_map_csv)
    split_summary, by_sample_index, by_source_sha256 = split_summary_and_index(split_csv)

    missing_columns = sorted(REQUIRED_PREFLIGHT_COLUMNS - set(fieldnames))
    blockers: list[str] = []
    if missing_columns:
        blockers.append("preflight_csv_missing_required_columns")
    if duplicate_private_ids:
        blockers.append("private_map_duplicate_review_focus_id")

    plan_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    replacement_counts: Counter[str] = Counter()
    split_action_counts: Counter[str] = Counter()
    missing_private_map_rows = 0
    missing_split_rows = 0
    split_mismatch_rows = 0
    label_mismatch_rows = 0
    invalid_status_rows = 0
    invalid_label_rows = 0
    duplicate_plan_keys = 0
    seen_plan_keys: set[str] = set()

    for row in rows:
        status = normalize(row.get("loop126_status"))
        status_counts[status or "blank"] += 1
        verdict_counts[normalize(row.get("manual_label_verdict")) or "blank"] += 1
        if status == "invalid" or normalize(row.get("loop126_issue_flags")):
            invalid_status_rows += 1
        if not normalize_bool(row.get("loop126_replacement_required")):
            continue

        focus_id = normalize(row.get("review_focus_id"))
        private_row = private_by_id.get(focus_id)
        if private_row is None:
            missing_private_map_rows += 1
            continue

        split_row = find_split_row(
            private_row=private_row,
            by_sample_index=by_sample_index,
            by_source_sha256=by_source_sha256,
        )
        if split_row is None:
            missing_split_rows += 1
            split = normalize(private_row.get("split") or row.get("split"))
            label_text = normalize_label(row.get("current_label") or private_row.get("label"))
            source_path = private_row.get("source_path", "")
            source_sha256 = private_row.get("source_sha256", "")
            sample_index = private_row.get("sample_index", "")
        else:
            split = normalize(split_row.get("split"))
            label_text = normalize_label(split_row.get("label"))
            source_path = split_row.get("source_path", private_row.get("source_path", ""))
            source_sha256 = split_row.get("source_sha256", private_row.get("source_sha256", ""))
            sample_index = split_row.get("sample_index", private_row.get("sample_index", ""))
            private_split = normalize(private_row.get("split"))
            if private_split and private_split != split:
                split_mismatch_rows += 1
            for label_source in [row.get("current_label"), private_row.get("label")]:
                candidate_label = normalize_label(label_source)
                if candidate_label and candidate_label != label_text:
                    label_mismatch_rows += 1

        if label_text not in {"0", "1"}:
            invalid_label_rows += 1
            replacement_label = ""
        else:
            replacement_label = label_text

        plan_key = sample_index or normalize(source_sha256).casefold() or focus_id
        if plan_key in seen_plan_keys:
            duplicate_plan_keys += 1
        seen_plan_keys.add(plan_key)

        reason_verdict = normalize(row.get("manual_label_verdict")) or "confirmed_bad_row"
        plan_row = {
            "source_path": source_path,
            "source_sha256": source_sha256,
            "sample_index": sample_index,
            "split": split,
            "original_label": replacement_label,
            "planned_label": replacement_label,
            "plan_action": "exclude_and_replace",
            "reason": f"loop150_val_focus_{reason_verdict}_fresh_same_original_label_redraw",
            "manual_label_verdict": row.get("manual_label_verdict", ""),
            "recommended_action": row.get("recommended_action", ""),
            "manual_verdict_note": row.get("manual_verdict_note", ""),
            "replacement_required": "true",
            "replacement_label": replacement_label,
            "usable_for_training_policy": "false",
        }
        plan_rows.append(plan_row)
        if replacement_label:
            replacement_counts[replacement_label] += 1
        split_action_counts[f"{split}:exclude_and_replace"] += 1

    if invalid_status_rows:
        blockers.append("preflight_csv_contains_invalid_rows")
    if missing_private_map_rows:
        blockers.append("adjustment_plan_missing_private_map_rows")
    if missing_split_rows:
        blockers.append("adjustment_plan_has_missing_split_rows")
    if split_mismatch_rows:
        blockers.append("adjustment_plan_split_alignment_mismatch")
    if label_mismatch_rows:
        blockers.append("adjustment_plan_label_alignment_mismatch")
    if duplicate_plan_keys:
        blockers.append("adjustment_plan_duplicate_review_rows")
    if invalid_label_rows:
        blockers.append("adjustment_plan_invalid_original_label")

    payload = {
        "schema": "axon_loop152_loop150_val_focus_redraw_adjustment_plan_v1",
        "protocol": PROTOCOL,
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "review_rows": len(rows),
        "planned_rows": len(plan_rows),
        "ignored_rows": max(0, len(rows) - len(plan_rows)),
        "unknown_verdict_rows": 0,
        "missing_split_rows": missing_split_rows,
        "duplicate_review_rows": duplicate_plan_keys,
        "replacement_required": len(plan_rows),
        "replacement_counts_by_original_label": dict(sorted(replacement_counts.items())),
        "training_policy_rows": 0,
        "review_rows_in_test_split": sum(1 for item in plan_rows if item.get("split") == "test"),
        "action_counts": {"exclude_and_replace": len(plan_rows)} if plan_rows else {},
        "split_action_counts": dict(sorted(split_action_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "alignment_quality": {
            "missing_private_map_rows": missing_private_map_rows,
            "missing_split_rows": missing_split_rows,
            "split_mismatch_rows": split_mismatch_rows,
            "label_mismatch_rows": label_mismatch_rows,
            "duplicate_plan_keys": duplicate_plan_keys,
            "invalid_label_rows": invalid_label_rows,
        },
        "split_summary": split_summary,
        "notes": [
            "All confirmed bad rows are converted to exclude_and_replace; direct relabel is forbidden in this protocol.",
            "source_path/source_sha256/sample_index/split are copied only so downstream audits can find the row.",
            "replacement_label is always the original locked split label; the bad row never fills its own slot.",
        ],
    }
    return payload, plan_rows, sorted(set(blockers))


def blocked_summary(
    *,
    decision: str,
    blockers: Sequence[str],
    preflight_json: Path,
    output_dir: Path,
    output_json: Path,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": "axon_loop152_loop150_val_focus_redraw_readiness_v1",
        "protocol": PROTOCOL,
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {"preflight_json": str(preflight_json)},
        "output_dir": str(output_dir),
        "decision": decision,
        "blockers": sorted(set(blockers)),
        "ready_for": {
            "fresh_redraw": False,
            "cache_recovery": False,
            "train_val_only": False,
            "test10k": False,
            "full_test": False,
        },
        "decisions": dict(TRAINING_ALLOWED),
        "outputs": {},
        "notes": [
            "Preflight, private-map alignment, and split alignment must pass before Loop76 redraw readiness.",
            "No training, Test-10k, or full-test evaluation is authorized from this state.",
        ],
    }
    if extra:
        summary.update(extra)
    write_json(output_json, summary)
    return summary


def write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Loop152 Loop150 Val Focus Redraw Readiness",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Blockers: `{payload.get('blockers', [])}`",
        f"- Replacement required: `{payload.get('counts', {}).get('replacement_required', 0)}`",
        f"- Training allowed: `{payload.get('decisions', {}).get('training_allowed', False)}`",
        f"- Test-10k allowed: `{payload.get('decisions', {}).get('test10k_allowed', False)}`",
        f"- Full test allowed: `{payload.get('decisions', {}).get('full_test_allowed', False)}`",
        "",
        "## Policy",
        "",
        IDENTITY_FEATURE_POLICY,
        "",
    ]
    loop76 = payload.get("loop76_readiness", {})
    if loop76:
        lines.extend(
            [
                "## Loop76",
                "",
                f"- Decision: `{loop76.get('decision')}`",
                f"- Next step: `{loop76.get('next_step')}`",
                "",
            ]
        )
    resolved.write_text("\n".join(lines), encoding="utf-8")


def run_loop150_val_focus_redraw_readiness(
    *,
    preflight_csv: Path,
    preflight_json: Path,
    private_map_csv: Path,
    split_csv: Path,
    output_dir: Path,
    output_json: Path,
    output_md: Optional[Path] = None,
    candidate_pool_json: Optional[Path] = None,
    corrected_split_json: Optional[Path] = None,
    replacement_audit_json: Optional[Path] = None,
    cache_ready_json: Optional[Path] = None,
    split_metadata_json: Optional[Path] = None,
    manifest_json: Optional[Path] = None,
    data_dir: Path = Path("data"),
    output_prefix: Optional[Path] = None,
    candidate_csv: Optional[Path] = None,
    corrected_split_csv: Optional[Path] = None,
    enforce_label_balance: bool = True,
) -> dict[str, Any]:
    resolved_output_dir = resolve_path(output_dir)
    assert resolved_output_dir is not None
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix if output_prefix is not None else resolved_output_dir / "loop152"
    strict_import_json = resolved_output_dir / "loop152_strict_import.json"
    loop76_json = resolved_output_dir / "loop152_loop76_readiness.json"
    loop76_md = resolved_output_dir / "loop152_loop76_readiness.md"
    plan_csv = resolved_output_dir / "loop152_strict_adjustment_plan.csv"
    plan_json = resolved_output_dir / "loop152_strict_adjustment_plan.json"

    preflight = read_json(preflight_json)
    preflight_ready, preflight_blockers = preflight_is_ready(preflight)
    adjustment_payload, plan_rows, plan_blockers = build_strict_adjustment_plan(
        preflight_csv=preflight_csv,
        private_map_csv=private_map_csv,
        split_csv=split_csv,
    )
    all_blockers = sorted(set(preflight_blockers + plan_blockers))
    strict_import = build_strict_import_payload(
        preflight_json=preflight_json,
        preflight=preflight,
        split_summary=adjustment_payload["split_summary"],
        blockers=all_blockers,
    )
    write_json(strict_import_json, strict_import)
    write_csv_rows(plan_csv, plan_rows, PLAN_FIELDNAMES)
    write_json(plan_json, adjustment_payload)

    if not preflight_ready or all_blockers:
        return blocked_summary(
            decision="blocked_before_loop76_redraw_readiness",
            blockers=all_blockers,
            preflight_json=preflight_json,
            output_dir=output_dir,
            output_json=output_json,
            extra={
                "outputs": {
                    "strict_import_json": str(strict_import_json),
                    "adjustment_plan_csv": str(plan_csv),
                    "adjustment_plan_json": str(plan_json),
                },
                "counts": {
                    "replacement_required": adjustment_payload["replacement_required"],
                    "training_policy_rows": adjustment_payload["training_policy_rows"],
                },
            },
        )

    loop76_payload = build_readiness(
        strict_import_json=strict_import_json,
        adjustment_plan_json=plan_json,
        candidate_pool_json=candidate_pool_json,
        corrected_split_json=corrected_split_json,
        replacement_audit_json=replacement_audit_json,
        cache_ready_json=cache_ready_json,
        split_metadata_json=split_metadata_json,
        split_csv=split_csv,
        plan_csv=plan_csv,
        candidate_csv=candidate_csv,
        corrected_split_csv=corrected_split_csv,
        manifest_json=manifest_json,
        data_dir=data_dir,
        output_prefix=prefix,
        enforce_label_balance=enforce_label_balance,
    )
    write_json(loop76_json, loop76_payload)
    write_loop76_markdown(loop76_md, loop76_payload)

    summary = {
        "schema": "axon_loop152_loop150_val_focus_redraw_readiness_v1",
        "protocol": PROTOCOL,
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {
            "preflight_csv": str(preflight_csv),
            "preflight_json": str(preflight_json),
            "private_map_csv": str(private_map_csv),
            "split_csv": str(split_csv),
        },
        "output_dir": str(output_dir),
        "decision": loop76_payload["decision"],
        "blockers": list(loop76_payload.get("strict_failures", [])),
        "counts": {
            "preflight_rows": preflight.get("rows"),
            "preflight_annotated_rows": preflight.get("annotated_rows"),
            "preflight_replacement_required_rows": preflight.get("replacement_required_rows"),
            "replacement_required": adjustment_payload["replacement_required"],
            "training_policy_rows": adjustment_payload["training_policy_rows"],
            "plan_rows": len(plan_rows),
        },
        "ready_for": loop76_payload["ready_for"],
        "decisions": {
            **TRAINING_ALLOWED,
            "fresh_redraw_allowed": bool(loop76_payload["ready_for"].get("fresh_redraw")),
            "next_allowed_step": loop76_payload.get("next_step"),
        },
        "outputs": {
            "strict_import_json": str(strict_import_json),
            "adjustment_plan_csv": str(plan_csv),
            "adjustment_plan_json": str(plan_json),
            "loop76_readiness_json": str(loop76_json),
            "loop76_readiness_md": str(loop76_md),
        },
        "loop76_readiness": {
            "decision": loop76_payload["decision"],
            "strict_failures": loop76_payload.get("strict_failures", []),
            "next_step": loop76_payload.get("next_step"),
            "ready_for": loop76_payload.get("ready_for", {}),
        },
        "notes": [
            "Loop152 does not sample replacement files. It only proves whether replacement sampling is the next allowed step.",
            "label_wrong, feature_broken, and out_of_scope verdicts are all handled as exclude_and_replace, never direct relabel.",
            "Loop150 full-test focus rows must not be used for model or threshold selection; this bridge is intended for Val focus governance.",
        ],
    }
    write_json(output_json, summary)
    if output_md is not None:
        write_summary_markdown(output_md, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge Loop150 Val focus preflight output into Loop76 redraw readiness.")
    parser.add_argument("--preflight-csv", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--private-map-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--candidate-pool-json", type=Path, default=None)
    parser.add_argument("--corrected-split-json", type=Path, default=None)
    parser.add_argument("--replacement-audit-json", type=Path, default=None)
    parser.add_argument("--cache-ready-json", type=Path, default=None)
    parser.add_argument("--split-metadata-json", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--corrected-split-csv", type=Path, default=None)
    parser.add_argument("--no-enforce-label-balance", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_loop150_val_focus_redraw_readiness(
        preflight_csv=args.preflight_csv,
        preflight_json=args.preflight_json,
        private_map_csv=args.private_map_csv,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        output_json=args.output_json,
        output_md=args.output_md,
        candidate_pool_json=args.candidate_pool_json,
        corrected_split_json=args.corrected_split_json,
        replacement_audit_json=args.replacement_audit_json,
        cache_ready_json=args.cache_ready_json,
        split_metadata_json=args.split_metadata_json,
        manifest_json=args.manifest_json,
        data_dir=args.data_dir,
        output_prefix=args.output_prefix,
        candidate_csv=args.candidate_csv,
        corrected_split_csv=args.corrected_split_csv,
        enforce_label_balance=not bool(args.no_enforce_label_balance),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not str(summary["decision"]).startswith("blocked") else 2


if __name__ == "__main__":
    raise SystemExit(main())
