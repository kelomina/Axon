#!/usr/bin/env python3
"""Bridge Loop112 external verdict output into Loop76 redraw readiness.

This command is intentionally read-only. It does not train, evaluate, load a
checkpoint, open NPZ feature arrays, sample replacements, or mutate split/cache
state. It only converts strict Loop87 replacement verdicts into a non-destructive
fresh-redraw plan and then asks Loop76 what the next allowed step is.
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

from build_loop76_redraw_readiness import (  # noqa: E402
    build_readiness,
    write_markdown as write_loop76_markdown,
)


PROTOCOL = (
    "Loop114 read-only bridge from Loop112/Loop87 strict external verdicts to Loop76 redraw readiness; "
    "confirmed bad rows become quarantine plus fresh same-original-label redraw requests only"
)
IDENTITY_FEATURE_POLICY = (
    "filename/path/extension/directory/source_sha256/sample_index/split/row order/model score are logistics, "
    "alignment, duplicate-check, and audit fields only; they are never verdict evidence, model evidence, "
    "feature-selection evidence, threshold evidence, or production inference inputs"
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
REQUIRED_LOOP87_COLUMNS = {
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "label",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
    "loop87_status",
    "loop87_issue_flags",
    "loop87_replacement_required",
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
    return json.loads(resolved.read_text(encoding="utf-8"))


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


def parse_label(value: object) -> Optional[int]:
    text = normalize(value).casefold()
    if text in {"0", "benign", "white", "clean"}:
        return 0
    if text in {"1", "malicious", "black", "malware"}:
        return 1
    return None


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
            "label_split_counts": {
                split: dict(sorted(counts.items()))
                for split, counts in label_split_counts.items()
            },
        },
        by_sample_index,
        by_source_sha256,
    )


def find_split_row(
    row: dict[str, str],
    by_sample_index: dict[str, dict[str, str]],
    by_source_sha256: dict[str, dict[str, str]],
) -> Optional[dict[str, str]]:
    sample_index = normalize(row.get("sample_index"))
    if sample_index and sample_index in by_sample_index:
        return by_sample_index[sample_index]
    source_sha256 = normalize(row.get("source_sha256")).casefold()
    if source_sha256 and source_sha256 in by_source_sha256:
        return by_source_sha256[source_sha256]
    return None


def loop112_is_ready(loop112: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers = list(loop112.get("blockers", []))
    if blockers:
        return False, [f"loop112:{item}" for item in blockers]
    loop110_stage = loop112.get("stages", {}).get("loop110_focus_verdict_pipeline", {})
    if loop110_stage and not loop110_stage.get("passed", False):
        return False, ["loop112:loop110_stage_not_passed"]
    if not loop112.get("outputs", {}).get("loop110_json"):
        return False, ["loop112:missing_loop110_json"]
    return True, []


def loop110_is_ready(loop110: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers = list(loop110.get("blockers", []))
    if blockers:
        return False, [f"loop110:{item}" for item in blockers]
    outputs = loop110.get("outputs", {})
    missing = [name for name in ["loop87_validated_csv", "loop87_json"] if not outputs.get(name)]
    if missing:
        return False, [f"loop110:missing_{name}" for name in missing]
    return True, []


def loop87_is_ready(loop87: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers = [f"loop87:{item}" for item in loop87.get("blocking_issues", [])]
    if not loop87.get("import_ready", False):
        blockers.append("loop87:import_not_ready")
    if int(loop87.get("invalid_rows", 0) or 0) > 0:
        blockers.append("loop87:invalid_rows_present")
    manual_quality = loop87.get("manual_quality", {})
    if int(manual_quality.get("evidence_note_identity_or_score_only_rows", 0) or 0) > 0:
        blockers.append("loop87:identity_or_model_score_only_evidence")
    if int(manual_quality.get("evidence_note_missing_content_or_external_rows", 0) or 0) > 0:
        blockers.append("loop87:missing_content_or_external_evidence")
    return not blockers, blockers


def build_strict_adjustment_plan(
    *,
    loop87_validated_csv: Path,
    split_csv: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    rows, fieldnames = read_csv_rows(loop87_validated_csv)
    missing_columns = sorted(REQUIRED_LOOP87_COLUMNS - set(fieldnames))
    blockers: list[str] = []
    if missing_columns:
        blockers.append("loop87_validated_csv_missing_required_columns")

    split_summary, by_sample_index, by_source_sha256 = split_summary_and_index(split_csv)
    plan_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    replacement_counts: Counter[str] = Counter()
    split_action_counts: Counter[str] = Counter()
    duplicate_plan_keys = 0
    missing_split_rows = 0
    split_mismatch_rows = 0
    label_mismatch_rows = 0
    invalid_status_rows = 0
    invalid_label_rows = 0
    seen_plan_keys: set[str] = set()

    for row in rows:
        status = normalize(row.get("loop87_status"))
        status_counts[status or "blank"] += 1
        verdict_counts[normalize(row.get("manual_label_verdict")) or "blank"] += 1
        if status == "invalid" or normalize(row.get("loop87_issue_flags")):
            invalid_status_rows += 1
        if not normalize_bool(row.get("loop87_replacement_required")):
            continue

        split_row = find_split_row(row, by_sample_index, by_source_sha256)
        if split_row is None:
            missing_split_rows += 1
            split = normalize(row.get("split"))
            label_value = row.get("label")
        else:
            split = normalize(split_row.get("split"))
            label_value = split_row.get("label")
            if normalize(row.get("split")) and normalize(row.get("split")) != split:
                split_mismatch_rows += 1
            if normalize(row.get("label")) and normalize(row.get("label")) != normalize(split_row.get("label")):
                label_mismatch_rows += 1

        original_label = parse_label(label_value)
        if original_label is None:
            invalid_label_rows += 1
            original_label_text = ""
        else:
            original_label_text = str(original_label)

        sample_index = normalize(row.get("sample_index"))
        source_sha256 = normalize(row.get("source_sha256"))
        plan_key = sample_index or source_sha256 or f"row:{len(plan_rows) + 1}"
        if plan_key in seen_plan_keys:
            duplicate_plan_keys += 1
        seen_plan_keys.add(plan_key)

        reason_verdict = normalize(row.get("manual_label_verdict")) or "confirmed_bad_row"
        plan_row = {
            "source_path": row.get("source_path", ""),
            "source_sha256": source_sha256,
            "sample_index": sample_index,
            "split": split,
            "original_label": original_label_text,
            "planned_label": original_label_text,
            "plan_action": "exclude_and_replace",
            "reason": f"loop112_content_verdict_{reason_verdict}_fresh_same_original_label_redraw",
            "manual_label_verdict": row.get("manual_label_verdict", ""),
            "recommended_action": row.get("recommended_action", ""),
            "manual_verdict_note": row.get("manual_verdict_note", ""),
            "replacement_required": "true",
            "replacement_label": original_label_text,
            "usable_for_training_policy": "false",
        }
        plan_rows.append(plan_row)
        if original_label_text:
            replacement_counts[original_label_text] += 1
        split_action_counts[f"{split}:exclude_and_replace"] += 1

    if invalid_status_rows:
        blockers.append("loop87_validated_csv_contains_invalid_rows")
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

    action_counts = {"exclude_and_replace": len(plan_rows)} if plan_rows else {}
    review_rows_in_test_split = sum(1 for row in plan_rows if row.get("split") == "test")
    payload = {
        "schema": "axon_loop114_strict_redraw_adjustment_plan_v1",
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
        "review_rows_in_test_split": review_rows_in_test_split,
        "action_counts": action_counts,
        "split_action_counts": dict(sorted(split_action_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "alignment_quality": {
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
    loop112_summary_json: Path,
    output_dir: Path,
    output_json: Path,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": "axon_loop114_loop112_redraw_readiness_v1",
        "protocol": PROTOCOL,
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {"loop112_summary_json": str(loop112_summary_json)},
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
            "Upstream verdict gates must pass before Loop114 writes a redraw plan.",
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
        "# Loop114 Loop112 Redraw Readiness",
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


def run_loop112_redraw_readiness(
    *,
    loop112_summary_json: Path,
    split_csv: Path,
    output_dir: Path,
    output_json: Path,
    output_md: Optional[Path] = None,
    candidate_pool_json: Optional[Path] = None,
    corrected_split_json: Optional[Path] = None,
    replacement_audit_json: Optional[Path] = None,
    cache_ready_json: Optional[Path] = None,
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
    prefix = output_prefix if output_prefix is not None else resolved_output_dir / "loop114"
    loop76_json = resolved_output_dir / "loop114_loop76_readiness.json"
    loop76_md = resolved_output_dir / "loop114_loop76_readiness.md"
    plan_csv = resolved_output_dir / "loop114_strict_adjustment_plan.csv"
    plan_json = resolved_output_dir / "loop114_strict_adjustment_plan.json"

    loop112 = read_json(loop112_summary_json)
    ready, blockers = loop112_is_ready(loop112)
    if not ready:
        return blocked_summary(
            decision="blocked_upstream_loop112",
            blockers=blockers,
            loop112_summary_json=loop112_summary_json,
            output_dir=output_dir,
            output_json=output_json,
        )

    loop110_json = Path(loop112["outputs"]["loop110_json"])
    loop110 = read_json(loop110_json)
    ready, blockers = loop110_is_ready(loop110)
    if not ready:
        return blocked_summary(
            decision="blocked_upstream_loop110",
            blockers=blockers,
            loop112_summary_json=loop112_summary_json,
            output_dir=output_dir,
            output_json=output_json,
            extra={"loop110_summary_json": str(loop110_json)},
        )

    loop87_json = Path(loop110["outputs"]["loop87_json"])
    loop87_validated_csv = Path(loop110["outputs"]["loop87_validated_csv"])
    loop87 = read_json(loop87_json)
    ready, blockers = loop87_is_ready(loop87)
    if not ready:
        return blocked_summary(
            decision="blocked_upstream_loop87",
            blockers=blockers,
            loop112_summary_json=loop112_summary_json,
            output_dir=output_dir,
            output_json=output_json,
            extra={"loop110_summary_json": str(loop110_json), "loop87_summary_json": str(loop87_json)},
        )

    adjustment_payload, plan_rows, plan_blockers = build_strict_adjustment_plan(
        loop87_validated_csv=loop87_validated_csv,
        split_csv=split_csv,
    )
    write_csv_rows(plan_csv, plan_rows, PLAN_FIELDNAMES)
    write_json(plan_json, adjustment_payload)
    if plan_blockers:
        return blocked_summary(
            decision="blocked_before_loop76_redraw_readiness",
            blockers=plan_blockers,
            loop112_summary_json=loop112_summary_json,
            output_dir=output_dir,
            output_json=output_json,
            extra={
                "loop110_summary_json": str(loop110_json),
                "loop87_summary_json": str(loop87_json),
                "outputs": {
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
        strict_import_json=loop87_json,
        adjustment_plan_json=plan_json,
        candidate_pool_json=candidate_pool_json,
        corrected_split_json=corrected_split_json,
        replacement_audit_json=replacement_audit_json,
        cache_ready_json=cache_ready_json,
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
        "schema": "axon_loop114_loop112_redraw_readiness_v1",
        "protocol": PROTOCOL,
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {
            "loop112_summary_json": str(loop112_summary_json),
            "loop110_summary_json": str(loop110_json),
            "loop87_summary_json": str(loop87_json),
            "loop87_validated_csv": str(loop87_validated_csv),
            "split_csv": str(split_csv),
        },
        "output_dir": str(output_dir),
        "decision": loop76_payload["decision"],
        "blockers": list(loop76_payload.get("strict_failures", [])),
        "counts": {
            "loop112_actionable_rows": loop112.get("counts", {}).get("loop87_actionable_rows"),
            "loop112_replacement_required_rows": loop112.get("counts", {}).get("loop87_replacement_required_rows"),
            "loop87_rows": loop87.get("rows"),
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
            "Loop114 does not sample replacement files. It only proves whether replacement sampling is the next allowed step.",
            "label_wrong, feature_broken, and out_of_scope verdicts are all handled as exclude_and_replace, never direct relabel.",
            "Identity and model-score fields remain logistics only and are not used as verdict evidence.",
        ],
    }
    write_json(output_json, summary)
    if output_md is not None:
        write_summary_markdown(output_md, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge Loop112 output into Loop76 redraw readiness.")
    parser.add_argument("--loop112-summary-json", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--candidate-pool-json", type=Path, default=None)
    parser.add_argument("--corrected-split-json", type=Path, default=None)
    parser.add_argument("--replacement-audit-json", type=Path, default=None)
    parser.add_argument("--cache-ready-json", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--corrected-split-csv", type=Path, default=None)
    parser.add_argument("--no-enforce-label-balance", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_loop112_redraw_readiness(
        loop112_summary_json=args.loop112_summary_json,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        output_json=args.output_json,
        output_md=args.output_md,
        candidate_pool_json=args.candidate_pool_json,
        corrected_split_json=args.corrected_split_json,
        replacement_audit_json=args.replacement_audit_json,
        cache_ready_json=args.cache_ready_json,
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
