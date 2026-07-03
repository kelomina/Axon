#!/usr/bin/env python3
"""Build a read-only readiness gate for Loop72/74/75 redraw workflow.

The gate does not train, evaluate, scan raw files, extract cache, or mutate a
split. It inspects machine-readable artifacts and tells the operator which
single next step is allowed before any corrected-split experiment can proceed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOTAL = 200000
EXPECTED_SPLIT_COUNTS = {"train": 20000, "val": 20000, "test": 160000}
IDENTITY_FEATURE_POLICY = (
    "filename/path/extension/directory/source hash/cache_path/sample_index/split/row order are loading, "
    "alignment, cache-audit, duplicate-review, and manual-review fields only; they are not model evidence "
    "and must not drive thresholds, relabeling, feature engineering, or production inference"
)


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Optional[Path]) -> dict[str, Any]:
    resolved = resolve_path(path)
    if resolved is None:
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).strip().casefold() == "true"


def split_counts_ok(summary: dict[str, Any]) -> bool:
    return _int(summary.get("rows")) == EXPECTED_TOTAL and summary.get("split_counts") == EXPECTED_SPLIT_COUNTS


def summarize_import(import_payload: dict[str, Any]) -> dict[str, Any]:
    split_summary = import_payload.get("split_summary", {})
    input_alignment = import_payload.get("input_alignment", {})
    return {
        "schema": import_payload.get("schema", ""),
        "decision": import_payload.get("decision", ""),
        "import_ready": bool(import_payload.get("import_ready", False)),
        "review_rows": _int(import_payload.get("review_rows")),
        "expected_rows": import_payload.get("expected_rows"),
        "sample_index_match_count": _int(input_alignment.get("sample_index_match_count")),
        "missing_split_rows": _int(input_alignment.get("missing_split_rows")),
        "duplicate_review_rows": _int(input_alignment.get("duplicate_review_rows")),
        "invalid_rows": _int(import_payload.get("invalid_rows")),
        "training_policy_rows": _int(import_payload.get("training_policy_rows")),
        "blocking_issues": list(import_payload.get("blocking_issues", [])),
        "split_summary": split_summary,
        "split_counts_ok": split_counts_ok(split_summary),
        "target_feasibility": import_payload.get("target_feasibility", {}),
        "confirmed_bad_rows": import_payload.get("confirmed_bad_rows", {}),
        "manual_quality": import_payload.get("manual_quality", {}),
    }


def summarize_adjustment(adjustment_payload: dict[str, Any]) -> dict[str, Any]:
    split_summary = adjustment_payload.get("split_summary", {})
    return {
        "schema": adjustment_payload.get("schema", ""),
        "review_rows": _int(adjustment_payload.get("review_rows")),
        "planned_rows": _int(adjustment_payload.get("planned_rows")),
        "ignored_rows": _int(adjustment_payload.get("ignored_rows")),
        "unknown_verdict_rows": _int(adjustment_payload.get("unknown_verdict_rows")),
        "missing_split_rows": _int(adjustment_payload.get("missing_split_rows")),
        "duplicate_review_rows": _int(adjustment_payload.get("duplicate_review_rows")),
        "replacement_required": _int(adjustment_payload.get("replacement_required")),
        "replacement_counts_by_original_label": dict(adjustment_payload.get("replacement_counts_by_original_label", {})),
        "training_policy_rows": _int(adjustment_payload.get("training_policy_rows")),
        "review_rows_in_test_split": _int(adjustment_payload.get("review_rows_in_test_split")),
        "split_summary": split_summary,
        "split_counts_ok": split_counts_ok(split_summary),
    }


def summarize_candidate_pool(candidate_payload: dict[str, Any]) -> dict[str, Any]:
    if not candidate_payload:
        return {"provided": False}
    return {
        "provided": True,
        "schema": candidate_payload.get("schema", ""),
        "rows": _int(candidate_payload.get("rows")),
        "label_counts": dict(candidate_payload.get("label_counts", {})),
        "required_replacements": dict(candidate_payload.get("required_replacements", {})),
        "replacement_shortfall": dict(candidate_payload.get("replacement_shortfall", {})),
        "enough_for_required_replacements": bool(candidate_payload.get("enough_for_required_replacements", False)),
    }


def summarize_corrected_split(corrected_payload: dict[str, Any]) -> dict[str, Any]:
    if not corrected_payload:
        return {"provided": False}
    corrected_summary = corrected_payload.get("corrected_summary", {})
    replacement_summary = corrected_payload.get("replacement_summary", {})
    return {
        "provided": True,
        "schema": corrected_payload.get("schema", ""),
        "allow_test_replacements": bool(corrected_payload.get("allow_test_replacements", False)),
        "excluded_rows": _int(corrected_payload.get("excluded_rows")),
        "relabeled_rows": _int(corrected_payload.get("relabeled_rows")),
        "corrected_summary": corrected_summary,
        "corrected_split_counts_ok": split_counts_ok(corrected_summary),
        "replacement_summary": replacement_summary,
        "replacement_shortfall": dict(replacement_summary.get("shortfall", {})),
    }


def summarize_replacement_audit(replacement_payload: dict[str, Any]) -> dict[str, Any]:
    if not replacement_payload:
        return {"provided": False}
    corrected_summary = replacement_payload.get("corrected_summary", {})
    return {
        "provided": True,
        "schema": replacement_payload.get("schema", ""),
        "replacement_integrity_ok": bool(replacement_payload.get("replacement_integrity_ok", False)),
        "integrity_failures": list(replacement_payload.get("integrity_failures", [])),
        "row_count_ok": bool(replacement_payload.get("row_count_ok", False)),
        "label_balance_enforced": bool(replacement_payload.get("label_balance_enforced", False)),
        "fresh_replacement_rows": _int(replacement_payload.get("fresh_replacement_rows")),
        "replacement_requests": _int(replacement_payload.get("replacement_requests")),
        "test_replacement_requests": _int(replacement_payload.get("test_replacement_requests")),
        "corrected_summary": corrected_summary,
        "corrected_split_counts_ok": split_counts_ok(corrected_summary),
    }


def summarize_cache_ready(cache_payload: dict[str, Any]) -> dict[str, Any]:
    if not cache_payload:
        return {"provided": False}
    return {
        "provided": True,
        "schema": cache_payload.get("schema", ""),
        "cache_ready": bool(cache_payload.get("cache_ready", False)),
        "total_rows": _int(cache_payload.get("total_rows")),
        "covered_rows": _int(cache_payload.get("covered_rows")),
        "missing_rows": _int(cache_payload.get("missing_rows")),
        "coverage_ratio": float(cache_payload.get("coverage_ratio", 0.0) or 0.0),
        "shape_failures": list(cache_payload.get("shape_failures", [])),
        "missing_label_counts": dict(cache_payload.get("missing_label_counts", {})),
        "missing_split_counts": dict(cache_payload.get("missing_split_counts", {})),
        "missing_reason_counts": dict(cache_payload.get("missing_reason_counts", {})),
        "missing_cache_output": cache_payload.get("missing_cache_output"),
        "label_balance_enforced": bool(cache_payload.get("label_balance_enforced", False)),
        "label_balance_drift": list(cache_payload.get("label_balance_drift", [])),
    }


def command_for_candidate_pool(
    *,
    data_dir: Path,
    split_csv: Path,
    manifest_json: Optional[Path],
    output_prefix: Path,
    replacement_counts: dict[str, Any],
) -> str:
    manifest_part = f" --manifest-json {manifest_json}" if manifest_json is not None else ""
    return (
        ".\\vnev\\Scripts\\python.exe scripts\\build_replacement_candidate_pool.py"
        f" --data-dir {data_dir}"
        f" --split-csv {split_csv}"
        f"{manifest_part}"
        f" --required-label0 {_int(replacement_counts.get('0'))}"
        f" --required-label1 {_int(replacement_counts.get('1'))}"
        f" --output-csv {output_prefix}_candidate_pool.csv"
        f" --output-json {output_prefix}_candidate_pool.json"
    )


def command_for_corrected_split(
    *,
    split_csv: Path,
    plan_csv: Path,
    candidate_csv: Path,
    output_prefix: Path,
    allow_test_replacements: bool,
) -> str:
    allow = " --allow-test-replacements" if allow_test_replacements else ""
    return (
        ".\\vnev\\Scripts\\python.exe scripts\\build_corrected_split_from_plan.py"
        f" --split-csv {split_csv}"
        f" --plan-csv {plan_csv}"
        f" --candidate-csv {candidate_csv}"
        f" --output-csv {output_prefix}_corrected_split.csv"
        f" --output-json {output_prefix}_corrected_split.json"
        f"{allow}"
    )


def command_for_replacement_audit(
    *,
    original_split_csv: Path,
    corrected_split_csv: Path,
    plan_csv: Path,
    output_prefix: Path,
    allow_test_replacements: bool,
    enforce_label_balance: bool,
) -> str:
    allow = " --allow-test-replacements" if allow_test_replacements else ""
    balance = " --enforce-label-balance" if enforce_label_balance else ""
    return (
        ".\\vnev\\Scripts\\python.exe scripts\\audit_corrected_split_replacements.py"
        f" --original-split-csv {original_split_csv}"
        f" --corrected-split-csv {corrected_split_csv}"
        f" --plan-csv {plan_csv}"
        f" --detail-output-csv {output_prefix}_replacement_audit_detail.csv"
        f" --output-json {output_prefix}_replacement_audit.json"
        f" --strict{allow}{balance}"
    )


def command_for_cache_ready(
    *,
    corrected_split_csv: Path,
    manifest_json: Path,
    output_prefix: Path,
    enforce_label_balance: bool,
) -> str:
    balance = " --enforce-label-balance" if enforce_label_balance else ""
    return (
        ".\\vnev\\Scripts\\python.exe scripts\\audit_corrected_split_cache_ready.py"
        f" --split-csv {corrected_split_csv}"
        f" --manifest-json {manifest_json}"
        f" --missing-cache-output {output_prefix}_missing_cache.csv"
        f" --output-json {output_prefix}_cache_ready.json"
        f" --strict{balance}"
    )


def decide(payload: dict[str, Any]) -> tuple[str, list[str], str]:
    failures: list[str] = []
    imp = payload["strict_import"]
    adj = payload["adjustment_plan"]
    cand = payload["candidate_pool"]
    corrected = payload["corrected_split"]
    repl = payload["replacement_audit"]
    cache = payload["cache_ready"]

    if not imp["import_ready"]:
        failures.append("strict_import_not_ready")
    if imp["blocking_issues"]:
        failures.append("strict_import_has_blocking_issues")
    if imp["expected_rows"] is not None and imp["review_rows"] != _int(imp["expected_rows"]):
        failures.append("strict_import_review_row_count_mismatch")
    if imp["expected_rows"] is not None and imp["sample_index_match_count"] != _int(imp["expected_rows"]):
        failures.append("strict_import_sample_index_alignment_incomplete")
    if imp["missing_split_rows"] or imp["duplicate_review_rows"]:
        failures.append("strict_import_row_identity_unresolved")
    if imp["training_policy_rows"] != 0:
        failures.append("strict_import_training_policy_rows_nonzero")
    if not imp["split_counts_ok"]:
        failures.append("strict_import_split_shape_invalid")
    manual_quality = imp.get("manual_quality", {})
    if _int(manual_quality.get("actionable_verdict_missing_note_rows")):
        failures.append("strict_import_missing_actionable_evidence_notes")
    if _int(manual_quality.get("evidence_note_missing_content_or_external_rows")):
        failures.append("strict_import_evidence_note_missing_content_or_external")
    if _int(manual_quality.get("evidence_note_identity_or_score_only_rows")):
        failures.append("strict_import_evidence_note_identity_or_score_only")
    if adj["missing_split_rows"] or adj["duplicate_review_rows"] or adj["unknown_verdict_rows"]:
        failures.append("adjustment_plan_has_unresolved_rows")
    if adj["training_policy_rows"] != 0:
        failures.append("adjustment_plan_training_policy_rows_nonzero")
    if not adj["split_counts_ok"]:
        failures.append("adjustment_plan_split_shape_invalid")
    if failures:
        return "blocked_before_redraw", failures, "fix_strict_import_or_adjustment_plan"

    replacement_required = adj["replacement_required"]
    if replacement_required <= 0:
        return "await_external_verdicts", [], "no_redraw_required_until_actionable_verdicts"

    if not cand["provided"]:
        return "needs_replacement_candidate_pool", [], "build_replacement_candidate_pool"
    if not cand["enough_for_required_replacements"]:
        return "blocked_candidate_shortfall", ["replacement_candidate_shortfall"], "collect_more_valid_same_label_candidates"
    if not corrected["provided"]:
        return "needs_corrected_split", [], "build_corrected_split_from_plan"
    if not corrected["corrected_split_counts_ok"] or corrected["replacement_shortfall"]:
        return "blocked_corrected_split_invalid", ["corrected_split_invalid_or_shortfall"], "fix_corrected_split"
    if not repl["provided"]:
        return "needs_replacement_integrity_audit", [], "audit_corrected_split_replacements"
    if not repl["replacement_integrity_ok"]:
        return "blocked_replacement_integrity", ["replacement_integrity_failed"], "fix_replacement_integrity"
    if not repl["label_balance_enforced"]:
        return "blocked_replacement_integrity", ["replacement_integrity_label_balance_not_enforced"], "rerun_replacement_audit_with_label_balance"
    if not cache["provided"]:
        return "needs_cache_readiness_audit", [], "audit_corrected_split_cache_ready"
    if not cache["cache_ready"]:
        return "needs_cache_recovery", [], "recover_missing_cache_then_rerun_cache_ready"
    if not cache["label_balance_enforced"]:
        return "blocked_cache_readiness", ["cache_ready_label_balance_not_enforced"], "rerun_cache_ready_with_label_balance"
    return "ready_for_val_first_reverification", [], "restart_val_first_funnel"


def ready_for_matrix(decision: str) -> dict[str, bool]:
    return {
        "fresh_redraw": decision in {"needs_replacement_candidate_pool", "needs_corrected_split"},
        "cache_recovery": decision == "needs_cache_recovery",
        "train_val_only": decision == "ready_for_val_first_reverification",
        "test10k": False,
        "full_test": False,
    }


def build_readiness(
    *,
    strict_import_json: Path,
    adjustment_plan_json: Path,
    candidate_pool_json: Optional[Path],
    corrected_split_json: Optional[Path],
    replacement_audit_json: Optional[Path],
    cache_ready_json: Optional[Path],
    split_csv: Path,
    plan_csv: Path,
    candidate_csv: Optional[Path],
    corrected_split_csv: Optional[Path],
    manifest_json: Optional[Path],
    data_dir: Path,
    output_prefix: Path,
    enforce_label_balance: bool = True,
) -> dict[str, Any]:
    strict_import = summarize_import(read_json(strict_import_json))
    adjustment_plan = summarize_adjustment(read_json(adjustment_plan_json))
    candidate_pool = summarize_candidate_pool(read_json(candidate_pool_json))
    corrected_split = summarize_corrected_split(read_json(corrected_split_json))
    replacement_audit = summarize_replacement_audit(read_json(replacement_audit_json))
    cache_ready = summarize_cache_ready(read_json(cache_ready_json))

    payload = {
        "schema": "axon_loop76_redraw_readiness_orchestration_gate_v1",
        "protocol": "orchestrates strict Loop75 import, fresh same-original-label redraw, replacement integrity, cache readiness, and Val-first funnel only",
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "memory_leak_profile": {
            "risk": "low_read_only_metadata",
            "loads_model": False,
            "uses_cuda": False,
            "reads_npz_feature_arrays": False,
            "scans_raw_data": False,
            "requires_pre_run_resource_guard": True,
        },
        "inputs": {
            "strict_import_json": str(resolve_path(strict_import_json)),
            "adjustment_plan_json": str(resolve_path(adjustment_plan_json)),
            "candidate_pool_json": str(resolve_path(candidate_pool_json)) if candidate_pool_json else None,
            "corrected_split_json": str(resolve_path(corrected_split_json)) if corrected_split_json else None,
            "replacement_audit_json": str(resolve_path(replacement_audit_json)) if replacement_audit_json else None,
            "cache_ready_json": str(resolve_path(cache_ready_json)) if cache_ready_json else None,
        },
        "strict_import": strict_import,
        "loop75_import": strict_import,
        "adjustment_plan": adjustment_plan,
        "redraw_requirements": {
            "replacement_required": adjustment_plan["replacement_required"],
            "replacement_counts_by_original_label": adjustment_plan["replacement_counts_by_original_label"],
            "redraw_rule": "fresh_same_original_label_only",
            "test_rows_policy": "data_hygiene_only_not_training_policy",
            "corrected_label_policy": "target_feasibility_evidence_only",
        },
        "candidate_pool": candidate_pool,
        "corrected_split": corrected_split,
        "replacement_audit": replacement_audit,
        "corrected_split_integrity": replacement_audit,
        "cache_ready": cache_ready,
        "cache_readiness": cache_ready,
        "val_first_policy": {
            "threshold_selection_allowed_sources": ["train_oof", "val"],
            "forbid_test10k_threshold_selection": True,
            "forbid_full_test_threshold_selection": True,
            "test10k_allowed_only_after_val_gate": True,
            "full_test_allowed_only_after_frozen_test10k_pass": True,
        },
        "forbidden_uses": [
            "Do not use Loop72/74/75/76 full-test review artifacts for threshold selection.",
            "Do not train on held-out test verdicts.",
            "Do not treat filename/path/hash/sample_index/split/review rank/model score as label evidence.",
            "Do not claim F1>=0.999 until a Val-selected candidate passes Test-10k and full-test evaluation.",
        ],
    }
    decision, failures, next_step = decide(payload)
    payload["decision"] = decision
    payload["strict_failures"] = failures
    payload["blocked_reasons"] = failures
    payload["next_step"] = next_step
    payload["ready_for"] = ready_for_matrix(decision)
    payload["forbidden_for_training_or_threshold_selection"] = payload["forbidden_uses"]

    replacement_counts = adjustment_plan["replacement_counts_by_original_label"]
    allow_test_replacements = adjustment_plan["review_rows_in_test_split"] > 0 and adjustment_plan["replacement_required"] > 0
    candidate_csv_for_command = candidate_csv if candidate_csv is not None else Path(f"{output_prefix}_candidate_pool.csv")
    corrected_split_csv_for_command = corrected_split_csv if corrected_split_csv is not None else Path(f"{output_prefix}_corrected_split.csv")
    commands = {
        "build_replacement_candidate_pool": command_for_candidate_pool(
            data_dir=data_dir,
            split_csv=split_csv,
            manifest_json=manifest_json,
            output_prefix=output_prefix,
            replacement_counts=replacement_counts,
        ),
        "build_corrected_split": command_for_corrected_split(
            split_csv=split_csv,
            plan_csv=plan_csv,
            candidate_csv=candidate_csv_for_command,
            output_prefix=output_prefix,
            allow_test_replacements=allow_test_replacements,
        ),
        "audit_replacements": command_for_replacement_audit(
            original_split_csv=split_csv,
            corrected_split_csv=corrected_split_csv_for_command,
            plan_csv=plan_csv,
            output_prefix=output_prefix,
            allow_test_replacements=allow_test_replacements,
            enforce_label_balance=enforce_label_balance,
        ),
    }
    if manifest_json is not None:
        commands["audit_cache_ready"] = command_for_cache_ready(
            corrected_split_csv=corrected_split_csv_for_command,
            manifest_json=manifest_json,
            output_prefix=output_prefix,
            enforce_label_balance=enforce_label_balance,
        )
    else:
        commands["audit_cache_ready"] = "manifest_json_required_before_cache_readiness_audit"
    payload["commands"] = commands
    payload["notes"] = [
        "This gate is intentionally read-only. Run only the command named by next_step.",
        "Every downstream command still needs its own resource/static leak guard before execution.",
        "If replacement_required=0, there is no data-cleaning action to perform; wait for external verdicts.",
    ]
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Loop76 Redraw Readiness",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Next step: `{payload['next_step']}`",
        f"- Strict failures: `{payload['strict_failures']}`",
        f"- Replacement required: `{payload['adjustment_plan']['replacement_required']}`",
        f"- Training policy rows: `{payload['adjustment_plan']['training_policy_rows']}`",
        "",
        "## Allowed Next Command",
        "",
        "```powershell",
        payload["commands"].get(payload["next_step"], ""),
        "```",
        "",
        "## Forbidden Uses",
        "",
    ]
    for item in payload["forbidden_uses"]:
        lines.append(f"- {item}")
    lines.append("")
    resolved.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop76 redraw readiness gate.")
    parser.add_argument("--strict-import-json", type=Path, required=True)
    parser.add_argument("--adjustment-plan-json", type=Path, required=True)
    parser.add_argument("--candidate-pool-json", type=Path, default=None)
    parser.add_argument("--corrected-split-json", type=Path, default=None)
    parser.add_argument("--replacement-audit-json", type=Path, default=None)
    parser.add_argument("--cache-ready-json", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--corrected-split-csv", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--no-enforce-label-balance",
        action="store_true",
        help="Relax final replacement/cache balance checks. Not recommended for strict 20w protocol.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_readiness(
        strict_import_json=args.strict_import_json,
        adjustment_plan_json=args.adjustment_plan_json,
        candidate_pool_json=args.candidate_pool_json,
        corrected_split_json=args.corrected_split_json,
        replacement_audit_json=args.replacement_audit_json,
        cache_ready_json=args.cache_ready_json,
        split_csv=args.split_csv,
        plan_csv=args.plan_csv,
        candidate_csv=args.candidate_csv,
        corrected_split_csv=args.corrected_split_csv,
        manifest_json=args.manifest_json,
        data_dir=args.data_dir,
        output_prefix=args.output_prefix,
        enforce_label_balance=not bool(args.no_enforce_label_balance),
    )
    write_json(args.output_json, payload)
    if args.output_md is not None:
        write_markdown(args.output_md, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not payload["strict_failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
