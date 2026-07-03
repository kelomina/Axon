#!/usr/bin/env python3
"""Preflight check for ML authorization packages without running heavy work."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence


EXPECTED_TOTAL_ROWS = 200000
EXPECTED_SPLIT_COUNTS = {
    "train": 20000,
    "val": 20000,
    "test": 160000,
}
EXPECTED_LABEL_SPLIT_COUNTS = {
    "train": {"0": 10000, "1": 10000},
    "val": {"0": 10000, "1": 10000},
    "test": {"0": 80000, "1": 80000},
}
IDENTITY_FIELDS = [
    "filename",
    "path",
    "extension",
    "directory",
    "hash",
    "source_sha256",
    "sample_index",
    "split",
    "row_order",
    "model_score",
]
ALLOWED_IDENTITY_USES = [
    "loading",
    "alignment",
    "cache audit",
    "duplicate detection",
    "manual/external review indexing",
]
DEFAULT_ROUTE_AUDIT = Path("reports/random_20w_split/loop101_identity_safe_route_audit.json")
DEFAULT_CURRENT_STATE_GATE = Path("reports/random_20w_split/loop101_current_state_gate_metadata.json")
DEFAULT_CACHE_READY = Path("reports/random_20w_split/loop100_cache_ready_metadata.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _int(value: object, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _optional_json_status(path: Optional[Path], root: Path) -> tuple[Optional[dict], dict]:
    if path is None:
        return None, {"path": None, "exists": False, "loaded": False}
    resolved = _resolve(root, path)
    status = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "loaded": False,
    }
    if not resolved.exists():
        return None, status
    payload = load_json(resolved)
    status["loaded"] = True
    status["schema"] = payload.get("schema")
    return payload, status


def _csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _path_status(path_text: str, root: Path) -> dict:
    path = Path(path_text)
    full_path = path if path.is_absolute() else root / path
    status = {
        "path": path_text,
        "exists": full_path.exists(),
        "type": "missing",
    }
    if full_path.exists():
        status["type"] = "dir" if full_path.is_dir() else "file"
        if full_path.is_file() and full_path.suffix.lower() == ".csv":
            status["rows"] = _csv_count(full_path)
    return status


def _check_package(package: dict, root: Path) -> dict:
    if package.get("completed"):
        return {
            "id": package["id"],
            "recommendation_id": package.get("recommendation_id"),
            "heavy_authorization_required": bool(package.get("heavy_authorization_required")),
            "completed": True,
            "ready_for_authorization": False,
            "missing_inputs": [],
            "acceptance_criteria": package.get("acceptance_criteria", []),
        }
    inputs = package.get("bounded_recovery_inputs", {})
    input_status = {
        name: _path_status(path, root)
        for name, path in inputs.items()
    }
    missing = [
        name
        for name, status in input_status.items()
        if not status["exists"]
    ]
    return {
        "id": package["id"],
        "recommendation_id": package.get("recommendation_id"),
        "heavy_authorization_required": bool(package.get("heavy_authorization_required")),
        "bounded_recovery_inputs": input_status,
        "ready_for_authorization": not missing,
        "missing_inputs": missing,
        "acceptance_criteria": package.get("acceptance_criteria", []),
    }


def _check_cache_ready(cache_ready: Optional[dict]) -> dict:
    blockers: list[str] = []
    evidence: dict[str, Any] = {}
    if not cache_ready:
        blockers.append("cache_ready_report_missing")
        return {"status": "missing", "blockers": blockers, "evidence": evidence}

    split_summary = cache_ready.get("split_summary", {}) if isinstance(cache_ready, dict) else {}
    evidence = {
        "cache_ready": cache_ready.get("cache_ready"),
        "total_rows": cache_ready.get("total_rows"),
        "covered_rows": cache_ready.get("covered_rows"),
        "missing_rows": cache_ready.get("missing_rows"),
        "label_balance_enforced": cache_ready.get("label_balance_enforced"),
        "cache_metadata_validation_enabled": cache_ready.get("cache_metadata_validation_enabled"),
        "metadata_checked_rows": cache_ready.get("metadata_checked_rows"),
        "metadata_failure_rows": cache_ready.get("metadata_failure_rows"),
        "split_counts": split_summary.get("split_counts"),
        "label_split_counts": split_summary.get("label_split_counts"),
    }
    if cache_ready.get("cache_ready") is not True:
        blockers.append("cache_ready_not_true")
    if _int(cache_ready.get("total_rows")) != EXPECTED_TOTAL_ROWS:
        blockers.append("cache_ready_total_rows_not_200000")
    if _int(cache_ready.get("covered_rows")) != EXPECTED_TOTAL_ROWS:
        blockers.append("cache_ready_covered_rows_not_200000")
    if _int(cache_ready.get("missing_rows")) != 0:
        blockers.append("cache_ready_missing_rows_present")
    if cache_ready.get("label_balance_enforced") is not True:
        blockers.append("cache_ready_label_balance_not_enforced")
    if cache_ready.get("cache_metadata_validation_enabled") is not True:
        blockers.append("cache_ready_metadata_validation_not_enabled")
    if _int(cache_ready.get("metadata_checked_rows")) != EXPECTED_TOTAL_ROWS:
        blockers.append("cache_ready_metadata_not_fully_checked")
    if _int(cache_ready.get("metadata_failure_rows")) != 0:
        blockers.append("cache_ready_metadata_failures_present")
    if split_summary.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        blockers.append("cache_ready_split_counts_not_1_1_8_20w")
    if split_summary.get("label_split_counts") != EXPECTED_LABEL_SPLIT_COUNTS:
        blockers.append("cache_ready_label_split_counts_not_balanced")
    return {
        "status": "pass" if not blockers else "block",
        "blockers": blockers,
        "evidence": evidence,
    }


def _check_current_state_gate(current_state: Optional[dict]) -> dict:
    blockers: list[str] = []
    evidence: dict[str, Any] = {}
    if not current_state:
        blockers.append("current_state_gate_missing")
        return {"status": "missing", "blockers": blockers, "evidence": evidence}

    cache_section = _get(current_state, "sections", "current_split_cache", default={}) or {}
    replacement_section = _get(current_state, "sections", "fixed_v2_replacement_130", default={}) or {}
    evidence = {
        "decision": current_state.get("decision"),
        "replacement_rows": replacement_section.get("replacement_rows"),
        "self_replacements": replacement_section.get("self_replacements"),
        "selection_status_counts": replacement_section.get("selection_status_counts"),
        "total_rows": cache_section.get("total_rows"),
        "covered_rows": cache_section.get("covered_rows"),
        "missing_rows": cache_section.get("missing_rows"),
        "label_balance_enforced": cache_section.get("label_balance_enforced"),
        "cache_metadata_validation_enabled": cache_section.get("cache_metadata_validation_enabled"),
        "metadata_checked_rows": cache_section.get("metadata_checked_rows"),
        "metadata_failure_rows": cache_section.get("metadata_failure_rows"),
        "sampled_rows": cache_section.get("sampled_rows"),
        "sample_failed_rows": cache_section.get("sample_failed_rows"),
    }
    if current_state.get("decision") != "pass":
        blockers.append("current_state_gate_not_pass")
    if _int(replacement_section.get("replacement_rows")) != 130:
        blockers.append("current_state_replacement_130_not_proven")
    if _int(replacement_section.get("self_replacements")) != 0:
        blockers.append("current_state_self_replacements_detected")
    if _int(cache_section.get("total_rows")) != EXPECTED_TOTAL_ROWS:
        blockers.append("current_state_total_rows_not_200000")
    if _int(cache_section.get("covered_rows")) != EXPECTED_TOTAL_ROWS:
        blockers.append("current_state_cache_not_fully_covered")
    if _int(cache_section.get("missing_rows")) != 0:
        blockers.append("current_state_cache_missing_rows")
    if cache_section.get("label_balance_enforced") is not True:
        blockers.append("current_state_label_balance_not_enforced")
    if cache_section.get("cache_metadata_validation_enabled") is not True:
        blockers.append("current_state_metadata_validation_not_enabled")
    if _int(cache_section.get("metadata_checked_rows")) != EXPECTED_TOTAL_ROWS:
        blockers.append("current_state_metadata_not_fully_checked")
    if _int(cache_section.get("metadata_failure_rows")) != 0:
        blockers.append("current_state_metadata_failures_present")
    if _int(cache_section.get("sample_failed_rows"), 0) != 0:
        blockers.append("current_state_sample_integrity_failures_present")
    return {
        "status": "pass" if not blockers else "block",
        "blockers": blockers,
        "evidence": evidence,
    }


def _check_route_audit(route_audit: Optional[dict]) -> dict:
    blockers: list[str] = []
    evidence: dict[str, Any] = {}
    if not route_audit:
        blockers.append("route_audit_missing")
        return {"status": "missing", "blockers": blockers, "evidence": evidence}

    decisions = route_audit.get("decisions", {}) if isinstance(route_audit.get("decisions"), dict) else {}
    full_queue = _get(route_audit, "route_sections", "full_queue_review", "evidence", default={}) or {}
    identity_policy = route_audit.get("identity_feature_policy", {})
    forbidden = set(identity_policy.get("forbidden_as_model_or_verdict_evidence", []))
    allowed = set(identity_policy.get("allowed_uses", []))
    missing_forbidden = [field for field in IDENTITY_FIELDS if field not in forbidden]
    missing_allowed = [use for use in ALLOWED_IDENTITY_USES if use not in allowed]

    evidence = {
        "decision": route_audit.get("decision"),
        "blockers": route_audit.get("blockers", []),
        "actionable_rows": full_queue.get("actionable_rows"),
        "replacement_required_rows": full_queue.get("replacement_required_rows"),
        "training_policy_rows": full_queue.get("training_policy_rows"),
        "training_allowed_now": decisions.get("training_allowed_now"),
        "test10k_allowed_now": decisions.get("test10k_allowed_now"),
        "full_test_allowed_now": decisions.get("full_test_allowed_now"),
        "ready_for_redraw_preflight": decisions.get("ready_for_redraw_preflight"),
        "next_allowed_step": decisions.get("next_allowed_step"),
        "missing_forbidden_identity_fields": missing_forbidden,
        "missing_allowed_identity_uses": missing_allowed,
    }
    if route_audit.get("decision") == "await_independent_blinded_verdicts":
        blockers.append("route_audit_awaits_independent_blinded_verdicts")
    if route_audit.get("decision") not in {
        "await_independent_blinded_verdicts",
        "ready_for_non_destructive_redraw_preflight",
    }:
        blockers.append("route_audit_decision_unrecognized")
    if route_audit.get("blockers"):
        blockers.append("route_audit_has_blockers")
    if _int(full_queue.get("actionable_rows"), 0) == 0:
        blockers.append("no_actionable_independent_verdicts")
    if _int(full_queue.get("training_policy_rows"), 0) != 0:
        blockers.append("route_audit_training_policy_rows_present")
    if missing_forbidden:
        blockers.append("identity_policy_missing_required_forbidden_fields")
    if missing_allowed:
        blockers.append("identity_policy_missing_required_allowed_uses")
    return {
        "status": "pass" if not blockers else "block",
        "blockers": blockers,
        "evidence": evidence,
    }


def _build_operation_authorization(
    *,
    route_audit: Optional[dict],
    route_gate: dict,
    current_state_gate: dict,
    cache_ready_gate: dict,
) -> dict:
    route_decision = route_audit.get("decision") if route_audit else None
    route_decisions = route_audit.get("decisions", {}) if route_audit else {}
    shared_blockers = (
        cache_ready_gate["blockers"]
        + current_state_gate["blockers"]
        + route_gate["blockers"]
    )

    operation_required_flags = {
        "train_val": "training_allowed_now",
        "threshold_sweep": "training_allowed_now",
        "test10k": "test10k_allowed_now",
        "full_test": "full_test_allowed_now",
    }
    operation_blockers: dict[str, list[str]] = {}
    decisions: dict[str, bool] = {
        "read_only_review_allowed": True,
        "package_completion_grants_operations": False,
    }
    for operation, flag in operation_required_flags.items():
        blockers = list(shared_blockers)
        if route_decision == "ready_for_non_destructive_redraw_preflight":
            blockers.append("route_audit_only_allows_redraw_preflight")
        if route_decisions.get(flag) is not True:
            blockers.append(f"route_audit_{flag}_false")
        operation_blockers[operation] = sorted(set(blockers))
        decisions[f"{operation}_allowed"] = not blockers

    redraw_blockers = list(shared_blockers)
    if route_decisions.get("ready_for_redraw_preflight") is not True:
        redraw_blockers.append("route_audit_ready_for_redraw_preflight_false")
    operation_blockers["redraw_preflight"] = sorted(set(redraw_blockers))
    decisions["redraw_preflight_allowed"] = not redraw_blockers
    decisions["cache_mutation_allowed"] = False

    return {
        "source_of_truth": {
            "route_audit": "Loop98/101 identity-safe route audit decisions",
            "current_state_gate": "Loop79/101 fixed-v2 current-state metadata gate",
            "cache_ready": "Loop100 full cache metadata readiness",
            "completed_authorization_packages": "records only; not heavy-operation authorization",
        },
        "decisions": decisions,
        "shared_blockers": sorted(set(shared_blockers)),
        "operation_blockers": operation_blockers,
        "identity_feature_policy": {
            "forbidden_as_model_or_verdict_evidence": IDENTITY_FIELDS,
            "allowed_uses": ALLOWED_IDENTITY_USES,
            "strict_boundary": (
                "Filename, path, extension, directory, hash, source_sha256, sample_index, split, "
                "row_order, and model_score are logistics metadata only. They must not drive training, "
                "feature masks, threshold tuning, fusion, relabeling, replacement sampling, or production inference."
            ),
            "label_boundary": (
                "If original labels were bootstrapped from curated directories, that step ends at the locked "
                "manifest/split label. Redraws use the locked manifest label pool, never fresh name/path inference."
            ),
        },
        "next_allowed_step": (
            _get(route_audit or {}, "decisions", "next_allowed_step")
            or "Only read-only review/preflight is allowed until route evidence is complete."
        ),
    }


def build_preflight(
    root: Path,
    authorization_plan: Path,
    status_path: Path,
    *,
    route_audit_path: Optional[Path] = None,
    current_state_gate_path: Optional[Path] = None,
    cache_ready_path: Optional[Path] = None,
) -> dict:
    authorization_plan = _resolve(root, authorization_plan)
    status_path = _resolve(root, status_path)
    plan = load_json(authorization_plan)
    status = load_json(status_path)
    route_audit, route_audit_status = _optional_json_status(route_audit_path, root)
    current_state, current_state_status = _optional_json_status(current_state_gate_path, root)
    cache_ready, cache_ready_status = _optional_json_status(cache_ready_path, root)
    route_gate = _check_route_audit(route_audit)
    current_state_gate = _check_current_state_gate(current_state)
    cache_ready_gate = _check_cache_ready(cache_ready)
    packages = plan.get("authorization_packages", [])
    package_checks = [_check_package(package, root) for package in packages]
    active_checks = [check for check in package_checks if not check.get("completed")]
    package_scope_audit = {
        "included_package_ids": [check.get("id") for check in package_checks],
        "ignored_package_ids": [],
        "completed_records": [
            check.get("id")
            for check in package_checks
            if check.get("completed")
        ],
        "open_heavy_packages": [
            check.get("id")
            for check in package_checks
            if not check.get("completed") and check.get("heavy_authorization_required")
        ],
        "open_light_or_review_packages": [
            check.get("id")
            for check in package_checks
            if not check.get("completed") and not check.get("heavy_authorization_required")
        ],
        "completed_records_do_not_authorize_operations": True,
    }
    operation_authorization = _build_operation_authorization(
        route_audit=route_audit,
        route_gate=route_gate,
        current_state_gate=current_state_gate,
        cache_ready_gate=cache_ready_gate,
    )
    allowed_operations = [
        name.removesuffix("_allowed")
        for name, allowed in operation_authorization["decisions"].items()
        if name.endswith("_allowed") and allowed is True and name not in {"read_only_review_allowed"}
    ]
    ml_gate_result = {
        "passed": bool(allowed_operations),
        "allowed_operations": allowed_operations,
        "blocking_reasons": operation_authorization["shared_blockers"],
        "required_for_commands": ["train", "eval", "test10k", "full_test"],
        "read_only_review_allowed": True,
    }
    return {
        "schema": "axon_ml_authorization_preflight_v2",
        "authorization_plan": str(authorization_plan),
        "status": str(status_path),
        "status_summary": status.get("summary", {}),
        "evidence_inputs": {
            "route_audit": route_audit_status,
            "current_state_gate": current_state_status,
            "cache_ready": cache_ready_status,
        },
        "package_checks": package_checks,
        "package_scope_audit": package_scope_audit,
        "all_bounded_inputs_present": all(check["ready_for_authorization"] for check in active_checks),
        "ml_gate_result": ml_gate_result,
        "route_gate": route_gate,
        "current_state_gate": current_state_gate,
        "cache_ready_gate": cache_ready_gate,
        "operation_authorization": operation_authorization,
        "guardrails": [
            "This preflight does not rebuild cache, train models, evaluate checkpoints, or delete files.",
            "Completed A/B/C package records do not authorize training, threshold sweeps, Test-10k, or full-test.",
            "Heavy operations require this preflight to allow the exact operation, not just user intent.",
            "Do not use scripts/rebuild_cache_64.py or scripts/rebuild_cache_8192.py for current fixed-v2 cache recovery.",
            "Filename/path/directory/extension/hash/source_sha256/sample_index/split/row order/model score fields are not malware evidence.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ML authorization preflight JSON.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--authorization-plan", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, default=DEFAULT_ROUTE_AUDIT)
    parser.add_argument("--current-state-gate", type=Path, default=DEFAULT_CURRENT_STATE_GATE)
    parser.add_argument("--cache-ready", type=Path, default=DEFAULT_CACHE_READY)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    preflight = build_preflight(
        root=root,
        authorization_plan=args.authorization_plan,
        status_path=args.status_json,
        route_audit_path=args.route_audit,
        current_state_gate_path=args.current_state_gate,
        cache_ready_path=args.cache_ready,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
