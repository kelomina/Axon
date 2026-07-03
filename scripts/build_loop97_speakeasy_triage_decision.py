#!/usr/bin/env python3
"""Summarize Speakeasy-X evidence into an automatic-merge decision gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence


DEFAULT_RULE = "timeout_filter_score_lt_0.95"
PROTOCOL = (
    "read-only Speakeasy-X triage decision summary; no emulation, no model fitting, no threshold selection, "
    "no split/cache mutation"
)
IDENTITY_POLICY = (
    "Speakeasy dynamic behavior evidence is content/runtime evidence. File names, paths, hashes, and row ids are "
    "not verdict evidence and are not model features."
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric(summary: dict[str, Any], rule_name: str) -> dict[str, Any]:
    rules = summary.get("rule_comparison") or {}
    if rule_name not in rules:
        raise KeyError(f"Rule {rule_name!r} not found in summary")
    return dict(rules[rule_name])


def _baseline(summary: dict[str, Any]) -> dict[str, Any]:
    rules = summary.get("rule_comparison") or {}
    if "calibrator_fixed_threshold" not in rules:
        raise KeyError("calibrator_fixed_threshold not found in summary")
    return dict(rules["calibrator_fixed_threshold"])


def _role(summary: dict[str, Any], role_name: str) -> dict[str, Any]:
    return dict((summary.get("by_role") or {}).get(role_name) or {})


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _delta(rule: dict[str, Any], field: str) -> int:
    return _int((rule.get("delta_vs_calibrator") or {}).get(field))


def build_decision(
    *,
    val_summary_json: Path,
    test_confirmation_json: Path,
    random_val_summary_json: Path,
    output_json: Path,
    rule_name: str = DEFAULT_RULE,
    max_allowed_new_fn_confirmation: int = 0,
    max_allowed_new_fn_rate_confirmation: float = 0.01,
) -> dict[str, Any]:
    val_summary = read_json(val_summary_json)
    test_summary = read_json(test_confirmation_json)
    random_val_summary = read_json(random_val_summary_json)
    val_rule = _metric(val_summary, rule_name)
    test_rule = _metric(test_summary, rule_name)
    val_baseline = _baseline(val_summary)
    test_baseline = _baseline(test_summary)

    test_samples = _int((test_summary.get("sample_counts") or {}).get("total"))
    new_fn_confirmation = _int(test_rule.get("new_fn_from_baseline_tp"))
    new_fn_rate_confirmation = new_fn_confirmation / test_samples if test_samples else 0.0
    fixed_fp_confirmation = _int(test_rule.get("fixed_baseline_fp"))
    fp_delta_confirmation = _delta(test_rule, "false_positive")
    fn_delta_confirmation = _delta(test_rule, "false_negative")
    error_delta_confirmation = _delta(test_rule, "errors")

    blockers: list[str] = []
    warnings: list[str] = []
    if new_fn_confirmation > max_allowed_new_fn_confirmation:
        blockers.append("confirmation_new_fn_exceeds_zero_tolerance")
    if new_fn_rate_confirmation > max_allowed_new_fn_rate_confirmation:
        blockers.append("confirmation_new_fn_rate_too_high")
    if fn_delta_confirmation > 0:
        blockers.append("confirmation_fn_delta_positive")
    if fixed_fp_confirmation <= 0:
        warnings.append("confirmation_did_not_fix_fp")
    if error_delta_confirmation >= 0:
        warnings.append("confirmation_errors_not_reduced")

    test_roles = {
        "calibrator_FP": _role(test_summary, "calibrator_FP"),
        "matched_correct_malicious_for_FN": _role(test_summary, "matched_correct_malicious_for_FN"),
        "ordinary_malicious": _role(test_summary, "ordinary_malicious"),
        "rule_risk_correct_malicious": _role(test_summary, "rule_risk_correct_malicious"),
    }
    malicious_timeout_counts = {
        role: _int(payload.get("timeouts"))
        for role, payload in test_roles.items()
        if role != "calibrator_FP"
    }
    if sum(malicious_timeout_counts.values()) > 0:
        blockers.append("timeout_signal_also_hits_true_malicious_rows")

    random_val_comparison = dict((random_val_summary.get("model_comparison") or {}))
    random_val_calibrator = random_val_comparison.get("existing_probability_calibrator_fixed_threshold_on_val_subset")
    if random_val_calibrator and _int(random_val_calibrator.get("errors")) == 0:
        warnings.append("random_val_calibrator_already_perfect_on_small_probe")

    decision = {
        "schema": "axon_loop97_speakeasy_triage_decision_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "rule_name": rule_name,
        "inputs": {
            "val_summary_json": str(val_summary_json),
            "test_confirmation_json": str(test_confirmation_json),
            "random_val_summary_json": str(random_val_summary_json),
        },
        "limits": {
            "max_allowed_new_fn_confirmation": max_allowed_new_fn_confirmation,
            "max_allowed_new_fn_rate_confirmation": max_allowed_new_fn_rate_confirmation,
        },
        "val_evidence": {
            "sample_count": _int((val_summary.get("sample_counts") or {}).get("total")),
            "baseline": {
                "f1": _float(val_baseline.get("f1")),
                "errors": _int(val_baseline.get("errors")),
                "false_positive": _int(val_baseline.get("false_positive")),
                "false_negative": _int(val_baseline.get("false_negative")),
            },
            "rule": {
                "f1": _float(val_rule.get("f1")),
                "errors": _int(val_rule.get("errors")),
                "false_positive": _int(val_rule.get("false_positive")),
                "false_negative": _int(val_rule.get("false_negative")),
                "error_delta": _delta(val_rule, "errors"),
                "fp_delta": _delta(val_rule, "false_positive"),
                "fn_delta": _delta(val_rule, "false_negative"),
                "new_fn_from_baseline_tp": _int(val_rule.get("new_fn_from_baseline_tp")),
                "fixed_baseline_fp": _int(val_rule.get("fixed_baseline_fp")),
            },
        },
        "confirmation_evidence": {
            "sample_count": test_samples,
            "baseline": {
                "f1": _float(test_baseline.get("f1")),
                "errors": _int(test_baseline.get("errors")),
                "false_positive": _int(test_baseline.get("false_positive")),
                "false_negative": _int(test_baseline.get("false_negative")),
            },
            "rule": {
                "f1": _float(test_rule.get("f1")),
                "errors": _int(test_rule.get("errors")),
                "false_positive": _int(test_rule.get("false_positive")),
                "false_negative": _int(test_rule.get("false_negative")),
                "error_delta": error_delta_confirmation,
                "fp_delta": fp_delta_confirmation,
                "fn_delta": fn_delta_confirmation,
                "new_fn_from_baseline_tp": new_fn_confirmation,
                "new_fn_rate": new_fn_rate_confirmation,
                "fixed_baseline_fp": fixed_fp_confirmation,
            },
            "malicious_timeout_counts": malicious_timeout_counts,
        },
        "random_val_sanity": {
            "sample_count": _int((random_val_summary.get("sample_counts") or {}).get("total")),
            "calibrator_errors": _int(random_val_calibrator.get("errors")) if random_val_calibrator else None,
            "calibrator_f1": _float(random_val_calibrator.get("f1")) if random_val_calibrator else None,
        },
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "decisions": {
            "automatic_classifier_merge_allowed": False,
            "automatic_threshold_override_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "manual_external_review_signal_allowed": True,
            "recommended_use": (
                "Use Speakeasy timeout/dynamic behavior only as manual or external-review context for likely FP "
                "triage. Do not automatically downgrade malicious predictions because confirmation added FNs."
            ),
        },
        "notes": [
            "The confirmation subset is intentionally enriched and is not a full-test F1 estimate.",
            "The rule was selected before this summary; this script does not tune thresholds.",
            "New false negatives are treated as a blocker for automatic malware-detector merge.",
        ],
    }
    write_json(output_json, decision)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop97 Speakeasy triage decision summary.")
    parser.add_argument("--val-summary-json", type=Path, required=True)
    parser.add_argument("--test-confirmation-json", type=Path, required=True)
    parser.add_argument("--random-val-summary-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rule-name", default=DEFAULT_RULE)
    parser.add_argument("--max-allowed-new-fn-confirmation", type=int, default=0)
    parser.add_argument("--max-allowed-new-fn-rate-confirmation", type=float, default=0.01)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    decision = build_decision(
        val_summary_json=args.val_summary_json,
        test_confirmation_json=args.test_confirmation_json,
        random_val_summary_json=args.random_val_summary_json,
        output_json=args.output_json,
        rule_name=args.rule_name,
        max_allowed_new_fn_confirmation=args.max_allowed_new_fn_confirmation,
        max_allowed_new_fn_rate_confirmation=args.max_allowed_new_fn_rate_confirmation,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
