#!/usr/bin/env python3
"""Build a Test-10k promotion-margin guard for near-best candidates.

This guard turns the loose phrase "Test-10k confirmed" into an explicit rule:
after a Val win, the candidate must also improve Test-10k by a meaningful error
margin before it deserves a full-test run. The guard is read-only and never
trains, tunes, relabels, samples replacements, or mutates split/cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_BEST = {
    "val": {"f1": 0.9919193934557063, "errors": 162, "fp": 105, "fn": 57},
    "test10k": {"f1": 0.9921921921921922, "errors": 78, "fp": 49, "fn": 29},
}
IDENTITY_FEATURE_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/split are alignment and audit fields only; "
    "Loop161 only reads aggregate metrics and never uses identity fields, model probabilities, row order, "
    "or full-test observations for model, threshold, feature-mask, or replacement selection"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def metric_summary(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "f1": float(metric["f1"]),
        "errors": int(metric["errors"]),
        "fp": int(metric.get("false_positive", metric.get("fp", 0))),
        "fn": int(metric.get("false_negative", metric.get("fn", 0))),
    }


def delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "f1": float(candidate["f1"]) - float(reference["f1"]),
        "errors": int(candidate["errors"]) - int(reference["errors"]),
        "fp": int(candidate["fp"]) - int(reference["fp"]),
        "fn": int(candidate["fn"]) - int(reference["fn"]),
    }


def from_authenticode_eval(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(path)
    return metric_summary(payload["baseline"]), metric_summary(payload["candidate"])


def from_loop160(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = read_json(path)
    val = payload["evaluations"]["val"]
    test10k = payload["evaluations"]["test10k"]
    return (
        metric_summary(val["baseline"]),
        metric_summary(val["candidate"]),
        metric_summary(test10k["baseline"]),
        metric_summary(test10k["candidate"]),
    )


def from_loop159(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = read_json(path)
    return (
        dict(payload["current_best_reference"]["val"]),
        dict(payload["metrics"]["val"]),
        dict(payload["current_best_reference"]["test10k"]),
        dict(payload["metrics"]["test10k"]),
    )


def normalize_metric_names(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "f1": float(metric["f1"]),
        "errors": int(metric["errors"]),
        "fp": int(metric.get("fp", metric.get("false_positive", 0))),
        "fn": int(metric.get("fn", metric.get("false_negative", 0))),
    }


def candidate_record(
    *,
    candidate_id: str,
    source: str,
    val_reference: dict[str, Any],
    val_candidate: dict[str, Any],
    test10k_reference: dict[str, Any],
    test10k_candidate: dict[str, Any],
    min_val_error_improvement: int,
    min_test10k_error_improvement: int,
    full_test_json: Optional[Path] = None,
) -> dict[str, Any]:
    val_reference = normalize_metric_names(val_reference)
    val_candidate = normalize_metric_names(val_candidate)
    test10k_reference = normalize_metric_names(test10k_reference)
    test10k_candidate = normalize_metric_names(test10k_candidate)
    val_delta = delta(val_candidate, val_reference)
    test_delta = delta(test10k_candidate, test10k_reference)
    val_pass = val_delta["errors"] <= -int(min_val_error_improvement)
    test10k_margin_pass = test_delta["errors"] <= -int(min_test10k_error_improvement)
    if not val_pass:
        decision = "reject_val_margin"
    elif not test10k_margin_pass:
        decision = "reject_test10k_margin_too_small"
    else:
        decision = "allow_full_test_confirmation"

    full_test = None
    if full_test_json is not None:
        full_payload = read_json(full_test_json)
        if "metrics" in full_payload and "full_test" in full_payload["metrics"]:
            full_test = full_payload["metrics"]["full_test"]
        elif "evaluations" in full_payload and "full_test" in full_payload["evaluations"]:
            full_test = full_payload["evaluations"]["full_test"]["candidate"]
        elif "candidate" in full_payload:
            full_test = metric_summary(full_payload["candidate"])

    return {
        "candidate_id": candidate_id,
        "source": source,
        "val": {
            "reference": val_reference,
            "candidate": val_candidate,
            "delta_vs_reference": val_delta,
            "pass": val_pass,
        },
        "test10k": {
            "reference": test10k_reference,
            "candidate": test10k_candidate,
            "delta_vs_reference": test_delta,
            "pass": test10k_margin_pass,
        },
        "full_test_observed_posthoc": full_test,
        "decision": decision,
    }


def build_loop161_guard(
    *,
    loop151_val_eval: Path,
    loop151_test10k_eval: Path,
    loop144_val_eval: Path,
    loop144_test10k_eval: Path,
    loop159_audit: Path,
    loop160_audit: Path,
    output_json: Path,
    output_md: Optional[Path] = None,
    min_val_error_improvement: int = 3,
    min_test10k_error_improvement: int = 3,
) -> dict[str, Any]:
    loop151_val_ref, loop151_val_cand = from_authenticode_eval(loop151_val_eval)
    loop151_test_ref, loop151_test_cand = from_authenticode_eval(loop151_test10k_eval)
    _loop144_val_ref, loop144_val_cand = from_authenticode_eval(loop144_val_eval)
    _loop144_test_ref, loop144_test_cand = from_authenticode_eval(loop144_test10k_eval)
    loop159_val_ref, loop159_val_cand, loop159_test_ref, loop159_test_cand = from_loop159(loop159_audit)
    loop160_val_ref, loop160_val_cand, loop160_test_ref, loop160_test_cand = from_loop160(loop160_audit)

    candidates = [
        candidate_record(
            candidate_id="loop151_trusted_signer_guard",
            source="Loop151 vs Loop136 reference",
            val_reference=loop151_val_ref,
            val_candidate=loop151_val_cand,
            test10k_reference=loop151_test_ref,
            test10k_candidate=loop151_test_cand,
            min_val_error_improvement=min_val_error_improvement,
            min_test10k_error_improvement=min_test10k_error_improvement,
        ),
        candidate_record(
            candidate_id="loop144_union_trusted_signer",
            source="Loop144 union + Loop151 trusted signer",
            val_reference=CURRENT_BEST["val"],
            val_candidate=loop144_val_cand,
            test10k_reference=CURRENT_BEST["test10k"],
            test10k_candidate=loop144_test_cand,
            min_val_error_improvement=min_val_error_improvement,
            min_test10k_error_improvement=min_test10k_error_improvement,
        ),
        candidate_record(
            candidate_id="loop159_r11_only_trusted_signer",
            source="Loop159 R11-only + trusted signer",
            val_reference=loop159_val_ref,
            val_candidate=loop159_val_cand,
            test10k_reference=loop159_test_ref,
            test10k_candidate=loop159_test_cand,
            min_val_error_improvement=min_val_error_improvement,
            min_test10k_error_improvement=min_test10k_error_improvement,
        ),
        candidate_record(
            candidate_id="loop160_lowprob_r11_gate",
            source="Loop160 low-probability R11 gate",
            val_reference=loop160_val_ref,
            val_candidate=loop160_val_cand,
            test10k_reference=loop160_test_ref,
            test10k_candidate=loop160_test_cand,
            min_val_error_improvement=min_val_error_improvement,
            min_test10k_error_improvement=min_test10k_error_improvement,
        ),
    ]
    payload = {
        "schema": "axon_loop161_test10k_promotion_margin_guard_v1",
        "protocol": (
            "Read-only promotion-margin guard: Val winners need a material Test-10k error improvement "
            "before full-test confirmation is justified"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "thresholds": {
            "min_val_error_improvement": int(min_val_error_improvement),
            "min_test10k_error_improvement": int(min_test10k_error_improvement),
        },
        "candidates": candidates,
        "summary": {
            "allow_full_test_confirmation": sum(1 for row in candidates if row["decision"] == "allow_full_test_confirmation"),
            "rejected_test10k_margin": sum(1 for row in candidates if row["decision"] == "reject_test10k_margin_too_small"),
            "rejected_val_margin": sum(1 for row in candidates if row["decision"] == "reject_val_margin"),
        },
        "decision": "guard_active",
        "next_action": (
            "Use this guard before future full-test runs. Loop160's one-error Test-10k gain is too small to justify "
            "full-test promotion under the stricter margin."
        ),
    }
    write_json(output_json, payload)
    if output_md is not None:
        write_markdown(output_md, payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Loop161 Test-10k Promotion Margin Guard",
        "",
        f"- Min Val error improvement: `{payload['thresholds']['min_val_error_improvement']}`",
        f"- Min Test-10k error improvement: `{payload['thresholds']['min_test10k_error_improvement']}`",
        f"- Allow full-test confirmation: `{payload['summary']['allow_full_test_confirmation']}`",
        f"- Rejected by Test-10k margin: `{payload['summary']['rejected_test10k_margin']}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Val delta errors | Test-10k delta errors | Decision |",
        "|---|---:|---:|---|",
    ]
    for row in payload["candidates"]:
        lines.append(
            f"| `{row['candidate_id']}` | `{row['val']['delta_vs_reference']['errors']}` | "
            f"`{row['test10k']['delta_vs_reference']['errors']}` | `{row['decision']}` |"
        )
    lines.extend(["", "## Policy", "", IDENTITY_FEATURE_POLICY, ""])
    resolved.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop161 Test-10k promotion-margin guard.")
    parser.add_argument("--loop151-val-eval", type=Path, required=True)
    parser.add_argument("--loop151-test10k-eval", type=Path, required=True)
    parser.add_argument("--loop144-val-eval", type=Path, required=True)
    parser.add_argument("--loop144-test10k-eval", type=Path, required=True)
    parser.add_argument("--loop159-audit", type=Path, required=True)
    parser.add_argument("--loop160-audit", type=Path, required=True)
    parser.add_argument("--min-val-error-improvement", type=int, default=3)
    parser.add_argument("--min-test10k-error-improvement", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop161_guard(
        loop151_val_eval=args.loop151_val_eval,
        loop151_test10k_eval=args.loop151_test10k_eval,
        loop144_val_eval=args.loop144_val_eval,
        loop144_test10k_eval=args.loop144_test10k_eval,
        loop159_audit=args.loop159_audit,
        loop160_audit=args.loop160_audit,
        min_val_error_improvement=args.min_val_error_improvement,
        min_test10k_error_improvement=args.min_test10k_error_improvement,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "summary": payload["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
