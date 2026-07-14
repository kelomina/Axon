#!/usr/bin/env python3
"""Summarize Loop159 R11-only trusted-signer candidate governance.

Loop159 isolates the R11-filtered branch from the broader Loop144 union, then
keeps the Loop151 frozen trusted-signer guard. It is a model-selection audit:
Val and Test-10k are inspected before the pre-existing frozen full-test result
is considered. No fitting, threshold search, relabeling, or split/cache mutation
is performed here.
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
    "full_test": {"f1": 0.9908541911012403, "errors": 1466, "fp": 879, "fn": 587},
}
IDENTITY_FEATURE_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/split are alignment and audit fields only; "
    "Loop159 uses the frozen R11 branch prediction plus the preapproved Loop151 Authenticode trusted-signer guard, "
    "and never uses identity fields for feature selection, threshold selection, or production inference evidence"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric_block(payload: dict[str, Any], split: str) -> dict[str, Any]:
    candidate = payload["candidate"]
    reference = CURRENT_BEST[split]
    return {
        "f1": float(candidate["f1"]),
        "errors": int(candidate["errors"]),
        "fp": int(candidate["false_positive"]),
        "fn": int(candidate["false_negative"]),
        "delta_vs_loop151": {
            "f1": float(candidate["f1"]) - float(reference["f1"]),
            "errors": int(candidate["errors"]) - int(reference["errors"]),
            "fp": int(candidate["false_positive"]) - int(reference["fp"]),
            "fn": int(candidate["false_negative"]) - int(reference["fn"]),
        },
        "fixed_fp_by_signer": int(payload.get("fixed_fp", 0)),
        "introduced_fn_by_signer": int(payload.get("introduced_fn", 0)),
        "decision": payload.get("decision", ""),
    }


def build_loop159_audit(
    *,
    val_eval_json: Path,
    test10k_eval_json: Path,
    full_eval_json: Path,
    output_json: Path,
    output_md: Optional[Path] = None,
) -> dict[str, Any]:
    val = read_json(val_eval_json)
    test10k = read_json(test10k_eval_json)
    full = read_json(full_eval_json)
    metrics = {
        "val": _metric_block(val, "val"),
        "test10k": _metric_block(test10k, "test10k"),
        "full_test": _metric_block(full, "full_test"),
    }
    val_pass = metrics["val"]["delta_vs_loop151"]["errors"] < 0
    test10k_error_pass = metrics["test10k"]["delta_vs_loop151"]["errors"] < 0
    test10k_f1_non_regression = metrics["test10k"]["delta_vs_loop151"]["f1"] >= 0.0
    full_pass = metrics["full_test"]["delta_vs_loop151"]["errors"] < 0 and metrics["full_test"]["delta_vs_loop151"]["f1"] > 0.0
    decision = (
        "reject_full_test_confirmation_not_strict_best"
        if val_pass and test10k_f1_non_regression and not full_pass
        else "reject_test10k_gate"
        if val_pass and not test10k_error_pass
        else "reject_val_gate"
        if not val_pass
        else "candidate_requires_manual_review"
    )
    payload: dict[str, Any] = {
        "schema": "axon_loop159_r11_only_trusted_signer_candidate_audit_v1",
        "protocol": (
            "Val-first audit of R11-only recall recovery plus frozen Loop151 trusted-signer FP guard; "
            "no training, no threshold search, no split/cache mutation"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {
            "val_eval_json": str(resolve_path(val_eval_json)),
            "test10k_eval_json": str(resolve_path(test10k_eval_json)),
            "full_eval_json": str(resolve_path(full_eval_json)),
        },
        "current_best_reference": CURRENT_BEST,
        "metrics": metrics,
        "gate_review": {
            "val_pass": val_pass,
            "test10k_error_pass": test10k_error_pass,
            "test10k_f1_non_regression": test10k_f1_non_regression,
            "full_pass": full_pass,
        },
        "decision": decision,
        "next_action": (
            "Do not replace Loop151 for strict F1/total-error. Keep Loop159 only as a high-recall trade-off record: "
            "full-test FN improves, but FP and total errors regress."
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
        "# Loop159 R11-Only Trusted-Signer Candidate Audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Val delta errors vs Loop151: `{payload['metrics']['val']['delta_vs_loop151']['errors']}`",
        f"- Test-10k delta errors vs Loop151: `{payload['metrics']['test10k']['delta_vs_loop151']['errors']}`",
        f"- Full-test delta errors vs Loop151: `{payload['metrics']['full_test']['delta_vs_loop151']['errors']}`",
        f"- Full-test delta FP/FN vs Loop151: `{payload['metrics']['full_test']['delta_vs_loop151']['fp']}` / `{payload['metrics']['full_test']['delta_vs_loop151']['fn']}`",
        "",
        "## Policy",
        "",
        IDENTITY_FEATURE_POLICY,
        "",
    ]
    resolved.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop159 R11-only candidate audit.")
    parser.add_argument("--val-eval-json", type=Path, required=True)
    parser.add_argument("--test10k-eval-json", type=Path, required=True)
    parser.add_argument("--full-eval-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop159_audit(
        val_eval_json=args.val_eval_json,
        test10k_eval_json=args.test10k_eval_json,
        full_eval_json=args.full_eval_json,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "val_delta_errors": payload["metrics"]["val"]["delta_vs_loop151"]["errors"],
                "test10k_delta_errors": payload["metrics"]["test10k"]["delta_vs_loop151"]["errors"],
                "full_delta_errors": payload["metrics"]["full_test"]["delta_vs_loop151"]["errors"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
