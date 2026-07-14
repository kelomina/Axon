#!/usr/bin/env python3
"""Build a conservative low-probability R11 rescue gate.

Loop159 showed that accepting every R11 0->1 rescue is a high-recall trade-off
that regresses full-test total errors. Loop160 searches only on Val for the
smallest baseline-probability threshold that gives a limited recall rescue
without increasing Val FP, then applies that frozen threshold to Test-10k and
optionally full-test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FEATURE_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/split are alignment and audit fields only; "
    "Loop160 selects a threshold only from Val metrics on non-identity probability columns and applies the frozen "
    "rule to later splits without using Test-10k or full-test for threshold selection"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["sample_index", "source_sha256", "label", "loop160_prediction"]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _prediction(row: dict[str, str]) -> int:
    if str(row.get("trusted_signer_guard_prediction") or "").strip():
        return _int(row.get("trusted_signer_guard_prediction"))
    return _int(row.get("prediction"))


def _key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("sample_index") or "").strip(), str(row.get("source_sha256") or "").strip().casefold())


def align_rows(base_csv: Path, candidate_csv: Path) -> list[tuple[dict[str, str], dict[str, str]]]:
    base_rows = read_rows(base_csv)
    candidate_rows = read_rows(candidate_csv)
    candidate_by_key = {_key(row): row for row in candidate_rows}
    if len(candidate_by_key) != len(candidate_rows):
        raise ValueError("candidate_csv has duplicate sample_index/source_sha256 keys")
    aligned: list[tuple[dict[str, str], dict[str, str]]] = []
    missing = 0
    label_mismatch = 0
    for row in base_rows:
        candidate = candidate_by_key.get(_key(row))
        if candidate is None:
            missing += 1
            continue
        if str(row.get("label")) != str(candidate.get("label")):
            label_mismatch += 1
        aligned.append((row, candidate))
    if missing or label_mismatch or len(aligned) != len(base_rows):
        raise ValueError(
            f"alignment failed: missing={missing}, label_mismatch={label_mismatch}, aligned={len(aligned)}, base={len(base_rows)}"
        )
    return aligned


def metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "errors": int(fp + fn),
    }


def apply_gate(
    aligned: Sequence[tuple[dict[str, str], dict[str, str]]],
    *,
    threshold: Optional[float],
    score_column: str,
    output_predictions_csv: Optional[Path] = None,
) -> dict[str, Any]:
    labels: list[int] = []
    base_predictions: list[int] = []
    gated_predictions: list[int] = []
    output_rows: list[dict[str, Any]] = []
    accepted = 0
    accepted_correct = 0
    accepted_wrong = 0
    candidate_0_to_1 = 0

    for base, candidate in aligned:
        label = _int(base.get("label"))
        base_pred = _prediction(base)
        candidate_pred = _prediction(candidate)
        labels.append(label)
        base_predictions.append(base_pred)
        score = _float(candidate.get(score_column), default=1.0)
        is_candidate_0_to_1 = base_pred == 0 and candidate_pred == 1
        if is_candidate_0_to_1:
            candidate_0_to_1 += 1
        accept = bool(threshold is not None and is_candidate_0_to_1 and score <= float(threshold))
        gated_pred = candidate_pred if accept else base_pred
        gated_predictions.append(gated_pred)
        if accept:
            accepted += 1
            if gated_pred == label:
                accepted_correct += 1
            else:
                accepted_wrong += 1
        if output_predictions_csv is not None:
            item = dict(base)
            item["loop160_score_column"] = score_column
            item["loop160_gate_score"] = f"{score:.10f}"
            item["loop160_selected_threshold"] = "" if threshold is None else f"{float(threshold):.10f}"
            item["loop160_candidate_prediction"] = str(candidate_pred)
            item["loop160_accept_candidate"] = str(accept)
            item["loop160_prediction"] = str(gated_pred)
            item["loop160_correct"] = str(gated_pred == label)
            output_rows.append(item)

    base_metrics = metrics(labels, base_predictions)
    gated_metrics = metrics(labels, gated_predictions)
    delta = {
        "errors": int(gated_metrics["errors"]) - int(base_metrics["errors"]),
        "false_positive": int(gated_metrics["false_positive"]) - int(base_metrics["false_positive"]),
        "false_negative": int(gated_metrics["false_negative"]) - int(base_metrics["false_negative"]),
        "f1": float(gated_metrics["f1"]) - float(base_metrics["f1"]),
    }
    if output_predictions_csv is not None:
        write_rows(output_predictions_csv, output_rows)
    return {
        "rows": len(aligned),
        "threshold": threshold,
        "score_column": score_column,
        "candidate_0_to_1_rows": candidate_0_to_1,
        "accepted_rows": accepted,
        "accepted_correct": accepted_correct,
        "accepted_wrong": accepted_wrong,
        "baseline": base_metrics,
        "candidate": gated_metrics,
        "delta_vs_baseline": delta,
    }


def threshold_candidates(aligned: Sequence[tuple[dict[str, str], dict[str, str]]], *, score_column: str) -> list[float]:
    values: set[float] = set()
    for base, candidate in aligned:
        if _prediction(base) == 0 and _prediction(candidate) == 1:
            values.add(_float(candidate.get(score_column), default=1.0))
    return sorted(values)


def select_threshold(
    aligned: Sequence[tuple[dict[str, str], dict[str, str]]],
    *,
    score_column: str,
    min_val_error_improvement: int,
    max_val_fp_delta: int,
) -> tuple[Optional[float], list[dict[str, Any]]]:
    sweep: list[dict[str, Any]] = []
    selected: Optional[float] = None
    for threshold in threshold_candidates(aligned, score_column=score_column):
        result = apply_gate(aligned, threshold=threshold, score_column=score_column)
        row = {
            "threshold": threshold,
            "accepted_rows": result["accepted_rows"],
            "errors": result["candidate"]["errors"],
            "fp": result["candidate"]["false_positive"],
            "fn": result["candidate"]["false_negative"],
            "f1": result["candidate"]["f1"],
            "delta_errors": result["delta_vs_baseline"]["errors"],
            "delta_fp": result["delta_vs_baseline"]["false_positive"],
            "delta_fn": result["delta_vs_baseline"]["false_negative"],
        }
        sweep.append(row)
        if (
            selected is None
            and row["delta_errors"] <= -int(min_val_error_improvement)
            and row["delta_fp"] <= int(max_val_fp_delta)
        ):
            selected = threshold
    return selected, sweep


def build_loop160_gate(
    *,
    val_base_csv: Path,
    val_candidate_csv: Path,
    test10k_base_csv: Path,
    test10k_candidate_csv: Path,
    output_json: Path,
    output_md: Optional[Path] = None,
    output_dir: Path,
    full_base_csv: Optional[Path] = None,
    full_candidate_csv: Optional[Path] = None,
    score_column: str = "baseline_prob_malicious",
    min_val_error_improvement: int = 3,
    max_val_fp_delta: int = 0,
) -> dict[str, Any]:
    output_root = resolve_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    val_aligned = align_rows(val_base_csv, val_candidate_csv)
    selected_threshold, val_sweep = select_threshold(
        val_aligned,
        score_column=score_column,
        min_val_error_improvement=min_val_error_improvement,
        max_val_fp_delta=max_val_fp_delta,
    )
    val_eval = apply_gate(
        val_aligned,
        threshold=selected_threshold,
        score_column=score_column,
        output_predictions_csv=output_root / "loop160_lowprob_r11_val_predictions.csv",
    )
    test10k_eval = apply_gate(
        align_rows(test10k_base_csv, test10k_candidate_csv),
        threshold=selected_threshold,
        score_column=score_column,
        output_predictions_csv=output_root / "loop160_lowprob_r11_test10k_predictions.csv",
    )
    full_eval: Optional[dict[str, Any]] = None
    if full_base_csv is not None and full_candidate_csv is not None:
        full_eval = apply_gate(
            align_rows(full_base_csv, full_candidate_csv),
            threshold=selected_threshold,
            score_column=score_column,
            output_predictions_csv=output_root / "loop160_lowprob_r11_full_predictions.csv",
        )

    val_pass = selected_threshold is not None and val_eval["delta_vs_baseline"]["errors"] < 0
    test10k_pass = bool(test10k_eval["delta_vs_baseline"]["errors"] < 0)
    full_pass = bool(full_eval and full_eval["delta_vs_baseline"]["errors"] < 0)
    if selected_threshold is None:
        decision = "reject_no_val_threshold"
    elif not test10k_pass:
        decision = "reject_test10k_confirmation"
    elif full_eval is not None and not full_pass:
        decision = "reject_full_test_confirmation"
    else:
        decision = "allow_next_funnel_step"

    payload: dict[str, Any] = {
        "schema": "axon_loop160_lowprob_r11_gate_v1",
        "protocol": (
            "Val-only conservative threshold selection for R11 0->1 rescue rows; threshold is frozen before "
            "Test-10k/full-test confirmation"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {
            "val_base_csv": str(resolve_path(val_base_csv)),
            "val_candidate_csv": str(resolve_path(val_candidate_csv)),
            "test10k_base_csv": str(resolve_path(test10k_base_csv)),
            "test10k_candidate_csv": str(resolve_path(test10k_candidate_csv)),
            "full_base_csv": str(resolve_path(full_base_csv)) if full_base_csv else None,
            "full_candidate_csv": str(resolve_path(full_candidate_csv)) if full_candidate_csv else None,
        },
        "selection_policy": {
            "score_column": score_column,
            "direction": "base_prediction_0_to_candidate_prediction_1",
            "mode": "smallest_val_threshold_meeting_constraints",
            "min_val_error_improvement": int(min_val_error_improvement),
            "max_val_fp_delta": int(max_val_fp_delta),
            "selected_threshold": selected_threshold,
        },
        "val_sweep": val_sweep,
        "evaluations": {
            "val": val_eval,
            "test10k": test10k_eval,
            "full_test": full_eval,
        },
        "gate_review": {
            "val_pass": val_pass,
            "test10k_pass": test10k_pass,
            "full_pass": full_pass,
        },
        "decision": decision,
        "outputs": {
            "val_predictions_csv": str(output_root / "loop160_lowprob_r11_val_predictions.csv"),
            "test10k_predictions_csv": str(output_root / "loop160_lowprob_r11_test10k_predictions.csv"),
            "full_predictions_csv": str(output_root / "loop160_lowprob_r11_full_predictions.csv")
            if full_eval is not None
            else None,
        },
    }
    write_json(output_json, payload)
    if output_md is not None:
        write_markdown(output_md, payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Loop160 Low-Probability R11 Gate",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Selected threshold: `{payload['selection_policy']['selected_threshold']}`",
        f"- Val delta errors: `{payload['evaluations']['val']['delta_vs_baseline']['errors']}`",
        f"- Test-10k delta errors: `{payload['evaluations']['test10k']['delta_vs_baseline']['errors']}`",
    ]
    full_eval = payload["evaluations"].get("full_test")
    if full_eval:
        lines.append(f"- Full-test delta errors: `{full_eval['delta_vs_baseline']['errors']}`")
    lines.extend(["", "## Policy", "", IDENTITY_FEATURE_POLICY, ""])
    resolved.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop160 low-probability R11 rescue gate.")
    parser.add_argument("--val-base-csv", type=Path, required=True)
    parser.add_argument("--val-candidate-csv", type=Path, required=True)
    parser.add_argument("--test10k-base-csv", type=Path, required=True)
    parser.add_argument("--test10k-candidate-csv", type=Path, required=True)
    parser.add_argument("--full-base-csv", type=Path, default=None)
    parser.add_argument("--full-candidate-csv", type=Path, default=None)
    parser.add_argument("--score-column", default="baseline_prob_malicious")
    parser.add_argument("--min-val-error-improvement", type=int, default=3)
    parser.add_argument("--max-val-fp-delta", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop160_gate(
        val_base_csv=args.val_base_csv,
        val_candidate_csv=args.val_candidate_csv,
        test10k_base_csv=args.test10k_base_csv,
        test10k_candidate_csv=args.test10k_candidate_csv,
        full_base_csv=args.full_base_csv,
        full_candidate_csv=args.full_candidate_csv,
        score_column=args.score_column,
        min_val_error_improvement=args.min_val_error_improvement,
        max_val_fp_delta=args.max_val_fp_delta,
        output_dir=args.output_dir,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "selected_threshold": payload["selection_policy"]["selected_threshold"],
                "val_delta_errors": payload["evaluations"]["val"]["delta_vs_baseline"]["errors"],
                "test10k_delta_errors": payload["evaluations"]["test10k"]["delta_vs_baseline"]["errors"],
                "full_delta_errors": payload["evaluations"].get("full_test", {}).get("delta_vs_baseline", {}).get("errors"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
