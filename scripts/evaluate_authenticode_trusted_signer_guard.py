#!/usr/bin/env python3
"""Evaluate a frozen Authenticode trusted-signer FP guard.

The guard only downgrades already-malicious predictions to benign when Windows
Authenticode reports a valid signature and the signer subject matches a
predeclared trusted-publisher term. Paths and hashes are used only for alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IDENTITY_FEATURE_POLICY = (
    "source_path/source_sha256/sample_index/split are alignment and audit fields only; "
    "only Authenticode status plus signer certificate subject are used by this guard"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0])
    else:
        fieldnames = [
            "source_path",
            "source_sha256",
            "sample_index",
            "label",
            "split",
            "prediction",
            "trusted_signer_guard_prediction",
        ]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def _row_keys(row: dict[str, str]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    sample_index = str(row.get("sample_index") or "").strip()
    if sample_index:
        keys.append(("sample_index", sample_index))
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    if source_sha:
        keys.append(("source_sha256", source_sha))
    source_path = str(row.get("source_path") or "").strip()
    if source_path:
        keys.append(("source_path", source_path.casefold()))
    return keys


def build_signature_lookup(signature_rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in signature_rows:
        for key in _row_keys(row):
            lookup.setdefault(key, row)
    return lookup


def find_signature(row: dict[str, str], lookup: dict[tuple[str, str], dict[str, str]]) -> dict[str, str]:
    for key in _row_keys(row):
        match = lookup.get(key)
        if match is not None:
            return match
    return {}


def parse_terms(items: Sequence[str]) -> list[str]:
    terms: list[str] = []
    for item in items:
        for part in str(item).split("|"):
            text = part.strip()
            if text:
                terms.append(text)
    if not terms:
        raise ValueError("At least one trusted signer term is required")
    lowered = []
    for term in terms:
        folded = term.casefold()
        if folded not in lowered:
            lowered.append(folded)
    return lowered


def metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, object]:
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


def evaluate_trusted_signer_guard(
    *,
    predictions_csv: Path,
    signature_csv: Path,
    output_json: Path,
    output_predictions_csv: Optional[Path],
    trusted_terms: Sequence[str],
    score_threshold: float = 1.0,
    score_column: str = "stage2_prob_malicious",
    reference_errors: Optional[int] = None,
    min_error_improvement: int = 10,
) -> dict[str, object]:
    terms = parse_terms(trusted_terms)
    rows = read_rows(predictions_csv)
    signature_rows = read_rows(signature_csv)
    lookup = build_signature_lookup(signature_rows)

    labels: list[int] = []
    baseline_predictions: list[int] = []
    candidate_predictions: list[int] = []
    output_rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    matched_term_counts: Counter[str] = Counter()
    valid_signed_predicted_positive = 0
    trusted_signed_predicted_positive = 0
    missing_signature_rows_for_predicted_positive = 0
    fixed_fp = 0
    introduced_fn = 0

    for row in rows:
        label = _int(row.get("label"))
        baseline_pred = _int(row.get("prediction"))
        labels.append(label)
        baseline_predictions.append(baseline_pred)
        sig = find_signature(row, lookup)
        auth_status = str(sig.get("auth_status") or "")
        signer_subject = str(sig.get("signer_subject") or "")
        status_counts[auth_status or "missing"] += 1
        if baseline_pred == 1 and not sig:
            missing_signature_rows_for_predicted_positive += 1
        if baseline_pred == 1 and auth_status == "Valid":
            valid_signed_predicted_positive += 1
        score = _float(row.get(score_column), _float(row.get("prob_malicious"), 1.0))
        matched_terms = [term for term in terms if term in signer_subject.casefold()]
        should_downgrade = bool(
            baseline_pred == 1
            and auth_status == "Valid"
            and matched_terms
            and score <= float(score_threshold)
        )
        if should_downgrade:
            trusted_signed_predicted_positive += 1
            for term in matched_terms:
                matched_term_counts[term] += 1
        candidate_pred = 0 if should_downgrade else baseline_pred
        candidate_predictions.append(candidate_pred)
        if should_downgrade and label == 0:
            fixed_fp += 1
        elif should_downgrade and label == 1:
            introduced_fn += 1
        if output_predictions_csv is not None:
            item = dict(row)
            item["auth_status"] = auth_status
            item["signer_subject"] = signer_subject
            item["trusted_signer_terms"] = "|".join(matched_terms)
            item["trusted_signer_guard_downgrade"] = str(should_downgrade)
            item["trusted_signer_guard_prediction"] = str(candidate_pred)
            item["trusted_signer_guard_correct"] = str(candidate_pred == label)
            output_rows.append(item)

    baseline = metrics(labels, baseline_predictions)
    candidate = metrics(labels, candidate_predictions)
    ref_errors = int(reference_errors) if reference_errors is not None else int(baseline["errors"])
    error_improvement = ref_errors - int(candidate["errors"])
    decision = "allow_next_funnel_step" if error_improvement >= int(min_error_improvement) else "reject_val_margin_too_small"
    report = {
        "schema": "axon_authenticode_trusted_signer_guard_v1",
        "protocol": "frozen trusted-signer guard; no fitting, no relabeling, no split/cache mutation",
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "predictions_csv": str(resolve_path(predictions_csv)),
        "signature_csv": str(resolve_path(signature_csv)),
        "trusted_terms": list(trusted_terms),
        "score_column": score_column,
        "score_threshold": float(score_threshold),
        "baseline": baseline,
        "candidate": candidate,
        "reference_errors": ref_errors,
        "min_error_improvement": int(min_error_improvement),
        "error_improvement_vs_reference": int(error_improvement),
        "valid_signed_predicted_positive": int(valid_signed_predicted_positive),
        "trusted_signed_predicted_positive": int(trusted_signed_predicted_positive),
        "fixed_fp": int(fixed_fp),
        "introduced_fn": int(introduced_fn),
        "missing_signature_rows_for_predicted_positive": int(missing_signature_rows_for_predicted_positive),
        "signature_status_counts": dict(sorted(status_counts.items())),
        "matched_term_counts": dict(sorted(matched_term_counts.items())),
        "decision": decision,
        "artifacts": {
            "output_json": str(resolve_path(output_json)),
            "output_predictions_csv": str(resolve_path(output_predictions_csv)) if output_predictions_csv else "",
        },
    }
    resolved_json = resolve_path(output_json)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_predictions_csv is not None:
        write_rows(output_predictions_csv, output_rows)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an Authenticode trusted-signer FP guard.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--signature-csv", type=Path, required=True)
    parser.add_argument("--trusted-term", action="append", required=True)
    parser.add_argument("--score-column", default="stage2_prob_malicious")
    parser.add_argument("--score-threshold", type=float, default=1.0)
    parser.add_argument("--reference-errors", type=int, default=None)
    parser.add_argument("--min-error-improvement", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_trusted_signer_guard(
        predictions_csv=args.predictions_csv,
        signature_csv=args.signature_csv,
        output_json=args.output_json,
        output_predictions_csv=args.output_predictions_csv,
        trusted_terms=args.trusted_term,
        score_column=args.score_column,
        score_threshold=args.score_threshold,
        reference_errors=args.reference_errors,
        min_error_improvement=args.min_error_improvement,
    )
    print(
        json.dumps(
            {
                "baseline_errors": report["baseline"]["errors"],
                "candidate_errors": report["candidate"]["errors"],
                "fixed_fp": report["fixed_fp"],
                "introduced_fn": report["introduced_fn"],
                "decision": report["decision"],
                "output_json": str(resolve_path(args.output_json)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
