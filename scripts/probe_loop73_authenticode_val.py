#!/usr/bin/env python3
"""Probe Authenticode trust status as a Val-only FP guard.

Paths are used only to open files for Windows signature verification. Filename,
directory, extension, path text, source hash, sample index, split, and row order
are alignment/audit fields only and are not model evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence


IDENTITY_FEATURE_POLICY = (
    "filename/path/extension/directory/source hash/cache_path/sample_index/split/row order are loading, "
    "alignment, cache-audit, duplicate-review, and manual-review fields only; they are not model evidence "
    "and must not drive thresholds, relabeling, feature engineering, or production inference"
)
PROTOCOL = (
    "Val-only Authenticode trust probe; paths are used only to open files for signature verification; "
    "no Test-10k or full-test access, no model fitting, no automatic relabeling, no split mutation"
)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
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


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def metrics_from_predictions(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": len(labels),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": fp + fn,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_thresholds(spec: str) -> list[float]:
    text = str(spec or "").strip()
    if not text:
        return [1.0]
    if ":" in text:
        start_text, stop_text, step_text = text.split(":", 2)
        start = float(start_text)
        stop = float(stop_text)
        step = float(step_text)
        if step <= 0:
            raise ValueError("Threshold step must be positive")
        values = []
        current = start
        while current <= stop + 1e-12:
            values.append(round(current, 10))
            current += step
        return values
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _row_key(row: dict) -> tuple[str, str]:
    sample_index = str(row.get("sample_index") or "").strip()
    if sample_index:
        return "sample_index", sample_index
    sha = str(row.get("source_sha256") or "").strip().casefold()
    if sha:
        return "source_sha256", sha
    return "source_path", str(row.get("source_path") or "").strip().casefold()


def _signature_lookup(rows: Sequence[dict]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for row in rows:
        for key_name in ("sample_index", "source_sha256", "source_path"):
            value = str(row.get(key_name) or "").strip()
            if not value:
                continue
            key_value = value.casefold() if key_name != "sample_index" else value
            lookup.setdefault((key_name, key_value), row)
    return lookup


def _find_signature(row: dict, lookup: dict[tuple[str, str], dict]) -> dict:
    sample_index = str(row.get("sample_index") or "").strip()
    if sample_index and ("sample_index", sample_index) in lookup:
        return lookup[("sample_index", sample_index)]
    sha = str(row.get("source_sha256") or "").strip().casefold()
    if sha and ("source_sha256", sha) in lookup:
        return lookup[("source_sha256", sha)]
    source_path = str(row.get("source_path") or "").strip().casefold()
    if source_path and ("source_path", source_path) in lookup:
        return lookup[("source_path", source_path)]
    return {}


def _selected_signature_rows(
    prediction_rows: Sequence[dict],
    *,
    only_predicted_positive: bool,
    max_rows: Optional[int],
) -> list[dict]:
    selected = []
    for ordinal, row in enumerate(prediction_rows):
        if only_predicted_positive and _int(row.get("prediction")) != 1:
            continue
        selected.append(
            {
                "row_ordinal": str(ordinal),
                "source_path": row.get("source_path", ""),
                "source_sha256": row.get("source_sha256", ""),
                "sample_index": row.get("sample_index", ""),
                "prediction": row.get("prediction", ""),
                "label": row.get("label", ""),
            }
        )
        if max_rows is not None and len(selected) >= max_rows:
            break
    return selected


def build_signature_cache(
    *,
    predictions_csv: Path,
    output_csv: Path,
    only_predicted_positive: bool,
    max_rows: Optional[int],
    powershell_exe: str,
) -> dict[str, Any]:
    prediction_rows = read_rows(predictions_csv)
    selected = _selected_signature_rows(
        prediction_rows,
        only_predicted_positive=only_predicted_positive,
        max_rows=max_rows,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "authenticode_input.csv"
        write_rows(
            input_csv,
            selected,
            ["row_ordinal", "source_path", "source_sha256", "sample_index", "prediction", "label"],
        )
        ps_script = r"""
$InputCsv = $env:AXON_AUTH_INPUT
$OutputCsv = $env:AXON_AUTH_OUTPUT
Import-Module Microsoft.PowerShell.Security -ErrorAction Stop
$rows = Import-Csv -LiteralPath $InputCsv
$out = foreach ($row in $rows) {
  try {
    $sig = Get-AuthenticodeSignature -LiteralPath $row.source_path
    [PSCustomObject]@{
      row_ordinal = $row.row_ordinal
      source_path = $row.source_path
      source_sha256 = $row.source_sha256
      sample_index = $row.sample_index
      prediction = $row.prediction
      label = $row.label
      auth_status = [string]$sig.Status
      auth_status_message = [string]$sig.StatusMessage
      signer_subject = if ($sig.SignerCertificate) { [string]$sig.SignerCertificate.Subject } else { "" }
      signer_issuer = if ($sig.SignerCertificate) { [string]$sig.SignerCertificate.Issuer } else { "" }
      signer_thumbprint = if ($sig.SignerCertificate) { [string]$sig.SignerCertificate.Thumbprint } else { "" }
      timestamper_subject = if ($sig.TimeStamperCertificate) { [string]$sig.TimeStamperCertificate.Subject } else { "" }
      collection_error = ""
    }
  } catch {
    [PSCustomObject]@{
      row_ordinal = $row.row_ordinal
      source_path = $row.source_path
      source_sha256 = $row.source_sha256
      sample_index = $row.sample_index
      prediction = $row.prediction
      label = $row.label
      auth_status = "CollectionError"
      auth_status_message = ""
      signer_subject = ""
      signer_issuer = ""
      signer_thumbprint = ""
      timestamper_subject = ""
      collection_error = [string]$_.Exception.Message
    }
  }
}
$out | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
"""
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["AXON_AUTH_INPUT"] = str(input_csv)
        env["AXON_AUTH_OUTPUT"] = str(output_csv)
        # Avoid loading PowerShell 7 modules into Windows PowerShell, which can
        # break Microsoft.PowerShell.Security type data and hide
        # Get-AuthenticodeSignature.
        env["PSModulePath"] = ";".join(
            [
                str(Path.home() / "Documents" / "WindowsPowerShell" / "Modules"),
                r"C:\Program Files\WindowsPowerShell\Modules",
                r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\Modules",
            ]
        )
        subprocess.run(
            [
                powershell_exe,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_script,
            ],
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
    return {
        "predictions_csv": str(predictions_csv),
        "output_csv": str(output_csv),
        "selected_rows": len(selected),
        "only_predicted_positive": bool(only_predicted_positive),
        "max_rows": max_rows,
    }


def evaluate_valid_signature_downgrade(
    *,
    predictions_csv: Path,
    signature_csv: Path,
    output_json: Path,
    output_predictions_csv: Optional[Path],
    reference_val_errors: Optional[int],
    min_val_error_improvement: int,
    score_thresholds: Sequence[float] = (1.0,),
) -> dict[str, Any]:
    rows = read_rows(predictions_csv)
    signature_rows = read_rows(signature_csv)
    sig_lookup = _signature_lookup(signature_rows)
    labels = [_int(row.get("label")) for row in rows]
    baseline_predictions = [_int(row.get("prediction")) for row in rows]
    baseline = metrics_from_predictions(labels, baseline_predictions)

    missing_signature_rows = 0
    row_signatures = []
    for row, label, baseline_pred in zip(rows, labels, baseline_predictions):
        sig = _find_signature(row, sig_lookup)
        auth_status = str(sig.get("auth_status") or "")
        if baseline_pred == 1 and not sig:
            missing_signature_rows += 1
        row_signatures.append((sig, auth_status))

    threshold_reports = []
    best_predictions: list[int] = list(baseline_predictions)
    best_threshold = None
    for threshold in score_thresholds:
        candidate_predictions = []
        valid_signed_predicted_positive = 0
        valid_signed_flips = 0
        fixed_fp = 0
        introduced_fn = 0
        for row, label, baseline_pred, (_, auth_status) in zip(rows, labels, baseline_predictions, row_signatures):
            score = _float(row.get("final_prob_malicious"), _float(row.get("prob_malicious"), 1.0))
            is_valid_predicted_positive = baseline_pred == 1 and auth_status == "Valid"
            should_flip = is_valid_predicted_positive and score <= float(threshold)
            pred = 0 if should_flip else baseline_pred
            if is_valid_predicted_positive:
                valid_signed_predicted_positive += 1
            if should_flip:
                valid_signed_flips += 1
                if label == 0:
                    fixed_fp += 1
                elif label == 1:
                    introduced_fn += 1
            candidate_predictions.append(pred)
        candidate_metrics = metrics_from_predictions(labels, candidate_predictions)
        threshold_report = {
            "score_threshold": float(threshold),
            "metrics": candidate_metrics,
            "valid_signed_predicted_positive": valid_signed_predicted_positive,
            "valid_signed_flips": valid_signed_flips,
            "fixed_fp": fixed_fp,
            "introduced_fn": introduced_fn,
        }
        threshold_reports.append(threshold_report)
        if best_threshold is None or (
            candidate_metrics["errors"],
            -candidate_metrics["f1"],
            introduced_fn,
        ) < (
            best_threshold["metrics"]["errors"],
            -best_threshold["metrics"]["f1"],
            best_threshold["introduced_fn"],
        ):
            best_threshold = threshold_report
            best_predictions = candidate_predictions

    if best_threshold is None:
        best_threshold = {
            "score_threshold": 1.0,
            "metrics": baseline,
            "valid_signed_predicted_positive": 0,
            "valid_signed_flips": 0,
            "fixed_fp": 0,
            "introduced_fn": 0,
        }

    output_rows = []
    if output_predictions_csv is not None:
        best_threshold_value = float(best_threshold["score_threshold"])
        for row, label, baseline_pred, (sig, auth_status), pred in zip(
            rows,
            labels,
            baseline_predictions,
            row_signatures,
            best_predictions,
        ):
            score = _float(row.get("final_prob_malicious"), _float(row.get("prob_malicious"), 1.0))
            should_flip = baseline_pred == 1 and auth_status == "Valid" and score <= best_threshold_value
            item = dict(row)
            item.update(
                {
                    "auth_status": auth_status,
                    "signer_thumbprint": sig.get("signer_thumbprint", ""),
                    "authenticode_valid_downgrade": str(should_flip),
                    "loop73_prediction": str(pred),
                    "loop73_correct": str(pred == label),
                }
            )
            output_rows.append(item)

    candidate = best_threshold["metrics"]
    reference_errors = int(reference_val_errors) if reference_val_errors is not None else int(baseline["errors"])
    error_improvement = int(reference_errors) - int(candidate["errors"])
    decision = (
        "allow_test10k_candidate"
        if error_improvement >= int(min_val_error_improvement)
        else "reject_val_margin_too_small"
    )
    report = {
        "schema": "axon_loop73_authenticode_val_probe_v1",
        "protocol": PROTOCOL,
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "predictions_csv": str(predictions_csv),
        "signature_csv": str(signature_csv),
        "candidate_rule": (
            "if baseline prediction is malicious, Authenticode Status == Valid, and final_prob_malicious <= "
            "selected score threshold, downgrade to benign"
        ),
        "baseline": baseline,
        "candidate": candidate,
        "selected_score_threshold": float(best_threshold["score_threshold"]),
        "threshold_candidates": threshold_reports,
        "reference_val_errors": reference_errors,
        "min_val_error_improvement": int(min_val_error_improvement),
        "error_improvement_vs_reference": error_improvement,
        "valid_signed_predicted_positive": int(best_threshold["valid_signed_predicted_positive"]),
        "valid_signed_flips": int(best_threshold["valid_signed_flips"]),
        "fixed_fp": int(best_threshold["fixed_fp"]),
        "introduced_fn": int(best_threshold["introduced_fn"]),
        "missing_signature_rows_for_predicted_positive": missing_signature_rows,
        "signature_status_counts": dict(
            sorted(
                {
                    status: sum(1 for row in signature_rows if str(row.get("auth_status") or "") == status)
                    for status in {str(row.get("auth_status") or "") for row in signature_rows}
                }.items()
            )
        ),
        "decision": decision,
        "artifacts": {
            "output_json": str(output_json),
            "output_predictions_csv": str(output_predictions_csv) if output_predictions_csv else "",
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_predictions_csv is not None:
        fieldnames = list(output_rows[0].keys()) if output_rows else []
        write_rows(output_predictions_csv, output_rows, fieldnames)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Authenticode Valid signature as a Val-only FP guard.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--signature-csv", type=Path)
    parser.add_argument("--build-signature-cache-csv", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path)
    parser.add_argument("--reference-val-errors", type=int)
    parser.add_argument("--min-val-error-improvement", type=int, default=10)
    parser.add_argument("--score-thresholds", default="0.50:1.00:0.05")
    parser.add_argument("--max-signature-rows", type=int)
    parser.add_argument("--all-rows", action="store_true", help="Collect signatures for every prediction row; default only prediction==1 rows.")
    parser.add_argument("--powershell-exe", default="powershell")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    signature_csv = args.signature_csv
    cache_report = None
    if signature_csv is None:
        if args.build_signature_cache_csv is None:
            raise ValueError("Provide --signature-csv or --build-signature-cache-csv")
        signature_csv = args.build_signature_cache_csv
        cache_report = build_signature_cache(
            predictions_csv=args.predictions_csv,
            output_csv=signature_csv,
            only_predicted_positive=not bool(args.all_rows),
            max_rows=args.max_signature_rows,
            powershell_exe=args.powershell_exe,
        )
    report = evaluate_valid_signature_downgrade(
        predictions_csv=args.predictions_csv,
        signature_csv=signature_csv,
        output_json=args.output_json,
        output_predictions_csv=args.output_predictions_csv,
        reference_val_errors=args.reference_val_errors,
        min_val_error_improvement=args.min_val_error_improvement,
        score_thresholds=parse_thresholds(args.score_thresholds),
    )
    if cache_report is not None:
        report["signature_cache_build"] = cache_report
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline_errors": report["baseline"]["errors"],
                "candidate_errors": report["candidate"]["errors"],
                "fixed_fp": report["fixed_fp"],
                "introduced_fn": report["introduced_fn"],
                "decision": report["decision"],
                "signature_csv": str(signature_csv),
                "output_json": str(args.output_json),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
