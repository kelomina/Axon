#!/usr/bin/env python3
"""Evaluate a trained probability calibrator on exported predictions."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def _is_valid_source_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _npz_scalar_to_text(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def _load_cache_features(cache_path: Path, *, expected_label: int, expected_source_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as data:
        required = {"pe_features", "label", "source_sha256"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Cache missing required fields {missing}: {cache_path}")
        cache_label = int(data["label"])
        if cache_label != int(expected_label):
            raise ValueError(f"Cache label mismatch for {cache_path}: expected {expected_label}, got {cache_label}")
        cache_sha = _npz_scalar_to_text(data["source_sha256"]).strip().casefold()
        if cache_sha != expected_source_sha256:
            raise ValueError(f"Cache source_sha256 mismatch for {cache_path}")
        pe_features = data["pe_features"].astype(np.float32)
        stat_features = data.get("stat_features", np.zeros(49, dtype=np.float32)).astype(np.float32)
    return pe_features, stat_features


def _write_missing_cache_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source_path",
        "source_sha256",
        "cache_path",
        "label",
        "split",
        "sample_index",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_calibrated_prediction_rows(
    path: Path,
    *,
    rows: list[dict],
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    calibrator_model_scores: np.ndarray,
    calibrator_scores: np.ndarray,
    baseline_threshold: float,
    calibrator_threshold: float,
    blend_model_weight: float,
) -> None:
    fieldnames = [
        "source_path",
        "source_sha256",
        "cache_path",
        "label",
        "split",
        "sample_index",
        "baseline_prob_malicious",
        "baseline_threshold",
        "baseline_prediction",
        "baseline_correct",
        "calibrator_model_prob_malicious",
        "blend_model_weight",
        "calibrated_prob_malicious",
        "calibrated_threshold",
        "calibrated_minus_baseline",
        "calibrated_prediction",
        "calibrated_correct",
        "error_transition",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row, label, baseline_score, calibrator_model_score, calibrator_score in zip(
            rows,
            labels,
            baseline_scores,
            calibrator_model_scores,
            calibrator_scores,
        ):
            label = int(label)
            baseline_prediction = int(float(baseline_score) >= float(baseline_threshold))
            calibrator_prediction = int(float(calibrator_score) >= float(calibrator_threshold))
            baseline_correct = baseline_prediction == label
            calibrator_correct = calibrator_prediction == label
            if baseline_correct and calibrator_correct:
                transition = "both_correct"
            elif (not baseline_correct) and calibrator_correct:
                transition = "fixed_by_calibrator"
            elif baseline_correct and (not calibrator_correct):
                transition = "broken_by_calibrator"
            else:
                transition = "persistent_error"
            writer.writerow(
                {
                    "source_path": row.get("source_path", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "cache_path": row.get("cache_path", ""),
                    "label": label,
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                    "baseline_prob_malicious": float(baseline_score),
                    "baseline_threshold": float(baseline_threshold),
                    "baseline_prediction": baseline_prediction,
                    "baseline_correct": baseline_correct,
                    "calibrator_model_prob_malicious": float(calibrator_model_score),
                    "blend_model_weight": float(blend_model_weight),
                    "calibrated_prob_malicious": float(calibrator_score),
                    "calibrated_threshold": float(calibrator_threshold),
                    "calibrated_minus_baseline": float(calibrator_score) - float(baseline_score),
                    "calibrated_prediction": calibrator_prediction,
                    "calibrated_correct": calibrator_correct,
                    "error_transition": transition,
                }
            )


def _load_prediction_features(
    predictions_path: Path,
    *,
    allow_missing_cache: bool = False,
    missing_cache_output: Path | None = None,
):
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    features = []
    labels = []
    probabilities = []
    kept_rows = []
    skipped_missing_cache = 0
    missing_cache_examples = []
    missing_cache_rows = []

    for row in rows:
        source_sha = str(row.get("source_sha256") or "").strip().casefold()
        if not _is_valid_source_sha256(source_sha):
            raise ValueError(f"Prediction row has invalid source_sha256 in {predictions_path}: {source_sha!r}")
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"Prediction row has invalid label in {predictions_path}: {row.get('label')!r}")
        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            skipped_missing_cache += 1
            if len(missing_cache_examples) < 10:
                missing_cache_examples.append(str(cache_path))
            missing_cache_rows.append(
                {
                    "source_path": row.get("source_path", ""),
                    "source_sha256": source_sha,
                    "cache_path": str(cache_path),
                    "label": row.get("label", ""),
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                }
            )
            continue

        pe_features, stat_features = _load_cache_features(
            cache_path,
            expected_label=label,
            expected_source_sha256=source_sha,
        )

        prob = float(row["prob_malicious"])
        probability_features = np.array(
            [
                prob,
                prob * prob,
                np.log(max(prob, 1e-6)),
                np.log(max(1.0 - prob, 1e-6)),
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([probability_features, stat_features, pe_features]))
        labels.append(label)
        probabilities.append(prob)
        kept_rows.append(row)

    if missing_cache_output is not None:
        _write_missing_cache_rows(missing_cache_output, missing_cache_rows)
    if skipped_missing_cache and not allow_missing_cache:
        raise FileNotFoundError(
            "Prediction CSV references missing cache files: "
            f"{skipped_missing_cache}/{len(rows)} missing. "
            "Regenerate predictions/cache before using this as a complete confirmation, "
            "or pass --allow-missing-cache only for diagnostic subset runs. "
            f"Examples: {missing_cache_examples}"
        )
    if not features:
        raise ValueError(f"No usable prediction rows with cache features were loaded from {predictions_path}")

    return (
        np.vstack(features),
        np.asarray(labels),
        np.asarray(probabilities),
        kept_rows,
        {
            "total": len(rows),
            "kept": len(labels),
            "skipped_missing_cache": skipped_missing_cache,
            "missing_cache_examples": missing_cache_examples,
            "missing_cache_output": str(missing_cache_output) if missing_cache_output is not None else None,
        },
    )


def _metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": auc,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "errors": int(fp + fn),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
    }


def _slice_metrics(
    *,
    rows: list[dict],
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    calibrator_scores: np.ndarray,
    baseline_threshold: float,
    calibrator_threshold: float,
) -> dict:
    _ = rows
    slices = {}
    definitions = {
        "benign_label_0": np.asarray([label == 0 for label in labels], dtype=bool),
        "malicious_label_1": np.asarray([label == 1 for label in labels], dtype=bool),
        "baseline_near_threshold_0.40_0.60": np.asarray(
            (baseline_scores >= 0.40) & (baseline_scores <= 0.60),
            dtype=bool,
        ),
    }
    for name, mask in definitions.items():
        if not bool(mask.any()):
            continue
        baseline = _metrics(baseline_scores[mask], labels[mask], baseline_threshold)
        calibrator = _metrics(calibrator_scores[mask], labels[mask], calibrator_threshold)
        slices[name] = {
            "rows": int(mask.sum()),
            "baseline": baseline,
            "calibrator_metrics": calibrator,
            "delta_errors_vs_baseline": calibrator["errors"] - baseline["errors"],
            "delta_f1_vs_baseline": calibrator["f1"] - baseline["f1"],
        }
    return slices


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a train-split probability calibrator.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--baseline-threshold", type=float, default=0.53)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-cache",
        action="store_true",
        help="Allow diagnostic subset evaluation when some prediction cache files are missing.",
    )
    parser.add_argument(
        "--missing-cache-output",
        type=Path,
        default=None,
        help="Optional CSV output listing every prediction row whose cache_path is missing.",
    )
    args = parser.parse_args()

    with args.model.open("rb") as handle:
        payload = pickle.load(handle)

    model = payload["model"]
    blend_weight = float(payload["blend_model_weight"])
    threshold = args.threshold
    if threshold is None:
        threshold = float(payload["val_selected_threshold"])

    features, labels, baseline_probs, kept_rows, counts = _load_prediction_features(
        args.predictions,
        allow_missing_cache=args.allow_missing_cache,
        missing_cache_output=args.missing_cache_output,
    )
    calibrator_probs = model.predict_proba(features)[:, 1]
    blended_probs = blend_weight * calibrator_probs + (1.0 - blend_weight) * baseline_probs

    baseline_metrics = _metrics(baseline_probs, labels, args.baseline_threshold)
    calibrator_metrics = _metrics(blended_probs, labels, threshold)
    if args.output_predictions_csv is not None:
        _write_calibrated_prediction_rows(
            args.output_predictions_csv,
            rows=kept_rows,
            labels=labels,
            baseline_scores=baseline_probs,
            calibrator_model_scores=calibrator_probs,
            calibrator_scores=blended_probs,
            baseline_threshold=args.baseline_threshold,
            calibrator_threshold=threshold,
            blend_model_weight=blend_weight,
        )
    report = {
        "protocol": (
            "fixed train-split calibrator evaluated on exported predictions; "
            "source_sha256 validates cache identity only; path/name/directory/extension are not slices or evidence; "
            "threshold is provided or stored from val selection"
        ),
        "predictions": str(args.predictions),
        "calibrated_predictions_csv": str(args.output_predictions_csv) if args.output_predictions_csv else None,
        "rows": counts,
        "calibrator": {
            "model": str(args.model),
            "features": payload.get("features"),
            "C": payload.get("C"),
            "class_weight": payload.get("class_weight"),
            "blend_model_weight": blend_weight,
            "threshold": threshold,
        },
        "baseline": baseline_metrics,
        "calibrator_metrics": calibrator_metrics,
        "slices": _slice_metrics(
            rows=kept_rows,
            labels=labels,
            baseline_scores=baseline_probs,
            calibrator_scores=blended_probs,
            baseline_threshold=args.baseline_threshold,
            calibrator_threshold=threshold,
        ),
        "delta_f1_vs_baseline": calibrator_metrics["f1"] - baseline_metrics["f1"],
        "delta_errors_vs_baseline": calibrator_metrics["errors"] - baseline_metrics["errors"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
