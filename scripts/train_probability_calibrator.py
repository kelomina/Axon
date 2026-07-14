#!/usr/bin/env python3
"""Train a lightweight probability calibrator from exported predictions."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


def _probability_features(prob: float) -> np.ndarray:
    return np.array(
        [
            prob,
            prob * prob,
            np.log(max(prob, 1e-6)),
            np.log(max(1.0 - prob, 1e-6)),
        ],
        dtype=np.float32,
    )


def _feature_names(*, stat_dim: int, pe_dim: int) -> list[str]:
    return (
        ["prob_malicious", "prob_malicious_squared", "log_prob_malicious", "log_prob_benign"]
        + [f"stat_{idx}" for idx in range(stat_dim)]
        + [f"pe_{idx}" for idx in range(pe_dim)]
    )


def _feature_names_hash(feature_names: list[str]) -> str:
    payload = json.dumps(feature_names, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_prediction_row(row: dict, predictions_path: Path, *, expected_split: str | None = None) -> tuple[str, int, Path, float]:
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    if not _is_valid_source_sha256(source_sha):
        raise ValueError(f"Prediction row has invalid source_sha256 in {predictions_path}: {source_sha!r}")
    if expected_split is not None:
        split_value = str(row.get("split") or "").strip().casefold()
        if split_value != expected_split:
            raise ValueError(
                f"Prediction row split mismatch in {predictions_path}: "
                f"expected {expected_split!r}, got {split_value!r}"
            )
    label = int(row["label"])
    if label not in {0, 1}:
        raise ValueError(f"Prediction row has invalid label in {predictions_path}: {row.get('label')!r}")
    return source_sha, label, Path(row["cache_path"]), float(row["prob_malicious"])


def _load_prediction_features(
    predictions_path: Path,
    *,
    allow_missing_cache: bool = False,
    expected_split: str | None = None,
    return_source_sha256s: bool = False,
):
    total = 0
    kept = 0
    skipped_missing_cache = 0
    missing_cache_examples = []
    feature_dim = None
    stat_dim = None
    pe_dim = None

    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            source_sha, label, cache_path, prob = _validate_prediction_row(
                row,
                predictions_path,
                expected_split=expected_split,
            )
            if not cache_path.exists():
                skipped_missing_cache += 1
                if len(missing_cache_examples) < 10:
                    missing_cache_examples.append(str(cache_path))
                continue
            kept += 1
            if feature_dim is None:
                pe_features, stat_features = _load_cache_features(
                    cache_path,
                    expected_label=label,
                    expected_source_sha256=source_sha,
                )
                stat_dim = int(stat_features.shape[0])
                pe_dim = int(pe_features.shape[0])
                feature_dim = int(_probability_features(prob).shape[0] + stat_dim + pe_dim)

    if skipped_missing_cache and not allow_missing_cache:
        raise FileNotFoundError(
            "Prediction CSV references missing cache files: "
            f"{skipped_missing_cache}/{total} missing. "
            "Regenerate predictions/cache before training this calibrator, "
            "or pass --allow-missing-cache only for diagnostic subset runs. "
            f"Examples: {missing_cache_examples}"
        )
    if kept <= 0 or feature_dim is None or stat_dim is None or pe_dim is None:
        raise ValueError(f"No usable prediction rows with cache features were loaded from {predictions_path}")

    features = np.empty((kept, feature_dim), dtype=np.float32)
    labels = np.empty(kept, dtype=np.int64)
    probabilities = np.empty(kept, dtype=np.float32)
    source_sha256s = set()
    fill_idx = 0
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_sha, label, cache_path, prob = _validate_prediction_row(
                row,
                predictions_path,
                expected_split=expected_split,
            )
            if not cache_path.exists():
                continue
            pe_features, stat_features = _load_cache_features(
                cache_path,
                expected_label=label,
                expected_source_sha256=source_sha,
            )
            vector = np.concatenate([_probability_features(prob), stat_features, pe_features])
            if vector.shape[0] != feature_dim:
                raise ValueError(
                    f"Feature dimension changed while reading {predictions_path}: "
                    f"expected {feature_dim}, got {vector.shape[0]}"
                )
            features[fill_idx] = vector
            labels[fill_idx] = label
            probabilities[fill_idx] = prob
            source_sha256s.add(source_sha)
            fill_idx += 1

    counts = {
        "total": total,
        "kept": int(fill_idx),
        "skipped_missing_cache": skipped_missing_cache,
        "missing_cache_examples": missing_cache_examples,
        "stat_feature_dim": int(stat_dim),
        "pe_feature_dim": int(pe_dim),
    }
    result = (features[:fill_idx], labels[:fill_idx], probabilities[:fill_idx], counts)
    if return_source_sha256s:
        return (*result, source_sha256s)
    return result


def _threshold_metrics(scores: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    auc = float(roc_auc_score(labels, scores))
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
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
        )
    return rows


def _parse_thresholds(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a train-split-only logistic probability calibrator.")
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--thresholds", default=",".join(f"{value:.3f}" for value in np.arange(0.20, 0.801, 0.005)))
    parser.add_argument("--candidates", default="0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--blend-weights", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--allow-missing-cache",
        action="store_true",
        help="Allow diagnostic subset training when some prediction cache files are missing.",
    )
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds)
    c_values = _parse_thresholds(args.candidates)
    blend_weights = _parse_thresholds(args.blend_weights)

    train_x, train_y, train_probs, train_counts, train_source_sha256s = _load_prediction_features(
        args.train_predictions,
        allow_missing_cache=args.allow_missing_cache,
        expected_split="train",
        return_source_sha256s=True,
    )
    val_x, val_y, val_probs, val_counts, val_source_sha256s = _load_prediction_features(
        args.val_predictions,
        allow_missing_cache=args.allow_missing_cache,
        expected_split="val",
        return_source_sha256s=True,
    )
    overlap = sorted(train_source_sha256s & val_source_sha256s)
    if overlap:
        raise ValueError(
            "Train/val prediction CSVs overlap by source_sha256; "
            f"overlap_count={len(overlap)}, examples={overlap[:5]}"
        )
    if int(train_x.shape[1]) != int(val_x.shape[1]):
        raise ValueError(f"Train/val feature dimension mismatch: {train_x.shape[1]} != {val_x.shape[1]}")

    baseline_rows = _threshold_metrics(val_probs, val_y, thresholds)
    baseline_best = max(baseline_rows, key=lambda row: row["f1"])

    results = []
    best_model = None
    best_result = None
    for c_value in c_values:
        for class_weight in [None, "balanced"]:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=5000,
                    class_weight=class_weight,
                    C=c_value,
                    solver="liblinear",
                ),
            )
            model.fit(train_x, train_y)
            train_scores = model.predict_proba(train_x)[:, 1]
            val_scores = model.predict_proba(val_x)[:, 1]
            model_selected = False

            for blend_weight in blend_weights:
                blended_train = blend_weight * train_scores + (1.0 - blend_weight) * train_probs
                blended_val = blend_weight * val_scores + (1.0 - blend_weight) * val_probs
                train_best = max(_threshold_metrics(blended_train, train_y, thresholds), key=lambda row: row["f1"])
                val_best = max(_threshold_metrics(blended_val, val_y, thresholds), key=lambda row: row["f1"])
                result = {
                    "C": c_value,
                    "class_weight": class_weight,
                    "blend_model_weight": blend_weight,
                    "train_best": train_best,
                    "val_best": val_best,
                    "delta_val_f1_vs_baseline": val_best["f1"] - baseline_best["f1"],
                }
                results.append(result)
                if best_result is None or val_best["f1"] > best_result["val_best"]["f1"]:
                    previous_best_model = best_model
                    best_model = model
                    best_result = result
                    model_selected = True
                    if previous_best_model is not None and previous_best_model is not model:
                        del previous_best_model
            del train_scores, val_scores
            if not model_selected:
                del model

    results.sort(key=lambda row: row["val_best"]["f1"], reverse=True)
    if best_model is None or best_result is None:
        raise RuntimeError("No calibrator candidate was selected")
    feature_names = _feature_names(
        stat_dim=int(train_counts["stat_feature_dim"]),
        pe_dim=int(train_counts["pe_feature_dim"]),
    )
    feature_names_hash = _feature_names_hash(feature_names)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("wb") as handle:
        pickle.dump(
            {
                "schema": "axon_probability_calibrator_v1",
                "model": best_model,
                "blend_model_weight": best_result["blend_model_weight"],
                "C": best_result["C"],
                "class_weight": best_result["class_weight"],
                "features": "probability+stat_features+pe_features only",
                "feature_count": int(train_x.shape[1]),
                "feature_names_hash": feature_names_hash,
                "stat_feature_dim": int(train_counts["stat_feature_dim"]),
                "pe_feature_dim": int(train_counts["pe_feature_dim"]),
                "train_source_sha256_count": len(train_source_sha256s),
                "val_source_sha256_count": len(val_source_sha256s),
                "val_selected_threshold": best_result["val_best"]["threshold"],
            },
            handle,
        )

    report = {
        "protocol": (
            "train split trains calibrator; val split selects model and threshold; "
            "source_sha256 validates cache identity only; no path/name/directory/extension metadata; no test used"
        ),
        "train_rows": train_counts,
        "val_rows": val_counts,
        "feature_count": int(train_x.shape[1]),
        "feature_names_hash": feature_names_hash,
        "train_val_source_sha256_overlap_count": 0,
        "baseline_val_best": baseline_best,
        "selected": best_result,
        "top_results": results[:30],
        "output_model": str(args.output_model),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report["selected"], indent=2))
    print(f"Model saved to: {args.output_model}")
    print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
