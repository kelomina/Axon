#!/usr/bin/env python3
"""One frozen Train-to-Val evaluation of Authenticode-enhanced calibration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/roadmap_9997/loop169_train_authenticode/train_to_val_calibrator.json"
TRAIN_BASE = ROOT / "reports/phase3_loop133/loop130_content_string_guard_r5_train_predictions.csv"
TRAIN_CANDIDATE = ROOT / "reports/phase3_loop134/oof_fixed_v2_string_noise_valonly/stage2_oof_stacker_train_oof_predictions.csv"
VAL = ROOT / "reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_valonly/loop135_pairwise_selector_val_predictions.csv"
TRAIN_AUTH = ROOT / "reports/roadmap_9997/loop169_train_authenticode/train_authenticode.csv"
VAL_AUTH = ROOT / "reports/roadmap_9997/loop169_train_authenticode/val_authenticode.csv"
TERMS = ("Microsoft Corporation", "Microsoft Windows", "Seagate Technology", "FinalWire", "NetEase", "Beijing Sogou", "Beijing Kingsoft", "Beijing Qihu", "Wondershare", "IObit", "Yozosoft", "Huya")
STATUSES = ("Valid", "NotSigned", "HashMismatch", "UnknownError", "CollectionError")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def keyed(path: Path) -> dict[str, dict[str, str]]:
    result = {str(row["source_sha256"]).casefold(): row for row in rows(path)}
    if len(result) != 20_000:
        raise ValueError(f"{path} denominator or SHA uniqueness drifted")
    return result


def external(row: dict[str, str]) -> list[float]:
    status = str(row.get("auth_status") or "CollectionError")
    subject = str(row.get("signer_subject") or "").casefold()
    return [float(status == item) for item in STATUSES] + [float(term.casefold() in subject) for term in TERMS]


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int | float]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"f1": float(f1_score(labels, predictions, zero_division=0)), "precision": float(precision_score(labels, predictions, zero_division=0)), "recall": float(recall_score(labels, predictions, zero_division=0)), "true_positive": int(tp), "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "errors": int(fp + fn)}


def main() -> None:
    train_base, train_candidate = keyed(TRAIN_BASE), keyed(TRAIN_CANDIDATE)
    val = keyed(VAL)
    train_auth, val_auth = keyed(TRAIN_AUTH), keyed(VAL_AUTH)
    if set(train_base) != set(train_candidate) or set(train_base) != set(train_auth):
        raise ValueError("Train alignment drifted")
    if set(val) != set(val_auth):
        raise ValueError("Val alignment drifted")
    train_keys, val_keys = sorted(train_base), sorted(val)
    train_x = np.asarray([[float(train_base[key]["stage2_prob_malicious"]), float(train_candidate[key]["stage2_prob_malicious"]), *external(train_auth[key])] for key in train_keys], dtype=np.float64)
    train_y = np.asarray([int(train_base[key]["label"]) for key in train_keys], dtype=np.int8)
    val_x = np.asarray([[float(val[key]["baseline_prob_malicious"]), float(val[key]["candidate_prob_malicious"]), *external(val_auth[key])] for key in val_keys], dtype=np.float64)
    val_y = np.asarray([int(val[key]["label"]) for key in val_keys], dtype=np.int8)
    model = LogisticRegression(C=0.1, max_iter=1000, random_state=169, solver="lbfgs")
    model.fit(train_x, train_y)
    calibrated = (model.predict_proba(val_x)[:, 1] >= 0.5).astype(np.int8)
    baseline = np.asarray([int(val[key]["prediction"]) for key in val_keys], dtype=np.int8)
    report = {"schema": "axon_loop169_frozen_train_to_val_calibrator_v1", "scope": {"train_access": True, "val_access": True, "test10k_access": False, "full_test_access": False, "threshold_selection": False, "hyperparameter_search": False}, "train_rows": int(train_y.size), "val_rows": int(val_y.size), "features": ["loop130_score", "loop134_score", "auth_status_onehot", "frozen_trusted_signer_terms"], "recipe": {"estimator": "LogisticRegression", "C": 0.1, "threshold": 0.5, "seed": 169}, "baseline": metric(val_y, baseline), "candidate": metric(val_y, calibrated)}
    report["net_error_reduction"] = report["baseline"]["errors"] - report["candidate"]["errors"]
    REPORT.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
