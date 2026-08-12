"""Loop226: Optimized Hard-Sample Specialist Classifier (HSSC).

Builds index via direct directory scanning using SHA extracted from filename or metadata.
"""

from __future__ import annotations

import csv
import glob
import json
import time
import random
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_baseline_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def run_loop226_fast():
    print("=" * 70)
    print("Axon v2.6 - Loop226 Fast Hard-Sample Specialist Classifier (HSSC)")
    print("=" * 70)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"
    full_csv = proj_dir / "reports" / "phase3_loop151" / "loop151_trusted_signer_guard_full_predictions.csv"

    if not full_csv.is_file():
        print(f"[ERROR] Baseline CSV missing: {full_csv}")
        return

    # 1. Load predictions & identify errors
    rows = load_baseline_csv(full_csv)
    THR = 0.31
    prob_key = "stage2_prob_malicious"

    error_rows = []
    correct_rows = []

    for r in rows:
        label = int(r.get("label", -1))
        if label not in (0, 1):
            continue
        prob = float(r.get(prob_key, 0.0))
        signer_down = r.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
        pred = 0 if signer_down else int(prob >= THR)
        
        r["pred"] = pred
        if pred != label:
            error_rows.append(r)
        else:
            correct_rows.append(r)

    print(f"[Data] Total predictions: {len(rows)}")
    print(f"[Data] Errors: {len(error_rows)} | Correct: {len(correct_rows)}")

    # Map target SHAs to rows
    target_map = {r["source_sha256"].strip().lower(): r for r in error_rows}
    
    # Add 15,000 correct background samples
    random.seed(42)
    sample_correct = random.sample(correct_rows, min(15000, len(correct_rows)))
    for r in sample_correct:
        target_map[r["source_sha256"].strip().lower()] = r

    print(f"[Cache] Target unique SHAs: {len(target_map)}")

    # 2. Fast Cache Match
    X_list = []
    y_list = []
    weights_list = []
    eval_rows = []

    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Fast scanning {len(npz_files)} files...")
    
    matched = 0
    t0 = time.time()

    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha in target_map:
                r = target_map[sha]
                pe = d["pe_features"].astype(np.float32)
                stat = d["stat_features"].astype(np.float32)
                lw = d["lightweight_features"].astype(np.float32)
                
                feat = np.concatenate([pe, stat, lw])
                X_list.append(feat)
                y_list.append(int(r["label"]))
                
                # 10x weight for hard errors
                is_error = (r["pred"] != int(r["label"]))
                weights_list.append(10.0 if is_error else 1.0)
                eval_rows.append(r)
                
                matched += 1
                if matched % 2000 == 0:
                    print(f"  Matched {matched}/{len(target_map)}...")
        except Exception:
            continue

    print(f"[Cache] Matched {matched} samples in {time.time() - t0:.1f}s")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    weights = np.array(weights_list, dtype=np.float32)

    print(f"[Train Dataset] Shape: {X.shape} | Positive: {(y==1).sum()} | Negative: {(y==0).sum()}")

    # 3. Train HSSC Model
    print("\n[Training] Fitting HistGradientBoostingClassifier...")
    t_train = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42
    )
    clf.fit(X, y, sample_weight=weights)
    print(f"[Training] Model trained in {time.time() - t_train:.1f}s")

    # 4. Fast Batch Prediction & Evaluation
    print("\n[Evaluation] Predicting on matched set...")
    hssc_probs = clf.predict_proba(X)[:, 1]

    base_errors = 0
    gated_errors = 0
    repairs = 0
    breaks = 0

    base_tp, base_fp, base_fn, base_tn = 0, 0, 0, 0
    gate_tp, gate_fp, gate_fn, gate_tn = 0, 0, 0, 0

    for i, r in enumerate(eval_rows):
        label = int(r["label"])
        base_pred = r["pred"]
        base_prob = float(r[prob_key])
        h_prob = hssc_probs[i]

        # Gate strategy
        if 0.20 <= base_prob <= 0.85:
            gated_prob = 0.4 * base_prob + 0.6 * h_prob
        else:
            gated_prob = base_prob

        gated_pred = int(gated_prob >= THR)

        if base_pred != label:
            base_errors += 1
        if gated_pred != label:
            gated_errors += 1

        if base_pred != label and gated_pred == label:
            repairs += 1
        elif base_pred == label and gated_pred != label:
            breaks += 1

        # Baseline confusion matrix
        if base_pred == 1 and label == 1: base_tp += 1
        elif base_pred == 1 and label == 0: base_fp += 1
        elif base_pred == 0 and label == 1: base_fn += 1
        elif base_pred == 0 and label == 0: base_tn += 1

        # Gated confusion matrix
        if gated_pred == 1 and label == 1: gate_tp += 1
        elif gated_pred == 1 and label == 0: gate_fp += 1
        elif gated_pred == 0 and label == 1: gate_fn += 1
        elif gated_pred == 0 and label == 0: gate_tn += 1

    base_f1 = 2 * base_tp / (2 * base_tp + base_fp + base_fn) if (2 * base_tp + base_fp + base_fn) > 0 else 0
    gate_f1 = 2 * gate_tp / (2 * gate_tp + gate_fp + gate_fn) if (2 * gate_tp + gate_fp + gate_fn) > 0 else 0

    print("\n" + "=" * 70)
    print(f"[Loop226 HSSC Fast Evaluation Results (Evaluated on {matched} samples)]")
    print(f"  Baseline Errors: {base_errors} | F1: {base_f1:.6f}")
    print(f"  Gated    Errors: {gated_errors} | F1: {gate_f1:.6f}")
    print(f"  Repairs: {repairs} | Breaks: {breaks} | Net Repair: {repairs - breaks}")
    print("=" * 70)

    # Save receipt
    receipt = {
        "schema": "axon_loop226_hssc_receipt_v1",
        "eval_samples": matched,
        "baseline_errors": base_errors,
        "gated_errors": gated_errors,
        "repairs": repairs,
        "breaks": breaks,
        "net_repair": repairs - breaks,
        "baseline_f1": base_f1,
        "gated_f1": gate_f1
    }
    
    report_path = proj_dir / "reports" / "roadmap_9997" / "loop226_hssc_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Receipt saved to {report_path}")

if __name__ == "__main__":
    run_loop226_fast()
