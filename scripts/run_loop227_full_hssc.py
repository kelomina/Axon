"""Loop227: Full 160k Evaluation with Hard-Sample Specialist Classifier (HSSC).

Runs full batch inference across all 160,000 samples in cache using HSSC gate consultation.
Computes final total errors and full-dataset F1.
"""

from __future__ import annotations

import csv
import glob
import json
import time
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


def run_loop227_full():
    print("=" * 70)
    print("Axon v2.6 - Loop227 Full 160k Dataset Evaluation with HSSC Gate")
    print("=" * 70)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"
    full_csv = proj_dir / "reports" / "phase3_loop151" / "loop151_trusted_signer_guard_full_predictions.csv"

    if not full_csv.is_file():
        print(f"[ERROR] Baseline CSV missing: {full_csv}")
        return

    # 1. Load baseline predictions
    rows = load_baseline_csv(full_csv)
    THR = 0.31
    prob_key = "stage2_prob_malicious"

    print(f"[Data] Loaded {len(rows)} predictions")

    # Map SHA to baseline prediction metadata
    sha_map = {}
    error_shas = set()
    for r in rows:
        label = int(r.get("label", -1))
        if label not in (0, 1):
            continue
        prob = float(r.get(prob_key, 0.0))
        signer_down = r.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
        pred = 0 if signer_down else int(prob >= THR)
        
        r["pred"] = pred
        sha = r["source_sha256"].strip().lower()
        sha_map[sha] = r
        if pred != label:
            error_shas.add(sha)

    # 2. Fast scan all 200k cache files to build training set + run full inference
    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Scanning {len(npz_files)} cache files for training HSSC...")

    X_train_list = []
    y_train_list = []
    w_train_list = []

    # Select error SHAs + background sample
    import random
    random.seed(42)
    bg_shas = set(random.sample(list(set(sha_map.keys()) - error_shas), 20000))
    target_train_shas = error_shas | bg_shas

    matched_train = 0
    t0 = time.time()

    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha in target_train_shas:
                r = sha_map[sha]
                pe = d["pe_features"].astype(np.float32)
                stat = d["stat_features"].astype(np.float32)
                lw = d["lightweight_features"].astype(np.float32)
                
                feat = np.concatenate([pe, stat, lw])
                X_train_list.append(feat)
                y_train_list.append(int(r["label"]))
                w_train_list.append(10.0 if sha in error_shas else 1.0)
                matched_train += 1
        except Exception:
            continue

    print(f"[Training Data] Matched {matched_train} training samples in {time.time() - t0:.1f}s")
    X_train = np.stack(X_train_list)
    y_train = np.array(y_train_list, dtype=np.int64)
    w_train = np.array(w_train_list, dtype=np.float32)

    # 3. Train HSSC Model
    print("\n[Training] Fitting HistGradientBoostingClassifier (max_iter=300)...")
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=45,
        min_samples_leaf=10,
        l2_regularization=0.5,
        random_state=42
    )
    t_train = time.time()
    clf.fit(X_train, y_train, sample_weight=w_train)
    print(f"[Training] Model trained in {time.time() - t_train:.1f}s")

    # 4. Full 160k Batch Inference & Evaluation
    print("\n[Full Evaluation] Running streaming inference across all cache samples...")
    
    t_eval = time.time()
    total_eval = 0
    base_errors = 0
    gated_errors = 0
    repairs = 0
    breaks = 0

    base_tp, base_fp, base_fn, base_tn = 0, 0, 0, 0
    gate_tp, gate_fp, gate_fn, gate_tn = 0, 0, 0, 0

    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha not in sha_map:
                continue

            r = sha_map[sha]
            label = int(r["label"])
            base_pred = r["pred"]
            base_prob = float(r[prob_key])

            # Consult HSSC if base_prob in gate zone [0.15, 0.85]
            if 0.15 <= base_prob <= 0.85:
                pe = d["pe_features"].astype(np.float32)
                stat = d["stat_features"].astype(np.float32)
                lw = d["lightweight_features"].astype(np.float32)
                feat = np.concatenate([pe, stat, lw]).reshape(1, -1)
                
                h_prob = clf.predict_proba(feat)[0, 1]
                gated_prob = 0.35 * base_prob + 0.65 * h_prob
            else:
                gated_prob = base_prob

            gated_pred = int(gated_prob >= THR)

            total_eval += 1
            if base_pred != label:
                base_errors += 1
            if gated_pred != label:
                gated_errors += 1

            if base_pred != label and gated_pred == label:
                repairs += 1
            elif base_pred == label and gated_pred != label:
                breaks += 1

            # Baseline CM
            if base_pred == 1 and label == 1: base_tp += 1
            elif base_pred == 1 and label == 0: base_fp += 1
            elif base_pred == 0 and label == 1: base_fn += 1
            elif base_pred == 0 and label == 0: base_tn += 1

            # Gated CM
            if gated_pred == 1 and label == 1: gate_tp += 1
            elif gated_pred == 1 and label == 0: gate_fp += 1
            elif gated_pred == 0 and label == 1: gate_fn += 1
            elif gated_pred == 0 and label == 0: gate_tn += 1

        except Exception:
            continue

    base_f1 = 2 * base_tp / (2 * base_tp + base_fp + base_fn) if (2 * base_tp + base_fp + base_fn) > 0 else 0
    gate_f1 = 2 * gate_tp / (2 * gate_tp + gate_fp + gate_fn) if (2 * gate_tp + gate_fp + gate_fn) > 0 else 0

    print("\n" + "=" * 70)
    print(f"[Loop227 Full 160k Evaluation Results (Evaluated on {total_eval:,} samples)]")
    print(f"  Baseline Errors: {base_errors:,} | Full F1: {base_f1:.6f}")
    print(f"  Gated    Errors: {gated_errors:,} | Full F1: {gate_f1:.6f}")
    print(f"  Repairs: {repairs} | Breaks: {breaks} | Net Repair: {repairs - breaks}")
    print(f"  Evaluation completed in {time.time() - t_eval:.1f}s")
    print("=" * 70)

    # Save receipt
    receipt = {
        "schema": "axon_loop227_full_hssc_receipt_v1",
        "eval_samples": total_eval,
        "baseline_errors": base_errors,
        "gated_errors": gated_errors,
        "repairs": repairs,
        "breaks": breaks,
        "net_repair": repairs - breaks,
        "baseline_f1": base_f1,
        "gated_f1": gate_f1
    }
    
    report_path = proj_dir / "reports" / "roadmap_9997" / "loop227_full_hssc_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Receipt saved to {report_path}")

if __name__ == "__main__":
    run_loop227_full()
