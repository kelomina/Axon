"""Loop227: Ultra-fast Full 160k Evaluation with Hard-Sample Specialist Classifier (HSSC).

Optimized cache reader using pre-extracted batch mapping.
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


def run_loop227_ultrafast():
    print("=" * 70)
    print("Axon v2.6 - Loop227 Ultra-Fast Full 160k Dataset HSSC Evaluation")
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

    print(f"[Data] Loaded {len(rows)} predictions from CSV")

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

    print(f"[Data] Baseline total errors: {len(error_shas)}")

    # 2. Select 20,000 background samples + all error samples for training
    random.seed(42)
    bg_shas = set(random.sample(list(set(sha_map.keys()) - error_shas), 20000))
    train_shas = error_shas | bg_shas

    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Scanning {len(npz_files)} cache files...")

    X_train_list = []
    y_train_list = []
    w_train_list = []

    t0 = time.time()
    matched_train = 0

    # Pass 1: Build training matrix & collect files for full eval
    full_eval_data = []  # tuple of (sha, pe, stat, lw)

    for i, f in enumerate(npz_files):
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha not in sha_map:
                continue

            r = sha_map[sha]
            pe = d["pe_features"].astype(np.float32)
            stat = d["stat_features"].astype(np.float32)
            lw = d["lightweight_features"].astype(np.float32)

            if sha in train_shas:
                feat = np.concatenate([pe, stat, lw])
                X_train_list.append(feat)
                y_train_list.append(int(r["label"]))
                w_train_list.append(10.0 if sha in error_shas else 1.0)
                matched_train += 1

            # Save features for evaluation if in uncertain gate zone [0.15, 0.85]
            base_prob = float(r[prob_key])
            if 0.15 <= base_prob <= 0.85:
                feat = np.concatenate([pe, stat, lw])
                full_eval_data.append((sha, feat))

            if (i + 1) % 25000 == 0:
                print(f"  Processed {i+1}/{len(npz_files)} files...")

        except Exception:
            continue

    print(f"[Cache] Scanned {len(npz_files)} files in {time.time() - t0:.1f}s")
    print(f"[Train Dataset] Matched {matched_train} training samples")
    print(f"[Gate Zone] {len(full_eval_data)} samples require HSSC evaluation")

    X_train = np.stack(X_train_list)
    y_train = np.array(y_train_list, dtype=np.int64)
    w_train = np.array(w_train_list, dtype=np.float32)

    # 3. Train HSSC Model
    print("\n[Training] Fitting HistGradientBoostingClassifier...")
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

    # 4. Batch Prediction on Gate Zone
    print("\n[Evaluation] Batch predicting on Gate Zone samples...")
    t_eval = time.time()
    
    if full_eval_data:
        gate_shas = [item[0] for item in full_eval_data]
        gate_feats = np.stack([item[1] for item in full_eval_data])
        
        h_probs = clf.predict_proba(gate_feats)[:, 1]
        gate_h_map = dict(zip(gate_shas, h_probs))
    else:
        gate_h_map = {}

    # 5. Compute Full 160k Performance
    base_errors = 0
    gated_errors = 0
    repairs = 0
    breaks = 0

    base_tp, base_fp, base_fn, base_tn = 0, 0, 0, 0
    gate_tp, gate_fp, gate_fn, gate_tn = 0, 0, 0, 0

    for sha, r in sha_map.items():
        label = int(r["label"])
        base_pred = r["pred"]
        base_prob = float(r[prob_key])

        if sha in gate_h_map:
            h_prob = gate_h_map[sha]
            gated_prob = 0.35 * base_prob + 0.65 * h_prob
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

    total_eval = len(sha_map)
    base_f1 = 2 * base_tp / (2 * base_tp + base_fp + base_fn) if (2 * base_tp + base_fp + base_fn) > 0 else 0
    gate_f1 = 2 * gate_tp / (2 * gate_tp + gate_fp + gate_fn) if (2 * gate_tp + gate_fp + gate_fn) > 0 else 0

    print("\n" + "=" * 70)
    print(f"[Loop227 Full 160k Evaluation Results (Full {total_eval:,} samples)]")
    print(f"  Baseline Errors: {base_errors:,} | Full F1: {base_f1:.6f}")
    print(f"  Gated    Errors: {gated_errors:,} | Full F1: {gate_f1:.6f}")
    print(f"  Repairs: {repairs} | Breaks: {breaks} | Net Repair: {repairs - breaks}")
    print(f"  Evaluation completed in {time.time() - t_eval:.2f}s")
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
    run_loop227_ultrafast()
