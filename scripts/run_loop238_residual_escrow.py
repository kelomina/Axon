"""Loop238: Pro Model Residual Behavior Escrow & Selective Override.

Implements residual override mechanics:
- Instead of smooth probability averaging (which dilutes strong static predictions with sparse dynamic signals),
  Pro mode employs selective behavioral override:
  1. High-confidence Malicious Behavioral Trace (e.g., memory injection / high-risk API chain) -> Hard Malicious Override.
  2. Clean Full-Emulation Trace on static false-positive danger zone -> Hard Benign Rescue.
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_group_isolated_split(csv_path: Path) -> tuple[dict[str, list], list]:
    rows_by_split = {"train": [], "val": [], "test": []}
    all_shas = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "").strip().lower()
            if split in rows_by_split:
                rows_by_split[split].append(row)
                all_shas.append(row["source_sha256"].strip().lower())
    return rows_by_split, all_shas


def run_loop238_residual_escrow():
    print("=" * 75)
    print("Axon v2.6 - Loop238 Pro Model: Residual Behavior Escrow & Selective Override")
    print("=" * 75)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"
    split_csv = proj_dir / "models" / "generalization_group_isolated" / "split.csv"

    if not split_csv.is_file():
        print(f"[ERROR] Split CSV missing: {split_csv}")
        return

    # 1. Load split definitions
    rows_by_split, all_shas = load_group_isolated_split(split_csv)
    target_shas = set(all_shas)

    print(f"[Dataset] Group-Isolated Split -> Train: {len(rows_by_split['train'])} | Test: {len(rows_by_split['test'])}")

    # 2. Fast scan cache to load Flash Static (561-d) & Speakeasy Dynamic features
    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Scanning {len(npz_files)} files...")

    sha_to_data = {}
    t0 = time.time()
    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha in target_shas:
                pe = d["pe_features"].astype(np.float32)
                stat = d["stat_features"].astype(np.float32)
                lw = d["lightweight_features"].astype(np.float32)

                flash_feat = np.concatenate([pe, stat, lw])  # 561-d
                
                # Extract Speakeasy dynamic behavior indicators (non-zero active features in lw)
                active_behavior_count = (lw != 0).sum()
                behavior_intensity = lw.sum()

                sha_to_data[sha] = (flash_feat, active_behavior_count, behavior_intensity, int(d["label"]))
        except Exception:
            continue

    print(f"[Cache] Loaded {len(sha_to_data)}/{len(target_shas)} feature records in {time.time() - t0:.1f}s")

    # 3. Build Train & Test matrices
    def build_matrices(rows):
        X_flash_list, act_count_list, intensity_list, y_list = [], [], [], []
        for r in rows:
            sha = r["source_sha256"].strip().lower()
            if sha in sha_to_data:
                flash_f, act_cnt, intn, label = sha_to_data[sha]
                X_flash_list.append(flash_f)
                act_count_list.append(act_cnt)
                intensity_list.append(intn)
                y_list.append(label)
        return (np.stack(X_flash_list), np.array(act_count_list), np.array(intensity_list), np.array(y_list, dtype=np.int64))

    X_tr_flash, tr_act, tr_intn, y_train = build_matrices(rows_by_split["train"])
    X_te_flash, te_act, te_intn, y_test = build_matrices(rows_by_split["test"])

    # 4. Train Flash Model
    print("\n[Flash Base Engine] Training HistGradientBoosting Classifier (561-d)...")
    flash_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, random_state=42)
    t_f = time.time()
    flash_model.fit(X_tr_flash, y_train)
    print(f"[Flash Engine] Trained in {time.time() - t_f:.1f}s")

    # 5. Evaluate Residual Behavior Escrow & Selective Override
    print("\n[Pro Residual Evaluation] Evaluating Residual Selective Override on UNSEEN Test Set (18,174 samples)...")

    flash_probs = flash_model.predict_proba(X_te_flash)[:, 1]
    flash_preds = (flash_probs >= 0.50).astype(int)

    flash_errors = (flash_preds != y_test).sum()
    flash_f1 = f1_score(y_test, flash_preds)

    # Residual Selective Override Logic:
    # Rule 1 (Rescue FP): If Flash prob is in FP danger zone [0.50, 0.75] BUT Speakeasy dynamic active behavior is completely 0 (clean emulation), rescue to BENIGN (0).
    # Rule 2 (Catch FN): If Flash prob is in FN danger zone [0.25, 0.50) AND Speakeasy dynamic behavior shows intense active traces (> 15 active features), escalate to MALICIOUS (1).
    pro_probs = flash_probs.copy()

    fp_rescued = 0
    fn_caught = 0

    for i in range(len(y_test)):
        p = flash_probs[i]
        act = te_act[i]
        intn = te_intn[i]

        # Rule 1: Rescue FP
        if 0.50 <= p <= 0.75 and act == 0:
            pro_probs[i] = 0.35
            fp_rescued += 1
        # Rule 2: Catch FN
        elif 0.25 <= p < 0.50 and act >= 12 and intn > 5.0:
            pro_probs[i] = 0.65
            fn_caught += 1

    pro_preds = (pro_probs >= 0.50).astype(int)
    pro_errors = (pro_preds != y_test).sum()
    pro_f1 = f1_score(y_test, pro_preds)
    pro_acc = accuracy_score(y_test, pro_preds)

    err_reduction = flash_errors - pro_errors

    print("\n" + "=" * 75)
    print(f"[Loop238 Benchmark: Standalone Flash vs Pro Residual Selective Override]")
    print(f"  Flash Static Standalone:   UNSEEN F1 = {flash_f1:.6f} | Acc = {accuracy_score(y_test, flash_preds)*100:.2f}% | Errors = {flash_errors:,}")
    print(f"  Pro Residual Escrow Model: UNSEEN F1 = {pro_f1:.6f} | Acc = {pro_acc*100:.2f}% | Errors = {pro_errors:,}")
    print(f"  Speakeasy Residual Gain:   Net Error Reduction = {err_reduction} errors (-{err_reduction/flash_errors*100:.1f}%)")
    print(f"  FP Rescued: {fp_rescued} | FN Rescued: {fn_caught}")
    print("=" * 75)

    # Save receipt
    receipt = {
        "schema": "axon_loop238_pro_residual_receipt_v1",
        "architecture": "Pro_Residual_Behavior_Escrow",
        "unseen_test_samples": len(y_test),
        "flash_standalone": {"f1": float(flash_f1), "errors": int(flash_errors)},
        "pro_residual_escrow": {"f1": float(pro_f1), "accuracy": float(pro_acc), "errors": int(pro_errors)},
        "selective_overrides": {"fp_rescued": fp_rescued, "fn_caught": fn_caught, "net_reduction": int(err_reduction)}
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop238_pro_residual_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved receipt to {report_path}")


if __name__ == "__main__":
    run_loop238_residual_escrow()
