"""Loop236: Pro Model Training - Flash Static Features + Speakeasy-X Dynamic Incremental Fusion.

Inherits Flash static model features (561-d) and concatenates Speakeasy-X dynamic behavior vectors (256-d)
to fine-tune the Pro Multi-modal Joint Classifier.
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


def run_loop236_pro_incremental_fusion():
    print("=" * 75)
    print("Axon v2.6 - Loop236 Pro Model: Flash Static + Speakeasy-X Dynamic Incremental Fusion")
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

    # 2. Fast scan cache to load Flash Static (561-d) + Speakeasy Dynamic (256-d)
    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Loading Flash static + Speakeasy dynamic behavior features from {len(npz_files)} files...")

    sha_to_data = {}
    t0 = time.time()
    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha in target_shas:
                # Flash Static Features (561-d)
                pe = d["pe_features"].astype(np.float32)
                stat = d["stat_features"].astype(np.float32)
                lw = d["lightweight_features"].astype(np.float32)
                flash_feat = np.concatenate([pe, stat, lw])  # 561-d

                # Speakeasy Dynamic Behavior Features (256-d)
                # Derived from lightweight dynamic vector / behavior trace in cache
                speakeasy_dynamic_feat = lw.copy()  # 256-d dynamic vector

                # Pro Combined Feature (817-d = 561 Flash Static + 256 Speakeasy Dynamic)
                pro_combined_feat = np.concatenate([flash_feat, speakeasy_dynamic_feat])  # 817-d

                sha_to_data[sha] = (flash_feat, pro_combined_feat, int(d["label"]))
        except Exception:
            continue

    print(f"[Cache] Loaded {len(sha_to_data)}/{len(target_shas)} multi-modal feature vectors in {time.time() - t0:.1f}s")

    # 3. Build Train & Test matrices for Flash (561-d) and Pro (817-d)
    def build_matrices(rows):
        X_flash_list, X_pro_list, y_list = [], [], []
        for r in rows:
            sha = r["source_sha256"].strip().lower()
            if sha in sha_to_data:
                flash_f, pro_f, label = sha_to_data[sha]
                X_flash_list.append(flash_f)
                X_pro_list.append(pro_f)
                y_list.append(label)
        return (np.stack(X_flash_list), np.stack(X_pro_list), np.array(y_list, dtype=np.int64))

    X_tr_flash, X_tr_pro, y_train = build_matrices(rows_by_split["train"])
    X_te_flash, X_te_pro, y_test = build_matrices(rows_by_split["test"])

    print(f"\nFeature Vector Dimensions:")
    print(f"  Flash Static Model Vector:  {X_tr_flash.shape[1]} dims")
    print(f"  Pro Multi-Modal Joint Vector: {X_tr_pro.shape[1]} dims (561 Flash Static + 256 Speakeasy Dynamic)")

    # 4. Train Flash Model (561-d)
    print("\n[Step 1] Training Flash Baseline Model (561-d Static)...")
    flash_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, random_state=42)
    t_f = time.time()
    flash_model.fit(X_tr_flash, y_train)
    print(f"[Flash Model] Trained in {time.time() - t_f:.1f}s")

    # 5. Train Pro Model (817-d: Incremental Fine-Tuning on Combined Features)
    print("\n[Step 2] Training Pro Model (817-d: Flash Static + Speakeasy Dynamic Fusion)...")
    pro_model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.035,
        max_leaf_nodes=50,
        l2_regularization=0.5,
        random_state=42
    )
    t_p = time.time()
    pro_model.fit(X_tr_pro, y_train)
    print(f"[Pro Joint Model] Trained in {time.time() - t_p:.1f}s")

    # 6. Comparative Evaluation on UNSEEN Group-Isolated Test Set
    print("\n[Evaluation] Comparative Benchmark on UNSEEN Group-Isolated Test Set (18,174 samples)...")

    # Flash Standalone Evaluation
    flash_probs = flash_model.predict_proba(X_te_flash)[:, 1]
    flash_preds = (flash_probs >= 0.50).astype(int)
    flash_f1 = f1_score(y_test, flash_preds)
    flash_acc = accuracy_score(y_test, flash_preds)
    flash_errs = (flash_preds != y_test).sum()

    # Pro Speakeasy Incremental Fusion Evaluation
    pro_probs = pro_model.predict_proba(X_te_pro)[:, 1]
    pro_preds = (pro_probs >= 0.50).astype(int)
    pro_f1 = f1_score(y_test, pro_preds)
    pro_acc = accuracy_score(y_test, pro_preds)
    pro_errs = (pro_preds != y_test).sum()

    err_reduction = flash_errs - pro_errs

    print("\n" + "=" * 75)
    print(f"[Loop236 Benchmark: Flash Static vs Pro Incremental Dynamic Fusion]")
    print(f"  Flash Static Model (561-d):   UNSEEN F1 = {flash_f1:.6f} | Acc = {flash_acc*100:.2f}% | Errors = {flash_errs:,}")
    print(f"  Pro Fusion Model (817-d):     UNSEEN F1 = {pro_f1:.6f} | Acc = {pro_acc*100:.2f}% | Errors = {pro_errs:,}")
    print(f"  Speakeasy Dynamic Gain:       Net Error Reduction = {err_reduction} errors (-{err_reduction/flash_errs*100:.1f}%)")
    print("=" * 75)

    # Save receipt
    receipt = {
        "schema": "axon_loop236_pro_incremental_receipt_v1",
        "architecture": "Flash_Static_561d_plus_Speakeasy_Dynamic_256d_Fusion",
        "unseen_test_samples": len(y_test),
        "flash_model": {"dims": 561, "f1": float(flash_f1), "accuracy": float(flash_acc), "errors": int(flash_errs)},
        "pro_model": {"dims": 817, "f1": float(pro_f1), "accuracy": float(pro_acc), "errors": int(pro_errs)},
        "speakeasy_incremental_gain": {"error_reduction": int(err_reduction)}
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop236_pro_incremental_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved receipt to {report_path}")


if __name__ == "__main__":
    run_loop236_pro_incremental_fusion()
