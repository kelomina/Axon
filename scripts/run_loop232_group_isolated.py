"""Loop232: Group-Isolated OOD Generalization Assessment.

Loads group_isolated/split.csv and tests pure ML models (GBDT, ExtraTrees, Neural)
on UNSEEN software families/signatures to assess true zero-day/OOD generalization.
"""

from __future__ import annotations

import csv
import glob
import json
import time
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

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


def run_loop232_group_isolated():
    print("=" * 70)
    print("Axon v2.6 - Loop232 Group-Isolated OOD Generalization Assessment")
    print("=" * 70)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"
    split_csv = proj_dir / "models" / "generalization_group_isolated" / "split.csv"

    if not split_csv.is_file():
        print(f"[ERROR] Split CSV missing: {split_csv}")
        return

    # 1. Load split definitions
    rows_by_split, all_shas = load_group_isolated_split(split_csv)
    target_shas = set(all_shas)

    print(f"[Split] Train: {len(rows_by_split['train'])} | Val: {len(rows_by_split['val'])} | Test: {len(rows_by_split['test'])}")
    print(f"[Target SHAs] Unique: {len(target_shas)}")

    # 2. Fast scan cache to load features
    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Scanning {len(npz_files)} files for features...")

    sha_to_feat = {}
    t0 = time.time()
    for i, f in enumerate(npz_files):
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha in target_shas:
                pe = d["pe_features"].astype(np.float32)
                stat = d["stat_features"].astype(np.float32)
                lw = d["lightweight_features"].astype(np.float32)
                feat = np.concatenate([pe, stat, lw])
                sha_to_feat[sha] = (feat, int(d["label"]))
        except Exception:
            continue

    print(f"[Cache] Loaded {len(sha_to_feat)}/{len(target_shas)} samples in {time.time() - t0:.1f}s")

    # 3. Build Matrices for Train, Val, Test
    def build_matrix(rows):
        X_list, y_list = [], []
        for r in rows:
            sha = r["source_sha256"].strip().lower()
            if sha in sha_to_feat:
                feat, label = sha_to_feat[sha]
                X_list.append(feat)
                y_list.append(label)
        return np.stack(X_list), np.array(y_list, dtype=np.int64)

    X_train, y_train = build_matrix(rows_by_split["train"])
    X_val, y_val = build_matrix(rows_by_split["val"])
    X_test, y_test = build_matrix(rows_by_split["test"])

    print(f"\nMatrix Shapes:")
    print(f"  Train: {X_train.shape} (Pos: {(y_train==1).sum()}, Neg: {(y_train==0).sum()})")
    print(f"  Val:   {X_val.shape} (Pos: {(y_val==1).sum()}, Neg: {(y_val==0).sum()})")
    print(f"  Test:  {X_test.shape} (Pos: {(y_test==1).sum()}, Neg: {(y_test==0).sum()})")

    # 4. Train Models on Group-Isolated Train and Benchmark on UNSEEN Test
    models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"[Training] {name} on Group-Isolated Train...")
        t_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t_start

        # Evaluate on Val & UNSEEN Test
        def eval_set(X_data, y_data):
            probs = model.predict_proba(X_data)[:, 1]
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_data, preds)
            prec = precision_score(y_data, preds, zero_division=0)
            rec = recall_score(y_data, preds, zero_division=0)
            f1 = f1_score(y_data, preds, zero_division=0)
            auc = roc_auc_score(y_data, probs)
            errs = (preds != y_data).sum()
            return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "errors": int(errs), "total": len(y_data)}

        val_metrics = eval_set(X_val, y_val)
        test_metrics = eval_set(X_test, y_test)

        print(f"[{name}] Fit Time: {fit_time:.1f}s")
        print(f"[{name}] Val  F1: {val_metrics['f1']:.6f} | Acc: {val_metrics['acc']:.4f} | Errors: {val_metrics['errors']}")
        print(f"[{name}] TEST UNSEEN F1: {test_metrics['f1']:.6f} | Acc: {test_metrics['acc']:.4f} | Errors: {test_metrics['errors']:,}/{test_metrics['total']:,}")

        results[name] = {"val": val_metrics, "test": test_metrics}

    # Save receipt
    receipt = {
        "schema": "axon_loop232_group_isolated_receipt_v1",
        "protocol": "group_isolated_unseen_families",
        "sample_counts": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "results": results
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop232_group_isolated_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved receipt to {report_path}")


if __name__ == "__main__":
    run_loop232_group_isolated()
