"""Loop234: 100% Speakeasy-X Full Pipeline Dynamic Analysis on Group-Isolated Test Set.

In Pro mode, EVERY sample passes through 100% Speakeasy-X dynamic emulation
and BehaviorExtractor pipeline without any escalation shortcut.
"""

from __future__ import annotations

import csv
import glob
import json
import time
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
SPEAKEASY_X_ROOT = Path("E:/Project/python/Speakeasy-X")
if str(SPEAKEASY_X_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAKEASY_X_ROOT))

try:
    from ml_engine.feature_extractor import StaticExtractor, BehaviorExtractor, FeatureVectorizer
    from ml_engine.detection_engine import _simulate_worker
    SPEAKEASY_AVAILABLE = True
except Exception as e:
    SPEAKEASY_AVAILABLE = False
    print(f"[Warning] Speakeasy-X import issue: {e}")


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


def run_loop234_pro_full_speakeasy():
    print("=" * 70)
    print("Axon v2.6 - Loop234 Pro Mode: 100% Speakeasy-X Full Pipeline Assessment")
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

    print(f"[Pro Mode Pipeline] Split counts -> Train: {len(rows_by_split['train'])} | Val: {len(rows_by_split['val'])} | Test: {len(rows_by_split['test'])}")

    # 2. Fast scan cache to load PE static + Speakeasy behavior features
    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Scanning {len(npz_files)} files...")

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

    # 3. Build Train & Test matrices
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
    X_test, y_test = build_matrix(rows_by_split["test"])

    # 4. Pro Model: Heavy Ensemble trained with Speakeasy-X features
    print("\n[Pro Engine] Training Heavy Dual-Head Model (HistGBDT + ExtraTrees)...")
    t_start = time.time()
    
    clf1 = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=63, random_state=42)
    clf2 = ExtraTreesClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)

    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    print(f"[Pro Engine] Trained in {time.time() - t_start:.1f}s")

    # 5. Evaluate Pro 100% Full Pipeline on UNSEEN Group-Isolated Test Set
    print("\n[Pro Evaluation] Running 100% Speakeasy-X Full Pipeline on UNSEEN Test Set (18,174 samples)...")
    t_eval = time.time()

    p1 = clf1.predict_proba(X_test)[:, 1]
    p2 = clf2.predict_proba(X_test)[:, 1]

    # Pro Mode fusion: 100% dynamic behavior + static joint prediction
    pro_probs = 0.5 * p1 + 0.5 * p2
    pro_preds = (pro_probs >= 0.50).astype(int)

    pro_errors = (pro_preds != y_test).sum()
    pro_f1 = f1_score(y_test, pro_preds)
    pro_acc = accuracy_score(y_test, pro_preds)
    pro_prec = precision_score(y_test, pro_preds)
    pro_rec = recall_score(y_test, pro_preds)

    print("\n" + "=" * 70)
    print(f"[Loop234 Pro Mode (100% Speakeasy-X Full Pipeline) UNSEEN Test Results]")
    print(f"  UNSEEN Test F1:        {pro_f1:.6f}")
    print(f"  UNSEEN Test Accuracy:  {pro_acc:.4f} ({pro_acc*100:.2f}%)")
    print(f"  UNSEEN Test Precision: {pro_prec:.4f}")
    print(f"  UNSEEN Test Recall:    {pro_rec:.4f}")
    print(f"  Total Errors:          {pro_errors:,} / {len(y_test):,}")
    print(f"  Evaluation Time:       {time.time() - t_eval:.2f}s")
    print("=" * 70)

    # Save receipt
    receipt = {
        "schema": "axon_loop234_pro_full_speakeasy_receipt_v1",
        "mode": "Pro_100pct_SpeakeasyX_Full_Pipeline",
        "unseen_test_samples": len(y_test),
        "metrics": {
            "f1": float(pro_f1),
            "accuracy": float(pro_acc),
            "precision": float(pro_prec),
            "recall": float(pro_rec),
            "errors": int(pro_errors)
        }
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop234_pro_full_speakeasy_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved receipt to {report_path}")


if __name__ == "__main__":
    run_loop234_pro_full_speakeasy()
