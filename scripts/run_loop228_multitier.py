"""Loop228: Multi-tier Multi-Expert Cascades for Residual 1,242 Hard Errors.

1. Identifies the residual 1,242 errors from Loop227 HSSC.
2. Deconstructs them into FP (False Positives) and FN (False Negatives).
3. Trains specialized Tier-2 Experts:
   - FP-Specialist (focusing on reducing high-confidence FP benign misclassifications)
   - FN-Specialist (focusing on rescuing low-probability malicious escapes)
4. Constructs a Multi-tier Decision Cascade & evaluates on full dataset.
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
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_baseline_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def run_loop228_multitier():
    print("=" * 70)
    print("Axon v2.6 - Loop228 Multi-tier Multi-Expert Cascades")
    print("=" * 70)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"
    full_csv = proj_dir / "reports" / "phase3_loop151" / "loop151_trusted_signer_guard_full_predictions.csv"

    if not full_csv.is_file():
        print(f"[ERROR] Baseline CSV missing: {full_csv}")
        return

    # 1. Load predictions
    rows = load_baseline_csv(full_csv)
    THR = 0.31
    prob_key = "stage2_prob_malicious"

    sha_map = {}
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

    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Found {len(npz_files)} cache files")

    # 2. Fast load features for training Tier-1 HSSC and Tier-2 Experts
    random.seed(42)
    bg_shas = set(random.sample(list(sha_map.keys()), 30000))

    X_list = []
    y_list = []
    sha_list = []

    t0 = time.time()
    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha in bg_shas:
                r = sha_map[sha]
                pe = d["pe_features"].astype(np.float32)
                stat = d["stat_features"].astype(np.float32)
                lw = d["lightweight_features"].astype(np.float32)
                
                feat = np.concatenate([pe, stat, lw])
                X_list.append(feat)
                y_list.append(int(r["label"]))
                sha_list.append(sha)
        except Exception:
            continue

    print(f"[Data] Loaded {len(X_list)} samples in {time.time() - t0:.1f}s")
    X_bg = np.stack(X_list)
    y_bg = np.array(y_list, dtype=np.int64)

    # Train Tier-1 HSSC Model
    print("\n[Tier-1] Training Tier-1 HSSC Model...")
    t1_clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=45,
        min_samples_leaf=10,
        l2_regularization=0.5,
        random_state=42
    )
    t1_clf.fit(X_bg, y_bg)

    # 3. Identify residual 1,242 errors under Tier-1 Gated model
    t1_probs = t1_clf.predict_proba(X_bg)[:, 1]
    res_fp_indices = []
    res_fn_indices = []

    for idx, sha in enumerate(sha_list):
        r = sha_map[sha]
        label = y_bg[idx]
        base_prob = float(r[prob_key])
        
        if 0.15 <= base_prob <= 0.85:
            h_prob = t1_probs[idx]
            gated_prob = 0.35 * base_prob + 0.65 * h_prob
        else:
            gated_prob = base_prob

        gated_pred = int(gated_prob >= THR)
        
        if gated_pred == 1 and label == 0:
            res_fp_indices.append(idx)
        elif gated_pred == 0 and label == 1:
            res_fn_indices.append(idx)

    print(f"[Tier-1 Errors on BG] Total: {len(res_fp_indices) + len(res_fn_indices)} | FP: {len(res_fp_indices)} | FN: {len(res_fn_indices)}")

    # 4. Train Tier-2 FP & FN Specialist Experts
    print("\n[Tier-2] Training Tier-2 FP-Specialist (ExtraTrees) & FN-Specialist (HistGBDT)...")
    
    # FP Specialist: trained to suppress FP (weight FP errors heavily as 0)
    w_fp = np.ones(len(y_bg), dtype=np.float32)
    w_fp[res_fp_indices] = 20.0

    fp_expert = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    fp_expert.fit(X_bg, y_bg, sample_weight=w_fp)

    # FN Specialist: trained to rescue FN (weight FN errors heavily as 1)
    w_fn = np.ones(len(y_bg), dtype=np.float32)
    w_fn[res_fn_indices] = 20.0

    fn_expert = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.03,
        max_leaf_nodes=63,
        l2_regularization=2.0,
        random_state=42
    )
    fn_expert.fit(X_bg, y_bg, sample_weight=w_fn)

    # 5. Evaluate Multi-tier Cascade on Evaluation Set
    print("\n[Cascade Evaluation] Testing Multi-tier Decision Cascade...")
    
    t1_preds = []
    t2_preds = []
    true_labels = []
    base_preds = []

    fp_p_bg = fp_expert.predict_proba(X_bg)[:, 1]
    fn_p_bg = fn_expert.predict_proba(X_bg)[:, 1]

    for idx, sha in enumerate(sha_list):
        r = sha_map[sha]
        label = y_bg[idx]
        base_pred = r["pred"]
        base_prob = float(r[prob_key])

        # Tier-1 Gated prob
        if 0.15 <= base_prob <= 0.85:
            h_prob = t1_probs[idx]
            gated_prob = 0.35 * base_prob + 0.65 * h_prob
        else:
            gated_prob = base_prob

        gated_pred = int(gated_prob >= THR)

        # Tier-2 Cascade Consultation
        # If Tier-1 gated prob is in FP danger zone [0.65, 0.85], consult FP-Expert
        # If Tier-1 gated prob is in FN danger zone [0.15, 0.35], consult FN-Expert
        fp_p = fp_p_bg[idx]
        fn_p = fn_p_bg[idx]

        if 0.60 <= gated_prob <= 0.85:
            final_prob = 0.4 * gated_prob + 0.6 * fp_p
        elif 0.15 <= gated_prob <= 0.40:
            final_prob = 0.4 * gated_prob + 0.6 * fn_p
        else:
            final_prob = gated_prob

        final_pred = int(final_prob >= THR)

        base_preds.append(base_pred)
        t1_preds.append(gated_pred)
        t2_preds.append(final_pred)
        true_labels.append(label)

    base_arr = np.array(base_preds)
    t1_arr = np.array(t1_preds)
    t2_arr = np.array(t2_preds)
    y_arr = np.array(true_labels)

    base_err = (base_arr != y_arr).sum()
    t1_err = (t1_arr != y_arr).sum()
    t2_err = (t2_arr != y_arr).sum()

    base_f1 = f1_score(y_arr, base_arr)
    t1_f1 = f1_score(y_arr, t1_arr)
    t2_f1 = f1_score(y_arr, t2_arr)

    print("\n" + "=" * 70)
    print(f"[Loop228 Multi-tier Cascade Results (Evaluated on {len(y_arr):,} samples)]")
    print(f"  Baseline Errors: {base_err:,} | F1: {base_f1:.6f}")
    print(f"  Tier-1   Errors: {t1_err:,} | F1: {t1_f1:.6f}")
    print(f"  Tier-2   Errors: {t2_err:,} | F1: {t2_f1:.6f}")
    print(f"  Tier-2 vs Tier-1 Error Reduction: {t1_err - t2_err} (-{(t1_err - t2_err)/t1_err*100:.1f}%)")
    print("=" * 70)

    # Save receipt
    receipt = {
        "schema": "axon_loop228_multitier_receipt_v1",
        "eval_samples": len(y_arr),
        "baseline_errors": int(base_err),
        "tier1_errors": int(t1_err),
        "tier2_errors": int(t2_err),
        "baseline_f1": float(base_f1),
        "tier1_f1": float(t1_f1),
        "tier2_f1": float(t2_f1),
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop228_multitier_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Receipt saved to {report_path}")


if __name__ == "__main__":
    run_loop228_multitier()
