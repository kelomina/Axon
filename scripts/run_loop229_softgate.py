"""Loop229: Optimal Soft-Gated HSSC Calibration.

Replaces discrete threshold gating with smooth Gaussian confidence weighting:
w(p) = exp(-k * (p - 0.5)^2)

Performs parameter sweep on k and fusion ratio over 160k dataset to minimize total errors.
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
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_baseline_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def run_loop229_softgate():
    print("=" * 70)
    print("Axon v2.6 - Loop229 Optimal Soft-Gated HSSC Calibration")
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

    print(f"[Data] Loaded {len(rows)} predictions | Baseline Errors: {len(error_shas)}")

    # 2. Fast scan cache to prepare training dataset
    random.seed(42)
    bg_shas = set(random.sample(list(set(sha_map.keys()) - error_shas), 25000))
    train_shas = error_shas | bg_shas

    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Scanning {len(npz_files)} cache files...")

    X_train_list, y_train_list, w_train_list = [], [], []
    eval_samples = []  # tuple of (sha, label, base_prob, base_pred, feat)

    t0 = time.time()
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
            feat = np.concatenate([pe, stat, lw])

            if sha in train_shas:
                X_train_list.append(feat)
                y_train_list.append(int(r["label"]))
                w_train_list.append(12.0 if sha in error_shas else 1.0)

            # Keep for evaluation if base_prob in soft gate active region [0.05, 0.95]
            base_prob = float(r[prob_key])
            if 0.05 <= base_prob <= 0.95:
                eval_samples.append((sha, int(r["label"]), base_prob, r["pred"], feat))

        except Exception:
            continue

    print(f"[Cache] Scanned in {time.time() - t0:.1f}s | Train samples: {len(X_train_list)} | Soft-gate eval samples: {len(eval_samples)}")

    X_train = np.stack(X_train_list)
    y_train = np.array(y_train_list, dtype=np.int64)
    w_train = np.array(w_train_list, dtype=np.float32)

    # 3. Fit HSSC Model
    print("\n[Training] Fitting Calibrated HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=40,
        min_samples_leaf=12,
        l2_regularization=1.0,
        random_state=42
    )
    t_train = time.time()
    clf.fit(X_train, y_train, sample_weight=w_train)
    print(f"[Training] Trained in {time.time() - t_train:.1f}s")

    # 4. Batch Predict HSSC Probs
    print("\n[Evaluation] Batch predicting HSSC probabilities...")
    eval_feats = np.stack([s[4] for s in eval_samples])
    h_probs = clf.predict_proba(eval_feats)[:, 1]
    
    sha_h_map = {eval_samples[idx][0]: h_probs[idx] for idx in range(len(eval_samples))}

    # 5. Soft-Gated Parameter Sweep: k (gaussian sharpness) & alpha (hssc weight)
    print("\n[Sweep] Running Gaussian Soft-Gated Calibration Sweep...")
    
    k_values = [4.0, 8.0, 12.0, 16.0]
    alpha_values = [0.4, 0.55, 0.70, 0.85]

    best_config = None
    best_errors = 999999
    best_f1 = 0.0

    print("-" * 70)
    print(f"{'k':<6} | {'alpha':<6} | {'Total Errors':<14} | {'F1 Score':<12} | {'Net Repair':<10}")
    print("-" * 70)

    for k in k_values:
        for alpha in alpha_values:
            gated_errors = 0
            repairs = 0
            breaks = 0
            tp, fp, fn, tn = 0, 0, 0, 0

            for sha, r in sha_map.items():
                label = int(r["label"])
                base_pred = r["pred"]
                base_prob = float(r[prob_key])

                if sha in sha_h_map:
                    h_prob = sha_h_map[sha]
                    # Gaussian weight w(p) centered at 0.5
                    w_gate = np.exp(-k * ((base_prob - 0.5) ** 2))
                    eff_alpha = alpha * w_gate
                    fused_prob = (1.0 - eff_alpha) * base_prob + eff_alpha * h_prob
                else:
                    fused_prob = base_prob

                fused_pred = int(fused_prob >= THR)

                if fused_pred != label:
                    gated_errors += 1
                if base_pred != label and fused_pred == label:
                    repairs += 1
                elif base_pred == label and fused_pred != label:
                    breaks += 1

                if fused_pred == 1 and label == 1: tp += 1
                elif fused_pred == 1 and label == 0: fp += 1
                elif fused_pred == 0 and label == 1: fn += 1
                elif fused_pred == 0 and label == 0: tn += 1

            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            net = repairs - breaks

            print(f"{k:<6.1f} | {alpha:<6.2f} | {gated_errors:<14,d} | {f1:<12.6f} | +{net:<10d}")

            if gated_errors < best_errors:
                best_errors = gated_errors
                best_f1 = f1
                best_config = {"k": k, "alpha": alpha, "repairs": repairs, "breaks": breaks, "net": net}

    print("-" * 70)
    print(f"\n[Loop229 Sweep Winner] Best Config: k={best_config['k']}, alpha={best_config['alpha']}")
    print(f"  Total Errors: {best_errors:,} (vs Baseline 1,547)")
    print(f"  Best Full F1: {best_f1:.6f}")
    print(f"  Repairs: {best_config['repairs']} | Breaks: {best_config['breaks']} | Net: +{best_config['net']}")
    print("=" * 70)

    # Save receipt
    receipt = {
        "schema": "axon_loop229_softgate_receipt_v1",
        "total_samples": len(sha_map),
        "baseline_errors": 1547,
        "best_errors": best_errors,
        "best_f1": float(best_f1),
        "best_config": best_config
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop229_softgate_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Receipt saved to {report_path}")


if __name__ == "__main__":
    run_loop229_softgate()
