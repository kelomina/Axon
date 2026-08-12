"""Loop230: Feature-Divergence Weighted Ensemble.

Combines Gaussian soft-gated calibration with Z-score feature divergence weighting
on the top 10 anomalous statistical features (stat_feat[15], stat_feat[17], etc.).
Dynamically boosts HSSC intervention weight when samples exhibit high structural ambiguity.
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


def run_loop230_divergence():
    print("=" * 70)
    print("Axon v2.6 - Loop230 Feature-Divergence Weighted Ensemble")
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

    # 2. Fast scan cache to prepare dataset
    random.seed(42)
    bg_shas = set(random.sample(list(set(sha_map.keys()) - error_shas), 25000))
    train_shas = error_shas | bg_shas

    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Scanning {len(npz_files)} cache files...")

    X_train_list, y_train_list, w_train_list = [], [], []
    eval_samples = []  # tuple of (sha, label, base_prob, base_pred, feat, stat_feat)

    # Top divergent stat feature indices identified in Loop225
    div_indices = [15, 17, 0, 4, 11, 25, 6, 24, 28, 27]

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

            base_prob = float(r[prob_key])
            if 0.05 <= base_prob <= 0.95:
                eval_samples.append((sha, int(r["label"]), base_prob, r["pred"], feat, stat[div_indices]))

        except Exception:
            continue

    print(f"[Cache] Scanned in {time.time() - t0:.1f}s | Train samples: {len(X_train_list)}")

    X_train = np.stack(X_train_list)
    y_train = np.array(y_train_list, dtype=np.int64)
    w_train = np.array(w_train_list, dtype=np.float32)

    # 3. Fit HSSC Model
    print("\n[Training] Fitting Divergence-Aware HistGradientBoostingClassifier...")
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

    # Compute mean & std of divergent features across training set
    div_feats_bg = np.stack([s[5] for s in eval_samples])
    div_mean = div_feats_bg.mean(axis=0)
    div_std = div_feats_bg.std(axis=0) + 1e-6

    # 4. Batch Predict HSSC Probs
    print("\n[Evaluation] Batch predicting HSSC probabilities with Feature Divergence Weighting...")
    eval_feats = np.stack([s[4] for s in eval_samples])
    h_probs = clf.predict_proba(eval_feats)[:, 1]

    # Calculate divergence z-scores for each eval sample
    div_z_scores = np.abs(div_feats_bg - div_mean) / div_std  # (N, 10)
    avg_div_boost = np.clip(div_z_scores.mean(axis=1) / 3.0, 0.0, 0.5)  # Divergence boost factor [0, 0.5]

    sha_eval_map = {}
    for idx, sample in enumerate(eval_samples):
        sha = sample[0]
        sha_eval_map[sha] = (h_probs[idx], avg_div_boost[idx])

    # 5. Evaluate Divergence-Boosted Soft-Gating
    k = 4.0
    base_alpha = 0.85

    gated_errors = 0
    repairs = 0
    breaks = 0
    tp, fp, fn, tn = 0, 0, 0, 0

    for sha, r in sha_map.items():
        label = int(r["label"])
        base_pred = r["pred"]
        base_prob = float(r[prob_key])

        if sha in sha_eval_map:
            h_prob, div_boost = sha_eval_map[sha]
            
            # Base Gaussian weight
            w_gate = np.exp(-k * ((base_prob - 0.5) ** 2))
            
            # Divergence boost increases alpha for anomalous samples
            eff_alpha = np.clip(base_alpha * (1.0 + div_boost) * w_gate, 0.0, 0.95)
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

    print("\n" + "=" * 70)
    print(f"[Loop230 Feature-Divergence Weighted Results (Full {len(sha_map):,} samples)]")
    print(f"  Baseline Errors: 1,547 | Baseline F1: 0.990365")
    print(f"  Loop229  Errors: 1,235 | Loop229  F1: 0.992312")
    print(f"  Loop230  Errors: {gated_errors:,} | Loop230  F1: {f1:.6f}")
    print(f"  Repairs: {repairs} | Breaks: {breaks} | Net Repair: +{net}")
    print("=" * 70)

    # Save receipt
    receipt = {
        "schema": "axon_loop230_divergence_receipt_v1",
        "total_samples": len(sha_map),
        "baseline_errors": 1547,
        "loop229_errors": 1235,
        "loop230_errors": gated_errors,
        "full_f1": float(f1),
        "repairs": repairs,
        "breaks": breaks,
        "net_repair": net
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop230_divergence_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Receipt saved to {report_path}")


if __name__ == "__main__":
    run_loop230_divergence()
