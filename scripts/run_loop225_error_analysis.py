"""Loop225: Error Analysis of GBDT Baseline Errors.

Analyzes the 1,544 errors from the Loop151 full 160k predictions:
- Error type distribution (FP vs FN)
- Feature statistics of error samples vs correct samples  
- StreamGNN agreement/disagreement patterns
- Identifies clusters of similar errors for targeted fixing
"""

from __future__ import annotations

import csv
import glob
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_baseline_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def analyze_errors():
    print("=" * 70)
    print("Axon v2.6 - Loop225 Error Analysis")
    print("=" * 70)

    proj_dir = Path(__file__).resolve().parent.parent

    # Load full predictions
    full_csv = proj_dir / "reports" / "phase3_loop151" / "loop151_trusted_signer_guard_full_predictions.csv"
    if not full_csv.is_file():
        print(f"[ERROR] CSV not found: {full_csv}")
        return

    rows = load_baseline_csv(full_csv)
    print(f"Loaded {len(rows)} predictions")
    print(f"Available columns: {list(rows[0].keys())}")

    # Parse predictions
    THR = 0.31
    errors_fp = []  # Predicted malicious, actually benign
    errors_fn = []  # Predicted benign, actually malicious
    correct = []

    prob_key = "stage2_prob_malicious"

    for row in rows:
        label = int(row.get("label", -1))
        if label not in (0, 1):
            continue
        prob = float(row.get(prob_key, 0.0))
        signer_down = row.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
        pred = 0 if signer_down else int(prob >= THR)

        if pred == 1 and label == 0:
            errors_fp.append(row)
        elif pred == 0 and label == 1:
            errors_fn.append(row)
        else:
            correct.append(row)

    total_errors = len(errors_fp) + len(errors_fn)
    print(f"\n--- Error Summary ---")
    print(f"Total errors: {total_errors}")
    print(f"False Positives (FP, benign predicted malicious): {len(errors_fp)}")
    print(f"False Negatives (FN, malicious predicted benign): {len(errors_fn)}")
    print(f"Correct: {len(correct)}")

    # Analyze probability distribution of errors
    print(f"\n--- FP Probability Distribution ---")
    fp_probs = [float(r[prob_key]) for r in errors_fp]
    if fp_probs:
        fp_probs_arr = np.array(fp_probs)
        print(f"  Count: {len(fp_probs)}")
        print(f"  Mean: {fp_probs_arr.mean():.4f}")
        print(f"  Median: {np.median(fp_probs_arr):.4f}")
        print(f"  Min: {fp_probs_arr.min():.4f}")
        print(f"  Max: {fp_probs_arr.max():.4f}")
        print(f"  Borderline (0.31-0.50): {((fp_probs_arr >= 0.31) & (fp_probs_arr < 0.50)).sum()}")
        print(f"  Medium (0.50-0.80):     {((fp_probs_arr >= 0.50) & (fp_probs_arr < 0.80)).sum()}")
        print(f"  High (0.80-1.00):       {(fp_probs_arr >= 0.80).sum()}")

    print(f"\n--- FN Probability Distribution ---")
    fn_probs = [float(r[prob_key]) for r in errors_fn]
    if fn_probs:
        fn_probs_arr = np.array(fn_probs)
        print(f"  Count: {len(fn_probs)}")
        print(f"  Mean: {fn_probs_arr.mean():.4f}")
        print(f"  Median: {np.median(fn_probs_arr):.4f}")
        print(f"  Min: {fn_probs_arr.min():.4f}")
        print(f"  Max: {fn_probs_arr.max():.4f}")
        print(f"  Borderline (0.20-0.31): {((fn_probs_arr >= 0.20) & (fn_probs_arr < 0.31)).sum()}")
        print(f"  Low (0.10-0.20):        {((fn_probs_arr >= 0.10) & (fn_probs_arr < 0.20)).sum()}")
        print(f"  Very Low (0.00-0.10):   {(fn_probs_arr < 0.10).sum()}")
        print(f"  Signer downgraded (prob may be high but signer guard overrode):")
        fn_signer_down = [r for r in errors_fn if r.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"]
        print(f"    Count: {len(fn_signer_down)}")
        if fn_signer_down:
            fn_down_probs = np.array([float(r[prob_key]) for r in fn_signer_down])
            print(f"    Prob range: [{fn_down_probs.min():.4f}, {fn_down_probs.max():.4f}]")
            print(f"    Prob >= THR (would be correct w/o guard): {(fn_down_probs >= THR).sum()}")

    # Analyze signer-guard impact
    print(f"\n--- Signer Guard Analysis ---")
    signer_downgraded = [r for r in rows if r.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"]
    signer_correct = [r for r in signer_downgraded if int(r["label"]) == 0]
    signer_wrong = [r for r in signer_downgraded if int(r["label"]) == 1]
    print(f"Total signer-downgraded: {len(signer_downgraded)}")
    print(f"  Correct downgrade (really benign): {len(signer_correct)}")
    print(f"  Wrong downgrade (actually malicious): {len(signer_wrong)}")

    # Feature statistics comparison: errors vs correct
    print(f"\n--- Feature-Level Error Analysis (from npz) ---")
    cache_dir = proj_dir / "data" / ".cache"
    
    # Build SHA index for error samples only
    error_shas = set()
    for r in errors_fp + errors_fn:
        error_shas.add(r.get("source_sha256", "").strip().lower())

    # Also sample some correct predictions for comparison
    import random
    random.seed(42)
    correct_sample = random.sample(correct, min(2000, len(correct)))
    correct_shas = set()
    for r in correct_sample:
        correct_shas.add(r.get("source_sha256", "").strip().lower())

    all_shas = error_shas | correct_shas
    
    # Scan cache to find matching files
    sha_to_path = {}
    npz_files = sorted(glob.glob(str(cache_dir / "*.npz")))
    print(f"Scanning {len(npz_files)} npz files for {len(all_shas)} target SHAs...")
    found = 0
    for f in npz_files:
        if found >= len(all_shas):
            break
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha in all_shas:
                sha_to_path[sha] = f
                found += 1
        except Exception:
            continue
    print(f"Found {found}/{len(all_shas)} target SHAs in cache")

    # Load features for error vs correct
    def load_features_for_rows(rows_list, tag):
        pe_list, stat_list, lw_list = [], [], []
        for r in rows_list:
            sha = r.get("source_sha256", "").strip().lower()
            path = sha_to_path.get(sha)
            if path is None:
                continue
            try:
                d = np.load(path, allow_pickle=True)
                pe_list.append(d["pe_features"].astype(np.float32))
                stat_list.append(d["stat_features"].astype(np.float32))
                lw_list.append(d["lightweight_features"].astype(np.float32))
            except Exception:
                continue
        if pe_list:
            pe = np.stack(pe_list)
            stat = np.stack(stat_list)
            lw = np.stack(lw_list)
            print(f"  [{tag}] Loaded {len(pe_list)} samples")
            print(f"    pe_features: mean={pe.mean():.4f}, std={pe.std():.4f}, nonzero_frac={(pe != 0).mean():.4f}")
            print(f"    stat_features: mean={stat.mean():.4f}, std={stat.std():.4f}")
            print(f"    lightweight_features: mean={lw.mean():.4f}, std={lw.std():.4f}")
            return pe, stat, lw
        return None, None, None

    print(f"\nFalse Positives (benign predicted malicious):")
    fp_pe, fp_stat, fp_lw = load_features_for_rows(errors_fp, "FP")
    
    print(f"\nFalse Negatives (malicious predicted benign):")
    fn_pe, fn_stat, fn_lw = load_features_for_rows(errors_fn, "FN")
    
    print(f"\nCorrect Predictions (sample):")
    c_pe, c_stat, c_lw = load_features_for_rows(correct_sample, "Correct")

    # Feature divergence analysis
    if fp_stat is not None and c_stat is not None:
        print(f"\n--- Stat Feature Divergence (FP vs Correct) ---")
        fp_mean = fp_stat.mean(axis=0)
        c_mean = c_stat.mean(axis=0)
        c_std = c_stat.std(axis=0) + 1e-8
        z_scores = np.abs(fp_mean - c_mean) / c_std
        top_divergent = np.argsort(z_scores)[::-1][:10]
        for idx in top_divergent:
            print(f"  stat_feat[{idx:2d}]: FP_mean={fp_mean[idx]:.4f}, Correct_mean={c_mean[idx]:.4f}, Z={z_scores[idx]:.2f}")

    if fn_stat is not None and c_stat is not None:
        print(f"\n--- Stat Feature Divergence (FN vs Correct) ---")
        fn_mean = fn_stat.mean(axis=0)
        c_mean = c_stat.mean(axis=0)
        c_std = c_stat.std(axis=0) + 1e-8
        z_scores = np.abs(fn_mean - c_mean) / c_std
        top_divergent = np.argsort(z_scores)[::-1][:10]
        for idx in top_divergent:
            print(f"  stat_feat[{idx:2d}]: FN_mean={fn_mean[idx]:.4f}, Correct_mean={c_mean[idx]:.4f}, Z={z_scores[idx]:.2f}")

    # Receipt
    receipt = {
        "schema": "axon_loop225_error_analysis_receipt_v1",
        "total_errors": total_errors,
        "fp_count": len(errors_fp),
        "fn_count": len(errors_fn),
        "signer_guard_wrong_downgrade": len(signer_wrong),
        "matched_in_cache": found,
    }
    report_path = proj_dir / "reports" / "roadmap_9997" / "loop225_error_analysis_receipt.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nSaved receipt to {report_path}")


if __name__ == "__main__":
    analyze_errors()
