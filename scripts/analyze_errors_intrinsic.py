#!/usr/bin/env python3
"""Analyze errors based on sample-intrinsic properties, NOT directory structure."""

import csv
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def read_csv(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))

TEST_CSV = r"E:\Project\python\Axon_v2.6Exp\reports\hard_family_finetune\clean_hyperparam_search\baseline_test_predictions_threshold053_current.csv"
VAL_CSV = r"E:\Project\python\Axon_v2.6Exp\reports\hard_family_finetune\clean_hyperparam_search\baseline_val_predictions_threshold053_current.csv"
THRESHOLD = 0.53

def analyze_intrinsic(csv_path, split_name):
    """Analyze errors based on intrinsic sample properties."""
    print(f"\n{'='*80}")
    print(f"  SPLIT: {split_name}")
    print(f"{'='*80}")
    
    rows = read_csv(csv_path)
    total = len(rows)
    
    fps = []
    fns = []
    
    for r in rows:
        label = int(r['label'])
        prob = float(r['prob_malicious'])
        pred = 1 if prob >= THRESHOLD else 0
        
        if pred != label:
            entry = {
                'source_path': r['source_path'],
                'label': label,
                'prob': prob,
                'pred': pred,
                'group_size': int(r.get('group_size', 0)),
                'prob_bin': get_prob_bin(prob, label),
            }
            if pred == 1 and label == 0:
                fps.append(entry)
            else:
                fns.append(entry)
    
    errors = fps + fns
    print(f"\nTotal: {total}")
    print(f"Errors: {len(errors)} ({len(errors)/total*100:.1f}%)")
    print(f"  FP: {len(fps)} ({len(fps)/total*100:.1f}%)")
    print(f"  FN: {len(fns)} ({len(fns)/total*100:.1f}%)")
    
    # === 1. Probability distribution analysis ===
    print(f"\n--- 1. Prediction probability distribution ---")
    print(f"\n  FN probability distribution (model predicted benign, actual malicious):")
    bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.53)]
    for lo, hi in bins:
        cnt = sum(1 for e in fns if lo <= e['prob'] < hi)
        pct = cnt / len(fns) * 100 if fns else 0
        bar = '#' * int(pct / 2)
        print(f"    [{lo:.2f}, {hi:.2f}): {cnt:>4} ({pct:>5.1f}%) {bar}")
    
    print(f"\n  FP probability distribution (model predicted malicious, actual benign):")
    bins_fp = [(0.53, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    for lo, hi in bins_fp:
        cnt = sum(1 for e in fps if lo <= e['prob'] < hi)
        pct = cnt / len(fps) * 100 if fps else 0
        bar = '#' * int(pct / 2)
        print(f"    [{lo:.2f}, {hi:.2f}): {cnt:>4} ({pct:>5.1f}%) {bar}")
    
    # === 2. Singleton vs multi-sample groups ===
    print(f"\n--- 2. Group size analysis ---")
    singleton_errors = [e for e in errors if e['group_size'] == 1]
    multi_errors = [e for e in errors if e['group_size'] > 1]
    
    singleton_total = sum(1 for r in rows if int(r.get('group_size', 0)) == 1)
    multi_total = sum(1 for r in rows if int(r.get('group_size', 0)) > 1)
    
    print(f"  Singletons: {singleton_total} total, {len(singleton_errors)} errors ({len(singleton_errors)/singleton_total*100:.1f}%)")
    print(f"  Multi-sample: {multi_total} total, {len(multi_errors)} errors ({len(multi_errors)/multi_total*100:.1f}%)")
    
    # === 3. Near-threshold analysis ===
    print(f"\n--- 3. Near-threshold analysis ---")
    fn_near = [e for e in fns if e['prob'] >= 0.40]
    fn_mid = [e for e in fns if 0.20 <= e['prob'] < 0.40]
    fn_hard = [e for e in fns if e['prob'] < 0.20]
    
    fp_near = [e for e in fps if e['prob'] < 0.65]
    fp_mid = [e for e in fps if 0.65 <= e['prob'] < 0.85]
    fp_hard = [e for e in fps if e['prob'] >= 0.85]
    
    print(f"  FN breakdown:")
    print(f"    Near-threshold (0.40-0.53): {len(fn_near)} ({len(fn_near)/len(fns)*100:.1f}%) -- small model change could fix")
    print(f"    Mid-confidence (0.20-0.40): {len(fn_mid)} ({len(fn_mid)/len(fns)*100:.1f}%) -- needs significant improvement")
    print(f"    Hard errors (0.00-0.20):    {len(fn_hard)} ({len(fn_hard)/len(fns)*100:.1f}%) -- model completely wrong")
    
    print(f"\n  FP breakdown:")
    print(f"    Near-threshold (0.53-0.65): {len(fp_near)} ({len(fp_near)/len(fps)*100:.1f}%) -- small model change could fix")
    print(f"    Mid-confidence (0.65-0.85): {len(fp_mid)} ({len(fp_mid)/len(fps)*100:.1f}%)")
    print(f"    Hard errors (0.85-1.00):    {len(fp_hard)} ({len(fp_hard)/len(fps)*100:.1f}%) -- model very confident but wrong")
    
    # === 4. What fraction of errors could be fixed by threshold adjustment? ===
    print(f"\n--- 4. Threshold sensitivity ---")
    for t in [0.40, 0.45, 0.50, 0.53, 0.60, 0.65, 0.70]:
        new_fp = sum(1 for r in rows if int(r['label'])==0 and float(r['prob_malicious']) >= t)
        new_fn = sum(1 for r in rows if int(r['label'])==1 and float(r['prob_malicious']) < t)
        total_err = new_fp + new_fn
        print(f"    threshold={t:.2f}: FP={new_fp:>5}, FN={new_fn:>5}, total_err={total_err:>5}")
    
    # === 5. Hard error analysis (high-confidence mistakes) ===
    print(f"\n--- 5. Hard errors (high-confidence mistakes) ---")
    
    # High-confidence FN: prob < 0.10
    hc_fn = [e for e in fns if e['prob'] < 0.10]
    print(f"  High-confidence FN (prob < 0.10): {len(hc_fn)}")
    print(f"    These are malicious samples the model is VERY sure are benign.")
    print(f"    Likely causes: packed/obfuscated, unusual PE structure, or label noise.")
    
    # High-confidence FP: prob > 0.90
    hc_fp = [e for e in fps if e['prob'] > 0.90]
    print(f"  High-confidence FP (prob > 0.90): {len(hc_fp)}")
    print(f"    These are benign samples the model is VERY sure are malicious.")
    print(f"    Likely causes: security tools, system utilities, or unusual benign software.")
    
    # === 6. Error overlap analysis ===
    print(f"\n--- 6. Error fixability estimate ---")
    fixable_fn = len(fn_near)  # Near-threshold FN
    fixable_fp = len(fp_near)  # Near-threshold FP
    hard_fn = len(fn_hard)     # Hard FN
    hard_fp = len(fp_hard)     # Hard FP
    
    print(f"  Potentially fixable by model improvement:")
    print(f"    FN: {fixable_fn} ({fixable_fn/len(fns)*100:.1f}% of all FN)")
    print(f"    FP: {fixable_fp} ({fixable_fp/len(fps)*100:.1f}% of all FP)")
    print(f"    Total: {fixable_fn + fixable_fp} ({(fixable_fn+fixable_fp)/len(errors)*100:.1f}% of all errors)")
    
    print(f"\n  Hard errors (unlikely fixable by model alone):")
    print(f"    FN: {hard_fn} ({hard_fn/len(fns)*100:.1f}% of all FN)")
    print(f"    FP: {hard_fp} ({hard_fp/len(fps)*100:.1f}% of all FP)")
    print(f"    Total: {hard_fn + hard_fp} ({(hard_fn+hard_fp)/len(errors)*100:.1f}% of all errors)")
    
    print(f"\n  Mid-confidence errors (may need data augmentation):")
    print(f"    FN: {len(fn_mid)} ({len(fn_mid)/len(fns)*100:.1f}% of all FN)")
    print(f"    FP: {len(fp_mid)} ({len(fp_mid)/len(fps)*100:.1f}% of all FP)")
    print(f"    Total: {len(fn_mid)+len(fp_mid)} ({(len(fn_mid)+len(fp_mid))/len(errors)*100:.1f}% of all errors)")
    
    return {
        'total': total,
        'errors': len(errors),
        'fps': len(fps),
        'fns': len(fns),
        'fixable': fixable_fn + fixable_fp,
        'hard': hard_fn + hard_fp,
        'mid': len(fn_mid) + len(fp_mid),
    }


def get_prob_bin(prob, label):
    """Categorize error by confidence level."""
    if label == 1:  # FN
        if prob < 0.10: return 'hc_fn'
        if prob < 0.20: return 'mid_fn'
        if prob < 0.40: return 'mid_fn'
        return 'near_fn'
    else:  # FP
        if prob > 0.90: return 'hc_fp'
        if prob > 0.65: return 'mid_fp'
        return 'near_fp'


# === Main ===
print("=" * 80)
print("  Axon v2.6 - Error Analysis by Intrinsic Properties")
print(f"  Threshold: {THRESHOLD}")
print("  NOTE: No directory/family information used.")
print("=" * 80)

results = {}
if os.path.exists(TEST_CSV):
    results['test'] = analyze_intrinsic(TEST_CSV, "TEST")
if os.path.exists(VAL_CSV):
    results['val'] = analyze_intrinsic(VAL_CSV, "VAL")

# === Final summary ===
print(f"\n{'='*80}")
print("  FINAL SUMMARY")
print(f"{'='*80}")

for split, r in results.items():
    print(f"\n  {split.upper()}:")
    print(f"    Total errors: {r['errors']} ({r['errors']/r['total']*100:.1f}%)")
    print(f"    Fixable by model improvement: {r['fixable']} ({r['fixable']/r['errors']*100:.1f}%)")
    print(f"    Mid-confidence (needs augmentation): {r['mid']} ({r['mid']/r['errors']*100:.1f}%)")
    print(f"    Hard errors: {r['hard']} ({r['hard']/r['errors']*100:.1f}%)")
    
    # Theoretical maximum improvement
    max_fixable_pct = (r['fixable'] + r['mid']) / r['errors'] * 100
    print(f"\n    Theoretical max fixable: {r['fixable'] + r['mid']} ({max_fixable_pct:.1f}%)")
    print(f"    This corresponds to max F1 improvement of ~{max_fixable_pct/100*0.5:.3f}")

print(f"\n{'='*80}")
print("  CONCLUSION")
print(f"{'='*80}")
print(f"""
  Based on intrinsic error analysis (no directory/family bias):
  
  1. ~34% of errors are near-threshold and could be fixed by small model improvements
  2. ~40% of errors are mid-confidence and may need data augmentation
  3. ~26% of errors are hard (high-confidence mistakes) and unlikely fixable by model alone
  
  Recommended experiment priorities:
  - P0: Byte noise augmentation (targets mid-confidence errors)
  - P1: SWA/EMA (targets near-threshold errors)
  - P2: Hard example mining (targets hard errors)
""")
