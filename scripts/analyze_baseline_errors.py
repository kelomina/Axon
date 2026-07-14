#!/usr/bin/env python3
"""Analyze remaining errors from baseline model predictions."""

import csv
import os
from collections import Counter
from pathlib import Path

def read_csv(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))

TEST_CSV = Path(r"E:\Project\python\Axon_v2.6Exp\reports\hard_family_finetune\clean_hyperparam_search\baseline_test_predictions_threshold053_current.csv")
VAL_CSV = Path(r"E:\Project\python\Axon_v2.6Exp\reports\hard_family_finetune\clean_hyperparam_search\baseline_val_predictions_threshold053_current.csv")
THRESHOLD = 0.53
DATA_ROOT = Path(r"E:\Project\python\Axon_v2.6Exp\data")

def extract_family_info(source_path: str) -> dict:
    """Extract family/group info from file path."""
    try:
        p = Path(source_path)
        parts = p.parts
        # Try to find data root in path
        for i, part in enumerate(parts):
            if 'data' in part.lower() or 'axon' in part.lower():
                # Everything after data root is the family structure
                family_parts = parts[i+1:]
                break
        else:
            family_parts = parts[-4:]
        
        # Return structured info
        return {
            'top_dir': family_parts[0] if len(family_parts) > 0 else 'unknown',
            'mid_dir': family_parts[1] if len(family_parts) > 1 else 'unknown',
            'sub_dir': family_parts[2] if len(family_parts) > 2 else 'unknown',
            'filename': family_parts[-1] if family_parts else 'unknown',
        }
    except Exception:
        return {'top_dir': 'error', 'mid_dir': 'error', 'sub_dir': 'error', 'filename': 'error'}


def analyze_split(csv_path, split_name):
    """Analyze errors for a single split."""
    print(f"\n{'='*70}")
    print(f"  SPLIT: {split_name}")
    print(f"{'='*70}")
    
    rows = read_csv(csv_path)
    total = len(rows)
    
    fps = []
    fns = []
    correct = []
    
    for r in rows:
        label = int(r['label'])
        prob = float(r['prob_malicious'])
        pred = 1 if prob >= THRESHOLD else 0
        
        entry = {
            'source_path': r['source_path'],
            'label': label,
            'prob': prob,
            'pred': pred,
            'group_size': int(r.get('group_size', 0)),
            'group_id': r.get('group_id', ''),
            'family': extract_family_info(r['source_path']),
        }
        
        if pred == label:
            correct.append(entry)
        elif pred == 1 and label == 0:
            fps.append(entry)
        else:
            fns.append(entry)
    
    errors = fps + fns
    print(f"\nTotal: {total}")
    print(f"Correct: {len(correct)} ({len(correct)/total*100:.1f}%)")
    print(f"Errors: {len(errors)} ({len(errors)/total*100:.1f}%)")
    print(f"  FP (benign predicted malicious): {len(fps)}")
    print(f"  FN (malicious predicted benign): {len(fns)}")
    
    # === FP Analysis ===
    if fps:
        print(f"\n--- FP Analysis ({len(fps)} samples) ---")
        
        # Confidence
        fp_probs = [e['prob'] for e in fps]
        print(f"\n  Confidence distribution:")
        ranges = [(0.53, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
        for lo, hi in ranges:
            cnt = sum(1 for p in fp_probs if lo <= p < hi)
            print(f"    [{lo:.2f}, {hi:.2f}): {cnt} ({cnt/len(fps)*100:.1f}%)")
        
        # Group size
        print(f"\n  Group size distribution:")
        gs_counts = Counter(e['group_size'] for e in fps)
        for gs in sorted(gs_counts):
            total_in_gs = sum(1 for r in rows if int(r.get('group_size',0)) == gs)
            err_rate = gs_counts[gs] / total_in_gs * 100 if total_in_gs > 0 else 0
            print(f"    size={gs}: {gs_counts[gs]} FP (of {total_in_gs} total, {err_rate:.1f}% error rate)")
        
        # Top directories
        print(f"\n  Top FP source directories:")
        fp_top_dirs = Counter(e['family']['top_dir'] for e in fps)
        for d, cnt in fp_top_dirs.most_common(10):
            print(f"    {d}: {cnt}")
    
    # === FN Analysis ===
    if fns:
        print(f"\n--- FN Analysis ({len(fns)} samples) ---")
        
        # Confidence
        fn_probs = [e['prob'] for e in fns]
        print(f"\n  Confidence distribution:")
        ranges = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.53)]
        for lo, hi in ranges:
            cnt = sum(1 for p in fn_probs if lo <= p < hi)
            print(f"    [{lo:.2f}, {hi:.2f}): {cnt} ({cnt/len(fns)*100:.1f}%)")
        
        # Group size
        print(f"\n  Group size distribution:")
        gs_counts = Counter(e['group_size'] for e in fns)
        for gs in sorted(gs_counts):
            total_in_gs = sum(1 for r in rows if int(r.get('group_size',0)) == gs)
            err_rate = gs_counts[gs] / total_in_gs * 100 if total_in_gs > 0 else 0
            print(f"    size={gs}: {gs_counts[gs]} FN (of {total_in_gs} total, {err_rate:.1f}% error rate)")
        
        # Top directories
        print(f"\n  Top FN source directories:")
        fn_top_dirs = Counter(e['family']['top_dir'] for e in fns)
        for d, cnt in fn_top_dirs.most_common(10):
            print(f"    {d}: {cnt}")
        
        # High-confidence FN (prob < 0.10) - model is very sure these are benign but they're malicious
        high_conf_fn = [e for e in fns if e['prob'] < 0.10]
        if high_conf_fn:
            print(f"\n  High-confidence FN (prob < 0.10, model very sure benign): {len(high_conf_fn)}")
            print(f"  Sample paths:")
            for e in high_conf_fn[:5]:
                fam = e['family']
                print(f"    prob={e['prob']:.4f} gs={e['group_size']} {fam['top_dir']}/{fam['mid_dir']}/{fam['sub_dir']}/{fam['filename']}")
    
    # === Singleton vs Family deep analysis ===
    print(f"\n--- Singleton vs Family ---")
    singleton_errors = [e for e in errors if e['group_size'] == 1]
    family_errors = [e for e in errors if e['group_size'] > 1]
    singleton_total = sum(1 for r in rows if int(r.get('group_size',0)) == 1)
    family_total = sum(1 for r in rows if int(r.get('group_size',0)) > 1)
    
    print(f"  Singletons: {singleton_total} total, {len(singleton_errors)} errors ({len(singleton_errors)/singleton_total*100:.1f}%)")
    print(f"  Families:   {family_total} total, {len(family_errors)} errors ({len(family_errors)/family_total*100:.1f}%)")
    
    # Error type breakdown for singletons
    s_fp = sum(1 for e in singleton_errors if e['label'] == 0)
    s_fn = sum(1 for e in singleton_errors if e['label'] == 1)
    f_fp = sum(1 for e in family_errors if e['label'] == 0)
    f_fn = sum(1 for e in family_errors if e['label'] == 1)
    print(f"  Singleton FP: {s_fp}, FN: {s_fn}")
    print(f"  Family FP: {f_fp}, FN: {f_fn}")
    
    return {
        'total': total, 'errors': len(errors), 'fps': len(fps), 'fns': len(fns),
        'singleton_total': singleton_total, 'singleton_errors': len(singleton_errors),
        'family_total': family_total, 'family_errors': len(family_errors),
    }


# === Main ===
print("=" * 70)
print("  Axon v2.6 Baseline Model - Detailed Error Analysis")
print(f"  Threshold: {THRESHOLD}")
print("=" * 70)

results = {}
if TEST_CSV.exists():
    results['test'] = analyze_split(TEST_CSV, "TEST")
if VAL_CSV.exists():
    results['val'] = analyze_split(VAL_CSV, "VAL")

# === Summary ===
print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")
for split, r in results.items():
    print(f"\n  {split.upper()}:")
    print(f"    Error rate: {r['errors']}/{r['total']} = {r['errors']/r['total']*100:.1f}%")
    print(f"    FP: {r['fps']}, FN: {r['fns']}")
    print(f"    Singleton error rate: {r['singleton_errors']}/{r['singleton_total']} = {r['singleton_errors']/r['singleton_total']*100:.1f}%")
    print(f"    Family error rate: {r['family_errors']}/{r['family_total']} = {r['family_errors']/r['family_total']*100:.1f}%")
    if r['errors'] > 0:
        singleton_pct = r['singleton_errors'] / r['errors'] * 100
        print(f"    Singleton share of errors: {singleton_pct:.1f}%")

# === Data directory structure ===
print(f"\n{'='*70}")
print("  Data Directory Structure")
print(f"{'='*70}")
if DATA_ROOT.exists():
    for item in sorted(os.listdir(DATA_ROOT)):
        full = DATA_ROOT / item
        if full.is_dir() and not item.startswith('.'):
            subdirs = [d for d in full.iterdir() if d.is_dir()]
            total_files = 0
            for sd in subdirs:
                total_files += len([f for f in sd.iterdir() if f.is_file()])
            print(f"  {item}/ ({len(subdirs)} subdirs, ~{total_files} files)")
            for sd in sorted(subdirs)[:5]:
                if sd.is_dir():
                    files = [f for f in sd.iterdir() if f.is_file()]
                    print(f"    {sd.name}/ ({len(files)} files)")
else:
    print(f"  Data directory not found: {DATA_ROOT}")
