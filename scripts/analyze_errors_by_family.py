#!/usr/bin/env python3
"""Analyze errors by malware collection date (family proxy)."""

import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

def read_csv(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))

TEST_CSV = r"E:\Project\python\Axon_v2.6Exp\reports\hard_family_finetune\clean_hyperparam_search\baseline_test_predictions_threshold053_current.csv"
THRESHOLD = 0.53

DATE_DIRS = [
    '2020-02','2020-03','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11',
    '2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12',
    '2025-03','2025-10','2025-11','2025-12','2026-01','2026-02','2026-03',
    '黑文件1',
]

def get_date_dir(source_path):
    parts = source_path.replace('\\', '/').split('/')
    for p in parts:
        if p in DATE_DIRS:
            return p
    if '白名单' in source_path:
        return 'benign_flat'
    return 'unknown'

test = read_csv(TEST_CSV)

# Aggregate
fn_by_date = {}
fp_by_date = {}
total_by_date = {}
fn_probs_by_date = {}
hc_fn_by_date = {}  # high-confidence FN (prob < 0.10)

for r in test:
    label = int(r['label'])
    prob = float(r['prob_malicious'])
    pred = 1 if prob >= THRESHOLD else 0
    date_dir = get_date_dir(r['source_path'])
    
    total_by_date[date_dir] = total_by_date.get(date_dir, 0) + 1
    
    if pred != label:
        if pred == 1 and label == 0:
            fp_by_date[date_dir] = fp_by_date.get(date_dir, 0) + 1
        else:
            fn_by_date[date_dir] = fn_by_date.get(date_dir, 0) + 1
            if date_dir not in fn_probs_by_date:
                fn_probs_by_date[date_dir] = []
            fn_probs_by_date[date_dir].append(prob)
            if prob < 0.10:
                hc_fn_by_date[date_dir] = hc_fn_by_date.get(date_dir, 0) + 1

print("=" * 80)
print("  FN by collection date (malicious families)")
print("=" * 80)
header = f"{'Date':<15} {'Total':>8} {'FN':>6} {'FN%':>7} {'AvgProb':>8} {'HiConf':>7}"
print(header)
print("-" * 80)

for date_dir in sorted(total_by_date):
    total = total_by_date[date_dir]
    fn = fn_by_date.get(date_dir, 0)
    if fn > 0:
        fn_pct = fn / total * 100
        avg_prob = sum(fn_probs_by_date.get(date_dir, [0])) / max(len(fn_probs_by_date.get(date_dir, [1])), 1)
        hc = hc_fn_by_date.get(date_dir, 0)
        print(f"{date_dir:<15} {total:>8} {fn:>6} {fn_pct:>6.1f}% {avg_prob:>8.3f} {hc:>7}")

total_fn = sum(fn_by_date.values())
total_all = sum(total_by_date.values())
print("-" * 80)
print(f"{'TOTAL':<15} {total_all:>8} {total_fn:>6} {total_fn/total_all*100:>6.1f}%")

print()
print("=" * 80)
print("  FP by source")
print("=" * 80)
header2 = f"{'Source':<15} {'Total':>8} {'FP':>6} {'FP%':>7}"
print(header2)
print("-" * 80)

for date_dir in sorted(total_by_date):
    total = total_by_date[date_dir]
    fp = fp_by_date.get(date_dir, 0)
    if fp > 0:
        fp_pct = fp / total * 100
        print(f"{date_dir:<15} {total:>8} {fp:>6} {fp_pct:>6.1f}%")

total_fp = sum(fp_by_date.values())
print("-" * 80)
print(f"{'TOTAL':<15} {total_all:>8} {total_fp:>6} {total_fp/total_all*100:>6.1f}%")

# Near-threshold analysis
print()
print("=" * 80)
print("  Near-threshold errors (could flip with small model change)")
print("=" * 80)

# FN near threshold: prob in [0.40, 0.53)
fn_near = sum(1 for r in test if int(r['label'])==1 and 0.40 <= float(r['prob_malicious']) < THRESHOLD)
# FP near threshold: prob in [0.53, 0.65)
fp_near = sum(1 for r in test if int(r['label'])==0 and THRESHOLD <= float(r['prob_malicious']) < 0.65)

print(f"  FN near-threshold (0.40-0.53): {fn_near} / {total_fn} = {fn_near/total_fn*100:.1f}% of all FN")
print(f"  FP near-threshold (0.53-0.65): {fp_near} / {total_fp} = {fp_near/total_fp*100:.1f}% of all FP")
print()
print(f"  If threshold moved to 0.40:")
new_fn = sum(1 for r in test if int(r['label'])==1 and float(r['prob_malicious']) < 0.40)
new_fp = sum(1 for r in test if int(r['label'])==0 and float(r['prob_malicious']) >= 0.40)
print(f"    FN would be: {new_fn} (currently {total_fn}, change: {new_fn - total_fn:+d})")
print(f"    FP would be: {new_fp} (currently {total_fp}, change: {new_fp - total_fp:+d})")

print(f"\n  If threshold moved to 0.65:")
new_fn2 = sum(1 for r in test if int(r['label'])==1 and float(r['prob_malicious']) < 0.65)
new_fp2 = sum(1 for r in test if int(r['label'])==0 and float(r['prob_malicious']) >= 0.65)
print(f"    FN would be: {new_fn2} (currently {total_fn}, change: {new_fn2 - total_fn:+d})")
print(f"    FP would be: {new_fp2} (currently {total_fp}, change: {new_fp2 - total_fp:+d})")

# Check .NET rate among FN
print()
print("=" * 80)
print("  Checking .NET samples among errors")
print("=" * 80)
import os
from pathlib import Path

# Sample some FN files and check if they're .NET
fn_files = [r['source_path'] for r in test 
            if int(r['label'])==1 and float(r['prob_malicious']) < 0.10]

print(f"  Checking {len(fn_files)} high-confidence FN files for .NET...")
dotnet_count = 0
checked = 0
for f in fn_files[:50]:  # Check first 50
    try:
        import pefile
        pe = pefile.PE(f, fast_load=True)
        # Check for .NET CLR header
        if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
            pass
        # Check for mscoree.dll import (strong .NET indicator)
        is_dotnet = False
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8', errors='ignore').lower()
                if 'mscoree' in dll_name or 'mscoree.dll' in dll_name:
                    is_dotnet = True
                    break
        if is_dotnet:
            dotnet_count += 1
        checked += 1
    except Exception as e:
        pass

print(f"  Checked {checked} files, {dotnet_count} are .NET ({dotnet_count/max(checked,1)*100:.1f}%)")
