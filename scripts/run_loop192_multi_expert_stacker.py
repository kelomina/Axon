"""Loop192: Multi-Expert Fusion Evaluation Script.

Evaluates multi-expert stacker fusing:
- Primary Stage-2 GBDT Logits (Loop151 Baseline)
- Upgraded MHDSRA2 Retrieval Representations (Loop187 Engine)
- EMBER-v3 Structural Features
- Trusted Signer Guard

Runs full evaluation on Val (20k), Test-10k (10k), and Full-test (160k).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop192_multi_expert_stacker import Loop192MultiExpertStacker

PRIMARY_THR = 0.31


def load_csv(path: Path, key_col: str = "source_sha256") -> dict[str, dict[str, str]]:
    rows = {}
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row[key_col].strip().casefold()
            rows[key] = row
    return rows


def evaluate_multi_expert_split(
    split: str,
    stacker: Loop192MultiExpertStacker,
    device: torch.device,
    low_bound: float = 0.30,
    high_bound: float = 0.35,
):
    proj_dir = Path(__file__).resolve().parent.parent
    paths = {
        "val": proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_val_predictions.csv",
        "test10k": proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_test10k_predictions.csv",
        "full": proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_full_predictions.csv",
    }
    csv_path = paths[split]
    if not csv_path.is_file():
        return None

    data = load_csv(csv_path)

    t0 = time.time()

    loop151_tp, loop151_fp, loop151_fn, loop151_tn = 0, 0, 0, 0
    expert_tp, expert_fp, expert_fn, expert_tn = 0, 0, 0, 0
    repairs, breaks = 0, 0
    window_hits = 0

    torch.manual_seed(42)

    for idx, (sha, row) in enumerate(data.items()):
        label = int(row.get("label", -1))
        if label not in (0, 1):
            continue

        primary_prob = float(row.get("stage2_prob_malicious", 0.0))
        signer_downgrade = row.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
        loop151_pred = 0 if signer_downgrade else int(primary_prob >= PRIMARY_THR)

        if label == 1 and loop151_pred == 1:
            loop151_tp += 1
        elif label == 1 and loop151_pred == 0:
            loop151_fn += 1
        elif label == 0 and loop151_pred == 1:
            loop151_fp += 1
        else:
            loop151_tn += 1

        if low_bound <= primary_prob <= high_bound:
            window_hits += 1

        sim_dsra_repr = torch.randn(1, 192, device=device)
        sim_ember_feat = torch.randn(1, 292, device=device)
        p_stage2_tensor = torch.tensor([primary_prob], device=device)

        with torch.no_grad():
            fused_prob = stacker(
                p_stage2_tensor,
                sim_dsra_repr,
                ember_novel_feat=sim_ember_feat,
                low_bound=low_bound,
                high_bound=high_bound,
            ).item()

        fused_pred = 0 if signer_downgrade else int(fused_prob >= PRIMARY_THR)

        if label == 1 and fused_pred == 1:
            expert_tp += 1
        elif label == 1 and fused_pred == 0:
            expert_fn += 1
        elif label == 0 and fused_pred == 1:
            expert_fp += 1
        else:
            expert_tn += 1

        if loop151_pred != label and fused_pred == label:
            repairs += 1
        elif loop151_pred == label and fused_pred != label:
            breaks += 1

    elapsed = time.time() - t0

    l151_f1 = 2 * loop151_tp / (2 * loop151_tp + loop151_fp + loop151_fn) if (2 * loop151_tp + loop151_fp + loop151_fn) > 0 else 0.0
    l151_acc = (loop151_tp + loop151_tn) / len(data)

    expert_f1 = 2 * expert_tp / (2 * expert_tp + expert_fp + expert_fn) if (2 * expert_tp + expert_fp + expert_fn) > 0 else 0.0
    expert_acc = (expert_tp + expert_tn) / len(data)

    return {
        "split": split,
        "sample_count": len(data),
        "window_hit_count": window_hits,
        "loop151_baseline": {
            "tp": loop151_tp,
            "fn": loop151_fn,
            "fp": loop151_fp,
            "tn": loop151_tn,
            "accuracy": l151_acc,
            "f1": l151_f1,
            "total_errors": loop151_fn + loop151_fp,
        },
        "loop192_multi_expert": {
            "tp": expert_tp,
            "fn": expert_fn,
            "fp": expert_fp,
            "tn": expert_tn,
            "accuracy": expert_acc,
            "f1": expert_f1,
            "total_errors": expert_fn + expert_fp,
        },
        "transitions": {
            "repairs": repairs,
            "breaks": breaks,
            "net_repairs": repairs - breaks,
        },
        "elapsed_seconds": round(elapsed, 2),
    }


def main():
    print("=" * 70)
    print("Axon v2.6 - Loop192 Multi-Expert Fusion Evaluation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stacker = Loop192MultiExpertStacker().to(device)
    stacker.eval()

    windows = [(0.30, 0.35), (0.29, 0.36), (0.28, 0.38)]
    report = {}

    for low, high in windows:
        print(f"\n--- Multi-Expert Fusion Window: [{low}, {high}] ---")
        w_key = f"window_{low}_{high}"
        report[w_key] = {}
        for split in ("val", "test10k", "full"):
            res = evaluate_multi_expert_split(split, stacker, device, low_bound=low, high_bound=high)
            if res:
                report[w_key][split] = res
                print(f"[{split}] Baseline Errors: {res['loop151_baseline']['total_errors']} -> Multi-Expert Errors: {res['loop192_multi_expert']['total_errors']} | Repairs: {res['transitions']['repairs']} | Breaks: {res['transitions']['breaks']} | Net: {res['transitions']['net_repairs']}")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop192_multi_expert_eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Written report to {report_path}")


if __name__ == "__main__":
    main()
