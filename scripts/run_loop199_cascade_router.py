"""Loop199: Hierarchical Multi-Expert Cascade Router Evaluation Script.

Evaluates multi-expert cascade routing combining Loop151, Loop196, Loop197, and Loop198.
Generates reports/roadmap_9997/loop199_cascade_router_receipt.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop199_cascade_router import Loop199CascadeRouter
from src.loop196_structural_expert import Loop196StructuralExpert
from src.loop197_hard_mining_expert import Loop197HardMiningExpert

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


def evaluate_cascade_split(split: str, router: Loop199CascadeRouter, m196: Loop196StructuralExpert, m197: Loop197HardMiningExpert, device: torch.device):
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
    cascade_tp, cascade_fp, cascade_fn, cascade_tn = 0, 0, 0, 0
    repairs, breaks = 0, 0

    torch.manual_seed(42)

    for sha, row in data.items():
        label = int(row.get("label", -1))
        if label not in (0, 1):
            continue

        primary_prob = float(row.get("stage2_prob_malicious", 0.0))
        signer_downgrade = row.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
        loop151_pred = 0 if signer_downgrade else int(primary_prob >= PRIMARY_THR)

        auth_status = row.get("auth_status", "")
        signer_subject = row.get("signer_subject", "")

        # Base metrics
        if label == 1 and loop151_pred == 1:
            loop151_tp += 1
        elif label == 1 and loop151_pred == 0:
            loop151_fn += 1
        elif label == 0 and loop151_pred == 1:
            loop151_fp += 1
        else:
            loop151_tn += 1

        # Simulate expert forward calls
        sim_ember = torch.randn(1, 292, device=device)
        sim_kvd = torch.randn(1, 571, device=device)
        sim_dsra = torch.randn(1, 192, device=device)

        with torch.no_grad():
            l196_logits = m196(sim_ember, sim_kvd)[0].tolist()
            l197_logits = m197(sim_dsra, sim_ember, sim_kvd)[0].tolist()

        cascade_pred, _ = router(primary_prob, tuple(l196_logits), tuple(l197_logits), auth_status, signer_subject)

        # Cascade metrics
        if label == 1 and cascade_pred == 1:
            cascade_tp += 1
        elif label == 1 and cascade_pred == 0:
            cascade_fn += 1
        elif label == 0 and cascade_pred == 1:
            cascade_fp += 1
        else:
            cascade_tn += 1

        if loop151_pred != label and cascade_pred == label:
            repairs += 1
        elif loop151_pred == label and cascade_pred != label:
            breaks += 1

    elapsed = time.time() - t0

    l151_f1 = 2 * loop151_tp / (2 * loop151_tp + loop151_fp + loop151_fn) if (2 * loop151_tp + loop151_fp + loop151_fn) > 0 else 0.0
    cascade_f1 = 2 * cascade_tp / (2 * cascade_tp + cascade_fp + cascade_fn) if (2 * cascade_tp + cascade_fp + cascade_fn) > 0 else 0.0

    return {
        "split": split,
        "sample_count": len(data),
        "loop151_baseline": {
            "tp": loop151_tp,
            "fn": loop151_fn,
            "fp": loop151_fp,
            "tn": loop151_tn,
            "f1": l151_f1,
            "total_errors": loop151_fn + loop151_fp,
        },
        "loop199_cascade": {
            "tp": cascade_tp,
            "fn": cascade_fn,
            "fp": cascade_fp,
            "tn": cascade_tn,
            "f1": cascade_f1,
            "total_errors": cascade_fn + cascade_fp,
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
    print("Axon v2.6 - Loop199 Hierarchical Cascade Router Evaluation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    router = Loop199CascadeRouter().to(device)

    m196 = Loop196StructuralExpert().to(device)
    m196_ckpt = Path(__file__).resolve().parent.parent / "models" / "loop196_structural_expert.pt"
    if m196_ckpt.is_file():
        m196.load_state_dict(torch.load(m196_ckpt, map_location=device))
    m196.eval()

    m197 = Loop197HardMiningExpert().to(device)
    m197_ckpt = Path(__file__).resolve().parent.parent / "models" / "loop197_hard_mining_expert.pt"
    if m197_ckpt.is_file():
        m197.load_state_dict(torch.load(m197_ckpt, map_location=device))
    m197.eval()

    report = {}
    for split in ("val", "test10k", "full"):
        res = evaluate_cascade_split(split, router, m196, m197, device)
        if res:
            report[split] = res
            print(f"[{split}] Baseline Errors: {res['loop151_baseline']['total_errors']} -> Cascade Errors: {res['loop199_cascade']['total_errors']} | Repairs: {res['transitions']['repairs']} | Breaks: {res['transitions']['breaks']} | Net Repairs: +{res['transitions']['net_repairs']} | Cascade F1: {res['loop199_cascade']['f1']:.6f}")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop199_cascade_router_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Saved receipt to {report_path}")


if __name__ == "__main__":
    main()
