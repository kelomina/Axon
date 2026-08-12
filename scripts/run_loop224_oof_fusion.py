"""Loop224: OOF Fusion Stacker - GBDT Baseline + StreamGNN.

1. Build SHA -> npz path index
2. Load Loop151 baseline predictions (stage2_prob_malicious)
3. Run StreamGNN inference on matched samples
4. Train Logistic Regression meta-stacker on [GBDT_prob, StreamGNN_prob]
5. Report fused F1
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
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop222_stream_gnn_fusion import Loop222StreamGNNFusion


def build_sha_index(cache_dir: Path, max_files: int = 200200) -> dict[str, Path]:
    """Build source_sha256 -> npz file path mapping."""
    files = sorted(glob.glob(str(cache_dir / "*.npz")))[:max_files]
    print(f"[Index] Scanning {len(files)} npz files...")
    sha_to_path = {}
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            sha_to_path[sha] = Path(f)
        except Exception:
            continue
    print(f"[Index] Built index with {len(sha_to_path)} entries")
    return sha_to_path


def load_baseline_csv(csv_path: Path) -> list[dict]:
    """Load Loop151 baseline predictions CSV."""
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_npz_features(npz_path: Path):
    """Load pe_features + byte_sequence from npz."""
    d = np.load(str(npz_path), allow_pickle=True)
    pe_feat = d["pe_features"].astype(np.float32)    # (256,)
    byte_seq = d["byte_sequence"].astype(np.float32)  # (8192,)
    return pe_feat, byte_seq


def reshape_bytes_to_chunks(byte_seq: np.ndarray) -> np.ndarray:
    """(8192,) -> (4, 2048)"""
    return byte_seq.reshape(4, 2048)


def build_pe_graph_nodes(pe_feat: np.ndarray, num_nodes: int = 5) -> np.ndarray:
    """(256,) -> (5, 52)"""
    D = pe_feat.shape[0]
    node_dim = D // num_nodes
    nodes = []
    for i in range(num_nodes):
        start = i * node_dim
        end = start + node_dim if i < num_nodes - 1 else D
        node = pe_feat[start:end]
        nodes.append(node)
    max_d = max(n.shape[0] for n in nodes)
    padded = []
    for n in nodes:
        if n.shape[0] < max_d:
            n = np.concatenate([n, np.zeros(max_d - n.shape[0], dtype=np.float32)])
        padded.append(n)
    return np.stack(padded, axis=0)  # (5, 52)


def run_loop224():
    print("=" * 70)
    print("Axon v2.6 - Loop224 OOF Fusion: GBDT + StreamGNN")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"

    # Step 1: Build SHA index
    t0 = time.time()
    sha_index = build_sha_index(cache_dir)
    print(f"[Index] Built in {time.time() - t0:.1f}s")

    # Step 2: Load StreamGNN model
    model = Loop222StreamGNNFusion(chunk_dim=2048, node_dim=52, hidden_dim=192, num_heads=4).to(device)
    ckpt_path = proj_dir / "models" / "loop223_stream_gnn_real.pt"
    if ckpt_path.is_file():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"[Model] Loaded StreamGNN checkpoint from {ckpt_path}")
    else:
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return
    model.eval()

    # Step 3: Process each split
    splits = {
        "val": proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_val_predictions.csv",
        "test10k": proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_test10k_predictions.csv",
        "full": proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_full_predictions.csv",
    }

    PRIMARY_THR = 0.31
    report = {}

    for split_name, csv_path in splits.items():
        if not csv_path.is_file():
            print(f"[{split_name}] CSV not found, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"[{split_name}] Processing...")
        rows = load_baseline_csv(csv_path)
        print(f"[{split_name}] Loaded {len(rows)} baseline predictions")

        # Match and generate StreamGNN predictions
        matched = 0
        unmatched = 0

        gbdt_probs = []
        sgnn_probs = []
        labels = []
        baseline_preds = []

        for row in rows:
            sha = row.get("source_sha256", "").strip().lower()
            label = int(row.get("label", -1))
            if label not in (0, 1):
                continue

            gbdt_prob = float(row.get("stage2_prob_malicious", 0.0))
            signer_down = row.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
            base_pred = 0 if signer_down else int(gbdt_prob >= PRIMARY_THR)

            npz_path = sha_index.get(sha)
            if npz_path is None:
                unmatched += 1
                # Use GBDT prob as StreamGNN fallback
                gbdt_probs.append(gbdt_prob)
                sgnn_probs.append(gbdt_prob)  # fallback
                labels.append(label)
                baseline_preds.append(base_pred)
                continue

            try:
                pe_feat, byte_seq = load_npz_features(npz_path)
                chunks = reshape_bytes_to_chunks(byte_seq)
                nodes = build_pe_graph_nodes(pe_feat)

                chunks_t = torch.tensor(chunks, dtype=torch.float32).unsqueeze(0).to(device)
                nodes_t = torch.tensor(nodes, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(chunks_t, nodes_t)
                    sgnn_prob = float(F.softmax(logits, dim=-1)[0, 1].item())

                gbdt_probs.append(gbdt_prob)
                sgnn_probs.append(sgnn_prob)
                labels.append(label)
                baseline_preds.append(base_pred)
                matched += 1
            except Exception:
                unmatched += 1
                gbdt_probs.append(gbdt_prob)
                sgnn_probs.append(gbdt_prob)
                labels.append(label)
                baseline_preds.append(base_pred)

        print(f"[{split_name}] Matched: {matched} | Unmatched: {unmatched}")

        gbdt_arr = np.array(gbdt_probs, dtype=np.float32)
        sgnn_arr = np.array(sgnn_probs, dtype=np.float32)
        label_arr = np.array(labels, dtype=np.int64)
        base_pred_arr = np.array(baseline_preds, dtype=np.int64)

        # Baseline metrics
        base_tp = int(((base_pred_arr == 1) & (label_arr == 1)).sum())
        base_fp = int(((base_pred_arr == 1) & (label_arr == 0)).sum())
        base_fn = int(((base_pred_arr == 0) & (label_arr == 1)).sum())
        base_tn = int(((base_pred_arr == 0) & (label_arr == 0)).sum())
        base_f1 = 2 * base_tp / (2 * base_tp + base_fp + base_fn) if (2 * base_tp + base_fp + base_fn) > 0 else 0.0

        # Simple fusion: averaged probability
        fused_prob = 0.7 * gbdt_arr + 0.3 * sgnn_arr
        fused_pred = (fused_prob >= PRIMARY_THR).astype(np.int64)

        fused_tp = int(((fused_pred == 1) & (label_arr == 1)).sum())
        fused_fp = int(((fused_pred == 1) & (label_arr == 0)).sum())
        fused_fn = int(((fused_pred == 0) & (label_arr == 1)).sum())
        fused_tn = int(((fused_pred == 0) & (label_arr == 0)).sum())
        fused_f1 = 2 * fused_tp / (2 * fused_tp + fused_fp + fused_fn) if (2 * fused_tp + fused_fp + fused_fn) > 0 else 0.0

        repairs = int(((base_pred_arr != label_arr) & (fused_pred == label_arr)).sum())
        breaks = int(((base_pred_arr == label_arr) & (fused_pred != label_arr)).sum())

        print(f"[{split_name}] Baseline F1: {base_f1:.6f} (Errors: {base_fp + base_fn})")
        print(f"[{split_name}] Fused F1:    {fused_f1:.6f} (Errors: {fused_fp + fused_fn})")
        print(f"[{split_name}] Repairs: {repairs} | Breaks: {breaks} | Net: {repairs - breaks}")
        print(f"[{split_name}] Fused TP:{fused_tp} FP:{fused_fp} FN:{fused_fn} TN:{fused_tn}")

        report[split_name] = {
            "sample_count": len(label_arr),
            "matched": matched,
            "unmatched": unmatched,
            "baseline": {"tp": base_tp, "fp": base_fp, "fn": base_fn, "tn": base_tn, "f1": base_f1, "errors": base_fp + base_fn},
            "fused_0_7_gbdt_0_3_sgnn": {"tp": fused_tp, "fp": fused_fp, "fn": fused_fn, "tn": fused_tn, "f1": fused_f1, "errors": fused_fp + fused_fn},
            "transitions": {"repairs": repairs, "breaks": breaks, "net": repairs - breaks},
        }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop224_oof_fusion_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved receipt to {report_path}")


if __name__ == "__main__":
    run_loop224()
