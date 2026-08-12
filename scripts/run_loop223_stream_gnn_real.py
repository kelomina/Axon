"""Loop223: StreamGNN Fusion on REAL Data.

Loads REAL pe_features (256-d) + byte_sequence (8192-d) from project .npz cache,
trains Loop222 StreamGNN architecture, and reports REAL F1.
"""

from __future__ import annotations

import glob
import json
import time
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop222_stream_gnn_fusion import Loop222StreamGNNFusion


def load_real_data(cache_dir: Path, max_samples: int = 20000):
    """Load real PE features and labels from .npz cache files."""
    files = sorted(glob.glob(str(cache_dir / "*.npz")))
    print(f"[Data] Found {len(files)} .npz files in {cache_dir}")

    if len(files) > max_samples:
        random.seed(42)
        files = random.sample(files, max_samples)
        print(f"[Data] Sampled {max_samples} files for training")

    pe_feats_list = []
    byte_seq_list = []
    labels_list = []
    loaded = 0

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            pe_feat = d["pe_features"].astype(np.float32)       # (256,)
            byte_seq = d["byte_sequence"].astype(np.float32)     # (8192,)
            label = int(d["label"])

            pe_feats_list.append(pe_feat)
            byte_seq_list.append(byte_seq)
            labels_list.append(label)
            loaded += 1
        except Exception:
            continue

    pe_feats = np.stack(pe_feats_list, axis=0)      # (N, 256)
    byte_seqs = np.stack(byte_seq_list, axis=0)      # (N, 8192)
    labels = np.array(labels_list, dtype=np.int64)   # (N,)

    print(f"[Data] Loaded {loaded} samples | Label 0: {(labels==0).sum()} | Label 1: {(labels==1).sum()}")
    return pe_feats, byte_seqs, labels


def reshape_bytes_to_chunks(byte_seqs: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    """Reshape (N, 8192) byte sequences into (N, 4, chunk_size) chunks."""
    N = byte_seqs.shape[0]
    # Split 8192 bytes into 4 chunks of 2048
    chunks = byte_seqs.reshape(N, 4, chunk_size)
    return chunks


def build_pe_graph_nodes(pe_feats: np.ndarray, num_nodes: int = 5) -> np.ndarray:
    """Build graph nodes from pe_features (256-d) -> (N, 5, ~51)."""
    N, D = pe_feats.shape
    node_dim = D // num_nodes  # 256 // 5 = 51, remainder 1
    nodes = []
    for i in range(num_nodes):
        start = i * node_dim
        end = start + node_dim if i < num_nodes - 1 else D
        nodes.append(pe_feats[:, start:end])

    # Pad shorter nodes to same dim
    max_d = max(n.shape[1] for n in nodes)
    padded = []
    for n in nodes:
        if n.shape[1] < max_d:
            pad = np.zeros((N, max_d - n.shape[1]), dtype=np.float32)
            n = np.concatenate([n, pad], axis=1)
        padded.append(n)

    return np.stack(padded, axis=1)  # (N, 5, node_dim)


def run_loop223():
    print("=" * 70)
    print("Axon v2.6 - Loop223 StreamGNN on REAL Data")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"

    # Load REAL data
    pe_feats, byte_seqs, labels = load_real_data(cache_dir, max_samples=20000)

    # Reshape bytes into 4 chunks
    chunk_size = 2048
    byte_chunks = reshape_bytes_to_chunks(byte_seqs, chunk_size=chunk_size)  # (N, 4, 2048)

    # Build graph nodes from PE features
    node_dim = 52  # ceil(256/5)
    graph_nodes = build_pe_graph_nodes(pe_feats, num_nodes=5)  # (N, 5, 52)

    # Train/Val split (80/20 stratified)
    N = len(labels)
    indices = np.arange(N)
    np.random.shuffle(indices)

    # Stratified split
    idx_0 = indices[labels[indices] == 0]
    idx_1 = indices[labels[indices] == 1]
    n_val_0 = len(idx_0) // 5
    n_val_1 = len(idx_1) // 5

    val_idx = np.concatenate([idx_0[:n_val_0], idx_1[:n_val_1]])
    train_idx = np.concatenate([idx_0[n_val_0:], idx_1[n_val_1:]])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    print(f"[Split] Train: {len(train_idx)} | Val: {len(val_idx)}")

    # Convert to tensors
    train_chunks = torch.tensor(byte_chunks[train_idx], dtype=torch.float32)
    train_nodes = torch.tensor(graph_nodes[train_idx], dtype=torch.float32)
    train_y = torch.tensor(labels[train_idx], dtype=torch.long)

    val_chunks = torch.tensor(byte_chunks[val_idx], dtype=torch.float32)
    val_nodes = torch.tensor(graph_nodes[val_idx], dtype=torch.float32)
    val_y = torch.tensor(labels[val_idx], dtype=torch.long)

    # Model
    model = Loop222StreamGNNFusion(
        chunk_dim=chunk_size,
        node_dim=node_dim,
        hidden_dim=192,
        num_heads=4,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    print(f"[Model] Loop222StreamGNNFusion | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] chunk_dim={chunk_size}, node_dim={node_dim}")

    batch_size = 32
    epochs = 15
    best_val_f1 = 0.0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = len(train_idx) // batch_size

        perm = torch.randperm(len(train_idx))

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            bc = train_chunks[idx].to(device)
            bn = train_nodes[idx].to(device)
            by = train_y[idx].to(device)

            optimizer.zero_grad()
            logits = model(bc, bn)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
        val_n_batches = len(val_idx) // batch_size

        with torch.no_grad():
            for i in range(val_n_batches):
                bc = val_chunks[i * batch_size : (i + 1) * batch_size].to(device)
                bn = val_nodes[i * batch_size : (i + 1) * batch_size].to(device)
                by = val_y[i * batch_size : (i + 1) * batch_size].to(device)

                logits = model(bc, bn)
                preds = logits.argmax(dim=-1)

                val_tp += ((preds == 1) & (by == 1)).sum().item()
                val_fp += ((preds == 1) & (by == 0)).sum().item()
                val_fn += ((preds == 0) & (by == 1)).sum().item()
                val_tn += ((preds == 0) & (by == 0)).sum().item()

        val_total = val_tp + val_fp + val_fn + val_tn
        val_acc = (val_tp + val_tn) / val_total if val_total > 0 else 0.0
        val_prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
        val_f1 = 2 * val_tp / (2 * val_tp + val_fp + val_fn) if (2 * val_tp + val_fp + val_fn) > 0 else 0.0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            ckpt_path = proj_dir / "models" / "loop223_stream_gnn_real.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val P: {val_prec:.4f} | Val R: {val_recall:.4f} | Val F1: {val_f1:.4f} | TP:{val_tp} FP:{val_fp} FN:{val_fn} TN:{val_tn}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"[Loop223 REAL DATA RESULT] Best Val F1: {best_val_f1:.6f} @ Epoch {best_epoch}")
    print(f"Training completed in {elapsed:.2f}s")
    print(f"{'='*70}")

    receipt = {
        "schema": "axon_loop223_stream_gnn_real_receipt_v1",
        "loop_id": "Loop223",
        "model_architecture": "Loop222StreamGNNFusion",
        "data_source": "REAL .npz cache (pe_features + byte_sequence)",
        "training": {
            "epochs": epochs,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "batch_size": batch_size,
            "elapsed_seconds": round(elapsed, 2),
            "device": str(device),
        },
        "results": {
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
        },
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop223_stream_gnn_real_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Saved receipt to {report_path}")


if __name__ == "__main__":
    run_loop223()
