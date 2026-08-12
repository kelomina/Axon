#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 813k 语料的同一 seed-42 划分上评估任意 checkpoint 的 val/test 完整指标（F1/召回/误报率/AUC）。

与 eval_benign_misrate.py 互补：它只统计良性误判比例，这里给出完整分类指标，
用于核对 compile 重训是否不降低精度。

用法：python scripts/eval_full_test_metrics.py <checkpoint.pt> [--compile]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split, SubDataset  # noqa: E402
from model import AxonMalwareModel  # noqa: E402

TRUNCATE_BYTE_LENGTH = 4096
BATCH_SIZE = 64
NUM_WORKERS = 8


class _TruncatedByteDataset(torch.utils.data.Dataset):
    def __init__(self, base, max_len: int = TRUNCATE_BYTE_LENGTH):
        self.base = base
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        byte_seq = item[0]
        if byte_seq.shape[0] > self.max_len:
            byte_seq = byte_seq[: self.max_len]
        return (byte_seq,) + tuple(item[1:])


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_compile = "--compile" in sys.argv
    skip_test = "--skip-test" in sys.argv
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
    if not args or not Path(args[0]).exists():
        raise SystemExit("usage: eval_full_test_metrics.py <checkpoint.pt> [--compile] [--skip-test] [--limit=N]")
    ckpt_path = Path(args[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]
    config = AxonExperimentConfig.from_dict(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    try:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    dataset = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"),
        cache_dir=str(PROJECT_ROOT / "data" / ".cache"),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=None,
        axon_config=config,
    )
    label_arr = np.array(dataset.label_list)
    print(f"[Dataset] {len(dataset):,}  benign={int((label_arr == 0).sum()):,}")

    model = AxonMalwareModel(config)
    sd = ckpt["model_state_dict"]
    # torch.compile 训练保存的 checkpoint 键带 _orig_mod. 前缀，剥掉
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
        print("[eval] stripped _orig_mod. prefix from state_dict")
    model.load_state_dict(sd)
    model.to(device).eval()
    if do_compile:
        model = torch.compile(model, dynamic=True)
        print("[eval] torch.compile enabled")

    _, val_ds, test_ds = create_stratified_split(
        dataset, val_ratio=0.10, test_ratio=0.20, seed=42, axon_config=config)
    print(f"[Split] val={len(val_ds):,} test={len(test_ds):,}")

    # --limit=N：对 val/test 做确定性等距子采样（跨全划分，避免只取前段），得到快速信号
    def _subsample(sub_ds, n):
        if n is None or len(sub_ds) <= n:
            return sub_ds
        stride = len(sub_ds) // n
        return SubDataset(sub_ds.base_dataset, sub_ds.indices[::max(stride, 1)])

    val_ds = _subsample(val_ds, limit)
    if not skip_test:
        test_ds = _subsample(test_ds, limit)
    if limit is not None:
        test_desc = "n/a" if skip_test else f"{len(test_ds):,}"
        print(f"[subsampled] val={len(val_ds):,} test={test_desc}")

    def eval_split(sub_ds, name):
        ds = _TruncatedByteDataset(sub_ds, TRUNCATE_BYTE_LENGTH)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
            persistent_workers=NUM_WORKERS > 0,
        )
        all_prob = np.zeros(len(ds), dtype=np.float32)
        with torch.no_grad():
            for step, (byte_seq, pe, stat, _) in enumerate(loader):
                out = model(byte_seq.to(device), pe.to(device), stat.to(device))
                p = torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
                all_prob[step * BATCH_SIZE: step * BATCH_SIZE + len(p)] = p
                if step % 200 == 0 and step > 0:
                    print(f"  [{name} {step} batches]")
        y = None
        base = getattr(sub_ds, "base_dataset", None)
        if base is not None and hasattr(base, "label_list"):
            y = np.asarray([base.label_list[i] for i in sub_ds.indices], dtype=np.int64)
        elif hasattr(sub_ds, "label_list"):
            y = np.asarray(sub_ds.label_list, dtype=np.int64)
        if y is None:
            print(f"[{name}] no labels")
            return
        # threshold 0.5 判定
        pred = (all_prob >= 0.5).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(tp + fn, 1)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        # AUC (按样本概率，非 batch)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, all_prob))
        print(f"[{name}] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} "
              f"AUC={auc:.4f} | FP={fp} FN={fn} | FPR={fpr:.4f} FNR={fnr:.4f}")
        # 良性误判率（与 eval_benign_misrate.py 同口径）：良性样本 prob>=阈值 的比例
        ben = all_prob[y == 0]
        n_ben = int(ben.size)
        if n_ben > 0:
            fracs = {th: float((ben >= th).mean()) for th in (0.5, 0.7, 0.9)}
            print(f"[{name}] benign n={n_ben}  >=0.5: {fracs[0.5]*100:.3f}%  "
                  f">=0.7: {fracs[0.7]*100:.3f}%  >=0.9: {fracs[0.9]*100:.3f}%")

    eval_split(val_ds, "val")
    if not skip_test:
        eval_split(test_ds, "test")


if __name__ == "__main__":
    main()
