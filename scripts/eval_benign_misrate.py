#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""难例重训后的基座良性误判率验证（val + test，快速，~10min）。

统计新基座模型在 val/test 良性上 base_prob>=0.5 / >=0.7 / >=0.9 的比例，
对比旧模型基线（val 良性 6.95%>=0.5，0.95%>=0.9）。若难例加权有效，比例应下降。

用法：python scripts/eval_benign_misrate.py <checkpoint.pt>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split  # noqa: E402
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
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit(f"usage: python eval_benign_misrate.py <checkpoint.pt>  ({ckpt_path})")

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
    print(f"[Dataset] {len(dataset):,}  benign={int((label_arr==0).sum()):,}")

    model = AxonMalwareModel(config)
    sd = ckpt["model_state_dict"]
    # torch.compile 训练保存的 checkpoint 键带 _orig_mod. 前缀（OptimizedModule 包装泄漏），剥掉
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
        print("[eval] stripped _orig_mod. prefix from state_dict")
    model.load_state_dict(sd)
    model.to(device).eval()
    if "--compile" in sys.argv:
        model = torch.compile(model, dynamic=True)
        print("[eval] torch.compile enabled")

    # 基座 7:1:2 划分（seed 42），与训练一致
    _, val_ds, test_ds = create_stratified_split(
        dataset, val_ratio=0.10, test_ratio=0.20, seed=42, axon_config=config)
    print(f"[Split] val={len(val_ds):,} test={len(test_ds):,}")

    def eval_split(sub_ds, name):
        ds = _TruncatedByteDataset(sub_ds, TRUNCATE_BYTE_LENGTH)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
            persistent_workers=NUM_WORKERS > 0,
        )
        probs = np.zeros(len(ds), dtype=np.float32)
        with torch.no_grad():
            for step, (byte_seq, pe, stat, label) in enumerate(loader):
                byte_seq = byte_seq.to(device)
                pe = pe.to(device)
                stat = stat.to(device)
                out = model(byte_seq, pe, stat)
                p = torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
                probs[step * BATCH_SIZE: step * BATCH_SIZE + len(p)] = p
                if step % 100 == 0 and step > 0:
                    print(f"  [{name} {step} batches]")
        y = None
        base = getattr(sub_ds, "base_dataset", None)
        if base is not None and hasattr(base, "label_list"):
            y = np.asarray([base.label_list[i] for i in sub_ds.indices], dtype=np.int64)
        elif hasattr(sub_ds, "label_list"):
            y = np.asarray(sub_ds.label_list, dtype=np.int64)
        ben_mask = y == 0
        n_ben = int(ben_mask.sum())
        b = probs[ben_mask]
        gt5 = int((b >= 0.5).sum()); gt7 = int((b >= 0.7).sum()); gt9 = int((b >= 0.9).sum())
        print(f"[{name}] benign n={n_ben}  base>=0.5: {gt5} ({gt5/max(n_ben,1)*100:.3f}%)  "
              f">=0.7: {gt7} ({gt7/max(n_ben,1)*100:.3f}%)  >=0.9: {gt9} ({gt9/max(n_ben,1)*100:.3f}%)")
        return {"name": name, "n_benign": n_ben,
                "p50": float(np.median(b)), "p90": float(np.percentile(b, 90)),
                "ge05": float((b >= 0.5).mean()), "ge07": float((b >= 0.7).mean()),
                "ge09": float((b >= 0.9).mean())}

    rv = eval_split(val_ds, "val")
    rt = eval_split(test_ds, "test")

    print("\n=== 旧基线（full_739k_benign best_model_739k.pt） ===")
    print("  val  良性: 6.95% >=0.5, 0.95% >=0.9")
    print("  test 良性: 7.16% >=0.5, 0.90% >=0.9")
    out = {"checkpoint": str(ckpt_path), "val": rv, "test": rt}
    (PROJECT_ROOT / "reports" / "full_739k_benign_hardneg" / "benign_misrate_eval.json").write_text(
        __import__("json").dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
