#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""强化版 Stage-2：完整特征 + HGB 集成（杠杆①+②）。

特征：base-prob 派生(6) + content_pe_v1(100) + content_pe_v2(182) + content_string(43) = 331。
模型：3 个 HGB（不同 seed）集成取平均概率。
协议：VAL 训练 + VAL 选阈值，TEST 一次性评估。
输入：base_prob/、content_pe_v1/、content_v2string/（extract_content_v2string_739k.py 产物）。
输出：stage2_report.json 追加 "stage2_v2" 段 + 控制台对比。
"""
from __future__ import annotations

import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

BASE_DIR = PROJECT_ROOT / "reports" / "full_739k" / "base_prob"
CONTENT_V1_DIR = PROJECT_ROOT / "reports" / "full_739k" / "content_pe_v1"
V2_DIR = PROJECT_ROOT / "reports" / "full_739k" / "content_v2string"
CERT_DIR = PROJECT_ROOT / "reports" / "full_739k" / "content_cert"
REPORT_JSON = PROJECT_ROOT / "reports" / "full_739k" / "stage2_report.json"
CHECKPOINT = PROJECT_ROOT / "models" / "full_739k" / "best_model_739k.pt"
SEEDS = (0, 1, 2)


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def main() -> None:
    t0 = time.time()
    print("=== Stage-2 v2: derived + content_v1 (+v2/string if ready) + HGB ensemble ===")

    base_prob = load_chunks(BASE_DIR)
    content_v1 = load_chunks(CONTENT_V1_DIR)
    v2string_ready = (V2_DIR / "val.npy").exists()
    cert_ready = (CERT_DIR / "val.npy").exists()
    if v2string_ready:
        v2 = np.load(V2_DIR / "val.npy").astype(np.float32)
        st = np.load(V2_DIR / "test.npy").astype(np.float32)
        with open(V2_DIR / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        val_idx = np.asarray(meta["val_indices"], dtype=np.int64)
        test_idx = np.asarray(meta["test_indices"], dtype=np.int64)
    else:
        print("[warn] content_v2string not ready; falling back to derived + content_v1")
        from dataset import FeatureCacheDataset, create_stratified_split
        from config import AxonExperimentConfig
        import torch
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        raw = ckpt["config"]
        config = AxonExperimentConfig.from_dict(raw) if isinstance(raw, dict) else raw
        ds = FeatureCacheDataset(
            data_dir=str(PROJECT_ROOT / "data"), cache_dir=str(PROJECT_ROOT / "data" / ".cache"),
            max_byte_length=config.max_byte_length, pe_feature_dim=config.pe_feature_dim,
            stat_feature_dim=config.stat_feature_dim, max_samples_per_class=None, axon_config=config)
        _, val_ds, test_ds = create_stratified_split(ds, val_ratio=0.10, test_ratio=0.20, seed=42, axon_config=config)
        val_idx = np.asarray(val_ds.indices, dtype=np.int64)
        test_idx = np.asarray(test_ds.indices, dtype=np.int64)
        v2 = st = None
    if cert_ready:
        cert_v = np.load(CERT_DIR / "val.npy").astype(np.float32)
        cert_t = np.load(CERT_DIR / "test.npy").astype(np.float32)
    else:
        cert_v = cert_t = None
    print(f"base={base_prob.shape} v1={content_v1.shape} v2string_ready={v2string_ready} cert_ready={cert_ready}")

    with open(CONTENT_V1_DIR / "meta.csv", encoding="utf-8") as f:
        labels_all = np.asarray([int(r["label"]) for r in csv.DictReader(f)], dtype=np.int64)
    yv = labels_all[val_idx]
    yt = labels_all[test_idx]

    # ---- 特征矩阵（顺序对齐 val/test）----
    def feat(p_vec, v1, v2s, cert):
        parts = [derived(p_vec), v1]
        if v2s is not None:
            parts.append(v2s)
        if cert is not None:
            parts.append(cert)
        return np.column_stack(parts).astype(np.float32)

    Xv = feat(base_prob[val_idx], content_v1[val_idx], v2, cert_v)
    Xt = feat(base_prob[test_idx], content_v1[test_idx], st, cert_t)
    print(f"Xv={Xv.shape} Xt={Xt.shape}  ({Xv.shape[1]} features)")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score

    # ---- 三 seed HGB 集成 ----
    clfs = []
    for seed in SEEDS:
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        clf.fit(Xv, yv)
        clfs.append(clf)
        print(f"  [seed {seed}] trained")
    pv = np.mean([c.predict_proba(Xv)[:, 1] for c in clfs], axis=0)
    pt = np.mean([c.predict_proba(Xt)[:, 1] for c in clfs], axis=0)

    best = (0.0, 0.5)
    for t in [x / 100 for x in range(20, 90)]:
        f1 = f1_score(yv, (pv >= t).astype(int))
        if f1 > best[0]:
            best = (f1, t)
    thr = best[1]
    pred = (pt >= thr).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f1t = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0

    print(f"\n[val] bestF1={best[0]:.5f}@t={thr:.2f}")
    print(f"[TEST] F1={f1t:.5f} P={p_:.4f} R={r_:.4f} errors={fp+fn} "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")
    print(f"[对比] 上一版(100特征) TEST F1=0.99282 errors=1647")

    rep = json.loads(REPORT_JSON.read_text(encoding="utf-8")) if REPORT_JSON.exists() else {}
    rep["stage2_v2"] = {
        "features": "base_prob_derived6 + content_v1_100 + content_v2_182 + string_43 = 331",
        "model": f"HGB ensemble seeds={list(SEEDS)}",
        "val_best_f1": best[0], "threshold": float(thr),
        "test_f1": f1t, "precision": p_, "recall": r_,
        "errors": int(fp + fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "prev_stage2_errors": 1647, "prev_stage2_f1": 0.99282,
        "elapsed_sec": time.time() - t0,
    }
    REPORT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {REPORT_JSON}")
    print(f"[done] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
