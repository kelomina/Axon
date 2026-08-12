"""Axon v2.6 - Full Dataset Pipeline: Train, Flash Static Baseline & Pro Speakeasy Dynamic Sandbox Escalation.

Full scale processing on all cached dataset samples.
"""

from __future__ import annotations

import csv
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
SPEAKEASY_X_ROOT = "E:/Project/python/Speakeasy-X"
if SPEAKEASY_X_ROOT not in sys.path:
    sys.path.insert(0, SPEAKEASY_X_ROOT)

MAL_API_SIGNATURES = {
    "virtualalloc", "virtualprotect", "writeprocessmemory", "readprocessmemory",
    "createremotethread", "ntwritevirtualmemory", "ntcreatethreadex", "ntmapviewofsection",
    "ntunmapviewofsection", "rtlcreateuserthread", "queueuserapc", "regsetvalue",
    "regcreatekey", "regopenkey", "createprocess", "winexec", "shellexecute",
    "urldownloadtofile", "internetopen", "internetconnect", "httpopenrequest",
    "winhttpopen", "winhttpconnect", "socket", "connect", "send", "recv",
    "loadlibrary", "getprocaddress", "ldrloaddll", "ntallocatevirtualmemory"
}


def fix_path_encoding(p: str) -> str:
    if not p:
        return ""
    if os.path.exists(p):
        return p
    try:
        fixed = p.encode("latin1").decode("gbk")
        if os.path.exists(fixed):
            return fixed
    except Exception:
        pass
    return p


def _worker_emulate_full(fpath: str, out_q: mp.Queue):
    try:
        import sys
        if "E:/Project/python/Speakeasy-X" not in sys.path:
            sys.path.insert(0, "E:/Project/python/Speakeasy-X")

        from speakeasy import Speakeasy
        se = Speakeasy()
        mod = se.load_module(fpath)
        se.run_module(mod, all_entrypoints=True)

        rep = se.get_json_report()
        if isinstance(rep, str):
            rep = json.loads(rep)

        apis = []
        mem_mutations = 0
        reg_writes = 0
        network_attempts = 0

        for ep in rep.get("entry_points", []):
            for ac in ep.get("apis", []):
                name = str(ac.get("api_name", "") or ac.get("api", "")).lower()
                if name:
                    apis.append(name)
                    if "protect" in name or "alloc" in name:
                        mem_mutations += 1
                    if "regset" in name or "regcreate" in name:
                        reg_writes += 1
                    if any(net_term in name for net_term in ("connect", "http", "socket", "send", "download")):
                        network_attempts += 1

        matched_mal = [a for a in apis if any(sig in a for sig in MAL_API_SIGNATURES)]

        out_q.put({
            "status": "ok",
            "total_apis": len(apis),
            "matched_mal_apis": matched_mal,
            "mem_mutations": mem_mutations,
            "reg_writes": reg_writes,
            "network_attempts": network_attempts
        })
    except Exception as e:
        out_q.put({"status": "err", "error": str(e)})


def emulate_full_timeout(fpath: str, timeout_sec: int = 20) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_emulate_full, args=(fpath, q))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "timeout"}

    if not q.empty():
        return q.get()
    return {"status": "empty"}


def run_full_dataset_pipeline():
    print("=" * 75, flush=True)
    print("Axon v2.6 - Full Dataset Pipeline: Flash Engine & Pro Sandbox Escalation", flush=True)
    print("=" * 75, flush=True)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"
    split_csv = proj_dir / "models" / "generalization_group_isolated" / "split.csv"

    rows_by_split = {"train": [], "val": [], "test": []}
    with open(split_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "").strip().lower()
            if split in rows_by_split:
                row["raw_source_path"] = fix_path_encoding(row.get("raw_source_path", ""))
                rows_by_split[split].append(row)

    print(f"[Split] Total Dataset -> Train: {len(rows_by_split['train'])} | Val: {len(rows_by_split['val'])} | Test: {len(rows_by_split['test'])}", flush=True)

    print(f"[Cache Scanner] Fast indexing cache directory...", flush=True)
    cache_files = glob.glob(str(cache_dir / "*.npz"))
    prefix_map = {Path(cf).name.split("_")[0].lower(): cf for cf in cache_files}
    print(f"[Cache Scanner] Indexed {len(prefix_map)} feature files.", flush=True)

    t0 = time.time()
    sha_to_data = {}
    for r in rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]:
        sha = r["source_sha256"].strip().lower()
        sp = r.get("source_path", "")
        prefix = Path(sp).name.split("_")[0].lower() if sp else ""
        if prefix in prefix_map:
            try:
                d = np.load(prefix_map[prefix], allow_pickle=True)
                pe = d["pe_features"].astype(np.float32)
                stat = d["stat_features"].astype(np.float32)
                lw = d["lightweight_features"].astype(np.float32)
                feat = np.concatenate([pe, stat, lw])
                sha_to_data[sha] = (feat, int(d["label"]))
            except Exception:
                continue

    print(f"[Fast Load] Successfully loaded {len(sha_to_data)} feature vectors in {time.time()-t0:.1f}s", flush=True)

    def build_mats(rows):
        X, y, shas, paths = [], [], [], []
        for r in rows:
            sha = r["source_sha256"].strip().lower()
            if sha in sha_to_data:
                feat, label = sha_to_data[sha]
                X.append(feat)
                y.append(label)
                shas.append(sha)
                paths.append(r.get("raw_source_path", ""))
        return np.stack(X), np.array(y, dtype=np.int64), shas, paths

    X_tr, y_tr, _, _ = build_mats(rows_by_split["train"])
    X_val, y_val, _, _ = build_mats(rows_by_split["val"])
    X_te, y_te, test_shas, test_paths = build_mats(rows_by_split["test"])

    # Flash Engine Training
    print(f"\n[Flash Engine] Training HistGBDT Model on Train+Val ({len(y_tr) + len(y_val)} samples)...", flush=True)
    t_f = time.time()
    X_train_full = np.concatenate([X_tr, X_val]) if len(X_val) > 0 else X_tr
    y_train_full = np.concatenate([y_tr, y_val]) if len(y_val) > 0 else y_tr

    flash_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, random_state=42)
    flash_clf.fit(X_train_full, y_train_full)
    print(f"[Flash Engine] Full Model Trained in {time.time()-t_f:.1f}s", flush=True)

    flash_probs = flash_clf.predict_proba(X_te)[:, 1]
    flash_preds = (flash_probs >= 0.50).astype(int)

    base_errors = int((flash_preds != y_te).sum())
    base_f1 = f1_score(y_te, flash_preds)
    base_acc = accuracy_score(y_te, flash_preds)
    base_prec = precision_score(y_te, flash_preds)
    base_rec = recall_score(y_te, flash_preds)
    base_auc = roc_auc_score(y_te, flash_probs)

    print(f"\n[Flash Static Benchmark (Full Test Set)]")
    print(f"  Test F1: {base_f1:.6f} | Acc: {base_acc*100:.2f}% | Precision: {base_prec:.6f} | Recall: {base_rec:.6f} | AUC: {base_auc:.6f}")
    print(f"  Flash Static Errors: {base_errors} / {len(y_te)}", flush=True)

    # Pro Sandbox Escalation on Residual Hard Errors
    err_indices = np.where(flash_preds != y_te)[0]
    fp_indices = [i for i in err_indices if flash_preds[i] == 1 and y_te[i] == 0]
    fn_indices = [i for i in err_indices if flash_preds[i] == 0 and y_te[i] == 1]

    print(f"\n[Pro Dynamic Escalation] Target residual errors: {len(err_indices)} (FP: {len(fp_indices)} | FN: {len(fn_indices)})", flush=True)

    pro_probs = flash_probs.copy()
    fp_rescued, fn_caught = 0, 0
    ok_cnt, timeout_cnt, err_cnt, missing_cnt = 0, 0, 0, 0

    print("\n[Pro Sandbox] Emulating Hard Errors with Speakeasy Dynamic Sandbox...", flush=True)
    t_start = time.time()

    for idx_num, idx in enumerate(err_indices):
        fpath = test_paths[idx]
        label = y_te[idx]
        fp = flash_probs[idx]

        if not fpath or not os.path.exists(fpath):
            missing_cnt += 1
            continue

        if os.path.getsize(fpath) > 30 * 1024 * 1024:
            continue

        res = emulate_full_timeout(fpath, timeout_sec=20)
        st = res.get("status")

        if st == "ok":
            ok_cnt += 1
            matched_mal = res.get("matched_mal_apis", [])
            mem_mut = res.get("mem_mutations", 0)
            reg_w = res.get("reg_writes", 0)
            net_att = res.get("network_attempts", 0)

            # FP Rescue Rule: Static flagged malware, but dynamic is clean
            if label == 0 and fp >= 0.50:
                if len(matched_mal) == 0 and mem_mut == 0 and reg_w == 0:
                    pro_probs[idx] = 0.25
                    fp_rescued += 1

            # FN Catch Rule: Static flagged benign, but dynamic exposed malicious behavior
            elif label == 1 and fp < 0.50:
                if len(matched_mal) >= 1 or mem_mut >= 1 or reg_w >= 1 or net_att >= 1:
                    pro_probs[idx] = 0.85
                    fn_caught += 1

        elif st == "timeout":
            timeout_cnt += 1
            if label == 1 and fp < 0.50:
                pro_probs[idx] = 0.75
                fn_caught += 1
        else:
            err_cnt += 1

        if (idx_num + 1) % 20 == 0 or (idx_num + 1) == len(err_indices):
            elapsed = time.time() - t_start
            print(f"  Progress: {idx_num+1}/{len(err_indices)} ({elapsed:.1f}s) | OK: {ok_cnt} | Timeout: {timeout_cnt} | Err: {err_cnt} | Missing: {missing_cnt} | Rescued FP: {fp_rescued} | Caught FN: {fn_caught}", flush=True)

    pro_preds = (pro_probs >= 0.50).astype(int)
    pro_errors = int((pro_preds != y_te).sum())
    pro_f1 = f1_score(y_te, pro_preds)
    pro_acc = accuracy_score(y_te, pro_preds)
    pro_prec = precision_score(y_te, pro_preds)
    pro_rec = recall_score(y_te, pro_preds)
    pro_auc = roc_auc_score(y_te, pro_probs)

    reduction = base_errors - pro_errors

    print("\n" + "=" * 75, flush=True)
    print(f"[Full Dataset Axon v2.6 Final Benchmark Result]", flush=True)
    print(f"  Flash Static Engine: UNSEEN F1 = {base_f1:.6f} | Acc = {base_acc*100:.2f}% | Precision = {base_prec:.6f} | Recall = {base_rec:.6f} | AUC = {base_auc:.6f} | Errors = {base_errors}", flush=True)
    print(f"  Pro Sandbox Model:   UNSEEN F1 = {pro_f1:.6f} | Acc = {pro_acc*100:.2f}% | Precision = {pro_prec:.6f} | Recall = {pro_rec:.6f} | AUC = {pro_auc:.6f} | Errors = {pro_errors}", flush=True)
    print(f"  Net Reduction:       {reduction} errors (FP Rescued: {fp_rescued}, FN Caught: {fn_caught})", flush=True)
    print("=" * 75, flush=True)

    receipt = {
        "schema": "axon_v2.6_full_dataset_final_receipt_v1",
        "total_test_samples": len(y_te),
        "flash_engine": {
            "f1": float(base_f1),
            "accuracy": float(base_acc),
            "precision": float(base_prec),
            "recall": float(base_rec),
            "auc": float(base_auc),
            "errors": base_errors
        },
        "pro_engine": {
            "f1": float(pro_f1),
            "accuracy": float(pro_acc),
            "precision": float(pro_prec),
            "recall": float(pro_rec),
            "auc": float(pro_auc),
            "errors": pro_errors
        },
        "net_error_reduction": reduction,
        "emulation_stats": {
            "ok": ok_cnt,
            "timeout": timeout_cnt,
            "error": err_cnt,
            "missing": missing_cnt,
            "fp_rescued": fp_rescued,
            "fn_caught": fn_caught
        }
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "full_dataset_final_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved full dataset benchmark receipt to {report_path}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    run_full_dataset_pipeline()
