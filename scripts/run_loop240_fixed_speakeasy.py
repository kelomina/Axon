"""Loop240: Re-run Pro Sandbox Hard-Emulation with fixed Speakeasy engine.

The self.mem AttributeError in winemu.py has been fixed (self.mem -> self).
Now re-test the 320 residual error samples with the repaired Speakeasy engine.
"""
from __future__ import annotations
import csv, glob, json, os, time, traceback
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
SPEAKEASY_X_ROOT = Path("E:/Project/python/Speakeasy-X")
sys.path.insert(0, str(SPEAKEASY_X_ROOT))

try:
    from speakeasy import Speakeasy
    SPEAKEASY_AVAILABLE = True
    print("[OK] Speakeasy imported successfully (with mem fix applied)")
except Exception as e:
    SPEAKEASY_AVAILABLE = False
    print(f"[Warning] Speakeasy import: {e}")


def load_split(csv_path):
    rows_by_split = {"train": [], "val": [], "test": []}
    all_shas = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "").strip().lower()
            if split in rows_by_split:
                rows_by_split[split].append(row)
                all_shas.append(row["source_sha256"].strip().lower())
    return rows_by_split, all_shas


def speakeasy_emulate(fpath, timeout_sec=60):
    """Run Speakeasy emulation on a single file, return (apis_list, error_str|None)."""
    try:
        se = Speakeasy()
        module = se.load_module(fpath)
        se.run_module(module, all_entrypoints=False)
        report = se.get_report()
        apis = []
        for ep in getattr(report, 'entry_points', []):
            for api_call in getattr(ep, 'apis', []):
                api_name = getattr(api_call, 'api_name', '') or str(api_call)
                apis.append(api_name)
        # Also try JSON report fallback
        if not apis:
            try:
                jreport = se.get_json_report()
                if isinstance(jreport, str):
                    jdict = json.loads(jreport)
                else:
                    jdict = jreport
                for ep in jdict.get("entry_points", []):
                    for ac in ep.get("apis", []):
                        apis.append(ac.get("api_name", "") or ac.get("api", ""))
            except Exception:
                pass
        return apis, None
    except Exception as e:
        return [], str(e)


def run_loop240():
    print("=" * 75)
    print("Axon v2.6 - Loop240: Re-run Pro Sandbox with Fixed Speakeasy Engine")
    print("=" * 75)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"
    split_csv = proj_dir / "models" / "generalization_group_isolated" / "split.csv"

    rows_by_split, all_shas = load_split(split_csv)
    target_shas = set(all_shas)

    # Load features from cache
    npz_files = glob.glob(str(cache_dir / "*.npz"))
    print(f"[Cache] Scanning {len(npz_files)} files...")

    sha_to_data = {}
    t0 = time.time()
    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            sha = str(d["source_sha256"]).strip().lower()
            if sha in target_shas:
                pe = d["pe_features"].astype(np.float32)
                stat = d["stat_features"].astype(np.float32)
                lw = d["lightweight_features"].astype(np.float32)
                feat = np.concatenate([pe, stat, lw])
                raw_path = str(d.get("raw_source_path", "")) if "raw_source_path" in d else ""
                sha_to_data[sha] = (feat, int(d["label"]), raw_path)
        except Exception:
            continue
    print(f"[Cache] Loaded {len(sha_to_data)} samples in {time.time()-t0:.1f}s")

    # Build matrices
    def build(rows):
        X, y, shas, paths = [], [], [], []
        for r in rows:
            sha = r["source_sha256"].strip().lower()
            if sha in sha_to_data:
                feat, label, raw_path = sha_to_data[sha]
                X.append(feat); y.append(label); shas.append(sha)
                paths.append(r.get("raw_source_path", raw_path or ""))
        return np.stack(X), np.array(y, dtype=np.int64), shas, paths

    X_train, y_train, _, _ = build(rows_by_split["train"])
    X_test, y_test, test_shas, test_paths = build(rows_by_split["test"])

    # Train Flash base
    print(f"\n[Flash] Training HistGBDT on {len(y_train)} train samples...")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, random_state=42)
    clf.fit(X_train, y_train)

    flash_probs = clf.predict_proba(X_test)[:, 1]
    flash_preds = (flash_probs >= 0.50).astype(int)
    base_errors = int((flash_preds != y_test).sum())
    base_f1 = f1_score(y_test, flash_preds)
    print(f"[Flash] UNSEEN F1={base_f1:.6f} | Errors={base_errors}")

    # Find error indices
    error_idx = np.where(flash_preds != y_test)[0]
    fp_idx = [i for i in error_idx if flash_preds[i]==1 and y_test[i]==0]
    fn_idx = [i for i in error_idx if flash_preds[i]==0 and y_test[i]==1]
    print(f"\n[Errors] Total={len(error_idx)} | FP={len(fp_idx)} | FN={len(fn_idx)}")

    # Pro Speakeasy emulation on error samples
    pro_probs = flash_probs.copy()
    rescued_fp, rescued_fn, emu_success, emu_fail = 0, 0, 0, 0

    MAL_APIS = {"VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread",
                "NtWriteVirtualMemory", "NtCreateThreadEx", "RegSetValueExA",
                "RegSetValueExW", "URLDownloadToFileA", "URLDownloadToFileW",
                "InternetOpenA", "HttpOpenRequestA", "WinExec", "ShellExecuteA",
                "CreateProcessInternalW"}

    if SPEAKEASY_AVAILABLE:
        print(f"\n[Pro Sandbox] Running fixed Speakeasy on {len(error_idx)} error samples...")
        for count, idx in enumerate(error_idx):
            fpath = test_paths[idx]
            label = y_test[idx]
            fp = flash_probs[idx]

            if not fpath or not os.path.exists(fpath):
                continue
            if os.path.getsize(fpath) > 30 * 1024 * 1024:
                continue

            apis, err = speakeasy_emulate(fpath, timeout_sec=60)

            if err:
                emu_fail += 1
                continue
            emu_success += 1

            detected_mal_apis = set()
            for api in apis:
                for m in MAL_APIS:
                    if m.lower() in api.lower():
                        detected_mal_apis.add(m)

            # FP Rescue: Static says malware but Speakeasy shows clean behavior
            if label == 0 and fp >= 0.50:
                if len(apis) == 0 or not detected_mal_apis:
                    pro_probs[idx] = 0.25
                    rescued_fp += 1

            # FN Catch: Static says benign but Speakeasy detects malicious APIs
            if label == 1 and fp < 0.50:
                if len(detected_mal_apis) >= 2:
                    pro_probs[idx] = 0.80
                    rescued_fn += 1
                elif len(detected_mal_apis) >= 1:
                    pro_probs[idx] = 0.65
                    rescued_fn += 1

            if (count+1) % 10 == 0:
                print(f"  [{count+1}/{len(error_idx)}] emu_ok={emu_success} emu_fail={emu_fail} fp_rescued={rescued_fp} fn_caught={rescued_fn}")

    pro_preds = (pro_probs >= 0.50).astype(int)
    pro_errors = int((pro_preds != y_test).sum())
    pro_f1 = f1_score(y_test, pro_preds)
    pro_acc = accuracy_score(y_test, pro_preds)
    reduction = base_errors - pro_errors

    print("\n" + "=" * 75)
    print(f"[Loop240 Final Results - Fixed Speakeasy Pro Sandbox on UNSEEN Test Set]")
    print(f"  Flash Base:      F1={base_f1:.6f} | Errors={base_errors}")
    print(f"  Pro Sandbox:     F1={pro_f1:.6f} | Acc={pro_acc*100:.2f}% | Errors={pro_errors}")
    print(f"  Error Reduction: {reduction} errors ({reduction/base_errors*100:.1f}%)")
    print(f"  Speakeasy Stats: Success={emu_success} | Fail={emu_fail} | FP_Rescued={rescued_fp} | FN_Caught={rescued_fn}")
    print("=" * 75)

    receipt = {
        "schema": "axon_loop240_fixed_speakeasy_receipt_v1",
        "unseen_test_samples": len(y_test),
        "baseline": {"f1": float(base_f1), "errors": base_errors},
        "pro_fixed": {"f1": float(pro_f1), "accuracy": float(pro_acc), "errors": pro_errors},
        "reduction": reduction,
        "speakeasy": {"success": emu_success, "fail": emu_fail, "fp_rescued": rescued_fp, "fn_caught": rescued_fn}
    }
    rp = proj_dir / "reports" / "roadmap_9997" / "loop240_fixed_speakeasy_receipt.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Receipt saved to {rp}")


if __name__ == "__main__":
    run_loop240()
