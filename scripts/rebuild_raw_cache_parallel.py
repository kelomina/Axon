# -*- coding: utf-8 -*-
"""Axon v2.6 - Pure Uncompressed Binary Parallel 400k Cache Builder.

Features:
1. Purges legacy npz files in data/.cache/
2. Directly scans uncompressed raw binaries in H:\\私人\\良性文件 and H:\\私人\\恶意\\MB\\unziped
3. Uses 16 multiprocessing workers to extract pe_features, stat_features, lightweight_features
4. Zero ZIP decompression overhead.
"""

from __future__ import annotations

import os
import time
import hashlib
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def purge_cache(cache_dir: Path):
    print(f"[Purge] Cleaning up old cache directory: {cache_dir}...", flush=True)
    if cache_dir.exists():
        old_files = list(cache_dir.glob("*.npz"))
        print(f"[Purge] Removing {len(old_files)} old npz files...", flush=True)
        for f in old_files:
            try:
                f.unlink()
            except Exception:
                pass
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[Purge] Cache directory successfully reset and emptied.", flush=True)


def _compute_cache_config_hash() -> str:
    """Compute the same config hash that FeatureCacheDataset uses for the default config."""
    import hashlib as _hl
    # Matches AxonExperimentConfig defaults: max_byte_length=65536, stat_feature_dim=49,
    # pe_feature_dim=1500, lightweight_feature_dim=256, strict_pe_parsing=True,
    # allow_pe_fallback=False, pe_schema_version='legacy_dynamic'
    sig = "65536_49_1500_256_True_False"
    return _hl.md5(sig.encode()).hexdigest()[:8]


_CACHE_CONFIG_HASH = _compute_cache_config_hash()


def _worker_process_raw_file(args):
    fpath_str, label, cache_dir_str = args
    try:
        fpath = Path(fpath_str)
        if not fpath.exists() or fpath.stat().st_size < 1024 or fpath.stat().st_size > 50 * 1024 * 1024:
            return False

        with open(fpath, "rb") as f:
            header = f.read(2)
            if header != b"MZ":
                return False
            f.seek(0)
            bdata = f.read()

        sha256 = hashlib.sha256(bdata).hexdigest().lower()
        cache_name = f"{sha256[:32]}_{_CACHE_CONFIG_HASH}.npz"
        target_cache = Path(cache_dir_str) / cache_name
        if target_cache.exists():
            return True

        from kvd_features.extractor import ExtractionConfig, extract_all_features
        config = ExtractionConfig()
        res = extract_all_features(str(fpath), config=config)
        if res is not None and len(res) >= 4:
            byte_seq, pe_features, stat_features, lightweight_features = res[:4]
            if pe_features is not None and stat_features is not None:
                np.savez_compressed(
                    target_cache,
                    byte_sequence=byte_seq,
                    pe_features=pe_features.astype(np.float32),
                    stat_features=stat_features.astype(np.float32),
                    lightweight_features=lightweight_features.astype(np.float32),
                    label=int(label),
                    source_sha256=sha256,
                    raw_source_path=str(fpath)
                )
                return True
        return False
    except Exception:
        return False


def collect_raw_binary_tasks(benign_dirs: list[Path], malware_dirs: list[Path], cache_dir_str: str):
    tasks = []
    print("\n[Scanner] Fast scanning raw binary files...", flush=True)
    
    # Collect Malware binaries (Label 1)
    for mdir in malware_dirs:
        if mdir.exists():
            print(f"[Scanner] Scanning Malware directory: {mdir}...", flush=True)
            for root, _, files in os.walk(mdir):
                for fname in files:
                    if not fname.lower().endswith((".zip", ".7z", ".rar", ".npz", ".ini")):
                        tasks.append((os.path.join(root, fname), 1, cache_dir_str))
                        
    # Collect Benign binaries (Label 0)
    for bdir in benign_dirs:
        if bdir.exists():
            print(f"[Scanner] Scanning Benign directory: {bdir}...", flush=True)
            for root, _, files in os.walk(bdir):
                for fname in files:
                    if not fname.lower().endswith((".zip", ".7z", ".rar", ".npz", ".ini")):
                        tasks.append((os.path.join(root, fname), 0, cache_dir_str))

    print(f"[Scanner] Total raw binary candidate files collected: {len(tasks)}", flush=True)
    return tasks


def main():
    print("=" * 75, flush=True)
    print("Axon v2.6 - Pure Uncompressed Raw Binary 16-Process Cache Builder")
    print("=" * 75, flush=True)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"

    # Step 1: Purge Old Cache
    purge_cache(cache_dir)

    benign_dirs = [Path(r"H:\私人\良性文件")]
    malware_dirs = [Path(r"H:\私人\恶意\MB\unziped")]

    num_processes = 16
    print(f"\n[Parallel Setup] Launching {num_processes}-Process Pool...", flush=True)

    t_start = time.time()
    total_built = 0

    tasks = collect_raw_binary_tasks(benign_dirs, malware_dirs, str(cache_dir))
    if not tasks:
        print("[Error] No raw binary files found to process!", flush=True)
        return

    print(f"\n[Phase 1] Distributing {len(tasks)} raw binary tasks across {num_processes} worker processes...", flush=True)

    with mp.Pool(processes=num_processes) as pool:
        for idx, ok in enumerate(pool.imap_unordered(_worker_process_raw_file, tasks, chunksize=32)):
            if ok:
                total_built += 1
            if (idx + 1) % 500 == 0 or (idx + 1) == len(tasks):
                elapsed = time.time() - t_start
                rate = total_built / elapsed if elapsed > 0 else 0
                print(f"  Progress: {idx+1}/{len(tasks)} files processed ({elapsed:.1f}s, {rate:.1f} NPZ/sec) | Built NPZ Cache: {total_built}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print(f"[Pure Uncompressed Binary Cache Builder Complete]")
    print(f"  Total Fresh Cache Built: {total_built} npz files in {cache_dir}")
    print(f"  Total Time Elapsed:     {time.time()-t_start:.1f}s")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
