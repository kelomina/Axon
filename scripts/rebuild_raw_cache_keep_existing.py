# -*- coding: utf-8 -*-
"""Axon v2.6 - Keep Cache & Fast Skip Raw Binary 16-Process Cache Builder.

Features:
1. DOES NOT PURGE EXISTING CACHE! Keeps all existing .npz files.
2. Pre-loads existing SHA256 / filenames into a set for INSTANT O(1) SKIPPING.
3. Directly scans uncompressed raw binaries in H:\\私人\\良性文件 and H:\\私人\\恶意\\MB\\unziped.
4. Uses 16 multiprocessing workers to extract features only for NEW un-cached binaries.
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


def get_existing_cache_hashes(cache_dir: Path) -> set[str]:
    print(f"[Cache Load] Scanning existing .npz cache in {cache_dir}...", flush=True)
    existing_hashes = set()
    if cache_dir.exists():
        for f in cache_dir.glob("*.npz"):
            # cache file format: {sha256[:32]}_38672ba0.npz
            name = f.name
            if "_" in name:
                sha_prefix = name.split("_")[0]
                existing_hashes.add(sha_prefix)
    print(f"[Cache Load] Found {len(existing_hashes)} existing cache hashes (Will be SKIPPED instantly).", flush=True)
    return existing_hashes


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
        cache_name = f"{sha256[:32]}_38672ba0.npz"
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


def collect_raw_binary_tasks(benign_dirs: list[Path], malware_dirs: list[Path], cache_dir_str: str, existing_hashes: set[str]):
    tasks = []
    skipped_count = 0
    print("\n[Scanner] Fast scanning raw binary files...", flush=True)
    
    # Collect Malware binaries (Label 1)
    for mdir in malware_dirs:
        if mdir.exists():
            print(f"[Scanner] Scanning Malware directory: {mdir}...", flush=True)
            for root, _, files in os.walk(mdir):
                for fname in files:
                    if not fname.lower().endswith((".zip", ".7z", ".rar", ".npz", ".ini")):
                        # Fast check if filename is SHA256
                        name_no_ext = os.path.splitext(fname)[0].lower()
                        if len(name_no_ext) >= 32 and name_no_ext[:32] in existing_hashes:
                            skipped_count += 1
                            continue
                        tasks.append((os.path.join(root, fname), 1, cache_dir_str))
                        
    # Collect Benign binaries (Label 0)
    for bdir in benign_dirs:
        if bdir.exists():
            print(f"[Scanner] Scanning Benign directory: {bdir}...", flush=True)
            for root, _, files in os.walk(bdir):
                for fname in files:
                    if not fname.lower().endswith((".zip", ".7z", ".rar", ".npz", ".ini")):
                        name_no_ext = os.path.splitext(fname)[0].lower()
                        if len(name_no_ext) >= 32 and name_no_ext[:32] in existing_hashes:
                            skipped_count += 1
                            continue
                        tasks.append((os.path.join(root, fname), 0, cache_dir_str))

    print(f"[Scanner] Fast-skipped {skipped_count} already-cached files!", flush=True)
    print(f"[Scanner] Total NEW un-cached binary tasks to process: {len(tasks)}", flush=True)
    return tasks


def main():
    print("=" * 75, flush=True)
    print("Axon v2.6 - Keep Cache & Fast Skip Raw Binary 16-Process Builder")
    print("=" * 75, flush=True)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"

    # Step 1: Keep & scan existing cache
    existing_hashes = get_existing_cache_hashes(cache_dir)

    benign_dirs = [Path(r"H:\私人\良性文件")]
    malware_dirs = [Path(r"H:\私人\恶意\MB\unziped")]

    num_processes = 16
    print(f"\n[Parallel Setup] Launching {num_processes}-Process Pool with Instant-Skip...", flush=True)

    t_start = time.time()
    total_new_built = 0

    tasks = collect_raw_binary_tasks(benign_dirs, malware_dirs, str(cache_dir), existing_hashes)
    if not tasks:
        print("[Info] All binaries are already in cache or no new files found!", flush=True)
        return

    print(f"\n[Phase 1] Distributing {len(tasks)} NEW raw binary tasks across {num_processes} worker processes...", flush=True)

    with mp.Pool(processes=num_processes) as pool:
        for idx, ok in enumerate(pool.imap_unordered(_worker_process_raw_file, tasks, chunksize=32)):
            if ok:
                total_new_built += 1
            if (idx + 1) % 500 == 0 or (idx + 1) == len(tasks):
                elapsed = time.time() - t_start
                rate = total_new_built / elapsed if elapsed > 0 else 0
                print(f"  Progress: {idx+1}/{len(tasks)} files processed ({elapsed:.1f}s, {rate:.1f} new NPZ/sec) | Total New Built: {total_new_built}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print(f"[Incremental Raw Binary Cache Builder Complete]")
    print(f"  Existing Cache Retained: {len(existing_hashes)}")
    print(f"  New Cache Built:         {total_new_built}")
    print(f"  Total Cache Files Now:   {len(list(cache_dir.glob('*.npz')))}")
    print(f"  Total Time Elapsed:     {time.time()-t_start:.1f}s")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
