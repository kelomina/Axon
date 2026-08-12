"""Axon v2.6 - Instant 16-Process Zip-Parallel 400k Cache Builder.

Features:
1. Purges legacy npz files in data/.cache/
2. Instant top-level listdir scanning for zero latency.
3. 16 Worker processes unpack & extract features in parallel.
"""

from __future__ import annotations

import os
import time
import zipfile
import tempfile
import hashlib
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PASSWORDS = [b"infected", b"infected123", b"malware", b"virus", b"pwd", b"123", b"123456", None]


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


def _worker_process_zip(args):
    zip_path, label, cache_dir_str = args
    built_count = 0
    try:
        from kvd_features.extractor import ExtractionConfig, extract_all_features
        config = ExtractionConfig()

        with zipfile.ZipFile(zip_path, "r") as zf:
            for zinfo in zf.infolist():
                if zinfo.is_dir() or zinfo.file_size < 1024:
                    continue

                bdata = None
                for pwd in PASSWORDS:
                    try:
                        bdata = zf.read(zinfo, pwd=pwd)
                        if bdata and bdata.startswith(b"MZ"):
                            break
                    except Exception:
                        bdata = None

                if not bdata or len(bdata) > 50 * 1024 * 1024:
                    continue

                sha256 = hashlib.sha256(bdata).hexdigest().lower()
                cache_name = f"{sha256[:32]}_38672ba0.npz"
                target_cache = Path(cache_dir_str) / cache_name

                if target_cache.exists():
                    built_count += 1
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp:
                    tmp.write(bdata)
                    tmp_path = tmp.name

                try:
                    res = extract_all_features(tmp_path, config=config)
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
                                raw_source_path=f"{zip_path}::{zinfo.filename}"
                            )
                            built_count += 1
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
    except Exception:
        pass
    return built_count


def _worker_process_raw_file(args):
    fpath_str, label, cache_dir_str = args
    try:
        fpath = Path(fpath_str)
        if not fpath.exists() or fpath.stat().st_size < 1024:
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


def collect_zip_tasks(malware_dir: Path, cache_dir_str: str) -> list[tuple[str, int, str]]:
    zip_tasks = []
    if not malware_dir.exists():
        return zip_tasks

    print(f"[Scanner] Fast listing Malware directory: {malware_dir}...", flush=True)
    for entry in os.scandir(malware_dir):
        if entry.is_file() and entry.name.lower().endswith(".zip"):
            zip_tasks.append((entry.path, 1, cache_dir_str))
        elif entry.is_dir():
            for sub_entry in os.scandir(entry.path):
                if sub_entry.is_file() and sub_entry.name.lower().endswith(".zip"):
                    zip_tasks.append((sub_entry.path, 1, cache_dir_str))

    print(f"[Scanner] Found {len(zip_tasks)} .zip malware packages.", flush=True)
    return zip_tasks


def main():
    print("=" * 75, flush=True)
    print("Axon v2.6 - Instant 16-Process Zip-Parallel 400k Cache Builder")
    print("=" * 75, flush=True)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"

    # Step 1: Purge Old Cache
    purge_cache(cache_dir)

    benign_dir = Path(r"H:\私人\良性文件")
    malware_dir = Path(r"H:\私人\恶意\MB")

    num_processes = 16
    print(f"\n[Parallel Setup] Launching {num_processes}-Process Pool...", flush=True)

    t_start = time.time()
    total_built = 0

    with mp.Pool(processes=num_processes) as pool:
        # Phase 1: Malware Zips (Instant dispatch)
        zip_tasks = collect_zip_tasks(malware_dir, str(cache_dir))
        if zip_tasks:
            print(f"\n[Phase 1] Distributing {len(zip_tasks)} .zip packages to 16 processes...", flush=True)
            for idx, built in enumerate(pool.imap_unordered(_worker_process_zip, zip_tasks, chunksize=1)):
                total_built += built
                if (idx + 1) % 10 == 0 or (idx + 1) == len(zip_tasks):
                    elapsed = time.time() - t_start
                    rate = total_built / elapsed if elapsed > 0 else 0
                    print(f"  Zip Progress: {idx+1}/{len(zip_tasks)} packages ({elapsed:.1f}s, {rate:.1f} NPZ/sec) | Built NPZ Cache: {total_built}", flush=True)

        # Phase 2: Benign Directories
        if benign_dir.exists():
            print(f"\n[Phase 2] Scanning & Processing Benign binaries in parallel...", flush=True)
            b_tasks = []
            for root, _, files in os.walk(benign_dir):
                for fname in files:
                    b_tasks.append((os.path.join(root, fname), 0, str(cache_dir)))

            print(f"[Phase 2] Found {len(b_tasks)} benign files. Distributing across {num_processes} processes...", flush=True)

            for idx, ok in enumerate(pool.imap_unordered(_worker_process_raw_file, b_tasks, chunksize=32)):
                if ok:
                    total_built += 1
                if (idx + 1) % 1000 == 0 or (idx + 1) == len(b_tasks):
                    elapsed = time.time() - t_start
                    rate = total_built / elapsed if elapsed > 0 else 0
                    print(f"  Benign Progress: {idx+1}/{len(b_tasks)} files ({elapsed:.1f}s) | Total Built Cache: {total_built}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print(f"[Instant 400k Parallel Cache Builder Complete]")
    print(f"  Total Fresh Cache Built: {total_built} npz files in {cache_dir}")
    print(f"  Total Time Elapsed:     {time.time()-t_start:.1f}s")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
