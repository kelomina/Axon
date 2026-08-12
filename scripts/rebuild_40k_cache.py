"""Axon v2.6 - Purge Old Cache & Rebuild 40k Cache Dataset.

1. Completely purges all legacy npz files in data/.cache/
2. Scans H:\\私人\\良性文件 and H:\\私人\\恶意\\MB
3. Extracts PE, Stat, and Lightweight features using kvd_features.extractor
4. Generates fresh 40k npz cache records.
"""

from __future__ import annotations

import os
import shutil
import time
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kvd_features.extractor import ExtractionConfig, extract_all_features


def purge_cache(cache_dir: Path):
    print(f"[Purge] Resetting and cleaning up old cache directory: {cache_dir}...", flush=True)
    if cache_dir.exists():
        old_files = list(cache_dir.glob("*.npz"))
        print(f"[Purge] Removing {len(old_files)} legacy npz cache files...", flush=True)
        for f in old_files:
            try:
                f.unlink()
            except Exception:
                pass
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[Purge] Cache directory completely reset and emptied.", flush=True)


def scan_source_files(target_dir: Path, max_count: int = 25000) -> list[Path]:
    print(f"[Scanner] Recursively scanning binary files under {target_dir}...", flush=True)
    found = []
    t0 = time.time()

    for root, _, files in os.walk(target_dir):
        for fname in files:
            fpath = Path(root) / fname
            # Check PE extension / binary file size
            if fpath.is_file():
                ext = fpath.suffix.lower()
                if ext in (".exe", ".dll", ".sys", ".drv", ".ocx", ".bin", ""):
                    try:
                        sz = fpath.stat().st_size
                        if 1024 <= sz <= 50 * 1024 * 1024:  # 1KB to 50MB
                            found.append(fpath)
                            if len(found) >= max_count:
                                break
                    except Exception:
                        continue
        if len(found) >= max_count:
            break

    print(f"[Scanner] Discovered {len(found)} candidate binary files in {time.time()-t0:.1f}s under {target_dir}", flush=True)
    return found


def calculate_sha256(fpath: Path) -> str:
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main():
    print("=" * 75, flush=True)
    print("Axon v2.6 - Purge Old Cache & Rebuild Fresh 40k Cache Dataset")
    print("=" * 75, flush=True)

    proj_dir = Path(__file__).resolve().parent.parent
    cache_dir = proj_dir / "data" / ".cache"

    # Step 1: Purge Old Cache
    purge_cache(cache_dir)

    # Step 2: Discover Sources
    benign_dir = Path(r"H:\私人\良性文件")
    malware_dir = Path(r"H:\私人\恶意\MB")

    b_files = scan_source_files(benign_dir, max_count=22000)
    m_files = scan_source_files(malware_dir, max_count=22000)

    all_targets = [(f, 0) for f in b_files] + [(f, 1) for f in m_files]
    print(f"\n[Dataset Summary] Total candidates to process: {len(all_targets)} (Benign: {len(b_files)}, Malware: {len(m_files)})", flush=True)

    # Step 3: Feature Extraction & Cache Rebuilding
    config = ExtractionConfig()
    t_start = time.time()
    success_cnt, fail_cnt = 0, 0

    print("\n[Cache Builder] Extracting Features and Writing Fresh NPZ Cache...", flush=True)
    for idx, (fpath, label) in enumerate(all_targets):
        try:
            res = extract_all_features(str(fpath), config=config)
            if res is not None and len(res) >= 4:
                byte_seq, pe_features, stat_features, lightweight_features = res[:4]
                if pe_features is not None and stat_features is not None:
                    sha256 = calculate_sha256(fpath)
                    cache_name = f"{sha256[:32]}_38672ba0.npz"
                    target_cache = cache_dir / cache_name

                    np.savez_compressed(
                        target_cache,
                        byte_sequence=byte_seq,
                        pe_features=pe_features.astype(np.float32),
                        stat_features=stat_features.astype(np.float32),
                        lightweight_features=lightweight_features.astype(np.float32),
                        label=int(label),
                        source_sha256=sha256,
                        raw_source_path=str(fpath.resolve())
                    )
                    success_cnt += 1
                else:
                    fail_cnt += 1
            else:
                fail_cnt += 1
        except Exception:
            fail_cnt += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_targets):
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Progress: {idx+1}/{len(all_targets)} ({elapsed:.1f}s, {rate:.1f} s/s) | Built: {success_cnt} | Failed: {fail_cnt}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print(f"[Cache Rebuild Complete]")
    print(f"  Total Fresh NPZ Cache Files Created: {success_cnt} in {cache_dir}")
    print(f"  Failed / Unsupported Non-PE Files: {fail_cnt}")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
