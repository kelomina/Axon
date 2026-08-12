"""Loop175 region cache 重新生成脚本（输出到 D: 盘）。

背景：
- 原始 phase_b_region_cache_v1.npz (926MB) 被误删
- E: 盘空间不足（458MB free），无法在原位置重建
- D: 盘有 217GB 可用空间
- 源 PE 文件仍存在于 data/random_20w_worktree/

用法:
    python scripts/regen_loop175_region_cache_to_d.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.phase_b_cache_builder import (  # noqa: E402
    build_region_cache,
    load_region_sources,
)

FOLD_MANIFEST = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_train_diagnostic_folds.jsonl"
OUTPUT_DIR = Path("D:/axon_loop185_region_cache")
OUTPUT_CACHE = OUTPUT_DIR / "phase_b_region_cache_v1.npz"
STAGING_DIR = OUTPUT_DIR / "staging"


def main() -> int:
    print("=== Loop175 Region Cache 重新生成（输出到 D: 盘）===")
    print(f"[input] fold_manifest: {FOLD_MANIFEST}")
    print(f"[output] cache: {OUTPUT_CACHE}")
    print(f"[output] staging: {STAGING_DIR}")

    if not FOLD_MANIFEST.exists():
        print(f"[ERROR] fold manifest not found: {FOLD_MANIFEST}")
        return 1

    if OUTPUT_CACHE.exists():
        print(f"[ERROR] output cache already exists: {OUTPUT_CACHE}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[step 1] 加载 region sources...")
    t0 = time.time()
    sources = load_region_sources(FOLD_MANIFEST)
    print(f"[step 1] 加载完成: {len(sources)} sources, 耗时 {time.time() - t0:.1f}s")

    print("\n[step 2] 构建 region cache（可能需要 3-10 分钟）...")
    t0 = time.time()
    result = build_region_cache(
        sources,
        output_cache=OUTPUT_CACHE,
        staging_directory=STAGING_DIR,
        block_rows=64,
    )
    elapsed = time.time() - t0
    print(f"[step 2] 构建完成: 耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  cache path: {result.cache.path}")
    print(f"  cache sha256: {result.cache.sha256}")
    print(f"  cache bytes: {result.cache.size_bytes:,} ({result.cache.size_bytes/1024/1024:.1f} MB)")
    print(f"  cache rows: {result.cache.row_count}")
    print(f"  cache regions: {result.cache.region_count}")
    print(f"  cache tokens: {result.cache.token_count}")
    print(f"  decision: {result.decision}")
    print(f"  attempted: {result.attempted}")
    print(f"  supported: {result.supported}")
    print(f"  coverage: {result.supported / result.attempted:.4f}")
    print(f"  class_coverage: {dict(result.class_coverage)}")
    print(f"  status_counts: {dict(result.status_counts)}")
    print(f"  blockers: {list(result.blockers)}")

    # 验证 SHA256
    print("\n[step 3] 验证 SHA256...")
    t0 = time.time()
    h = hashlib.sha256()
    with OUTPUT_CACHE.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    actual_sha256 = h.hexdigest()
    print(f"[step 3] 实际 SHA256: {actual_sha256}")
    print(f"[step 3] 期望 SHA256: 6e4ffb2382b986b1c4bd8bd1ac8ca211e3ca01f28643d7303c2baec1d338249d")
    print(f"[step 3] 匹配: {actual_sha256 == '6e4ffb2382b986b1c4bd8bd1ac8ca211e3ca01f28643d7303c2baec1d338249d'}")
    print(f"[step 3] 耗时 {time.time() - t0:.1f}s")

    print(f"\n[DONE] Region cache 已重建")
    print(f"  path: {OUTPUT_CACHE}")
    print(f"  sha256: {actual_sha256}")
    print(f"  size: {result.cache.size_bytes / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
