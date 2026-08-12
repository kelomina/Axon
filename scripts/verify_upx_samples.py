#!/usr/bin/env python3
import os
import sys
import math
import hashlib
import collections
from pathlib import Path

out_dir = Path(r"H:\私人\良性文件\待加入白名单_upx")

samples = []
with os.scandir(out_dir) as entries:
    for entry in entries:
        if entry.is_file() and not entry.name.startswith("_temp") and not entry.name.endswith(".json"):
            samples.append(Path(entry.path))
            if len(samples) >= 10:
                break

print(f"========== 随机抽查 10 个加壳成功文件校验 ==========", flush=True)

for i, p in enumerate(samples, 1):
    with open(p, "rb") as f:
        data = f.read()
    
    file_sha256 = hashlib.sha256(data).hexdigest().lower()
    expected_sha256 = p.stem.lower()
    sha_match = (file_sha256 == expected_sha256)
    
    counts = collections.Counter(data)
    total = len(data)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values()) if total > 0 else 0.0
    
    has_upx_sig = any(sig in data for sig in [b"UPX0", b"UPX1", b"UPX!"])
    
    print(f"样本 [{i}/10]: {p.name}", flush=True)
    print(f"  |- 文件大小  : {len(data)} bytes", flush=True)
    print(f"  |- SHA256匹配: {'[PASS] 一致' if sha_match else '[FAIL] 不一致'} ({file_sha256[:16]}...)", flush=True)
    print(f"  |- 字节熵值  : {entropy:.4f} bits/byte (加壳后高熵特性)", flush=True)
    print(f"  |- UPX特征  : {'[PASS] 发现 UPX 标志头' if has_upx_sig else '[WARN] 未匹配明文UPX头'}", flush=True)
    print("", flush=True)
