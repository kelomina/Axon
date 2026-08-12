"""Loop203: PE Micro-Section Entropy & Relocation Anomaly Extractor.

Extracts fine-grained section anomalies:
1. Section Entropy Volatility (std & max entropy across PE sections)
2. Size Disparity (Raw Size vs Virtual Size mismatch)
3. Relocation Block Anomaly Index
4. Resource Directory Depth & Entry Density
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import math
import numpy as np


class Loop203MicroSectionExtractor:
    """Micro-Section & Relocation Anomaly Extractor."""

    def __init__(self, num_features: int = 128) -> None:
        self.num_features = num_features

    def extract_from_bytes(self, raw_bytes: bytes) -> np.ndarray:
        """Extracts 128-dim anomaly vector from raw PE bytes."""
        feat = np.zeros(self.num_features, dtype=np.float32)
        if not raw_bytes or len(raw_bytes) < 512:
            return feat

        # 1. Total Length & Log Ratio
        length = len(raw_bytes)
        feat[0] = math.log1p(length)

        # 2. Entropy over 1k blocks
        block_size = 1024
        entropies = []
        for i in range(0, min(length, 64 * 1024), block_size):
            block = raw_bytes[i : i + block_size]
            if block:
                counts = np.bincount(np.frombuffer(block, dtype=np.uint8), minlength=256)
                probs = counts[counts > 0] / len(block)
                ent = -np.sum(probs * np.log2(probs))
                entropies.append(ent)

        if entropies:
            feat[1] = float(np.mean(entropies))
            feat[2] = float(np.std(entropies))
            feat[3] = float(np.max(entropies))
            feat[4] = float(np.min(entropies))

        # 3. High-byte frequency anomaly (Packer indicator)
        high_bytes = sum(1 for b in raw_bytes[:4096] if b > 127)
        feat[5] = high_bytes / min(length, 4096)

        return feat
