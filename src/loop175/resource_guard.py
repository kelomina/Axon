"""Resource measurements and fail-closed limits for Loop175 probes."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceLimits:
    maximum_rss_bytes: int = 11 * 1024**3
    maximum_gpu_allocated_bytes: int = int(6.5 * 1024**3)
    maximum_wall_seconds_per_fold: int = 6 * 60 * 60
    maximum_new_disk_bytes: int = 30 * 1024**3


def process_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value if os.name == "posix" and value > 1024**2 else value * 1024
        except (ImportError, OSError):
            return 0


def gpu_allocated_bytes() -> int:
    try:
        import torch

        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError:
        return 0


@dataclass
class ResourceGuard:
    limits: ResourceLimits = ResourceLimits()
    started_at: float = 0.0

    def start(self) -> None:
        self.started_at = time.monotonic()

    def snapshot(self, *, new_disk_bytes: int = 0) -> dict[str, int | float]:
        if self.started_at == 0.0:
            raise RuntimeError("resource guard must be started before sampling")
        snapshot = {
            "wall_seconds": time.monotonic() - self.started_at,
            "rss_bytes": process_rss_bytes(),
            "gpu_allocated_bytes": gpu_allocated_bytes(),
            "new_disk_bytes": int(new_disk_bytes),
        }
        if snapshot["rss_bytes"] > self.limits.maximum_rss_bytes:
            raise RuntimeError("Loop175 RSS limit exceeded")
        if snapshot["gpu_allocated_bytes"] > self.limits.maximum_gpu_allocated_bytes:
            raise RuntimeError("Loop175 GPU allocation limit exceeded")
        if snapshot["wall_seconds"] > self.limits.maximum_wall_seconds_per_fold:
            raise RuntimeError("Loop175 wall-time limit exceeded")
        if snapshot["new_disk_bytes"] > self.limits.maximum_new_disk_bytes:
            raise RuntimeError("Loop175 disk limit exceeded")
        return snapshot
