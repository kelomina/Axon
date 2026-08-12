"""Protocol helpers for Loop175 counterfactual and coverage accounting."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .region_extractor import RegionExtractionResult


def deterministic_region_permutation(size: int, *, seed: int, fold: int, role: str) -> tuple[int, ...]:
    if size < 0:
        raise ValueError("size must be non-negative")
    if role not in {"fit", "holdout"}:
        raise ValueError("role must be fit or holdout")
    material = f"loop175|{seed}|{fold}|{role}|{size}".encode("ascii")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    indices = list(range(size))
    random.Random(derived_seed).shuffle(indices)
    return tuple(indices)


@dataclass
class CoverageAccounting:
    attempted: int = 0
    supported: int = 0
    silent_drops: int = 0
    bytes_read: int = 0
    model_region_bytes: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    label_attempted: dict[int, int] = field(default_factory=dict)
    label_supported: dict[int, int] = field(default_factory=dict)

    def observe(self, result: RegionExtractionResult, *, label: int | None = None) -> None:
        self.attempted += 1
        self.supported += int(result.supported)
        self.bytes_read += result.bytes_read
        self.model_region_bytes += result.model_region_bytes
        self.status_counts[result.status] = self.status_counts.get(result.status, 0) + 1
        if len(result.regions) == 0:
            self.silent_drops += 1
        if label is not None:
            if label not in {0, 1}:
                raise ValueError("label must be binary")
            self.label_attempted[label] = self.label_attempted.get(label, 0) + 1
            self.label_supported[label] = self.label_supported.get(label, 0) + int(result.supported)

    def summary(self) -> dict[str, object]:
        coverage = self.supported / self.attempted if self.attempted else 0.0
        label_coverage = {
            str(label): self.label_supported.get(label, 0) / attempted
            for label, attempted in sorted(self.label_attempted.items())
            if attempted
        }
        class_gap = max(label_coverage.values()) - min(label_coverage.values()) if label_coverage else 0.0
        return {
            "attempted": self.attempted,
            "supported": self.supported,
            "coverage": coverage,
            "silent_drops": self.silent_drops,
            "bytes_read": self.bytes_read,
            "model_region_bytes": self.model_region_bytes,
            "status_counts": dict(sorted(self.status_counts.items())),
            "label_coverage": label_coverage,
            "class_coverage_gap": class_gap,
        }


def assert_coverage_gate(
    accounting: CoverageAccounting,
    *,
    minimum_coverage: float = 0.995,
    maximum_class_gap: float = 0.02,
) -> dict[str, object]:
    summary = accounting.summary()
    if summary["coverage"] < minimum_coverage:
        raise RuntimeError("Loop175 coverage is below the frozen gate")
    if summary["silent_drops"] != 0:
        raise RuntimeError("Loop175 silent drop accounting is nonzero")
    if summary["class_coverage_gap"] > maximum_class_gap:
        raise RuntimeError("Loop175 class coverage gap exceeds the frozen gate")
    return summary
