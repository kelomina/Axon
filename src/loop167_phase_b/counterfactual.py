"""Deterministic no-label counterfactual permutation for Loop167 novel blocks."""

from __future__ import annotations

import hashlib
import inspect
import re

import numpy as np

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_ROLES = frozenset({"fit", "holdout"})


def _derived_seed(protocol_sha256: str, seed: int, outer_fold: int, role: str) -> int:
    if not isinstance(protocol_sha256, str) or not SHA256_PATTERN.fullmatch(protocol_sha256):
        raise ValueError("protocol_sha256 must be lowercase SHA-256")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int) or outer_fold < 0:
        raise ValueError("outer_fold must be a non-negative integer")
    if role not in ALLOWED_ROLES:
        raise ValueError("role must be fit or holdout")
    material = f"loop167-cf-v1\0{protocol_sha256}\0{seed}\0{outer_fold}\0{role}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def permute_complete_novel_blocks(
    novel_blocks: np.ndarray,
    novel_complete_mask: np.ndarray,
    *,
    protocol_sha256: str,
    seed: int,
    outer_fold: int,
    role: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute full 292-column blocks only among complete rows in one partition role."""

    blocks = np.asarray(novel_blocks, dtype=np.float32)
    complete_mask = np.asarray(novel_complete_mask, dtype=bool)
    if blocks.ndim != 2 or blocks.shape[1] != 292:
        raise ValueError("novel_blocks must have shape [rows, 292]")
    if complete_mask.shape != (blocks.shape[0],):
        raise ValueError("novel_complete_mask must have one value per row")
    if not np.isfinite(blocks).all():
        raise ValueError("novel_blocks must be finite before counterfactual permutation")
    complete_positions = np.flatnonzero(complete_mask)
    permutation = np.arange(blocks.shape[0], dtype=np.int64)
    if complete_positions.size > 1:
        generator = np.random.Generator(np.random.PCG64(_derived_seed(protocol_sha256, seed, outer_fold, role)))
        shuffled = complete_positions[generator.permutation(complete_positions.size)]
        permutation[complete_positions] = shuffled
    result = blocks.copy()
    result[complete_positions] = blocks[permutation[complete_positions]]
    return result, permutation


def assert_counterfactual_api_has_no_identity_or_label_surface() -> None:
    parameters = inspect.signature(permute_complete_novel_blocks).parameters
    forbidden = {"label", "target", "identity", "sha", "path", "row", "family", "time", "score"}
    if any(name in forbidden for name in parameters):
        raise RuntimeError("Counterfactual API has a forbidden parameter")
