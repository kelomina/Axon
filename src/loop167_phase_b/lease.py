"""Fail-closed one-shot lease primitive for a future Loop167 Phase-B run."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class LeaseError(RuntimeError):
    """Raised when a Phase-B lease cannot be consumed exactly once."""


@dataclass(frozen=True)
class ConsumedLease:
    marker_path: Path
    marker_sha256: str
    payload: dict[str, Any]


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise LeaseError("Lease payload is not canonical JSON") from exc


def consume_lease(marker_path: Path, payload: Mapping[str, Any]) -> ConsumedLease:
    """Atomically create and fsync a lease marker; never delete a failed marker."""

    marker_path = Path(marker_path)
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(marker_path, flags, 0o600)
    except FileExistsError as exc:
        raise LeaseError(f"Lease marker already exists: {marker_path}") from exc
    except OSError as exc:
        raise LeaseError(f"Lease marker cannot be created: {marker_path}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        raise LeaseError("Lease marker write failed; marker remains consumed") from exc
    return ConsumedLease(marker_path, hashlib.sha256(content).hexdigest(), dict(payload))
