"""Canonical artifact and path validation primitives for Loop167 Phase B."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PhaseBContractError(ValueError):
    """Raised when a Phase-B static artifact is missing, drifted, or unsafe."""


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise PhaseBContractError("Artifact is not canonical JSON") from exc


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise PhaseBContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def resolve_project_file(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise PhaseBContractError("Artifact path must be a nonempty relative string")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise PhaseBContractError("Artifact path must be project-relative")
    resolved_root = root.resolve(strict=True)
    resolved_path = (resolved_root / candidate).resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_root):
        raise PhaseBContractError("Artifact path escapes the project root")
    return resolved_path


def require_canonical_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PhaseBContractError(f"Artifact is missing or unsafe: {path}")
    content = path.read_bytes()
    try:
        payload = json.loads(content)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseBContractError(f"Artifact is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != content:
        raise PhaseBContractError(f"Artifact is not canonical JSON: {path}")
    return payload


def verify_file_binding(root: Path, binding: object, *, label: str) -> tuple[Path, str]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise PhaseBContractError(f"{label} binding must contain exactly path and sha256")
    path = resolve_project_file(root, binding["path"])
    expected_sha256 = require_sha256(binding["sha256"], field=f"{label}.sha256")
    if not path.is_file() or path.is_symlink():
        raise PhaseBContractError(f"{label} file is missing or unsafe")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise PhaseBContractError(f"{label} hash mismatch")
    return path, observed_sha256


def canonical_argv_sha256(argv: Sequence[str]) -> str:
    if isinstance(argv, (str, bytes)) or not all(isinstance(value, str) and value for value in argv):
        raise PhaseBContractError("Canonical argv must contain nonempty strings")
    return sha256_bytes(canonical_json_bytes({"argv": list(argv)}))
