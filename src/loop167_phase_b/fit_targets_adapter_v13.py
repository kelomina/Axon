"""Read only Train-only labels, folds and component IDs for v13 fitting.

Unlike the raw manifest adapter, this parser never resolves, stats, hashes or
opens a sample path.  The v12 ledger already attests the sole permitted raw
scan; v13 consumes only sealed authority metadata and numeric cache values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import PhaseBContractError, canonical_json_bytes
from .path_safety_v4 import safe_project_path

ROWS = 20_000
FOLDS = 5
ROWS_PER_FOLD = 4_000
PROTOCOL_RELATIVE_PATH = "manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_protocol.json"


@dataclass(frozen=True, slots=True)
class V13FitTargets:
    labels: np.ndarray
    folds: np.ndarray
    component_ids: tuple[str, ...]
    protocol_sha256: str
    fold_manifest_sha256: str


def _canonical_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseBContractError(f"v13 {label} is not canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise PhaseBContractError(f"v13 {label} is not canonical JSON")
    return value


def load_fit_targets_v13(project_root: Path | str, *, protocol_sha256: str) -> V13FitTargets:
    """Load the fixed 20k Train-only target vectors without touching raw inputs."""

    root = Path(project_root).resolve(strict=True)
    protocol_path = safe_project_path(root, PROTOCOL_RELATIVE_PATH, require_exists=True, require_regular_file=True)
    protocol_raw = protocol_path.read_bytes()
    observed_protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    if observed_protocol_sha256 != protocol_sha256:
        raise PhaseBContractError("v13 protocol binding drifted")
    protocol = _canonical_object(protocol_raw, label="protocol")
    input_contract = protocol.get("input_contract")
    if not isinstance(input_contract, dict):
        raise PhaseBContractError("v13 protocol lacks its Train-only input contract")
    fold_contract = input_contract.get("folds")
    if not isinstance(fold_contract, dict) or fold_contract.get("rows") != ROWS or fold_contract.get("folds") != FOLDS:
        raise PhaseBContractError("v13 fold contract drifted")
    manifest_relative = fold_contract.get("path")
    manifest_sha256 = fold_contract.get("sha256")
    if not isinstance(manifest_relative, str) or not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise PhaseBContractError("v13 fold manifest binding is invalid")
    manifest_path = safe_project_path(root, manifest_relative, require_exists=True, require_regular_file=True)
    manifest_raw = manifest_path.read_bytes()
    observed_manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if observed_manifest_sha256 != manifest_sha256 or not manifest_raw.endswith(b"\n"):
        raise PhaseBContractError("v13 fold manifest binding drifted")
    lines = manifest_raw[:-1].split(b"\n")
    if len(lines) != ROWS:
        raise PhaseBContractError("v13 fold manifest denominator drifted")
    labels = np.empty(ROWS, dtype=np.uint8)
    folds = np.empty(ROWS, dtype=np.int8)
    components: list[str] = []
    component_folds: dict[str, int] = {}
    for ordinal, raw_line in enumerate(lines):
        record = _canonical_object(raw_line + b"\n", label="fold record")
        label, fold, component = record.get("label"), record.get("diagnostic_fold"), record.get("content_component_id")
        if record.get("split_role") != "train" or record.get("train_row_index") != ordinal or record.get("sample_index") != ordinal:
            raise PhaseBContractError("v13 fold record ordering or scope drifted")
        if label not in (0, 1) or not isinstance(fold, int) or fold not in range(FOLDS) or not isinstance(component, str):
            raise PhaseBContractError("v13 fold record targets drifted")
        previous = component_folds.setdefault(component, fold)
        if previous != fold:
            raise PhaseBContractError("v13 component crosses outer folds")
        labels[ordinal], folds[ordinal] = label, fold
        components.append(component)
    if not np.all(np.bincount(folds, minlength=FOLDS) == ROWS_PER_FOLD) or not np.all(np.bincount(labels, minlength=2) == ROWS // 2):
        raise PhaseBContractError("v13 Train-only target balance drifted")
    return V13FitTargets(labels, folds, tuple(components), observed_protocol_sha256, observed_manifest_sha256)
