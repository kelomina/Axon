from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.phase_b_contract import write_exclusive_json  # noqa: E402
from src.loop175.phase_b_controller import (  # noqa: E402
    PhaseBControllerError,
    _worker_receipt_commitment,
    directory_size_bytes,
    validate_pilot_receipt,
    validate_worker_receipt,
)

PROTOCOL_SHA = "a" * 64
CACHE_SHA = "b" * 64


def test_pilot_receipt_scope_and_epoch_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "pilot.json"
    payload = {
        "schema": "axon_loop175_epoch_pilot_receipt_v1",
        "arm": "C",
        "seed": 41,
        "outer_fold_never_read": 0,
        "pilot_fit_folds": [2, 3, 4],
        "inner_selection_fold": 1,
        "pilot_fit_rows": 12_000,
        "selection_rows": 4_000,
        "selected_epoch": 4,
        "selection_losses": [1.0 - index * 0.01 for index in range(12)],
        "protocol_sha256": PROTOCOL_SHA,
        "cache_sha256": CACHE_SHA,
        "raw_rows_opened": 0,
        "val_test_or_full_rows_opened": 0,
        "decision": "pilot_pass_freeze_epoch_for_all_outer_folds",
        "resources": {},
    }
    write_exclusive_json(path, payload)
    assert validate_pilot_receipt(
        path,
        arm="C",
        protocol_sha256=PROTOCOL_SHA,
        cache_sha256=CACHE_SHA,
    )["selected_epoch"] == 4
    payload["selected_epoch"] = 13
    changed = tmp_path / "pilot_bad.json"
    write_exclusive_json(changed, payload)
    with pytest.raises(PhaseBControllerError, match="selection"):
        validate_pilot_receipt(
            changed,
            arm="C",
            protocol_sha256=PROTOCOL_SHA,
            cache_sha256=CACHE_SHA,
        )


def test_worker_receipt_binds_runtime_and_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "schema": "axon_loop175_arm_fold_worker_receipt_v1",
        "arm": "C",
        "fold": 2,
        "seed": 41,
        "fit_rows": 16_000,
        "holdout_rows": 4_000,
        "protocol_sha256": PROTOCOL_SHA,
        "cache_sha256": CACHE_SHA,
        "model_commitment": hashlib.sha256(b"checkpoint").hexdigest(),
        "checkpoint_path": str(checkpoint),
        "resources": {
            "wall_seconds": 1.0,
            "rss_bytes": 10,
            "gpu_allocated_bytes": 20,
            "new_disk_bytes": len(b"checkpoint"),
        },
        "raw_rows_opened": 0,
        "val_test_or_full_rows_opened": 0,
        "decision": "arm_fold_outer_oof_complete",
        "artifact_numeric_commitment": "c" * 64,
    }
    payload["runtime_commitment"] = _worker_receipt_commitment(payload)
    receipt = tmp_path / "worker.json"
    write_exclusive_json(receipt, payload)
    assert validate_worker_receipt(
        receipt,
        arm="C",
        fold=2,
        protocol_sha256=PROTOCOL_SHA,
        cache_sha256=CACHE_SHA,
    )["runtime_commitment"] == payload["runtime_commitment"]
    tampered = json.loads(receipt.read_text(encoding="ascii"))
    tampered["resources"]["wall_seconds"] = 2.0
    changed = tmp_path / "worker_bad.json"
    write_exclusive_json(changed, tampered)
    with pytest.raises(PhaseBControllerError, match="commitment"):
        validate_worker_receipt(
            changed,
            arm="C",
            fold=2,
            protocol_sha256=PROTOCOL_SHA,
            cache_sha256=CACHE_SHA,
        )


def test_directory_size_counts_nested_artifacts(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b").write_bytes(b"12345")
    assert directory_size_bytes(tmp_path) == 8

