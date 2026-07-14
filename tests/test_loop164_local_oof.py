from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop164.local_oof import (  # noqa: E402
    FOLD_CLAIM_SCOPE,
    FOLD_RECORD_SCHEMA,
    FOLD_SUMMARY_SCHEMA,
    IDENTITY_METADATA_FIELDS,
    LOOP_ID,
    REQUIRED_LIMITATIONS,
    LocalOOFContractError,
    fixed_binary_metrics,
    load_local_diagnostic_folds,
)

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_loop164_local_whole_file_oof as oof_controller  # noqa: E402
from pre_run_resource_leak_guard import GUARD_SCHEMA, build_guard_receipt  # noqa: E402


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _case(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    records = []
    for row_index in range(4):
        component_id = _digest(f"component-{row_index}")[:24]
        records.append(
            {
                "schema": FOLD_RECORD_SCHEMA,
                "loop_id": LOOP_ID,
                "claim_scope": FOLD_CLAIM_SCOPE,
                "split_role": "train",
                "train_row_index": row_index,
                "sample_index": row_index,
                "source_path": str(data_root / f"sample-{row_index}.bin"),
                "source_sha256": _digest(f"source-{row_index}"),
                "source_size_bytes": 32,
                "label": row_index % 2,
                "availability": "supported",
                "missing_reason": None,
                "content_component_id": component_id,
                "content_component_size": 1,
                "diagnostic_fold": row_index // 2,
                "identity_metadata_not_model_features": IDENTITY_METADATA_FIELDS,
            }
        )
    folds_path = tmp_path / "folds.jsonl"
    folds_raw = ("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n").encode()
    folds_path.write_bytes(folds_raw)
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema": FOLD_SUMMARY_SCHEMA,
        "loop_id": LOOP_ID,
        "claim_scope": FOLD_CLAIM_SCOPE,
        "parameters": {
            "fold_count": 2,
            "seed": 164,
            "max_supported_file_bytes": 64,
        },
        "inputs": {
            "canonical_split_train_prefix": {
                "heldout_rows_read": 0,
                "stopped_before_next_line": True,
                "train_rows": 4,
            }
        },
        "limitations": REQUIRED_LIMITATIONS,
        "time_stress_metadata": {"used_for_fold_assignment": False},
        "aggregate": {
            "canonical_train_rows": 4,
            "availability_counts": {"supported": 4},
            "label_counts": {"0": 2, "1": 2},
            "cross_label_components": 0,
            "split_rows_by_role_read": {"train": 4},
        },
        "output": {
            "path": str(folds_path),
            "sha256": hashlib.sha256(folds_raw).hexdigest(),
            "record_count": 4,
            "record_schema": FOLD_RECORD_SCHEMA,
        },
        "folds": {
            "component_cross_fold_count": 0,
            "fold_label_counts": {
                "0": {"0": 1, "1": 1},
                "1": {"0": 1, "1": 1},
            },
            "fold_total_counts": {"0": 2, "1": 2},
        },
        "ready_for": {
            "a2_training_authority": False,
            "local_whole_file_randomized_oof_diagnostic": True,
            "loop164_production_oof": False,
            "val_or_test_access": False,
            "candidate_promotion": False,
        },
        "decision": "local_content_group_diagnostic_folds_ready_not_production_scope",
    }
    summary_path.write_text(json.dumps(summary))
    return folds_path, summary_path, data_root


def test_loads_complete_train_only_component_isolated_folds(tmp_path: Path):
    folds_path, summary_path, data_root = _case(tmp_path)

    records, _summary = load_local_diagnostic_folds(
        folds_path=folds_path,
        summary_path=summary_path,
        data_root=data_root,
        expected_rows=4,
        fold_count=2,
        expected_seed=164,
        max_supported_file_bytes=64,
        expected_rows_per_fold=2,
        expected_rows_per_label_per_fold=1,
    )

    assert len(records) == 4
    assert [record.fold for record in records] == [0, 0, 1, 1]


def test_rejects_component_crossing_folds_even_when_summary_hash_matches(tmp_path: Path):
    folds_path, summary_path, data_root = _case(tmp_path)
    rows = [json.loads(line) for line in folds_path.read_text().splitlines()]
    rows[1]["content_component_id"] = rows[0]["content_component_id"]
    rows[0]["content_component_size"] = 2
    rows[1]["content_component_size"] = 2
    rows[1]["diagnostic_fold"] = 1
    folds_raw = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    folds_path.write_bytes(folds_raw)
    summary = json.loads(summary_path.read_text())
    summary["output"]["sha256"] = hashlib.sha256(folds_raw).hexdigest()
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(LocalOOFContractError, match="component crosses"):
        load_local_diagnostic_folds(
            folds_path=folds_path,
            summary_path=summary_path,
            data_root=data_root,
            expected_rows=4,
            fold_count=2,
            expected_seed=164,
            max_supported_file_bytes=64,
            expected_rows_per_fold=2,
            expected_rows_per_label_per_fold=1,
        )


def test_rejects_availability_size_that_disagrees_with_frozen_cap(tmp_path: Path):
    folds_path, summary_path, data_root = _case(tmp_path)
    rows = [json.loads(line) for line in folds_path.read_text().splitlines()]
    rows[0]["availability"] = "oversize"
    rows[0]["missing_reason"] = "oversize"
    rows[0]["source_size_bytes"] = 32
    folds_raw = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    folds_path.write_bytes(folds_raw)
    summary = json.loads(summary_path.read_text())
    summary["output"]["sha256"] = hashlib.sha256(folds_raw).hexdigest()
    summary["aggregate"]["availability_counts"] = {"oversize": 1, "supported": 3}
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(LocalOOFContractError, match="Oversize-row size"):
        load_local_diagnostic_folds(
            folds_path=folds_path,
            summary_path=summary_path,
            data_root=data_root,
            expected_rows=4,
            fold_count=2,
            expected_seed=164,
            max_supported_file_bytes=64,
            expected_rows_per_fold=2,
            expected_rows_per_label_per_fold=1,
        )


def test_fixed_metrics_use_preregistered_threshold_without_sweep():
    metrics = fixed_binary_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.2], threshold=0.5)

    assert metrics["f1"] == 0.5
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1

    tie = fixed_binary_metrics([0], [0.5], threshold=0.5)
    assert tie["false_positive"] == 1


def _resource_guard_payload(*, command: list[str], created_at: float) -> dict[str, object]:
    return {
        "schema": GUARD_SCHEMA,
        "guard_ready": True,
        "receipt": build_guard_receipt(
            target_scripts=oof_controller._resource_guard_targets(),
            command=command,
            created_at=created_at,
            cwd=PROJECT_ROOT,
        ),
    }


def test_resource_guard_is_fresh_and_bound_to_exact_oof_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    guard_path = tmp_path / "guard.json"
    argv: list[str] = []
    command = oof_controller._expected_resource_guard_command(argv)
    guard_path.write_text(json.dumps(_resource_guard_payload(command=command, created_at=1000.0)))
    monkeypatch.setattr(oof_controller, "RESOURCE_GUARD", guard_path)

    validation = oof_controller._validate_resource_guard(guard_path, argv=argv, now=1001.0)

    assert validation["validation"]["valid"] is True
    assert validation["validation"]["age_seconds"] == 1.0


@pytest.mark.parametrize("failure", ["expired", "wrong_command"])
def test_resource_guard_rejects_stale_or_wrong_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    guard_path = tmp_path / "guard.json"
    argv: list[str] = []
    command = oof_controller._expected_resource_guard_command(argv)
    if failure == "wrong_command":
        command = [sys.executable, "wrong-controller.py"]
    created_at = time.time() - (
        oof_controller.RESOURCE_GUARD_MAX_AGE_SECONDS + 1 if failure == "expired" else 0
    )
    guard_path.write_text(json.dumps(_resource_guard_payload(command=command, created_at=created_at)))
    monkeypatch.setattr(oof_controller, "RESOURCE_GUARD", guard_path)

    with pytest.raises(oof_controller.OOFRunError, match="resource guard was rejected"):
        oof_controller._validate_resource_guard(guard_path, argv=argv, now=time.time())


def _local_authorization_payload(resource_guard_path: Path) -> dict[str, object]:
    argv: list[str] = []
    frozen_paths = oof_controller._frozen_protocol_paths(
        oof_controller.DEFAULT_CONFIG,
        oof_controller.DEFAULT_FOLDS,
        oof_controller.DEFAULT_FOLDS_SUMMARY,
    )
    binding_paths = {
        **frozen_paths,
        "controller": Path(oof_controller.__file__).resolve(),
        "resource_guard": resource_guard_path,
    }
    return {
        "schema": "axon_loop164_local_whole_file_oof_authorization_v1",
        "loop_id": "loop164_whole_file_residual_expert",
        "claim_scope": oof_controller.CLAIM_SCOPE,
        "authorization": {
            "authorized": True,
            "authority_type": "user_explicit_local_custody_delegation",
            "authorization_date": "2026-07-13",
            "public_key_required": False,
            "external_a2_training_authority": False,
            "val_test_or_full_access": False,
            "candidate_promotion": False,
            "checkpoint_or_model_state_write": False,
            "threshold_selection": False,
        },
        "command": oof_controller._expected_resource_guard_command(argv),
        "bindings": {
            name: {
                "path": str(path.resolve(strict=True)),
                "sha256": oof_controller.sha256_file(path.resolve(strict=True)),
            }
            for name, path in binding_paths.items()
        },
        "outputs": {
            "predictions": str(oof_controller.DEFAULT_PREDICTIONS.resolve(strict=False)),
            "report": str(oof_controller.DEFAULT_REPORT.resolve(strict=False)),
        },
        "decision": "authorized_local_train_diagnostic_only",
    }


def test_local_authorization_requires_no_public_key_and_forbids_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resource_guard_path = tmp_path / "guard.json"
    resource_guard_path.write_text("{}")
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(_local_authorization_payload(resource_guard_path)))
    monkeypatch.setattr(oof_controller, "LOCAL_AUTHORIZATION", authorization_path)

    validation = oof_controller._validate_local_authorization(
        authorization_path,
        config_path=oof_controller.DEFAULT_CONFIG,
        folds_path=oof_controller.DEFAULT_FOLDS,
        folds_summary_path=oof_controller.DEFAULT_FOLDS_SUMMARY,
        resource_guard_path=resource_guard_path,
        argv=[],
    )

    assert validation["public_key_required"] is False
    assert validation["external_a2_training_authority"] is False


def test_local_authorization_rejects_fabricated_external_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resource_guard_path = tmp_path / "guard.json"
    resource_guard_path.write_text("{}")
    authorization_path = tmp_path / "authorization.json"
    payload = _local_authorization_payload(resource_guard_path)
    payload["authorization"]["public_key_required"] = True  # type: ignore[index]
    payload["authorization"]["external_a2_training_authority"] = True  # type: ignore[index]
    authorization_path.write_text(json.dumps(payload))
    monkeypatch.setattr(oof_controller, "LOCAL_AUTHORIZATION", authorization_path)

    with pytest.raises(oof_controller.OOFRunError, match="authorization scope drifted"):
        oof_controller._validate_local_authorization(
            authorization_path,
            config_path=oof_controller.DEFAULT_CONFIG,
            folds_path=oof_controller.DEFAULT_FOLDS,
            folds_summary_path=oof_controller.DEFAULT_FOLDS_SUMMARY,
            resource_guard_path=resource_guard_path,
            argv=[],
        )
