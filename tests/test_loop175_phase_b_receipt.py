from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import src.loop175.phase_b_receipt as phase_b_receipt
from src.loop175.phase_b_receipt import (
    ARM_FOLD_SCHEMA,
    PhaseBReceiptError,
    aggregate_seed41_receipt,
    artifact_paths,
    load_arm_fold_artifact,
    write_arm_fold_artifact,
)

COMMITMENT = "a" * 64
RUNTIME = {
    "coverage": 1.0,
    "class_coverage_gap": 0.0,
    "silent_drops": 0,
    "oom": False,
    "timeout": False,
    "nonfinite": False,
    "gpu_allocated_bytes": 0,
    "rss_bytes": 0,
    "new_disk_bytes": 0,
    "maximum_fold_wall_seconds": 1.0,
    "seed_wall_seconds": 5.0,
}


def _write_fixture(directory: Path, *, rows: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    folds = np.repeat(np.arange(5, dtype=np.int8), rows // 5)
    labels = np.tile(np.asarray([0, 1], dtype=np.uint8), rows // 2)
    for arm in ("A", "B", "C", "D", "E"):
        for fold in range(5):
            indices = np.flatnonzero(folds == fold).astype(np.int64)
            if arm == "C":
                scores = labels[indices].astype(np.float64)
            elif arm == "B":
                scores = np.full(indices.size, 0.5, dtype=np.float64)
            else:
                scores = 1.0 - labels[indices].astype(np.float64)
            write_arm_fold_artifact(
                directory,
                arm=arm,
                fold=fold,
                holdout_indices=indices,
                scores=scores,
                fit_count=rows - indices.size,
                protocol_commitment=COMMITMENT,
                cache_commitment="b" * 64,
                config_commitment="c" * 64,
                model_commitment=(f"{ord(arm):02x}{fold:02x}" + "d" * 60),
                runtime_commitment=hashlib.sha256(
                    f"runtime-{arm}-{fold}".encode("ascii")
                ).hexdigest(),
            )
    return labels, folds


def test_arm_fold_artifact_is_numeric_canonical_and_immutable(tmp_path: Path) -> None:
    indices = np.arange(4_000, dtype=np.int64)
    scores = np.linspace(0.0, 1.0, indices.size, dtype=np.float64)
    artifact = write_arm_fold_artifact(
        tmp_path,
        arm="A",
        fold=0,
        holdout_indices=indices,
        scores=scores,
        fit_count=16_000,
        protocol_commitment=COMMITMENT,
        cache_commitment="b" * 64,
        config_commitment="c" * 64,
        model_commitment="d" * 64,
        runtime_commitment="e" * 64,
    )
    loaded = load_arm_fold_artifact(artifact.npz_path, artifact.metadata_path)
    np.testing.assert_array_equal(loaded.holdout_indices, indices)
    np.testing.assert_array_equal(loaded.scores, scores)
    metadata = json.loads(artifact.metadata_path.read_text(encoding="ascii"))
    assert metadata["schema"] == ARM_FOLD_SCHEMA
    assert not {"path", "sha256", "source_sha"}.intersection(metadata)
    with pytest.raises(PhaseBReceiptError, match="overwrite"):
        write_arm_fold_artifact(
            tmp_path,
            arm="A",
            fold=0,
            holdout_indices=indices,
            scores=scores,
            fit_count=16_000,
            protocol_commitment=COMMITMENT,
            cache_commitment="b" * 64,
            config_commitment="c" * 64,
            model_commitment="d" * 64,
            runtime_commitment="e" * 64,
        )


def test_numeric_and_metadata_tamper_fail_closed(tmp_path: Path) -> None:
    indices = np.arange(4_000, dtype=np.int64)
    artifact = write_arm_fold_artifact(
        tmp_path,
        arm="A",
        fold=0,
        holdout_indices=indices,
        scores=np.zeros(indices.size, dtype=np.float64),
        fit_count=16_000,
        protocol_commitment=COMMITMENT,
        cache_commitment="b" * 64,
        config_commitment="c" * 64,
        model_commitment="d" * 64,
        runtime_commitment="e" * 64,
    )
    with np.load(artifact.npz_path, allow_pickle=False) as archive:
        changed = np.asarray(archive["scores"]).copy()
        changed[0] = 1.0
        np.savez(artifact.npz_path, holdout_indices=archive["holdout_indices"], scores=changed)
    with pytest.raises(PhaseBReceiptError, match="commitment"):
        load_arm_fold_artifact(artifact.npz_path, artifact.metadata_path)

    metadata = json.loads(artifact.metadata_path.read_text(encoding="ascii"))
    metadata["schema"] = "tampered"
    artifact.metadata_path.write_bytes(
        (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    )
    with pytest.raises(PhaseBReceiptError, match="schema"):
        load_arm_fold_artifact(artifact.npz_path, artifact.metadata_path)

    metadata["schema"] = ARM_FOLD_SCHEMA
    metadata["unexpected_path"] = "not-allowed"
    artifact.metadata_path.write_bytes(
        (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    )
    with pytest.raises(PhaseBReceiptError, match="fields|path"):
        load_arm_fold_artifact(artifact.npz_path, artifact.metadata_path)


def test_aggregate_rejects_missing_or_wrong_fold_artifact(tmp_path: Path) -> None:
    labels, folds = _write_fixture(tmp_path)
    component_ids = np.asarray([f"component-{index}" for index in range(labels.size)], dtype=object)
    numeric_path, _metadata_path = artifact_paths(tmp_path, arm="E", fold=4)
    numeric_path.unlink()
    with pytest.raises(PhaseBReceiptError, match="missing|inaccessible"):
        aggregate_seed41_receipt(
            tmp_path,
            labels=labels,
            folds=folds,
            component_ids=component_ids,
            protocol_sha256=COMMITMENT,
            runtime=RUNTIME,
            output=tmp_path / "receipt.json",
            bootstrap_replicates=32,
        )

    _write_fixture(tmp_path / "wrong")
    wrong_path, wrong_metadata = artifact_paths(tmp_path / "wrong", arm="A", fold=0)
    with np.load(wrong_path, allow_pickle=False) as archive:
        wrong_indices = archive["holdout_indices"].copy()
        wrong_scores = archive["scores"].copy()
    wrong_indices += 4_000
    np.savez(wrong_path, holdout_indices=wrong_indices, scores=wrong_scores)
    wrong_payload = json.loads(wrong_metadata.read_text(encoding="ascii"))
    wrong_payload["numeric_commitment"] = phase_b_receipt._numeric_commitment(
        wrong_indices, wrong_scores
    )
    wrong_metadata.write_bytes(
        (json.dumps(wrong_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    )
    with pytest.raises(PhaseBReceiptError, match="fold authority"):
        aggregate_seed41_receipt(
            tmp_path / "wrong",
            labels=labels,
            folds=folds,
            component_ids=component_ids,
            protocol_sha256=COMMITMENT,
            runtime=RUNTIME,
            output=tmp_path / "wrong_receipt.json",
            bootstrap_replicates=32,
        )


def test_successful_seed41_aggregation_is_train_only_and_one_time(tmp_path: Path) -> None:
    labels, folds = _write_fixture(tmp_path)
    component_ids = np.asarray([f"component-{index}" for index in range(labels.size)], dtype=object)
    output = tmp_path / "seed41_receipt.json"
    receipt = aggregate_seed41_receipt(
        tmp_path,
        labels=labels,
        folds=folds,
        component_ids=component_ids,
        protocol_sha256=COMMITMENT,
        runtime=RUNTIME,
        output=output,
        bootstrap_replicates=64,
    )
    assert receipt["decision"] == "seed41_pass_allow_seed42_43"
    assert receipt["artifact_count"] == 25
    assert len(receipt["runtime_commitments"]) == 25
    assert len(set(receipt["runtime_commitments"].values())) == 25
    assert receipt["val_rows_opened"] == 0
    assert receipt["test10k_rows_opened"] == 0
    assert receipt["full_test_rows_opened"] == 0
    assert receipt["val_test_or_full_rows_opened"] == 0
    assert output.read_bytes() == output.read_bytes()
    with pytest.raises(PhaseBReceiptError, match="overwrite"):
        aggregate_seed41_receipt(
            tmp_path,
            labels=labels,
            folds=folds,
            component_ids=component_ids,
            protocol_sha256=COMMITMENT,
            runtime=RUNTIME,
            output=output,
            bootstrap_replicates=32,
        )
