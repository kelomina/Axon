from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.materialize_loop127_content_pe_sidecars import materialize_loop127_content_pe_sidecars
from scripts.train_stage2_cache_matrix import CONTENT_PE_V2_FEATURE_NAMES
from src.kvd_features.content_pe_v1 import CONTENT_PE_FEATURE_NAMES


SHA_A = "a" * 64
SHA_B = "b" * 64


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "cache_path", "label", "split", "sample_index"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _row(tmp_path: Path, *, sha: str, split: str, sample_index: str, label: str = "1") -> dict[str, str]:
    source_path = tmp_path / f"{split}-{sample_index}.bin"
    source_path.write_bytes(b"not-a-pe")
    return {
        "source_path": str(source_path),
        "source_sha256": sha,
        "cache_path": str(tmp_path / f"{split}-{sample_index}.npz"),
        "label": label,
        "split": split,
        "sample_index": sample_index,
    }


def test_materialize_loop127_blocks_invalid_hash_without_path_hash_fallback(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_predictions(train_csv, [_row(tmp_path, sha="bad", split="train", sample_index="0")])
    _write_predictions(val_csv, [_row(tmp_path, sha=SHA_B, split="val", sample_index="1")])

    report = materialize_loop127_content_pe_sidecars(
        train_predictions=train_csv,
        val_predictions=val_csv,
        content_pe_cache_dir=tmp_path / "v1",
        content_pe_v2_cache_dir=tmp_path / "v2",
        output_json=tmp_path / "report.json",
        workers=1,
    )

    assert report["ready_for_readiness_recheck"] is False
    assert report["train"]["issue_counts"]["invalid_source_sha256"] == 1
    assert "train_inputs_not_materializable" in report["blockers"]
    assert not list((tmp_path / "v1").glob("*.npz"))


def test_materialize_loop127_refreshes_invalid_sidecars_and_reports_zero_features(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_predictions(train_csv, [_row(tmp_path, sha=SHA_A, split="train", sample_index="0")])
    _write_predictions(val_csv, [_row(tmp_path, sha=SHA_B, split="val", sample_index="1", label="0")])
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    np.savez(tmp_path / "v1" / f"{SHA_A}.npz", features=np.zeros(3, dtype=np.float32))
    np.savez(tmp_path / "v2" / f"{SHA_A}.npz", features=np.zeros(3, dtype=np.float32))

    report = materialize_loop127_content_pe_sidecars(
        train_predictions=train_csv,
        val_predictions=val_csv,
        content_pe_cache_dir=tmp_path / "v1",
        content_pe_v2_cache_dir=tmp_path / "v2",
        output_json=tmp_path / "report.json",
        workers=1,
    )

    assert report["ready_for_readiness_recheck"] is True
    assert report["counts"]["v1_refreshed_invalid"] == 1
    assert report["counts"]["v2_refreshed_invalid"] == 1
    with np.load(tmp_path / "v1" / f"{SHA_A}.npz", allow_pickle=False) as data:
        assert data["features"].shape == (len(CONTENT_PE_FEATURE_NAMES),)
    with np.load(tmp_path / "v2" / f"{SHA_A}.npz", allow_pickle=False) as data:
        assert data["features"].shape == (len(CONTENT_PE_V2_FEATURE_NAMES),)


def test_materialize_loop127_blocks_duplicate_sample_index_but_allows_duplicate_hash_audit(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_predictions(
        train_csv,
        [
            _row(tmp_path, sha=SHA_A, split="train", sample_index="0"),
            _row(tmp_path, sha=SHA_B, split="train", sample_index="0"),
        ],
    )
    _write_predictions(val_csv, [_row(tmp_path, sha="c" * 64, split="val", sample_index="1")])

    report = materialize_loop127_content_pe_sidecars(
        train_predictions=train_csv,
        val_predictions=val_csv,
        content_pe_cache_dir=tmp_path / "v1",
        content_pe_v2_cache_dir=tmp_path / "v2",
        output_json=tmp_path / "report.json",
        workers=1,
    )

    assert report["ready_for_readiness_recheck"] is False
    assert report["train"]["issue_counts"]["duplicate_sample_index"] == 1
    assert "train_inputs_not_materializable" in report["blockers"]
