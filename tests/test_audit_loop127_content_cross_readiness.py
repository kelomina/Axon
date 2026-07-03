from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.audit_loop127_content_cross_readiness import (
    audit_loop127_content_cross_readiness,
    sidecar_cache_path,
)


FIELDS = ["source_path", "source_sha256", "cache_path", "label", "split", "sample_index", "prob_malicious"]
SHA_A = "a" * 64
SHA_B = "b" * 64


def _write_main_cache(path: Path, *, sha: str, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        label=np.array(int(label), dtype=np.int64),
        source_sha256=np.array(sha),
    )
    return path


def _write_sidecar(path: Path, *, dim: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=np.zeros(dim, dtype=np.float32))
    return path


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _row(tmp_path: Path, *, sha: str, split: str, label: str = "1", sample_index: str = "0") -> dict[str, str]:
    cache_path = _write_main_cache(tmp_path / "feature_cache" / f"{split}-{sample_index}.npz", sha=sha, label=label)
    return {
        "source_path": str(tmp_path / "raw" / f"{split}-{sample_index}.bin"),
        "source_sha256": sha,
        "cache_path": str(cache_path),
        "label": label,
        "split": split,
        "sample_index": sample_index,
        "prob_malicious": "0.75",
    }


def _touch_sidecars(tmp_path: Path, sha: str) -> tuple[Path, Path]:
    v1_dir = tmp_path / "content_pe_cache_v1"
    v2_dir = tmp_path / "content_pe_v2_cache"
    _write_sidecar(v1_dir / f"{sha}.npz", dim=100)
    _write_sidecar(v2_dir / f"{sha}.npz", dim=182)
    return v1_dir, v2_dir


def _audit(tmp_path: Path, train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> dict:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_csv(train_csv, train_rows)
    _write_csv(val_csv, val_rows)
    return audit_loop127_content_cross_readiness(
        train_predictions=train_csv,
        val_predictions=val_csv,
        content_pe_cache_dir=tmp_path / "content_pe_cache_v1",
        content_pe_v2_cache_dir=tmp_path / "content_pe_v2_cache",
        output_json=tmp_path / "readiness.json",
        expected_train_rows=len(train_rows),
        expected_val_rows=len(val_rows),
        expected_test_rows=0,
        expected_total_rows=len(train_rows) + len(val_rows),
    )


def test_loop127_readiness_allows_complete_train_and_val_inputs(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_A)
    _touch_sidecars(tmp_path, SHA_B)

    payload = _audit(
        tmp_path,
        [_row(tmp_path, sha=SHA_A, split="train", sample_index="0")],
        [_row(tmp_path, sha=SHA_B, split="val", sample_index="1", label="0")],
    )

    assert payload["ready_for_loop43_val_only"] is True
    assert payload["blockers"] == []
    assert json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))["train"]["ready"] is True


def test_loop127_readiness_blocks_missing_prob_malicious_column(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_A)
    _touch_sidecars(tmp_path, SHA_B)
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_row = _row(tmp_path, sha=SHA_A, split="train")
    train_row.pop("prob_malicious")
    _write_csv(train_csv, [train_row], [field for field in FIELDS if field != "prob_malicious"])
    _write_csv(val_csv, [_row(tmp_path, sha=SHA_B, split="val")])

    payload = audit_loop127_content_cross_readiness(
        train_predictions=train_csv,
        val_predictions=val_csv,
        content_pe_cache_dir=tmp_path / "content_pe_cache_v1",
        content_pe_v2_cache_dir=tmp_path / "content_pe_v2_cache",
        output_json=tmp_path / "readiness.json",
        expected_train_rows=1,
        expected_val_rows=1,
        expected_test_rows=0,
        expected_total_rows=2,
    )

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["missing_columns"] == ["prob_malicious"]
    assert payload["train"]["issue_counts"]["missing_required_columns"] == 1


def test_loop127_readiness_blocks_missing_content_sidecar(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_B)
    _write_sidecar(tmp_path / "content_pe_cache_v1" / f"{SHA_A}.npz", dim=100)

    payload = _audit(
        tmp_path,
        [_row(tmp_path, sha=SHA_A, split="train")],
        [_row(tmp_path, sha=SHA_B, split="val", sample_index="1")],
    )

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["issue_counts"]["content_pe_v2_missing"] == 1
    assert payload["blockers"] == ["train_inputs_not_ready"]


def test_loop127_readiness_blocks_unexpected_split(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_A)
    _touch_sidecars(tmp_path, SHA_B)

    payload = _audit(
        tmp_path,
        [_row(tmp_path, sha=SHA_A, split="val")],
        [_row(tmp_path, sha=SHA_B, split="val")],
    )

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["issue_counts"]["unexpected_split"] == 1


def test_loop127_readiness_blocks_invalid_hash_without_source_path_fallback(tmp_path: Path):
    bad_sha = "not-a-sha"
    stem_cache = _write_sidecar(tmp_path / "content_pe_cache_v1" / "train-0.npz", dim=100)
    row = _row(tmp_path, sha=bad_sha, split="train")
    row["source_path"] = str(tmp_path / "raw" / "train-0.exe")
    _touch_sidecars(tmp_path, SHA_B)

    payload = _audit(tmp_path, [row], [_row(tmp_path, sha=SHA_B, split="val")])

    assert sidecar_cache_path(stem_cache.parent, row) is None
    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["issue_counts"]["invalid_source_sha256"] == 1


def test_loop127_readiness_blocks_blank_cache_path_even_when_cwd_exists(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_A)
    _touch_sidecars(tmp_path, SHA_B)
    train_row = _row(tmp_path, sha=SHA_A, split="train")
    train_row["cache_path"] = ""

    payload = _audit(tmp_path, [train_row], [_row(tmp_path, sha=SHA_B, split="val")])

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["issue_counts"]["blank_cache_path"] == 1


def test_loop127_readiness_blocks_nan_probability_and_bad_sample_index(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_A)
    _touch_sidecars(tmp_path, SHA_B)
    train_row = _row(tmp_path, sha=SHA_A, split="train", sample_index="x")
    train_row["prob_malicious"] = "nan"

    payload = _audit(tmp_path, [train_row], [_row(tmp_path, sha=SHA_B, split="val")])

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["issue_counts"]["invalid_sample_index"] == 1
    assert payload["train"]["issue_counts"]["prob_malicious_out_of_range"] == 1


def test_loop127_readiness_blocks_cache_hash_mismatch(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_A)
    _touch_sidecars(tmp_path, SHA_B)
    train_row = _row(tmp_path, sha=SHA_A, split="train")
    _write_main_cache(Path(train_row["cache_path"]), sha=SHA_B, label="1")

    payload = _audit(tmp_path, [train_row], [_row(tmp_path, sha=SHA_B, split="val")])

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["issue_counts"]["cache_source_sha256_mismatch"] == 1


def test_loop127_readiness_blocks_sidecar_shape_and_nonfinite_values(tmp_path: Path):
    _write_sidecar(tmp_path / "content_pe_cache_v1" / f"{SHA_A}.npz", dim=99)
    path = tmp_path / "content_pe_v2_cache" / f"{SHA_A}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=np.array([np.nan] + [0.0] * 181, dtype=np.float32))
    _touch_sidecars(tmp_path, SHA_B)

    payload = _audit(tmp_path, [_row(tmp_path, sha=SHA_A, split="train")], [_row(tmp_path, sha=SHA_B, split="val")])

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["issue_counts"]["content_pe_v1_feature_shape_mismatch"] == 1
    assert payload["train"]["issue_counts"]["content_pe_v2_feature_nonfinite"] == 1


def test_loop127_readiness_blocks_duplicate_and_cross_split_identity(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_A)
    row_a = _row(tmp_path, sha=SHA_A, split="train", sample_index="0")
    row_b = _row(tmp_path, sha=SHA_A, split="train", sample_index="0")
    row_b["sample_index"] = "1"
    val_row = _row(tmp_path, sha=SHA_A, split="val", sample_index="0")

    payload = _audit(tmp_path, [row_a, row_b], [val_row])

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["issue_counts"]["duplicate_source_sha256"] == 1
    assert payload["cross_split_identity"]["issue_counts"]["train_val_source_sha256_overlap"] == 1
    assert payload["cross_split_identity"]["issue_counts"]["train_val_sample_index_overlap"] == 1
    assert "train_val_identity_overlap" in payload["blockers"]


def test_loop127_readiness_blocks_duplicate_sample_index(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_A)
    _touch_sidecars(tmp_path, SHA_B)
    _touch_sidecars(tmp_path, "c" * 64)
    row_a = _row(tmp_path, sha=SHA_A, split="train", sample_index="0")
    row_b = _row(tmp_path, sha=SHA_B, split="train", sample_index="0")

    payload = _audit(tmp_path, [row_a, row_b], [_row(tmp_path, sha="c" * 64, split="val", sample_index="2")])

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["issue_counts"]["duplicate_sample_index"] == 1


def test_loop127_readiness_blocks_forbidden_extra_columns(tmp_path: Path):
    _touch_sidecars(tmp_path, SHA_A)
    _touch_sidecars(tmp_path, SHA_B)
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_row = _row(tmp_path, sha=SHA_A, split="train")
    train_row["filename_hint"] = "sample.exe"
    _write_csv(train_csv, [train_row], FIELDS + ["filename_hint"])
    _write_csv(val_csv, [_row(tmp_path, sha=SHA_B, split="val")])

    payload = audit_loop127_content_cross_readiness(
        train_predictions=train_csv,
        val_predictions=val_csv,
        content_pe_cache_dir=tmp_path / "content_pe_cache_v1",
        content_pe_v2_cache_dir=tmp_path / "content_pe_v2_cache",
        output_json=tmp_path / "readiness.json",
        expected_train_rows=1,
        expected_val_rows=1,
        expected_test_rows=0,
        expected_total_rows=2,
    )

    assert payload["ready_for_loop43_val_only"] is False
    assert payload["train"]["forbidden_extra_columns"] == ["filename_hint"]
    assert payload["train"]["issue_counts"]["forbidden_extra_columns"] == 1
