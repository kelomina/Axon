from __future__ import annotations

import csv
import hashlib
import inspect
from pathlib import Path
from concurrent.futures import Future

import numpy as np

from scripts import materialize_loop127_content_pe_sidecars as loop127_sidecars
from scripts.materialize_loop127_content_pe_sidecars import materialize_loop127_content_pe_sidecars
from scripts.train_stage2_cache_matrix import CONTENT_PE_V2_FEATURE_NAMES
from src.kvd_features.content_pe_v1 import CONTENT_PE_FEATURE_NAMES


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "cache_path", "label", "split", "sample_index"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _row(tmp_path: Path, *, sha: str | None = None, split: str, sample_index: str, label: str = "1") -> dict[str, str]:
    source_path = tmp_path / f"{split}-{sample_index}.bin"
    content = f"not-a-pe-{split}-{sample_index}".encode("utf-8")
    source_path.write_bytes(content)
    source_sha = sha if sha is not None else hashlib.sha256(content).hexdigest()
    return {
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "cache_path": str(tmp_path / f"{split}-{sample_index}.npz"),
        "label": label,
        "split": split,
        "sample_index": sample_index,
    }


def test_materialize_loop127_blocks_invalid_hash_without_path_hash_fallback(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_predictions(train_csv, [_row(tmp_path, sha="bad", split="train", sample_index="0")])
    _write_predictions(val_csv, [_row(tmp_path, split="val", sample_index="1")])

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
    train_row = _row(tmp_path, split="train", sample_index="0")
    train_sha = train_row["source_sha256"]
    _write_predictions(train_csv, [train_row])
    _write_predictions(val_csv, [_row(tmp_path, split="val", sample_index="1", label="0")])
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    np.savez(tmp_path / "v1" / f"{train_sha}.npz", features=np.zeros(3, dtype=np.float32))
    np.savez(tmp_path / "v2" / f"{train_sha}.npz", features=np.zeros(3, dtype=np.float32))

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
    with np.load(tmp_path / "v1" / f"{train_sha}.npz", allow_pickle=False) as data:
        assert data["features"].shape == (len(CONTENT_PE_FEATURE_NAMES),)
    with np.load(tmp_path / "v2" / f"{train_sha}.npz", allow_pickle=False) as data:
        assert data["features"].shape == (len(CONTENT_PE_V2_FEATURE_NAMES),)


def test_materialize_loop127_blocks_duplicate_sample_index_but_allows_duplicate_hash_audit(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_predictions(
        train_csv,
        [
            _row(tmp_path, split="train", sample_index="0"),
            _row(tmp_path, split="train", sample_index="0"),
        ],
    )
    _write_predictions(val_csv, [_row(tmp_path, split="val", sample_index="1")])

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


def test_materialize_loop127_cross_split_overlap_is_bounded(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    shared_rows = [_row(tmp_path, split="train", sample_index=str(idx)) for idx in range(12)]
    val_rows = [
        {**row, "split": "val", "sample_index": str(idx)}
        for idx, row in enumerate(shared_rows)
    ]
    _write_predictions(train_csv, shared_rows)
    _write_predictions(val_csv, val_rows)

    report = materialize_loop127_content_pe_sidecars(
        train_predictions=train_csv,
        val_predictions=val_csv,
        content_pe_cache_dir=tmp_path / "v1",
        content_pe_v2_cache_dir=tmp_path / "v2",
        output_json=tmp_path / "report.json",
        workers=1,
    )

    assert report["ready_for_readiness_recheck"] is False
    assert report["cross_split_source_sha256_overlap_count"] == 12
    assert len(report["cross_split_source_sha256_overlap_examples"]) == loop127_sidecars.MAX_FAILURE_EXAMPLES
    assert report["unique_source_sha256_rows"] == 12
    assert "train_val_source_sha256_overlap" in report["blockers"]


def test_materialize_loop127_records_worker_exceptions(tmp_path: Path, monkeypatch):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_predictions(train_csv, [_row(tmp_path, split="train", sample_index="0")])
    _write_predictions(val_csv, [_row(tmp_path, split="val", sample_index="1")])

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, payload):
            future = Future()
            future.set_exception(OSError("simulated worker failure"))
            return future

    monkeypatch.setattr(loop127_sidecars, "ProcessPoolExecutor", FakeExecutor)

    report = materialize_loop127_content_pe_sidecars(
        train_predictions=train_csv,
        val_predictions=val_csv,
        content_pe_cache_dir=tmp_path / "v1",
        content_pe_v2_cache_dir=tmp_path / "v2",
        output_json=tmp_path / "report.json",
        workers=2,
    )

    assert report["ready_for_readiness_recheck"] is False
    assert "sidecar_materialization_failures" in report["blockers"]
    assert report["failure_examples"][0]["failure_reason"].startswith("worker_exception:OSError")


def test_materialize_loop127_avoids_full_payload_and_result_lists():
    source = inspect.getsource(loop127_sidecars.materialize_loop127_content_pe_sidecars)
    module_source = inspect.getsource(loop127_sidecars)

    assert "train_rows + val_rows" not in source
    assert "payloads" not in source
    assert "executor.map" not in source
    assert "results =" not in source
    assert "pending[future]" in source
    assert "sorted(train_sha & val_sha)" not in source
    assert "train_sha | val_sha" not in source
    assert "list(reader)" not in module_source
