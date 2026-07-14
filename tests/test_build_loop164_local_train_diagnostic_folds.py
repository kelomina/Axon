from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_loop164_local_train_diagnostic_folds import (  # noqa: E402
    build_local_train_diagnostic_folds,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_split(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "label", "sample_index", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_builds_balanced_train_only_content_group_folds(tmp_path: Path):
    canonical_root = tmp_path / "canonical"
    data_root = tmp_path / "worktree"
    canonical_root.mkdir()
    data_root.mkdir()
    rows: list[dict[str, object]] = []
    shared_chunk = b"A" * 16
    sample_index = 0
    for label in (0, 1):
        for index in range(5):
            if label == 0 and index == 0:
                raw = shared_chunk + b"B" * 16
            elif label == 1 and index == 0:
                raw = shared_chunk + b"C" * 16
            else:
                raw = bytes([label * 10 + index + 1]) * 32
            relative = Path(f"class-{label}") / f"sample-{index}.bin"
            materialized = data_root / relative
            materialized.parent.mkdir(parents=True, exist_ok=True)
            materialized.write_bytes(raw)
            rows.append(
                {
                    "source_path": str(canonical_root / relative),
                    "source_sha256": _sha256(raw),
                    "label": label,
                    "sample_index": sample_index,
                    "split": "train",
                }
            )
            sample_index += 1
    rows.append(
        {
            "source_path": str(canonical_root / "heldout.bin"),
            "source_sha256": "f" * 64,
            "label": 0,
            "sample_index": 999,
            "split": "val",
        }
    )
    split_path = tmp_path / "split.csv"
    _write_split(split_path, rows)
    output_path = tmp_path / "folds.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = build_local_train_diagnostic_folds(
        split_csv=split_path,
        canonical_source_root=canonical_root,
        data_root=data_root,
        output_jsonl=output_path,
        summary_json=summary_path,
        fold_count=5,
        chunk_size=16,
        signature_size=8,
        band_size=1,
        max_bucket_size=16,
        similarity_threshold=0.5,
        max_supported_file_bytes=1024,
        expected_train_rows=10,
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 10
    assert summary["aggregate"]["split_rows_by_role_read"] == {"train": 10}
    assert summary["inputs"]["canonical_split_train_prefix"]["heldout_rows_read"] == 0
    assert summary["aggregate"]["verified_source_sha256"] == 10
    assert summary["folds"]["component_cross_fold_count"] == 0
    assert all(
        counts == {"0": 1, "1": 1}
        for counts in summary["folds"]["fold_label_counts"].values()
    )
    assert summary["ready_for"]["loop164_production_oof"] is False
    by_sample = {record["sample_index"]: record for record in records}
    assert by_sample[0]["content_component_id"] == by_sample[5]["content_component_id"]
    assert by_sample[0]["diagnostic_fold"] == by_sample[5]["diagnostic_fold"]
    assert all(record["split_role"] == "train" for record in records)
    assert "heldout" not in output_path.read_text()


def test_missing_train_source_stays_in_denominator_as_read_failure(tmp_path: Path):
    canonical_root = tmp_path / "canonical"
    data_root = tmp_path / "worktree"
    canonical_root.mkdir()
    data_root.mkdir()
    rows = []
    sample_index = 0
    for label in (0, 1):
        for index in range(2):
            relative = Path(f"class-{label}") / f"sample-{index}.bin"
            raw = bytes([label * 10 + index + 1]) * 32
            if not (label == 1 and index == 1):
                materialized = data_root / relative
                materialized.parent.mkdir(parents=True, exist_ok=True)
                materialized.write_bytes(raw)
            rows.append(
                {
                    "source_path": str(canonical_root / relative),
                    "source_sha256": _sha256(raw),
                    "label": label,
                    "sample_index": sample_index,
                    "split": "train",
                }
            )
            sample_index += 1
    split_path = tmp_path / "split.csv"
    _write_split(split_path, rows)

    summary = build_local_train_diagnostic_folds(
        split_csv=split_path,
        canonical_source_root=canonical_root,
        data_root=data_root,
        output_jsonl=tmp_path / "folds.jsonl",
        summary_json=tmp_path / "summary.json",
        fold_count=2,
        chunk_size=16,
        signature_size=4,
        band_size=1,
        max_bucket_size=8,
        similarity_threshold=1.0,
        max_supported_file_bytes=1024,
        expected_train_rows=4,
    )

    assert summary["aggregate"]["canonical_train_rows"] == 4
    assert summary["aggregate"]["availability_counts"] == {
        "read_failure": 1,
        "supported": 3,
    }
    assert summary["output"]["record_count"] == 4
