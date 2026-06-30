from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_replacement_candidate_pool import build_candidate_pool  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_pe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ" + b"\0" * 32)


def test_candidate_pool_excludes_current_split_rows_and_counts_labels():
    used_sha = "a" * 64
    unused_benign_sha = "b" * 64
    unused_mal_sha = "c" * 64
    with _case_dir("replacement_pool_basic") as tmp_path:
        data_dir = tmp_path / "data"
        _write_pe(data_dir / "待加入白名单" / used_sha)
        _write_pe(data_dir / "待加入白名单" / unused_benign_sha)
        _write_pe(data_dir / "待拉黑" / f"{unused_mal_sha}.exe")
        split_csv = tmp_path / "split.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": str(data_dir / "待加入白名单" / used_sha), "label": "0", "sample_index": "0", "split": "train"}],
        )

        rows, summary = build_candidate_pool(data_dir=data_dir, split_csv=split_csv)

    by_path = {Path(row["source_path"]).name: row for row in rows}
    assert used_sha not in by_path
    assert unused_benign_sha in by_path
    assert f"{unused_mal_sha}.exe" in by_path
    assert summary["label_counts"] == {"0": 1, "1": 1}
    assert summary["per_label"]["0"]["already_used_in_split"] == 1


def test_candidate_pool_reports_required_replacement_shortfall():
    with _case_dir("replacement_pool_shortfall") as tmp_path:
        data_dir = tmp_path / "data"
        _write_pe(data_dir / "待加入白名单" / ("b" * 64))
        split_csv = tmp_path / "split.csv"
        _write_csv(split_csv, ["source_path", "label", "sample_index", "split"], [])

        _rows, summary = build_candidate_pool(
            data_dir=data_dir,
            split_csv=split_csv,
            required_label0=2,
            required_label1=1,
        )

    assert summary["per_label"]["0"]["available_candidates"] == 1
    assert summary["per_label"]["1"]["available_candidates"] == 0
    assert summary["replacement_shortfall"] == {"0": 1, "1": 1}
    assert summary["enough_for_required_replacements"] is False


def test_candidate_pool_can_limit_candidates_per_label():
    with _case_dir("replacement_pool_limit") as tmp_path:
        data_dir = tmp_path / "data"
        _write_pe(data_dir / "待加入白名单" / ("b" * 64))
        _write_pe(data_dir / "待加入白名单" / ("c" * 64))
        split_csv = tmp_path / "split.csv"
        _write_csv(split_csv, ["source_path", "label", "sample_index", "split"], [])

        rows, summary = build_candidate_pool(
            data_dir=data_dir,
            split_csv=split_csv,
            max_candidates_per_label=1,
        )

    assert len(rows) == 1
    assert summary["label_counts"] == {"0": 1}
