from __future__ import annotations

import csv
import hashlib
import json
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


def _write_pe(path: Path, payload: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ" + payload + b"\0" * 32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_pool_excludes_current_split_rows_and_counts_labels():
    used_sha = "a" * 64
    unused_benign_sha = "b" * 64
    unused_mal_sha = "c" * 64
    with _case_dir("replacement_pool_basic") as tmp_path:
        data_dir = tmp_path / "data"
        _write_pe(data_dir / "待加入白名单" / used_sha, b"used")
        _write_pe(data_dir / "待加入白名单" / unused_benign_sha, b"unused-benign")
        _write_pe(data_dir / "待拉黑" / f"{unused_mal_sha}.exe", b"unused-mal")
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
    assert summary["source_sha256_origin_counts"] == {"content_hash": 2}


def test_candidate_pool_reports_required_replacement_shortfall():
    with _case_dir("replacement_pool_shortfall") as tmp_path:
        data_dir = tmp_path / "data"
        _write_pe(data_dir / "待加入白名单" / ("b" * 64), b"only-benign")
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
        _write_pe(data_dir / "待加入白名单" / ("b" * 64), b"first")
        _write_pe(data_dir / "待加入白名单" / ("c" * 64), b"second")
        split_csv = tmp_path / "split.csv"
        _write_csv(split_csv, ["source_path", "label", "sample_index", "split"], [])

        rows, summary = build_candidate_pool(
            data_dir=data_dir,
            split_csv=split_csv,
            max_candidates_per_label=1,
        )

    assert len(rows) == 1
    assert summary["label_counts"] == {"0": 1}


def test_candidate_pool_defaults_to_bounded_scan_when_replacements_are_required():
    with _case_dir("replacement_pool_default_required_limit") as tmp_path:
        data_dir = tmp_path / "data"
        for index in range(70):
            _write_pe(data_dir / "待加入白名单" / f"candidate-{index}.bin", f"payload-{index}".encode("ascii"))
        split_csv = tmp_path / "split.csv"
        _write_csv(split_csv, ["source_path", "label", "sample_index", "split"], [])

        rows, summary = build_candidate_pool(
            data_dir=data_dir,
            split_csv=split_csv,
            required_label0=5,
        )

    assert len(rows) == 55
    assert summary["effective_candidate_limits"]["0"] == 55
    assert summary["per_label"]["0"]["scan_limit"] == 55
    assert summary["per_label"]["0"]["stopped_after_reaching_limit"] is True


def test_candidate_pool_hashes_content_and_excludes_renamed_manifest_duplicate():
    with _case_dir("replacement_pool_manifest_content_hash") as tmp_path:
        data_dir = tmp_path / "data"
        used_original = data_dir / "待加入白名单" / "original-name.exe"
        renamed_duplicate = data_dir / "待加入白名单" / "renamed-copy.bin"
        fresh_candidate = data_dir / "待加入白名单" / "fresh.bin"
        _write_pe(used_original, b"same-content")
        _write_pe(renamed_duplicate, b"same-content")
        _write_pe(fresh_candidate, b"fresh-content")
        used_sha = _sha256(used_original)
        fresh_sha = _sha256(fresh_candidate)

        manifest_json = tmp_path / "manifest.json"
        manifest_json.write_text(
            json.dumps({"samples": [{"source_path": str(used_original), "source_sha256": used_sha, "cache_path": "used.npz"}]}),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": str(used_original), "label": "0", "sample_index": "0", "split": "train"}],
        )

        rows, summary = build_candidate_pool(data_dir=data_dir, split_csv=split_csv, manifest_json=manifest_json)

    names = {Path(row["source_path"]).name for row in rows}
    assert "renamed-copy.bin" not in names
    assert "fresh.bin" in names
    assert rows[0]["source_sha256"] == fresh_sha
    assert rows[0]["source_sha256_origin"] == "content_hash"
    assert summary["manifest_sha_count"] == 1
    assert summary["per_label"]["0"]["already_used_in_split"] == 2


def test_candidate_pool_no_hash_files_is_explicit_compatibility_mode():
    with _case_dir("replacement_pool_no_hash_compat") as tmp_path:
        data_dir = tmp_path / "data"
        candidate = data_dir / "待加入白名单" / "not-a-sha-name.bin"
        _write_pe(candidate, b"payload")
        split_csv = tmp_path / "split.csv"
        _write_csv(split_csv, ["source_path", "label", "sample_index", "split"], [])

        rows, summary = build_candidate_pool(data_dir=data_dir, split_csv=split_csv, hash_files=False)

    assert len(rows) == 1
    assert rows[0]["source_sha256"] == ""
    assert rows[0]["source_sha256_origin"] == "missing"
    assert summary["content_hash_required_for_strict_redraw"] is False
    assert summary["per_label"]["0"]["unhashed_candidates"] == 1
