import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_strict_split_from_cache import (  # noqa: E402
    collect_strict_records,
    compute_metrics,
    select_manifest_sample,
)


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_split(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "label", "sample_index", "split"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, samples: list[dict]) -> None:
    path.write_text(json.dumps({"samples": samples}), encoding="utf-8")


def test_collect_strict_records_matches_by_source_sha256_only():
    with _case_dir("strict_eval_sha_only") as tmp_path:
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "sample.npz"
        cache_path.write_bytes(b"placeholder")
        split_csv = tmp_path / "split.csv"
        manifest_json = cache_dir / "manifest.json"
        _write_split(
            split_csv,
            [
                {
                    "source_path": "renamed-anything.exe",
                    "source_sha256": "a" * 64,
                    "label": "1",
                    "sample_index": "1",
                    "split": "val",
                }
            ],
        )
        _write_manifest(
            manifest_json,
            [
                {
                    "source_path": "different-name.exe",
                    "cache_path": str(cache_path),
                    "label": 1,
                    "source_sha256": "a" * 64,
                }
            ],
        )

        records, summary = collect_strict_records(
            split_csv=split_csv,
            manifest_json=manifest_json,
            split="val",
        )

    assert len(records) == 1
    assert records[0]["source_sha256"] == "a" * 64
    assert summary["manifest_match_counts"] == {"source_sha256": 1}
    assert summary["issue_rows"] == 0


def test_collect_strict_records_rejects_path_match_when_hash_differs():
    with _case_dir("strict_eval_no_path_fallback") as tmp_path:
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "sample.npz"
        cache_path.write_bytes(b"placeholder")
        source_path = str(tmp_path / "same-name.exe")
        split_csv = tmp_path / "split.csv"
        manifest_json = cache_dir / "manifest.json"
        _write_split(
            split_csv,
            [
                {
                    "source_path": source_path,
                    "source_sha256": "b" * 64,
                    "label": "0",
                    "sample_index": "2",
                    "split": "val",
                }
            ],
        )
        _write_manifest(
            manifest_json,
            [
                {
                    "source_path": source_path,
                    "cache_path": str(cache_path),
                    "label": 0,
                    "source_sha256": "a" * 64,
                }
            ],
        )

        records, summary = collect_strict_records(
            split_csv=split_csv,
            manifest_json=manifest_json,
            split="val",
        )

    assert records == []
    assert summary["manifest_match_counts"] == {}
    assert summary["issue_counts"]["manifest_missing_source_sha256"] == 1


def test_select_manifest_sample_rejects_conflicting_manifest_labels():
    sample_sha = "c" * 64

    sample, issues = select_manifest_sample(
        {"source_sha256": sample_sha, "label": "1"},
        {
            sample_sha: [
                {"source_sha256": sample_sha, "label": "1"},
                {"source_sha256": sample_sha, "label": "0"},
            ]
        },
    )

    assert sample is None
    assert issues == ["manifest_conflicting_labels_for_source_sha256"]


def test_compute_metrics_reports_confusion_counts():
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], threshold=0.5)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["errors"] == 2
