import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_high_value_benign_manifest import build_manifest, write_manifest_csv  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, rows: list[dict]) -> Path:
    fieldnames = ["source_path", "cache_path", "label", "split", "sample_index"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_build_high_value_benign_manifest_filters_and_deduplicates_rows():
    with _case_dir("high_value_benign_manifest") as tmp_path:
        shared = {
            "source_path": r"E:\Project\python\Axon_v2.6Exp\data\待加入白名单\a.exe",
            "cache_path": str(tmp_path / "missing.npz"),
            "label": "0",
            "split": "test",
            "sample_index": "1",
        }
        official = _write_csv(
            tmp_path / "official.csv",
            [
                shared,
                {
                    "source_path": r"E:\Project\python\Axon_v2.6Exp\data\待拉黑\b.exe",
                    "cache_path": "",
                    "label": "1",
                    "split": "test",
                    "sample_index": "2",
                },
            ],
        )
        hard_missing = _write_csv(tmp_path / "hard_missing.csv", [shared])
        hard_predictions = _write_csv(
            tmp_path / "hard_predictions.csv",
            [
                {
                    "source_path": r"E:\Project\python\Axon_v2.6Exp\data\待加入白名单\c.exe",
                    "cache_path": str(tmp_path / "present.npz"),
                    "label": "0",
                    "split": "test",
                    "sample_index": "3",
                }
            ],
        )
        (tmp_path / "present.npz").write_bytes(b"placeholder")

        rows, summary = build_manifest(
            official_test_missing=official,
            hard_error_missing=hard_missing,
            hard_error_predictions=hard_predictions,
        )
        output_csv = tmp_path / "manifest.csv"
        write_manifest_csv(output_csv, rows)
        output_exists = output_csv.exists()

    assert len(rows) == 2
    by_path = {row["source_path"]: row for row in rows}
    assert set(by_path[shared["source_path"]]["manifest_sources"].split(";")) == {
        "official_test_missing_cache",
        "hard_error_missing_cache",
    }
    assert summary["summary"]["cache_path_present"] == 1
    assert summary["summary"]["cache_path_missing"] == 1
    assert output_exists
