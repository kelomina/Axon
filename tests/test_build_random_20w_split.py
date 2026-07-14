import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_random_20w_split import (  # noqa: E402
    build_summary,
    pick_balanced_samples,
    scan_valid_raw_samples,
    write_split_csv,
)


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_file(path: Path, header: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"payload")
    return path


def test_scan_valid_raw_samples_filters_non_pe_and_limits_reservoir():
    with _case_dir("random_20w_scan") as tmp_path:
        root = tmp_path / "待加入白名单"
        valid1 = _write_file(root / "a.exe", b"MZ")
        valid2 = _write_file(root / "b.exe", b"MZ")
        valid3 = _write_file(root / "c.exe", b"MZ")
        _write_file(root / "d.txt", b"ZZ")

        result = scan_valid_raw_samples(root, label=0, max_file_size=1024, seed=7, sample_limit=2)

    assert result.valid_pe_count == 3
    assert result.files_seen == 4
    assert len(result.selected_rows) == 2
    assert {row["label"] for row in result.selected_rows} == {0}
    assert {row["source_path"] for row in result.selected_rows}.issubset(
        {str(valid1), str(valid2), str(valid3)}
    )


def test_pick_balanced_samples_and_write_split_csv_are_reproducible():
    benign_rows = [{"source_path": f"benign_{idx}.exe", "label": 0} for idx in range(10)]
    malicious_rows = [{"source_path": f"mal_{idx}.exe", "label": 1} for idx in range(10)]

    rows1, summary1 = pick_balanced_samples(benign_rows, malicious_rows, total_samples=20, seed=42)
    rows2, summary2 = pick_balanced_samples(benign_rows, malicious_rows, total_samples=20, seed=42)

    assert rows1 == rows2
    assert summary1 == summary2
    assert summary1["split_counts"] == {"train": 2, "val": 2, "test": 16}
    assert summary1["label_counts"] == {"0": 10, "1": 10}
    assert summary1["label_split_counts"]["train"] == {"0": 1, "1": 1}

    with _case_dir("random_20w_write") as tmp_path:
        split_csv = tmp_path / "random_20w_split.csv"
        write_split_csv(split_csv, rows1)
        with split_csv.open("r", encoding="utf-8-sig", newline="") as f:
            parsed = list(csv.DictReader(f))

    assert len(parsed) == 20
    assert set(parsed[0].keys()) == {"source_path", "label", "sample_index", "split"}


def test_build_summary_marks_shortfall_without_rows():
    class ScanResult:
        def __init__(self, files_seen: int, valid_pe_count: int):
            self.files_seen = files_seen
            self.valid_pe_count = valid_pe_count

    summary = build_summary(
        seed=42,
        total_samples=200000,
        benign_scan=ScanResult(10, 5),
        malicious_scan=ScanResult(10, 5),
        rows=[],
        output_csv=Path("reports/random_20w_split/random_20w_split.csv"),
        shortfall={"benign_shortfall": 99995, "malicious_shortfall": 99995},
    )

    assert summary["shortfall"]["benign_shortfall"] == 99995
    assert summary["label_counts"] == {}
    assert summary["split_counts"] == {}
