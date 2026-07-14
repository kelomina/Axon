import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_strict_test10k_split import build_test10k  # noqa: E402


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
            fieldnames=["source_path", "source_sha256", "label", "split", "sample_index"],
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_test10k_balances_labels_and_ignores_path_names():
    with _case_dir("strict_test10k") as tmp_path:
        split_csv = tmp_path / "split.csv"
        output_csv = tmp_path / "test10k.csv"
        rows = [
            {"source_path": "looks-malicious.exe", "source_sha256": "a" * 64, "label": "0", "split": "test", "sample_index": "1"},
            {"source_path": "looks-benign.exe", "source_sha256": "b" * 64, "label": "1", "split": "test", "sample_index": "2"},
            {"source_path": "ignored-train.exe", "source_sha256": "c" * 64, "label": "1", "split": "train", "sample_index": "3"},
        ]
        _write_split(split_csv, rows)

        payload = build_test10k(split_csv=split_csv, output_csv=output_csv, per_label=1)
        selected = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig", newline="")))

    assert payload["selected_rows"] == 2
    assert payload["label_counts"] == {"0": 1, "1": 1}
    assert [row["split"] for row in selected] == ["test10k", "test10k"]


def test_build_test10k_rejects_invalid_hash():
    with _case_dir("strict_test10k_invalid_hash") as tmp_path:
        split_csv = tmp_path / "split.csv"
        output_csv = tmp_path / "test10k.csv"
        _write_split(
            split_csv,
            [
                {"source_path": "a.exe", "source_sha256": "not-a-sha", "label": "0", "split": "test", "sample_index": "1"},
                {"source_path": "b.exe", "source_sha256": "b" * 64, "label": "1", "split": "test", "sample_index": "2"},
            ],
        )

        try:
            build_test10k(split_csv=split_csv, output_csv=output_csv, per_label=1)
        except ValueError as exc:
            message = str(exc)
        else:
            message = ""

    assert "Could not build balanced Test-10k" in message or "Strict Test-10k source rows had issues" in message
