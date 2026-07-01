import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_manual_review_sources import build_source_summary  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source_path",
        "source_sha256",
        "label",
        "priority",
        "error_type",
        "prob_malicious",
        "nearest_similarity",
        "opposite_label_ratio",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_source_summary_groups_flat_benign_and_malicious_date_batches():
    with _case_dir("source_summary") as tmp_path:
        review_csv = tmp_path / "review.csv"
        output_csv = tmp_path / "groups.csv"
        output_json = tmp_path / "summary.json"
        _write_rows(
            review_csv,
            [
                {
                    "source_path": r"E:\repo\data\待加入白名单\a.exe",
                    "source_sha256": "a" * 64,
                    "label": "0",
                    "priority": "0",
                    "error_type": "FP",
                    "prob_malicious": "0.99",
                    "nearest_similarity": "0.96",
                    "opposite_label_ratio": "1.0",
                },
                {
                    "source_path": r"E:\repo\data\待加入白名单\b",
                    "source_sha256": "b" * 64,
                    "label": "0",
                    "priority": "1",
                    "error_type": "FP",
                    "prob_malicious": "0.90",
                    "nearest_similarity": "0.91",
                    "opposite_label_ratio": "0.8",
                },
                {
                    "source_path": r"E:\repo\data\待拉黑\2026-03\2026-03-01\c.dll",
                    "source_sha256": "c" * 64,
                    "label": "1",
                    "priority": "2",
                    "error_type": "FN",
                    "prob_malicious": "0.10",
                    "nearest_similarity": "0.70",
                    "opposite_label_ratio": "0.9",
                },
            ],
        )

        summary = build_source_summary(
            review_csv=review_csv,
            output_csv=output_csv,
            output_json=output_json,
            prefix_depth=3,
        )

        persisted = json.loads(output_json.read_text(encoding="utf-8"))
        group_rows = list(csv.DictReader(output_csv.open(encoding="utf-8-sig")))

    assert summary["rows"] == 3
    assert persisted["data_dir_counts"] == {"待加入白名单": 2, "待拉黑": 1}
    assert persisted["source_prefix_counts"]["待加入白名单/<flat>"] == 2
    assert persisted["source_prefix_counts"]["待拉黑/2026-03/2026-03-01"] == 1
    assert persisted["high_similarity_conflicts_ge_0.90"] == 2
    assert persisted["critical_conflicts_ge_0.95"] == 1
    flat_group = [
        row for row in group_rows
        if row["dimension"] == "source_prefix" and row["value"] == "待加入白名单/<flat>"
    ][0]
    assert flat_group["count"] == "2"
    assert flat_group["fp_count"] == "2"
    assert flat_group["critical_conflicts_ge_0.95"] == "1"
