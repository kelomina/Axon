from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_manual_review_adjudication_guide import build_guide  # noqa: E402


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


def test_adjudication_guide_is_read_only_and_marks_critical_conflict():
    with _case_dir("adjudication_guide") as tmp_path:
        readiness_csv = tmp_path / "readiness.csv"
        _write_csv(
            readiness_csv,
            [
                "review_rank",
                "priority",
                "error_type",
                "source_path",
                "source_sha256",
                "label",
                "prediction",
                "prob_malicious",
                "base_prob_malicious",
                "nearest_similarity",
                "opposite_label_ratio",
                "neighbor_label_counts",
                "top5_neighbor_labels",
                "top5_neighbor_similarities",
                "top5_neighbor_paths",
                "source_exists",
                "source_sha256_ok",
                "cache_exists",
                "npz_shape_ok",
                "is_pe",
                "file_size",
                "machine",
                "number_of_sections",
                "section_names",
                "max_section_entropy",
                "overlay_size",
                "manual_review_ready",
                "readiness_reasons",
                "manual_label_verdict",
                "recommended_action",
            ],
            [
                {
                    "review_rank": "1",
                    "priority": "0",
                    "error_type": "FP",
                    "source_path": "data/white/a.exe",
                    "source_sha256": "a" * 64,
                    "label": "0",
                    "prediction": "1",
                    "prob_malicious": "0.99",
                    "base_prob_malicious": "0.97",
                    "nearest_similarity": "0.96",
                    "opposite_label_ratio": "1.0",
                    "neighbor_label_counts": "1:25",
                    "top5_neighbor_labels": "1|1|1|1|1",
                    "top5_neighbor_similarities": "0.96|0.95|0.94|0.93|0.92",
                    "top5_neighbor_paths": "n1|n2|n3|n4|n5",
                    "source_exists": "True",
                    "source_sha256_ok": "True",
                    "cache_exists": "True",
                    "npz_shape_ok": "True",
                    "is_pe": "True",
                    "file_size": "1024",
                    "machine": "0x014c",
                    "number_of_sections": "5",
                    "section_names": ".text|.rsrc",
                    "max_section_entropy": "6.5",
                    "overlay_size": "0",
                    "manual_review_ready": "True",
                    "readiness_reasons": "",
                    "manual_label_verdict": "",
                    "recommended_action": "",
                }
            ],
        )

        summary = build_guide(
            readiness_csv=readiness_csv,
            output_csv=tmp_path / "guide.csv",
            output_json=tmp_path / "guide.json",
            output_md=tmp_path / "guide.md",
            markdown_rows=1,
        )
        rows = list(csv.DictReader((tmp_path / "guide.csv").open("r", encoding="utf-8-sig", newline="")))
        json_payload = json.loads((tmp_path / "guide.json").read_text(encoding="utf-8"))
        markdown = (tmp_path / "guide.md").read_text(encoding="utf-8")

    assert summary["rows"] == 1
    assert summary["manual_review_ready_rows"] == 1
    assert summary["suspicion_level_counts"] == {"critical_label_conflict": 1}
    assert json_payload["notes"][0].startswith("This guide is read-only")
    assert rows[0]["guide_suspicion_level"] == "critical_label_conflict"
    assert "manual_label_verdict" not in rows[0]
    assert "recommended_action" not in rows[0]
    assert rows[0]["allowed_manual_label_verdicts"] == "label_correct|label_wrong|out_of_scope|feature_broken|uncertain"
    assert "feature_broken" in markdown
