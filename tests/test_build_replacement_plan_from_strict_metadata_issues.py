import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_replacement_plan_from_strict_metadata_issues import (  # noqa: E402
    build_replacement_plan_from_issues,
)


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_summary(path: Path, rows: list[dict], *, row_issue_count: int | None = None) -> Path:
    payload = {
        "schema": "axon_strict_split_metadata_enrichment_v1",
        "row_issue_count": len(rows) if row_issue_count is None else row_issue_count,
        "row_issue_examples": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_builds_exclude_and_replace_plan_with_original_label_only():
    with _case_dir("replacement_plan_from_metadata") as tmp_path:
        summary_json = _write_summary(
            tmp_path / "summary.json",
            [
                {
                    "sample_index": "7",
                    "split": "test",
                    "label": "0",
                    "source_path": "looks-malicious.exe",
                    "source_sha256": "a" * 64,
                    "issues": ["manifest_conflicting_labels_for_source_sha256"],
                }
            ],
        )

        rows, summary = build_replacement_plan_from_issues(summary_json=summary_json)

    assert summary["plan_ready"] is True
    assert summary["replacement_counts_by_split_label"] == {"test:0": 1}
    assert "does not relabel" in summary["identity_feature_policy"]
    assert rows == [
        {
            "source_path": "looks-malicious.exe",
            "source_sha256": "a" * 64,
            "sample_index": "7",
            "split": "test",
            "original_label": "0",
            "planned_label": "0",
            "plan_action": "exclude_and_replace",
            "replacement_required": "true",
            "replacement_label": "0",
            "usable_for_training_policy": "false",
            "metadata_issue_flags": "manifest_conflicting_labels_for_source_sha256",
        }
    ]


def test_blocks_unsupported_issue_instead_of_auto_replacing():
    with _case_dir("replacement_plan_from_metadata_block") as tmp_path:
        summary_json = _write_summary(
            tmp_path / "summary.json",
            [
                {
                    "sample_index": "8",
                    "split": "val",
                    "label": "1",
                    "source_path": "sample.exe",
                    "source_sha256": "b" * 64,
                    "issues": ["manifest_missing_source_sha256"],
                }
            ],
        )

        rows, summary = build_replacement_plan_from_issues(summary_json=summary_json)

    assert rows == []
    assert summary["plan_ready"] is False
    assert summary["blocked_rows"] == 1
    assert summary["blocked_examples"][0]["unsupported_issues"] == ["manifest_missing_source_sha256"]


def test_requires_full_issue_export_before_plan_build():
    with _case_dir("replacement_plan_from_metadata_truncated") as tmp_path:
        summary_json = _write_summary(
            tmp_path / "summary.json",
            [
                {
                    "sample_index": "8",
                    "split": "val",
                    "label": "1",
                    "source_path": "sample.exe",
                    "source_sha256": "b" * 64,
                    "issues": ["manifest_conflicting_labels_for_source_sha256"],
                }
            ],
            row_issue_count=2,
        )

        with pytest.raises(ValueError, match="does not contain every issue row"):
            build_replacement_plan_from_issues(summary_json=summary_json)
