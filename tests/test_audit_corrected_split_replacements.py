from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_corrected_split_replacements import audit_corrected_split_replacements  # noqa: E402


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


SPLIT_FIELDS = ["source_path", "label", "sample_index", "split"]
PLAN_FIELDS = [
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "original_label",
    "planned_label",
    "plan_action",
    "replacement_required",
    "replacement_label",
    "usable_for_training_policy",
]


def test_empty_plan_reports_integrity_ok_with_shape_disabled():
    with _case_dir("replacement_audit_empty") as tmp_path:
        original_csv = tmp_path / "original.csv"
        corrected_csv = tmp_path / "corrected.csv"
        plan_csv = tmp_path / "plan.csv"
        rows = [
            {"source_path": "data/a.exe", "label": "0", "sample_index": "0", "split": "train"},
            {"source_path": "data/b.exe", "label": "1", "sample_index": "1", "split": "val"},
        ]
        _write_csv(original_csv, SPLIT_FIELDS, rows)
        _write_csv(corrected_csv, SPLIT_FIELDS, rows)
        _write_csv(plan_csv, PLAN_FIELDS, [])

        payload = audit_corrected_split_replacements(
            original_split_csv=original_csv,
            corrected_split_csv=corrected_csv,
            plan_csv=plan_csv,
            enforce_shape=False,
        )

    assert payload["replacement_integrity_ok"] is True
    assert payload["replacement_requests"] == 0
    assert payload["fresh_replacement_rows"] == 0
    assert payload["unplanned_original_rows_removed"] == 0


def test_replacement_audit_accepts_fresh_same_split_label_replacement():
    with _case_dir("replacement_audit_fresh") as tmp_path:
        original_csv = tmp_path / "original.csv"
        corrected_csv = tmp_path / "corrected.csv"
        plan_csv = tmp_path / "plan.csv"
        _write_csv(
            original_csv,
            SPLIT_FIELDS,
            [
                {"source_path": "data/good.exe", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/bad.exe", "label": "1", "sample_index": "1", "split": "val"},
            ],
        )
        _write_csv(
            corrected_csv,
            SPLIT_FIELDS,
            [
                {"source_path": "data/good.exe", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/fresh.exe", "label": "1", "sample_index": "1", "split": "val"},
            ],
        )
        _write_csv(
            plan_csv,
            PLAN_FIELDS,
            [
                {
                    "source_path": "data/bad.exe",
                    "source_sha256": "",
                    "sample_index": "1",
                    "split": "val",
                    "original_label": "1",
                    "planned_label": "1",
                    "plan_action": "exclude_and_replace",
                    "replacement_required": "true",
                    "replacement_label": "1",
                    "usable_for_training_policy": "false",
                }
            ],
        )

        payload = audit_corrected_split_replacements(
            original_split_csv=original_csv,
            corrected_split_csv=corrected_csv,
            plan_csv=plan_csv,
            enforce_shape=False,
        )

    assert payload["replacement_integrity_ok"] is True
    assert payload["replacement_requests"] == 1
    assert payload["excluded_rows_present_after_correction"] == 0
    assert payload["planned_excluded_rows_removed"] == 1
    assert payload["fresh_replacement_rows"] == 1
    assert payload["replacement_request_counts_by_split_label"] == {"val:1": 1}
    assert payload["fresh_replacement_counts_by_split_label"] == {"val:1": 1}


def test_replacement_audit_rejects_self_replacement():
    with _case_dir("replacement_audit_self") as tmp_path:
        original_csv = tmp_path / "original.csv"
        corrected_csv = tmp_path / "corrected.csv"
        plan_csv = tmp_path / "plan.csv"
        row = {"source_path": "data/bad.exe", "label": "1", "sample_index": "0", "split": "val"}
        _write_csv(original_csv, SPLIT_FIELDS, [row])
        _write_csv(corrected_csv, SPLIT_FIELDS, [row])
        _write_csv(
            plan_csv,
            PLAN_FIELDS,
            [
                {
                    "source_path": "data/bad.exe",
                    "source_sha256": "",
                    "sample_index": "0",
                    "split": "val",
                    "original_label": "1",
                    "planned_label": "1",
                    "plan_action": "exclude_and_replace",
                    "replacement_required": "true",
                    "replacement_label": "1",
                    "usable_for_training_policy": "false",
                }
            ],
        )

        payload = audit_corrected_split_replacements(
            original_split_csv=original_csv,
            corrected_split_csv=corrected_csv,
            plan_csv=plan_csv,
            enforce_shape=False,
        )

    assert payload["replacement_integrity_ok"] is False
    assert payload["excluded_rows_present_after_correction"] == 1
    assert payload["fresh_replacement_rows"] == 0
    assert "fresh replacement counts do not match replacement requests" in payload["integrity_failures"]


def test_replacement_audit_rejects_unplanned_removed_original_row():
    with _case_dir("replacement_audit_unplanned_removed") as tmp_path:
        original_csv = tmp_path / "original.csv"
        corrected_csv = tmp_path / "corrected.csv"
        plan_csv = tmp_path / "plan.csv"
        _write_csv(
            original_csv,
            SPLIT_FIELDS,
            [
                {"source_path": "data/a.exe", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/b.exe", "label": "1", "sample_index": "1", "split": "val"},
            ],
        )
        _write_csv(
            corrected_csv,
            SPLIT_FIELDS,
            [
                {"source_path": "data/a.exe", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/c.exe", "label": "1", "sample_index": "1", "split": "val"},
            ],
        )
        _write_csv(plan_csv, PLAN_FIELDS, [])

        payload = audit_corrected_split_replacements(
            original_split_csv=original_csv,
            corrected_split_csv=corrected_csv,
            plan_csv=plan_csv,
            enforce_shape=False,
        )

    assert payload["replacement_integrity_ok"] is False
    assert payload["unplanned_original_rows_removed"] == 1
    assert payload["fresh_replacement_rows"] == 1


def test_replacement_audit_checks_relabel_applied():
    with _case_dir("replacement_audit_relabel") as tmp_path:
        original_csv = tmp_path / "original.csv"
        corrected_csv = tmp_path / "corrected.csv"
        plan_csv = tmp_path / "plan.csv"
        _write_csv(original_csv, SPLIT_FIELDS, [{"source_path": "data/a.exe", "label": "0", "sample_index": "0", "split": "train"}])
        _write_csv(corrected_csv, SPLIT_FIELDS, [{"source_path": "data/a.exe", "label": "1", "sample_index": "0", "split": "train"}])
        _write_csv(
            plan_csv,
            PLAN_FIELDS,
            [
                {
                    "source_path": "data/a.exe",
                    "source_sha256": "",
                    "sample_index": "0",
                    "split": "train",
                    "original_label": "0",
                    "planned_label": "1",
                    "plan_action": "relabel",
                    "replacement_required": "false",
                    "replacement_label": "",
                    "usable_for_training_policy": "true",
                }
            ],
        )

        payload = audit_corrected_split_replacements(
            original_split_csv=original_csv,
            corrected_split_csv=corrected_csv,
            plan_csv=plan_csv,
            enforce_shape=False,
        )

    assert payload["replacement_integrity_ok"] is True
    assert payload["relabel_requests"] == 1
    assert payload["relabel_label_mismatch"] == 0


def test_replacement_audit_reports_existing_duplicates_without_blocking_when_not_increased():
    with _case_dir("replacement_audit_existing_duplicates") as tmp_path:
        original_csv = tmp_path / "original.csv"
        corrected_csv = tmp_path / "corrected.csv"
        plan_csv = tmp_path / "plan.csv"
        rows = [
            {"source_path": "data/dup.exe", "label": "0", "sample_index": "0", "split": "train"},
            {"source_path": "data/dup.exe", "label": "0", "sample_index": "1", "split": "train"},
        ]
        _write_csv(original_csv, SPLIT_FIELDS, rows)
        _write_csv(corrected_csv, SPLIT_FIELDS, rows)
        _write_csv(plan_csv, PLAN_FIELDS, [])

        payload = audit_corrected_split_replacements(
            original_split_csv=original_csv,
            corrected_split_csv=corrected_csv,
            plan_csv=plan_csv,
            enforce_shape=False,
        )

    assert payload["replacement_integrity_ok"] is True
    assert payload["original_duplicate_key_rows"] == 1
    assert payload["corrected_duplicate_key_rows"] == 1
    assert payload["duplicate_key_row_delta"] == 0


def test_replacement_audit_rejects_new_duplicates_added_by_corrected_split():
    with _case_dir("replacement_audit_new_duplicates") as tmp_path:
        original_csv = tmp_path / "original.csv"
        corrected_csv = tmp_path / "corrected.csv"
        plan_csv = tmp_path / "plan.csv"
        _write_csv(original_csv, SPLIT_FIELDS, [{"source_path": "data/a.exe", "label": "0", "sample_index": "0", "split": "train"}])
        _write_csv(
            corrected_csv,
            SPLIT_FIELDS,
            [
                {"source_path": "data/a.exe", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/a.exe", "label": "0", "sample_index": "1", "split": "train"},
            ],
        )
        _write_csv(plan_csv, PLAN_FIELDS, [])

        payload = audit_corrected_split_replacements(
            original_split_csv=original_csv,
            corrected_split_csv=corrected_csv,
            plan_csv=plan_csv,
            enforce_shape=False,
        )

    assert payload["replacement_integrity_ok"] is False
    assert payload["duplicate_key_row_delta"] == 1
    assert "corrected split introduced duplicate source keys: +1" in payload["integrity_failures"]
