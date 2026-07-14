from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_corrected_split_cache_ready import audit_corrected_split_cache_ready  # noqa: E402
from audit_corrected_split_replacements import audit_corrected_split_replacements  # noqa: E402
from build_corrected_split_from_plan import build_corrected_split, write_split_csv  # noqa: E402


SPLIT_FIELDS = ["source_path", "source_sha256", "label", "sample_index", "split"]
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
CANDIDATE_FIELDS = ["source_path", "label", "source_sha256"]


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _sha(char: str) -> str:
    return char * 64


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_cache_npz(path: Path, *, label: int, source_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        byte_sequence=np.zeros(8, dtype=np.uint8),
        pe_features=np.zeros(4, dtype=np.float32),
        stat_features=np.zeros(3, dtype=np.float32),
        label=np.asarray(label, dtype=np.int64),
        source_sha256=np.asarray(source_sha256),
    )


def _patch_strict_shape(monkeypatch) -> None:
    expected_total = 2
    expected_split_counts = {"train": 1, "val": 1}
    expected_label_split_counts = {"train": {"0": 1}, "val": {"1": 1}, "test": {}}

    import audit_corrected_split_cache_ready as cache_ready_module
    import audit_corrected_split_replacements as replacement_audit_module
    import build_corrected_split_from_plan as corrected_split_module

    for module in [cache_ready_module, replacement_audit_module, corrected_split_module]:
        monkeypatch.setattr(module, "EXPECTED_TOTAL", expected_total)
        monkeypatch.setattr(module, "EXPECTED_SPLIT_COUNTS", expected_split_counts)
        monkeypatch.setattr(module, "EXPECTED_LABEL_SPLIT_COUNTS", expected_label_split_counts)


def test_redraw_hash_flows_from_candidate_to_corrected_split_replacement_audit_and_cache_ready(monkeypatch):
    _patch_strict_shape(monkeypatch)
    with _case_dir("redraw_hash_e2e") as tmp_path:
        bad_sha = _sha("b")
        good_sha = _sha("a")
        fresh_sha = _sha("c")
        original_split_csv = _write_csv(
            tmp_path / "original_split.csv",
            SPLIT_FIELDS,
            [
                {"source_path": "data/good.exe", "source_sha256": good_sha, "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/bad.exe", "source_sha256": bad_sha, "label": "1", "sample_index": "1", "split": "val"},
            ],
        )
        plan_csv = _write_csv(
            tmp_path / "plan.csv",
            PLAN_FIELDS,
            [
                {
                    "source_path": "data/bad.exe",
                    "source_sha256": bad_sha,
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
        candidate_csv = _write_csv(
            tmp_path / "candidates.csv",
            CANDIDATE_FIELDS,
            [{"source_path": "data/fresh.exe", "label": "1", "source_sha256": fresh_sha}],
        )

        corrected_rows, corrected_summary = build_corrected_split(
            split_csv=original_split_csv,
            plan_csv=plan_csv,
            candidate_csv=candidate_csv,
            strict_20w=True,
        )
        corrected_split_csv = tmp_path / "corrected_split.csv"
        write_split_csv(corrected_split_csv, corrected_rows)
        corrected_by_path = {row["source_path"]: row for row in _read_csv(corrected_split_csv)}

        replacement_detail_csv = tmp_path / "replacement_detail.csv"
        replacement_audit = audit_corrected_split_replacements(
            original_split_csv=original_split_csv,
            corrected_split_csv=corrected_split_csv,
            plan_csv=plan_csv,
            detail_output_csv=replacement_detail_csv,
            enforce_shape=True,
            enforce_label_balance=True,
        )
        replacement_detail_rows = _read_csv(replacement_detail_csv)

        cache_good = tmp_path / "cache" / "good.npz"
        cache_fresh = tmp_path / "cache" / "fresh.npz"
        _write_cache_npz(cache_good, label=0, source_sha256=good_sha)
        _write_cache_npz(cache_fresh, label=1, source_sha256=fresh_sha)
        manifest_json = tmp_path / "manifest.json"
        manifest_json.write_text(
            json.dumps(
                {
                    "max_byte_length": 8,
                    "pe_feature_dim": 4,
                    "stat_feature_dim": 3,
                    "samples": [
                        {"source_path": "data/good.exe", "source_sha256": good_sha, "label": 0, "cache_path": str(cache_good)},
                        {"source_path": "data/fresh.exe", "source_sha256": fresh_sha, "label": 1, "cache_path": str(cache_fresh)},
                    ],
                }
            ),
            encoding="utf-8",
        )
        cache_ready = audit_corrected_split_cache_ready(
            split_csv=corrected_split_csv,
            manifest_json=manifest_json,
            metadata_issue_output=tmp_path / "metadata_issues.csv",
            enforce_shape=True,
            enforce_label_balance=True,
        )

    assert corrected_summary["corrected_summary"]["rows"] == 2
    assert "data/bad.exe" not in corrected_by_path
    assert corrected_by_path["data/fresh.exe"]["source_sha256"] == fresh_sha
    assert replacement_audit["replacement_integrity_ok"] is True
    assert replacement_audit["fresh_replacement_rows"] == 1
    assert replacement_audit["fresh_replacement_counts_by_split_label"] == {"val:1": 1}
    assert any(
        row["record_type"] == "fresh_replacement" and row["source_sha256"] == fresh_sha
        for row in replacement_detail_rows
    )
    assert cache_ready["cache_ready"] is True
    assert cache_ready["manifest_match_counts"] == {"source_sha256": 2}
    assert cache_ready["metadata_failure_rows"] == 0


def test_redraw_cache_ready_rejects_corrected_split_that_loses_candidate_hash(monkeypatch):
    _patch_strict_shape(monkeypatch)
    with _case_dir("redraw_hash_missing_e2e") as tmp_path:
        good_sha = _sha("a")
        fresh_sha = _sha("c")
        cache_good = tmp_path / "cache" / "good.npz"
        cache_fresh = tmp_path / "cache" / "fresh.npz"
        _write_cache_npz(cache_good, label=0, source_sha256=good_sha)
        _write_cache_npz(cache_fresh, label=1, source_sha256=fresh_sha)
        manifest_json = tmp_path / "manifest.json"
        manifest_json.write_text(
            json.dumps(
                {
                    "max_byte_length": 8,
                    "pe_feature_dim": 4,
                    "stat_feature_dim": 3,
                    "samples": [
                        {"source_path": "data/good.exe", "source_sha256": good_sha, "label": 0, "cache_path": str(cache_good)},
                        {"source_path": "data/fresh.exe", "source_sha256": fresh_sha, "label": 1, "cache_path": str(cache_fresh)},
                    ],
                }
            ),
            encoding="utf-8",
        )
        corrected_split_csv = _write_csv(
            tmp_path / "corrected_split_missing_hash.csv",
            SPLIT_FIELDS,
            [
                {"source_path": "data/good.exe", "source_sha256": good_sha, "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/fresh.exe", "source_sha256": "", "label": "1", "sample_index": "1", "split": "val"},
            ],
        )

        cache_ready = audit_corrected_split_cache_ready(
            split_csv=corrected_split_csv,
            manifest_json=manifest_json,
            enforce_shape=True,
            enforce_label_balance=True,
        )

    assert cache_ready["cache_ready"] is False
    assert cache_ready["manifest_match_counts"] == {"source_path": 1, "source_sha256": 1}
    assert cache_ready["metadata_issue_counts"]["split_missing_source_sha256"] == 1


def test_redraw_cache_ready_rejects_corrected_split_hash_drift_even_when_path_matches(monkeypatch):
    _patch_strict_shape(monkeypatch)
    with _case_dir("redraw_hash_drift_e2e") as tmp_path:
        good_sha = _sha("a")
        fresh_sha = _sha("c")
        drift_sha = _sha("d")
        cache_good = tmp_path / "cache" / "good.npz"
        cache_fresh = tmp_path / "cache" / "fresh.npz"
        _write_cache_npz(cache_good, label=0, source_sha256=good_sha)
        _write_cache_npz(cache_fresh, label=1, source_sha256=fresh_sha)
        manifest_json = tmp_path / "manifest.json"
        manifest_json.write_text(
            json.dumps(
                {
                    "max_byte_length": 8,
                    "pe_feature_dim": 4,
                    "stat_feature_dim": 3,
                    "samples": [
                        {"source_path": "data/good.exe", "source_sha256": good_sha, "label": 0, "cache_path": str(cache_good)},
                        {"source_path": "data/fresh.exe", "source_sha256": fresh_sha, "label": 1, "cache_path": str(cache_fresh)},
                    ],
                }
            ),
            encoding="utf-8",
        )
        corrected_split_csv = _write_csv(
            tmp_path / "corrected_split_hash_drift.csv",
            SPLIT_FIELDS,
            [
                {"source_path": "data/good.exe", "source_sha256": good_sha, "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/fresh.exe", "source_sha256": drift_sha, "label": "1", "sample_index": "1", "split": "val"},
            ],
        )

        cache_ready = audit_corrected_split_cache_ready(
            split_csv=corrected_split_csv,
            manifest_json=manifest_json,
            enforce_shape=True,
            enforce_label_balance=True,
        )

    assert cache_ready["cache_ready"] is False
    assert cache_ready["manifest_match_counts"] == {"source_path": 1, "source_sha256": 1}
    assert cache_ready["metadata_issue_counts"]["source_sha256_mismatch_split_manifest"] == 1
