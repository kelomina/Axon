import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_strict_split_metadata import audit_strict_split_metadata  # noqa: E402


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
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_npz(path: Path, *, label: int, source_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        byte_sequence=np.asarray([77, 90], dtype=np.uint8),
        pe_features=np.asarray([1.0, 2.0], dtype=np.float32),
        label=np.asarray(label, dtype=np.int64),
        source_sha256=np.asarray(source_sha256),
    )


def _write_manifest(path: Path, samples: list[dict]) -> None:
    path.write_text(json.dumps({"samples": samples}), encoding="utf-8")


def test_strict_split_metadata_audit_passes_with_explicit_label_and_hash():
    with _case_dir("strict_split_metadata_ok") as tmp_path:
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        cache_path = tmp_path / "cache" / "a.npz"
        _write_npz(cache_path, label=1, source_sha256="a" * 64)
        _write_split(
            split_csv,
            [{"source_path": str(tmp_path / "renamed.bin"), "source_sha256": "a" * 64, "label": "1", "sample_index": "7", "split": "test"}],
        )
        _write_manifest(
            manifest_json,
            [{"source_path": "not-used-for-verdict.exe", "cache_path": str(cache_path), "label": 1, "source_sha256": "a" * 64}],
        )

        payload = audit_strict_split_metadata(split_csv=split_csv, manifest_json=manifest_json)

    assert payload["audit_ready"] is True
    assert payload["match_counts"] == {"source_sha256": 1}
    assert payload["ready_for"]["test10k"] is False
    assert "alignment and loading fields only" in payload["identity_feature_policy"]


def test_strict_split_metadata_audit_rejects_missing_split_hash():
    with _case_dir("strict_split_metadata_missing_hash") as tmp_path:
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        _write_split(
            split_csv,
            [{"source_path": "malicious_name.exe", "source_sha256": "", "label": "1", "sample_index": "1", "split": "train"}],
        )
        _write_manifest(manifest_json, [])

        payload = audit_strict_split_metadata(split_csv=split_csv, manifest_json=manifest_json)

    assert payload["audit_ready"] is False
    assert payload["metadata_issue_counts"]["split_invalid_source_sha256"] == 1
    assert payload["metadata_issue_counts"]["manifest_missing_source_sha256"] == 1


def test_strict_split_metadata_audit_rejects_manifest_label_mismatch():
    with _case_dir("strict_split_metadata_label_mismatch") as tmp_path:
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        cache_path = tmp_path / "cache" / "a.npz"
        _write_npz(cache_path, label=0, source_sha256="a" * 64)
        _write_split(
            split_csv,
            [{"source_path": "any-name.exe", "source_sha256": "a" * 64, "label": "1", "sample_index": "1", "split": "val"}],
        )
        _write_manifest(
            manifest_json,
            [{"source_path": "different-name.exe", "cache_path": str(cache_path), "label": 0, "source_sha256": "a" * 64}],
        )

        payload = audit_strict_split_metadata(split_csv=split_csv, manifest_json=manifest_json)

    assert payload["audit_ready"] is False
    assert payload["metadata_issue_counts"]["label_mismatch_split_manifest"] == 1


def test_strict_split_metadata_audit_rejects_manifest_conflicting_labels_for_same_hash():
    with _case_dir("strict_split_metadata_conflicting_manifest_labels") as tmp_path:
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        cache_path = tmp_path / "cache" / "a.npz"
        _write_npz(cache_path, label=1, source_sha256="a" * 64)
        _write_split(
            split_csv,
            [{"source_path": "renamed.exe", "source_sha256": "a" * 64, "label": "1", "sample_index": "1", "split": "val"}],
        )
        _write_manifest(
            manifest_json,
            [
                {"source_path": "one.exe", "cache_path": str(cache_path), "label": 1, "source_sha256": "a" * 64},
                {"source_path": "two.exe", "cache_path": str(cache_path), "label": 0, "source_sha256": "a" * 64},
            ],
        )

        payload = audit_strict_split_metadata(split_csv=split_csv, manifest_json=manifest_json)

    assert payload["audit_ready"] is False
    assert payload["metadata_issue_counts"]["manifest_conflicting_labels_for_source_sha256"] == 1


def test_strict_split_metadata_audit_rejects_npz_hash_drift():
    with _case_dir("strict_split_metadata_npz_hash_drift") as tmp_path:
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        cache_path = tmp_path / "cache" / "a.npz"
        _write_npz(cache_path, label=1, source_sha256="b" * 64)
        _write_split(
            split_csv,
            [{"source_path": "sample.exe", "source_sha256": "a" * 64, "label": "1", "sample_index": "1", "split": "test"}],
        )
        _write_manifest(
            manifest_json,
            [{"source_path": "sample.exe", "cache_path": str(cache_path), "label": 1, "source_sha256": "a" * 64}],
        )

        payload = audit_strict_split_metadata(split_csv=split_csv, manifest_json=manifest_json)

    assert payload["audit_ready"] is False
    assert payload["metadata_issue_counts"]["source_sha256_mismatch_split_npz"] == 1
