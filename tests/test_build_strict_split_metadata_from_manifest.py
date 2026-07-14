import csv
import hashlib
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_strict_split_metadata_from_manifest import enrich_strict_split_metadata  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_split(path: Path, rows: list[dict], *, include_hash: bool = False) -> None:
    fieldnames = ["source_path", "label", "sample_index", "split"]
    if include_hash:
        fieldnames.insert(1, "source_sha256")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, samples: list[dict]) -> None:
    path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")


def test_enriches_missing_hash_by_hashing_file_content_not_filename():
    with _case_dir("strict_split_enrich_hash_file") as tmp_path:
        source = tmp_path / "definitely_malicious_name.exe"
        payload = b"MZbenign-content"
        source.write_bytes(payload)
        source_sha = _sha_bytes(payload)
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        _write_split(
            split_csv,
            [{"source_path": str(source), "label": "0", "sample_index": "1", "split": "val"}],
        )
        _write_manifest(
            manifest_json,
            [{"source_path": "different-name.bin", "label": 0, "source_sha256": source_sha}],
        )

        rows, summary = enrich_strict_split_metadata(split_csv=split_csv, manifest_json=manifest_json)

    assert summary["enrichment_ready"] is True
    assert summary["hash_source_counts"] == {
        "computed_from_source_file": 1,
        "missing_split_source_sha256": 1,
    }
    assert summary["manifest_match_counts"] == {"source_sha256": 1}
    assert rows[0]["source_sha256"] == source_sha
    assert rows[0]["label"] == "0"
    assert "never infers labels from names" in summary["identity_feature_policy"]


def test_rejects_path_only_manifest_match_when_content_hash_is_missing():
    with _case_dir("strict_split_enrich_no_path_fallback") as tmp_path:
        source = tmp_path / "same-path.exe"
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        _write_split(
            split_csv,
            [{"source_path": str(source), "label": "1", "sample_index": "2", "split": "train"}],
        )
        _write_manifest(
            manifest_json,
            [{"source_path": str(source), "label": 1, "source_sha256": "a" * 64}],
        )

        rows, summary = enrich_strict_split_metadata(split_csv=split_csv, manifest_json=manifest_json)

    assert rows[0]["source_sha256"] == ""
    assert summary["enrichment_ready"] is False
    assert summary["issue_counts"]["source_file_missing_for_hash"] == 1
    assert summary["row_issue_examples"][0]["issues"] == ["source_file_missing_for_hash", "split_missing_source_sha256"]
    assert summary["manifest_match_counts"] == {}


def test_rejects_computed_hash_that_does_not_exist_in_manifest_even_if_path_matches():
    with _case_dir("strict_split_enrich_hash_not_manifest") as tmp_path:
        source = tmp_path / "same-path.exe"
        source.write_bytes(b"MZactual")
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        _write_split(
            split_csv,
            [{"source_path": str(source), "label": "1", "sample_index": "3", "split": "test"}],
        )
        _write_manifest(
            manifest_json,
            [{"source_path": str(source), "label": 1, "source_sha256": "b" * 64}],
        )

        _rows, summary = enrich_strict_split_metadata(split_csv=split_csv, manifest_json=manifest_json)

    assert summary["enrichment_ready"] is False
    assert summary["issue_counts"]["manifest_missing_source_sha256"] == 1
    assert summary["manifest_match_counts"] == {}


def test_rejects_split_manifest_label_mismatch_after_hash_match():
    with _case_dir("strict_split_enrich_label_mismatch") as tmp_path:
        source = tmp_path / "renamed-anything.bin"
        payload = b"MZcontent"
        source.write_bytes(payload)
        source_sha = _sha_bytes(payload)
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        _write_split(
            split_csv,
            [{"source_path": str(source), "label": "1", "sample_index": "4", "split": "val"}],
        )
        _write_manifest(
            manifest_json,
            [{"source_path": str(source), "label": 0, "source_sha256": source_sha}],
        )

        rows, summary = enrich_strict_split_metadata(split_csv=split_csv, manifest_json=manifest_json)

    assert rows[0]["source_sha256"] == source_sha
    assert summary["enrichment_ready"] is False
    assert summary["issue_counts"]["label_mismatch_split_manifest"] == 1
    assert summary["manifest_match_counts"] == {}


def test_existing_hash_can_be_verified_against_file_content():
    with _case_dir("strict_split_enrich_verify_existing") as tmp_path:
        source = tmp_path / "sample.exe"
        source.write_bytes(b"MZactual")
        actual_sha = _sha_bytes(b"MZactual")
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        _write_split(
            split_csv,
            [{
                "source_path": str(source),
                "source_sha256": actual_sha,
                "label": "1",
                "sample_index": "5",
                "split": "test",
            }],
            include_hash=True,
        )
        _write_manifest(
            manifest_json,
            [{"source_path": "other.exe", "label": 1, "source_sha256": actual_sha}],
        )

        _rows, summary = enrich_strict_split_metadata(
            split_csv=split_csv,
            manifest_json=manifest_json,
            verify_existing_hash_from_source=True,
        )

    assert summary["enrichment_ready"] is True
    assert summary["hash_source_counts"]["existing_split_source_sha256"] == 1
    assert summary["hash_source_counts"]["verified_existing_hash_from_source_file"] == 1
