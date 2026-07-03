from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path

from scripts.build_loop86_review_evidence_package import build_evidence_package


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _minimal_pe_bytes() -> bytes:
    data = bytearray(0x400)
    data[0:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\x00\x00"
    coff_offset = 0x84
    struct.pack_into("<HHIIIHH", data, coff_offset, 0x14C, 1, 123456, 0, 0, 0xE0, 0x010F)
    optional_offset = 0x98
    struct.pack_into("<H", data, optional_offset, 0x10B)
    struct.pack_into("<H", data, optional_offset + 68, 3)
    struct.pack_into("<H", data, optional_offset + 70, 0x8140)
    struct.pack_into("<I", data, optional_offset + 92, 16)
    # Import directory and security directory sizes.
    struct.pack_into("<II", data, optional_offset + 96 + 1 * 8, 0x2000, 40)
    struct.pack_into("<II", data, optional_offset + 96 + 4 * 8, 0x300, 32)
    section_offset = optional_offset + 0xE0
    data[section_offset : section_offset + 8] = b".text\x00\x00\x00"
    struct.pack_into("<IIII", data, section_offset + 8, 0x100, 0x1000, 0x200, 0x200)
    struct.pack_into("<I", data, section_offset + 36, 0x60000020)
    data[0x200:0x400] = bytes([0x90]) * 0x200
    data.extend(bytes(range(64)))
    return bytes(data)


def test_loop86_builds_content_evidence_without_using_identity_as_evidence(tmp_path: Path):
    sample = tmp_path / "sample.exe"
    sample.write_bytes(_minimal_pe_bytes())
    sha = hashlib.sha256(sample.read_bytes()).hexdigest()
    cache = tmp_path / "sample.npz"
    cache.write_bytes(b"cache")
    review_csv = tmp_path / "review.csv"
    output_csv = tmp_path / "evidence.csv"
    output_json = tmp_path / "summary.json"
    _write_csv(
        review_csv,
        [
            {
                "review_batch_rank": "1",
                "review_category": "a_severe_persistent_fn",
                "review_priority_rank": "1",
                "loop57_error_type": "FN",
                "label": "1",
                "source_path": str(sample),
                "cache_path": str(cache),
                "source_sha256": sha,
                "sample_index": "177755",
                "split": "test",
                "loop57_final_prob": "0.001",
                "duplicate_manifest_sha_group": "false",
                "manual_label_verdict": "",
                "manual_verdict_note": "",
                "recommended_action": "",
            }
        ],
    )

    summary = build_evidence_package(
        review_csv=review_csv,
        output_csv=output_csv,
        output_json=output_json,
        max_entropy_bytes=1024,
    )
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig", newline="")))

    assert summary["rows"] == 1
    assert summary["manual_fields_blank"] is True
    assert summary["source_exists_count"] == 1
    assert summary["source_sha256_mismatch_count"] == 0
    assert summary["pe_parse_status_counts"] == {"ok": 1}
    assert summary["decisions"]["training_allowed_from_this_package"] is False
    assert "source_path" in summary["identity_feature_policy"]["identity_columns"]
    assert "training features" in summary["identity_feature_policy"]["forbidden_identity_uses"]
    assert rows[0]["identity_columns_are_not_evidence"] == "true"
    assert rows[0]["model_score_columns_are_not_verdict_evidence"] == "true"
    assert rows[0]["source_sha256_match"] == "true"
    assert rows[0]["pe_parse_status"] == "ok"
    assert rows[0]["pe_has_import_directory"] == "true"
    assert rows[0]["pe_has_security_directory"] == "true"
    assert "overlay_present" in rows[0]["review_tags"]
    assert "source_path" in rows[0]


def test_loop86_missing_source_is_review_evidence_not_auto_replacement(tmp_path: Path):
    review_csv = tmp_path / "review.csv"
    output_csv = tmp_path / "evidence.csv"
    output_json = tmp_path / "summary.json"
    _write_csv(
        review_csv,
        [
            {
                "review_batch_rank": "1",
                "review_category": "b_severe_persistent_fp",
                "loop57_error_type": "FP",
                "label": "0",
                "source_path": str(tmp_path / "missing.exe"),
                "cache_path": "",
                "source_sha256": "abc",
                "sample_index": "5",
                "split": "test",
            }
        ],
    )

    summary = build_evidence_package(
        review_csv=review_csv,
        output_csv=output_csv,
        output_json=output_json,
        max_entropy_bytes=1024,
    )
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig", newline="")))
    saved = json.loads(output_json.read_text(encoding="utf-8"))

    assert summary == saved
    assert summary["source_exists_count"] == 0
    assert summary["source_sha256_mismatch_count"] == 1
    assert summary["decisions"]["automatic_replacement_allowed"] is False
    assert summary["decisions"]["automatic_relabel_allowed"] is False
    assert rows[0]["source_exists"] == "false"
    assert rows[0]["pe_parse_status"] == "source_missing"
    assert "source_missing" in rows[0]["review_tags"]
