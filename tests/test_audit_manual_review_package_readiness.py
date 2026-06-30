from __future__ import annotations

import csv
import json
import shutil
import struct
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_manual_review_package_readiness import audit_manual_review_package  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _minimal_pe_bytes() -> bytes:
    data = bytearray(1024)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\x00\x00"
    struct.pack_into("<HHIIIHH", data, 0x84, 0x14C, 1, 0, 0, 0, 0xE0, 0x010F)
    opt = 0x98
    struct.pack_into("<H", data, opt, 0x10B)
    struct.pack_into("<I", data, opt + 16, 0x1000)
    struct.pack_into("<I", data, opt + 28, 0x400000)
    struct.pack_into("<I", data, opt + 56, 0x2000)
    struct.pack_into("<H", data, opt + 68, 3)
    struct.pack_into("<H", data, opt + 70, 0)
    section = 0x80 + 24 + 0xE0
    data[section:section + 8] = b".text\x00\x00\x00"
    struct.pack_into("<IIIIIIHHI", data, section + 8, 0x200, 0x1000, 0x200, 0x200, 0, 0, 0, 0, 0x60000020)
    return bytes(data)


def _write_review(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_rank",
                "support_bucket",
                "priority",
                "reason",
                "error_type",
                "source_path",
                "source_sha256",
                "label",
                "prediction",
                "prob_malicious",
                "score_column",
                "manual_label_verdict",
                "recommended_action",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_manual_review_readiness_reports_ready_row():
    with _case_dir("manual_readiness_ready") as tmp_path:
        source_path = tmp_path / "data" / "sample.exe"
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(_minimal_pe_bytes())
        sha = "not-computed-yet"
        import hashlib

        sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        cache_path = tmp_path / "cache" / "sample.npz"
        cache_path.parent.mkdir()
        np.savez_compressed(
            cache_path,
            byte_sequence=np.array([77, 90], dtype=np.uint8),
            pe_features=np.array([1.0, 2.0], dtype=np.float32),
            stat_features=np.array([0.1], dtype=np.float32),
            lightweight_features=np.array([0.2, 0.3, 0.4], dtype=np.float32),
            label=np.array(0, dtype=np.int64),
            source_sha256=np.array(sha),
        )
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "max_byte_length": 2,
                    "pe_feature_dim": 2,
                    "stat_feature_dim": 1,
                    "lightweight_feature_dim": 3,
                    "samples": [
                        {
                            "source_path": str(source_path),
                            "source_sha256": sha,
                            "cache_path": str(cache_path),
                            "label": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        review_csv = tmp_path / "review.csv"
        _write_review(
            review_csv,
            [
                {
                    "review_rank": 1,
                    "support_bucket": "neighbors_support_model_prediction",
                    "priority": 0,
                    "reason": "severe_fp_prob_ge_0.95",
                    "error_type": "FP",
                    "source_path": str(source_path),
                    "source_sha256": sha,
                    "label": 0,
                    "prediction": 1,
                    "prob_malicious": "0.99",
                    "score_column": "blend_prob_malicious",
                    "manual_label_verdict": "label_correct",
                    "recommended_action": "keep",
                }
            ],
        )

        summary = audit_manual_review_package(
            review_csv=review_csv,
            manifest_json=manifest_path,
            output_csv=tmp_path / "readiness.csv",
            output_json=tmp_path / "readiness.json",
        )
        rows = list(csv.DictReader((tmp_path / "readiness.csv").open("r", encoding="utf-8-sig", newline="")))

    assert summary["total_rows"] == 1
    assert summary["ready_rows"] == 1
    assert summary["review_queue_ready"] is True
    assert summary["verdict_package_ready"] is True
    assert summary["manual_review_ready"] is True
    assert rows[0]["manual_review_ready"] == "True"
    assert rows[0]["readiness_reasons"] == ""


def test_manual_review_readiness_reports_sha_and_cache_failures():
    with _case_dir("manual_readiness_failures") as tmp_path:
        source_path = tmp_path / "data" / "sample.exe"
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(_minimal_pe_bytes())
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "source_path": str(source_path),
                            "source_sha256": "a" * 64,
                            "cache_path": str(tmp_path / "missing.npz"),
                            "label": 1,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        review_csv = tmp_path / "review.csv"
        _write_review(
            review_csv,
            [
                {
                    "review_rank": 1,
                    "source_path": str(source_path),
                    "source_sha256": "b" * 64,
                    "label": 1,
                    "prediction": 0,
                    "prob_malicious": "0.01",
                }
            ],
        )

        summary = audit_manual_review_package(
            review_csv=review_csv,
            manifest_json=manifest_path,
            output_csv=tmp_path / "readiness.csv",
            output_json=tmp_path / "readiness.json",
        )

    assert summary["ready_rows"] == 0
    assert summary["not_ready_rows"] == 1
    assert summary["review_queue_ready"] is False
    assert summary["verdict_package_ready"] is False
    assert summary["readiness_reason_counts"]["source_sha256_mismatch"] == 1
    assert summary["readiness_reason_counts"]["cache_missing"] == 1


def test_manual_review_readiness_distinguishes_blank_verdict_package():
    with _case_dir("manual_readiness_blank_verdict") as tmp_path:
        source_path = tmp_path / "data" / "sample.exe"
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(_minimal_pe_bytes())
        import hashlib

        sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        cache_path = tmp_path / "cache" / "sample.npz"
        cache_path.parent.mkdir()
        np.savez_compressed(
            cache_path,
            byte_sequence=np.array([77, 90], dtype=np.uint8),
            pe_features=np.array([1.0], dtype=np.float32),
            stat_features=np.array([0.1], dtype=np.float32),
            lightweight_features=np.array([0.2], dtype=np.float32),
            label=np.array(0, dtype=np.int64),
            source_sha256=np.array(sha),
        )
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "max_byte_length": 2,
                    "pe_feature_dim": 1,
                    "stat_feature_dim": 1,
                    "lightweight_feature_dim": 1,
                    "samples": [
                        {
                            "source_path": str(source_path),
                            "source_sha256": sha,
                            "cache_path": str(cache_path),
                            "label": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        review_csv = tmp_path / "review.csv"
        _write_review(
            review_csv,
            [
                {
                    "review_rank": 1,
                    "source_path": str(source_path),
                    "source_sha256": sha,
                    "label": 0,
                    "prediction": 1,
                    "prob_malicious": "0.99",
                    "manual_label_verdict": "",
                    "recommended_action": "",
                }
            ],
        )

        summary = audit_manual_review_package(
            review_csv=review_csv,
            manifest_json=manifest_path,
            output_csv=tmp_path / "readiness.csv",
            output_json=tmp_path / "readiness.json",
        )

    assert summary["review_queue_ready"] is True
    assert summary["verdict_package_ready"] is False
    assert summary["manual_label_verdict_blank_count"] == 1
    assert summary["recommended_action_blank_count"] == 1
    assert summary["blocking_issues"] == ["manual_verdict_empty", "recommended_action_empty"]
