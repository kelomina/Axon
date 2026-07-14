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

from audit_manual_review_package_readiness import audit_manual_review_package, main  # noqa: E402


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
                "top5_neighbor_labels",
                "top5_neighbor_similarities",
                "top5_neighbor_sha256",
                "top5_neighbor_paths",
                "manual_label_verdict",
                "recommended_action",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_cache(cache_path: Path, *, label: int, sha: str, pe_dim: int = 2, stat_dim: int = 1, light_dim: int = 3) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        byte_sequence=np.array([77, 90], dtype=np.uint8),
        pe_features=np.arange(pe_dim, dtype=np.float32),
        stat_features=np.arange(stat_dim, dtype=np.float32),
        lightweight_features=np.arange(light_dim, dtype=np.float32),
        label=np.array(label, dtype=np.int64),
        source_sha256=np.array(sha),
    )


def _make_source(tmp_path: Path, relative_name: str) -> tuple[Path, str]:
    source_path = tmp_path / "data" / relative_name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(_minimal_pe_bytes())
    import hashlib

    return source_path, hashlib.sha256(source_path.read_bytes()).hexdigest()


def _make_neighbor_samples(tmp_path: Path, count: int = 5) -> tuple[list[dict], dict]:
    samples = []
    labels = []
    similarities = []
    shas = []
    paths = []
    for index in range(count):
        source_path, sha = _make_source(tmp_path, f"neighbor-{index}.exe")
        cache_path = tmp_path / "cache" / f"neighbor-{index}.npz"
        _write_cache(cache_path, label=1, sha=sha)
        samples.append({"source_path": str(source_path), "source_sha256": sha, "cache_path": str(cache_path), "label": 1})
        labels.append("1")
        similarities.append(f"{0.99 - index * 0.01:.8f}")
        shas.append(sha)
        paths.append(str(source_path))
    return samples, {
        "top5_neighbor_labels": "|".join(labels),
        "top5_neighbor_similarities": "|".join(similarities),
        "top5_neighbor_sha256": " | ".join(shas),
        "top5_neighbor_paths": " | ".join(paths),
    }


def test_manual_review_readiness_reports_ready_row():
    with _case_dir("manual_readiness_ready") as tmp_path:
        source_path, sha = _make_source(tmp_path, "sample.exe")
        cache_path = tmp_path / "cache" / "sample.npz"
        _write_cache(cache_path, label=0, sha=sha)
        neighbor_samples, neighbor_fields = _make_neighbor_samples(tmp_path)
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
                    ]
                    + neighbor_samples,
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
                    **neighbor_fields,
                    "manual_label_verdict": "label_correct",
                    "recommended_action": "keep_label",
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
    assert summary["manual_label_verdict_invalid_count"] == 0
    assert summary["recommended_action_invalid_count"] == 0
    assert summary["top5_neighbor_evidence_ok_count"] == 1
    assert rows[0]["manual_review_ready"] == "True"
    assert rows[0]["readiness_reasons"] == ""


def test_manual_review_readiness_reports_sha_and_cache_failures():
    with _case_dir("manual_readiness_failures") as tmp_path:
        source_path, _actual_sha = _make_source(tmp_path, "sample.exe")
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
        source_path, sha = _make_source(tmp_path, "sample.exe")
        cache_path = tmp_path / "cache" / "sample.npz"
        _write_cache(cache_path, label=0, sha=sha, pe_dim=1, stat_dim=1, light_dim=1)
        neighbor_samples, neighbor_fields = _make_neighbor_samples(tmp_path)
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
                    ]
                    + neighbor_samples,
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
                    **neighbor_fields,
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
    assert summary["manual_label_verdict_invalid_count"] == 0
    assert summary["recommended_action_invalid_count"] == 0
    assert summary["blocking_issues"] == ["manual_verdict_empty", "recommended_action_empty"]


def test_manual_review_readiness_flags_incomplete_neighbor_evidence():
    with _case_dir("manual_readiness_bad_neighbors") as tmp_path:
        source_path, sha = _make_source(tmp_path, "sample.exe")
        cache_path = tmp_path / "cache" / "sample.npz"
        _write_cache(cache_path, label=0, sha=sha)
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
                    "source_path": str(source_path),
                    "source_sha256": sha,
                    "label": 0,
                    "prediction": 1,
                    "prob_malicious": "0.99",
                    "top5_neighbor_labels": "1|1",
                    "top5_neighbor_similarities": "0.9|0.8",
                    "top5_neighbor_sha256": "a" * 64,
                    "top5_neighbor_paths": "missing-a.exe",
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

    assert summary["review_queue_ready"] is False
    assert summary["top5_neighbor_evidence_ok_count"] == 0
    assert summary["readiness_reason_counts"]["top5_neighbor_evidence_incomplete"] == 1


def test_manual_review_readiness_rejects_unknown_manual_field_values():
    with _case_dir("manual_readiness_bad_manual_fields") as tmp_path:
        source_path, sha = _make_source(tmp_path, "sample.exe")
        cache_path = tmp_path / "cache" / "sample.npz"
        _write_cache(cache_path, label=0, sha=sha)
        neighbor_samples, neighbor_fields = _make_neighbor_samples(tmp_path)
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
                    ]
                    + neighbor_samples,
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
                    **neighbor_fields,
                    "manual_label_verdict": "definitely_bad",
                    "recommended_action": "magic_action",
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
    assert summary["manual_label_verdict_invalid_count"] == 1
    assert summary["recommended_action_invalid_count"] == 1
    assert summary["blocking_issues"] == ["manual_verdict_invalid", "recommended_action_invalid"]


def test_manual_review_readiness_rejects_inconsistent_manual_field_pair():
    with _case_dir("manual_readiness_inconsistent_manual_fields") as tmp_path:
        source_path, sha = _make_source(tmp_path, "sample.exe")
        cache_path = tmp_path / "cache" / "sample.npz"
        _write_cache(cache_path, label=0, sha=sha)
        neighbor_samples, neighbor_fields = _make_neighbor_samples(tmp_path)
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
                    ]
                    + neighbor_samples,
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
                    **neighbor_fields,
                    "manual_label_verdict": "feature_broken",
                    "recommended_action": "relabel_train_only",
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

    assert summary["review_queue_ready"] is True
    assert summary["verdict_package_ready"] is False
    assert summary["manual_label_verdict_invalid_count"] == 0
    assert summary["recommended_action_invalid_count"] == 0
    assert summary["manual_fields_inconsistent_count"] == 1
    assert summary["blocking_issues"] == ["manual_fields_inconsistent"]
    assert rows[0]["manual_verdict_category"] == "exclude"
    assert rows[0]["recommended_action_category"] == "relabel"
    assert rows[0]["manual_fields_consistent"] == "False"


def test_strict_cli_fails_when_review_ready_but_manual_verdicts_blank():
    with _case_dir("manual_readiness_strict_blank_verdict") as tmp_path:
        source_path, sha = _make_source(tmp_path, "sample.exe")
        cache_path = tmp_path / "cache" / "sample.npz"
        _write_cache(cache_path, label=0, sha=sha, pe_dim=1, stat_dim=1, light_dim=1)
        neighbor_samples, neighbor_fields = _make_neighbor_samples(tmp_path)
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
                    ]
                    + neighbor_samples,
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
                    **neighbor_fields,
                    "manual_label_verdict": "",
                    "recommended_action": "",
                }
            ],
        )

        exit_code = main(
            [
                "--review-csv",
                str(review_csv),
                "--manifest-json",
                str(manifest_path),
                "--output-csv",
                str(tmp_path / "readiness.csv"),
                "--output-json",
                str(tmp_path / "readiness.json"),
                "--strict",
            ]
        )
        summary = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))

    assert summary["review_queue_ready"] is True
    assert summary["verdict_package_ready"] is False
    assert summary["blocking_issues"] == ["manual_verdict_empty", "recommended_action_empty"]
    assert exit_code == 2


def test_strict_cli_passes_when_verdict_package_ready():
    with _case_dir("manual_readiness_strict_ready") as tmp_path:
        source_path, sha = _make_source(tmp_path, "sample.exe")
        cache_path = tmp_path / "cache" / "sample.npz"
        _write_cache(cache_path, label=0, sha=sha, pe_dim=1, stat_dim=1, light_dim=1)
        neighbor_samples, neighbor_fields = _make_neighbor_samples(tmp_path)
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
                    ]
                    + neighbor_samples,
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
                    **neighbor_fields,
                    "manual_label_verdict": "label_correct",
                    "recommended_action": "keep_label",
                }
            ],
        )

        exit_code = main(
            [
                "--review-csv",
                str(review_csv),
                "--manifest-json",
                str(manifest_path),
                "--output-csv",
                str(tmp_path / "readiness.csv"),
                "--output-json",
                str(tmp_path / "readiness.json"),
                "--strict",
            ]
        )
        summary = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))

    assert summary["review_queue_ready"] is True
    assert summary["verdict_package_ready"] is True
    assert summary["blocking_issues"] == []
    assert exit_code == 0


def test_non_strict_cli_reports_blank_verdicts_without_failing():
    with _case_dir("manual_readiness_non_strict_blank_verdict") as tmp_path:
        source_path, sha = _make_source(tmp_path, "sample.exe")
        cache_path = tmp_path / "cache" / "sample.npz"
        _write_cache(cache_path, label=0, sha=sha, pe_dim=1, stat_dim=1, light_dim=1)
        neighbor_samples, neighbor_fields = _make_neighbor_samples(tmp_path)
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
                    ]
                    + neighbor_samples,
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
                    **neighbor_fields,
                    "manual_label_verdict": "",
                    "recommended_action": "",
                }
            ],
        )

        exit_code = main(
            [
                "--review-csv",
                str(review_csv),
                "--manifest-json",
                str(manifest_path),
                "--output-csv",
                str(tmp_path / "readiness.csv"),
                "--output-json",
                str(tmp_path / "readiness.json"),
            ]
        )
        summary = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))

    assert summary["review_queue_ready"] is True
    assert summary["verdict_package_ready"] is False
    assert summary["blocking_issues"] == ["manual_verdict_empty", "recommended_action_empty"]
    assert exit_code == 0


def test_strict_cli_fails_when_manual_fields_are_inconsistent():
    with _case_dir("manual_readiness_strict_inconsistent_fields") as tmp_path:
        source_path, sha = _make_source(tmp_path, "sample.exe")
        cache_path = tmp_path / "cache" / "sample.npz"
        _write_cache(cache_path, label=0, sha=sha, pe_dim=1, stat_dim=1, light_dim=1)
        neighbor_samples, neighbor_fields = _make_neighbor_samples(tmp_path)
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
                    ]
                    + neighbor_samples,
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
                    **neighbor_fields,
                    "manual_label_verdict": "feature_broken",
                    "recommended_action": "relabel_train_only",
                }
            ],
        )

        exit_code = main(
            [
                "--review-csv",
                str(review_csv),
                "--manifest-json",
                str(manifest_path),
                "--output-csv",
                str(tmp_path / "readiness.csv"),
                "--output-json",
                str(tmp_path / "readiness.json"),
                "--strict",
            ]
        )
        summary = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))

    assert summary["review_queue_ready"] is True
    assert summary["verdict_package_ready"] is False
    assert summary["blocking_issues"] == ["manual_fields_inconsistent"]
    assert exit_code == 2
