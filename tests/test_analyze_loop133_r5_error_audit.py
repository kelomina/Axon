from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import analyze_loop133_r5_error_audit as loop133
from scripts import train_stage2_cache_matrix as stage2
from src.kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES


def _feature_table(row_count: int) -> dict[str, np.ndarray]:
    return {
        name: np.arange(row_count, dtype=np.float32) + offset
        for offset, name in enumerate(loop133.FEATURE_NAMES)
    }


def test_loop133_audit_counts_guard_harm_and_extra_flips():
    rows = [
        {"sample_index": "0", "source_sha256": "a" * 64, "label": "0", "prediction": "0", "stage2_prob_malicious": "0.20"},
        {"sample_index": "1", "source_sha256": "b" * 64, "label": "1", "prediction": "0", "stage2_prob_malicious": "0.05"},
        {"sample_index": "2", "source_sha256": "c" * 64, "label": "0", "prediction": "1", "stage2_prob_malicious": "0.95"},
        {"sample_index": "3", "source_sha256": "d" * 64, "label": "1", "prediction": "1", "stage2_prob_malicious": "0.80"},
    ]
    primary_rows = [{"prediction": "1"} for _ in rows]
    reference_rows = [
        {"prediction": "0"},
        {"prediction": "1"},
        {"prediction": "1"},
        {"prediction": "1"},
    ]

    summary, review_rows, flip_rows = loop133.build_audit(
        rows=rows,
        primary_rows=primary_rows,
        reference_rows=reference_rows,
        feature_table=_feature_table(len(rows)),
        max_review_rows=0,
    )

    assert summary["metrics"]["errors"] == 2
    assert summary["counts"]["fp"] == 1
    assert summary["counts"]["fn"] == 1
    assert summary["counts"]["guard_flips"] == 2
    assert summary["counts"]["guard_repaired_fp"] == 1
    assert summary["counts"]["guard_harmful_fn"] == 1
    assert summary["counts"]["extra_flips_over_reference"] == 1
    assert summary["counts"]["extra_harmful_fn"] == 1
    assert summary["counts"]["high_conf_fp_ge_0_90"] == 1
    assert summary["counts"]["high_conf_fn_lt_0_10"] == 1
    assert summary["feature_summary"]["guard_harmful_fn"]["count"] == 1

    assert [row["sample_index"] for row in review_rows] == ["1", "2"]
    assert review_rows[0]["reasons"].startswith("guard_harmful_fn;r5_extra_harmful_fn;high_conf_fn_lt_0.10")
    assert {row["outcome"] for row in flip_rows} == {"repaired_fp", "harmful_fn"}


def test_loop133_build_feature_table_loads_numeric_sidecars_by_sha(tmp_path: Path):
    source_sha = "a" * 64
    rows = [{"source_sha256": source_sha}]
    pe_v1_dir = tmp_path / "pe_v1"
    pe_v2_dir = tmp_path / "pe_v2"
    string_dir = tmp_path / "string"
    for directory in (pe_v1_dir, pe_v2_dir, string_dir):
        directory.mkdir()

    pe_v1 = np.zeros(len(CONTENT_PE_V1_FEATURE_NAMES), dtype=np.float32)
    pe_v2 = np.zeros(len(stage2.CONTENT_PE_V2_FEATURE_NAMES), dtype=np.float32)
    string = np.zeros(len(stage2.CONTENT_STRING_FEATURE_NAMES), dtype=np.float32)

    pe_v1[CONTENT_PE_V1_FEATURE_NAMES.index("content_overlay_log_size")] = 7.0
    pe_v2[stage2.CONTENT_PE_V2_FEATURE_NAMES.index("v2_resource_data_entry_count_log")] = 3.0
    string[stage2.CONTENT_STRING_FEATURE_NAMES.index("string_benign_vendor_count_log")] = 5.0

    np.savez(pe_v1_dir / f"{source_sha}.npz", features=pe_v1)
    np.savez(pe_v2_dir / f"{source_sha}.npz", features=pe_v2)
    np.savez(string_dir / f"{source_sha}.npz", features=string)

    table = loop133.build_feature_table(rows, pe_v1_dir, pe_v2_dir, string_dir)

    assert table["content_overlay_log_size"].tolist() == [7.0]
    assert table["v2_resource_data_entry_count_log"].tolist() == [3.0]
    assert table["string_benign_vendor_count_log"].tolist() == [5.0]


def test_loop133_align_rows_rejects_duplicate_keys():
    rows = [{"sample_index": "1", "source_sha256": "a"}]
    other_rows = [
        {"sample_index": "1", "source_sha256": "a"},
        {"sample_index": "1", "source_sha256": "a"},
    ]

    try:
        loop133.align_rows(rows, other_rows, ("sample_index", "source_sha256"), "primary")
    except ValueError as exc:
        assert "duplicate alignment keys" in str(exc)
    else:
        raise AssertionError("duplicate alignment keys should fail")
