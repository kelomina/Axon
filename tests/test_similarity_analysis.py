import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_similarity import AnalysisOptions, analyze_similarity, select_manifest  # noqa: E402
from config import AxonExperimentConfig  # noqa: E402


def _write_npz(path, byte_sequence, pe_features, stat_features, label):
    np.savez_compressed(
        path,
        byte_sequence=np.array(byte_sequence, dtype=np.uint8),
        pe_features=np.array(pe_features, dtype=np.float32),
        stat_features=np.array(stat_features, dtype=np.float32),
        label=int(label),
    )


def _write_manifest(path, samples, *, max_byte_length=8, pe_feature_dim=4, stat_feature_dim=2):
    manifest = {
        "version": 1,
        "data_dir": str(path.parents[1]),
        "cache_config_hash": path.stem.replace("manifest_", ""),
        "max_byte_length": max_byte_length,
        "pe_feature_dim": pe_feature_dim,
        "stat_feature_dim": stat_feature_dim,
        "lightweight_feature_dim": 256,
        "strict_pe_parsing": True,
        "allow_pe_fallback": False,
        "pe_schema_version": "fixed_v2",
        "pe_fixed_section_slots": 32,
        "samples": samples,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest


def test_similarity_analysis_reports_duplicate_and_feature_pairs(tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    sample_paths = [cache_dir / f"sample_{idx}.npz" for idx in range(4)]

    _write_npz(sample_paths[0], [77, 90, 1, 2], [1.0, 1.0, 0.0, 0.0], [0.2, 0.2], 0)
    _write_npz(sample_paths[1], [77, 90, 1, 2], [1.0, 1.0, 0.0, 0.0], [0.2, 0.2], 0)
    _write_npz(sample_paths[2], [77, 90, 3, 4], [0.0, 0.0, 1.0, 1.0], [0.7, 0.7], 1)
    _write_npz(sample_paths[3], [77, 90, 5, 6], [0.0, 0.0, 0.9, 0.9], [0.69, 0.69], 1)

    samples = [
        {"source_path": f"source_{idx}.exe", "cache_path": str(path), "label": idx // 2}
        for idx, path in enumerate(sample_paths)
    ]
    manifest_path = cache_dir / "manifest_test.json"
    _write_manifest(manifest_path, samples)

    output_dir = tmp_path / "reports"
    summary = analyze_similarity(
        AnalysisOptions(
            manifest_path=manifest_path,
            output_dir=output_dir,
            similarity_threshold=0.95,
            simhash_bits=16,
            lsh_band_size=4,
            max_bucket_size=10,
            seed=123,
        )
    )

    assert summary["analyzed_samples"] == 4
    assert summary["pair_counts"]["byte_duplicate"] >= 1
    assert summary["pair_counts"]["feature_similar"] >= 1
    assert summary["group_count"] >= 1

    pair_path = output_dir / "sample_similarity_pairs.csv"
    group_path = output_dir / "sample_similarity_groups.csv"
    summary_path = output_dir / "sample_similarity_summary.json"
    assert pair_path.exists()
    assert group_path.exists()
    assert summary_path.exists()

    with pair_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    duplicate_rows = [row for row in rows if "byte_duplicate" in row["methods"]]
    feature_rows = [row for row in rows if "feature_similar" in row["methods"]]
    assert duplicate_rows
    assert feature_rows
    assert {duplicate_rows[0]["source_path_i"], duplicate_rows[0]["source_path_j"]} == {
        "source_0.exe",
        "source_1.exe",
    }
    assert {"split_i", "split_j", "similarity", "label_i", "label_j"}.issubset(rows[0])


def test_similarity_analysis_does_not_report_unrelated_samples(tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    sample_paths = [cache_dir / f"sample_{idx}.npz" for idx in range(2)]

    _write_npz(sample_paths[0], [77, 90, 1, 2], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0], 0)
    _write_npz(sample_paths[1], [77, 90, 9, 9], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0], 1)

    samples = [
        {"source_path": f"source_{idx}.exe", "cache_path": str(path), "label": idx}
        for idx, path in enumerate(sample_paths)
    ]
    manifest_path = cache_dir / "manifest_test.json"
    _write_manifest(manifest_path, samples)

    summary = analyze_similarity(
        AnalysisOptions(
            manifest_path=manifest_path,
            output_dir=tmp_path / "reports",
            similarity_threshold=0.99,
            simhash_bits=16,
            lsh_band_size=4,
            max_bucket_size=10,
            seed=456,
        )
    )

    assert summary["pair_counts"]["total"] == 0
    assert summary["group_count"] == 0


def test_auto_manifest_selection_prefers_matching_fixed_v2_manifest(tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    sample_path = cache_dir / "sample.npz"
    _write_npz(
        sample_path,
        [77, 90],
        np.pad(np.array([1.0], dtype=np.float32), (0, 255)),
        np.zeros(49, dtype=np.float32),
        0,
    )

    fixed_manifest = cache_dir / "manifest_fixed.json"
    legacy_manifest = cache_dir / "manifest_legacy.json"
    sample = {"source_path": "source.exe", "cache_path": str(sample_path), "label": 0}
    _write_manifest(fixed_manifest, [sample], pe_feature_dim=256, stat_feature_dim=49)
    legacy = _write_manifest(legacy_manifest, [sample], pe_feature_dim=1500, stat_feature_dim=49)
    legacy["pe_schema_version"] = "legacy_dynamic"
    legacy_manifest.write_text(json.dumps(legacy), encoding="utf-8")

    config = AxonExperimentConfig(
        max_byte_length=8,
        pe_feature_dim=256,
        stat_feature_dim=49,
        pe_schema_version="fixed_v2",
    )
    manifest_path, _manifest, reason = select_manifest({}, config, tmp_path / "data", None)

    assert manifest_path == fixed_manifest
    assert reason == "auto-matched-config"
