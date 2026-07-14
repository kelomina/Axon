import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_raw_similarity import RawAnalysisOptions, analyze_raw_similarity  # noqa: E402


def _write_config(path, data_dir):
    path.write_text(
        f"""
[experiment]
name = "raw_similarity_test"
seed = 42

[model]
pe_feature_dim = 256
pe_schema_version = "fixed_v2"

[data]
data_dir = "{data_dir.as_posix()}"
val_ratio = 0.2
test_ratio = 0.2
benign_dir_names_fs = ["benign"]
malicious_dir_names_fs = ["malicious"]
""",
        encoding="utf-8",
    )


def _write_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_raw_similarity_reports_duplicates_and_chunk_similarity(tmp_path):
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir)

    shared_a = b"A" * 4096
    shared_b = b"B" * 4096
    _write_file(data_dir / "benign" / "a.exe", b"MZ" + shared_a + shared_b)
    _write_file(data_dir / "benign" / "b.exe", b"MZ" + shared_a + shared_b)
    _write_file(data_dir / "malicious" / "c.exe", b"MZ" + shared_a + b"C" * 4096)
    _write_file(data_dir / "malicious" / "d.exe", b"MZ" + b"X" * 4096 + b"Y" * 4096)

    output_dir = tmp_path / "reports"
    summary = analyze_raw_similarity(
        RawAnalysisOptions(
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            similarity_threshold=0.5,
            chunk_size=4096,
            minhash_size=8,
            lsh_band_size=1,
            max_bucket_size=10,
        )
    )

    assert summary["mode"] == "raw_files"
    assert summary["analyzed_samples"] == 4
    assert summary["pair_counts"]["raw_duplicate"] >= 1
    assert summary["pair_counts"]["chunk_similar"] >= 1
    assert summary["group_count"] >= 1

    pair_path = output_dir / "raw_similarity_pairs.csv"
    group_path = output_dir / "raw_similarity_groups.csv"
    summary_path = output_dir / "raw_similarity_summary.json"
    skipped_path = output_dir / "raw_similarity_skipped.csv"
    assert pair_path.exists()
    assert group_path.exists()
    assert summary_path.exists()
    assert skipped_path.exists()

    with pair_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert any("raw_duplicate" in row["methods"] for row in rows)
    assert any("chunk_similar" in row["methods"] for row in rows)
    assert {"split_i", "split_j", "chunk_similarity", "source_path_i", "source_path_j"}.issubset(rows[0])


def test_raw_similarity_skips_non_pe_by_default(tmp_path):
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir)

    _write_file(data_dir / "benign" / "a.exe", b"MZ" + b"A" * 4096)
    _write_file(data_dir / "benign" / "note.txt", b"not a PE file")
    _write_file(data_dir / "malicious" / "b.exe", b"MZ" + b"B" * 4096)

    output_dir = tmp_path / "reports"
    summary = analyze_raw_similarity(
        RawAnalysisOptions(
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            chunk_size=4096,
            minhash_size=8,
            lsh_band_size=1,
            max_bucket_size=10,
        )
    )

    assert summary["analyzed_samples"] == 2
    assert summary["skipped_samples"] == 1
    skipped = (output_dir / "raw_similarity_skipped.csv").read_text(encoding="utf-8-sig")
    assert "not_pe_mz" in skipped


def test_raw_similarity_does_not_follow_symlinked_dirs_by_default(tmp_path):
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir)

    _write_file(data_dir / "benign" / "a.exe", b"MZ" + b"A" * 4096)
    _write_file(data_dir / "malicious" / "b.exe", b"MZ" + b"B" * 4096)
    external_dir = tmp_path / "external_samples"
    _write_file(external_dir / "outside.exe", b"MZ" + b"C" * 4096)

    symlink_path = data_dir / "benign" / "linked_external"
    try:
        symlink_path.symlink_to(external_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable in this environment: {exc}")

    output_dir = tmp_path / "reports"
    summary = analyze_raw_similarity(
        RawAnalysisOptions(
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            chunk_size=4096,
            minhash_size=8,
            lsh_band_size=1,
            max_bucket_size=10,
        )
    )

    assert summary["analyzed_samples"] == 2
    records = (output_dir / "raw_similarity_summary.json").read_text(encoding="utf-8")
    assert "outside.exe" not in records


def test_raw_similarity_summary_is_json_serializable(tmp_path):
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir)

    _write_file(data_dir / "benign" / "a.exe", b"MZ" + b"A" * 4096)
    _write_file(data_dir / "malicious" / "b.exe", b"MZ" + b"B" * 4096)

    output_dir = tmp_path / "reports"
    analyze_raw_similarity(
        RawAnalysisOptions(
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            chunk_size=4096,
            minhash_size=8,
            lsh_band_size=1,
            max_bucket_size=10,
        )
    )

    summary = json.loads((output_dir / "raw_similarity_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "raw_files"
    assert summary["outputs"]["pairs"].endswith("raw_similarity_pairs.csv")
