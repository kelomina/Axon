import csv
from pathlib import Path

import pytest

from scripts.build_feature_cache_from_split import (
    cache_hash_for_config,
    load_extraction_config,
    load_split_rows,
)


def write_split(path: Path, duplicate: bool = False) -> None:
    digest = "a" * 64
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_path", "sha256", "label", "split"])
        writer.writerow(["C:/samples/a.exe", digest, 0, "train"])
        if duplicate:
            writer.writerow(["C:/samples/b.exe", digest, 1, "test"])


def test_load_split_rows_accepts_legacy_sha_column(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    write_split(split)
    rows = load_split_rows(split)
    assert rows == [
        {
            "source_path": "C:/samples/a.exe",
            "source_sha256": "a" * 64,
            "label": "0",
            "split": "train",
        }
    ]


def test_load_split_rows_rejects_duplicate_hash(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    write_split(split, duplicate=True)
    with pytest.raises(ValueError, match="duplicate source SHA-256"):
        load_split_rows(split)


def test_funnel_config_is_fixed_v2_and_cache_bound() -> None:
    config = load_extraction_config(Path("config/funnel_712_fixedv2.toml"))
    assert config.pe_schema_version == "fixed_v2"
    assert config.max_byte_length == 8192
    assert config.cache_dir == "data/.cache_712_fixedv2"
    assert len(cache_hash_for_config(config)) == 8
