import csv
from pathlib import Path

import pytest

from scripts.build_712_funnel_manifests import build_funnel, parse_stage_sizes


def write_source(path: Path, rows_per_cell: int = 100) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "sha256", "label", "split", "date", "source_path"])
        index = 0
        for split in ("train", "val", "test"):
            for label in (0, 1):
                for ordinal in range(rows_per_cell):
                    digest = f"{index:064x}"
                    writer.writerow([index, digest, label, split, "", f"C:/samples/{digest}.exe"])
                    index += 1


def test_build_funnel_is_nested_and_normalizes_legacy_sha_column(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    write_source(source)
    payload = build_funnel(
        source_split=source,
        output_dir=tmp_path / "out",
        stage_sizes=(20, 100, 200),
        selection_seed=9997,
        excluded_sha256=set(),
    )

    assert payload["artifacts"]["20"]["rows"] == 20
    assert payload["artifacts"]["100"]["rows"] == 100
    assert payload["artifacts"]["200"]["rows"] == 200
    assert all(item["missing_rows"] == 0 for item in payload["nesting"])
    with (tmp_path / "out" / "split_20.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "source_sha256" in (reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == 20
    assert {row["label"] for row in rows} == {"0", "1"}


def test_parse_stage_sizes_rejects_non_nested_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_stage_sizes("100,20")
