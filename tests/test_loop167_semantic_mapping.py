from __future__ import annotations

from src.loop167.semantic_mapping import (
    CATEGORY_EXACT,
    CATEGORY_FORBIDDEN,
    CATEGORY_NOVEL,
    CATEGORY_PARTIAL,
    build_frozen_baseline_allowlist,
    build_semantic_delta_mapping,
    control_indices,
    novel_indices,
    semantic_mapping_rows,
)
from src.loop167.semantic_schema import OFFICIAL_DIMENSION, OFFICIAL_GROUPS


def test_source_order_and_category_conservation_are_frozen() -> None:
    rows = semantic_mapping_rows()
    assert len(rows) == OFFICIAL_DIMENSION == 2568
    assert [row["index"] for row in rows] == list(range(2568))
    assert [(group.name, group.start, group.stop) for group in OFFICIAL_GROUPS] == [
        ("general", 0, 7),
        ("histogram", 7, 263),
        ("byteentropy", 263, 519),
        ("strings", 519, 696),
        ("header", 696, 770),
        ("section", 770, 994),
        ("imports", 994, 2276),
        ("exports", 2276, 2405),
        ("datadirectories", 2405, 2439),
        ("richheader", 2439, 2472),
        ("authenticode", 2472, 2480),
        ("pefilewarnings", 2480, 2568),
    ]
    mapping = build_semantic_delta_mapping()
    assert mapping["category_counts"] == {
        CATEGORY_EXACT: 49,
        CATEGORY_PARTIAL: 487,
        CATEGORY_NOVEL: 292,
        CATEGORY_FORBIDDEN: 1740,
    }
    assert rows[696]["official_group"] == "header"
    assert rows[770]["official_group"] == "section"
    assert rows[994]["official_group"] == "imports"
    assert rows[2276]["official_group"] == "exports"


def test_known_source_traps_are_explicitly_frozen() -> None:
    rows = semantic_mapping_rows()
    assert [rows[index]["category"] for index in range(3, 7)] == [CATEGORY_NOVEL] * 4
    assert [rows[index]["category"] for index in range(263, 519)] == [CATEGORY_NOVEL] * 256
    assert rows[2439]["category"] == CATEGORY_NOVEL
    assert rows[2276]["category"] == CATEGORY_EXACT
    assert rows[2276]["official_name"] == "export_hash_vector_length_sentinel"
    assert [rows[index]["category"] for index in (2435, 2436)] == [CATEGORY_FORBIDDEN] * 2
    assert all(
        row["category"] != CATEGORY_NOVEL
        for row in rows
        if row["official_group"] in {"datadirectories", "authenticode"}
    )


def test_selected_sets_are_disjoint_and_forbidden_columns_cannot_leak() -> None:
    rows = semantic_mapping_rows()
    novel = set(novel_indices())
    controls = set(control_indices())
    forbidden = {int(row["index"]) for row in rows if row["category"] == CATEGORY_FORBIDDEN}
    assert len(novel) == 292
    assert len(controls) == 536
    assert not novel & controls
    assert not novel & forbidden
    assert not controls & forbidden
    assert {rows[index]["official_group"] for index in novel} >= {
        "general",
        "byteentropy",
        "header",
        "richheader",
    }
    assert all(
        row["implementation_status"] == "implemented_native_phase_a"
        for row in rows
        if row["category"] == CATEGORY_NOVEL
    )


def test_baseline_allowlist_is_named_and_exact_duplicate_only() -> None:
    allowlist = build_frozen_baseline_allowlist()
    assert allowlist["source_inventory_dimension"] == 572
    assert allowlist["frozen_allowlist_dimension"] == 571
    assert len(allowlist["feature_names"]) == 571
    assert all("." in name for name in allowlist["feature_names"])
    dropped = [column for column in allowlist["columns"] if not column["included"]]
    assert dropped == [
        {
            "inventory_index": 192,
            "source_family": "content_pe_v1",
            "source_index": 0,
            "feature_name": "content_file_log_size",
            "semantic_key": "content_pe_v1.content_file_log_size",
            "canonical_source": "fixed_v2.fixed_v2_log_size",
            "included": False,
            "dropped_duplicate_of": "fixed_v2.fixed_v2_log_size",
        }
    ]
    excluded = set(allowlist["excluded_feature_families"])
    assert {"checkpoint_score", "path_filename_extension", "source_sha256_row_fold_label_family_time"} <= excluded
