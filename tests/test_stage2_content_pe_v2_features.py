from pathlib import Path

import numpy as np

from scripts.train_stage2_cache_matrix import (
    CONTENT_PE_V2_FEATURE_NAMES,
    content_pe_v2_group_indices,
    content_pe_v2_selected_feature_names,
    parse_content_pe_v2_groups,
    _content_pe_v2_features_from_path,
)


def test_content_pe_v2_features_do_not_depend_on_filename(tmp_path: Path):
    payload = b"MZ" + bytes(range(64)) + b"same-invalid-pe-content"
    first = tmp_path / "benign-looking-name.exe"
    second = tmp_path / "random_hash_without_extension"
    first.write_bytes(payload)
    second.write_bytes(payload)

    first_features = _content_pe_v2_features_from_path(first)
    second_features = _content_pe_v2_features_from_path(second)

    assert first_features.shape == (len(CONTENT_PE_V2_FEATURE_NAMES),)
    assert second_features.shape == (len(CONTENT_PE_V2_FEATURE_NAMES),)
    np.testing.assert_array_equal(first_features, second_features)


def test_content_pe_v2_group_selection_is_stable():
    assert parse_content_pe_v2_groups("dll,apis,sections") == ("import_dll", "api", "section")

    all_indices = content_pe_v2_group_indices("all")
    imports_indices = content_pe_v2_group_indices("imports")
    section_names = content_pe_v2_selected_feature_names("section")

    assert len(all_indices) == len(CONTENT_PE_V2_FEATURE_NAMES)
    assert len(imports_indices) > 0
    assert len(imports_indices) < len(all_indices)
    assert all(
        name.startswith("v2_section_")
        or name.startswith("v2_ep_")
        or name.startswith("v2_first_section_")
        or name.startswith("v2_last_section_")
        for name in section_names
    )
