from pathlib import Path
import struct

import numpy as np

from scripts.train_stage2_cache_matrix import (
    CONTENT_PE_V2_FEATURE_NAMES,
    content_pe_v2_group_indices,
    content_pe_v2_selected_feature_names,
    parse_content_pe_v2_groups,
    _content_pe_v2_features_from_path,
)


def _minimal_pe32() -> bytes:
    data = bytearray(0x400)
    data[0:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    pe_offset = 0x80
    data[pe_offset : pe_offset + 4] = b"PE\0\0"
    coff_offset = pe_offset + 4
    struct.pack_into("<HHIIIHH", data, coff_offset, 0x14C, 1, 0, 0, 0, 0xE0, 0x010F)
    optional_offset = coff_offset + 20
    struct.pack_into(
        "<HBBIIIIII",
        data,
        optional_offset,
        0x10B,
        14,
        0,
        0x200,
        0x200,
        0,
        0x1000,
        0x1000,
        0x2000,
    )
    struct.pack_into("<I", data, optional_offset + 28, 0x400000)
    struct.pack_into("<II", data, optional_offset + 32, 0x1000, 0x200)
    struct.pack_into("<HHHHHH", data, optional_offset + 40, 6, 0, 0, 0, 6, 0)
    struct.pack_into("<I", data, optional_offset + 52, 0)
    struct.pack_into("<III", data, optional_offset + 56, 0x3000, 0x200, 0)
    struct.pack_into("<HH", data, optional_offset + 68, 3, 0x8140)
    struct.pack_into("<IIIIII", data, optional_offset + 72, 0x100000, 0x1000, 0x100000, 0x1000, 0, 16)
    section_offset = optional_offset + 0xE0
    data[section_offset : section_offset + 8] = b".text\0\0\0"
    struct.pack_into(
        "<IIIIIIHHI",
        data,
        section_offset + 8,
        0x1000,
        0x1000,
        0x200,
        0x200,
        0,
        0,
        0,
        0,
        0x60000020,
    )
    data[0x200:0x400] = b"\x90" * 0x200
    return bytes(data)


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


def test_content_pe_v2_features_extract_real_pe_section_signals(tmp_path: Path):
    sample = tmp_path / "sample.exe"
    sample.write_bytes(_minimal_pe32())

    features = _content_pe_v2_features_from_path(sample)
    by_name = dict(zip(CONTENT_PE_V2_FEATURE_NAMES, features))

    assert features.shape == (len(CONTENT_PE_V2_FEATURE_NAMES),)
    assert np.count_nonzero(features) > 0
    assert by_name["v2_section_exec_count_log"] > 0.0
    assert by_name["v2_ep_in_exec_section"] == 1.0
    assert by_name["v2_section_name_group_code_ratio"] == 1.0


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
