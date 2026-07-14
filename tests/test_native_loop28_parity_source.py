from __future__ import annotations

import ctypes
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from src.kvd_features.content_pe_v1 import (
    CONTENT_PE_V1_FEATURE_NAMES,
    extract_content_pe_v1_features,
)
from src.kvd_features.extractor import (
    ExtractionConfig,
    PEFeatureExtractor,
    calculate_byte_entropy,
    extract_statistical_features,
)
from src.kvd_features.schema_names import fixed_v2_feature_names

ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = ROOT / "tools" / "axon_onnx_dll" / "src" / "axon_onnx_predict.cpp"
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_localized_feature_indices_remain_schema_bound() -> None:
    pe_names = fixed_v2_feature_names(section_slots=32, pe_feature_dim=256)
    assert pe_names[117] == "fixed_v2_section_entropy_std"
    assert pe_names[125] == "fixed_v2_section_raw_size_std"
    assert pe_names[126] == "fixed_v2_section_raw_size_cv"
    assert pe_names[135] == "fixed_v2_api_network_ratio"
    assert CONTENT_PE_V1_FEATURE_NAMES[76:81] == [
        "content_resource_entry_count_log",
        "content_resource_type_count_log",
        "content_tls_callback_count_log",
        "content_reloc_block_count_log",
        "content_reloc_entry_count_log",
    ]


def test_native_source_contains_exact_remediation_paths() -> None:
    source = NATIVE_SOURCE.read_text(encoding="utf-8")
    required_fragments = (
        "float numpy_pairwise_sum_f32(",
        "double numpy_pairwise_sum_f64(",
        "std::pair<double, double> numpy_mean_std_u8(",
        "float numpy_entropy_from_f32_counts(",
        "std::optional<std::size_t> parsed_rva_to_offset(",
        "pe.number_of_rva_and_sizes",
        "kMaxRelocationBlocks",
        'keyword == "connect" || keyword == "send" || keyword == "recv"',
        "ResourceDirectoryStats collect_resource_directory_stats(",
        "RelocationDirectoryStats collect_relocation_directory_stats(",
        "TlsDirectoryStats collect_tls_directory_stats(",
        "std::vector<double> node_values_;",
        "std::vector<double> node_num_thresholds_;",
        "double baseline_prediction_ = 0.0;",
        "SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL)",
    )
    for fragment in required_fragments:
        assert fragment in source
    assert "Resource/TLS/reloc fine-grained counts require deeper directory parsing" not in source


def _put_u16(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", buffer, offset, value)


def _put_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


def _put_u64(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", buffer, offset, value)


def _synthetic_pe_bytes(
    *,
    pe_plus: bool = False,
    named_resource: bool = False,
    tls_callbacks: bool = True,
) -> bytes:
    data = bytearray(0x2000)
    data[0:2] = b"MZ"
    _put_u32(data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    coff = 0x84
    optional = coff + 20
    optional_size = 0xF0 if pe_plus else 0xE0
    section_table = optional + optional_size
    machine = 0x8664 if pe_plus else 0x14C
    characteristics = 0x000F if pe_plus else 0x010F
    struct.pack_into(
        "<HHIIIHH",
        data,
        coff,
        machine,
        3,
        123456789,
        0,
        0,
        optional_size,
        characteristics,
    )

    _put_u16(data, optional, 0x20B if pe_plus else 0x10B)
    data[optional + 2] = 14
    data[optional + 3] = 7
    _put_u32(data, optional + 4, 0x600)
    _put_u32(data, optional + 8, 0xA00)
    _put_u32(data, optional + 16, 0x1000)
    if pe_plus:
        _put_u64(data, optional + 24, 0x140000000)
    else:
        _put_u32(data, optional + 28, 0x400000)
    _put_u32(data, optional + 32, 0x1000)
    _put_u32(data, optional + 36, 0x200)
    _put_u32(data, optional + 56, 0x4000)
    _put_u32(data, optional + 60, 0x200)
    _put_u16(data, optional + 68, 3)
    _put_u16(data, optional + 70, 0x8140)
    _put_u32(data, optional + (108 if pe_plus else 92), 16)
    data_directory = optional + (112 if pe_plus else 96)
    for index, rva, size in (
        (1, 0x2000, 0x100),
        (2, 0x2100, 0x100),
        (5, 0x2200, 0x10),
        (9, 0x2300, 0x28 if pe_plus else 0x18),
    ):
        _put_u32(data, data_directory + index * 8, rva)
        _put_u32(data, data_directory + index * 8 + 4, size)

    sections = (
        (b".text\0\0\0", 0x500, 0x1000, 0x600, 0x200, 0x60000020),
        (b".rdata\0\0", 0x600, 0x2000, 0x600, 0x800, 0x40000040),
        (b".data\0\0\0", 0x380, 0x3000, 0x400, 0xE00, 0xC0000040),
    )
    for index, (name, virtual_size, rva, raw_size, raw_offset, flags) in enumerate(sections):
        offset = section_table + index * 40
        data[offset : offset + 8] = name
        struct.pack_into("<IIII", data, offset + 8, virtual_size, rva, raw_size, raw_offset)
        _put_u32(data, offset + 36, flags)

    for index in range(0x600):
        data[0x200 + index] = (index * 29 + 7) & 0xFF
    for index in range(0x600):
        data[0x800 + index] = (index * 17 + 11) & 0xFF
    for index in range(0x400):
        data[0xE00 + index] = (index * 5 + 19) & 0xFF
    for index in range(0x1200, len(data)):
        data[index] = (index * 13 + 3) & 0xFF

    struct.pack_into("<IIIII", data, 0x800, 0x2040, 0, 0, 0x2060, 0x2040)
    struct.pack_into("<IIIII", data, 0x814, 0, 0, 0, 0, 0)
    if pe_plus:
        struct.pack_into("<QQQ", data, 0x840, 0x2080, 0x20A0, 0)
    else:
        struct.pack_into("<III", data, 0x840, 0x2080, 0x20A0, 0)
    data[0x860:0x86D] = b"kernel32.dll\0"
    _put_u16(data, 0x880, 0)
    data[0x882:0x88F] = b"MySendHelper\0"
    _put_u16(data, 0x8A0, 0)
    data[0x8A2:0x8B3] = b"ConnectNamedPipe\0"

    for directory_offset in (0x900, 0x920, 0x940):
        data[directory_offset : directory_offset + 16] = bytes(16)
        _put_u16(data, directory_offset + 14, 1)
    resource_name = 0x80000070 if named_resource else 10
    struct.pack_into("<II", data, 0x910, resource_name, 0x80000020)
    struct.pack_into("<II", data, 0x930, 7, 0x80000040)
    struct.pack_into("<II", data, 0x950, 1033, 0x60)
    struct.pack_into("<IIII", data, 0x960, 0x2180, 4, 0, 0)
    if named_resource:
        _put_u16(data, 0x970, 4)
        data[0x972:0x97A] = "TEST".encode("utf-16le")

    struct.pack_into("<IIHHHH", data, 0xA00, 0x1000, 16, 0x3001, 0x3002, 0x0000, 0x3003)
    callback_address = (0x140003000 if pe_plus else 0x403000) if tls_callbacks else 0
    if pe_plus:
        struct.pack_into(
            "<QQQQII",
            data,
            0xB00,
            0x140001000,
            0x140001100,
            0x140002000,
            callback_address,
            0,
            0,
        )
    else:
        struct.pack_into(
            "<IIIIII",
            data,
            0xB00,
            0x401000,
            0x401100,
            0x402000,
            callback_address,
            0,
            0,
        )
    return bytes(data)


def _write_synthetic_pe(tmp_path: Path, name: str, **kwargs) -> Path:
    sample = tmp_path / name
    sample.write_bytes(_synthetic_pe_bytes(**kwargs))
    return sample


def _python_feature_vectors(sample: Path) -> tuple[np.ndarray, np.ndarray]:
    config = ExtractionConfig(
        max_file_size=8192,
        pe_feature_dim=256,
        pe_schema_version="fixed_v2",
        allow_pe_fallback=False,
    )
    pe_features = PEFeatureExtractor(config=config).extract(str(sample))
    assert pe_features is not None
    return pe_features, extract_content_pe_v1_features(sample)


def _resource_summary(sample: Path) -> tuple[int, set[int | None]]:
    pefile = pytest.importorskip("pefile")
    pe = pefile.PE(str(sample), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]]
        )
        assert hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"), pe.get_warnings()
        stack = list(pe.DIRECTORY_ENTRY_RESOURCE.entries)
        entry_count = 0
        ids: set[int | None] = set()
        while stack:
            entry = stack.pop()
            entry_count += 1
            assert hasattr(entry, "id")
            ids.add(entry.id)
            if hasattr(entry, "directory"):
                stack.extend(entry.directory.entries)
        return entry_count, ids
    finally:
        pe.close()


def _f32_bits(value: float | np.floating) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _legacy_serial_mean_f32(values: np.ndarray) -> np.float32:
    accumulator = np.float32(-0.0)
    for value in np.asarray(values, dtype=np.float32):
        accumulator = np.float32(accumulator + value)
    return np.float32(accumulator / np.float32(len(values)))


def _legacy_serial_std_f32(values: np.ndarray) -> np.float32:
    values = np.asarray(values, dtype=np.float32)
    mean = _legacy_serial_mean_f32(values)
    variance = np.float32(-0.0)
    for value in values:
        difference = np.float32(value - mean)
        variance = np.float32(variance + np.float32(difference * difference))
    variance = np.float32(variance / np.float32(len(values)))
    return np.float32(np.sqrt(variance))


def _legacy_entropy_f64_to_f32(byte_values: np.ndarray) -> np.float32:
    counts = np.bincount(np.asarray(byte_values, dtype=np.uint8), minlength=256)
    total = float(len(byte_values))
    entropy = (
        -sum((int(count) / total) * math.log2(int(count) / total) for count in counts if count)
        / 8.0
    )
    return np.float32(entropy)


def _stat45_reduction_fixture() -> np.ndarray:
    chunk_lengths = [819] * 9 + [821]
    high_counts = [12, 559, 486, 44, 745, 180, 209, 151, 273, 144]
    chunks = [
        np.concatenate(
            [
                np.full(high_count, 255, dtype=np.uint8),
                np.zeros(length - high_count, dtype=np.uint8),
            ]
        )
        for length, high_count in zip(chunk_lengths, high_counts)
    ]
    return np.concatenate(chunks)


def _scalar_entropy_reduction_fixture() -> np.ndarray:
    state = 2
    values = np.empty(8192, dtype=np.uint8)
    for index in range(values.size):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        values[index] = (state >> 24) & 0xFF
    return values


@pytest.mark.parametrize(
    ("named_resource", "expected_ids"),
    [
        (False, {10, 7, 1033}),
        (True, {None, 7, 1033}),
    ],
)
def test_synthetic_resource_tree_is_really_parsed(
    tmp_path: Path,
    named_resource: bool,
    expected_ids: set[int | None],
) -> None:
    sample = _write_synthetic_pe(
        tmp_path,
        f"resource_{'named' if named_resource else 'numeric'}.exe",
        named_resource=named_resource,
    )

    entry_count, ids = _resource_summary(sample)
    _, content_features = _python_feature_vectors(sample)
    columns = {name: index for index, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}

    assert entry_count == 3
    assert ids == expected_ids
    assert content_features[columns["content_resource_entry_count_log"]] == np.float32(
        math.log1p(3)
    )
    assert content_features[columns["content_resource_type_count_log"]] == np.float32(math.log1p(3))


@pytest.mark.parametrize("pe_plus", [False, True])
@pytest.mark.parametrize("tls_callbacks", [False, True])
def test_synthetic_pe_variants_exercise_api_tls_and_relocations(
    tmp_path: Path,
    pe_plus: bool,
    tls_callbacks: bool,
) -> None:
    sample = _write_synthetic_pe(
        tmp_path,
        f"pe{'64' if pe_plus else '32'}_tls_{int(tls_callbacks)}.exe",
        pe_plus=pe_plus,
        tls_callbacks=tls_callbacks,
    )
    pe_features, content_features = _python_feature_vectors(sample)
    pe_columns = {name: index for index, name in enumerate(fixed_v2_feature_names(32, 256))}
    content_columns = {name: index for index, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}

    assert pe_features[pe_columns["fixed_v2_api_network_ratio"]] == np.float32(0.5)
    assert pe_features[pe_columns["fixed_v2_has_tls"]] == np.float32(1.0)
    assert content_features[content_columns["content_tls_callback_count_log"]] == np.float32(
        math.log1p(1) if tls_callbacks else 0.0
    )
    assert content_features[content_columns["content_reloc_block_count_log"]] == np.float32(
        math.log1p(1)
    )
    assert content_features[content_columns["content_reloc_entry_count_log"]] == np.float32(
        math.log1p(4)
    )


def test_synthetic_fixture_is_mutation_sensitive_for_localized_pe_indices(
    tmp_path: Path,
) -> None:
    pefile = pytest.importorskip("pefile")
    sample = _write_synthetic_pe(tmp_path, "pe_numeric_precision.exe")
    pe_features, _ = _python_feature_vectors(sample)
    pe = pefile.PE(str(sample), fast_load=True)
    try:
        section_entropies = []
        section_sizes = []
        for section in pe.sections:
            raw_size = int(section.SizeOfRawData)
            section_data = section.get_data(length=min(raw_size, 256))
            section_entropies.append(
                calculate_byte_entropy(np.frombuffer(section_data, dtype=np.uint8))
            )
            section_sizes.append(raw_size)
    finally:
        pe.close()

    legacy_entropy_std = _legacy_serial_std_f32(np.asarray(section_entropies))
    legacy_size_std = _legacy_serial_std_f32(np.asarray(section_sizes))
    legacy_size_mean = _legacy_serial_mean_f32(np.asarray(section_sizes))
    legacy_size_cv = np.float32(legacy_size_std / max(legacy_size_mean, np.float32(1.0)))

    assert _f32_bits(legacy_entropy_std) != _f32_bits(pe_features[117])
    assert _f32_bits(legacy_size_std) != _f32_bits(pe_features[125])
    assert _f32_bits(legacy_size_cv) != _f32_bits(pe_features[126])
    assert pe_features[135] == np.float32(0.5)


def test_deterministic_numeric_fixtures_trigger_stat45_and_scalar_entropy() -> None:
    from src.predict_api import _byte_summary_features

    stat_bytes = _stat45_reduction_fixture()
    stat_features = extract_statistical_features(
        stat_bytes,
        len(stat_bytes),
        config=ExtractionConfig(stat_segment_count=3, stat_chunk_count=10),
    )
    std_differences = np.abs(np.diff(stat_features[31:41].astype(np.float32)))
    legacy_stat45 = _legacy_serial_mean_f32(std_differences)

    assert _f32_bits(legacy_stat45) != _f32_bits(stat_features[45])
    assert _f32_bits(stat_features[45]) == _f32_bits(np.mean(std_differences))

    entropy_bytes = _scalar_entropy_reduction_fixture()
    byte_summary = np.asarray(
        _byte_summary_features(entropy_bytes, prefix_len=256, chunk_count=16),
        dtype=np.float32,
    )
    legacy_entropy = _legacy_entropy_f64_to_f32(entropy_bytes)
    legacy_chunk_entropy = np.asarray(
        [_legacy_entropy_f64_to_f32(chunk) for chunk in np.array_split(entropy_bytes, 16)],
        dtype=np.float32,
    )
    current_chunk_entropy = byte_summary[768 + 2 : 848 : 5]
    chunk_mismatches = np.flatnonzero(
        legacy_chunk_entropy.view(np.uint32) != current_chunk_entropy.view(np.uint32)
    )

    assert _f32_bits(legacy_entropy) != _f32_bits(byte_summary[848])
    assert chunk_mismatches.tolist() == [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]


def test_primary_fixture_triggers_float32_log_path() -> None:
    from src.predict_api import _byte_summary_features

    byte_values = np.frombuffer(_synthetic_pe_bytes(), dtype=np.uint8)
    byte_summary = np.asarray(
        _byte_summary_features(byte_values, prefix_len=256, chunk_count=16),
        dtype=np.float32,
    )
    counts = np.bincount(byte_values, minlength=256)
    legacy_log_hist = np.asarray(
        [math.log1p(int(count)) / math.log1p(float(byte_values.size)) for count in counts],
        dtype=np.float32,
    )
    log_mismatches = np.flatnonzero(
        legacy_log_hist.view(np.uint32) != byte_summary[256:512].view(np.uint32)
    )

    assert log_mismatches.tolist() == [0, 16, 32]


def _invalidate_content_directory_rvas(*, pe_plus: bool = False) -> bytes:
    data = bytearray(_synthetic_pe_bytes(pe_plus=pe_plus))
    optional = 0x84 + 20
    data_directory = optional + (112 if pe_plus else 96)
    for index in (2, 5, 9):
        _put_u32(data, data_directory + index * 8, 0x70000000 + index * 0x1000)
    return bytes(data)


def _limit_declared_data_directories(*, pe_plus: bool = False) -> bytes:
    data = bytearray(_synthetic_pe_bytes(pe_plus=pe_plus))
    optional = 0x84 + 20
    _put_u32(data, optional + (108 if pe_plus else 92), 2)
    return bytes(data)


def test_invalid_directory_rva_fixture_is_rejected_by_python_reference(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "invalid_directory_rvas.exe"
    sample.write_bytes(_invalidate_content_directory_rvas())
    pe_features, content_features = _python_feature_vectors(sample)
    pe_columns = {name: index for index, name in enumerate(fixed_v2_feature_names(32, 256))}
    content_columns = {name: index for index, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}

    assert pe_features[pe_columns["fixed_v2_has_relocs"]] == np.float32(0.0)
    assert pe_features[pe_columns["fixed_v2_has_tls"]] == np.float32(0.0)
    for name in (
        "content_resource_entry_count_log",
        "content_resource_type_count_log",
        "content_tls_callback_count_log",
        "content_reloc_block_count_log",
        "content_reloc_entry_count_log",
    ):
        assert content_features[content_columns[name]] == np.float32(0.0)


@pytest.mark.parametrize("pe_plus", [False, True])
def test_declared_directory_count_blocks_ghost_directories(
    tmp_path: Path,
    pe_plus: bool,
) -> None:
    sample = tmp_path / f"declared_directories_pe{64 if pe_plus else 32}.exe"
    sample.write_bytes(_limit_declared_data_directories(pe_plus=pe_plus))
    pe_features, content_features = _python_feature_vectors(sample)
    pe_columns = {name: index for index, name in enumerate(fixed_v2_feature_names(32, 256))}
    content_columns = {name: index for index, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}

    assert pe_features[pe_columns["fixed_v2_has_relocs"]] == np.float32(0.0)
    assert pe_features[pe_columns["fixed_v2_has_tls"]] == np.float32(0.0)
    for name in (
        "content_resource_entry_count_log",
        "content_resource_type_count_log",
        "content_tls_callback_count_log",
        "content_reloc_block_count_log",
        "content_reloc_entry_count_log",
    ):
        assert content_features[content_columns[name]] == np.float32(0.0)


def test_truncated_relocation_fixture_documents_pefile_partial_parse() -> None:
    pefile = pytest.importorskip("pefile")
    truncated = _synthetic_pe_bytes()[:0xA08]
    pe = pefile.PE(data=truncated, fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_BASERELOC"]]
        )
        assert hasattr(pe, "DIRECTORY_ENTRY_BASERELOC")
        assert len(pe.DIRECTORY_ENTRY_BASERELOC) == 1
        assert len(pe.DIRECTORY_ENTRY_BASERELOC[0].entries) == 0
    finally:
        pe.close()


def _native_fixed_pe_features(sample: Path) -> np.ndarray:
    dll_path = (
        ROOT / "tools" / "axon_onnx_dll" / "build" / "bin" / "Release" / "axon_onnx_predict.dll"
    )
    if not dll_path.is_file():
        pytest.skip(f"native DLL is not built: {dll_path}")
    library = ctypes.WinDLL(str(dll_path))
    extract = library.kvd_extract_pe_features
    extract.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    extract.restype = ctypes.c_int
    output = (ctypes.c_float * 256)()
    result = extract(os.fsencode(sample), output, len(output))
    assert result == 0
    return np.ctypeslib.as_array(output).copy()


@pytest.mark.skipif(
    os.name != "nt" or os.environ.get("AXON_RUN_NATIVE_FEATURE_INTEGRATION") != "1",
    reason="native feature-only integration is opt-in and requires Windows",
)
@pytest.mark.parametrize("pe_plus", [False, True])
def test_native_fixed_pe_features_match_valid_synthetic_variants(
    tmp_path: Path,
    pe_plus: bool,
) -> None:
    sample = _write_synthetic_pe(
        tmp_path,
        f"native_pe{'64' if pe_plus else '32'}.exe",
        pe_plus=pe_plus,
    )
    python_features, _ = _python_feature_vectors(sample)
    native_features = _native_fixed_pe_features(sample)

    np.testing.assert_array_equal(native_features, python_features)


@pytest.mark.skipif(
    os.name != "nt" or os.environ.get("AXON_RUN_NATIVE_FEATURE_INTEGRATION") != "1",
    reason="native feature-only integration is opt-in and requires Windows",
)
def test_native_invalid_rvas_and_truncated_relocation_fail_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "native_invalid_rvas.exe"
    invalid.write_bytes(_invalidate_content_directory_rvas())
    invalid_features = _native_fixed_pe_features(invalid)

    truncated = tmp_path / "native_truncated_reloc.exe"
    truncated.write_bytes(_synthetic_pe_bytes()[:0xA08])
    truncated_features = _native_fixed_pe_features(truncated)

    assert invalid_features[13] == np.float32(0.0)
    assert invalid_features[14] == np.float32(0.0)
    assert truncated_features[13] == np.float32(0.0)


@pytest.mark.skipif(
    os.name != "nt" or os.environ.get("AXON_RUN_NATIVE_FEATURE_INTEGRATION") != "1",
    reason="native feature-only integration is opt-in and requires Windows",
)
@pytest.mark.parametrize("pe_plus", [False, True])
def test_native_respects_declared_directory_count(tmp_path: Path, pe_plus: bool) -> None:
    sample = tmp_path / f"native_declared_directories_pe{64 if pe_plus else 32}.exe"
    sample.write_bytes(_limit_declared_data_directories(pe_plus=pe_plus))

    python_features, _ = _python_feature_vectors(sample)
    native_features = _native_fixed_pe_features(sample)

    np.testing.assert_array_equal(native_features, python_features)


def _assert_trace_exercises_content_semantics(
    trace,
    *,
    tls_callbacks: bool,
) -> None:
    pe_features = trace.components["pe_features"]
    stage2_features = trace.components["stage2_features"]

    assert pe_features[135] == np.float32(0.5)
    assert stage2_features[1496] == np.float32(math.log1p(3))
    assert stage2_features[1497] == np.float32(math.log1p(3))
    assert stage2_features[1498] == np.float32(math.log1p(1) if tls_callbacks else 0.0)
    assert stage2_features[1499] == np.float32(math.log1p(1))
    assert stage2_features[1500] == np.float32(math.log1p(4))


@pytest.mark.skipif(
    os.environ.get("AXON_RUN_NATIVE_PARITY_INTEGRATION") != "1",
    reason="native model integration is opt-in",
)
def test_native_synthetic_feature_parity(tmp_path: Path) -> None:
    import diagnose_loop28_parity as diagnostic

    sample = tmp_path / "synthetic_parity.exe"
    sample.write_bytes(_synthetic_pe_bytes())
    trace = diagnostic.build_python_trace(
        project_root=ROOT,
        sample_path=sample,
        checkpoint_path=ROOT / diagnostic.DEFAULT_CHECKPOINT,
        stage2_path=ROOT / diagnostic.DEFAULT_PYTHON_STAGE2,
    )
    _assert_trace_exercises_content_semantics(trace, tls_callbacks=True)
    output_sizes: list[int] = []

    def native_runner(key: bytes, component: str | None, block_elements: int | None):
        return diagnostic.run_native_diagnostics(
            sample_path=sample,
            allowed_raw_root=tmp_path,
            selftest_path=ROOT / diagnostic.DEFAULT_NATIVE_SELFTEST,
            dll_path=ROOT / diagnostic.DEFAULT_NATIVE_DLL,
            onnx_path=ROOT / diagnostic.DEFAULT_NATIVE_ONNX,
            stage2_path=ROOT / diagnostic.DEFAULT_NATIVE_STAGE2,
            key=key,
            timeout_seconds=120,
            max_output_bytes=64 * 1024 * 1024,
            output_sizes=output_sizes,
            component=component,
            block_elements=block_elements,
        )

    result = diagnostic.diagnose_trace(trace, native_runner=native_runner)
    by_name = {row["name"]: row for row in result["component_results"]}
    assert by_name["byte_seq"]["whole_match"]
    assert by_name["pe_features"]["whole_match"]
    assert by_name["stat_features"]["whole_match"]
    assert all(index < 6 for index in by_name["stage2_features"].get("mismatch_indices", []))
    assert result["predictions"]["base_decision_match"]
    assert result["stage2_inference"]["decision_match"]
    deltas = {
        "base": result["predictions"]["absolute_probability_deltas"]["base_prob_malicious"],
        "stage2": result["stage2_inference"]["absolute_probability_delta"],
        "stage2_mismatches": by_name["stage2_features"].get("mismatch_indices", []),
    }
    print(json.dumps(deltas, sort_keys=True))
    assert deltas["base"] <= 1.0e-6 and deltas["stage2"] <= 1.0e-6, deltas


@pytest.mark.skipif(
    os.environ.get("AXON_RUN_NATIVE_PARITY_VARIANTS") != "1",
    reason="native model parity variants are opt-in",
)
@pytest.mark.parametrize(
    ("pe_plus", "named_resource", "tls_callbacks"),
    [
        (False, True, True),
        (False, False, False),
        (True, True, False),
    ],
)
def test_native_synthetic_feature_parity_variants(
    tmp_path: Path,
    pe_plus: bool,
    named_resource: bool,
    tls_callbacks: bool,
) -> None:
    import diagnose_loop28_parity as diagnostic

    sample = _write_synthetic_pe(
        tmp_path,
        f"variant_{int(pe_plus)}_{int(named_resource)}_{int(tls_callbacks)}.exe",
        pe_plus=pe_plus,
        named_resource=named_resource,
        tls_callbacks=tls_callbacks,
    )
    trace = diagnostic.build_python_trace(
        project_root=ROOT,
        sample_path=sample,
        checkpoint_path=ROOT / diagnostic.DEFAULT_CHECKPOINT,
        stage2_path=ROOT / diagnostic.DEFAULT_PYTHON_STAGE2,
    )
    _assert_trace_exercises_content_semantics(trace, tls_callbacks=tls_callbacks)
    output_sizes: list[int] = []

    def native_runner(key: bytes, component: str | None, block_elements: int | None):
        return diagnostic.run_native_diagnostics(
            sample_path=sample,
            allowed_raw_root=tmp_path,
            selftest_path=ROOT / diagnostic.DEFAULT_NATIVE_SELFTEST,
            dll_path=ROOT / diagnostic.DEFAULT_NATIVE_DLL,
            onnx_path=ROOT / diagnostic.DEFAULT_NATIVE_ONNX,
            stage2_path=ROOT / diagnostic.DEFAULT_NATIVE_STAGE2,
            key=key,
            timeout_seconds=120,
            max_output_bytes=64 * 1024 * 1024,
            output_sizes=output_sizes,
            component=component,
            block_elements=block_elements,
        )

    result = diagnostic.diagnose_trace(trace, native_runner=native_runner)
    by_name = {row["name"]: row for row in result["component_results"]}

    assert by_name["byte_seq"]["whole_match"]
    assert by_name["pe_features"]["whole_match"]
    assert by_name["stat_features"]["whole_match"]
    stage2_mismatches = by_name["stage2_features"].get("mismatch_indices", [])
    assert all(index < 6 for index in stage2_mismatches)
    assert result["predictions"]["base_decision_match"]
    assert result["stage2_inference"]["decision_match"]
    deltas = {
        "pe_plus": pe_plus,
        "named_resource": named_resource,
        "tls_callbacks": tls_callbacks,
        "base": result["predictions"]["absolute_probability_deltas"]["base_prob_malicious"],
        "stage2": result["stage2_inference"]["absolute_probability_delta"],
        "stage2_mismatches": stage2_mismatches,
    }
    print(json.dumps(deltas, sort_keys=True))
    assert deltas["base"] <= 1.0e-6 and deltas["stage2"] <= 1.0e-6, deltas
