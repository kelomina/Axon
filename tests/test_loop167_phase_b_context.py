from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.loop167_phase_b.b0_projector import extract_b0_projection
from src.loop167_phase_b.ember_controls import extract_context_features
from src.loop167_phase_b.raw_context import RawFeatureContext


class _FakeDosHeader:
    def dump_dict(self) -> dict[str, dict[str, int]]:
        return {
            "e_magic": {"Value": 0x5A4D},
            "e_cblp": {"Value": 144},
            "e_cp": {"Value": 3},
            "e_crlc": {"Value": 0},
            "e_cparhdr": {"Value": 4},
            "e_minalloc": {"Value": 0},
            "e_maxalloc": {"Value": 0xFFFF},
            "e_ss": {"Value": 0},
            "e_sp": {"Value": 184},
            "e_csum": {"Value": 0},
            "e_ip": {"Value": 0},
            "e_cs": {"Value": 0},
            "e_lfarlc": {"Value": 64},
            "e_ovno": {"Value": 0},
            "e_oemid": {"Value": 0},
            "e_oeminfo": {"Value": 0},
            "e_lfanew": {"Value": 128},
        }


class _FakeFileHeader:
    TimeDateStamp = 123
    NumberOfSections = 1
    NumberOfSymbols = 7
    SizeOfOptionalHeader = 240
    PointerToSymbolTable = 288
    Machine = 0x14C
    Characteristics = 0x2002


class _FakeOptionalHeader:
    Subsystem = 3
    MajorImageVersion = 5
    MinorImageVersion = 6
    MajorLinkerVersion = 14
    MinorLinkerVersion = 1
    MajorOperatingSystemVersion = 10
    MinorOperatingSystemVersion = 0
    MajorSubsystemVersion = 11
    MinorSubsystemVersion = 0
    SizeOfCode = 1024
    SizeOfHeaders = 512
    SizeOfImage = 4096
    SizeOfInitializedData = 256
    SizeOfUninitializedData = 0
    SizeOfStackReserve = 4096
    SizeOfStackCommit = 1024
    SizeOfHeapReserve = 8192
    SizeOfHeapCommit = 2048
    AddressOfEntryPoint = 4096
    BaseOfCode = 512
    ImageBase = 0x400000
    SectionAlignment = 4096
    CheckSum = 0
    NumberOfRvaAndSizes = 16
    DllCharacteristics = 0x140

    def __init__(self) -> None:
        self.DATA_DIRECTORY = [
            type(
                "Directory",
                (),
                {
                    "name": f"IMAGE_DIRECTORY_ENTRY_{name}",
                    "Size": index + 1,
                    "VirtualAddress": 100 + index,
                },
            )()
            for index, name in enumerate(
                (
                    "EXPORT",
                    "IMPORT",
                    "RESOURCE",
                    "EXCEPTION",
                    "SECURITY",
                    "BASERELOC",
                    "DEBUG",
                    "COPYRIGHT",
                    "GLOBALPTR",
                    "TLS",
                    "LOAD_CONFIG",
                    "BOUND_IMPORT",
                    "IAT",
                    "DELAY_IMPORT",
                    "COM_DESCRIPTOR",
                    "RESERVED",
                )
            )
        ]


class _FakeSection:
    Name = b".text\x00"
    SizeOfRawData = 16
    Misc_VirtualSize = 16
    Characteristics = 0x60000000
    PointerToRawData = 0

    def get_entropy(self) -> float:
        raise AssertionError("bounded Context extraction must not call section.get_entropy")

    def get_data(self) -> bytes:
        raise AssertionError("bounded Context extraction must not call section.get_data")


class _FakeImport:
    name = b"CreateFileA"
    ordinal = None


class _FakeImportEntry:
    dll = b"KERNEL32.dll"
    imports = [_FakeImport()]


class _FakeExportSymbol:
    name = b"ExportedThing"
    ordinal = 1


class _FakeExportDirectory:
    symbols = [_FakeExportSymbol()]


class _FakeRichHeader:
    values = [1, 2, 3, 4]


class _FakePe:
    def __init__(self) -> None:
        self.FILE_HEADER = _FakeFileHeader()
        self.OPTIONAL_HEADER = _FakeOptionalHeader()
        self.DOS_HEADER = _FakeDosHeader()
        self.sections = [_FakeSection()]
        self.DIRECTORY_ENTRY_IMPORT = [_FakeImportEntry()]
        self.DIRECTORY_ENTRY_EXPORT = _FakeExportDirectory()
        self.RICH_HEADER = _FakeRichHeader()
        self.directory_parse_calls = 0
        self.closed = 0

    def parse_data_directories(self, **_: object) -> None:
        self.directory_parse_calls += 1

    def get_overlay(self) -> bytes:
        raise AssertionError("bounded Context extraction must not call pe.get_overlay")

    def get_overlay_data_start_offset(self) -> int:
        return 4

    def has_relocs(self) -> bool:
        return True

    def has_dynamic_relocs(self) -> bool:
        return False

    def close(self) -> None:
        self.closed += 1


class _DirectoryFailurePe(_FakePe):
    def __init__(self) -> None:
        super().__init__()
        self.OPTIONAL_HEADER.DATA_DIRECTORY[4].VirtualAddress = 0
        self.OPTIONAL_HEADER.DATA_DIRECTORY[4].Size = 0

    def parse_data_directories(self, **_: object) -> None:
        self.directory_parse_calls += 1
        raise ValueError("synthetic directory failure")


class _CloseFailurePe(_FakePe):
    def close(self) -> None:
        super().close()
        raise RuntimeError("synthetic close failure")


def test_context_projects_controls_and_novel_values_without_a_second_parse() -> None:
    calls: list[_FakePe] = []

    def factory(**_: object) -> _FakePe:
        parsed = _FakePe()
        calls.append(parsed)
        return parsed

    context = RawFeatureContext.from_bytes(
        b"MZ\x90\x00synthetic", maximum_input_bytes=1024, pe_factory=factory
    )
    assert len(calls) == 1
    assert context.pe_parse_attempts == 1
    assert context.directory_parse_attempts == 1
    assert calls[0].directory_parse_calls == 1

    features = extract_context_features(context)
    assert len(calls) == 1
    assert features.controls.values.shape == (536,)
    assert features.controls.missing_indicator_names == (
        "missing_b1_byte_context",
        "missing_b1_pe_context",
        "missing_b1_directory_context",
        "missing_b1_authenticode",
    )
    assert features.controls.missing_indicators.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert features.controls.sampling_indicators.shape == (3,)
    assert features.novel.values.shape == (292,)
    assert features.novel.missing_indicators.tolist() == [0.0]
    assert np.isfinite(features.controls.values).all()
    assert np.isfinite(features.novel.values).all()
    assert 263 not in features.controls.original_indices
    assert 2439 not in features.controls.original_indices
    assert 2435 not in features.controls.original_indices
    assert set(features.controls.original_indices).isdisjoint(features.novel.original_indices)

    novel_by_index = dict(zip(features.novel.original_indices, features.novel.values, strict=True))
    assert [novel_by_index[index] for index in range(3, 7)] == [77.0, 90.0, 144.0, 0.0]
    assert novel_by_index[698] == 7.0
    assert novel_by_index[2439] == 2.0
    assert "authenticode:certificate_directory_out_of_bounds" in features.controls.missing_reasons
    assert features.controls.complete is False

    context.close()
    assert calls[0].closed == 1
    assert context.bytez == b""
    assert context.pe is None
    with pytest.raises(RuntimeError, match="closed"):
        extract_context_features(context)


def test_context_parse_failure_zero_fills_and_preserves_reasons() -> None:
    context = RawFeatureContext.from_bytes(
        b"not-a-pe", maximum_input_bytes=1024, pe_factory=lambda **_: (_ for _ in ()).throw(ValueError())
    )
    features = extract_context_features(context)
    assert context.pe_parse_attempts == 1
    assert context.directory_parse_attempts == 0
    assert context.missing_reasons == ("pe_parse_failure",)
    assert "pe_parse_failure" in features.controls.missing_reasons
    assert "pe_parse_failure" in features.novel.missing_reasons
    assert features.controls.complete is False
    assert features.novel.complete is False
    assert features.controls.missing_indicators.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert features.novel.missing_indicators.tolist() == [1.0]
    assert np.isfinite(features.controls.values).all()
    assert np.isfinite(features.novel.values).all()

    b0 = extract_b0_projection(context)
    assert b0.values.shape == (571,)
    assert b0.missing_indicators.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert "fixed_v2:pe_parse_failure" in b0.missing_reasons


def test_b0_projection_keeps_frozen_name_order_and_family_indicators() -> None:
    context = RawFeatureContext.from_bytes(
        b"MZ\x90\x00synthetic", maximum_input_bytes=1024, pe_factory=lambda **_: _FakePe()
    )
    b0 = extract_b0_projection(context)
    assert b0.values.shape == (571,)
    assert b0.missing_indicators.shape == (6,)
    assert b0.missing_indicator_names == (
        "missing_fixed_v2",
        "missing_stat",
        "missing_content_pe_v1",
        "missing_content_pe_v2",
        "missing_content_string",
        "missing_content_cert",
    )
    assert "fixed_v2.fixed_v2_log_size" in b0.feature_names
    assert "content_pe_v1.content_file_log_size" not in b0.feature_names
    assert len(set(b0.feature_names)) == 571
    assert np.isfinite(b0.values).all()
    assert np.isfinite(b0.missing_indicators).all()


def test_raw_context_has_no_path_or_reopen_api_surface() -> None:
    signature = inspect.signature(RawFeatureContext.from_bytes)
    assert tuple(signature.parameters) == ("bytez", "maximum_input_bytes", "pe_factory")
    source = inspect.getsource(RawFeatureContext)
    for forbidden in ("Path", ".open(", "label", "score", "source_sha256", "row_index"):
        assert forbidden not in source


def test_oversize_context_skips_parse_and_releases_input_bytes() -> None:
    calls = 0

    def factory(**_: object) -> _FakePe:
        nonlocal calls
        calls += 1
        return _FakePe()

    context = RawFeatureContext.from_bytes(b"012345", maximum_input_bytes=5, pe_factory=factory)
    assert calls == 0
    assert context.source_length == 6
    assert context.bytez == b""
    assert context.missing_reasons == ("oversize_input",)
    features = extract_context_features(context)
    assert np.all(features.controls.values == 0.0)
    assert np.all(features.novel.values == 0.0)
    assert features.controls.missing_indicators.tolist() == [1.0, 1.0, 0.0, 1.0]


def test_directory_failure_preserves_compatible_b0_partial_values_and_marks_b1_context() -> None:
    context = RawFeatureContext.from_bytes(
        b"MZ" + b"\x00" * 256,
        maximum_input_bytes=1024,
        pe_factory=lambda **_: _DirectoryFailurePe(),
    )
    b0 = extract_b0_projection(context)
    features = extract_context_features(context)

    assert context.directory_parse_reason == "directory_parse_failure"
    assert b0.missing_indicators.tolist() == [0.0] * 6
    assert "directory_parse_failure" in b0.missing_reasons
    assert features.controls.missing_indicators.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert features.controls.complete is False
    assert features.novel.complete is True


def test_b1_string_sampling_has_frozen_provenance() -> None:
    sample_bytes = 256 * 1024 + 64 * 1024 + 1
    context = RawFeatureContext.from_bytes(
        b"A" * sample_bytes,
        maximum_input_bytes=sample_bytes + 1,
        pe_factory=lambda **_: _FakePe(),
    )
    features = extract_context_features(context)

    assert features.controls.sampling_indicator_names == (
        "b1_string_sampled_to_native_cap",
        "b1_string_candidate_cap_reached",
        "b1_section_or_overlay_entropy_sampled",
    )
    assert features.controls.sampling_indicators.tolist() == [1.0, 0.0, 1.0]
    assert "b1_string_sampled_to_native_cap" in features.controls.sampling_reasons


def test_context_releases_bytes_when_parser_close_fails() -> None:
    parsed: list[_CloseFailurePe] = []

    def factory(**_: object) -> _CloseFailurePe:
        value = _CloseFailurePe()
        parsed.append(value)
        return value

    context = RawFeatureContext.from_bytes(b"MZsynthetic", maximum_input_bytes=1024, pe_factory=factory)
    with pytest.raises(RuntimeError, match="synthetic close failure"):
        context.close()

    assert parsed[0].closed == 1
    assert context.bytez == b""
    assert context.pe is None
    with pytest.raises(RuntimeError, match="closed"):
        context.require_open()
