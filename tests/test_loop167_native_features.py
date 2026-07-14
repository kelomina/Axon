from __future__ import annotations

import inspect

import numpy as np

from src.loop167 import ember_v3_native
from src.loop167.ember_v3_native import extract_novel_delta


class _FakeDosHeader:
    def __init__(self) -> None:
        self._values = {
            "e_magic": 0x5A4D,
            "e_cblp": 144,
            "e_cp": 3,
            "e_crlc": 0,
            "e_cparhdr": 4,
            "e_minalloc": 0,
            "e_maxalloc": 0xFFFF,
            "e_ss": 0,
            "e_sp": 184,
            "e_csum": 0,
            "e_ip": 0,
            "e_cs": 0,
            "e_lfarlc": 64,
            "e_ovno": 0,
            "e_oemid": 0,
            "e_oeminfo": 0,
            "e_lfanew": 128,
        }

    def dump_dict(self) -> dict[str, dict[str, int]]:
        return {name: {"Value": value} for name, value in self._values.items()}


class _FakeHeader:
    NumberOfSymbols = 7
    PointerToSymbolTable = 288


class _FakeOptionalHeader:
    MajorImageVersion = 5
    MinorImageVersion = 6
    MajorOperatingSystemVersion = 10
    MinorOperatingSystemVersion = 1
    MajorSubsystemVersion = 11
    MinorSubsystemVersion = 2
    SizeOfStackReserve = 4096
    SizeOfStackCommit = 1024
    SizeOfHeapReserve = 8192
    SizeOfHeapCommit = 2048
    BaseOfCode = 512
    NumberOfRvaAndSizes = 16


class _FakeRichHeader:
    values = [1, 2, 3, 4, 5, 6]


class _FakePe:
    def __init__(self) -> None:
        self.FILE_HEADER = _FakeHeader()
        self.OPTIONAL_HEADER = _FakeOptionalHeader()
        self.DOS_HEADER = _FakeDosHeader()
        self.RICH_HEADER = _FakeRichHeader()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fake_factory(**_: object) -> _FakePe:
    return _FakePe()


def _values_by_original_index(result) -> dict[int, float]:
    return dict(zip(result.original_indices, result.values.tolist(), strict=True))


def test_native_vector_is_source_ordered_and_header_values_are_at_true_offsets() -> None:
    result = extract_novel_delta(b"\x11\x22\x33\x44", pe_factory=_fake_factory)
    values = _values_by_original_index(result)
    assert result.values.shape == (292,)
    assert result.pe_parse_succeeded is True
    assert result.missing_reasons == ()
    assert [values[index] for index in range(3, 7)] == [17.0, 34.0, 51.0, 68.0]
    assert values[698] == 7.0
    assert values[700] == 288.0
    assert values[703] == 5.0
    assert values[716] == 4096.0
    assert values[721] == 512.0
    assert values[725] == 16.0
    assert values[753] == float(0x5A4D)
    assert values[769] == 128.0
    assert values[2439] == 3.0
    assert np.isfinite(result.values).all()


def test_native_byteentropy_uses_pinned_nonempty_window_semantics() -> None:
    result = extract_novel_delta(bytes(range(256)) * 8, pe_factory=_fake_factory)
    values = _values_by_original_index(result)
    entropy_values = np.asarray([values[index] for index in range(263, 519)], dtype=np.float32).reshape(16, 16)
    assert np.count_nonzero(entropy_values[:15]) == 0
    assert np.allclose(entropy_values[15], np.full(16, 1.0 / 16.0, dtype=np.float32))


def test_native_boundary_and_missing_contracts_are_finite() -> None:
    for length in (1, 1023, 1024, 2047, 2048, 2049):
        result = extract_novel_delta(bytes(index % 251 for index in range(length)), pe_factory=_fake_factory)
        assert result.values.shape == (292,)
        assert np.isfinite(result.values).all()

    empty = extract_novel_delta(b"", pe_factory=_fake_factory)
    assert empty.missing_reasons == ("empty_input",)
    assert empty.pe_parse_succeeded is False
    assert np.all(empty.values == 0.0)

    failed = extract_novel_delta(b"not-a-pe", pe_factory=lambda **_: (_ for _ in ()).throw(ValueError()))
    assert failed.missing_reasons == ("pe_parse_failure",)
    assert failed.pe_parse_succeeded is False
    assert np.isfinite(failed.values).all()
    failed_values = _values_by_original_index(failed)
    assert np.isclose(sum(failed_values[index] for index in range(263, 519)), 1.0)


def test_native_api_has_no_path_identity_label_or_score_surface() -> None:
    signature = inspect.signature(extract_novel_delta)
    assert tuple(signature.parameters) == ("bytez", "pe_factory")
    source = inspect.getsource(ember_v3_native)
    for forbidden in ("Path", ".open(", "label", "score", "source_sha256", "row_index"):
        assert forbidden not in source
