from __future__ import annotations

import numpy as np

from src.loop167_phase_b.authenticode import extract_authenticode_control
from src.loop167_phase_b.raw_context import RawFeatureContext


class _OptionalHeader:
    def __init__(self, security_offset: int, security_size: int) -> None:
        self.DATA_DIRECTORY = [
            type("Directory", (), {"VirtualAddress": 0, "Size": 0})()
            for _ in range(4)
        ] + [type("Directory", (), {"VirtualAddress": security_offset, "Size": security_size})()]


class _Pe:
    def __init__(self, security_offset: int, security_size: int) -> None:
        self.OPTIONAL_HEADER = _OptionalHeader(security_offset, security_size)

    def parse_data_directories(self, **_: object) -> None:
        return None

    def close(self) -> None:
        return None


def test_unsigned_file_is_a_complete_zero_control() -> None:
    context = RawFeatureContext.from_bytes(
        b"MZsynthetic", maximum_input_bytes=1024, pe_factory=lambda **_: _Pe(0, 0)
    )
    control = extract_authenticode_control(context)
    assert control.complete is True
    assert control.reason is None
    assert control.values.shape == (8,)
    assert np.all(control.values == 0.0)


def test_signed_or_malformed_certificate_never_invents_cms_fields() -> None:
    certificate = (8).to_bytes(4, "little") + b"\x00\x02\x02\x00"
    context = RawFeatureContext.from_bytes(
        b"MZ" + b"\x00" * 6 + certificate,
        maximum_input_bytes=1024,
        pe_factory=lambda **_: _Pe(8, len(certificate)),
    )
    control = extract_authenticode_control(context)
    assert control.complete is False
    assert control.reason == "cms_fields_unavailable_without_pinned_parser"
    assert np.all(control.values == 0.0)

    malformed = RawFeatureContext.from_bytes(
        b"MZ\x01\x00",
        maximum_input_bytes=1024,
        pe_factory=lambda **_: _Pe(2, 2),
    )
    malformed_control = extract_authenticode_control(malformed)
    assert malformed_control.complete is False
    assert malformed_control.reason == "win_certificate_header_truncated"
