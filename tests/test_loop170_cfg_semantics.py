from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import capstone

from src.loop170.cfg_semantics import _category, _disassembler, _summarize_spans


def test_x86_disassembly_summary_supports_the_known_machine() -> None:
    disassembler, architecture = _disassembler(0x014C)

    assert disassembler is not None
    assert architecture == "x86"


def test_unsupported_machine_is_rejected() -> None:
    disassembler, architecture = _disassembler(0xFFFF)

    assert disassembler is None
    assert architecture is None


def test_opcode_categories_are_stable() -> None:
    assert _category("movzx") == "data_move"
    assert _category("cmp") == "compare"
    assert _category("ret") == "other"


def test_direct_and_indirect_control_flow_are_not_double_counted() -> None:
    disassembler, _ = _disassembler(0x014C)
    assert disassembler is not None
    disassembler.detail = True
    summary = _summarize_spans(disassembler, (b"\xe8\x00\x00\x00\x00\xff\xd0\xc3",))

    assert summary is not None
    _, _, blocks, calls, direct_branches, indirect, returns, _, _ = summary
    assert calls == 2
    assert direct_branches == 0
    assert indirect == 1
    assert returns == 1
    assert blocks == 2


def test_partial_decode_is_detectable_per_span() -> None:
    disassembler, _ = _disassembler(0xAA64)
    assert disassembler is not None
    disassembler.detail = True

    assert _summarize_spans(disassembler, (b"\x00",)) is None
