"""Fail-closed static control-flow summaries for PE executable sections."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import capstone
import pefile

from loop166.code_sections import plan_executable_spans


MACHINE_32 = 0x014C
MACHINE_64 = 0x8664
MACHINE_ARM64 = 0xAA64
MISSING_REASONS = (
    "parse_failure",
    "no_executable_section",
    "zero_raw_executable_section",
    "invalid_executable_section_span",
    "unsupported_machine",
    "disassembly_failure",
    "partial_decode",
    "worker_timeout",
    "worker_crash",
)


@dataclass(frozen=True)
class CFGSemanticFeatures:
    """Aggregate-only semantic counts; no bytes, strings, identities, or targets."""

    architecture: str
    instruction_count: int
    decoded_byte_count: int
    estimated_block_count: int
    call_count: int
    direct_branch_count: int
    indirect_control_count: int
    return_count: int
    interrupt_count: int
    category_counts: tuple[tuple[str, int], ...]
    missing_reason: Optional[str]

    @property
    def available(self) -> bool:
        return self.missing_reason is None


def _disassembler(machine: int) -> tuple[capstone.Cs, str] | tuple[None, None]:
    if machine == MACHINE_32:
        return capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32), "x86"
    if machine == MACHINE_64:
        return capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64), "x64"
    if machine == MACHINE_ARM64:
        return capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM), "arm64"
    return None, None


def _category(mnemonic: str) -> str:
    value = mnemonic.casefold()
    if value.startswith(("mov", "lea", "ldr", "str")):
        return "data_move"
    if value.startswith(("add", "sub", "mul", "div", "inc", "dec", "adc", "sbb")):
        return "arithmetic"
    if value.startswith(("and", "or", "xor", "not", "neg", "shl", "shr", "rol", "ror")):
        return "bitwise"
    if value.startswith(("cmp", "test", "cmn", "tst")):
        return "compare"
    if value.startswith(("push", "pop", "enter", "leave", "stp", "ldp")):
        return "stack"
    if value.startswith(("sys", "int", "svc", "hvc")):
        return "system"
    return "other"


def _empty(reason: str) -> CFGSemanticFeatures:
    return CFGSemanticFeatures("unknown", 0, 0, 0, 0, 0, 0, 0, 0, (), reason)


def _summarize_spans(disassembler: capstone.Cs, spans: tuple[bytes, ...]) -> tuple[int, int, int, int, int, int, int, int, tuple[tuple[str, int], ...]] | None:
    """Decode each PE span independently so a decoder cannot cross section gaps."""
    categories: Counter[str] = Counter()
    instruction_count = decoded_bytes = calls = direct_branches = indirect = returns = interrupts = boundaries = 0
    for span in spans:
        span_decoded = 0
        for instruction in disassembler.disasm(span, 0):
            instruction_count += 1
            size = int(instruction.size)
            decoded_bytes += size
            span_decoded += size
            categories[_category(str(instruction.mnemonic))] += 1
            groups = set(instruction.groups)
            is_call = capstone.CS_GRP_CALL in groups
            is_return = capstone.CS_GRP_RET in groups
            is_interrupt = capstone.CS_GRP_INT in groups
            is_jump = capstone.CS_GRP_JUMP in groups and not is_call and not is_return
            calls += int(is_call)
            returns += int(is_return)
            interrupts += int(is_interrupt)
            boundaries += int(is_jump or is_return or is_interrupt)
            if is_jump:
                operands = getattr(instruction, "operands", ())
                direct = bool(operands) and getattr(operands[-1], "type", None) == capstone.CS_OP_IMM
                direct_branches += int(direct)
                indirect += int(not direct)
            elif is_call:
                operands = getattr(instruction, "operands", ())
                indirect += int(bool(operands) and getattr(operands[-1], "type", None) != capstone.CS_OP_IMM)
        if span_decoded != len(span):
            return None
    if instruction_count == 0:
        return None
    return (instruction_count, decoded_bytes, 1 + boundaries, calls, direct_branches, indirect, returns, interrupts, tuple(sorted(categories.items())))


def extract_cfg_semantics(bytez: bytes) -> CFGSemanticFeatures:
    """Summarize fixed PE executable spans without retaining code or operand values."""
    parsed = None
    try:
        parsed = pefile.PE(data=bytez, fast_load=True)
        plan = plan_executable_spans(parsed.sections, file_size=len(bytez))
        if plan.missing_reason is not None:
            return _empty(plan.missing_reason)
        disassembler, architecture = _disassembler(int(parsed.FILE_HEADER.Machine))
        if disassembler is None or architecture is None:
            return _empty("unsupported_machine")
        disassembler.detail = True
        summary = _summarize_spans(disassembler, tuple(bytez[start:end] for start, end in plan.spans))
        if summary is None:
            return _empty("partial_decode")
        return CFGSemanticFeatures(architecture, *summary, None)
    except (capstone.CsError, pefile.PEFormatError, AttributeError, TypeError, ValueError):
        return _empty("disassembly_failure")
    finally:
        close = getattr(parsed, "close", None)
        if callable(close):
            close()
