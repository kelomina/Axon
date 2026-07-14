"""Bounded structural Authenticode availability contract for Loop167 controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .raw_context import RawFeatureContext

AUTHENTICODE_DIMENSION = 8
MAX_CERTIFICATE_BYTES = 4 * 1024 * 1024
SECURITY_DIRECTORY_INDEX = 4


@dataclass(frozen=True)
class AuthenticodeControl:
    values: np.ndarray
    complete: bool
    reason: str | None


def _safe_int(value: object, attribute: str) -> int:
    try:
        return int(getattr(value, attribute, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _security_directory(context: RawFeatureContext) -> object | None:
    directories = list(getattr(getattr(context.pe, "OPTIONAL_HEADER", None), "DATA_DIRECTORY", []) or [])
    return directories[SECURITY_DIRECTORY_INDEX] if len(directories) > SECURITY_DIRECTORY_INDEX else None


def extract_authenticode_control(context: RawFeatureContext) -> AuthenticodeControl:
    """Return a complete zero for unsigned files, otherwise fail closed on CMS fields."""

    values = np.zeros(AUTHENTICODE_DIMENSION, dtype=np.float32)
    if context.parse_reason or context.directory_parse_reason or context.pe is None:
        return AuthenticodeControl(values, False, context.parse_reason or context.directory_parse_reason or "pe_unavailable")
    directory = _security_directory(context)
    offset = _safe_int(directory, "VirtualAddress")
    declared_size = _safe_int(directory, "Size")
    if offset <= 0 or declared_size <= 0:
        return AuthenticodeControl(values, True, None)
    if offset > len(context.bytez) or offset + declared_size > len(context.bytez):
        return AuthenticodeControl(values, False, "certificate_directory_out_of_bounds")
    if declared_size > MAX_CERTIFICATE_BYTES:
        return AuthenticodeControl(values, False, "certificate_blob_exceeds_native_cap")
    cursor = offset
    end = offset + declared_size
    certificate_count = 0
    while cursor < end:
        if end - cursor < 8:
            return AuthenticodeControl(values, False, "win_certificate_header_truncated")
        length = int.from_bytes(context.bytez[cursor : cursor + 4], "little", signed=False)
        if length < 8 or cursor + length > end:
            return AuthenticodeControl(values, False, "win_certificate_length_invalid")
        certificate_count += 1
        cursor = (cursor + length + 7) & ~7
        if cursor > end:
            return AuthenticodeControl(values, False, "win_certificate_alignment_invalid")
    if certificate_count == 0:
        return AuthenticodeControl(values, True, None)
    # The remaining official fields require a pinned CMS parser. Do not invent them.
    return AuthenticodeControl(values, False, "cms_fields_unavailable_without_pinned_parser")
