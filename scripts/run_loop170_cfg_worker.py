#!/usr/bin/env python3
"""One isolated, aggregate-only CFG extraction worker for Loop170."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import stat
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loop170.cfg_semantics import extract_cfg_semantics  # noqa: E402


class SourceIntegrityError(ValueError):
    """Raised before parsing when the source binding is not exact."""


def _read_verified(path: Path, *, expected_sha256: str, expected_size: int, max_bytes: int) -> bytes:
    if expected_size <= 0 or expected_size > max_bytes:
        raise SourceIntegrityError("declared source size is outside the fixed bound")
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise SourceIntegrityError("source file metadata drifted")
        payload = bytearray(expected_size)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            view = memoryview(payload)
            offset = 0
            while offset < expected_size:
                read = handle.readinto(view[offset:])
                if not isinstance(read, int) or read <= 0:
                    raise SourceIntegrityError("source read truncated")
                offset += read
            if handle.read(1):
                raise SourceIntegrityError("source size drifted while reading")
            after = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise SourceIntegrityError("source changed while reading")
        raw = bytes(payload)
        if hashlib.sha256(raw).hexdigest() != expected_sha256.casefold():
            raise SourceIntegrityError("source SHA-256 drifted")
        return raw
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _peak_rss_bytes() -> int:
    if os.name != "nt":
        return 0

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess  # type: ignore[attr-defined]
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
    get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    get_process_memory_info.restype = ctypes.c_int
    if not get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
        raise RuntimeError("cannot read worker RSS")
    return int(counters.PeakWorkingSetSize)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    args = parser.parse_args()
    try:
        raw = _read_verified(args.source, expected_sha256=args.sha256, expected_size=args.expected_size, max_bytes=args.max_bytes)
    except SourceIntegrityError as error:
        print(json.dumps({"status": "integrity_error", "detail": str(error)}, ensure_ascii=True))
        raise SystemExit(2)
    # 子进程只返回聚合数值；即便原生反汇编器崩溃也不会带走父进程或其它样本。
    print(json.dumps({"status": "ok", "feature": asdict(extract_cfg_semantics(raw)), "peak_rss_bytes": _peak_rss_bytes()}, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
