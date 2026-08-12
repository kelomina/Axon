#!/usr/bin/env python3
"""Guest-only Loop171 capa runner; never run this on the Windows host."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loop171.guest_capa import (  # noqa: E402
    GuestCapaError,
    GuestCapaOutputLimitError,
    GuestCapaTimeoutError,
    _write_new_receipt,
    install_linux_capa_zip,
    receipt_payload,
    run_guest_capa,
)


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one aggregate-only Linux capa guest invocation.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--max-input-bytes", type=int, required=True)
    parser.add_argument("--capa", type=Path, required=True)
    parser.add_argument("--capa-sha256", required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--rules-sha256", required=True)
    parser.add_argument("--toolchain-archive", type=Path, required=True)
    parser.add_argument("--toolchain-archive-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mountinfo", type=Path, default=Path("/proc/self/mountinfo"))
    return parser


def _install_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a SHA-bound Linux capa archive inside the future guest.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mountinfo", type=Path, default=Path("/proc/self/mountinfo"))
    return parser


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", parents=[_run_parser()], add_help=False)
    commands.add_parser("install-toolchain", parents=[_install_parser()], add_help=False)
    args = parser.parse_args()
    if args.command == "install-toolchain":
        try:
            payload = install_linux_capa_zip(
                archive_path=args.archive,
                archive_sha256=args.archive_sha256,
                destination=args.destination,
                mountinfo_path=args.mountinfo,
            )
            _write_new_receipt(args.receipt, payload)
        except GuestCapaError as error:
            _write_new_receipt(args.receipt, receipt_payload(None, failure=str(error)))
            raise SystemExit(2) from None
        return
    try:
        result = run_guest_capa(
            source=args.source,
            source_sha256=args.source_sha256,
            expected_size=args.expected_size,
            max_input_bytes=args.max_input_bytes,
            capa=args.capa,
            capa_sha256=args.capa_sha256,
            rules=args.rules,
            rules_sha256=args.rules_sha256,
            toolchain_archive=args.toolchain_archive,
            toolchain_archive_sha256=args.toolchain_archive_sha256,
            timeout_seconds=args.timeout_seconds,
            mountinfo_path=args.mountinfo,
        )
        _write_new_receipt(args.receipt, receipt_payload(result))
    except GuestCapaError as error:
        _write_new_receipt(args.receipt, receipt_payload(None, failure=str(error)))
        if isinstance(error, GuestCapaTimeoutError):
            raise SystemExit(3) from None
        if isinstance(error, GuestCapaOutputLimitError):
            raise SystemExit(4) from None
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
