#!/usr/bin/env python3
"""One contained, aggregate-only capa invocation for Loop171."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from loop171.capa_aggregate import CapaAggregateError, aggregate_capa_json  # noqa: E402
from loop166.windows_job import (  # noqa: E402
    WindowsJobError,
    WindowsJobTimeoutError,
    WindowsKillOnCloseJob,
    run_subprocess_in_job,
)
from run_loop170_cfg_worker import SourceIntegrityError, _read_verified  # noqa: E402


CAPA = ROOT / ".cache/loop171_capa/capa-v9.4.0-windows/capa.exe"
RULES = ROOT / ".cache/loop171_capa/capa-rules"
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
PROCESS_MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({"NO_PROXY": "*", "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"})
    return environment


class CapaOutputLimitError(RuntimeError):
    """Raised before capa JSON can grow beyond the fixed temporary-file limit."""


def _read_bounded(path: Path) -> bytes:
    if path.stat().st_size > MAX_OUTPUT_BYTES:
        raise CapaOutputLimitError("capa JSON exceeded the fixed output cap")
    return path.read_bytes()


def _assert_output_within_limit(path: Path) -> None:
    if path.stat().st_size > MAX_OUTPUT_BYTES:
        raise CapaOutputLimitError("capa JSON exceeded the fixed output cap")


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    _write_receipt(args.receipt, {"status": "started"})
    try:
        _read_verified(args.source, expected_sha256=args.sha256, expected_size=args.expected_size, max_bytes=args.max_bytes)
    except SourceIntegrityError as error:
        payload = {"status": "integrity_error", "detail": str(error)}
        _write_receipt(args.receipt, payload)
        print(json.dumps(payload, ensure_ascii=True))
        raise SystemExit(2)
    with tempfile.TemporaryDirectory(prefix="axon-loop171-capa-") as directory:
        stdout_path = Path(directory) / "stdout.json"
        stderr_path = Path(directory) / "stderr.log"
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                result = run_subprocess_in_job(
                    (str(CAPA), "-j", "-r", str(RULES), str(args.source)),
                    cwd=ROOT,
                    env=_environment(),
                    timeout_seconds=args.timeout_seconds,
                    stdout=stdout,
                    stderr=stderr,
                    job_factory=lambda: WindowsKillOnCloseJob(
                        memory_limit_bytes=PROCESS_MEMORY_LIMIT_BYTES
                    ),
                    monitor_callback=lambda: _assert_output_within_limit(stdout_path),
                    monitor_interval_seconds=0.25,
                )
        except WindowsJobTimeoutError as error:
            payload = {"status": "capa_timeout", "tree_termination_confirmed": bool(error.termination.get("tree_termination_confirmed"))}
            _write_receipt(args.receipt, payload)
            print(json.dumps(payload, ensure_ascii=True))
            raise SystemExit(3)
        except CapaOutputLimitError:
            payload = {"status": "capa_output_limit"}
            _write_receipt(args.receipt, payload)
            print(json.dumps(payload, ensure_ascii=True))
            raise SystemExit(4)
        except WindowsJobError:
            payload = {"status": "capa_job_error"}
            _write_receipt(args.receipt, payload)
            print(json.dumps(payload, ensure_ascii=True))
            raise SystemExit(4)
        try:
            _read_verified(args.source, expected_sha256=args.sha256, expected_size=args.expected_size, max_bytes=args.max_bytes)
        except SourceIntegrityError as error:
            payload = {"status": "integrity_error", "detail": str(error)}
            _write_receipt(args.receipt, payload)
            print(json.dumps(payload, ensure_ascii=True))
            raise SystemExit(2)
        if result.returncode != 0:
            payload = {"status": "capa_error", "returncode": result.returncode, "job": result.job_audit}
            _write_receipt(args.receipt, payload)
            print(json.dumps(payload, ensure_ascii=True))
            raise SystemExit(5)
        try:
            aggregate = aggregate_capa_json(json.loads(_read_bounded(stdout_path).decode("utf-8", "strict")))
        except (UnicodeDecodeError, json.JSONDecodeError, CapaAggregateError, CapaOutputLimitError):
            payload = {"status": "capa_schema_error"}
            _write_receipt(args.receipt, payload)
            print(json.dumps(payload, ensure_ascii=True))
            raise SystemExit(6)
        payload = {"status": "ok", "aggregate": asdict(aggregate), "job": result.job_audit}
        _write_receipt(args.receipt, payload)
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
