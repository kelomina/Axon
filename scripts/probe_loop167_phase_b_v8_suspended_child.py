#!/usr/bin/env python3
"""Run a zero-input v8 suspended-child Windows Job probe without project artifacts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop167_phase_b.windows_job_v8 import WindowsJobV8  # noqa: E402


def main() -> int:
    if os.name != "nt":
        raise SystemExit("The v8 suspended-child probe requires Windows")
    with tempfile.TemporaryDirectory(prefix="axon-loop167-v8-probe-") as temporary_directory:
        job = WindowsJobV8.create(memory_limit_bytes=64 * 1024 * 1024, kill_on_close=True)
        child = None
        try:
            child = job.spawn_suspended_assigned(
                (sys.executable, "-I", "-c", "import sys; sys.exit(0)"),
                cwd=temporary_directory,
                environment={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            audit = job.assignment_audit(child)
            child.resume()
            returncode = child.wait(30)
            active_after = job.active_processes()
            if returncode != 0 or active_after != 0:
                raise RuntimeError("v8 suspended-child probe did not close its contained process tree")
            print(
                json.dumps(
                    {
                        "decision": "pass",
                        "assignment": audit,
                        "active_processes_after": active_after,
                        "raw_open_attempts": 0,
                    },
                    sort_keys=True,
                )
            )
            return 0
        finally:
            if child is not None:
                child.close()
            job.close()


if __name__ == "__main__":
    raise SystemExit(main())
