#!/usr/bin/env python3
"""Run the outer Windows Job supervisor for the Loop167 Phase-B v7 route."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop167_phase_b.contracts import PhaseBContractError, sha256_file  # noqa: E402
from loop167_phase_b.execution_authorization_v7 import (  # noqa: E402
    validate_execution_authorization_v7,
)
from loop167_phase_b.execution_contract_v7 import (  # noqa: E402
    CONTROLLER_RELATIVE_PATH,
    RUN_AUTHORIZATION_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    SUPERVISOR_RELATIVE_PATH,
    VNEV_PYTHON_RELATIVE_PATH,
    verify_execution_contract_v7,
)
from loop167_phase_b.invocation_v7 import (  # noqa: E402
    bootstrap_thread_environment_v7,
    validate_current_runtime_invocation_v7,
)
from loop167_phase_b.path_safety_v4 import safe_project_path  # noqa: E402
from loop167_phase_b.preflight_v7 import validate_static_preflight_v7  # noqa: E402
from loop167_phase_b.supervisor_v7 import (  # noqa: E402
    SupervisorConfigV7,
    run_supervised_v7,
)


def _binding(path: Path) -> dict[str, str]:
    relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    safe_path = safe_project_path(
        PROJECT_ROOT,
        relative_path,
        require_exists=True,
        require_regular_file=True,
    )
    return {
        "path": relative_path,
        "sha256": sha256_file(safe_path),
    }


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _static_preflight(mode: str) -> Any:
    bootstrap_thread_environment_v7()
    validate_current_runtime_invocation_v7(PROJECT_ROOT, role="supervisor", mode=mode)
    return validate_static_preflight_v7(
        PROJECT_ROOT,
        source_closure_binding=_binding(PROJECT_ROOT / SOURCE_CLOSURE_RELATIVE_PATH),
        controller_binding=_binding(PROJECT_ROOT / CONTROLLER_RELATIVE_PATH),
        supervisor_binding=_binding(PROJECT_ROOT / SUPERVISOR_RELATIVE_PATH),
        phase="prelaunch",
    )


def run_execute() -> dict[str, Any]:
    static_receipt = _static_preflight("execute")
    authorization = validate_execution_authorization_v7(
        PROJECT_ROOT,
        PROJECT_ROOT / RUN_AUTHORIZATION_RELATIVE_PATH,
        now_utc=_utc_now(),
        phase="prelaunch",
    )
    expected = {
        "source_closure_binding": static_receipt.source_closure_binding,
        "execution_contract_binding": static_receipt.execution_contract_binding,
        "runtime_lock_binding": static_receipt.runtime_lock_binding,
        "controller_binding": static_receipt.controller_binding,
        "supervisor_binding": static_receipt.supervisor_binding,
        "loop166_windows_job_binding": static_receipt.loop166_windows_job_binding,
        "loop166_windows_process_lineage_binding": static_receipt.loop166_windows_process_lineage_binding,
    }
    for name, binding in expected.items():
        if dict(getattr(authorization, name)) != dict(binding):
            raise PhaseBContractError(f"v7 supervisor authorization {name} drifted from static preflight")
    contract = verify_execution_contract_v7(PROJECT_ROOT, authorization.execution_contract_binding)
    command = (
        str((PROJECT_ROOT / VNEV_PYTHON_RELATIVE_PATH).resolve(strict=True)),
        "-I",
        CONTROLLER_RELATIVE_PATH,
        "--execute",
    )
    outputs = authorization.output_paths
    result = run_supervised_v7(
        SupervisorConfigV7(
            project_root=PROJECT_ROOT,
            mode="execute",
            command=command,
            launch_receipt=outputs["supervisor_launch_receipt"],
            exit_receipt=outputs["supervisor_exit_receipt"],
            failure_receipt=outputs["supervisor_failure_receipt"],
            memory_limit_bytes=int(contract.resource_contract["maximum_training_peak_rss_bytes"]),
            timeout_seconds=int(contract.resource_contract["maximum_total_wall_seconds"]),
            static_bindings={
                "source_closure": dict(authorization.source_closure_binding),
                "execution_contract": dict(authorization.execution_contract_binding),
                "runtime_lock": dict(authorization.runtime_lock_binding),
                "controller": dict(authorization.controller_binding),
                "supervisor": dict(authorization.supervisor_binding),
                "loop166_windows_job": dict(authorization.loop166_windows_job_binding),
                "loop166_windows_process_lineage": dict(authorization.loop166_windows_process_lineage_binding),
            },
        )
    )
    return {
        "returncode": result.returncode,
        "launch_receipt_sha256": result.launch_receipt_sha256,
        "exit_receipt_sha256": result.exit_receipt_sha256,
        "raw_open_attempts": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        receipt = _static_preflight("preflight")
        print(
            json.dumps(
                {
                    "schema": "axon_loop167_phase_b_supervisor_preflight_receipt_v7",
                    "decision": "pass_static_preflight_raw_open_attempts_zero",
                    "source_closure_sha256": receipt.source_closure_sha256,
                    "execution_contract_sha256": receipt.execution_contract_sha256,
                    "runtime_lock_sha256": receipt.runtime_lock_sha256,
                    "raw_open_attempts": receipt.raw_open_attempts,
                },
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(run_execute(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
