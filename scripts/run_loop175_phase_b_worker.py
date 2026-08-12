#!/usr/bin/env python3
"""Launch one isolated Loop175 Phase-B pilot or outer-fold unit."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.phase_b_contract import write_exclusive_json  # noqa: E402
from src.loop175.phase_b_worker import run_outer_fold, run_pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("pilot", "outer"), required=True)
    parser.add_argument("--arm", choices=("A", "B", "C", "D", "E"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--source-closure", type=Path, required=True)
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument("--pilot-receipt", type=Path)
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--worker-receipt", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure-receipt", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("--execute is required")
    try:
        if arguments.mode == "pilot":
            if arguments.arm == "A" or arguments.output is None:
                raise ValueError("pilot requires arm B-E and --output")
            result = run_pilot(
                project_root=PROJECT_ROOT,
                source_closure_path=arguments.source_closure.resolve(),
                cache_receipt_path=arguments.cache_receipt.resolve(),
                arm=arguments.arm,
                output=arguments.output.resolve(),
                microbatch=arguments.microbatch,
                gradient_accumulation=arguments.gradient_accumulation,
            )
        else:
            required = {
                "artifact_directory": arguments.artifact_directory,
                "worker_receipt": arguments.worker_receipt,
                "checkpoint": arguments.checkpoint,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"outer mode lacks required paths: {missing}")
            result = run_outer_fold(
                project_root=PROJECT_ROOT,
                source_closure_path=arguments.source_closure.resolve(),
                cache_receipt_path=arguments.cache_receipt.resolve(),
                pilot_receipt_path=(
                    None if arguments.pilot_receipt is None else arguments.pilot_receipt.resolve()
                ),
                artifact_directory=arguments.artifact_directory.resolve(),
                worker_receipt_path=arguments.worker_receipt.resolve(),
                checkpoint_path=arguments.checkpoint.resolve(),
                arm=arguments.arm,
                outer_fold=arguments.fold,
                microbatch=arguments.microbatch,
                gradient_accumulation=arguments.gradient_accumulation,
            )
    except Exception as error:
        failure = {
            "schema": "axon_loop175_phase_b_worker_failure_v1",
            "mode": arguments.mode,
            "arm": arguments.arm,
            "fold": arguments.fold,
            "error_type": type(error).__name__,
            "detail": str(error),
            "traceback": traceback.format_exc(),
            "decision": "worker_failed_no_success_claim",
        }
        if not arguments.failure_receipt.exists():
            write_exclusive_json(arguments.failure_receipt.resolve(), failure)
        print(json.dumps(failure, ensure_ascii=True, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

