#!/usr/bin/env python3
"""Run or resume the complete Loop175 seed-41 Train-only OOF experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.phase_b_controller import run_seed41_controller  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-closure", type=Path, required=True)
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--final-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("--execute is required")
    receipt = run_seed41_controller(
        project_root=PROJECT_ROOT,
        source_closure_path=arguments.source_closure.resolve(),
        cache_receipt_path=arguments.cache_receipt.resolve(),
        output_directory=arguments.output_directory.resolve(),
        final_receipt_path=arguments.final_receipt.resolve(),
    )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0 if receipt["decision"] == "seed41_pass_allow_seed42_43" else 2


if __name__ == "__main__":
    raise SystemExit(main())

