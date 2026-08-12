#!/usr/bin/env python3
"""Seal the static Loop175 Phase-B source closure after focused validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.phase_b_source_closure import seal_phase_b_source_closure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-relative", required=True)
    arguments = parser.parse_args()
    receipt = seal_phase_b_source_closure(PROJECT_ROOT, arguments.output_relative)
    print(
        json.dumps(
            {
                "path": str(receipt.path),
                "sha256": receipt.sha256,
                "bytes": receipt.size_bytes,
                "source_count": receipt.source_count,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
