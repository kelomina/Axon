#!/usr/bin/env python3
"""Run the SHA-bound raw-file Loop151 champion reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop151_runtime.raw_runtime import Loop151Runtime, Loop151RuntimeError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Loop151 raw-file champion.")
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    args = parser.parse_args()
    try:
        runtime = Loop151Runtime(device=args.device, artifact_manifest=args.artifact_manifest)
        prediction = runtime.predict_path(args.file)
    except Loop151RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "loop_id": "Loop151",
                "prediction": prediction.prediction,
                "probability": prediction.probability,
                "primary_probability": prediction.primary_probability,
                "conservative_probability": prediction.conservative_probability,
                "content_cross_probability": prediction.content_cross_probability,
                "loop130_prediction": prediction.loop130_prediction,
                "loop134_probability": prediction.loop134_probability,
                "loop136_prediction": prediction.loop136_prediction,
                "selector_score": prediction.selector_score,
                "trusted_signer_downgrade": prediction.signer.downgraded,
                "trusted_signer_terms": list(prediction.signer.matched_terms),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
