#!/usr/bin/env python3
"""Periodic, read-only watcher for the Loop175 seed-41 OOF controller."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ARMS = ("A", "B", "C", "D", "E")
FOLDS = range(5)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fold_status(output: Path, arm: str, fold: int) -> dict[str, object]:
    checkpoint_candidates = (
        output / "checkpoints" / f"arm_{arm}_fold_{fold}.pt",
        output / "checkpoints" / f"arm_{arm}_fold_{fold}.joblib",
    )
    names = {
        "oof_npz": output / "oof" / f"arm_{arm}_fold_{fold}.npz",
        "oof_json": output / "oof" / f"arm_{arm}_fold_{fold}.json",
        "worker_receipt": output / "workers" / f"arm_{arm}_fold_{fold}.json",
        "failure_receipt": output / "failures" / f"arm_{arm}_fold_{fold}_attempt1.json",
    }
    present = {key: path.exists() for key, path in names.items()}
    present["checkpoint"] = any(path.exists() for path in checkpoint_candidates)
    complete = all(present[key] for key in ("checkpoint", "oof_npz", "oof_json", "worker_receipt"))
    return {"arm": arm, "fold": fold, "complete": complete, "present": present}


def snapshot(output: Path, final_receipt: Path) -> dict[str, object]:
    folds = [_fold_status(output, arm, fold) for arm in ARMS for fold in FOLDS]
    completed = [item for item in folds if item["complete"]]
    failures = [item for item in folds if item["present"]["failure_receipt"]]
    payload: dict[str, object] = {
        "schema": "axon_loop175_seed41_watch_status_v1",
        "observed_at": _now(),
        "pid": os.getpid(),
        "output_directory": str(output),
        "final_receipt": str(final_receipt),
        "completed_fold_count": len(completed),
        "completed_folds": [f"{item['arm']}{item['fold']}" for item in completed],
        "failure_receipt_count": len(failures),
        "failure_receipts": [f"{item['arm']}{item['fold']}" for item in failures],
        "final_receipt_present": final_receipt.exists(),
        "folds": folds,
    }
    if final_receipt.exists():
        try:
            receipt = json.loads(final_receipt.read_text(encoding="utf-8"))
            payload["decision"] = receipt.get("decision")
            payload["final_receipt_schema"] = receipt.get("schema")
        except (OSError, json.JSONDecodeError) as exc:
            payload["final_receipt_error"] = repr(exc)
    return payload


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--final-receipt", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 10:
        raise SystemExit("--interval-seconds must be >= 10")

    while True:
        status = snapshot(args.output_directory.resolve(), args.final_receipt.resolve())
        write_atomic(args.status_file.resolve(), status)
        print(json.dumps(status, ensure_ascii=True, sort_keys=True), flush=True)
        if args.once or status["final_receipt_present"] or status["failure_receipt_count"]:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
