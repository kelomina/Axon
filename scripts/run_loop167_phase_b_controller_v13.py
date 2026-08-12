"""Run the independent Loop167 fit-only OOF over the sealed v12 cache."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.loop167_phase_b.contracts import canonical_json_bytes, sha256_file
from src.loop167_phase_b.feature_cache_v4 import make_phase_b_fit_payload
from src.loop167_phase_b.fit_cache_loader_v13 import load_verified_v12_cache_for_v13
from src.loop167_phase_b.fit_targets_adapter_v13 import load_fit_targets_v13
from src.loop167_phase_b.fit_worker import run_phase_b_fit
from src.loop167_phase_b.progress_ledger import FitLedger

REPORT = ROOT / "reports/roadmap_9997/loop167/phase_b_v13_cache_only_fit"
LEASE = REPORT / "phase_b_execution_consumed_v13.json"


def consume_lease() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "axon_loop167_phase_b_execution_consumed_v13", "status": "consumed", "raw_open_attempts": 0}
    data = canonical_json_bytes(payload)
    try:
        fd = os.open(LEASE, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError("v13 lease already consumed; retry is forbidden") from error
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    if sys.argv[1:] != ["--execute"]:
        raise SystemExit("usage: run_loop167_phase_b_controller_v13.py --execute")
    consume_lease()
    verified = load_verified_v12_cache_for_v13(ROOT)
    protocol = ROOT / "manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_protocol.json"
    targets = load_fit_targets_v13(ROOT, protocol_sha256=sha256_file(protocol))
    payload = make_phase_b_fit_payload(verified.loaded_cache.cache, targets.labels, targets.folds)
    ledger_path = REPORT / "phase_b_fit_progress_v13.jsonl"
    with FitLedger.create(ledger_path) as ledger:
        result = run_phase_b_fit(payload.cache, payload.labels, payload.folds, ledger,
            fit_protocol_commitment_sha256=targets.protocol_sha256,
            feature_rows_commitment_sha256=verified.raw_validation.feature_rows_commitment_sha256,
            raw_ledger_final_record_sha256=verified.raw_validation.final_record_sha256)
    receipt = {"schema": "axon_loop167_phase_b_execution_receipt_v13", "status": "fit_completed_train_only",
               "raw_open_attempts": 0, "heldout_access": False, "fit_units": result.total_fit_units,
               "fit_ledger_sha256": result.fit_ledger_final_record_sha256}
    (REPORT / "phase_b_execution_receipt_v13.json").write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
