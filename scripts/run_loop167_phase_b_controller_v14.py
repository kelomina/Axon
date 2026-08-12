"""One-shot fit plus evaluation over the validated Train-only cache."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.loop167_phase_b.contracts import canonical_json_bytes, sha256_file
from src.loop167_phase_b.evaluation_v4 import evaluate_phase_b_fit
from src.loop167_phase_b.feature_cache_v4 import make_phase_b_fit_payload
from src.loop167_phase_b.fit_cache_loader_v13 import load_verified_v12_cache_for_v13
from src.loop167_phase_b.fit_targets_adapter_v13 import load_fit_targets_v13
from src.loop167_phase_b.fit_worker import run_phase_b_fit
from src.loop167_phase_b.progress_ledger import FitLedger

REPORT = ROOT / "reports/roadmap_9997/loop167/phase_b_v14_cache_only_fit"
LEASE = REPORT / "phase_b_execution_consumed_v14.json"

def main() -> int:
    if sys.argv[1:] != ["--execute"]: raise SystemExit("--execute required")
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LEASE, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error: raise RuntimeError("v14 lease already consumed") from error
    with os.fdopen(fd, "wb") as handle:
        handle.write(canonical_json_bytes({"schema":"axon_loop167_phase_b_execution_consumed_v14","raw_open_attempts":0}))
        handle.flush(); os.fsync(handle.fileno())
    verified = load_verified_v12_cache_for_v13(ROOT)
    protocol = ROOT / "manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_protocol.json"
    protocol_sha = sha256_file(protocol)
    targets = load_fit_targets_v13(ROOT, protocol_sha256=protocol_sha)
    payload = make_phase_b_fit_payload(verified.loaded_cache.cache, targets.labels, targets.folds)
    with FitLedger.create(REPORT / "phase_b_fit_progress_v14.jsonl") as ledger:
        fit = run_phase_b_fit(payload.cache, payload.labels, payload.folds, ledger,
            fit_protocol_commitment_sha256=protocol_sha,
            feature_rows_commitment_sha256=verified.raw_validation.feature_rows_commitment_sha256,
            raw_ledger_final_record_sha256=verified.raw_validation.final_record_sha256)
    evaluation = evaluate_phase_b_fit(fit, payload.labels, payload.folds, targets.component_ids, protocol_sha256=protocol_sha)
    receipt = {"schema":"axon_loop167_phase_b_execution_receipt_v14","status":"fit_and_oof_evaluation_completed_train_only","raw_open_attempts":0,"heldout_access":False,"fit_units":fit.total_fit_units,"fit_ledger_sha256":fit.fit_ledger_final_record_sha256,"evaluation":{str(seed):dict(summary) for seed,summary in evaluation.items()}}
    (REPORT / "phase_b_execution_receipt_v14.json").write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
