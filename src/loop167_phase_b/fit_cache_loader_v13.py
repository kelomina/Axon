"""Load the completed v12 cache for the independent v13 fit-only lineage.

This module deliberately has no raw-worker or raw-manifest dependency.  The
v13 controller may consume only the complete v12 ledger and its immutable
numeric cache after the v12 controller failed at its post-cache time gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import PhaseBContractError, require_canonical_json, sha256_file
from .feature_cache_v4 import LoadedFeatureCacheV4, load_phase_b_feature_cache_v4
from .progress_ledger import RawScanLedgerValidation, validate_raw_scan_ledger


V12_REPORT_DIRECTORY = "reports/roadmap_9997/loop167/phase_b_v12_dual_identity_job_attestation_remediation"
V12_CACHE_RELATIVE_PATH = f"{V12_REPORT_DIRECTORY}/phase_b_feature_cache_v12.npz"
V12_LEDGER_RELATIVE_PATH = f"{V12_REPORT_DIRECTORY}/phase_b_raw_progress_v12.jsonl"
V12_FAILURE_RELATIVE_PATH = f"{V12_REPORT_DIRECTORY}/phase_b_controller_failure_v12.json"
V12_CACHE_SHA256 = "7826abfc76e04f93ea4b6ee4bc31cf25e651dab4355fa989f29e5488c7fda18b"
V12_LEDGER_FINAL_SHA256 = "c8c4b6f4a963cc538438bcb0d23e8611c5c4240e162e1518e64fb6cd78d0103f"


@dataclass(frozen=True, slots=True)
class VerifiedV12CacheForV13:
    loaded_cache: LoadedFeatureCacheV4
    raw_validation: RawScanLedgerValidation
    cache_sha256: str


def load_verified_v12_cache_for_v13(project_root: Path | str) -> VerifiedV12CacheForV13:
    """Verify and load the only v12 cache eligible for a v13 fit-only run."""

    root = Path(project_root).resolve(strict=True)
    failure = require_canonical_json(root / V12_FAILURE_RELATIVE_PATH)
    if (
        failure.get("schema") != "axon_loop167_phase_b_controller_failure_receipt_v12"
        or failure.get("error_type") != "PhaseBContractError"
        or failure.get("detail") != "v12 Phase-B extraction wall-clock budget was exceeded"
    ):
        raise PhaseBContractError("v13 requires the sealed v12 post-cache budget failure")
    raw_validation = validate_raw_scan_ledger(root / V12_LEDGER_RELATIVE_PATH)
    if not raw_validation.complete or raw_validation.final_record_sha256 != V12_LEDGER_FINAL_SHA256:
        raise PhaseBContractError("v13 requires the complete sealed v12 raw ledger")
    cache_path = root / V12_CACHE_RELATIVE_PATH
    cache_sha256 = sha256_file(cache_path)
    if cache_sha256 != V12_CACHE_SHA256:
        raise PhaseBContractError("v13 cache digest differs from the sealed v12 cache")
    loaded_cache = load_phase_b_feature_cache_v4(
        cache_path,
        expected_cache_sha256=cache_sha256,
        expected_raw_scope_commitment_sha256=raw_validation.raw_scope_commitment_sha256,
        expected_feature_rows_commitment_sha256=raw_validation.feature_rows_commitment_sha256,
        expected_raw_ledger_final_record_sha256=raw_validation.final_record_sha256,
    )
    if loaded_cache.cache.b0_values.shape[0] != 20_000:
        raise PhaseBContractError("v13 cache row count is not the sealed Train-only denominator")
    return VerifiedV12CacheForV13(loaded_cache, raw_validation, cache_sha256)
