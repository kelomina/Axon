#!/usr/bin/env python3
"""Fail-closed aggregate-only verifier for future Loop164 dual-window certification."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "loop164_whole_file_residual_expert"
POWER_SCHEMA = "axon_loop164_certification_power_analysis_v1"
WINDOW_MANIFEST_SCHEMA = "axon_loop164_sealed_window_manifest_v1"
WINDOW_RECEIPT_SCHEMA = "axon_loop164_sealed_window_evaluation_receipt_v1"
REPLICATION_SCHEMA = "axon_loop164_certification_replication_receipt_v1"
VALIDATION_SCHEMA = "axon_loop164_certification_evidence_validation_v1"
TARGET_F1 = Fraction(9997, 10000)
BOOTSTRAP_REPLICATES = 200000
ONE_SIDED_ALPHA = Fraction(1, 40)
MINIMUM_SIMULATIONS = 50000
MINIMUM_JOINT_POWER = Fraction(9, 10)
REQUIRED_GATE_NAMES = (
    "FPR",
    "FNR",
    "FP_per_million_benign",
    "FN_per_1000_malicious",
    "coverage",
    "P95_latency",
    "cost",
    "critical_slice_error_share",
)
FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "sample_uid",
        "source_sha256",
        "source_path",
        "filename",
        "directory",
        "sample_index",
        "row_order",
        "prediction",
        "prediction_rows",
    }
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.casefold()
    )


def _fraction(value: object) -> Optional[Fraction]:
    try:
        parsed = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if 0 <= parsed <= 1 else None


def _positive_int(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _nonnegative_int(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _exact_keys(payload: object, expected: set[str], label: str, blockers: list[str]) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        blockers.append(f"{label}_not_object")
        return None
    if set(payload) != expected:
        blockers.append(f"{label}_shape_invalid")
        return None
    return payload


def _contains_identity(value: object) -> bool:
    if isinstance(value, dict):
        return any(key in FORBIDDEN_IDENTITY_KEYS or _contains_identity(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_identity(item) for item in value)
    return False


def _binding(payload: dict[str, Any], field: str, expected: str, label: str, blockers: list[str]) -> None:
    if payload.get(field) != expected:
        blockers.append(f"{label}_{field}_mismatch")


def _validate_power(payload: object, *, protocol_sha256: str, blockers: list[str]) -> Optional[dict[str, Any]]:
    power = _exact_keys(
        payload,
        {
            "schema", "loop_id", "protocol_sha256", "bundle_sha256", "statistics_runner_sha256",
            "cluster_profile_sha256", "simulation_method", "prng", "simulation_count",
            "planned_point_floor", "joint_pass_count", "joint_simulation_count",
            "joint_power_mc_lower_bound", "unknown_components", "decision",
        },
        "certification_power",
        blockers,
    )
    if power is None:
        return None
    _binding(power, "schema", POWER_SCHEMA, "certification_power", blockers)
    _binding(power, "loop_id", LOOP_ID, "certification_power", blockers)
    _binding(power, "protocol_sha256", protocol_sha256, "certification_power", blockers)
    for field in ("bundle_sha256", "statistics_runner_sha256", "cluster_profile_sha256"):
        if not _is_sha256(power.get(field)):
            blockers.append(f"certification_power_{field}_invalid")
    if power.get("simulation_method") != "aggregate_only_component_bootstrap_monte_carlo_v1":
        blockers.append("certification_power_method_invalid")
    if power.get("prng") != "PCG64DXSM":
        blockers.append("certification_power_prng_invalid")
    simulations = _positive_int(power.get("simulation_count"))
    joint_simulations = _positive_int(power.get("joint_simulation_count"))
    pass_count = _nonnegative_int(power.get("joint_pass_count"))
    if simulations is None or simulations < MINIMUM_SIMULATIONS or joint_simulations != simulations:
        blockers.append("certification_power_simulation_count_invalid")
    if pass_count is None or simulations is None or pass_count > simulations:
        blockers.append("certification_power_pass_count_invalid")
    floor = _fraction(power.get("planned_point_floor"))
    if floor is None or floor <= TARGET_F1:
        blockers.append("certification_power_point_floor_invalid")
    lower = _fraction(power.get("joint_power_mc_lower_bound"))
    if lower is None or lower < MINIMUM_JOINT_POWER:
        blockers.append("certification_power_joint_lower_bound_invalid")
    elif simulations is not None and pass_count is not None and lower > Fraction(pass_count, simulations):
        blockers.append("certification_power_lower_bound_exceeds_observed")
    if power.get("unknown_components") != 0:
        blockers.append("certification_power_unknown_components")
    if power.get("decision") != "pass":
        blockers.append("certification_power_not_pass")
    return power


def _validate_manifest(payload: object, *, window_id: str, protocol_sha256: str, power_sha256: str, power: dict[str, Any], blockers: list[str]) -> Optional[dict[str, Any]]:
    manifest = _exact_keys(
        payload,
        {
            "schema", "loop_id", "window_id", "window_start_utc", "window_end_utc",
            "protocol_sha256", "power_analysis_sha256", "manifest_root_sha256", "label_provenance_sha256",
            "isolation_closure_sha256", "bundle_sha256", "statistics_runner_sha256",
            "a3_authorization_sha256", "a3_lease_consumption_sha256", "evaluation_generation",
            "component_overlap_count", "replacement_count", "pooling_enabled", "unknown_components",
        },
        f"{window_id}_manifest",
        blockers,
    )
    if manifest is None:
        return None
    for field, expected in (("schema", WINDOW_MANIFEST_SCHEMA), ("loop_id", LOOP_ID), ("window_id", window_id), ("protocol_sha256", protocol_sha256), ("power_analysis_sha256", power_sha256), ("bundle_sha256", power.get("bundle_sha256")), ("statistics_runner_sha256", power.get("statistics_runner_sha256"))):
        _binding(manifest, field, expected, f"{window_id}_manifest", blockers)
    for field in ("power_analysis_sha256", "manifest_root_sha256", "label_provenance_sha256", "isolation_closure_sha256", "a3_authorization_sha256", "a3_lease_consumption_sha256"):
        if not _is_sha256(manifest.get(field)):
            blockers.append(f"{window_id}_manifest_{field}_invalid")
    try:
        start = datetime.fromisoformat(str(manifest.get("window_start_utc")).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(manifest.get("window_end_utc")).replace("Z", "+00:00"))
    except ValueError:
        blockers.append(f"{window_id}_manifest_time_invalid")
    else:
        if start >= end:
            blockers.append(f"{window_id}_manifest_time_order_invalid")
    if manifest.get("evaluation_generation") != 1:
        blockers.append(f"{window_id}_manifest_generation_invalid")
    if manifest.get("component_overlap_count") != 0 or manifest.get("replacement_count") != 0:
        blockers.append(f"{window_id}_manifest_overlap_or_replacement")
    if manifest.get("pooling_enabled") is not False or manifest.get("unknown_components") != 0:
        blockers.append(f"{window_id}_manifest_isolation_invalid")
    return manifest


def _validate_receipt(payload: object, *, window_id: str, protocol_sha256: str, power_sha256: str, manifest: dict[str, Any], manifest_sha256: str, blockers: list[str]) -> Optional[dict[str, Any]]:
    receipt = _exact_keys(
        payload,
        {
            "schema", "loop_id", "window_id", "protocol_sha256", "power_analysis_sha256",
            "window_manifest_sha256", "bundle_sha256", "statistics_runner_sha256",
            "a3_authorization_sha256", "a3_lease_consumption_sha256", "evaluation_generation",
            "confusion_matrix", "denominator", "failure_reason_counts", "unmapped_failure_rows",
            "point_f1", "f1_lcb_97_5", "bootstrap", "cluster_time_summary", "operational_gates", "decision",
        },
        f"{window_id}_receipt",
        blockers,
    )
    if receipt is None:
        return None
    for field, expected in (("schema", WINDOW_RECEIPT_SCHEMA), ("loop_id", LOOP_ID), ("window_id", window_id), ("protocol_sha256", protocol_sha256), ("power_analysis_sha256", power_sha256), ("window_manifest_sha256", manifest_sha256), ("bundle_sha256", manifest.get("bundle_sha256")), ("statistics_runner_sha256", manifest.get("statistics_runner_sha256")), ("a3_authorization_sha256", manifest.get("a3_authorization_sha256")), ("a3_lease_consumption_sha256", manifest.get("a3_lease_consumption_sha256"))):
        _binding(receipt, field, expected, f"{window_id}_receipt", blockers)
    if receipt.get("evaluation_generation") != 1:
        blockers.append(f"{window_id}_receipt_generation_invalid")
    confusion = _exact_keys(receipt.get("confusion_matrix"), {"TP", "TN", "FP", "FN"}, f"{window_id}_receipt_confusion", blockers)
    denominator = _exact_keys(receipt.get("denominator"), {"eligible_rows", "scored_rows"}, f"{window_id}_receipt_denominator", blockers)
    if confusion is not None and denominator is not None:
        values = {name: _nonnegative_int(confusion.get(name)) for name in confusion}
        if any(value is None for value in values.values()):
            blockers.append(f"{window_id}_receipt_confusion_invalid")
        else:
            total = sum(value for value in values.values() if value is not None)
            if denominator.get("eligible_rows") != total or denominator.get("scored_rows") != total:
                blockers.append(f"{window_id}_receipt_denominator_mismatch")
            calculated = Fraction(2 * values["TP"], 2 * values["TP"] + values["FP"] + values["FN"]) if 2 * values["TP"] + values["FP"] + values["FN"] else Fraction(0)
            if _fraction(receipt.get("point_f1")) != calculated:
                blockers.append(f"{window_id}_receipt_point_f1_mismatch")
    reasons = receipt.get("failure_reason_counts")
    if not isinstance(reasons, dict) or set(reasons) != {"abstain", "timeout", "missing_feature", "unsupported", "parser_failure", "runtime_failure"} or any(_nonnegative_int(value) is None for value in reasons.values()):
        blockers.append(f"{window_id}_receipt_failure_reasons_invalid")
    if receipt.get("unmapped_failure_rows") != 0:
        blockers.append(f"{window_id}_receipt_unmapped_failures")
    lcb = _fraction(receipt.get("f1_lcb_97_5"))
    if lcb is None or lcb < TARGET_F1:
        blockers.append(f"{window_id}_receipt_lcb_below_target")
    bootstrap = _exact_keys(receipt.get("bootstrap"), {"replicates", "one_sided_alpha", "prng", "seed_sha256", "lower_quantile_rule", "lower_quantile_rank", "distribution_sha256", "conservative_guard_pass"}, f"{window_id}_receipt_bootstrap", blockers)
    if bootstrap is not None:
        if bootstrap.get("replicates") != BOOTSTRAP_REPLICATES or _fraction(bootstrap.get("one_sided_alpha")) != ONE_SIDED_ALPHA or bootstrap.get("prng") != "PCG64DXSM" or bootstrap.get("lower_quantile_rule") != "order_statistic_ceiling(alpha_times_B_plus_1)" or bootstrap.get("lower_quantile_rank") != 5001 or bootstrap.get("conservative_guard_pass") is not True or not _is_sha256(bootstrap.get("seed_sha256")) or not _is_sha256(bootstrap.get("distribution_sha256")):
            blockers.append(f"{window_id}_receipt_bootstrap_invalid")
    summary = _exact_keys(receipt.get("cluster_time_summary"), {"component_algorithm", "component_time", "calendar_blocking", "resampling_unit", "cross_window_component_overlap", "unknown_or_empty_state"}, f"{window_id}_receipt_cluster_time", blockers)
    if summary is not None and summary != {"component_algorithm": "union_find_exact_near_family_campaign_source_v1", "component_time": "max_first_seen_time_utc", "calendar_blocking": "preregistered_calendar_blocks", "resampling_unit": "whole_component_with_replacement_within_block", "cross_window_component_overlap": 0, "unknown_or_empty_state": "insufficient_evidence"}:
        blockers.append(f"{window_id}_receipt_cluster_time_invalid")
    gates = receipt.get("operational_gates")
    if not isinstance(gates, dict) or set(gates) != set(REQUIRED_GATE_NAMES) or any(value is not True for value in gates.values()):
        blockers.append(f"{window_id}_receipt_operational_gates_invalid")
    if receipt.get("decision") != "pass":
        blockers.append(f"{window_id}_receipt_not_pass")
    return receipt


def validate_certification_evidence(*, proposal_json: Path, power_json: Path, w1_manifest_json: Path, w1_receipt_json: Path, w2_manifest_json: Path, w2_receipt_json: Path, replication_json: Path) -> dict[str, Any]:
    blockers: list[str] = []
    proposal = _read_json(proposal_json)
    if proposal is None:
        return {"schema": VALIDATION_SCHEMA, "decision": "block", "certification_status": "not_certified", "blockers": ["certification_proposal_invalid"]}
    try:
        from build_loop164_mainline_preflight import validate_certification_protocol

        validate_certification_protocol(proposal)
    except (ImportError, ValueError):
        return {"schema": VALIDATION_SCHEMA, "decision": "block", "certification_status": "not_certified", "blockers": ["certification_protocol_invalid"]}
    protocol_sha256 = sha256_file(proposal_json)
    documents = [power_json, w1_manifest_json, w1_receipt_json, w2_manifest_json, w2_receipt_json, replication_json]
    payloads = [_read_json(path) for path in documents]
    if any(payload is None for payload in payloads):
        return {"schema": VALIDATION_SCHEMA, "decision": "block", "certification_status": "not_certified", "blockers": ["certification_evidence_unreadable"]}
    if any(_contains_identity(payload) for payload in payloads):
        blockers.append("certification_evidence_contains_identity_payload")
    power = _validate_power(payloads[0], protocol_sha256=protocol_sha256, blockers=blockers)
    if power is not None:
        power_sha256 = sha256_file(power_json)
        w1_manifest = _validate_manifest(payloads[1], window_id="W1_certification", protocol_sha256=protocol_sha256, power_sha256=power_sha256, power=power, blockers=blockers)
        w2_manifest = _validate_manifest(payloads[3], window_id="W2_later_replication", protocol_sha256=protocol_sha256, power_sha256=power_sha256, power=power, blockers=blockers)
        if w1_manifest is not None and w2_manifest is not None:
            try:
                w1_end = datetime.fromisoformat(str(w1_manifest["window_end_utc"]).replace("Z", "+00:00"))
                w2_start = datetime.fromisoformat(str(w2_manifest["window_start_utc"]).replace("Z", "+00:00"))
                if w2_start <= w1_end:
                    blockers.append("certification_windows_not_temporally_ordered")
            except ValueError:
                pass
            w1_receipt = _validate_receipt(payloads[2], window_id="W1_certification", protocol_sha256=protocol_sha256, power_sha256=power_sha256, manifest=w1_manifest, manifest_sha256=sha256_file(w1_manifest_json), blockers=blockers)
            w2_receipt = _validate_receipt(payloads[4], window_id="W2_later_replication", protocol_sha256=protocol_sha256, power_sha256=power_sha256, manifest=w2_manifest, manifest_sha256=sha256_file(w2_manifest_json), blockers=blockers)
            replication = _exact_keys(payloads[5], {"schema", "loop_id", "protocol_sha256", "power_analysis_sha256", "bundle_sha256", "w1_receipt_sha256", "w2_receipt_sha256", "pooling_used", "replacement_count", "decision"}, "certification_replication", blockers)
            if replication is not None and w1_receipt is not None and w2_receipt is not None:
                expected = {"schema": REPLICATION_SCHEMA, "loop_id": LOOP_ID, "protocol_sha256": protocol_sha256, "power_analysis_sha256": power_sha256, "bundle_sha256": power["bundle_sha256"], "w1_receipt_sha256": sha256_file(w1_receipt_json), "w2_receipt_sha256": sha256_file(w2_receipt_json), "pooling_used": False, "replacement_count": 0, "decision": "certified_99_97"}
                if replication != expected:
                    blockers.append("certification_replication_invalid")
    return {"schema": VALIDATION_SCHEMA, "loop_id": LOOP_ID, "decision": "pass" if not blockers else "block", "certification_status": "certified_99_97" if not blockers else "not_certified", "blockers": sorted(set(blockers))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Loop164 dual-window aggregate certification evidence.")
    for name in ("proposal", "power", "w1-manifest", "w1-receipt", "w2-manifest", "w2-receipt", "replication"):
        parser.add_argument(f"--{name}-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_certification_evidence(proposal_json=args.proposal_json, power_json=args.power_json, w1_manifest_json=args.w1_manifest_json, w1_receipt_json=args.w1_receipt_json, w2_manifest_json=args.w2_manifest_json, w2_receipt_json=args.w2_receipt_json, replication_json=args.replication_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
