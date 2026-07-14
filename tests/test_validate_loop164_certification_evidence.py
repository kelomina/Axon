from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_loop164_certification_evidence import (  # noqa: E402
    LOOP_ID,
    PROJECT_ROOT,
    REQUIRED_GATE_NAMES,
    validate_certification_evidence,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_case(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / f"{name}.json" for name in ("power", "w1_manifest", "w1_receipt", "w2_manifest", "w2_receipt", "replication")}
    paths["proposal"] = PROJECT_ROOT / "manifests/roadmap_9997/loop164_whole_file_residual_expert/proposal.json"
    protocol_sha256 = sha256_file(paths["proposal"])
    write_json(
        paths["power"],
        {
            "schema": "axon_loop164_certification_power_analysis_v1",
            "loop_id": LOOP_ID,
            "protocol_sha256": protocol_sha256,
            "bundle_sha256": digest("bundle"),
            "statistics_runner_sha256": digest("statistics"),
            "cluster_profile_sha256": digest("profile"),
            "simulation_method": "aggregate_only_component_bootstrap_monte_carlo_v1",
            "prng": "PCG64DXSM",
            "simulation_count": 50000,
            "planned_point_floor": "9998/10000",
            "joint_pass_count": 49500,
            "joint_simulation_count": 50000,
            "joint_power_mc_lower_bound": "99/100",
            "unknown_components": 0,
            "decision": "pass",
        },
    )
    power_sha256 = sha256_file(paths["power"])
    manifests = {}
    for window_id, start, end, name in (
        ("W1_certification", "2026-01-01T00:00:00Z", "2026-01-31T00:00:00Z", "w1_manifest"),
        ("W2_later_replication", "2026-02-01T00:00:00Z", "2026-02-28T00:00:00Z", "w2_manifest"),
    ):
        manifests[name] = {
            "schema": "axon_loop164_sealed_window_manifest_v1",
            "loop_id": LOOP_ID,
            "window_id": window_id,
            "window_start_utc": start,
            "window_end_utc": end,
            "protocol_sha256": protocol_sha256,
            "power_analysis_sha256": power_sha256,
            "manifest_root_sha256": digest(f"{window_id}-root"),
            "label_provenance_sha256": digest(f"{window_id}-labels"),
            "isolation_closure_sha256": digest(f"{window_id}-isolation"),
            "bundle_sha256": digest("bundle"),
            "statistics_runner_sha256": digest("statistics"),
            "a3_authorization_sha256": digest(f"{window_id}-auth"),
            "a3_lease_consumption_sha256": digest(f"{window_id}-lease"),
            "evaluation_generation": 1,
            "component_overlap_count": 0,
            "replacement_count": 0,
            "pooling_enabled": False,
            "unknown_components": 0,
        }
        write_json(paths[name], manifests[name])
    for window_id, manifest_name, receipt_name in (
        ("W1_certification", "w1_manifest", "w1_receipt"),
        ("W2_later_replication", "w2_manifest", "w2_receipt"),
    ):
        manifest = manifests[manifest_name]
        write_json(
            paths[receipt_name],
            {
                "schema": "axon_loop164_sealed_window_evaluation_receipt_v1",
                "loop_id": LOOP_ID,
                "window_id": window_id,
                "protocol_sha256": protocol_sha256,
                "power_analysis_sha256": power_sha256,
                "window_manifest_sha256": sha256_file(paths[manifest_name]),
                "bundle_sha256": manifest["bundle_sha256"],
                "statistics_runner_sha256": manifest["statistics_runner_sha256"],
                "a3_authorization_sha256": manifest["a3_authorization_sha256"],
                "a3_lease_consumption_sha256": manifest["a3_lease_consumption_sha256"],
                "evaluation_generation": 1,
                "confusion_matrix": {"TP": 100000, "TN": 100000, "FP": 0, "FN": 0},
                "denominator": {"eligible_rows": 200000, "scored_rows": 200000},
                "failure_reason_counts": {"abstain": 0, "timeout": 0, "missing_feature": 0, "unsupported": 0, "parser_failure": 0, "runtime_failure": 0},
                "unmapped_failure_rows": 0,
                "point_f1": "1",
                "f1_lcb_97_5": "9998/10000",
                "bootstrap": {"replicates": 200000, "one_sided_alpha": "1/40", "prng": "PCG64DXSM", "seed_sha256": digest(f"{window_id}-seed"), "lower_quantile_rule": "order_statistic_ceiling(alpha_times_B_plus_1)", "lower_quantile_rank": 5001, "distribution_sha256": digest(f"{window_id}-distribution"), "conservative_guard_pass": True},
                "cluster_time_summary": {"component_algorithm": "union_find_exact_near_family_campaign_source_v1", "component_time": "max_first_seen_time_utc", "calendar_blocking": "preregistered_calendar_blocks", "resampling_unit": "whole_component_with_replacement_within_block", "cross_window_component_overlap": 0, "unknown_or_empty_state": "insufficient_evidence"},
                "operational_gates": {name: True for name in REQUIRED_GATE_NAMES},
                "decision": "pass",
            },
        )
    write_json(
        paths["replication"],
        {
            "schema": "axon_loop164_certification_replication_receipt_v1",
            "loop_id": LOOP_ID,
            "protocol_sha256": protocol_sha256,
            "power_analysis_sha256": power_sha256,
            "bundle_sha256": digest("bundle"),
            "w1_receipt_sha256": sha256_file(paths["w1_receipt"]),
            "w2_receipt_sha256": sha256_file(paths["w2_receipt"]),
            "pooling_used": False,
            "replacement_count": 0,
            "decision": "certified_99_97",
        },
    )
    return paths


def validate(paths: dict[str, Path]) -> dict:
    return validate_certification_evidence(
        proposal_json=paths["proposal"], power_json=paths["power"],
        w1_manifest_json=paths["w1_manifest"], w1_receipt_json=paths["w1_receipt"],
        w2_manifest_json=paths["w2_manifest"], w2_receipt_json=paths["w2_receipt"],
        replication_json=paths["replication"],
    )


def test_valid_aggregate_dual_window_evidence_certifies(tmp_path: Path):
    result = validate(create_case(tmp_path))

    assert result["decision"] == "pass"
    assert result["certification_status"] == "certified_99_97"


def test_certification_rejects_lcb_below_target(tmp_path: Path):
    paths = create_case(tmp_path)
    receipt = json.loads(paths["w2_receipt"].read_text(encoding="utf-8"))
    receipt["f1_lcb_97_5"] = "9996/10000"
    write_json(paths["w2_receipt"], receipt)

    result = validate(paths)

    assert result["decision"] == "block"
    assert "W2_later_replication_receipt_lcb_below_target" in result["blockers"]


def test_certification_rejects_cross_window_time_overlap(tmp_path: Path):
    paths = create_case(tmp_path)
    manifest = json.loads(paths["w2_manifest"].read_text(encoding="utf-8"))
    manifest["window_start_utc"] = "2026-01-30T00:00:00Z"
    write_json(paths["w2_manifest"], manifest)

    result = validate(paths)

    assert result["decision"] == "block"
    assert "certification_windows_not_temporally_ordered" in result["blockers"]


def test_certification_rejects_identity_payload(tmp_path: Path):
    paths = create_case(tmp_path)
    receipt = json.loads(paths["w1_receipt"].read_text(encoding="utf-8"))
    receipt["sample_uid"] = "must-not-appear"
    write_json(paths["w1_receipt"], receipt)

    result = validate(paths)

    assert result["decision"] == "block"
    assert "certification_evidence_contains_identity_payload" in result["blockers"]
    assert "W1_certification_receipt_shape_invalid" in result["blockers"]


def test_certification_rejects_replication_pooling(tmp_path: Path):
    paths = create_case(tmp_path)
    replication = json.loads(paths["replication"].read_text(encoding="utf-8"))
    replication["pooling_used"] = True
    write_json(paths["replication"], replication)

    result = validate(paths)

    assert result["decision"] == "block"
    assert "certification_replication_invalid" in result["blockers"]
