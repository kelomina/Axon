from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_loop164_isolation_contract as isolation_contract  # noqa: E402
from pre_run_resource_leak_guard import GUARD_SCHEMA  # noqa: E402
from validate_loop164_isolation_contract import (  # noqa: E402
    CONTRACT_SCHEMA,
    FEATURE_CONTRACT_SCHEMA,
    GROUP_FIELDS,
    IMPLEMENTATION_BINDING_PHASE,
    LOOP_ID,
    METADATA_AUTHORITY_SCOPE,
    METADATA_AUTHORIZATION_SCHEMA,
    METADATA_TRUST_ANCHOR_SCHEMA,
    METADATA_VALIDATOR_CLOSURE_SCHEMA,
    RESIDUAL_FUSION_INPUT_FIELDS,
    ROLE_ORDER,
    consume_a2_metadata_lease,
    main,
    stable_component_id,
    validate_a2_metadata_authorization,
    validate_loop164_isolation_contract,
)

CANONICAL_ARGV = ["synthetic-python", "synthetic-metadata-validator"]
TRUSTED_KEY_FINGERPRINT = hashlib.sha256(b"synthetic-metadata-custodian-key").hexdigest()
VERIFICATION_RECEIPT_SHA256 = hashlib.sha256(b"synthetic-metadata-verification").hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_a2_metadata_authorization_case(
    *,
    authorization_path: Path,
    contract_path: Path,
    rows_path: Path,
    output_path: Path,
    resource_guard_path: Path,
) -> Path:
    validator_path = SCRIPTS_DIR / "validate_loop164_isolation_contract.py"
    resource_guard_path.write_text(
        json.dumps(
            {
                "schema": GUARD_SCHEMA,
                "guard_ready": True,
                "decision": "pass",
                "receipt": {
                    "created_at_unix": 1783814400.0,
                    "cwd": str(Path(__file__).resolve().parents[1]),
                    "command": CANONICAL_ARGV,
                    "target_sha256": {str(validator_path.resolve()): sha256_file(validator_path)},
                    "missing_targets": [],
                },
            }
        ),
        encoding="utf-8",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    trust_anchor_path = authorization_path.parent / "external_custody" / "trust_anchor.json"
    trust_anchor_path.parent.mkdir(parents=True, exist_ok=True)
    trust_anchor_path.write_text(
        json.dumps(
            {
                "schema": METADATA_TRUST_ANCHOR_SCHEMA,
                "loop_id": LOOP_ID,
                "trusted_key_fingerprint": TRUSTED_KEY_FINGERPRINT,
                "verification_receipt_sha256": VERIFICATION_RECEIPT_SHA256,
                "root_state": "externally_verified",
            }
        ),
        encoding="utf-8",
    )
    closure_files, closure_sha256 = isolation_contract._validator_source_closure()
    authorization_path.write_text(
        json.dumps(
            {
                "schema": METADATA_AUTHORIZATION_SCHEMA,
                "loop_id": LOOP_ID,
                "authorization_level": "A2_metadata_isolation_validation_only",
                "decision": "allow_single_metadata_isolation_validation",
                "execution_environment": "custodian_side_metadata_only",
                "operation": "loop164_metadata_isolation_validation",
                "authority_scope": METADATA_AUTHORITY_SCOPE,
                "issued_at_utc": "2026-07-12T00:00:00Z",
                "not_before_utc": "2026-07-12T00:00:00Z",
                "expires_at_utc": "2026-07-12T01:00:00Z",
                "authority_attestation": {
                    "trusted_key_fingerprint": TRUSTED_KEY_FINGERPRINT,
                    "trust_anchor_sha256": sha256_file(trust_anchor_path),
                    "verification_receipt_sha256": VERIFICATION_RECEIPT_SHA256,
                },
                "runtime_binding": {
                    "cwd": str(Path(__file__).resolve().parents[1]),
                    "python_executable": sys.executable,
                    "python_sha256": sha256_file(Path(sys.executable)),
                },
                "validator_binding": {
                    "path": str(validator_path),
                    "sha256": sha256_file(validator_path),
                },
                "validator_source_closure": {
                    "schema": METADATA_VALIDATOR_CLOSURE_SCHEMA,
                    "files": closure_files,
                    "closure_sha256": closure_sha256,
                },
                "canonical_argv": CANONICAL_ARGV,
                "contract_binding": {
                    "path": str(contract_path),
                    "sha256": sha256_file(contract_path),
                },
                "rows_artifact_binding": {
                    "path": str(rows_path),
                    "sha256": contract["rows_artifact"]["sha256"],
                },
                "metadata_root_binding": {"path": str(rows_path.parent)},
                "output_binding": {"path": str(output_path)},
                "resource_guard_binding": {
                    "path": str(resource_guard_path),
                    "sha256": sha256_file(resource_guard_path),
                },
                "max_resource_guard_age_seconds": 3600,
                "one_shot_lease": {
                    "lease_id": "synthetic-lease",
                    "state": "ready",
                    "purpose": "single_metadata_isolation_validation",
                },
            }
        ),
        encoding="utf-8",
    )
    return trust_anchor_path


def validate_authorization_case(
    *,
    authorization_path: Path,
    contract_path: Path,
    output_path: Path,
    resource_guard_path: Path,
    trust_anchor_path: Path,
    lease_directory: Path,
    actual_argv: list[str] | None = None,
):
    return validate_a2_metadata_authorization(
        authorization_json=authorization_path,
        contract_json=contract_path,
        output_json=output_path,
        now_utc=datetime(2026, 7, 12, tzinfo=timezone.utc),
        expected_authorization_json=authorization_path,
        expected_output_json=output_path,
        expected_resource_guard_json=resource_guard_path,
        lease_directory=lease_directory,
        trust_anchor_json=trust_anchor_path,
        expected_trusted_key_fingerprint=TRUSTED_KEY_FINGERPRINT,
        actual_argv=actual_argv or CANONICAL_ARGV,
    )


def make_record(
    *,
    index: int,
    role: str,
    first_seen_day: int,
    outer_fold_id: int | None = None,
    inner_fold_id: int | None = None,
    near_duplicate_cluster_id: str | None = None,
    family_id: str | None = None,
    campaign_id: str | None = None,
    source_group_id: str | None = None,
) -> dict:
    source_sha256 = f"{index:064x}"
    record = {
        "record_type": "sample",
        "sample_uid": f"record-{index}",
        "source_sha256": source_sha256,
        "locked_label": index % 2,
        "label_provenance": "custody-ledger-v1",
        "label_evidence_version": "labels-2026-07",
        "label_frozen_at_utc": "2026-01-01T00:00:00Z",
        "schema_version": "isolation-row-v1",
        "acquisition_time_utc": f"2025-01-{first_seen_day + 1:02d}T00:00:00Z",
        "first_seen_time_utc": f"2025-01-{first_seen_day:02d}T00:00:00Z",
        "timestamp_provenance": "custodian_verified",
        "source_id": "provider-a",
        "source_group_id": source_group_id or f"collection-{index}",
        "exact_cluster_id": f"exact-{index}",
        "near_duplicate_cluster_id": near_duplicate_cluster_id or f"near-{index}",
        "family_id": family_id or f"family-{index}",
        "family_evidence_version": "family-ledger-v1",
        "campaign_id": campaign_id or f"campaign-{index}",
        "campaign_evidence_version": "campaign-ledger-v1",
        "isolation_component_id": stable_component_id([source_sha256]),
        "parser_status": "parsed",
        "grouping_status": "resolved",
        "feature_schema_version": "fixed-v3",
        "split_role": role,
        "oof_role": "not_applicable",
        "outer_fold_id": None,
        "inner_fold_id": None,
        "calibration_role": "not_applicable",
        "evaluation_generation": "loop164-a2",
        "denominator_status": "included",
    }
    if role == "train_anchor":
        record["oof_role"] = "warmup_not_meta_eligible"
        record["calibration_role"] = "fit_only"
    elif role == "train_oof":
        record["oof_role"] = "eligible"
        record["outer_fold_id"] = outer_fold_id
        record["inner_fold_id"] = inner_fold_id
        record["calibration_role"] = "outer_holdout"
    return record


def refresh_component_ids(records: list[dict]) -> None:
    groups: list[set[int]] = []
    pending = set(range(len(records)))
    while pending:
        component = {pending.pop()}
        changed = True
        while changed:
            changed = False
            for candidate in list(pending):
                if any(
                    any(
                        records[candidate][field] == records[current][field]
                        for field in GROUP_FIELDS
                    )
                    for current in component
                ):
                    pending.remove(candidate)
                    component.add(candidate)
                    changed = True
        groups.append(component)
    for component in groups:
        component_hash = stable_component_id(records[index]["source_sha256"] for index in component)
        for index in component:
            records[index]["isolation_component_id"] = component_hash


def write_contract_case(
    tmp_path: Path, records: list[dict], *, mutate_contract=None
) -> tuple[Path, Path]:
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(
        json.dumps(
            {
                "loop_id": LOOP_ID,
                "decision": "propose_loop164_whole_file_residual_expert_no_execution",
            }
        ),
        encoding="utf-8",
    )
    rows_path = tmp_path / "full_pool_isolation_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    contract = {
        "schema": CONTRACT_SCHEMA,
        "loop_id": LOOP_ID,
        "pool_scope": "full_pool",
        "required_roles": ["train_anchor", "train_oof"],
        "manifest_version": "loop164-contract-v1",
        "proposal_binding": {
            "path": str(proposal_path),
            "sha256": sha256_file(proposal_path),
        },
        "inventory": {"expected_active_rows": len(records), "inventory_sha256": "b" * 64},
        "rows_artifact": {
            "path": rows_path.name,
            "sha256": sha256_file(rows_path),
            "rows": len(records),
        },
        "grouping": {
            "algorithm": "multi_relation_union_find",
            "version": "1",
            "parameters_sha256": "c" * 64,
            "required_relation_fields": list(GROUP_FIELDS),
            "path_derived_groups_forbidden": True,
            "unresolved_grouping_policy": "block",
            "candidate_coverage_complete": True,
            "oversized_bucket_policy": "block",
            "source_group_provenance": "custodian_provided",
        },
        "temporal_policy": {
            "event_time_field": "first_seen_time_utc",
            "component_time": "max_first_seen_time_utc",
            "roles_in_order": list(ROLE_ORDER),
            "embargo_seconds": 1,
            "allowed_timestamp_provenance": ["custodian_verified"],
        },
        "oof_policy": {
            "mode": "purged_forward_group",
            "fold_count": 5,
            "warmup_required": True,
            "eligible_once": True,
            "fold_manifest_shared_across_seeds": True,
            "seeds": [41, 42, 43],
            "minimum_fit_rows_per_label": 1,
            "minimum_holdout_rows_per_label": 1,
        },
        "model_input_fields": list(RESIDUAL_FUSION_INPUT_FIELDS),
        "feature_contract": {
            "schema": FEATURE_CONTRACT_SCHEMA,
            "feature_fields": list(RESIDUAL_FUSION_INPUT_FIELDS),
            "feature_matrix_receipt_required": True,
            "implementation_binding_phase": IMPLEMENTATION_BINDING_PHASE,
        },
        "identity_feature_policy": "Identity, grouping, provenance, time, labels, and split metadata are forbidden model inputs.",
    }
    if mutate_contract is not None:
        mutate_contract(contract)
    contract_path = tmp_path / "full_pool_group_manifest.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return contract_path, rows_path


def validate_case(contract_path: Path, rows_path: Path) -> dict:
    return validate_loop164_isolation_contract(
        contract_json=contract_path,
        rows_jsonl=rows_path,
        expected_proposal_json=contract_path.parent / "proposal.json",
        minimum_full_pool_rows=1,
        required_embargo_seconds=1,
        required_minimum_fit_rows_per_label=1,
        required_minimum_holdout_rows_per_label=1,
        required_roles=("train_anchor", "train_oof"),
        allow_synthetic_test_inputs=True,
    )


def valid_records() -> list[dict]:
    records = [
        make_record(index=1, role="train_anchor", first_seen_day=1),
        make_record(index=2, role="train_anchor", first_seen_day=2),
    ]
    next_index = 3
    for fold_index in range(5):
        first_day = 5 + fold_index * 4
        records.extend(
            [
                make_record(
                    index=next_index,
                    role="train_oof",
                    first_seen_day=first_day,
                    outer_fold_id=fold_index,
                    inner_fold_id=fold_index,
                ),
                make_record(
                    index=next_index + 1,
                    role="train_oof",
                    first_seen_day=first_day + 1,
                    outer_fold_id=fold_index,
                    inner_fold_id=fold_index,
                ),
            ]
        )
        next_index += 2
    refresh_component_ids(records)
    return records


def records_for_fold(records: list[dict], fold_index: int) -> list[dict]:
    return [record for record in records if record["outer_fold_id"] == fold_index]


def test_valid_contract_passes_with_forward_warmup_and_aggregate_only_receipt(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "pass"
    assert payload["ready_for"]["full_pool_isolation_contract"] is True
    assert payload["ready_for"]["loop164_train_oof_partition"] is True
    assert payload["ready_for"]["loop164_train_oof_data_boundary"] is False
    assert payload["ready_for"]["a2_training_authorization"] is False
    assert payload["counts"]["roles"] == {"train_anchor": 2, "train_oof": 10}
    assert payload["oof"]["eligible_rows"] == 10
    assert payload["oof"]["warmup_rows"] == 2
    assert payload["feature_contract"] == {
        "schema": FEATURE_CONTRACT_SCHEMA,
        "feature_fields": list(RESIDUAL_FUSION_INPUT_FIELDS),
        "feature_matrix_receipt_required": True,
        "implementation_binding_phase": IMPLEMENTATION_BINDING_PHASE,
    }
    assert "source_sha256" not in json.dumps(payload)


def test_contract_rejects_legacy_implementation_manifest_placeholder(tmp_path: Path):
    contract_path, rows_path = write_contract_case(
        tmp_path,
        valid_records(),
        mutate_contract=lambda contract: contract.update(
            {
                "feature_contract": {
                    "schema": "axon_loop164_residual_fusion_feature_contract_v1",
                    "implementation_manifest_sha256": "d" * 64,
                    "feature_matrix_receipt_required": True,
                }
            }
        ),
    )

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert "contract_feature_contract_shape_invalid" in payload["blockers"]
    assert "contract_feature_contract_schema_invalid" in payload["blockers"]
    assert "contract_feature_contract_implementation_binding_phase_invalid" in payload["blockers"]


def test_contract_rejects_wrong_deferred_phase_or_feature_allowlist(tmp_path: Path):
    contract_path, rows_path = write_contract_case(
        tmp_path,
        valid_records(),
        mutate_contract=lambda contract: contract["feature_contract"].update(
            {
                "feature_fields": [*RESIDUAL_FUSION_INPUT_FIELDS[:-1], "family_id"],
                "implementation_binding_phase": "metadata_receipt_binds_implementation",
            }
        ),
    )

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert "contract_feature_contract_feature_allowlist_invalid" in payload["blockers"]
    assert "contract_feature_contract_implementation_binding_phase_invalid" in payload["blockers"]


def test_contract_blocks_transitive_hard_group_component_crossing_roles(tmp_path: Path):
    records = valid_records()
    records.append(
        make_record(
            index=99,
            role="val_a",
            first_seen_day=20,
            family_id=records[1]["family_id"],
        )
    )
    refresh_component_ids(records)
    contract_path, rows_path = write_contract_case(tmp_path, records)

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert "isolation_component_cross_split_role" in payload["blockers"]


def test_contract_blocks_unresolved_group_path_time_and_identity_features(tmp_path: Path):
    records = valid_records()
    records[1]["source_group_id"] = "unknown"
    records[1]["timestamp_provenance"] = "path_date_proxy"
    refresh_component_ids(records)
    contract_path, rows_path = write_contract_case(
        tmp_path,
        records,
        mutate_contract=lambda contract: contract["model_input_fields"].append("source_group_id"),
    )

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert "identity_feature_input_detected" in payload["blockers"]
    assert "source_group_id" in payload["identity_feature_violations"]


def test_contract_blocks_nonforward_outer_fold_or_embargo_order(tmp_path: Path):
    records = valid_records()
    fold_one_records = records_for_fold(records, 1)
    fold_one_records[0]["first_seen_time_utc"] = "2025-01-04T00:00:00Z"
    fold_one_records[0]["acquisition_time_utc"] = "2025-01-05T00:00:00Z"
    refresh_component_ids(records)
    contract_path, rows_path = write_contract_case(tmp_path, records)

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert "outer_fold_temporal_or_embargo_violation" in payload["blockers"]


def test_contract_uses_component_max_time_for_forward_fold_order(tmp_path: Path):
    records = valid_records()
    fold_one_records = records_for_fold(records, 1)
    fold_one_records[0]["first_seen_time_utc"] = "2025-01-04T00:00:00Z"
    fold_one_records[0]["acquisition_time_utc"] = "2025-01-05T00:00:00Z"
    fold_one_records[1]["first_seen_time_utc"] = "2025-01-12T00:00:00Z"
    fold_one_records[1]["acquisition_time_utc"] = "2025-01-13T00:00:00Z"
    fold_one_records[0]["family_id"] = "family-late-component"
    fold_one_records[1]["family_id"] = "family-late-component"
    refresh_component_ids(records)
    contract_path, rows_path = write_contract_case(tmp_path, records)

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "pass"
    assert "outer_fold_temporal_or_embargo_violation" not in payload["blockers"]


def test_contract_blocks_duplicate_sha_and_conflicting_locked_label(tmp_path: Path):
    records = valid_records()
    duplicate = dict(records[2])
    duplicate["sample_uid"] = "record-duplicate"
    duplicate["locked_label"] = 1 - int(records[2]["locked_label"])
    records.append(duplicate)
    refresh_component_ids(records)
    contract_path, rows_path = write_contract_case(tmp_path, records)

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert "duplicate_source_sha256" in payload["blockers"]
    assert "conflicting_locked_label_for_source_sha256" in payload["blockers"]


def test_contract_blocks_conflicting_labels_inside_one_exact_cluster(tmp_path: Path):
    records = valid_records()
    fold_zero_records = records_for_fold(records, 0)
    fold_zero_records[1]["exact_cluster_id"] = fold_zero_records[0]["exact_cluster_id"]
    refresh_component_ids(records)
    contract_path, rows_path = write_contract_case(tmp_path, records)

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert "conflicting_locked_label_for_exact_cluster" in payload["blockers"]


def test_contract_blocks_untrusted_timestamp_provenance(tmp_path: Path):
    records = valid_records()
    records_for_fold(records, 0)[0]["timestamp_provenance"] = "provider_unverified"
    refresh_component_ids(records)
    contract_path, rows_path = write_contract_case(tmp_path, records)

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert "timestamp_provenance_untrusted" in payload["blockers"]


def test_default_validator_requires_authoritative_full_pool_minimum(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())

    payload = validate_loop164_isolation_contract(
        contract_json=contract_path,
        rows_jsonl=rows_path,
        expected_proposal_json=contract_path.parent / "proposal.json",
        allow_synthetic_test_inputs=True,
    )

    assert payload["decision"] == "block"
    assert "contract_full_pool_rows_below_minimum" in payload["blockers"]


def test_direct_validator_requires_explicit_synthetic_fixture_mode(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())

    try:
        validate_loop164_isolation_contract(contract_json=contract_path, rows_jsonl=rows_path)
    except RuntimeError as error:
        assert "Direct metadata validation is disabled" in str(error)
    else:
        raise AssertionError("direct metadata validation unexpectedly opened synthetic rows")


def test_a2_metadata_authorization_binds_paths_before_rows_and_consumes_lease(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )

    authorization = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=tmp_path / "metadata_leases",
    )

    assert authorization["ready"] is True
    assert consume_a2_metadata_lease(
        authorization,
        actual_argv=CANONICAL_ARGV,
        now_utc=datetime(2026, 7, 12, tzinfo=timezone.utc),
    ) is True
    assert consume_a2_metadata_lease(
        authorization,
        actual_argv=CANONICAL_ARGV,
        now_utc=datetime(2026, 7, 12, tzinfo=timezone.utc),
    ) is False


def test_cli_blocks_before_opening_rows_when_a2_authorization_is_missing(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    rows_path.unlink()
    output_path = tmp_path / "blocked_receipt.json"

    exit_code = main(
        [
            "--contract-json",
            str(contract_path),
            "--a2-authorization-json",
            str(tmp_path / "missing_authorization.json"),
            "--output-json",
            str(output_path),
        ]
    )
    assert exit_code == 2
    assert not output_path.exists()


def test_metadata_lease_is_content_addressed_and_clone_replay_is_blocked(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    lease_directory = tmp_path / "metadata_leases"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )
    original = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=lease_directory,
    )

    assert original["ready"] is True
    assert consume_a2_metadata_lease(
        original,
        actual_argv=CANONICAL_ARGV,
        now_utc=datetime(2026, 7, 12, tzinfo=timezone.utc),
    ) is True

    clone_path = tmp_path / "copied_a2_authorization.json"
    clone_path.write_bytes(authorization_path.read_bytes())
    clone = validate_authorization_case(
        authorization_path=clone_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=lease_directory,
    )

    assert clone["ready"] is False
    assert clone["failures"] == ["a2_authorization_lease_already_consumed"]


def test_metadata_lease_rejects_reformatted_authorization_replay(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    lease_directory = tmp_path / "metadata_leases"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )
    authorization = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=lease_directory,
    )

    assert authorization["ready"] is True
    assert consume_a2_metadata_lease(
        authorization,
        actual_argv=CANONICAL_ARGV,
        now_utc=datetime(2026, 7, 12, tzinfo=timezone.utc),
    ) is True
    authorization_path.write_text(
        json.dumps(json.loads(authorization_path.read_text(encoding="utf-8")), separators=(",", ":")),
        encoding="utf-8",
    )

    replay = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=lease_directory,
    )

    assert replay["ready"] is False
    assert replay["failures"] == ["a2_authorization_lease_already_consumed"]


def test_metadata_authorization_rejects_argv_drift_before_lease(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )

    authorization = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=tmp_path / "metadata_leases",
        actual_argv=[*CANONICAL_ARGV, "--drift"],
    )

    assert authorization["ready"] is False
    assert "a2_authorization_canonical_argv_mismatch" in authorization["failures"]


def test_metadata_authorization_rejects_training_or_granted_scope(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )
    authorization_payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization_payload["authority_scope"] = {
        "tier": "A2",
        "operation": "training",
        "protected_input_scope": "metadata_only",
        "grants": ["train_oof"],
    }
    authorization_path.write_text(
        json.dumps(authorization_payload), encoding="utf-8"
    )

    authorization = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=tmp_path / "metadata_leases",
    )

    assert authorization["ready"] is False
    assert "a2_authorization_authority_scope_invalid" in authorization["failures"]


def test_metadata_lease_refuses_binding_drift_before_metadata_open(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )
    authorization = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=tmp_path / "metadata_leases",
    )
    contract_path.write_text(contract_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert authorization["ready"] is True
    assert consume_a2_metadata_lease(
        authorization,
        actual_argv=CANONICAL_ARGV,
        now_utc=datetime(2026, 7, 12, tzinfo=timezone.utc),
    ) is False
    assert not authorization["lease_marker_path"].exists()


def test_metadata_authorization_refuses_existing_output_without_burning_lease(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )
    output_path.write_text("pre-existing receipt\n", encoding="utf-8")

    authorization = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=tmp_path / "metadata_leases",
    )

    assert authorization["ready"] is False
    assert authorization["failures"] == ["a2_authorization_output_already_exists"]
    assert not (tmp_path / "metadata_leases").exists()


def test_metadata_lease_keeps_marker_after_fsync_failure(tmp_path: Path, monkeypatch):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )
    authorization = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=tmp_path / "metadata_leases",
    )

    def fail_fsync(_: int) -> None:
        raise OSError("synthetic fsync failure")

    assert authorization["ready"] is True
    with monkeypatch.context() as context:
        context.setattr(isolation_contract.os, "fsync", fail_fsync)
        assert consume_a2_metadata_lease(
            authorization,
            actual_argv=CANONICAL_ARGV,
            now_utc=datetime(2026, 7, 12, tzinfo=timezone.utc),
        ) is False
    assert authorization["lease_marker_path"].exists()
    assert consume_a2_metadata_lease(
        authorization,
        actual_argv=CANONICAL_ARGV,
        now_utc=datetime(2026, 7, 12, tzinfo=timezone.utc),
    ) is False


def test_contract_blocks_rows_drift_detected_after_parsing(tmp_path: Path, monkeypatch):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    original_sha256_open_file = isolation_contract.sha256_open_file
    calls = 0

    def drift_on_final_hash(handle):
        nonlocal calls
        calls += 1
        digest = original_sha256_open_file(handle)
        return "f" * 64 if calls == 3 else digest

    monkeypatch.setattr(isolation_contract, "sha256_open_file", drift_on_final_hash)
    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert "rows_artifact_changed_during_parse" in payload["blockers"]


def test_metadata_authorization_blocks_contract_mutation_after_binding(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["manifest_version"] = "mutated-after-authorization"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    authorization = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=tmp_path / "metadata_leases",
    )

    assert authorization["ready"] is False
    assert "a2_authorization_contract_sha256_mismatch" in authorization["failures"]


def test_metadata_authorization_rejects_bare_resource_guard(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "isolation_receipt.json"
    resource_guard_path = tmp_path / "resource_guard.json"
    authorization_path = tmp_path / "a2_authorization.json"
    trust_anchor_path = write_a2_metadata_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        rows_path=rows_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
    )
    resource_guard_path.write_text(
        json.dumps(
            {
                "guard_ready": True,
                "decision": "pass",
                "receipt": {"created_at_unix": 1783814400.0},
            }
        ),
        encoding="utf-8",
    )
    authorization_payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization_payload["resource_guard_binding"]["sha256"] = sha256_file(resource_guard_path)
    authorization_path.write_text(json.dumps(authorization_payload), encoding="utf-8")

    authorization = validate_authorization_case(
        authorization_path=authorization_path,
        contract_path=contract_path,
        output_path=output_path,
        resource_guard_path=resource_guard_path,
        trust_anchor_path=trust_anchor_path,
        lease_directory=tmp_path / "metadata_leases",
    )

    assert authorization["ready"] is False
    assert "a2_authorization_resource_guard_not_ready" in authorization["failures"]


def test_invalid_cli_authorization_cannot_overwrite_existing_output(tmp_path: Path):
    contract_path, _ = write_contract_case(tmp_path, valid_records())
    output_path = tmp_path / "existing_receipt.json"
    output_path.write_text("immutable-existing-receipt\n", encoding="utf-8")

    exit_code = main(
        [
            "--contract-json",
            str(contract_path),
            "--a2-authorization-json",
            str(tmp_path / "missing_authorization.json"),
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert output_path.read_text(encoding="utf-8") == "immutable-existing-receipt\n"


def test_contract_rejects_artifact_hash_mismatch_before_reading_rows(tmp_path: Path):
    contract_path, rows_path = write_contract_case(tmp_path, valid_records())
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["rows_artifact"]["sha256"] = "d" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    payload = validate_case(contract_path, rows_path)

    assert payload["decision"] == "block"
    assert payload["rows_read"] == 0
    assert "rows_artifact_sha256_mismatch" in payload["blockers"]
