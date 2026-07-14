from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_loop164_a2_request import (  # noqa: E402
    METADATA_AUTHORIZATION_PATH,
    METADATA_REQUEST_SCHEMA,
    TRAINING_REQUEST_SCHEMA,
    validate_a2_request_file,
    validate_a2_request_payload,
)
from validate_loop164_isolation_contract import (  # noqa: E402
    validate_a2_metadata_authorization,
)

TEMPLATES = (
    Path("manifests/roadmap_9997/loop164_whole_file_residual_expert/templates")
    / "a2_metadata_request.template.json",
    Path("manifests/roadmap_9997/loop164_whole_file_residual_expert/templates")
    / "a2_training_request.template.json",
)


def read_template(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_a2_request_templates_are_valid_and_never_authorize():
    metadata_request = read_template(TEMPLATES[0])
    training_request = read_template(TEMPLATES[1])

    assert metadata_request["schema"] == METADATA_REQUEST_SCHEMA
    assert training_request["schema"] == TRAINING_REQUEST_SCHEMA
    assert validate_a2_request_payload(metadata_request) == []
    assert validate_a2_request_payload(training_request) == []
    assert metadata_request["authorization_granted"] is False
    assert training_request["authorization_granted"] is False


def test_metadata_request_rejects_authorization_decision_and_scope_expansion():
    payload = read_template(TEMPLATES[0])
    payload["decision"] = "allow_single_metadata_isolation_validation"
    payload["authority_scope"]["grants"] = ["train_oof"]

    blockers = validate_a2_request_payload(payload)

    assert "metadata_a2_request_unexpected_fields" in blockers
    assert "a2_request_contains_authorization_field" in blockers
    assert "metadata_request_scope_invalid" in blockers


def test_training_request_rejects_ready_state_and_heldout_role():
    payload = read_template(TEMPLATES[1])
    payload["request_state"] = "draft"
    payload["allowed_split_roles"].append("val_a")

    blockers = validate_a2_request_payload(payload)

    assert "a2_request_state_invalid" in blockers
    assert "training_request_allowed_roles_invalid" in blockers


def test_request_validator_rejects_canonical_authorization_location(tmp_path: Path):
    request_path = tmp_path / METADATA_AUTHORIZATION_PATH
    request_path.parent.mkdir(parents=True)
    request_path.write_text(TEMPLATES[0].read_text(encoding="utf-8"), encoding="utf-8")

    result = validate_a2_request_file(request_path, root=tmp_path)

    assert result["decision"] == "block"
    assert result["blockers"] == ["a2_request_path_is_authorization_path"]


def test_request_file_reports_pass_without_granting_authorization(tmp_path: Path):
    request_path = tmp_path / "metadata_request.json"
    request_path.write_text(TEMPLATES[0].read_text(encoding="utf-8"), encoding="utf-8")

    result = validate_a2_request_file(request_path, root=tmp_path)

    assert result["decision"] == "pass"
    assert result["authorization_granted"] is False


def test_metadata_gate_rejects_request_template_before_rows_are_opened(tmp_path: Path):
    authorization_path = tmp_path / METADATA_AUTHORIZATION_PATH
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text(TEMPLATES[0].read_text(encoding="utf-8"), encoding="utf-8")

    result = validate_a2_metadata_authorization(
        authorization_json=authorization_path,
        contract_json=tmp_path / "missing-contract.json",
        output_json=tmp_path / "missing-output.json",
        expected_authorization_json=authorization_path,
        expected_output_json=tmp_path / "missing-output.json",
        expected_resource_guard_json=tmp_path / "missing-resource-guard.json",
        lease_directory=tmp_path / "metadata-leases",
        trust_anchor_json=tmp_path / "missing-trust-anchor.json",
        expected_trusted_key_fingerprint="0" * 64,
        actual_argv=["synthetic-metadata-validator"],
        now_utc=datetime(2026, 7, 12, tzinfo=timezone.utc),
    )

    assert result["ready"] is False
    assert "a2_authorization_unexpected_fields" in result["failures"]
    assert not (tmp_path / "metadata-leases").exists()
