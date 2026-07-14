#!/usr/bin/env python3
"""Fail closed when frozen TopK ties make the ONNX remediation gate unreachable."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "p0_loop28_onnx_operator_remediation_001"

DEFAULT_PROPOSAL = Path("manifests/roadmap_9997/p0_loop28_onnx_operator_remediation/proposal.json")
DEFAULT_AUTHORIZATION = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_operator_remediation/authorization.json"
)
DEFAULT_AMENDMENT = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_operator_remediation/preflight_amendment.json"
)
DEFAULT_PARENT_CLOSURE = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/post_manifest.json")
DEFAULT_LOCALIZATION_EVIDENCE = Path(
    "reports/roadmap_9997/p0_loop28_onnx_fidelity/localization_evidence.final.json"
)
DEFAULT_OUTPUT = Path("manifests/roadmap_9997/p0_loop28_onnx_operator_remediation/preflight.json")

EXPECTED_FIXTURES = (
    "pe32_numeric_resource_tls_callbacks",
    "pe32_named_resource_tls_callbacks",
    "pe32_numeric_resource_zero_tls_callbacks",
    "pe32_plus_named_resource_zero_tls_callbacks",
)
EXPECTED_TIED_OCCURRENCE = 15
EXPECTED_TIED_NODE_INDEX = 1739
EXPECTED_ROUTE_COUNT = 62
EXPECTED_MARGIN_MULTIPLIER = 8.0


class OperatorPreflightError(RuntimeError):
    """Raised when the frozen no-execution proof cannot be reproduced."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise OperatorPreflightError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorPreflightError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise OperatorPreflightError(f"JSON artifact must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_within(project_root: Path, relative: Path, *, must_exist: bool = True) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise OperatorPreflightError(f"Path must remain project-relative: {relative}")
    root = project_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise OperatorPreflightError(f"Path escapes project root: {relative}") from exc
    if must_exist and not candidate.is_file():
        raise OperatorPreflightError(f"Required artifact is not a file: {relative}")
    return candidate


def _require_mapping(value: object, purpose: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OperatorPreflightError(f"{purpose} must be an object")
    return value


def _require_list(value: object, purpose: str) -> list[Any]:
    if not isinstance(value, list):
        raise OperatorPreflightError(f"{purpose} must be a list")
    return value


def _validate_timestamp(value: str) -> str:
    if not value or not value.endswith("Z"):
        raise OperatorPreflightError("generated_at_utc must be a UTC timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise OperatorPreflightError("generated_at_utc is invalid") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise OperatorPreflightError("generated_at_utc must use UTC")
    return value


def _verify_bound_artifact(
    project_root: Path,
    record: Mapping[str, Any],
    *,
    expected_path: Path,
    purpose: str,
) -> tuple[Path, str]:
    if record.get("path") != expected_path.as_posix():
        raise OperatorPreflightError(f"{purpose} path binding drifted")
    path = _resolve_within(project_root, expected_path)
    digest = sha256_file(path)
    if record.get("sha256") != digest:
        raise OperatorPreflightError(f"{purpose} SHA-256 binding drifted")
    return path, digest


def _routing_rows(fixture: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    profiles = _require_mapping(fixture.get("profiles"), "fixture profiles")
    routing = _require_mapping(profiles.get("routing"), "routing profile")
    stability = _require_mapping(routing.get("stability"), "routing stability")
    if stability.get("route_count") != EXPECTED_ROUTE_COUNT:
        raise OperatorPreflightError("Frozen route count drifted")
    rows = _require_list(stability.get("rows"), "routing stability rows")
    if len(rows) != EXPECTED_ROUTE_COUNT:
        raise OperatorPreflightError("Frozen routing row inventory drifted")
    mapped_rows = [_require_mapping(row, "routing row") for row in rows]
    occurrences = [row.get("occurrence") for row in mapped_rows]
    expected_occurrences = [*range(61), 62]
    if occurrences != expected_occurrences:
        raise OperatorPreflightError("Frozen routing occurrences are not canonical 0..60,62")
    return mapped_rows


def prove_frozen_tie_blocker(
    evidence: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    if evidence.get("decision") != "localized_negative_no_raw":
        raise OperatorPreflightError("Localization evidence decision drifted")
    if evidence.get("determinism_all_passed") is not True:
        raise OperatorPreflightError("Localization determinism evidence is not complete")

    fixtures = _require_list(evidence.get("fixtures"), "localization fixtures")
    if len(fixtures) != len(EXPECTED_FIXTURES):
        raise OperatorPreflightError("Frozen fixture count drifted")

    rows_by_name: dict[str, Mapping[str, Any]] = {}
    for fixture_value in fixtures:
        fixture = _require_mapping(fixture_value, "fixture evidence")
        identity = _require_mapping(fixture.get("fixture"), "fixture identity")
        name = identity.get("name")
        if not isinstance(name, str) or name in rows_by_name:
            raise OperatorPreflightError("Fixture identity is missing or duplicated")
        rows_by_name[name] = fixture
    if tuple(rows_by_name) != EXPECTED_FIXTURES:
        raise OperatorPreflightError("Frozen fixture ordering or identity drifted")

    fixture_proofs: list[dict[str, Any]] = []
    for name in EXPECTED_FIXTURES:
        fixture = rows_by_name[name]
        base_probability = _require_mapping(
            fixture.get("base_probability"), "base probability control"
        )
        if base_probability.get("control_reproduced") is not True:
            raise OperatorPreflightError(f"Baseline control did not reproduce: {name}")
        rows = _routing_rows(fixture)
        tied = rows[EXPECTED_TIED_OCCURRENCE]
        if tied.get("node_index") != EXPECTED_TIED_NODE_INDEX:
            raise OperatorPreflightError(f"Tied route node drifted: {name}")
        if tied.get("k") != 4 or tied.get("query_count") != 2048:
            raise OperatorPreflightError(f"Tied route contract drifted: {name}")
        if tied.get("margin_guard_multiplier") != EXPECTED_MARGIN_MULTIPLIER:
            raise OperatorPreflightError(f"Margin guard multiplier drifted: {name}")
        tie_count = tied.get("exact_support_tie_count")
        margin = tied.get("minimum_pytorch_support_margin")
        if not isinstance(tie_count, int) or tie_count < 1:
            raise OperatorPreflightError(f"Expected exact support tie is absent: {name}")
        if not isinstance(margin, (int, float)) or not math.isfinite(float(margin)):
            raise OperatorPreflightError(f"Support margin is invalid: {name}")
        if float(margin) != 0.0:
            raise OperatorPreflightError(f"Frozen support margin is not zero: {name}")
        all_ties = [
            {
                "occurrence": int(row["occurrence"]),
                "node_index": int(row["node_index"]),
                "k": int(row["k"]),
                "exact_support_tie_count": int(row["exact_support_tie_count"]),
            }
            for row in rows
            if int(row.get("exact_support_tie_count", 0)) > 0
        ]
        fixture_proofs.append(
            {
                "fixture": name,
                "control_reproduced": True,
                "route_count": len(rows),
                "shared_blocking_route": {
                    "occurrence": EXPECTED_TIED_OCCURRENCE,
                    "node_index": EXPECTED_TIED_NODE_INDEX,
                    "k": 4,
                    "query_count": 2048,
                    "exact_support_tie_count": tie_count,
                    "minimum_pytorch_support_margin": float(margin),
                    "margin_guard_multiplier": EXPECTED_MARGIN_MULTIPLIER,
                },
                "all_exact_support_ties": all_ties,
            }
        )

    success_gate = _require_mapping(proposal.get("success_gate"), "proposal success gate")
    if success_gate.get("candidate_margin_guard_violations") != 0:
        raise OperatorPreflightError("Proposal no-violation success gate drifted")
    candidate = _require_mapping(proposal.get("single_candidate"), "single candidate")
    if candidate.get("semantic_candidates", 1) not in (None, 1):
        raise OperatorPreflightError("Candidate count drifted")
    if candidate.get("selection_is_frozen") is not True:
        raise OperatorPreflightError("Candidate selection is not frozen")

    # 冻结判据使用严格大于号。参考 margin 已经是 0，而任何绝对 delta 都非负，
    # 所以即便候选逐 bit 相同，0 > 8 * 0 仍然为假，成功分支不可达。
    delta_examples = (0.0, 1.0e-12, 1.0e-6)
    evaluations = [
        {
            "candidate_score_delta": delta,
            "left_margin": 0.0,
            "right_bound": EXPECTED_MARGIN_MULTIPLIER * delta,
            "strict_guard_passed": 0.0 > EXPECTED_MARGIN_MULTIPLIER * delta,
        }
        for delta in delta_examples
    ]
    if any(row["strict_guard_passed"] for row in evaluations):
        raise OperatorPreflightError("Formal strict-margin proof is inconsistent")

    return {
        "fixture_count": len(fixture_proofs),
        "shared_tied_occurrence": EXPECTED_TIED_OCCURRENCE,
        "shared_tied_node_index": EXPECTED_TIED_NODE_INDEX,
        "fixture_proofs": fixture_proofs,
        "formal_proof": {
            "reference_support_margin": 0.0,
            "candidate_score_delta_domain": "delta >= 0",
            "frozen_guard": "margin > 8 * delta",
            "reduced_guard": "0 > 8 * delta",
            "evaluations": evaluations,
            "universal_result": "false for every delta >= 0",
            "success_branch_reachable": False,
        },
    }


def build_preflight(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    paths = {
        "proposal": _resolve_within(root, DEFAULT_PROPOSAL),
        "authorization": _resolve_within(root, DEFAULT_AUTHORIZATION),
        "amendment": _resolve_within(root, DEFAULT_AMENDMENT),
        "parent_closure": _resolve_within(root, DEFAULT_PARENT_CLOSURE),
        "localization_evidence": _resolve_within(root, DEFAULT_LOCALIZATION_EVIDENCE),
    }
    payloads = {name: load_json_strict(path) for name, path in paths.items()}
    hashes = {name: sha256_file(path) for name, path in paths.items()}

    proposal = payloads["proposal"]
    authorization = payloads["authorization"]
    amendment = payloads["amendment"]
    parent = payloads["parent_closure"]
    evidence = payloads["localization_evidence"]
    if proposal.get("schema") != "axon_loop28_onnx_operator_remediation_proposal_v1":
        raise OperatorPreflightError("Proposal schema mismatch")
    if proposal.get("loop_id") != LOOP_ID:
        raise OperatorPreflightError("Proposal loop mismatch")
    if (
        authorization.get("decision")
        != "authorize_single_bounded_operator_remediation_implementation"
    ):
        raise OperatorPreflightError("Authorization decision mismatch")
    if amendment.get("decision") != (
        "authorize_fail_closed_exact_tie_preflight_and_forbid_candidate_execution"
    ):
        raise OperatorPreflightError("Preflight amendment decision mismatch")
    _verify_bound_artifact(
        root,
        _require_mapping(authorization.get("proposal"), "authorization proposal"),
        expected_path=DEFAULT_PROPOSAL,
        purpose="Proposal",
    )
    _verify_bound_artifact(
        root,
        _require_mapping(amendment.get("proposal"), "amendment proposal"),
        expected_path=DEFAULT_PROPOSAL,
        purpose="Amended proposal",
    )
    _verify_bound_artifact(
        root,
        _require_mapping(amendment.get("authorization"), "amendment authorization"),
        expected_path=DEFAULT_AUTHORIZATION,
        purpose="Authorization",
    )
    if parent.get("decision") != "post_fidelity_closure_frozen_localized_negative_no_raw":
        raise OperatorPreflightError("Parent closure decision mismatch")
    if hashes["parent_closure"] != proposal["parent_closure"]["sha256"]:
        raise OperatorPreflightError("Parent closure hash drifted")
    if hashes["localization_evidence"] != proposal["parent_localization_evidence"]["sha256"]:
        raise OperatorPreflightError("Localization evidence hash drifted")
    if hashes["localization_evidence"] != amendment["new_preflight_fact"]["source_sha256"]:
        raise OperatorPreflightError("Amendment evidence hash drifted")
    scope_reduction = _require_mapping(amendment.get("scope_reduction"), "scope reduction")
    forbidden_execution_fields = (
        "candidate_generation_allowed",
        "checkpoint_or_onnx_model_load_allowed",
        "native_probe_execution_allowed",
        "generation_or_verification_lease_allowed",
    )
    if any(scope_reduction.get(field) is not False for field in forbidden_execution_fields):
        raise OperatorPreflightError("Amendment does not fail closed on execution")

    proof = prove_frozen_tie_blocker(evidence, proposal)
    return {
        "schema": "axon_loop28_onnx_operator_remediation_preflight_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            name: {"path": path.relative_to(root).as_posix(), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "frozen_gate": {
            "tolerance": 1.0e-6,
            "topk_indices_exact_required": True,
            "support_margin_guard": "pytorch_support_margin > 8 * measured_candidate_score_delta",
            "tie_exception_allowed": False,
            "gate_relaxation_allowed": False,
        },
        "proof": proof,
        "execution_audit": {
            "checkpoint_load_count": 0,
            "onnx_graph_load_count": 0,
            "native_probe_subprocess_count": 0,
            "candidate_graph_count": 0,
            "lease_count": 0,
            "raw_split_cache_heldout_access_count": 0,
            "quality_metric_count": 0,
            "f1_computation_count": 0,
        },
        "next_route": {
            "runtime_family": "pytorch_compatible_native_runtime",
            "required_new_proposal": True,
            "onnx_candidate_generation_forbidden": True,
            "reuse_same_intermediate_and_topk_gates": True,
        },
        "claim_boundary": amendment["claim_boundary"],
        "decision": amendment["added_exit_decision"],
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise OperatorPreflightError(f"Output already exists: {path}") from exc


def verify_preflight(project_root: Path, output: Path) -> dict[str, Any]:
    output_path = _resolve_within(project_root, output)
    payload = load_json_strict(output_path)
    if payload.get("schema") != "axon_loop28_onnx_operator_remediation_preflight_v1":
        raise OperatorPreflightError("Preflight schema mismatch")
    rebuilt = build_preflight(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc") or ""),
    )
    if payload != rebuilt:
        raise OperatorPreflightError("Preflight no longer matches frozen evidence")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve(strict=True)
    if args.verify:
        payload = verify_preflight(project_root, args.output)
    else:
        if not args.generated_at_utc:
            raise OperatorPreflightError("--generated-at-utc is required when building")
        output = _resolve_within(project_root, args.output, must_exist=False)
        payload = build_preflight(project_root, generated_at_utc=args.generated_at_utc)
        _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "fixture_count": payload["proof"]["fixture_count"],
                "candidate_graph_count": payload["execution_audit"]["candidate_graph_count"],
                "decision": payload["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
