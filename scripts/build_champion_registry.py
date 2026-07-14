#!/usr/bin/env python3
"""Generate the Axon research/native/connected champion registry from manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = {
    "truth_manifest": Path("manifests/roadmap_9997/p0_truth_freeze/loop151_truth_manifest.json"),
    "native_smoke_receipt": Path("manifests/roadmap_9997/p0_raw_replay/train_smoke_receipt.json"),
    "native_parity_receipt": Path(
        "manifests/roadmap_9997/p0_raw_replay/native_parity_receipt.json"
    ),
    "loop28_pause": Path(
        "manifests/roadmap_9997/p0_loop28_pytorch_native_decode_compat/paused_without_execution.json"
    ),
    "recommendation_status": Path(
        "reports/model_review/final_model_selection/ml_recommendation_status.json"
    ),
}


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def candidate_metrics(metric_payload: dict[str, Any]) -> dict[str, object]:
    candidate = metric_payload.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Truth manifest metric is missing candidate")
    required = ("f1", "errors", "false_positive", "false_negative", "sample_count")
    missing = [key for key in required if key not in candidate]
    if missing:
        raise ValueError(f"Truth manifest candidate is missing: {missing}")
    return {
        "f1": float(candidate["f1"]),
        "errors": int(candidate["errors"]),
        "fp": int(candidate["false_positive"]),
        "fn": int(candidate["false_negative"]),
        "rows": int(candidate["sample_count"]),
    }


def find_recommendation(payload: dict[str, Any], recommendation_id: str) -> dict[str, Any]:
    rows = payload.get("recommendations")
    if not isinstance(rows, list):
        raise ValueError("Recommendation status is missing recommendations")
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == recommendation_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one recommendation {recommendation_id}, found {len(matches)}")
    return matches[0]


def find_stage(receipt: dict[str, Any], stage_id: str) -> dict[str, Any]:
    dag = receipt.get("dag")
    if not isinstance(dag, dict) or not isinstance(dag.get("stages"), list):
        raise ValueError("Native receipt is missing DAG stages")
    matches = [
        stage
        for stage in dag["stages"]
        if isinstance(stage, dict) and stage.get("stage_id") == stage_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one DAG stage {stage_id}, found {len(matches)}")
    return matches[0]


def evidence_index(root: Path) -> dict[str, dict[str, str]]:
    evidence = {}
    for evidence_id, relative_path in SOURCE_PATHS.items():
        resolved = resolve_path(root, relative_path)
        if not resolved.is_file():
            raise ValueError(f"Missing champion registry source: {relative_path}")
        evidence[evidence_id] = {"path": relative_path.as_posix(), "sha256": sha256_file(resolved)}
    return evidence


def build_registry(root: Path, generated_at_utc: Optional[str] = None) -> dict[str, Any]:
    root = root.resolve()

    # 只解析已经冻结的聚合 manifest/receipt；不打开逐行预测、raw、split、cache 或模型载荷。
    sources = {name: read_json(resolve_path(root, path)) for name, path in SOURCE_PATHS.items()}
    truth = sources["truth_manifest"]
    smoke = sources["native_smoke_receipt"]
    parity = sources["native_parity_receipt"]
    pause = sources["loop28_pause"]
    recommendations = sources["recommendation_status"]

    require_equal(
        truth.get("decision"), "artifact_freeze_complete_raw_replay_pending", "truth decision"
    )
    truth_registry = truth.get("champion_registry")
    if not isinstance(truth_registry, dict):
        raise ValueError("Truth manifest is missing champion_registry")
    require_equal(
        truth_registry.get("champion_id"), "current_strict_best_loop151", "truth champion id"
    )
    require_equal(truth_registry.get("status"), "current_strict_best", "truth champion status")

    research_status = find_recommendation(recommendations, "current_strict_best_loop151")
    require_equal(
        research_status.get("status"), "current_strict_best", "recommendation champion status"
    )
    parked_candidate = find_recommendation(recommendations, "loop164_whole_file_residual_expert")
    require_equal(
        parked_candidate.get("status"),
        "parked_surrogate_negative_formal_gate_not_run",
        "parked candidate status",
    )
    closed_candidate = find_recommendation(recommendations, "loop166_code_section_foundation")
    require_equal(
        closed_candidate.get("status"),
        "closed_b1_nonfinite_current_recipe_retired",
        "closed candidate status",
    )
    next_candidate = find_recommendation(recommendations, "loop167_ember_v3_novel_delta")
    require_equal(
        next_candidate.get("status"),
        "preregistered_phase_a_static_only_phase_b_source_closure_pending",
        "next candidate status",
    )

    require_equal(
        smoke.get("decision"),
        "native_loop28_train_smoke_passed_full_loop151_blocked",
        "smoke decision",
    )
    require_equal(parity.get("decision"), "native_loop28_parity_blocked", "parity decision")
    require_equal(
        pause.get("decision"),
        "pause_native_decode_compat_without_execution_resume_loop151_f1_mainline",
        "Loop28 pause decision",
    )

    metrics = truth.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Truth manifest is missing metrics")
    research_metrics = {
        "val": candidate_metrics(metrics.get("val", {})),
        "test10k": candidate_metrics(metrics.get("test10k", {})),
        "legacy_full_test": candidate_metrics(metrics.get("legacy_full_test", {})),
    }

    capability = truth.get("capability_boundary")
    if not isinstance(capability, dict):
        raise ValueError("Truth manifest is missing capability_boundary")
    native_stage = find_stage(smoke, "loop28_native")
    require_equal(native_stage.get("raw_executable"), True, "Loop28 raw executable")
    native_artifacts = native_stage.get("artifacts")
    if not isinstance(native_artifacts, list) or not native_artifacts:
        raise ValueError("Loop28 native stage has no artifacts")
    native_artifact_index = [
        {
            "path": str(artifact.get("path")),
            "sha256": str(artifact.get("sha256")),
            "size_bytes": int(artifact.get("size_bytes", 0)),
        }
        for artifact in native_artifacts
        if isinstance(artifact, dict) and artifact.get("exists") is True
    ]

    parity_summary = parity.get("parity")
    if not isinstance(parity_summary, dict):
        raise ValueError("Native parity receipt is missing parity summary")
    claim_scope = smoke.get("claim_scope")
    if not isinstance(claim_scope, dict):
        raise ValueError("Native smoke receipt is missing claim scope")

    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    return {
        "schema": "axon_champion_registry_v1",
        "generated_at_utc": generated,
        "generation_policy": (
            "Generated only from frozen aggregate manifests and receipts. Research, native, and connected scopes are "
            "not comparable or interchangeable; no metric is hand-entered by this builder."
        ),
        "source_artifacts": evidence_index(root),
        "champions": {
            "research": {
                "candidate_id": "Loop151",
                "status": "current_strict_best",
                "protocol_role": "legacy_development_leaderboard",
                "classification_scope": capability.get("classification"),
                "metrics": research_metrics,
                "lineage": ["Loop127", "Loop130", "Loop134", "Loop136", "Loop151"],
                "raw_file_to_report_ready": bool(capability.get("raw_file_to_report_replay_ready")),
                "native_ready": bool(capability.get("native_loop151_ready")),
                "deployment_parity": False,
                "next_candidate": {
                    "candidate_id": "Loop167",
                    "status": next_candidate.get("status"),
                    "decision": next_candidate.get("phase_review", {}).get("decision"),
                },
                "recently_closed_candidate": {
                    "candidate_id": "Loop166",
                    "status": closed_candidate.get("status"),
                    "decision": closed_candidate.get("phase_review", {}).get("decision"),
                },
                "recently_parked_candidate": {
                    "candidate_id": "Loop164",
                    "status": parked_candidate.get("status"),
                    "decision": parked_candidate.get("preflight_review", {}).get("decision"),
                },
            },
            "native_offline": {
                "candidate_id": "Loop28",
                "status": "native_reference_parity_blocked_not_quality_champion",
                "protocol_role": "train_only_native_smoke",
                "raw_executable": bool(native_stage.get("raw_executable")),
                "smoke_decision": smoke.get("decision"),
                "parity_decision": parity.get("decision"),
                "parity_requested": int(parity_summary.get("requested_count", 0)),
                "parity_passed": int(parity_summary.get("passed_count", 0)),
                "quality_claim_allowed": bool(claim_scope.get("quality_claim_allowed")),
                "artifacts": native_artifact_index,
                "decode_compat_branch": "paused_without_execution",
                "deployment_parity": False,
            },
            "connected_system": {
                "candidate_id": None,
                "status": "no_connected_champion",
                "ready": bool(capability.get("connected_system_ready")),
                "certification_ready": bool(capability.get("certification_ready")),
                "blockers": list(capability.get("blockers") or []),
            },
        },
        "invariants": {
            "research_champion_is_not_native": True,
            "native_reference_is_not_research_champion": True,
            "no_connected_champion_without_full_runtime": True,
            "legacy_metrics_are_development_only": True,
            "loop164_is_parked_not_champion": True,
            "loop166_is_closed_not_champion": True,
            "loop167_is_candidate_not_champion": True,
        },
        "decision": "registry_current_scopes_separated_no_connected_champion",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the three-scope Axon champion registry.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--manifest-output-json", type=Path, required=True)
    parser.add_argument("--generated-at-utc", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    payload = build_registry(root, generated_at_utc=args.generated_at_utc)
    write_json(resolve_path(root, args.output_json), payload)
    write_json(resolve_path(root, args.manifest_output_json), payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "research": payload["champions"]["research"]["candidate_id"],
                "native_offline": payload["champions"]["native_offline"]["candidate_id"],
                "connected_system": payload["champions"]["connected_system"]["candidate_id"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
