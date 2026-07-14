#!/usr/bin/env python3
"""Freeze the completed Loop28 parity diagnostic evidence closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

SCHEMA = "axon_loop28_parity_post_diagnostic_manifest_v1"
LOOP_ID = "p0_loop28_parity_diagnostic_001"
DEFAULT_OUTPUT = Path(
    "manifests/roadmap_9997/p0_loop28_parity_diagnostic/post_diagnostic_manifest.json"
)


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    role: str
    path: Path
    expected_sha256: Optional[str]


ARTIFACTS = (
    ArtifactSpec(
        "historical_truth",
        "historical_parent_evidence",
        Path("manifests/roadmap_9997/p0_truth_freeze/loop151_truth_manifest.json"),
        "174861be850a681025a7040798c59b7157cc67ab5503437088359692dad5659d",
    ),
    ArtifactSpec(
        "prereg_authorization",
        "diagnostic_preregistration",
        Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/authorization.json"),
        "3de5c204c37e300c552eba16fd2b633fda5b8bcc68cbdb2b58a7d00b24cb201f",
    ),
    ArtifactSpec(
        "implementation_manifest",
        "hash_only_implementation_closure",
        Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/implementation_manifest.json"),
        "a581f00503d62117576e0f2721de72da049ea8fe7786788db63a653ace207a49",
    ),
    ArtifactSpec(
        "run_authorization",
        "bounded_run_authorization",
        Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_authorization.json"),
        "aa07e16765e23b6b0e2e07dad5ffa06ed2036d064d25940b4c0f36f6f4eda359",
    ),
    ArtifactSpec(
        "consumed_lease",
        "one_shot_attempt_lease",
        Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_attempt.final.json"),
        "69d2fbbf1a9e31e542dd61541befc0b11c973c9023e187f8b0e87a9825f91b19",
    ),
    ArtifactSpec(
        "final_receipt",
        "diagnostic_result",
        Path("reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.final.json"),
        "de8f0c5885df08646298f67f59b5427696252f7cb921d84eeb6527bef6878bc7",
    ),
    ArtifactSpec(
        "experiment_journal",
        "durable_experiment_record",
        Path("reports/hard_family_finetune/experiment_journal.md"),
        None,
    ),
    ArtifactSpec(
        "recommendations",
        "owner_facing_research_status",
        Path("docs/ml_improvement_recommendations.md"),
        None,
    ),
)


class ManifestError(RuntimeError):
    pass


def _validate_generated_at(value: str) -> str:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ManifestError("generated_at_utc must be RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ManifestError("generated_at_utc must include a timezone")
    return value


def _sha256_stable_file(project_root: Path, relative_path: Path) -> tuple[int, str]:
    root = project_root.resolve(strict=True)
    path = project_root / relative_path
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ManifestError(f"Artifact must be a regular non-symlink file: {relative_path}")
    resolved = path.resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise ManifestError(f"Artifact escapes project root: {relative_path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    after = path.lstat()
    # 哈希前后同时核对文件身份，避免把并发替换误记成一个稳定 artifact。
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ManifestError(f"Artifact changed while hashing: {relative_path}")
    return before.st_size, digest.hexdigest()


def build_manifest(
    project_root: Path,
    *,
    generated_at_utc: str,
    artifacts: Sequence[ArtifactSpec] = ARTIFACTS,
) -> dict:
    generated_at_utc = _validate_generated_at(generated_at_utc)
    rows = []
    blockers = []
    verified_predeclared = 0
    for spec in artifacts:
        try:
            size_bytes, sha256 = _sha256_stable_file(project_root, spec.path)
        except (FileNotFoundError, ManifestError, OSError) as exc:
            blockers.append({"artifact": spec.name, "reason": str(exc)})
            continue
        expected_match = None
        if spec.expected_sha256 is not None:
            expected_match = sha256 == spec.expected_sha256
            if expected_match:
                verified_predeclared += 1
            else:
                blockers.append(
                    {
                        "artifact": spec.name,
                        "reason": "predeclared_sha256_mismatch",
                        "expected_sha256": spec.expected_sha256,
                        "actual_sha256": sha256,
                    }
                )
        rows.append(
            {
                "name": spec.name,
                "role": spec.role,
                "path": spec.path.as_posix(),
                "required": True,
                "expected_sha256": spec.expected_sha256,
                "exists": True,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "expected_sha256_match": expected_match,
            }
        )

    predeclared_count = sum(spec.expected_sha256 is not None for spec in artifacts)
    manifest = {
        "schema": SCHEMA,
        "loop_id": LOOP_ID,
        "generated_at_utc": generated_at_utc,
        "contract": {
            "operation": "opaque_file_stat_and_streaming_sha256_only",
            "bound_artifact_payloads_parsed": False,
            "manifest_self_hashed": False,
            "output_replace_allowed": False,
            "supplemental_artifacts_allowed": False,
        },
        "claim_scope": {
            "diagnostic_closure_only": True,
            "first_divergence_stage": "feature_extraction",
            "quality_claim_allowed": False,
            "parity_claim_allowed": False,
            "certification_claim_allowed": False,
            "remediation_execution_authorized": False,
        },
        "artifacts": rows,
        "integrity": {
            "artifact_count": len(rows),
            "required_artifact_count": len(artifacts),
            "present_required_artifact_count": len(rows),
            "predeclared_sha256_count": predeclared_count,
            "verified_predeclared_sha256_count": verified_predeclared,
            "blockers": blockers,
        },
        "decision": (
            "diagnostic_closure_frozen_remediation_requires_new_authorization"
            if not blockers and len(rows) == len(artifacts)
            else "diagnostic_closure_blocked"
        ),
    }
    return manifest


def write_manifest_exclusive(output_path: Path, manifest: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ManifestError(f"Output already exists: {output_path}") from exc


def verify_manifest(
    project_root: Path,
    manifest_path: Path,
    *,
    artifacts: Sequence[ArtifactSpec] = ARTIFACTS,
) -> dict:
    payload = json.loads((project_root / manifest_path).read_text(encoding="utf-8"))
    rebuilt = build_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc", "")),
        artifacts=artifacts,
    )
    if payload != rebuilt:
        raise ManifestError("Manifest does not match the current fixed artifact closure")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve(strict=True)
    if args.verify:
        manifest = verify_manifest(project_root, args.output)
    else:
        if not args.generated_at_utc:
            raise ManifestError("--generated-at-utc is required when building")
        manifest = build_manifest(project_root, generated_at_utc=args.generated_at_utc)
        if manifest["integrity"]["blockers"]:
            raise ManifestError("Diagnostic closure contains blockers")
        write_manifest_exclusive(project_root / args.output, manifest)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "artifact_count": manifest["integrity"]["artifact_count"],
                "blocker_count": len(manifest["integrity"]["blockers"]),
                "decision": manifest["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
