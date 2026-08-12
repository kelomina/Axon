"""Fail-closed protocol and evidence bindings for Loop175 Phase B."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROTOCOL_RELATIVE_PATH = Path(
    "manifests/roadmap_9997/loop175_section_region_moe/phase_b_protocol.json"
)
PROTOCOL_SCHEMA = "axon_loop175_phase_b_protocol_v1"
EXPECTED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "loop_id",
        "status",
        "claim_scope",
        "evidence_bindings",
        "inputs",
        "region_cache",
        "arms",
        "hgb",
        "extreme_weights",
        "epoch_selection",
        "training",
        "seed41_gate",
        "resource_contract",
        "forbidden",
        "ready_for",
    }
)


class PhaseBContractError(RuntimeError):
    """Raised when a Phase-B input or evidence binding drifts."""


@dataclass(frozen=True, slots=True)
class LoadedPhaseBProtocol:
    payload: Mapping[str, Any]
    path: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseBContractError(f"Phase-B JSON repeats key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise PhaseBContractError(f"Phase-B JSON contains non-finite value: {value}")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseBContractError(f"Cannot load Phase-B JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise PhaseBContractError(f"Phase-B JSON root is not an object: {path}")
    return payload


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseBContractError(f"{name} must be an object")
    return value


def validate_protocol_structure(payload: Mapping[str, Any]) -> None:
    if set(payload) != EXPECTED_TOP_LEVEL_FIELDS:
        raise PhaseBContractError("Phase-B protocol top-level fields drifted")
    if payload.get("schema") != PROTOCOL_SCHEMA or payload.get("loop_id") != "Loop175":
        raise PhaseBContractError("Phase-B protocol identity drifted")
    if payload.get("status") != "protocol_frozen_implementation_and_source_closure_required":
        raise PhaseBContractError("Phase-B protocol status drifted")

    arms = _mapping(payload.get("arms"), name="arms")
    if set(arms) != {"A", "B", "C", "D", "E"}:
        raise PhaseBContractError("Phase-B arm set drifted")
    b0 = _mapping(_mapping(payload.get("inputs"), name="inputs").get("b0_cache"), name="b0_cache")
    if b0.get("shape") != [20_000, 571] or b0.get("missing_indicators_used") is not False:
        raise PhaseBContractError("Phase-B B0 values-only contract drifted")
    canonical_split = _mapping(
        _mapping(payload.get("inputs"), name="inputs").get("canonical_split"),
        name="canonical_split",
    )
    if (
        canonical_split.get("train_rows") != 20_000
        or canonical_split.get("train_prefix_bytes") != 4_120_895
        or canonical_split.get("train_prefix_sha256")
        != "dfbad6994605aa0fd9b7fa049b19cd87f15e50e37490a60efc43696c540dd54a"
    ):
        raise PhaseBContractError("Phase-B canonical Train prefix contract drifted")

    training = _mapping(payload.get("training"), name="training")
    if (
        training.get("seeds") != [41, 42, 43]
        or training.get("seed41_only_until_gate") is not True
        or training.get("threshold") != 0.5
        or training.get("hard_decision") != "probability_strictly_greater_than_0.5"
    ):
        raise PhaseBContractError("Phase-B training decision contract drifted")
    weights = _mapping(payload.get("extreme_weights"), name="extreme_weights")
    if (
        weights.get("residual_source") != "B0_only_outer_fit_inner_OOF"
        or weights.get("near_boundary_inclusive") != [0.35, 0.65]
        or weights.get("error_weight") != 8.0
        or weights.get("search_allowed") is not False
    ):
        raise PhaseBContractError("Phase-B extreme-weight contract drifted")

    ready = _mapping(payload.get("ready_for"), name="ready_for")
    if ready.get("implementation") is not True or any(
        value is not False for key, value in ready.items() if key != "implementation"
    ):
        raise PhaseBContractError("Phase-B readiness must remain implementation-only")


def load_phase_b_protocol(
    project_root: Path | str,
    protocol_relative_path: Path = PROTOCOL_RELATIVE_PATH,
) -> LoadedPhaseBProtocol:
    root = Path(project_root).resolve(strict=True)
    protocol_path = (root / protocol_relative_path).resolve(strict=True)
    try:
        protocol_path.relative_to(root)
    except ValueError as error:
        raise PhaseBContractError("Phase-B protocol escapes the project root") from error
    if not protocol_path.is_file() or protocol_path.is_symlink():
        raise PhaseBContractError("Phase-B protocol must be a regular non-symlink file")
    payload = load_json_object(protocol_path)
    validate_protocol_structure(payload)
    return LoadedPhaseBProtocol(payload, protocol_path, sha256_file(protocol_path))


def validate_bound_evidence(project_root: Path | str, protocol: LoadedPhaseBProtocol) -> None:
    root = Path(project_root).resolve(strict=True)
    bindings = _mapping(protocol.payload.get("evidence_bindings"), name="evidence_bindings")
    if set(bindings) != {"proposal", "phase0_receipt", "phase_a_receipt", "execution_plan"}:
        raise PhaseBContractError("Phase-B evidence binding set drifted")
    for name, raw_binding in bindings.items():
        binding = _mapping(raw_binding, name=f"evidence_bindings.{name}")
        if set(binding) != {"path", "sha256"}:
            raise PhaseBContractError(f"Phase-B evidence binding fields drifted: {name}")
        path = (root / str(binding["path"])).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise PhaseBContractError(f"Phase-B evidence path escapes root: {name}") from error
        if not path.is_file() or path.is_symlink() or sha256_file(path) != binding["sha256"]:
            raise PhaseBContractError(f"Phase-B evidence binding drifted: {name}")

    phase_a_binding = _mapping(bindings["phase_a_receipt"], name="phase_a_receipt")
    phase0_binding = _mapping(bindings["phase0_receipt"], name="phase0_receipt")
    phase0 = load_json_object(root / str(phase0_binding["path"]))
    sources = phase0.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PhaseBContractError("Phase-0 receipt has no source closure")
    for raw_source in sources:
        source = _mapping(raw_source, name="phase0.sources")
        if set(source) != {"path", "sha256", "bytes"}:
            raise PhaseBContractError("Phase-0 source binding fields drifted")
        source_path = (root / str(source["path"])).resolve(strict=True)
        try:
            source_path.relative_to(root)
        except ValueError as error:
            raise PhaseBContractError("Phase-0 source path escapes root") from error
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or source_path.stat().st_size != source["bytes"]
            or sha256_file(source_path) != source["sha256"]
        ):
            raise PhaseBContractError(f"Phase-0 source closure drifted: {source['path']}")
    phase_a = load_json_object(root / str(phase_a_binding["path"]))
    if (
        phase_a.get("decision") != "phase_a_pass_seed41_implementation_may_begin"
        or phase_a.get("training_runs") != 0
        or phase_a.get("val_test_or_full_rows_opened") != 0
    ):
        raise PhaseBContractError("Phase-A receipt does not authorize Phase-B implementation")


def write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise PhaseBContractError(f"Refusing to overwrite Phase-B artifact: {path}") from error
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
