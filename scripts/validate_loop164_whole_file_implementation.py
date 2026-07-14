#!/usr/bin/env python3
"""Fail-closed static validator for a future Loop164 whole-file expert.

The validator only inspects manifest JSON and source/configuration bytes. It
never opens a raw sample, cache, checkpoint, prediction, or split artifact.
Its pass means that a future implementation has a reviewable static contract;
it does not authorize execution or establish any model-quality claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "loop164_whole_file_residual_expert"
IMPLEMENTATION_MANIFEST_SCHEMA = "axon_loop164_whole_file_implementation_manifest_v2"

REQUIRED_SOURCE_ROLE_PATHS = {
    "controller": "scripts/run_loop164_train_oof_controller.py",
    "package_init": "src/loop164/__init__.py",
    "model": "src/loop164/whole_file_gcg.py",
    "input_loader": "src/loop164/authorized_input.py",
    "oof_protocol": "src/loop164/oof_protocol.py",
    "fusion": "src/loop164/fusion.py",
    "implementation_validator": "scripts/validate_loop164_whole_file_implementation.py",
    "training_authority_validator": "scripts/validate_loop164_training_authority.py",
    "nested_receipt_validator": "scripts/validate_loop164_nested_oof_execution_receipt.py",
    "dense_equivalence_test": "tests/test_loop164_whole_file_gcg.py",
}
CONFIG_PATH = "config/loop164_whole_file.toml"
RUNTIME_LOCK_PATH = "requirements.txt"
MISSINGNESS_REASONS = (
    "timeout",
    "unsupported",
    "read_failure",
    "parse_failure",
    "oversize",
)
IDENTITY_FEATURE_FIELDS = (
    "path",
    "filename",
    "directory",
    "extension",
    "source_sha256",
    "sample_index",
    "row_id",
    "split",
    "row_order",
    "family_id",
    "campaign_id",
    "source_group_id",
    "first_seen_time_utc",
)
STATIC_RUNTIME_ROLES = {"controller", "model", "input_loader", "oof_protocol", "fusion"}
STATIC_FEATURE_ROLES = {"model", "oof_protocol", "fusion"}
REQUIRED_TEST_CLASSES = (
    "dense_reference_equivalence",
    "gradient_reference_equivalence",
    "winner_global_position_equivalence",
    "chunk_boundary_equivalence",
    "tail_coverage",
    "all_negative_activation_winner",
    "zero_byte_and_eof_semantics",
    "two_pass_context_equivalence",
    "deterministic_repeated_run",
    "missingness_denominator",
    "noncontiguous_winner_independence",
)


@dataclass(frozen=True)
class WholeFileImplementationManifestResult:
    ready: bool
    blockers: tuple[str, ...]
    source_closure_sha256: Optional[str] = None
    config_sha256: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None
    input_contract_sha256: Optional[str] = None
    missingness_contract_sha256: Optional[str] = None
    memory_contract_sha256: Optional[str] = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Non-finite JSON value: {value}")


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _append_if_false(blockers: list[str], condition: bool, code: str) -> None:
    if not condition:
        blockers.append(code)


def _require_exact_keys(
    payload: object, expected: set[str], *, label: str, blockers: list[str]
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        blockers.append(f"{label}_not_object")
        return None
    actual = set(payload)
    if expected - actual:
        blockers.append(f"{label}_missing_fields")
    if actual - expected:
        blockers.append(f"{label}_unexpected_fields")
    return payload


def _require_no_symlink(path: Path) -> None:
    candidate = path.absolute()
    for ancestor in (candidate, *candidate.parents):
        if ancestor.is_symlink():
            raise ValueError("Symbolic-link path bindings are forbidden")


def _resolve_project_file(root: Path, value: object, *, expected_path: str) -> Path:
    text = str(value or "")
    candidate = Path(text)
    if not text or candidate.is_absolute() or candidate.as_posix() != expected_path:
        raise ValueError("Path binding is not the canonical relative path")
    resolved_root = root.resolve()
    candidate_path = (root / candidate).absolute()
    _require_no_symlink(candidate_path)
    resolved = candidate_path.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Path binding escapes the project root") from exc
    if not resolved.is_file():
        raise ValueError("Bound source file is missing")
    return resolved


def calculate_source_closure_sha256(files: Sequence[dict[str, object]]) -> str:
    normalized = [
        {
            "role": entry["role"],
            "path": entry["path"],
            "sha256": entry["sha256"],
            "bytes": entry["bytes"],
        }
        for entry in sorted(files, key=lambda item: str(item["role"]))
    ]
    return sha256_json(normalized)


def _module_aliases(relative_path: str) -> set[str]:
    path = Path(relative_path)
    if path.suffix != ".py":
        return set()
    without_suffix = path.with_suffix("").as_posix().replace("/", ".")
    aliases = {without_suffix, path.stem}
    if without_suffix.startswith("src."):
        aliases.add(without_suffix.removeprefix("src."))
    return aliases


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_nonpositive_integer_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value <= 0
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return True
    return False


def _local_imports(tree: ast.AST, *, current_path: str) -> set[str]:
    imports: set[str] = set()
    current_module = Path(current_path).with_suffix("").as_posix().replace("/", ".")
    current_package = current_module.rsplit(".", 1)[0] if "." in current_module else ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = current_package.split(".") if current_package else []
                parent_parts = package_parts[: max(0, len(package_parts) - node.level + 1)]
                prefix = ".".join(parent_parts)
                if node.module:
                    imports.add(".".join(part for part in (prefix, node.module) if part))
                else:
                    imports.update(
                        ".".join(part for part in (prefix, alias.name) if part)
                        for alias in node.names
                    )
            elif node.module:
                imports.add(node.module)
    return imports


def _validate_source_ast(
    *,
    source_paths: dict[str, Path],
    blockers: list[str],
) -> tuple[int, int, int, set[str]]:
    alias_to_role: dict[str, str] = {}
    for role, relative_path in REQUIRED_SOURCE_ROLE_PATHS.items():
        for alias in _module_aliases(relative_path):
            alias_to_role[alias] = role

    dynamic_import_count = 0
    forbidden_io_call_count = 0
    forbidden_identity_feature_count = 0
    oracle_test_names: set[str] = set()
    oracle_test_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    known_local_prefixes = ("loop164", "src.loop164", "validate_loop164", "run_loop164")
    for role, path in source_paths.items():
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            blockers.append("implementation_source_ast_unreadable")
            continue
        relative_path = REQUIRED_SOURCE_ROLE_PATHS[role]
        for imported in _local_imports(tree, current_path=relative_path):
            if imported in alias_to_role:
                continue
            if imported.startswith(known_local_prefixes):
                blockers.append("implementation_source_closure_local_import_missing")
        for node in ast.walk(tree):
            if role == "dense_equivalence_test" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                oracle_test_names.add(node.name)
                oracle_test_nodes[node.name] = node
            if isinstance(node, ast.Call):
                name = _attribute_name(node.func)
                if name in {"__import__", "exec", "eval", "compile", "import_module"}:
                    dynamic_import_count += 1
                if role in STATIC_RUNTIME_ROLES:
                    if name in {"read_bytes", "read_text", "readlines", "fromfile", "memmap", "mmap"}:
                        forbidden_io_call_count += 1
                    elif name == "read" and (
                        (not node.args and not node.keywords)
                        or (bool(node.args) and _is_nonpositive_integer_literal(node.args[0]))
                    ):
                        forbidden_io_call_count += 1
            if role in STATIC_FEATURE_ROLES and isinstance(node, ast.Constant):
                if isinstance(node.value, str) and node.value in IDENTITY_FEATURE_FIELDS:
                    forbidden_identity_feature_count += 1
    _append_if_false(
        blockers, dynamic_import_count == 0, "implementation_source_dynamic_import_detected"
    )
    _append_if_false(
        blockers, forbidden_io_call_count == 0, "implementation_source_unbounded_io_detected"
    )
    _append_if_false(
        blockers,
        forbidden_identity_feature_count == 0,
        "implementation_source_identity_feature_detected",
    )
    _append_if_false(
        blockers,
        {f"test_{name}" for name in REQUIRED_TEST_CLASSES}.issubset(oracle_test_names),
        "implementation_source_required_oracle_test_missing",
    )
    required_oracle_nodes = [oracle_test_nodes.get(f"test_{name}") for name in REQUIRED_TEST_CLASSES]
    _append_if_false(
        blockers,
        all(
            node is not None
            and any(isinstance(descendant, ast.Call) for descendant in ast.walk(node))
            and any(
                isinstance(descendant, ast.Assert)
                and not (
                    isinstance(descendant.test, ast.Constant)
                    and descendant.test.value is True
                )
                for descendant in ast.walk(node)
            )
            for node in required_oracle_nodes
        ),
        "implementation_source_oracle_test_body_invalid",
    )
    return dynamic_import_count, forbidden_io_call_count, forbidden_identity_feature_count, oracle_test_names


def _validate_source_closure(
    payload: object, *, root: Path, blockers: list[str]
) -> tuple[Optional[str], dict[str, Path]]:
    closure = _require_exact_keys(
        payload, {"files", "closure_sha256"}, label="implementation_source_closure", blockers=blockers
    )
    if closure is None:
        return None, {}
    entries = closure.get("files")
    if not isinstance(entries, list) or len(entries) != len(REQUIRED_SOURCE_ROLE_PATHS):
        blockers.append("implementation_source_closure_file_set_invalid")
        return None, {}
    by_role: dict[str, dict[str, Any]] = {}
    for entry in entries:
        record = _require_exact_keys(
            entry,
            {"role", "path", "sha256", "bytes"},
            label="implementation_source_closure_file",
            blockers=blockers,
        )
        if record is None:
            continue
        role = record.get("role")
        if not isinstance(role, str) or role in by_role:
            blockers.append("implementation_source_closure_role_invalid")
            continue
        by_role[role] = record
    if set(by_role) != set(REQUIRED_SOURCE_ROLE_PATHS):
        blockers.append("implementation_source_closure_role_set_invalid")
        return None, {}

    source_paths: dict[str, Path] = {}
    normalized_entries: list[dict[str, object]] = []
    for role, expected_path in REQUIRED_SOURCE_ROLE_PATHS.items():
        entry = by_role[role]
        try:
            path = _resolve_project_file(root, entry.get("path"), expected_path=expected_path)
            expected_size = path.stat().st_size
            actual_sha256 = sha256_file(path)
        except (OSError, ValueError):
            blockers.append("implementation_source_closure_path_invalid")
            continue
        _append_if_false(
            blockers,
            isinstance(entry.get("bytes"), int) and entry.get("bytes") == expected_size,
            "implementation_source_closure_size_mismatch",
        )
        _append_if_false(
            blockers,
            is_sha256(entry.get("sha256"))
            and str(entry.get("sha256")).casefold() == actual_sha256,
            "implementation_source_closure_sha256_mismatch",
        )
        source_paths[role] = path
        normalized_entries.append(
            {
                "role": role,
                "path": expected_path,
                "sha256": actual_sha256,
                "bytes": expected_size,
            }
        )
    if len(source_paths) != len(REQUIRED_SOURCE_ROLE_PATHS):
        return None, source_paths
    closure_sha256 = calculate_source_closure_sha256(normalized_entries)
    _append_if_false(
        blockers,
        is_sha256(closure.get("closure_sha256"))
        and str(closure.get("closure_sha256")).casefold() == closure_sha256,
        "implementation_source_closure_fingerprint_mismatch",
    )
    return closure_sha256, source_paths


def _validate_bound_file(
    payload: object,
    *,
    root: Path,
    expected_path: str,
    label: str,
    blockers: list[str],
) -> Optional[str]:
    binding = _require_exact_keys(payload, {"path", "sha256"}, label=label, blockers=blockers)
    if binding is None:
        return None
    try:
        path = _resolve_project_file(root, binding.get("path"), expected_path=expected_path)
        digest = sha256_file(path)
    except (OSError, ValueError):
        blockers.append(f"{label}_path_invalid")
        return None
    _append_if_false(
        blockers,
        is_sha256(binding.get("sha256")) and str(binding.get("sha256")).casefold() == digest,
        f"{label}_sha256_mismatch",
    )
    return digest


def _validate_model_contract(payload: object, blockers: list[str]) -> None:
    contract = _require_exact_keys(
        payload,
        {
            "architecture",
            "prefix_only",
            "input_representation",
            "vocab_size",
            "pad_token",
            "raw_byte_offset",
            "eof_policy",
            "archive_policy",
            "pooling_equivalence_mode",
            "context_pass_count",
            "winner_initialization",
            "dense_equivalence_policy",
        },
        label="implementation_model_contract",
        blockers=blockers,
    )
    if contract is None:
        return
    _append_if_false(
        blockers,
        contract == {
            "architecture": "malconv2_style_low_memory_gcg",
            "prefix_only": False,
            "input_representation": "raw_byte_plus_one_reserved_pad_token",
            "vocab_size": 257,
            "pad_token": 0,
            "raw_byte_offset": 1,
            "eof_policy": "explicit_valid_length_mask",
            "archive_policy": "unsupported_explicit_missingness",
            "pooling_equivalence_mode": "exact_independent_regions",
            "context_pass_count": 2,
            "winner_initialization": "negative_infinity",
            "dense_equivalence_policy": "required_before_a2_execution",
        },
        "implementation_model_contract_invalid",
    )


def _validate_input_contract(payload: object, blockers: list[str]) -> Optional[str]:
    contract = _require_exact_keys(
        payload,
        {
            "allowed_split_roles",
            "protected_input_open_policy",
            "whole_file_input_policy",
            "supported_file_policy",
            "oversize_policy",
            "padding_policy",
            "identity_feature_count",
            "forbidden_identity_fields",
        },
        label="implementation_input_contract",
        blockers=blockers,
    )
    if contract is None:
        return None
    _append_if_false(
        blockers,
        contract == {
            "allowed_split_roles": ["train_anchor", "train_oof"],
            "protected_input_open_policy": "after_final_lease_only",
            "whole_file_input_policy": "all_bytes_chunked_no_silent_truncation",
            "supported_file_policy": "stream_all_bytes",
            "oversize_policy": "explicit_missingness_no_prefix_fallback",
            "padding_policy": "reserved_pad_token_or_explicit_length_mask",
            "identity_feature_count": 0,
            "forbidden_identity_fields": list(IDENTITY_FEATURE_FIELDS),
        },
        "implementation_input_contract_invalid",
    )
    return sha256_json(contract)


def _validate_timeout_contract(payload: object, blockers: list[str]) -> None:
    contract = _require_exact_keys(
        payload,
        {
            "per_file_timeout_policy",
            "worker_failure_policy",
            "oom_policy",
            "run_failure_policy",
            "retry_policy",
        },
        label="implementation_timeout_contract",
        blockers=blockers,
    )
    if contract is None:
        return
    _append_if_false(
        blockers,
        contract == {
            "per_file_timeout_policy": "neutral_score_deterministic_uncertainty_explicit_missingness",
            "worker_failure_policy": "abort_no_execution_receipt",
            "oom_policy": "abort_no_execution_receipt",
            "run_failure_policy": "abort_no_execution_receipt",
            "retry_policy": "no_silent_retry",
        },
        "implementation_timeout_contract_invalid",
    )


def _validate_missingness_contract(payload: object, blockers: list[str]) -> Optional[str]:
    contract = _require_exact_keys(
        payload,
        {
            "reasons",
            "denominator_policy",
            "dropped_rows",
            "score_fallback_policy",
            "nonfinite_policy",
        },
        label="implementation_missingness_contract",
        blockers=blockers,
    )
    if contract is None:
        return None
    _append_if_false(
        blockers,
        contract == {
            "reasons": list(MISSINGNESS_REASONS),
            "denominator_policy": "every_eligible_row_exactly_once",
            "dropped_rows": 0,
            "score_fallback_policy": "neutral_not_loop151_substitution",
            "nonfinite_policy": "abort_no_execution_receipt",
        },
        "implementation_missingness_contract_invalid",
    )
    return sha256_json(contract)


def _validate_memory_contract(payload: object, blockers: list[str]) -> Optional[str]:
    contract = _require_exact_keys(
        payload,
        {
            "chunk_bytes",
            "receptive_field_bytes",
            "output_stride_bytes",
            "overlap_bytes",
            "top_k_chunks",
            "max_supported_file_bytes",
            "max_inflight_samples",
            "max_workers",
            "prefetch_factor",
            "max_host_buffer_bytes",
            "max_device_buffer_bytes",
            "max_cpu_memory_bytes",
            "max_gpu_memory_bytes",
            "pass_count",
            "tail_policy",
            "bounded_read_bytes",
        },
        label="implementation_memory_contract",
        blockers=blockers,
    )
    if contract is None:
        return None
    positive_names = {
        "chunk_bytes",
        "receptive_field_bytes",
        "output_stride_bytes",
        "top_k_chunks",
        "max_supported_file_bytes",
        "max_inflight_samples",
        "max_workers",
        "prefetch_factor",
        "max_host_buffer_bytes",
        "max_device_buffer_bytes",
        "max_cpu_memory_bytes",
        "max_gpu_memory_bytes",
        "bounded_read_bytes",
    }
    _append_if_false(
        blockers,
        all(isinstance(contract.get(name), int) and int(contract[name]) > 0 for name in positive_names),
        "implementation_memory_contract_positive_values_invalid",
    )
    try:
        shape_valid = (
            int(contract["receptive_field_bytes"]) <= int(contract["chunk_bytes"])
            and int(contract["output_stride_bytes"]) <= int(contract["chunk_bytes"])
            and int(contract["output_stride_bytes"])
            <= int(contract["receptive_field_bytes"])
            and int(contract["overlap_bytes"])
            == int(contract["receptive_field_bytes"])
            - int(contract["output_stride_bytes"])
            and int(contract["bounded_read_bytes"]) == int(contract["chunk_bytes"])
            and int(contract["max_supported_file_bytes"]) >= int(contract["chunk_bytes"])
            and int(contract["max_host_buffer_bytes"])
            >= int(contract["top_k_chunks"]) * int(contract["chunk_bytes"])
            and contract["pass_count"] == 2
            and contract["tail_policy"] == "include_tail_or_explicit_missingness_no_drop"
        )
    except (KeyError, TypeError, ValueError):
        shape_valid = False
    _append_if_false(blockers, shape_valid, "implementation_memory_contract_bounds_invalid")
    return sha256_json(contract)


def _validate_static_safety_audit(
    payload: object,
    *,
    source_counts: tuple[int, int, int, set[str]],
    blockers: list[str],
) -> None:
    audit = _require_exact_keys(
        payload,
        {
            "source_closure_complete",
            "symlink_free",
            "dynamic_import_count",
            "forbidden_io_call_count",
            "forbidden_identity_feature_count",
            "required_test_classes",
        },
        label="implementation_static_safety_audit",
        blockers=blockers,
    )
    if audit is None:
        return
    dynamic_count, io_count, identity_count, _ = source_counts
    _append_if_false(
        blockers,
        audit.get("source_closure_complete") is True
        and audit.get("symlink_free") is True
        and audit.get("dynamic_import_count") == dynamic_count == 0
        and audit.get("forbidden_io_call_count") == io_count == 0
        and audit.get("forbidden_identity_feature_count") == identity_count == 0
        and tuple(audit.get("required_test_classes") or []) == REQUIRED_TEST_CLASSES,
        "implementation_static_safety_audit_invalid",
    )


def validate_implementation_manifest_payload(
    payload: object, *, root: Path = PROJECT_ROOT
) -> WholeFileImplementationManifestResult:
    """Validate a manifest object without opening any protected ML/data input."""

    blockers: list[str] = []
    manifest = _require_exact_keys(
        payload,
        {
            "schema",
            "loop_id",
            "review_state",
            "claim_scope",
            "source_closure",
            "model_contract",
            "config",
            "runtime_lock",
            "input_contract",
            "timeout_contract",
            "missingness_contract",
            "memory_contract",
            "static_safety_audit",
            "decision",
        },
        label="implementation_manifest",
        blockers=blockers,
    )
    if manifest is None:
        return WholeFileImplementationManifestResult(False, tuple(sorted(set(blockers))))
    _append_if_false(
        blockers,
        manifest.get("schema") == IMPLEMENTATION_MANIFEST_SCHEMA
        and manifest.get("loop_id") == LOOP_ID
        and manifest.get("review_state") == "reviewed"
        and manifest.get("claim_scope") == "static_source_contract_no_data_or_model_execution"
        and manifest.get("decision") == "reviewed_static_only_execution_not_authorized",
        "implementation_manifest_identity_or_scope_invalid",
    )

    source_closure_sha256, source_paths = _validate_source_closure(
        manifest.get("source_closure"), root=root.resolve(), blockers=blockers
    )
    source_counts = _validate_source_ast(source_paths=source_paths, blockers=blockers)
    config_sha256 = _validate_bound_file(
        manifest.get("config"),
        root=root.resolve(),
        expected_path=CONFIG_PATH,
        label="implementation_config",
        blockers=blockers,
    )
    runtime_lock_sha256 = _validate_bound_file(
        manifest.get("runtime_lock"),
        root=root.resolve(),
        expected_path=RUNTIME_LOCK_PATH,
        label="implementation_runtime_lock",
        blockers=blockers,
    )
    _validate_model_contract(manifest.get("model_contract"), blockers)
    input_contract_sha256 = _validate_input_contract(manifest.get("input_contract"), blockers)
    _validate_timeout_contract(manifest.get("timeout_contract"), blockers)
    missingness_contract_sha256 = _validate_missingness_contract(
        manifest.get("missingness_contract"), blockers
    )
    memory_contract_sha256 = _validate_memory_contract(manifest.get("memory_contract"), blockers)
    _validate_static_safety_audit(
        manifest.get("static_safety_audit"), source_counts=source_counts, blockers=blockers
    )
    return WholeFileImplementationManifestResult(
        not blockers,
        tuple(sorted(set(blockers))),
        source_closure_sha256=source_closure_sha256,
        config_sha256=config_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        input_contract_sha256=input_contract_sha256,
        missingness_contract_sha256=missingness_contract_sha256,
        memory_contract_sha256=memory_contract_sha256,
    )


def validate_implementation_manifest_path(
    manifest_json: Path, *, root: Path = PROJECT_ROOT
) -> WholeFileImplementationManifestResult:
    try:
        payload = read_json_object(manifest_json)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return WholeFileImplementationManifestResult(False, ("implementation_manifest_unreadable",))
    return validate_implementation_manifest_payload(payload, root=root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a future Loop164 whole-file implementation manifest without execution."
    )
    parser.add_argument("--check", action="store_true", help="Validate only; never write or execute.")
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.check:
        raise SystemExit("Only --check is available; this validator never generates an artifact.")
    result = validate_implementation_manifest_path(args.manifest_json, root=args.project_root)
    print(json.dumps({"ready": result.ready, "blockers": list(result.blockers)}, indent=2))
    return 0 if result.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
