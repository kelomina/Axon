from __future__ import annotations

import hashlib
import json
from pathlib import Path

from validate_loop164_whole_file_implementation import (
    CONFIG_PATH,
    IDENTITY_FEATURE_FIELDS,
    IMPLEMENTATION_MANIFEST_SCHEMA,
    LOOP_ID,
    MISSINGNESS_REASONS,
    REQUIRED_SOURCE_ROLE_PATHS,
    REQUIRED_TEST_CLASSES,
    RUNTIME_LOCK_PATH,
    calculate_source_closure_sha256,
    sha256_file,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_whole_file_implementation_fixture(root: Path, manifest_path: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'synthetic'\n", encoding="utf-8")
    (root / CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / CONFIG_PATH).write_text("[loop164]\nchunk_bytes = 4096\n", encoding="utf-8")
    (root / RUNTIME_LOCK_PATH).write_text("torch==synthetic\n", encoding="utf-8")
    source_text = {
        "controller": (
            "from validate_loop164_training_authority import validate_training_authority\n"
            "def main():\n    return validate_training_authority\n"
        ),
        "package_init": "",
        "model": "def encode(chunk):\n    return chunk\n",
        "input_loader": "def read_chunk(handle, chunk_bytes):\n    return handle.read(chunk_bytes)\n",
        "oof_protocol": "from .whole_file_gcg import encode\n",
        "fusion": "def combine(scores):\n    return scores\n",
        "implementation_validator": "def validate():\n    return True\n",
        "training_authority_validator": "def validate_training_authority():\n    return True\n",
        "nested_receipt_validator": "def validate_receipt():\n    return True\n",
        "dense_equivalence_test": "def _oracle():\n    return 1\n\n"
        + "\n".join(
            f"def test_{test_name}():\n    assert _oracle() == 1\n"
            for test_name in REQUIRED_TEST_CLASSES
        ),
    }
    entries = []
    for role, relative_path in REQUIRED_SOURCE_ROLE_PATHS.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source_text[role], encoding="utf-8")
        entries.append(
            {
                "role": role,
                "path": relative_path,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema": IMPLEMENTATION_MANIFEST_SCHEMA,
        "loop_id": LOOP_ID,
        "review_state": "reviewed",
        "claim_scope": "static_source_contract_no_data_or_model_execution",
        "source_closure": {
            "files": entries,
            "closure_sha256": calculate_source_closure_sha256(entries),
        },
        "model_contract": {
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
        "config": {"path": CONFIG_PATH, "sha256": sha256_file(root / CONFIG_PATH)},
        "runtime_lock": {"path": RUNTIME_LOCK_PATH, "sha256": sha256_file(root / RUNTIME_LOCK_PATH)},
        "input_contract": {
            "allowed_split_roles": ["train_anchor", "train_oof"],
            "protected_input_open_policy": "after_final_lease_only",
            "whole_file_input_policy": "all_bytes_chunked_no_silent_truncation",
            "supported_file_policy": "stream_all_bytes",
            "oversize_policy": "explicit_missingness_no_prefix_fallback",
            "padding_policy": "reserved_pad_token_or_explicit_length_mask",
            "identity_feature_count": 0,
            "forbidden_identity_fields": list(IDENTITY_FEATURE_FIELDS),
        },
        "timeout_contract": {
            "per_file_timeout_policy": "neutral_score_deterministic_uncertainty_explicit_missingness",
            "worker_failure_policy": "abort_no_execution_receipt",
            "oom_policy": "abort_no_execution_receipt",
            "run_failure_policy": "abort_no_execution_receipt",
            "retry_policy": "no_silent_retry",
        },
        "missingness_contract": {
            "reasons": list(MISSINGNESS_REASONS),
            "denominator_policy": "every_eligible_row_exactly_once",
            "dropped_rows": 0,
            "score_fallback_policy": "neutral_not_loop151_substitution",
            "nonfinite_policy": "abort_no_execution_receipt",
        },
        "memory_contract": {
            "chunk_bytes": 4096,
            "receptive_field_bytes": 512,
            "output_stride_bytes": 256,
            "overlap_bytes": 256,
            "top_k_chunks": 8,
            "max_supported_file_bytes": 16 * 1024 * 1024,
            "max_inflight_samples": 1,
            "max_workers": 1,
            "prefetch_factor": 1,
            "max_host_buffer_bytes": 64 * 1024,
            "max_device_buffer_bytes": 256 * 1024,
            "max_cpu_memory_bytes": 512 * 1024 * 1024,
            "max_gpu_memory_bytes": 2 * 1024 * 1024 * 1024,
            "pass_count": 2,
            "tail_policy": "include_tail_or_explicit_missingness_no_drop",
            "bounded_read_bytes": 4096,
        },
        "static_safety_audit": {
            "source_closure_complete": True,
            "symlink_free": True,
            "dynamic_import_count": 0,
            "forbidden_io_call_count": 0,
            "forbidden_identity_feature_count": 0,
            "required_test_classes": list(REQUIRED_TEST_CLASSES),
        },
        "decision": "reviewed_static_only_execution_not_authorized",
    }
    write_json(manifest_path, payload)
    return payload


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()
