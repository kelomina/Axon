"""Verify the frozen KVD C ABI required by future Axon champion DLLs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence

EXPECTED_EXPORTS = (
    "axon_predict_json",
    "axon_string_free",
    "axon_version",
    "kvd_create",
    "kvd_destroy",
    "kvd_extract_pe_features",
    "kvd_extract_pe_features_batch",
    "kvd_free",
    "kvd_get_pe_feature_dimension",
    "kvd_parity_diagnostics_path_v1",
    "kvd_scan_bytes",
    "kvd_scan_path",
    "kvd_scan_paths",
    "kvd_signature_flush",
    "kvd_train_from_path",
    "kvd_train_path",
    "kvd_train_paths",
    "kvd_validate_models",
)
EXPECTED_CONFIG_FIELDS = (
    "model_path",
    "model_normal_path",
    "model_packed_path",
    "family_classifier_json_path",
    "allowed_scan_root",
    "max_file_size",
    "prediction_threshold",
    "onnx_model_path",
    "onnx_model_normal_path",
    "onnx_model_packed_path",
    "stage2_model_json_path",
    "archive_scanner_path",
    "scan_nested",
)


class AbiVerificationError(ValueError):
    """Raised when a DLL no longer satisfies the frozen C ABI."""


def parse_dumpbin_exports(output: str) -> tuple[tuple[int, str], ...]:
    """Extract ordinal/name pairs from `dumpbin /exports` output."""
    pattern = re.compile(r"^\s*(\d+)\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)\s*$", re.MULTILINE)
    return tuple((int(ordinal), name) for ordinal, name in pattern.findall(output))


def verify_header(header: Path) -> None:
    source = header.read_text(encoding="utf-8")
    if "#define KVD_CALL __cdecl" not in source or 'extern "C"' not in source:
        raise AbiVerificationError("header no longer freezes the Windows __cdecl C ABI")
    match = re.search(r"typedef struct kvd_config \{(?P<body>.*?)\} kvd_config;", source, flags=re.DOTALL)
    if match is None:
        raise AbiVerificationError("kvd_config declaration is missing")
    fields = tuple(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*;", match.group("body")))
    if fields != EXPECTED_CONFIG_FIELDS:
        raise AbiVerificationError("kvd_config field order or layout changed")
    for symbol in EXPECTED_EXPORTS:
        if symbol not in source:
            raise AbiVerificationError(f"header no longer declares {symbol}")


def verify_exports(exports: Sequence[tuple[int, str]]) -> None:
    expected = tuple(enumerate(EXPECTED_EXPORTS, start=1))
    if tuple(exports) != expected:
        raise AbiVerificationError("DLL export names or ordinals changed")


def run_dumpbin(dumpbin: Path, dll: Path) -> str:
    completed = subprocess.run(
        [str(dumpbin), "/exports", str(dll)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AbiVerificationError(completed.stderr.strip() or "dumpbin /exports failed")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--dumpbin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    verify_header(arguments.header)
    exports = parse_dumpbin_exports(run_dumpbin(arguments.dumpbin, arguments.dll))
    verify_exports(exports)
    payload = {
        "schema": "axon_kvd_dll_abi_verification_v1",
        "header": str(arguments.header),
        "dll": str(arguments.dll),
        "calling_convention": "__cdecl",
        "exports": [{"ordinal": ordinal, "name": name} for ordinal, name in exports],
        "kvd_config_x64_size": 96,
        "kvd_parity_diagnostics_options_v1_x64_size": 56,
        "decision": "abi_verified_not_a_model_quality_or_loop151_runtime_claim",
    }
    arguments.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
