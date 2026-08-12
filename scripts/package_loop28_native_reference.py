#!/usr/bin/env python3
"""Package the runnable KVD-ABI Axon native reference without quality relabeling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

PACKAGE_SCHEMA = "axon_native_reference_package_v1"
ABI_SCHEMA = "axon_kvd_dll_abi_verification_v1"
ASSETS = (
    ("bin/axon_native_scanner.dll", "tools/axon_onnx_dll/build/bin/Release/axon_onnx_predict.dll"),
    ("bin/axon_native_scanner.lib", "tools/axon_onnx_dll/build/lib/Release/axon_onnx_predict.lib"),
    ("bin/axon_native_selftest.exe", "tools/axon_onnx_dll/build/bin/Release/axon_onnx_selftest.exe"),
    ("bin/onnxruntime.dll", "tools/axon_onnx_dll/build/bin/Release/onnxruntime.dll"),
    ("include/axon_native_scanner.h", "tools/axon_onnx_dll/include/axon_onnx_predict.h"),
    ("models/axon_loop28_base.onnx", "models/random_20w_8192/axon_loop28_base.onnx"),
    ("models/axon_loop28_base.onnx.data", "models/random_20w_8192/axon_loop28_base.onnx.data"),
    ("models/loop28_stage2_hgb.json", "models/random_20w_8192/loop28_stage2_hgb.json"),
    ("resources/family_classifier.json", "resources/axon_family/family_classifier.json"),
    ("tools/axon-archive-scanner.exe", "tools/archive_scanner/target/release/axon-archive-scanner.exe"),
)


class PackageError(ValueError):
    """Raised when a native-reference package cannot be safely published."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def load_abi_receipt(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackageError(f"ABI receipt is unreadable: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema") != ABI_SCHEMA:
        raise PackageError("ABI receipt schema is invalid")
    if payload.get("calling_convention") != "__cdecl":
        raise PackageError("ABI receipt does not prove the frozen __cdecl convention")
    exports = payload.get("exports")
    if not isinstance(exports, list) or len(exports) != 18:
        raise PackageError("ABI receipt does not prove all 18 exports")
    return payload


def resolve_assets(project_root: Path) -> tuple[tuple[str, Path], ...]:
    resolved: list[tuple[str, Path]] = []
    for destination, source_relative in ASSETS:
        source = (project_root / source_relative).resolve()
        if not source.is_file():
            raise PackageError(f"required package asset is missing: {source_relative}")
        resolved.append((destination, source))
    return tuple(resolved)


def write_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_deterministic_zip(source_directory: Path, zip_path: Path, members: Iterable[str]) -> None:
    temporary = zip_path.with_name(f".{zip_path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative in sorted(members):
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (source_directory / relative).read_bytes())
        os.link(temporary, zip_path)
    except FileExistsError as error:
        raise PackageError("package zip overwrite is forbidden") from error
    finally:
        temporary.unlink(missing_ok=True)


def package_native_reference(*, project_root: Path, output_directory: Path, zip_path: Path, abi_receipt: Path) -> dict[str, object]:
    """Create a no-overwrite native-reference package from verified assets."""
    if output_directory.exists() or output_directory.is_symlink() or zip_path.exists() or zip_path.is_symlink():
        raise PackageError("package output or zip already exists")
    if not output_directory.parent.is_dir() or not zip_path.parent.is_dir():
        raise PackageError("package output parents must already exist")
    abi_payload = load_abi_receipt(abi_receipt)
    assets = resolve_assets(project_root)
    output_directory.mkdir()
    try:
        records: list[dict[str, object]] = []
        for destination, source in assets:
            target = output_directory / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            records.append({"path": destination, "bytes": target.stat().st_size, "sha256": sha256_file(target)})
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "package_id": "axon_loop28_native_reference",
            "runtime_classification": "native_reference_not_loop151_research_champion",
            "quality_claim_allowed": False,
            "abi": {
                "calling_convention": "__cdecl",
                "exports": abi_payload["exports"],
                "kvd_config_x64_size": abi_payload["kvd_config_x64_size"],
                "kvd_parity_diagnostics_options_v1_x64_size": abi_payload[
                    "kvd_parity_diagnostics_options_v1_x64_size"
                ],
                "verification_receipt_sha256": sha256_file(abi_receipt),
            },
            "assets": records,
            "runtime_requirements": {
                "platform": "Windows x64",
                "vc_runtime": "Microsoft Visual C++ 2015-2022 x64",
                "onnx_external_data_must_remain_adjacent": True,
                "third_party_entrypoint": "kvd_create/kvd_scan_path/kvd_scan_bytes",
            },
        }
        write_file(output_directory / "manifest.json", canonical_json(manifest))
        sums = "".join(f"{record['sha256']}  {record['path']}\n" for record in records)
        write_file(output_directory / "SHA256SUMS.txt", sums.encode("ascii"))
        readme = (
            "# Axon Loop28 Native Reference\n\n"
            "This is a runnable KVD-ABI reference package, not the Loop151 research champion.\n"
            "Its C ABI is frozen as extern C plus __cdecl; use kvd_free for output strings.\n"
            "Keep models/axon_loop28_base.onnx.data adjacent to its ONNX file.\n"
        )
        write_file(output_directory / "README.md", readme.encode("ascii"))
        members = [record["path"] for record in records] + ["manifest.json", "SHA256SUMS.txt", "README.md"]
        write_deterministic_zip(output_directory, zip_path, members)
        return {"manifest": manifest, "output_directory": str(output_directory), "zip": str(zip_path), "zip_sha256": sha256_file(zip_path)}
    except Exception:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--abi-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    result = package_native_reference(
        project_root=arguments.project_root.resolve(),
        output_directory=arguments.output_directory,
        zip_path=arguments.zip_path,
        abi_receipt=arguments.abi_receipt,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
