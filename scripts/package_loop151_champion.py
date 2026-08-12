#!/usr/bin/env python3
"""Package the Loop151 native-only champion delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCHEMA = "axon_loop151_native_champion_package_v1"
ABI_RECEIPT = Path("reports/roadmap_9997/loop151_native_loader_abi_20260717.json")
PARITY_RECEIPT = Path("reports/roadmap_9997/loop151_raw_parity_receipt_20260716.json")
NATIVE_ASSET_MANIFEST = Path("reports/roadmap_9997/loop151_native_assets/manifest.json")

ASSETS = (
    ("bin/axon_loop151_champion.dll", Path("tools/axon_onnx_dll/build/bin/Release/axon_loop151_champion.dll")),
    ("bin/axon_loop151_example.exe", Path("tools/axon_onnx_dll/build/bin/Release/axon_loop151_example.exe")),
    ("bin/onnxruntime.dll", Path("tools/axon_onnx_dll/build/bin/Release/onnxruntime.dll")),
    ("lib/axon_loop151_champion.lib", Path("tools/axon_onnx_dll/build/lib/Release/axon_loop151_champion.lib")),
    ("include/axon_onnx_predict.h", Path("tools/axon_onnx_dll/include/axon_onnx_predict.h")),
    ("models/axon_loop151_base.onnx", Path("models/random_20w_8192/axon_loop151_base.onnx")),
    ("models/axon_loop151_base.onnx.data", Path("models/random_20w_8192/axon_loop151_base.onnx.data")),
    ("models/axon_loop151_base.onnx.json", Path("models/random_20w_8192/axon_loop151_base.onnx.json")),
    ("models/loop151_primary.native.json", Path("reports/roadmap_9997/loop151_native_assets/primary.native.json")),
    ("models/loop151_conservative.native.json", Path("reports/roadmap_9997/loop151_native_assets/conservative.native.json")),
    ("models/loop151_content_cross.native.json", Path("reports/roadmap_9997/loop151_native_assets/content_cross.native.json")),
    ("models/loop151_noise.native.json", Path("reports/roadmap_9997/loop151_native_assets/noise.native.json")),
    ("models/loop151_selector.native.json", Path("reports/roadmap_9997/loop151_native_assets/selector.native.json")),
)

EXAMPLES = (
    ("examples/cpp/axon_onnx_predict.h", Path("tools/axon_onnx_dll/include/axon_onnx_predict.h")),
    ("examples/cpp/axon_loop151_example.cpp", Path("tools/axon_onnx_dll/examples/axon_loop151_example.cpp")),
    ("examples/rust/Cargo.toml", Path("tools/axon_onnx_dll/examples/axon_loop151_rust/Cargo.toml")),
    ("examples/rust/Cargo.lock", Path("tools/axon_onnx_dll/examples/axon_loop151_rust/Cargo.lock")),
    ("examples/rust/README.md", Path("tools/axon_onnx_dll/examples/axon_loop151_rust/README.md")),
    ("examples/rust/src/main.rs", Path("tools/axon_onnx_dll/examples/axon_loop151_rust/src/main.rs")),
    ("examples/js/package.json", Path("tools/axon_onnx_dll/examples/js/package.json")),
    ("examples/js/README.md", Path("tools/axon_onnx_dll/examples/js/README.md")),
    ("examples/js/axon_loop151_call.js", Path("tools/axon_onnx_dll/examples/js/axon_loop151_call.js")),
)


class PackageError(RuntimeError):
    """Raised when a native package input is missing or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackageError(f"cannot read JSON input: {path}") from error
    if not isinstance(payload, dict):
        raise PackageError(f"JSON input is not an object: {path}")
    return payload


def copy_asset(source: Path, target: Path, package_path: str) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {"path": package_path, "bytes": target.stat().st_size, "sha256": sha256(target)}


def deterministic_zip(source: Path, target: Path) -> str:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for item in sorted(path for path in source.rglob("*") if path.is_file()):
                relative = item.relative_to(source).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, item.read_bytes())
        os.link(temporary, target)
    except FileExistsError as error:
        raise PackageError("package zip already exists") from error
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(target)


def validate_inputs() -> tuple[dict, dict, dict]:
    abi = read_json(PROJECT_ROOT / ABI_RECEIPT)
    if abi.get("schema") != "axon_kvd_dll_abi_verification_v1":
        raise PackageError("ABI receipt schema is invalid")
    if abi.get("calling_convention") != "__cdecl" or abi.get("kvd_config_x64_size") != 96:
        raise PackageError("ABI receipt does not prove the frozen x64 KVD ABI")
    exports = abi.get("exports")
    if not isinstance(exports, list) or len(exports) != 18:
        raise PackageError("ABI receipt does not list all 18 exports")
    parity = read_json(PROJECT_ROOT / PARITY_RECEIPT)
    if parity.get("passed") is not True:
        raise PackageError("raw parity receipt does not pass")
    native_manifest = read_json(PROJECT_ROOT / NATIVE_ASSET_MANIFEST)
    if native_manifest.get("schema") != "axon_loop151_native_asset_manifest_v1":
        raise PackageError("native asset manifest schema is invalid")
    return abi, parity, native_manifest


def package(*, output_directory: Path, output_zip: Path) -> dict[str, object]:
    output_directory = output_directory.resolve()
    output_zip = output_zip.resolve()
    if output_directory.exists() or output_zip.exists():
        raise PackageError("package output already exists; overwrite is forbidden")
    if not output_directory.parent.is_dir() or not output_zip.parent.is_dir():
        raise PackageError("package output parents must already exist")
    abi, parity, native_manifest = validate_inputs()
    for _, relative in ASSETS + EXAMPLES:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            raise PackageError(f"required package asset is missing: {relative}")

    output_directory.mkdir(parents=True)
    records: list[dict[str, object]] = []
    try:
        for destination, relative in ASSETS + EXAMPLES:
            records.append(copy_asset(PROJECT_ROOT / relative, output_directory / destination, destination))

        runtime = {
            "schema": "axon_loop151_native_runtime_v1",
            "base_onnx_path": "../models/axon_loop151_base.onnx",
            "primary_model_path": "../models/loop151_primary.native.json",
            "conservative_model_path": "../models/loop151_conservative.native.json",
            "content_cross_model_path": "../models/loop151_content_cross.native.json",
            "noise_model_path": "../models/loop151_noise.native.json",
            "selector_model_path": "../models/loop151_selector.native.json",
            "python_required": False,
            "external_ml_runtime_required": False,
        }
        runtime_path = output_directory / "runtime/loop151_native_runtime.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="ascii")
        records.append({"path": "runtime/loop151_native_runtime.json", "bytes": runtime_path.stat().st_size, "sha256": sha256(runtime_path)})

        readme = """# Axon Loop151 Native Champion DLL

This is the Loop151 champion packaged in the same native-only form as the
Loop28 delivery. It requires Windows x64, but it does not require Python,
PyTorch, scikit-learn, or the repository. The ONNX Runtime DLL is bundled in
`bin/`; the ONNX external-data file stays beside its model in `models/`.

The frozen KVD ABI is unchanged: Windows `__cdecl`, 18 exports, and a 96-byte
x64 `kvd_config`. Set `stage2_model_json_path` to
`runtime/loop151_native_runtime.json` and keep calling
`kvd_create -> kvd_validate_models -> kvd_scan_path` or `kvd_scan_bytes`.
`onnx_model_path` may remain null because the native runtime config binds the
base ONNX model and all five exported Loop151 model assets.

The package contains only native binaries, model weights, caller examples,
the runtime manifest, and this documentation. The source parity receipt is
provenance for the frozen Python reference only; it is not a native DLL parity
or full-test quality claim.
"""
        readme_path = output_directory / "README.md"
        readme_path.write_text(readme, encoding="ascii")
        records.append({"path": "README.md", "bytes": readme_path.stat().st_size, "sha256": sha256(readme_path)})

        manifest = {
            "schema": PACKAGE_SCHEMA,
            "package_id": "axon_loop151_native_champion",
            "runtime_classification": "native_only_loop151_research_champion",
            "quality_claim": "No native full-test quality claim; the referenced raw parity receipt covers only the frozen Python reference samples.",
            "abi": {
                "calling_convention": "__cdecl",
                "exports": len(abi["exports"]),
                "kvd_config_x64_size": abi["kvd_config_x64_size"],
            },
            "runtime": {
                "python_required": False,
                "pytorch_required": False,
                "sklearn_required": False,
                "onnxruntime_bundled": True,
                "onnx_external_data_adjacent": True,
                "config": "runtime/loop151_native_runtime.json",
            },
            "frozen_chain": ["Loop127", "Loop130_R5", "Loop134", "Loop136", "Loop151_trusted_signer"],
            "native_asset_manifest_sha256": sha256(PROJECT_ROOT / NATIVE_ASSET_MANIFEST),
            "parity_receipt_sha256": sha256(PROJECT_ROOT / PARITY_RECEIPT),
            "assets": records,
        }
        manifest_path = output_directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
        records.append({"path": "manifest.json", "bytes": manifest_path.stat().st_size, "sha256": sha256(manifest_path)})
        sums_path = output_directory / "SHA256SUMS.txt"
        sums_path.write_text("".join(f"{record['sha256']}  {record['path']}\n" for record in records), encoding="ascii")
        zip_sha256 = deterministic_zip(output_directory, output_zip)
        return {
            "output_directory": str(output_directory),
            "zip": str(output_zip),
            "zip_sha256": zip_sha256,
            "members": len(records) + 1,
            "python_required": False,
        }
    except Exception:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--zip", dest="output_zip", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(package(output_directory=arguments.output_directory, output_zip=arguments.output_zip), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
