#!/usr/bin/env python3
"""Package only the Loop151 DLL, frozen weights, caller examples, and README."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DLL = Path("tools/axon_onnx_dll/build/bin/Release/axon_loop151_champion.dll")
HEADER = Path("tools/axon_onnx_dll/include/axon_onnx_predict.h")
ARTIFACT_MANIFEST = Path("manifests/roadmap_9997/loop151_runtime/frozen_artifacts.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def zip_directory(source: Path, target: Path) -> str:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(item for item in source.rglob("*") if item.is_file()):
                relative = path.relative_to(source).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(target)


def package(output_directory: Path, output_zip: Path) -> dict[str, str | int]:
    output_directory = output_directory.resolve()
    output_zip = output_zip.resolve()
    if output_directory.exists() or output_zip.exists():
        raise RuntimeError("Minimal package output already exists; overwrite is forbidden")
    manifest = json.loads((PROJECT_ROOT / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != "axon_loop151_raw_runtime_artifacts_v1":
        raise RuntimeError("Unexpected Loop151 artifact manifest schema")
    output_directory.mkdir(parents=True)
    try:
        copy_file(PROJECT_ROOT / DLL, output_directory / "axon_loop151_champion.dll")
        copy_file(PROJECT_ROOT / HEADER, output_directory / "examples/cpp/axon_onnx_predict.h")
        copy_file(
            PROJECT_ROOT / "tools/axon_onnx_dll/examples/axon_loop151_example.cpp",
            output_directory / "examples/cpp/axon_loop151_example.cpp",
        )
        rust_source = PROJECT_ROOT / "tools/axon_onnx_dll/examples/axon_loop151_rust"
        for relative in ("Cargo.toml", "Cargo.lock", "README.md", "src/main.rs"):
            copy_file(rust_source / relative, output_directory / f"examples/rust/{relative}")
        js_source = PROJECT_ROOT / "tools/axon_onnx_dll/examples/js"
        for relative in ("package.json", "axon_loop151_call.js", "README.md"):
            copy_file(js_source / relative, output_directory / f"examples/js/{relative}")
        for artifact_name, item in manifest["artifacts"].items():
            source = PROJECT_ROOT / item["path"]
            if sha256(source) != item["sha256"]:
                raise RuntimeError(f"Frozen artifact digest changed: {item['path']}")
            copy_file(
                source,
                output_directory / "models" / f"{artifact_name}_{Path(item['path']).name}",
            )
        readme = """# Axon Loop151 minimal DLL package

This artifact-only package contains exactly the Loop151 champion DLL, its
frozen checkpoint/Stage-2 weights, three caller examples, and this README.
It preserves the KVD `extern \"C\"` Windows `__cdecl` ABI, 18 exports, and the
96-byte x64 `kvd_config` layout.

The current DLL is a native ABI bridge, not a self-contained Python-free
runtime. It requires the already prepared Loop151 Python runtime and a bridge
JSON. Set `kvd_config.stage2_model_json_path` to the `runtime/loop151_bridge.json`
from the full champion package/project, then pass the DLL path from this
package. Do not pass a Loop28 Stage-2 HGB JSON.

Examples:

- `examples/cpp/axon_loop151_example.cpp`
- `examples/rust/src/main.rs`
- `examples/js/axon_loop151_call.js`

The model files in `models/` are the frozen Loop151 weights, named with their
stage (`checkpoint`, `primary`, `conservative`, `content_cross`, `noise`, and
`selector`) to avoid filename collisions. They are included
for deployment/audit handoff; the bridge's configured runtime must resolve
them using its SHA-bound artifact paths.
"""
        (output_directory / "README.md").write_text(readme, encoding="utf-8")
        zip_sha = zip_directory(output_directory, output_zip)
        member_count = sum(1 for path in output_directory.rglob("*") if path.is_file())
        return {"output_directory": str(output_directory), "zip": str(output_zip), "zip_sha256": zip_sha, "members": member_count}
    except Exception:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    arguments = parser.parse_args()
    print(package(arguments.output_directory, arguments.zip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
