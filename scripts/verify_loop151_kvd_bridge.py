#!/usr/bin/env python3
"""Exercise the frozen KVD ABI against the Loop151 champion bridge DLL."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path


class KvdConfig(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("model_normal_path", ctypes.c_char_p),
        ("model_packed_path", ctypes.c_char_p),
        ("family_classifier_json_path", ctypes.c_char_p),
        ("allowed_scan_root", ctypes.c_char_p),
        ("max_file_size", ctypes.c_uint),
        ("prediction_threshold", ctypes.c_float),
        ("onnx_model_path", ctypes.c_char_p),
        ("onnx_model_normal_path", ctypes.c_char_p),
        ("onnx_model_packed_path", ctypes.c_char_p),
        ("stage2_model_json_path", ctypes.c_char_p),
        ("archive_scanner_path", ctypes.c_char_p),
        ("scan_nested", ctypes.c_int),
    ]


def _read_response(library: ctypes.CDLL, pointer: ctypes.c_void_p, length: int) -> dict:
    try:
        return json.loads(ctypes.string_at(pointer, length).decode("utf-8"))
    finally:
        library.kvd_free(pointer)


def _scan_path(library: ctypes.CDLL, handle: ctypes.c_void_p, path: Path) -> dict:
    pointer = ctypes.c_void_p()
    length = ctypes.c_size_t()
    result = library.kvd_scan_path(handle, str(path).encode("utf-8"), ctypes.byref(pointer), ctypes.byref(length))
    if result != 0:
        raise RuntimeError(f"kvd_scan_path failed: {result}")
    return _read_response(library, pointer, length.value)


def _scan_bytes(library: ctypes.CDLL, handle: ctypes.c_void_p, path: Path) -> dict:
    source = path.read_bytes()
    buffer = (ctypes.c_ubyte * len(source)).from_buffer_copy(source)
    pointer = ctypes.c_void_p()
    length = ctypes.c_size_t()
    result = library.kvd_scan_bytes(handle, buffer, len(source), ctypes.byref(pointer), ctypes.byref(length))
    if result != 0:
        raise RuntimeError(f"kvd_scan_bytes failed: {result}")
    return _read_response(library, pointer, length.value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--bridge-config", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    args = parser.parse_args()
    if ctypes.sizeof(KvdConfig) != 96:
        raise SystemExit(f"Unexpected x64 kvd_config size: {ctypes.sizeof(KvdConfig)}")
    library = ctypes.CDLL(str(args.dll))
    library.kvd_create.argtypes = [ctypes.POINTER(KvdConfig)]
    library.kvd_create.restype = ctypes.c_void_p
    library.kvd_destroy.argtypes = [ctypes.c_void_p]
    library.kvd_free.argtypes = [ctypes.c_void_p]
    library.kvd_scan_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    library.kvd_scan_path.restype = ctypes.c_int
    library.kvd_scan_bytes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    library.kvd_scan_bytes.restype = ctypes.c_int
    config = KvdConfig()
    config.stage2_model_json_path = str(args.bridge_config).encode("utf-8")
    handle = library.kvd_create(ctypes.byref(config))
    if not handle:
        raise SystemExit("kvd_create rejected the Loop151 bridge configuration")
    try:
        path_result = _scan_path(library, handle, args.sample)
        bytes_result = _scan_bytes(library, handle, args.sample)
    finally:
        library.kvd_destroy(handle)
    for result in (path_result, bytes_result):
        if result.get("loop_id") != "Loop151" or "loop151" not in result:
            raise SystemExit(f"Bridge did not return a Loop151 result: {result}")
    if path_result["prediction"] != bytes_result["prediction"]:
        raise SystemExit("Path and byte scans disagree on the final Loop151 prediction")
    print(
        json.dumps(
            {
                "schema": "axon_loop151_kvd_bridge_integration_v1",
                "kvd_config_x64_size": ctypes.sizeof(KvdConfig),
                "path_prediction": path_result["prediction"],
                "bytes_prediction": bytes_result["prediction"],
                "path_loop151": path_result["loop151"],
                "bytes_loop151": bytes_result["loop151"],
                "passed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
