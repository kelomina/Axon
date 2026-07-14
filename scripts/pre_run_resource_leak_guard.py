#!/usr/bin/env python3
"""Pre-run resource and static leak guard for heavy Axon operations.

The guard is deliberately read-only. It does not import training modules, load
models, touch CUDA libraries, open NPZ feature arrays, scan raw sample trees, or
start worker pools. It checks current host pressure and scans target scripts for
patterns that require explicit review before execution.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUARD_SCHEMA = "axon_loop77_pre_run_resource_leak_guard_v1"

STATIC_RISK_PATTERNS: dict[str, re.Pattern[str]] = {
    "infinite_loop": re.compile(r"\bwhile\s+True\s*:", re.IGNORECASE),
    "torch_import": re.compile(r"^\s*(import\s+torch|from\s+torch\b)", re.IGNORECASE | re.MULTILINE),
    "cuda_usage": re.compile(r"(\btorch\.cuda\b|\.cuda\s*\()", re.IGNORECASE),
    "npz_array_load": re.compile(r"\bnp\.load\s*\(", re.IGNORECASE),
    "process_pool": re.compile(r"\b(ProcessPoolExecutor|multiprocessing|mp\.Pool)\b", re.IGNORECASE),
    "thread_pool": re.compile(r"\b(ThreadPoolExecutor|threading\.Thread)\b", re.IGNORECASE),
    "torch_dataloader": re.compile(r"\bDataLoader\s*\(", re.IGNORECASE),
    "persistent_workers": re.compile(r"\bpersistent_workers\s*=\s*True\b", re.IGNORECASE),
    "unbounded_spawn": re.compile(r"\b(subprocess\.Popen|Start-Process)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class MemorySnapshot:
    total_mb: float
    available_mb: float

    @property
    def used_pct(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return (1.0 - self.available_mb / self.total_mb) * 100.0


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_git_status() -> dict[str, Any]:
    head = _run_command(["git", "rev-parse", "--short", "HEAD"], timeout=5).strip()
    status = _run_command(["git", "status", "--short"], timeout=10)
    return {
        "head": head or None,
        "dirty": bool(status.strip()),
        "status_line_count": len([line for line in status.splitlines() if line.strip()]),
    }


def collect_memory_snapshot() -> MemorySnapshot:
    if platform.system().lower() == "windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return MemorySnapshot(
                total_mb=float(status.ullTotalPhys / (1024 * 1024)),
                available_mb=float(status.ullAvailPhys / (1024 * 1024)),
            )

    if hasattr(os, "sysconf"):
        page_size = float(os.sysconf("SC_PAGE_SIZE"))
        total_pages = float(os.sysconf("SC_PHYS_PAGES"))
        available_pages = float(os.sysconf("SC_AVPHYS_PAGES"))
        return MemorySnapshot(
            total_mb=page_size * total_pages / (1024 * 1024),
            available_mb=page_size * available_pages / (1024 * 1024),
        )
    return MemorySnapshot(total_mb=0.0, available_mb=0.0)


def _run_command(command: list[str], timeout: int = 10) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout or ""


def collect_python_processes() -> list[dict[str, Any]]:
    if platform.system().lower() == "windows":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-Process -Name python,python3 -ErrorAction SilentlyContinue | "
                "Select-Object Id,ProcessName,WorkingSet64,CPU | ConvertTo-Json -Compress"
            ),
        ]
        raw = _run_command(command)
        if not raw.strip():
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        rows = payload if isinstance(payload, list) else [payload]
        return [
            {
                "pid": int(row.get("Id", 0) or 0),
                "name": str(row.get("ProcessName", "")),
                "rss_mb": float(row.get("WorkingSet64", 0) or 0) / (1024 * 1024),
                "cpu_seconds": float(row.get("CPU", 0.0) or 0.0),
            }
            for row in rows
        ]

    raw = _run_command(["ps", "-eo", "pid=,comm=,rss="])
    rows = []
    for line in raw.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid, name, rss_kb = parts
        if "python" not in name.casefold():
            continue
        try:
            rss_mb = float(rss_kb) / 1024.0
        except ValueError:
            rss_mb = 0.0
        rows.append({"pid": int(pid), "name": name, "rss_mb": rss_mb, "cpu_seconds": 0.0})
    return rows


def collect_gpu_summary() -> dict[str, Any]:
    raw_gpu = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if not raw_gpu.strip():
        return {"available": False}
    devices = []
    for index, line in enumerate(raw_gpu.splitlines()):
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            memory_used_mb = int(parts[0].strip())
            memory_total_mb = int(parts[1].strip())
            utilization_pct = int(parts[2].strip())
        except ValueError:
            continue
        devices.append(
            {
                "index": index,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "memory_used_pct": float(memory_used_mb / memory_total_mb * 100.0) if memory_total_mb else 0.0,
                "utilization_pct": utilization_pct,
            }
        )
    if not devices:
        return {"available": False}
    highest_pressure = max(devices, key=lambda item: float(item["memory_used_pct"]))

    raw_apps = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    compute_apps = [line.strip() for line in raw_apps.splitlines() if line.strip()]
    python_compute_apps = [line for line in compute_apps if "python" in line.casefold()]
    return {
        "available": True,
        "memory_used_mb": highest_pressure["memory_used_mb"],
        "memory_total_mb": highest_pressure["memory_total_mb"],
        "memory_used_pct": highest_pressure["memory_used_pct"],
        "utilization_pct": max(int(item["utilization_pct"]) for item in devices),
        "devices": devices,
        "compute_app_count": len(compute_apps),
        "python_compute_app_count": len(python_compute_apps),
        "python_compute_apps": python_compute_apps[:10],
    }


def _max_gpu_memory_used_pct(gpu_summary: dict[str, Any]) -> float:
    devices = gpu_summary.get("devices")
    if isinstance(devices, list) and devices:
        return max(float(device.get("memory_used_pct", 0.0)) for device in devices)
    return float(gpu_summary.get("memory_used_pct", 0.0))


def _is_relative_to_path(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _candidate_local_module_paths(module_name: str, current_file: Path) -> list[Path]:
    parts = [part for part in module_name.split(".") if part]
    if not parts:
        return []
    roots = [current_file.parent, PROJECT_ROOT / "scripts", PROJECT_ROOT / "src", PROJECT_ROOT]
    candidates = []
    for root in roots:
        module_path = root.joinpath(*parts)
        candidates.append(module_path.with_suffix(".py"))
        candidates.append(module_path / "__init__.py")
    return candidates


def _resolve_local_import(module_name: str, current_file: Path) -> Optional[Path]:
    project_root = PROJECT_ROOT.resolve(strict=False)
    for candidate in _candidate_local_module_paths(module_name, current_file):
        resolved = candidate.resolve(strict=False)
        if not _is_relative_to_path(resolved, project_root) and not _is_relative_to_path(resolved, current_file.parent):
            continue
        if resolved.exists() and resolved.suffix == ".py":
            return resolved
    return None


def _local_imports_from_text(text: str, current_file: Path) -> list[Path]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _resolve_local_import(alias.name, current_file)
                if resolved is not None:
                    imports.append(resolved)
        elif isinstance(node, ast.ImportFrom) and node.module:
            resolved = _resolve_local_import(node.module, current_file)
            if resolved is not None:
                imports.append(resolved)
    return imports


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_attr(node: ast.AST) -> str:
    return node.attr if isinstance(node, ast.Attribute) else ""


def _is_numeric_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _is_open_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node.func)
    return name == "open" or name.endswith(".open")


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for element in target.elts:
            names.extend(_target_names(element))
        return names
    return []


def _collect_ast_aliases(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    reader_vars: set[str] = set()
    file_handle_vars: set[str] = set()
    executor_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None and _is_open_call(item.context_expr):
                    file_handle_vars.update(_target_names(item.optional_vars))
                if item.optional_vars is not None and _call_name(getattr(item.context_expr, "func", ast.Name(""))).endswith(
                    ("Executor", "Pool")
                ):
                    executor_vars.update(_target_names(item.optional_vars))
        elif isinstance(node, ast.Assign):
            targets = [name for target in node.targets for name in _target_names(target)]
            value_name = _call_name(getattr(node.value, "func", ast.Name("")))
            if value_name in {"csv.reader", "csv.DictReader"}:
                reader_vars.update(targets)
            elif _is_open_call(node.value):
                file_handle_vars.update(targets)
            elif value_name.endswith(("Executor", "Pool")):
                executor_vars.update(targets)
        elif isinstance(node, ast.AnnAssign):
            targets = _target_names(node.target)
            value_name = _call_name(getattr(node.value, "func", ast.Name(""))) if node.value is not None else ""
            if value_name in {"csv.reader", "csv.DictReader"}:
                reader_vars.update(targets)
            elif node.value is not None and _is_open_call(node.value):
                file_handle_vars.update(targets)
            elif value_name.endswith(("Executor", "Pool")):
                executor_vars.update(targets)
    return reader_vars, file_handle_vars, executor_vars


def _is_csv_reader_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_name(node.func) in {"csv.reader", "csv.DictReader"}


def _is_safe_bounded_islice(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or _call_name(node.func) not in {"islice", "itertools.islice"}:
        return False
    return any(_is_numeric_constant(arg) for arg in node.args[1:])


def _is_reader_or_file_iterable(node: ast.AST, reader_vars: set[str], file_handle_vars: set[str]) -> bool:
    if _is_safe_bounded_islice(node):
        return False
    if _is_csv_reader_call(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in reader_vars or node.id in file_handle_vars
    return False


def _is_directory_iterator_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node.func)
    return name in {"os.scandir", "os.walk"} or _call_attr(node.func) in {"iterdir", "glob", "rglob"}


def _is_unbounded_read_call(node: ast.Call) -> bool:
    attr = _call_attr(node.func)
    if attr in {"read_text", "read_bytes"}:
        return True
    if attr not in {"read", "readlines"}:
        return False
    if not node.args:
        return True
    return not _is_numeric_constant(node.args[0])


def _is_array_or_object_load_call(node: ast.Call, load_aliases: dict[str, str]) -> bool:
    name = _call_name(node.func)
    if name in {
        "np.load",
        "numpy.load",
        "torch.load",
        "pickle.load",
        "joblib.load",
        "json.load",
    }:
        return True
    return isinstance(node.func, ast.Name) and node.func.id in load_aliases


def _collect_load_aliases(tree: ast.AST) -> dict[str, str]:
    load_aliases: dict[str, str] = {}
    load_modules = {"numpy", "torch", "pickle", "joblib", "json"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module not in load_modules:
            continue
        for alias in node.names:
            if alias.name == "load":
                load_aliases[alias.asname or alias.name] = node.module
    return load_aliases


def _is_executor_map_call(node: ast.Call, executor_vars: set[str]) -> bool:
    if _call_attr(node.func) not in {"map", "starmap", "imap", "imap_unordered"}:
        return False
    value = node.func.value if isinstance(node.func, ast.Attribute) else None
    return isinstance(value, ast.Name) and value.id in executor_vars


def _is_executor_submit_comprehension(node: ast.comprehension, elt: ast.AST, executor_vars: set[str]) -> bool:
    if not isinstance(elt, ast.Call) or _call_attr(elt.func) != "submit":
        return False
    value = elt.func.value if isinstance(elt.func, ast.Attribute) else None
    return isinstance(value, ast.Name) and value.id in executor_vars and not _is_safe_bounded_islice(node.iter)


def _append_ast_finding(
    findings: list[dict[str, Any]],
    *,
    risk_id: str,
    path: Path,
    line_number: int,
    lines: list[str],
    allowed_risks: set[str],
) -> None:
    if risk_id in allowed_risks:
        return
    snippet = lines[line_number - 1].strip()[:240] if 0 < line_number <= len(lines) else ""
    findings.append(
        {
            "risk_id": risk_id,
            "path": str(path),
            "line": line_number,
            "snippet": snippet,
        }
    )


def scan_ast_risks(text: str, path: Path, allowed_risks: set[str]) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    findings: list[dict[str, Any]] = []
    reader_vars, file_handle_vars, executor_vars = _collect_ast_aliases(tree)
    load_aliases = _collect_load_aliases(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            risk_id: Optional[str] = None
            if call_name == "os.listdir":
                risk_id = "directory_materialization"
            elif _is_unbounded_read_call(node):
                risk_id = "whole_file_read"
            elif _is_array_or_object_load_call(node, load_aliases):
                risk_id = "array_or_object_load"
            elif _is_executor_map_call(node, executor_vars):
                risk_id = "executor_map_unbounded"
            elif call_name in {"list", "tuple", "set", "sorted", "dict"} and node.args:
                first_arg = node.args[0]
                if _is_reader_or_file_iterable(first_arg, reader_vars, file_handle_vars):
                    risk_id = "reader_materialization"
                elif _is_directory_iterator_call(first_arg):
                    risk_id = "directory_materialization"

            if risk_id is not None:
                _append_ast_finding(
                    findings,
                    risk_id=risk_id,
                    path=path,
                    line_number=getattr(node, "lineno", 1),
                    lines=lines,
                    allowed_risks=allowed_risks,
                )
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in node.generators:
                risk_id = None
                if _is_reader_or_file_iterable(generator.iter, reader_vars, file_handle_vars):
                    risk_id = "reader_materialization"
                elif _is_directory_iterator_call(generator.iter):
                    risk_id = "directory_materialization"
                elif _is_executor_submit_comprehension(generator, getattr(node, "elt", ast.Name("")), executor_vars):
                    risk_id = "executor_map_unbounded"
                if risk_id is not None:
                    _append_ast_finding(
                        findings,
                        risk_id=risk_id,
                        path=path,
                        line_number=getattr(node, "lineno", 1),
                        lines=lines,
                        allowed_risks=allowed_risks,
                    )
    return findings


def _expand_static_scan_paths(paths: Sequence[Path], *, follow_local_imports: bool) -> tuple[list[Path], list[str]]:
    ordered_paths: list[Path] = []
    missing_files: list[str] = []
    queue = [resolve_path(path) for path in paths]
    seen: set[Path] = set()
    while queue:
        resolved = queue.pop(0).resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            missing_files.append(str(resolved))
            continue
        ordered_paths.append(resolved)
        if not follow_local_imports:
            continue
        text = resolved.read_text(encoding="utf-8", errors="replace")
        for imported in _local_imports_from_text(text, resolved):
            if imported not in seen:
                queue.append(imported)
    return ordered_paths, missing_files


def scan_static_risks(
    paths: Sequence[Path],
    allowed_risks: set[str],
    *,
    follow_local_imports: bool = False,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    findings: list[dict[str, Any]] = []
    expanded_paths, missing_files = _expand_static_scan_paths(paths, follow_local_imports=follow_local_imports)
    for resolved in expanded_paths:
        text = resolved.read_text(encoding="utf-8", errors="replace")
        scan_text = strip_python_strings_and_comments(text)
        for risk_id, pattern in STATIC_RISK_PATTERNS.items():
            if risk_id in allowed_risks:
                continue
            for match in pattern.finditer(scan_text):
                line_number = scan_text.count("\n", 0, match.start()) + 1
                line_start = scan_text.rfind("\n", 0, match.start()) + 1
                line_end = scan_text.find("\n", match.start())
                if line_end == -1:
                    line_end = len(scan_text)
                findings.append(
                    {
                        "risk_id": risk_id,
                        "path": str(resolved),
                        "line": line_number,
                        "snippet": text[line_start:line_end].strip()[:240],
                    }
                )
        findings.extend(scan_ast_risks(text, resolved, allowed_risks))
    return findings, missing_files, [str(path) for path in expanded_paths]


def strip_python_strings_and_comments(text: str) -> str:
    lines = text.splitlines(keepends=True)
    mutable = [list(line) for line in lines]
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type not in {tokenize.STRING, tokenize.COMMENT}:
                continue
            (start_line, start_col), (end_line, end_col) = token.start, token.end
            for line_index in range(start_line - 1, end_line):
                if line_index < 0 or line_index >= len(mutable):
                    continue
                start = start_col if line_index == start_line - 1 else 0
                end = end_col if line_index == end_line - 1 else len(mutable[line_index])
                for col in range(start, min(end, len(mutable[line_index]))):
                    if mutable[line_index][col] not in "\r\n":
                        mutable[line_index][col] = " "
    except tokenize.TokenError:
        return text
    return "".join("".join(line) for line in mutable)


def build_guard_receipt(
    *,
    target_scripts: Sequence[Path],
    command: Optional[Sequence[str]] = None,
    created_at: Optional[float] = None,
    cwd: Optional[Path] = None,
    python_executable: Optional[str] = None,
) -> dict[str, Any]:
    resolved_targets = [resolve_path(path).resolve(strict=False) for path in target_scripts]
    target_hashes = {}
    missing_targets = []
    for target in resolved_targets:
        if target.exists():
            target_hashes[str(target)] = file_sha256(target)
        else:
            missing_targets.append(str(target))
    return {
        "created_at_unix": float(time.time() if created_at is None else created_at),
        "cwd": str((cwd or Path.cwd()).resolve(strict=False)),
        "python_executable": python_executable or sys.executable,
        "command": list(command) if command is not None else None,
        "target_sha256": target_hashes,
        "missing_targets": missing_targets,
        "git": collect_git_status(),
    }


def validate_guard_receipt(
    payload: dict[str, Any],
    *,
    expected_target_scripts: Sequence[Path],
    expected_command: Optional[Sequence[str]] = None,
    expected_cwd: Optional[Path] = None,
    max_age_seconds: float = 3600.0,
    now: Optional[float] = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if payload.get("schema") != GUARD_SCHEMA:
        failures.append("guard_schema_mismatch")
    if payload.get("guard_ready") is not True:
        failures.append("guard_not_ready")

    receipt = payload.get("receipt")
    if not isinstance(receipt, dict):
        return {"valid": False, "failures": failures + ["receipt_missing"]}

    created_at = receipt.get("created_at_unix")
    current_time = float(time.time() if now is None else now)
    try:
        age_seconds = current_time - float(created_at)
    except (TypeError, ValueError):
        age_seconds = float("inf")
        failures.append("receipt_created_at_invalid")
    else:
        if age_seconds < -300:
            failures.append("receipt_created_in_future")
        if age_seconds > float(max_age_seconds):
            failures.append("receipt_expired")

    expected_cwd_text = str((expected_cwd or Path.cwd()).resolve(strict=False))
    if str(receipt.get("cwd")) != expected_cwd_text:
        failures.append("receipt_cwd_mismatch")

    if expected_command is not None and list(expected_command) != receipt.get("command"):
        failures.append("receipt_command_mismatch")

    recorded_hashes = receipt.get("target_sha256")
    if not isinstance(recorded_hashes, dict):
        recorded_hashes = {}
        failures.append("receipt_target_hashes_missing")

    expected_targets = [resolve_path(path).resolve(strict=False) for path in expected_target_scripts]
    recorded_targets = set(str(path) for path in recorded_hashes)
    expected_target_texts = set(str(path) for path in expected_targets)
    if recorded_targets != expected_target_texts:
        failures.append("receipt_target_set_mismatch")

    changed_targets = []
    missing_targets = []
    for target in expected_targets:
        target_text = str(target)
        if not target.exists():
            missing_targets.append(target_text)
            continue
        expected_hash = recorded_hashes.get(target_text)
        if expected_hash is None:
            continue
        actual_hash = file_sha256(target)
        if actual_hash != expected_hash:
            changed_targets.append(target_text)
    if missing_targets:
        failures.append("receipt_target_missing")
    if changed_targets:
        failures.append("receipt_target_hash_mismatch")

    return {
        "valid": not failures,
        "failures": failures,
        "age_seconds": age_seconds,
        "changed_targets": changed_targets,
        "missing_targets": missing_targets,
    }


def evaluate_guard(
    *,
    target_scripts: Sequence[Path],
    max_system_used_pct: float = 90.0,
    max_python_rss_mb: float = 8192.0,
    max_python_process_count: int = 32,
    max_total_python_rss_mb: float = 16384.0,
    max_gpu_memory_used_pct: float = 95.0,
    max_gpu_python_apps: int = 0,
    follow_local_imports: bool = False,
    allowed_risks: Optional[set[str]] = None,
    memory_snapshot: Optional[MemorySnapshot] = None,
    python_processes: Optional[list[dict[str, Any]]] = None,
    gpu_summary: Optional[dict[str, Any]] = None,
    command: Optional[Sequence[str]] = None,
    created_at: Optional[float] = None,
) -> dict[str, Any]:
    allowed = set(allowed_risks or set())
    memory = memory_snapshot or collect_memory_snapshot()
    python_rows = python_processes if python_processes is not None else collect_python_processes()
    gpu = gpu_summary if gpu_summary is not None else collect_gpu_summary()
    static_findings, missing_files, scanned_files = scan_static_risks(
        target_scripts,
        allowed,
        follow_local_imports=follow_local_imports,
    )

    failures: list[str] = []
    warnings: list[str] = []
    if memory.used_pct > max_system_used_pct:
        failures.append("system_memory_used_pct_exceeds_limit")

    heavy_python = [row for row in python_rows if float(row.get("rss_mb", 0.0)) > max_python_rss_mb]
    total_python_rss_mb = sum(float(row.get("rss_mb", 0.0)) for row in python_rows)
    if heavy_python:
        failures.append("python_process_rss_exceeds_limit")
    if max_python_process_count >= 0 and len(python_rows) > max_python_process_count:
        failures.append("python_process_count_exceeds_limit")
    if total_python_rss_mb > max_total_python_rss_mb:
        failures.append("python_total_rss_exceeds_limit")

    if gpu.get("available"):
        if _max_gpu_memory_used_pct(gpu) > max_gpu_memory_used_pct:
            failures.append("gpu_memory_used_pct_exceeds_limit")
        if int(gpu.get("python_compute_app_count", 0)) > max_gpu_python_apps:
            failures.append("python_gpu_compute_app_detected")
    else:
        warnings.append("gpu_status_unavailable")

    if static_findings:
        failures.append("static_risk_patterns_detected")
    if missing_files:
        failures.append("target_script_missing")

    return {
        "schema": GUARD_SCHEMA,
        "protocol": "read-only pre-run resource and static leak guard; no model loading, no CUDA import, no NPZ feature loading",
        "guard_ready": not failures,
        "decision": "pass" if not failures else "block",
        "failures": failures,
        "warnings": warnings,
        "limits": {
            "max_system_used_pct": max_system_used_pct,
            "max_python_rss_mb": max_python_rss_mb,
            "max_python_process_count": max_python_process_count,
            "max_total_python_rss_mb": max_total_python_rss_mb,
            "max_gpu_memory_used_pct": max_gpu_memory_used_pct,
            "max_gpu_python_apps": max_gpu_python_apps,
            "follow_local_imports": follow_local_imports,
            "allowed_static_risks": sorted(allowed),
        },
        "system_memory": {
            "total_mb": round(memory.total_mb, 2),
            "available_mb": round(memory.available_mb, 2),
            "used_pct": round(memory.used_pct, 2),
        },
        "python_processes": {
            "count": len(python_rows),
            "total_rss_mb": round(total_python_rss_mb, 2),
            "heavy_count": len(heavy_python),
            "heavy_examples": heavy_python[:10],
        },
        "gpu": gpu,
        "static_scan": {
            "target_scripts": [str(resolve_path(path)) for path in target_scripts],
            "scanned_files": scanned_files,
            "missing_files": missing_files,
            "finding_count": len(static_findings),
            "findings": static_findings[:50],
        },
        "receipt": build_guard_receipt(
            target_scripts=target_scripts,
            command=command,
            created_at=created_at,
        ),
        "notes": [
            "Run this guard before training, evaluation, cache recovery, or any script that can allocate substantial memory.",
            "Static findings are conservative. If a risk is intentionally acceptable, pass --allow-risk with a reason in the experiment notes.",
            "Passing this guard does not prove absence of leaks; it blocks known high-risk conditions before execution.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-run resource/static leak guard.")
    parser.add_argument("--target-script", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-system-used-pct", type=float, default=90.0)
    parser.add_argument("--max-python-rss-mb", type=float, default=8192.0)
    parser.add_argument("--max-python-process-count", type=int, default=32)
    parser.add_argument("--max-total-python-rss-mb", type=float, default=16384.0)
    parser.add_argument("--max-gpu-memory-used-pct", type=float, default=95.0)
    parser.add_argument("--max-gpu-python-apps", type=int, default=0)
    parser.add_argument("--follow-local-imports", action="store_true")
    parser.add_argument("--allow-risk", action="append", default=[])
    parser.add_argument(
        "--receipt-command",
        action="append",
        default=None,
        help="Repeat for each token of the heavy command this guard receipt authorizes.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_guard(
        target_scripts=args.target_script,
        max_system_used_pct=float(args.max_system_used_pct),
        max_python_rss_mb=float(args.max_python_rss_mb),
        max_python_process_count=int(args.max_python_process_count),
        max_total_python_rss_mb=float(args.max_total_python_rss_mb),
        max_gpu_memory_used_pct=float(args.max_gpu_memory_used_pct),
        max_gpu_python_apps=int(args.max_gpu_python_apps),
        follow_local_imports=bool(args.follow_local_imports),
        allowed_risks=set(args.allow_risk or []),
        command=args.receipt_command,
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["guard_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
