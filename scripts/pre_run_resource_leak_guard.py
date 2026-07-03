#!/usr/bin/env python3
"""Pre-run resource and static leak guard for heavy Axon operations.

The guard is deliberately read-only. It does not import training modules, load
models, touch CUDA libraries, open NPZ feature arrays, scan raw sample trees, or
start worker pools. It checks current host pressure and scans target scripts for
patterns that require explicit review before execution.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import os
import platform
import re
import subprocess
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    first = raw_gpu.splitlines()[0].split(",")
    if len(first) < 3:
        return {"available": False}
    try:
        memory_used_mb = int(first[0].strip())
        memory_total_mb = int(first[1].strip())
        utilization_pct = int(first[2].strip())
    except ValueError:
        return {"available": False}

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
        "memory_used_mb": memory_used_mb,
        "memory_total_mb": memory_total_mb,
        "memory_used_pct": float(memory_used_mb / memory_total_mb * 100.0) if memory_total_mb else 0.0,
        "utilization_pct": utilization_pct,
        "compute_app_count": len(compute_apps),
        "python_compute_app_count": len(python_compute_apps),
        "python_compute_apps": python_compute_apps[:10],
    }


def scan_static_risks(paths: Sequence[Path], allowed_risks: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[dict[str, Any]] = []
    missing_files: list[str] = []
    for path in paths:
        resolved = resolve_path(path)
        if not resolved.exists():
            missing_files.append(str(resolved))
            continue
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
    return findings, missing_files


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


def evaluate_guard(
    *,
    target_scripts: Sequence[Path],
    max_system_used_pct: float = 90.0,
    max_python_rss_mb: float = 8192.0,
    max_gpu_memory_used_pct: float = 95.0,
    max_gpu_python_apps: int = 0,
    allowed_risks: Optional[set[str]] = None,
    memory_snapshot: Optional[MemorySnapshot] = None,
    python_processes: Optional[list[dict[str, Any]]] = None,
    gpu_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    allowed = set(allowed_risks or set())
    memory = memory_snapshot or collect_memory_snapshot()
    python_rows = python_processes if python_processes is not None else collect_python_processes()
    gpu = gpu_summary if gpu_summary is not None else collect_gpu_summary()
    static_findings, missing_files = scan_static_risks(target_scripts, allowed)

    failures: list[str] = []
    warnings: list[str] = []
    if memory.used_pct > max_system_used_pct:
        failures.append("system_memory_used_pct_exceeds_limit")

    heavy_python = [row for row in python_rows if float(row.get("rss_mb", 0.0)) > max_python_rss_mb]
    if heavy_python:
        failures.append("python_process_rss_exceeds_limit")

    if gpu.get("available"):
        if float(gpu.get("memory_used_pct", 0.0)) > max_gpu_memory_used_pct:
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
        "schema": "axon_loop77_pre_run_resource_leak_guard_v1",
        "protocol": "read-only pre-run resource and static leak guard; no model loading, no CUDA import, no NPZ feature loading",
        "guard_ready": not failures,
        "decision": "pass" if not failures else "block",
        "failures": failures,
        "warnings": warnings,
        "limits": {
            "max_system_used_pct": max_system_used_pct,
            "max_python_rss_mb": max_python_rss_mb,
            "max_gpu_memory_used_pct": max_gpu_memory_used_pct,
            "max_gpu_python_apps": max_gpu_python_apps,
            "allowed_static_risks": sorted(allowed),
        },
        "system_memory": {
            "total_mb": round(memory.total_mb, 2),
            "available_mb": round(memory.available_mb, 2),
            "used_pct": round(memory.used_pct, 2),
        },
        "python_processes": {
            "count": len(python_rows),
            "heavy_count": len(heavy_python),
            "heavy_examples": heavy_python[:10],
        },
        "gpu": gpu,
        "static_scan": {
            "target_scripts": [str(resolve_path(path)) for path in target_scripts],
            "missing_files": missing_files,
            "finding_count": len(static_findings),
            "findings": static_findings[:50],
        },
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
    parser.add_argument("--max-gpu-memory-used-pct", type=float, default=95.0)
    parser.add_argument("--max-gpu-python-apps", type=int, default=0)
    parser.add_argument("--allow-risk", action="append", default=[])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_guard(
        target_scripts=args.target_script,
        max_system_used_pct=float(args.max_system_used_pct),
        max_python_rss_mb=float(args.max_python_rss_mb),
        max_gpu_memory_used_pct=float(args.max_gpu_memory_used_pct),
        max_gpu_python_apps=int(args.max_gpu_python_apps),
        allowed_risks=set(args.allow_risk or []),
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["guard_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
