"""Axon nested archive scanner integration.

这个模块只负责调用 Rust 解包器并整理 JSON 结果。它不做训练标签传播：
压缩包/MSI 内层文件默认都是 unknown_training_label，避免把白+黑混合 MSI
错误地整体继承为黑样本或白样本。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCANNER_BINARY = (
    PROJECT_ROOT / "tools" / "archive_scanner" / "target" / "release" / "axon-archive-scanner.exe"
)
DEBUG_SCANNER_BINARY = (
    PROJECT_ROOT / "tools" / "archive_scanner" / "target" / "debug" / "axon-archive-scanner.exe"
)
MAX_SCANNER_OUTPUT_CHARS = 16 * 1024
MAX_SCANNER_ERROR_CHARS = 4 * 1024
SCANNER_TIMEOUT_SECONDS = 300
HARD_MAX_ARCHIVE_DEPTH = 8
HARD_MAX_ARCHIVE_FILES = 10_000
HARD_MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
HARD_MAX_ARCHIVE_FILE_BYTES = 128 * 1024 * 1024
SCANNER_TEMP_ROOT_PREFIX = "axon-archive-scanner-root-"


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _create_scanner_temp_root() -> Path:
    return Path(tempfile.mkdtemp(prefix=SCANNER_TEMP_ROOT_PREFIX))


def _cleanup_owned_temp_root(temp_root: Path) -> None:
    try:
        if temp_root.exists():
            shutil.rmtree(temp_root)
    except Exception:
        pass


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_scan_options(options: ArchiveScanOptions) -> None:
    """Apply Python-side hard limits before invoking the external scanner."""
    if not 1 <= int(options.max_depth) <= HARD_MAX_ARCHIVE_DEPTH:
        raise ValueError(f"archive max_depth must be in [1, {HARD_MAX_ARCHIVE_DEPTH}]")
    if not 1 <= int(options.max_files) <= HARD_MAX_ARCHIVE_FILES:
        raise ValueError(f"archive max_files must be in [1, {HARD_MAX_ARCHIVE_FILES}]")
    if not 1 <= int(options.max_total_bytes) <= HARD_MAX_ARCHIVE_TOTAL_BYTES:
        raise ValueError(
            f"archive max_total_bytes must be in [1, {HARD_MAX_ARCHIVE_TOTAL_BYTES}]"
        )
    if not 1 <= int(options.max_file_bytes) <= HARD_MAX_ARCHIVE_FILE_BYTES:
        raise ValueError(
            f"archive max_file_bytes must be in [1, {HARD_MAX_ARCHIVE_FILE_BYTES}]"
        )


@dataclass(frozen=True)
class ArchiveScanOptions:
    max_depth: int = 4
    max_files: int = 4096
    max_total_bytes: int = 512 * 1024 * 1024
    max_file_bytes: int = 128 * 1024 * 1024
    keep_temp: bool = False
    scanner_binary: Optional[Path] = None


@dataclass(frozen=True)
class ScannerProcessResult:
    returncode: int
    stdout: str
    stderr: str
    stdout_exceeded: bool = False
    stderr_exceeded: bool = False


def resolve_scanner_binary(explicit: Optional[str | Path] = None) -> Path:
    """找到 Rust 解包器二进制。

    优先级：
    1. 用户显式传入的路径；
    2. release 构建产物；
    3. debug 构建产物。
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([DEFAULT_SCANNER_BINARY, DEBUG_SCANNER_BINARY])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Archive scanner binary not found. Build it with: "
        'cd tools\\archive_scanner; cargo build --release'
    )


def _read_pipe_with_limit(pipe, *, limit: int, process: subprocess.Popen, exceeded: threading.Event, chunks: list[bytes]) -> None:
    total = 0
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            next_total = total + len(chunk)
            if next_total > limit:
                remaining = max(0, limit - total)
                if remaining:
                    chunks.append(chunk[:remaining])
                exceeded.set()
                try:
                    process.kill()
                except Exception:
                    pass
                break
            chunks.append(chunk)
            total = next_total
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _run_scanner_process(command: Sequence[str]) -> ScannerProcessResult:
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_exceeded = threading.Event()
    stderr_exceeded = threading.Event()
    stdout_thread = threading.Thread(
        target=_read_pipe_with_limit,
        kwargs={
            "pipe": process.stdout,
            "limit": MAX_SCANNER_OUTPUT_CHARS,
            "process": process,
            "exceeded": stdout_exceeded,
            "chunks": stdout_chunks,
        },
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_pipe_with_limit,
        kwargs={
            "pipe": process.stderr,
            "limit": MAX_SCANNER_ERROR_CHARS,
            "process": process,
            "exceeded": stderr_exceeded,
            "chunks": stderr_chunks,
        },
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=SCANNER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        finally:
            process.wait()
        raise
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    return ScannerProcessResult(
        returncode=int(returncode),
        stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
        stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        stdout_exceeded=stdout_exceeded.is_set(),
        stderr_exceeded=stderr_exceeded.is_set(),
    )


def run_archive_scan(file_path: str | Path, options: Optional[ArchiveScanOptions] = None) -> dict[str, Any]:
    """调用 Rust 解包器，返回 JSON 报告。"""
    options = options or ArchiveScanOptions()
    validate_scan_options(options)
    scanner_binary = resolve_scanner_binary(options.scanner_binary)
    temp_root = _create_scanner_temp_root()
    command = [
        str(scanner_binary),
        "--input",
        str(file_path),
        "--output",
        "json",
        "--max-depth",
        str(options.max_depth),
        "--max-files",
        str(options.max_files),
        "--max-total-bytes",
        str(options.max_total_bytes),
        "--max-file-bytes",
        str(options.max_file_bytes),
        "--temp-root",
        str(temp_root),
    ]
    if options.keep_temp:
        command.append("--keep-temp")

    try:
        result = _run_scanner_process(command)
        if result.stdout_exceeded:
            raise ValueError(
                "Archive scanner JSON output exceeded limit: "
                f">{MAX_SCANNER_OUTPUT_CHARS} chars"
            )
        if result.returncode != 0:
            raise RuntimeError(
                "Archive scanner failed:\n"
                f"command: {' '.join(command)}\n"
                f"stdout: {_truncate_text(result.stdout.strip(), MAX_SCANNER_ERROR_CHARS)}\n"
                f"stderr: {_truncate_text(result.stderr.strip(), MAX_SCANNER_ERROR_CHARS)}"
            )

        if len(result.stdout) > MAX_SCANNER_OUTPUT_CHARS:
            raise ValueError(
                "Archive scanner JSON output exceeded limit: "
                f"{len(result.stdout)} chars > {MAX_SCANNER_OUTPUT_CHARS}"
            )
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            preview = _truncate_text(result.stdout, MAX_SCANNER_ERROR_CHARS)
            raise ValueError(f"Archive scanner returned invalid JSON: {preview}") from exc
        validate_scan_report(report, options=options, temp_root=temp_root)
        if options.keep_temp:
            report["_scanner_temp_root"] = str(temp_root)
        else:
            _cleanup_owned_temp_root(temp_root)
        return report
    except BaseException:
        _cleanup_owned_temp_root(temp_root)
        raise


def validate_scan_report(
    report: dict[str, Any],
    *,
    options: Optional[ArchiveScanOptions] = None,
    temp_root: Optional[Path] = None,
) -> None:
    """做最小 schema 校验，防止后续对接误信坏 JSON。"""
    if not isinstance(report, dict):
        raise ValueError("Archive scan report must be a JSON object")
    if report.get("version") != 1:
        raise ValueError(f"Unsupported archive scan report version: {report.get('version')}")
    entries = report.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Archive scan report must contain entries list")
    if options is not None and len(entries) > int(options.max_files):
        raise ValueError(f"Archive scan report entries exceed max_files: {len(entries)} > {options.max_files}")
    limits = report.get("limits")
    if options is not None and isinstance(limits, dict):
        for field in ("max_depth", "max_files", "max_total_bytes", "max_file_bytes"):
            if field in limits and int(limits[field]) > int(getattr(options, field)):
                raise ValueError(f"Archive scan report limit {field} exceeds requested value")
    summary = report.get("summary")
    if options is not None and isinstance(summary, dict):
        observed = int(summary.get("total_observed_bytes") or 0)
        if observed > int(options.max_total_bytes):
            raise ValueError(
                "Archive scan report total_observed_bytes exceeds max_total_bytes: "
                f"{observed} > {options.max_total_bytes}"
            )
    resolved_temp_root = Path(temp_root).resolve(strict=False) if temp_root is not None else None
    temp_dir = report.get("temp_dir")
    if resolved_temp_root is not None and temp_dir:
        resolved_temp_dir = Path(temp_dir).resolve(strict=False)
        if not (resolved_temp_dir == resolved_temp_root or _path_is_relative_to(resolved_temp_dir, resolved_temp_root)):
            raise ValueError("Archive scan temp_dir is outside the owned scanner temp root")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Archive scan entry #{index} must be an object")
        for field in ("id", "depth", "logical_path", "kind", "candidate_for_axon", "archive", "status"):
            if field not in entry:
                raise ValueError(f"Archive scan entry #{index} missing field: {field}")
        if options is not None:
            depth = int(entry.get("depth") or 0)
            if depth > int(options.max_depth):
                raise ValueError(f"Archive scan entry #{index} exceeds max_depth")
            size = entry.get("size")
            if size is not None and int(size) > int(options.max_file_bytes):
                raise ValueError(f"Archive scan entry #{index} exceeds max_file_bytes")
        extracted_path = entry.get("extracted_path")
        if (
            resolved_temp_root is not None
            and extracted_path
            and entry.get("candidate_for_axon")
            and entry.get("kind") == "pe"
        ):
            resolved_extracted = Path(extracted_path).resolve(strict=False)
            if not _path_is_relative_to(resolved_extracted, resolved_temp_root):
                raise ValueError(f"Archive scan entry #{index} extracted_path is outside scanner temp root")


def iter_axon_candidates(report: dict[str, Any], *, kinds: Optional[set[str]] = None) -> Iterable[dict[str, Any]]:
    """遍历可交给 Axon 或继续审计的内层候选文件。"""
    allowed = kinds or {"pe", "msi", "zip", "7z", "rar", "cab"}
    for entry in report.get("entries", []):
        if not entry.get("candidate_for_axon"):
            continue
        if entry.get("kind") not in allowed:
            continue
        if entry.get("status") not in {"candidate", "scanned"}:
            continue
        yield entry


def iter_pe_prediction_targets(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """只返回已经落盘、可交给现有 PE 模型预测的内层 PE 文件。"""
    for entry in iter_axon_candidates(report, kinds={"pe"}):
        extracted_path = entry.get("extracted_path")
        if extracted_path:
            yield entry


def cleanup_scan_temp(report: dict[str, Any], *, trusted_roots: Optional[Sequence[Path]] = None) -> dict[str, Any]:
    """删除 Rust 解包器保留下来的临时目录，并把失败降级为状态信息。"""
    temp_dir = report.get("temp_dir")
    status: dict[str, Any] = {
        "attempted": False,
        "deleted": False,
        "temp_dir": str(temp_dir) if temp_dir else None,
    }
    if not temp_dir:
        scanner_temp_root = report.get("_scanner_temp_root")
        if not scanner_temp_root:
            status["reason"] = "no_temp_dir"
            return status
        roots = list(trusted_roots or (Path(tempfile.gettempdir()),))
        try:
            root = Path(scanner_temp_root).resolve(strict=False)
            resolved_roots = [Path(item).resolve(strict=False) for item in roots]
        except OSError as exc:
            status["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            return status
        status["attempted"] = True
        status["temp_root"] = str(root)
        if not root.name.startswith(SCANNER_TEMP_ROOT_PREFIX):
            status["reason"] = "unexpected_temp_root_prefix"
            return status
        if not any(root == trusted_root or _path_is_relative_to(root, trusted_root) for trusted_root in resolved_roots):
            status["reason"] = "temp_root_outside_trusted_roots"
            return status
        try:
            if root.exists():
                shutil.rmtree(root)
                status["temp_root_deleted"] = True
            else:
                status["reason"] = "temp_root_missing"
        except Exception as exc:  # noqa: BLE001 - cleanup must not hide prediction results.
            status["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        return status

    path = Path(temp_dir)
    scanner_temp_root = report.get("_scanner_temp_root")
    roots = list(trusted_roots or (Path(tempfile.gettempdir()),))
    try:
        resolved_path = path.resolve(strict=False)
        resolved_roots = [Path(root).resolve(strict=False) for root in roots]
    except OSError as exc:
        status["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        return status

    if not resolved_path.is_absolute():
        status["reason"] = "temp_dir_not_absolute"
        return status
    if not resolved_path.name.startswith("axon-archive-scanner-"):
        status["reason"] = "unexpected_temp_dir_prefix"
        return status
    if not any(resolved_path == root or resolved_path.is_relative_to(root) for root in resolved_roots):
        status["reason"] = "temp_dir_outside_trusted_roots"
        return status

    status["attempted"] = True
    status["temp_dir"] = str(resolved_path)
    try:
        if resolved_path.exists():
            shutil.rmtree(resolved_path)
            status["deleted"] = True
        else:
            status["reason"] = "temp_dir_missing"
    except Exception as exc:  # noqa: BLE001 - cleanup must not hide prediction results.
        status["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    if scanner_temp_root:
        root = Path(scanner_temp_root).resolve(strict=False)
        status["temp_root"] = str(root)
        if root.name.startswith(SCANNER_TEMP_ROOT_PREFIX) and any(
            root == trusted_root or root.is_relative_to(trusted_root) for trusted_root in resolved_roots
        ):
            try:
                root.rmdir()
                status["temp_root_deleted"] = True
            except OSError as exc:
                status["temp_root_deleted"] = False
                status["temp_root_cleanup_error"] = f"{type(exc).__name__}: {exc}"
    return status
