"""Fail-closed Linux guest adapter for aggregate-only Loop171 capa evidence."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .capa_aggregate import CapaAggregateError, CapabilityAggregate, aggregate_capa_json

SHA256_HEX_LENGTH = 64
MAX_CAPA_JSON_BYTES = 64 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_TOOLCHAIN_MEMBERS = 10_000
MAX_TOOLCHAIN_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
GUEST_RECEIPT_SCHEMA = "axon_loop171_guest_capa_receipt_v1"
TOOLCHAIN_RECEIPT_SCHEMA = "axon_loop171_guest_capa_toolchain_receipt_v1"


class GuestCapaError(RuntimeError):
    """Raised when the Linux guest contract cannot be proved before capa runs."""


class GuestCapaTimeoutError(GuestCapaError):
    """Raised when the capa process group exceeds its fixed time budget."""


class GuestCapaOutputLimitError(GuestCapaError):
    """Raised when capa attempts to emit more than the fixed JSON budget."""


@dataclass(frozen=True)
class GuestCapaResult:
    status: str
    aggregate: CapabilityAggregate | None


def _is_sha256(value: str) -> bool:
    return len(value) == SHA256_HEX_LENGTH and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: str, name: str) -> None:
    if not _is_sha256(value):
        raise GuestCapaError(f"{name}_invalid")


def _sha256_file(path: Path, *, expected_size: int | None = None, max_bytes: int | None = None) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise GuestCapaError("regular_file_open_failed") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise GuestCapaError("not_regular_file")
        if expected_size is not None and file_stat.st_size != expected_size:
            raise GuestCapaError("size_mismatch")
        if max_bytes is not None and (file_stat.st_size < 0 or file_stat.st_size > max_bytes):
            raise GuestCapaError("size_out_of_bounds")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _assert_existing_path_has_no_symlink(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            path_stat = os.lstat(current)
        except OSError as error:
            raise GuestCapaError("path_component_unreadable") from error
        if stat.S_ISLNK(path_stat.st_mode):
            raise GuestCapaError("symlink_path_forbidden")


def _decode_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _readonly_mount_for(path: Path, *, mountinfo_path: Path = Path("/proc/self/mountinfo")) -> None:
    try:
        rows = mountinfo_path.read_text(encoding="utf-8", errors="strict").splitlines()
    except OSError as error:
        raise GuestCapaError("mountinfo_unavailable") from error
    target = path.resolve(strict=True)
    candidate: tuple[int, bool] | None = None
    for row in rows:
        fields = row.split()
        try:
            separator = fields.index("-")
            mountpoint = Path(_decode_mount_path(fields[4]))
            mount_options = set(fields[5].split(","))
            super_options = set(fields[separator + 3].split(","))
        except (IndexError, ValueError):
            continue
        try:
            target.relative_to(mountpoint)
        except ValueError:
            continue
        score = len(mountpoint.parts)
        read_only = "ro" in mount_options or "ro" in super_options
        if candidate is None or score > candidate[0]:
            candidate = (score, read_only)
    if candidate is None or not candidate[1]:
        raise GuestCapaError("readonly_mount_required")


def _assert_readonly_regular_input(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    max_bytes: int,
    mountinfo_path: Path,
) -> None:
    _require_sha256(expected_sha256, "source_sha256")
    if expected_size < 0 or max_bytes <= 0 or expected_size > max_bytes:
        raise GuestCapaError("input_bounds_invalid")
    _assert_existing_path_has_no_symlink(path)
    _readonly_mount_for(path, mountinfo_path=mountinfo_path)
    if _sha256_file(path, expected_size=expected_size, max_bytes=max_bytes) != expected_sha256:
        raise GuestCapaError("source_sha256_mismatch")


def _tree_sha256(root: Path) -> str:
    _assert_existing_path_has_no_symlink(root)
    try:
        root_stat = root.stat()
    except OSError as error:
        raise GuestCapaError("toolchain_root_unreadable") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise GuestCapaError("rules_directory_required")
    digest = hashlib.sha256()
    files: list[Path] = []
    try:
        entries = sorted(root.rglob("*"))
    except OSError as error:
        raise GuestCapaError("rules_tree_unreadable") from error
    for path in entries:
        try:
            path_stat = path.lstat()
        except OSError as error:
            raise GuestCapaError("rules_tree_unreadable") from error
        if stat.S_ISLNK(path_stat.st_mode):
            raise GuestCapaError("rules_symlink_forbidden")
        if stat.S_ISDIR(path_stat.st_mode):
            continue
        if not stat.S_ISREG(path_stat.st_mode):
            raise GuestCapaError("rules_nonregular_file_forbidden")
        files.append(path)
    if not files:
        raise GuestCapaError("rules_directory_empty")
    for path in files:
        relative = path.relative_to(root).as_posix()
        if relative.startswith("../") or "\x00" in relative:
            raise GuestCapaError("rules_path_invalid")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _assert_linux_capa_binary(path: Path, *, expected_sha256: str) -> None:
    _require_sha256(expected_sha256, "capa_sha256")
    _assert_existing_path_has_no_symlink(path)
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise GuestCapaError("capa_binary_unreadable") from error
    if not stat.S_ISREG(mode) or (os.name == "posix" and not mode & stat.S_IXUSR):
        raise GuestCapaError("linux_capa_executable_required")
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"\x7fELF":
                raise GuestCapaError("linux_elf_capa_required")
    except OSError as error:
        raise GuestCapaError("capa_binary_unreadable") from error
    if _sha256_file(path) != expected_sha256:
        raise GuestCapaError("capa_sha256_mismatch")


def _offline_environment() -> dict[str, str]:
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "NO_PROXY": "*",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
    }
    return environment


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise GuestCapaError("process_group_termination_unconfirmed") from error


def _run_capa_json(
    *,
    capa: Path,
    rules: Path,
    source: Path,
    timeout_seconds: float,
    output_limit: int = MAX_CAPA_JSON_BYTES,
) -> bytes:
    if timeout_seconds <= 0 or output_limit <= 0:
        raise GuestCapaError("runtime_bounds_invalid")
    try:
        process = subprocess.Popen(
            [str(capa), "-j", "-r", str(rules), str(source)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd="/",
            env=_offline_environment(),
            start_new_session=True,
            close_fds=True,
        )
    except OSError as error:
        raise GuestCapaError("capa_launch_failed") from error
    assert process.stdout is not None
    chunks: list[bytes] = []
    total = 0
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_group(process)
                raise GuestCapaTimeoutError("capa_timeout")
            events = selector.select(timeout=remaining)
            if not events:
                continue
            for key, _ in events:
                chunk = os.read(key.fd, min(1024 * 1024, output_limit - total + 1))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > output_limit:
                    _terminate_process_group(process)
                    raise GuestCapaOutputLimitError("capa_output_limit")
                chunks.append(chunk)
        remaining = max(0.0, deadline - time.monotonic())
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            raise GuestCapaTimeoutError("capa_timeout") from None
    finally:
        selector.close()
        process.stdout.close()
    if returncode != 0:
        raise GuestCapaError("capa_nonzero_exit")
    return b"".join(chunks)


def _write_new_receipt(path: Path, payload: Mapping[str, object]) -> None:
    _assert_existing_path_has_no_symlink(path.parent)
    if path.exists() or path.is_symlink():
        raise GuestCapaError("receipt_overwrite_forbidden")
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise GuestCapaError("receipt_output_limit")
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".loop171-", dir=path.parent)
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        temporary = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def run_guest_capa(
    *,
    source: Path,
    source_sha256: str,
    expected_size: int,
    max_input_bytes: int,
    capa: Path,
    capa_sha256: str,
    rules: Path,
    rules_sha256: str,
    toolchain_archive: Path,
    toolchain_archive_sha256: str,
    timeout_seconds: float,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
    runner: Callable[..., bytes] = _run_capa_json,
) -> GuestCapaResult:
    """Run one SHA-bound input and retain only a fixed aggregate in process memory."""
    _require_sha256(toolchain_archive_sha256, "toolchain_archive_sha256")
    _require_sha256(rules_sha256, "rules_sha256")
    _assert_readonly_regular_input(
        source,
        expected_sha256=source_sha256,
        expected_size=expected_size,
        max_bytes=max_input_bytes,
        mountinfo_path=mountinfo_path,
    )
    _assert_existing_path_has_no_symlink(toolchain_archive)
    _readonly_mount_for(toolchain_archive, mountinfo_path=mountinfo_path)
    if _sha256_file(toolchain_archive) != toolchain_archive_sha256:
        raise GuestCapaError("toolchain_archive_sha256_mismatch")
    _assert_linux_capa_binary(capa, expected_sha256=capa_sha256)
    if _tree_sha256(rules) != rules_sha256:
        raise GuestCapaError("rules_sha256_mismatch")
    if timeout_seconds <= 0:
        raise GuestCapaError("runtime_bounds_invalid")
    raw_json = runner(capa=capa, rules=rules, source=source, timeout_seconds=timeout_seconds)
    if len(raw_json) > MAX_CAPA_JSON_BYTES:
        raise GuestCapaOutputLimitError("capa_output_limit")
    try:
        aggregate = aggregate_capa_json(json.loads(raw_json.decode("utf-8", "strict")))
    except (UnicodeDecodeError, json.JSONDecodeError, CapaAggregateError) as error:
        raise GuestCapaError("capa_json_schema_invalid") from error
    _assert_readonly_regular_input(
        source,
        expected_sha256=source_sha256,
        expected_size=expected_size,
        max_bytes=max_input_bytes,
        mountinfo_path=mountinfo_path,
    )
    return GuestCapaResult(status="ok", aggregate=aggregate)


def receipt_payload(result: GuestCapaResult | None, *, failure: str | None = None) -> dict[str, object]:
    """Return the only allowed persistent guest result schema, without identities or raw capa JSON."""
    if result is not None:
        return {
            "schema": GUEST_RECEIPT_SCHEMA,
            "status": result.status,
            "aggregate": asdict(result.aggregate) if result.aggregate is not None else None,
            "raw_or_match_location_persisted": False,
            "source_identity_persisted": False,
            "network_policy": "zero_nic_required_by_guest_acceptance_and_offline_environment",
        }
    return {
        "schema": GUEST_RECEIPT_SCHEMA,
        "status": "blocked",
        "reason": failure or "guest_contract_failed",
        "raw_or_match_location_persisted": False,
        "source_identity_persisted": False,
        "network_policy": "zero_nic_required_by_guest_acceptance_and_offline_environment",
    }


def _safe_zip_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    members = archive.infolist()
    if not 1 <= len(members) <= MAX_TOOLCHAIN_MEMBERS:
        raise GuestCapaError("toolchain_member_count_invalid")
    observed: set[str] = set()
    total = 0
    for member in members:
        name = member.filename
        normalized = Path(name)
        mode = member.external_attr >> 16
        if (
            not name
            or "\\" in name
            or normalized.is_absolute()
            or ".." in normalized.parts
            or name in observed
            or stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}
        ):
            raise GuestCapaError("toolchain_archive_path_invalid")
        observed.add(name)
        total += member.file_size
        if member.file_size < 0 or total > MAX_TOOLCHAIN_UNCOMPRESSED_BYTES:
            raise GuestCapaError("toolchain_archive_size_invalid")
        yield member


def install_linux_capa_zip(
    *,
    archive_path: Path,
    archive_sha256: str,
    destination: Path,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> dict[str, object]:
    """Extract a SHA-bound Linux capa archive only into a new, empty guest-local directory."""
    _require_sha256(archive_sha256, "toolchain_archive_sha256")
    _assert_existing_path_has_no_symlink(archive_path)
    _readonly_mount_for(archive_path, mountinfo_path=mountinfo_path)
    if _sha256_file(archive_path) != archive_sha256:
        raise GuestCapaError("toolchain_archive_sha256_mismatch")
    _assert_existing_path_has_no_symlink(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise GuestCapaError("toolchain_destination_must_be_new")
    try:
        destination.mkdir(mode=0o700, parents=False)
        with zipfile.ZipFile(archive_path) as archive:
            members = tuple(_safe_zip_members(archive))
            for member in members:
                target = destination / member.filename
                if member.is_dir():
                    target.mkdir(mode=0o700, parents=True, exist_ok=False)
                    continue
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                os.chmod(target, 0o700 if target.name == "capa" else 0o600)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise GuestCapaError("toolchain_archive_extract_failed") from error
    candidates = tuple(path for path in destination.rglob("capa") if path.is_file())
    if len(candidates) != 1:
        raise GuestCapaError("toolchain_capa_binary_ambiguous")
    rules_candidates = tuple(path for path in destination.rglob("rules") if path.is_dir())
    if len(rules_candidates) != 1:
        raise GuestCapaError("toolchain_rules_directory_ambiguous")
    capa = candidates[0]
    rules = rules_candidates[0]
    capa_sha256 = _sha256_file(capa)
    _assert_linux_capa_binary(capa, expected_sha256=capa_sha256)
    return {
        "schema": TOOLCHAIN_RECEIPT_SCHEMA,
        "toolchain_archive_sha256": archive_sha256,
        "capa_sha256": capa_sha256,
        "rules_sha256": _tree_sha256(rules),
        "raw_or_match_location_persisted": False,
        "source_identity_persisted": False,
    }
