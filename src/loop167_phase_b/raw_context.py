"""One-pass, in-memory raw feature context for the future Loop167 controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    import pefile
except ImportError:  # pragma: no cover - covered by the controller's runtime lock.
    pefile = None


DIRECTORY_ENTRY_NAMES = (
    "IMAGE_DIRECTORY_ENTRY_IMPORT",
    "IMAGE_DIRECTORY_ENTRY_EXPORT",
    "IMAGE_DIRECTORY_ENTRY_RESOURCE",
    "IMAGE_DIRECTORY_ENTRY_BASERELOC",
    "IMAGE_DIRECTORY_ENTRY_TLS",
    "IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT",
    "IMAGE_DIRECTORY_ENTRY_SECURITY",
    "IMAGE_DIRECTORY_ENTRY_EXCEPTION",
    "IMAGE_DIRECTORY_ENTRY_DEBUG",
)


def _default_pe_factory(*, data: bytes, fast_load: bool) -> Any:
    if pefile is None:
        raise RuntimeError("pefile is unavailable")
    return pefile.PE(data=data, fast_load=fast_load)


@dataclass
class RawFeatureContext:
    """Transient bytes plus at most one PE parse and one directory parse attempt."""

    bytez: bytes
    source_length: int
    maximum_input_bytes: int
    pe: object | None
    parse_reason: str | None
    directory_parse_reason: str | None
    pe_parse_attempts: int
    directory_parse_attempts: int
    _closed: bool = False

    @classmethod
    def from_bytes(
        cls,
        bytez: bytes | bytearray | memoryview,
        *,
        maximum_input_bytes: int,
        pe_factory: Callable[..., Any] = _default_pe_factory,
    ) -> "RawFeatureContext":
        """Create a context from one already-read byte stream without filesystem access."""

        if isinstance(bytez, memoryview):
            bytez = bytez.tobytes()
        elif isinstance(bytez, bytearray):
            bytez = bytes(bytez)
        if not isinstance(bytez, bytes):
            raise TypeError("RawFeatureContext accepts bytes-like content only")
        if isinstance(maximum_input_bytes, bool) or maximum_input_bytes <= 0:
            raise ValueError("maximum_input_bytes must be a positive integer")
        source_length = len(bytez)
        if source_length > maximum_input_bytes:
            return cls(b"", source_length, maximum_input_bytes, None, "oversize_input", None, 0, 0)
        if not bytez:
            return cls(bytez, source_length, maximum_input_bytes, None, "empty_input", None, 0, 0)
        try:
            parsed = pe_factory(data=bytez, fast_load=True)
        except Exception:
            return cls(bytez, source_length, maximum_input_bytes, None, "pe_parse_failure", None, 1, 0)

        directory_reason = None
        directory_attempts = 0
        parse_directories = getattr(parsed, "parse_data_directories", None)
        if callable(parse_directories):
            directory_attempts = 1
            try:
                directory_ids = []
                if pefile is not None:
                    directory_ids = [
                        pefile.DIRECTORY_ENTRY[name]
                        for name in DIRECTORY_ENTRY_NAMES
                        if name in pefile.DIRECTORY_ENTRY
                    ]
                parse_directories(directories=directory_ids)
            except Exception:
                directory_reason = "directory_parse_failure"
        return cls(
            bytez,
            source_length,
            maximum_input_bytes,
            parsed,
            None,
            directory_reason,
            1,
            directory_attempts,
        )

    @property
    def pe_parse_succeeded(self) -> bool:
        return self.pe is not None and self.parse_reason is None

    @property
    def missing_reasons(self) -> tuple[str, ...]:
        return tuple(reason for reason in (self.parse_reason, self.directory_parse_reason) if reason)

    def require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RawFeatureContext is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        parsed = self.pe
        try:
            close = getattr(parsed, "close", None)
            if callable(close):
                close()
        finally:
            self.bytez = b""
            self.pe = None

    def __enter__(self) -> "RawFeatureContext":
        self.require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
