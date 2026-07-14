"""In-memory Loop167 EMBER control and novel projections from one parsed context."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

try:
    from loop167.ember_v3_native import (
        BYTE_ENTROPY_GLOBAL_START,
        GENERAL_START_BYTE_INDICES,
        RICH_PAIR_COUNT_INDEX,
        _byte_entropy_vector,
        _header_novel_values,
        _rich_pair_count,
    )
    from loop167.semantic_mapping import control_indices, novel_indices
except ModuleNotFoundError:  # Supports repository-root test imports.
    from src.loop167.ember_v3_native import (
        BYTE_ENTROPY_GLOBAL_START,
        GENERAL_START_BYTE_INDICES,
        RICH_PAIR_COUNT_INDEX,
        _byte_entropy_vector,
        _header_novel_values,
        _rich_pair_count,
    )
    from src.loop167.semantic_mapping import control_indices, novel_indices

from .authenticode import extract_authenticode_control
from .raw_context import RawFeatureContext

OFFICIAL_DIMENSION = 2568
HEADER_START = 696
SECTION_START = 770
IMPORT_START = 994
EXPORT_START = 2276
DIRECTORY_START = 2405
AUTHENTICODE_START = 2472
SECTION_ENTROPY_SAMPLE_BYTES = 4096
OVERLAY_ENTROPY_SAMPLE_BYTES = 65536
B1_STRING_HEAD_BYTES = 256 * 1024
B1_STRING_TAIL_BYTES = 64 * 1024
B1_STRING_MAX_CANDIDATES = 4096
B1_CONTROL_MISSING_INDICATOR_NAMES = (
    "missing_b1_byte_context",
    "missing_b1_pe_context",
    "missing_b1_directory_context",
    "missing_b1_authenticode",
)
B1_CONTROL_SAMPLING_INDICATOR_NAMES = (
    "b1_string_sampled_to_native_cap",
    "b1_string_candidate_cap_reached",
    "b1_section_or_overlay_entropy_sampled",
)

MACHINE_TYPES = (
    "IMAGE_FILE_MACHINE_UNKNOWN",
    "IMAGE_FILE_MACHINE_I386",
    "IMAGE_FILE_MACHINE_R3000",
    "IMAGE_FILE_MACHINE_R4000",
    "IMAGE_FILE_MACHINE_R10000",
    "IMAGE_FILE_MACHINE_WCEMIPSV2",
    "IMAGE_FILE_MACHINE_ALPHA",
    "IMAGE_FILE_MACHINE_SH3",
    "IMAGE_FILE_MACHINE_SH3DSP",
    "IMAGE_FILE_MACHINE_SH3E",
    "IMAGE_FILE_MACHINE_SH4",
    "IMAGE_FILE_MACHINE_SH5",
    "IMAGE_FILE_MACHINE_ARM",
    "IMAGE_FILE_MACHINE_THUMB",
    "IMAGE_FILE_MACHINE_ARMNT",
    "IMAGE_FILE_MACHINE_AM33",
    "IMAGE_FILE_MACHINE_POWERPC",
    "IMAGE_FILE_MACHINE_POWERPCFP",
    "IMAGE_FILE_MACHINE_IA64",
    "IMAGE_FILE_MACHINE_MIPS16",
    "IMAGE_FILE_MACHINE_ALPHA64",
    "IMAGE_FILE_MACHINE_AXP64",
    "IMAGE_FILE_MACHINE_MIPSFPU",
    "IMAGE_FILE_MACHINE_MIPSFPU16",
    "IMAGE_FILE_MACHINE_TRICORE",
    "IMAGE_FILE_MACHINE_CEF",
    "IMAGE_FILE_MACHINE_EBC",
    "IMAGE_FILE_MACHINE_RISCV32",
    "IMAGE_FILE_MACHINE_RISCV64",
    "IMAGE_FILE_MACHINE_RISCV128",
    "IMAGE_FILE_MACHINE_LOONGARCH32",
    "IMAGE_FILE_MACHINE_LOONGARCH64",
    "IMAGE_FILE_MACHINE_AMD64",
    "IMAGE_FILE_MACHINE_M32R",
    "IMAGE_FILE_MACHINE_ARM64",
    "IMAGE_FILE_MACHINE_CEE",
)
SUBSYSTEM_TYPES = (
    "IMAGE_SUBSYSTEM_UNKNOWN",
    "IMAGE_SUBSYSTEM_NATIVE",
    "IMAGE_SUBSYSTEM_WINDOWS_GUI",
    "IMAGE_SUBSYSTEM_WINDOWS_CUI",
    "IMAGE_SUBSYSTEM_OS2_CUI",
    "IMAGE_SUBSYSTEM_POSIX_CUI",
    "IMAGE_SUBSYSTEM_NATIVE_WINDOWS",
    "IMAGE_SUBSYSTEM_WINDOWS_CE_GUI",
    "IMAGE_SUBSYSTEM_EFI_APPLICATION",
    "IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER",
    "IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER",
    "IMAGE_SUBSYSTEM_EFI_ROM",
    "IMAGE_SUBSYSTEM_XBOX",
    "IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION",
)
COFF_CHARACTERISTIC_MASKS = (
    0x0001,
    0x0002,
    0x0004,
    0x0008,
    0x0010,
    0x0020,
    0x0040,
    0x0080,
    0x0100,
    0x0200,
    0x0400,
    0x0800,
    0x1000,
    0x2000,
    0x4000,
    0x8000,
)
DLL_CHARACTERISTIC_MASKS = (
    0x0020,
    0x0040,
    0x0080,
    0x0100,
    0x0200,
    0x0400,
    0x0800,
    0x1000,
    0x2000,
    0x4000,
    0x8000,
)
DOS_MEMBERS = (
    "e_magic",
    "e_cblp",
    "e_cp",
    "e_crlc",
    "e_cparhdr",
    "e_minalloc",
    "e_maxalloc",
    "e_ss",
    "e_sp",
    "e_csum",
    "e_ip",
    "e_cs",
    "e_lfarlc",
    "e_ovno",
    "e_oemid",
    "e_oeminfo",
    "e_lfanew",
)
DATA_DIRECTORY_NAMES = (
    "EXPORT",
    "IMPORT",
    "RESOURCE",
    "EXCEPTION",
    "SECURITY",
    "BASERELOC",
    "DEBUG",
    "COPYRIGHT",
    "GLOBALPTR",
    "TLS",
    "LOAD_CONFIG",
    "BOUND_IMPORT",
    "IAT",
    "DELAY_IMPORT",
    "COM_DESCRIPTOR",
    "RESERVED",
)


@dataclass(frozen=True)
class ContextProjection:
    """A source-ordered finite vector and the reasons a group was zero-filled."""

    values: np.ndarray
    original_indices: tuple[int, ...]
    missing_reasons: tuple[str, ...]
    complete: bool
    missing_indicators: np.ndarray
    missing_indicator_names: tuple[str, ...]
    sampling_indicators: np.ndarray
    sampling_indicator_names: tuple[str, ...]
    sampling_reasons: tuple[str, ...]


@dataclass(frozen=True)
class Loop167ContextFeatures:
    """B1 controls and M/A/CF novel values derived without another parse."""

    controls: ContextProjection
    novel: ContextProjection


def _safe_int(value: object, attribute: str) -> int:
    try:
        return int(getattr(value, attribute, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _decode_text(value: bytes | str | None) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value.lower().strip()
    try:
        return value.decode("utf-8", errors="ignore").lower().strip()
    except Exception:
        return ""


def _entropy_bits(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0].astype(np.float64) / float(len(data))
    return float(-(probabilities * np.log2(probabilities)).sum())


def _bounded_entropy_bits(bytez: bytes, *, offset: int, length: int, maximum_bytes: int) -> float:
    if offset < 0 or length <= 0 or maximum_bytes <= 0 or offset >= len(bytez):
        return 0.0
    end = min(len(bytez), offset + min(length, maximum_bytes))
    if end <= offset:
        return 0.0
    values = np.frombuffer(bytez, dtype=np.uint8, count=end - offset, offset=offset)
    counts = np.bincount(values, minlength=256)
    probabilities = counts[counts > 0].astype(np.float64) / float(values.size)
    return float(-(probabilities * np.log2(probabilities)).sum())


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _string_regexes() -> dict[str, re.Pattern[str]]:
    return {
        "url": re.compile(r"\b(?:http|https|ftp):\/\/[a-zA-Z0-9-._~:?#[\]@!$&'()*+,;=]+"),
        "ipv4_addr": re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
        "ipv6_addr": re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b|\b(?:[A-Fa-f0-9]{1,4}:){1,7}:\b|\b:[A-Fa-f0-9]{1,4}(?::[A-Fa-f0-9]{1,4}){1,6}\b"),
        "mac_addr": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b"),
        "email_addr": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b"),
        "btc_wallet": re.compile(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}"),
        "file_path": re.compile(r"\bC:/"),
        "dos_msg": re.compile(r"!This program "),
        "registry_key": re.compile(r"\b(?:KHEY_|KHLM|HKCU)"),
        "/dev/": re.compile(r"/dev/"),
        "/proc/": re.compile(r"/proc/"),
        "/bin/": re.compile(r"/bin/"),
        "/usr/": re.compile(r"/usr/"),
        "/tmp/": re.compile(r"/tmp/"),
        "/URI": re.compile(r"/URI"),
        "/FlateDecode": re.compile(r"/FlateDecode"),
        "/EmbeddedFile": re.compile(r"/EmbeddedFile"),
        "html": re.compile(r"html", re.IGNORECASE),
        "javascript": re.compile(r"javascript", re.IGNORECASE),
        "<script": re.compile(r"<script", re.IGNORECASE),
        ".click(": re.compile(r".click", re.IGNORECASE),
        "onlick": re.compile(r"onclick", re.IGNORECASE),
        "powershell": re.compile(r"powershell", re.IGNORECASE),
        "Invoke-Expression": re.compile(r"Invoke-Expression"),
        "Invoke-Command": re.compile(r"Invoke-Command"),
        "Start-process": re.compile(r"Start-process"),
        "get": re.compile(r"GET /", re.IGNORECASE),
        "post": re.compile(r"POST /", re.IGNORECASE),
        "http": re.compile(r"HTTP/", re.IGNORECASE),
        "http://": re.compile(r"http://", re.IGNORECASE),
        "https://": re.compile(r"https://", re.IGNORECASE),
        "ftp": re.compile(r"ftp:", re.IGNORECASE),
        "useragent": re.compile(r"User-Agent", re.IGNORECASE),
        "cookie": re.compile(r"cookie", re.IGNORECASE),
        "internet": re.compile(r"internet", re.IGNORECASE),
        "download": re.compile(r"download", re.IGNORECASE),
        "connect": re.compile(r"connect", re.IGNORECASE),
        "base64": re.compile(r"base64", re.IGNORECASE),
        "base64string": re.compile(r"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"),
        "crypt": re.compile(r"crypt"),
        "encode": re.compile(r"encode", re.IGNORECASE),
        "decode": re.compile(r"decode", re.IGNORECASE),
        "cache": re.compile(r"cache", re.IGNORECASE),
        "certificate": re.compile(r"certificate", re.IGNORECASE),
        "clipboard": re.compile(r"clipboard", re.IGNORECASE),
        "command": re.compile(r"command", re.IGNORECASE),
        "create": re.compile(r"create", re.IGNORECASE),
        "debug": re.compile(r"debug", re.IGNORECASE),
        "delete": re.compile(r"delete", re.IGNORECASE),
        "desktop": re.compile(r"desktop", re.IGNORECASE),
        "directory": re.compile(r"directory", re.IGNORECASE),
        "disk": re.compile(r"disk", re.IGNORECASE),
        "environment": re.compile(r"environment", re.IGNORECASE),
        "enum": re.compile(r"enum", re.IGNORECASE),
        "exit": re.compile(r"exit", re.IGNORECASE),
        "file": re.compile(r"file", re.IGNORECASE),
        "hostname": re.compile(r"hostname", re.IGNORECASE),
        "install": re.compile(r"install", re.IGNORECASE),
        "hidden": re.compile(r"hidden", re.IGNORECASE),
        "keyboard": re.compile(r"keyboard", re.IGNORECASE),
        "memory": re.compile(r"memory", re.IGNORECASE),
        "module": re.compile(r"module", re.IGNORECASE),
        "mutex": re.compile(r"mutex", re.IGNORECASE),
        "password": re.compile(r"password", re.IGNORECASE),
        "privilege": re.compile(r"privilege", re.IGNORECASE),
        "process": re.compile(r"process", re.IGNORECASE),
        "remote": re.compile(r"remote", re.IGNORECASE),
        "resource": re.compile(r"resource", re.IGNORECASE),
        "security": re.compile(r"security", re.IGNORECASE),
        "service": re.compile(r"service", re.IGNORECASE),
        "shell": re.compile(r"shell", re.IGNORECASE),
        "snapshot": re.compile(r"snapshot", re.IGNORECASE),
        "system": re.compile(r"system", re.IGNORECASE),
        "thread": re.compile(r"thread", re.IGNORECASE),
        "token": re.compile(r"token", re.IGNORECASE),
        "wallet": re.compile(r"wallet", re.IGNORECASE),
        "window": re.compile(r"window", re.IGNORECASE),
    }


STRING_REGEXES = _string_regexes()
if len(STRING_REGEXES) != 77:
    raise RuntimeError("Pinned EMBER string-regex schema drift")
STRING_REGEX_INDEX = {name: index for index, name in enumerate(sorted(STRING_REGEXES))}


def _bounded_string_sample(bytez: bytes) -> tuple[bytes, bool]:
    sample_limit = B1_STRING_HEAD_BYTES + B1_STRING_TAIL_BYTES
    if len(bytez) <= sample_limit:
        return bytez, False
    return bytez[:B1_STRING_HEAD_BYTES] + bytez[-B1_STRING_TAIL_BYTES:], True


def _string_vector_with_audit(bytez: bytes) -> tuple[np.ndarray, bool, bool]:
    vector = np.zeros(177, dtype=np.float32)
    if not bytez:
        return vector, False, False
    sample, sampled = _bounded_string_sample(bytez)
    histogram = np.zeros(96, dtype=np.float64)
    string_counts = np.zeros(77, dtype=np.float32)
    string_count = 0
    total_length = 0
    candidate_cap_reached = False
    for match in re.finditer(rb"[\x20-\x7f]{5,}", sample):
        if string_count >= B1_STRING_MAX_CANDIDATES:
            candidate_cap_reached = True
            break
        raw_string = match.group(0)
        string_count += 1
        total_length += len(raw_string)
        shifted = np.frombuffer(raw_string, dtype=np.uint8).astype(np.int16) - 0x20
        histogram += np.bincount(shifted, minlength=96)
        text = raw_string.decode("ascii", errors="ignore")
        for name, regex in STRING_REGEXES.items():
            if regex.search(text):
                string_counts[STRING_REGEX_INDEX[name]] += 1.0
    if string_count:
        printable_count = float(histogram.sum())
        probabilities = histogram[histogram > 0.0] / printable_count
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        vector[0] = float(string_count)
        vector[1] = float(total_length / string_count)
        vector[2] = printable_count
        vector[3:99] = (histogram / printable_count).astype(np.float32)
        vector[99] = entropy
    vector[100:] = string_counts
    return vector, sampled, candidate_cap_reached


def _string_vector(bytez: bytes) -> np.ndarray:
    return _string_vector_with_audit(bytez)[0]


def _category_index(raw_value: int, lookup: tuple[str, ...], namespace: object, attribute: str) -> float:
    mapping = getattr(namespace, attribute, {}) if namespace is not None else {}
    name = mapping.get(raw_value, lookup[0]) if isinstance(mapping, dict) else lookup[0]
    try:
        return float(lookup.index(name))
    except ValueError:
        return 0.0


def _dos_value(pe: object, member: str) -> int:
    dos_header = getattr(pe, "DOS_HEADER", None)
    dump_dict = getattr(dos_header, "dump_dict", None)
    if callable(dump_dict):
        try:
            return int(dump_dict().get(member, {}).get("Value", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            return 0
    return _safe_int(dos_header, member)


def _header_vector(pe: object | None) -> np.ndarray:
    vector = np.zeros(74, dtype=np.float32)
    if pe is None:
        return vector
    file_header = getattr(pe, "FILE_HEADER", None)
    optional_header = getattr(pe, "OPTIONAL_HEADER", None)
    vector[:30] = (
        _safe_int(file_header, "TimeDateStamp"),
        _safe_int(file_header, "NumberOfSections"),
        _safe_int(file_header, "NumberOfSymbols"),
        _safe_int(file_header, "SizeOfOptionalHeader"),
        _safe_int(file_header, "PointerToSymbolTable"),
        _category_index(_safe_int(file_header, "Machine"), MACHINE_TYPES, __import__("pefile"), "MACHINE_TYPE"),
        _category_index(_safe_int(optional_header, "Subsystem"), SUBSYSTEM_TYPES, __import__("pefile"), "SUBSYSTEM_TYPE"),
        _safe_int(optional_header, "MajorImageVersion"),
        _safe_int(optional_header, "MinorImageVersion"),
        _safe_int(optional_header, "MajorLinkerVersion"),
        _safe_int(optional_header, "MinorLinkerVersion"),
        _safe_int(optional_header, "MajorOperatingSystemVersion"),
        _safe_int(optional_header, "MinorOperatingSystemVersion"),
        _safe_int(optional_header, "MajorSubsystemVersion"),
        _safe_int(optional_header, "MinorSubsystemVersion"),
        _safe_int(optional_header, "SizeOfCode"),
        _safe_int(optional_header, "SizeOfHeaders"),
        _safe_int(optional_header, "SizeOfImage"),
        _safe_int(optional_header, "SizeOfInitializedData"),
        _safe_int(optional_header, "SizeOfUninitializedData"),
        _safe_int(optional_header, "SizeOfStackReserve"),
        _safe_int(optional_header, "SizeOfStackCommit"),
        _safe_int(optional_header, "SizeOfHeapReserve"),
        _safe_int(optional_header, "SizeOfHeapCommit"),
        _safe_int(optional_header, "AddressOfEntryPoint"),
        _safe_int(optional_header, "BaseOfCode"),
        _safe_int(optional_header, "ImageBase"),
        _safe_int(optional_header, "SectionAlignment"),
        _safe_int(optional_header, "CheckSum"),
        _safe_int(optional_header, "NumberOfRvaAndSizes"),
    )
    characteristics = _safe_int(file_header, "Characteristics")
    vector[30:46] = [1.0 if characteristics & mask else 0.0 for mask in COFF_CHARACTERISTIC_MASKS]
    dll_characteristics = _safe_int(optional_header, "DllCharacteristics")
    vector[46:57] = [1.0 if dll_characteristics & mask else 0.0 for mask in DLL_CHARACTERISTIC_MASKS]
    vector[57:74] = [_dos_value(pe, member) for member in DOS_MEMBERS]
    return vector


def _section_vector(context: RawFeatureContext) -> tuple[np.ndarray, bool]:
    vector = np.zeros(224, dtype=np.float32)
    pe = context.pe
    if pe is None:
        return vector, False
    sections = list(getattr(pe, "sections", []) or [])
    raw_length = len(context.bytez)
    section_records = []
    for section in sections:
        name = _decode_text(getattr(section, "Name", b"")).strip("\x00")
        raw_size = float(_safe_int(section, "SizeOfRawData"))
        virtual_size = float(_safe_int(section, "Misc_VirtualSize"))
        characteristics = _safe_int(section, "Characteristics")
        raw_offset = _safe_int(section, "PointerToRawData")
        entropy = _bounded_entropy_bits(
            context.bytez,
            offset=raw_offset,
            length=int(raw_size),
            maximum_bytes=SECTION_ENTROPY_SAMPLE_BYTES,
        )
        section_records.append(
            {
                "name": name,
                "raw_size": raw_size,
                "virtual_size": virtual_size,
                "entropy": entropy,
                "read": bool(characteristics & 0x40000000),
                "write": bool(characteristics & 0x80000000),
                "execute": bool(characteristics & 0x20000000),
            }
        )
    overlay_offset = None
    get_overlay_offset = getattr(pe, "get_overlay_data_start_offset", None)
    if callable(get_overlay_offset):
        try:
            candidate = get_overlay_offset()
            overlay_offset = int(candidate) if candidate is not None else None
        except Exception:
            overlay_offset = None
    overlay_size = (
        max(len(context.bytez) - overlay_offset, 0)
        if overlay_offset is not None and 0 <= overlay_offset <= len(context.bytez)
        else 0
    )
    overlay_entropy = _bounded_entropy_bits(
        context.bytez,
        offset=overlay_offset or 0,
        length=overlay_size,
        maximum_bytes=OVERLAY_ENTROPY_SAMPLE_BYTES,
    )
    entropies = [record["entropy"] for record in section_records] + [overlay_entropy, 0.0]
    size_ratios = [_safe_ratio(record["raw_size"], raw_length) for record in section_records] + [
        _safe_ratio(overlay_size, raw_length),
        0.0,
    ]
    virtual_ratios = [_safe_ratio(record["raw_size"], record["virtual_size"]) for record in section_records] + [0.0]
    vector[:11] = (
        len(section_records),
        sum(record["raw_size"] == 0.0 for record in section_records),
        sum(record["name"] == "" for record in section_records),
        sum(record["read"] and record["execute"] for record in section_records),
        sum(record["write"] for record in section_records),
        max(entropies),
        min(entropies),
        max(size_ratios),
        min(size_ratios),
        max(virtual_ratios),
        min(virtual_ratios),
    )
    vector[221:224] = (overlay_size, _safe_ratio(overlay_size, raw_length), overlay_entropy)
    sampled = any(record["raw_size"] > SECTION_ENTROPY_SAMPLE_BYTES for record in section_records)
    sampled = sampled or overlay_size > OVERLAY_ENTROPY_SAMPLE_BYTES
    return vector, sampled


def _import_vector(pe: object | None) -> np.ndarray:
    vector = np.zeros(1282, dtype=np.float32)
    if pe is None:
        return vector
    libraries: dict[str, list[str]] = {}
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
        dll_name = _decode_text(getattr(entry, "dll", None))
        if not dll_name:
            continue
        values: list[str] = []
        for imported in getattr(entry, "imports", []) or []:
            imported_name = getattr(imported, "name", None)
            ordinal = getattr(imported, "ordinal", None)
            if imported_name:
                values.append(_decode_text(imported_name)[:10000])
            elif ordinal is not None:
                values.append(f"{dll_name}:ordinal{int(ordinal)}")
        libraries[dll_name] = values
    if not libraries:
        return vector
    normalized_libraries = {name.lower() for name in libraries}
    imports = [name.lower() + ":" + value for name, values in libraries.items() for value in values]
    vector[:2] = (len(imports), len(normalized_libraries))
    return vector


def _export_vector(pe: object | None) -> np.ndarray:
    vector = np.zeros(129, dtype=np.float32)
    if pe is None:
        return vector
    exports = []
    directory = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
    for symbol in getattr(directory, "symbols", []) or []:
        name = getattr(symbol, "name", None)
        ordinal = getattr(symbol, "ordinal", None)
        if name:
            exports.append(_decode_text(name)[:10000])
        elif ordinal is not None:
            exports.append(f"ordinal{int(ordinal)}")
    if exports:
        vector[0] = 128.0
    return vector


def _directory_vector(pe: object | None) -> np.ndarray:
    vector = np.zeros(34, dtype=np.float32)
    if pe is None:
        return vector
    optional_header = getattr(pe, "OPTIONAL_HEADER", None)
    directories = list(getattr(optional_header, "DATA_DIRECTORY", []) or [])
    # Pin the official loop bound: only indices 0..14 are written, index 15 is dead.
    for index, directory in enumerate(directories[:15]):
        name = str(getattr(directory, "name", "")).replace("IMAGE_DIRECTORY_ENTRY_", "")
        if name not in DATA_DIRECTORY_NAMES:
            continue
        output_index = DATA_DIRECTORY_NAMES.index(name)
        vector[2 * output_index] = _safe_int(directory, "Size")
        vector[2 * output_index + 1] = _safe_int(directory, "VirtualAddress")
    has_relocs = getattr(pe, "has_relocs", None)
    has_dynamic_relocs = getattr(pe, "has_dynamic_relocs", None)
    try:
        vector[-2] = 1.0 if callable(has_relocs) and has_relocs() else 0.0
    except Exception:
        vector[-2] = 0.0
    try:
        vector[-1] = 1.0 if callable(has_dynamic_relocs) and has_dynamic_relocs() else 0.0
    except Exception:
        vector[-1] = 0.0
    return vector


def _general_vector(context: RawFeatureContext) -> np.ndarray:
    vector = np.zeros(7, dtype=np.float32)
    if not context.bytez:
        return vector
    vector[0] = float(len(context.bytez))
    vector[1] = _entropy_bits(context.bytez)
    vector[2] = 1.0 if context.pe_parse_succeeded else 0.0
    for offset in range(4):
        vector[3 + offset] = float(context.bytez[offset]) if offset < len(context.bytez) else 0.0
    return vector


def _histogram_vector(bytez: bytes) -> np.ndarray:
    if not bytez:
        return np.zeros(256, dtype=np.float32)
    counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256).astype(np.float32)
    return counts / float(counts.sum())


def _control_full_vector(
    context: RawFeatureContext,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, tuple[str, ...]]:
    context.require_open()
    full = np.zeros(OFFICIAL_DIMENSION, dtype=np.float32)
    full[:7] = _general_vector(context)
    full[7:263] = _histogram_vector(context.bytez)
    string_vector, string_sampled, string_candidate_cap_reached = _string_vector_with_audit(context.bytez)
    full[519:696] = string_vector
    full[HEADER_START : HEADER_START + 74] = _header_vector(context.pe)
    section_vector, section_entropy_sampled = _section_vector(context)
    full[SECTION_START : SECTION_START + 224] = section_vector
    full[IMPORT_START : IMPORT_START + 1282] = _import_vector(context.pe)
    full[EXPORT_START : EXPORT_START + 129] = _export_vector(context.pe)
    full[DIRECTORY_START : DIRECTORY_START + 34] = _directory_vector(context.pe)
    authenticode = extract_authenticode_control(context)
    full[AUTHENTICODE_START : AUTHENTICODE_START + 8] = authenticode.values
    reasons = list(context.missing_reasons)
    if section_entropy_sampled:
        reasons.append("section_or_overlay_entropy_sampled_to_resource_bound")
    if authenticode.reason:
        reasons.append(f"authenticode:{authenticode.reason}")
    missing_indicators = np.asarray(
        (
            1.0 if not context.bytez else 0.0,
            1.0 if not context.pe_parse_succeeded else 0.0,
            1.0 if context.directory_parse_reason else 0.0,
            1.0 if not authenticode.complete else 0.0,
        ),
        dtype=np.float32,
    )
    sampling_indicators = np.asarray(
        (
            1.0 if string_sampled else 0.0,
            1.0 if string_candidate_cap_reached else 0.0,
            1.0 if section_entropy_sampled else 0.0,
        ),
        dtype=np.float32,
    )
    sampling_reasons = []
    if string_sampled:
        sampling_reasons.append("b1_string_sampled_to_native_cap")
    if string_candidate_cap_reached:
        sampling_reasons.append("b1_string_candidate_cap_reached")
    if section_entropy_sampled:
        sampling_reasons.append("b1_section_or_overlay_entropy_sampled")
    return (
        full,
        tuple(dict.fromkeys(reasons)),
        missing_indicators,
        sampling_indicators,
        tuple(sampling_reasons),
    )


def extract_context_features(context: RawFeatureContext) -> Loop167ContextFeatures:
    """Project B1 controls and novel delta from one existing context, never reparsing bytes."""

    (
        full_controls,
        control_reasons,
        control_missing,
        control_sampling,
        control_sampling_reasons,
    ) = _control_full_vector(context)
    control_original_indices = control_indices()
    control_values = np.asarray([full_controls[index] for index in control_original_indices], dtype=np.float32)

    novel_values_by_index: dict[int, float] = {}
    for offset, index in enumerate(GENERAL_START_BYTE_INDICES):
        novel_values_by_index[index] = float(context.bytez[offset]) if offset < len(context.bytez) else 0.0
    for offset, value in enumerate(_byte_entropy_vector(context.bytez)):
        novel_values_by_index[BYTE_ENTROPY_GLOBAL_START + offset] = float(value)
    novel_values_by_index.update(_header_novel_values(context.pe))
    novel_values_by_index[RICH_PAIR_COUNT_INDEX] = _rich_pair_count(context.pe)
    novel_original_indices = novel_indices()
    novel_values = np.asarray(
        [novel_values_by_index[index] for index in novel_original_indices], dtype=np.float32
    )
    if not np.isfinite(control_values).all() or not np.isfinite(novel_values).all():
        raise ValueError("Loop167 context projection produced a non-finite value")
    novel_reasons = context.missing_reasons
    return Loop167ContextFeatures(
        controls=ContextProjection(
            control_values,
            control_original_indices,
            control_reasons,
            not control_missing.any(),
            control_missing,
            B1_CONTROL_MISSING_INDICATOR_NAMES,
            control_sampling,
            B1_CONTROL_SAMPLING_INDICATOR_NAMES,
            control_sampling_reasons,
        ),
        novel=ContextProjection(
            novel_values,
            novel_original_indices,
            novel_reasons,
            context.pe_parse_succeeded,
            np.asarray((0.0 if context.pe_parse_succeeded else 1.0,), dtype=np.float32),
            ("missing_novel_pe_context",),
            np.zeros(0, dtype=np.float32),
            (),
            (),
        ),
    )
