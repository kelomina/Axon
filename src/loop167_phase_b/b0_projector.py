"""One-context B0 projection for Loop167's frozen 571+6 baseline contract."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

try:
    from kvd_features.content_pe_v1 import (
        CONTENT_API_CATEGORIES,
        DATA_DIRECTORY_INDEXES,
        SECTION_COMBO_NAMES,
        SYSTEM_DLLS,
    )
    from kvd_features.extractor import extract_statistical_features
    from loop167.semantic_mapping import frozen_baseline_columns
except ModuleNotFoundError:  # Supports repository-root test imports.
    from src.kvd_features.content_pe_v1 import (
        CONTENT_API_CATEGORIES,
        DATA_DIRECTORY_INDEXES,
        SECTION_COMBO_NAMES,
        SYSTEM_DLLS,
    )
    from src.kvd_features.extractor import extract_statistical_features
    from src.loop167.semantic_mapping import frozen_baseline_columns

from .ember_controls import _decode_text, _safe_int, _safe_ratio
from .raw_context import RawFeatureContext

STAT_PREFIX_BYTES = 65536
FIXED_SECTION_ENTROPY_BYTES = 256
CONTENT_SECTION_ENTROPY_BYTES = 4096
CONTENT_OVERLAY_ENTROPY_BYTES = 65536
CONTENT_RESOURCE_ENTROPY_BYTES = 4096
CONTENT_RESOURCE_ENTROPY_MAX_ITEMS = 64
CONTENT_STRING_HEAD_BYTES = 2 * 1024 * 1024
CONTENT_STRING_TAIL_BYTES = 512 * 1024
CONTENT_CERT_MAX_BYTES = 4 * 1024 * 1024

FIXED_PACKER_KEYWORDS = (
    "upx",
    "aspack",
    "petite",
    "pecompact",
    "themida",
    "vmprotect",
    "enigma",
    "obsidium",
    "armadillo",
    "safengine",
    "orion",
    "execryptor",
    "pelock",
    "npack",
    "nspack",
    "wwpack",
    "diminuto",
    "upack",
    "kkrunchy",
    "joexe",
    "fsg",
    "stunnix",
    "winlicense",
    "packed",
)
FIXED_API_CATEGORIES = {
    "network": ("internet", "http", "socket", "connect", "recv", "send", "url", "download", "upload", "proxy", "wsa", "ftp", "smtp"),
    "process": ("createprocess", "openprocess", "virtualalloc", "virtualprotect", "writeprocessmemory", "readprocessmemory", "createremotethread", "shellexecute", "winexec", "loadlibrary", "getprocaddress"),
    "filesystem": ("createfile", "readfile", "writefile", "deletefile", "movefile", "copyfile", "getfilesize", "setfilepointer", "findfirstfile", "findnextfile", "gettemppath"),
    "registry": ("regopenkey", "regsetvalue", "regcreatekey", "regdeletekey", "regqueryvalue", "regclosekey", "savekey", "restorekey"),
    "crypto": ("cryptencrypt", "cryptdecrypt", "cryptderivekey", "cryptgenkey", "cryptcreatehash", "crypthashdata", "cryptsignhash", "cryptverify"),
    "injection": ("createremotethread", "virtualallocex", "writeprocessmemory", "readprocessmemory", "queueuserapc", "setwindowshookex", "rtlcreateuserthread", "ntcreatethreadex"),
}
FIXED_PREFIX_ONLY_KEYWORDS = frozenset({"connect", "send", "recv"})

V2_IMPORT_DLLS = (
    "kernel32.dll",
    "ntdll.dll",
    "user32.dll",
    "advapi32.dll",
    "shell32.dll",
    "ole32.dll",
    "oleaut32.dll",
    "gdi32.dll",
    "comctl32.dll",
    "comdlg32.dll",
    "shlwapi.dll",
    "version.dll",
    "setupapi.dll",
    "msvcrt.dll",
    "ucrtbase.dll",
    "vcruntime140.dll",
    "ws2_32.dll",
    "wininet.dll",
    "winhttp.dll",
    "urlmon.dll",
    "dnsapi.dll",
    "iphlpapi.dll",
    "netapi32.dll",
    "crypt32.dll",
    "bcrypt.dll",
    "secur32.dll",
    "psapi.dll",
    "wtsapi32.dll",
    "dbghelp.dll",
    "imagehlp.dll",
    "mpr.dll",
    "wintrust.dll",
)
V2_API_CATEGORIES = {
    "service": ("openscmanager", "createservice", "startservice", "controlservice", "deleteservice"),
    "driver": ("ntloaddriver", "zwloaddriver", "deviceiocontrol", "createsymboliclink", "ioctl"),
    "privilege": ("adjusttokenprivileges", "openprocesstoken", "lookupprivilege", "impersonate"),
    "antidebug": ("isdebuggerpresent", "checkremotedebugger", "ntqueryinformationprocess", "outputdebugstring"),
    "memory": ("virtualalloc", "virtualallocex", "virtualprotect", "virtualprotectex", "heapalloc"),
    "thread": ("createthread", "createremotethread", "queueuserapc", "rtlcreateuserthread", "setthreadcontext"),
    "module": ("loadlibrary", "getprocaddress", "ldrloaddll", "freelibrary"),
    "process_enum": ("createtoolhelp32snapshot", "process32first", "process32next", "enumprocesses"),
    "persistence": ("regsetvalue", "regcreatekey", "createservice", "schtasks", "startup"),
    "network_http": ("internetopen", "internetconnect", "httpopenrequest", "httpsendrequest", "winhttp"),
    "network_socket": ("socket", "connect", "bind", "listen", "accept", "wsastartup"),
    "file_mutation": ("createfile", "writefile", "deletefile", "movefile", "copyfile", "setfileattributes"),
    "crypto_cert": ("crypt", "bcrypt", "cert", "winverifytrust"),
    "resource": ("findresource", "loadresource", "lockresource", "sizeofresource", "beginupdateresource"),
    "installer": ("msi", "setup", "install", "uninstall"),
    "com": ("cocreateinstance", "coinitialize", "clsidfromprogid", "regsvr"),
}
V2_RESOURCE_TYPES = {
    "cursor": 1,
    "bitmap": 2,
    "icon": 3,
    "menu": 4,
    "dialog": 5,
    "string": 6,
    "rcdata": 10,
    "group_cursor": 12,
    "group_icon": 14,
    "version": 16,
    "manifest": 24,
}
V2_EXPORT_PATTERNS = {
    "com": ("dllgetclassobject", "dllcanunloadnow", "dllregisterserver", "dllunregisterserver"),
    "control_panel": ("cplapplet",),
    "service": ("servicemain", "handler", "startservice"),
    "plugin": ("plugin", "initialize", "init", "register"),
}
V2_SECTION_NAME_GROUPS = {
    "code": (".text", "code"),
    "data": (".data", ".rdata", ".bss"),
    "resource": (".rsrc",),
    "import": (".idata",),
    "export": (".edata",),
    "reloc": (".reloc",),
    "tls": (".tls",),
    "packer": ("upx", "aspack", "themida", "vmprotect", "enigma", "packed", "nspack", "upack"),
}
STRING_PATTERNS = {
    "url": (b"http://", b"https://", b"www.", b"ftp://"),
    "network": (b"socket", b"connect", b"recv", b"send", b"wininet", b"ws2_32", b"internetopen", b"urldownload"),
    "script_exec": (b"powershell", b"cmd.exe", b"wscript", b"cscript", b"mshta", b"rundll32", b"regsvr32"),
    "persistence": (b"currentversion\\run", b"runonce", b"\\services\\", b"startup", b"schtasks", b"autostart"),
    "injection": (b"createremotethread", b"virtualalloc", b"virtualprotect", b"writeprocessmemory", b"queueuserapc"),
    "credential": (b"password", b"credential", b"token", b"cookie", b"browser", b"wallet"),
    "crypto": (b"cryptencrypt", b"cryptdecrypt", b"bcrypt", b"advapi32", b"base64", b"aes", b"rsa"),
    "evasion": (b"isdebuggerpresent", b"checkremotedebugger", b"ntqueryinformationprocess", b"sleep", b"sandbox"),
    "vm": (b"vmware", b"virtualbox", b"vbox", b"qemu", b"wine_get_unix_file_name"),
    "packer": (b"upx", b"themida", b"vmprotect", b"aspack", b"enigma", b"packed"),
    "file_ops": (b"createfile", b"writefile", b"deletefile", b"copyfile", b"movefile", b"findfirstfile"),
    "registry": (b"regopenkey", b"regsetvalue", b"regcreatekey", b"regdeletekey", b"regqueryvalue"),
    "benign_vendor": (b"microsoft", b"windows", b"google", b"adobe", b"intel", b"nvidia", b"mozilla", b"oracle"),
    "version_resource": (b"companyname", b"productname", b"filedescription", b"originalfilename", b"copyright"),
}
CERT_VENDOR_PATTERNS = {
    "microsoft": (b"microsoft", b"windows"),
    "digicert": (b"digicert",),
    "sectigo": (b"sectigo", b"comodo"),
    "globalsign": (b"globalsign",),
    "verisign": (b"verisign", b"symantec", b"thawte"),
    "entrust": (b"entrust",),
    "ssl_com": (b"ssl.com",),
    "google": (b"google",),
    "adobe": (b"adobe",),
    "intel": (b"intel",),
    "nvidia": (b"nvidia",),
    "oracle": (b"oracle",),
    "mozilla": (b"mozilla",),
    "kaspersky": (b"kaspersky",),
    "avast": (b"avast", b"avg technologies"),
}
CERT_OID_PATTERNS = (
    b"\x06\t*\x86H\x86\xf7\r\x01\x07\x02",
    b"\x06\x08+\x06\x01\x05\x05\x07\x03\x03",
    b"\x06\x08+\x06\x01\x05\x05\x07\x03\x08",
    b"\x06\x05+\x0e\x03\x02\x1a",
    b"\x06\t`\x86H\x01e\x03\x04\x02\x01",
    b"\x06\t`\x86H\x01e\x03\x04\x02\x02",
    b"\x06\t*\x86H\x86\xf7\r\x01\x01\x01",
    b"\x06\x08*\x86H\xce=\x04\x03\x02",
)


@dataclass(frozen=True)
class B0Projection:
    """Frozen B0 values and the six required family-missing indicators."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    missing_indicators: np.ndarray
    missing_indicator_names: tuple[str, ...]
    missing_reasons: tuple[str, ...]


def _normalized_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0].astype(np.float64) / float(len(data))
    return float(-(probabilities * np.log2(probabilities)).sum() / 8.0)


def _span_bytes(context: RawFeatureContext, offset: int, length: int, maximum_bytes: int) -> bytes:
    if offset < 0 or length <= 0 or maximum_bytes <= 0 or offset >= len(context.bytez):
        return b""
    end = min(len(context.bytez), offset + min(length, maximum_bytes))
    return context.bytez[offset:end]


def _section_bytes(context: RawFeatureContext, section: object, maximum_bytes: int) -> bytes:
    return _span_bytes(
        context,
        _safe_int(section, "PointerToRawData"),
        _safe_int(section, "SizeOfRawData"),
        maximum_bytes,
    )


def _data_directory(pe: object, index: int) -> object | None:
    directories = list(getattr(getattr(pe, "OPTIONAL_HEADER", None), "DATA_DIRECTORY", []) or [])
    return directories[index] if 0 <= index < len(directories) else None


def _overlay_span(context: RawFeatureContext) -> tuple[int, int]:
    get_offset = getattr(context.pe, "get_overlay_data_start_offset", None)
    if not callable(get_offset):
        return 0, 0
    try:
        offset = get_offset()
        offset = int(offset) if offset is not None else -1
    except Exception:
        return 0, 0
    if offset < 0 or offset > len(context.bytez):
        return 0, 0
    return offset, len(context.bytez) - offset


def _pe_family_available(context: RawFeatureContext) -> tuple[bool, str | None]:
    if context.parse_reason:
        return False, context.parse_reason
    if context.pe is None:
        return False, "pe_unavailable"
    return True, None


def _fixed_v2_features(context: RawFeatureContext) -> tuple[np.ndarray, bool, str | None]:
    vector = np.zeros(143, dtype=np.float32)
    available, reason = _pe_family_available(context)
    if not available:
        return vector, True, reason
    try:
        pe = context.pe
        file_header = getattr(pe, "FILE_HEADER", None)
        optional = getattr(pe, "OPTIONAL_HEADER", None)
        file_size = context.source_length
        vector[:18] = (
            float(file_size),
            np.log1p(float(file_size)),
            _safe_int(file_header, "SizeOfOptionalHeader"),
            _safe_ratio(_safe_int(file_header, "SizeOfOptionalHeader") + 24, file_size),
            _safe_int(optional, "Subsystem"),
            _safe_int(optional, "DllCharacteristics"),
            _safe_int(optional, "CheckSum"),
            1.0 if _safe_int(optional, "CheckSum") == 0 else 0.0,
            1.0 if _safe_int(optional, "DllCharacteristics") & 0x0040 else 0.0,
            1.0 if _safe_int(optional, "DllCharacteristics") & 0x0080 else 0.0,
            1.0 if _safe_int(optional, "DllCharacteristics") & 0x4000 else 0.0,
            1.0 if _safe_int(file_header, "Characteristics") & 0x0004 else 0.0,
            1.0 if hasattr(pe, "DIRECTORY_ENTRY_DEBUG") else 0.0,
            1.0 if hasattr(pe, "DIRECTORY_ENTRY_BASERELOC") else 0.0,
            1.0 if hasattr(pe, "DIRECTORY_ENTRY_TLS") else 0.0,
            1.0 if hasattr(pe, "DIRECTORY_ENTRY_EXCEPTION") else 0.0,
            1.0 if hasattr(pe, "DIRECTORY_ENTRY_SECURITY") else 0.0,
            _safe_int(file_header, "NumberOfSections"),
        )
        sections = list(getattr(pe, "sections", []) or [])
        section_sizes: list[float] = []
        section_virtual_sizes: list[float] = []
        section_entropies: list[float] = []
        section_names: list[str] = []
        for slot in range(32):
            if slot < len(sections):
                chars = _safe_int(sections[slot], "Characteristics")
                vector[18 + 3 * slot : 21 + 3 * slot] = (
                    1.0 if chars & 0x20000000 else 0.0,
                    1.0 if chars & 0x80000000 else 0.0,
                    1.0 if chars & 0x40000000 else 0.0,
                )
        for section in sections:
            raw_size = float(_safe_int(section, "SizeOfRawData"))
            virtual_size = float(_safe_int(section, "Misc_VirtualSize"))
            section_sizes.append(raw_size)
            section_virtual_sizes.append(virtual_size)
            section_names.append(_decode_text(getattr(section, "Name", b"")).strip("\x00"))
            if 0 < raw_size < 10 * 1024 * 1024:
                sample = _section_bytes(context, section, FIXED_SECTION_ENTROPY_BYTES)
                if sample:
                    section_entropies.append(_normalized_entropy(sample))
        aggregate_start = 114
        if section_entropies:
            vector[aggregate_start : aggregate_start + 5] = (
                max(section_entropies),
                min(section_entropies),
                float(np.mean(section_entropies)),
                float(np.std(section_entropies)),
                _safe_ratio(sum(value > 0.8 for value in section_entropies), len(section_entropies)),
            )
        if section_sizes:
            average_raw = float(np.mean(section_sizes))
            vector[119:127] = (
                sum(section_sizes),
                sum(section_virtual_sizes),
                average_raw,
                float(np.mean(section_virtual_sizes)),
                min(section_sizes),
                max(section_sizes),
                float(np.std(section_sizes)),
                _safe_ratio(float(np.std(section_sizes)), max(average_raw, 1.0)),
            )
        valid_names = [name for name in section_names if name]
        vector[127] = len(valid_names)
        if valid_names:
            lengths = [len(name) for name in valid_names]
            vector[128:131] = (float(np.mean(lengths)), max(lengths), min(lengths))
        if section_sizes and float(np.mean(section_sizes)) > 0.0:
            average_raw = float(np.mean(section_sizes))
            long_sections = sum(size > 2.0 * average_raw for size in section_sizes)
            short_sections = sum(size < 0.5 * average_raw for size in section_sizes)
            vector[131:135] = (
                long_sections,
                _safe_ratio(long_sections, len(section_sizes)),
                short_sections,
                _safe_ratio(short_sections, len(section_sizes)),
            )
        category_counts = {name: 0 for name in FIXED_API_CATEGORIES}
        total_apis = 0
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
            for imported in getattr(entry, "imports", []) or []:
                api_name = _decode_text(getattr(imported, "name", None))
                if not api_name:
                    continue
                total_apis += 1
                for category, keywords in FIXED_API_CATEGORIES.items():
                    if any(
                        api_name.startswith(keyword) if keyword in FIXED_PREFIX_ONLY_KEYWORDS else keyword in api_name
                        for keyword in keywords
                    ):
                        category_counts[category] += 1
        for offset, category in enumerate(("network", "process", "filesystem", "registry", "crypto", "injection")):
            vector[135 + offset] = _safe_ratio(category_counts[category], total_apis)
        packer_hits = sum(
            any(keyword in name.lower() for keyword in FIXED_PACKER_KEYWORDS) for name in section_names
        )
        vector[141:143] = (packer_hits, _safe_ratio(packer_hits, max(_safe_int(file_header, "NumberOfSections"), 1)))
        return np.nan_to_num(vector, copy=False), False, None
    except Exception:
        return np.zeros(143, dtype=np.float32), True, "fixed_v2_native_failure"


def _stat_features(context: RawFeatureContext) -> tuple[np.ndarray, bool, str | None]:
    if not context.bytez:
        return np.zeros(49, dtype=np.float32), True, context.parse_reason or "empty_input"
    try:
        prefix = context.bytez[:STAT_PREFIX_BYTES]
        values = extract_statistical_features(
            np.frombuffer(prefix, dtype=np.uint8), orig_length=len(prefix)
        ).astype(np.float32, copy=False)
        if values.shape != (49,) or not np.isfinite(values).all():
            raise ValueError("stat feature shape or finiteness drift")
        return values, False, None
    except Exception:
        return np.zeros(49, dtype=np.float32), True, "stat_native_failure"


def _content_string_sample(bytez: bytes) -> bytes:
    if len(bytez) > CONTENT_STRING_HEAD_BYTES + CONTENT_STRING_TAIL_BYTES:
        return bytez[:CONTENT_STRING_HEAD_BYTES] + bytez[-CONTENT_STRING_TAIL_BYTES:]
    return bytez[:CONTENT_STRING_HEAD_BYTES]


def _ascii_run_lengths(data: bytes, *, min_len: int = 4) -> list[int]:
    values: list[int] = []
    current = 0
    for value in data:
        if 32 <= value <= 126:
            current += 1
        else:
            if current >= min_len:
                values.append(current)
            current = 0
    if current >= min_len:
        values.append(current)
    return values


def _utf16_ascii_run_lengths(data: bytes, *, min_len: int = 4) -> list[int]:
    values: list[int] = []
    current = 0
    index = 0
    while index < len(data) - 1:
        if 32 <= data[index] <= 126 and data[index + 1] == 0:
            current += 1
            index += 2
        else:
            if current >= min_len:
                values.append(current)
            current = 0
            index += 1
    if current >= min_len:
        values.append(current)
    return values


def _content_string_features(context: RawFeatureContext) -> tuple[np.ndarray, bool, str | None]:
    data = _content_string_sample(context.bytez)
    if not data:
        return np.zeros(43, dtype=np.float32), True, context.parse_reason or "empty_input"
    try:
        byte_values = np.frombuffer(data, dtype=np.uint8)
        lowered = data.lower()
        ascii_printable = int(np.count_nonzero((byte_values >= 32) & (byte_values <= 126)))
        null_count = int(np.count_nonzero(byte_values == 0))
        high_byte_count = int(np.count_nonzero(byte_values >= 128))
        ascii_runs = _ascii_run_lengths(data)
        utf16_runs = _utf16_ascii_run_lengths(data)
        features = [
            math.log1p(len(data)),
            _safe_ratio(ascii_printable, len(data)),
            _safe_ratio(null_count, len(data)),
            _safe_ratio(high_byte_count, len(data)),
            math.log1p(len(ascii_runs)),
            _safe_ratio(len(ascii_runs), len(data) / 1024.0),
            min(float(np.mean(ascii_runs)) if ascii_runs else 0.0, 512.0) / 512.0,
            min(max(ascii_runs) if ascii_runs else 0, 4096.0) / 4096.0,
            math.log1p(len(utf16_runs)),
            _safe_ratio(len(utf16_runs), len(data) / 1024.0),
            math.log1p(len(re.findall(rb"https?://[^\s\x00\"']+", lowered))),
            math.log1p(len(re.findall(rb"\b(?:\d{1,3}\.){3}\d{1,3}\b", lowered))),
            math.log1p(lowered.count(b"\\software\\") + lowered.count(b"\\registry\\") + lowered.count(b"hkey_")),
            math.log1p(lowered.count(b"c:\\") + lowered.count(b"\\windows\\") + lowered.count(b"\\system32\\")),
            _normalized_entropy(data),
        ]
        for patterns in STRING_PATTERNS.values():
            count = sum(lowered.count(pattern.lower()) for pattern in patterns)
            features.extend((math.log1p(count), 1.0 if count else 0.0))
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (43,) or not np.isfinite(vector).all():
            raise ValueError("content string feature shape or finiteness drift")
        return vector, False, None
    except Exception:
        return np.zeros(43, dtype=np.float32), True, "content_string_native_failure"


def _certificate_blob(context: RawFeatureContext) -> tuple[bytes, int, int, int, bool]:
    if context.pe is None:
        return b"", 0, 0, 0, True
    directory = _data_directory(context.pe, DATA_DIRECTORY_INDEXES["security"])
    offset = _safe_int(directory, "VirtualAddress")
    declared_size = _safe_int(directory, "Size")
    if offset <= 0 or declared_size <= 0:
        return b"", 0, 0, 0, False
    if offset > len(context.bytez) or offset + declared_size > len(context.bytez):
        return b"", declared_size, 0, 0, True
    blob = context.bytez[offset : offset + min(declared_size, CONTENT_CERT_MAX_BYTES)]
    revision = int.from_bytes(blob[4:6], "little", signed=False) if len(blob) >= 6 else 0
    cert_type = int.from_bytes(blob[6:8], "little", signed=False) if len(blob) >= 8 else 0
    return blob, declared_size, revision, cert_type, False


def _content_cert_features(context: RawFeatureContext) -> tuple[np.ndarray, bool, str | None]:
    available, reason = _pe_family_available(context)
    if not available:
        return np.zeros(55, dtype=np.float32), True, reason
    try:
        blob, declared_size, revision, cert_type, malformed = _certificate_blob(context)
        if malformed:
            return np.zeros(55, dtype=np.float32), True, "certificate_directory_out_of_bounds"
        if not blob:
            return np.zeros(55, dtype=np.float32), False, None
        byte_values = np.frombuffer(blob, dtype=np.uint8)
        ascii_printable = int(np.count_nonzero((byte_values >= 32) & (byte_values <= 126)))
        null_count = int(np.count_nonzero(byte_values == 0))
        high_byte_count = int(np.count_nonzero(byte_values >= 128))
        ascii_runs = _ascii_run_lengths(blob)
        utf16_runs = _utf16_ascii_run_lengths(blob)
        lowered = blob.lower()
        utf16_text = blob.decode("utf-16le", errors="ignore").lower().encode("utf-8", errors="ignore")
        searchable = lowered + b"\n" + utf16_text
        features = [
            1.0,
            math.log1p(declared_size or len(blob)),
            _safe_ratio(declared_size or len(blob), context.source_length),
            _safe_ratio(int.from_bytes(blob[:4], "little", signed=False) if len(blob) >= 4 else 0, declared_size or len(blob)),
            revision / 65535.0,
            cert_type / 65535.0,
            _normalized_entropy(blob),
            _safe_ratio(ascii_printable, len(blob)),
            _safe_ratio(null_count, len(blob)),
            _safe_ratio(high_byte_count, len(blob)),
            math.log1p(len(ascii_runs)),
            min(float(np.mean(ascii_runs)) if ascii_runs else 0.0, 512.0) / 512.0,
            min(max(ascii_runs) if ascii_runs else 0, 4096.0) / 4096.0,
            math.log1p(len(utf16_runs)),
            math.log1p(blob.count(b"\x30\x82")),
            1.0 if b"timestamp" in searchable or b"time stamp" in searchable else 0.0,
            1.0 if b"countersign" in searchable or b"counter sign" in searchable else 0.0,
        ]
        features.extend(1.0 if pattern in blob else 0.0 for pattern in CERT_OID_PATTERNS)
        for patterns in CERT_VENDOR_PATTERNS.values():
            count = sum(searchable.count(pattern.lower()) for pattern in patterns)
            features.extend((1.0 if count else 0.0, math.log1p(count)))
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (55,) or not np.isfinite(vector).all():
            raise ValueError("content cert feature shape or finiteness drift")
        return vector, False, None
    except Exception:
        return np.zeros(55, dtype=np.float32), True, "content_cert_native_failure"


def _content_pe_v1_features(context: RawFeatureContext) -> tuple[np.ndarray, bool, str | None]:
    available, reason = _pe_family_available(context)
    if not available:
        return np.zeros(100, dtype=np.float32), True, reason
    try:
        pe = context.pe
        file_header = getattr(pe, "FILE_HEADER", None)
        optional = getattr(pe, "OPTIONAL_HEADER", None)
        file_size = context.source_length
        characteristics = _safe_int(file_header, "Characteristics")
        timestamp = _safe_int(file_header, "TimeDateStamp")
        timestamp_year = 1970.0 + timestamp / 31557600.0 if timestamp > 0 else 0.0
        timestamp_valid = 1.0 if 1970.0 <= timestamp_year <= 2099.0 else 0.0
        timestamp_norm = (min(max(timestamp_year, 1970.0), 2099.0) - 1970.0) / 129.0 if timestamp_valid else 0.0
        size_of_code = float(_safe_int(optional, "SizeOfCode"))
        size_initialized = float(_safe_int(optional, "SizeOfInitializedData"))
        size_uninitialized = float(_safe_int(optional, "SizeOfUninitializedData"))
        entry_point = float(_safe_int(optional, "AddressOfEntryPoint"))
        image_base = float(_safe_int(optional, "ImageBase"))
        section_alignment = float(_safe_int(optional, "SectionAlignment"))
        file_alignment = float(_safe_int(optional, "FileAlignment"))
        size_of_image = float(_safe_int(optional, "SizeOfImage"))
        size_of_headers = float(_safe_int(optional, "SizeOfHeaders"))
        features: list[float] = [
            math.log1p(float(file_size)),
            min(float(file_size), 100.0 * 1024 * 1024) / (100.0 * 1024 * 1024),
            _safe_int(file_header, "Machine") / 65535.0,
            characteristics / 65535.0,
            min(float(_safe_int(file_header, "NumberOfSections")), 64.0) / 64.0,
            timestamp_valid,
            timestamp_norm,
            _safe_int(optional, "Magic") / 65535.0,
            min(float(_safe_int(optional, "MajorLinkerVersion")), 255.0) / 255.0,
            min(float(_safe_int(optional, "MinorLinkerVersion")), 255.0) / 255.0,
            _safe_ratio(size_of_code, file_size),
            _safe_ratio(size_initialized, file_size),
            _safe_ratio(size_uninitialized, file_size),
            _safe_ratio(entry_point, max(size_of_image, file_size, 1.0)),
            math.log1p(max(image_base, 0.0)) / 64.0,
            math.log1p(max(section_alignment, 0.0)) / 16.0,
            math.log1p(max(file_alignment, 0.0)) / 16.0,
            _safe_ratio(size_of_image, file_size),
            _safe_ratio(size_of_headers, file_size),
            _safe_int(optional, "Subsystem") / 32.0,
            _safe_int(optional, "DllCharacteristics") / 65535.0,
            1.0 if characteristics & 0x2000 else 0.0,
            1.0 if characteristics & 0x0002 else 0.0,
            1.0 if characteristics & 0x1000 else 0.0,
            1.0 if characteristics & 0x0020 else 0.0,
            1.0 if characteristics & 0x0100 else 0.0,
            1.0 if characteristics & 0x0001 else 0.0,
            1.0 if characteristics & 0x0200 else 0.0,
        ]
        for index in DATA_DIRECTORY_INDEXES.values():
            directory = _data_directory(pe, index)
            size = float(_safe_int(directory, "Size"))
            address = _safe_int(directory, "VirtualAddress")
            features.extend((1.0 if size > 0.0 or address > 0 else 0.0, math.log1p(max(size, 0.0)), _safe_ratio(size, file_size)))

        import_dlls: list[str] = []
        import_names: list[str] = []
        imports_per_dll: list[int] = []
        ordinal_imports = 0
        api_counts = {name: 0 for name in CONTENT_API_CATEGORIES}
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
            dll_name = _decode_text(getattr(entry, "dll", None))
            if dll_name:
                import_dlls.append(dll_name)
            entry_count = 0
            for imported in getattr(entry, "imports", []) or []:
                entry_count += 1
                api_name = _decode_text(getattr(imported, "name", None))
                if api_name:
                    import_names.append(api_name)
                    for category, keywords in CONTENT_API_CATEGORIES.items():
                        if any(keyword in api_name for keyword in keywords):
                            api_counts[category] += 1
                else:
                    ordinal_imports += 1
            imports_per_dll.append(entry_count)
        total_imports = len(import_names) + ordinal_imports
        unique_imports = len(set(import_names))
        unique_dlls = set(import_dlls)
        features.extend(
            (
                math.log1p(len(unique_dlls)),
                math.log1p(total_imports),
                math.log1p(unique_imports),
                _safe_ratio(ordinal_imports, total_imports),
                _safe_ratio(sum(dll in SYSTEM_DLLS for dll in unique_dlls), len(unique_dlls)),
                _safe_ratio(total_imports, len(imports_per_dll)),
                _safe_ratio(max(imports_per_dll) if imports_per_dll else 0, 512.0),
            )
        )
        features.extend(_safe_ratio(api_counts[category], total_imports) for category in CONTENT_API_CATEGORIES)

        exports = list(getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", []) or [])
        export_count = len(exports)
        export_name_count = sum(bool(getattr(symbol, "name", None)) for symbol in exports)
        features.extend((math.log1p(export_count), _safe_ratio(export_name_count, export_count)))

        resource_entry_count = 0
        resource_types: set[object] = set()
        stack = list(getattr(getattr(pe, "DIRECTORY_ENTRY_RESOURCE", None), "entries", []) or [])
        while stack:
            entry = stack.pop()
            resource_entry_count += 1
            if hasattr(entry, "id"):
                resource_types.add(getattr(entry, "id"))
            stack.extend(list(getattr(getattr(entry, "directory", None), "entries", []) or []))
        features.extend((math.log1p(resource_entry_count), math.log1p(len(resource_types))))

        tls_struct = getattr(getattr(pe, "DIRECTORY_ENTRY_TLS", None), "struct", None)
        features.append(math.log1p(1 if _safe_int(tls_struct, "AddressOfCallBacks") else 0))
        relocation_blocks = list(getattr(pe, "DIRECTORY_ENTRY_BASERELOC", []) or [])
        features.extend((math.log1p(len(relocation_blocks)), math.log1p(sum(len(getattr(block, "entries", []) or []) for block in relocation_blocks))))

        overlay_offset, overlay_size = _overlay_span(context)
        overlay = _span_bytes(context, overlay_offset, overlay_size, CONTENT_OVERLAY_ENTROPY_BYTES)
        features.extend((1.0 if overlay_size else 0.0, math.log1p(float(overlay_size)), _safe_ratio(overlay_size, file_size), _normalized_entropy(overlay)))

        common_sections = {".text", ".data", ".rdata", ".rsrc", ".idata", ".edata", ".bss", ".reloc", ".tls"}
        packer_keywords = ("upx", "aspack", "themida", "vmprotect", "enigma", "packed", "nspack", "upack")
        combo_counts = {name: 0 for name in SECTION_COMBO_NAMES}
        section_entropies: list[float] = []
        nonstandard_names = 0
        raw_virtual_mismatch = 0
        zero_raw = 0
        packer_hits = 0
        sections = list(getattr(pe, "sections", []) or [])
        for section in sections:
            chars = _safe_int(section, "Characteristics")
            is_execute = bool(chars & 0x20000000)
            is_write = bool(chars & 0x80000000)
            is_read = bool(chars & 0x40000000)
            if is_execute and is_read and is_write:
                combo_counts["rwx"] += 1
            elif is_execute and is_write:
                combo_counts["wx"] += 1
            elif is_execute and is_read:
                combo_counts["rx"] += 1
            elif is_read and is_write:
                combo_counts["rw"] += 1
            elif is_execute:
                combo_counts["exec_only"] += 1
            elif is_read:
                combo_counts["read_only"] += 1
            elif is_write:
                combo_counts["write_only"] += 1
            else:
                combo_counts["none"] += 1
            section_name = _decode_text(getattr(section, "Name", b"")).strip("\x00")
            nonstandard_names += int(bool(section_name and section_name not in common_sections))
            packer_hits += int(any(keyword in section_name for keyword in packer_keywords))
            raw_size = float(_safe_int(section, "SizeOfRawData"))
            virtual_size = float(_safe_int(section, "Misc_VirtualSize"))
            zero_raw += int(raw_size <= 0.0)
            if max(raw_size, virtual_size) > 0.0 and abs(raw_size - virtual_size) / max(raw_size, virtual_size) > 0.5:
                raw_virtual_mismatch += 1
            if raw_size > 0.0:
                section_entropies.append(_normalized_entropy(_section_bytes(context, section, CONTENT_SECTION_ENTROPY_BYTES)))
        section_count = max(len(sections), 1)
        features.extend(_safe_ratio(combo_counts[name], section_count) for name in SECTION_COMBO_NAMES)
        features.extend(
            (
                _safe_ratio(nonstandard_names, section_count),
                _safe_ratio(sum(value >= 0.8 for value in section_entropies), len(section_entropies)),
                _safe_ratio(raw_virtual_mismatch, section_count),
                _safe_ratio(zero_raw, section_count),
                float(np.mean(section_entropies)) if section_entropies else 0.0,
                float(np.max(section_entropies)) if section_entropies else 0.0,
                _safe_ratio(packer_hits, section_count),
            )
        )
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (100,) or not np.isfinite(vector).all():
            raise ValueError("content PE v1 feature shape or finiteness drift")
        return vector, False, None
    except Exception:
        return np.zeros(100, dtype=np.float32), True, "content_pe_v1_native_failure"


def _iter_import_entries(pe: object) -> list[tuple[str, object]]:
    entries: list[tuple[str, object]] = []
    for directory_name in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
        for entry in getattr(pe, directory_name, []) or []:
            entries.append((directory_name, entry))
    return entries


def _content_pe_v2_features(context: RawFeatureContext) -> tuple[np.ndarray, bool, str | None]:
    available, reason = _pe_family_available(context)
    if not available:
        return np.zeros(182, dtype=np.float32), True, reason
    try:
        pe = context.pe
        features: list[float] = []
        import_dlls: list[str] = []
        import_names: list[str] = []
        ordinal_imports = 0
        dll_api_counts = {name: 0 for name in V2_IMPORT_DLLS}
        delay_import_dlls: set[str] = set()
        delay_import_api_count = 0
        category_counts = {name: 0 for name in V2_API_CATEGORIES}
        for directory_name, entry in _iter_import_entries(pe):
            dll_name = _decode_text(getattr(entry, "dll", None)).split("\\")[-1]
            if dll_name:
                import_dlls.append(dll_name)
                if directory_name == "DIRECTORY_ENTRY_DELAY_IMPORT":
                    delay_import_dlls.add(dll_name)
            entry_count = 0
            for imported in getattr(entry, "imports", []) or []:
                entry_count += 1
                if directory_name == "DIRECTORY_ENTRY_DELAY_IMPORT":
                    delay_import_api_count += 1
                api_name = _decode_text(getattr(imported, "name", None))
                if api_name:
                    import_names.append(api_name)
                    for category, keywords in V2_API_CATEGORIES.items():
                        if any(keyword in api_name for keyword in keywords):
                            category_counts[category] += 1
                else:
                    ordinal_imports += 1
            if dll_name in dll_api_counts:
                dll_api_counts[dll_name] += entry_count
        total_imports = len(import_names) + ordinal_imports
        imported_dlls = set(import_dlls)
        for dll_name in V2_IMPORT_DLLS:
            features.extend((1.0 if dll_name in imported_dlls else 0.0, _safe_ratio(dll_api_counts[dll_name], total_imports)))
        for category in V2_API_CATEGORIES:
            count = category_counts[category]
            features.extend((1.0 if count else 0.0, math.log1p(count), _safe_ratio(count, total_imports)))
        features.extend((math.log1p(len(delay_import_dlls)), math.log1p(delay_import_api_count), _safe_ratio(delay_import_api_count, total_imports)))

        exports = list(getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", []) or [])
        export_count = len(exports)
        export_name_count = 0
        export_forwarder_count = 0
        export_name_lengths: list[int] = []
        export_ordinals: list[int] = []
        export_pattern_hits = {name: 0 for name in V2_EXPORT_PATTERNS}
        for symbol in exports:
            ordinal = getattr(symbol, "ordinal", None)
            if ordinal is not None:
                export_ordinals.append(int(ordinal))
            if getattr(symbol, "forwarder", None):
                export_forwarder_count += 1
            export_name = _decode_text(getattr(symbol, "name", None))
            if export_name:
                export_name_count += 1
                export_name_lengths.append(len(export_name))
                for pattern_name, keywords in V2_EXPORT_PATTERNS.items():
                    if any(keyword in export_name for keyword in keywords):
                        export_pattern_hits[pattern_name] += 1
        ordinal_span = max(export_ordinals) - min(export_ordinals) + 1 if export_ordinals else 0
        features.extend(
            (
                _safe_ratio(export_count - export_name_count, export_count),
                _safe_ratio(export_forwarder_count, export_count),
                _safe_ratio(float(np.mean(export_name_lengths)) if export_name_lengths else 0.0, 128.0),
                _safe_ratio(max(export_name_lengths) if export_name_lengths else 0, 256.0),
                math.log1p(ordinal_span),
            )
        )
        features.extend(1.0 if export_pattern_hits[name] else 0.0 for name in V2_EXPORT_PATTERNS)

        resource_entries = 0
        resource_named_entries = 0
        resource_type_counts = {name: 0 for name in V2_RESOURCE_TYPES}
        resource_languages: set[int] = set()
        resource_sizes: list[int] = []
        resource_entropies: list[float] = []
        stack = [(entry, 0, getattr(entry, "id", None)) for entry in getattr(getattr(pe, "DIRECTORY_ENTRY_RESOURCE", None), "entries", []) or []]
        while stack:
            entry, depth, root_type = stack.pop()
            resource_entries += 1
            if getattr(entry, "name", None) is not None:
                resource_named_entries += 1
            if depth == 0:
                root_type = getattr(entry, "id", root_type)
            if depth == 2 and getattr(entry, "id", None) is not None:
                resource_languages.add(int(entry.id))
            for resource_name, resource_id in V2_RESOURCE_TYPES.items():
                if root_type == resource_id:
                    resource_type_counts[resource_name] += 1
            data_struct = getattr(getattr(entry, "data", None), "struct", None)
            size = _safe_int(data_struct, "Size")
            resource_rva = _safe_int(data_struct, "OffsetToData")
            if size > 0:
                resource_sizes.append(size)
                if len(resource_entropies) < CONTENT_RESOURCE_ENTROPY_MAX_ITEMS:
                    try:
                        offset = int(pe.get_offset_from_rva(resource_rva))
                        resource_entropies.append(
                            _normalized_entropy(_span_bytes(context, offset, size, CONTENT_RESOURCE_ENTROPY_BYTES))
                        )
                    except Exception:
                        pass
            stack.extend(
                (child, depth + 1, root_type)
                for child in getattr(getattr(entry, "directory", None), "entries", []) or []
            )
        resource_total_size = sum(resource_sizes)
        features.extend(
            (
                math.log1p(len(resource_sizes)),
                _safe_ratio(resource_named_entries, resource_entries),
                math.log1p(len(resource_languages)),
                math.log1p(resource_total_size),
                _safe_ratio(max(resource_sizes) if resource_sizes else 0, context.source_length),
                float(np.mean(resource_entropies)) if resource_entropies else 0.0,
                float(np.max(resource_entropies)) if resource_entropies else 0.0,
            )
        )
        for resource_name in V2_RESOURCE_TYPES:
            count = resource_type_counts[resource_name]
            features.extend((1.0 if count else 0.0, math.log1p(count)))

        optional = getattr(pe, "OPTIONAL_HEADER", None)
        entry_point = _safe_int(optional, "AddressOfEntryPoint")
        section_infos = []
        group_hits = {name: 0 for name in V2_SECTION_NAME_GROUPS}
        for section in getattr(pe, "sections", []) or []:
            characteristics = _safe_int(section, "Characteristics")
            raw_size = float(_safe_int(section, "SizeOfRawData"))
            virtual_size = float(_safe_int(section, "Misc_VirtualSize"))
            virtual_address = _safe_int(section, "VirtualAddress")
            virtual_span = max(int(virtual_size), int(raw_size), 1)
            name = _decode_text(getattr(section, "Name", b"")).strip("\x00")
            for group_name, keywords in V2_SECTION_NAME_GROUPS.items():
                if any(keyword in name for keyword in keywords):
                    group_hits[group_name] += 1
            section_infos.append(
                {
                    "exec": bool(characteristics & 0x20000000),
                    "read": bool(characteristics & 0x40000000),
                    "write": bool(characteristics & 0x80000000),
                    "zero_raw": raw_size <= 0.0,
                    "entropy": _normalized_entropy(_section_bytes(context, section, CONTENT_SECTION_ENTROPY_BYTES)),
                    "contains_ep": virtual_address <= entry_point < virtual_address + virtual_span,
                    "raw_virtual_delta": abs(raw_size - virtual_size) / max(raw_size, virtual_size, 1.0),
                    "virtual_raw_ratio": virtual_size / max(raw_size, 1.0),
                }
            )
        section_count = max(len(section_infos), 1)
        executable = [item for item in section_infos if item["exec"]]
        writable = [item for item in section_infos if item["write"]]
        readable = [item for item in section_infos if item["read"]]
        executable_writable = [item for item in section_infos if item["exec"] and item["write"]]
        entry_section = next((item for item in section_infos if item["contains_ep"]), None)
        first_section = section_infos[0] if section_infos else None
        last_section = section_infos[-1] if section_infos else None
        deltas = [float(item["raw_virtual_delta"]) for item in section_infos]
        ratios = [float(item["virtual_raw_ratio"]) for item in section_infos]
        features.extend(
            (
                math.log1p(len(executable)),
                math.log1p(len(writable)),
                math.log1p(len(readable)),
                math.log1p(len(executable_writable)),
                _safe_ratio(sum(item["entropy"] >= 0.8 for item in executable), len(executable)),
                _safe_ratio(sum(item["entropy"] >= 0.8 for item in writable), len(writable)),
                _safe_ratio(sum(item["zero_raw"] for item in executable), len(executable)),
                _safe_ratio(sum(item["zero_raw"] for item in writable), len(writable)),
                max(deltas) if deltas else 0.0,
                float(np.mean(deltas)) if deltas else 0.0,
                math.log1p(max(ratios) if ratios else 0.0),
                1.0 if entry_section and entry_section["exec"] else 0.0,
                1.0 if entry_section and entry_section["write"] else 0.0,
                float(entry_section["entropy"]) if entry_section else 0.0,
                float(entry_section["raw_virtual_delta"]) if entry_section else 0.0,
                float(first_section["entropy"]) if first_section else 0.0,
                1.0 if first_section and first_section["exec"] else 0.0,
                1.0 if first_section and first_section["write"] else 0.0,
                float(last_section["entropy"]) if last_section else 0.0,
                1.0 if last_section and last_section["exec"] else 0.0,
                1.0 if last_section and last_section["write"] else 0.0,
            )
        )
        features.extend(_safe_ratio(group_hits[name], section_count) for name in V2_SECTION_NAME_GROUPS)
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (182,) or not np.isfinite(vector).all():
            raise ValueError("content PE v2 feature shape or finiteness drift")
        return vector, False, None
    except Exception:
        return np.zeros(182, dtype=np.float32), True, "content_pe_v2_native_failure"


def extract_b0_projection(context: RawFeatureContext) -> B0Projection:
    """Build the frozen B0 value order plus six family-missing indicators."""

    context.require_open()
    family_outputs = {
        "fixed_v2": _fixed_v2_features(context),
        "stat": _stat_features(context),
        "content_pe_v1": _content_pe_v1_features(context),
        "content_pe_v2": _content_pe_v2_features(context),
        "content_string": _content_string_features(context),
        "content_cert": _content_cert_features(context),
    }
    columns = frozen_baseline_columns()
    selected_columns = [column for column in columns if column.included]
    values = np.asarray(
        [family_outputs[column.source_family][0][column.source_index] for column in selected_columns],
        dtype=np.float32,
    )
    missing_names = (
        "missing_fixed_v2",
        "missing_stat",
        "missing_content_pe_v1",
        "missing_content_pe_v2",
        "missing_content_string",
        "missing_content_cert",
    )
    missing_indicators = np.asarray(
        [1.0 if family_outputs[name][1] else 0.0 for name in ("fixed_v2", "stat", "content_pe_v1", "content_pe_v2", "content_string", "content_cert")],
        dtype=np.float32,
    )
    reasons = list(context.missing_reasons)
    for family, (_, missing, reason) in family_outputs.items():
        if missing and reason:
            reasons.append(f"{family}:{reason}")
    if values.shape != (571,) or missing_indicators.shape != (6,):
        raise ValueError("Loop167 frozen B0 schema dimension drift")
    if not np.isfinite(values).all() or not np.isfinite(missing_indicators).all():
        raise ValueError("Loop167 B0 projection produced a non-finite value")
    return B0Projection(
        values=values,
        feature_names=tuple(column.semantic_key for column in selected_columns),
        missing_indicators=missing_indicators,
        missing_indicator_names=missing_names,
        missing_reasons=tuple(dict.fromkeys(reasons)),
    )
