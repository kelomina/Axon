"""Stable content-derived PE metadata schema used by Loop28.

The features in this module are computed from file bytes and PE structures.
Paths are accepted only so the extractor can open the file; filename,
extension, directory, path text, hashes, sample ids, split names, and row order
are not encoded as feature values.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import pefile

    PEFILE_AVAILABLE = True
except ImportError:
    pefile = None
    PEFILE_AVAILABLE = False


SYSTEM_DLLS = {
    "kernel32.dll",
    "user32.dll",
    "advapi32.dll",
    "shell32.dll",
    "ole32.dll",
    "oleaut32.dll",
    "msvcrt.dll",
    "ntdll.dll",
    "ws2_32.dll",
    "wininet.dll",
    "urlmon.dll",
    "crypt32.dll",
    "secur32.dll",
    "netapi32.dll",
    "dnsapi.dll",
    "iphlpapi.dll",
    "gdi32.dll",
    "comdlg32.dll",
    "comctl32.dll",
    "shlwapi.dll",
    "version.dll",
    "setupapi.dll",
    "imm32.dll",
}

CONTENT_API_CATEGORIES = {
    "network": ["internet", "http", "socket", "connect", "recv", "send", "url", "download", "upload", "wsa"],
    "process": [
        "createprocess",
        "openprocess",
        "virtualalloc",
        "virtualprotect",
        "writeprocessmemory",
        "readprocessmemory",
        "createremotethread",
        "shellexecute",
        "winexec",
        "loadlibrary",
        "getprocaddress",
    ],
    "filesystem": ["createfile", "readfile", "writefile", "deletefile", "movefile", "copyfile", "findfirstfile"],
    "registry": ["regopenkey", "regsetvalue", "regcreatekey", "regdeletekey", "regqueryvalue"],
    "crypto": ["cryptencrypt", "cryptdecrypt", "cryptderivekey", "cryptgenkey", "cryptcreatehash", "crypthashdata"],
    "injection": ["createremotethread", "virtualallocex", "writeprocessmemory", "queueuserapc", "setwindowshookex"],
}

DATA_DIRECTORY_INDEXES = {
    "export": 0,
    "import": 1,
    "resource": 2,
    "exception": 3,
    "security": 4,
    "basereloc": 5,
    "debug": 6,
    "tls": 9,
    "iat": 12,
    "delay_import": 13,
    "clr": 14,
}

SECTION_COMBO_NAMES = ["rx", "rw", "rwx", "wx", "exec_only", "read_only", "write_only", "none"]
SECTION_ENTROPY_SAMPLE_BYTES = 4096

CONTENT_PE_V1_FEATURE_NAMES = [
    "content_file_log_size",
    "content_file_size_norm_100mb",
    "content_machine",
    "content_characteristics",
    "content_num_sections_norm",
    "content_timestamp_valid",
    "content_timestamp_year_norm",
    "content_optional_magic",
    "content_major_linker_norm",
    "content_minor_linker_norm",
    "content_size_of_code_ratio",
    "content_size_init_data_ratio",
    "content_size_uninit_data_ratio",
    "content_entry_point_ratio",
    "content_image_base_log",
    "content_section_alignment_log",
    "content_file_alignment_log",
    "content_size_of_image_ratio",
    "content_size_of_headers_ratio",
    "content_subsystem",
    "content_dll_characteristics",
    "content_is_dll",
    "content_is_executable_image",
    "content_is_system",
    "content_large_address_aware",
    "content_32bit_machine",
    "content_relocs_stripped",
    "content_debug_stripped",
]
for directory_name in DATA_DIRECTORY_INDEXES:
    CONTENT_PE_V1_FEATURE_NAMES.extend(
        [
            f"content_dir_{directory_name}_present",
            f"content_dir_{directory_name}_log_size",
            f"content_dir_{directory_name}_size_ratio",
        ]
    )
CONTENT_PE_V1_FEATURE_NAMES.extend(
    [
        "content_import_dll_count_log",
        "content_import_api_count_log",
        "content_unique_import_api_count_log",
        "content_import_ordinal_ratio",
        "content_system_dll_ratio",
        "content_avg_imports_per_dll",
        "content_max_imports_per_dll_norm",
    ]
)
for category_name in CONTENT_API_CATEGORIES:
    CONTENT_PE_V1_FEATURE_NAMES.append(f"content_api_{category_name}_ratio")
CONTENT_PE_V1_FEATURE_NAMES.extend(
    [
        "content_export_count_log",
        "content_export_name_ratio",
        "content_resource_entry_count_log",
        "content_resource_type_count_log",
        "content_tls_callback_count_log",
        "content_reloc_block_count_log",
        "content_reloc_entry_count_log",
        "content_overlay_present",
        "content_overlay_log_size",
        "content_overlay_ratio",
        "content_overlay_entropy",
    ]
)
for combo_name in SECTION_COMBO_NAMES:
    CONTENT_PE_V1_FEATURE_NAMES.append(f"content_section_combo_{combo_name}_ratio")
CONTENT_PE_V1_FEATURE_NAMES.extend(
    [
        "content_section_nonstandard_name_ratio",
        "content_section_high_entropy_ratio",
        "content_section_raw_virtual_mismatch_ratio",
        "content_section_zero_raw_ratio",
        "content_section_mean_entropy",
        "content_section_max_entropy",
        "content_section_name_packer_hit_ratio",
    ]
)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum() / 8.0)


def _entropy_from_bytes(data: bytes) -> float:
    if not data:
        return 0.0
    return _entropy_from_counts(np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _safe_lower_bytes(value: bytes | None) -> str:
    if not value:
        return ""
    try:
        return value.decode("utf-8", errors="ignore").lower().strip()
    except Exception:
        return ""


def _section_data_prefix(section, file_path: Path) -> bytes:
    """Read only the prefix needed for entropy, never the full section."""

    try:
        return section.get_data(length=SECTION_ENTROPY_SAMPLE_BYTES)
    except TypeError:
        pointer = int(getattr(section, "PointerToRawData", 0) or 0)
        raw_size = int(getattr(section, "SizeOfRawData", 0) or 0)
        if pointer < 0 or raw_size <= 0:
            return b""
        try:
            with Path(file_path).open("rb") as handle:
                handle.seek(pointer)
                return handle.read(4096)
        except OSError:
            return b""
    except Exception:
        return b""


def _data_directory(pe, index: int):
    directories = getattr(pe.OPTIONAL_HEADER, "DATA_DIRECTORY", [])
    if index < 0 or index >= len(directories):
        return None
    return directories[index]


def extract_content_pe_v1_features(file_path: Path) -> np.ndarray:
    features: list[float] = []
    if not PEFILE_AVAILABLE:
        return np.zeros(len(CONTENT_PE_V1_FEATURE_NAMES), dtype=np.float32)

    file_path = Path(file_path)
    try:
        file_size = file_path.stat().st_size
    except OSError:
        return np.zeros(len(CONTENT_PE_V1_FEATURE_NAMES), dtype=np.float32)

    try:
        pe = pefile.PE(str(file_path), fast_load=True)
    except Exception:
        return np.zeros(len(CONTENT_PE_V1_FEATURE_NAMES), dtype=np.float32)

    try:
        try:
            pe.parse_data_directories(
                directories=[
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_BASERELOC"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_TLS"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXCEPTION"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DEBUG"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
                ]
            )
        except Exception:
            pass

        file_header = pe.FILE_HEADER
        optional = pe.OPTIONAL_HEADER
        characteristics = int(getattr(file_header, "Characteristics", 0))
        timestamp = int(getattr(file_header, "TimeDateStamp", 0))
        timestamp_year = 1970 + timestamp / 31557600.0 if timestamp > 0 else 0.0
        timestamp_valid = 1.0 if 1970 <= timestamp_year <= 2099 else 0.0
        timestamp_year_norm = (min(max(timestamp_year, 1970.0), 2099.0) - 1970.0) / 129.0 if timestamp_valid else 0.0

        size_of_code = float(getattr(optional, "SizeOfCode", 0))
        size_init = float(getattr(optional, "SizeOfInitializedData", 0))
        size_uninit = float(getattr(optional, "SizeOfUninitializedData", 0))
        entry_point = float(getattr(optional, "AddressOfEntryPoint", 0))
        image_base = float(getattr(optional, "ImageBase", 0))
        section_alignment = float(getattr(optional, "SectionAlignment", 0))
        file_alignment = float(getattr(optional, "FileAlignment", 0))
        size_of_image = float(getattr(optional, "SizeOfImage", 0))
        size_of_headers = float(getattr(optional, "SizeOfHeaders", 0))

        features.extend(
            [
                math.log1p(float(file_size)),
                min(float(file_size), 100.0 * 1024 * 1024) / (100.0 * 1024 * 1024),
                float(getattr(file_header, "Machine", 0)) / 65535.0,
                float(characteristics) / 65535.0,
                min(float(getattr(file_header, "NumberOfSections", 0)), 64.0) / 64.0,
                timestamp_valid,
                timestamp_year_norm,
                float(getattr(optional, "Magic", 0)) / 65535.0,
                min(float(getattr(optional, "MajorLinkerVersion", 0)), 255.0) / 255.0,
                min(float(getattr(optional, "MinorLinkerVersion", 0)), 255.0) / 255.0,
                _safe_ratio(size_of_code, file_size),
                _safe_ratio(size_init, file_size),
                _safe_ratio(size_uninit, file_size),
                _safe_ratio(entry_point, max(size_of_image, file_size, 1.0)),
                math.log1p(max(image_base, 0.0)) / 64.0,
                math.log1p(max(section_alignment, 0.0)) / 16.0,
                math.log1p(max(file_alignment, 0.0)) / 16.0,
                _safe_ratio(size_of_image, file_size),
                _safe_ratio(size_of_headers, file_size),
                float(getattr(optional, "Subsystem", 0)) / 32.0,
                float(getattr(optional, "DllCharacteristics", 0)) / 65535.0,
                1.0 if characteristics & 0x2000 else 0.0,
                1.0 if characteristics & 0x0002 else 0.0,
                1.0 if characteristics & 0x1000 else 0.0,
                1.0 if characteristics & 0x0020 else 0.0,
                1.0 if characteristics & 0x0100 else 0.0,
                1.0 if characteristics & 0x0001 else 0.0,
                1.0 if characteristics & 0x0200 else 0.0,
            ]
        )

        for _name, index in DATA_DIRECTORY_INDEXES.items():
            directory = _data_directory(pe, index)
            size = float(getattr(directory, "Size", 0) if directory is not None else 0)
            rva = int(getattr(directory, "VirtualAddress", 0) if directory is not None else 0)
            present = 1.0 if size > 0 or rva > 0 else 0.0
            features.extend([present, math.log1p(max(size, 0.0)), _safe_ratio(size, file_size)])

        import_dlls: list[str] = []
        import_names: list[str] = []
        ordinal_imports = 0
        imports_per_dll: list[int] = []
        category_counts = {category: 0 for category in CONTENT_API_CATEGORIES}
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = _safe_lower_bytes(getattr(entry, "dll", None))
                if dll_name:
                    import_dlls.append(dll_name)
                entry_count = 0
                for imp in getattr(entry, "imports", []):
                    entry_count += 1
                    if getattr(imp, "name", None):
                        api_name = _safe_lower_bytes(imp.name)
                        if api_name:
                            import_names.append(api_name)
                            for category, keywords in CONTENT_API_CATEGORIES.items():
                                if any(keyword in api_name for keyword in keywords):
                                    category_counts[category] += 1
                    else:
                        ordinal_imports += 1
                imports_per_dll.append(entry_count)

        total_imports = len(import_names) + ordinal_imports
        unique_imports = len(set(import_names))
        system_dlls = sum(1 for dll in set(import_dlls) if dll in SYSTEM_DLLS)
        features.extend(
            [
                math.log1p(len(set(import_dlls))),
                math.log1p(total_imports),
                math.log1p(unique_imports),
                _safe_ratio(ordinal_imports, total_imports),
                _safe_ratio(system_dlls, len(set(import_dlls))),
                _safe_ratio(total_imports, len(imports_per_dll)),
                _safe_ratio(max(imports_per_dll) if imports_per_dll else 0, 512.0),
            ]
        )
        for category in CONTENT_API_CATEGORIES:
            features.append(_safe_ratio(category_counts[category], total_imports))

        export_count = 0
        export_name_count = 0
        if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            for symbol in getattr(pe.DIRECTORY_ENTRY_EXPORT, "symbols", []):
                export_count += 1
                if getattr(symbol, "name", None):
                    export_name_count += 1
        features.extend([math.log1p(export_count), _safe_ratio(export_name_count, export_count)])

        resource_entry_count = 0
        resource_type_ids = set()
        if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
            stack = list(getattr(pe.DIRECTORY_ENTRY_RESOURCE, "entries", []))
            while stack:
                entry = stack.pop()
                resource_entry_count += 1
                if hasattr(entry, "id"):
                    resource_type_ids.add(entry.id)
                if hasattr(entry, "directory"):
                    stack.extend(getattr(entry.directory, "entries", []))
        features.extend([math.log1p(resource_entry_count), math.log1p(len(resource_type_ids))])

        tls_callbacks = 0
        if hasattr(pe, "DIRECTORY_ENTRY_TLS"):
            struct_obj = getattr(pe.DIRECTORY_ENTRY_TLS, "struct", None)
            callback_array = int(getattr(struct_obj, "AddressOfCallBacks", 0) or 0)
            tls_callbacks = 1 if callback_array else 0
        features.append(math.log1p(tls_callbacks))

        reloc_blocks = 0
        reloc_entries = 0
        if hasattr(pe, "DIRECTORY_ENTRY_BASERELOC"):
            reloc_blocks = len(pe.DIRECTORY_ENTRY_BASERELOC)
            reloc_entries = sum(len(getattr(block, "entries", [])) for block in pe.DIRECTORY_ENTRY_BASERELOC)
        features.extend([math.log1p(reloc_blocks), math.log1p(reloc_entries)])

        overlay_offset = pe.get_overlay_data_start_offset()
        overlay_size = max(file_size - int(overlay_offset), 0) if overlay_offset is not None else 0
        overlay_entropy = 0.0
        if overlay_size > 0 and overlay_offset is not None:
            try:
                with file_path.open("rb") as handle:
                    handle.seek(int(overlay_offset))
                    overlay_entropy = _entropy_from_bytes(handle.read(65536)[: min(overlay_size, 65536)])
            except OSError:
                overlay_entropy = 0.0
        features.extend([1.0 if overlay_size > 0 else 0.0, math.log1p(float(overlay_size)), _safe_ratio(overlay_size, file_size), overlay_entropy])

        common_sections = {".text", ".data", ".rdata", ".rsrc", ".idata", ".edata", ".bss", ".reloc", ".tls"}
        packer_keywords = ("upx", "aspack", "themida", "vmprotect", "enigma", "packed", "nspack", "upack")
        combo_counts = {combo: 0 for combo in SECTION_COMBO_NAMES}
        section_entropies = []
        nonstandard_names = 0
        raw_virtual_mismatch = 0
        zero_raw = 0
        packer_hits = 0
        for section in pe.sections:
            chars = int(getattr(section, "Characteristics", 0))
            is_exec = bool(chars & 0x20000000)
            is_write = bool(chars & 0x80000000)
            is_read = bool(chars & 0x40000000)
            if is_exec and is_read and is_write:
                combo_counts["rwx"] += 1
            elif is_exec and is_write:
                combo_counts["wx"] += 1
            elif is_exec and is_read:
                combo_counts["rx"] += 1
            elif is_read and is_write:
                combo_counts["rw"] += 1
            elif is_exec:
                combo_counts["exec_only"] += 1
            elif is_read:
                combo_counts["read_only"] += 1
            elif is_write:
                combo_counts["write_only"] += 1
            else:
                combo_counts["none"] += 1

            section_name = _safe_lower_bytes(getattr(section, "Name", b"")).strip("\x00")
            if section_name and section_name not in common_sections:
                nonstandard_names += 1
            if any(keyword in section_name for keyword in packer_keywords):
                packer_hits += 1
            raw_size = float(getattr(section, "SizeOfRawData", 0))
            virt_size = float(getattr(section, "Misc_VirtualSize", 0))
            if raw_size <= 0:
                zero_raw += 1
            if max(raw_size, virt_size) > 0 and abs(raw_size - virt_size) / max(raw_size, virt_size) > 0.50:
                raw_virtual_mismatch += 1
            if raw_size > 0:
                section_entropies.append(_entropy_from_bytes(_section_data_prefix(section, file_path)))

        section_count = max(len(pe.sections), 1)
        for combo_name in SECTION_COMBO_NAMES:
            features.append(_safe_ratio(combo_counts[combo_name], section_count))
        high_entropy_sections = sum(1 for value in section_entropies if value >= 0.80)
        features.extend(
            [
                _safe_ratio(nonstandard_names, section_count),
                _safe_ratio(high_entropy_sections, len(section_entropies)),
                _safe_ratio(raw_virtual_mismatch, section_count),
                _safe_ratio(zero_raw, section_count),
                float(np.mean(section_entropies)) if section_entropies else 0.0,
                float(np.max(section_entropies)) if section_entropies else 0.0,
                _safe_ratio(packer_hits, section_count),
            ]
        )
        if len(features) != len(CONTENT_PE_V1_FEATURE_NAMES):
            raise ValueError(f"Content PE v1 feature length mismatch: {len(features)} != {len(CONTENT_PE_V1_FEATURE_NAMES)}")
        return np.nan_to_num(np.asarray(features, dtype=np.float32), copy=False)
    except Exception:
        return np.zeros(len(CONTENT_PE_V1_FEATURE_NAMES), dtype=np.float32)
    finally:
        pe.close()


# Backward-compatible names for Loop28/Stage-2 scripts.
CONTENT_PE_FEATURE_NAMES = CONTENT_PE_V1_FEATURE_NAMES
_content_pe_features_from_path = extract_content_pe_v1_features
