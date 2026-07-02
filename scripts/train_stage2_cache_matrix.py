#!/usr/bin/env python3
"""Run cache-backed stage-2 validation matrix for Axon predictions.

The script is intentionally cache-first: it consumes exported prediction CSVs
and feature-cache NPZ files, then runs many cheap train/val candidates before
optionally confirming the best val candidate on a fixed test CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import pefile

    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from dataset import _load_cached_feature_npz  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from kvd_features.content_pe_v1 import (  # noqa: E402
    CONTENT_API_CATEGORIES,
    CONTENT_PE_FEATURE_NAMES,
    DATA_DIRECTORY_INDEXES,
    SECTION_COMBO_NAMES,
    SYSTEM_DLLS,
    _content_pe_features_from_path,
)


CONTENT_PE_V2_IMPORT_DLLS = [
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
]

CONTENT_PE_V2_API_CATEGORIES = {
    "service": ["openscmanager", "createservice", "startservice", "controlservice", "deleteservice"],
    "driver": ["ntloaddriver", "zwloaddriver", "deviceiocontrol", "createsymboliclink", "ioctl"],
    "privilege": ["adjusttokenprivileges", "openprocesstoken", "lookupprivilege", "impersonate"],
    "antidebug": [
        "isdebuggerpresent",
        "checkremotedebugger",
        "ntqueryinformationprocess",
        "outputdebugstring",
    ],
    "memory": ["virtualalloc", "virtualallocex", "virtualprotect", "virtualprotectex", "heapalloc"],
    "thread": ["createthread", "createremotethread", "queueuserapc", "rtlcreateuserthread", "setthreadcontext"],
    "module": ["loadlibrary", "getprocaddress", "ldrloaddll", "freelibrary"],
    "process_enum": ["createtoolhelp32snapshot", "process32first", "process32next", "enumprocesses"],
    "persistence": ["regsetvalue", "regcreatekey", "createservice", "schtasks", "startup"],
    "network_http": ["internetopen", "internetconnect", "httpopenrequest", "httpsendrequest", "winhttp"],
    "network_socket": ["socket", "connect", "bind", "listen", "accept", "wsastartup"],
    "file_mutation": ["createfile", "writefile", "deletefile", "movefile", "copyfile", "setfileattributes"],
    "crypto_cert": ["crypt", "bcrypt", "cert", "winverifytrust"],
    "resource": ["findresource", "loadresource", "lockresource", "sizeofresource", "beginupdateresource"],
    "installer": ["msi", "setup", "install", "uninstall"],
    "com": ["cocreateinstance", "coinitialize", "clsidfromprogid", "regsvr"],
}

CONTENT_PE_V2_RESOURCE_TYPES = {
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

CONTENT_PE_V2_EXPORT_PATTERNS = {
    "com": ["dllgetclassobject", "dllcanunloadnow", "dllregisterserver", "dllunregisterserver"],
    "control_panel": ["cplapplet"],
    "service": ["servicemain", "handler", "startservice"],
    "plugin": ["plugin", "initialize", "init", "register"],
}

CONTENT_PE_V2_SECTION_NAME_GROUPS = {
    "code": [".text", "code"],
    "data": [".data", ".rdata", ".bss"],
    "resource": [".rsrc"],
    "import": [".idata"],
    "export": [".edata"],
    "reloc": [".reloc"],
    "tls": [".tls"],
    "packer": ["upx", "aspack", "themida", "vmprotect", "enigma", "packed", "nspack", "upack"],
}

CONTENT_PE_V2_FEATURE_NAMES = []
for dll_name in CONTENT_PE_V2_IMPORT_DLLS:
    stem = dll_name[:-4].replace(".", "_")
    CONTENT_PE_V2_FEATURE_NAMES.extend([f"v2_import_dll_{stem}_present", f"v2_import_dll_{stem}_api_ratio"])
for category_name in CONTENT_PE_V2_API_CATEGORIES:
    CONTENT_PE_V2_FEATURE_NAMES.extend(
        [
            f"v2_api_{category_name}_present",
            f"v2_api_{category_name}_count_log",
            f"v2_api_{category_name}_ratio",
        ]
    )
CONTENT_PE_V2_FEATURE_NAMES.extend(
    [
        "v2_delay_import_dll_count_log",
        "v2_delay_import_api_count_log",
        "v2_delay_import_ratio",
        "v2_export_ordinal_only_ratio",
        "v2_export_forwarder_ratio",
        "v2_export_mean_name_len_norm",
        "v2_export_max_name_len_norm",
        "v2_export_ordinal_span_log",
    ]
)
for pattern_name in CONTENT_PE_V2_EXPORT_PATTERNS:
    CONTENT_PE_V2_FEATURE_NAMES.append(f"v2_export_pattern_{pattern_name}_present")
CONTENT_PE_V2_FEATURE_NAMES.extend(
    [
        "v2_resource_data_entry_count_log",
        "v2_resource_named_entry_ratio",
        "v2_resource_language_count_log",
        "v2_resource_data_size_log",
        "v2_resource_max_data_size_ratio",
        "v2_resource_mean_entropy",
        "v2_resource_max_entropy",
    ]
)
for resource_name in CONTENT_PE_V2_RESOURCE_TYPES:
    CONTENT_PE_V2_FEATURE_NAMES.extend(
        [f"v2_resource_type_{resource_name}_present", f"v2_resource_type_{resource_name}_count_log"]
    )
CONTENT_PE_V2_FEATURE_NAMES.extend(
    [
        "v2_section_exec_count_log",
        "v2_section_write_count_log",
        "v2_section_read_count_log",
        "v2_section_exec_write_count_log",
        "v2_section_exec_high_entropy_ratio",
        "v2_section_write_high_entropy_ratio",
        "v2_section_zero_raw_exec_ratio",
        "v2_section_zero_raw_write_ratio",
        "v2_section_max_raw_virtual_delta",
        "v2_section_mean_raw_virtual_delta",
        "v2_section_max_virtual_raw_ratio_log",
        "v2_ep_in_exec_section",
        "v2_ep_in_write_section",
        "v2_ep_section_entropy",
        "v2_ep_section_raw_virtual_delta",
        "v2_first_section_entropy",
        "v2_first_section_exec",
        "v2_first_section_write",
        "v2_last_section_entropy",
        "v2_last_section_exec",
        "v2_last_section_write",
    ]
)
for group_name in CONTENT_PE_V2_SECTION_NAME_GROUPS:
    CONTENT_PE_V2_FEATURE_NAMES.append(f"v2_section_name_group_{group_name}_ratio")


def _content_pe_v2_group_map() -> dict[str, tuple[int, ...]]:
    groups: dict[str, list[int]] = {
        "import_dll": [],
        "api": [],
        "delay_import": [],
        "export": [],
        "resource": [],
        "section": [],
    }
    for index, name in enumerate(CONTENT_PE_V2_FEATURE_NAMES):
        if name.startswith("v2_import_dll_"):
            groups["import_dll"].append(index)
        elif name.startswith("v2_api_"):
            groups["api"].append(index)
        elif name.startswith("v2_delay_import_"):
            groups["delay_import"].append(index)
        elif name.startswith("v2_export_"):
            groups["export"].append(index)
        elif name.startswith("v2_resource_"):
            groups["resource"].append(index)
        elif (
            name.startswith("v2_section_")
            or name.startswith("v2_ep_")
            or name.startswith("v2_first_section_")
            or name.startswith("v2_last_section_")
        ):
            groups["section"].append(index)
        else:
            raise ValueError(f"Unassigned content PE v2 feature: {name}")

    result = {name: tuple(indexes) for name, indexes in groups.items()}
    result["imports"] = tuple(sorted(set(result["import_dll"]) | set(result["api"]) | set(result["delay_import"])))
    result["all"] = tuple(range(len(CONTENT_PE_V2_FEATURE_NAMES)))
    return result


CONTENT_PE_V2_GROUPS = _content_pe_v2_group_map()


def parse_content_pe_v2_groups(text: str | Sequence[str] | None) -> tuple[str, ...]:
    if text is None:
        return ("all",)
    if isinstance(text, str):
        raw_values = [item.strip().lower() for item in text.split(",") if item.strip()]
    else:
        raw_values = [str(item).strip().lower() for item in text if str(item).strip()]
    if not raw_values:
        return ("all",)
    aliases = {
        "dll": "import_dll",
        "import": "imports",
        "imports": "imports",
        "apis": "api",
        "sections": "section",
        "resources": "resource",
        "exports": "export",
        "delay": "delay_import",
    }
    values = tuple(aliases.get(item, item) for item in raw_values)
    unknown = sorted(set(values) - set(CONTENT_PE_V2_GROUPS))
    if unknown:
        available = ", ".join(sorted(CONTENT_PE_V2_GROUPS))
        raise ValueError(f"Unknown content PE v2 group(s): {unknown}. Available: {available}")
    if "all" in values:
        return ("all",)
    return tuple(dict.fromkeys(values))


def content_pe_v2_group_indices(groups: str | Sequence[str] | None) -> tuple[int, ...]:
    selected_groups = parse_content_pe_v2_groups(groups)
    if selected_groups == ("all",):
        return CONTENT_PE_V2_GROUPS["all"]
    selected: set[int] = set()
    for group in selected_groups:
        selected.update(CONTENT_PE_V2_GROUPS[group])
    return tuple(sorted(selected))


def content_pe_v2_selected_feature_names(groups: str | Sequence[str] | None) -> list[str]:
    return [CONTENT_PE_V2_FEATURE_NAMES[index] for index in content_pe_v2_group_indices(groups)]

CONTENT_STRING_PATTERNS = {
    "url": [b"http://", b"https://", b"www.", b"ftp://"],
    "network": [b"socket", b"connect", b"recv", b"send", b"wininet", b"ws2_32", b"internetopen", b"urldownload"],
    "script_exec": [b"powershell", b"cmd.exe", b"wscript", b"cscript", b"mshta", b"rundll32", b"regsvr32"],
    "persistence": [b"currentversion\\run", b"runonce", b"\\services\\", b"startup", b"schtasks", b"autostart"],
    "injection": [b"createremotethread", b"virtualalloc", b"virtualprotect", b"writeprocessmemory", b"queueuserapc"],
    "credential": [b"password", b"credential", b"token", b"cookie", b"browser", b"wallet"],
    "crypto": [b"cryptencrypt", b"cryptdecrypt", b"bcrypt", b"advapi32", b"base64", b"aes", b"rsa"],
    "evasion": [b"isdebuggerpresent", b"checkremotedebugger", b"ntqueryinformationprocess", b"sleep", b"sandbox"],
    "vm": [b"vmware", b"virtualbox", b"vbox", b"qemu", b"wine_get_unix_file_name"],
    "packer": [b"upx", b"themida", b"vmprotect", b"aspack", b"enigma", b"packed"],
    "file_ops": [b"createfile", b"writefile", b"deletefile", b"copyfile", b"movefile", b"findfirstfile"],
    "registry": [b"regopenkey", b"regsetvalue", b"regcreatekey", b"regdeletekey", b"regqueryvalue"],
    "benign_vendor": [b"microsoft", b"windows", b"google", b"adobe", b"intel", b"nvidia", b"mozilla", b"oracle"],
    "version_resource": [b"companyname", b"productname", b"filedescription", b"originalfilename", b"copyright"],
}

CONTENT_STRING_FEATURE_NAMES = [
    "string_sample_log_size",
    "string_ascii_printable_ratio",
    "string_null_ratio",
    "string_high_byte_ratio",
    "string_ascii_run_count_log",
    "string_ascii_run_density",
    "string_ascii_run_mean_len_norm",
    "string_ascii_run_max_len_norm",
    "string_utf16_ascii_run_count_log",
    "string_utf16_ascii_run_density",
    "string_url_regex_count_log",
    "string_ipv4_regex_count_log",
    "string_registry_path_count_log",
    "string_windows_path_count_log",
    "string_entropy",
]
for category_name in CONTENT_STRING_PATTERNS:
    CONTENT_STRING_FEATURE_NAMES.extend(
        [
            f"string_{category_name}_count_log",
            f"string_{category_name}_present",
        ]
    )

CERT_VENDOR_PATTERNS = {
    "microsoft": [b"microsoft", b"windows"],
    "digicert": [b"digicert"],
    "sectigo": [b"sectigo", b"comodo"],
    "globalsign": [b"globalsign"],
    "verisign": [b"verisign", b"symantec", b"thawte"],
    "entrust": [b"entrust"],
    "ssl_com": [b"ssl.com"],
    "google": [b"google"],
    "adobe": [b"adobe"],
    "intel": [b"intel"],
    "nvidia": [b"nvidia"],
    "oracle": [b"oracle"],
    "mozilla": [b"mozilla"],
    "kaspersky": [b"kaspersky"],
    "avast": [b"avast", b"avg technologies"],
}

CERT_OID_PATTERNS = {
    "pkcs7_signed_data": b"\x06\t*\x86H\x86\xf7\r\x01\x07\x02",
    "code_signing": b"\x06\x08+\x06\x01\x05\x05\x07\x03\x03",
    "timestamping": b"\x06\x08+\x06\x01\x05\x05\x07\x03\x08",
    "sha1": b"\x06\x05+\x0e\x03\x02\x1a",
    "sha256": b"\x06\t`\x86H\x01e\x03\x04\x02\x01",
    "sha384": b"\x06\t`\x86H\x01e\x03\x04\x02\x02",
    "rsa": b"\x06\t*\x86H\x86\xf7\r\x01\x01\x01",
    "ecdsa_sha256": b"\x06\x08*\x86H\xce=\x04\x03\x02",
}

CONTENT_CERT_FEATURE_NAMES = [
    "cert_present",
    "cert_log_size",
    "cert_size_ratio",
    "cert_win_length_ratio",
    "cert_revision",
    "cert_type",
    "cert_entropy",
    "cert_ascii_printable_ratio",
    "cert_null_ratio",
    "cert_high_byte_ratio",
    "cert_ascii_run_count_log",
    "cert_ascii_run_mean_len_norm",
    "cert_ascii_run_max_len_norm",
    "cert_utf16_run_count_log",
    "cert_sequence_marker_count_log",
    "cert_timestamp_text_present",
    "cert_counter_signature_text_present",
]


def stage2_feature_name_groups(
    feature_config: "FeatureConfig",
    knn_feature_names: Sequence[str] = (),
    checkpoint_config: Optional[AxonExperimentConfig] = None,
) -> dict[str, list[str]]:
    config = checkpoint_config or AxonExperimentConfig()
    names: dict[str, list[str]] = {
        "base_probability_features": [
            "base_prob_malicious",
            "base_prob_malicious_squared",
            "base_prob_distance_to_0_5",
            "base_prob_log_malicious",
            "base_prob_log_benign",
            "base_prob_logit",
        ],
        "stat_features": (
            [f"stat_{index}" for index in range(config.stat_feature_dim)]
            if feature_config.include_stat
            else []
        ),
        "pe_features": (
            [f"pe_{index}" for index in range(config.pe_feature_dim)]
            if feature_config.include_pe
            else []
        ),
        "lightweight_features": (
            [f"lightweight_{index}" for index in range(config.lightweight_feature_dim)]
            if feature_config.include_lightweight
            else []
        ),
        "byte_summary_features": [],
        "content_pe_feature_names": CONTENT_PE_FEATURE_NAMES if feature_config.include_content_pe else [],
        "content_pe_v2_feature_names": (
            content_pe_v2_selected_feature_names(getattr(feature_config, "content_pe_v2_groups", ("all",)))
            if getattr(feature_config, "include_content_pe_v2", False)
            else []
        ),
        "content_string_feature_names": CONTENT_STRING_FEATURE_NAMES if feature_config.include_content_string else [],
        "content_cert_feature_names": CONTENT_CERT_FEATURE_NAMES if feature_config.include_content_cert else [],
        "knn_feature_names": list(knn_feature_names),
    }
    if feature_config.include_byte_summary:
        names["byte_summary_features"] = (
            [f"byte_hist_{index}" for index in range(256)]
            + [f"byte_log_hist_{index}" for index in range(256)]
            + [f"byte_prefix_{index}" for index in range(max(0, int(feature_config.prefix_len)))]
            + [
                f"byte_chunk_{chunk}_{stat_name}"
                for chunk in range(max(1, int(feature_config.chunk_count)))
                for stat_name in ("mean", "std", "entropy", "nonzero_ratio", "max_byte_ratio")
            ]
            + [
                "byte_entropy",
                "byte_nonzero_ratio",
                "byte_mean",
                "byte_std",
                "byte_max_frequency_ratio",
            ]
        )
    return names


def assert_stage2_feature_names_safe(
    feature_config: "FeatureConfig",
    knn_feature_names: Sequence[str] = (),
    checkpoint_config: Optional[AxonExperimentConfig] = None,
) -> dict[str, list[str]]:
    groups = stage2_feature_name_groups(feature_config, knn_feature_names, checkpoint_config)
    for group_name, feature_names in groups.items():
        assert_no_identity_feature_names(feature_names, context=f"Stage-2 {group_name}")
    return groups


for name in CERT_OID_PATTERNS:
    CONTENT_CERT_FEATURE_NAMES.append(f"cert_oid_{name}_present")
for name in CERT_VENDOR_PATTERNS:
    CONTENT_CERT_FEATURE_NAMES.extend([f"cert_vendor_{name}_present", f"cert_vendor_{name}_count_log"])


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_thresholds(text: str) -> list[float]:
    if ":" in text:
        start_text, stop_text, step_text = text.split(":")
        start = float(start_text)
        stop = float(stop_text)
        step = float(step_text)
        count = int(math.floor((stop - start) / step)) + 1
        return [round(start + step * index, 10) for index in range(count)]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"All values must be positive: {values}")
    return sorted(set(values))


def read_prediction_rows(path: Path, max_rows: Optional[int] = None) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped))


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


def _section_entropy(section) -> float:
    raw_size = int(getattr(section, "SizeOfRawData", 0) or 0)
    if raw_size <= 0:
        return 0.0
    try:
        return _entropy_from_bytes(section.get_data()[:4096])
    except Exception:
        return 0.0


def _iter_import_directory_entries(pe) -> list[tuple[str, object]]:
    entries: list[tuple[str, object]] = []
    for attr_name in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
        for entry in getattr(pe, attr_name, []) or []:
            entries.append((attr_name, entry))
    return entries


def _content_pe_v2_features_from_path(file_path: Path) -> np.ndarray:
    features: list[float] = []
    if not PEFILE_AVAILABLE:
        return np.zeros(len(CONTENT_PE_V2_FEATURE_NAMES), dtype=np.float32)

    try:
        file_size = file_path.stat().st_size
    except OSError:
        return np.zeros(len(CONTENT_PE_V2_FEATURE_NAMES), dtype=np.float32)

    try:
        pe = pefile.PE(str(file_path), fast_load=True)
    except Exception:
        return np.zeros(len(CONTENT_PE_V2_FEATURE_NAMES), dtype=np.float32)

    try:
        try:
            pe.parse_data_directories(
                directories=[
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_BASERELOC"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_TLS"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
                ]
            )
        except Exception:
            pass

        import_dlls: list[str] = []
        import_names: list[str] = []
        ordinal_imports = 0
        dll_api_counts = {name: 0 for name in CONTENT_PE_V2_IMPORT_DLLS}
        delay_import_api_count = 0
        delay_import_dlls = set()
        category_counts = {category: 0 for category in CONTENT_PE_V2_API_CATEGORIES}

        for directory_name, entry in _iter_import_directory_entries(pe):
            dll_name = _safe_lower_bytes(getattr(entry, "dll", None)).split("\\")[-1]
            if dll_name:
                import_dlls.append(dll_name)
                if directory_name == "DIRECTORY_ENTRY_DELAY_IMPORT":
                    delay_import_dlls.add(dll_name)
            entry_import_count = 0
            for imp in getattr(entry, "imports", []) or []:
                entry_import_count += 1
                if directory_name == "DIRECTORY_ENTRY_DELAY_IMPORT":
                    delay_import_api_count += 1
                if getattr(imp, "name", None):
                    api_name = _safe_lower_bytes(imp.name)
                    if api_name:
                        import_names.append(api_name)
                        for category, keywords in CONTENT_PE_V2_API_CATEGORIES.items():
                            if any(keyword in api_name for keyword in keywords):
                                category_counts[category] += 1
                else:
                    ordinal_imports += 1
            if dll_name in dll_api_counts:
                dll_api_counts[dll_name] += entry_import_count

        total_imports = len(import_names) + ordinal_imports
        imported_dll_set = set(import_dlls)
        for dll_name in CONTENT_PE_V2_IMPORT_DLLS:
            features.extend(
                [
                    1.0 if dll_name in imported_dll_set else 0.0,
                    _safe_ratio(dll_api_counts[dll_name], total_imports),
                ]
            )
        for category in CONTENT_PE_V2_API_CATEGORIES:
            count = category_counts[category]
            features.extend([1.0 if count > 0 else 0.0, math.log1p(count), _safe_ratio(count, total_imports)])

        features.extend(
            [
                math.log1p(len(delay_import_dlls)),
                math.log1p(delay_import_api_count),
                _safe_ratio(delay_import_api_count, total_imports),
            ]
        )

        export_count = 0
        export_name_count = 0
        export_forwarder_count = 0
        export_name_lengths: list[int] = []
        export_ordinals: list[int] = []
        export_pattern_hits = {name: 0 for name in CONTENT_PE_V2_EXPORT_PATTERNS}
        if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            for symbol in getattr(pe.DIRECTORY_ENTRY_EXPORT, "symbols", []) or []:
                export_count += 1
                ordinal = getattr(symbol, "ordinal", None)
                if ordinal is not None:
                    export_ordinals.append(int(ordinal))
                if getattr(symbol, "forwarder", None):
                    export_forwarder_count += 1
                export_name = _safe_lower_bytes(getattr(symbol, "name", None))
                if export_name:
                    export_name_count += 1
                    export_name_lengths.append(len(export_name))
                    for pattern_name, keywords in CONTENT_PE_V2_EXPORT_PATTERNS.items():
                        if any(keyword in export_name for keyword in keywords):
                            export_pattern_hits[pattern_name] += 1
        ordinal_span = (max(export_ordinals) - min(export_ordinals) + 1) if export_ordinals else 0
        features.extend(
            [
                _safe_ratio(export_count - export_name_count, export_count),
                _safe_ratio(export_forwarder_count, export_count),
                _safe_ratio(float(np.mean(export_name_lengths)) if export_name_lengths else 0.0, 128.0),
                _safe_ratio(max(export_name_lengths) if export_name_lengths else 0, 256.0),
                math.log1p(ordinal_span),
            ]
        )
        for pattern_name in CONTENT_PE_V2_EXPORT_PATTERNS:
            features.append(1.0 if export_pattern_hits[pattern_name] > 0 else 0.0)

        resource_entries = 0
        resource_named_entries = 0
        resource_type_counts = {name: 0 for name in CONTENT_PE_V2_RESOURCE_TYPES}
        resource_languages = set()
        resource_sizes: list[int] = []
        resource_entropies: list[float] = []
        if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
            stack = [(entry, 0, getattr(entry, "id", None)) for entry in getattr(pe.DIRECTORY_ENTRY_RESOURCE, "entries", [])]
            while stack:
                entry, depth, root_type = stack.pop()
                resource_entries += 1
                if getattr(entry, "name", None) is not None:
                    resource_named_entries += 1
                if depth == 0:
                    root_type = getattr(entry, "id", root_type)
                if depth == 2 and getattr(entry, "id", None) is not None:
                    resource_languages.add(int(entry.id))
                for resource_name, resource_id in CONTENT_PE_V2_RESOURCE_TYPES.items():
                    if root_type == resource_id:
                        resource_type_counts[resource_name] += 1
                if hasattr(entry, "data"):
                    data_struct = getattr(entry.data, "struct", None)
                    size = int(getattr(data_struct, "Size", 0) or 0)
                    rva = int(getattr(data_struct, "OffsetToData", 0) or 0)
                    if size > 0:
                        resource_sizes.append(size)
                        if len(resource_entropies) < 64:
                            try:
                                offset = pe.get_offset_from_rva(rva)
                                with file_path.open("rb") as handle:
                                    handle.seek(offset)
                                    resource_entropies.append(_entropy_from_bytes(handle.read(min(size, 4096))))
                            except Exception:
                                pass
                if hasattr(entry, "directory"):
                    for child in getattr(entry.directory, "entries", []) or []:
                        stack.append((child, depth + 1, root_type))

        resource_total_size = sum(resource_sizes)
        features.extend(
            [
                math.log1p(len(resource_sizes)),
                _safe_ratio(resource_named_entries, resource_entries),
                math.log1p(len(resource_languages)),
                math.log1p(resource_total_size),
                _safe_ratio(max(resource_sizes) if resource_sizes else 0, file_size),
                float(np.mean(resource_entropies)) if resource_entropies else 0.0,
                float(np.max(resource_entropies)) if resource_entropies else 0.0,
            ]
        )
        for resource_name in CONTENT_PE_V2_RESOURCE_TYPES:
            count = resource_type_counts[resource_name]
            features.extend([1.0 if count > 0 else 0.0, math.log1p(count)])

        optional = pe.OPTIONAL_HEADER
        entry_point_rva = int(getattr(optional, "AddressOfEntryPoint", 0) or 0)
        section_infos = []
        group_hits = {name: 0 for name in CONTENT_PE_V2_SECTION_NAME_GROUPS}
        for section in pe.sections:
            chars = int(getattr(section, "Characteristics", 0) or 0)
            is_exec = bool(chars & 0x20000000)
            is_read = bool(chars & 0x40000000)
            is_write = bool(chars & 0x80000000)
            raw_size = float(getattr(section, "SizeOfRawData", 0) or 0)
            virt_size = float(getattr(section, "Misc_VirtualSize", 0) or 0)
            entropy = _section_entropy(section)
            virtual_address = int(getattr(section, "VirtualAddress", 0) or 0)
            virtual_span = max(int(virt_size), int(raw_size), 1)
            contains_ep = virtual_address <= entry_point_rva < (virtual_address + virtual_span)
            raw_virtual_delta = abs(raw_size - virt_size) / max(raw_size, virt_size, 1.0)
            virtual_raw_ratio = virt_size / max(raw_size, 1.0)
            section_name = _safe_lower_bytes(getattr(section, "Name", b"")).strip("\x00")
            for group_name, keywords in CONTENT_PE_V2_SECTION_NAME_GROUPS.items():
                if any(keyword in section_name for keyword in keywords):
                    group_hits[group_name] += 1
            section_infos.append(
                {
                    "exec": is_exec,
                    "read": is_read,
                    "write": is_write,
                    "zero_raw": raw_size <= 0,
                    "entropy": entropy,
                    "contains_ep": contains_ep,
                    "raw_virtual_delta": raw_virtual_delta,
                    "virtual_raw_ratio": virtual_raw_ratio,
                }
            )

        section_count = max(len(section_infos), 1)
        exec_sections = [info for info in section_infos if info["exec"]]
        write_sections = [info for info in section_infos if info["write"]]
        read_sections = [info for info in section_infos if info["read"]]
        exec_write_sections = [info for info in section_infos if info["exec"] and info["write"]]
        ep_section = next((info for info in section_infos if info["contains_ep"]), None)
        first_section = section_infos[0] if section_infos else None
        last_section = section_infos[-1] if section_infos else None
        deltas = [float(info["raw_virtual_delta"]) for info in section_infos]
        virtual_raw_ratios = [float(info["virtual_raw_ratio"]) for info in section_infos]
        features.extend(
            [
                math.log1p(len(exec_sections)),
                math.log1p(len(write_sections)),
                math.log1p(len(read_sections)),
                math.log1p(len(exec_write_sections)),
                _safe_ratio(sum(1 for info in exec_sections if info["entropy"] >= 0.80), len(exec_sections)),
                _safe_ratio(sum(1 for info in write_sections if info["entropy"] >= 0.80), len(write_sections)),
                _safe_ratio(sum(1 for info in exec_sections if info["zero_raw"]), len(exec_sections)),
                _safe_ratio(sum(1 for info in write_sections if info["zero_raw"]), len(write_sections)),
                max(deltas) if deltas else 0.0,
                float(np.mean(deltas)) if deltas else 0.0,
                math.log1p(max(virtual_raw_ratios) if virtual_raw_ratios else 0.0),
                1.0 if ep_section and ep_section["exec"] else 0.0,
                1.0 if ep_section and ep_section["write"] else 0.0,
                float(ep_section["entropy"]) if ep_section else 0.0,
                float(ep_section["raw_virtual_delta"]) if ep_section else 0.0,
                float(first_section["entropy"]) if first_section else 0.0,
                1.0 if first_section and first_section["exec"] else 0.0,
                1.0 if first_section and first_section["write"] else 0.0,
                float(last_section["entropy"]) if last_section else 0.0,
                1.0 if last_section and last_section["exec"] else 0.0,
                1.0 if last_section and last_section["write"] else 0.0,
            ]
        )
        for group_name in CONTENT_PE_V2_SECTION_NAME_GROUPS:
            features.append(_safe_ratio(group_hits[group_name], section_count))

        if len(features) != len(CONTENT_PE_V2_FEATURE_NAMES):
            raise ValueError(
                f"Content PE v2 feature length mismatch: {len(features)} != {len(CONTENT_PE_V2_FEATURE_NAMES)}"
            )
        return np.nan_to_num(np.asarray(features, dtype=np.float32), copy=False)
    except Exception:
        return np.zeros(len(CONTENT_PE_V2_FEATURE_NAMES), dtype=np.float32)
    finally:
        pe.close()


def _content_cache_path(row: dict, cache_dir: Optional[str]) -> Optional[Path]:
    if not cache_dir:
        return None
    key = (row.get("source_sha256") or "").strip().lower()
    if not key:
        source_path = row.get("source_path", "")
        key = hashlib.sha256(str(resolve_path(Path(source_path))).encode("utf-8", errors="ignore")).hexdigest()
    return resolve_path(Path(cache_dir)) / f"{key}.npz"


def content_pe_features_for_row(row: dict, cache_dir: Optional[str]) -> np.ndarray:
    cache_path = _content_cache_path(row, cache_dir)
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32, copy=False)
        if features.shape == (len(CONTENT_PE_FEATURE_NAMES),):
            return features

    source_path = resolve_path(Path(row["source_path"]))
    features = _content_pe_features_from_path(source_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_feature_npz_atomic(cache_path, features)
    return features


def _content_pe_v2_cache_path(row: dict, cache_dir: Optional[str]) -> Optional[Path]:
    if not cache_dir:
        return None
    key = (row.get("source_sha256") or "").strip().lower()
    if not key:
        source_path = row.get("source_path", "")
        key = hashlib.sha256(str(resolve_path(Path(source_path))).encode("utf-8", errors="ignore")).hexdigest()
    return resolve_path(Path(cache_dir)) / f"{key}.npz"


def content_pe_v2_features_for_row(row: dict, cache_dir: Optional[str]) -> np.ndarray:
    cache_path = _content_pe_v2_cache_path(row, cache_dir)
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32, copy=False)
        if features.shape == (len(CONTENT_PE_V2_FEATURE_NAMES),):
            return features

    source_path = resolve_path(Path(row["source_path"]))
    features = _content_pe_v2_features_from_path(source_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_feature_npz_atomic(cache_path, features)
    return features


def save_feature_npz_atomic(cache_path: Path, features: np.ndarray) -> None:
    temp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.{time.time_ns()}.tmp.npz")
    np.savez(temp_path, features=features)
    temp_path.replace(cache_path)


def _read_binary_sample(file_path: Path, *, head_bytes: int = 2 * 1024 * 1024, tail_bytes: int = 512 * 1024) -> bytes:
    try:
        file_size = file_path.stat().st_size
        with file_path.open("rb") as handle:
            head = handle.read(head_bytes)
            if file_size > head_bytes + tail_bytes:
                handle.seek(max(file_size - tail_bytes, 0))
                tail = handle.read(tail_bytes)
                return head + tail
            return head
    except OSError:
        return b""


def _count_regex(data: bytes, pattern: bytes) -> int:
    import re

    return len(re.findall(pattern, data))


def _ascii_run_lengths(data: bytes, *, min_len: int = 4) -> list[int]:
    lengths = []
    current = 0
    for value in data:
        if 32 <= value <= 126:
            current += 1
        else:
            if current >= min_len:
                lengths.append(current)
            current = 0
    if current >= min_len:
        lengths.append(current)
    return lengths


def _utf16_ascii_run_lengths(data: bytes, *, min_len: int = 4) -> list[int]:
    lengths = []
    current = 0
    index = 0
    limit = len(data) - 1
    while index < limit:
        if 32 <= data[index] <= 126 and data[index + 1] == 0:
            current += 1
            index += 2
        else:
            if current >= min_len:
                lengths.append(current)
            current = 0
            index += 1
    if current >= min_len:
        lengths.append(current)
    return lengths


def _content_string_features_from_path(file_path: Path) -> np.ndarray:
    data = _read_binary_sample(file_path)
    length = len(data)
    if length == 0:
        return np.zeros(len(CONTENT_STRING_FEATURE_NAMES), dtype=np.float32)

    lowered = data.lower()
    byte_values = np.frombuffer(data, dtype=np.uint8)
    ascii_printable = int(np.count_nonzero((byte_values >= 32) & (byte_values <= 126)))
    null_count = int(np.count_nonzero(byte_values == 0))
    high_byte_count = int(np.count_nonzero(byte_values >= 128))
    ascii_runs = _ascii_run_lengths(data)
    utf16_runs = _utf16_ascii_run_lengths(data)
    ascii_count = len(ascii_runs)
    utf16_count = len(utf16_runs)
    mean_ascii = float(np.mean(ascii_runs)) if ascii_runs else 0.0
    max_ascii = float(np.max(ascii_runs)) if ascii_runs else 0.0

    features = [
        math.log1p(length),
        _safe_ratio(ascii_printable, length),
        _safe_ratio(null_count, length),
        _safe_ratio(high_byte_count, length),
        math.log1p(ascii_count),
        _safe_ratio(ascii_count, length / 1024.0),
        min(mean_ascii, 512.0) / 512.0,
        min(max_ascii, 4096.0) / 4096.0,
        math.log1p(utf16_count),
        _safe_ratio(utf16_count, length / 1024.0),
        math.log1p(_count_regex(lowered, rb"https?://[^\s\x00\"']+")),
        math.log1p(_count_regex(lowered, rb"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
        math.log1p(lowered.count(b"\\software\\") + lowered.count(b"\\registry\\") + lowered.count(b"hkey_")),
        math.log1p(lowered.count(b"c:\\") + lowered.count(b"\\windows\\") + lowered.count(b"\\system32\\")),
        _entropy_from_counts(np.bincount(byte_values, minlength=256)),
    ]

    for patterns in CONTENT_STRING_PATTERNS.values():
        count = sum(lowered.count(pattern.lower()) for pattern in patterns)
        features.extend([math.log1p(count), 1.0 if count > 0 else 0.0])

    if len(features) != len(CONTENT_STRING_FEATURE_NAMES):
        raise ValueError(
            f"Content string feature length mismatch: {len(features)} != {len(CONTENT_STRING_FEATURE_NAMES)}"
        )
    return np.nan_to_num(np.asarray(features, dtype=np.float32), copy=False)


def _string_cache_path(row: dict, cache_dir: Optional[str]) -> Optional[Path]:
    if not cache_dir:
        return None
    key = (row.get("source_sha256") or "").strip().lower()
    if not key:
        source_path = row.get("source_path", "")
        key = hashlib.sha256(str(resolve_path(Path(source_path))).encode("utf-8", errors="ignore")).hexdigest()
    return resolve_path(Path(cache_dir)) / f"{key}.npz"


def content_string_features_for_row(row: dict, cache_dir: Optional[str]) -> np.ndarray:
    cache_path = _string_cache_path(row, cache_dir)
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32, copy=False)
        if features.shape == (len(CONTENT_STRING_FEATURE_NAMES),):
            return features

    source_path = resolve_path(Path(row["source_path"]))
    features = _content_string_features_from_path(source_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_feature_npz_atomic(cache_path, features)
    return features


def _read_certificate_blob(file_path: Path) -> tuple[bytes, int, int, int]:
    if not PEFILE_AVAILABLE:
        return b"", 0, 0, 0
    try:
        pe = pefile.PE(str(file_path), fast_load=True)
    except Exception:
        return b"", 0, 0, 0
    try:
        directory = _data_directory(pe, DATA_DIRECTORY_INDEXES["security"])
        offset = int(getattr(directory, "VirtualAddress", 0) if directory is not None else 0)
        size = int(getattr(directory, "Size", 0) if directory is not None else 0)
        if offset <= 0 or size <= 0:
            return b"", 0, 0, 0
        with file_path.open("rb") as handle:
            handle.seek(offset)
            blob = handle.read(min(size, 4 * 1024 * 1024))
        revision = int.from_bytes(blob[4:6], "little", signed=False) if len(blob) >= 6 else 0
        cert_type = int.from_bytes(blob[6:8], "little", signed=False) if len(blob) >= 8 else 0
        return blob, size, revision, cert_type
    except Exception:
        return b"", 0, 0, 0
    finally:
        pe.close()


def _content_cert_features_from_path(file_path: Path) -> np.ndarray:
    try:
        file_size = file_path.stat().st_size
    except OSError:
        file_size = 0
    blob, declared_size, revision, cert_type = _read_certificate_blob(file_path)
    length = len(blob)
    if length == 0:
        return np.zeros(len(CONTENT_CERT_FEATURE_NAMES), dtype=np.float32)

    byte_values = np.frombuffer(blob, dtype=np.uint8)
    ascii_printable = int(np.count_nonzero((byte_values >= 32) & (byte_values <= 126)))
    null_count = int(np.count_nonzero(byte_values == 0))
    high_byte_count = int(np.count_nonzero(byte_values >= 128))
    ascii_runs = _ascii_run_lengths(blob)
    utf16_runs = _utf16_ascii_run_lengths(blob)
    mean_ascii = float(np.mean(ascii_runs)) if ascii_runs else 0.0
    max_ascii = float(np.max(ascii_runs)) if ascii_runs else 0.0
    lowered = blob.lower()
    try:
        utf16_text = blob.decode("utf-16le", errors="ignore").lower().encode("utf-8", errors="ignore")
    except Exception:
        utf16_text = b""
    searchable = lowered + b"\n" + utf16_text

    features = [
        1.0,
        math.log1p(declared_size or length),
        _safe_ratio(declared_size or length, file_size),
        _safe_ratio(int.from_bytes(blob[:4], "little", signed=False) if len(blob) >= 4 else 0, declared_size or length),
        float(revision) / 65535.0,
        float(cert_type) / 65535.0,
        _entropy_from_counts(np.bincount(byte_values, minlength=256)),
        _safe_ratio(ascii_printable, length),
        _safe_ratio(null_count, length),
        _safe_ratio(high_byte_count, length),
        math.log1p(len(ascii_runs)),
        min(mean_ascii, 512.0) / 512.0,
        min(max_ascii, 4096.0) / 4096.0,
        math.log1p(len(utf16_runs)),
        math.log1p(blob.count(b"\x30\x82")),
        1.0 if b"timestamp" in searchable or b"time stamp" in searchable else 0.0,
        1.0 if b"countersign" in searchable or b"counter sign" in searchable else 0.0,
    ]

    for pattern in CERT_OID_PATTERNS.values():
        features.append(1.0 if pattern in blob else 0.0)

    for patterns in CERT_VENDOR_PATTERNS.values():
        count = sum(searchable.count(pattern.lower()) for pattern in patterns)
        features.extend([1.0 if count > 0 else 0.0, math.log1p(count)])

    if len(features) != len(CONTENT_CERT_FEATURE_NAMES):
        raise ValueError(f"Content cert feature length mismatch: {len(features)} != {len(CONTENT_CERT_FEATURE_NAMES)}")
    return np.nan_to_num(np.asarray(features, dtype=np.float32), copy=False)


def _cert_cache_path(row: dict, cache_dir: Optional[str]) -> Optional[Path]:
    if not cache_dir:
        return None
    key = (row.get("source_sha256") or "").strip().lower()
    if not key:
        source_path = row.get("source_path", "")
        key = hashlib.sha256(str(resolve_path(Path(source_path))).encode("utf-8", errors="ignore")).hexdigest()
    return resolve_path(Path(cache_dir)) / f"{key}.npz"


def content_cert_features_for_row(row: dict, cache_dir: Optional[str]) -> np.ndarray:
    cache_path = _cert_cache_path(row, cache_dir)
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32, copy=False)
        if features.shape == (len(CONTENT_CERT_FEATURE_NAMES),):
            return features

    source_path = resolve_path(Path(row["source_path"]))
    features = _content_cert_features_from_path(source_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_feature_npz_atomic(cache_path, features)
    return features


def _byte_summary_features(byte_seq: np.ndarray, prefix_len: int, chunk_count: int) -> np.ndarray:
    byte_values = byte_seq.astype(np.uint8, copy=False)
    counts = np.bincount(byte_values, minlength=256).astype(np.float32)
    hist = counts / max(float(byte_values.shape[0]), 1.0)
    log_hist = np.log1p(counts) / np.log1p(max(float(byte_values.shape[0]), 1.0))

    prefix = byte_values[:prefix_len].astype(np.float32) / 255.0
    if prefix.shape[0] < prefix_len:
        prefix = np.pad(prefix, (0, prefix_len - prefix.shape[0]))

    chunks = np.array_split(byte_values, max(1, chunk_count))
    chunk_features = []
    for chunk in chunks:
        if chunk.size == 0:
            chunk_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        chunk_counts = np.bincount(chunk, minlength=256).astype(np.float32)
        chunk_features.extend(
            [
                float(np.mean(chunk) / 255.0),
                float(np.std(chunk) / 255.0),
                _entropy_from_counts(chunk_counts),
                float(np.count_nonzero(chunk) / max(chunk.size, 1)),
                float(np.max(chunk_counts) / max(chunk.size, 1)),
            ]
        )

    scalar = np.asarray(
        [
            _entropy_from_counts(counts),
            float(np.count_nonzero(byte_values) / max(byte_values.shape[0], 1)),
            float(np.mean(byte_values) / 255.0),
            float(np.std(byte_values) / 255.0),
            float(np.max(counts) / max(byte_values.shape[0], 1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([hist, log_hist, prefix, np.asarray(chunk_features, dtype=np.float32), scalar])


@dataclass(frozen=True)
class FeatureConfig:
    prefix_len: int
    chunk_count: int
    include_pe: bool
    include_stat: bool
    include_lightweight: bool
    include_byte_summary: bool
    include_content_pe: bool = False
    content_cache_dir: Optional[str] = None
    include_content_pe_v2: bool = False
    content_pe_v2_cache_dir: Optional[str] = None
    content_pe_v2_groups: tuple[str, ...] = ("all",)
    include_content_string: bool = False
    content_string_cache_dir: Optional[str] = None
    include_content_cert: bool = False
    content_cert_cache_dir: Optional[str] = None


def build_matrix(
    rows: Sequence[dict],
    checkpoint_config: AxonExperimentConfig,
    feature_config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    features = []
    labels = []
    base_probs = []
    kept_rows = []
    skipped_missing_cache = 0
    content_pe_v2_indices = None
    if getattr(feature_config, "include_content_pe_v2", False):
        content_pe_v2_indices = np.asarray(
            content_pe_v2_group_indices(getattr(feature_config, "content_pe_v2_groups", ("all",))),
            dtype=np.int64,
        )
    for row in rows:
        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            skipped_missing_cache += 1
            continue
        label = int(row["label"])
        byte_seq, pe_feat, stat_feat, lightweight_feat, cached_label = _load_cached_feature_npz(
            cache_path,
            checkpoint_config.max_byte_length,
            checkpoint_config.pe_feature_dim,
            checkpoint_config.stat_feature_dim,
            checkpoint_config.lightweight_feature_dim,
            expected_label=label,
        )
        if cached_label != label:
            raise ValueError(f"Cache label mismatch: {cache_path}")

        prob = float(row["prob_malicious"])
        prob_arr = np.asarray(
            [
                prob,
                prob * prob,
                abs(prob - 0.5),
                math.log(max(prob, 1.0e-6)),
                math.log(max(1.0 - prob, 1.0e-6)),
                float(_safe_logit(np.asarray([prob]))[0]),
            ],
            dtype=np.float32,
        )
        parts = [prob_arr]
        if feature_config.include_stat:
            parts.append(np.nan_to_num(stat_feat.astype(np.float32, copy=False), copy=False))
        if feature_config.include_pe:
            parts.append(np.nan_to_num(pe_feat.astype(np.float32, copy=False), copy=False))
        if feature_config.include_lightweight:
            parts.append(np.nan_to_num(lightweight_feat.astype(np.float32, copy=False), copy=False))
        if feature_config.include_byte_summary:
            parts.append(_byte_summary_features(byte_seq, feature_config.prefix_len, feature_config.chunk_count))
        if getattr(feature_config, "include_content_pe", False):
            parts.append(content_pe_features_for_row(row, getattr(feature_config, "content_cache_dir", None)))
        if getattr(feature_config, "include_content_pe_v2", False):
            v2_features = content_pe_v2_features_for_row(row, getattr(feature_config, "content_pe_v2_cache_dir", None))
            parts.append(v2_features[content_pe_v2_indices])
        if getattr(feature_config, "include_content_string", False):
            parts.append(
                content_string_features_for_row(row, getattr(feature_config, "content_string_cache_dir", None))
            )
        if getattr(feature_config, "include_content_cert", False):
            parts.append(content_cert_features_for_row(row, getattr(feature_config, "content_cert_cache_dir", None)))
        features.append(np.concatenate(parts).astype(np.float32, copy=False))
        labels.append(label)
        base_probs.append(prob)
        kept_rows.append(row)
    if not features:
        raise ValueError("No usable rows were loaded")
    return (
        np.vstack(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(base_probs, dtype=np.float32),
        kept_rows,
        {"total": len(rows), "kept": len(labels), "skipped_missing_cache": skipped_missing_cache},
    )


def _fit_standard_l2_reference(matrix: np.ndarray) -> dict:
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    centered = (matrix.astype(np.float32, copy=False) - mean) / std
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1.0e-8)
    return {"mean": mean, "std": std, "normalized": normalized.astype(np.float32, copy=False)}


def _normalize_with_reference(matrix: np.ndarray, reference: dict) -> np.ndarray:
    centered = (matrix.astype(np.float32, copy=False) - reference["mean"]) / reference["std"]
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return (centered / np.maximum(norms, 1.0e-8)).astype(np.float32, copy=False)


def _knn_feature_names(top_ks: Sequence[int]) -> list[str]:
    names = []
    for top_k in top_ks:
        names.extend(
            [
                f"knn{top_k}_mal_ratio",
                f"knn{top_k}_benign_ratio",
                f"knn{top_k}_label_margin",
                f"knn{top_k}_weighted_mal_ratio",
                f"knn{top_k}_mean_similarity",
                f"knn{top_k}_min_similarity",
            ]
        )
    names.extend(["knn_top1_label", "knn_top1_similarity", "knn_top1_top2_gap"])
    return names


def _knn_support_features_from_norm(
    query_norm: np.ndarray,
    memory_norm: np.ndarray,
    memory_labels: np.ndarray,
    top_ks: Sequence[int],
    *,
    batch_size: int,
) -> np.ndarray:
    if memory_norm.shape[0] == 0:
        raise ValueError("kNN memory is empty")
    top_ks = [min(int(top_k), int(memory_norm.shape[0])) for top_k in top_ks]
    max_k = max(top_ks)
    feature_dim = len(_knn_feature_names(top_ks))
    features = np.empty((query_norm.shape[0], feature_dim), dtype=np.float32)
    memory_labels = memory_labels.astype(np.float32, copy=False)
    batch_size = max(1, int(batch_size))

    for start in range(0, query_norm.shape[0], batch_size):
        stop = min(start + batch_size, query_norm.shape[0])
        similarities = query_norm[start:stop] @ memory_norm.T
        top_unsorted = np.argpartition(-similarities, max_k - 1, axis=1)[:, :max_k]
        top_sim_unsorted = np.take_along_axis(similarities, top_unsorted, axis=1)
        top_order = np.argsort(-top_sim_unsorted, axis=1)
        top_idx = np.take_along_axis(top_unsorted, top_order, axis=1)
        top_sim = np.take_along_axis(similarities, top_idx, axis=1).astype(np.float32, copy=False)
        top_labels = memory_labels[top_idx]

        batch_features = np.empty((stop - start, feature_dim), dtype=np.float32)
        column = 0
        for top_k in top_ks:
            labels_k = top_labels[:, :top_k]
            sim_k = top_sim[:, :top_k]
            mal_ratio = labels_k.mean(axis=1)
            weights = np.clip((sim_k + 1.0) * 0.5, 1.0e-6, None)
            weighted_mal_ratio = (labels_k * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0e-6)
            batch_features[:, column] = mal_ratio
            batch_features[:, column + 1] = 1.0 - mal_ratio
            batch_features[:, column + 2] = 2.0 * mal_ratio - 1.0
            batch_features[:, column + 3] = weighted_mal_ratio
            batch_features[:, column + 4] = sim_k.mean(axis=1)
            batch_features[:, column + 5] = sim_k[:, -1]
            column += 6

        top2_index = 1 if top_sim.shape[1] > 1 else 0
        batch_features[:, column] = top_labels[:, 0]
        batch_features[:, column + 1] = top_sim[:, 0]
        batch_features[:, column + 2] = top_sim[:, 0] - top_sim[:, top2_index]
        features[start:stop] = batch_features

    return features


def build_oof_knn_features(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    top_ks: Sequence[int],
    folds: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    if folds < 2:
        raise ValueError("OOF kNN requires at least 2 folds")
    folds = min(int(folds), int(np.bincount(labels).min()))
    if folds < 2:
        raise ValueError("Not enough samples per class for OOF kNN")

    reference = _fit_standard_l2_reference(matrix)
    normalized = reference["normalized"]
    features = np.empty((matrix.shape[0], len(_knn_feature_names(top_ks))), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_sizes = []
    for fold_index, (memory_idx, query_idx) in enumerate(splitter.split(matrix, labels)):
        fold_sizes.append(int(query_idx.shape[0]))
        features[query_idx] = _knn_support_features_from_norm(
            normalized[query_idx],
            normalized[memory_idx],
            labels[memory_idx],
            top_ks,
            batch_size=batch_size,
        )
        print(
            f"[knn-oof] fold={fold_index + 1}/{folds} query={query_idx.shape[0]} memory={memory_idx.shape[0]}",
            flush=True,
        )
    return features, {"folds": folds, "fold_sizes": fold_sizes}


def build_frozen_knn_reference(matrix: np.ndarray, labels: np.ndarray) -> dict:
    reference = _fit_standard_l2_reference(matrix)
    return {
        "mean": reference["mean"],
        "std": reference["std"],
        "memory_norm": reference["normalized"],
        "memory_labels": labels.astype(np.int64, copy=False),
    }


def append_frozen_knn_features(
    matrix: np.ndarray,
    frozen_reference: dict,
    top_ks: Sequence[int],
    *,
    batch_size: int,
) -> np.ndarray:
    query_norm = _normalize_with_reference(
        matrix,
        {
            "mean": frozen_reference["mean"],
            "std": frozen_reference["std"],
        },
    )
    knn_features = _knn_support_features_from_norm(
        query_norm,
        frozen_reference["memory_norm"],
        frozen_reference["memory_labels"],
        top_ks,
        batch_size=batch_size,
    )
    return np.hstack([matrix, knn_features]).astype(np.float32, copy=False)


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "errors": int(fp + fn),
    }


def select_best_threshold(scores: np.ndarray, labels: np.ndarray, thresholds: Sequence[float]) -> dict:
    rows = [metrics_at_threshold(scores, labels, threshold) for threshold in thresholds]
    rows.sort(key=lambda row: (row["f1"], -row["errors"], row["threshold"]), reverse=True)
    return rows[0]


def suspected_noise_mask(labels: np.ndarray, base_probs: np.ndarray, *, low: float = 0.05, high: float = 0.95) -> np.ndarray:
    return ((labels == 1) & (base_probs <= low)) | ((labels == 0) & (base_probs >= high))


def _knn_reference_columns(knn_feature_names: Sequence[str]) -> dict:
    top_ks = []
    for name in knn_feature_names:
        if not name.startswith("knn") or not name.endswith("_mal_ratio"):
            continue
        value = name[3 : -len("_mal_ratio")]
        if value.isdigit():
            top_ks.append(int(value))
    if not top_ks:
        raise ValueError("No kNN malicious-ratio feature columns were found")

    ref_k = 25 if 25 in top_ks else max(top_ks)
    aux_k = 10 if 10 in top_ks else min(top_ks)
    columns = {name: index for index, name in enumerate(knn_feature_names)}
    required = [
        f"knn{ref_k}_mal_ratio",
        f"knn{ref_k}_weighted_mal_ratio",
        f"knn{aux_k}_mal_ratio",
        f"knn{aux_k}_weighted_mal_ratio",
    ]
    for name in required:
        if name not in columns:
            raise ValueError(f"Missing kNN feature column: {name}")
    return {
        "ref_k": ref_k,
        "aux_k": aux_k,
        "ref_mal_ratio": columns[f"knn{ref_k}_mal_ratio"],
        "ref_weighted_mal_ratio": columns[f"knn{ref_k}_weighted_mal_ratio"],
        "aux_mal_ratio": columns[f"knn{aux_k}_mal_ratio"],
        "aux_weighted_mal_ratio": columns[f"knn{aux_k}_weighted_mal_ratio"],
        "top1_label": columns.get("knn_top1_label"),
        "top1_similarity": columns.get("knn_top1_similarity"),
        "top1_top2_gap": columns.get("knn_top1_top2_gap"),
    }


def knn_conflict_masks(
    labels: np.ndarray,
    knn_features: np.ndarray,
    knn_feature_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict]:
    columns = _knn_reference_columns(knn_feature_names)
    labels = labels.astype(np.int64, copy=False)
    label0 = labels == 0
    label1 = labels == 1

    ref_mal_ratio = knn_features[:, columns["ref_mal_ratio"]]
    ref_weighted_mal_ratio = knn_features[:, columns["ref_weighted_mal_ratio"]]
    aux_mal_ratio = knn_features[:, columns["aux_mal_ratio"]]
    aux_weighted_mal_ratio = knn_features[:, columns["aux_weighted_mal_ratio"]]

    ref_opp_ratio = np.where(label1, 1.0 - ref_mal_ratio, ref_mal_ratio)
    ref_weighted_opp_ratio = np.where(label1, 1.0 - ref_weighted_mal_ratio, ref_weighted_mal_ratio)
    aux_opp_ratio = np.where(label1, 1.0 - aux_mal_ratio, aux_mal_ratio)
    aux_weighted_opp_ratio = np.where(label1, 1.0 - aux_weighted_mal_ratio, aux_weighted_mal_ratio)

    if columns["top1_similarity"] is not None:
        top1_similarity = knn_features[:, columns["top1_similarity"]]
    else:
        top1_similarity = np.ones(labels.shape[0], dtype=np.float32)

    strong = (ref_opp_ratio >= 0.80) & (ref_weighted_opp_ratio >= 0.80) & (top1_similarity >= 0.95)
    medium = strong | (
        (top1_similarity >= 0.90)
        & (
            ((aux_opp_ratio >= 0.70) & (aux_weighted_opp_ratio >= 0.70))
            | ((ref_opp_ratio >= 0.70) & (ref_weighted_opp_ratio >= 0.65))
        )
    )

    if columns["top1_label"] is not None:
        top1_label = np.rint(knn_features[:, columns["top1_label"]]).astype(np.int64)
        top1_opposes_label = top1_label != labels
    else:
        top1_opposes_label = np.zeros(labels.shape[0], dtype=bool)
    exact_opposite = (
        top1_opposes_label
        & (ref_opp_ratio >= 0.90)
        & (ref_weighted_opp_ratio >= 0.85)
        & (top1_similarity >= 0.95)
    )

    masks = {
        "medium": medium,
        "strong": strong,
        "exact_opposite": exact_opposite,
    }
    metadata = {
        "rule_version": "train_oof_knn_conflict_v2",
        "ref_k": int(columns["ref_k"]),
        "aux_k": int(columns["aux_k"]),
        "rules": {
            "opposite_ratio": "label1 uses 1-malicious_ratio; label0 uses malicious_ratio",
            "medium": "top1_similarity>=0.90 and (aux opposite ratio>=0.70/weighted>=0.70 or ref opposite ratio>=0.70/weighted>=0.65)",
            "strong": "ref opposite ratio>=0.80 and weighted opposite ratio>=0.80 and top1_similarity>=0.95",
            "exact_opposite": "top1 label opposes dataset label, ref opposite ratio>=0.90, weighted opposite ratio>=0.85, top1_similarity>=0.95",
        },
    }
    metadata["top1_similarity_mean"] = float(top1_similarity.mean())
    metadata["ref_opposite_ratio_mean"] = float(ref_opp_ratio.mean())
    metadata["ref_opposite_ratio_p95"] = float(np.quantile(ref_opp_ratio, 0.95))
    metadata["aux_opposite_ratio_mean"] = float(aux_opp_ratio.mean())
    metadata["aux_opposite_ratio_p95"] = float(np.quantile(aux_opp_ratio, 0.95))
    return masks, metadata


def summarize_knn_conflicts(
    labels: np.ndarray,
    knn_features: Optional[np.ndarray],
    knn_feature_names: Sequence[str],
) -> dict:
    if knn_features is None:
        return {"enabled": False}
    masks, metadata = knn_conflict_masks(labels, knn_features, knn_feature_names)
    summary = {"enabled": True, **metadata}
    for name, mask in masks.items():
        summary[f"{name}_count"] = int(mask.sum())
        summary[f"{name}_ratio"] = float(mask.mean())
        summary[f"{name}_label0"] = int((mask & (labels == 0)).sum())
        summary[f"{name}_label1"] = int((mask & (labels == 1)).sum())
    return summary


def summarize_weights(labels: np.ndarray, weights: np.ndarray) -> dict:
    positive = weights > 0.0
    return {
        "effective_train_rows": int(np.count_nonzero(positive)),
        "zero_weight_rows": int(np.count_nonzero(~positive)),
        "mean_weight": float(weights.mean()),
        "min_weight": float(weights.min()),
        "label0_zero_weight": int((~positive & (labels == 0)).sum()),
        "label1_zero_weight": int((~positive & (labels == 1)).sum()),
    }


def sample_weights(
    labels: np.ndarray,
    base_probs: np.ndarray,
    mode: str,
    *,
    knn_features: Optional[np.ndarray] = None,
    knn_feature_names: Sequence[str] = (),
) -> np.ndarray:
    weights = np.ones(labels.shape[0], dtype=np.float32)
    if mode == "none":
        return weights
    if mode == "soft_conflict_downweight":
        severe = suspected_noise_mask(labels, base_probs, low=0.05, high=0.95)
        medium = ((labels == 1) & (base_probs <= 0.15)) | ((labels == 0) & (base_probs >= 0.85))
        weights[medium] = 0.5
        weights[severe] = 0.15
        return weights
    if mode == "trim_extreme_conflict":
        severe = suspected_noise_mask(labels, base_probs, low=0.03, high=0.97)
        weights[severe] = 0.0
        return weights
    if mode.startswith("knn_"):
        if knn_features is None:
            raise ValueError(f"Noise mode {mode} requires --knn-features")
        masks, _metadata = knn_conflict_masks(labels, knn_features, knn_feature_names)
        if mode == "knn_soft_conflict_downweight":
            weights[masks["medium"]] = 0.60
            weights[masks["strong"]] = 0.25
            weights[masks["exact_opposite"]] = 0.10
            return weights
        if mode == "knn_trim_strong_conflict":
            weights[masks["strong"]] = 0.0
            return weights
        if mode == "knn_trim_exact_opposite":
            weights[masks["exact_opposite"]] = 0.0
            return weights
    raise ValueError(f"Unknown noise mode: {mode}")


def model_candidates(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "hgb_lr0.04_leaf15_l2_0",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.0,
                max_iter=320,
                random_state=seed,
            ),
        ),
        (
            "hgb_lr0.06_leaf31_l2_0",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.0,
                max_iter=260,
                random_state=seed,
            ),
        ),
        (
            "hgb_lr0.08_leaf31_l2_1e-3",
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0e-3,
                max_iter=220,
                random_state=seed,
            ),
        ),
        (
            "hgb_lr0.10_leaf63_l2_1e-3",
            HistGradientBoostingClassifier(
                learning_rate=0.10,
                max_leaf_nodes=63,
                l2_regularization=1.0e-3,
                max_iter=180,
                random_state=seed,
            ),
        ),
        (
            "extra_trees_300_leaf1",
            ExtraTreesClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=seed,
                class_weight=None,
            ),
        ),
        (
            "extra_trees_500_leaf2",
            ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
                class_weight=None,
            ),
        ),
        (
            "rf_300_leaf2",
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
                class_weight=None,
            ),
        ),
        (
            "logreg_l2_c1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", C=1.0),
            ),
        ),
    ]


def filter_model_candidates(candidates: list[tuple[str, object]], names: str) -> list[tuple[str, object]]:
    if names.strip() == "__none__":
        return []
    selected_names = [name.strip() for name in names.split(",") if name.strip()]
    if not selected_names:
        return candidates
    selected = [(name, model) for name, model in candidates if name in selected_names]
    missing = sorted(set(selected_names) - {name for name, _model in selected})
    if missing:
        available = ", ".join(name for name, _model in candidates)
        raise ValueError(f"Unknown model candidate(s): {missing}. Available: {available}")
    return selected


def predict_scores(model, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1].astype(np.float32, copy=False)
    scores = model.decision_function(matrix)
    scores = np.clip(scores, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32, copy=False)


def write_predictions(path: Path, rows: Sequence[dict], labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "stage2_prob_malicious",
        "prediction",
        "correct",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row, label, score in zip(rows, labels, scores):
            prediction = int(score >= threshold)
            writer.writerow(
                {
                    "source_path": row.get("source_path", ""),
                    "cache_path": row.get("cache_path", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "label": int(label),
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                    "stage2_prob_malicious": f"{float(score):.10f}",
                    "prediction": prediction,
                    "correct": prediction == int(label),
                }
            )


def summarize_noise(labels: np.ndarray, base_probs: np.ndarray) -> dict:
    severe = suspected_noise_mask(labels, base_probs, low=0.05, high=0.95)
    medium = ((labels == 1) & (base_probs <= 0.15)) | ((labels == 0) & (base_probs >= 0.85))
    return {
        "medium_conflict_count": int(medium.sum()),
        "severe_conflict_count": int(severe.sum()),
        "medium_conflict_ratio": float(medium.mean()),
        "severe_conflict_ratio": float(severe.mean()),
        "label0_severe": int((severe & (labels == 0)).sum()),
        "label1_severe": int((severe & (labels == 1)).sum()),
    }


def clean_slice_metrics(scores: np.ndarray, labels: np.ndarray, base_probs: np.ndarray, threshold: float) -> dict:
    severe = suspected_noise_mask(labels, base_probs, low=0.05, high=0.95)
    clean = ~severe
    if clean.sum() == 0:
        return {"samples": 0}
    result = metrics_at_threshold(scores[clean], labels[clean], threshold)
    result["samples"] = int(clean.sum())
    result["excluded_suspected_noise"] = int(severe.sum())
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run cache-backed stage-2 validation matrix.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.05:0.95:0.005")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--chunk-count", type=int, default=16)
    parser.add_argument("--feature-set", choices=["tabular", "extended"], default="extended")
    parser.add_argument(
        "--content-pe-features",
        action="store_true",
        help="Append production-stable PE metadata extracted from file content only.",
    )
    parser.add_argument(
        "--content-pe-cache-dir",
        type=Path,
        default=None,
        help="Optional sidecar cache for content-only PE metadata features.",
    )
    parser.add_argument(
        "--content-pe-v2-features",
        action="store_true",
        help="Append expanded content-only PE import/export/resource/section features.",
    )
    parser.add_argument(
        "--content-pe-v2-cache-dir",
        type=Path,
        default=None,
        help="Optional sidecar cache for expanded content-only PE v2 features.",
    )
    parser.add_argument(
        "--content-pe-v2-groups",
        default="all",
        help=(
            "Comma-separated content PE v2 feature groups. "
            "Available: all, import_dll, api, delay_import, imports, export, resource, section."
        ),
    )
    parser.add_argument(
        "--content-string-features",
        action="store_true",
        help="Append production-stable binary string/keyword features extracted from file content only.",
    )
    parser.add_argument(
        "--content-string-cache-dir",
        type=Path,
        default=None,
        help="Optional sidecar cache for content-only string features.",
    )
    parser.add_argument(
        "--content-cert-features",
        action="store_true",
        help="Append production-stable Authenticode certificate blob features extracted from file content only.",
    )
    parser.add_argument(
        "--content-cert-cache-dir",
        type=Path,
        default=None,
        help="Optional sidecar cache for content-only certificate features.",
    )
    parser.add_argument("--noise-modes", default="none,soft_conflict_downweight,trim_extreme_conflict")
    parser.add_argument("--test-val-f1-gate", type=float, default=0.980)
    parser.add_argument("--knn-features", action="store_true", help="Append train-only kNN label-support features.")
    parser.add_argument("--knn-top-k", default="5,10,25,50")
    parser.add_argument("--knn-folds", type=int, default=5)
    parser.add_argument("--knn-batch-size", type=int, default=2048)
    parser.add_argument(
        "--model-candidates",
        default="",
        help="Comma-separated model candidate names. Empty keeps the full default matrix.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    checkpoint = load_safe_checkpoint(resolve_path(args.checkpoint), map_location="cpu")
    checkpoint_config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    content_cache_dir = None
    if args.content_pe_features:
        content_cache_dir = resolve_path(args.content_pe_cache_dir or (output_dir / "content_pe_cache_v1"))
    content_pe_v2_cache_dir = None
    if args.content_pe_v2_features:
        content_pe_v2_cache_dir = resolve_path(args.content_pe_v2_cache_dir or (output_dir / "content_pe_v2_cache"))
    content_pe_v2_groups = parse_content_pe_v2_groups(args.content_pe_v2_groups)
    content_string_cache_dir = None
    if args.content_string_features:
        content_string_cache_dir = resolve_path(args.content_string_cache_dir or (output_dir / "content_string_cache_v1"))
    content_cert_cache_dir = None
    if args.content_cert_features:
        content_cert_cache_dir = resolve_path(args.content_cert_cache_dir or (output_dir / "content_cert_cache_v1"))
    feature_config = FeatureConfig(
        prefix_len=max(0, int(args.prefix_len)),
        chunk_count=max(1, int(args.chunk_count)),
        include_pe=True,
        include_stat=True,
        include_lightweight=args.feature_set == "extended",
        include_byte_summary=args.feature_set == "extended",
        include_content_pe=bool(args.content_pe_features),
        content_cache_dir=str(content_cache_dir) if content_cache_dir is not None else None,
        include_content_pe_v2=bool(args.content_pe_v2_features),
        content_pe_v2_cache_dir=str(content_pe_v2_cache_dir) if content_pe_v2_cache_dir is not None else None,
        content_pe_v2_groups=content_pe_v2_groups,
        include_content_string=bool(args.content_string_features),
        content_string_cache_dir=str(content_string_cache_dir) if content_string_cache_dir is not None else None,
        include_content_cert=bool(args.content_cert_features),
        content_cert_cache_dir=str(content_cert_cache_dir) if content_cert_cache_dir is not None else None,
    )

    train_rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_prediction_rows(args.val_predictions, args.max_val_rows)
    print(f"[load] train rows={len(train_rows)} val rows={len(val_rows)}", flush=True)
    train_x, train_y, train_base, train_kept_rows, train_counts = build_matrix(train_rows, checkpoint_config, feature_config)
    val_x, val_y, val_base, val_kept_rows, val_counts = build_matrix(val_rows, checkpoint_config, feature_config)
    print(f"[matrix] train={train_x.shape} val={val_x.shape}", flush=True)

    base_feature_dim = int(train_x.shape[1])
    knn_config = {
        "enabled": bool(args.knn_features),
        "top_ks": parse_int_list(args.knn_top_k),
        "folds": int(args.knn_folds),
        "batch_size": int(args.knn_batch_size),
        "feature_names": [],
        "oof": None,
    }
    frozen_knn_reference = None
    train_knn = None
    if args.knn_features:
        top_ks = knn_config["top_ks"]
        knn_config["feature_names"] = _knn_feature_names(top_ks)
        print(
            f"[knn] building OOF train features top_k={top_ks} folds={args.knn_folds} batch={args.knn_batch_size}",
            flush=True,
        )
        train_knn, oof_info = build_oof_knn_features(
            train_x,
            train_y,
            top_ks=top_ks,
            folds=int(args.knn_folds),
            seed=int(args.seed),
            batch_size=int(args.knn_batch_size),
        )
        frozen_knn_reference = build_frozen_knn_reference(train_x, train_y)
        val_x = append_frozen_knn_features(
            val_x,
            frozen_knn_reference,
            top_ks,
            batch_size=int(args.knn_batch_size),
        )
        train_x = np.hstack([train_x, train_knn]).astype(np.float32, copy=False)
        knn_config["oof"] = oof_info
        print(f"[knn] augmented train={train_x.shape} val={val_x.shape}", flush=True)
    safe_feature_name_groups = assert_stage2_feature_names_safe(
        feature_config,
        knn_config["feature_names"],
        checkpoint_config,
    )

    thresholds = parse_thresholds(args.thresholds)
    baseline_val_best = select_best_threshold(val_base, val_y, thresholds)
    results = []
    fitted = []
    noise_modes = [item.strip() for item in args.noise_modes.split(",") if item.strip()]
    candidates = filter_model_candidates(model_candidates(int(args.seed)), args.model_candidates)
    for noise_mode in noise_modes:
        weights = sample_weights(
            train_y,
            train_base,
            noise_mode,
            knn_features=train_knn,
            knn_feature_names=knn_config["feature_names"],
        )
        weight_summary = summarize_weights(train_y, weights)
        effective_train_rows = int(weight_summary["effective_train_rows"])
        for model_name, model in candidates:
            start = time.perf_counter()
            fit_kwargs = {}
            if not isinstance(model, type(make_pipeline(StandardScaler(), LogisticRegression()))):
                fit_kwargs["sample_weight"] = weights
            try:
                model.fit(train_x, train_y, **fit_kwargs)
            except TypeError:
                model.fit(train_x, train_y)
            fit_sec = time.perf_counter() - start
            val_scores = predict_scores(model, val_x)
            val_best = select_best_threshold(val_scores, val_y, thresholds)
            clean_val = clean_slice_metrics(val_scores, val_y, val_base, float(val_best["threshold"]))
            result = {
                "name": f"{model_name}__noise_{noise_mode}",
                "base_model": model_name,
                "noise_mode": noise_mode,
                "fit_sec": fit_sec,
                "effective_train_rows": effective_train_rows,
                "weight_summary": weight_summary,
                "val_best": val_best,
                "clean_val_at_val_threshold": clean_val,
                "delta_val_f1_vs_baseline": val_best["f1"] - baseline_val_best["f1"],
            }
            results.append(result)
            fitted.append((val_best["f1"], -val_best["errors"], result, model, val_scores))
            print(
                f"[val] {result['name']} f1={val_best['f1']:.6f} errors={val_best['errors']} "
                f"threshold={val_best['threshold']:.4f} fit_sec={fit_sec:.1f}",
                flush=True,
            )

    fitted.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_f1, _neg_errors, selected, selected_model, selected_val_scores = fitted[0]
    report = {
        "schema": "axon_stage2_cache_matrix_v1",
        "protocol": "train predictions/cache fit candidates; val selects model/noise mode/threshold; test10k only if val gate passes",
        "checkpoint": str(resolve_path(args.checkpoint)),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "test_predictions": str(resolve_path(args.test_predictions)) if args.test_predictions else None,
        "feature_config": feature_config.__dict__,
        "identity_feature_policy": (
            "source_path/source_sha256/cache_path/sample_index/split/filename/extension/directory are identity "
            "or audit fields only and are forbidden as model features"
        ),
        "feature_name_groups": safe_feature_name_groups,
        "content_pe_feature_names": CONTENT_PE_FEATURE_NAMES if feature_config.include_content_pe else [],
        "content_pe_v2_feature_names": (
            content_pe_v2_selected_feature_names(getattr(feature_config, "content_pe_v2_groups", ("all",)))
            if getattr(feature_config, "include_content_pe_v2", False)
            else []
        ),
        "content_pe_v2_group_feature_counts": {
            group: len(indexes) for group, indexes in sorted(CONTENT_PE_V2_GROUPS.items())
        },
        "content_string_feature_names": CONTENT_STRING_FEATURE_NAMES if feature_config.include_content_string else [],
        "content_cert_feature_names": CONTENT_CERT_FEATURE_NAMES if feature_config.include_content_cert else [],
        "records": {"train": train_counts, "val": val_counts},
        "base_feature_dim": base_feature_dim,
        "feature_dim": int(train_x.shape[1]),
        "knn_config": knn_config,
        "noise_summary": {
            "train": summarize_noise(train_y, train_base),
            "val": summarize_noise(val_y, val_base),
        },
        "knn_conflict_summary": summarize_knn_conflicts(train_y, train_knn, knn_config["feature_names"]),
        "baseline_val_best": baseline_val_best,
        "models": sorted(results, key=lambda row: (row["val_best"]["f1"], -row["val_best"]["errors"]), reverse=True),
        "selected_by_val": selected,
    }

    selected_threshold = float(selected["val_best"]["threshold"])
    write_predictions(output_dir / "stage2_val_predictions.csv", val_kept_rows, val_y, selected_val_scores, selected_threshold)

    test_ran = False
    if args.test_predictions is not None and selected_f1 >= float(args.test_val_f1_gate):
        test_rows = read_prediction_rows(args.test_predictions, args.max_test_rows)
        test_x, test_y, test_base, test_kept_rows, test_counts = build_matrix(test_rows, checkpoint_config, feature_config)
        if args.knn_features:
            test_x = append_frozen_knn_features(
                test_x,
                frozen_knn_reference,
                knn_config["top_ks"],
                batch_size=int(args.knn_batch_size),
            )
        test_scores = predict_scores(selected_model, test_x)
        test_metrics = metrics_at_threshold(test_scores, test_y, selected_threshold)
        report["records"]["test"] = test_counts
        report["test_at_val_threshold"] = test_metrics
        report["clean_test_at_val_threshold"] = clean_slice_metrics(test_scores, test_y, test_base, selected_threshold)
        report["noise_summary"]["test"] = summarize_noise(test_y, test_base)
        write_predictions(output_dir / "stage2_test_predictions.csv", test_kept_rows, test_y, test_scores, selected_threshold)
        test_ran = True
    else:
        report["test_skipped"] = {
            "reason": "selected val F1 below gate or no test predictions provided",
            "selected_val_f1": float(selected_f1),
            "gate": float(args.test_val_f1_gate),
        }

    model_path = output_dir / "stage2_selected_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": selected_model,
                "feature_config": feature_config,
                "threshold": selected_threshold,
                "selected": selected,
                "checkpoint_config": checkpoint_config.to_dict(),
                "identity_feature_policy": report["identity_feature_policy"],
                "feature_name_groups": safe_feature_name_groups,
                "content_pe_feature_names": CONTENT_PE_FEATURE_NAMES if feature_config.include_content_pe else [],
                "content_pe_v2_feature_names": (
                    content_pe_v2_selected_feature_names(getattr(feature_config, "content_pe_v2_groups", ("all",)))
                    if getattr(feature_config, "include_content_pe_v2", False)
                    else []
                ),
                "content_string_feature_names": (
                    CONTENT_STRING_FEATURE_NAMES if feature_config.include_content_string else []
                ),
                "content_cert_feature_names": CONTENT_CERT_FEATURE_NAMES if feature_config.include_content_cert else [],
                "knn": {
                    "enabled": bool(args.knn_features),
                    "top_ks": knn_config["top_ks"],
                    "batch_size": int(args.knn_batch_size),
                    "feature_names": knn_config["feature_names"],
                    "reference": frozen_knn_reference,
                },
            },
            handle,
        )
    report["model_path"] = str(model_path)
    report["test_ran"] = test_ran
    report_path = output_dir / "stage2_cache_matrix_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "test": report.get("test_at_val_threshold")}, indent=2, ensure_ascii=False))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
