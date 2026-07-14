"""Loop167 overlap policy and frozen Axon baseline registry."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

try:
    from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES
    from kvd_features.schema_names import fixed_v2_feature_names
except ModuleNotFoundError:  # Supports pytest imports rooted at the repository.
    from src.kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES
    from src.kvd_features.schema_names import fixed_v2_feature_names

from .semantic_schema import OFFICIAL_DIMENSION, OfficialColumn, canonical_official_columns

CATEGORY_EXACT = "exact_overlap"
CATEGORY_PARTIAL = "partial_overlap"
CATEGORY_NOVEL = "genuinely_novel"
CATEGORY_FORBIDDEN = "forbidden_or_unstable"
CATEGORIES = (CATEGORY_EXACT, CATEGORY_PARTIAL, CATEGORY_NOVEL, CATEGORY_FORBIDDEN)

EMBER2024_COMMIT = "0ef753e81d98bf209f71b03cd331dfc190b5b54d"
EMBER2024_FEATURES_SHA256 = "58a085e9ad307aa2c52e165985ff80db8fd5b763891c0cba2d1758a4825f7273"
EMBER2024_WARNINGS_SHA256 = "a23a9d0a7a938b19390a75fe0eb024dbc9bad7a134bb1511a2913f365a52e5fb"

HEADER_NOVEL_LOCAL_INDICES = frozenset(
    {
        2,
        4,
        7,
        8,
        11,
        12,
        13,
        14,
        20,
        21,
        22,
        23,
        25,
        29,
        *range(57, 74),
    }
)
HEADER_PARTIAL_LOCAL_INDICES = frozenset({0})
HEADER_EXACT_LOCAL_INDICES = frozenset(
    set(range(74)) - HEADER_NOVEL_LOCAL_INDICES - HEADER_PARTIAL_LOCAL_INDICES
)


@dataclass(frozen=True)
class BaselineColumn:
    """One named Axon structural source candidate before exact deduplication."""

    inventory_index: int
    source_family: str
    source_index: int
    feature_name: str
    semantic_key: str
    canonical_source: str
    included: bool
    dropped_duplicate_of: str | None


def _category_for(column: OfficialColumn) -> str:
    local = column.index
    group = column.official_group
    if group == "general":
        if local == 0:
            return CATEGORY_EXACT
        if local in {1, 2}:
            return CATEGORY_PARTIAL
        return CATEGORY_NOVEL
    if group == "histogram":
        return CATEGORY_PARTIAL
    if group == "byteentropy":
        return CATEGORY_NOVEL
    if group == "strings":
        return CATEGORY_PARTIAL
    if group == "header":
        local -= 696
        if local in HEADER_NOVEL_LOCAL_INDICES:
            return CATEGORY_NOVEL
        if local in HEADER_PARTIAL_LOCAL_INDICES:
            return CATEGORY_PARTIAL
        if local in HEADER_EXACT_LOCAL_INDICES:
            return CATEGORY_EXACT
    if group == "section":
        local -= 770
        if local < 11:
            return CATEGORY_PARTIAL
        if local < 221:
            return CATEGORY_FORBIDDEN
        return CATEGORY_EXACT
    if group == "imports":
        return CATEGORY_EXACT if local - 994 < 2 else CATEGORY_FORBIDDEN
    if group == "exports":
        return CATEGORY_EXACT if local - 2276 == 0 else CATEGORY_FORBIDDEN
    if group == "datadirectories":
        local -= 2405
        return CATEGORY_FORBIDDEN if local in {30, 31} else CATEGORY_PARTIAL
    if group == "richheader":
        return CATEGORY_NOVEL if local - 2439 == 0 else CATEGORY_FORBIDDEN
    if group == "authenticode":
        return CATEGORY_PARTIAL
    if group == "pefilewarnings":
        return CATEGORY_FORBIDDEN
    raise ValueError(f"Cannot classify official column {column.index}")


def _overlap_targets(column: OfficialColumn, category: str) -> tuple[str, ...]:
    if category == CATEGORY_NOVEL:
        return ()
    if category == CATEGORY_FORBIDDEN:
        return ()
    group = column.official_group
    local = column.index
    if group == "general":
        if local == 0:
            return ("fixed_v2.fixed_v2_file_size",)
        if local == 1:
            return ("stat.global_entropy_normalized", "content_string.string_entropy")
        return ("phase_b.parse_success_missing_indicator",)
    if group == "histogram":
        return ("axon_byte_sequence_distribution", "stat.byte_count_summary")
    if group == "strings":
        return ("content_string",)
    if group == "header":
        if local - 696 == 0:
            return ("content_pe_v1.content_timestamp_valid", "content_pe_v1.content_timestamp_year_norm")
        return ("fixed_v2", "content_pe_v1")
    if group == "section":
        if local - 770 >= 221:
            return ("content_pe_v1.content_overlay_*",)
        return ("fixed_v2", "content_pe_v1", "content_pe_v2")
    if group == "imports":
        return ("content_pe_v1.content_import_*",)
    if group == "exports":
        return ("content_pe_v1.content_export_*",)
    if group == "datadirectories":
        return ("content_pe_v1.content_dir_*", "fixed_v2")
    if group == "authenticode":
        return ("content_cert", "fixed_v2.fixed_v2_has_signature")
    raise ValueError(f"Unexpected overlap group: {group}")


def _transform_relation(column: OfficialColumn, category: str) -> str:
    if category == CATEGORY_EXACT:
        return "same_source_semantics_or_literal_output"
    if category == CATEGORY_PARTIAL:
        return "related_existing_semantics_with_different_scope_or_transform"
    if category == CATEGORY_NOVEL:
        return "not_present_in_frozen_572_dim_axonal_structural_inventory"
    if column.official_group in {"section", "imports", "exports", "richheader"}:
        return "unfrozen_feature_hasher_semantics"
    if column.official_group == "pefilewarnings":
        return "unfrozen_parser_warning_vocabulary_and_runtime"
    return "known_dead_or_unstable_official_column"


def _missing_policy(column: OfficialColumn, category: str) -> str:
    if category == CATEGORY_FORBIDDEN:
        return "excluded_from_all_fit_matrices"
    if column.official_group in {"general", "histogram", "byteentropy", "strings"}:
        return "empty_input_zero_vector_with_explicit_empty_input_reason"
    if category == CATEGORY_NOVEL:
        return "zero_fill_with_novel_missing_reason_then_M_and_CF_fallback_to_B0"
    return "zero_fill_with_frozen_missing_indicator"


def _finite_policy(category: str) -> str:
    if category == CATEGORY_FORBIDDEN:
        return "not_materialized"
    return "float32_finite_or_fail_closed_before_cache_write"


def _implementation_status(column: OfficialColumn, category: str) -> str:
    if category == CATEGORY_NOVEL:
        return "implemented_native_phase_a"
    if category == CATEGORY_FORBIDDEN:
        return "excluded_pending_new_source_closed_parity_revision"
    if column.official_group == "authenticode":
        return "semantic_control_only_pending_pinned_native_authenticode_contract"
    return "semantic_control_only_pending_phase_b_one_pass_extractor"


def semantic_mapping_rows() -> tuple[dict[str, object], ...]:
    """Classify every official column exactly once under the Phase-A policy."""

    rows: list[dict[str, object]] = []
    for column in canonical_official_columns():
        category = _category_for(column)
        rows.append(
            {
                "index": column.index,
                "official_group": column.official_group,
                "official_name": column.official_name,
                "source_semantics": column.source_semantics,
                "category": category,
                "axon_overlap_targets": list(_overlap_targets(column, category)),
                "transform_relation": _transform_relation(column, category),
                "missing_policy": _missing_policy(column, category),
                "finite_policy": _finite_policy(category),
                "implementation_status": _implementation_status(column, category),
            }
        )
    validate_semantic_mapping_rows(rows)
    return tuple(rows)


def validate_semantic_mapping_rows(rows: Iterable[dict[str, object]]) -> None:
    rows = tuple(rows)
    if len(rows) != OFFICIAL_DIMENSION:
        raise ValueError("Loop167 mapping must contain all 2568 official columns")
    indexes = [row.get("index") for row in rows]
    if indexes != list(range(OFFICIAL_DIMENSION)):
        raise ValueError("Loop167 official indices must be contiguous and source ordered")
    categories = [row.get("category") for row in rows]
    if set(categories) - set(CATEGORIES):
        raise ValueError("Loop167 mapping contains an unknown category")
    counts = Counter(categories)
    expected = {
        CATEGORY_EXACT: 49,
        CATEGORY_PARTIAL: 487,
        CATEGORY_NOVEL: 292,
        CATEGORY_FORBIDDEN: 1740,
    }
    if counts != expected:
        raise ValueError(f"Loop167 mapping category conservation failed: {counts}")
    novel_groups = {row["official_group"] for row in rows if row["category"] == CATEGORY_NOVEL}
    if len(novel_groups) < 3:
        raise ValueError("Loop167 novel set must span at least three semantic groups")
    for row in rows:
        if row["official_group"] in {"datadirectories", "authenticode"} and row["category"] == CATEGORY_NOVEL:
            raise ValueError("Forced overlap control entered the novel set")


def novel_indices() -> tuple[int, ...]:
    return tuple(
        int(row["index"])
        for row in semantic_mapping_rows()
        if row["category"] == CATEGORY_NOVEL
    )


def control_indices() -> tuple[int, ...]:
    return tuple(
        int(row["index"])
        for row in semantic_mapping_rows()
        if row["category"] in {CATEGORY_EXACT, CATEGORY_PARTIAL}
    )


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
V2_API_CATEGORIES = (
    "service",
    "driver",
    "privilege",
    "antidebug",
    "memory",
    "thread",
    "module",
    "process_enum",
    "persistence",
    "network_http",
    "network_socket",
    "file_mutation",
    "crypto_cert",
    "resource",
    "installer",
    "com",
)
V2_RESOURCE_TYPES = (
    "cursor",
    "bitmap",
    "icon",
    "menu",
    "dialog",
    "string",
    "rcdata",
    "group_cursor",
    "group_icon",
    "version",
    "manifest",
)
V2_EXPORT_PATTERNS = ("com", "control_panel", "service", "plugin")
V2_SECTION_NAME_GROUPS = ("code", "data", "resource", "import", "export", "reloc", "tls", "packer")
STRING_PATTERN_NAMES = (
    "url",
    "network",
    "script_exec",
    "persistence",
    "injection",
    "credential",
    "crypto",
    "evasion",
    "vm",
    "packer",
    "file_ops",
    "registry",
    "benign_vendor",
    "version_resource",
)
CERT_OID_NAMES = (
    "pkcs7_signed_data",
    "code_signing",
    "timestamping",
    "sha1",
    "sha256",
    "sha384",
    "rsa",
    "ecdsa_sha256",
)
CERT_VENDOR_NAMES = (
    "microsoft",
    "digicert",
    "sectigo",
    "globalsign",
    "verisign",
    "entrust",
    "ssl_com",
    "google",
    "adobe",
    "intel",
    "nvidia",
    "oracle",
    "mozilla",
    "kaspersky",
    "avast",
)


def _stat_feature_names() -> tuple[str, ...]:
    names = [
        "stat_byte_mean",
        "stat_byte_std",
        "stat_byte_min",
        "stat_byte_max",
        "stat_byte_median",
        "stat_byte_q25",
        "stat_byte_q75",
        "stat_count_0x00",
        "stat_count_0xff",
        "stat_count_0x90",
        "stat_ascii_count",
        "stat_global_entropy_normalized",
    ]
    for segment in range(3):
        names.extend(
            (
                f"stat_segment_{segment}_mean",
                f"stat_segment_{segment}_std",
                f"stat_segment_{segment}_entropy_normalized",
            )
        )
    names.extend(f"stat_chunk_{chunk}_mean" for chunk in range(10))
    names.extend(f"stat_chunk_{chunk}_std" for chunk in range(10))
    names.extend(
        (
            "stat_chunk_mean_abs_diff_mean",
            "stat_chunk_mean_diff_std",
            "stat_chunk_mean_diff_max",
            "stat_chunk_mean_diff_min",
            "stat_chunk_std_abs_diff_mean",
            "stat_chunk_std_diff_std",
            "stat_chunk_std_diff_max",
            "stat_chunk_std_diff_min",
        )
    )
    if len(names) != 49:
        raise RuntimeError("Axon stat schema dimension drift")
    return tuple(names)


def _v2_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for dll_name in V2_IMPORT_DLLS:
        stem = dll_name[:-4].replace(".", "_")
        names.extend((f"v2_import_dll_{stem}_present", f"v2_import_dll_{stem}_api_ratio"))
    for category_name in V2_API_CATEGORIES:
        names.extend(
            (
                f"v2_api_{category_name}_present",
                f"v2_api_{category_name}_count_log",
                f"v2_api_{category_name}_ratio",
            )
        )
    names.extend(
        (
            "v2_delay_import_dll_count_log",
            "v2_delay_import_api_count_log",
            "v2_delay_import_ratio",
            "v2_export_ordinal_only_ratio",
            "v2_export_forwarder_ratio",
            "v2_export_mean_name_len_norm",
            "v2_export_max_name_len_norm",
            "v2_export_ordinal_span_log",
        )
    )
    names.extend(f"v2_export_pattern_{name}_present" for name in V2_EXPORT_PATTERNS)
    names.extend(
        (
            "v2_resource_data_entry_count_log",
            "v2_resource_named_entry_ratio",
            "v2_resource_language_count_log",
            "v2_resource_data_size_log",
            "v2_resource_max_data_size_ratio",
            "v2_resource_mean_entropy",
            "v2_resource_max_entropy",
        )
    )
    for resource_name in V2_RESOURCE_TYPES:
        names.extend(
            (
                f"v2_resource_type_{resource_name}_present",
                f"v2_resource_type_{resource_name}_count_log",
            )
        )
    names.extend(
        (
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
        )
    )
    names.extend(f"v2_section_name_group_{name}_ratio" for name in V2_SECTION_NAME_GROUPS)
    if len(names) != 182:
        raise RuntimeError("Axon content_pe_v2 schema dimension drift")
    return tuple(names)


def _string_feature_names() -> tuple[str, ...]:
    names = [
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
    for name in STRING_PATTERN_NAMES:
        names.extend((f"string_{name}_count_log", f"string_{name}_present"))
    if len(names) != 43:
        raise RuntimeError("Axon content string schema dimension drift")
    return tuple(names)


def _cert_feature_names() -> tuple[str, ...]:
    names = [
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
    names.extend(f"cert_oid_{name}_present" for name in CERT_OID_NAMES)
    for name in CERT_VENDOR_NAMES:
        names.extend((f"cert_vendor_{name}_present", f"cert_vendor_{name}_count_log"))
    if len(names) != 55:
        raise RuntimeError("Axon content certificate schema dimension drift")
    return tuple(names)


def _baseline_source_families() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("fixed_v2", tuple(fixed_v2_feature_names(section_slots=32, pe_feature_dim=143))),
        ("stat", _stat_feature_names()),
        ("content_pe_v1", tuple(CONTENT_PE_V1_FEATURE_NAMES)),
        ("content_pe_v2", _v2_feature_names()),
        ("content_string", _string_feature_names()),
        ("content_cert", _cert_feature_names()),
    )


def frozen_baseline_columns() -> tuple[BaselineColumn, ...]:
    """Return the source-named 572-column inventory with exact duplicate decisions."""

    columns: list[BaselineColumn] = []
    duplicate_target = "fixed_v2.fixed_v2_log_size"
    for source_family, names in _baseline_source_families():
        for source_index, feature_name in enumerate(names):
            full_name = f"{source_family}.{feature_name}"
            dropped_duplicate_of = (
                duplicate_target
                if full_name == "content_pe_v1.content_file_log_size"
                else None
            )
            columns.append(
                BaselineColumn(
                    inventory_index=len(columns),
                    source_family=source_family,
                    source_index=source_index,
                    feature_name=feature_name,
                    semantic_key=full_name,
                    canonical_source=duplicate_target if dropped_duplicate_of else full_name,
                    included=dropped_duplicate_of is None,
                    dropped_duplicate_of=dropped_duplicate_of,
                )
            )
    if len(columns) != 572:
        raise RuntimeError(f"Axon baseline inventory drift: {len(columns)} != 572")
    if sum(column.included for column in columns) != 571:
        raise RuntimeError("Loop167 exact baseline deduplication drift")
    return tuple(columns)


def build_frozen_baseline_allowlist() -> dict[str, object]:
    columns = frozen_baseline_columns()
    return {
        "schema": "axon_loop167_frozen_deduplicated_baseline_allowlist_v1",
        "source_inventory_dimension": len(columns),
        "frozen_allowlist_dimension": sum(column.included for column in columns),
        "deduplication_policy": (
            "Remove only a proven bit-equivalent transform; retain correlated or recoverable "
            "features when their source scope or transform differs."
        ),
        "excluded_feature_families": [
            "base_probability",
            "checkpoint_score",
            "knn_similarity",
            "lightweight_hash",
            "path_filename_extension",
            "source_sha256_row_fold_label_family_time",
        ],
        "required_missing_indicator_names": [
            "missing_fixed_v2",
            "missing_stat",
            "missing_content_pe_v1",
            "missing_content_pe_v2",
            "missing_content_string",
            "missing_content_cert",
        ],
        "feature_names": [column.semantic_key for column in columns if column.included],
        "columns": [asdict(column) for column in columns],
    }


def build_semantic_delta_mapping() -> dict[str, object]:
    rows = semantic_mapping_rows()
    counts = Counter(str(row["category"]) for row in rows)
    return {
        "schema": "axon_loop167_semantic_delta_mapping_v1",
        "external_source": {
            "repository": "https://github.com/FutureComputing4AI/EMBER2024",
            "commit": EMBER2024_COMMIT,
            "features_sha256": EMBER2024_FEATURES_SHA256,
            "warnings_sha256": EMBER2024_WARNINGS_SHA256,
            "reference_execution_allowed": False,
        },
        "official_dimension": OFFICIAL_DIMENSION,
        "category_counts": {category: counts[category] for category in CATEGORIES},
        "novel_indices": list(novel_indices()),
        "control_indices": list(control_indices()),
        "forbidden_policy": "Forbidden columns cannot enter B1, M, A, or CF without a new source-closed revision.",
        "columns": list(rows),
    }
