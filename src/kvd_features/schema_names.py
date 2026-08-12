"""Stable feature-name helpers for KVD extraction schemas."""

from __future__ import annotations

from typing import Optional


FIXED_V2_HEADER_FEATURE_NAMES = [
    "fixed_v2_file_size",
    "fixed_v2_log_size",
    "fixed_v2_size_of_optional_header",
    "fixed_v2_header_size_ratio",
    "fixed_v2_subsystem",
    "fixed_v2_dll_characteristics",
    "fixed_v2_checksum",
    "fixed_v2_checksum_zero_flag",
    "fixed_v2_has_aslr",
    "fixed_v2_has_nx_compat",
    "fixed_v2_has_guard_cf",
    "fixed_v2_has_seh",
    "fixed_v2_has_debug_info",
    "fixed_v2_has_relocs",
    "fixed_v2_has_tls",
    "fixed_v2_has_exceptions",
    "fixed_v2_has_signature",
    "fixed_v2_sections_count",
]


FIXED_V2_AGGREGATE_FEATURE_NAMES = [
    "fixed_v2_section_entropy_max",
    "fixed_v2_section_entropy_min",
    "fixed_v2_section_entropy_avg",
    "fixed_v2_section_entropy_std",
    "fixed_v2_section_high_entropy_ratio",
    "fixed_v2_section_total_raw_size",
    "fixed_v2_section_total_virtual_size",
    "fixed_v2_section_avg_raw_size",
    "fixed_v2_section_avg_virtual_size",
    "fixed_v2_section_min_raw_size",
    "fixed_v2_section_max_raw_size",
    "fixed_v2_section_raw_size_std",
    "fixed_v2_section_raw_size_cv",
    "fixed_v2_section_names_count",
    "fixed_v2_section_name_avg_length",
    "fixed_v2_section_name_max_length",
    "fixed_v2_section_name_min_length",
    "fixed_v2_long_sections_count",
    "fixed_v2_long_sections_ratio",
    "fixed_v2_short_sections_count",
    "fixed_v2_short_sections_ratio",
    "fixed_v2_api_network_ratio",
    "fixed_v2_api_process_ratio",
    "fixed_v2_api_filesystem_ratio",
    "fixed_v2_api_registry_ratio",
    "fixed_v2_api_crypto_ratio",
    "fixed_v2_api_injection_ratio",
    "fixed_v2_packer_keyword_hits_count",
    "fixed_v2_packer_keyword_hits_ratio",
]


# fixed_v3 is fixed_v2 without `has_signature`. That column was permanently 0:
# it tested `hasattr(pe, 'DIRECTORY_ENTRY_SECURITY')`, an attribute pefile never
# sets (the certificate table lives in OPTIONAL_HEADER.DATA_DIRECTORY[4]), so it
# held a single value across all 200,145 cached samples. Every later column
# shifts down by one; fixed_v2 is frozen for checkpoint reproducibility.
FIXED_V3_HEADER_FEATURE_NAMES = [
    name for name in FIXED_V2_HEADER_FEATURE_NAMES if name != "fixed_v2_has_signature"
]


def _fixed_schema_feature_names(
    version: str,
    header_names: list[str],
    aggregate_names: list[str],
    section_slots: int,
    pe_feature_dim: Optional[int],
) -> list[str]:
    if section_slots <= 0:
        raise ValueError("section_slots must be positive")

    def retag(name: str) -> str:
        return name.replace("fixed_v2_", f"{version}_", 1)

    names = [retag(name) for name in header_names]
    for slot in range(section_slots):
        names.extend(
            [
                f"{version}_section_{slot:02d}_is_executable",
                f"{version}_section_{slot:02d}_is_writable",
                f"{version}_section_{slot:02d}_is_readable",
            ]
        )
    names.extend(retag(name) for name in aggregate_names)

    used_dim = len(names)
    if pe_feature_dim is None:
        return names
    if pe_feature_dim < used_dim:
        raise ValueError(
            f"pe_feature_dim ({pe_feature_dim}) must be at least {used_dim} "
            f"for {version} PE schema"
        )
    names.extend(f"{version}_reserved_{idx:03d}" for idx in range(used_dim, pe_feature_dim))
    return names


def fixed_v2_feature_names(section_slots: int = 32, pe_feature_dim: Optional[int] = None) -> list[str]:
    """Return stable column names for the fixed_v2 PE feature vector."""

    return _fixed_schema_feature_names(
        "fixed_v2",
        FIXED_V2_HEADER_FEATURE_NAMES,
        FIXED_V2_AGGREGATE_FEATURE_NAMES,
        section_slots,
        pe_feature_dim,
    )


def fixed_v3_feature_names(section_slots: int = 32, pe_feature_dim: Optional[int] = None) -> list[str]:
    """Return stable column names for the fixed_v3 PE feature vector."""

    return _fixed_schema_feature_names(
        "fixed_v3",
        FIXED_V3_HEADER_FEATURE_NAMES,
        FIXED_V2_AGGREGATE_FEATURE_NAMES,
        section_slots,
        pe_feature_dim,
    )


def stat_feature_names(segment_count: int = 3, chunk_count: int = 10) -> list[str]:
    """Return stable column names for the statistical feature vector.

    Mirrors ``extract_statistical_features`` and must stay in step with
    ``AxonExperimentConfig.expected_stat_feature_dim``:
    ``7 + 4 + 1 + 3 * segment_count + 2 * chunk_count + 8``.
    """

    if segment_count <= 0 or chunk_count <= 0:
        raise ValueError("segment_count and chunk_count must be positive")

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
    for segment in range(segment_count):
        names.extend(
            (
                f"stat_segment_{segment}_mean",
                f"stat_segment_{segment}_std",
                f"stat_segment_{segment}_entropy_normalized",
            )
        )
    names.extend(f"stat_chunk_{chunk}_mean" for chunk in range(chunk_count))
    names.extend(f"stat_chunk_{chunk}_std" for chunk in range(chunk_count))
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

    expected = 7 + 4 + 1 + 3 * segment_count + 2 * chunk_count + 8
    if len(names) != expected:
        raise RuntimeError(
            f"stat feature name count {len(names)} drifted from expected {expected}"
        )
    return names
