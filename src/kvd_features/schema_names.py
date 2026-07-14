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


def fixed_v2_feature_names(section_slots: int = 32, pe_feature_dim: Optional[int] = None) -> list[str]:
    """Return stable column names for the fixed_v2 PE feature vector."""

    if section_slots <= 0:
        raise ValueError("section_slots must be positive")

    names = list(FIXED_V2_HEADER_FEATURE_NAMES)
    for slot in range(section_slots):
        names.extend(
            [
                f"fixed_v2_section_{slot:02d}_is_executable",
                f"fixed_v2_section_{slot:02d}_is_writable",
                f"fixed_v2_section_{slot:02d}_is_readable",
            ]
        )
    names.extend(FIXED_V2_AGGREGATE_FEATURE_NAMES)

    used_dim = len(names)
    if pe_feature_dim is None:
        return names
    if pe_feature_dim < used_dim:
        raise ValueError(
            f"pe_feature_dim ({pe_feature_dim}) must be at least {used_dim} "
            "for fixed_v2 PE schema"
        )
    names.extend(f"fixed_v2_reserved_{idx:03d}" for idx in range(used_dim, pe_feature_dim))
    return names
