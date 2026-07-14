#!/usr/bin/env python3
"""Audit Loop28 content PE features before promoting them into stable schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from kvd_features.schema_names import fixed_v2_feature_names  # noqa: E402
from train_stage2_cache_matrix import CONTENT_PE_FEATURE_NAMES  # noqa: E402


KNOWN_FIXED_V2_CONTENT_COVERAGE = {
    "content_file_log_size": ["fixed_v2_log_size"],
    "content_file_size_norm_100mb": ["fixed_v2_file_size"],
    "content_subsystem": ["fixed_v2_subsystem"],
    "content_dll_characteristics": ["fixed_v2_dll_characteristics"],
    "content_dir_debug_present": ["fixed_v2_has_debug_info"],
    "content_dir_tls_present": ["fixed_v2_has_tls"],
    "content_dir_exception_present": ["fixed_v2_has_exceptions"],
    "content_dir_security_present": ["fixed_v2_has_signature"],
    "content_dir_basereloc_present": ["fixed_v2_has_relocs"],
    "content_num_sections_norm": ["fixed_v2_sections_count"],
    "content_api_network_ratio": ["fixed_v2_api_network_ratio"],
    "content_api_process_ratio": ["fixed_v2_api_process_ratio"],
    "content_api_filesystem_ratio": ["fixed_v2_api_filesystem_ratio"],
    "content_api_registry_ratio": ["fixed_v2_api_registry_ratio"],
    "content_api_crypto_ratio": ["fixed_v2_api_crypto_ratio"],
    "content_api_injection_ratio": ["fixed_v2_api_injection_ratio"],
    "content_section_high_entropy_ratio": ["fixed_v2_section_high_entropy_ratio"],
    "content_section_mean_entropy": ["fixed_v2_section_entropy_avg"],
    "content_section_max_entropy": ["fixed_v2_section_entropy_max"],
    "content_section_name_packer_hit_ratio": ["fixed_v2_packer_keyword_hits_ratio"],
}


def _group_gap(feature_name: str) -> str:
    name = feature_name.removeprefix("content_")
    if name.startswith("dir_"):
        return "data_directory_size_ratio"
    if name.startswith("import_") or name.startswith("unique_import") or name.startswith("system_dll"):
        return "import_shape"
    if name.startswith("api_"):
        return "api_category"
    if name.startswith("export_"):
        return "export_shape"
    if name.startswith("resource_"):
        return "resource_shape"
    if name.startswith("overlay_"):
        return "overlay"
    if name.startswith("section_combo_"):
        return "section_permission_combo"
    if name.startswith("section_"):
        return "section_aggregate"
    if name in {
        "machine",
        "characteristics",
        "optional_magic",
        "major_linker_norm",
        "minor_linker_norm",
        "32bit_machine",
        "large_address_aware",
        "is_dll",
        "is_executable_image",
        "is_system",
        "relocs_stripped",
        "debug_stripped",
    }:
        return "header_flags"
    if "alignment" in name or name in {
        "entry_point_ratio",
        "image_base_log",
        "size_of_code_ratio",
        "size_init_data_ratio",
        "size_uninit_data_ratio",
        "size_of_image_ratio",
        "size_of_headers_ratio",
    }:
        return "layout_ratio"
    return "other"


def _count_by_group(features: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature_name in features:
        group = _group_gap(feature_name)
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def build_report(*, section_slots: int = 32, pe_feature_dim: int = 256) -> dict:
    fixed_names = fixed_v2_feature_names(section_slots=section_slots, pe_feature_dim=pe_feature_dim)
    content_names = list(CONTENT_PE_FEATURE_NAMES)

    assert_no_identity_feature_names(fixed_names, context="fixed_v2 PE schema")
    assert_no_identity_feature_names(content_names, context="Loop28 content PE schema")

    fixed_name_set = set(fixed_names)
    content_name_set = set(content_names)
    invalid_mapping = {
        content_name: mapped_names
        for content_name, mapped_names in KNOWN_FIXED_V2_CONTENT_COVERAGE.items()
        if content_name not in content_name_set or any(mapped_name not in fixed_name_set for mapped_name in mapped_names)
    }
    if invalid_mapping:
        raise ValueError(f"Known fixed_v2/content PE coverage mapping is stale: {invalid_mapping}")

    covered = sorted(KNOWN_FIXED_V2_CONTENT_COVERAGE)
    gaps = [name for name in content_names if name not in KNOWN_FIXED_V2_CONTENT_COVERAGE]
    exact_overlaps = sorted(content_name_set & fixed_name_set)

    high_value_gap_groups = {
        group: count
        for group, count in _count_by_group(gaps).items()
        if group
        in {
            "data_directory_size_ratio",
            "header_flags",
            "import_shape",
            "layout_ratio",
            "overlay",
            "resource_shape",
            "section_permission_combo",
        }
    }

    return {
        "schema": "axon_loop49_content_pe_productization_audit_v1",
        "identity_feature_policy": (
            "filename/path/extension/directory/hash/sample index/split/row order are audit-only, "
            "never model evidence"
        ),
        "fixed_v2": {
            "section_slots": section_slots,
            "pe_feature_dim": pe_feature_dim,
            "used_dim_without_reserved": 18 + 3 * section_slots + 29,
            "feature_name_count": len(fixed_names),
            "reserved_feature_count": max(0, pe_feature_dim - (18 + 3 * section_slots + 29)),
        },
        "loop28_content_pe": {
            "feature_count": len(content_names),
            "exact_name_overlap_count": len(exact_overlaps),
            "exact_name_overlaps": exact_overlaps,
            "covered_or_partial_count": len(covered),
            "covered_or_partial": [
                {"content_feature": name, "fixed_v2_features": KNOWN_FIXED_V2_CONTENT_COVERAGE[name]}
                for name in covered
            ],
            "productization_gap_count": len(gaps),
            "productization_gaps": gaps,
            "gap_groups": _count_by_group(gaps),
            "high_value_gap_groups": high_value_gap_groups,
        },
        "decision": {
            "loop49_action": "productization_preflight_only",
            "test10k_allowed": False,
            "recommendation": (
                "Promote Loop28 content PE as a named stable content-derived schema or fixed_v3 candidate; "
                "do not rely on external identity fields. Prioritize gap groups that Loop28 proved useful: "
                "import_shape, layout_ratio, overlay, data_directory_size_ratio, resource_shape, "
                "section_permission_combo, and header_flags."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section-slots", type=int, default=32)
    parser.add_argument("--pe-feature-dim", type=int, default=256)
    parser.add_argument(
        "--output-json",
        default="reports/random_20w_split/loop49_content_pe_productization_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(section_slots=args.section_slots, pe_feature_dim=args.pe_feature_dim)
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[loop49] wrote {output_path}")
    print(
        "[loop49] content PE gaps:",
        report["loop28_content_pe"]["productization_gap_count"],
        "high-value groups:",
        report["loop28_content_pe"]["high_value_gap_groups"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
