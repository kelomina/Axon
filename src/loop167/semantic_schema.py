"""Canonical EMBER2024-v3 column order frozen for Loop167."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OfficialGroup:
    """One contiguous block in the SHA-bound EMBER2024 feature vector."""

    name: str
    start: int
    stop: int

    @property
    def dimension(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class OfficialColumn:
    """A single official column before Loop167 overlap classification."""

    index: int
    official_group: str
    official_name: str
    source_semantics: str


# This is the executable source order from EMBER2024's PEFeatureExtractor,
# not the older prose order in its dimension summary.
OFFICIAL_GROUPS = (
    OfficialGroup("general", 0, 7),
    OfficialGroup("histogram", 7, 263),
    OfficialGroup("byteentropy", 263, 519),
    OfficialGroup("strings", 519, 696),
    OfficialGroup("header", 696, 770),
    OfficialGroup("section", 770, 994),
    OfficialGroup("imports", 994, 2276),
    OfficialGroup("exports", 2276, 2405),
    OfficialGroup("datadirectories", 2405, 2439),
    OfficialGroup("richheader", 2439, 2472),
    OfficialGroup("authenticode", 2472, 2480),
    OfficialGroup("pefilewarnings", 2480, 2568),
)
OFFICIAL_DIMENSION = 2568

GENERAL_NAMES = (
    "file_size",
    "file_entropy_bits",
    "is_pe",
    "start_byte_0",
    "start_byte_1",
    "start_byte_2",
    "start_byte_3",
)

HEADER_NAMES = (
    "coff_timestamp",
    "coff_number_of_sections",
    "coff_number_of_symbols",
    "coff_size_of_optional_header",
    "coff_pointer_to_symbol_table",
    "coff_machine_category",
    "optional_subsystem_category",
    "optional_major_image_version",
    "optional_minor_image_version",
    "optional_major_linker_version",
    "optional_minor_linker_version",
    "optional_major_operating_system_version",
    "optional_minor_operating_system_version",
    "optional_major_subsystem_version",
    "optional_minor_subsystem_version",
    "optional_size_of_code",
    "optional_size_of_headers",
    "optional_size_of_image",
    "optional_size_of_initialized_data",
    "optional_size_of_uninitialized_data",
    "optional_size_of_stack_reserve",
    "optional_size_of_stack_commit",
    "optional_size_of_heap_reserve",
    "optional_size_of_heap_commit",
    "optional_address_of_entrypoint",
    "optional_base_of_code",
    "optional_image_base",
    "optional_section_alignment",
    "optional_checksum",
    "optional_number_of_rvas_and_sizes",
)
COFF_CHARACTERISTIC_NAMES = (
    "relocs_stripped",
    "executable_image",
    "line_nums_stripped",
    "local_syms_stripped",
    "aggressive_ws_trim",
    "large_address_aware",
    "machine_16bit",
    "bytes_reversed_lo",
    "machine_32bit",
    "debug_stripped",
    "removable_run_from_swap",
    "net_run_from_swap",
    "system",
    "dll",
    "up_system_only",
    "bytes_reversed_hi",
)
DLL_CHARACTERISTIC_NAMES = (
    "high_entropy_va",
    "dynamic_base",
    "force_integrity",
    "nx_compat",
    "no_isolation",
    "no_seh",
    "no_bind",
    "appcontainer",
    "wdm_driver",
    "guard_cf",
    "terminal_server_aware",
)
DOS_HEADER_NAMES = (
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
HEADER_NAMES = (
    HEADER_NAMES
    + tuple(f"coff_characteristic_{name}" for name in COFF_CHARACTERISTIC_NAMES)
    + tuple(f"dll_characteristic_{name}" for name in DLL_CHARACTERISTIC_NAMES)
    + tuple(f"dos_{name}" for name in DOS_HEADER_NAMES)
)

SECTION_GENERAL_NAMES = (
    "section_count",
    "section_zero_raw_count",
    "section_empty_name_count",
    "section_read_execute_count",
    "section_writable_count",
    "section_entropy_max",
    "section_entropy_min",
    "section_raw_size_ratio_max",
    "section_raw_size_ratio_min",
    "section_virtual_size_ratio_max",
    "section_virtual_size_ratio_min",
)
DATA_DIRECTORY_NAMES = (
    "export",
    "import",
    "resource",
    "exception",
    "security",
    "basereloc",
    "debug",
    "copyright",
    "globalptr",
    "tls",
    "load_config",
    "bound_import",
    "iat",
    "delay_import",
    "com_descriptor",
    "reserved",
)
AUTHENTICODE_NAMES = (
    "certificate_count",
    "self_signed",
    "empty_program_name",
    "no_countersigner",
    "parse_error",
    "chain_max_depth",
    "latest_signing_time",
    "signing_time_difference",
)


def _column_name_and_semantics(group: OfficialGroup, local_index: int) -> tuple[str, str]:
    if group.name == "general":
        name = GENERAL_NAMES[local_index]
        return name, "GeneralFileInfo.process_raw_features"
    if group.name == "histogram":
        return (
            f"normalized_byte_count_{local_index:03d}",
            "ByteHistogram normalized full-byte frequency",
        )
    if group.name == "byteentropy":
        entropy_bin, nibble_bin = divmod(local_index, 16)
        return (
            f"local_entropy_bin_{entropy_bin:02d}_high_nibble_{nibble_bin:02d}",
            "ByteEntropyHistogram flattened 16x16 joint histogram",
        )
    if group.name == "strings":
        if local_index < 3:
            return (
                ("string_count", "string_average_length", "printable_count")[local_index],
                "StringExtractor summary",
            )
        if local_index < 99:
            return (
                f"printable_character_frequency_{local_index - 3:02d}",
                "StringExtractor printable 0x20..0x7f normalized histogram",
            )
        if local_index == 99:
            return "printable_character_entropy", "StringExtractor printable entropy"
        return (
            f"sorted_regex_match_count_{local_index - 100:02d}",
            "StringExtractor lexicographically sorted regex-key count",
        )
    if group.name == "header":
        return HEADER_NAMES[local_index], "HeaderFileInfo.process_raw_features"
    if group.name == "section":
        if local_index < 11:
            return SECTION_GENERAL_NAMES[local_index], "SectionInfo general aggregate"
        if local_index < 61:
            return f"section_size_feature_hash_{local_index - 11:02d}", "FeatureHasher(pair)"
        if local_index < 111:
            return f"section_virtual_size_feature_hash_{local_index - 61:02d}", "FeatureHasher(pair)"
        if local_index < 161:
            return f"section_entropy_feature_hash_{local_index - 111:02d}", "FeatureHasher(pair)"
        if local_index < 211:
            return f"section_characteristic_feature_hash_{local_index - 161:02d}", "FeatureHasher(string)"
        if local_index < 221:
            return f"entry_section_feature_hash_{local_index - 211:02d}", "FeatureHasher(string)"
        return (
            ("overlay_size", "overlay_size_ratio", "overlay_entropy")[local_index - 221],
            "SectionInfo overlay aggregate",
        )
    if group.name == "imports":
        if local_index == 0:
            return "import_function_count", "ImportsInfo count before hash blocks"
        if local_index == 1:
            return "import_library_count", "ImportsInfo count before hash blocks"
        if local_index < 258:
            return f"import_library_feature_hash_{local_index - 2:04d}", "FeatureHasher(string)"
        return f"import_api_feature_hash_{local_index - 258:04d}", "FeatureHasher(string)"
    if group.name == "exports":
        if local_index == 0:
            return "export_hash_vector_length_sentinel", "ExportsInfo literal len(exports_hashed)"
        return f"export_feature_hash_{local_index - 1:03d}", "FeatureHasher(string)"
    if group.name == "datadirectories":
        if local_index < 32:
            directory_name = DATA_DIRECTORY_NAMES[local_index // 2]
            field = "size" if local_index % 2 == 0 else "virtual_address"
            return f"directory_{directory_name}_{field}", "DataDirectories.process_raw_features"
        if local_index == 32:
            return "has_relocations", "DataDirectories parser predicate"
        return "has_dynamic_relocations", "DataDirectories parser predicate"
    if group.name == "richheader":
        if local_index == 0:
            return "rich_pair_count", "RichHeader number_of_pairs"
        return f"rich_pair_feature_hash_{local_index - 1:02d}", "FeatureHasher(pair)"
    if group.name == "authenticode":
        return AUTHENTICODE_NAMES[local_index], "AuthenticodeSignature.process_raw_features"
    if group.name == "pefilewarnings":
        if local_index == 87:
            return "normalized_warning_count", "PEFormatWarnings normalized warning count"
        return f"normalized_warning_indicator_{local_index:02d}", "PEFormatWarnings warning vocabulary"
    raise ValueError(f"Unknown official group: {group.name}")


def canonical_official_columns() -> tuple[OfficialColumn, ...]:
    """Return every official index once, in executable extractor order."""

    columns: list[OfficialColumn] = []
    for group in OFFICIAL_GROUPS:
        for index in range(group.start, group.stop):
            name, source_semantics = _column_name_and_semantics(group, index - group.start)
            columns.append(OfficialColumn(index, group.name, name, source_semantics))
    if len(columns) != OFFICIAL_DIMENSION:
        raise RuntimeError("Official EMBER2024-v3 dimension drift")
    return tuple(columns)
