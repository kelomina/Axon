#!/usr/bin/env python3
"""Guard against training on external identity fields.

Paths, filenames, extensions, source hashes, split names, and row ids are
allowed for loading, joining, auditing, and manual review. They are not valid
model features because deployment filenames and paths are attacker-controlled
and usually differ from the training corpus.
"""

from __future__ import annotations

import re
from typing import Iterable


FORBIDDEN_IDENTITY_FEATURES = {
    "basename",
    "cache_path",
    "directory",
    "dir_name",
    "dirname",
    "extension",
    "file_extension",
    "file_hash",
    "file_name",
    "file_path",
    "file_sha256",
    "file_stem",
    "filename",
    "filename_hash",
    "hash",
    "manifest_source_hash",
    "manifest_source_sha256",
    "materialized_source_path",
    "original_source_path",
    "parent_dir",
    "path",
    "path_hint",
    "record_id",
    "record_index",
    "row_id",
    "row_index",
    "row_number",
    "row_order",
    "sample_id",
    "sample_index",
    "sha256",
    "source_hash",
    "source_path",
    "source_sha256",
    "split",
    "suffix",
}

FORBIDDEN_IDENTITY_PREFIXES = (
    "basename_",
    "cache_path_",
    "directory_",
    "dir_name_",
    "dirname_",
    "extension_",
    "ext_",
    "file_extension_",
    "file_hash_",
    "file_name_",
    "file_path_",
    "file_stem_",
    "filename_",
    "filename_hash_",
    "filepath_",
    "hash_",
    "manifest_source_",
    "materialized_source_",
    "original_source_",
    "parent_dir_",
    "path_",
    "record_id_",
    "record_index_",
    "row_id_",
    "row_index_",
    "row_number_",
    "row_order_",
    "sample_id_",
    "sample_index_",
    "sha256_",
    "source_hash_",
    "source_",
    "split_",
    "suffix_",
)


def normalize_feature_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return normalized.strip("_")


def identity_feature_violations(feature_names: Iterable[str]) -> list[str]:
    """Return feature names that look like external identity metadata."""

    violations = []
    for raw_name in feature_names:
        name = normalize_feature_name(raw_name)
        if not name:
            continue
        if name in FORBIDDEN_IDENTITY_FEATURES or name.startswith(FORBIDDEN_IDENTITY_PREFIXES):
            violations.append(str(raw_name))
    return violations


def assert_no_identity_feature_names(feature_names: Iterable[str], *, context: str) -> None:
    violations = identity_feature_violations(feature_names)
    if violations:
        preview = ", ".join(violations[:20])
        extra = "" if len(violations) <= 20 else f", ... (+{len(violations) - 20})"
        raise ValueError(
            f"{context} contains forbidden identity-derived model feature(s): {preview}{extra}. "
            "Use paths, filenames, extensions, hashes, split names, and row ids only for loading, "
            "joining, auditing, or manual review."
        )
