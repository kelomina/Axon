"""Loop167 EMBER-v3 semantic-delta source closure primitives."""

from .ember_v3_native import NativeNovelDelta, extract_novel_delta
from .semantic_mapping import (
    CATEGORY_EXACT,
    CATEGORY_FORBIDDEN,
    CATEGORY_NOVEL,
    CATEGORY_PARTIAL,
    build_semantic_delta_mapping,
)

__all__ = [
    "CATEGORY_EXACT",
    "CATEGORY_FORBIDDEN",
    "CATEGORY_NOVEL",
    "CATEGORY_PARTIAL",
    "NativeNovelDelta",
    "build_semantic_delta_mapping",
    "extract_novel_delta",
]
