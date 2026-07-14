"""Loop164 whole-file byte expert primitives.

The package intentionally contains no filesystem, cache, split, or identity
handling. Authorized input and experiment orchestration live in separate
modules so this core can be verified entirely with in-memory token sources.
"""

from .whole_file_gcg import (
    PAD_TOKEN,
    RAW_BYTE_OFFSET,
    VOCAB_SIZE,
    InMemoryByteSource,
    OutputChunk,
    OutputPartition,
    WholeFileByteSource,
    WholeFileGCGClassifier,
    WinnerSelection,
    encode_raw_bytes,
    output_partitions,
    padded_length_for_valid_length,
)

__all__ = [
    "PAD_TOKEN",
    "RAW_BYTE_OFFSET",
    "VOCAB_SIZE",
    "InMemoryByteSource",
    "OutputChunk",
    "OutputPartition",
    "WholeFileByteSource",
    "WholeFileGCGClassifier",
    "WinnerSelection",
    "encode_raw_bytes",
    "output_partitions",
    "padded_length_for_valid_length",
]
