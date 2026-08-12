from __future__ import annotations

from pathlib import Path

from scripts.train_stage2_cache_matrix import (
    CONTENT_PE_V2_FEATURE_NAMES,
    CONTENT_STRING_FEATURE_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "tools" / "axon_onnx_dll" / "src" / "axon_loop151_content_features.h"
SOURCE = ROOT / "tools" / "axon_onnx_dll" / "src" / "axon_loop151_content_features.cpp"


def test_native_feature_dimensions_match_python_contract() -> None:
    header = HEADER.read_text(encoding="utf-8")
    assert len(CONTENT_PE_V2_FEATURE_NAMES) == 182
    assert len(CONTENT_STRING_FEATURE_NAMES) == 43
    assert "kContentPeV2FeatureDim = 182" in header
    assert "kContentStringFeatureDim = 43" in header
    assert "content_pe_v2_features" in header
    assert "content_string_features" in header


def test_native_content_source_preserves_order_and_fail_closed_semantics() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    required_fragments = (
        '"kernel32.dll"',
        '"wintrust.dll"',
        '"winverifytrust"',
        '"dllgetclassobject"',
        'executable_sections',
        'constexpr std::size_t head_size = 2 * 1024 * 1024',
        'constexpr std::size_t tail_size = 512 * 1024',
        'return zero_features(kContentPeV2FeatureDim);',
        'return zero_features(kContentStringFeatureDim);',
    )
    for fragment in required_fragments:
        assert fragment in source
    for forbidden in ("source_sha256", "sample_index", '"label"', '"split"'):
        assert forbidden not in source
