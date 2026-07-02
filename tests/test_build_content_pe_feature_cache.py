import pytest
import numpy as np

from scripts import build_content_pe_feature_cache
from kvd_features.content_pe_v1 import CONTENT_PE_FEATURE_NAMES


def test_content_pe_cache_limit_requires_smoke(tmp_path):
    with pytest.raises(ValueError, match="--limit requires --smoke"):
        build_content_pe_feature_cache.main(
            [
                "--predictions",
                str(tmp_path / "missing.csv"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--limit",
                "1",
                "--output-json",
                str(tmp_path / "report.json"),
            ]
        )


def test_content_pe_cache_negative_limit_rejected(tmp_path):
    with pytest.raises(ValueError, match="--limit must be non-negative"):
        build_content_pe_feature_cache.main(
            [
                "--predictions",
                str(tmp_path / "missing.csv"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--smoke",
                "--limit",
                "-1",
                "--output-json",
                str(tmp_path / "report.json"),
            ]
        )


def test_load_valid_cached_features_rejects_bad_shape(tmp_path):
    cache_path = tmp_path / "bad.npz"
    np.savez(cache_path, features=np.zeros(3, dtype=np.float32))

    assert build_content_pe_feature_cache._load_valid_cached_features(cache_path) is None


def test_load_valid_cached_features_accepts_content_pe_v1_shape(tmp_path):
    cache_path = tmp_path / "good.npz"
    features = np.ones(len(CONTENT_PE_FEATURE_NAMES), dtype=np.float32)
    np.savez(cache_path, features=features)

    loaded = build_content_pe_feature_cache._load_valid_cached_features(cache_path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, features)
