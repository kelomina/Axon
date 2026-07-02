import pytest

from scripts import build_content_pe_feature_cache


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
