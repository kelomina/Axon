import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.main as main  # noqa: E402


def test_resolve_config_preserves_feature_cache_dir(tmp_path):
    config_path = tmp_path / "custom_cache.toml"
    config_path.write_text(
        """
[experiment]
name = "cache_dir_probe"

[data]
data_dir = "data"
cache_dir = "data/.cache_custom_probe"
pe_schema_version = "fixed_v2"
pe_feature_dim = 256
stat_feature_dim = 49
strict_pe_parsing = true
allow_pe_fallback = false

[model]
max_byte_length = 32768
dsra_dim = 160
dsra_heads = 4
""".strip(),
        encoding="utf-8",
    )

    config, _train_config = main._resolve_config(argparse.Namespace(config=str(config_path)))

    assert config.cache_dir == "data/.cache_custom_probe"
    assert main._cache_dir_from_config(config) == "data/.cache_custom_probe"
