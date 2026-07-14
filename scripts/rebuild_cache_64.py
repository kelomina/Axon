"""单独构建64字节cache"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tomllib
import dataclasses
from config import AxonExperimentConfig
from dataset import MalwareDataset

# 加载配置
config_path = PROJECT_ROOT / "config" / "default_config.toml"
with open(config_path, 'rb') as f:
    toml_data = tomllib.load(f)

merged = {}
for section in toml_data.values():
    if isinstance(section, dict):
        merged.update(section)

field_names = {field.name for field in dataclasses.fields(AxonExperimentConfig)}
config = AxonExperimentConfig(**{k: v for k, v in merged.items() if k in field_names})
config.max_byte_length = 64

# 删除旧的64字节manifest
import os
cache_dir = PROJECT_ROOT / "data" / ".cache"
for f in cache_dir.glob("manifest_*.json"):
    content = f.read_text()
    if '"max_byte_length": 64' in content:
        print(f"删除旧manifest: {f.name}")
        f.unlink()
        break

# 构建新cache
print("构建64字节cache...")
dataset = MalwareDataset(
    data_dir=str(PROJECT_ROOT / "data"),
    max_byte_length=64,
    pe_feature_dim=config.pe_feature_dim,
    stat_feature_dim=config.stat_feature_dim,
    max_samples_per_class=10000,
    max_file_size=config.max_file_size,
    axon_config=config,
    extraction_workers=4,
    extraction_backend='thread',
)
print(f"构建了 {len(dataset)} 个样本")
