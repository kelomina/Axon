#!/usr/bin/env python3
"""重建8192字节cache

从data目录随机抽取20000个样本（良性10k + 恶意10k），
用max_byte_length=8192提取特征并保存到cache。
"""

import sys
import random
from pathlib import Path

# 添加src目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import MalwareDataset
from config import AxonExperimentConfig


def main():
    print("="*80)
    print("重建8192字节cache")
    print("="*80)
    
    # 加载配置
    import tomllib
    import dataclasses
    
    config_path = PROJECT_ROOT / "config" / "default_config.toml"
    with open(config_path, 'rb') as f:
        toml_data = tomllib.load(f)
    
    # 合并配置
    merged = {}
    for section in toml_data.values():
        if isinstance(section, dict):
            merged.update(section)
    
    field_names = {field.name for field in dataclasses.fields(AxonExperimentConfig)}
    config = AxonExperimentConfig(**{k: v for k, v in merged.items() if k in field_names})
    
    # 覆盖max_byte_length为8192
    config.max_byte_length = 8192
    print(f"✓ max_byte_length = {config.max_byte_length}")
    
    # 数据目录
    data_dir = PROJECT_ROOT / "data"
    cache_dir = data_dir / ".cache"
    
    print(f"\n步骤1: 扫描data目录")
    print("-"*80)
    
    # 创建MalwareDataset（会自动扫描并提取特征）
    # max_samples_per_class=10000 限制每类10000个样本
    dataset = MalwareDataset(
        data_dir=str(data_dir),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=10000,  # 每类10000个
        max_file_size=config.max_file_size,
        axon_config=config,
        extraction_workers=4,  # 4个进程并行提取
        extraction_backend='thread',  # 使用线程池（避免内存泄漏）
    )
    
    print(f"\n✓ 数据集创建完成:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  Cache目录: {cache_dir}")
    
    # 验证cache文件数量
    cache_files = list(cache_dir.glob("*.npz"))
    print(f"  Cache文件数: {len(cache_files)}")
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)


if __name__ == "__main__":
    main()
