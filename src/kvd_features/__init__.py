"""KVD 特征提取器包初始化模块。

本模块提供恶意软件检测所需的特征提取功能，包括：
- 字节序列提取
- PE 结构特征提取
- 统计特征提取
- 轻量级哈希特征

使用方法：
    from kvd_features.extractor import extract_all_features
    
    byte_seq, pe_features, stat_features, lightweight_features, orig_len = extract_all_features(file_path)
"""

from .extractor import (
    extract_all_features,
    extract_byte_sequence,
    extract_pe_features,
    extract_statistical_features,
    extract_lightweight_features,
    PEFeatureExtractor,
    ExtractionConfig,
)

__all__ = [
    'extract_all_features',
    'extract_byte_sequence',
    'extract_pe_features',
    'extract_statistical_features',
    'extract_lightweight_features',
    'PEFeatureExtractor',
    'ExtractionConfig',
]
