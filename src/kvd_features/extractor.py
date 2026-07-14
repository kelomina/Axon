"""KVD 恶意软件特征提取器核心模块。

本模块提供完整的恶意软件特征提取功能，包括：
- 字节序列提取（固定长度）
- PE 结构特征提取（1500维）
- 统计特征提取（~100维）
- 轻量级哈希特征（256维）

特征维度：
- 字节序列：max_file_size (默认 65536)
- PE结构特征：1500
- 统计特征：~100
- 轻量级哈希：256
"""

import os
import hashlib
import struct
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np

try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False


@dataclass
class ExtractionConfig:
    """特征提取配置

    当 axon_config 提供时，可从 AxonExperimentConfig 自动同步参数。
    显式传入的参数优先级高于 axon_config 中的值。
    """
    max_file_size: int = None
    byte_histogram_bins: int = None
    stat_chunk_count: int = None
    stat_segment_count: int = None
    entropy_block_size: int = None
    entropy_sample_size: int = None
    size_norm_max: int = None
    timestamp_year_base: int = None
    timestamp_year_max: int = None
    large_trailing_data_size: int = None
    entropy_high_threshold: float = None
    section_entropy_min_size: int = None
    overlay_entropy_min_size: int = None
    pe_feature_dim: int = None
    stat_feature_dim: int = None
    lightweight_feature_dim: int = None
    ascii_printable_min: int = None
    ascii_printable_max: int = None
    allow_pe_fallback: bool = None
    pe_schema_version: str = None
    pe_fixed_section_slots: int = None

    def __post_init__(self):
        # Fallback defaults: only used when not created via from_axon_config().
        # Keep in sync with AxonExperimentConfig defaults.
        self.max_file_size = 65536 if self.max_file_size is None else self.max_file_size
        self.byte_histogram_bins = 256 if self.byte_histogram_bins is None else self.byte_histogram_bins
        self.stat_chunk_count = 10 if self.stat_chunk_count is None else self.stat_chunk_count
        self.stat_segment_count = 3 if self.stat_segment_count is None else self.stat_segment_count
        self.entropy_block_size = 4096 if self.entropy_block_size is None else self.entropy_block_size
        self.entropy_sample_size = 4096 if self.entropy_sample_size is None else self.entropy_sample_size
        self.size_norm_max = 100 * 1024 * 1024 if self.size_norm_max is None else self.size_norm_max
        self.timestamp_year_base = 1970 if self.timestamp_year_base is None else self.timestamp_year_base
        self.timestamp_year_max = 2099 if self.timestamp_year_max is None else self.timestamp_year_max
        self.large_trailing_data_size = 1024 * 1024 if self.large_trailing_data_size is None else self.large_trailing_data_size
        self.entropy_high_threshold = 0.8 if self.entropy_high_threshold is None else self.entropy_high_threshold
        self.section_entropy_min_size = 256 if self.section_entropy_min_size is None else self.section_entropy_min_size
        self.overlay_entropy_min_size = 256 if self.overlay_entropy_min_size is None else self.overlay_entropy_min_size
        self.pe_feature_dim = 1500 if self.pe_feature_dim is None else self.pe_feature_dim
        self.stat_feature_dim = 49 if self.stat_feature_dim is None else self.stat_feature_dim
        self.lightweight_feature_dim = 256 if self.lightweight_feature_dim is None else self.lightweight_feature_dim
        self.ascii_printable_min = 32 if self.ascii_printable_min is None else self.ascii_printable_min
        self.ascii_printable_max = 127 if self.ascii_printable_max is None else self.ascii_printable_max
        self.allow_pe_fallback = True if self.allow_pe_fallback is None else self.allow_pe_fallback
        self.pe_schema_version = "legacy_dynamic" if self.pe_schema_version is None else self.pe_schema_version
        self.pe_fixed_section_slots = 32 if self.pe_fixed_section_slots is None else self.pe_fixed_section_slots

    @classmethod
    def from_axon_config(cls, axon_config, **overrides) -> 'ExtractionConfig':
        """从 AxonExperimentConfig 创建 ExtractionConfig

        Args:
            axon_config: AxonExperimentConfig 实例
            **overrides: 显式覆盖的参数

        Returns:
            ExtractionConfig 实例
        """
        mapping = {
            'max_file_size': 'extraction_max_file_size',
            'byte_histogram_bins': 'byte_histogram_bins',
            'stat_chunk_count': 'stat_chunk_count',
            'stat_segment_count': 'stat_segment_count',
            'entropy_block_size': 'entropy_block_size',
            'entropy_sample_size': 'entropy_sample_size',
            'size_norm_max': 'size_norm_max',
            'timestamp_year_base': 'timestamp_year_base',
            'timestamp_year_max': 'timestamp_year_max',
            'large_trailing_data_size': 'large_trailing_data_size',
            'entropy_high_threshold': 'entropy_high_threshold',
            'section_entropy_min_size': 'section_entropy_min_size',
            'overlay_entropy_min_size': 'overlay_entropy_min_size',
            'pe_feature_dim': 'pe_feature_dim',
            'stat_feature_dim': 'stat_feature_dim',
            'lightweight_feature_dim': 'lightweight_feature_dim',
            'ascii_printable_min': 'ascii_printable_min',
            'ascii_printable_max': 'ascii_printable_max',
            'allow_pe_fallback': 'allow_pe_fallback',
            'pe_schema_version': 'pe_schema_version',
            'pe_fixed_section_slots': 'pe_fixed_section_slots',
        }
        kwargs = {}
        for our_key, axon_key in mapping.items():
            if our_key in overrides:
                kwargs[our_key] = overrides[our_key]
            elif hasattr(axon_config, axon_key):
                kwargs[our_key] = getattr(axon_config, axon_key)
        return cls(**kwargs)


# 特征名称列表（对应 PE_FEATURE_VECTOR_DIM）
FEATURE_NAMES = [
    'size', 'log_size', 'entropy', 'section_entropy_max', 'section_entropy_min',
    'section_entropy_avg', 'section_entropy_std', 'packed_sections_ratio',
    'sections_count', 'section_total_size', 'section_total_vsize',
    'avg_section_size', 'avg_section_vsize', 'min_section_size', 'max_section_size',
    'section_size_std', 'section_size_cv', 'section_names_count',
    'section_name_avg_length', 'section_name_max_length', 'section_name_min_length',
    'long_sections_count', 'long_sections_ratio', 'short_sections_count', 'short_sections_ratio',
    'executable_sections_ratio', 'writable_sections_ratio', 'readable_sections_ratio',
    'rwx_sections_ratio', 'rwx_sections_count', 'executable_code_density',
    'executable_writable_sections', 'non_standard_executable_sections_count',
    'non_standard_executable_sections_ratio', 'imports_count', 'unique_imports',
    'unique_dlls', 'import_ordinal_only_count', 'import_ordinal_only_ratio',
    'avg_imports_per_dll', 'imported_system_dlls_count', 'imported_system_dlls_ratio',
    'dll_name_avg_length', 'dll_name_max_length', 'dll_name_min_length',
    'dll_imports_entropy', 'api_imports_entropy', 'imports_per_section',
    'syscall_api_ratio', 'exports_count', 'exports_density', 'export_name_avg_length',
    'export_name_max_length', 'export_name_min_length', 'exports_name_ratio',
    'has_resources', 'resources_count', 'resource_types_count', 'pe_header_size',
    'header_size_ratio', 'subsystem', 'dll_characteristics', 'checksum',
    'checksum_zero_flag', 'has_aslr', 'has_nx_compat', 'has_guard_cf', 'has_seh',
    'has_debug_info', 'has_relocs', 'has_tls', 'has_exceptions', 'has_signature',
    'entry_point_ratio', 'entry_in_nonstandard_section_flag', 'trailing_data_size',
    'trailing_data_ratio', 'has_large_trailing_data', 'overlay_entropy',
    'overlay_high_entropy_flag', 'tls_callbacks_count', 'reloc_blocks_count',
    'reloc_entries_count', 'alignment_mismatch_count', 'alignment_mismatch_ratio',
    'api_network_ratio', 'api_process_ratio', 'api_filesystem_ratio',
    'api_registry_ratio', 'packer_keyword_hits_count', 'packer_keyword_hits_ratio',
]


def _read_file_prefix(file_path: str, max_bytes: int) -> bytes:
    """Read up to max_bytes using fixed chunks so static guards see bounded reads."""

    limit = max(0, int(max_bytes))
    data = bytearray()
    with open(file_path, 'rb') as f:
        while len(data) < limit:
            chunk = f.read(65536)
            if not chunk:
                break
            remaining = limit - len(data)
            data.extend(chunk[:remaining])
    return bytes(data)


def extract_byte_sequence(
    file_path: str, 
    max_file_size: int = 65536
) -> Tuple[Optional[np.ndarray], int]:
    """从文件中提取固定长度的字节序列。
    
    Args:
        file_path: 文件路径
        max_file_size: 最大读取字节数
        
    Returns:
        Tuple of (字节序列 numpy 数组, 原始文件长度)
        如果失败则返回 (None, 0)
    """
    try:
        orig_len = os.path.getsize(file_path)
        raw_bytes = _read_file_prefix(file_path, max_file_size)

        padded_sequence = np.zeros(max_file_size, dtype=np.uint8)
        read_len = len(raw_bytes)
        padded_sequence[:read_len] = np.frombuffer(raw_bytes, dtype=np.uint8)
        return padded_sequence, orig_len
            
    except Exception as e:
        print(f"[Error] Failed to extract byte sequence: {e}")
        return None, 0


def calculate_byte_entropy(
    byte_sequence: np.ndarray, 
    block_size: int = 4096
) -> float:
    """计算字节序列的熵值。
    
    Args:
        byte_sequence: 字节序列
        block_size: 块大小
        
    Returns:
        熵值（0-1之间）
    """
    if byte_sequence is None or len(byte_sequence) == 0:
        return 0.0
    
    hist = np.bincount(byte_sequence, minlength=256)
    prob = hist / len(byte_sequence)
    prob = prob[prob > 0]
    
    if len(prob) == 0:
        return 0.0
    
    entropy = -np.sum(prob * np.log2(prob)) / 8.0
    return float(entropy)


def extract_statistical_features(
    byte_sequence: np.ndarray,
    orig_length: Optional[int] = None,
    axon_config=None,
    config: Optional[ExtractionConfig] = None
) -> np.ndarray:
    """提取字节序列的统计特征。
    
    Args:
        byte_sequence: 字节序列 [max_file_size]
        orig_length: 原始文件长度
        axon_config: 可选的 AxonExperimentConfig
        config: 可选的 ExtractionConfig（优先于 axon_config）
        
    Returns:
        统计特征向量（纯统计特征，不含PE特征）
    """
    if orig_length is not None and orig_length >= 0:
        byte_array = np.asarray(byte_sequence[:orig_length], dtype=np.uint8)
    else:
        byte_array = np.asarray(byte_sequence, dtype=np.uint8)
    
    length = len(byte_array)
    features = []
    
    if config is not None:
        _stat_segment_count = config.stat_segment_count
        _stat_chunk_count = config.stat_chunk_count
        _ascii_min = config.ascii_printable_min
        _ascii_max = config.ascii_printable_max
    else:
        _stat_segment_count = getattr(axon_config, 'stat_segment_count', 3) if axon_config else 3
        _stat_chunk_count = getattr(axon_config, 'stat_chunk_count', 10) if axon_config else 10
        _ascii_min = getattr(axon_config, 'ascii_printable_min', 32) if axon_config else 32
        _ascii_max = getattr(axon_config, 'ascii_printable_max', 127) if axon_config else 127

    counts = np.bincount(byte_array, minlength=256) if length > 0 else np.zeros(256, dtype=np.int64)
    
    if length > 0:
        value_axis = np.arange(256, dtype=np.float64)
        weighted_sum = float(np.dot(counts, value_axis))
        mean_val = weighted_sum / float(length)
        weighted_sq_sum = float(np.dot(counts, value_axis * value_axis))
        variance = max(0.0, weighted_sq_sum / float(length) - mean_val * mean_val)
        std_val = float(np.sqrt(variance))
        
        nonzero_indices = np.flatnonzero(counts)
        min_val = float(nonzero_indices[0]) if nonzero_indices.size > 0 else 0.0
        max_val = float(nonzero_indices[-1]) if nonzero_indices.size > 0 else 0.0
        
        cdf = np.cumsum(counts)
        median_val = float(np.searchsorted(cdf, int(np.ceil(0.50 * length))))
        q25 = float(np.searchsorted(cdf, int(np.ceil(0.25 * length))))
        q75 = float(np.searchsorted(cdf, int(np.ceil(0.75 * length))))
    else:
        mean_val = std_val = min_val = max_val = median_val = q25 = q75 = 0.0
    
    features.extend([mean_val, std_val, min_val, max_val, median_val, q25, q75])
    
    features.extend([
        int(counts[0]),
        int(counts[255]),
        int(counts[0x90]),
        int(np.sum(counts[_ascii_min:_ascii_max])),
    ])
    
    p = counts.astype(np.float64) / float(length) if length > 0 else np.zeros_like(counts, dtype=np.float64)
    p = p[p > 0]
    entropy = float((-np.sum(p * np.log2(p)) / 8.0) if p.size > 0 else 0.0)
    features.append(entropy)
    
    segment_count = _stat_segment_count
    if length >= segment_count:
        seg_len = length // segment_count
        segments = [byte_array[i * seg_len:(i + 1) * seg_len if i < segment_count - 1 else length].copy() for i in range(segment_count)]
    else:
        segments = [byte_array.copy()] * segment_count
    
    for seg in segments:
        if len(seg) == 0:
            seg_mean = seg_std = seg_entropy = 0.0
        else:
            seg_mean = float(np.mean(seg))
            seg_std = float(np.std(seg))
            seg_counts = np.bincount(seg, minlength=256)
            seg_p = seg_counts.astype(np.float64) / float(len(seg))
            seg_p = seg_p[seg_p > 0]
            seg_entropy = float((-np.sum(seg_p * np.log2(seg_p)) / 8.0) if seg_p.size > 0 else 0.0)
        features.extend([seg_mean, seg_std, seg_entropy])
    
    chunk_count = _stat_chunk_count
    chunk_size = max(1, length // chunk_count)
    chunk_means = []
    chunk_stds = []
    
    for i in range(chunk_count):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < chunk_count - 1 else length
        chunk = byte_array[start_idx:end_idx]
        
        if len(chunk) > 0:
            chunk_means.append(float(np.mean(chunk)))
            chunk_stds.append(float(np.std(chunk)))
        else:
            chunk_means.append(0.0)
            chunk_stds.append(0.0)
    
    features.extend(chunk_means)
    features.extend(chunk_stds)
    
    # 块间差异统计
    chunk_means = np.array(chunk_means, dtype=np.float32)
    chunk_stds = np.array(chunk_stds, dtype=np.float32)
    
    if len(chunk_means) > 1:
        mean_diffs = np.diff(chunk_means)
        std_diffs = np.diff(chunk_stds)
        
        features.extend([
            float(np.mean(np.abs(mean_diffs))),
            float(np.std(mean_diffs)),
            float(np.max(mean_diffs)),
            float(np.min(mean_diffs)),
            float(np.mean(np.abs(std_diffs))),
            float(np.std(std_diffs)),
            float(np.max(std_diffs)),
            float(np.min(std_diffs)),
        ])
    else:
        features.extend([0.0] * 8)
    
    return np.array(features, dtype=np.float32)


class PEFeatureExtractor:
    """PE结构特征提取器。
    
    提取 1500 维的 PE 结构特征，包括：
    - 文件统计特征
    - PE Header 特征
    - Section 特征
    - 导入表特征
    - 导出表特征
    - 安全标志特征
    - 尾部数据特征
    - 资源特征
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None, axon_config=None):
        self.config = config or ExtractionConfig()
        self.axon_config = axon_config
        
        self.common_sections = set(self._config_value(
            axon_config, 'common_section_names',
            ['.text', '.data', '.rdata', '.rsrc', '.idata', '.edata',
             '.bss', '.reloc', '.tls', '.gfids', '.00cfg']
        ))
        
        self.system_dlls = set(self._config_value(
            axon_config, 'system_dlls',
            ['kernel32.dll', 'user32.dll', 'advapi32.dll', 'shell32.dll',
             'ole32.dll', 'oleaut32.dll', 'msvcrt.dll', 'ntdll.dll',
             'ws2_32.dll', 'wininet.dll', 'urlmon.dll', 'crypt32.dll',
             'secur32.dll', 'netapi32.dll', 'dnsapi.dll', 'iphlpapi.dll',
             'gdi32.dll', 'comdlg32.dll', 'comctl32.dll', 'shlwapi.dll',
             'version.dll', 'setupapi.dll', 'imm32.dll', 'midimap.dll',
             'msacm32.dll', 'ddraw.dll', 'dinput.dll', 'dsound.dll']
        ))
        
        self.packer_keywords = set(self._config_value(
            axon_config, 'packer_keywords',
            ['upx', 'aspack', 'petite', 'pecompact', 'themida', 'vmprotect',
             'enigma', 'obsidium', 'armadillo', 'safengine',
             'orion', 'execryptor', 'pelock', 'npack', 'nspack', 'wwpack',
             'diminuto', 'upack', 'kkrunchy', 'joexe', 'fsg',
             'stunnix', 'winlicense', 'packed']
        ))
        
        self.api_categories = self._config_value(
            axon_config, 'api_categories',
            {
                'network': ['internet', 'http', 'socket', 'connect', 'recv', 'send',
                            'url', 'download', 'upload', 'proxy', 'wsa', 'ftp', 'smtp'],
                'process': ['createprocess', 'openprocess', 'virtualalloc', 'virtualprotect',
                            'writeprocessmemory', 'readprocessmemory', 'createremotethread',
                            'shellexecute', 'winexec', 'loadlibrary', 'getprocaddress'],
                'filesystem': ['createfile', 'readfile', 'writefile', 'deletefile',
                               'movefile', 'copyfile', 'getfilesize', 'setfilepointer',
                               'findfirstfile', 'findnextfile', 'gettemppath'],
                'registry': ['regopenkey', 'regsetvalue', 'regcreatekey', 'regdeletekey',
                             'regqueryvalue', 'regclosekey', 'savekey', 'restorekey'],
                'crypto': ['cryptencrypt', 'cryptdecrypt', 'cryptderivekey', 'cryptgenkey',
                           'cryptcreatehash', 'crypthashdata', 'cryptsignhash', 'cryptverify'],
                'injection': ['createremotethread', 'virtualallocex', 'writeprocessmemory',
                              'readprocessmemory', 'queueuserapc', 'setwindowshookex',
                              'rtlcreateuserthread', 'ntcreatethreadex'],
            }
        )
        self._prefix_only = set(self._config_value(axon_config, 'api_prefix_only_keywords', ['connect', 'send', 'recv']))

    @staticmethod
    def _config_value(axon_config, attr_name: str, fallback):
        if axon_config is not None and hasattr(axon_config, attr_name):
            return getattr(axon_config, attr_name)
        return fallback
    
    def _safe_dll_name(self, dll_name: bytes) -> str:
        """安全处理 DLL 名称"""
        try:
            return dll_name.decode('utf-8', errors='ignore').lower().strip()
        except Exception:
            return ""
    
    def _safe_api_name(self, api_name: bytes) -> str:
        """安全处理 API 名称"""
        try:
            return api_name.decode('utf-8', errors='ignore').lower().strip()
        except Exception:
            return ""

    def _read_section_entropy_sample(self, section, raw_size: int) -> bytes:
        """Read only the bytes needed for section entropy instead of the full section."""
        section_data_max = (
            getattr(self.axon_config, 'section_data_max_size', 10 * 1024 * 1024)
            if self.axon_config
            else 10 * 1024 * 1024
        )
        if raw_size <= 0 or raw_size >= section_data_max:
            return b""

        sample_size = min(int(raw_size), int(self.config.section_entropy_min_size))
        if sample_size <= 0:
            return b""

        try:
            return section.get_data(length=sample_size)
        except TypeError:
            return b""

    def _collect_section_and_import_stats(self, pe):
        """收集 PE section、import 和 packer 统计，供不同 schema 共用。"""
        num_sections = pe.FILE_HEADER.NumberOfSections
        section_sizes = []
        section_entropies = []
        section_vsize = []
        section_flags = []
        section_names = []

        for section in pe.sections:
            section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
            raw_size = section.SizeOfRawData
            virt_size = section.Misc_VirtualSize

            section_names.append(section_name)
            section_sizes.append(raw_size)
            section_vsize.append(virt_size)

            section_data = self._read_section_entropy_sample(section, raw_size)
            if len(section_data) > 0:
                entropy = calculate_byte_entropy(np.frombuffer(section_data, dtype=np.uint8))
                section_entropies.append(entropy)

            chars = section.Characteristics
            section_flags.append((
                bool(chars & 0x20000000),
                bool(chars & 0x80000000),
                bool(chars & 0x40000000),
            ))

        total_apis = 0
        category_counts = {cat: 0 for cat in self.api_categories}
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        api_name = self._safe_api_name(imp.name)
                        if api_name:
                            total_apis += 1
                            for cat, keywords in self.api_categories.items():
                                if any(
                                    (api_name.startswith(kw) if kw in self._prefix_only else kw in api_name)
                                    for kw in keywords
                                ):
                                    category_counts[cat] += 1

        packer_hits = 0
        for section_name in section_names:
            if any(kw in section_name.lower() for kw in self.packer_keywords):
                packer_hits += 1

        return {
            "num_sections": num_sections,
            "section_sizes": section_sizes,
            "section_entropies": section_entropies,
            "section_vsize": section_vsize,
            "section_flags": section_flags,
            "section_names": section_names,
            "total_apis": total_apis,
            "category_counts": category_counts,
            "packer_hits": packer_hits,
        }

    def _write_pe_header_features(self, features: np.ndarray, pe, file_path: str) -> int:
        """写入两种 PE schema 共享的前 18 个稳定字段。"""
        idx = 0
        file_size = os.path.getsize(file_path)
        features[idx] = float(file_size); idx += 1
        features[idx] = np.log1p(float(file_size)); idx += 1

        features[idx] = pe.FILE_HEADER.SizeOfOptionalHeader; idx += 1
        header_size = pe.FILE_HEADER.SizeOfOptionalHeader + 24
        features[idx] = header_size / max(file_size, 1); idx += 1
        features[idx] = float(pe.OPTIONAL_HEADER.Subsystem); idx += 1
        features[idx] = float(pe.OPTIONAL_HEADER.DllCharacteristics); idx += 1
        features[idx] = float(pe.OPTIONAL_HEADER.CheckSum); idx += 1
        features[idx] = 1.0 if pe.OPTIONAL_HEADER.CheckSum == 0 else 0.0; idx += 1

        dll_chars = pe.OPTIONAL_HEADER.DllCharacteristics
        features[idx] = 1.0 if (dll_chars & 0x0040) else 0.0; idx += 1
        features[idx] = 1.0 if (dll_chars & 0x0080) else 0.0; idx += 1
        features[idx] = 1.0 if (dll_chars & 0x4000) else 0.0; idx += 1
        features[idx] = 1.0 if (pe.FILE_HEADER.Characteristics & 0x0004) else 0.0; idx += 1
        features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') else 0.0; idx += 1
        features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') else 0.0; idx += 1
        features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_TLS') else 0.0; idx += 1
        features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_EXCEPTION') else 0.0; idx += 1
        features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY') else 0.0; idx += 1
        features[idx] = float(pe.FILE_HEADER.NumberOfSections); idx += 1
        return idx

    def _write_pe_aggregate_features(self, features: np.ndarray, idx: int, stats: dict) -> int:
        """写入 section 聚合、API 类别和 packer 统计。"""
        section_entropies = stats["section_entropies"]
        section_sizes = stats["section_sizes"]
        section_vsize = stats["section_vsize"]
        section_names = stats["section_names"]
        num_sections = stats["num_sections"]

        if section_entropies:
            features[idx] = max(section_entropies); idx += 1
            features[idx] = min(section_entropies); idx += 1
            features[idx] = np.mean(section_entropies); idx += 1
            features[idx] = np.std(section_entropies); idx += 1
            high_entropy_ratio = sum(1 for e in section_entropies if e > self.config.entropy_high_threshold) / len(section_entropies)
            features[idx] = high_entropy_ratio; idx += 1
        else:
            features[idx:idx+5] = 0.0; idx += 5

        avg_raw = 0.0
        if section_sizes:
            total_raw = sum(section_sizes)
            total_vsize = sum(section_vsize)
            avg_raw = np.mean(section_sizes)
            avg_vsize = np.mean(section_vsize)

            features[idx] = float(total_raw); idx += 1
            features[idx] = float(total_vsize); idx += 1
            features[idx] = float(avg_raw); idx += 1
            features[idx] = float(avg_vsize); idx += 1
            features[idx] = float(min(section_sizes)); idx += 1
            features[idx] = float(max(section_sizes)); idx += 1
            features[idx] = float(np.std(section_sizes)); idx += 1
            features[idx] = float(np.std(section_sizes) / max(avg_raw, 1)); idx += 1
        else:
            features[idx:idx+8] = 0.0; idx += 8

        valid_names = [n for n in section_names if n]
        features[idx] = len(valid_names); idx += 1
        if valid_names:
            name_lens = [len(n) for n in valid_names]
            features[idx] = np.mean(name_lens); idx += 1
            features[idx] = max(name_lens); idx += 1
            features[idx] = min(name_lens); idx += 1
        else:
            features[idx:idx+3] = 0.0; idx += 3

        if section_sizes and avg_raw > 0:
            size_ratio = getattr(self.axon_config, 'section_size_ratio_threshold', 2.0) if self.axon_config else 2.0
            size_half_ratio = getattr(self.axon_config, 'section_size_half_ratio', 0.5) if self.axon_config else 0.5
            long_count = sum(1 for s in section_sizes if s > size_ratio * avg_raw)
            short_count = sum(1 for s in section_sizes if s < size_half_ratio * avg_raw)
            features[idx] = float(long_count); idx += 1
            features[idx] = float(long_count / len(section_sizes)); idx += 1
            features[idx] = float(short_count); idx += 1
            features[idx] = float(short_count / len(section_sizes)); idx += 1
        else:
            features[idx:idx+4] = 0.0; idx += 4

        total_apis = stats["total_apis"]
        category_counts = stats["category_counts"]
        for cat in ['network', 'process', 'filesystem', 'registry', 'crypto', 'injection']:
            features[idx] = category_counts[cat] / max(total_apis, 1); idx += 1

        packer_hits = stats["packer_hits"]
        features[idx] = float(packer_hits); idx += 1
        features[idx] = float(packer_hits) / max(num_sections, 1); idx += 1
        return idx

    def _extract_fixed_v2_features(self, pe, file_path: str) -> np.ndarray:
        """固定 PE schema：每一列语义固定，不随 section_count 偏移。"""
        used_dim = 18 + 3 * self.config.pe_fixed_section_slots + 29
        if self.config.pe_feature_dim < used_dim:
            raise ValueError(
                f"pe_feature_dim ({self.config.pe_feature_dim}) must be at least {used_dim} "
                f"for fixed_v2 PE schema"
            )

        features = np.zeros(self.config.pe_feature_dim, dtype=np.float32)
        idx = self._write_pe_header_features(features, pe, file_path)
        stats = self._collect_section_and_import_stats(pe)

        for slot in range(self.config.pe_fixed_section_slots):
            if slot < len(stats["section_flags"]):
                is_exec, is_write, is_read = stats["section_flags"][slot]
                features[idx] = 1.0 if is_exec else 0.0; idx += 1
                features[idx] = 1.0 if is_write else 0.0; idx += 1
                features[idx] = 1.0 if is_read else 0.0; idx += 1
            else:
                idx += 3

        self._write_pe_aggregate_features(features, idx, stats)
        return features
    
    def extract(self, file_path: str, allow_fallback: Optional[bool] = None) -> Optional[np.ndarray]:
        """提取 PE 文件特征。
        
        Args:
            file_path: PE 文件路径
            
        Returns:
            1500 维特征向量，失败返回 None
        """
        allow_fallback = self.config.allow_pe_fallback if allow_fallback is None else allow_fallback

        if not PEFILE_AVAILABLE:
            if not allow_fallback:
                return None
            return self._extract_fallback(file_path)

        try:
            pe = pefile.PE(file_path, fast_load=True)
        except Exception as e:
            if not allow_fallback:
                return None
            print(f"[Warning] PE load failed, using fallback features: {e}")
            return self._extract_fallback(file_path)
        
        fallback_after_close = False
        try:
            try:
                pe.parse_data_directories(directories=[
                    pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
                    pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DEBUG'],
                    pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_BASERELOC'],
                    pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_TLS'],
                    pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION'],
                    pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_SECURITY'],
                ])
            except Exception:
                pass
            if self.config.pe_schema_version == "fixed_v2":
                return self._extract_fixed_v2_features(pe, file_path)

            features = np.zeros(self.config.pe_feature_dim, dtype=np.float32)
            
            idx = 0
            
            # 1. 文件级统计特征
            file_size = os.path.getsize(file_path)
            features[idx] = float(file_size); idx += 1
            features[idx] = np.log1p(float(file_size)); idx += 1
            
            # 2. PE Header 特征
            features[idx] = pe.FILE_HEADER.SizeOfOptionalHeader; idx += 1
            header_size = pe.FILE_HEADER.SizeOfOptionalHeader + 24  # DOS + PE
            features[idx] = header_size / max(file_size, 1); idx += 1
            features[idx] = float(pe.OPTIONAL_HEADER.Subsystem); idx += 1
            features[idx] = float(pe.OPTIONAL_HEADER.DllCharacteristics); idx += 1
            features[idx] = float(pe.OPTIONAL_HEADER.CheckSum); idx += 1
            features[idx] = 1.0 if pe.OPTIONAL_HEADER.CheckSum == 0 else 0.0; idx += 1
            
            # 3. 安全标志
            dll_chars = pe.OPTIONAL_HEADER.DllCharacteristics
            features[idx] = 1.0 if (dll_chars & 0x0040) else 0.0; idx += 1  # ASLR
            features[idx] = 1.0 if (dll_chars & 0x0080) else 0.0; idx += 1  # NX
            features[idx] = 1.0 if (dll_chars & 0x4000) else 0.0; idx += 1  # CFG
            features[idx] = 1.0 if (pe.FILE_HEADER.Characteristics & 0x0004) else 0.0; idx += 1  # SEH
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') else 0.0; idx += 1
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') else 0.0; idx += 1
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_TLS') else 0.0; idx += 1
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_EXCEPTION') else 0.0; idx += 1
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY') else 0.0; idx += 1
            
            # 4. Section 特征
            num_sections = pe.FILE_HEADER.NumberOfSections
            features[idx] = float(num_sections); idx += 1
            
            section_sizes = []
            section_entropies = []
            section_vsize = []
            
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                raw_size = section.SizeOfRawData
                virt_size = section.Misc_VirtualSize
                
                section_sizes.append(raw_size)
                section_vsize.append(virt_size)
                
                # 计算 section 熵
                section_data = self._read_section_entropy_sample(section, raw_size)
                if len(section_data) > 0:
                    entropy = calculate_byte_entropy(np.frombuffer(section_data, dtype=np.uint8))
                    section_entropies.append(entropy)
                
                # Section 属性
                chars = section.Characteristics
                is_exec = bool(chars & 0x20000000)
                is_write = bool(chars & 0x80000000)
                is_read = bool(chars & 0x40000000)
                
                features[idx] = 1.0 if is_exec else 0.0; idx += 1
                features[idx] = 1.0 if is_write else 0.0; idx += 1
                features[idx] = 1.0 if is_read else 0.0; idx += 1
            
            # Section 统计
            if section_entropies:
                features[idx] = max(section_entropies); idx += 1
                features[idx] = min(section_entropies); idx += 1
                features[idx] = np.mean(section_entropies); idx += 1
                features[idx] = np.std(section_entropies); idx += 1
                high_entropy_ratio = sum(1 for e in section_entropies if e > self.config.entropy_high_threshold) / len(section_entropies)
                features[idx] = high_entropy_ratio; idx += 1
            else:
                features[idx:idx+5] = 0.0; idx += 5
            
            if section_sizes:
                total_raw = sum(section_sizes)
                total_vsize = sum(section_vsize)
                avg_raw = np.mean(section_sizes)
                avg_vsize = np.mean(section_vsize)
                
                features[idx] = float(total_raw); idx += 1
                features[idx] = float(total_vsize); idx += 1
                features[idx] = float(avg_raw); idx += 1
                features[idx] = float(avg_vsize); idx += 1
                features[idx] = float(min(section_sizes)); idx += 1
                features[idx] = float(max(section_sizes)); idx += 1
                features[idx] = float(np.std(section_sizes)); idx += 1
                features[idx] = float(np.std(section_sizes) / max(avg_raw, 1)); idx += 1
            else:
                features[idx:idx+8] = 0.0; idx += 8
            
            # Section 名称统计
            section_names = [s.Name.decode('utf-8', errors='ignore').strip('\x00') for s in pe.sections]
            valid_names = [n for n in section_names if n]
            features[idx] = len(valid_names); idx += 1
            if valid_names:
                name_lens = [len(n) for n in valid_names]
                features[idx] = np.mean(name_lens); idx += 1
                features[idx] = max(name_lens); idx += 1
                features[idx] = min(name_lens); idx += 1
            else:
                features[idx:idx+3] = 0.0; idx += 3
            
            # 大小异常 section
            if section_sizes and avg_raw > 0:
                size_ratio = getattr(self.axon_config, 'section_size_ratio_threshold', 2.0) if self.axon_config else 2.0
                size_half_ratio = getattr(self.axon_config, 'section_size_half_ratio', 0.5) if self.axon_config else 0.5
                long_count = sum(1 for s in section_sizes if s > size_ratio * avg_raw)
                short_count = sum(1 for s in section_sizes if s < size_half_ratio * avg_raw)
                features[idx] = float(long_count); idx += 1
                features[idx] = float(long_count / len(section_sizes)); idx += 1
                features[idx] = float(short_count); idx += 1
                features[idx] = float(short_count / len(section_sizes)); idx += 1
            else:
                features[idx:idx+4] = 0.0; idx += 4
            
            # API 类别特征
            total_apis = 0
            category_counts = {cat: 0 for cat in self.api_categories}
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = self._safe_dll_name(entry.dll)
                    for imp in entry.imports:
                        if imp.name:
                            api_name = self._safe_api_name(imp.name)
                            if api_name:
                                total_apis += 1
                                for cat, keywords in self.api_categories.items():
                                    if any(
                                        (api_name.startswith(kw) if kw in self._prefix_only else kw in api_name)
                                        for kw in keywords
                                    ):
                                        category_counts[cat] += 1

            for cat in ['network', 'process', 'filesystem', 'registry', 'crypto', 'injection']:
                features[idx] = category_counts[cat] / max(total_apis, 1); idx += 1

            packer_hits = 0
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00').lower()
                if any(kw in section_name for kw in self.packer_keywords):
                    packer_hits += 1
            features[idx] = float(packer_hits); idx += 1
            features[idx] = float(packer_hits) / max(num_sections, 1); idx += 1

            # 填充剩余维度
            while idx < self.config.pe_feature_dim:
                features[idx] = 0.0
                idx += 1
            
            return features
            
        except Exception as e:
            if not allow_fallback:
                return None
            print(f"[Warning] PE extraction failed, using fallback features: {e}")
            fallback_after_close = True
        finally:
            pe.close()

        if fallback_after_close:
            return self._extract_fallback(file_path)
        return None
    
    def _extract_fallback(self, file_path: str) -> Optional[np.ndarray]:
        """PE 提取失败时的降级处理"""
        try:
            fallback_max_size = getattr(self.axon_config, 'fallback_max_read_size', 1 * 1024 * 1024) if self.axon_config else 1 * 1024 * 1024
            data = _read_file_prefix(file_path, fallback_max_size)
            
            features = np.zeros(self.config.pe_feature_dim, dtype=np.float32)
            
            # 基本统计
            file_size = os.path.getsize(file_path)
            features[0] = float(file_size)
            features[1] = np.log1p(float(file_size))
            
            # 简单熵值
            byte_arr = np.frombuffer(data, dtype=np.uint8)
            features[2] = calculate_byte_entropy(byte_arr)

            data_lower = data.lower()
            category_counts = {cat: 0 for cat in self.api_categories}
            total_apis = 0
            for cat, keywords in self.api_categories.items():
                for keyword in keywords:
                    pattern = str(keyword).lower().encode()
                    count = data_lower.count(pattern)
                    if count > 0:
                        category_counts[cat] += count
                        total_apis += count

            idx = 84
            for cat in ['network', 'process', 'filesystem', 'registry', 'crypto', 'injection']:
                features[idx] = category_counts[cat] / max(total_apis, 1)
                idx += 1
            packer_hits = sum(1 for keyword in self.packer_keywords if str(keyword).lower().encode() in data_lower)
            features[idx] = float(packer_hits)
            features[idx + 1] = float(packer_hits) / max(len(self.packer_keywords), 1)
            
            return features
        except Exception:
            return None


def extract_lightweight_features(
    file_path: str,
    feature_dim: int = 256,
    axon_config=None
) -> np.ndarray:
    """提取轻量级哈希特征。
    
    基于 DLL 名称、API 函数名和 Section 名称的哈希映射。
    
    Args:
        file_path: 文件路径
        feature_dim: 特征维度（默认 256）
        
    Returns:
        256 维二值特征向量
    """
    features = np.zeros(feature_dim, dtype=np.float32)
    
    try:
        read_size = getattr(axon_config, 'lightweight_read_size', 65536) if axon_config else 65536
        data = _read_file_prefix(file_path, read_size)
        
        data_lower = data.lower()
        
        dll_patterns = [
            p.encode() for p in (
                getattr(axon_config, 'lightweight_dll_patterns', None) if axon_config else None
                or ['kernel32.dll', 'user32.dll', 'ntdll.dll', 'advapi32.dll',
                    'ws2_32.dll', 'wininet.dll', 'ole32.dll', 'shell32.dll',
                    'msvcrt.dll', 'msvcrtd.dll', 'vcruntime.dll', 'ucrtbase.dll']
            )
        ]
        
        hash_mod = getattr(axon_config, 'lightweight_hash_mod', 128) if axon_config else 128
        hash_offset = getattr(axon_config, 'lightweight_hash_offset', 128) if axon_config else 128
        section_hash_mod = getattr(axon_config, 'lightweight_section_hash_mod', 32) if axon_config else 32
        section_hash_offset = getattr(axon_config, 'lightweight_section_hash_offset', 224) if axon_config else 224

        for pattern in dll_patterns:
            if pattern in data_lower:
                h = hashlib.sha256(pattern).digest()[0]
                features[h % hash_mod] = 1
        
        api_patterns = [
            str(p).lower().encode() for p in (
                getattr(axon_config, 'lightweight_api_patterns', None) if axon_config else None
                or ['VirtualAlloc', 'VirtualProtect', 'CreateRemoteThread',
                    'WriteProcessMemory', 'ReadProcessMemory', 'WinExec',
                    'ShellExecute', 'LoadLibrary', 'GetProcAddress',
                    'CreateProcess', 'InternetOpen', 'InternetReadFile',
                    'URLDownloadToFile', 'CreateFile', 'RegOpenKey']
            )
        ]
        
        for pattern in api_patterns:
            if pattern in data_lower:
                h = hashlib.sha256(pattern).digest()[0]
                features[hash_offset + (h % hash_mod)] = 1
        
        section_patterns = [
            p.encode() for p in (
                getattr(axon_config, 'lightweight_section_patterns', None) if axon_config else None
                or ['.text', '.data', '.rdata', '.rsrc', '.reloc',
                    '.code', '.idata', '.edata', '.tls', '.bss']
            )
        ]
        
        for pattern in section_patterns:
            if pattern in data_lower:
                h = hashlib.sha256(pattern).digest()[0]
                features[(h % section_hash_mod) + section_hash_offset] = 1
        
        # L2 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
            
    except Exception as e:
        print(f"[Error] Lightweight feature extraction failed: {e}")
    
    return features


def extract_pe_features(
    file_path: str,
    config: Optional[ExtractionConfig] = None,
    axon_config=None,
    allow_fallback: Optional[bool] = None,
) -> Optional[np.ndarray]:
    """提取 PE 结构特征。
    
    Args:
        file_path: 文件路径
        axon_config: 可选的 AxonExperimentConfig
        
    Returns:
        1500 维特征向量
    """
    extractor = PEFeatureExtractor(config=config, axon_config=axon_config)
    return extractor.extract(file_path, allow_fallback=allow_fallback)


def extract_all_features(
    file_path: str,
    config: Optional[ExtractionConfig] = None,
    axon_config=None,
    allow_pe_fallback: Optional[bool] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """提取所有特征。
    
    Args:
        file_path: 文件路径
        config: 提取配置
        axon_config: 可选的 AxonExperimentConfig
        
    Returns:
        Tuple of (字节序列, PE特征, 统计特征, 轻量级特征, 原始长度)
    """
    config = config or ExtractionConfig()
    
    byte_seq, orig_len = extract_byte_sequence(file_path, config.max_file_size)
    if byte_seq is None:
        return None, None, None, None, 0
    
    pe_features = extract_pe_features(
        file_path,
        config=config,
        axon_config=axon_config,
        allow_fallback=allow_pe_fallback,
    )
    if pe_features is None:
        return None, None, None, None, 0

    stat_features = extract_statistical_features(byte_seq, orig_len, axon_config=axon_config, config=config)
    
    lightweight_features = extract_lightweight_features(
        file_path, feature_dim=config.lightweight_feature_dim, axon_config=axon_config
    )
    
    return byte_seq, pe_features, stat_features, lightweight_features, orig_len


# 兼容性别名
FeatureExtractor = PEFeatureExtractor
