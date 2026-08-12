"""Axon v2.6 数据集和加载器模块。

提供恶意软件检测的数据集类和批量加载器。
"""

import os
import sys
import csv
import hashlib
import json
import threading
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Iterable, Iterator, Tuple, Optional, List, Dict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from ..kvd_features.extractor import (
        extract_all_features,
        ExtractionConfig
    )
except ImportError:
    from kvd_features.extractor import (
        extract_all_features,
        ExtractionConfig
    )


def _feature_cache_signature(
    max_byte_length: int,
    stat_feature_dim: int,
    pe_feature_dim: int,
    lightweight_feature_dim: int,
    strict_pe_parsing: bool,
    allow_pe_fallback: bool,
    pe_schema_version: str = "legacy_dynamic",
    pe_fixed_section_slots: int = 32,
) -> str:
    """生成特征缓存的配置签名。"""
    if pe_schema_version == "legacy_dynamic":
        return (
            f"{max_byte_length}_{stat_feature_dim}_{pe_feature_dim}_"
            f"{lightweight_feature_dim}_{strict_pe_parsing}_{allow_pe_fallback}"
        )
    return (
        f"{max_byte_length}_{stat_feature_dim}_{pe_feature_dim}_"
        f"{lightweight_feature_dim}_{strict_pe_parsing}_{allow_pe_fallback}_"
        f"{pe_schema_version}_{pe_fixed_section_slots}"
    )


def _feature_cache_hash(
    max_byte_length: int,
    stat_feature_dim: int,
    pe_feature_dim: int,
    lightweight_feature_dim: int,
    strict_pe_parsing: bool,
    allow_pe_fallback: bool,
    pe_schema_version: str = "legacy_dynamic",
    pe_fixed_section_slots: int = 32,
) -> str:
    """生成特征缓存文件名中的配置哈希。"""
    signature = _feature_cache_signature(
        max_byte_length,
        stat_feature_dim,
        pe_feature_dim,
        lightweight_feature_dim,
        strict_pe_parsing,
        allow_pe_fallback,
        pe_schema_version,
        pe_fixed_section_slots,
    )
    return hashlib.md5(signature.encode()).hexdigest()[:8]


def _is_relative_to_path(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _resolve_manifest_cache_path(cache_path_text: str, cache_dir: Path) -> Path:
    """Resolve a manifest cache path and require it to stay inside cache_dir."""
    if not cache_path_text:
        raise ValueError("Manifest cache_path is empty")

    raw_path = Path(cache_path_text)
    candidates = [raw_path] if raw_path.is_absolute() else [
        Path.cwd() / raw_path,
        cache_dir / raw_path,
        cache_dir.parent.parent / raw_path,
    ]
    cache_root = cache_dir.resolve(strict=False)
    inside_missing = None
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if _is_relative_to_path(resolved, cache_root):
            if resolved.exists():
                return resolved
            inside_missing = resolved
    if inside_missing is not None:
        raise FileNotFoundError(f"Manifest cache file not found: {inside_missing}")
    raise ValueError(f"Cache path is outside cache directory: {cache_path_text}")


def _manifest_cache_path_fast(cache_path_text: str, cache_dir: Path) -> Path:
    """Fast path for flat feature-cache manifest entries.

    Feature cache manifests store cache_path values such as
    data/.cache/<hash>_<config>.npz. For cache-only evaluation we only need to
    force the file back under cache_dir; the actual NPZ content is validated
    when the sample is read.
    """
    if not cache_path_text:
        raise ValueError("Manifest cache_path is empty")
    raw_path = Path(cache_path_text)
    if ".." in raw_path.parts or raw_path.suffix.lower() != ".npz" or not raw_path.name:
        return _resolve_manifest_cache_path(cache_path_text, cache_dir)
    return cache_dir / raw_path.name


def _resolve_source_path(source_path_text: str) -> Path:
    path = Path(source_path_text)
    return path if path.is_absolute() else Path.cwd() / path


def _file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid_source_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _safe_file_sha256(file_path: Path) -> Optional[str]:
    try:
        return _file_sha256(file_path)
    except (OSError, TypeError):
        return None


def _npz_scalar_to_text(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def _normalize_cached_arrays(
    byte_seq: np.ndarray,
    pe_feat: np.ndarray,
    stat_feat: np.ndarray,
    lightweight_feat: np.ndarray,
    max_byte_length: int,
    pe_feature_dim: int,
    stat_feature_dim: int,
    lightweight_feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把缓存里的数组整理成模型固定输入长度。"""
    if len(byte_seq) > max_byte_length:
        byte_seq = byte_seq[:max_byte_length]
    elif len(byte_seq) < max_byte_length:
        byte_seq = np.pad(byte_seq, (0, max_byte_length - len(byte_seq)))

    if len(pe_feat) < pe_feature_dim:
        pe_feat = np.pad(pe_feat, (0, pe_feature_dim - len(pe_feat)))
    elif len(pe_feat) > pe_feature_dim:
        pe_feat = pe_feat[:pe_feature_dim]

    if len(stat_feat) < stat_feature_dim:
        stat_feat = np.pad(stat_feat, (0, stat_feature_dim - len(stat_feat)))
    elif len(stat_feat) > stat_feature_dim:
        stat_feat = stat_feat[:stat_feature_dim]

    if len(lightweight_feat) < lightweight_feature_dim:
        lightweight_feat = np.pad(lightweight_feat, (0, lightweight_feature_dim - len(lightweight_feat)))
    elif len(lightweight_feat) > lightweight_feature_dim:
        lightweight_feat = lightweight_feat[:lightweight_feature_dim]

    return (
        byte_seq.astype(np.uint8, copy=False),
        pe_feat.astype(np.float32, copy=False),
        stat_feat.astype(np.float32, copy=False),
        lightweight_feat.astype(np.float32, copy=False),
    )


def _load_cached_feature_npz(
    cache_path: Path,
    max_byte_length: int,
    pe_feature_dim: int,
    stat_feature_dim: int,
    lightweight_feature_dim: int,
    expected_label: Optional[int] = None,
    expected_source_sha256: Optional[str] = None,
    allow_missing_source_sha256: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """读取单个缓存样本；缓存不完整时直接抛错。"""
    with np.load(cache_path, allow_pickle=False) as data:
        required_fields = {"byte_sequence", "pe_features", "label"}
        missing_fields = sorted(required_fields - set(data.files))
        if missing_fields:
            raise ValueError(f"Cache missing required fields {missing_fields}: {cache_path}")
        byte_seq = data['byte_sequence']
        pe_features = data['pe_features']
        stat_features = data.get('stat_features', np.zeros(stat_feature_dim, dtype=np.float32))
        lightweight_features = data.get('lightweight_features', np.zeros(lightweight_feature_dim, dtype=np.float32))
        label = int(data['label'])
        if label not in {0, 1}:
            raise ValueError(f"Cache label must be 0 or 1: {cache_path}")
        if expected_label is not None and label != int(expected_label):
            raise ValueError(
                f"Cache label mismatch for {cache_path}: expected {expected_label}, got {label}"
            )
        if expected_source_sha256:
            if 'source_sha256' not in data.files:
                if allow_missing_source_sha256:
                    cached_sha = None
                else:
                    raise ValueError(f"Cache missing source SHA for {cache_path}")
            else:
                cached_sha = _npz_scalar_to_text(data['source_sha256'])
            if cached_sha is not None and cached_sha != expected_source_sha256:
                raise ValueError(f"Cache source SHA mismatch for {cache_path}")
    byte_seq, pe_features, stat_features, lightweight_features = _normalize_cached_arrays(
        byte_seq,
        pe_features,
        stat_features,
        lightweight_features,
        max_byte_length,
        pe_feature_dim,
        stat_feature_dim,
        lightweight_feature_dim,
    )
    return byte_seq, pe_features, stat_features, lightweight_features, label


def _load_cache_metadata(cache_path: Path) -> Optional[Dict]:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            required_fields = {"byte_sequence", "pe_features", "label"}
            if required_fields - set(data.files):
                return None
            label = int(data['label'])
            if label not in {0, 1}:
                return None
            source_sha256 = (
                _npz_scalar_to_text(data['source_sha256']).strip().casefold()
                if 'source_sha256' in data.files
                else None
            )
            return {"label": label, "source_sha256": source_sha256}
    except Exception:
        return None


def _load_cache_label(cache_path: Path) -> Optional[int]:
    """只读取缓存里的标签，用于快速判断缓存是否可复用。

    npz 文件像一个小压缩包。完整读取 byte_sequence/pe_features 会解压大数组；
    这里只看 label，相当于只看包裹外的小标签，速度会快很多。
    """
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if 'label' not in data.files:
                return None
            return int(data['label'])
    except Exception:
        return None


def _is_valid_pe_sample_path(file_path: Path, max_file_size: int) -> bool:
    """检查文件是否像一个可处理的 PE 样本。"""
    try:
        file_size = file_path.stat().st_size
    except OSError:
        return False
    if file_size == 0 or file_size > max_file_size:
        return False
    try:
        with open(file_path, 'rb') as f:
            return f.read(2) == b'MZ'
    except Exception:
        return False


def _feature_cache_path_for_file(
    file_path: Path,
    cache_dir: Path,
    cache_config_hash: str,
    stat_result=None,
) -> Path:
    """按原始文件指纹和配置哈希生成缓存路径。"""
    try:
        stat = stat_result if stat_result is not None else file_path.stat()
        file_sig = f"{stat.st_size}_{int(stat.st_mtime_ns)}"
    except OSError:
        file_sig = "missing"
    file_hash = hashlib.md5(f"{file_path.resolve()}_{file_sig}".encode()).hexdigest()
    return cache_dir / f"{file_hash}_{cache_config_hash}.npz"


def _prepare_sample_cache_worker(payload: Dict) -> Dict:
    """在线程或进程 worker 中准备单个样本缓存。

    返回状态而不是直接改 Dataset 计数器，这样进程池也能安全工作。
    """
    file_path = Path(payload['file_path'])
    label = int(payload['label'])
    cache_dir = Path(payload['cache_dir'])

    try:
        stat_result = file_path.stat()
    except OSError:
        return {'status': 'non_pe_skipped', 'cache_path': None}

    cache_path = _feature_cache_path_for_file(
        file_path, cache_dir, payload['cache_config_hash'], stat_result=stat_result
    )

    max_file_size = int(payload['max_file_size'])
    if stat_result.st_size == 0 or stat_result.st_size > max_file_size:
        return {'status': 'non_pe_skipped', 'cache_path': None}
    try:
        with open(file_path, 'rb') as handle:
            if handle.read(2) != b'MZ':
                return {'status': 'non_pe_skipped', 'cache_path': None}
    except Exception:
        return {'status': 'non_pe_skipped', 'cache_path': None}

    trusted_sha = payload.get('trust_source_sha256')
    if trusted_sha and _is_valid_source_sha256(trusted_sha):
        source_sha256 = str(trusted_sha).strip().casefold()
    else:
        source_sha256 = _file_sha256(file_path)

    if payload['use_cache']:
        cached_meta = _load_cache_metadata(cache_path)
        if cached_meta is not None:
            if cached_meta.get("label") == label and cached_meta.get("source_sha256") == source_sha256:
                return {'status': 'cache_hits', 'cache_path': str(cache_path)}
            return {'status': 'other_failed_skipped', 'cache_path': None}

    try:
        byte_seq, pe_feat, stat_feat, lightweight_feat, _orig_len = extract_all_features(
            str(file_path),
            payload['extraction_config'],
            axon_config=payload['axon_config'],
            allow_pe_fallback=payload['allow_pe_fallback'],
        )
        if byte_seq is None or pe_feat is None:
            return {'status': 'pe_parse_failed_skipped', 'cache_path': None}

        byte_seq, pe_feat, stat_feat, lightweight_feat = _normalize_cached_arrays(
            byte_seq,
            pe_feat,
            stat_feat,
            lightweight_feat,
            int(payload['max_byte_length']),
            int(payload['pe_feature_dim']),
            int(payload['stat_feature_dim']),
            int(payload['lightweight_feature_dim']),
        )

        if payload['use_cache']:
            np.savez_compressed(
                cache_path,
                byte_sequence=byte_seq,
                pe_features=pe_feat,
                stat_features=stat_feat,
                lightweight_features=lightweight_feat,
                label=label,
                source_sha256=source_sha256,
            )
        return {'status': 'extracted', 'cache_path': str(cache_path), 'source_sha256': source_sha256}
    except Exception as e:
        return {
            'status': 'other_failed_skipped',
            'cache_path': None,
            'warning': f"{file_path} ({e})",
        }


def _limit_samples_per_class(samples: Iterable[Dict], max_samples_per_class: Optional[int]) -> List[Dict]:
    """按标签限制样本数，None 表示不限制。"""
    if max_samples_per_class is None:
        return list(samples)
    counts = {}
    limited = []
    for sample in samples:
        label = int(sample['label'])
        if counts.get(label, 0) >= max_samples_per_class:
            continue
        limited.append(sample)
        counts[label] = counts.get(label, 0) + 1
    return limited


def _iter_manifest_sample_entries(
    manifest_path: Path,
    *,
    max_sample_chars: int = 4 * 1024 * 1024,
) -> Iterator[Dict[str, Any]]:
    """Stream ``samples`` entries from a cache manifest without json.load().

    20w 样本的 manifest 不算巨型文件，但 ``json.load`` 会先把整棵 JSON 树
    变成 Python 对象；再被 Dataset 转成索引时又复制一遍。这里按对象流式解析
    ``samples`` 数组，只把当前样本留在内存里。
    """
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    buffer = ""
    found_samples = False
    eof = False

    with manifest_path.open("r", encoding="utf-8") as f:
        while True:
            if not eof and (not found_samples or len(buffer) < chunk_size):
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            if not found_samples:
                key_index = buffer.find('"samples"')
                if key_index < 0:
                    if eof:
                        return
                    buffer = buffer[-16:]
                    continue
                colon_index = buffer.find(":", key_index)
                bracket_index = buffer.find("[", colon_index + 1 if colon_index >= 0 else key_index)
                if colon_index < 0 or bracket_index < 0:
                    if eof:
                        raise ValueError(f"Manifest samples array is malformed: {manifest_path}")
                    buffer = buffer[key_index:]
                    continue
                buffer = buffer[bracket_index + 1:]
                found_samples = True

            while True:
                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                try:
                    sample, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    if len(buffer) >= max_sample_chars:
                        raise ValueError(
                            "Manifest sample entry is too large or malformed "
                            f"(>{max_sample_chars} chars): {manifest_path}"
                        )
                    chunk = f.read(chunk_size)
                    if chunk:
                        buffer += chunk
                        continue
                    eof = True
                    raise
                if isinstance(sample, dict):
                    yield sample
                buffer = buffer[end:]


def _write_cache_manifest_stream(
    manifest_path: Path,
    header: Dict[str, Any],
    samples: Iterable[Dict[str, Any]],
) -> None:
    """Write a manifest without materializing the full samples list."""
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write("{")
        first_field = True
        for key, value in header.items():
            if key == "samples":
                continue
            if not first_field:
                f.write(",")
            json.dump(str(key), f, ensure_ascii=False)
            f.write(":")
            json.dump(value, f, ensure_ascii=False)
            first_field = False
        if not first_field:
            f.write(",")
        f.write('"samples":[')
        first_sample = True
        for sample in samples:
            if not first_sample:
                f.write(",")
            json.dump(sample, f, ensure_ascii=False)
            first_sample = False
        f.write("]}")
    tmp_path.replace(manifest_path)


class _ColumnarSampleView:
    """List-like view over FeatureCacheDataset's columnar index.

    外部脚本仍可用 ``dataset.samples[i]`` 拿到 dict，但底层不再保存
    20w 个 dict 对象，避免和 ``file_list/cache_path_list/label_list`` 双份占内存。
    """

    def __init__(
        self,
        file_list: List[Path],
        cache_path_list: List[Path],
        label_list: List[int],
        source_sha256_list: List[str],
        allow_missing_source_sha256_list: List[bool],
    ):
        self._file_list = file_list
        self._cache_path_list = cache_path_list
        self._label_list = label_list
        self._source_sha256_list = source_sha256_list
        self._allow_missing_source_sha256_list = allow_missing_source_sha256_list

    def __len__(self) -> int:
        return len(self._cache_path_list)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._sample_at(i) for i in range(*index.indices(len(self)))]
        return self._sample_at(int(index))

    def __iter__(self):
        for index in range(len(self)):
            yield self._sample_at(index)

    def _sample_at(self, index: int) -> Dict[str, Any]:
        return {
            "source_path": str(self._file_list[index]),
            "cache_path": str(self._cache_path_list[index]),
            "label": int(self._label_list[index]),
            "source_sha256": self._source_sha256_list[index],
            "allow_missing_source_sha256": bool(self._allow_missing_source_sha256_list[index]),
        }


def _default_dataloader_num_workers(num_workers: Optional[int]) -> int:
    if num_workers is not None:
        return max(0, int(num_workers))
    # Windows 使用 spawn，会复制 Dataset 索引；默认 0 更适合 20w 缓存评估。
    return 0 if os.name == "nt" else 4


def _iter_npz_files(directory: Path, suffix: str = ".npz") -> Iterator[Path]:
    """Iterate matching files without sorted(glob()) directory materialization."""
    suffix = suffix.casefold()
    for path in directory.iterdir():
        if path.is_file() and path.suffix.casefold() == suffix:
            yield path


def _iter_cache_files_with_suffix(cache_dir: Path, suffix: str) -> Iterator[Path]:
    """Iterate cache files for a config hash without building a sorted glob list."""
    for path in cache_dir.iterdir():
        if path.is_file() and path.name.endswith(suffix):
            yield path


class MalwareDataset(Dataset):
    """恶意软件数据集
    
    支持两种数据格式：
    1. 原始文件目录（需要实时提取特征）
    2. NPZ 文件目录（预提取特征）
    
    数据格式：
        byte_sequence: [max_byte_length] uint8
        pe_features: [pe_feature_dim] float32
        stat_features: [stat_feature_dim] float32
        label: int (0=良性, 1=恶意)
    """
    
    def __init__(
        self,
        data_dir: str,
        max_byte_length: int = 65536,
        pe_feature_dim: int = 1500,
        stat_feature_dim: int = 49,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        transform=None,
        target_transform=None,
        label_inference: str = "directory",
        max_samples_per_class: Optional[int] = None,
        max_file_size: int = 1 * 1024 * 1024 * 1024,
        malicious_keywords: Optional[list] = None,
        benign_keywords: Optional[list] = None,
        malicious_dir_names: Optional[list] = None,
        benign_dir_names: Optional[list] = None,
        axon_config=None,
        extraction_workers: int = 1,
        extraction_backend: str = "thread",
    ):
        """
        Args:
            data_dir: 数据目录路径
            max_byte_length: 最大字节序列长度
            pe_feature_dim: PE 特征维度
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            transform: 数据转换
            target_transform: 标签转换
            label_inference: 标签清单生成方式。filename/directory 只允许在没有显式标签表时
                把人工整理好的样本归入 0/1 标签；这些文本绝不能作为模型输入特征。
            max_samples_per_class: 每类最大样本数，None表示不限制
            extraction_workers: 并发提取缓存的线程数，1 表示保持原来的单线程行为
            extraction_backend: 并发后端，thread 使用线程池，process 使用进程池
        """
        self.data_dir = Path(data_dir)
        self.max_byte_length = max_byte_length
        self.pe_feature_dim = pe_feature_dim
        self.stat_feature_dim = stat_feature_dim
        self.lightweight_feature_dim = self._config_value(
            None, axon_config, 'lightweight_feature_dim', 256
        )
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"
        self.transform = transform
        self.target_transform = target_transform
        self.label_inference = label_inference
        self.max_samples_per_class = max_samples_per_class
        self.max_file_size = max_file_size
        self.strict_pe_parsing = self._config_value(
            None, axon_config, 'strict_pe_parsing', True
        )
        configured_allow_pe_fallback = self._config_value(
            None, axon_config, 'allow_pe_fallback', False
        )
        self.allow_pe_fallback = False if self.strict_pe_parsing else configured_allow_pe_fallback
        self.pe_schema_version = self._config_value(
            None, axon_config, 'pe_schema_version', 'legacy_dynamic'
        )
        self.pe_fixed_section_slots = self._config_value(
            None, axon_config, 'pe_fixed_section_slots', 32
        )
        self.follow_symlinks = bool(self._config_value(
            None, axon_config, 'follow_symlinks', False
        ))
        allowed_roots = self._config_value(
            None, axon_config, 'allowed_symlink_roots', []
        )
        self.allowed_symlink_roots = [Path(root).resolve(strict=False) for root in allowed_roots]
        self.strict_unknown_labels = bool(self._config_value(
            None, axon_config, 'strict_unknown_labels', True
        ))
        self.extraction_workers = max(1, int(extraction_workers or 1))
        self.extraction_backend = str(extraction_backend or "thread").lower()
        if self.extraction_backend not in {"thread", "process"}:
            raise ValueError("extraction_backend must be 'thread' or 'process'")
        self.malicious_keywords = self._config_value(
            malicious_keywords, axon_config, 'malicious_keywords',
            ["malware", "malicious", "virus", "trojan", "ransomware", "spyware", "adware", "worm", "backdoor", "rootkit", "keylogger", "botnet", "exploit", "dropper", "loader", "miner"]
        )
        self.benign_keywords = self._config_value(
            benign_keywords, axon_config, 'benign_keywords',
            ["benign", "clean", "safe", "legitimate", "good", "normal", "harmless"]
        )
        self.malicious_dir_names = self._config_value(
            malicious_dir_names, axon_config, 'malicious_dir_names',
            ["malware", "malicious", "virus", "trojan", "ransomware", "spyware", "adware", "worm", "backdoor", "rootkit", "samples", "dirty"]
        )
        self.benign_dir_names = self._config_value(
            benign_dir_names, axon_config, 'benign_dir_names',
            ["benign", "clean", "safe", "legitimate", "good", "normal", "harmless", "white"]
        )
        self._axon_config = axon_config

        if self._axon_config is not None:
            self.extraction_config = ExtractionConfig.from_axon_config(
                self._axon_config,
                max_file_size=max_byte_length,
                pe_feature_dim=pe_feature_dim
            )
        else:
            self.extraction_config = ExtractionConfig(
                max_file_size=max_byte_length,
                pe_feature_dim=pe_feature_dim
            )
        
        # 扫描数据文件
        self.file_list = []
        self.label_list = []
        self.cache_path_list = []
        self.scan_stats = {
            'benign_valid': 0,
            'malicious_valid': 0,
            'cache_hits': 0,
            'extracted': 0,
            'non_pe_skipped': 0,
            'pe_parse_failed_skipped': 0,
            'other_failed_skipped': 0,
        }
        self._scan_stats_lock = threading.Lock()
        
        # 创建缓存目录
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        loaded_prepared_cache = self._load_prepared_cache_index_if_complete()
        if not loaded_prepared_cache:
            self._scan_directory()
            self._save_cache_manifest()
        self._print_scan_summary()
    
    def _scan_directory(self):
        """扫描数据目录，收集所有文件

        当 max_samples_per_class 设置时，每类收集到足够数量即停止扫描。
        extraction_workers > 1 时，文件扫描仍保持原来的确定性顺序，
        但耗时的 PE 特征提取和缓存写入会并发执行。
        """
        benign_dir_names = self._config_value(None, self._axon_config, 'benign_dir_names_fs', ['benign', '待加入白名单'])
        malicious_dir_names = self._config_value(None, self._axon_config, 'malicious_dir_names_fs', ['malicious', '待拉黑'])

        benign_dirs = [self.data_dir / name for name in benign_dir_names]
        malicious_dirs = [self.data_dir / name for name in malicious_dir_names]

        max_benign = self.max_samples_per_class
        max_malicious = self.max_samples_per_class
        
        # 扫描良性样本
        benign_candidates = (
            (file_path, 0)
            for bdir in benign_dirs
            if bdir.exists()
            for file_path in self._iter_files(bdir)
            if file_path.is_file()
        )
        self._append_prepared_samples(
            self._prepare_candidates(benign_candidates, {0: max_benign})
        )
        
        # 扫描恶意样本
        malicious_candidates = (
            (file_path, 1)
            for mdir in malicious_dirs
            if mdir.exists()
            for file_path in self._iter_files(mdir)
            if file_path.is_file()
        )
        self._append_prepared_samples(
            self._prepare_candidates(malicious_candidates, {1: max_malicious})
        )
        
        # 如果没有子目录，直接扫描根目录
        if self.scan_stats['benign_valid'] == 0 and self.scan_stats['malicious_valid'] == 0:
            root_candidates = (
                (file_path, self._infer_label(file_path))
                for file_path in self._iter_files(self.data_dir)
                if file_path.is_file()
            )
            self._append_prepared_samples(
                self._prepare_candidates(root_candidates, {0: max_benign, 1: max_malicious})
            )

    def _iter_files(self, root: Path):
        """按稳定顺序遍历目录，保证每次扫描看到的文件顺序一致。"""
        skip_dirs = {".cache", "__pycache__", ".git", ".pytest_cache", "reports", "models", "swanlog"}
        for dirpath, dirnames, filenames in os.walk(root, followlinks=self.follow_symlinks):
            filtered_dirs = []
            for dirname in sorted(dirnames):
                if dirname in skip_dirs:
                    continue
                child = Path(dirpath) / dirname
                if child.is_symlink() and not self._is_allowed_symlink(child):
                    continue
                filtered_dirs.append(dirname)
            dirnames[:] = filtered_dirs
            for filename in sorted(filenames):
                file_path = Path(dirpath) / filename
                if file_path.is_symlink() and not self._is_allowed_symlink(file_path):
                    continue
                yield file_path

    def _is_allowed_symlink(self, path: Path) -> bool:
        if not self.follow_symlinks:
            return False
        if not path.is_symlink():
            return True
        if not self.allowed_symlink_roots:
            return False
        try:
            target = path.resolve(strict=True)
        except OSError:
            return False
        return any(_is_relative_to_path(target, root) for root in self.allowed_symlink_roots)

    def _increment_scan_stat(self, key: str, amount: int = 1):
        """线程安全地累计扫描统计，避免多个 worker 同时写同一个计数器。"""
        with self._scan_stats_lock:
            self.scan_stats[key] += amount

    def _target_counts_reached(self, counts: Dict[int, int], label_limits: Dict[int, Optional[int]]) -> bool:
        """判断每个类别是否都已经凑够目标数量。None 表示该类别不限量。"""
        for label, limit in label_limits.items():
            if limit is not None and counts.get(label, 0) < limit:
                return False
        return any(limit is not None for limit in label_limits.values())

    def _prepare_candidates(
        self,
        candidates,
        label_limits: Dict[int, Optional[int]],
    ) -> List[Tuple[Path, int, Path]]:
        """准备候选文件缓存。

        这一步是数据提取最慢的地方：读取 PE 文件、解析结构、生成 numpy 特征、
        写入 .cache。extraction_workers 大于 1 时，这些独立文件会交给多个线程并发处理。
        """
        selected_counts: Dict[int, int] = {}
        prepared: List[Tuple[int, Path, int, Path]] = []

        def accept_result(order: int, file_path: Path, label: int, cache_path: Optional[Path]):
            if cache_path is None:
                return
            limit = label_limits.get(label)
            if limit is not None and selected_counts.get(label, 0) >= limit:
                return
            selected_counts[label] = selected_counts.get(label, 0) + 1
            prepared.append((order, file_path, label, cache_path))

        if self.extraction_workers <= 1:
            for order, (file_path, label) in enumerate(candidates):
                result = self._prepare_sample_cache_result(file_path, label)
                accept_result(order, file_path, label, result)
                if self._target_counts_reached(selected_counts, label_limits):
                    break
        else:
            print(
                f"[Dataset] Preparing feature cache with {self.extraction_workers} "
                f"{self.extraction_backend} workers"
            )
            candidate_iter = enumerate(candidates)
            pending = {}
            completed = {}
            next_accept_order = 0
            exhausted = False
            stop_submitting = False
            executor_cls = ThreadPoolExecutor if self.extraction_backend == "thread" else ProcessPoolExecutor

            def submit_until_full(executor):
                nonlocal exhausted
                # 只保持有限数量的任务在飞，避免百万级文件一次性占满内存。
                max_pending = max(self.extraction_workers, self.extraction_workers * 4)
                # completed 里存放“已完成但因顺序约束暂不能接收”的结果；
                # 它也必须计入背压，否则前序慢样本会让后续完成结果无限堆积。
                while not exhausted and not stop_submitting and (len(pending) + len(completed)) < max_pending:
                    try:
                        order, (file_path, label) = next(candidate_iter)
                    except StopIteration:
                        exhausted = True
                        return
                    future = executor.submit(
                        _prepare_sample_cache_worker,
                        self._build_prepare_sample_cache_payload(str(file_path), label),
                    )
                    pending[future] = (order, file_path, label)

            with executor_cls(max_workers=self.extraction_workers) as executor:
                submit_until_full(executor)
                while pending:
                    for future in as_completed(list(pending)):
                        order, file_path, label = pending.pop(future)
                        try:
                            cache_path = self._record_prepare_result(future.result())
                        except Exception as e:
                            self._increment_scan_stat('other_failed_skipped')
                            print(f"[Warning] Skipping sample after worker failure: {file_path} ({e})")
                            cache_path = None
                        completed[order] = (file_path, label, cache_path)
                        while next_accept_order in completed:
                            ready_path, ready_label, ready_cache_path = completed.pop(next_accept_order)
                            accept_result(next_accept_order, ready_path, ready_label, ready_cache_path)
                            next_accept_order += 1
                            if self._target_counts_reached(selected_counts, label_limits):
                                stop_submitting = True
                                break
                        if stop_submitting:
                            for pending_future in pending:
                                pending_future.cancel()
                        else:
                            submit_until_full(executor)
                        break
                    if stop_submitting:
                        break

        prepared.sort(key=lambda item: item[0])
        return [(file_path, label, cache_path) for _order, file_path, label, cache_path in prepared]

    def _append_prepared_samples(self, prepared_samples: List[Tuple[Path, int, Path]]):
        """把已经确认可用的样本登记进 Dataset 索引。"""
        for file_path, label, cache_path in prepared_samples:
            self.file_list.append(file_path)
            self.label_list.append(label)
            self.cache_path_list.append(cache_path)
            if label == 0:
                self.scan_stats['benign_valid'] += 1
            else:
                self.scan_stats['malicious_valid'] += 1

    def _build_prepare_sample_cache_payload(self, file_path: str, label: int) -> Dict:
        """给线程/进程 worker 准备一个可以独立执行的参数包。"""
        return {
            'file_path': file_path,
            'label': int(label),
            'cache_dir': str(self.cache_dir),
            'cache_config_hash': self._cache_config_hash(),
            'max_file_size': self.max_file_size,
            'max_byte_length': self.max_byte_length,
            'pe_feature_dim': self.pe_feature_dim,
            'stat_feature_dim': self.stat_feature_dim,
            'lightweight_feature_dim': self.lightweight_feature_dim,
            'use_cache': self.use_cache,
            'allow_pe_fallback': self.allow_pe_fallback,
            'extraction_config': self.extraction_config,
            'axon_config': self._axon_config,
        }

    def _record_prepare_result(self, result: Dict) -> Optional[Path]:
        """把 worker 返回的状态记入 Dataset 统计，并返回可用缓存路径。"""
        status = result.get('status', 'other_failed_skipped')
        if status in self.scan_stats:
            self._increment_scan_stat(status)
        else:
            self._increment_scan_stat('other_failed_skipped')

        warning = result.get('warning')
        if warning:
            print(f"[Warning] Skipping sample after extraction failure: {warning}")

        cache_path = result.get('cache_path')
        return Path(cache_path) if cache_path else None

    def _prepare_sample_cache_result(self, file_path: Path, label: int) -> Optional[Path]:
        """同步准备单个样本缓存，并统一记录状态。"""
        result = _prepare_sample_cache_worker(
            self._build_prepare_sample_cache_payload(str(file_path), label)
        )
        return self._record_prepare_result(result)

    def _load_prepared_cache_index_if_complete(self) -> bool:
        """如果缓存清单已经满足本次 samples_per_class，直接复用缓存索引。

        这就是“已经提取完成的 100k 数据集直接跳过”的快速通道：
        不重新打开原始 PE 文件，不重新提取特征，只确认已有缓存文件数量够用。
        """
        if not self.use_cache or self.max_samples_per_class is None:
            return False

        source = "manifest"
        selected = self._select_cache_samples_per_class(
            self._iter_manifest_cache_samples(),
            self.max_samples_per_class,
        )
        if not selected:
            source = "cache-scan"
            selected = self._select_cache_samples_per_class(
                self._iter_existing_cache_samples(),
                self.max_samples_per_class,
            )
        if not selected:
            return False

        for sample in selected:
            label = int(sample['label'])
            self.file_list.append(Path(sample.get('source_path', sample['cache_path'])))
            self.label_list.append(label)
            self.cache_path_list.append(Path(sample['cache_path']))
            if label == 0:
                self.scan_stats['benign_valid'] += 1
            else:
                self.scan_stats['malicious_valid'] += 1
        self.scan_stats['cache_hits'] += len(selected)
        print(f"[Dataset] Loaded prepared cache index from {source}: {len(selected)} samples")
        if source == "cache-scan":
            self._save_cache_manifest()
        return True

    def _iter_manifest_cache_samples(self) -> Iterator[Dict[str, Any]]:
        """读取 manifest 中记录的缓存样本，缺失的缓存文件会被忽略。"""
        manifest_path = self._cache_manifest_path()
        if not manifest_path.exists():
            return
        try:
            sample_iter = _iter_manifest_sample_entries(manifest_path)
        except Exception as e:
            print(f"[Warning] Failed to load cache manifest {manifest_path}: {e}")
            return

        try:
            for sample in sample_iter:
                try:
                    cache_path = _manifest_cache_path_fast(sample.get("cache_path", ""), self.cache_dir)
                    label = int(sample["label"])
                    if label not in {0, 1}:
                        continue
                    source_sha = sample.get("source_sha256")
                    if source_sha:
                        source_sha = str(source_sha).strip().casefold()
                        if not _is_valid_source_sha256(source_sha):
                            raise ValueError(f"Manifest source_sha256 is invalid for {cache_path}")
                        meta = _load_cache_metadata(cache_path)
                        if meta is None:
                            raise ValueError(f"Invalid cache metadata for {cache_path}")
                        if int(meta["label"]) != label:
                            raise ValueError(
                                f"Cache label mismatch for {cache_path}: expected {label}, got {meta['label']}"
                            )
                        if meta.get("source_sha256") != source_sha:
                            raise ValueError(f"Cache source SHA mismatch for {cache_path}")
                    yield {
                        "source_path": sample.get("source_path", str(cache_path)),
                        "cache_path": str(cache_path),
                        "label": label,
                        "source_sha256": source_sha,
                    }
                except Exception as e:
                    print(f"[Warning] Ignoring invalid cache manifest sample: {e}")
        except Exception as e:
            print(f"[Warning] Failed to stream cache manifest {manifest_path}: {e}")

    def _load_manifest_cache_samples(self) -> List[Dict]:
        """Compatibility wrapper for callers/tests that need a materialized list."""
        return list(self._iter_manifest_cache_samples())

    def _iter_existing_cache_samples(self) -> Iterator[Dict[str, Any]]:
        """没有 manifest 时，从 .cache 目录扫描兼容配置的缓存文件。"""
        suffix = f"_{self._cache_config_hash()}.npz"
        for cache_path in _iter_cache_files_with_suffix(self.cache_dir, suffix):
            meta = _load_cache_metadata(cache_path)
            if meta is None or meta.get("source_sha256") is None:
                continue
            yield {
                "source_path": str(cache_path),
                "cache_path": str(cache_path),
                "label": int(meta["label"]),
                "source_sha256": meta["source_sha256"],
            }

    def _scan_existing_cache_samples(self) -> List[Dict]:
        return list(self._iter_existing_cache_samples())

    def _select_cache_samples_per_class(self, samples: Iterable[Dict], per_class: int) -> List[Dict]:
        """按类别挑出足够数量的缓存；任一类别不足时返回空列表。"""
        counts = {0: 0, 1: 0}
        selected = []
        for sample in samples:
            label = int(sample['label'])
            if label not in counts or counts[label] >= per_class:
                continue
            selected.append(sample)
            counts[label] += 1
            if counts[0] >= per_class and counts[1] >= per_class:
                return selected
        return []
    
    def _is_valid_sample(self, file_path: Path) -> bool:
        """检查是否为有效的PE样本文件"""
        try:
            file_size = file_path.stat().st_size
        except OSError:
            return False
        if file_size == 0 or file_size > self.max_file_size:
            return False
        try:
            with open(file_path, 'rb') as f:
                if f.read(2) != b'MZ':
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _config_value(explicit_value, axon_config, attr_name: str, fallback):
        if explicit_value is not None:
            return explicit_value
        if axon_config is not None and hasattr(axon_config, attr_name):
            return getattr(axon_config, attr_name)
        return fallback
    
    def _infer_label(self, file_path: Path) -> int:
        """从人工命名/目录约定生成标签，不把命名作为模型证据。"""
        filename_lower = file_path.name.lower()
        
        if self.label_inference == "filename":
            for kw in self.malicious_keywords:
                if kw in filename_lower:
                    return 1
            for kw in self.benign_keywords:
                if kw in filename_lower:
                    return 0
        elif self.label_inference == "directory":
            directory_names = [parent.name.lower() for parent in file_path.parents if parent != self.data_dir.parent]
            for kw in self.malicious_dir_names:
                if any(kw in directory_name for directory_name in directory_names):
                    return 1
            for kw in self.benign_dir_names:
                if any(kw in directory_name for directory_name in directory_names):
                    return 0
        
        if self.strict_unknown_labels:
            raise ValueError(f"Cannot infer label for sample: {file_path}")
        return 0
    
    def _get_cache_path(self, file_path: Path) -> Path:
        """获取缓存文件路径"""
        try:
            stat = file_path.stat()
            file_sig = f"{stat.st_size}_{int(stat.st_mtime_ns)}"
        except OSError:
            file_sig = "missing"
        file_hash = hashlib.md5(f"{file_path.resolve()}_{file_sig}".encode()).hexdigest()
        return self.cache_dir / f"{file_hash}_{self._cache_config_hash()}.npz"

    def _cache_config_hash(self) -> str:
        return _feature_cache_hash(
            self.max_byte_length,
            self.stat_feature_dim,
            self.pe_feature_dim,
            self.lightweight_feature_dim,
            self.strict_pe_parsing,
            self.allow_pe_fallback,
            self.pe_schema_version,
            self.pe_fixed_section_slots,
        )

    def _cache_manifest_path(self) -> Path:
        return self.cache_dir / f"manifest_{self._cache_config_hash()}.json"
    
    def _load_from_cache(self, file_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(file_path)
        return self._load_cache_path(cache_path, file_path)

    def _load_cache_path(
        self,
        cache_path: Path,
        source_path: Optional[Path] = None,
        expected_label: Optional[int] = None,
        expected_source_sha256: Optional[str] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
        """从指定缓存文件加载数据。

        source_path 只用于打印警告。这样 Dataset 可以直接使用 manifest 里的
        cache_path，不必重新根据原始文件路径计算缓存名。
        """
        if cache_path.exists():
            try:
                byte_seq, pe_features, stat_features, lightweight_features, label = _load_cached_feature_npz(
                    cache_path,
                    self.max_byte_length,
                    self.pe_feature_dim,
                    self.stat_feature_dim,
                    self.lightweight_feature_dim,
                    expected_label=(
                        expected_label
                        if expected_label is not None
                        else self._infer_label(source_path) if source_path and source_path.exists() else None
                    ),
                    expected_source_sha256=(
                        expected_source_sha256
                        if expected_source_sha256 is not None
                        else _safe_file_sha256(source_path) if source_path and source_path.exists() else None
                    ),
                )
                if len(byte_seq) != self.max_byte_length:
                    return None
                if len(pe_features) != self.pe_feature_dim:
                    return None
                if len(stat_features) != self.stat_feature_dim:
                    return None
                if len(lightweight_features) != self.lightweight_feature_dim:
                    return None
                return (byte_seq, pe_features, stat_features, lightweight_features, label)
            except Exception as e:
                display_path = source_path if source_path is not None else cache_path
                print(f"[Warning] Failed to load cache for {display_path}: {e}")
        
        return None
    
    def _save_to_cache(self, file_path: Path, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(file_path)
        
        try:
            np.savez_compressed(
                cache_path,
                byte_sequence=data[0],
                pe_features=data[1],
                stat_features=data[2],
                lightweight_features=data[3],
                label=data[4],
                source_sha256=_safe_file_sha256(file_path) or "",
            )
        except Exception as e:
            print(f"[Warning] Failed to save cache for {file_path}: {e}")

    def _save_cache_manifest(self):
        """保存本次扫描得到的缓存清单，后续评估可跳过原始目录扫描。"""
        if not self.use_cache or not self.cache_path_list:
            return

        manifest_path = self._cache_manifest_path()
        header = {
            "version": 1,
            "data_dir": str(self.data_dir),
            "cache_config_hash": self._cache_config_hash(),
            "max_byte_length": self.max_byte_length,
            "pe_feature_dim": self.pe_feature_dim,
            "stat_feature_dim": self.stat_feature_dim,
            "lightweight_feature_dim": self.lightweight_feature_dim,
            "strict_pe_parsing": self.strict_pe_parsing,
            "allow_pe_fallback": self.allow_pe_fallback,
            "pe_schema_version": self.pe_schema_version,
            "pe_fixed_section_slots": self.pe_fixed_section_slots,
        }

        try:
            def iter_samples():
                for file_path, label, cache_path in zip(self.file_list, self.label_list, self.cache_path_list):
                    yield {
                        "source_path": str(file_path),
                        "cache_path": str(cache_path),
                        "label": int(label),
                        "source_sha256": _safe_file_sha256(file_path),
                    }

            _write_cache_manifest_stream(manifest_path, header, iter_samples())
            print(f"[Dataset] Cache manifest: {manifest_path}")
        except Exception as e:
            print(f"[Warning] Failed to save cache manifest: {e}")

    def _normalize_sample_features(
        self,
        byte_seq: np.ndarray,
        pe_feat: np.ndarray,
        stat_feat: np.ndarray,
        lightweight_feat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """把提取出的特征统一成模型需要的固定长度。"""
        return _normalize_cached_arrays(
            byte_seq,
            pe_feat,
            stat_feat,
            lightweight_feat,
            self.max_byte_length,
            self.pe_feature_dim,
            self.stat_feature_dim,
            self.lightweight_feature_dim,
        )

    def _extract_prepared_sample(
        self,
        file_path: Path,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """严格提取一个样本；PE 解析失败返回 None，不生成 fallback 特征。"""
        byte_seq, pe_feat, stat_feat, lightweight_feat, _orig_len = extract_all_features(
            str(file_path),
            self.extraction_config,
            axon_config=getattr(self, '_axon_config', None),
            allow_pe_fallback=self.allow_pe_fallback,
        )
        if byte_seq is None or pe_feat is None:
            return None
        return self._normalize_sample_features(byte_seq, pe_feat, stat_feat, lightweight_feat)

    def _prepare_sample_cache(self, file_path: Path, label: int) -> Optional[Path]:
        """确保样本可严格提取并已有缓存；失败则返回 None。"""
        return self._prepare_sample_cache_result(file_path, label)

    def _sample_to_tensors(
        self,
        byte_seq: np.ndarray,
        pe_feat: np.ndarray,
        stat_feat: np.ndarray,
        label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.transform:
            byte_seq, pe_feat, stat_feat = self.transform(byte_seq, pe_feat, stat_feat)
        if self.target_transform:
            label = self.target_transform(label)
        return (
            torch.from_numpy(byte_seq),
            torch.from_numpy(pe_feat).float(),
            torch.from_numpy(stat_feat).float(),
            torch.tensor(label, dtype=torch.long)
        )

    def _add_prepared_sample(self, file_path: Path, label: int) -> bool:
        cache_path = self._prepare_sample_cache(file_path, label)
        if cache_path is None:
            return False
        self.file_list.append(file_path)
        self.label_list.append(label)
        self.cache_path_list.append(cache_path)
        if label == 0:
            self.scan_stats['benign_valid'] += 1
        else:
            self.scan_stats['malicious_valid'] += 1
        return True

    def _print_scan_summary(self):
        print(
            "[Dataset] valid benign={benign_valid}, valid malicious={malicious_valid}, "
            "cache_hits={cache_hits}, extracted={extracted}, non_pe_skipped={non_pe_skipped}, "
            "pe_parse_failed_skipped={pe_parse_failed_skipped}, other_failed_skipped={other_failed_skipped}".format(
                **self.scan_stats
            )
        )
        if self.max_samples_per_class is not None:
            if self.scan_stats['benign_valid'] < self.max_samples_per_class:
                print(
                    f"[Warning] Requested {self.max_samples_per_class} benign samples, "
                    f"but only {self.scan_stats['benign_valid']} valid PE samples were found"
                )
            if self.scan_stats['malicious_valid'] < self.max_samples_per_class:
                print(
                    f"[Warning] Requested {self.max_samples_per_class} malicious samples, "
                    f"but only {self.scan_stats['malicious_valid']} valid PE samples were found"
                )
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            Tuple of (byte_sequence, pe_features, stat_features, label)
            - byte_sequence: [max_byte_length] torch.uint8
            - pe_features: [pe_feature_dim] torch.float32
            - stat_features: [stat_feature_dim] torch.float32
            - label: torch.long
        """
        file_path = self.file_list[idx]
        label = self.label_list[idx]
        cache_path = self.cache_path_list[idx] if idx < len(self.cache_path_list) else self._get_cache_path(file_path)
        
        # 尝试从缓存加载
        if self.use_cache:
            cached_data = self._load_cache_path(
                cache_path,
                file_path,
                expected_label=label,
                expected_source_sha256="",
            )
            if cached_data is not None:
                byte_seq, pe_feat, stat_feat, _lightweight_feat, cached_label = cached_data
                return self._sample_to_tensors(byte_seq, pe_feat, stat_feat, cached_label)

        if not self.use_cache:
            extracted = self._extract_prepared_sample(file_path)
            if extracted is None:
                raise ValueError(f"Strict PE feature extraction failed for dataset item: {file_path}")
            byte_seq, pe_feat, stat_feat, _lightweight_feat = extracted
            return self._sample_to_tensors(byte_seq, pe_feat, stat_feat, label)
        
        cache_path = self._prepare_sample_cache(file_path, label)
        if cache_path is None:
            raise ValueError(f"Prepared sample cache is unavailable for valid dataset item: {file_path}")
        cached_data = self._load_from_cache(file_path)
        if cached_data is None:
            raise ValueError(f"Failed to load prepared sample cache: {cache_path}")

        byte_seq, pe_feat, stat_feat, _lightweight_feat, cached_label = cached_data
        return self._sample_to_tensors(byte_seq, pe_feat, stat_feat, cached_label)


class NPZDataset(Dataset):
    """预提取的 NPZ 数据集
    
    适用于已经使用 KVD 特征提取器处理过的数据。
    """
    
    def __init__(
        self,
        npz_dir: str,
        split: str = "train",  # train, val, test
        max_byte_length: int = 65536,
        pe_feature_dim: int = 1500,
        stat_feature_dim: int = 49,
    ):
        self.npz_dir = Path(npz_dir) / split
        self.max_byte_length = max_byte_length
        self.pe_feature_dim = pe_feature_dim
        self.stat_feature_dim = stat_feature_dim
        
        # 收集所有 NPZ 文件
        self.npz_files = list(_iter_npz_files(self.npz_dir))
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found in {self.npz_dir}")
    
    def __len__(self) -> int:
        return len(self.npz_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        npz_path = self.npz_files[idx]
        
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                required_fields = {"byte_sequence", "pe_features", "label"}
                missing_fields = sorted(required_fields - set(data.files))
                if missing_fields:
                    raise ValueError(f"NPZ file {npz_path} is missing required fields: {missing_fields}")

                byte_seq = data['byte_sequence']
                pe_feat = data['pe_features']
                stat_feat = data.get('stat_features', np.zeros(self.stat_feature_dim, dtype=np.float32))
                label = int(data['label'])

            if label not in {0, 1}:
                raise ValueError(f"NPZ file {npz_path} has invalid label: {label}")

            byte_seq, pe_feat, stat_feat, _lightweight_feat = _normalize_cached_arrays(
                byte_seq,
                pe_feat,
                stat_feat,
                np.zeros(0, dtype=np.float32),
                self.max_byte_length,
                self.pe_feature_dim,
                self.stat_feature_dim,
                0,
            )

            return (
                torch.from_numpy(byte_seq),
                torch.from_numpy(pe_feat).float(),
                torch.from_numpy(stat_feat).float(),
                torch.tensor(label, dtype=torch.long)
            )
        except Exception as e:
            raise ValueError(f"Failed to load NPZ sample {npz_path}: {e}") from e


class FeatureCacheDataset(Dataset):
    """直接从 data/.cache 中读取已经提取好的样本。

    这个数据集不再访问原始 PE 文件，适合评估、阈值扫描和特征重要性分析。
    优先使用 MalwareDataset 写出的 manifest；没有 manifest 时，按当前配置哈希扫描
    data/.cache 下兼容的缓存文件。
    """

    def __init__(
        self,
        data_dir: str,
        max_byte_length: int = 65536,
        pe_feature_dim: int = 1500,
        stat_feature_dim: int = 49,
        lightweight_feature_dim: int = 256,
        strict_pe_parsing: bool = True,
        allow_pe_fallback: bool = False,
        cache_dir: Optional[str] = None,
        max_samples_per_class: Optional[int] = None,
        require_manifest: bool = False,
        axon_config=None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"
        self.max_byte_length = max_byte_length
        self.pe_feature_dim = pe_feature_dim
        self.stat_feature_dim = stat_feature_dim
        self.lightweight_feature_dim = lightweight_feature_dim
        if axon_config is not None and hasattr(axon_config, 'lightweight_feature_dim'):
            self.lightweight_feature_dim = getattr(axon_config, 'lightweight_feature_dim')
        self.strict_pe_parsing = strict_pe_parsing
        if axon_config is not None and hasattr(axon_config, 'strict_pe_parsing'):
            self.strict_pe_parsing = getattr(axon_config, 'strict_pe_parsing')
        self.allow_pe_fallback = allow_pe_fallback
        if axon_config is not None and hasattr(axon_config, 'allow_pe_fallback'):
            configured_fallback = getattr(axon_config, 'allow_pe_fallback')
            self.allow_pe_fallback = False if self.strict_pe_parsing else configured_fallback
        self.pe_schema_version = getattr(axon_config, 'pe_schema_version', 'legacy_dynamic') if axon_config is not None else 'legacy_dynamic'
        self.pe_fixed_section_slots = getattr(axon_config, 'pe_fixed_section_slots', 32) if axon_config is not None else 32
        self.max_samples_per_class = max_samples_per_class
        self.require_manifest = require_manifest

        if not self.cache_dir.exists():
            raise ValueError(f"Feature cache directory not found: {self.cache_dir}")

        self.cache_config_hash = _feature_cache_hash(
            self.max_byte_length,
            self.stat_feature_dim,
            self.pe_feature_dim,
            self.lightweight_feature_dim,
            self.strict_pe_parsing,
            self.allow_pe_fallback,
            self.pe_schema_version,
            self.pe_fixed_section_slots,
        )
        self.cache_path_list: List[Path] = []
        self.label_list: List[int] = []
        self.file_list: List[Path] = []
        self.source_sha256_list: List[str] = []
        self.allow_missing_source_sha256_list: List[bool] = []

        manifest_path = self._manifest_path()
        if manifest_path.exists():
            source = "manifest"
            self._load_manifest_index(manifest_path)
            if not self.cache_path_list and not require_manifest:
                source = "cache-scan"
                print(
                    "[FeatureCacheDataset] Manifest had no usable samples; "
                    "falling back to cache directory scan."
                )
                self._build_index_from_samples(self._iter_cache_samples())
        elif require_manifest:
            raise ValueError(f"No cache manifest found for config hash {self.cache_config_hash}")
        else:
            source = "cache-scan"
            self._build_index_from_samples(self._iter_cache_samples())

        self.samples = _ColumnarSampleView(
            self.file_list,
            self.cache_path_list,
            self.label_list,
            self.source_sha256_list,
            self.allow_missing_source_sha256_list,
        )
        if source == "cache-scan" and self.cache_path_list:
            if self.max_samples_per_class is None:
                self._save_cache_only_manifest(self.samples)
            else:
                print(
                    "[FeatureCacheDataset] Skipping manifest generation for a "
                    "max_samples_per_class-limited cache scan."
                )

        if not self.cache_path_list:
            raise ValueError(f"No compatible feature cache samples found in {self.cache_dir}")
        print(f"[FeatureCacheDataset] Loaded {len(self.cache_path_list)} samples from {source}: {self.cache_dir}")

    def _manifest_path(self) -> Path:
        return self.cache_dir / f"manifest_{self.cache_config_hash}.json"

    def _append_index_sample(self, sample: Dict[str, Any]) -> None:
        cache_path = Path(sample["cache_path"])
        self.cache_path_list.append(cache_path)
        self.label_list.append(int(sample["label"]))
        self.file_list.append(Path(sample.get("source_path", str(cache_path))))
        self.source_sha256_list.append(str(sample.get("source_sha256") or "").strip().casefold())
        self.allow_missing_source_sha256_list.append(bool(sample.get("allow_missing_source_sha256")))

    def _label_limit_reached(self, counts: Dict[int, int]) -> bool:
        if self.max_samples_per_class is None:
            return False
        limit = int(self.max_samples_per_class)
        return counts.get(0, 0) >= limit and counts.get(1, 0) >= limit

    def _build_index_from_samples(self, samples: Iterable[Dict[str, Any]]) -> None:
        counts: Dict[int, int] = {}
        for sample in samples:
            label = int(sample["label"])
            if self.max_samples_per_class is not None:
                limit = int(self.max_samples_per_class)
                if counts.get(label, 0) >= limit:
                    if self._label_limit_reached(counts):
                        break
                    continue
            self._append_index_sample(sample)
            counts[label] = counts.get(label, 0) + 1

    def _load_manifest_index(self, manifest_path: Path) -> None:
        legacy_missing_source_sha_count = 0
        invalid_count = 0
        invalid_reasons: List[str] = []

        def record_invalid_reason(error: Exception) -> None:
            nonlocal invalid_count
            invalid_count += 1
            reason = str(error)
            if len(invalid_reasons) < 5:
                invalid_reasons.append(reason)
            if invalid_count <= 20:
                print(f"[Warning] Ignoring invalid cache manifest sample: {reason}")
            elif invalid_count == 21:
                print("[Warning] Further invalid cache manifest sample warnings suppressed.")

        try:
            sample_iter = _iter_manifest_sample_entries(manifest_path)
        except Exception as e:
            print(f"[Warning] Failed to load cache manifest {manifest_path}: {e}")
            return

        counts: Dict[int, int] = {}
        try:
            for sample in sample_iter:
                try:
                    cache_path = _manifest_cache_path_fast(sample.get("cache_path", ""), self.cache_dir)
                    label = int(sample["label"])
                    if label not in {0, 1}:
                        raise ValueError(f"Manifest label must be 0 or 1: {label}")
                    source_path_text = sample.get("source_path", str(cache_path))
                    source_sha = sample.get("source_sha256")
                    source_sha_text = str(source_sha or "").strip().casefold()
                    allow_missing_source_sha = not bool(source_sha)
                    if self.require_manifest:
                        if not _is_valid_source_sha256(source_sha_text):
                            raise ValueError(f"Strict cache manifest requires valid source_sha256 for {cache_path}")
                        meta = _load_cache_metadata(cache_path)
                        if meta is None:
                            raise ValueError(f"Invalid cache metadata for {cache_path}")
                        if int(meta["label"]) != label:
                            raise ValueError(
                                f"Cache label mismatch for {cache_path}: expected {label}, got {meta['label']}"
                            )
                        cached_source_sha = meta.get("source_sha256")
                        if cached_source_sha != source_sha_text:
                            raise ValueError(f"Cache source SHA mismatch for {cache_path}")
                    if allow_missing_source_sha:
                        legacy_missing_source_sha_count += 1
                    if self.max_samples_per_class is not None:
                        limit = int(self.max_samples_per_class)
                        if counts.get(label, 0) >= limit:
                            if self._label_limit_reached(counts):
                                break
                            continue
                    self._append_index_sample({
                        "source_path": source_path_text,
                        "cache_path": str(cache_path),
                        "label": label,
                        "source_sha256": source_sha_text,
                        "allow_missing_source_sha256": allow_missing_source_sha,
                    })
                    counts[label] = counts.get(label, 0) + 1
                except Exception as e:
                    record_invalid_reason(e)
        except Exception as e:
            record_invalid_reason(e)
            print(f"[Warning] Failed to stream cache manifest {manifest_path}: {e}")
        if self.require_manifest and invalid_reasons and not self.cache_path_list:
            raise ValueError(
                "Strict cache manifest has no valid samples after metadata validation: "
                + "; ".join(invalid_reasons[:5])
            )
        if invalid_count:
            print(f"[FeatureCacheDataset] Ignored {invalid_count} invalid manifest samples.")
        if legacy_missing_source_sha_count:
            print(
                "[FeatureCacheDataset] Legacy cache manifest without source_sha256: "
                f"{legacy_missing_source_sha_count} samples will be loaded without source file fingerprint validation."
            )

    def _iter_cache_samples(self) -> Iterator[Dict[str, Any]]:
        suffix = f"_{self.cache_config_hash}.npz"
        for cache_path in _iter_cache_files_with_suffix(self.cache_dir, suffix):
            meta = _load_cache_metadata(cache_path)
            if meta is None or meta.get("source_sha256") is None:
                continue
            yield {
                "source_path": str(cache_path),
                "cache_path": str(cache_path),
                "label": int(meta["label"]),
                "source_sha256": meta["source_sha256"],
            }

    def _save_cache_only_manifest(self, samples: Iterable[Dict[str, Any]]):
        """旧缓存没有 manifest 时，基于 cache 目录补一份清单供下次复用。"""
        header = {
            "version": 1,
            "data_dir": str(self.data_dir),
            "cache_config_hash": self.cache_config_hash,
            "max_byte_length": self.max_byte_length,
            "pe_feature_dim": self.pe_feature_dim,
            "stat_feature_dim": self.stat_feature_dim,
            "lightweight_feature_dim": self.lightweight_feature_dim,
            "strict_pe_parsing": self.strict_pe_parsing,
            "allow_pe_fallback": self.allow_pe_fallback,
            "pe_schema_version": self.pe_schema_version,
            "pe_fixed_section_slots": self.pe_fixed_section_slots,
            "source": "cache-scan",
        }
        try:
            manifest_path = self._manifest_path()
            _write_cache_manifest_stream(manifest_path, header, samples)
            print(f"[FeatureCacheDataset] Cache manifest generated: {manifest_path}")
        except Exception as e:
            print(f"[Warning] Failed to save feature cache manifest: {e}")

    def __len__(self) -> int:
        return len(self.cache_path_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_path = self.cache_path_list[idx]
        byte_seq, pe_feat, stat_feat, _lightweight_feat, label = _load_cached_feature_npz(
            cache_path,
            self.max_byte_length,
            self.pe_feature_dim,
            self.stat_feature_dim,
            self.lightweight_feature_dim,
            expected_label=int(self.label_list[idx]),
            expected_source_sha256=self.source_sha256_list[idx],
            allow_missing_source_sha256=bool(self.allow_missing_source_sha256_list[idx]),
        )
        return (
            torch.from_numpy(byte_seq),
            torch.from_numpy(pe_feat).float(),
            torch.from_numpy(stat_feat).float(),
            torch.tensor(label, dtype=torch.long),
        )


class NPZDataLoader:
    """NPZ 数据加载器封装
    
    提供便捷的数据加载接口。
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        max_byte_length: int = 65536,
        pe_feature_dim: int = 1500,
        stat_feature_dim: int = 49,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        shuffle: bool = True,
        max_samples_per_class: Optional[int] = None,
        allow_raw_fallback: bool = True,
    ):
        """
        Args:
            data_dir: 数据目录路径
            batch_size: 批次大小
            max_byte_length: 最大字节序列长度
            pe_feature_dim: PE 特征维度
            stat_feature_dim: 统计特征维度
            num_workers: 数据加载线程数；None 时 Windows 默认 0，避免 spawn 复制大索引
            pin_memory: 是否固定内存
            shuffle: 是否打乱数据
            max_samples_per_class: 每类最大样本数，None表示不限制
            allow_raw_fallback: NPZ 不存在时是否回退扫描原始文件
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_byte_length = max_byte_length
        self.pe_feature_dim = pe_feature_dim
        self.stat_feature_dim = stat_feature_dim
        self.num_workers = _default_dataloader_num_workers(num_workers)
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.max_samples_per_class = max_samples_per_class
        self.allow_raw_fallback = allow_raw_fallback
        
        self.dataset = None
        self.loader = None
    
    def create_dataloader(self, split: str = "train") -> DataLoader:
        """创建数据加载器
        
        Args:
            split: 数据集划分 (train, val, test)
        """
        try:
            # 尝试使用 NPZ 数据集
            dataset = NPZDataset(
                npz_dir=self.data_dir,
                split=split,
                max_byte_length=self.max_byte_length,
                pe_feature_dim=self.pe_feature_dim,
                stat_feature_dim=self.stat_feature_dim,
            )
        except ValueError:
            if not self.allow_raw_fallback:
                raise
            # 回退到原始文件数据集
            dataset = MalwareDataset(
                data_dir=self.data_dir,
                max_byte_length=self.max_byte_length,
                pe_feature_dim=self.pe_feature_dim,
                stat_feature_dim=self.stat_feature_dim,
                max_samples_per_class=self.max_samples_per_class,
            )
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle and split == "train",
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=split == "train",
        )
        
        return loader
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        return self.create_dataloader("train")
    
    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        return self.create_dataloader("val")
    
    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器"""
        return self.create_dataloader("test")


class SubDataset(Dataset):
    """子数据集包装器"""

    def __init__(self, base_dataset, indices, sample_weights: Optional[List[float]] = None):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sample_weights = (
            np.asarray(sample_weights, dtype=np.float32)
            if sample_weights is not None
            else None
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.base_dataset[int(self.indices[idx])]
        if self.sample_weights is None:
            return sample
        weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        if isinstance(sample, tuple):
            return (*sample, weight)
        return sample, weight


def _normalize_split_path(path_text: str) -> str:
    """把路径整理成稳定可比较的文本。

    原始相似度报告里是绝对路径，manifest 里经常是 data/... 相对路径。
    这里统一大小写和路径分隔符，避免同一个文件因为写法不同而匹配失败。
    """
    return str(path_text).replace("\\", "/").casefold()


def _split_path_keys(path_text: str) -> List[str]:
    """返回一个文件路径可能出现的几种写法。"""
    path = Path(path_text)
    normalized = _normalize_split_path(path_text)
    keys = {normalized}
    try:
        cwd_text = _normalize_split_path(str(Path.cwd())).rstrip("/")
        if path.is_absolute():
            prefix = f"{cwd_text}/"
            if normalized.startswith(prefix):
                keys.add(normalized[len(prefix):])
        else:
            keys.add(_normalize_split_path(str(Path.cwd() / path)))
    except (OSError, RuntimeError):
        pass
    return list(keys)


def _read_split_file(split_file: Path) -> Dict[str, str]:
    """读取 group_isolated_split.csv，返回 source_path -> split 的映射。"""
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    assignments: Dict[str, str] = {}
    group_splits: Dict[str, str] = {}
    valid_splits = {"train", "val", "test"}
    with split_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "source_path" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError("Split file must contain source_path and split columns")

        for row in reader:
            source_path = row.get("source_path", "")
            split = row.get("split", "")
            if not source_path:
                continue
            if split not in valid_splits:
                raise ValueError(f"Invalid split value for {source_path}: {split}")

            group_id = row.get("group_id")
            if group_id:
                existing_split = group_splits.get(group_id)
                if existing_split is not None and existing_split != split:
                    raise ValueError(f"Group {group_id} appears in both {existing_split} and {split}")
                group_splits[group_id] = split

            for key in _split_path_keys(source_path):
                assignments[key] = split

    if not assignments:
        raise ValueError(f"Split file has no usable samples: {split_file}")
    return assignments


def _read_split_rows_by_key(split_file: Path) -> Dict[str, Dict[str, str]]:
    """读取 split CSV，返回用于严格身份校验的行元数据。"""
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    required_columns = {"source_path", "split", "label", "source_sha256"}
    rows_by_key: Dict[str, Dict[str, str]] = {}
    with split_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required_columns - fieldnames)
        if missing:
            raise ValueError(
                "Strict split metadata requires columns: "
                + ", ".join(sorted(required_columns))
                + f"; missing: {missing}"
            )

        for row in reader:
            source_path = row.get("source_path", "")
            if not source_path:
                continue
            split = row.get("split", "")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split value for {source_path}: {split}")
            label_text = str(row.get("label", "")).strip()
            if label_text not in {"0", "1"}:
                raise ValueError(f"Strict split metadata requires label 0/1 for {source_path}: {label_text}")
            source_sha = str(row.get("source_sha256", "")).strip().casefold()
            if not _is_valid_source_sha256(source_sha):
                raise ValueError(f"Strict split metadata requires valid source_sha256 for {source_path}")
            normalized = dict(row)
            normalized["label"] = label_text
            normalized["source_sha256"] = source_sha
            for key in _split_path_keys(source_path):
                rows_by_key[key] = normalized

    if not rows_by_key:
        raise ValueError(f"Split file has no usable samples: {split_file}")
    return rows_by_key


def _dataset_sample_metadata(dataset: Dataset, index: int) -> Dict[str, object]:
    """读取 dataset 暴露的样本标签和内容 hash 元数据。"""
    label_list = getattr(dataset, "label_list", None)
    if label_list is None:
        raise ValueError("Dataset does not expose label_list, cannot verify strict split labels")
    if index >= len(label_list):
        raise ValueError(f"Dataset label_list is shorter than file_list at index {index}")

    metadata = {"label": int(label_list[index])}
    samples = getattr(dataset, "samples", None)
    if samples is not None and index < len(samples):
        metadata["source_sha256"] = str(samples[index].get("source_sha256") or "").strip().casefold()
    return metadata


def _strict_split_metadata_failure(dataset: Dataset, index: int, split_row: Dict[str, str]) -> Optional[str]:
    metadata = _dataset_sample_metadata(dataset, index)
    split_label = int(split_row["label"])
    dataset_label = int(metadata["label"])
    if dataset_label != split_label:
        return f"label_mismatch:split={split_label}:dataset={dataset_label}"

    split_sha = str(split_row.get("source_sha256") or "").strip().casefold()
    dataset_sha = str(metadata.get("source_sha256") or "").strip().casefold()
    if not _is_valid_source_sha256(split_sha):
        return "split_invalid_source_sha256"
    if not _is_valid_source_sha256(dataset_sha):
        return "dataset_missing_source_sha256"
    if dataset_sha != split_sha:
        return "source_sha256_mismatch"
    return None


def _group_sample_weight(
    row: Dict,
    singleton_group_weight: float,
    rare_group_weight: float,
    medium_group_weight: float,
) -> float:
    group_size = int(row.get("group_size") or 1)
    if group_size <= 1:
        return float(singleton_group_weight)
    if group_size <= 5:
        return float(rare_group_weight)
    if group_size <= 20:
        return float(medium_group_weight)
    return 1.0


def _row_sample_weight(row: Dict) -> Optional[float]:
    """读取 split 文件里的显式样本权重；空值表示不覆盖默认权重。"""
    raw_value = row.get("sample_weight")
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        weight = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid sample_weight for {row.get('source_path', '')}: {raw_value}") from exc
    if weight <= 0:
        raise ValueError(f"sample_weight must be positive for {row.get('source_path', '')}: {raw_value}")
    return weight


def create_split_from_file(
    dataset: Dataset,
    split_file: Path,
    *,
    rare_group_weighting: bool = False,
    singleton_group_weight: float = 1.8,
    rare_group_weight: float = 1.5,
    medium_group_weight: float = 1.2,
    require_explicit_metadata: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """按外部 CSV 清单创建 train/val/test 数据集。

    这用于相似族群隔离：CSV 已经保证同一个相似组只进入一个 split。
    Dataset 只选择 CSV 中出现且能匹配到的样本，避免把未诊断样本混进真实评估。
    """
    assignments = _read_split_file(split_file)
    strict_rows_by_key = _read_split_rows_by_key(split_file) if require_explicit_metadata else {}
    train_weights_by_key: Dict[str, float] = {}
    has_explicit_train_weight = False
    with Path(split_file).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != "train":
                continue
            explicit_weight = _row_sample_weight(row)
            if explicit_weight is not None:
                weight = explicit_weight
                has_explicit_train_weight = True
            elif rare_group_weighting:
                weight = _group_sample_weight(
                    row,
                    singleton_group_weight,
                    rare_group_weight,
                    medium_group_weight,
                )
            else:
                continue
            for key in _split_path_keys(row.get("source_path", "")):
                train_weights_by_key[key] = weight
    use_sample_weights = rare_group_weighting or has_explicit_train_weight

    file_list = getattr(dataset, "file_list", None)
    if file_list is None:
        raise ValueError("Dataset does not expose file_list, cannot apply split file")

    split_indices = {"train": [], "val": [], "test": []}
    train_weights = []
    missing_count = 0
    strict_failures = []
    for index, file_path in enumerate(file_list):
        split = None
        matched_key = None
        for key in _split_path_keys(str(file_path)):
            split = assignments.get(key)
            if split is not None:
                matched_key = key
                break
        if split is None:
            missing_count += 1
            continue
        if require_explicit_metadata:
            split_row = strict_rows_by_key.get(matched_key or "")
            if split_row is None:
                strict_failures.append(f"{file_path}:missing_strict_split_row")
            else:
                failure = _strict_split_metadata_failure(dataset, index, split_row)
                if failure is not None:
                    strict_failures.append(f"{file_path}:{failure}")
        split_indices[split].append(index)
        if split == "train" and use_sample_weights:
            train_weights.append(train_weights_by_key.get(matched_key, 1.0))

    if strict_failures:
        examples = strict_failures[:20]
        raise ValueError(
            "Strict split metadata verification failed; filename/path/directory labels are not allowed "
            f"for this split workflow. failures={examples}, total_failures={len(strict_failures)}"
        )

    if not all(split_indices.values()):
        sizes = {name: len(indices) for name, indices in split_indices.items()}
        raise ValueError(f"Split file produced an empty dataset split: {sizes}")

    matched_count = sum(len(indices) for indices in split_indices.values())
    print(
        "[Dataset] Applied split file: "
        f"train={len(split_indices['train'])}, val={len(split_indices['val'])}, "
        f"test={len(split_indices['test'])}, matched={matched_count}, "
        f"unmatched_dataset_samples={missing_count}"
    )
    if rare_group_weighting:
        print(
            "[Dataset] Rare-group weighting enabled: "
            f"singleton={singleton_group_weight}, rare={rare_group_weight}, "
            f"medium={medium_group_weight}"
        )
    if has_explicit_train_weight:
        print("[Dataset] Explicit sample_weight column enabled for training samples")

    return (
        SubDataset(dataset, split_indices["train"], train_weights if use_sample_weights else None),
        SubDataset(dataset, split_indices["val"]),
        SubDataset(dataset, split_indices["test"]),
    )


def create_stratified_split(
    dataset: Dataset,
    val_ratio: float = None,
    test_ratio: float = None,
    seed: int = None,
    axon_config=None
) -> Tuple[Dataset, Dataset, Dataset]:
    """创建分层划分的数据集

    Args:
        dataset: 完整数据集
        val_ratio: 验证集比例（None时从axon_config读取，默认0.2）
        test_ratio: 测试集比例（None时从axon_config读取，默认0.1）
        seed: 随机种子（None时从axon_config读取，默认42）
        axon_config: 可选的AxonExperimentConfig

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if val_ratio is None:
        val_ratio = getattr(axon_config, 'val_ratio', 0.2) if axon_config else 0.2
    if test_ratio is None:
        test_ratio = getattr(axon_config, 'test_ratio', 0.1) if axon_config else 0.1
    if seed is None:
        seed = getattr(axon_config, 'seed', 42) if axon_config else 42
    rng = np.random.RandomState(seed)

    # 按标签分层
    labels = dataset.label_list
    unique_labels = list(set(labels))

    train_indices = []
    val_indices = []
    test_indices = []

    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        rng.shuffle(label_indices)

        n = len(label_indices)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        if val_ratio > 0 and n_val == 0 and n >= 3:
            n_val = 1
        if test_ratio > 0 and n_test == 0 and n - n_val >= 2:
            n_test = 1
        if n_val + n_test >= n:
            n_test = max(0, n - n_val - 1)

        val_indices.extend(label_indices[:n_val])
        test_indices.extend(label_indices[n_val:n_val + n_test])
        train_indices.extend(label_indices[n_val + n_test:])

    train_dataset = SubDataset(dataset, train_indices)
    val_dataset = SubDataset(dataset, val_indices)
    test_dataset = SubDataset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


class FastModeDataset(Dataset):
    """快速训练模式的小型数据集包装器"""

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[int(self.indices[idx])]


def create_fast_mode_dataset(
    dataset: Dataset,
    samples_per_class: int = None,
    seed: int = None,
    axon_config=None
) -> Dataset:
    """创建快速训练模式的小型数据集

    用于测试代码是否存在bug（如测评失真或训练崩溃）。
    从每类中随机抽取少量样本。

    Args:
        dataset: 完整数据集
        samples_per_class: 每类抽取的样本数量（None时从axon_config读取，默认50）
        seed: 随机种子（None时从axon_config读取，默认42）
        axon_config: 可选的AxonExperimentConfig

    Returns:
        小型数据集，包含指定数量的每类样本
    """
    if samples_per_class is None:
        samples_per_class = getattr(axon_config, 'fast_mode_samples', 50) if axon_config else 50
    if seed is None:
        seed = getattr(axon_config, 'seed', 42) if axon_config else 42
    rng = np.random.RandomState(seed)

    # 按标签分层
    labels = dataset.label_list
    unique_labels = list(set(labels))

    fast_indices = []

    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        rng.shuffle(label_indices)

        # 取前 samples_per_class 个样本
        selected = label_indices[:min(samples_per_class, len(label_indices))]
        fast_indices.extend(selected)

    # 打乱顺序
    np.random.shuffle(fast_indices)

    return FastModeDataset(dataset, fast_indices)


class AugmentedDataset(Dataset):
    """数据增强包装器

    在训练时对字节序列和 PE 特征施加随机扰动，模拟加壳/混淆/编译差异。
    仅在 enable=True 时应用增强，否则直接透传原始样本。

    支持的增强方式：
    - byte_dropout: 随机将字节置零（模拟截断/损坏）
    - byte_noise: 随机替换字节值（模拟多态变异）
    - feature_noise: PE 特征加高斯噪声（模拟编译器差异）
    """

    def __init__(self, base_dataset, augmentation_config):
        """
        Args:
            base_dataset: 底层数据集，返回 (byte_seq, pe_features, stat_features, label)
            augmentation_config: DataAugmentationConfig 实例
        """
        self.base_dataset = base_dataset
        self.config = augmentation_config
        self.enabled = augmentation_config.enable

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        if not self.enabled:
            return sample

        # 处理 SubDataset 返回 5-tuple 的情况
        if len(sample) == 5:
            byte_seq, pe_features, stat_features, label, weight = sample
        else:
            byte_seq, pe_features, stat_features, label = sample
            weight = None

        # 计算实际字节长度（排除零填充区域），防止增强污染 padding
        actual_byte_len = int((byte_seq != 0).sum().item())

        # 字节级增强
        if self.config.byte_dropout > 0:
            byte_seq = self._apply_byte_dropout(byte_seq, self.config.byte_dropout, actual_byte_len)
        if self.config.byte_noise > 0:
            byte_seq = self._apply_byte_noise(byte_seq, self.config.byte_noise, actual_byte_len)

        # 特征级增强
        if self.config.feature_noise > 0:
            pe_features = self._apply_feature_noise(pe_features, self.config.feature_noise)

        if weight is not None:
            return byte_seq, pe_features, stat_features, label, weight
        return byte_seq, pe_features, stat_features, label

    @staticmethod
    def _apply_byte_dropout(byte_seq: torch.Tensor, dropout_rate: float, actual_length: int = -1) -> torch.Tensor:
        """随机将字节置零（模拟截断/损坏）

        Args:
            byte_seq: [max_byte_length] integer tensor, 值域 0-255
            dropout_rate: 置零概率
            actual_length: 实际字节长度（不含 padding）；>=0 时仅在 [0, actual_length) 施加增强

        Returns:
            增强后的 byte_seq（新 tensor，不修改原始数据）
        """
        mask = torch.rand(byte_seq.shape, device=byte_seq.device) >= dropout_rate
        if actual_length >= 0 and actual_length < byte_seq.shape[0]:
            # padding 区域保持 0 不变，仅对实际内容区域做 dropout
            result = byte_seq.clone()
            result[:actual_length] = byte_seq[:actual_length] * mask[:actual_length].to(dtype=byte_seq.dtype)
            return result
        return byte_seq * mask.to(dtype=byte_seq.dtype)

    @staticmethod
    def _apply_byte_noise(byte_seq: torch.Tensor, noise_rate: float, actual_length: int = -1) -> torch.Tensor:
        """随机替换字节值（模拟多态变异）

        Args:
            byte_seq: [max_byte_length] integer tensor, 值域 0-255
            noise_rate: 替换概率
            actual_length: 实际字节长度（不含 padding）；>=0 时仅在 [0, actual_length) 施加增强

        Returns:
            增强后的 byte_seq（新 tensor，不修改原始数据）
        """
        mask = torch.rand(byte_seq.shape, device=byte_seq.device) < noise_rate
        random_bytes = torch.randint(0, 256, byte_seq.shape, device=byte_seq.device, dtype=byte_seq.dtype)
        noisy = torch.where(mask, random_bytes, byte_seq)
        if actual_length >= 0 and actual_length < byte_seq.shape[0]:
            # padding 区域保持 0 不变，仅对实际内容区域加噪声
            return torch.cat([noisy[:actual_length], byte_seq[actual_length:]], dim=0)
        return noisy

    @staticmethod
    def _apply_feature_noise(pe_features: torch.Tensor, noise_std: float) -> torch.Tensor:
        """PE 特征加高斯噪声（模拟编译器差异）

        Args:
            pe_features: [pe_feature_dim] torch.float32
            noise_std: 噪声标准差

        Returns:
            增强后的 pe_features（新 tensor，不修改原始数据）
        """
        noise = torch.randn(pe_features.shape, device=pe_features.device) * noise_std
        return pe_features + noise
