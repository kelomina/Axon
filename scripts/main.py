#!/usr/bin/env python3
"""Axon v2.6 实验主脚本。

用于训练、评估和测试 Axon 恶意软件检测模型。

使用方法：
    # 训练
    python main.py train --data-dir data/samples --epochs 50

    # 评估
    python main.py eval --checkpoint models/best_model.pt --data-dir data/test

    # 测试单个文件
    python main.py predict --file path/to/sample.exe

    # 特征提取
    python main.py extract --data-dir raw_samples --output-dir data/extracted
"""

import sys
import argparse
import dataclasses
import hashlib
import json
import os
import random
import tomllib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch.nn.functional as F
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT_PATH = Path(__file__).resolve()

# 添加 src/scripts 目录到路径
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from config import AxonExperimentConfig, DSRAArchitectureConfig, TrainingConfig, DataAugmentationConfig
from model import AxonMalwareModel
from dataset import MalwareDataset, NPZDataLoader, FeatureCacheDataset, create_split_from_file, create_stratified_split, AugmentedDataset
from trainer import AxonTrainer
from kvd_features import extract_all_features, ExtractionConfig
from security import load_safe_checkpoint
from archive_scanner import (
    ArchiveScanOptions,
    cleanup_scan_temp,
    iter_pe_prediction_targets,
    run_archive_scan,
)
from pre_run_resource_leak_guard import validate_guard_receipt


RESOURCE_GUARD_REQUIRED_COMMANDS = {"train", "eval", "extract", "importance"}


class FeatureMaskedModel(torch.nn.Module):
    """Apply an exported PE/stat feature mask before forwarding to the model."""

    def __init__(self, model: torch.nn.Module, pe_mask: torch.Tensor, stat_mask: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("pe_mask", pe_mask.float().view(1, -1))
        self.register_buffer("stat_mask", stat_mask.float().view(1, -1))

    def forward(self, byte_seq, pe_features, stat_features=None, **kwargs):
        pe_features = pe_features * self.pe_mask.to(device=pe_features.device, dtype=pe_features.dtype)
        if stat_features is not None:
            stat_features = stat_features * self.stat_mask.to(
                device=stat_features.device,
                dtype=stat_features.dtype,
            )
        return self.model(byte_seq, pe_features, stat_features=stat_features, **kwargs)


def _individual_from_mask_payload(payload: dict) -> list[bool]:
    if "individual" in payload:
        return [bool(value) for value in payload["individual"]]

    mask_spec = payload.get("mask_spec", {})
    pe_search_dim = int(mask_spec.get("pe_search_dim", 0))
    stat_feature_dim = int(mask_spec.get("stat_feature_dim", 0))
    if pe_search_dim <= 0 or stat_feature_dim <= 0:
        raise ValueError("Feature mask must contain individual or mask_spec dimensions")

    individual = [False] * (pe_search_dim + stat_feature_dim)
    for index in payload.get("selected_pe_indices", []):
        index = int(index)
        if index < 0 or index >= pe_search_dim:
            raise ValueError(f"PE feature index out of searched range: {index}")
        individual[index] = True
    for index in payload.get("selected_stat_indices", []):
        index = int(index)
        if index < 0 or index >= stat_feature_dim:
            raise ValueError(f"stat feature index out of range: {index}")
        individual[pe_search_dim + index] = True
    return individual


def _load_feature_mask_tensors(
    feature_mask_path: Optional[str],
    config: AxonExperimentConfig,
    device: str | torch.device,
) -> Optional[tuple[torch.Tensor, torch.Tensor, dict]]:
    if not feature_mask_path:
        return None

    path = Path(feature_mask_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mask_spec = payload.get("mask_spec", {})
    pe_search_dim = int(mask_spec.get("pe_search_dim", config.pe_feature_dim))
    pe_feature_dim = int(mask_spec.get("pe_feature_dim", config.pe_feature_dim))
    stat_feature_dim = int(mask_spec.get("stat_feature_dim", config.stat_feature_dim))

    if pe_feature_dim != config.pe_feature_dim:
        raise ValueError(
            f"Feature mask PE dim {pe_feature_dim} does not match model PE dim {config.pe_feature_dim}"
        )
    if stat_feature_dim != config.stat_feature_dim:
        raise ValueError(
            f"Feature mask stat dim {stat_feature_dim} does not match model stat dim {config.stat_feature_dim}"
        )
    if pe_search_dim <= 0 or pe_search_dim > config.pe_feature_dim:
        raise ValueError(f"Invalid feature mask pe_search_dim: {pe_search_dim}")

    individual = _individual_from_mask_payload(payload)
    expected_len = pe_search_dim + config.stat_feature_dim
    if len(individual) != expected_len:
        raise ValueError(
            f"Feature mask individual length {len(individual)} does not match expected {expected_len}"
        )

    pe_mask = torch.zeros(config.pe_feature_dim, dtype=torch.float32, device=device)
    stat_mask = torch.zeros(config.stat_feature_dim, dtype=torch.float32, device=device)
    pe_bits = individual[:pe_search_dim]
    stat_bits = individual[pe_search_dim:]
    pe_mask[:pe_search_dim] = torch.tensor(pe_bits, dtype=torch.float32, device=device)
    stat_mask[:] = torch.tensor(stat_bits, dtype=torch.float32, device=device)
    return pe_mask, stat_mask, payload


def _summarize_feature_mask(payload: dict) -> str:
    kept_total = payload.get("kept_total")
    kept_pe = payload.get("kept_pe")
    kept_stat = payload.get("kept_stat")
    return f"kept_total={kept_total}, kept_pe={kept_pe}, kept_stat={kept_stat}"


def _set_training_seed(seed: int) -> None:
    """Seed training randomness so hyperparameter comparisons are meaningful."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_train_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _adapt_widened_classifier_tensor(
    key: str,
    checkpoint_value: torch.Tensor,
    current_value: torch.Tensor,
) -> torch.Tensor | None:
    """Adapt the old classifier head when only its hidden width is widened."""
    checkpoint_shape = tuple(checkpoint_value.shape)
    current_shape = tuple(current_value.shape)
    checkpoint_value = checkpoint_value.to(device=current_value.device, dtype=current_value.dtype)

    if key == "classifier.2.weight":
        if len(checkpoint_shape) == 2 and len(current_shape) == 2:
            old_hidden, old_input = checkpoint_shape
            new_hidden, new_input = current_shape
            if new_hidden > old_hidden and new_input == old_input:
                adapted = current_value.clone()
                adapted[:old_hidden, :] = checkpoint_value
                return adapted

    if key == "classifier.2.bias":
        if len(checkpoint_shape) == 1 and len(current_shape) == 1:
            old_hidden = checkpoint_shape[0]
            new_hidden = current_shape[0]
            if new_hidden > old_hidden:
                adapted = current_value.clone()
                adapted[:old_hidden] = checkpoint_value
                return adapted

    if key == "classifier.5.weight":
        if len(checkpoint_shape) == 2 and len(current_shape) == 2:
            old_classes, old_hidden = checkpoint_shape
            new_classes, new_hidden = current_shape
            if new_classes == old_classes and new_hidden > old_hidden:
                adapted = current_value.clone()
                adapted[:, :old_hidden] = checkpoint_value
                adapted[:, old_hidden:] = 0
                return adapted

    return None


def _filter_partial_init_state_dict(
    model: torch.nn.Module,
    checkpoint_state_dict: dict,
) -> tuple[dict, list[tuple[str, tuple, tuple]], list[tuple[str, tuple, tuple]]]:
    """Filter checkpoint tensors that cannot be loaded into the current model shape."""
    current_state = model.state_dict()
    filtered = {}
    shape_mismatches = []
    adapted_shapes = []
    for key, value in checkpoint_state_dict.items():
        current_value = current_state.get(key)
        if current_value is None:
            filtered[key] = value
            continue
        if tuple(value.shape) != tuple(current_value.shape):
            adapted_value = _adapt_widened_classifier_tensor(key, value, current_value)
            if adapted_value is not None:
                filtered[key] = adapted_value
                adapted_shapes.append((key, tuple(value.shape), tuple(current_value.shape)))
                continue
            shape_mismatches.append((key, tuple(value.shape), tuple(current_value.shape)))
            continue
        filtered[key] = value
    return filtered, shape_mismatches, adapted_shapes


def _read_toml_config(config_path: Optional[str]) -> dict:
    """读取 TOML 配置文件；未传入时返回空配置。"""
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as f:
        return tomllib.load(f)


def _dataclass_from_sections(cls, *sections: dict):
    """把 TOML section 合并后，只取目标 dataclass 支持的字段。"""
    merged = {}
    for section in sections:
        if section:
            merged.update(section)
    field_names = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in merged.items() if key in field_names})


def _resolve_config(args):
    """生成实验配置和训练配置。

    配置文件提供默认值，命令行参数只覆盖用户在命令中明确传入的常用训练项。
    """
    raw_config = _read_toml_config(getattr(args, "config", None))

    experiment_section = raw_config.get("experiment", {})
    model_section = raw_config.get("model", {})
    dsra_section = raw_config.get("dsra", {})
    data_section = raw_config.get("data", {})
    training_section = raw_config.get("training", {})
    device_section = raw_config.get("device", {})

    config = _dataclass_from_sections(
        AxonExperimentConfig,
        experiment_section,
        model_section,
        data_section,
        device_section,
    )
    train_config = _dataclass_from_sections(TrainingConfig, training_section)

    if dsra_section:
        config.dsra_arch_config = _dataclass_from_sections(
            DSRAArchitectureConfig,
            {
                "dim": config.dsra_dim,
                "heads": config.dsra_heads,
                "slots": config.dsra_slots,
                "read_topk": config.dsra_read_topk,
                "write_topk": config.dsra_write_topk,
                "local_window": config.dsra_local_window,
            },
            dsra_section,
        )

    augmentation_section = raw_config.get("augmentation", {})
    if augmentation_section:
        config.augmentation = DataAugmentationConfig(**augmentation_section)

    if "name" in experiment_section:
        config.experiment_name = experiment_section["name"]
    if "device" in device_section:
        config.device = device_section["device"]
    if "output_dir" in data_section:
        config.model_save_dir = Path(data_section["output_dir"])
    if "log_dir" in data_section:
        config.log_dir = Path(data_section["log_dir"])
    if "eval_interval" in data_section:
        train_config.eval_interval = data_section["eval_interval"]

    return config, train_config


def _cache_dir_from_config(config: AxonExperimentConfig) -> Optional[str]:
    cache_dir = getattr(config, "cache_dir", None)
    return str(cache_dir) if cache_dir else None


def _stat_feature_metadata(config: AxonExperimentConfig):
    """生成统计特征 metadata，便于重要性报告可读。"""
    names = [
        "byte_mean",
        "byte_std",
        "byte_min",
        "byte_max",
        "byte_median",
        "byte_q25",
        "byte_q75",
        "count_0x00",
        "count_0xff",
        "count_0x90",
        "count_ascii_printable",
        "byte_entropy",
    ]

    for idx in range(config.stat_segment_count):
        names.extend([
            f"segment_{idx}_mean",
            f"segment_{idx}_std",
            f"segment_{idx}_entropy",
        ])
    for idx in range(config.stat_chunk_count):
        names.append(f"chunk_{idx}_mean")
    for idx in range(config.stat_chunk_count):
        names.append(f"chunk_{idx}_std")
    names.extend([
        "chunk_mean_abs_diff_mean",
        "chunk_mean_diff_std",
        "chunk_mean_diff_max",
        "chunk_mean_diff_min",
        "chunk_std_abs_diff_mean",
        "chunk_std_diff_std",
        "chunk_std_diff_max",
        "chunk_std_diff_min",
    ])

    names = names[:config.stat_feature_dim] + [
        f"stat_feature_{idx:04d}" for idx in range(len(names), config.stat_feature_dim)
    ]
    return [
        {
            "index": idx,
            "name": name,
            "group": "statistical",
            "stable_global": True,
            "stable_in_analyzed_data": True,
            "possible_names": [name],
            "note": "Stable statistical feature generated by extract_statistical_features().",
        }
        for idx, name in enumerate(names)
    ]


def _pe_fixed_feature_metadata():
    names = [
        ("file_size", "file"),
        ("log_file_size", "file"),
        ("optional_header_size", "pe_header"),
        ("header_size_ratio", "pe_header"),
        ("subsystem", "pe_header"),
        ("dll_characteristics", "pe_header"),
        ("checksum", "pe_header"),
        ("checksum_is_zero", "pe_header"),
        ("aslr_flag", "security_flags"),
        ("nx_flag", "security_flags"),
        ("cfg_flag", "security_flags"),
        ("file_characteristic_0x0004", "security_flags"),
        ("has_debug_directory", "data_directories"),
        ("has_base_reloc_directory", "data_directories"),
        ("has_tls_directory", "data_directories"),
        ("has_exception_directory", "data_directories"),
        ("has_security_directory", "data_directories"),
        ("section_count", "section"),
    ]
    return [
        {
            "index": idx,
            "name": name,
            "group": group,
            "stable_global": True,
            "stable_in_analyzed_data": True,
            "possible_names": [name],
            "note": "Stable PE feature generated before the dynamic section loop.",
        }
        for idx, (name, group) in enumerate(names)
    ]


def _pe_aggregate_feature_names():
    return [
        ("section_entropy_max", "section_entropy"),
        ("section_entropy_min", "section_entropy"),
        ("section_entropy_mean", "section_entropy"),
        ("section_entropy_std", "section_entropy"),
        ("section_high_entropy_ratio", "section_entropy"),
        ("section_raw_size_total", "section_size"),
        ("section_virtual_size_total", "section_size"),
        ("section_raw_size_mean", "section_size"),
        ("section_virtual_size_mean", "section_size"),
        ("section_raw_size_min", "section_size"),
        ("section_raw_size_max", "section_size"),
        ("section_raw_size_std", "section_size"),
        ("section_raw_size_cv", "section_size"),
        ("section_name_count", "section_name"),
        ("section_name_length_mean", "section_name"),
        ("section_name_length_max", "section_name"),
        ("section_name_length_min", "section_name"),
        ("section_large_raw_count", "section_size_anomaly"),
        ("section_large_raw_ratio", "section_size_anomaly"),
        ("section_small_raw_count", "section_size_anomaly"),
        ("section_small_raw_ratio", "section_size_anomaly"),
        ("import_api_network_ratio", "import_api_category"),
        ("import_api_process_ratio", "import_api_category"),
        ("import_api_filesystem_ratio", "import_api_category"),
        ("import_api_registry_ratio", "import_api_category"),
        ("import_api_crypto_ratio", "import_api_category"),
        ("import_api_injection_ratio", "import_api_category"),
        ("packer_section_name_hits", "packer"),
        ("packer_section_name_hit_ratio", "packer"),
    ]


def _pe_fixed_v2_feature_metadata(config: AxonExperimentConfig):
    metadata = _pe_fixed_feature_metadata()
    section_slots = getattr(config, "pe_fixed_section_slots", 32)
    for slot in range(section_slots):
        for attr in ["is_executable", "is_writable", "is_readable"]:
            idx = len(metadata)
            name = f"section_slot_{slot}_{attr}"
            metadata.append({
                "index": idx,
                "name": name,
                "group": "section_flags_fixed",
                "stable_global": True,
                "stable_in_analyzed_data": True,
                "possible_names": [name],
                "note": "Fixed PE schema section flag slot. Section slots beyond the file section count are zero.",
            })

    for name, group in _pe_aggregate_feature_names():
        idx = len(metadata)
        metadata.append({
            "index": idx,
            "name": name,
            "group": group,
            "stable_global": True,
            "stable_in_analyzed_data": True,
            "possible_names": [name],
            "note": "Fixed PE schema aggregate feature with stable index.",
        })

    while len(metadata) < config.pe_feature_dim:
        idx = len(metadata)
        metadata.append({
            "index": idx,
            "name": "zero_padding_after_written_pe_features",
            "group": "padding",
            "stable_global": True,
            "stable_in_analyzed_data": True,
            "possible_names": ["zero_padding_after_written_pe_features"],
            "note": "Unused fixed PE schema padding dimension.",
        })

    return metadata[:config.pe_feature_dim]


def _pe_semantic_for_section_count(index: int, section_count: int):
    fixed_metadata = _pe_fixed_feature_metadata()
    if index < len(fixed_metadata):
        item = fixed_metadata[index]
        return item["name"], item["group"]

    section_attr_start = len(fixed_metadata)
    section_attr_end = section_attr_start + max(section_count, 0) * 3
    if section_attr_start <= index < section_attr_end:
        section_idx = (index - section_attr_start) // 3
        attr_idx = (index - section_attr_start) % 3
        attr = ["is_executable", "is_writable", "is_readable"][attr_idx]
        return f"section_{section_idx}_{attr}", "section_flags_dynamic"

    aggregate_offset = index - section_attr_end
    aggregate_names = _pe_aggregate_feature_names()
    if 0 <= aggregate_offset < len(aggregate_names):
        return aggregate_names[aggregate_offset]

    return "zero_padding_after_written_pe_features", "padding"


def _pe_feature_metadata(config: AxonExperimentConfig, observed_section_counts=None):
    """建立 PE 特征 index 到实际含义的映射。

    注意：extract() 里 section flags 是按实际 section 数动态写入的，
    因此 index >= 18 的全局语义会随 section_count 偏移。
    """
    if getattr(config, "pe_schema_version", "legacy_dynamic") == "fixed_v2":
        return _pe_fixed_v2_feature_metadata(config)

    observed_counts = sorted({int(max(0, count)) for count in (observed_section_counts or [])})
    fixed_metadata = _pe_fixed_feature_metadata()
    metadata = []

    for index in range(config.pe_feature_dim):
        if index < len(fixed_metadata):
            metadata.append(fixed_metadata[index])
            continue

        possible = []
        possible_groups = []
        for section_count in observed_counts:
            name, group = _pe_semantic_for_section_count(index, section_count)
            if name not in possible:
                possible.append(name)
            if group not in possible_groups:
                possible_groups.append(group)

        if not observed_counts:
            name = f"pe_dynamic_index_{index:04d}"
            group = "dynamic_unknown"
            stable_in_analyzed_data = False
            note = (
                "No observed section_count values were provided. This index is after the "
                "dynamic section loop, so its semantic meaning depends on each file's section_count."
            )
            possible = []
        elif len(possible) == 1:
            name = possible[0]
            group = possible_groups[0]
            stable_in_analyzed_data = True
            note = (
                "Stable for the analyzed sample set, but globally dynamic because section flags "
                "shift later aggregate feature positions."
            )
        else:
            name = f"pe_dynamic_mixed_index_{index:04d}"
            group = "dynamic_mixed"
            stable_in_analyzed_data = False
            note = (
                "This index maps to multiple meanings in the analyzed sample set because PE files "
                "have different section_count values. Do not prune this index alone without a "
                "fixed-layout feature extractor or an ablation test."
            )

        metadata.append({
            "index": index,
            "name": name,
            "group": group,
            "stable_global": False,
            "stable_in_analyzed_data": stable_in_analyzed_data,
            "possible_names": possible[:12],
            "possible_name_count": len(possible),
            "observed_section_counts": observed_counts[:32],
            "observed_section_count_total": len(observed_counts),
            "note": note,
        })

    return metadata


def _rank_feature_scores(scores: torch.Tensor, metadata, feature_type: str, top_k: int, reverse: bool = True):
    order = torch.argsort(scores, descending=reverse).detach().cpu().tolist()
    rows = []
    for rank, index in enumerate(order[:top_k], start=1):
        feature_meta = metadata[index] if index < len(metadata) else {
            "name": f"{feature_type}_{index:04d}",
            "group": "unknown",
            "stable_global": False,
            "stable_in_analyzed_data": False,
            "possible_names": [],
            "note": "No metadata available.",
        }
        rows.append({
            "rank": rank,
            "feature_type": feature_type,
            "index": int(index),
            "name": feature_meta["name"],
            "group": feature_meta.get("group", "unknown"),
            "stable_global": bool(feature_meta.get("stable_global", False)),
            "stable_in_analyzed_data": bool(feature_meta.get("stable_in_analyzed_data", False)),
            "possible_names": feature_meta.get("possible_names", []),
            "note": feature_meta.get("note", ""),
            "importance": float(scores[index].detach().cpu().item()),
        })
    return rows


def _build_feature_importance_loader(args, config: AxonExperimentConfig, train_config: TrainingConfig):
    data_dir = args.data_dir or config.data_dir
    if data_dir is None:
        raise ValueError("Feature importance requires --data-dir or config data_dir")

    samples_per_class = args.samples_per_class
    if samples_per_class is None:
        samples_per_class = getattr(config, "fast_mode_samples", None)
    if samples_per_class is not None and samples_per_class <= 0:
        samples_per_class = None

    split = args.split
    npz_split_dir = Path(data_dir) / split
    if split != "all" and npz_split_dir.exists():
        data_loader = NPZDataLoader(
            data_dir=data_dir,
            batch_size=train_config.batch_size,
            max_byte_length=config.max_byte_length,
            pe_feature_dim=config.pe_feature_dim,
            stat_feature_dim=config.stat_feature_dim,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            shuffle=False,
            max_samples_per_class=samples_per_class,
            allow_raw_fallback=False,
        )
        return data_loader.create_dataloader(split)

    dataset = FeatureCacheDataset(
        data_dir=data_dir,
        cache_dir=_cache_dir_from_config(config),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=samples_per_class,
        axon_config=config,
    )
    if split == "all":
        selected_dataset = dataset
    else:
        train_dataset, val_dataset, test_dataset = create_stratified_split(dataset, axon_config=config)
        selected_dataset = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }[split]

    return torch.utils.data.DataLoader(
        selected_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    )


def _raw_eval_dataset_for_split(dataset, split: str, config: AxonExperimentConfig, split_file: Optional[str] = None):
    if split == "all":
        return dataset
    if split_file:
        train_dataset, val_dataset, test_dataset = create_split_from_file(
            dataset,
            Path(split_file),
            require_explicit_metadata=True,
        )
    else:
        train_dataset, val_dataset, test_dataset = create_stratified_split(dataset, axon_config=config)
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }[split]


def _limit_dataset_stratified(dataset, max_samples: Optional[int]):
    """按标签均衡截取一个小评估集，避免只截到单一类别。"""
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset

    labels = getattr(dataset, "label_list", None)
    if labels is None and hasattr(dataset, "indices") and hasattr(dataset, "base_dataset"):
        base_labels = getattr(dataset.base_dataset, "label_list", None)
        if base_labels is not None:
            labels = [base_labels[i] for i in dataset.indices]

    if labels is None:
        return torch.utils.data.Subset(dataset, list(range(max_samples)))

    by_label = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)

    if len(by_label) < 2:
        return torch.utils.data.Subset(dataset, list(range(max_samples)))

    per_label = max(1, max_samples // len(by_label))
    selected = []
    for label in sorted(by_label):
        selected.extend(by_label[label][:per_label])

    remaining = max_samples - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        for index in range(len(labels)):
            if index not in selected_set:
                selected.append(index)
                if len(selected) >= max_samples:
                    break

    selected = sorted(selected[:max_samples])
    return torch.utils.data.Subset(dataset, selected)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Axon v2.6 Malware Detection Training and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    def add_resource_guard_args(command_parser):
        command_parser.add_argument(
            '--resource-guard-json',
            type=str,
            default=None,
            help='Required for heavy commands: JSON receipt from scripts/pre_run_resource_leak_guard.py',
        )
        command_parser.add_argument(
            '--resource-guard-max-age-seconds',
            type=float,
            default=3600.0,
            help='Maximum accepted age for --resource-guard-json',
        )

    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train the model')
    add_resource_guard_args(train_parser)
    train_parser.add_argument('--config', type=str, default=None,
                              help='Path to TOML config file')
    train_parser.add_argument('--data-dir', type=str, default=None,
                              help='Training data directory; defaults to config data_dir')
    train_parser.add_argument('--samples-per-class', type=int, default=None,
                              help='Maximum raw samples to scan per class')
    train_parser.add_argument('--epochs', type=int, default=None,
                              help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=None,
                              help='Batch size')
    train_parser.add_argument('--lr', type=float, default=None,
                              help='Learning rate')
    train_parser.add_argument('--device', type=str, default=None,
                              choices=['cuda', 'cpu'],
                              help='Device to use')
    train_parser.add_argument('--output-dir', type=str, default=None,
                              help='Output directory for models')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Resume from checkpoint')
    train_parser.add_argument('--init-checkpoint', type=str, default=None,
                              help='Initialize model weights from a checkpoint but restart optimizer/scheduler')
    train_parser.add_argument('--partial-init', action='store_true', default=False,
                              help='Allow non-strict model weight initialization from --init-checkpoint')
    train_parser.add_argument('--fast', action='store_true', default=False,
                              help='Enable fast training mode for testing (uses small subset of data)')
    train_parser.add_argument('--fp16', action='store_true', default=False,
                              help='Enable mixed precision training (AMP)')
    train_parser.add_argument('--enable-swanlab', action='store_true', default=False,
                              help='Explicitly enable SwanLab experiment tracking')
    train_parser.add_argument('--extract-workers', type=int, default=1,
                              help='Number of workers used to prepare raw-file feature cache')
    train_parser.add_argument('--extract-backend', type=str, default='thread',
                              choices=['thread', 'process'],
                              help='Parallel backend for raw-file feature cache preparation')
    train_parser.add_argument('--split-file', type=str, default=None,
                              help='CSV split assignment file, e.g. reports/raw_group_diagnostics/group_isolated_split.csv')
    train_parser.add_argument('--rare-group-weighting', action='store_true', default=False,
                              help='Increase training loss weight for singleton and rare similarity groups')
    train_parser.add_argument('--singleton-group-weight', type=float, default=None,
                              help='Training weight for group_size=1 samples when --rare-group-weighting is enabled')
    train_parser.add_argument('--rare-group-weight', type=float, default=None,
                              help='Training weight for group_size=2..5 samples when --rare-group-weighting is enabled')
    train_parser.add_argument('--medium-group-weight', type=float, default=None,
                              help='Training weight for group_size=6..20 samples when --rare-group-weighting is enabled')
    train_parser.add_argument('--skip-test-eval', action='store_true', default=False,
                              help='Skip automatic test-set evaluation after training; use a separate eval/export command instead')

    # 评估命令
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    add_resource_guard_args(eval_parser)
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')
    eval_parser.add_argument('--data-dir', type=str, required=True,
                            help='Evaluation data directory')
    eval_parser.add_argument('--batch-size', type=int, default=16,
                            help='Batch size')
    eval_parser.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'],
                            help='Device to use')
    eval_parser.add_argument('--output', type=str, default='eval_report.json',
                            help='Output report file')
    eval_parser.add_argument('--sweep-thresholds', type=str, default=None,
                            help='Comma-separated decision thresholds to evaluate, e.g. 0.42,0.45,0.48,0.50')
    eval_parser.add_argument('--decision-threshold', type=float, default=None,
                            help='Override checkpoint decision threshold for the main evaluation metrics')
    eval_parser.add_argument('--split', type=str, default='test',
                            choices=['train', 'val', 'test', 'all'],
                            help='Raw-file split to evaluate when NPZ split files are unavailable')
    eval_parser.add_argument('--samples-per-class', type=int, default=None,
                            help='Maximum valid raw samples per class to scan; defaults to checkpoint fast_mode_samples when checkpoint was trained in fast mode')
    eval_parser.add_argument('--max-eval-samples', type=int, default=None,
                            help='Maximum already-split evaluation samples to use; useful for quick threshold sweeps')
    eval_parser.add_argument('--split-file', type=str, default=None,
                            help='CSV split assignment file used when evaluating cache/raw samples')
    eval_parser.add_argument('--feature-mask', type=str, default=None,
                            help='Path to exported feature mask JSON applied before evaluation')

    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='Predict on a file')
    add_resource_guard_args(predict_parser)
    predict_parser.add_argument('--file', type=str, required=True,
                               help='File to predict')
    predict_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Path to model checkpoint')
    predict_parser.add_argument('--device', type=str, default='cuda',
                               choices=['cuda', 'cpu'],
                               help='Device to use')
    predict_parser.add_argument('--feature-mask', type=str, default=None,
                               help='Path to exported feature mask JSON applied before prediction')
    predict_parser.add_argument('--scan-nested', action='store_true', default=False,
                               help='Scan zip/7z/cab/msi containers and predict inner PE files; rar is detected but not extracted')
    predict_parser.add_argument('--archive-scanner', type=str, default=None,
                               help='Path to axon-archive-scanner binary; defaults to tools/archive_scanner target')
    predict_parser.add_argument('--archive-max-depth', type=int, default=4,
                               help='Maximum nested archive depth')
    predict_parser.add_argument('--archive-max-files', type=int, default=4096,
                               help='Maximum total entries observed during nested scan')
    predict_parser.add_argument('--archive-max-total-bytes', type=int, default=512 * 1024 * 1024,
                               help='Maximum total extracted bytes across nested scan')
    predict_parser.add_argument('--archive-max-file-bytes', type=int, default=128 * 1024 * 1024,
                               help='Maximum single extracted file size')

    # 特征提取命令
    extract_parser = subparsers.add_parser('extract', help='Extract features')
    add_resource_guard_args(extract_parser)
    extract_parser.add_argument('--data-dir', type=str, required=True,
                               help='Input data directory')
    extract_parser.add_argument('--output-dir', type=str, required=True,
                               help='Output directory for extracted features')
    extract_parser.add_argument('--max-workers', type=int, default=4,
                               help='Number of parallel workers')

    # 特征重要性命令
    importance_parser = subparsers.add_parser('importance', help='Rank PE/stat feature importance from a checkpoint')
    add_resource_guard_args(importance_parser)
    importance_parser.add_argument('--checkpoint', type=str, required=True,
                                   help='Path to model checkpoint')
    importance_parser.add_argument('--data-dir', type=str, default=None,
                                   help='Data directory; defaults to checkpoint config data_dir')
    importance_parser.add_argument('--split', type=str, default='val',
                                   choices=['train', 'val', 'test', 'all'],
                                   help='Dataset split used for importance analysis')
    importance_parser.add_argument('--samples-per-class', type=int, default=200,
                                   help='Maximum valid samples per class for raw-file analysis; use 0 for no limit')
    importance_parser.add_argument('--batch-size', type=int, default=16,
                                   help='Batch size')
    importance_parser.add_argument('--max-batches', type=int, default=50,
                                   help='Maximum batches to analyze; use 0 for no limit')
    importance_parser.add_argument('--top-k', type=int, default=40,
                                   help='Number of high/low importance features to report per group')
    importance_parser.add_argument('--device', type=str, default='cuda',
                                   choices=['cuda', 'cpu'],
                                   help='Device to use')
    importance_parser.add_argument('--output-json', type=str, default='reports/feature_importance.json',
                                   help='Output JSON report path')
    importance_parser.add_argument('--output-csv', type=str, default='reports/feature_importance.csv',
                                   help='Output CSV report path')

    return parser.parse_args()


def _expected_resource_guard_command(argv: Optional[Sequence[str]] = None) -> list[str]:
    return [sys.executable, str(MAIN_SCRIPT_PATH), *(argv if argv is not None else sys.argv[1:])]


def _enforce_resource_guard(args, *, now: Optional[float] = None) -> None:
    command = getattr(args, "command", None)
    guard_required = command in RESOURCE_GUARD_REQUIRED_COMMANDS or (
        command == "predict" and bool(getattr(args, "scan_nested", False))
    )
    if not guard_required:
        return
    guard_json = getattr(args, "resource_guard_json", None)
    if not guard_json:
        raise SystemExit(
            f"{args.command} requires --resource-guard-json. "
            "Run scripts/pre_run_resource_leak_guard.py for scripts/main.py before this heavy command."
        )

    guard_path = Path(guard_json)
    if not guard_path.exists():
        raise SystemExit(f"Resource guard JSON not found: {guard_path}")
    try:
        payload = json.loads(guard_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid resource guard JSON: {guard_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Resource guard JSON must be an object: {guard_path}")

    validation = validate_guard_receipt(
        payload,
        expected_target_scripts=[MAIN_SCRIPT_PATH],
        expected_command=_expected_resource_guard_command(),
        max_age_seconds=float(getattr(args, "resource_guard_max_age_seconds", 3600.0)),
        now=now,
    )
    if not validation["valid"]:
        raise SystemExit(
            "Resource guard receipt rejected: "
            + ", ".join(str(item) for item in validation["failures"])
        )
    print(f"[Resource Guard] accepted receipt: {guard_path}")


def _enforce_eval_threshold_sweep_policy(args) -> None:
    if not getattr(args, "sweep_thresholds", None):
        return
    split = getattr(args, "split", "test")
    if split in {"test", "all"}:
        raise SystemExit(
            "Threshold sweep is blocked on test/all splits. "
            "Use --split val for threshold selection, then run test once with --decision-threshold."
        )


def train_command(args):
    """训练命令"""
    print("=" * 60)
    print("Axon v2.6 Training")
    if args.fast:
        print("[FAST MODE] Enabled - Testing with reduced samples and epochs")
    print("=" * 60)

    config, train_config = _resolve_config(args)
    _set_training_seed(config.seed)
    train_generator = _make_train_generator(config.seed)
    worker_init_fn = _seed_worker if train_config.num_workers > 0 else None
    config.experiment_name = config.experiment_name + ("_fast" if args.fast else "")
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        train_config.batch_size = args.batch_size
    else:
        config.batch_size = train_config.batch_size
    if args.device is not None:
        config.device = args.device
    if args.output_dir is not None:
        config.model_save_dir = Path(args.output_dir)
    config.fast_mode = args.fast
    if args.fast:
        config.max_byte_length = config.fast_mode_byte_length

    if args.lr is not None:
        train_config.learning_rate = args.lr
    if args.fp16:
        train_config.mixed_precision = True
    if args.enable_swanlab:
        train_config.enable_swanlab = True
    if args.fast:
        train_config.max_epochs = config.fast_mode_epochs
    if args.epochs is not None:
        train_config.max_epochs = args.epochs
    if args.rare_group_weighting:
        train_config.rare_group_weighting = True
    if args.singleton_group_weight is not None:
        train_config.singleton_group_weight = args.singleton_group_weight
    if args.rare_group_weight is not None:
        train_config.rare_group_weight = args.rare_group_weight
    if args.medium_group_weight is not None:
        train_config.medium_group_weight = args.medium_group_weight
    if train_config.lr_scheduler == "cosine" and train_config.warmup_epochs >= train_config.max_epochs:
        train_config.warmup_epochs = max(0, train_config.max_epochs - 1)

    data_dir = args.data_dir or config.data_dir
    if data_dir is None:
        raise ValueError("Training data directory is required via --data-dir or config data_dir")
    if not args.fast and not args.split_file:
        raise ValueError(
            "Strict training requires --split-file with explicit label/source_sha256 metadata; "
            "raw filename/directory label inference is only allowed for --fast development checks."
        )
    samples_per_class = args.samples_per_class

    batch_size = train_config.batch_size

    # 创建模型
    print("\nInitializing model...")
    model = AxonMalwareModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 加载数据
    print("\nLoading data...")

    if args.fast:
        # 快速模式：创建一个限量数据集，再按索引划分，避免数据泄露
        per_class = samples_per_class or config.fast_mode_samples
        print(f"[FAST MODE] Creating limited dataset with {per_class} samples per class...")

        full_dataset = MalwareDataset(
            data_dir=data_dir,
            cache_dir=_cache_dir_from_config(config),
            max_byte_length=config.max_byte_length,
            pe_feature_dim=config.pe_feature_dim,
            stat_feature_dim=config.stat_feature_dim,
            max_samples_per_class=per_class,
            max_file_size=config.max_file_size,
            axon_config=config,
            extraction_workers=args.extract_workers,
            extraction_backend=args.extract_backend,
        )

        if args.split_file:
            train_dataset, val_dataset, test_dataset = create_split_from_file(
                full_dataset,
                Path(args.split_file),
                rare_group_weighting=train_config.rare_group_weighting,
                singleton_group_weight=train_config.singleton_group_weight,
                rare_group_weight=train_config.rare_group_weight,
                medium_group_weight=train_config.medium_group_weight,
                require_explicit_metadata=True,
            )
        else:
            train_dataset, val_dataset, test_dataset = create_stratified_split(
                full_dataset, axon_config=config
            )

        # 应用数据增强（仅训练集）
        if config.augmentation is not None and config.augmentation.enable:
            train_dataset = AugmentedDataset(train_dataset, config.augmentation)
            print(f"[Augmentation] Enabled: byte_dropout={config.augmentation.byte_dropout}, "
                  f"byte_noise={config.augmentation.byte_noise}, "
                  f"feature_noise={config.augmentation.feature_noise}")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=train_generator,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        print(f"[FAST MODE] Training samples: {len(train_dataset)}")
        print(f"[FAST MODE] Validation samples: {len(val_dataset)}")
        print(f"[FAST MODE] Test samples: {len(test_dataset)}")
    else:
        if args.split_file:
            print(f"[Split File] Using group-isolated split file: {args.split_file}")
            dataset = FeatureCacheDataset(
                data_dir=data_dir,
                cache_dir=_cache_dir_from_config(config),
                max_byte_length=config.max_byte_length,
                pe_feature_dim=config.pe_feature_dim,
                stat_feature_dim=config.stat_feature_dim,
                max_samples_per_class=samples_per_class,
                require_manifest=True,
                axon_config=config,
            )

            train_dataset, val_dataset, test_dataset = create_split_from_file(
                dataset,
                Path(args.split_file),
                rare_group_weighting=train_config.rare_group_weighting,
                singleton_group_weight=train_config.singleton_group_weight,
                rare_group_weight=train_config.rare_group_weight,
                medium_group_weight=train_config.medium_group_weight,
                require_explicit_metadata=True,
            )

            # 应用数据增强（仅训练集）
            if config.augmentation is not None and config.augmentation.enable:
                train_dataset = AugmentedDataset(train_dataset, config.augmentation)
                print(f"[Augmentation] Enabled: byte_dropout={config.augmentation.byte_dropout}, "
                      f"byte_noise={config.augmentation.byte_noise}, "
                      f"feature_noise={config.augmentation.feature_noise}")

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=train_config.num_workers,
                generator=train_generator,
                worker_init_fn=worker_init_fn,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=train_config.num_workers
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=train_config.num_workers
            )

            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
        else:
            # 正常模式
            data_loader = NPZDataLoader(
                data_dir=data_dir,
                batch_size=batch_size,
                max_byte_length=config.max_byte_length,
                pe_feature_dim=config.pe_feature_dim,
                stat_feature_dim=config.stat_feature_dim,
                num_workers=train_config.num_workers,
                max_samples_per_class=samples_per_class,
            )

            train_loader = None
            val_loader = None
            test_loader = None

            try:
                train_loader = data_loader.get_train_loader()
                val_loader = data_loader.get_val_loader()
                test_loader = data_loader.get_test_loader()
                print(f"Training samples: {len(train_loader.dataset)}")
                print(f"Validation samples: {len(val_loader.dataset)}")
                print(f"Test samples: {len(test_loader.dataset)}")
            except Exception as e:
                print(f"[Error] Failed to load data: {e}")
                print("Creating dataset from raw files...")

                dataset = MalwareDataset(
                    data_dir=data_dir,
                    cache_dir=_cache_dir_from_config(config),
                    max_byte_length=config.max_byte_length,
                    pe_feature_dim=config.pe_feature_dim,
                    stat_feature_dim=config.stat_feature_dim,
                    max_samples_per_class=samples_per_class,
                    max_file_size=config.max_file_size,
                    axon_config=config,
                    extraction_workers=args.extract_workers,
                    extraction_backend=args.extract_backend,
                )

                train_dataset, val_dataset, test_dataset = create_stratified_split(
                    dataset, axon_config=config
                )

                # 应用数据增强（仅训练集）
                if config.augmentation is not None and config.augmentation.enable:
                    train_dataset = AugmentedDataset(train_dataset, config.augmentation)
                    print(f"[Augmentation] Enabled: byte_dropout={config.augmentation.byte_dropout}, "
                          f"byte_noise={config.augmentation.byte_noise}, "
                          f"feature_noise={config.augmentation.feature_noise}")

                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=train_config.num_workers,
                    generator=train_generator,
                    worker_init_fn=worker_init_fn,
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False, num_workers=train_config.num_workers
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False, num_workers=train_config.num_workers
                )

                print(f"Training samples: {len(train_dataset)}")
                print(f"Validation samples: {len(val_dataset)}")
                print(f"Test samples: {len(test_dataset)}")

    # 创建训练器
    trainer = AxonTrainer(model, config, train_config)

    if args.init_checkpoint:
        if args.resume:
            raise ValueError("--init-checkpoint and --resume cannot be used together")
        init_path = Path(args.init_checkpoint)
        print(f"\nInitializing model weights from checkpoint: {init_path}")
        checkpoint = load_safe_checkpoint(init_path, map_location="cpu")
        init_state_dict = checkpoint["model_state_dict"]
        shape_mismatches = []
        adapted_shapes = []
        if args.partial_init:
            init_state_dict, shape_mismatches, adapted_shapes = _filter_partial_init_state_dict(model, init_state_dict)
        load_result = model.load_state_dict(init_state_dict, strict=not args.partial_init)
        if args.partial_init:
            print(
                "[Partial init] "
                f"missing_keys={len(load_result.missing_keys)}, "
                f"unexpected_keys={len(load_result.unexpected_keys)}, "
                f"adapted_shapes={len(adapted_shapes)}, "
                f"shape_mismatches={len(shape_mismatches)}"
            )
            for key, checkpoint_shape, model_shape in adapted_shapes[:20]:
                print(
                    "[Partial init] adapted_widened_shape "
                    f"{key}: checkpoint={checkpoint_shape}, model={model_shape}"
                )
            if len(adapted_shapes) > 20:
                print(f"[Partial init] adapted_widened_shape ... {len(adapted_shapes) - 20} more")
            for key, checkpoint_shape, model_shape in shape_mismatches[:20]:
                print(
                    "[Partial init] skipped_shape_mismatch "
                    f"{key}: checkpoint={checkpoint_shape}, model={model_shape}"
                )
            if len(shape_mismatches) > 20:
                print(f"[Partial init] skipped_shape_mismatch ... {len(shape_mismatches) - 20} more")
        del init_state_dict
        del checkpoint
        if trainer.device.type == "cuda":
            torch.cuda.empty_cache()

    # 恢复检查点（如果指定）
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # 训练
    print("\nStarting training...")
    results = trainer.train(
        train_loader,
        val_loader,
        test_loader=None if args.skip_test_eval else test_loader,
        fast_mode=args.fast,
    )

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best F1 Score: {trainer.best_f1:.4f}")
    print(f"Best Epoch: {trainer.best_epoch}")
    print(f"Model saved to: {config.model_save_dir}")

    return results


def eval_command(args):
    """评估命令"""
    _enforce_eval_threshold_sweep_policy(args)
    print("=" * 60)
    print("Axon v2.6 Evaluation")
    print("=" * 60)

    # 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = load_safe_checkpoint(checkpoint_path, map_location="cpu")
    config = AxonExperimentConfig.from_dict(checkpoint['config'])
    saved_train_config = checkpoint.get('train_config', {})
    if saved_train_config:
        saved_train_config['batch_size'] = args.batch_size
        if args.decision_threshold is not None:
            saved_train_config['decision_threshold'] = args.decision_threshold
    eval_train_config = TrainingConfig(**saved_train_config) if saved_train_config else TrainingConfig(batch_size=args.batch_size)
    if args.decision_threshold is not None:
        eval_train_config.decision_threshold = args.decision_threshold
    # Evaluation must not allocate training-only shadow models from checkpoint config.
    eval_train_config.use_ema = False
    eval_train_config.use_swa = False
    eval_train_config.enable_swanlab = False

    # 创建模型
    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model.to(args.device)
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    feature_mask_payload = None
    if args.feature_mask:
        mask_data = _load_feature_mask_tensors(args.feature_mask, config, args.device)
        if mask_data is not None:
            pe_mask, stat_mask, feature_mask_payload = mask_data
            model = FeatureMaskedModel(model, pe_mask, stat_mask).to(args.device)
            model.eval()
            print(f"[Feature Mask] {args.feature_mask} ({_summarize_feature_mask(feature_mask_payload)})")

    # 加载数据
    print("\nLoading evaluation data...")
    samples_per_class = args.samples_per_class
    if samples_per_class is None and getattr(config, 'fast_mode', False):
        samples_per_class = getattr(config, 'fast_mode_samples', None)
    if samples_per_class is not None and samples_per_class <= 0:
        samples_per_class = None

    if args.split_file:
        test_loader = None
    else:
        data_loader = NPZDataLoader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            max_byte_length=config.max_byte_length,
            pe_feature_dim=config.pe_feature_dim,
            stat_feature_dim=config.stat_feature_dim,
            num_workers=eval_train_config.num_workers,
            shuffle=False,
            allow_raw_fallback=False,
        )
        try:
            test_loader = data_loader.create_dataloader(args.split)
        except Exception:
            test_loader = None

    if test_loader is None:
        if samples_per_class is not None:
            print(f"[Eval] Limiting cache samples to {samples_per_class} valid samples per class")
        dataset = FeatureCacheDataset(
            data_dir=args.data_dir,
            cache_dir=_cache_dir_from_config(config),
            max_byte_length=config.max_byte_length,
            pe_feature_dim=config.pe_feature_dim,
            stat_feature_dim=config.stat_feature_dim,
            max_samples_per_class=samples_per_class,
            require_manifest=bool(args.split_file),
            axon_config=config,
        )
        selected_dataset = _raw_eval_dataset_for_split(dataset, args.split, config, split_file=args.split_file)
        selected_dataset = _limit_dataset_stratified(selected_dataset, args.max_eval_samples)
        test_loader = torch.utils.data.DataLoader(
            selected_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=eval_train_config.num_workers,
        )
    print(f"[Eval] Evaluation samples: {len(test_loader.dataset)}")

    # 评估
    print("\nEvaluating...")
    trainer = AxonTrainer(model, config, train_config=eval_train_config, device=torch.device(args.device))
    results = trainer.evaluate(test_loader, 0, "test")
    sweep_results = None
    if args.sweep_thresholds:
        thresholds = [float(item.strip()) for item in args.sweep_thresholds.split(',') if item.strip()]
        sweep_results = trainer.threshold_sweep(test_loader, thresholds, epoch=0, phase="test")

    # 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Loss: {results.loss:.4f}")
    print(f"Accuracy: {results.accuracy:.4f}")
    print(f"Precision: {results.precision:.4f}")
    print(f"Recall: {results.recall:.4f}")
    print(f"F1 Score: {results.f1:.4f}")
    print(f"True Positive: {results.true_positive}")
    print(f"True Negative: {results.true_negative}")
    print(f"False Positive: {results.false_positive}")
    print(f"False Negative: {results.false_negative}")
    print(f"False Positive Rate: {results.false_positive_rate:.4f}")
    print(f"False Negative Rate: {results.false_negative_rate:.4f}")
    if results.auc is not None:
        print(f"AUC: {results.auc:.4f}")
    if sweep_results is not None:
        print("\nThreshold Sweep")
        print("threshold | accuracy | precision | recall | f1 | fp | fn | fp_rate | fn_rate | auc")
        for row in sweep_results:
            auc = row['auc'] if row['auc'] is not None else 0.0
            print(
                f"{row['threshold']:.3f} | {row['accuracy']:.4f} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['false_positive']} | "
                f"{row['false_negative']} | {row['false_positive_rate']:.4f} | "
                f"{row['false_negative_rate']:.4f} | {auc:.4f}"
            )

    # 保存报告
    import json
    report = {
        'loss': float(results.loss),
        'accuracy': float(results.accuracy),
        'precision': float(results.precision),
        'recall': float(results.recall),
        'f1': float(results.f1),
        'auc': float(results.auc) if results.auc is not None else None,
        'true_positive': int(results.true_positive),
        'true_negative': int(results.true_negative),
        'false_positive': int(results.false_positive),
        'false_negative': int(results.false_negative),
        'false_positive_rate': float(results.false_positive_rate),
        'false_negative_rate': float(results.false_negative_rate),
        'threshold_sweep': sweep_results,
        'feature_mask': args.feature_mask,
        'feature_mask_summary': (
            {
                'kept_total': feature_mask_payload.get('kept_total'),
                'kept_pe': feature_mask_payload.get('kept_pe'),
                'kept_stat': feature_mask_payload.get('kept_stat'),
                'note': feature_mask_payload.get('note'),
            }
            if feature_mask_payload is not None
            else None
        ),
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")


def _apply_feature_mask_to_tensors(
    pe_tensor: torch.Tensor,
    stat_tensor: torch.Tensor,
    feature_mask: Optional[tuple[torch.Tensor, torch.Tensor, dict]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_mask is None:
        return pe_tensor, stat_tensor
    pe_mask, stat_mask, _payload = feature_mask
    return (
        pe_tensor * pe_mask.to(device=pe_tensor.device, dtype=pe_tensor.dtype).view(1, -1),
        stat_tensor * stat_mask.to(device=stat_tensor.device, dtype=stat_tensor.dtype).view(1, -1),
    )


def _predict_pe_file(
    model,
    config,
    file_path: Path,
    device: str,
    feature_mask: Optional[tuple[torch.Tensor, torch.Tensor, dict]] = None,
) -> Optional[dict]:
    """对一个已经确定为 PE 的文件执行现有 Axon 预测流程。"""
    extraction_config = ExtractionConfig.from_axon_config(
        config,
        max_file_size=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim
    )

    byte_seq, pe_features, stat_features, _, _ = extract_all_features(
        str(file_path),
        extraction_config,
        axon_config=config,
        allow_pe_fallback=config.allow_pe_fallback,
    )

    if byte_seq is None or pe_features is None:
        return None

    byte_tensor = torch.from_numpy(byte_seq).long().unsqueeze(0).to(device)
    pe_tensor = torch.from_numpy(pe_features).float().unsqueeze(0).to(device)
    stat_tensor = torch.from_numpy(stat_features).float().unsqueeze(0).to(device)
    pe_tensor, stat_tensor = _apply_feature_mask_to_tensors(pe_tensor, stat_tensor, feature_mask)

    with torch.no_grad():
        logits = model(byte_tensor, pe_tensor, stat_features=stat_tensor)['logits']
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    return {
        "prediction": int(pred),
        "label": "Malicious" if pred == 1 else "Benign",
        "confidence": float(confidence),
        "prob_benign": float(probs[0, 0].item()),
        "prob_malicious": float(probs[0, 1].item()),
    }


def _load_prediction_model(checkpoint_path: Path, device: str):
    checkpoint = load_safe_checkpoint(checkpoint_path, map_location="cpu")
    config = AxonExperimentConfig.from_dict(checkpoint['config'])

    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model.to(device)
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    return model, config


def _predict_nested_archive(args, file_path: Path, checkpoint_path: Path):
    """扫描嵌套包并对内层 PE 做预测。

    运行时策略是：只要任一内层 PE 被判为恶意，外层包就报警。
    训练策略不同：报告中的 training_label_policy 会明确提示不要继承父包标签。
    """
    print("\nScanning nested archive/MSI contents...")
    scan_options = ArchiveScanOptions(
        max_depth=args.archive_max_depth,
        max_files=args.archive_max_files,
        max_total_bytes=args.archive_max_total_bytes,
        max_file_bytes=args.archive_max_file_bytes,
        keep_temp=True,
        scanner_binary=Path(args.archive_scanner) if args.archive_scanner else None,
    )
    report = None
    try:
        report = run_archive_scan(file_path, scan_options)
        pe_targets = list(iter_pe_prediction_targets(report))
        print(f"Nested entries: {report['summary']['total_entries']}")
        print(f"Axon PE prediction targets: {len(pe_targets)}")
        print("Training label policy: inner archive/MSI contents are unknown unless explicitly labeled.")

        if not pe_targets:
            print("[Warning] No inner PE files were found for Axon prediction.")
            return

        model, config = _load_prediction_model(checkpoint_path, args.device)
        feature_mask = _load_feature_mask_tensors(args.feature_mask, config, args.device)
        if feature_mask is not None:
            print(f"[Feature Mask] {args.feature_mask} ({_summarize_feature_mask(feature_mask[2])})")
        predictions = []

        for entry in pe_targets:
            inner_path = Path(entry["extracted_path"])
            result = _predict_pe_file(model, config, inner_path, args.device, feature_mask=feature_mask)
            if result is None:
                predictions.append({
                    "logical_path": entry["logical_path"],
                    "sha256": entry.get("sha256"),
                    "status": "feature_extraction_failed",
                })
                continue
            predictions.append({
                "logical_path": entry["logical_path"],
                "sha256": entry.get("sha256"),
                "status": "predicted",
                **result,
            })

        malicious_hits = [item for item in predictions if item.get("prediction") == 1]

        print("\n" + "=" * 60)
        print("Nested Prediction Results")
        print("=" * 60)
        print(f"File: {file_path.name}")
        print(f"Parent Verdict: {'Malicious' if malicious_hits else 'Benign/No malicious inner PE detected'}")
        print(f"Runtime rule: any malicious inner PE triggers parent alert")
        print(f"Inner PE predicted: {sum(1 for item in predictions if item.get('status') == 'predicted')}")
        print(f"Malicious inner PE: {len(malicious_hits)}")
        print("\nInner results:")
        for item in predictions:
            if item.get("status") != "predicted":
                print(f"- {item['logical_path']}: {item['status']}")
                continue
            print(
                f"- {item['logical_path']}: {item['label']} "
                f"(malicious={item['prob_malicious']:.4f}, benign={item['prob_benign']:.4f})"
            )
    finally:
        if report is not None:
            cleanup_scan_temp(report)


def predict_command(args):
    """预测命令"""
    print("=" * 60)
    print("Axon v2.6 Prediction")
    print("=" * 60)

    # 检查文件
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        return

    # 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return

    if getattr(args, "scan_nested", False):
        _predict_nested_archive(args, file_path, checkpoint_path)
        return

    model, config = _load_prediction_model(checkpoint_path, args.device)
    feature_mask = _load_feature_mask_tensors(args.feature_mask, config, args.device)
    if feature_mask is not None:
        print(f"[Feature Mask] {args.feature_mask} ({_summarize_feature_mask(feature_mask[2])})")

    # 提取特征
    print("\nExtracting features...")
    result = _predict_pe_file(model, config, file_path, args.device, feature_mask=feature_mask)
    if result is None:
        print("[Error] 无法可靠判断：PE 解析或特征提取失败，需要人工复核。")
        return

    # 输出结果
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"File: {file_path.name}")
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: Benign={result['prob_benign']:.4f}, Malicious={result['prob_malicious']:.4f}")


def extract_command(args):
    """特征提取命令"""
    print("=" * 60)
    print("Axon v2.6 Feature Extraction")
    print("=" * 60)

    input_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[Error] Input directory not found: {input_dir}")
        return

    # 统计
    total_files = 0
    success_files = 0
    failed_files = []

    # 遍历所有文件
    print(f"\nExtracting features from: {input_dir}")
    print(f"Output directory: {output_dir}")

    default_config = AxonExperimentConfig()
    extraction_config = ExtractionConfig.from_axon_config(
        default_config,
        max_file_size=default_config.extraction_max_file_size,
        pe_feature_dim=default_config.pe_feature_dim
    )

    skip_dirs = {".cache", "__pycache__", ".git", ".pytest_cache", "reports", "models", "swanlog"}
    input_root = input_dir.resolve(strict=False)

    def iter_input_files():
        for dirpath, dirnames, filenames in os.walk(input_dir, followlinks=False):
            dirnames[:] = sorted(dirname for dirname in dirnames if dirname not in skip_dirs)
            for filename in sorted(filenames):
                file_path = Path(dirpath) / filename
                if file_path.is_symlink() or not file_path.is_file():
                    continue
                yield file_path

    def output_name_for(file_path: Path) -> str:
        try:
            relative_text = file_path.resolve(strict=False).relative_to(input_root).as_posix()
        except ValueError:
            relative_text = file_path.resolve(strict=False).as_posix()
        path_hash = hashlib.sha256(relative_text.encode("utf-8")).hexdigest()[:12]
        return f"{path_hash}_{file_path.stem}.npz"

    for file_path in iter_input_files():

        total_files += 1
        try:
            # 提取特征
            byte_seq, pe_features, stat_features, _lightweight_features, orig_len = extract_all_features(
                str(file_path), extraction_config, axon_config=default_config
            )

            if byte_seq is None:
                raise ValueError("Feature extraction failed")

            # 保存到 NPZ
            output_file = output_dir / output_name_for(file_path)
            import numpy as np
            np.savez_compressed(
                output_file,
                byte_sequence=byte_seq,
                pe_features=pe_features,
                stat_features=stat_features,
                orig_length=orig_len
            )

            success_files += 1

            if total_files % 100 == 0:
                print(f"Processed: {total_files} files, Success: {success_files}")

        except Exception as e:
            failed_files.append((str(file_path), str(e)))

    # 打印统计
    print("\n" + "=" * 60)
    print("Extraction Complete")
    print("=" * 60)
    print(f"Total files: {total_files}")
    print(f"Success: {success_files}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for path, error in failed_files[:10]:
            print(f"  - {path}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")


def feature_importance_command(args):
    """基于输入梯度统计 PE/stat 特征重要性。"""
    print("=" * 60)
    print("Axon v2.6 Feature Importance")
    print("=" * 60)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    checkpoint = load_safe_checkpoint(checkpoint_path, map_location="cpu")
    config = AxonExperimentConfig.from_dict(checkpoint['config'])
    saved_train_config = checkpoint.get('train_config', {})
    if saved_train_config:
        saved_train_config['batch_size'] = args.batch_size
    train_config = TrainingConfig(**saved_train_config) if saved_train_config else TrainingConfig(batch_size=args.batch_size)
    train_config.batch_size = args.batch_size
    train_config.num_workers = 0

    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.requires_grad_(False)
    model.eval()

    loader = _build_feature_importance_loader(args, config, train_config)
    max_batches = None if args.max_batches == 0 else args.max_batches

    pe_scores = torch.zeros(config.pe_feature_dim, device=device)
    stat_scores = torch.zeros(config.stat_feature_dim, device=device)
    analyzed_batches = 0
    analyzed_samples = 0
    observed_section_counts = set()

    for batch_idx, (byte_seq, pe_features, stat_features, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        byte_seq = byte_seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pe_features = pe_features.to(device, non_blocking=True).detach().requires_grad_(True)
        stat_features = stat_features.to(device, non_blocking=True).detach().requires_grad_(True)

        outputs = model(byte_seq, pe_features, stat_features=stat_features)
        loss = F.cross_entropy(outputs['logits'], labels)
        pe_grad, stat_grad = torch.autograd.grad(
            loss,
            (pe_features, stat_features),
            retain_graph=False,
            create_graph=False,
        )

        pe_scores += (pe_grad.detach().abs() * pe_features.detach().abs()).sum(dim=0)
        stat_scores += (stat_grad.detach().abs() * stat_features.detach().abs()).sum(dim=0)
        if config.pe_feature_dim > 17:
            section_counts = pe_features.detach()[:, 17].round().clamp(min=0).long().cpu().tolist()
            observed_section_counts.update(int(count) for count in section_counts)
        analyzed_batches += 1
        analyzed_samples += int(labels.shape[0])

    if analyzed_batches == 0:
        raise ValueError("No batches were available for feature importance analysis")

    pe_scores = pe_scores / max(analyzed_samples, 1)
    stat_scores = stat_scores / max(analyzed_samples, 1)

    top_k = max(1, args.top_k)
    pe_metadata = _pe_feature_metadata(config, observed_section_counts)
    stat_metadata = _stat_feature_metadata(config)

    report = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "analyzed_samples": analyzed_samples,
        "analyzed_batches": analyzed_batches,
        "observed_section_counts": sorted(observed_section_counts),
        "score": "mean(abs(gradient) * abs(feature_value))",
        "pe_feature_metadata": pe_metadata,
        "stat_feature_metadata": stat_metadata,
        "pe_top": _rank_feature_scores(pe_scores, pe_metadata, "pe", top_k, reverse=True),
        "pe_low": _rank_feature_scores(pe_scores, pe_metadata, "pe", top_k, reverse=False),
        "stat_top": _rank_feature_scores(stat_scores, stat_metadata, "stat", top_k, reverse=True),
        "stat_low": _rank_feature_scores(stat_scores, stat_metadata, "stat", top_k, reverse=False),
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with output_csv.open('w', encoding='utf-8') as f:
        f.write("group,rank,feature_type,index,name,semantic_group,stable_global,stable_in_analyzed_data,possible_names,importance,note\n")
        for group in ["pe_top", "pe_low", "stat_top", "stat_low"]:
            for row in report[group]:
                name = str(row['name']).replace('"', '""')
                note = str(row.get('note', '')).replace('"', '""')
                possible_names = "|".join(str(v) for v in row.get('possible_names', [])).replace('"', '""')
                f.write(
                    f"{group},{row['rank']},{row['feature_type']},"
                    f"{row['index']},\"{name}\",\"{row.get('group', '')}\","
                    f"{row.get('stable_global', False)},{row.get('stable_in_analyzed_data', False)},"
                    f"\"{possible_names}\",{row['importance']:.12g},\"{note}\"\n"
                )

    print(f"Analyzed samples: {analyzed_samples}")
    print(f"Analyzed batches: {analyzed_batches}")
    print(f"JSON report: {output_json}")
    print(f"CSV report: {output_csv}")
    print("\nTop PE features:")
    for row in report["pe_top"][:10]:
        print(f"  #{row['rank']:02d} [{row['index']:04d}] {row['name']}: {row['importance']:.6g}")
    print("\nTop stat features:")
    for row in report["stat_top"][:10]:
        print(f"  #{row['rank']:02d} [{row['index']:04d}] {row['name']}: {row['importance']:.6g}")


def main():
    """主函数"""
    args = parse_args()
    _enforce_resource_guard(args)

    if args.command == 'train':
        train_command(args)
    elif args.command == 'eval':
        eval_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'extract':
        extract_command(args)
    elif args.command == 'importance':
        feature_importance_command(args)
    else:
        print("Please specify a command. Use --help for usage information.")
        sys.exit(1)


if __name__ == '__main__':
    main()
