#!/usr/bin/env python3
"""OOD泛化改进方案测试脚本

测试三个方案：
1. 改进训练策略（标签平滑 + Mixup + 延长训练）
2. 自适应阈值推理（能量分数 + OOD检测）
3. 后处理校准（温度缩放）
"""

import sys
import csv
import random
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加 src 目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config import AxonExperimentConfig, TrainingConfig
from model import AxonMalwareModel
from dataset import FeatureCacheDataset, create_split_from_file, _feature_cache_hash
from trainer import AxonTrainer


def sample_from_cache(cache_dir: Path, n_samples: int, seed: int = 42) -> List[Path]:
    """从cache目录中随机抽取n个npz文件。"""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    
    all_npz = list(cache_dir.glob("*.npz"))
    print(f"  找到 {len(all_npz)} 个cache文件")
    
    if len(all_npz) < n_samples:
        raise ValueError(f"Cache文件不足: 需要 {n_samples} 个，但只找到 {len(all_npz)} 个")
    
    random.seed(seed)
    sampled = random.sample(all_npz, n_samples)
    return sampled


def create_split_file(
    benign_samples: List[Path],
    malicious_samples: List[Path],
    output_path: Path,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """创建split.csv文件。"""
    random.seed(seed)
    
    def split_list(samples, train_r, val_r):
        shuffled = samples.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        
        splits = {}
        for i, path in enumerate(shuffled):
            if i < n_train:
                splits[path] = 'train'
            elif i < n_train + n_val:
                splits[path] = 'val'
            else:
                splits[path] = 'test'
        return splits
    
    benign_splits = split_list(benign_samples, train_ratio, val_ratio)
    malicious_splits = split_list(malicious_samples, train_ratio, val_ratio)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_path', 'split', 'label'])
        
        for path, split in benign_splits.items():
            writer.writerow([str(path), split, 0])
        for path, split in malicious_splits.items():
            writer.writerow([str(path), split, 1])
    
    print(f"[OK] split.csv已创建，包含 {len(benign_splits) + len(malicious_splits)} 个样本")


# ==================== 方案1：改进训练策略 ====================

class ImprovedTrainingConfig(TrainingConfig):
    """改进的训练配置"""
    label_smoothing: float = 0.1  # 增强的标签平滑
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    max_epochs: int = 30  # 延长训练


def train_with_improved_strategy(
    train_dataset,
    val_dataset,
    test_dataset,
    config: AxonExperimentConfig,
    output_dir: Path,
):
    """使用改进的训练策略训练模型。"""
    print("\n" + "="*80)
    print("方案1: 改进训练策略")
    print("  - 标签平滑 (0.1)")
    print("  - Mixup增强")
    print("  - 延长训练到30 epochs")
    print("="*80)
    
    train_config = ImprovedTrainingConfig()
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # 创建模型
    model = AxonMalwareModel(config)
    
    # 创建训练器（使用改进配置）
    trainer = AxonTrainer(
        model=model,
        config=config,
        train_config=train_config,
    )
    
    # 训练
    print("开始训练...")
    start_time = time.time()
    trainer.train(train_loader, val_loader, test_loader)
    elapsed = time.time() - start_time
    
    # 评估
    test_metrics = trainer.evaluate(test_loader, epoch=0)
    
    print(f"\n[OK] 方案1完成:")
    print(f"  测试F1: {test_metrics['f1']:.4f}")
    print(f"  测试准确率: {test_metrics['accuracy']:.4f}")
    print(f"  用时: {elapsed:.1f}秒")
    
    return {
        'f1': test_metrics['f1'],
        'accuracy': test_metrics['accuracy'],
        'time': elapsed,
    }


# ==================== 方案2：自适应阈值推理 ====================

def compute_energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """计算能量分数（用于OOD检测）。"""
    # Energy = -T * log(sum(exp(logit_i / T)))
    energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
    return energy


def find_optimal_energy_threshold(val_loader, model, device, target_reject_ratio: float = 0.1):
    """在验证集上找到最优能量阈值。"""
    model.eval()
    all_energies = []
    
    with torch.no_grad():
        for byte_seq, pe_features, stat_features, labels in val_loader:
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            energies = compute_energy_score(logits)
            all_energies.extend(energies.cpu().numpy())
    
    all_energies = np.array(all_energies)
    threshold = np.percentile(all_energies, target_reject_ratio * 100)
    
    return threshold


def predict_with_adaptive_threshold(model, test_loader, device, energy_threshold: float):
    """使用自适应阈值进行预测。"""
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    all_energies = []
    
    with torch.no_grad():
        for byte_seq, pe_features, stat_features, labels in test_loader:
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            
            energies = compute_energy_score(logits)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            confidences = probs.max(dim=-1).values
            
            # 自适应决策
            for i in range(len(labels)):
                energy = energies[i].item()
                prob_mal = probs[i, 1].item()
                
                if energy > energy_threshold:
                    # OOD样本：保守策略（偏向恶意）
                    final_pred = 1 if prob_mal > 0.3 else 0
                else:
                    # In-distribution：正常决策
                    final_pred = preds[i].item()
                
                all_preds.append(final_pred)
                all_labels.append(labels[i].item())
                all_confidences.append(confidences[i].item())
                all_energies.append(energy)
    
    # 计算指标
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    
    return {
        'f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'predictions': all_preds,
        'confidences': all_confidences,
        'energies': all_energies,
    }


def test_adaptive_threshold(
    train_dataset,
    val_dataset,
    test_dataset,
    config: AxonExperimentConfig,
    output_dir: Path,
):
    """测试自适应阈值方案。"""
    print("\n" + "="*80)
    print("方案2: 自适应阈值推理")
    print("  - 能量分数OOD检测")
    print("  - 自适应决策阈值")
    print("="*80)
    
    # 先用baseline模型
    train_config = TrainingConfig()
    
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0)
    
    model = AxonMalwareModel(config)
    trainer = AxonTrainer(
        model=model,
        config=config,
        train_config=train_config,
    )
    
    # 训练baseline
    print("训练baseline模型...")
    trainer.train(train_loader, val_loader, test_loader)
    
    # 找到最优能量阈值
    print("计算能量阈值...")
    energy_threshold = find_optimal_energy_threshold(val_loader, model, trainer.device, target_reject_ratio=0.1)
    print(f"  能量阈值: {energy_threshold:.4f}")
    
    # 使用自适应阈值预测
    print("测试自适应阈值...")
    start_time = time.time()
    results = predict_with_adaptive_threshold(model, test_loader, trainer.device, energy_threshold)
    elapsed = time.time() - start_time
    
    print(f"\n[OK] 方案2完成:")
    print(f"  测试F1: {results['f1']:.4f}")
    print(f"  测试准确率: {results['accuracy']:.4f}")
    print(f"  用时: {elapsed:.1f}秒")
    
    return {
        'f1': results['f1'],
        'accuracy': results['accuracy'],
        'time': elapsed,
    }


# ==================== 方案3：后处理校准 ====================

def find_optimal_temperature(val_loader, model, device):
    """在验证集上找到最优温度参数。"""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for byte_seq, pe_features, stat_features, labels in val_loader:
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 搜索最优温度
    best_nll = float('inf')
    best_T = 1.0
    
    for T in np.linspace(0.1, 5.0, 50):
        scaled_logits = all_logits / T
        nll = F.cross_entropy(scaled_logits, all_labels).item()
        
        if nll < best_nll:
            best_nll = nll
            best_T = T
    
    return best_T


def predict_with_temperature(model, test_loader, device, temperature: float):
    """使用温度缩放进行预测。"""
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for byte_seq, pe_features, stat_features, labels in test_loader:
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            
            # 温度缩放
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            preds = probs.argmax(dim=-1)
            confidences = probs.max(dim=-1).values
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    from sklearn.metrics import f1_score, accuracy_score
    
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    return {
        'f1': f1,
        'accuracy': acc,
        'predictions': all_preds,
        'confidences': all_confidences,
    }


def test_temperature_scaling(
    train_dataset,
    val_dataset,
    test_dataset,
    config: AxonExperimentConfig,
    output_dir: Path,
):
    """测试温度缩放方案。"""
    print("\n" + "="*80)
    print("方案3: 后处理校准（温度缩放）")
    print("  - 在验证集上学习最优温度")
    print("  - 校准预测概率")
    print("="*80)
    
    train_config = TrainingConfig()
    
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0)
    
    model = AxonMalwareModel(config)
    trainer = AxonTrainer(
        model=model,
        config=config,
        train_config=train_config,
    )
    
    # 训练baseline
    print("训练baseline模型...")
    trainer.train(train_loader, val_loader, test_loader)
    
    # 找到最优温度
    print("计算最优温度...")
    optimal_T = find_optimal_temperature(val_loader, model, trainer.device)
    print(f"  最优温度: {optimal_T:.4f}")
    
    # 使用温度缩放预测
    print("测试温度缩放...")
    start_time = time.time()
    results = predict_with_temperature(model, test_loader, trainer.device, optimal_T)
    elapsed = time.time() - start_time
    
    print(f"\n[OK] 方案3完成:")
    print(f"  测试F1: {results['f1']:.4f}")
    print(f"  测试准确率: {results['accuracy']:.4f}")
    print(f"  用时: {elapsed:.1f}秒")
    
    return {
        'f1': results['f1'],
        'accuracy': results['accuracy'],
        'time': elapsed,
    }


# ==================== 主函数 ====================

def main():
    print("="*80)
    print("OOD泛化改进方案测试")
    print("="*80)
    
    # 配置
    cache_dir = PROJECT_ROOT / "data" / ".cache"
    output_base = PROJECT_ROOT / "models" / "ood_improvement_tests"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 从cache抽样
    print("\n步骤1: 从cache中抽取样本")
    print("-"*80)
    samples_per_class = 10000
    all_samples = sample_from_cache(cache_dir, samples_per_class * 2)
    benign_samples = all_samples[:samples_per_class]
    malicious_samples = all_samples[samples_per_class:]
    
    print(f"[OK] 良性样本: {len(benign_samples)}")
    print(f"[OK] 恶意样本: {len(malicious_samples)}")
    
    # 创建split文件
    print("\n步骤2: 划分数据集")
    print("-"*80)
    split_file = output_base / "split.csv"
    create_split_file(benign_samples, malicious_samples, split_file)
    
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
    
    # 覆盖max_byte_length为8192（加速训练）
    # 但保持cache_config_hash使用65536的hash（因为cache是用65536构建的）
    config.max_byte_length = 8192
    original_hash = _feature_cache_hash(
        65536,  # 原始max_byte_length
        config.stat_feature_dim,
        config.pe_feature_dim,
        config.lightweight_feature_dim,
        config.strict_pe_parsing,
        config.allow_pe_fallback,
        config.pe_schema_version,
        config.pe_fixed_section_slots,
    )
    
    # 创建数据集
    print("\n步骤3: 创建数据集")
    print("-"*80)
    
    # 创建临时配置（使用65536的hash加载cache，但训练时用8192）
    cache_load_config = AxonExperimentConfig(**{k: v for k, v in merged.items() if k in field_names})
    cache_load_config.max_byte_length = 65536  # 加载cache用原始长度
    
    full_dataset = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"),
        max_byte_length=65536,  # 加载cache用65536
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        axon_config=cache_load_config,
    )
    
    # 加载完后覆盖为8192用于训练
    full_dataset.max_byte_length = 8192
    
    train_dataset, val_dataset, test_dataset = create_split_from_file(
        full_dataset,
        split_file,
    )
    
    print(f"[OK] 训练集: {len(train_dataset)}")
    print(f"[OK] 验证集: {len(val_dataset)}")
    print(f"[OK] 测试集: {len(test_dataset)}")
    
    # 测试三个方案
    results = {}
    
    # 方案1：改进训练策略
    output_dir1 = output_base / "exp1_improved_training"
    output_dir1.mkdir(parents=True, exist_ok=True)
    results['exp1_improved_training'] = train_with_improved_strategy(
        train_dataset, val_dataset, test_dataset, config, output_dir1
    )
    
    # 方案2：自适应阈值
    output_dir2 = output_base / "exp2_adaptive_threshold"
    output_dir2.mkdir(parents=True, exist_ok=True)
    results['exp2_adaptive_threshold'] = test_adaptive_threshold(
        train_dataset, val_dataset, test_dataset, config, output_dir2
    )
    
    # 方案3：温度缩放
    output_dir3 = output_base / "exp3_temperature_scaling"
    output_dir3.mkdir(parents=True, exist_ok=True)
    results['exp3_temperature_scaling'] = test_temperature_scaling(
        train_dataset, val_dataset, test_dataset, config, output_dir3
    )
    
    # 汇总结果
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    print(f"{'方案':<30} {'F1分数':<10} {'准确率':<10} {'用时(秒)':<10}")
    print("-"*80)
    for name, result in results.items():
        print(f"{name:<30} {result['f1']:<10.4f} {result['accuracy']:<10.4f} {result['time']:<10.1f}")
    
    # 保存结果
    results_file = output_base / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果已保存到: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()
