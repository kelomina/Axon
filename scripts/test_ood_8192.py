#!/usr/bin/env python3
"""三个OOD改进方案测试（使用8192 cache）"""

import sys
import csv
import random
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加 src 目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import AxonExperimentConfig, TrainingConfig
from model import AxonMalwareModel
from dataset import FeatureCacheDataset, create_split_from_file
from trainer import AxonTrainer


def create_experiment_datasets(cache_dir: Path, output_dir: Path, seed: int = 42):
    """从8192 cache创建实验数据集"""
    import tomllib
    import dataclasses
    
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
    config.max_byte_length = 8192  # 使用8192
    
    # 创建数据集（使用8192 cache）
    print("创建数据集（max_byte_length=8192）...")
    full_dataset = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"),
        max_byte_length=8192,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        axon_config=config,
    )
    
    print(f"  加载了 {len(full_dataset)} 个样本")
    
    # 随机划分 1:1:8
    random.seed(seed)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    
    n_total = len(indices)
    n_train = int(n_total * 0.1)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    
    train_subset = torch.utils.data.Subset(full_dataset, indices[:n_train])
    val_subset = torch.utils.data.Subset(full_dataset, indices[n_train:n_train+n_val])
    test_subset = torch.utils.data.Subset(full_dataset, indices[n_train+n_val:])
    
    print(f"  训练集: {n_train}")
    print(f"  验证集: {n_val}")
    print(f"  测试集: {n_test}")
    
    return train_subset, val_subset, test_subset, config


# 方案1：改进训练策略
def test_improved_training(train_dataset, val_dataset, test_dataset, config, output_dir):
    """改进训练策略：标签平滑(0.1) + 30 epochs"""
    print("\n" + "="*80)
    print("方案1: 改进训练策略")
    print("  - 标签平滑 (0.1)")
    print("  - 30 epochs")
    print("="*80)
    
    train_config = TrainingConfig()
    train_config.label_smoothing = 0.1
    train_config.max_epochs = 30
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model = AxonMalwareModel(config)
    trainer = AxonTrainer(model=model, config=config, train_config=train_config)
    
    print("开始训练...")
    start_time = time.time()
    trainer.train(train_loader, val_loader, test_loader)
    elapsed = time.time() - start_time
    
    # 测试集评估
    test_metrics = trainer.evaluate(test_loader, epoch=0)
    
    print(f"\n[OK] 方案1完成:")
    print(f"  测试F1: {test_metrics.f1:.4f}")
    print(f"  测试准确率: {test_metrics.accuracy:.4f}")
    
    return {'f1': test_metrics.f1, 'accuracy': test_metrics.accuracy, 'time': elapsed}


# 方案2：自适应阈值
def test_adaptive_threshold(train_dataset, val_dataset, test_dataset, config, output_dir):
    """自适应阈值：能量分数OOD检测"""
    print("\n" + "="*80)
    print("方案2: 自适应阈值推理")
    print("  - 能量分数OOD检测")
    print("  - 自适应决策阈值")
    print("="*80)
    
    train_config = TrainingConfig()
    train_config.label_smoothing = 0.03
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model = AxonMalwareModel(config)
    trainer = AxonTrainer(model=model, config=config, train_config=train_config)
    
    print("训练baseline...")
    trainer.train(train_loader, val_loader, test_loader)
    
    # 计算能量阈值
    print("计算能量阈值...")
    model.eval()
    all_energies = []
    device = trainer.device
    
    with torch.no_grad():
        for batch in val_loader:
            byte_seq, pe_features, stat_features, labels = batch
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            energies = -torch.logsumexp(logits, dim=-1)
            all_energies.extend(energies.cpu().numpy())
    
    threshold = np.percentile(all_energies, 10)
    print(f"  能量阈值 (10%分位): {threshold:.4f}")
    
    # 测试
    print("测试自适应阈值...")
    start_time = time.time()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            byte_seq, pe_features, stat_features, labels = batch
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            energies = -torch.logsumexp(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            
            for i in range(len(labels)):
                energy = energies[i].item()
                prob_mal = probs[i, 1].item()
                
                if energy > threshold:
                    final_pred = 1 if prob_mal > 0.3 else 0
                else:
                    final_pred = 1 if prob_mal > 0.5 else 0
                
                all_preds.append(final_pred)
                all_labels.append(labels[i].item())
    
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    elapsed = time.time() - start_time
    
    print(f"\n[OK] 方案2完成:")
    print(f"  测试F1: {f1:.4f}")
    print(f"  测试准确率: {acc:.4f}")
    
    return {'f1': f1, 'accuracy': acc, 'time': elapsed}


# 方案3：温度缩放
def test_temperature_scaling(train_dataset, val_dataset, test_dataset, config, output_dir):
    """后处理校准：温度缩放"""
    print("\n" + "="*80)
    print("方案3: 后处理校准（温度缩放）")
    print("  - 在验证集上学习最优温度")
    print("  - 校准预测概率")
    print("="*80)
    
    train_config = TrainingConfig()
    train_config.label_smoothing = 0.03
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model = AxonMalwareModel(config)
    trainer = AxonTrainer(model=model, config=config, train_config=train_config)
    
    print("训练baseline...")
    trainer.train(train_loader, val_loader, test_loader)
    
    # 找最优温度
    print("计算最优温度...")
    model.eval()
    all_logits = []
    all_labels = []
    device = trainer.device
    
    with torch.no_grad():
        for batch in val_loader:
            byte_seq, pe_features, stat_features, labels = batch
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    best_nll = float('inf')
    best_T = 1.0
    for T in np.linspace(0.1, 5.0, 50):
        scaled_logits = all_logits / T
        nll = F.cross_entropy(scaled_logits, all_labels).item()
        if nll < best_nll:
            best_nll = nll
            best_T = T
    
    print(f"  最优温度: {best_T:.4f}")
    
    # 测试
    print("测试温度缩放...")
    start_time = time.time()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            byte_seq, pe_features, stat_features, labels = batch
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            scaled_logits = logits / best_T
            probs = F.softmax(scaled_logits, dim=-1)
            preds = probs.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    elapsed = time.time() - start_time
    
    print(f"\n[OK] 方案3完成:")
    print(f"  测试F1: {f1:.4f}")
    print(f"  测试准确率: {acc:.4f}")
    
    return {'f1': f1, 'accuracy': acc, 'time': elapsed}


# 主函数
def main():
    print("="*80)
    print("OOD泛化改进方案测试（8192字节输入）")
    print("="*80)
    
    output_base = PROJECT_ROOT / "models" / "ood_improvement_tests_8192"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset, config = create_experiment_datasets(
        PROJECT_ROOT / "data" / ".cache",
        output_base,
    )
    
    # 测试三个方案
    results = {}
    
    # 方案1
    output_dir1 = output_base / "exp1_improved_training"
    output_dir1.mkdir(parents=True, exist_ok=True)
    results['exp1_improved_training'] = test_improved_training(
        train_dataset, val_dataset, test_dataset, config, output_dir1
    )
    
    # 方案2
    output_dir2 = output_base / "exp2_adaptive_threshold"
    output_dir2.mkdir(parents=True, exist_ok=True)
    results['exp2_adaptive_threshold'] = test_adaptive_threshold(
        train_dataset, val_dataset, test_dataset, config, output_dir2
    )
    
    # 方案3
    output_dir3 = output_base / "exp3_temperature_scaling"
    output_dir3.mkdir(parents=True, exist_ok=True)
    results['exp3_temperature_scaling'] = test_temperature_scaling(
        train_dataset, val_dataset, test_dataset, config, output_dir3
    )
    
    # 汇总
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    print(f"{'方案':<30} {'F1':<10} {'准确率':<10} {'用时(s)':<10}")
    print("-"*80)
    for name, result in results.items():
        print(f"{name:<30} {result['f1']:<10.4f} {result['accuracy']:<10.4f} {result['time']:<10.1f}")
    
    # 保存结果
    results_file = output_base / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果已保存: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()
