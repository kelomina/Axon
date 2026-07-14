"""
重测baseline（使用8192缓存）
"""
import time
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import AxonExperimentConfig, TrainingConfig
from dataset import FeatureCacheDataset
from model import AxonMalwareModel
from trainer import AxonTrainer


def create_datasets(cache_dir: Path, seed: int = 42):
    """从8192 cache创建数据集"""
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
    config.max_byte_length = 8192
    
    full_dataset = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"),
        max_byte_length=8192,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        axon_config=config,
    )
    
    print(f"加载了 {len(full_dataset)} 个样本")
    
    # 随机划分 1:1:8
    random = np.random.RandomState(seed)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    
    n_total = len(full_dataset)
    n_train = int(n_total * 0.1)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    
    train_dataset = Subset(full_dataset, indices[:n_train])
    val_dataset = Subset(full_dataset, indices[n_train:n_train+n_val])
    test_dataset = Subset(full_dataset, indices[n_train+n_val:])
    
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, config


def test_baseline():
    """测试标准baseline"""
    print("="*80)
    print("Baseline重测（8192字节输入）")
    print("="*80)
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset, config = create_datasets(
        PROJECT_ROOT / "data" / ".cache"
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    
    # 创建模型
    model = AxonMalwareModel(config)
    
    # 标准训练配置
    train_config = TrainingConfig()
    train_config.max_epochs = 50
    train_config.early_stopping_patience = 10
    train_config.learning_rate = 1e-4
    train_config.label_smoothing = 0.0  # 无标签平滑
    train_config.focal_gamma = 0.5
    train_config.class_weights = [0.8, 1.2]
    
    # 训练
    output_dir = PROJECT_ROOT / "models" / "baseline_8192_retest"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = AxonTrainer(
        model=model,
        config=config,
        train_config=train_config,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    
    print("\n开始训练baseline...")
    start_time = time.time()
    trainer.train(train_loader, val_loader, test_loader)
    elapsed = time.time() - start_time
    
    # 测试集评估
    test_metrics = trainer.evaluate(test_loader, epoch=0)
    
    print(f"\nBaseline完成:")
    print(f"  测试F1: {test_metrics.f1:.4f}")
    print(f"  测试准确率: {test_metrics.accuracy:.4f}")
    print(f"  训练时间: {elapsed:.1f}s")
    
    # 保存结果
    results = {
        'baseline_retest': {
            'f1': test_metrics.f1,
            'accuracy': test_metrics.accuracy,
            'time': elapsed
        }
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_file}")
    
    return results


if __name__ == "__main__":
    test_baseline()
