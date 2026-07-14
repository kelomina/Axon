"""
测试不同输入长度的baseline性能
"""
import time
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tomllib
import dataclasses
from config import AxonExperimentConfig, TrainingConfig
from dataset import MalwareDataset, FeatureCacheDataset
from model import AxonMalwareModel
from trainer import AxonTrainer


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_config(max_byte_length: int):
    config_path = PROJECT_ROOT / "config" / "default_config.toml"
    with open(config_path, 'rb') as f:
        toml_data = tomllib.load(f)
    merged = {}
    for section in toml_data.values():
        if isinstance(section, dict):
            merged.update(section)
    field_names = {field.name for field in dataclasses.fields(AxonExperimentConfig)}
    config = AxonExperimentConfig(**{k: v for k, v in merged.items() if k in field_names})
    config.max_byte_length = max_byte_length
    return config


def build_cache(max_byte_length: int):
    """为指定长度构建cache"""
    config = get_config(max_byte_length)
    data_dir = PROJECT_ROOT / "data"
    
    print(f"\n构建 cache (max_byte_length={max_byte_length})...")
    dataset = MalwareDataset(
        data_dir=str(data_dir),
        max_byte_length=max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=10000,
        max_file_size=config.max_file_size,
        axon_config=config,
        extraction_workers=4,
        extraction_backend='thread',
    )
    print(f"  构建了 {len(dataset)} 个样本的cache")
    return len(dataset)


def test_length(max_byte_length: int, seed: int = 42):
    """测试单个长度"""
    config = get_config(max_byte_length)
    
    # 加载cache
    full_dataset = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"),
        max_byte_length=max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        axon_config=config,
    )
    
    n_total = len(full_dataset)
    print(f"  加载了 {n_total} 个样本")
    
    # 划分 1:1:8
    random = np.random.RandomState(seed)
    indices = list(range(n_total))
    random.shuffle(indices)
    
    n_train = int(n_total * 0.1)
    n_val = int(n_total * 0.1)
    
    train_dataset = Subset(full_dataset, indices[:n_train])
    val_dataset = Subset(full_dataset, indices[n_train:n_train+n_val])
    test_dataset = Subset(full_dataset, indices[n_train+n_val:])
    
    print(f"  训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    
    model = AxonMalwareModel(config)
    train_config = TrainingConfig()
    train_config.max_epochs = 50
    train_config.early_stopping_patience = 10
    train_config.learning_rate = 1e-4
    train_config.label_smoothing = 0.0
    train_config.focal_gamma = 0.5
    train_config.class_weights = [0.8, 1.2]
    
    trainer = AxonTrainer(
        model=model, config=config, train_config=train_config,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    
    start_time = time.time()
    trainer.train(train_loader, val_loader, test_loader)
    elapsed = time.time() - start_time
    
    test_metrics = trainer.evaluate(test_loader, epoch=0)
    
    return {
        'max_byte_length': max_byte_length,
        'f1': float(test_metrics.f1),
        'accuracy': float(test_metrics.accuracy),
        'auc': float(test_metrics.auc),
        'fpr': float(test_metrics.false_positive_rate),
        'fnr': float(test_metrics.false_negative_rate),
        'time': elapsed,
    }


def main():
    lengths = [2048, 1024, 512, 256, 128, 64]
    
    print("="*80)
    print("不同输入长度性能测试")
    print("="*80)
    
    output_dir = PROJECT_ROOT / "models" / "length_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for length in lengths:
        print(f"\n{'='*80}")
        print(f"测试 max_byte_length = {length}")
        print(f"{'='*80}")
        
        # 构建cache（如果不存在）
        try:
            n_samples = build_cache(length)
        except Exception as e:
            print(f"  Cache构建失败: {e}")
            results[str(length)] = {'error': str(e)}
            continue
        
        if n_samples == 0:
            print(f"  跳过: 没有可用样本")
            results[str(length)] = {'error': 'no samples'}
            continue
        
        # 测试
        try:
            result = test_length(length)
            results[str(length)] = result
            print(f"\n[OK] length={length}: F1={result['f1']:.4f}, Acc={result['accuracy']:.4f}, AUC={result['auc']:.4f}, Time={result['time']:.1f}s")
        except Exception as e:
            print(f"  测试失败: {e}")
            results[str(length)] = {'error': str(e)}
            continue
        
        # 每轮保存一次
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, cls=SafeEncoder)
    
    # 最终汇总
    print("\n" + "="*80)
    print("最终汇总")
    print("="*80)
    print(f"{'长度':<10} {'F1':<10} {'准确率':<10} {'AUC':<10} {'FPR':<10} {'FNR':<10} {'用时(s)':<10}")
    print("-"*70)
    
    for length in lengths:
        r = results.get(str(length), {})
        if 'error' in r:
            print(f"{length:<10} ERROR: {r['error']}")
        else:
            print(f"{length:<10} {r['f1']:<10.4f} {r['accuracy']:<10.4f} {r['auc']:<10.4f} {r['fpr']:<10.4f} {r['fnr']:<10.4f} {r['time']:<10.1f}")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=SafeEncoder)
    print(f"\n结果已保存: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
