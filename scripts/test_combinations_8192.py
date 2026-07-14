"""
测试所有方案组合（使用8192缓存）
"""
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tomllib
import dataclasses
from config import AxonExperimentConfig, TrainingConfig
from dataset import FeatureCacheDataset
from model import AxonMalwareModel
from trainer import AxonTrainer


class SafeEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def create_datasets(seed: int = 42):
    """从8192 cache创建数据集"""
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


def train_model(train_dataset, val_dataset, test_dataset, config, train_config, output_dir):
    """训练模型并返回结果"""
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
    
    model = AxonMalwareModel(config)
    
    trainer = AxonTrainer(
        model=model,
        config=config,
        train_config=train_config,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    
    start_time = time.time()
    trainer.train(train_loader, val_loader, test_loader)
    elapsed = time.time() - start_time
    
    test_metrics = trainer.evaluate(test_loader, epoch=0)
    
    return trainer, test_metrics, elapsed, test_loader


def apply_temperature_scaling(trainer, test_loader, val_dataset, config):
    """应用温度缩放"""
    device = trainer.device
    
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    
    model = trainer.model
    model.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            byte_seq, pe_features, stat_features, labels = batch
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            all_logits.append(output['logits'].detach().cpu())
            all_labels.append(labels.detach().cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    best_nll = float('inf')
    best_T = 1.0
    for T in np.linspace(0.1, 5.0, 50):
        T_val = float(T)
        scaled_logits = all_logits / T_val
        nll = F.cross_entropy(scaled_logits, all_labels).item()
        if nll < best_nll:
            best_nll = nll
            best_T = T_val
    
    all_preds = []
    all_labels_test = []
    
    with torch.no_grad():
        for batch in test_loader:
            byte_seq, pe_features, stat_features, labels = batch
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            scaled_logits = output['logits'] / best_T
            probs = F.softmax(scaled_logits, dim=-1)
            preds = probs.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels_test.extend(labels.tolist())
    
    from sklearn.metrics import f1_score, accuracy_score
    f1 = float(f1_score(all_labels_test, all_preds))
    acc = float(accuracy_score(all_labels_test, all_preds))
    
    return f1, acc, float(best_T)


def apply_adaptive_threshold(trainer, test_loader, val_dataset, config):
    """应用自适应阈值"""
    device = trainer.device
    
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    
    model = trainer.model
    model.eval()
    
    all_energies = []
    with torch.no_grad():
        for batch in val_loader:
            byte_seq, pe_features, stat_features, labels = batch
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            energies = -torch.logsumexp(logits, dim=-1)
            all_energies.extend(energies.cpu().tolist())
    
    threshold = float(np.percentile(all_energies, 10))
    
    all_preds = []
    all_labels_test = []
    
    with torch.no_grad():
        for batch in test_loader:
            byte_seq, pe_features, stat_features, labels = batch
            byte_seq = byte_seq.to(device)
            pe_features = pe_features.to(device)
            stat_features = stat_features.to(device)
            
            output = model(byte_seq, pe_features, stat_features)
            logits = output['logits']
            probs = F.softmax(logits, dim=-1)
            energies = -torch.logsumexp(logits, dim=-1)
            
            prob_mal = probs[:, 1].cpu().numpy()
            energy = energies.cpu().numpy()
            
            final_pred = np.zeros(len(labels))
            for i in range(len(labels)):
                if energy[i] > threshold:
                    final_pred[i] = 1 if prob_mal[i] > 0.3 else 0
                else:
                    final_pred[i] = 1 if prob_mal[i] > 0.5 else 0
            
            all_preds.extend(final_pred.tolist())
            all_labels_test.extend(labels.tolist())
    
    from sklearn.metrics import f1_score, accuracy_score
    f1 = float(f1_score(all_labels_test, all_preds))
    acc = float(accuracy_score(all_labels_test, all_preds))
    
    return f1, acc, threshold


def main():
    print("="*80)
    print("组合方案测试（8192字节输入）")
    print("="*80)
    
    train_dataset, val_dataset, test_dataset, config = create_datasets()
    
    output_dir = PROJECT_ROOT / "models" / "combination_tests_8192"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 实验1: Baseline
    print("\n" + "="*80)
    print("实验1: Baseline + 后处理")
    print("="*80)
    
    train_config1 = TrainingConfig()
    train_config1.max_epochs = 50
    train_config1.early_stopping_patience = 10
    train_config1.learning_rate = 1e-4
    train_config1.label_smoothing = 0.0
    train_config1.focal_gamma = 0.5
    train_config1.class_weights = [0.8, 1.2]
    
    trainer1, metrics1, time1, test_loader1 = train_model(
        train_dataset, val_dataset, test_dataset, config, train_config1, output_dir / "exp1_baseline"
    )
    
    f1_ts, acc_ts, best_T = apply_temperature_scaling(trainer1, test_loader1, val_dataset, config)
    f1_at, acc_at, threshold_at = apply_adaptive_threshold(trainer1, test_loader1, val_dataset, config)
    
    print(f"\n[OK] Baseline 原始: F1={float(metrics1.f1):.4f}, Acc={float(metrics1.accuracy):.4f}")
    print(f"[OK] Baseline + 温度缩放: F1={f1_ts:.4f} (T={best_T:.2f}), Acc={acc_ts:.4f}")
    print(f"[OK] Baseline + 自适应阈值: F1={f1_at:.4f} (阈值={threshold_at:.4f}), Acc={acc_at:.4f}")
    print(f"  训练时间: {time1:.1f}s")
    
    results['baseline'] = {
        'f1': float(metrics1.f1),
        'accuracy': float(metrics1.accuracy),
        'time': time1
    }
    results['baseline_temp_scaling'] = {
        'f1': f1_ts, 'accuracy': acc_ts,
        'temperature': best_T, 'time': time1
    }
    results['baseline_adaptive_threshold'] = {
        'f1': f1_at, 'accuracy': acc_at,
        'threshold': threshold_at, 'time': time1
    }
    
    # 实验2: 改进训练
    print("\n" + "="*80)
    print("实验2: 改进训练（标签平滑0.05）+ 后处理")
    print("="*80)
    
    train_config2 = TrainingConfig()
    train_config2.max_epochs = 50
    train_config2.early_stopping_patience = 10
    train_config2.learning_rate = 1e-4
    train_config2.label_smoothing = 0.05
    train_config2.focal_gamma = 0.5
    train_config2.class_weights = [0.8, 1.2]
    
    trainer2, metrics2, time2, test_loader2 = train_model(
        train_dataset, val_dataset, test_dataset, config, train_config2, output_dir / "exp2_improved"
    )
    
    f1_ts2, acc_ts2, best_T2 = apply_temperature_scaling(trainer2, test_loader2, val_dataset, config)
    f1_at2, acc_at2, threshold_at2 = apply_adaptive_threshold(trainer2, test_loader2, val_dataset, config)
    
    print(f"\n[OK] 改进训练 原始: F1={float(metrics2.f1):.4f}, Acc={float(metrics2.accuracy):.4f}")
    print(f"[OK] 改进训练 + 温度缩放: F1={f1_ts2:.4f} (T={best_T2:.2f}), Acc={acc_ts2:.4f}")
    print(f"[OK] 改进训练 + 自适应阈值: F1={f1_at2:.4f} (阈值={threshold_at2:.4f}), Acc={acc_at2:.4f}")
    print(f"  训练时间: {time2:.1f}s")
    
    results['improved'] = {
        'f1': float(metrics2.f1),
        'accuracy': float(metrics2.accuracy),
        'time': time2
    }
    results['improved_temp_scaling'] = {
        'f1': f1_ts2, 'accuracy': acc_ts2,
        'temperature': best_T2, 'time': time2
    }
    results['improved_adaptive_threshold'] = {
        'f1': f1_at2, 'accuracy': acc_at2,
        'threshold': threshold_at2, 'time': time2
    }
    
    # 保存结果（使用SafeEncoder处理所有numpy类型）
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, cls=SafeEncoder)
    print(f"\n[OK] 结果已保存: {results_file}")
    
    # 汇总表格
    print("\n" + "="*80)
    print("最终结果汇总")
    print("="*80)
    print(f"{'实验':<45} {'F1':<10} {'准确率':<10} {'用时(s)':<10}")
    print("-"*85)
    
    experiments = [
        ('Baseline', results['baseline']['f1'], results['baseline']['accuracy'], results['baseline']['time']),
        ('Baseline + 温度缩放', results['baseline_temp_scaling']['f1'], results['baseline_temp_scaling']['accuracy'], results['baseline_temp_scaling']['time']),
        ('Baseline + 自适应阈值', results['baseline_adaptive_threshold']['f1'], results['baseline_adaptive_threshold']['accuracy'], results['baseline_adaptive_threshold']['time']),
        ('改进训练(LS=0.05)', results['improved']['f1'], results['improved']['accuracy'], results['improved']['time']),
        ('改进训练 + 温度缩放', results['improved_temp_scaling']['f1'], results['improved_temp_scaling']['accuracy'], results['improved_temp_scaling']['time']),
        ('改进训练 + 自适应阈值', results['improved_adaptive_threshold']['f1'], results['improved_adaptive_threshold']['accuracy'], results['improved_adaptive_threshold']['time']),
    ]
    
    for name, f1, acc, t in experiments:
        print(f"{name:<45} {f1:<10.4f} {acc:<10.4f} {t:<10.1f}")


if __name__ == "__main__":
    main()
