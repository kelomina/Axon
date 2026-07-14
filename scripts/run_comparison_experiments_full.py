#!/usr/bin/env python3
"""完整对比实验脚本

严格按照以下流程执行：
1. 清除所有缓存
2. 从两个目录各随机抽取10000个样本
3. 使用32进程并行提取特征
4. 按1:1:8划分数据集（训练2000 + 验证2000 + 测试16000）
5. 运行6个实验并对比F1分数
"""

import sys
import csv
import random
import shutil
import time
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

# 添加 src 目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def sample_files_from_directory(directory: Path, n_samples: int, seed: int = 42) -> List[Path]:
    """从目录中随机抽取n个文件。"""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # 直接使用所有文件
    all_files = [f for f in directory.rglob("*") if f.is_file()]
    
    print(f"  找到 {len(all_files)} 个文件在 {directory.name}")
    
    if len(all_files) < n_samples:
        raise ValueError(f"文件不足: 需要 {n_samples} 个，但只找到 {len(all_files)} 个")
    
    # 随机抽样
    random.seed(seed)
    sampled = random.sample(all_files, n_samples)
    return sampled


def create_experiment_dataset(
    benign_dir: Path,
    malicious_dir: Path,
    samples_per_class: int = 10000,
    seed: int = 42,
    data_dir: Path = None,
) -> Tuple[List[Path], List[Path]]:
    """创建实验数据集：从两个目录各抽取样本。"""
    print("\n" + "=" * 80)
    print("步骤1: 创建实验数据集")
    print("=" * 80)
    
    print(f"\n抽取良性样本（从 {benign_dir.name}）...")
    benign_samples = sample_files_from_directory(benign_dir, samples_per_class, seed)
    
    print(f"\n抽取恶意样本（从 {malicious_dir.name}）...")
    malicious_samples = sample_files_from_directory(malicious_dir, samples_per_class, seed)
    
    print(f"\n✓ 数据集创建完成:")
    print(f"  良性样本: {len(benign_samples)} 个")
    print(f"  恶意样本: {len(malicious_samples)} 个")
    print(f"  总计: {len(benign_samples) + len(malicious_samples)} 个")
    
    # 清空cache，确保重新构建
    if data_dir is not None:
        cache_dir = data_dir / ".cache"
        if cache_dir.exists():
            print(f"\n清空旧cache: {cache_dir}")
            shutil.rmtree(cache_dir)
            print(f"✓ Cache已清空，将重新提取特征")
    
    return benign_samples, malicious_samples


def split_dataset(
    benign_samples: List[Path],
    malicious_samples: List[Path],
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """划分数据集：训练/验证/测试。"""
    print("\n" + "=" * 80)
    print("步骤2: 划分数据集 (1:1:8)")
    print("=" * 80)
    
    random.seed(seed)
    
    # 分别对两类样本进行划分
    def split_samples(samples: List[Path], label: str):
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_test = n_total - n_train - n_val
        
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]
        
        print(f"  {label}: 训练={len(train)}, 验证={len(val)}, 测试={len(test)}")
        return train, val, test
    
    benign_train, benign_val, benign_test = split_samples(benign_samples, "良性")
    mal_train, mal_val, mal_test = split_samples(malicious_samples, "恶意")
    
    split = {
        'train': {'benign': benign_train, 'malicious': mal_train},
        'val': {'benign': benign_val, 'malicious': mal_val},
        'test': {'benign': benign_test, 'malicious': mal_test},
    }
    
    total_train = len(benign_train) + len(mal_train)
    total_val = len(benign_val) + len(mal_val)
    total_test = len(benign_test) + len(mal_test)
    
    print(f"\n✓ 划分完成:")
    print(f"  训练集: {total_train} 个 ({total_train/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"  验证集: {total_val} 个 ({total_val/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"  测试集: {total_test} 个 ({total_test/(total_train+total_val+total_test)*100:.1f}%)")
    
    # 验证满足最小要求
    assert total_train >= 2000, f"训练集{total_train} < 2000"
    assert total_val >= 2000, f"验证集{total_val} < 2000"
    assert total_test >= 16000, f"测试集{total_test} < 16000"
    print(f"\n✓ 数据集大小验证通过")
    
    return split


def create_split_csv(split: dict, output_path: Path):
    """创建split.csv文件供main.py使用。"""
    print(f"\n创建split文件: {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['source_path', 'split'])  # 只需要这两列
        
        for split_name in ['train', 'val', 'test']:
            for label_name in ['benign', 'malicious']:
                for file_path in split[split_name][label_name]:
                    writer.writerow([str(file_path), split_name])
    
    total = sum(len(files) for split_data in split.values() for files in split_data.values())
    print(f"✓ split.csv已创建，包含 {total} 个样本")


def run_experiment(
    experiment_name: str,
    config_path: Path,
    data_dir: Path,
    split_file: Path,
    output_dir: Path,
) -> dict:
    """运行单个实验并返回结果。"""
    print("\n" + "=" * 80)
    print(f"实验: {experiment_name}")
    print(f"配置: {config_path}")
    print("=" * 80)
    
    # 导入main模块
    from main import train_command
    
    start_time = time.time()
    
    try:
        # 构造训练参数（使用fast_mode限制样本数）
        args = argparse.Namespace()
        args.command = 'train'
        args.config = str(config_path)
        args.data_dir = str(data_dir)
        args.split_file = str(split_file)
        args.output_dir = str(output_dir / experiment_name)
        args.fast = True  # 使用fast_mode限制样本数，避免扫描整个目录
        args.batch_size = None
        args.device = None
        args.lr = None
        args.fp16 = False
        args.enable_swanlab = False
        args.epochs = 10  # 使用10个epochs加速
        args.samples_per_class = 10000  # 限制每类10000个样本（匹配split.csv）
        args.extract_workers = 4  # 降低进程数避免内存问题
        args.extract_backend = 'process'
        args.rare_group_weighting = False
        args.singleton_group_weight = None
        args.rare_group_weight = None
        args.medium_group_weight = None
        args.skip_test_eval = False
        args.resume = None
        args.no_resume = True
        args.init_checkpoint = None
        args.partial_init = False
        
        # 运行训练
        print(f"\n开始训练 {experiment_name}...")
        result = train_command(args)
        
        elapsed = time.time() - start_time
        
        # 提取F1分数
        f1_score = 0.0
        if result and 'val' in result and len(result['val']) > 0:
            best_val_metrics = max(result['val'], key=lambda m: m.f1)
            f1_score = best_val_metrics.f1
        
        print(f"\n✓ {experiment_name} 完成:")
        print(f"  F1分数: {f1_score:.4f}")
        print(f"  用时: {elapsed:.1f}秒")
        
        return {
            'experiment': experiment_name,
            'config': str(config_path),
            'f1': f1_score,
            'elapsed': elapsed,
            'status': 'success',
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {experiment_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'experiment': experiment_name,
            'config': str(config_path),
            'f1': 0.0,
            'elapsed': elapsed,
            'status': f'failed: {str(e)}',
        }


def print_comparison_table(results: List[dict]):
    """打印对比表格。"""
    print("\n" + "=" * 100)
    print("实验对比结果")
    print("=" * 100)
    
    # 表头
    print(f"{'实验':<25} {'F1分数':<12} {'提升':<15} {'用时(秒)':<15} {'状态':<20}")
    print("-" * 100)
    
    # 基线F1
    baseline_f1 = results[0]['f1'] if results else 0.0
    
    # 数据行
    for r in results:
        improvement = r['f1'] - baseline_f1 if baseline_f1 > 0 else 0.0
        improvement_str = f"{improvement:+.4f}" if baseline_f1 > 0 else "N/A"
        
        print(f"{r['experiment']:<25} {r['f1']:<12.4f} {improvement_str:<15} {r['elapsed']:<15.1f} {r['status']:<20}")
    
    print("=" * 100)
    
    # 总结
    if results and len(results) > 1:
        best_exp = max(results[1:], key=lambda x: x['f1'])
        print(f"\n最佳实验: {best_exp['experiment']} (F1={best_exp['f1']:.4f}, 提升={best_exp['f1']-baseline_f1:+.4f})")


def main():
    """主函数：运行所有对比实验。"""
    print("=" * 100)
    print("Axon v2.6 泛化增强对比实验 - 完整版")
    print("=" * 100)
    
    # 配置路径
    data_dir = PROJECT_ROOT / "data"
    benign_dir = data_dir / "待加入白名单"
    malicious_dir = data_dir / "待拉黑"
    
    config_dir = PROJECT_ROOT / "config"
    output_dir = PROJECT_ROOT / "models" / "comparison_experiments_full"
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 实验配置（6个实验）
    experiments = [
        {
            'name': 'exp0_baseline',
            'config': config_dir / "default_config.toml",
        },
        {
            'name': 'exp1_byte_noise',
            'config': config_dir / "exp1_byte_noise.toml",
        },
        {
            'name': 'exp2_swa',
            'config': config_dir / "exp2_swa.toml",
        },
        {
            'name': 'exp3_ema',
            'config': config_dir / "exp3_ema.toml",
        },
        {
            'name': 'exp4_near_threshold',
            'config': config_dir / "exp4_near_threshold.toml",
        },
        {
            'name': 'exp5_all_combined',
            'config': config_dir / "generalization_enhanced.toml",
        },
    ]
    
    print(f"\n将运行 {len(experiments)} 个实验:")
    for exp in experiments:
        print(f"  - {exp['name']}")
    
    # 步骤1: 创建数据集
    benign_samples, malicious_samples = create_experiment_dataset(
        benign_dir=benign_dir,
        malicious_dir=malicious_dir,
        samples_per_class=10000,
        seed=42,
        data_dir=data_dir,
    )
    
    # 步骤2: 划分数据集
    split = split_dataset(
        benign_samples=benign_samples,
        malicious_samples=malicious_samples,
        train_ratio=0.1,
        val_ratio=0.1,
        test_ratio=0.8,
        seed=42,
    )
    
    # 步骤3: 创建split.csv
    split_file = output_dir / "split.csv"
    create_split_csv(split, split_file)
    
    # 步骤4: 运行实验（不使用fast_mode，让MalwareDataset从split.csv加载）
    print(f"\n" + "=" * 100)
    print(f"步骤4: 运行实验")
    print("=" * 100)
    print(f"注意: MalwareDataset将从split.csv加载样本，第一个实验会自动提取特征并保存到cache")
    
    # 运行所有实验
    results = []
    
    for exp in experiments:
        result = run_experiment(
            experiment_name=exp['name'],
            config_path=exp['config'],
            data_dir=data_dir,
            split_file=split_file,
            output_dir=output_dir,
        )
        results.append(result)
        
        # 保存中间结果
        results_file = output_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 中间结果已保存到 {results_file}")
    
    # 步骤5: 打印对比表格
    print_comparison_table(results)
    
    # 步骤6: 保存最终结果
    results_file = output_dir / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    summary_file = output_dir / "summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# 泛化增强对比实验结果（完整版）\n\n")
        f.write("| 实验 | F1分数 | 提升 | 用时(秒) | 状态 |\n")
        f.write("|------|--------|------|----------|------|\n")
        
        baseline_f1 = results[0]['f1'] if results else 0.0
        for r in results:
            improvement = r['f1'] - baseline_f1
            f.write(f"| {r['experiment']} | {r['f1']:.4f} | {improvement:+.4f} | {r['elapsed']:.1f} | {r['status']} |\n")
        
        f.write(f"\n## 数据集信息\n")
        f.write(f"- 总样本数: {len(benign_samples) + len(malicious_samples)}\n")
        f.write(f"- 良性样本: {len(benign_samples)}\n")
        f.write(f"- 恶意样本: {len(malicious_samples)}\n")
        f.write(f"- 训练集: {len(split['train']['benign']) + len(split['train']['malicious'])}\n")
        f.write(f"- 验证集: {len(split['val']['benign']) + len(split['val']['malicious'])}\n")
        f.write(f"- 测试集: {len(split['test']['benign']) + len(split['test']['malicious'])}\n")
    
    print(f"\n✓ 最终结果已保存到:")
    print(f"  JSON: {results_file}")
    print(f"  Markdown: {summary_file}")
    
    print("\n" + "=" * 100)
    print("所有实验完成！")
    print("=" * 100)


if __name__ == "__main__":
    import argparse
    
    main()
