#!/usr/bin/env python3
"""对比实验脚本：基线 vs 4个增强实验 vs 全部叠加

实验设计：
1. 从'待加入白名单'和'待拉黑'各随机抽取10000个样本
2. 按1:1:8划分训练/验证/测试集（种子42确保可复现）
3. 使用fast_mode加速（samples=10000, epochs=10, byte_length=8192）
4. 依次运行6个实验并记录F1分数

实验列表：
- baseline: default_config.toml
- exp1: 字节噪声增强
- exp2: SWA权重平均
- exp3: EMA模型
- exp4: 近阈值样本加权
- exp_all: 全部4个实验叠加
"""

import sys
import json
import random
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# 添加 src 目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def sample_files_from_directory(directory: Path, n_samples: int, seed: int = 42) -> List[Path]:
    """从目录中随机抽取n个PE文件。"""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # 直接使用所有文件（PE文件可能没有标准扩展名）
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
    print("\n" + "=" * 60)
    print("步骤1: 创建实验数据集")
    print("=" * 60)
    
    print(f"\n抽取良性样本（从 {benign_dir.name}）...")
    benign_samples = sample_files_from_directory(benign_dir, samples_per_class, seed)
    
    print(f"\n抽取恶意样本（从 {malicious_dir.name}）...")
    malicious_samples = sample_files_from_directory(malicious_dir, samples_per_class, seed)
    
    print(f"\n✓ 数据集创建完成:")
    print(f"  良性样本: {len(benign_samples)} 个")
    print(f"  恶意样本: {len(malicious_samples)} 个")
    print(f"  总计: {len(benign_samples) + len(malicious_samples)} 个")
    
    # 清空cache，确保重新构建（解决fast_mode样本匹配问题）
    if data_dir is not None:
        cache_dir = data_dir / ".cache"
        if cache_dir.exists():
            print(f"\n清空旧cache: {cache_dir}")
            import shutil
            shutil.rmtree(cache_dir)
            print(f"✓ Cache已清空，训练时将重新提取特征")
    
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
    print("\n" + "=" * 60)
    print("步骤2: 划分数据集 (1:1:8)")
    print("=" * 60)
    
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
    
    return split


def create_split_csv(split: dict, output_path: Path):
    """创廻split.csv文件供main.py使用。"""
    import csv
    
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
    print("\n" + "=" * 60)
    print(f"实验: {experiment_name}")
    print(f"配置: {config_path}")
    print("=" * 60)
    
    # 导入main模块
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from main import train_command
    from config import AxonExperimentConfig
    
    start_time = time.time()
    
    try:
        # 构造训练参数
        args = argparse.Namespace()
        args.command = 'train'
        args.config = str(config_path)
        args.data_dir = str(data_dir)
        args.split_file = str(split_file)
        args.output_dir = str(output_dir / experiment_name)
        args.fast = True  # fast_mode开关
        args.batch_size = None
        args.device = None
        args.lr = None
        args.fp16 = False
        args.enable_swanlab = False
        args.epochs = None
        args.samples_per_class = None
        args.extract_workers = 1
        args.extract_backend = 'thread'
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
        
        # 提取F1分数（从trainer.best_f1获取）
        # train_command返回的是trainer.train()的结果：Dict[str, List[TrainingMetrics]]
        # 我们需要从最后一个val metrics中获取最佳F1
        f1_score = 0.0
        if result and 'val' in result and len(result['val']) > 0:
            # 获取验证集最佳F1
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
    print("\n" + "=" * 80)
    print("实验对比结果")
    print("=" * 80)
    
    # 表头
    print(f"{'实验':<20} {'F1分数':<12} {'提升':<12} {'用时(秒)':<12} {'状态':<15}")
    print("-" * 80)
    
    # 基线F1
    baseline_f1 = results[0]['f1'] if results else 0.0
    
    # 数据行
    for r in results:
        improvement = r['f1'] - baseline_f1 if baseline_f1 > 0 else 0.0
        improvement_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
        
        print(f"{r['experiment']:<20} {r['f1']:<12.4f} {improvement_str:<12} {r['elapsed']:<12.1f} {r['status']:<15}")
    
    print("=" * 80)
    
    # 总结
    if results:
        best_exp = max(results[1:], key=lambda x: x['f1']) if len(results) > 1 else results[0]
        print(f"\n最佳实验: {best_exp['experiment']} (F1={best_exp['f1']:.4f}, 提升={best_exp['f1']-baseline_f1:+.4f})")


def main():
    """主函数：运行所有对比实验。"""
    print("=" * 80)
    print("Axon v2.6 泛化增强对比实验")
    print("=" * 80)
    
    # 配置路径
    data_dir = PROJECT_ROOT / "data"
    benign_dir = data_dir / "待加入白名单"
    malicious_dir = data_dir / "待拉黑"
    
    config_dir = PROJECT_ROOT / "config"
    output_dir = PROJECT_ROOT / "models" / "comparison_experiments"
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 实验配置（测试模式：只运行前2个）
    experiments = [
        {
            'name': 'baseline',
            'config': config_dir / "default_config.toml",
        },
        {
            'name': 'exp1_byte_noise',
            'config': config_dir / "exp1_byte_noise.toml",
        },
        # 以下实验待测试通过后再运行
        # {
        #     'name': 'exp2_swa',
        #     'config': config_dir / "exp2_swa.toml",
        # },
        # {
        #     'name': 'exp3_ema',
        #     'config': config_dir / "exp3_ema.toml",
        # },
        # {
        #     'name': 'exp4_near_threshold',
        #     'config': config_dir / "exp4_near_threshold.toml",
        # },
        # {
        #     'name': 'exp_all_combined',
        #     'config': config_dir / "generalization_enhanced.toml",
        # },
    ]
    
    print(f"\n测试模式: 只运行前 {len(experiments)} 个实验")
    
    # 步骤1: 创建数据集
    benign_samples, malicious_samples = create_experiment_dataset(
        benign_dir=benign_dir,
        malicious_dir=malicious_dir,
        samples_per_class=10000,
        seed=42,
        data_dir=data_dir,  # 传入data_dir用于清空cache
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
    
    # 步骤4: 运行所有实验
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
        f.write("# 泛化增强对比实验结果\n\n")
        f.write("| 实验 | F1分数 | 提升 | 用时(秒) | 状态 |\n")
        f.write("|------|--------|------|----------|------|\n")
        
        baseline_f1 = results[0]['f1'] if results else 0.0
        for r in results:
            improvement = r['f1'] - baseline_f1
            f.write(f"| {r['experiment']} | {r['f1']:.4f} | {improvement:+.4f} | {r['elapsed']:.1f} | {r['status']} |\n")
    
    print(f"\n✓ 最终结果已保存到:")
    print(f"  JSON: {results_file}")
    print(f"  Markdown: {summary_file}")
    
    print("\n" + "=" * 80)
    print("所有实验完成！")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    main()
