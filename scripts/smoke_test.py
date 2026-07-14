#!/usr/bin/env python3
"""Axon v2.6 冒烟测试脚本

验证所有核心组件功能正常：
1. 特征提取器
2. 数据集加载器
3. 模型初始化
4. 前向传播
5. 训练器
"""

import sys
import os
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

def test_feature_extractor():
    """测试特征提取器"""
    print("=" * 60)
    print("测试 1: 特征提取器")
    print("=" * 60)
    
    try:
        from kvd_features.extractor import (
            extract_byte_sequence,
            extract_pe_features,
            extract_statistical_features,
            PEFeatureExtractor,
            ExtractionConfig
        )
        
        # 创建测试文件
        test_file = Path("test_sample.exe")
        
        # 创建一个小型测试文件（模拟 PE 文件结构）
        pe_header = b'MZ' + b'\x00' * 60 + b'PE\x00\x00'
        section_data = b'\x90' * 1024  # NOP 指令
        
        with open(test_file, 'wb') as f:
            f.write(pe_header + section_data)
        
        # 测试字节序列提取
        byte_seq, orig_len = extract_byte_sequence(str(test_file), max_file_size=65536)
        assert byte_seq is not None, "字节序列提取失败"
        assert len(byte_seq) == 65536, f"字节序列长度不对: {len(byte_seq)}"
        assert orig_len == len(pe_header) + len(section_data), "原始长度错误"
        print("  ✓ 字节序列提取成功")
        
        # 测试 PE 特征提取
        pe_features = extract_pe_features(str(test_file))
        assert pe_features is not None, "PE特征提取失败"
        assert len(pe_features) == 1500, f"PE特征维度不对: {len(pe_features)}"
        print("  ✓ PE特征提取成功")
        
        # 测试统计特征提取。统计特征只需要字节序列和原始长度；
        # PE 特征是另一条输入，不再混到这个函数里。
        stat_features = extract_statistical_features(byte_seq, orig_len)
        assert stat_features is not None, "统计特征提取失败"
        assert len(stat_features) > 0, "统计特征为空"
        print("  ✓ 统计特征提取成功")
        
        # 测试特征提取器类
        extractor = PEFeatureExtractor()
        features = extractor.extract(str(test_file))
        assert features is not None, "PEFeatureExtractor 失败"
        print("  ✓ PEFeatureExtractor 类成功")
        
        # 清理测试文件
        test_file.unlink()
        
        print("  ✓ 特征提取器测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ✗ 特征提取器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """测试数据集加载器"""
    print("=" * 60)
    print("测试 2: 数据集加载器")
    print("=" * 60)
    
    try:
        from config import AxonExperimentConfig
        from dataset import MalwareDataset, NPZDataset
        
        # 创建临时数据目录
        test_dir = Path("test_data")
        benign_dir = test_dir / "benign"
        malicious_dir = test_dir / "malicious"
        benign_dir.mkdir(parents=True, exist_ok=True)
        malicious_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试文件
        for i in range(3):
            with open(benign_dir / f"benign_{i}.exe", 'wb') as f:
                f.write(b'MZ' + b'\x00' * 100)
            with open(malicious_dir / f"malicious_{i}.exe", 'wb') as f:
                f.write(b'MZ' + b'\x90' * 100)
        
        # 这些测试文件只是最小 MZ 合成样本，不是真实 PE。
        # 冒烟测试里显式开启 fallback，真实配置仍默认严格解析。
        smoke_config = AxonExperimentConfig(
            max_byte_length=4096,
            pe_feature_dim=1500,
            strict_pe_parsing=False,
            allow_pe_fallback=True,
        )

        # 测试数据集初始化
        dataset = MalwareDataset(
            data_dir=str(test_dir),
            max_byte_length=smoke_config.max_byte_length,
            pe_feature_dim=smoke_config.pe_feature_dim,
            stat_feature_dim=smoke_config.stat_feature_dim,
            use_cache=False,
            axon_config=smoke_config,
        )
        
        assert len(dataset) == 6, f"数据集大小不对: {len(dataset)}"
        print("  ✓ 数据集初始化成功")
        
        # 测试数据加载
        byte_seq, pe_features, stat_features, label = dataset[0]
        
        assert byte_seq.shape == (4096,), f"字节序列形状不对: {byte_seq.shape}"
        assert pe_features.shape == (1500,), f"PE特征形状不对: {pe_features.shape}"
        assert stat_features.shape[0] > 0, "统计特征为空"
        assert label in [0, 1], f"标签值不对: {label}"
        print("  ✓ 数据加载成功")
        
        # 测试数据类型
        assert byte_seq.dtype == torch.uint8, f"字节序列类型不对: {byte_seq.dtype}"
        assert pe_features.dtype == torch.float32, f"PE特征类型不对: {pe_features.dtype}"
        assert label.dtype == torch.int64, f"标签类型不对: {label.dtype}"
        print("  ✓ 数据类型正确")
        
        # 清理测试目录
        import shutil
        shutil.rmtree(test_dir)
        
        print("  ✓ 数据集测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ✗ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """测试模型初始化和前向传播"""
    print("=" * 60)
    print("测试 3: 模型初始化和前向传播")
    print("=" * 60)
    
    try:
        from config import AxonExperimentConfig, DSRAArchitectureConfig
        from model import AxonMalwareModel, MalwareDSRAEncoder
        
        # 创建配置
        config = AxonExperimentConfig(
            max_byte_length=4096,
            batch_size=2,
            device="cpu"
        )
        
        # 测试模型初始化
        model = AxonMalwareModel(config)
        assert model is not None, "模型初始化失败"
        print("  ✓ 模型初始化成功")
        
        # 测试参数数量
        params = sum(p.numel() for p in model.parameters())
        assert params > 0, "模型无参数"
        print(f"  ✓ 模型参数数量: {params:,}")
        
        # 测试前向传播
        batch_size = 2
        byte_seq = torch.randint(0, 256, (batch_size, config.max_byte_length)).long()
        pe_features = torch.randn(batch_size, config.pe_feature_dim).float()
        
        outputs = model(byte_seq, pe_features)
        
        assert 'logits' in outputs, "输出缺少 logits"
        assert outputs['logits'].shape == (batch_size, config.num_classes), \
            f"logits形状不对: {outputs['logits'].shape}"
        print("  ✓ 前向传播成功")
        
        # 测试返回特征
        outputs = model(byte_seq, pe_features, return_features=True)
        assert 'features' in outputs, "输出缺少 features"
        assert 'byte_repr' in outputs, "输出缺少 byte_repr"
        assert 'pe_repr' in outputs, "输出缺少 pe_repr"
        print("  ✓ 返回特征成功")
        
        # 测试预测
        preds = model.predict(byte_seq, pe_features)
        assert preds.shape == (batch_size,), f"预测形状不对: {preds.shape}"
        assert torch.all(preds >= 0) and torch.all(preds < config.num_classes), "预测值范围不对"
        print("  ✓ 预测功能成功")
        
        # 测试概率预测
        probs = model.predict_proba(byte_seq, pe_features)
        assert probs.shape == (batch_size, config.num_classes), \
            f"概率形状不对: {probs.shape}"
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size)), "概率和不为1"
        print("  ✓ 概率预测成功")
        
        # 测试 DSRA 编码器
        dsra_config = DSRAArchitectureConfig(dim=64, heads=2, slots=32)
        encoder = MalwareDSRAEncoder(
            byte_embedding_dim=64,
            max_byte_length=4096,
            dsra_config=dsra_config,
            pe_feature_dim=1500,
            pe_projection_dim=64
        )
        
        byte_repr, pe_repr, state = encoder(byte_seq[:, :1024], pe_features)
        assert byte_repr.shape == (batch_size, 64), f"字节表示形状不对: {byte_repr.shape}"
        assert pe_repr.shape == (batch_size, 64), f"PE表示形状不对: {pe_repr.shape}"
        print("  ✓ DSRA编码器测试通过")
        
        print("  ✓ 模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer():
    """测试训练器"""
    print("=" * 60)
    print("测试 4: 训练器")
    print("=" * 60)
    
    try:
        from config import AxonExperimentConfig, TrainingConfig
        from model import AxonMalwareModel
        from trainer import AxonTrainer, EarlyStopping, MetricsTracker
        
        # 创建配置
        config = AxonExperimentConfig(
            max_byte_length=1024,
            batch_size=2,
            device="cpu"
        )
        
        train_config = TrainingConfig(
            max_epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            lr_scheduler="none",
            early_stopping_patience=1
        )
        
        # 创建模型
        model = AxonMalwareModel(config)
        
        # 测试训练器初始化
        trainer = AxonTrainer(model, config, train_config)
        assert trainer is not None, "训练器初始化失败"
        print("  ✓ 训练器初始化成功")
        
        # 测试早停机制
        early_stopping = EarlyStopping(patience=2)
        assert not early_stopping(0.5), "早停机制异常"
        assert not early_stopping(0.6), "早停机制异常"
        assert not early_stopping(0.6), "早停机制异常"
        assert early_stopping(0.6), "早停机制未触发"
        print("  ✓ 早停机制测试通过")
        
        # 测试指标追踪器
        tracker = MetricsTracker()
        
        from trainer import TrainingMetrics
        metrics = TrainingMetrics(
            epoch=1,
            phase="train",
            loss=0.5,
            accuracy=0.8,
            precision=0.7,
            recall=0.9,
            f1=0.8,
            auc=0.85,
            learning_rate=1e-4
        )
        
        tracker.update(metrics)
        assert len(tracker.history) == 1, "指标追踪器未记录"
        assert tracker.get_best('f1') == 0.8, "最佳指标获取失败"
        print("  ✓ 指标追踪器测试通过")
        
        # 创建虚拟数据集进行训练测试
        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            
            def __getitem__(self, idx):
                byte_seq = torch.randint(0, 256, (config.max_byte_length,)).long()
                pe_features = torch.randn(config.pe_feature_dim).float()
                stat_features = torch.randn(config.stat_feature_dim).float()
                label = torch.tensor(idx % 2, dtype=torch.long)
                return byte_seq, pe_features, stat_features, label
        
        mock_dataset = MockDataset()
        mock_loader = torch.utils.data.DataLoader(
            mock_dataset, batch_size=2, shuffle=True
        )
        
        # 测试单轮训练
        train_metrics = trainer.train_epoch(mock_loader, epoch=1)
        assert train_metrics.loss > 0, "训练损失异常"
        assert 0 <= train_metrics.accuracy <= 1, "准确率范围异常"
        print("  ✓ 单轮训练成功")
        
        # 测试评估
        val_metrics = trainer.evaluate(mock_loader, epoch=1, phase="val")
        assert val_metrics.loss > 0, "评估损失异常"
        print("  ✓ 评估功能成功")
        
        # 测试检查点保存
        trainer.save_checkpoint("test_checkpoint.pt")
        checkpoint_path = trainer.output_dir / "test_checkpoint.pt"
        assert checkpoint_path.exists(), "检查点未保存"
        checkpoint_path.unlink()
        print("  ✓ 检查点保存成功")
        
        print("  ✓ 训练器测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ✗ 训练器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    """测试 GPU 支持"""
    print("=" * 60)
    print("测试 5: GPU 支持")
    print("=" * 60)
    
    try:
        if torch.cuda.is_available():
            print("  ✓ CUDA 可用")
            
            from config import AxonExperimentConfig
            from model import AxonMalwareModel
            
            config = AxonExperimentConfig(
                max_byte_length=1024,
                batch_size=1,
                device="cuda"
            )
            
            model = AxonMalwareModel(config)
            model.to("cuda")
            
            byte_seq = torch.randint(0, 256, (1, config.max_byte_length)).long().cuda()
            pe_features = torch.randn(1, config.pe_feature_dim).float().cuda()
            
            outputs = model(byte_seq, pe_features)
            assert outputs['logits'].device.type == 'cuda', "输出不在 GPU 上"
            print("  ✓ 模型在 GPU 上运行成功")
            
            print("  ✓ GPU 测试通过\n")
        else:
            print("  ⚠ CUDA 不可用，跳过 GPU 测试\n")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GPU 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有冒烟测试"""
    print("\n" + "=" * 60)
    print("Axon v2.6 冒烟测试")
    print("=" * 60 + "\n")
    
    tests = [
        ("特征提取器", test_feature_extractor),
        ("数据集加载器", test_dataset),
        ("模型", test_model),
        ("训练器", test_trainer),
        ("GPU 支持", test_gpu),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")
    
    if failed > 0:
        print("\n✗ 存在失败的测试，请修复后再运行训练")
        sys.exit(1)
    else:
        print("\n✓ 所有测试通过！可以开始训练")
        sys.exit(0)


if __name__ == "__main__":
    main()
