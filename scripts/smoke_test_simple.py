#!/usr/bin/env python3
"""Axon v2.6 简化冒烟测试

验证项目结构和代码语法是否正确，不依赖外部库。
"""

import sys
import os
import importlib.util
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def iter_project_python_files(project_root: Path):
    """只扫描项目代码目录，避免递归进入 vnev、缓存和数据目录。"""
    for dir_name in ["src", "scripts", "tests"]:
        root = project_root / dir_name
        if root.exists():
            yield from root.rglob("*.py")
    yield from project_root.glob("*.py")

def test_project_structure():
    """测试项目结构"""
    print("=" * 60)
    print("测试 1: 项目结构")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        'src', 'scripts', 'config', 'data', 'models', 'reports', 'tests'
    ]
    
    required_files = [
        'pyproject.toml',
        'requirements.txt',
        'src/axon_exp.py',
        'src/config.py',
        'src/dataset.py',
        'src/model.py',
        'src/trainer.py',
        'src/dsra/__init__.py',
        'src/dsra/mhdsra2/__init__.py',
        'src/dsra/mhdsra2/improved_dsra_mha.py',
        'src/kvd_features/__init__.py',
        'src/kvd_features/extractor.py',
        'scripts/main.py',
        'config/default_config.toml',
    ]
    
    all_ok = True
    
    # 检查目录
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - 缺失")
            all_ok = False
    
    # 检查文件
    print()
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✓ {file_name} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_name} - 缺失")
            all_ok = False
    
    if all_ok:
        print("\n  ✓ 项目结构测试通过\n")
    else:
        print("\n  ✗ 项目结构不完整\n")
    
    return all_ok


def test_syntax():
    """测试 Python 文件语法"""
    print("=" * 60)
    print("测试 2: Python 语法检查")
    print("=" * 60)
    
    import ast
    
    project_root = Path(__file__).parent.parent
    py_files = list(iter_project_python_files(project_root))
    
    success = 0
    failed = []
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            
            ast.parse(source)
            success += 1
            
        except SyntaxError as e:
            failed.append((py_file, f"语法错误: {e.msg}"))
        except Exception as e:
            failed.append((py_file, str(e)))
    
    print(f"  检查文件数: {len(py_files)}")
    print(f"  通过: {success}")
    
    if failed:
        print(f"  失败: {len(failed)}")
        for file_path, error in failed[:3]:
            print(f"    - {file_path}: {error}")
        if len(failed) > 3:
            print(f"    ... 还有 {len(failed) - 3} 个错误")
        print("\n  ✗ 语法检查失败\n")
        return False
    else:
        print("\n  ✓ 语法检查通过\n")
        return True


def test_imports():
    """测试基本导入"""
    print("=" * 60)
    print("测试 3: 模块导入")
    print("=" * 60)
    
    modules = [
        'config',
        'dataset',
        'kvd_features',
        'kvd_features.extractor',
        'dsra.domain',
        'dsra.mhdsra2',
    ]
    
    success = 0
    failed = []
    
    for module_name in modules:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                success += 1
                print(f"  ✓ {module_name}")
            else:
                failed.append(module_name)
        except Exception as e:
            failed.append(f"{module_name}: {e}")
    
    if failed:
        print(f"\n  ✗ 导入失败:")
        for item in failed:
            print(f"    - {item}")
        print()
        return False
    else:
        print("\n  ✓ 模块导入测试通过\n")
        return True


def test_dsra_module():
    """测试 DSRA 核心模块"""
    print("=" * 60)
    print("测试 4: DSRA 核心模块")
    print("=" * 60)
    
    try:
        from dsra.mhdsra2.improved_dsra_mha import (
            MHDSRA2Config,
            MHDSRA2State,
            MultiHeadDSRA2
        )
        
        print("  ✓ 导入 MHDSRA2 模块成功")
        
        # 测试配置类
        config = MHDSRA2Config(
            dim=64,
            heads=2,
            slots=32,
            read_topk=4,
            write_topk=2,
            local_window=64
        )
        print("  ✓ MHDSRA2Config 创建成功")
        
        print("  ✓ DSRA 核心模块测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ✗ DSRA 核心模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_feature_extractor_module():
    """测试特征提取器模块"""
    print("=" * 60)
    print("测试 5: 特征提取器模块")
    print("=" * 60)
    
    try:
        from kvd_features.extractor import (
            ExtractionConfig,
            PEFeatureExtractor,
            extract_byte_sequence
        )
        
        print("  ✓ 导入特征提取器成功")
        
        # 测试配置
        config = ExtractionConfig(
            max_file_size=4096,
            pe_feature_dim=1500
        )
        print("  ✓ ExtractionConfig 创建成功")
        
        # 测试提取器
        extractor = PEFeatureExtractor(config)
        print("  ✓ PEFeatureExtractor 创建成功")
        
        print("  ✓ 特征提取器模块测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ✗ 特征提取器模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """运行所有冒烟测试"""
    print("\n" + "=" * 60)
    print("Axon v2.6 简化冒烟测试")
    print("=" * 60)
    print("注意: 此测试不依赖外部库 (torch等)")
    print("=" * 60 + "\n")
    
    tests = [
        ("项目结构", test_project_structure),
        ("Python语法", test_syntax),
        ("模块导入", test_imports),
        ("DSRA核心模块", test_dsra_module),
        ("特征提取器模块", test_feature_extractor_module),
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
        print("\n建议检查:")
        print("1. 确保所有文件都已正确复制")
        print("2. 检查 Python 语法错误")
        print("3. 确保依赖已安装")
        sys.exit(1)
    else:
        print("\n✓ 所有简化测试通过！")
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 运行完整冒烟测试: python scripts/smoke_test.py")
        print("3. 开始训练: python scripts/main.py train --data-dir data")
        sys.exit(0)


if __name__ == "__main__":
    main()
