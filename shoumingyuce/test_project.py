#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目功能测试脚本

该脚本用于测试项目的各个组件是否正常工作，包括：
1. 依赖包导入测试
2. 配置文件加载测试
3. 数据加载器测试
4. 模型创建测试
5. 基本前向传播测试
"""

import os
from pathlib import Path
import sys
import json
import traceback

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """
    测试依赖包导入
    """
    print("=== 测试依赖包导入 ===")

    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False

    try:
        import torchvision
        print(f"✓ TorchVision版本: {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision导入失败: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy导入失败: {e}")
        return False

    try:
        import matplotlib
        print(f"✓ Matplotlib版本: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib导入失败: {e}")
        return False

    try:
        import sklearn
        print(f"✓ Scikit-learn版本: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn导入失败: {e}")
        return False

    try:
        from PIL import Image
        print(f"✓ Pillow导入成功")
    except ImportError as e:
        print(f"✗ Pillow导入失败: {e}")
        return False

    print("✓ 所有依赖包导入成功\n")
    return True

def test_config_loading():
    """
    测试配置文件加载
    """
    print("=== 测试配置文件加载 ===")

    config_path = str(Path(__file__).resolve().parents[0] / 'configs/config.json')

    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for key in ('data_path', 'damage_model_path'):
            config['data'][key] = str((Path(__file__).resolve().parent / config['data'][key]).resolve())

        # 检查必要的配置项
        required_sections = ['model', 'training', 'data']
        for section in required_sections:
            if section not in config:
                print(f"✗ 配置文件缺少必要部分: {section}")
                return False
            print(f"✓ 配置部分 '{section}' 存在")

        # 检查模型配置
        model_config = config['model']
        required_model_params = ['embedding_dim', 'hidden_dim', 'num_heads', 'num_layers', 'dropout', 'feature_dim']
        for param in required_model_params:
            if param not in model_config:
                print(f"✗ 模型配置缺少参数: {param}")
                return False

        print(f"✓ 配置文件加载成功")
        print(f"✓ 模型嵌入维度: {model_config['embedding_dim']}")
        print(f"✓ Transformer层数: {model_config['num_layers']}")
        print(f"✓ 注意力头数: {model_config['num_heads']}")

    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

    print("✓ 配置文件测试通过\n")
    return True

def test_project_modules():
    """
    测试项目模块导入
    """
    print("=== 测试项目模块导入 ===")

    try:
        from src.data_loader import extract_features_from_shuxing, LifetimePredictionDataset
        print("✓ 数据加载器模块导入成功")
    except ImportError as e:
        print(f"✗ 数据加载器模块导入失败: {e}")
        return False

    try:
        from src.model import FeatureEmbedding, TransformerDecoder, LifetimePredictionModel, create_lifetime_model
        print("✓ 模型模块导入成功")
    except ImportError as e:
        print(f"✗ 模型模块导入失败: {e}")
        return False

    try:
        from src.train import EarlyStopping, calculate_metrics
        print("✓ 训练模块导入成功")
    except ImportError as e:
        print(f"✗ 训练模块导入失败: {e}")
        return False

    print("✓ 所有项目模块导入成功\n")
    return True

def test_feature_extraction():
    """
    测试特征提取功能
    """
    print("=== 测试特征提取功能 ===")

    try:
        from src.data_loader import extract_features_from_shuxing

        # 创建测试数据
        test_shuxing = {
            "moshuo": 2.5,
            "chishu": 30,
            "chikuan": 20,
            "fenduyuanzhijing": 75,
            "zhuansu": 1500,
            "niuju": 100,
            "gonglv": 15,
            "yingdu": 250,
            "tanxingmoliang": 210000,
            "bosongbi": 0.3
        }

        size_features, operation_features, material_features = extract_features_from_shuxing(test_shuxing)

        print(f"✓ 尺寸特征提取成功: {len(size_features)}维")
        print(f"  尺寸特征: {size_features}")
        print(f"✓ 工况特征提取成功: {len(operation_features)}维")
        print(f"  工况特征: {operation_features}")
        print(f"✓ 材料特征提取成功: {len(material_features)}维")
        print(f"  材料特征: {material_features}")

    except Exception as e:
        print(f"✗ 特征提取测试失败: {e}")
        traceback.print_exc()
        return False

    print("✓ 特征提取测试通过\n")
    return True

def test_model_creation():
    """
    测试模型创建
    """
    print("=== 测试模型创建 ===")

    try:
        import torch
        from src.model import create_lifetime_model

        # 加载配置
        with open(Path(__file__).resolve().parent / 'configs/config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        for key in ('data_path', 'damage_model_path'):
            config['data'][key] = str((Path(__file__).resolve().parent / config['data'][key]).resolve())

        # 创建模型
        model = create_lifetime_model(config)

        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✓ 模型创建成功")
        print(f"✓ 总参数数: {total_params:,}")
        print(f"✓ 可训练参数数: {trainable_params:,}")

        # 测试模型结构
        print(f"✓ 模型类型: {type(model).__name__}")

    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        traceback.print_exc()
        return False

    print("✓ 模型创建测试通过\n")
    return True

def test_forward_pass():
    """
    测试模型前向传播
    """
    print("=== 测试模型前向传播 ===")

    try:
        import torch
        from src.model import create_lifetime_model

        # 加载配置
        with open(Path(__file__).resolve().parent / 'configs/config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        for key in ('data_path', 'damage_model_path'):
            config['data'][key] = str((Path(__file__).resolve().parent / config['data'][key]).resolve())

        # 创建模型
        model = create_lifetime_model(config)
        model.eval()

        # 创建测试输入
        batch_size = 2
        image_input = torch.randn(batch_size, 3, 224, 224)
        size_input = torch.randn(batch_size, 7)  # 尺寸特征：7维
        operation_input = torch.randn(batch_size, 2)  # 工况特征：2维
        material_input = torch.randn(batch_size, 8)  # 材料特征：8维

        print(f"✓ 测试输入创建成功")
        print(f"  图像输入形状: {image_input.shape}")
        print(f"  尺寸输入形状: {size_input.shape}")
        print(f"  工况输入形状: {operation_input.shape}")
        print(f"  材料输入形状: {material_input.shape}")

        # 前向传播
        with torch.no_grad():
            output = model(image_input, size_input, operation_input, material_input)

        print(f"✓ 前向传播成功")
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # 检查输出形状
        expected_shape = (batch_size, 1)
        if output.shape == expected_shape:
            print(f"✓ 输出形状正确: {output.shape}")
        else:
            print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {output.shape}")
            return False

    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        traceback.print_exc()
        return False

    print("✓ 前向传播测试通过\n")
    return True

def test_data_path():
    """
    测试数据路径
    """
    print("=== 测试数据路径 ===")

    # 加载配置
    with open(Path(__file__).resolve().parent / 'configs/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        for key in ('data_path', 'damage_model_path'):
            config['data'][key] = str((Path(__file__).resolve().parent / config['data'][key]).resolve())

    data_path = config['data']['data_path']
    damage_model_path = config['data']['damage_model_path']

    # 检查数据路径
    if os.path.exists(data_path):
        print(f"✓ 数据路径存在: {data_path}")

        # 检查DATA.json
        json_path = os.path.join(data_path, 'DATA.json')
        if os.path.exists(json_path):
            print(f"✓ DATA.json文件存在")

            # 检查数据格式
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                damage_types = ['boluo', 'cashang', 'dianshi', 'yashang']
                for damage_type in damage_types:
                    if damage_type in data:
                        sample_count = len(data[damage_type])
                        print(f"✓ {damage_type}损伤样本数: {sample_count}")
                    else:
                        print(f"⚠ {damage_type}损伤类型不存在")

            except Exception as e:
                print(f"✗ DATA.json格式错误: {e}")
                return False
        else:
            print(f"✗ DATA.json文件不存在: {json_path}")
            return False
    else:
        print(f"✗ 数据路径不存在: {data_path}")
        return False

    # 检查损伤模型路径
    if os.path.exists(damage_model_path):
        print(f"✓ 损伤识别模型存在: {damage_model_path}")
    else:
        print(f"⚠ 损伤识别模型不存在: {damage_model_path}")
        print(f"  将使用预训练ResNet模型")

    print("✓ 数据路径测试完成\n")
    return True

def main():
    """
    主测试函数
    """
    print("齿轮寿命预测项目 - 功能测试")
    print("=" * 50)

    tests = [
        ("依赖包导入", test_imports),
        ("配置文件加载", test_config_loading),
        ("项目模块导入", test_project_modules),
        ("特征提取功能", test_feature_extraction),
        ("模型创建", test_model_creation),
        ("模型前向传播", test_forward_pass),
        ("数据路径检查", test_data_path)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ {test_name} 测试失败\n")
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}\n")
            traceback.print_exc()

    print("=" * 50)
    print(f"测试结果: {passed_tests}/{total_tests} 通过")

    if passed_tests == total_tests:
        print("🎉 所有测试通过！项目配置正确，可以开始训练。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查配置和依赖。")
        return 1

if __name__ == "__main__":
    exit(main())
