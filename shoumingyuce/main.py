#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
齿轮寿命预测项目主运行脚本

该脚本提供了完整的训练和预测流程，包括：
1. 数据加载和预处理
2. 模型训练
3. 模型评估
4. 寿命预测

使用方法:
    python main.py --mode train    # 训练模式
    python main.py --mode predict  # 预测模式
    python main.py --mode both     # 训练+预测模式
"""

import os
from pathlib import Path
import sys
import json
import argparse
import torch
import numpy as np
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train_model
from src.predict import LifetimePredictor
from src.data_loader import create_lifetime_dataloaders
from src.model import create_lifetime_model

def load_config(config_path='configs/config.json'):
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    module_root = Path(__file__).resolve().parent
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = module_root / config_path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    for key in ('data_path', 'damage_model_path'):
        value = Path(config['data'][key])
        if not value.is_absolute():
            config['data'][key] = str((module_root / value).resolve())
    return config

def setup_directories(config):
    """
    创建必要的目录

    Args:
        config: 配置字典
    """
    directories = [
        str(Path(__file__).resolve().parent / 'output'),
        str(Path(__file__).resolve().parent / 'output/models'),
        str(Path(__file__).resolve().parent / 'output/logs'),
        str(Path(__file__).resolve().parent / 'output/predictions'),
        str(Path(__file__).resolve().parent / 'output/visualizations')
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def check_data_availability(config):
    """
    检查数据文件是否存在

    Args:
        config: 配置字典

    Returns:
        bool: 数据是否可用
    """
    data_path = config['data']['data_path']
    damage_model_path = config['data']['damage_model_path']

    # 检查数据路径
    if not os.path.exists(data_path):
        print(f"错误: 数据路径不存在: {data_path}")
        return False

    # 检查DATA.json文件
    json_path = os.path.join(data_path, 'DATA.json')
    if not os.path.exists(json_path):
        print(f"错误: 数据文件不存在: {json_path}")
        return False

    # 检查损伤识别模型
    if not os.path.exists(damage_model_path):
        print(f"警告: 损伤识别模型不存在: {damage_model_path}")
        print("将使用预训练的ResNet模型")

    print("数据检查通过")
    return True

def train_mode(config):
    """
    训练模式

    Args:
        config: 配置字典

    Returns:
        训练好的模型路径
    """
    print("\n=== 开始训练模式 ===")

    # 定义要排除的异常值样本
    exclude_outliers = ['yashang_figure1', 'yashang_figure2', 'yashang_figure5']
    print(f"排除异常值样本: {exclude_outliers}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    print("创建数据加载器...")
    dataloaders, dataset_sizes, all_data = create_lifetime_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        exclude_outliers=exclude_outliers
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    print(f"数据集大小: {dataset_sizes}")

    # 创建模型
    print("创建模型...")
    model = create_lifetime_model(config)
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    # 获取损伤识别模型路径
    damage_model_path = config['data']['damage_model_path']

    # 训练模型
    print("开始训练...")
    best_model, training_history, test_results = train_model(
        config=config,
        data_path=config['data']['data_path'],
        output_dir=str(Path(__file__).resolve().parent / 'output'),
        pretrained_damage_model_path=damage_model_path if os.path.exists(damage_model_path) else None,
        exclude_outliers=exclude_outliers
    )

    best_model_path = str(Path(__file__).resolve().parent / 'output/models/best_lifetime_model.pth')

    print(f"\n训练完成！最佳模型保存在: {best_model_path}")

    # 保存训练历史
    history_path = str(Path(__file__).resolve().parent / 'output/training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    print(f"训练历史保存在: {history_path}")

    return best_model_path

def predict_mode(config, model_path=None):
    """
    预测模式

    Args:
        config: 配置字典
        model_path: 模型路径，如果为None则使用默认路径
    """
    print("\n=== 开始预测模式 ===")

    # 定义要排除的异常值样本
    exclude_outliers = ['yashang_figure1', 'yashang_figure2', 'yashang_figure5']
    print(f"排除异常值样本: {exclude_outliers}")

    # 确定模型路径
    if model_path is None:
        model_path = str(Path(__file__).resolve().parent / 'output/models/best_lifetime_model.pth')

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练模式生成模型")
        return

    # 创建预测器
    print(f"加载模型: {model_path}")
    predictor = LifetimePredictor(model_path)

    # 预测数据集中的样本
    print("开始预测...")
    results = predictor.predict_from_dataset(
        data_path=config['data']['data_path'],
        sample_ids=None,  # 预测所有样本
        photo_type='hongguan',
        exclude_outliers=exclude_outliers
    )

    print(f"预测完成，共处理 {len(results)} 个样本")

    # 检查是否有预测结果
    if len(results) == 0:
        print("\n警告: 没有找到任何样本进行预测")
        print("请检查数据路径和数据格式是否正确")
        return

    # 分析预测结果
    analysis = predictor.analyze_predictions(results)

    # 检查分析结果是否包含错误
    if 'error' in analysis:
        print(f"\n预测分析错误: {analysis['error']}")
        return

    # 打印分析结果
    print("\n=== 预测结果分析 ===")
    print(f"总样本数: {analysis['total_samples']}")
    print(f"成功预测: {analysis['successful_predictions']}")
    print(f"预测失败: {analysis['failed_predictions']}")

    if 'overall_metrics' in analysis:
        metrics = analysis['overall_metrics']
        print(f"\n整体性能指标:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")

        print(f"\n各损伤类型性能:")
        for damage_type, stats in analysis['damage_type_stats'].items():
            print(f"  {damage_type}: 样本数={stats['count']}, MAE={stats['mae']:.2f}, MAPE={stats['mape']:.2f}%")

    # 保存结果
    output_dir = str(Path(__file__).resolve().parent / 'output/predictions')
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细预测结果
    results_path = os.path.join(output_dir, 'prediction_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存分析结果
    analysis_path = os.path.join(output_dir, 'prediction_analysis.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # 可视化结果
    predictor.visualize_predictions(results, output_dir)

    print(f"\n预测结果已保存到: {output_dir}")

def demo_single_prediction(config):
    """
    演示单个样本预测

    Args:
        config: 配置字典
    """
    print("\n=== 单个样本预测演示 ===")

    model_path = str(Path(__file__).resolve().parent / 'output/models/best_lifetime_model.pth')
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    # 创建预测器
    predictor = LifetimePredictor(model_path)

    # 加载数据集获取示例
    data_path = config['data']['data_path']
    json_path = os.path.join(data_path, 'DATA.json')

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 选择一个示例进行预测
    for damage_type in ['boluo', 'cashang', 'dianshi', 'yashang']:
        if damage_type in metadata:
            sample_id = list(metadata[damage_type].keys())[0]
            sample_data = metadata[damage_type][sample_id]

            # 构建图片路径
            image_filename = f"{sample_data['path']}.jpg"
            image_path = os.path.join(data_path, damage_type, 'hongguan', image_filename)

            if os.path.exists(image_path):
                print(f"\n预测样本: {sample_id} (损伤类型: {damage_type})")
                print(f"图片路径: {image_path}")

                # 预测
                pred_lifetime = predictor.predict_single(image_path, sample_data['shuxing'])
                true_lifetime = sample_data['shouming']

                print(f"真实寿命: {true_lifetime}")
                print(f"预测寿命: {pred_lifetime:.2f}")
                print(f"绝对误差: {abs(pred_lifetime - true_lifetime):.2f}")
                print(f"相对误差: {abs(pred_lifetime - true_lifetime) / true_lifetime * 100:.2f}%")

                break

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='齿轮寿命预测项目')
    parser.add_argument('--mode', choices=['train', 'predict', 'both', 'demo'],
                       default='both', help='运行模式')
    parser.add_argument('--config', default='configs/config.json',
                       help='配置文件路径')
    parser.add_argument('--model', help='预测模式下使用的模型路径')

    args = parser.parse_args()

    print("=" * 60)
    print("齿轮寿命预测项目")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行模式: {args.mode}")
    print("=" * 60)

    try:
        # 加载配置
        print(f"加载配置文件: {args.config}")
        config = load_config(args.config)

        # 创建目录
        setup_directories(config)

        # 检查数据
        if not check_data_availability(config):
            return

        # 根据模式执行相应操作
        if args.mode == 'train':
            train_mode(config)

        elif args.mode == 'predict':
            predict_mode(config, args.model)

        elif args.mode == 'both':
            # 先训练后预测
            model_path = train_mode(config)
            predict_mode(config, model_path)

        elif args.mode == 'demo':
            demo_single_prediction(config)

        print("\n=== 程序执行完成 ===")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
