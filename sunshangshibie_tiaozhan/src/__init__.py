#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
齿轮损伤智能识别系统核心模块

本包提供了齿轮损伤分类的完整解决方案，包括:
- 深度学习模型 (ResNet, VGG, 多模态融合)
- 数据加载和预处理
- 训练和评估工具
- 预测接口
"""

from .model import (
    DamageClassifier,
    VGGDamageClassifier,
    MultiModalDamageClassifier,
    create_damage_classifier,
    create_vgg_damage_classifier,
    load_model,
    save_model
)

from .data_loader import (
    GearDamageDataset,
    load_dataset_with_metadata,
    create_data_loaders,
    get_data_transforms
)

from .predict import DamagePredictor

__version__ = "1.0.0"
__author__ = "齿轮损伤识别项目组"
__description__ = "基于深度学习的齿轮损伤智能识别系统"

__all__ = [
    # 模型相关
    "DamageClassifier",
    "VGGDamageClassifier",
    "MultiModalDamageClassifier",
    "create_damage_classifier",
    "create_vgg_damage_classifier",
    "load_model",
    "save_model",

    # 数据处理相关
    "GearDamageDataset",
    "load_dataset_with_metadata",
    "create_data_loaders",
    "get_data_transforms",

    # 预测相关
    "DamagePredictor"
]

# 包级别配置
DEFAULT_CONFIG = {
    "model_type": "resnet34",
    "num_classes": 4,
    "damage_types": ["boluo", "cashang", "dianshi", "yashang"],
    "photo_types": ["hongguan", "tishijing"],
    "input_size": (224, 224)
}

# 版本兼容性检查
import sys
if sys.version_info < (3, 7):
    raise RuntimeError("本项目需要 Python 3.7 或更高版本")

# 依赖检查
try:
    import torch
    import torchvision
    import numpy
    import sklearn
    import matplotlib
    import seaborn
except ImportError as e:
    raise ImportError(f"缺少必要依赖: {e}. 请运行 'pip install -r requirements.txt' 安装依赖")

print(f"齿轮损伤智能识别系统 v{__version__} 已加载")
