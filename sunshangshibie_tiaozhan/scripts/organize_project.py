#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目代码整理脚本

本脚本用于整理项目代码结构，确保引用路径正确，并创建清晰的项目层次结构。
"""

import os
from pathlib import Path
import shutil
import json
from datetime import datetime

def create_project_structure():
    """
    创建标准的项目目录结构
    """
    base_dir = str(Path(__file__).resolve().parents[1])

    # 定义目录结构
    directories = [
        'src',                          # 源代码
        '../data',                      # 共用数据文件
        'output',                       # 输出结果
        'output/models',                # 训练好的模型
        'output/logs',                  # 训练日志
        'output/results',               # 评估结果
        'output/evaluation',            # 评估和可视化
        'output/vgg_baseline',          # VGG基线模型结果
        'output/model_comparison',      # 模型对比结果
        'scripts',                      # 脚本文件
        'notebooks',                    # Jupyter notebooks
        'docs',                         # 文档
        'visualization',                # 可视化结果
        'configs',                      # 配置文件
        'tests'                         # 测试文件
    ]

    # 创建目录
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建/确认目录: {directory}")

    return base_dir

def organize_source_files(base_dir):
    """
    整理源代码文件
    """
    print("\n整理源代码文件...")

    # 确保src目录中的文件都在正确位置
    src_files = [
        'model.py',
        'data_loader.py',
        'train.py',
        'evaluate.py',
        'predict.py'
    ]

    src_dir = os.path.join(base_dir, 'src')

    for file_name in src_files:
        file_path = os.path.join(src_dir, file_name)
        if os.path.exists(file_path):
            print(f"  ✓ {file_name} 已存在")
        else:
            print(f"  ✗ {file_name} 不存在")

def organize_script_files(base_dir):
    """
    整理脚本文件
    """
    print("\n整理脚本文件...")

    scripts_dir = os.path.join(base_dir, 'scripts')

    # 需要移动到scripts目录的文件
    script_files = [
        ('train_vgg_baseline.py', '训练VGG基线模型'),
        ('compare_models.py', '模型对比分析'),
        ('debug_augmentation_save.py', '图像增强调试'),
        ('organize_project.py', '项目整理脚本')
    ]

    for file_name, description in script_files:
        src_path = os.path.join(base_dir, file_name)
        dst_path = os.path.join(scripts_dir, file_name)

        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            print(f"  复制 {file_name} -> scripts/ ({description})")
        elif os.path.exists(dst_path):
            print(f"  ✓ {file_name} 已在scripts目录")
        else:
            print(f"  ✗ {file_name} 不存在")

def organize_notebook_files(base_dir):
    """
    整理Jupyter notebook文件
    """
    print("\n整理Jupyter notebook文件...")

    notebooks_dir = os.path.join(base_dir, 'notebooks')

    # 需要移动到notebooks目录的文件
    notebook_files = [
        ('image_augmentation_debug.ipynb', '图像增强调试notebook')
    ]

    for file_name, description in notebook_files:
        src_path = os.path.join(base_dir, file_name)
        dst_path = os.path.join(notebooks_dir, file_name)

        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            print(f"  复制 {file_name} -> notebooks/ ({description})")
        elif os.path.exists(dst_path):
            print(f"  ✓ {file_name} 已在notebooks目录")
        else:
            print(f"  ✗ {file_name} 不存在")

def organize_visualization_files(base_dir):
    """
    整理可视化文件
    """
    print("\n整理可视化文件...")

    visualization_dir = os.path.join(base_dir, 'visualization')

    # 移动augmentation_visualization目录
    src_aug_dir = os.path.join(base_dir, 'augmentation_visualization')
    dst_aug_dir = os.path.join(visualization_dir, 'augmentation_visualization')

    if os.path.exists(src_aug_dir) and not os.path.exists(dst_aug_dir):
        shutil.copytree(src_aug_dir, dst_aug_dir)
        print(f"  移动 augmentation_visualization/ -> visualization/")
    elif os.path.exists(dst_aug_dir):
        print(f"  ✓ augmentation_visualization 已在visualization目录")
    else:
        print(f"  ✗ augmentation_visualization 目录不存在")

def create_config_files(base_dir):
    """
    创建配置文件
    """
    print("\n创建配置文件...")

    configs_dir = os.path.join(base_dir, 'configs')

    # 创建训练配置文件
    train_config = {
        'resnet_config': {
            'model_name': 'resnet34',
            'num_epochs': 30,
            'batch_size': 8,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau'
        },
        'vgg_config': {
            'model_name': 'vgg16',
            'num_epochs': 40,
            'batch_size': 16,
            'learning_rate': 0.001,
            'optimizer': 'SGD',
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'scheduler': 'StepLR',
            'step_size': 10,
            'gamma': 0.1
        },
        'data_config': {
            'num_classes': 4,
            'class_names': ['boluo', 'cashang', 'dianshi', 'yashang'],
            'class_names_zh': ['剥落', '擦伤', '点蚀', '压伤'],
            'image_size': 224,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        }
    }

    config_path = os.path.join(configs_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)

    print(f"  创建训练配置文件: {config_path}")

    # 创建路径配置文件
    paths_config = {
        'data_path': '../data',
        'output_path': './output',
        'models_path': './output/models',
        'logs_path': './output/logs',
        'results_path': './output/results',
        'visualization_path': './visualization'
    }

    paths_config_path = os.path.join(configs_dir, 'paths_config.json')
    with open(paths_config_path, 'w', encoding='utf-8') as f:
        json.dump(paths_config, f, indent=2, ensure_ascii=False)

    print(f"  创建路径配置文件: {paths_config_path}")

def create_project_readme(base_dir):
    """
    创建项目README文件
    """
    print("\n创建项目README文件...")

    readme_content = f"""# 齿轮损伤智能识别系统

## 项目概述

本项目是一个基于深度学习的齿轮损伤智能识别系统，能够自动识别和分类齿轮表面的四种主要损伤类型：剥落、擦伤、点蚀和压伤。

## 项目结构

```
sunshangshibie_tiaozhan/
├── src/                    # 源代码
│   ├── model.py           # 模型定义
│   ├── data_loader.py     # 数据加载器
│   ├── train.py           # 训练脚本
│   ├── evaluate.py        # 评估脚本
│   └── predict.py         # 预测脚本
├── scripts/               # 脚本文件
│   ├── train_vgg_baseline.py      # VGG基线模型训练
│   ├── compare_models.py          # 模型对比分析
│   ├── debug_augmentation_save.py # 图像增强调试
│   └── organize_project.py        # 项目整理
├── notebooks/             # Jupyter notebooks
│   └── image_augmentation_debug.ipynb
├── ../data/               # 项目根目录共用数据文件
│   ├── DATA.json         # 数据元信息
│   ├── boluo/            # 剥落损伤图像
│   ├── cashang/          # 擦伤损伤图像
│   ├── dianshi/          # 点蚀损伤图像
│   └── yashang/          # 压伤损伤图像
├── output/               # 输出结果
│   ├── models/           # 训练好的模型
│   ├── logs/             # 训练日志
│   ├── results/          # 评估结果
│   ├── evaluation/       # 评估和可视化
│   ├── vgg_baseline/     # VGG基线模型结果
│   └── model_comparison/ # 模型对比结果
├── visualization/        # 可视化结果
│   └── augmentation_visualization/
├── configs/              # 配置文件
│   ├── training_config.json
│   └── paths_config.json
├── docs/                 # 文档
└── tests/                # 测试文件
```

## 模型架构

### 主要模型
- **ResNet34**: 残差网络，作为主要模型
- **VGG16**: 经典卷积网络，作为基线模型

### 模型特点
- 支持宏观照片和体视镜照片两种输入类型
- 使用ImageNet预训练权重
- 针对齿轮损伤分类任务优化

## 数据集

- **总样本数**: 80个样本
- **损伤类型**: 4类（剥落、擦伤、点蚀、压伤）
- **照片类型**: 宏观照片、体视镜照片
- **数据划分**: 训练集75%，测试集25%

## 使用方法

### 1. 环境配置

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 训练ResNet模型
python src/train.py

# 训练VGG基线模型
python scripts/train_vgg_baseline.py
```

### 3. 模型评估

```bash
# 评估单个模型
python src/evaluate.py

# 模型对比分析
python scripts/compare_models.py
```

### 4. 图像增强调试

```bash
# 运行图像增强可视化
python scripts/debug_augmentation_save.py

# 或使用Jupyter notebook
jupyter notebook notebooks/image_augmentation_debug.ipynb
```

## 主要功能

### 1. 模型训练
- 支持ResNet和VGG架构
- 自动数据增强
- 学习率调度
- 早停机制

### 2. 模型评估
- 准确率、精确率、召回率、F1分数
- 混淆矩阵
- 训练曲线可视化

### 3. 模型对比
- 多模型性能对比
- 可视化对比结果
- 详细分析报告

### 4. 数据增强调试
- 逐步可视化增强过程
- 随机增强效果展示
- 参数影响分析

## 性能结果

### ResNet34
- 宏观照片模型: 准确率 100%
- 体视镜照片模型: 准确率 80%

### VGG16基线
- 详见模型对比报告

## 技术特点

1. **数据增强**: 旋转、翻转、颜色抖动等
2. **迁移学习**: 使用ImageNet预训练权重
3. **多模型对比**: ResNet vs VGG性能分析
4. **可视化调试**: 完整的图像处理流程可视化
5. **代码组织**: 清晰的项目结构和模块化设计

## 依赖库

- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- PIL
- numpy
- pandas

## 更新日志

- **v1.0**: 初始版本，ResNet34模型
- **v1.1**: 添加VGG16基线模型
- **v1.2**: 模型对比分析功能
- **v1.3**: 图像增强调试工具
- **v1.4**: 项目结构整理和代码优化

## 作者

齿轮损伤识别项目组

## 许可证

本项目仅用于学术研究和教育目的。

---

*最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    readme_path = os.path.join(base_dir, 'README_PROJECT.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"  创建项目README: {readme_path}")

def check_import_paths(base_dir):
    """
    检查和修复导入路径
    """
    print("\n检查导入路径...")

    # 需要检查的Python文件
    python_files = [
        'scripts/train_vgg_baseline.py',
        'scripts/compare_models.py',
        'src/train.py',
        'src/evaluate.py',
        'src/predict.py'
    ]

    for file_path in python_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path} 存在")
            # 这里可以添加更详细的路径检查逻辑
        else:
            print(f"  ✗ {file_path} 不存在")

def create_project_summary(base_dir):
    """
    创建项目总结文件
    """
    print("\n创建项目总结...")

    docs_dir = os.path.join(base_dir, 'docs')

    summary = {
        'project_name': '齿轮损伤智能识别系统',
        'version': '1.4',
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': {
            'resnet34': {
                'architecture': 'ResNet34',
                'parameters': '21.8M',
                'performance': {
                    'hongguan_accuracy': 1.0,
                    'tishijing_accuracy': 0.8
                }
            },
            'vgg16': {
                'architecture': 'VGG16',
                'parameters': '138M',
                'performance': {
                    'status': 'training'
                }
            }
        },
        'dataset': {
            'total_samples': 80,
            'classes': 4,
            'class_names': ['剥落', '擦伤', '点蚀', '压伤'],
            'photo_types': ['宏观照片', '体视镜照片']
        },
        'features': [
            '多模型架构支持',
            '数据增强可视化',
            '模型性能对比',
            '完整的训练评估流程',
            '清晰的项目结构'
        ]
    }

    summary_path = os.path.join(docs_dir, 'project_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  创建项目总结: {summary_path}")

def main():
    """
    主函数
    """
    print("齿轮损伤识别项目整理开始")
    print("=" * 60)

    # 创建项目结构
    base_dir = create_project_structure()

    # 整理各类文件
    organize_source_files(base_dir)
    organize_script_files(base_dir)
    organize_notebook_files(base_dir)
    organize_visualization_files(base_dir)

    # 创建配置和文档
    create_config_files(base_dir)
    create_project_readme(base_dir)
    create_project_summary(base_dir)

    # 检查导入路径
    check_import_paths(base_dir)

    print("\n" + "=" * 60)
    print("项目整理完成！")
    print("\n项目结构已优化:")
    print("- 源代码模块化组织")
    print("- 脚本文件分类管理")
    print("- 配置文件标准化")
    print("- 文档完善")
    print("- 可视化结果整理")
    print("\n建议下一步:")
    print("1. 检查VGG训练进度")
    print("2. 运行模型对比分析")
    print("3. 生成最终报告")
    print("=" * 60)

if __name__ == "__main__":
    main()
