#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
齿轮损伤智能识别系统 - 脚本工具包

本包提供了项目相关的各种脚本工具:
- 模型训练脚本
- 模型对比分析
- 数据增强调试
- 项目组织工具
- 可视化结果整理
"""

__version__ = "1.0.0"
__description__ = "齿轮损伤识别系统脚本工具集"

# 脚本模块列表
SCRIPT_MODULES = [
    "compare_models",           # 模型对比分析
    "train_vgg_baseline",       # VGG基线训练
    "debug_augmentation_save",  # 数据增强调试
    "organize_project",         # 项目组织
    "organize_visualization_results",  # 可视化整理
    "run_pipeline"              # 流水线执行
]

# 使用说明
USAGE_INFO = """
脚本使用说明:

1. 模型对比分析:
   python scripts/compare_models.py

2. VGG基线训练:
   python scripts/train_vgg_baseline.py

3. 数据增强调试:
   python scripts/debug_augmentation_save.py

4. 项目组织:
   python scripts/organize_project.py

5. 可视化结果整理:
   python scripts/organize_visualization_results.py

6. 完整流水线:
   python scripts/run_pipeline.py
"""

def show_usage():
    """显示脚本使用说明"""
    print(USAGE_INFO)

if __name__ == "__main__":
    show_usage()
