#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
齿轮损伤图像预处理与数据增强可视化脚本（保存版本）

本脚本用于调试和可视化图像预处理的每个步骤，帮助理解数据增强对图像的影响。
运行此脚本将生成多个可视化图表并保存为图片文件。
"""

import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import random

# 设置matplotlib后端为非交互式
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可重现
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def load_sample_images(data_root):
    """加载一些示例图像"""
    json_path = os.path.join(data_root, 'DATA.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 选择每个类别的一张图片作为示例
    sample_images = {}
    damage_types = ['boluo', 'cashang', 'dianshi', 'yashang']
    damage_types_zh = ['剥落', '擦伤', '点蚀', '压伤']

    for i, damage_type in enumerate(damage_types):
        if damage_type in metadata:
            # 获取该类别的第一个样本
            first_sample_key = list(metadata[damage_type].keys())[0]
            first_sample = metadata[damage_type][first_sample_key]

            # 构建图像路径
            path_info = first_sample['path']
            # 解析路径信息: data_boluo_hongguan_train_4052-1
            parts = path_info.split('_')
            if len(parts) >= 4:
                category = parts[1]  # boluo
                photo_type = parts[2]  # hongguan
                split = parts[3]  # train
                filename = parts[4] if len(parts) > 4 else '4052-1'  # 4052-1

                # 构建完整路径
                image_path = os.path.join(data_root, category, photo_type, split, f"{filename}.jpg")

                if os.path.exists(image_path):
                    sample_images[damage_types_zh[i]] = {
                        'path': image_path,
                        'label': damage_type,
                        'label_zh': damage_types_zh[i]
                    }

    return sample_images

def create_individual_transforms():
    """创建单个变换步骤用于可视化"""
    return {
        '原始图像': transforms.Compose([]),
        '调整尺寸(256x256)': transforms.Compose([
            transforms.Resize(256)
        ]),
        '随机旋转(±15°)': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(15)
        ]),
        '随机水平翻转': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=1.0)  # 强制翻转以显示效果
        ]),
        '随机垂直翻转': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=1.0)  # 强制翻转以显示效果
        ]),
        '颜色抖动': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)  # 增强效果以便观察
        ]),
        '中心裁剪(224x224)': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.CenterCrop(224)
        ])
    }

def visualize_augmentation_steps(image_path, label_zh, transform_steps, output_dir):
    """可视化图像增强的每个步骤"""
    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')

    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # 应用每个变换步骤并显示
    for i, (step_name, transform) in enumerate(transform_steps.items()):
        if i >= 8:  # 只显示前8个步骤
            break

        # 应用变换
        if step_name == '原始图像':
            transformed_image = original_image
        else:
            # 设置随机种子以确保一致性
            torch.manual_seed(42 + i)
            random.seed(42 + i)
            transformed_image = transform(original_image)

        # 显示图像
        axes[i].imshow(transformed_image)
        axes[i].set_title(f'{step_name}\n尺寸: {transformed_image.size}', fontsize=12)
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(len(transform_steps), 8):
        axes[i].axis('off')

    plt.suptitle(f'{label_zh} - 图像增强步骤可视化', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, f'{label_zh}_增强步骤.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存: {output_path}")
    return transformed_image

def visualize_complete_pipeline(image_path, label_zh, output_dir):
    """可视化完整的训练数据预处理流程"""
    # 定义完整的训练变换
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 定义测试变换
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')

    # 应用变换
    torch.manual_seed(42)
    train_tensor = train_transform(original_image)
    test_tensor = test_transform(original_image)

    # 反归一化以便显示
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    train_denorm = denormalize(train_tensor)
    test_denorm = denormalize(test_tensor)

    # 转换为numpy数组用于显示
    train_np = train_denorm.permute(1, 2, 0).numpy()
    test_np = test_denorm.permute(1, 2, 0).numpy()

    # 裁剪到[0,1]范围
    train_np = np.clip(train_np, 0, 1)
    test_np = np.clip(test_np, 0, 1)

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title(f'原始图像\n尺寸: {original_image.size}', fontsize=14)
    axes[0].axis('off')

    # 训练变换后
    axes[1].imshow(train_np)
    axes[1].set_title(f'训练变换后\n尺寸: {train_tensor.shape[1:]}\n(包含数据增强)', fontsize=14)
    axes[1].axis('off')

    # 测试变换后
    axes[2].imshow(test_np)
    axes[2].set_title(f'测试变换后\n尺寸: {test_tensor.shape[1:]}\n(无数据增强)', fontsize=14)
    axes[2].axis('off')

    plt.suptitle(f'{label_zh} - 完整预处理流程对比', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, f'{label_zh}_完整流程.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存: {output_path}")

    # 打印张量信息
    print(f"  训练张量形状: {train_tensor.shape}")
    print(f"  训练张量数据类型: {train_tensor.dtype}")
    print(f"  训练张量值范围: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
    print(f"  测试张量形状: {test_tensor.shape}")
    print(f"  测试张量数据类型: {test_tensor.dtype}")
    print(f"  测试张量值范围: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")

def show_multiple_augmentations(image_path, label_zh, output_dir, num_samples=6):
    """展示同一张图片的多次随机增强结果"""
    # 定义训练变换
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.CenterCrop(224)
    ])

    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 生成多个随机增强版本
    for i in range(num_samples):
        # 使用不同的随机种子
        torch.manual_seed(i * 10)
        random.seed(i * 10)

        # 应用变换
        augmented_image = train_transform(original_image)

        # 显示图像
        axes[i].imshow(augmented_image)
        axes[i].set_title(f'随机增强 #{i+1}', fontsize=12)
        axes[i].axis('off')

    plt.suptitle(f'{label_zh} - 多次随机增强效果对比', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, f'{label_zh}_多次随机增强.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存: {output_path}")

def analyze_augmentation_parameters():
    """分析当前使用的数据增强参数"""
    print("\n当前数据增强参数分析:")
    print("=" * 50)

    augmentation_info = {
        'Resize(256)': '将图像调整为256x256像素，保持宽高比',
        'RandomRotation(15)': '随机旋转±15度，增加旋转不变性',
        'RandomHorizontalFlip()': '50%概率水平翻转，增加水平对称性',
        'RandomVerticalFlip()': '50%概率垂直翻转，增加垂直对称性',
        'ColorJitter(0.1, 0.1, 0.1)': '轻微调整亮度、对比度、饱和度(±10%)',
        'CenterCrop(224)': '中心裁剪为224x224像素，匹配模型输入',
        'ToTensor()': '转换为PyTorch张量，像素值归一化到[0,1]',
        'Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])': 'ImageNet标准化，匹配预训练模型'
    }

    for i, (transform, description) in enumerate(augmentation_info.items(), 1):
        print(f"{i}. {transform}")
        print(f"   {description}")

    print("\n数据增强的作用:")
    print("- 增加数据多样性，提高模型泛化能力")
    print("- 模拟真实场景中的图像变化")
    print("- 减少过拟合风险")
    print("- 提高模型对几何变换和光照变化的鲁棒性")

def main():
    """主函数"""
    # 定义数据路径
    data_root = str(Path(__file__).resolve().parents[2] / 'data')
    output_dir = str(Path(__file__).resolve().parents[1] / 'augmentation_visualization')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("齿轮损伤图像预处理与数据增强可视化")
    print("=" * 60)
    print(f"输出目录: {output_dir}")

    # 加载示例图像
    print("\n1. 加载示例图像...")
    sample_images = load_sample_images(data_root)
    print("加载的示例图像:")
    for label, info in sample_images.items():
        print(f"  {label}: {info['path']}")

    if not sample_images:
        print("错误: 未找到示例图像，请检查数据路径")
        return

    # 创建变换步骤
    transform_steps = create_individual_transforms()

    # 可视化每个损伤类别的图像增强过程
    print("\n2. 可视化图像增强步骤...")
    for label_zh, info in sample_images.items():
        print(f"\n处理: {label_zh} ({info['label']})")
        visualize_augmentation_steps(info['path'], label_zh, transform_steps, output_dir)

    # 可视化完整的预处理流程
    print("\n3. 可视化完整预处理流程...")
    for label_zh, info in sample_images.items():
        print(f"\n处理: {label_zh} ({info['label']})")
        visualize_complete_pipeline(info['path'], label_zh, output_dir)

    # 展示多次随机增强效果
    print("\n4. 展示多次随机增强效果...")
    first_sample = list(sample_images.items())[0]
    label_zh, info = first_sample

    print(f"处理: {label_zh} 的多次随机增强效果")
    show_multiple_augmentations(info['path'], label_zh, output_dir)

    # 分析数据增强参数
    analyze_augmentation_parameters()

    print("\n" + "="*60)
    print("可视化完成！")
    print(f"所有图片已保存到: {output_dir}")
    print("\n生成的文件包括:")
    print("- 各类别的图像增强步骤可视化")
    print("- 各类别的完整预处理流程对比")
    print("- 多次随机增强效果对比")
    print("\n通过这些可视化，您可以清楚地看到:")
    print("- 每个数据增强步骤对图像的具体影响")
    print("- 训练和测试时的不同预处理流程")
    print("- 随机增强如何增加数据多样性")
    print("- 输入神经网络前图像的最终形态")
    print("="*60)

if __name__ == "__main__":
    main()
