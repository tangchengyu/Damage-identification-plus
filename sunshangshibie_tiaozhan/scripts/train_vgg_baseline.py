#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGG基线模型训练脚本

本脚本用于训练基于VGG的齿轮损伤分类基线模型，并与ResNet模型进行对比。
使用针对VGG优化的超参数设置。
"""

import os
import copy
from pathlib import Path
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import create_vgg_damage_classifier
from data_loader import create_data_loaders

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def train_vgg_model(data_path, output_dir, photo_type='hongguan',
                   num_epochs=40, batch_size=16, learning_rate=0.001):
    """
    训练VGG基线模型

    Args:
        data_path: 数据路径
        output_dir: 输出目录
        photo_type: 照片类型 ('hongguan' 或 'tishijing')
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率

    Returns:
        dict: 训练结果
    """
    print(f"开始训练VGG基线模型 - {photo_type}")
    print(f"超参数设置: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    dataloaders, dataset_sizes = create_data_loaders(
        data_path, photo_type=photo_type, batch_size=batch_size
    )

    print(f"数据集大小: 训练集 {dataset_sizes['train']}, 测试集 {dataset_sizes['test']}")
    if 'val' not in dataloaders or dataset_sizes.get('val', 0) <= 0:
        raise ValueError('Provide a reviewed val split before VGG training; test cannot select checkpoints')

    # 创建VGG模型
    model = create_vgg_damage_classifier(num_classes=4, pretrained=True, model_name='vgg16')
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VGG16模型参数量: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # VGG适用的优化器设置：较小的学习率，使用SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # 学习率调度器：每10个epoch衰减0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练历史记录
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # 训练循环
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        train_history['learning_rates'].append(current_lr)
        print(f'当前学习率: {current_lr:.6f}')

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式
            else:
                model.eval()   # 设置为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零参数梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化（仅在训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录训练历史
            if phase == 'train':
                train_history['train_loss'].append(epoch_loss)
                train_history['train_acc'].append(epoch_acc.item())
            else:
                train_history['val_loss'].append(epoch_loss)
                train_history['val_acc'].append(epoch_acc.item())

                # 保存最佳模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f'新的最佳验证准确率: {best_acc:.4f}')

        # 更新学习率
        scheduler.step()

    training_time = time.time() - start_time
    print(f'\n训练完成，用时: {training_time//60:.0f}m {training_time%60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 保存模型
    model_save_path = os.path.join(output_dir, f'vgg16_{photo_type}_best.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'最佳模型已保存: {model_save_path}')

    # 评估模型
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 保存训练历史
    history_path = os.path.join(output_dir, f'vgg16_{photo_type}_training_log.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        history_to_save = {
            'train_loss': [float(x) for x in train_history['train_loss']],
            'train_acc': [float(x) for x in train_history['train_acc']],
            'val_loss': [float(x) for x in train_history['val_loss']],
            'val_acc': [float(x) for x in train_history['val_acc']],
            'learning_rates': [float(x) for x in train_history['learning_rates']],
            'best_accuracy': float(best_acc),
            'training_time': training_time,
            'hyperparameters': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'optimizer': 'SGD',
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'scheduler': 'StepLR',
                'step_size': 10,
                'gamma': 0.1
            }
        }
        json.dump(history_to_save, f, indent=2, ensure_ascii=False)

    # 保存评估结果
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'model_name': 'VGG16',
        'photo_type': photo_type,
        'dataset_sizes': dataset_sizes
    }

    results_path = os.path.join(output_dir, f'vgg16_{photo_type}_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 绘制训练曲线
    plot_training_curves(train_history, output_dir, photo_type, 'VGG16')

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, output_dir, photo_type, 'VGG16')

    print(f'\n=== VGG16 {photo_type} 模型评估结果 ===')
    print(f'准确率: {accuracy:.4f}')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1分数: {f1:.4f}')

    return results

def plot_training_curves(history, output_dir, photo_type, model_name):
    """绘制训练曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失')
    ax1.set_title(f'{model_name} {photo_type} - 损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
    ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率')
    ax2.set_title(f'{model_name} {photo_type} - 准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # 学习率曲线
    ax3.plot(epochs, history['learning_rates'], 'g-', label='学习率')
    ax3.set_title(f'{model_name} {photo_type} - 学习率变化')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)

    # 验证准确率详细视图
    ax4.plot(epochs, history['val_acc'], 'r-', linewidth=2)
    ax4.set_title(f'{model_name} {photo_type} - 验证准确率详细')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy')
    ax4.grid(True)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(output_dir, f'{model_name.lower()}_{photo_type}_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'训练曲线已保存: {save_path}')

def plot_confusion_matrix(cm, output_dir, photo_type, model_name):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))

    # 类别标签
    class_names = ['剥落', '擦伤', '点蚀', '压伤']

    # 绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title(f'{model_name} {photo_type} - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 保存图片
    save_path = os.path.join(output_dir, f'{model_name.lower()}_{photo_type}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'混淆矩阵已保存: {save_path}')

def main():
    """主函数"""
    # 配置路径
    data_path = str(Path(__file__).resolve().parents[2] / 'data')
    output_dir = str(Path(__file__).resolve().parents[1] / 'output/vgg_baseline')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("VGG基线模型训练开始")
    print("=" * 60)

    # VGG优化的超参数
    vgg_hyperparams = {
        'num_epochs': 40,      # VGG通常需要更多epoch
        'batch_size': 16,      # 较小的batch size适合VGG
        'learning_rate': 0.001 # 较小的学习率
    }

    print(f"VGG超参数设置: {vgg_hyperparams}")

    # 训练宏观照片模型
    print("\n训练宏观照片VGG模型...")
    hongguan_results = train_vgg_model(
        data_path=data_path,
        output_dir=output_dir,
        photo_type='hongguan',
        **vgg_hyperparams
    )

    # 训练体视镜照片模型
    print("\n训练体视镜照片VGG模型...")
    tishijing_results = train_vgg_model(
        data_path=data_path,
        output_dir=output_dir,
        photo_type='tishijing',
        **vgg_hyperparams
    )

    # 保存总体结果
    overall_results = {
        'vgg_baseline_results': {
            'hongguan': hongguan_results,
            'tishijing': tishijing_results
        },
        'hyperparameters': vgg_hyperparams,
        'model_architecture': 'VGG16',
        'training_completed': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    overall_results_path = os.path.join(output_dir, 'vgg_baseline_overall_results.json')
    with open(overall_results_path, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("VGG基线模型训练完成！")
    print(f"结果保存在: {output_dir}")
    print("\n训练结果摘要:")
    print(f"宏观照片模型 - 准确率: {hongguan_results['accuracy']:.4f}")
    print(f"体视镜照片模型 - 准确率: {tishijing_results['accuracy']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
