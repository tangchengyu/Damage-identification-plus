import os
from pathlib import Path
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

from data_loader import create_data_loaders
from model import create_damage_classifier, save_model

class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_acc = 0.0
        self.best_epoch = 0

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """记录每个epoch的结果"""
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = epoch

        print(f'Epoch [{epoch+1}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    def save_plots(self, photo_type):
        """保存训练过程图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', color='blue')
        ax1.plot(self.val_losses, label='验证损失', color='red')
        ax1.set_title(f'{photo_type} - 损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率', color='blue')
        ax2.plot(self.val_accuracies, label='验证准确率', color='red')
        ax2.set_title(f'{photo_type} - 准确率曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'{photo_type}_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"训练曲线已保存: {os.path.join(self.log_dir, f'{photo_type}_training_curves.png')}")

    def save_log(self, photo_type):
        """保存训练日志"""
        # 将所有数值转换为Python原生类型，避免JSON序列化错误
        def convert_to_python_type(value):
            if hasattr(value, 'item'):  # 处理Tensor标量
                return value.item()
            elif isinstance(value, list):
                return [convert_to_python_type(v) for v in value]
            else:
                return float(value) if isinstance(value, (int, float)) else value

        log_data = {
            'photo_type': photo_type,
            'best_epoch': int(self.best_epoch),
            'best_accuracy': convert_to_python_type(self.best_acc),
            'train_losses': [convert_to_python_type(loss) for loss in self.train_losses],
            'train_accuracies': [convert_to_python_type(acc) for acc in self.train_accuracies],
            'val_losses': [convert_to_python_type(loss) for loss in self.val_losses],
            'val_accuracies': [convert_to_python_type(acc) for acc in self.val_accuracies]
        }

        log_path = os.path.join(self.log_dir, f'{photo_type}_training_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"训练日志已保存: {log_path}")

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
                num_epochs=50, device='cpu', photo_type='', logger=None):
    """
    训练模型的主函数

    Args:
        model: 要训练的模型
        dataloaders: 数据加载器字典
        dataset_sizes: 数据集大小字典
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 设备
        photo_type: 照片类型
        logger: 日志记录器

    Returns:
        model: 训练好的模型
    """
    if 'val' not in dataloaders or 'val' not in dataset_sizes:
        raise ValueError('A separate val split is required; test cannot select checkpoints. '
                         'Provide a reviewed train/val/test partition before training.')
    if any(dataset_sizes.get(phase, 0) <= 0 for phase in ('train', 'val')):
        raise ValueError('train and val datasets must be non-empty')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 每个epoch有训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零化参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 只在训练时计算梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 后向传播和优化，只在训练阶段
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss, train_acc = epoch_loss, epoch_acc
            else:
                val_loss, val_acc = epoch_loss, epoch_acc

                # 深拷贝模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # 记录每个epoch的结果
        if logger:
            logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        epoch_time = time.time() - epoch_start
        print(f'Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s')
        print('-' * 60)

    time_elapsed = time.time() - since
    print(f'训练完成，用时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, device, class_names):
    """
    评估模型性能

    Args:
        model: 训练好的模型
        dataloader: 测试数据加载器
        device: 设备
        class_names: 类别名称列表

    Returns:
        dict: 评估结果
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels
    }

    return results

def plot_confusion_matrix(cm, class_names, title, save_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")

def train_single_model(data_path, photo_type, output_dir, num_epochs=50, batch_size=16, learning_rate=1e-4):
    """
    训练单个模型（宏观照片或体视镜照片）

    Args:
        data_path: 数据路径
        photo_type: 照片类型 ('hongguan' 或 'tishijing')
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率

    Returns:
        dict: 训练结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"开始训练 {photo_type} 模型...")

    # 创建数据加载器
    dataloaders, dataset_sizes = create_data_loaders(data_path, photo_type, batch_size)
    if 'val' not in dataloaders:
        raise ValueError('Provide a reviewed val split before training; test is reserved for final evaluation')
    print(f"数据集大小: {dataset_sizes}")

    # 创建模型
    model = create_damage_classifier(num_classes=4, pretrained=True, model_name='resnet34')
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 学习率调度器
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 创建日志记录器
    log_dir = os.path.join(output_dir, 'logs')
    logger = TrainingLogger(log_dir)

    # 训练模型
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer,
                       exp_lr_scheduler, num_epochs, device, photo_type, logger)

    # 保存模型
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'model_{photo_type}.pth')
    save_model(model, model_path)

    # 评估模型
    class_names = ['boluo', 'cashang', 'dianshi', 'yashang']
    results = evaluate_model(model, dataloaders['test'], device, class_names)

    print(f"\n{photo_type} 模型评估结果:")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1_score']:.4f}")

    # 保存混淆矩阵
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    cm_path = os.path.join(results_dir, f'{photo_type}_confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], class_names,
                         f'{photo_type} 模型混淆矩阵', cm_path)

    # 保存训练曲线和日志
    logger.save_plots(photo_type)
    logger.save_log(photo_type)

    # 保存评估结果
    results_to_save = {
        'photo_type': photo_type,
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'model_path': model_path
    }

    results_path = os.path.join(results_dir, f'{photo_type}_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    return results_to_save

def main():
    """主训练函数"""
    # 配置参数
    data_path = str(Path(__file__).resolve().parents[2] / 'data')
    output_dir = str(Path(__file__).resolve().parents[1] / 'output')

    # 训练参数
    num_epochs = 30  # 减少epochs用于快速测试
    batch_size = 8   # 适应小数据集
    learning_rate = 1e-4

    print("="*80)
    print("开始训练齿轮损伤识别模型")
    print("="*80)

    # 训练宏观照片模型
    print("\n1. 训练宏观照片模型...")
    hongguan_results = train_single_model(data_path, 'hongguan', output_dir,
                                         num_epochs, batch_size, learning_rate)

    print("\n" + "="*80)

    # 训练体视镜照片模型
    print("\n2. 训练体视镜照片模型...")
    tishijing_results = train_single_model(data_path, 'tishijing', output_dir,
                                          num_epochs, batch_size, learning_rate)

    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)

    # 汇总结果
    print("\n训练结果汇总:")
    print(f"宏观照片模型 - 准确率: {hongguan_results['accuracy']:.4f}, F1: {hongguan_results['f1_score']:.4f}")
    print(f"体视镜照片模型 - 准确率: {tishijing_results['accuracy']:.4f}, F1: {tishijing_results['f1_score']:.4f}")

    return hongguan_results, tishijing_results

if __name__ == "__main__":
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    hongguan_results, tishijing_results = main()
