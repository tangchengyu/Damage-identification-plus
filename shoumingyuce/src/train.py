import os
import copy
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from model import create_lifetime_model
from data_loader import create_lifetime_dataloaders

class EarlyStopping:
    """
    早停策略类
    当验证损失不再改善时停止训练
    """

    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        """
        初始化早停策略

        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善幅度
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        """
        检查是否应该早停

        Args:
            val_loss: 验证损失
            model: 模型

        Returns:
            是否应该早停
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        self.best_weights = copy.deepcopy(model.state_dict())

def convert_numpy_types(obj):
    """
    递归转换numpy类型为Python原生类型，用于JSON序列化

    Args:
        obj: 要转换的对象

    Returns:
        转换后的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def calculate_metrics(y_true, y_pred):
    """
    计算回归评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        指标字典
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 计算平均绝对百分比误差 (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """
    训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        writer: TensorBoard写入器

    Returns:
        平均损失
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # 将数据移到设备上
        image = batch['image'].to(device)
        size_features = batch['size_features'].to(device)
        operation_features = batch['operation_features'].to(device)
        material_features = batch['material_features'].to(device)
        lifetime = batch['lifetime'].to(device).unsqueeze(1)  # [batch_size, 1]

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(image, size_features, operation_features, material_features)

        # 计算损失
        loss = criterion(outputs, lifetime)

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

        # 记录到TensorBoard
        if writer is not None:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)

        # 打印进度
        if batch_idx % 1 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')

    avg_loss = running_loss / num_batches
    return avg_loss

def validate_epoch(model, dataloader, criterion, device, lifetime_scaler=None):
    """
    验证一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        lifetime_scaler: 寿命标准化器，用于逆变换

    Returns:
        平均损失和评估指标
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    all_predictions_original = []  # 逆变换后的预测值
    all_targets_original = []      # 逆变换后的真实值

    with torch.no_grad():
        for batch in dataloader:
            # 将数据移到设备上
            image = batch['image'].to(device)
            size_features = batch['size_features'].to(device)
            operation_features = batch['operation_features'].to(device)
            material_features = batch['material_features'].to(device)
            lifetime = batch['lifetime'].to(device).unsqueeze(1)

            # 前向传播
            outputs = model(image, size_features, operation_features, material_features)

            # 计算损失（在标准化空间中）
            loss = criterion(outputs, lifetime)
            running_loss += loss.item()

            # 收集标准化后的预测值和真实值
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(lifetime.cpu().numpy().flatten())

            # 如果有寿命标准化器，进行逆变换
            if lifetime_scaler is not None:
                # 逆变换预测值
                pred_original = lifetime_scaler.inverse_transform(
                    outputs.cpu().numpy().reshape(-1, 1)
                ).flatten()
                # 逆变换真实值
                target_original = lifetime_scaler.inverse_transform(
                    lifetime.cpu().numpy().reshape(-1, 1)
                ).flatten()

                all_predictions_original.extend(pred_original)
                all_targets_original.extend(target_original)

    avg_loss = running_loss / len(dataloader)

    # 如果有逆变换的值，使用原始尺度计算指标
    if lifetime_scaler is not None and all_predictions_original:
        metrics = calculate_metrics(np.array(all_targets_original), np.array(all_predictions_original))
        return avg_loss, metrics, all_predictions_original, all_targets_original
    else:
        # 否则使用标准化后的值计算指标
        metrics = calculate_metrics(np.array(all_targets), np.array(all_predictions))
        return avg_loss, metrics, all_predictions, all_targets

def train_model(config, data_path, output_dir, pretrained_damage_model_path=None, exclude_outliers=None):
    """
    训练寿命预测模型

    Args:
        config: 配置字典
        data_path: 数据路径
        output_dir: 输出目录
        pretrained_damage_model_path: 预训练损伤识别模型路径
        exclude_outliers: 要排除的异常值样本ID列表

    Returns:
        训练好的模型和训练历史
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    print("创建数据加载器...")
    if exclude_outliers:
        print(f"排除异常值样本: {exclude_outliers}")
    dataloaders, dataset_sizes, scalers = create_lifetime_dataloaders(
        data_path=data_path,
        photo_type='hongguan',
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        save_scalers=True,
        exclude_outliers=exclude_outliers
    )

    print(f"特征标准化器: {type(scalers['feature_scaler'])}")
    print(f"寿命标准化器: {type(scalers['lifetime_scaler'])}")
    print(f"异常值检测器: {type(scalers['outlier_detector'])}")

    print(f"数据集大小: {dataset_sizes}")

    # 创建模型
    print("创建模型...")
    model = create_lifetime_model(config, pretrained_damage_model_path)
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化可训练参数
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 早停策略
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        restore_best_weights=True
    )

    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    print("开始训练...")
    start_time = time.time()

    for epoch in range(config['training']['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["training"]["num_epochs"]}')
        print('-' * 50)

        # 训练阶段
        train_loss = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch, writer
        )

        # 验证阶段
        val_loss, val_metrics, val_predictions, val_targets = validate_epoch(
            model, dataloaders['val'], criterion, device, scalers['lifetime_scaler']
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/R2', val_metrics['r2'], epoch)
        writer.add_scalar('Metrics/RMSE', val_metrics['rmse'], epoch)
        writer.add_scalar('Metrics/MAE', val_metrics['mae'], epoch)
        writer.add_scalar('Metrics/MAPE', val_metrics['mape'], epoch)

        # 打印结果
        print(f'训练损失: {train_loss:.6f}')
        print(f'验证损失: {val_loss:.6f}')
        print(f'验证R²: {val_metrics["r2"]:.4f}')
        print(f'验证RMSE: {val_metrics["rmse"]:.2f}')
        print(f'验证MAE: {val_metrics["mae"]:.2f}')
        print(f'验证MAPE: {val_metrics["mape"]:.2f}%')

        # 早停检查
        if early_stopping(val_loss, model):
            print(f'\n早停触发，在第{epoch+1}轮停止训练')
            break

    if early_stopping.best_weights is None:
        raise ValueError('No validated checkpoint was produced')
    # Also restore when the epoch limit is reached before early stopping.
    model.load_state_dict(early_stopping.best_weights)
    training_time = time.time() - start_time
    print(f'\n训练完成，总用时: {training_time:.2f}秒')

    # 创建子文件夹
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')
    plots_dir = os.path.join(output_dir, 'plots')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 保存模型到models文件夹
    model_path = os.path.join(models_dir, 'best_lifetime_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_history': history,
        'dataset_sizes': dataset_sizes,
        'scalers': scalers  # 保存标准化器
    }, model_path)
    print(f'模型已保存到: {model_path}')
    print(f'标准化器已保存到模型文件中')

    # 保存训练历史到results文件夹
    history_path = os.path.join(results_dir, 'training_history.json')
    # 转换numpy类型为Python原生类型
    history_serializable = convert_numpy_types(history)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_serializable, f, indent=2, ensure_ascii=False)

    # 绘制训练曲线到plots文件夹
    plot_training_curves(history, plots_dir)

    # 最终测试
    print('\n在测试集上评估...')
    test_loss, test_metrics, test_predictions, test_targets = validate_epoch(
        model, dataloaders['test'], criterion, device, scalers['lifetime_scaler']
    )

    print('注意：以下指标基于逆变换后的真实寿命值计算')

    print(f'测试损失: {test_loss:.6f}')
    print(f'测试R²: {test_metrics["r2"]:.4f}')
    print(f'测试RMSE: {test_metrics["rmse"]:.2f}')
    print(f'测试MAE: {test_metrics["mae"]:.2f}')
    print(f'测试MAPE: {test_metrics["mape"]:.2f}%')

    # 保存测试结果到results文件夹
    test_results = {
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'predictions': test_predictions,
        'targets': test_targets
    }

    test_results_path = os.path.join(results_dir, 'test_results.json')
    # 转换numpy类型为Python原生类型
    test_results_serializable = convert_numpy_types(test_results)
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results_serializable, f, indent=2, ensure_ascii=False)

    # 绘制预测vs真实值散点图到plots文件夹
    plot_predictions_vs_targets(test_predictions, test_targets, plots_dir)

    writer.close()

    return model, history, test_results

def plot_training_curves(history, output_dir):
    """
    绘制训练曲线

    Args:
        history: 训练历史
        output_dir: 输出目录
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    # R²曲线
    plt.subplot(1, 3, 2)
    r2_scores = [metrics['r2'] for metrics in history['val_metrics']]
    plt.plot(epochs, r2_scores, 'g-', label='验证R²')
    plt.title('验证R²分数')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)

    # RMSE曲线
    plt.subplot(1, 3, 3)
    rmse_scores = [metrics['rmse'] for metrics in history['val_metrics']]
    plt.plot(epochs, rmse_scores, 'm-', label='验证RMSE')
    plt.title('验证RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions_vs_targets(predictions, targets, output_dir):
    """
    绘制预测值vs真实值散点图

    Args:
        predictions: 预测值
        targets: 真实值
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 8))

    plt.scatter(targets, predictions, alpha=0.6, s=50)

    # 绘制理想线（y=x）
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')

    plt.xlabel('真实寿命值')
    plt.ylabel('预测寿命值')
    plt.title('预测值 vs 真实值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加R²信息
    r2 = r2_score(targets, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_targets.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 加载配置
    config_path = str(Path(__file__).resolve().parents[1] / 'configs/config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 设置路径
    data_path = str(Path(__file__).resolve().parents[2] / 'data')
    output_dir = str(Path(__file__).resolve().parents[1] / 'output')
    pretrained_model_path = str((Path(__file__).resolve().parents[1] / config['data']['damage_model_path']).resolve())

    # 检查预训练模型是否存在
    if not os.path.exists(pretrained_model_path):
        print(f"警告: 预训练模型不存在: {pretrained_model_path}")
        print("将使用ImageNet预训练权重")
        pretrained_model_path = None

    # 开始训练
    model, history, test_results = train_model(
        config=config,
        data_path=data_path,
        output_dir=output_dir,
        pretrained_damage_model_path=pretrained_model_path
    )

    print("\n训练完成！")
