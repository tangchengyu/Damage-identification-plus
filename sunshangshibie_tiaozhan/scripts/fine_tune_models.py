import os
from pathlib import Path
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import create_data_loaders
from model import create_damage_classifier, save_model
from train import TrainingLogger, train_model, evaluate_model, plot_confusion_matrix

def load_pretrained_model(model_path, num_classes=4):
    """
    加载预训练模型

    Args:
        model_path: 预训练模型路径
        num_classes: 分类数量

    Returns:
        model: 加载的模型
    """
    # 创建模型架构
    model = create_damage_classifier(num_classes=num_classes, pretrained=False, model_name='resnet34')

    # 加载预训练权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    print(f"成功加载预训练模型: {model_path}")
    return model

def load_extra_dataset_with_metadata(data_path):
    """
    专门加载extra_data的数据集和元信息

    Args:
        data_path: extra_data目录路径

    Returns:
        list: 包含所有样本信息的列表
    """
    # 加载 DATA.json 文件
    json_path = os.path.join(data_path, 'DATA.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    dataset = []
    labels = ['boluo', 'cashang', 'dianshi', 'yashang']
    label_map = {label: i for i, label in enumerate(labels)}

    # 遍历 JSON 数据
    for label in labels:
        if label in metadata:
            for sample_id, sample_info in metadata[label].items():
                # 解析路径信息 - extra_data格式: "extra_data_dianshi_hongguan_train_1"
                path_parts = sample_info['path'].split('_')
                if len(path_parts) >= 5:
                    # path_parts: ['extra', 'data', 'dianshi', 'hongguan', 'train', '1']
                    damage_type = path_parts[2]  # dianshi 或 yashang
                    photo_type = path_parts[3]   # hongguan 或 tishijing
                    split = path_parts[4]        # train 或 test
                    img_filename_base = path_parts[5]  # 基础文件名

                    # extra_data中的实际文件名就是数字.jpg
                    img_filename = img_filename_base + '.jpg'
                    img_path = os.path.join(data_path, damage_type, photo_type, split, img_filename)

                    # 检查文件是否存在
                    if os.path.exists(img_path):
                        dataset.append({
                            'sample_id': sample_id,
                            'image_path': img_path,
                            'label_name': label,
                            'label_id': label_map[label],
                            'category_vector': sample_info['category-vector'],
                            'photo_type': photo_type,
                            'split': split,
                            'shouming': sample_info.get('shouming'),
                            'technical_params': sample_info.get('shuxing', {})
                        })
                    else:
                        print(f"警告: 文件不存在: {img_path}")

    return dataset

def create_combined_data_loaders(original_data_path, extra_data_path, photo_type, batch_size=16):
    """
    创建结合原始数据和额外数据的数据加载器

    Args:
        original_data_path: 原始数据路径
        extra_data_path: 额外数据路径
        photo_type: 照片类型 ('hongguan' 或 'tishijing')
        batch_size: 批次大小

    Returns:
        dataloaders: 数据加载器字典
        dataset_sizes: 数据集大小字典
    """
    from data_loader import load_dataset_with_metadata, get_data_transforms, GearDamageDataset
    from torch.utils.data import DataLoader

    # 加载原始数据
    original_data = load_dataset_with_metadata(original_data_path)
    print(f"原始数据样本数: {len(original_data)}")

    # 加载额外数据（使用专门的函数）
    extra_data = load_extra_dataset_with_metadata(extra_data_path)
    print(f"额外数据样本数: {len(extra_data)}")

    # 合并数据
    all_data = original_data + extra_data
    print(f"合并后总样本数: {len(all_data)}")

    # 按照照片类型和数据集划分过滤数据
    train_data = [item for item in all_data if item['photo_type'] == photo_type and item['split'] == 'train']
    test_data = [item for item in all_data if item['photo_type'] == photo_type and item['split'] == 'test']
    val_data = [item for item in all_data if item['photo_type'] == photo_type and item['split'] == 'val']
    if not val_data:
        raise ValueError('Provide a reviewed val split before fine-tuning; test cannot select checkpoints')

    print(f"{photo_type} 训练集样本数: {len(train_data)}")
    print(f"{photo_type} 测试集样本数: {len(test_data)}")

    # 打印数据分布
    train_labels = [item['label_name'] for item in train_data]
    test_labels = [item['label_name'] for item in test_data]

    from collections import Counter
    print(f"训练集标签分布: {Counter(train_labels)}")
    print(f"测试集标签分布: {Counter(test_labels)}")

    # 获取数据变换
    data_transforms = get_data_transforms()

    # 创建数据集
    train_dataset = GearDamageDataset(train_data, transform=data_transforms['train'])
    test_dataset = GearDamageDataset(test_data, transform=data_transforms['test'])
    val_dataset = GearDamageDataset(val_data, transform=data_transforms['test'])

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }

    return dataloaders, dataset_sizes

def fine_tune_model(pretrained_model_path, original_data_path, extra_data_path,
                   photo_type, output_dir, num_epochs=20, batch_size=16, learning_rate=1e-5):
    """
    对预训练模型进行fine-tuning

    Args:
        pretrained_model_path: 预训练模型路径
        original_data_path: 原始数据路径
        extra_data_path: 额外数据路径
        photo_type: 照片类型 ('hongguan' 或 'tishijing')
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率（通常比从头训练要小）

    Returns:
        dict: 训练结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"开始fine-tuning {photo_type} 模型...")

    # 加载预训练模型
    model = load_pretrained_model(pretrained_model_path)
    model = model.to(device)

    # 创建结合原始数据和额外数据的数据加载器
    dataloaders, dataset_sizes = create_combined_data_loaders(
        original_data_path, extra_data_path, photo_type, batch_size
    )
    print(f"数据集大小: {dataset_sizes}")

    # 定义损失函数和优化器（使用较小的学习率进行fine-tuning）
    criterion = nn.CrossEntropyLoss()

    # 可以选择只训练分类器层或者训练整个网络
    # 这里我们训练整个网络，但使用较小的学习率
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 学习率调度器（更温和的衰减）
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 创建日志记录器
    log_dir = os.path.join(output_dir, 'fine_tune_logs')
    logger = TrainingLogger(log_dir)

    # Fine-tuning模型
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer,
                       exp_lr_scheduler, num_epochs, device, f'{photo_type}_fine_tuned', logger)

    # 保存fine-tuned模型
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'model_{photo_type}_updated.pth')
    save_model(model, model_path)

    # 评估模型
    class_names = ['boluo', 'cashang', 'dianshi', 'yashang']
    results = evaluate_model(model, dataloaders['test'], device, class_names)

    print(f"\n{photo_type} fine-tuned模型评估结果:")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1_score']:.4f}")

    # 保存混淆矩阵
    results_dir = os.path.join(output_dir, 'fine_tune_results')
    os.makedirs(results_dir, exist_ok=True)
    cm_path = os.path.join(results_dir, f'{photo_type}_updated_confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], class_names,
                         f'{photo_type} Fine-tuned模型混淆矩阵', cm_path)

    # 保存训练曲线和日志
    logger.save_plots(f'{photo_type}_fine_tuned')
    logger.save_log(f'{photo_type}_fine_tuned')

    # 保存评估结果
    results_to_save = {
        'photo_type': photo_type,
        'model_type': 'fine_tuned',
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'model_path': model_path,
        'training_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }

    results_path = os.path.join(results_dir, f'{photo_type}_updated_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)

    return results_to_save

def main():
    """
    主函数：对宏观照片和体视镜照片模型进行fine-tuning
    """
    # 设置路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_data_path = str(Path(project_root).parent / 'data')
    extra_data_path = str(Path(project_root).parent / 'extra_data')
    output_dir = os.path.join(project_root, 'output')

    # 预训练模型路径
    pretrained_models = {
        'hongguan': os.path.join(output_dir, 'models', 'model_hongguan.pth'),
        'tishijing': os.path.join(output_dir, 'models', 'model_tishijing.pth')
    }

    # 检查预训练模型是否存在
    for photo_type, model_path in pretrained_models.items():
        if not os.path.exists(model_path):
            print(f"警告: 预训练模型不存在: {model_path}")
            return

    # Fine-tuning参数
    fine_tune_params = {
        'num_epochs': 20,  # 较少的epoch数
        'batch_size': 16,
        'learning_rate': 1e-5  # 较小的学习率
    }

    results = {}

    # Fine-tune宏观照片模型
    print("=" * 50)
    print("开始fine-tuning宏观照片模型")
    print("=" * 50)

    hongguan_results = fine_tune_model(
        pretrained_models['hongguan'],
        original_data_path,
        extra_data_path,
        'hongguan',
        output_dir,
        **fine_tune_params
    )
    results['hongguan'] = hongguan_results

    # Fine-tune体视镜照片模型
    print("\n" + "=" * 50)
    print("开始fine-tuning体视镜照片模型")
    print("=" * 50)

    tishijing_results = fine_tune_model(
        pretrained_models['tishijing'],
        original_data_path,
        extra_data_path,
        'tishijing',
        output_dir,
        **fine_tune_params
    )
    results['tishijing'] = tishijing_results

    # 保存总体结果
    overall_results_path = os.path.join(output_dir, 'fine_tune_overall_results.json')
    with open(overall_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("Fine-tuning完成！")
    print("=" * 50)
    print(f"宏观照片模型准确率: {hongguan_results['accuracy']:.4f}")
    print(f"体视镜照片模型准确率: {tishijing_results['accuracy']:.4f}")
    print(f"结果保存在: {output_dir}")

    return results

if __name__ == "__main__":
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    results = main()
