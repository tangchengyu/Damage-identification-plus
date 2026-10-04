import os
import hashlib
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import pickle

class LifetimePredictionDataset(Dataset):
    """齿轮寿命预测数据集类"""

    def __init__(self, dataset_items, image_transform=None, feature_scaler=None,
                 lifetime_scaler=None, outlier_detector=None, is_train=True):
        """
        初始化数据集

        Args:
            dataset_items: 数据集项目列表
            image_transform: 图像变换
            feature_scaler: 特征标准化器
            lifetime_scaler: 寿命标准化器
            outlier_detector: 异常值检测器
            is_train: 是否为训练集
        """
        self.dataset_items = dataset_items
        self.image_transform = image_transform
        self.feature_scaler = feature_scaler if feature_scaler else StandardScaler()
        self.lifetime_scaler = lifetime_scaler if lifetime_scaler else StandardScaler()
        self.outlier_detector = outlier_detector if outlier_detector else IsolationForest(
            contamination=0.1, random_state=42
        )
        self.is_train = is_train

        # 如果是训练集，进行数据预处理
        if is_train:
            self._preprocess_training_data()
        else:
            self._preprocess_test_data()

    def _preprocess_training_data(self):
        """预处理训练数据：异常值检测和特征标准化"""
        print(f"开始预处理训练数据，共{len(self.dataset_items)}个样本")

        # 提取所有特征和寿命
        all_features = []
        lifetimes = []

        for item in self.dataset_items:
            # 安全地转换特征为数值类型
            def safe_convert_to_float(feature_list):
                converted = []
                for x in feature_list:
                    if isinstance(x, (int, float)):
                        converted.append(float(x))
                    elif isinstance(x, list):
                        # 如果是列表，取平均值
                        numeric_values = [float(v) for v in x if isinstance(v, (int, float))]
                        converted.append(np.mean(numeric_values) if numeric_values else 0.0)
                    elif isinstance(x, str):
                        try:
                            converted.append(float(x))
                        except ValueError:
                            converted.append(0.0)
                    else:
                        converted.append(0.0)
                return converted

            size_features = safe_convert_to_float(item['size_features'])
            operation_features = safe_convert_to_float(item['operation_features'])
            material_features = safe_convert_to_float(item['material_features'])

            combined_features = size_features + operation_features + material_features
            all_features.append(combined_features)
            lifetimes.append(item['lifetime'])

        # 检查特征维度一致性
        feature_lengths = [len(features) for features in all_features]
        if len(set(feature_lengths)) > 1:
            print(f"警告：特征维度不一致: {set(feature_lengths)}")
            # 找到最常见的特征长度
            from collections import Counter
            most_common_length = Counter(feature_lengths).most_common(1)[0][0]
            print(f"使用最常见的特征长度: {most_common_length}")

            # 过滤掉维度不匹配的样本
            filtered_features = []
            filtered_lifetimes = []
            filtered_items = []

            for i, (features, lifetime, item) in enumerate(zip(all_features, lifetimes, self.dataset_items)):
                if len(features) == most_common_length:
                    filtered_features.append(features)
                    filtered_lifetimes.append(lifetime)
                    filtered_items.append(item)
                else:
                    print(f"跳过样本 {i}，特征维度为 {len(features)}")

            all_features = filtered_features
            lifetimes = filtered_lifetimes
            self.dataset_items = filtered_items
            print(f"过滤后剩余 {len(all_features)} 个样本")

        all_features = np.array(all_features)
        lifetimes = np.array(lifetimes).reshape(-1, 1)

        # 异常值检测
        outlier_mask = self.outlier_detector.fit_predict(all_features) == 1

        # 基于寿命进行异常值检测（使用IQR方法）
        Q1 = np.percentile(lifetimes, 25)
        Q3 = np.percentile(lifetimes, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        lifetime_outlier_mask = (lifetimes.flatten() >= lower_bound) & (lifetimes.flatten() <= upper_bound)

        # 综合异常值掩码
        valid_mask = outlier_mask & lifetime_outlier_mask

        print(f"原始数据: {len(self.dataset_items)} 样本")
        print(f"移除异常值后: {np.sum(valid_mask)} 样本")
        print(f"移除了 {len(self.dataset_items) - np.sum(valid_mask)} 个异常值")

        # 过滤异常值
        self.dataset_items = [self.dataset_items[i] for i in range(len(self.dataset_items)) if valid_mask[i]]
        filtered_features = all_features[valid_mask]
        filtered_lifetimes = lifetimes[valid_mask]

        # 特征标准化
        self.feature_scaler.fit(filtered_features)

        # 寿命标准化
        self.lifetime_scaler.fit(filtered_lifetimes)

    def _preprocess_test_data(self):
        """预处理测试数据：使用训练时的标准化器"""
        # 测试时不需要额外处理，直接使用传入的标准化器
        pass

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        item = self.dataset_items[idx]

        # 加载图片
        image = Image.open(item['image_path']).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        # 确保特征都是数字列表
        def ensure_numeric_list(features):
            numeric_features = []
            for f in features:
                if isinstance(f, list):
                    # 如果是列表，取平均值
                    numeric_values = [float(v) for v in f if isinstance(v, (int, float, str)) and str(v).replace('.', '').replace('-', '').isdigit()]
                    numeric_features.append(np.mean(numeric_values) if numeric_values else 0.0)
                else:
                    numeric_features.append(float(f) if f is not None else 0.0)
            return numeric_features

        # 获取特征
        size_features = ensure_numeric_list(item['size_features'])
        operation_features = ensure_numeric_list(item['operation_features'])
        material_features = ensure_numeric_list(item['material_features'])

        # 合并所有特征进行标准化
        combined_features = size_features + operation_features + material_features
        combined_features = np.array(combined_features).reshape(1, -1)

        # 应用特征标准化
        if self.feature_scaler:
            combined_features = self.feature_scaler.transform(combined_features)[0]

        # 分离标准化后的特征
        size_len = len(size_features)
        operation_len = len(operation_features)
        material_len = len(material_features)

        size_features = torch.tensor(combined_features[:size_len], dtype=torch.float32)
        operation_features = torch.tensor(combined_features[size_len:size_len+operation_len], dtype=torch.float32)
        material_features = torch.tensor(combined_features[size_len+operation_len:], dtype=torch.float32)

        # 获取寿命标签并标准化
        lifetime = item['lifetime']
        if self.lifetime_scaler:
            lifetime = self.lifetime_scaler.transform([[lifetime]])[0][0]
        lifetime = torch.tensor(lifetime, dtype=torch.float32)

        return {
            'image': image,
            'size_features': size_features,
            'operation_features': operation_features,
            'material_features': material_features,
            'lifetime': lifetime,
            'sample_id': item['sample_id']
        }

def extract_features_from_shuxing(shuxing):
    """
    从shuxing字段提取三类特征

    Args:
        shuxing: 齿轮属性字典

    Returns:
        tuple: (尺寸特征, 工况特征, 材料特征)
    """
    # 尺寸特征 (feature set 1)
    size_features = [
        shuxing.get('moshu', 0),  # 模数
        shuxing.get('chishu', 0),  # 齿数
        shuxing.get('bianweixishu', 0),  # 变位系数
        shuxing.get('luoxuanjiao', 0),  # 螺旋角
        shuxing.get('yalijiao', 0),  # 压力角
        shuxing.get('chidingyuan(mm)', 0),  # 齿顶圆直径
        shuxing.get('chikuan(mm)', 0)  # 齿宽
    ]

    # 工况特征 (feature set 2)
    operation_features = [
        shuxing.get('edingniujiu(N*m)', 0),  # 额定扭矩
        shuxing.get('edingzhuansu(r/min)', 0)  # 额定转速
    ]

    # 材料特征 (feature set 3)
    # 处理可能的列表值和字符串值
    def safe_extract_value(value, default=0):
        if isinstance(value, list):
            # 确保列表中的元素都是数字
            numeric_values = []
            for v in value:
                if isinstance(v, (int, float)):
                    numeric_values.append(v)
                elif isinstance(v, str):
                    try:
                        numeric_values.append(float(v))
                    except:
                        pass
            return np.mean(numeric_values) if numeric_values else default
        elif isinstance(value, str):
            # 处理类似"(205-210)*1000"的字符串
            if '*' in value:
                parts = value.split('*')
                if len(parts) == 2:
                    try:
                        base_str = parts[0].strip()
                        multiplier = float(parts[1])
                        # 处理括号内的范围，如(205-210)
                        if base_str.startswith('(') and base_str.endswith(')'):
                            range_str = base_str[1:-1]  # 去掉括号
                            if '-' in range_str:
                                range_parts = range_str.split('-')
                                if len(range_parts) == 2:
                                    min_val = float(range_parts[0])
                                    max_val = float(range_parts[1])
                                    base = (min_val + max_val) / 2  # 取平均值
                                else:
                                    base = float(range_str)
                            else:
                                base = float(range_str)
                        else:
                            base = float(base_str)
                        return base * multiplier
                    except:
                        return default
            # 处理单纯的范围字符串，如"950-1150"
            elif '-' in value and value.replace('-', '').replace('.', '').isdigit():
                try:
                    parts = value.split('-')
                    if len(parts) == 2:
                        min_val = float(parts[0])
                        max_val = float(parts[1])
                        return (min_val + max_val) / 2  # 取平均值
                except:
                    pass
            try:
                return float(value)
            except:
                return default
        else:
            return float(value) if value is not None else default

    material_features = [
        1 if shuxing.get('cailiao') == '12Cr2Ni4AGJB1951-1994' else 0,  # 材料类型编码
        safe_extract_value(shuxing.get('midu(g/cm3)')),  # 密度
        safe_extract_value(shuxing.get('tanxingmoliang(MPa)')),  # 弹性模量
        safe_extract_value(shuxing.get('bosongbi')),  # 泊松比
        safe_extract_value(shuxing.get('qufuyingli(MPa)')),  # 屈服应力
        safe_extract_value(shuxing.get('qianghuaxishu(MPa)')),  # 强化系数
        safe_extract_value(shuxing.get('yinghuazhishu')),  # 硬化指数
        safe_extract_value(shuxing.get('mocaxishu'))  # 摩擦系数
    ]

    return size_features, operation_features, material_features

def load_lifetime_dataset(data_path, photo_type='hongguan', exclude_outliers=None):
    """
    加载寿命预测数据集

    Args:
        data_path: 数据根目录路径
        photo_type: 照片类型 ('hongguan' 或 'tishijing')
        exclude_outliers: 要排除的异常值样本ID列表

    Returns:
        list: 包含所有样本信息的列表
    """
    if exclude_outliers is None:
        exclude_outliers = []
    # 加载 DATA.json 文件
    json_path = os.path.join(data_path, 'DATA.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    dataset = []
    damage_types = ['boluo', 'cashang', 'dianshi', 'yashang']

    # 遍历所有损伤类型
    for damage_type in damage_types:
        if damage_type in metadata:
            for sample_id, sample_data in metadata[damage_type].items():
                # 解析路径信息
                path_parts = sample_data['path'].split('_')
                if len(path_parts) < 5:
                    raise ValueError(f'Malformed lifetime image reference: {sample_id}')
                if path_parts[2] != photo_type:
                    continue
                # 路径格式: data_损伤类型_照片类型_训练集类型_图片名
                if len(path_parts) >= 5:
                    train_test_type = path_parts[3]  # train 或 test
                    image_name = path_parts[4]  # 图片名
                    image_filename = f"{image_name}.jpg"
                    image_path = os.path.join(data_path, damage_type, photo_type, train_test_type, image_filename)
                else:
                    # 备用方案：直接使用原路径
                    image_filename = f"{sample_data['path']}.jpg"
                    image_path = os.path.join(data_path, damage_type, photo_type, image_filename)

                # 检查图片是否存在且不在排除列表中
                if os.path.exists(image_path) and sample_id not in exclude_outliers:
                    # 提取特征
                    size_features, operation_features, material_features = extract_features_from_shuxing(
                        sample_data['shuxing']
                    )

                    # 获取寿命标签
                    lifetime = sample_data['shouming']

                    dataset.append({
                        'sample_id': sample_id,
                        'image_path': image_path,
                        'damage_type': damage_type,
                        'size_features': size_features,
                        'operation_features': operation_features,
                        'material_features': material_features,
                        'lifetime': lifetime,
                        'specimen_id': sample_data.get('specimen_id'),
                        'lifetime_verified': sample_data.get('lifetime_verified') is True,
                        'lifetime_source': sample_data.get('lifetime_source')
                    })
                elif sample_id in exclude_outliers:
                    print(f"排除异常值样本: {sample_id}")

    return dataset

def get_image_transforms():
    """
    获取图像变换

    Returns:
        dict: 包含训练和验证变换的字典
    """
    # 训练时的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证和测试时的变换
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }

def create_lifetime_dataloaders(data_path, photo_type='hongguan', batch_size=16,
                               train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, save_scalers=True, exclude_outliers=None):
    """
    创建寿命预测数据加载器

    Args:
        data_path: 数据根目录路径
        photo_type: 照片类型
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        save_scalers: 是否保存标准化器
        exclude_outliers: 要排除的异常值样本ID列表

    Returns:
        tuple: (数据加载器字典, 数据集大小字典, 标准化器字典)
    """
    # 加载数据集
    all_data = load_lifetime_dataset(data_path, photo_type, exclude_outliers)
    print(f"总样本数: {len(all_data)}")
    if not all_data:
        raise ValueError('No verified lifetime training data')
    if any(not x['lifetime_verified'] or not x['lifetime_source'] or not x['specimen_id'] for x in all_data):
        raise ValueError('Lifetime training requires verified labels, lifetime_source and specimen_id. '
                         'Historical labels are unverified; do not fabricate provenance or groups.')
    hashes = [hashlib.sha256(Path(x['image_path']).read_bytes()).hexdigest() for x in all_data]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate lifetime images must be reconciled before training')
    ratios = (train_ratio, val_ratio, test_ratio)
    if not all(np.isfinite(r) and r > 0 for r in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError('train/val/test ratios must be positive and sum to 1')

    # 划分数据集
    groups = [x['specimen_id'] for x in all_data]
    train_idx, temp_idx = next(GroupShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=42)
                               .split(all_data, groups=groups))
    train_data = [all_data[i] for i in train_idx]
    temp_data = [all_data[i] for i in temp_idx]
    val_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
                            .split(temp_data, groups=[x['specimen_id'] for x in temp_data]))
    val_data = [temp_data[i] for i in val_idx]
    test_data = [temp_data[i] for i in test_idx]

    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

    # 获取图像变换
    transforms_dict = get_image_transforms()

    # 创建训练数据集（会进行数据预处理）
    train_dataset = LifetimePredictionDataset(train_data, transforms_dict['train'], is_train=True)

    # 创建验证和测试数据集（使用训练集的标准化器）
    val_dataset = LifetimePredictionDataset(
        val_data, transforms_dict['val'],
        feature_scaler=train_dataset.feature_scaler,
        lifetime_scaler=train_dataset.lifetime_scaler,
        outlier_detector=train_dataset.outlier_detector,
        is_train=False
    )

    test_dataset = LifetimePredictionDataset(
        test_data, transforms_dict['test'],
        feature_scaler=train_dataset.feature_scaler,
        lifetime_scaler=train_dataset.lifetime_scaler,
        outlier_detector=train_dataset.outlier_detector,
        is_train=False
    )

    # 创建数据集字典
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    # 保存标准化器
    scalers = {
        'feature_scaler': train_dataset.feature_scaler,
        'lifetime_scaler': train_dataset.lifetime_scaler,
        'outlier_detector': train_dataset.outlier_detector
    }

    if save_scalers:
        scalers_path = os.path.join(data_path, 'scalers.pkl')
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"标准化器已保存到: {scalers_path}")

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    # 数据集大小
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }

    return dataloaders, dataset_sizes, scalers

if __name__ == "__main__":
    # 测试数据加载器
    data_root = str(Path(__file__).resolve().parents[2] / 'data')

    print("加载寿命预测数据集...")
    dataloaders, sizes, all_data = create_lifetime_dataloaders(data_root, 'hongguan', batch_size=4)

    print(f"\n数据集大小: {sizes}")

    # 测试一个批次
    for phase in ['train', 'val', 'test']:
        print(f"\n{phase}阶段第一个批次:")
        for batch in dataloaders[phase]:
            print(f"图像形状: {batch['image'].shape}")
            print(f"尺寸特征形状: {batch['size_features'].shape}")
            print(f"工况特征形状: {batch['operation_features'].shape}")
            print(f"材料特征形状: {batch['material_features'].shape}")
            print(f"寿命标签形状: {batch['lifetime'].shape}")
            print(f"寿命标签范围: {batch['lifetime'].min().item():.0f} - {batch['lifetime'].max().item():.0f}")
            break
