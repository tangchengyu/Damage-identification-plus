import os
from pathlib import Path
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class GearDamageDataset(Dataset):
    """齿轮损伤数据集类"""

    def __init__(self, dataset_items, transform=None):
        self.dataset_items = dataset_items
        self.transform = transform

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        item = self.dataset_items[idx]

        # 加载图片
        image = Image.open(item['image_path']).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 返回图片和标签
        return image, item['label_id']

def load_dataset_with_metadata(data_path):
    """
    加载数据集和元信息

    Args:
        data_path: 数据根目录路径

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
                # 解析路径信息
                path_parts = sample_info['path'].split('_')
                photo_type = path_parts[2]  # hongguan 或 tishijing
                split = path_parts[3]       # train 或 test
                img_filename_base = path_parts[4]  # 基础文件名

                # 根据照片类型调整文件名格式
                if photo_type == 'tishijing':
                    # 体视镜照片使用单横线格式
                    img_filename_single = img_filename_base + '.jpg'
                    img_path_single = os.path.join(data_path, label, photo_type, split, img_filename_single)

                    if os.path.exists(img_path_single):
                        img_filename = img_filename_single
                        img_path = img_path_single
                    else:
                        # 如果文件不存在，仍使用单横线格式
                        img_filename = img_filename_single
                        img_path = img_path_single
                else:
                    # 宏观照片使用单横线格式
                    img_filename = img_filename_base + '.jpg'
                    img_path = os.path.join(data_path, label, photo_type, split, img_filename)

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
                    print(f'Warning: unresolved metadata image {sample_id}: {img_path}')

    return dataset

def get_data_transforms():
    """
    获取数据预处理变换

    Returns:
        dict: 包含训练和测试变换的字典
    """
    data_transforms = {
        # 训练集使用数据增强
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # 测试集不做数据增强，只做必要的尺寸调整和归一化
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def create_data_loaders(data_path, photo_type='hongguan', batch_size=32):
    """
    创建数据加载器

    Args:
        data_path: 数据根目录路径
        photo_type: 照片类型 ('hongguan' 或 'tishijing')
        batch_size: 批次大小

    Returns:
        dict: 包含训练和测试数据加载器的字典
    """
    # 加载数据集
    all_data = load_dataset_with_metadata(data_path)

    # 根据照片类型和数据集划分过滤数据
    train_data = [item for item in all_data if item['photo_type'] == photo_type and item['split'] == 'train']
    test_data = [item for item in all_data if item['photo_type'] == photo_type and item['split'] == 'test']

    # 获取数据变换
    transforms_dict = get_data_transforms()

    # 创建数据集对象
    train_dataset = GearDamageDataset(train_data, transform=transforms_dict['train'])
    test_dataset = GearDamageDataset(test_data, transform=transforms_dict['test'])

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    # 输出数据集信息
    print(f"数据集信息 ({photo_type}):")
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")

    sizes = {'train': len(train_data), 'test': len(test_data)}
    val_data = [item for item in all_data if item['photo_type'] == photo_type and item['split'] == 'val']
    if val_data:
        val_dataset = GearDamageDataset(val_data, transform=transforms_dict['test'])
        dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        sizes['val'] = len(val_data)
    return dataloaders, sizes

if __name__ == "__main__":
    # 测试数据加载
    data_root = str(Path(__file__).resolve().parents[2] / 'data')

    # 测试元数据加载
    all_data = load_dataset_with_metadata(data_root)
    print(f"总样本数: {len(all_data)}")

    if all_data:
        print("\n第一个样本信息:")
        print(f"样本ID: {all_data[0]['sample_id']}")
        print(f"图片路径: {all_data[0]['image_path']}")
        print(f"损伤类型: {all_data[0]['label_name']}")
        print(f"照片类型: {all_data[0]['photo_type']}")
        print(f"数据集划分: {all_data[0]['split']}")

    # 测试数据加载器
    print("\n创建宏观照片数据加载器:")
    dataloaders_h, sizes_h = create_data_loaders(data_root, 'hongguan', batch_size=8)

    print("\n创建体视镜照片数据加载器:")
    dataloaders_t, sizes_t = create_data_loaders(data_root, 'tishijing', batch_size=8)
