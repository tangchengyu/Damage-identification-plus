import os
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle

from model import create_lifetime_model
from data_loader import extract_features_from_shuxing

class LifetimePredictor:
    """
    齿轮寿命预测器
    用于加载训练好的模型并进行推理
    """

    def __init__(self, model_path, device=None):
        """
        初始化预测器

        Args:
            model_path: 训练好的模型路径
            device: 计算设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型和标准化器
        self.model, self.config, self.scalers = self._load_model(model_path)

        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"模型已加载到设备: {self.device}")
        if self.scalers:
            print(f"标准化器已加载: 特征标准化器, 寿命标准化器")
        else:
            print("警告: 未找到标准化器，将使用原始特征值")

    def _load_model(self, model_path):
        """
        加载训练好的模型和标准化器

        Args:
            model_path: 模型路径

        Returns:
            tuple: (模型, 配置, 标准化器)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载检查点
        # Historical local checkpoints include sklearn scalers as well as tensors.
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 获取配置
        config = checkpoint['config']

        # 创建模型
        # The complete checkpoint replaces every backbone tensor; no download is needed.
        model = create_lifetime_model(config, pretrained_backbone=False)

        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # 加载标准化器
        scalers = None
        if 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
            print(f"成功加载标准化器")
        else:
            print("警告: 检查点中未找到标准化器")

        print(f"成功加载模型: {model_path}")
        if 'dataset_sizes' in checkpoint:
            print(f"训练数据集大小: {checkpoint['dataset_sizes']}")

        return model, config, scalers

    def predict_single(self, image_path, gear_attributes):
        """
        预测单个齿轮的寿命

        Args:
            image_path: 齿轮图像路径
            gear_attributes: 齿轮属性字典（shuxing格式）

        Returns:
            预测的寿命值（原始尺度）
        """
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        # 提取特征
        size_features, operation_features, material_features = extract_features_from_shuxing(gear_attributes)

        # 合并所有特征
        all_features = np.concatenate([size_features, operation_features, material_features])

        # 应用特征标准化（如果有标准化器）
        if self.scalers and 'feature_scaler' in self.scalers:
            all_features = self.scalers['feature_scaler'].transform([all_features])[0]

        # 重新分割特征
        size_dim = len(size_features)
        operation_dim = len(operation_features)
        material_dim = len(material_features)

        size_features_scaled = all_features[:size_dim]
        operation_features_scaled = all_features[size_dim:size_dim+operation_dim]
        material_features_scaled = all_features[size_dim+operation_dim:]

        # 转换为张量
        size_tensor = torch.tensor(size_features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        operation_tensor = torch.tensor(operation_features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        material_tensor = torch.tensor(material_features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            prediction = self.model(image_tensor, size_tensor, operation_tensor, material_tensor)
            lifetime_pred = prediction.cpu().numpy()[0, 0]

        # 应用寿命逆变换（如果有标准化器）
        if self.scalers and 'lifetime_scaler' in self.scalers:
            lifetime_pred = self.scalers['lifetime_scaler'].inverse_transform([[lifetime_pred]])[0, 0]

        return lifetime_pred

    def predict_batch(self, data_list):
        """
        批量预测齿轮寿命

        Args:
            data_list: 数据列表，每个元素包含 {'image_path': str, 'attributes': dict}

        Returns:
            预测结果列表
        """
        predictions = []

        for data in data_list:
            try:
                pred = self.predict_single(data['image_path'], data['attributes'])
                predictions.append({
                    'image_path': data['image_path'],
                    'predicted_lifetime': pred,
                    'status': 'success'
                })
            except Exception as e:
                predictions.append({
                    'image_path': data['image_path'],
                    'predicted_lifetime': None,
                    'status': 'error',
                    'error_message': str(e)
                })

        return predictions

    def predict_from_dataset(self, data_path, sample_ids=None, photo_type='hongguan', exclude_outliers=None):
        """
        从数据集中预测指定样本的寿命

        Args:
            data_path: 数据集路径
            sample_ids: 样本ID列表，如果为None则预测所有样本
            photo_type: 照片类型
            exclude_outliers: 要排除的异常值样本ID列表

        Returns:
            预测结果字典
        """
        if exclude_outliers is None:
            exclude_outliers = []
        # 加载数据集
        json_path = os.path.join(data_path, 'DATA.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        results = []
        damage_types = ['boluo', 'cashang', 'dianshi', 'yashang']

        for damage_type in damage_types:
            if damage_type in metadata:
                for sample_id, sample_data in metadata[damage_type].items():
                    # 如果指定了样本ID，只处理指定的样本
                    if sample_ids is not None and sample_id not in sample_ids:
                        continue

                    # 跳过异常值样本
                    if sample_id in exclude_outliers:
                        print(f"跳过异常值样本: {sample_id}")
                        continue

                    # 构建图片路径，尝试在train和test子目录中查找
                    # 从路径中提取实际的文件名，例如从"data_boluo_hongguan_train_4052-1"提取"4052-1"
                    path_parts = sample_data['path'].split('_')
                    if len(path_parts) < 5:
                        raise ValueError(f'Malformed image reference: {sample_id}')
                    if path_parts[2] != photo_type:
                        continue
                    if len(path_parts) >= 4:
                        actual_filename = path_parts[-1]  # 取最后一部分作为文件名
                    else:
                        actual_filename = sample_data['path']  # 如果格式不符合预期，使用原始路径

                    image_filename = f"{actual_filename}.jpg"

                    # 尝试在不同的子目录中查找图片
                    possible_paths = [
                        os.path.join(data_path, damage_type, photo_type, path_parts[3], image_filename)
                    ]

                    image_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            image_path = path
                            break

                    if image_path is None:
                        results.append({'sample_id': sample_id, 'status': 'error',
                                        'error_message': 'Recorded image reference is missing; no substitute used'})
                    else:
                        try:
                            # 预测寿命
                            pred_lifetime = self.predict_single(image_path, sample_data['shuxing'])
                            true_lifetime = sample_data['shouming']

                            # 计算误差
                            absolute_error = abs(pred_lifetime - true_lifetime)
                            relative_error = (absolute_error / true_lifetime) * 100

                            results.append({
                                'sample_id': sample_id,
                                'damage_type': damage_type,
                                'image_path': image_path,
                                'true_lifetime': true_lifetime,
                                'predicted_lifetime': pred_lifetime,
                                'absolute_error': absolute_error,
                                'relative_error': relative_error,
                                'status': 'success'
                            })
                        except Exception as e:
                            results.append({
                                'sample_id': sample_id,
                                'damage_type': damage_type,
                                'image_path': image_path,
                                'status': 'error',
                                'error_message': str(e)
                            })

        return results

    def analyze_predictions(self, results):
        """
        分析预测结果

        Args:
            results: 预测结果列表

        Returns:
            分析统计信息
        """
        successful_results = [r for r in results if r['status'] == 'success']

        if not successful_results:
            return {'error': '没有成功的预测结果'}

        true_values = [r['true_lifetime'] for r in successful_results]
        pred_values = [r['predicted_lifetime'] for r in successful_results]
        absolute_errors = [r['absolute_error'] for r in successful_results]
        relative_errors = [r['relative_error'] for r in successful_results]

        # 计算统计指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(true_values, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, pred_values)
        r2 = r2_score(true_values, pred_values)
        mape = np.mean(relative_errors)

        # 按损伤类型分析
        damage_type_stats = {}
        for damage_type in ['boluo', 'cashang', 'dianshi', 'yashang']:
            type_results = [r for r in successful_results if r['damage_type'] == damage_type]
            if type_results:
                type_true = [r['true_lifetime'] for r in type_results]
                type_pred = [r['predicted_lifetime'] for r in type_results]
                type_mae = mean_absolute_error(type_true, type_pred)
                type_mape = np.mean([r['relative_error'] for r in type_results])

                damage_type_stats[damage_type] = {
                    'count': len(type_results),
                    'mae': type_mae,
                    'mape': type_mape
                }

        analysis = {
            'total_samples': len(results),
            'successful_predictions': len(successful_results),
            'failed_predictions': len(results) - len(successful_results),
            'overall_metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            },
            'error_statistics': {
                'mean_absolute_error': np.mean(absolute_errors),
                'std_absolute_error': np.std(absolute_errors),
                'mean_relative_error': np.mean(relative_errors),
                'std_relative_error': np.std(relative_errors),
                'max_absolute_error': np.max(absolute_errors),
                'max_relative_error': np.max(relative_errors)
            },
            'damage_type_stats': damage_type_stats
        }

        return analysis

    def visualize_predictions(self, results, output_dir):
        """
        可视化预测结果

        Args:
            results: 预测结果列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        successful_results = [r for r in results if r['status'] == 'success']

        if not successful_results:
            print("没有成功的预测结果可以可视化")
            return

        true_values = [r['true_lifetime'] for r in successful_results]
        pred_values = [r['predicted_lifetime'] for r in successful_results]
        damage_types = [r['damage_type'] for r in successful_results]

        # 预测vs真实值散点图
        plt.figure(figsize=(12, 5))

        # 整体散点图
        plt.subplot(1, 2, 1)
        plt.scatter(true_values, pred_values, alpha=0.6, s=50)

        # 理想线
        min_val = min(min(true_values), min(pred_values))
        max_val = max(max(true_values), max(pred_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')

        plt.xlabel('真实寿命值')
        plt.ylabel('预测寿命值')
        plt.title('预测值 vs 真实值')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 添加R²信息
        from sklearn.metrics import r2_score
        r2 = r2_score(true_values, pred_values)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 按损伤类型分组的散点图
        plt.subplot(1, 2, 2)
        colors = {'boluo': 'red', 'cashang': 'blue', 'dianshi': 'green', 'yashang': 'orange'}

        for damage_type in ['boluo', 'cashang', 'dianshi', 'yashang']:
            type_indices = [i for i, dt in enumerate(damage_types) if dt == damage_type]
            if type_indices:
                type_true = [true_values[i] for i in type_indices]
                type_pred = [pred_values[i] for i in type_indices]
                plt.scatter(type_true, type_pred, c=colors[damage_type],
                           label=damage_type, alpha=0.6, s=50)

        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.5)
        plt.xlabel('真实寿命值')
        plt.ylabel('预测寿命值')
        plt.title('按损伤类型分组的预测结果')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 误差分布直方图
        plt.figure(figsize=(12, 5))

        relative_errors = [r['relative_error'] for r in successful_results]

        plt.subplot(1, 2, 1)
        plt.hist(relative_errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('相对误差 (%)')
        plt.ylabel('频次')
        plt.title('相对误差分布')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        absolute_errors = [r['absolute_error'] for r in successful_results]
        plt.hist(absolute_errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('绝对误差')
        plt.ylabel('频次')
        plt.title('绝对误差分布')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"可视化结果已保存到: {output_dir}")

def main():
    """
    主函数 - 演示预测器的使用
    """
    # 设置路径
    model_path = str(Path(__file__).resolve().parents[1] / 'output/models/best_lifetime_model.pth')
    data_path = str(Path(__file__).resolve().parents[2] / 'data')
    output_dir = str(Path(__file__).resolve().parents[1] / 'output/predictions')

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型")
        return

    # 创建预测器
    predictor = LifetimePredictor(model_path)

    # 预测数据集中的所有样本（可以指定部分样本）
    print("开始预测...")
    results = predictor.predict_from_dataset(data_path, sample_ids=None, photo_type='hongguan')

    print(f"预测完成，共处理 {len(results)} 个样本")

    # 分析预测结果
    analysis = predictor.analyze_predictions(results)

    print("\n=== 预测结果分析 ===")
    print(f"总样本数: {analysis['total_samples']}")
    print(f"成功预测: {analysis['successful_predictions']}")
    print(f"预测失败: {analysis['failed_predictions']}")

    if 'overall_metrics' in analysis:
        metrics = analysis['overall_metrics']
        print(f"\n整体性能指标:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")

        print(f"\n各损伤类型性能:")
        for damage_type, stats in analysis['damage_type_stats'].items():
            print(f"  {damage_type}: 样本数={stats['count']}, MAE={stats['mae']:.2f}, MAPE={stats['mape']:.2f}%")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细预测结果
    results_path = os.path.join(output_dir, 'prediction_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存分析结果
    analysis_path = os.path.join(output_dir, 'prediction_analysis.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # 可视化结果
    predictor.visualize_predictions(results, output_dir)

    print(f"\n结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
