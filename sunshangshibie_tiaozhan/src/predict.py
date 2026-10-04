import os
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
from torchvision import transforms

from model import load_model
from data_loader import get_data_transforms

class DamagePredictor:
    """齿轮损伤预测器"""

    def __init__(self, model_dir, device=None, model_variant='updated'):
        """
        初始化预测器

        Args:
            model_dir: 模型目录路径
            device: 计算设备
        """
        self.model_dir = model_dir
        if model_variant not in ('original', 'updated'):
            raise ValueError('model_variant must be original or updated')
        self.model_variant = model_variant
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['boluo', 'cashang', 'dianshi', 'yashang']
        self.class_names_zh = ['剥落', '擦伤', '点蚀', '压伤']

        # 加载模型
        self.model_hongguan = None
        self.model_tishijing = None
        self.load_models()

        # 获取数据变换
        self.transforms = get_data_transforms()['test']

    def load_models(self):
        """加载训练好的模型"""
        suffix = '_updated' if self.model_variant == 'updated' else ''
        hongguan_path = os.path.join(self.model_dir, f'model_hongguan{suffix}.pth')
        tishijing_path = os.path.join(self.model_dir, f'model_tishijing{suffix}.pth')
        self.checkpoints = {}
        missing = [path for path in (hongguan_path, tishijing_path) if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f'Missing {self.model_variant} checkpoints: {missing}. '
                                    'Install the selected pair or explicitly choose original weights.')

        if os.path.exists(hongguan_path):
            self.model_hongguan = load_model(hongguan_path, num_classes=4, model_name='resnet34')
            self.model_hongguan.to(self.device)
            self.model_hongguan.eval()
            self.checkpoints['hongguan'] = str(Path(hongguan_path).resolve())
            print(f"宏观照片模型已加载: {hongguan_path}")
        else:
            print(f"警告: 宏观照片模型未找到: {hongguan_path}")

        if os.path.exists(tishijing_path):
            self.model_tishijing = load_model(tishijing_path, num_classes=4, model_name='resnet34')
            self.model_tishijing.to(self.device)
            self.model_tishijing.eval()
            self.checkpoints['tishijing'] = str(Path(tishijing_path).resolve())
            print(f"体视镜照片模型已加载: {tishijing_path}")
        else:
            print(f"警告: 体视镜照片模型未找到: {tishijing_path}")

    def preprocess_image(self, image_path):
        """
        预处理图像

        Args:
            image_path: 图像路径

        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件未找到: {image_path}")

        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transforms(image).unsqueeze(0)  # 添加batch维度
        return image_tensor.to(self.device)

    def predict_single_model(self, image_tensor, model):
        """
        单个模型预测

        Args:
            image_tensor: 图像张量
            model: 模型

        Returns:
            torch.Tensor: 预测概率
        """
        if model is None:
            return torch.zeros(1, 4).to(self.device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        return probabilities

    def predict_damage_type(self, image_hongguan_path=None, image_tishijing_path=None,
                           weight_hongguan=0.3, weight_tishijing=0.7):
        """
        预测齿轮损伤类型（加权融合）

        Args:
            image_hongguan_path: 宏观照片路径
            image_tishijing_path: 体视镜照片路径
            weight_hongguan: 宏观照片权重
            weight_tishijing: 体视镜照片权重

        Returns:
            dict: 预测结果
        """
        if not all(math.isfinite(w) and w >= 0 for w in (weight_hongguan, weight_tishijing)):
            raise ValueError('Fusion weights must be finite and non-negative')
        for path, model in ((image_hongguan_path, self.model_hongguan),
                            (image_tishijing_path, self.model_tishijing)):
            if path:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f'Image file not found: {path}')
                if model is None:
                    raise RuntimeError(f'Model unavailable for supplied image: {path}')
        prob_hongguan = torch.zeros(1, 4).to(self.device)
        prob_tishijing = torch.zeros(1, 4).to(self.device)

        used_models = []

        # 宏观照片预测
        if image_hongguan_path:
            image_hongguan = self.preprocess_image(image_hongguan_path)
            prob_hongguan = self.predict_single_model(image_hongguan, self.model_hongguan)
            used_models.append('hongguan')
        else:
            weight_hongguan = 0

        # 体视镜照片预测
        if image_tishijing_path:
            image_tishijing = self.preprocess_image(image_tishijing_path)
            prob_tishijing = self.predict_single_model(image_tishijing, self.model_tishijing)
            used_models.append('tishijing')
        else:
            weight_tishijing = 0

        # 权重归一化
        total_weight = weight_hongguan + weight_tishijing
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("没有可用的模型进行预测")

        weight_hongguan /= total_weight
        weight_tishijing /= total_weight

        # 加权融合
        final_prob = prob_hongguan * weight_hongguan + prob_tishijing * weight_tishijing

        # 获取预测结果
        predicted_class_idx = torch.argmax(final_prob, dim=1).item()
        predicted_class_name = self.class_names[predicted_class_idx]
        predicted_class_name_zh = self.class_names_zh[predicted_class_idx]

        # 转换为numpy数组便于处理
        prob_hongguan_np = prob_hongguan.cpu().numpy()[0]
        prob_tishijing_np = prob_tishijing.cpu().numpy()[0]
        final_prob_np = final_prob.cpu().numpy()[0]

        results = {
            'model_variant': self.model_variant,
            'checkpoints': self.checkpoints.copy(),
            'predicted_class': predicted_class_name,
            'predicted_class_zh': predicted_class_name_zh,
            'predicted_class_idx': predicted_class_idx,
            'confidence': final_prob_np[predicted_class_idx],
            'probabilities': {
                self.class_names[i]: final_prob_np[i] for i in range(len(self.class_names))
            },
            'probabilities_zh': {
                self.class_names_zh[i]: final_prob_np[i] for i in range(len(self.class_names_zh))
            },
            'model_predictions': {
                'hongguan': {
                    'probabilities': prob_hongguan_np.tolist(),
                    'weight': weight_hongguan
                },
                'tishijing': {
                    'probabilities': prob_tishijing_np.tolist(),
                    'weight': weight_tishijing
                }
            },
            'used_models': used_models,
            'weights': {
                'hongguan': weight_hongguan,
                'tishijing': weight_tishijing
            }
        }

        return results

    def visualize_prediction(self, results, image_paths, save_path=None):
        """
        可视化预测结果

        Args:
            results: 预测结果
            image_paths: 图像路径字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 显示图像
        row_0_used = False
        if image_paths.get('hongguan') and os.path.exists(image_paths['hongguan']):
            img = Image.open(image_paths['hongguan'])
            axes[0, 0].imshow(img)
            axes[0, 0].set_title(f'宏观照片\n权重: {results["weights"]["hongguan"]:.2f}')
            axes[0, 0].axis('off')
            row_0_used = True

        if image_paths.get('tishijing') and os.path.exists(image_paths['tishijing']):
            img = Image.open(image_paths['tishijing'])
            axes[0, 1].imshow(img)
            axes[0, 1].set_title(f'体视镜照片\n权重: {results["weights"]["tishijing"]:.2f}')
            axes[0, 1].axis('off')
            row_0_used = True

        if not row_0_used:
            axes[0, 0].axis('off')
            axes[0, 1].axis('off')

        # 预测概率柱状图
        classes = list(results['probabilities_zh'].keys())
        probs = list(results['probabilities_zh'].values())

        bars = axes[1, 0].bar(classes, probs, color=['red' if i == results['predicted_class_idx'] else 'lightblue'
                                                    for i in range(len(classes))])
        axes[1, 0].set_title(f'预测结果: {results["predicted_class_zh"]}\n置信度: {results["confidence"]:.4f}')
        axes[1, 0].set_ylabel('概率')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, prob in zip(bars, probs):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom')

        # 模型对比图
        models = ['宏观照片', '体视镜照片', '融合结果']
        model_probs = [
            results['model_predictions']['hongguan']['probabilities'],
            results['model_predictions']['tishijing']['probabilities'],
            list(results['probabilities'].values())
        ]

        x = np.arange(len(classes))
        width = 0.25

        for i, (model, probs_list) in enumerate(zip(models, model_probs)):
            axes[1, 1].bar(x + i*width, probs_list, width, label=model, alpha=0.8)

        axes[1, 1].set_title('模型预测对比')
        axes[1, 1].set_xlabel('损伤类型')
        axes[1, 1].set_ylabel('概率')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels(classes, rotation=45)
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果可视化已保存: {save_path}")

        plt.show()

    def batch_predict(self, test_data_dir, output_dir):
        """
        批量预测测试集

        Args:
            test_data_dir: 测试数据目录
            output_dir: 输出目录

        Returns:
            list: 所有预测结果
        """
        results = []
        class_dirs = ['boluo', 'cashang', 'dianshi', 'yashang']

        for class_dir in class_dirs:
            for photo_type in ['hongguan', 'tishijing']:
                test_dir = os.path.join(test_data_dir, class_dir, photo_type, 'test')
                if not os.path.exists(test_dir):
                    continue

                for img_file in os.listdir(test_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(test_dir, img_file)

                        # 预测
                        if photo_type == 'hongguan':
                            pred_result = self.predict_damage_type(image_hongguan_path=img_path)
                        else:
                            pred_result = self.predict_damage_type(image_tishijing_path=img_path)

                        pred_result.update({
                            'image_path': img_path,
                            'true_class': class_dir,
                            'photo_type': photo_type,
                            'image_file': img_file
                        })

                        results.append(pred_result)

        # 保存批量预测结果
        results_path = os.path.join(output_dir, 'batch_prediction_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"批量预测结果已保存: {results_path}")
        return results

def main():
    """主预测函数"""
    # 配置路径
    model_dir = str(Path(__file__).resolve().parents[1] / 'output/models')
    data_dir = str(Path(__file__).resolve().parents[2] / 'data')
    output_dir = str(Path(__file__).resolve().parents[1] / 'output/results')

    os.makedirs(output_dir, exist_ok=True)

    # 创建预测器
    predictor = DamagePredictor(model_dir)

    # 示例预测
    print("进行示例预测...")

    # 查找测试图像进行示例预测
    test_image_paths = {}
    for class_name in ['boluo', 'cashang', 'dianshi', 'yashang']:
        for photo_type in ['hongguan', 'tishijing']:
            test_dir = os.path.join(data_dir, class_name, photo_type, 'test')
            if os.path.exists(test_dir):
                img_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if img_files:
                    test_image_paths[f'{class_name}_{photo_type}'] = os.path.join(test_dir, img_files[0])

    # 进行几个示例预测
    example_predictions = []
    for i, (key, img_path) in enumerate(test_image_paths.items()):
        if i >= 3:  # 只做3个示例
            break

        class_name, photo_type = key.split('_')
        print(f"\n示例 {i+1}: {class_name} - {photo_type}")
        print(f"图像路径: {img_path}")

        try:
            if photo_type == 'hongguan':
                result = predictor.predict_damage_type(image_hongguan_path=img_path)
            else:
                result = predictor.predict_damage_type(image_tishijing_path=img_path)

            print(f"预测结果: {result['predicted_class_zh']} (置信度: {result['confidence']:.4f})")
            print(f"真实标签: {class_name}")
            print(f"预测正确: {'是' if result['predicted_class'] == class_name else '否'}")

            result['true_class'] = class_name
            result['image_path'] = img_path
            example_predictions.append(result)

            # 可视化结果
            image_paths = {photo_type: img_path}
            save_path = os.path.join(output_dir, f'prediction_example_{i+1}.png')
            predictor.visualize_prediction(result, image_paths, save_path)

        except Exception as e:
            print(f"预测失败: {e}")

    # 保存示例预测结果
    if example_predictions:
        example_path = os.path.join(output_dir, 'example_predictions.json')
        with open(example_path, 'w', encoding='utf-8') as f:
            json.dump(example_predictions, f, indent=2, ensure_ascii=False)
        print(f"\n示例预测结果已保存: {example_path}")

    print("\n预测完成！")

if __name__ == "__main__":
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    main()
