import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           confusion_matrix, classification_report)
import json
import pandas as pd
from collections import defaultdict

from data_loader import create_data_loaders
from model import load_model
from predict import DamagePredictor

class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model_dir, data_dir, output_dir):
        """
        初始化评估器

        Args:
            model_dir: 模型目录
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['boluo', 'cashang', 'dianshi', 'yashang']
        self.class_names_zh = ['剥落', '擦伤', '点蚀', '压伤']

        os.makedirs(output_dir, exist_ok=True)

    def evaluate_single_model(self, photo_type, batch_size=16):
        """
        评估单个模型

        Args:
            photo_type: 照片类型 ('hongguan' 或 'tishijing')
            batch_size: 批次大小

        Returns:
            dict: 评估结果
        """
        print(f"评估 {photo_type} 模型...")

        # 加载数据
        dataloaders, dataset_sizes = create_data_loaders(self.data_dir, photo_type, batch_size)

        # 加载模型
        model_path = os.path.join(self.model_dir, f'model_{photo_type}.pth')
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None

        model = load_model(model_path, num_classes=4, model_name='resnet34')
        model.to(self.device)
        model.eval()

        # 进行预测
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None)

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        # 分类报告 - 处理类别数不匹配的问题
        unique_labels = sorted(list(set(all_labels)))
        if len(unique_labels) < len(self.class_names_zh):
            # 如果实际类别数少于预期类别数，只使用实际存在的类别
            actual_class_names = [self.class_names_zh[i] for i in unique_labels]
            report = classification_report(all_labels, all_preds,
                                         labels=unique_labels,
                                         target_names=actual_class_names,
                                         output_dict=True, zero_division=0)
        else:
            report = classification_report(all_labels, all_preds, target_names=self.class_names_zh,
                                         output_dict=True, zero_division=0)

        results = {
            'photo_type': photo_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs,
            'dataset_size': dataset_sizes['test']
        }

        print(f"{photo_type} 模型评估完成:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")

        return results

    def evaluate_fusion_model(self, weight_hongguan=0.3, weight_tishijing=0.7):
        """
        评估融合模型 - 根据齿轮技术属性匹配同一齿轮的不同照片类型

        Args:
            weight_hongguan: 宏观照片权重
            weight_tishijing: 体视镜照片权重

        Returns:
            dict: 评估结果
        """
        print("评估融合模型...")

        # 创建预测器
        predictor = DamagePredictor(self.model_dir, self.device)

        # 从DATA.json加载完整的数据集信息
        from data_loader import load_dataset_with_metadata
        all_data = load_dataset_with_metadata(self.data_dir)

        # 只保留测试集数据
        test_data = [item for item in all_data if item['split'] == 'test']

        # 根据技术属性分组齿轮
        gear_groups = {}
        for item in test_data:
            # 使用技术属性作为齿轮的唯一标识
            tech_params = item['technical_params']
            gear_key = json.dumps(tech_params, sort_keys=True)

            if gear_key not in gear_groups:
                gear_groups[gear_key] = {
                    'hongguan': [],
                    'tishijing': [],
                    'class': item['label_name'],
                    'tech_params': tech_params
                }

            gear_groups[gear_key][item['photo_type']].append(item)

        # 构建融合测试样本
        test_samples = []

        # 处理每个齿轮组
        for gear_key, gear_data in gear_groups.items():
            hongguan_items = gear_data['hongguan']
            tishijing_items = gear_data['tishijing']

            # 如果同时有宏观照片和体视镜照片，进行配对
            if hongguan_items and tishijing_items:
                # 配对所有可能的组合
                for h_item in hongguan_items:
                    for t_item in tishijing_items:
                        test_samples.append({
                            'class': gear_data['class'],
                            'hongguan_path': h_item['image_path'],
                            'tishijing_path': t_item['image_path'],
                            'gear_key': gear_key,
                            'fusion_type': 'both'
                        })

            # 只有宏观照片的情况
            elif hongguan_items:
                for h_item in hongguan_items:
                    test_samples.append({
                        'class': gear_data['class'],
                        'hongguan_path': h_item['image_path'],
                        'tishijing_path': None,
                        'gear_key': gear_key,
                        'fusion_type': 'hongguan_only'
                    })

            # 只有体视镜照片的情况
            elif tishijing_items:
                for t_item in tishijing_items:
                    test_samples.append({
                        'class': gear_data['class'],
                        'hongguan_path': None,
                        'tishijing_path': t_item['image_path'],
                        'gear_key': gear_key,
                        'fusion_type': 'tishijing_only'
                    })

        # 进行融合预测
        all_preds = []
        all_labels = []
        all_probs = []
        successful_predictions = 0
        fusion_type_counts = {'both': 0, 'hongguan_only': 0, 'tishijing_only': 0}

        for sample in test_samples:
            try:
                # 根据融合类型动态调整权重
                if sample['fusion_type'] == 'hongguan_only':
                    # 只有宏观照片，宏观权重为1，体视镜权重为0
                    current_weight_hongguan = 1.0
                    current_weight_tishijing = 0.0
                elif sample['fusion_type'] == 'tishijing_only':
                    # 只有体视镜照片，体视镜权重为1，宏观权重为0
                    current_weight_hongguan = 0.0
                    current_weight_tishijing = 1.0
                else:
                    # 两种照片都有，使用原始权重
                    current_weight_hongguan = weight_hongguan
                    current_weight_tishijing = weight_tishijing

                result = predictor.predict_damage_type(
                    image_hongguan_path=sample['hongguan_path'],
                    image_tishijing_path=sample['tishijing_path'],
                    weight_hongguan=current_weight_hongguan,
                    weight_tishijing=current_weight_tishijing
                )

                pred_class_idx = result['predicted_class_idx']
                true_class_idx = self.class_names.index(sample['class'])
                probs = list(result['probabilities'].values())

                all_preds.append(pred_class_idx)
                all_labels.append(true_class_idx)
                all_probs.append(probs)
                successful_predictions += 1
                fusion_type_counts[sample['fusion_type']] += 1

            except Exception as e:
                print(f"预测失败 ({sample['fusion_type']}): {e}")

        if successful_predictions == 0:
            print("没有成功的预测结果")
            return None

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None)

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        # 分类报告
        report = classification_report(all_labels, all_preds, target_names=self.class_names_zh,
                                     output_dict=True, zero_division=0)

        results = {
            'model_type': 'fusion',
            'weights': {'hongguan': weight_hongguan, 'tishijing': weight_tishijing},
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs,
            'successful_predictions': successful_predictions,
            'total_samples': len(test_samples),
            'fusion_type_counts': fusion_type_counts,
            'gear_groups_count': len(gear_groups)
        }

        print(f"融合模型评估完成:")
        print(f"  齿轮组数: {len(gear_groups)}")
        print(f"  融合类型分布:")
        print(f"    - 双照片融合: {fusion_type_counts['both']}")
        print(f"    - 仅宏观照片: {fusion_type_counts['hongguan_only']}")
        print(f"    - 仅体视镜照片: {fusion_type_counts['tishijing_only']}")
        print(f"  成功预测: {successful_predictions}/{len(test_samples)}")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")

        return results

    def plot_confusion_matrix(self, cm, title, filename, normalize=False):
        """绘制混淆矩阵"""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names_zh, yticklabels=self.class_names_zh)
        plt.title(title)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存: {save_path}")

    def plot_metrics_comparison(self, results_list):
        """绘制指标对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        models = [result.get('photo_type', result.get('model_type', 'unknown')) for result in results_list]

        # 整体指标对比
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['准确率', '精确率', '召回率', 'F1分数']

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [result[metric] for result in results_list]

            ax = axes[i//2, i%2]
            bars = ax.bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
            ax.set_title(f'{name}对比')
            ax.set_ylabel(name)
            ax.set_ylim(0, 1)

            # 添加数值标签
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"指标对比图已保存: {save_path}")

    def plot_per_class_metrics(self, results_list):
        """绘制各类别指标对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        models = [result.get('photo_type', result.get('model_type', 'unknown')) for result in results_list]
        metrics = ['precision_per_class', 'recall_per_class', 'f1_per_class']
        metric_names = ['精确率', '召回率', 'F1分数']

        x = np.arange(len(self.class_names_zh))
        width = 0.25

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]

            for j, (model, result) in enumerate(zip(models, results_list)):
                values = result[metric]
                ax.bar(x + j*width, values, width, label=model, alpha=0.8)

            ax.set_title(f'各类别{name}对比')
            ax.set_xlabel('损伤类型')
            ax.set_ylabel(name)
            ax.set_xticks(x + width)
            ax.set_xticklabels(self.class_names_zh)
            ax.legend()
            ax.set_ylim(0, 1)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'per_class_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"各类别指标对比图已保存: {save_path}")

    def convert_to_python_type(self, obj):
        """将numpy类型转换为Python原生类型"""
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_python_type(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_python_type(value) for key, value in obj.items()}
        else:
            return obj

    def generate_evaluation_report(self, results_list):
        """生成评估报告"""
        # 转换数据类型
        results_list = self.convert_to_python_type(results_list)

        report = {
            'evaluation_summary': {
                'total_models': len(results_list),
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'device': str(self.device)
            },
            'model_results': results_list,
            'best_model': None
        }

        # 找出最佳模型
        best_f1 = 0
        best_model = None
        for result in results_list:
            if result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                best_model = result

        if best_model:
            report['best_model'] = {
                'model_type': best_model.get('photo_type', best_model.get('model_type')),
                'f1_score': best_model['f1_score'],
                'accuracy': best_model['accuracy']
            }

        # 保存报告
        report_path = os.path.join(self.output_dir, 'evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"评估报告已保存: {report_path}")

        # 生成简要文本报告
        text_report = f"""
齿轮损伤识别模型评估报告
================================

评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
使用设备: {self.device}
评估模型数: {len(results_list)}

模型性能汇总:
"""

        for result in results_list:
            model_name = result.get('photo_type', result.get('model_type', '未知'))
            text_report += f"""
{model_name} 模型:
  - 准确率: {result['accuracy']:.4f}
  - 精确率: {result['precision']:.4f}
  - 召回率: {result['recall']:.4f}
  - F1分数: {result['f1_score']:.4f}
"""

        if best_model:
            best_name = best_model.get('photo_type', best_model.get('model_type', '未知'))
            text_report += f"""
最佳模型: {best_name}
  - F1分数: {best_model['f1_score']:.4f}
  - 准确率: {best_model['accuracy']:.4f}
"""

        text_report_path = os.path.join(self.output_dir, 'evaluation_summary.txt')
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(text_report)

        print(f"评估摘要已保存: {text_report_path}")

        return report

def main():
    """主评估函数"""
    # 配置路径
    model_dir = str(Path(__file__).resolve().parents[1] / 'output/models')
    data_dir = str(Path(__file__).resolve().parents[2] / 'data')
    output_dir = str(Path(__file__).resolve().parents[1] / 'output/evaluation')

    print("="*80)
    print("开始模型评估")
    print("="*80)

    # 创建评估器
    evaluator = ModelEvaluator(model_dir, data_dir, output_dir)

    results_list = []

    # 评估宏观照片模型
    print("\n1. 评估宏观照片模型...")
    hongguan_results = evaluator.evaluate_single_model('hongguan')
    if hongguan_results:
        results_list.append(hongguan_results)
        # 绘制混淆矩阵
        evaluator.plot_confusion_matrix(
            np.array(hongguan_results['confusion_matrix']),
            '宏观照片模型混淆矩阵',
            'hongguan_confusion_matrix.png'
        )

    print("\n" + "-"*60)

    # 评估体视镜照片模型
    print("\n2. 评估体视镜照片模型...")
    tishijing_results = evaluator.evaluate_single_model('tishijing')
    if tishijing_results:
        results_list.append(tishijing_results)
        # 绘制混淆矩阵
        evaluator.plot_confusion_matrix(
            np.array(tishijing_results['confusion_matrix']),
            '体视镜照片模型混淆矩阵',
            'tishijing_confusion_matrix.png'
        )

    print("\n" + "-"*60)

    # 评估融合模型
    print("\n3. 评估融合模型...")
    fusion_results = evaluator.evaluate_fusion_model(weight_hongguan=0.3, weight_tishijing=0.7)
    if fusion_results:
        results_list.append(fusion_results)
        # 绘制混淆矩阵
        evaluator.plot_confusion_matrix(
            np.array(fusion_results['confusion_matrix']),
            '融合模型混淆矩阵',
            'fusion_confusion_matrix.png'
        )

    print("\n" + "="*80)

    # 生成对比图表
    if len(results_list) > 1:
        print("\n4. 生成对比图表...")
        evaluator.plot_metrics_comparison(results_list)
        evaluator.plot_per_class_metrics(results_list)

    # 生成评估报告
    print("\n5. 生成评估报告...")
    report = evaluator.generate_evaluation_report(results_list)

    print("\n" + "="*80)
    print("评估完成！")
    print("="*80)

    # 打印最终结果摘要
    print("\n最终结果摘要:")
    for result in results_list:
        model_name = result.get('photo_type', result.get('model_type', '未知'))
        print(f"{model_name:10} - 准确率: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")

    return results_list

if __name__ == "__main__":
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    results = main()
