#!/usr/bin/env python3
"""
齿轮损伤识别系统自动化流水线
自动化执行数据加载、模型训练、预测和评估的完整流程
"""

import os
from pathlib import Path
import sys
import time
import json
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

# 导入项目模块
from data_loader import load_dataset_with_metadata, create_data_loaders
from train import main as train_main
from predict import main as predict_main
from evaluate import main as evaluate_main

class PipelineRunner:
    """流水线运行器"""

    def __init__(self, config):
        """
        初始化流水线运行器

        Args:
            config: 配置字典
        """
        self.config = config
        self.project_root = config['project_root']
        self.data_dir = config['data_dir']
        self.output_dir = config['output_dir']
        self.log_file = os.path.join(self.output_dir, 'pipeline_log.txt')

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'evaluation'), exist_ok=True)

        # 初始化日志
        self.start_time = time.time()
        self.log("="*80)
        self.log(f"齿轮损伤识别系统自动化流水线启动")
        self.log(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("="*80)

    def log(self, message, print_to_console=True):
        """
        记录日志

        Args:
            message: 日志消息
            print_to_console: 是否打印到控制台
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"

        # 写入日志文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

        # 打印到控制台
        if print_to_console:
            print(log_message)

    def check_data_availability(self):
        """检查数据可用性"""
        self.log("步骤 1: 检查数据可用性...")

        # 检查数据目录
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        # 检查DATA.json文件
        json_path = os.path.join(self.data_dir, 'DATA.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"数据元信息文件不存在: {json_path}")

        # 加载数据集统计信息
        try:
            all_data = load_dataset_with_metadata(self.data_dir)

            # 统计各类别数据
            stats = {}
            for item in all_data:
                key = f"{item['label_name']}_{item['photo_type']}_{item['split']}"
                stats[key] = stats.get(key, 0) + 1

            self.log(f"数据集加载成功，总样本数: {len(all_data)}")
            self.log("数据分布统计:")

            for class_name in ['boluo', 'cashang', 'dianshi', 'yashang']:
                for photo_type in ['hongguan', 'tishijing']:
                    train_count = stats.get(f"{class_name}_{photo_type}_train", 0)
                    test_count = stats.get(f"{class_name}_{photo_type}_test", 0)
                    self.log(f"  {class_name}_{photo_type}: 训练集={train_count}, 测试集={test_count}")

            # 保存数据统计
            stats_path = os.path.join(self.output_dir, 'data_statistics.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_samples': len(all_data),
                    'distribution': stats,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)

            self.log(f"数据统计已保存: {stats_path}")
            return True

        except Exception as e:
            self.log(f"数据检查失败: {e}")
            return False

    def run_training(self):
        """运行模型训练"""
        self.log("步骤 2: 开始模型训练...")

        try:
            # 保存当前工作目录
            original_cwd = os.getcwd()

            # 更改到src目录执行训练
            src_dir = os.path.join(self.project_root, 'src')
            os.chdir(src_dir)

            # 执行训练
            self.log("正在训练模型，这可能需要较长时间...")
            start_time = time.time()

            # 调用训练主函数
            hongguan_results, tishijing_results = train_main()

            training_time = time.time() - start_time
            self.log(f"模型训练完成，用时: {training_time:.2f}秒")
            self.log(f"宏观照片模型准确率: {hongguan_results['accuracy']:.4f}")
            self.log(f"体视镜照片模型准确率: {tishijing_results['accuracy']:.4f}")

            # 恢复工作目录
            os.chdir(original_cwd)

            return True

        except Exception as e:
            self.log(f"模型训练失败: {e}")
            # 恢复工作目录
            os.chdir(original_cwd)
            return False

    def run_prediction(self):
        """运行预测"""
        self.log("步骤 3: 开始预测测试...")

        try:
            # 保存当前工作目录
            original_cwd = os.getcwd()

            # 更改到src目录执行预测
            src_dir = os.path.join(self.project_root, 'src')
            os.chdir(src_dir)

            # 执行预测
            self.log("正在进行预测测试...")
            start_time = time.time()

            # 调用预测主函数
            predict_main()

            prediction_time = time.time() - start_time
            self.log(f"预测测试完成，用时: {prediction_time:.2f}秒")

            # 恢复工作目录
            os.chdir(original_cwd)

            return True

        except Exception as e:
            self.log(f"预测测试失败: {e}")
            # 恢复工作目录
            os.chdir(original_cwd)
            return False

    def run_evaluation(self):
        """运行模型评估"""
        self.log("步骤 4: 开始模型评估...")

        try:
            # 保存当前工作目录
            original_cwd = os.getcwd()

            # 更改到src目录执行评估
            src_dir = os.path.join(self.project_root, 'src')
            os.chdir(src_dir)

            # 执行评估
            self.log("正在进行模型性能评估...")
            start_time = time.time()

            # 调用评估主函数
            results = evaluate_main()

            evaluation_time = time.time() - start_time
            self.log(f"模型评估完成，用时: {evaluation_time:.2f}秒")

            # 记录评估结果
            if results:
                self.log("评估结果摘要:")
                for result in results:
                    model_name = result.get('photo_type', result.get('model_type', '未知'))
                    self.log(f"  {model_name}: 准确率={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")

            # 恢复工作目录
            os.chdir(original_cwd)

            return True

        except Exception as e:
            self.log(f"模型评估失败: {e}")
            # 恢复工作目录
            os.chdir(original_cwd)
            return False

    def generate_final_report(self):
        """生成最终报告"""
        self.log("步骤 5: 生成最终报告...")

        try:
            total_time = time.time() - self.start_time

            # 收集所有结果文件
            results_files = []
            results_dir = os.path.join(self.output_dir, 'results')
            evaluation_dir = os.path.join(self.output_dir, 'evaluation')

            # 扫描结果文件
            for directory in [results_dir, evaluation_dir]:
                if os.path.exists(directory):
                    for file in os.listdir(directory):
                        if file.endswith(('.json', '.png', '.txt')):
                            results_files.append(os.path.join(directory, file))

            # 生成最终报告
            final_report = {
                'pipeline_info': {
                    'execution_time': f"{total_time:.2f} seconds",
                    'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'project_root': self.project_root,
                    'data_dir': self.data_dir,
                    'output_dir': self.output_dir
                },
                'generated_files': results_files,
                'pipeline_status': 'completed'
            }

            # 尝试加载评估报告
            eval_report_path = os.path.join(evaluation_dir, 'evaluation_report.json')
            if os.path.exists(eval_report_path):
                with open(eval_report_path, 'r', encoding='utf-8') as f:
                    eval_report = json.load(f)
                    final_report['evaluation_summary'] = eval_report.get('evaluation_summary', {})
                    final_report['best_model'] = eval_report.get('best_model', {})

            # 保存最终报告
            report_path = os.path.join(self.output_dir, 'final_pipeline_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)

            self.log(f"最终报告已保存: {report_path}")
            self.log(f"流水线总耗时: {total_time:.2f}秒")
            self.log(f"生成文件数量: {len(results_files)}")

            return True

        except Exception as e:
            self.log(f"生成最终报告失败: {e}")
            return False

    def run_complete_pipeline(self):
        """运行完整流水线"""
        self.log("开始执行完整的自动化流水线...")

        success_steps = 0
        total_steps = 5

        # 步骤1: 检查数据
        if self.check_data_availability():
            success_steps += 1
        else:
            self.log("数据检查失败，流水线终止")
            return False

        # 步骤2: 训练模型
        if self.config.get('skip_training', False):
            self.log("跳过模型训练步骤")
            success_steps += 1
        else:
            if self.run_training():
                success_steps += 1
            else:
                self.log("模型训练失败，但继续执行后续步骤")

        # 步骤3: 预测测试
        if self.config.get('skip_prediction', False):
            self.log("跳过预测测试步骤")
            success_steps += 1
        else:
            if self.run_prediction():
                success_steps += 1
            else:
                self.log("预测测试失败，但继续执行后续步骤")

        # 步骤4: 模型评估
        if self.config.get('skip_evaluation', False):
            self.log("跳过模型评估步骤")
            success_steps += 1
        else:
            if self.run_evaluation():
                success_steps += 1
            else:
                self.log("模型评估失败，但继续执行后续步骤")

        # 步骤5: 生成最终报告
        if self.generate_final_report():
            success_steps += 1

        # 总结
        self.log("="*80)
        self.log(f"流水线执行完成！成功步骤: {success_steps}/{total_steps}")
        if success_steps == total_steps:
            self.log("所有步骤均成功完成！")
            return True
        else:
            self.log("部分步骤执行失败，请查看详细日志")
            return False

def create_default_config():
    """创建默认配置"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config = {
        'project_root': project_root,
        'data_dir': str(Path(project_root).parent / 'data'),
        'output_dir': os.path.join(project_root, 'output'),
        'skip_training': False,
        'skip_prediction': False,
        'skip_evaluation': False
    }

    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='齿轮损伤识别系统自动化流水线')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--skip-training', action='store_true', help='跳过训练步骤')
    parser.add_argument('--skip-prediction', action='store_true', help='跳过预测步骤')
    parser.add_argument('--skip-evaluation', action='store_true', help='跳过评估步骤')
    parser.add_argument('--data-dir', type=str, help='数据目录路径')
    parser.add_argument('--output-dir', type=str, help='输出目录路径')

    args = parser.parse_args()

    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = create_default_config()

    # 覆盖命令行参数
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.skip_training:
        config['skip_training'] = True
    if args.skip_prediction:
        config['skip_prediction'] = True
    if args.skip_evaluation:
        config['skip_evaluation'] = True

    # 创建并运行流水线
    pipeline = PipelineRunner(config)
    success = pipeline.run_complete_pipeline()

    # 退出状态
    exit_code = 0 if success else 1
    print(f"\n流水线执行{'成功' if success else '失败'}，退出代码: {exit_code}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
