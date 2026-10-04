import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取测试结果数据
with open('output/results/test_results.json', 'r', encoding='utf-8') as f:
    test_results = json.load(f)

predictions = test_results['predictions']
targets = test_results['targets']
test_metrics = test_results['test_metrics']

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. 预测值 vs 真实值散点图
ax1.scatter(targets, predictions, alpha=0.7, s=100)
ax1.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', label='Perfect Prediction')
ax1.set_xlabel('True Values')
ax1.set_ylabel('Predicted Values')
ax1.set_title('Predictions vs True Values')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 预测误差分布
errors = np.array(predictions) - np.array(targets)
ax2.hist(errors, bins=20, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Prediction Error')
ax2.set_ylabel('Frequency')
ax2.set_title('Prediction Error Distribution')
ax2.grid(True, alpha=0.3)

# 3. 样本索引 vs 预测值和真实值
sample_indices = range(len(predictions))
ax3.plot(sample_indices, targets, 'bo-', label='True Values', markersize=8)
ax3.plot(sample_indices, predictions, 'ro-', label='Predictions', markersize=8)
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Values')
ax3.set_title('Sample-wise Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 性能指标可视化
metrics_names = ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE']
metrics_values = [
    test_metrics['mse'],
    test_metrics['rmse'],
    test_metrics['mae'],
    test_metrics['r2'],
    test_metrics['mape']
]

# 对数值进行标准化显示（除了R²和MAPE）
normalized_values = []
for i, (name, value) in enumerate(zip(metrics_names, metrics_values)):
    if name in ['R²', 'MAPE']:
        normalized_values.append(value)
    else:
        normalized_values.append(np.log10(value) if value > 0 else 0)

colors = ['red' if name == 'R²' and value < 0 else 'blue' for name, value in zip(metrics_names, metrics_values)]
ax4.bar(metrics_names, normalized_values, color=colors, alpha=0.7)
ax4.set_ylabel('Log10(Value) or Raw Value')
ax4.set_title('Performance Metrics')
ax4.grid(True, alpha=0.3)

# 添加数值标签
for i, (name, raw_value, norm_value) in enumerate(zip(metrics_names, metrics_values, normalized_values)):
    if name in ['R²', 'MAPE']:
        ax4.text(i, norm_value + 0.1, f'{raw_value:.2f}', ha='center', va='bottom')
    else:
        ax4.text(i, norm_value + 0.1, f'{raw_value:.2e}', ha='center', va='bottom', rotation=45)

plt.tight_layout()

# 确保plots目录存在
import os
os.makedirs('output/plots', exist_ok=True)

# 保存图表
plt.savefig('output/plots/prediction_analysis.png', dpi=300, bbox_inches='tight')
print("\n图表已保存为: output/plots/prediction_analysis.png")
plt.show()

# 打印详细分析
print("=== 预测性能分析 ===")
print(f"测试样本数量: {len(predictions)}")
print(f"预测值范围: [{min(predictions):.2f}, {max(predictions):.2f}]")
print(f"真实值范围: [{min(targets):,}, {max(targets):,}]")
print(f"预测值标准差: {np.std(predictions):.4f}")
print(f"真实值标准差: {np.std(targets):,.0f}")

print("\n=== 性能指标 ===")
for name, value in zip(metrics_names, metrics_values):
    if name == 'R²':
        print(f"{name}: {value:.4f}")
    elif name == 'MAPE':
        print(f"{name}: {value:.2f}%")
    else:
        print(f"{name}: {value:.2e}")

print("\n=== 问题诊断 ===")
if test_metrics['r2'] < 0:
    print("⚠️  R²为负值，表明模型性能差于简单均值预测")
if test_metrics['mape'] > 50:
    print("⚠️  MAPE过高，预测误差极大")
if max(predictions) - min(predictions) < 1:
    print("⚠️  预测值范围过小，模型可能未学到有效模式")
if np.std(predictions) / np.std(targets) < 0.01:
    print("⚠️  预测值变异性远小于真实值，模型缺乏表达能力")
