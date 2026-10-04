import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# 读取训练历史数据
with open('output/results/training_history.json', 'r', encoding='utf-8') as f:
    history = json.load(f)

# 读取测试结果数据
with open('output/results/test_results.json', 'r', encoding='utf-8') as f:
    test_results = json.load(f)

train_loss = np.array(history['train_loss'])
val_loss = np.array(history['val_loss'])
epochs = np.arange(1, len(train_loss) + 1)

# 创建综合分析图表
fig = plt.figure(figsize=(20, 15))

# 1. 损失函数的Scaling Law (2x2的第一个)
ax1 = plt.subplot(3, 3, 1)
plt.loglog(epochs, train_loss, 'b-', alpha=0.7, label='Training Loss')
plt.loglog(epochs, val_loss, 'r-', alpha=0.7, label='Validation Loss')
plt.xlabel('Epoch (log scale)')
plt.ylabel('Loss (log scale)')
plt.title('Loss Scaling Law (Log-Log Plot)')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加趋势线
log_epochs = np.log10(epochs[10:])  # 跳过前几个epoch
log_train_loss = np.log10(train_loss[10:])
log_val_loss = np.log10(val_loss[10:])

# 训练损失趋势
slope_train, intercept_train, r_train, _, _ = stats.linregress(log_epochs, log_train_loss)
train_trend = 10**(slope_train * np.log10(epochs) + intercept_train)
plt.loglog(epochs, train_trend, 'b--', alpha=0.5, label=f'Train Trend (slope={slope_train:.3f})')

# 验证损失趋势
slope_val, intercept_val, r_val, _, _ = stats.linregress(log_epochs, log_val_loss)
val_trend = 10**(slope_val * np.log10(epochs) + intercept_val)
plt.loglog(epochs, val_trend, 'r--', alpha=0.5, label=f'Val Trend (slope={slope_val:.3f})')
plt.legend()

# 2. 线性尺度的损失变化
ax2 = plt.subplot(3, 3, 2)
plt.plot(epochs, train_loss, 'b-', alpha=0.7, label='Training Loss')
plt.plot(epochs, val_loss, 'r-', alpha=0.7, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Evolution (Linear Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

# 标记学习率调整点
lr_reduction_epochs = [45, 56, 67, 78, 89]
for epoch in lr_reduction_epochs:
    if epoch <= len(train_loss):
        plt.axvline(x=epoch, color='green', linestyle='--', alpha=0.5)

# 3. 损失比率分析
ax3 = plt.subplot(3, 3, 3)
loss_ratio = train_loss / val_loss
plt.plot(epochs, loss_ratio, 'purple', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Train Loss / Val Loss')
plt.title('Overfitting Analysis')
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Perfect Ratio')
plt.legend()

# 4. 损失变化率分析
ax4 = plt.subplot(3, 3, 4)
train_loss_change = np.diff(train_loss) / train_loss[:-1] * 100
val_loss_change = np.diff(val_loss) / val_loss[:-1] * 100
plt.plot(epochs[1:], train_loss_change, 'b-', alpha=0.7, label='Train Loss Change %')
plt.plot(epochs[1:], val_loss_change, 'r-', alpha=0.7, label='Val Loss Change %')
plt.xlabel('Epoch')
plt.ylabel('Loss Change (%)')
plt.title('Loss Change Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 5. 预测值分布
ax5 = plt.subplot(3, 3, 5)
predictions = test_results['predictions']
targets = test_results['targets']

plt.hist(predictions, bins=10, alpha=0.7, label='Predictions', color='blue')
plt.hist(targets, bins=10, alpha=0.7, label='True Values', color='red')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Value Distribution Comparison')
plt.legend()
plt.yscale('log')

# 6. 预测误差分析
ax6 = plt.subplot(3, 3, 6)
errors = np.array(predictions) - np.array(targets)
relative_errors = errors / np.array(targets) * 100

plt.scatter(targets, relative_errors, alpha=0.7, s=100)
plt.xlabel('True Values')
plt.ylabel('Relative Error (%)')
plt.title('Relative Error vs True Values')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# 7. 模型收敛性分析
ax7 = plt.subplot(3, 3, 7)
# 计算移动平均
window_size = 10
train_ma = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
val_ma = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
ma_epochs = epochs[window_size-1:]

plt.plot(ma_epochs, train_ma, 'b-', linewidth=2, label=f'Train MA({window_size})')
plt.plot(ma_epochs, val_ma, 'r-', linewidth=2, label=f'Val MA({window_size})')
plt.xlabel('Epoch')
plt.ylabel('Loss (Moving Average)')
plt.title('Convergence Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. 学习效率分析
ax8 = plt.subplot(3, 3, 8)
# 计算每个epoch的学习效率（损失下降量）
learning_efficiency = -np.diff(val_loss)
plt.plot(epochs[1:], learning_efficiency, 'green', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Efficiency')
plt.title('Learning Efficiency per Epoch')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# 9. 性能指标雷达图
ax9 = plt.subplot(3, 3, 9, projection='polar')
metrics = ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE']
values = [
    np.log10(test_results['test_metrics']['mse']) / 15,  # 归一化
    np.log10(test_results['test_metrics']['rmse']) / 10,
    np.log10(test_results['test_metrics']['mae']) / 10,
    (test_results['test_metrics']['r2'] + 5) / 10,  # 调整R²范围
    test_results['test_metrics']['mape'] / 100
]

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
values += values[:1]  # 闭合图形
angles += angles[:1]

ax9.plot(angles, values, 'o-', linewidth=2, color='red')
ax9.fill(angles, values, alpha=0.25, color='red')
ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(metrics)
ax9.set_title('Performance Metrics Radar')

plt.tight_layout()

# 确保plots目录存在
import os
os.makedirs('output/plots', exist_ok=True)

# 保存图表
plt.savefig('output/plots/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("\n综合分析图表已保存为: output/plots/comprehensive_analysis.png")
plt.show()

# 打印详细的Scaling Law分析
print("=== Scaling Law 分析 ===")
print(f"训练损失斜率: {slope_train:.4f} (R²={r_train**2:.4f})")
print(f"验证损失斜率: {slope_val:.4f} (R²={r_val**2:.4f})")
print(f"训练损失变异系数: {np.std(train_loss)/np.mean(train_loss):.4f}")
print(f"验证损失变异系数: {np.std(val_loss)/np.mean(val_loss):.4f}")

print("\n=== 收敛性分析 ===")
final_10_epochs = train_loss[-10:]
convergence_std = np.std(final_10_epochs)
print(f"最后10个epoch训练损失标准差: {convergence_std:.2e}")
print(f"是否收敛: {'是' if convergence_std < np.mean(final_10_epochs) * 0.05 else '否'}")

print("\n=== 学习效率分析 ===")
positive_learning = np.sum(learning_efficiency > 0)
total_epochs = len(learning_efficiency)
print(f"有效学习epoch数: {positive_learning}/{total_epochs} ({positive_learning/total_epochs*100:.1f}%)")
print(f"平均学习效率: {np.mean(learning_efficiency):.2e}")
