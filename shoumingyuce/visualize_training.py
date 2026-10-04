import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练历史数据
with open('output/results/training_history.json', 'r', encoding='utf-8') as f:
    history = json.load(f)

# 提取训练和验证损失
train_loss = history['train_loss']
val_loss = history['val_loss']
epochs = range(1, len(train_loss) + 1)

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 训练损失和验证损失对比
ax1.plot(epochs, train_loss, 'b-', label='训练损失', alpha=0.7)
ax1.plot(epochs, val_loss, 'r-', label='验证损失', alpha=0.7)
ax1.set_xlabel('训练轮数 (Epoch)')
ax1.set_ylabel('损失值')
ax1.set_title('训练过程损失变化 (Scaling Law)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')  # 使用对数坐标

# 验证损失的详细变化（放大显示）
ax2.plot(epochs, val_loss, 'r-', label='验证损失', linewidth=2)
ax2.set_xlabel('训练轮数 (Epoch)')
ax2.set_ylabel('验证损失值')
ax2.set_title('验证损失详细变化')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 添加学习率调整的标记
lr_reduction_epochs = [45, 56, 67, 78, 89]
for epoch in lr_reduction_epochs:
    if epoch <= len(train_loss):
        ax1.axvline(x=epoch, color='green', linestyle='--', alpha=0.5)
        ax2.axvline(x=epoch, color='green', linestyle='--', alpha=0.5)

# 添加图例说明
ax1.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='学习率调整')
ax1.legend()

plt.tight_layout()

# 确保plots目录存在
os.makedirs('output/plots', exist_ok=True)

# 保存图表
plt.savefig('output/plots/training_loss_scaling_law.png', dpi=300, bbox_inches='tight')
print("\n图表已保存为: output/plots/training_loss_scaling_law.png")
plt.show()

# 生成损失统计信息
print("=== 训练损失统计 ===")
print(f"初始训练损失: {train_loss[0]:.2e}")
print(f"最终训练损失: {train_loss[-1]:.2e}")
print(f"训练损失变化: {((train_loss[-1] - train_loss[0]) / train_loss[0] * 100):.2f}%")
print(f"训练损失最大值: {max(train_loss):.2e}")
print(f"训练损失最小值: {min(train_loss):.2e}")

print("\n=== 验证损失统计 ===")
print(f"初始验证损失: {val_loss[0]:.2e}")
print(f"最终验证损失: {val_loss[-1]:.2e}")
print(f"验证损失变化: {((val_loss[-1] - val_loss[0]) / val_loss[0] * 100):.2f}%")
print(f"验证损失最大值: {max(val_loss):.2e}")
print(f"验证损失最小值: {min(val_loss):.2e}")

# 分析训练稳定性
train_loss_std = np.std(train_loss)
val_loss_std = np.std(val_loss)
print(f"\n=== 训练稳定性分析 ===")
print(f"训练损失标准差: {train_loss_std:.2e}")
print(f"验证损失标准差: {val_loss_std:.2e}")
print(f"训练损失变异系数: {train_loss_std / np.mean(train_loss):.4f}")
print(f"验证损失变异系数: {val_loss_std / np.mean(val_loss):.4f}")
