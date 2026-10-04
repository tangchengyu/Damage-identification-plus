import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.data_loader import load_lifetime_dataset, extract_features_from_shuxing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_outlier_features():
    """
    详细分析异常值样本的特征
    """
    print("=== 异常值特征详细分析 ===")

    # 加载数据
    data_path = str(Path(__file__).resolve().parents[1] / 'data')
    dataset_items = load_lifetime_dataset(data_path, 'hongguan')

    # 异常值样本ID
    outlier_ids = ['yashang_figure1', 'yashang_figure2', 'yashang_figure5']

    # 提取所有样本的特征
    all_features = []
    all_sample_info = []
    feature_names = ['模数', '齿数', '变位系数', '螺旋角', '压力角', '齿顶圆', '齿宽',
                    '额定扭矩', '额定转速',
                    '密度', '弹性模量', '泊松比', '屈服应力', '强化系数', '硬化指数', '摩擦系数']

    for item in dataset_items:
        try:
            # 提取特征
            size_features = np.array(item['size_features'], dtype=float)
            operation_features = np.array(item['operation_features'], dtype=float)
            material_features = np.array(item['material_features'], dtype=float)

            # 合并特征
            features = np.concatenate([size_features, operation_features, material_features])

            if len(features) >= 10:
                all_features.append(features)
                all_sample_info.append({
                    'sample_id': item['sample_id'],
                    'damage_type': item['damage_type'],
                    'lifetime': item['lifetime'],
                    'is_outlier': item['sample_id'] in outlier_ids
                })
        except Exception as e:
            continue

    all_features = np.array(all_features)

    # 创建DataFrame便于分析
    df = pd.DataFrame(all_features[:, :len(feature_names)], columns=feature_names)
    df['sample_id'] = [info['sample_id'] for info in all_sample_info]
    df['damage_type'] = [info['damage_type'] for info in all_sample_info]
    df['lifetime'] = [info['lifetime'] for info in all_sample_info]
    df['is_outlier'] = [info['is_outlier'] for info in all_sample_info]

    print(f"总样本数: {len(df)}")
    print(f"异常值样本数: {df['is_outlier'].sum()}")

    # 分析异常值样本的特征
    print("\n=== 异常值样本详细信息 ===")
    outlier_samples = df[df['is_outlier']]
    normal_samples = df[~df['is_outlier']]

    for idx, row in outlier_samples.iterrows():
        print(f"\n样本: {row['sample_id']}")
        print(f"损伤类型: {row['damage_type']}")
        print(f"寿命: {row['lifetime']:,}")
        print("特征值:")
        for feature in feature_names:
            if feature in row:
                print(f"  {feature}: {row[feature]:.4f}")

    # 统计分析
    print("\n=== 特征统计对比 ===")
    print("\n异常值样本统计:")
    outlier_stats = outlier_samples[feature_names].describe()
    print(outlier_stats)

    print("\n正常样本统计:")
    normal_stats = normal_samples[feature_names].describe()
    print(normal_stats)

    print("\n压伤类型样本统计:")
    yashang_samples = df[df['damage_type'] == 'yashang']
    yashang_stats = yashang_samples[feature_names].describe()
    print(yashang_stats)

    # 找出异常值样本与正常样本差异最大的特征
    print("\n=== 特征差异分析 ===")
    feature_differences = []

    for feature in feature_names:
        outlier_mean = outlier_samples[feature].mean()
        normal_mean = normal_samples[feature].mean()
        yashang_mean = yashang_samples[feature].mean()

        # 计算相对差异
        if normal_mean != 0:
            relative_diff = abs(outlier_mean - normal_mean) / abs(normal_mean) * 100
        else:
            relative_diff = 0

        feature_differences.append({
            'feature': feature,
            'outlier_mean': outlier_mean,
            'normal_mean': normal_mean,
            'yashang_mean': yashang_mean,
            'absolute_diff': abs(outlier_mean - normal_mean),
            'relative_diff': relative_diff
        })

    # 按相对差异排序
    feature_differences.sort(key=lambda x: x['relative_diff'], reverse=True)

    print("特征差异排序（按相对差异）:")
    for diff in feature_differences[:10]:  # 显示前10个差异最大的特征
        print(f"{diff['feature']:12s}: 异常值={diff['outlier_mean']:8.3f}, 正常值={diff['normal_mean']:8.3f}, "
              f"压伤均值={diff['yashang_mean']:8.3f}, 相对差异={diff['relative_diff']:6.1f}%")

    # 可视化分析
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    # 1. 特征分布对比（选择差异最大的几个特征）
    top_diff_features = [diff['feature'] for diff in feature_differences[:6]]

    for i, feature in enumerate(top_diff_features):
        ax = axes[i]

        # 绘制直方图
        ax.hist(normal_samples[feature], bins=20, alpha=0.7, label='正常样本', color='blue', density=True)
        ax.hist(yashang_samples[feature], bins=10, alpha=0.7, label='压伤样本', color='orange', density=True)

        # 标记异常值
        for _, outlier in outlier_samples.iterrows():
            ax.axvline(outlier[feature], color='red', linestyle='--', alpha=0.8)

        ax.set_xlabel(feature)
        ax.set_ylabel('密度')
        ax.set_title(f'{feature} 分布对比')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 7. PCA分析
    ax = axes[6]
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)

    # 绘制不同损伤类型
    damage_types = df['damage_type'].unique()
    colors = ['blue', 'orange', 'green', 'purple']

    for i, damage_type in enumerate(damage_types):
        mask = df['damage_type'] == damage_type
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                  alpha=0.6, label=damage_type, color=colors[i % len(colors)])

    # 突出显示异常值
    outlier_mask = df['is_outlier']
    ax.scatter(features_2d[outlier_mask, 0], features_2d[outlier_mask, 1],
              color='red', s=200, marker='x', linewidth=3, label='异常值')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
    ax.set_title('PCA特征空间分析')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. 寿命vs特征散点图（选择一个重要特征）
    ax = axes[7]
    important_feature = feature_differences[0]['feature']

    for damage_type, color in zip(damage_types, colors):
        mask = df['damage_type'] == damage_type
        ax.scatter(df[mask][important_feature], df[mask]['lifetime'],
                  alpha=0.6, label=damage_type, color=color)

    # 突出显示异常值
    ax.scatter(outlier_samples[important_feature], outlier_samples['lifetime'],
              color='red', s=200, marker='x', linewidth=3, label='异常值')

    ax.set_xlabel(important_feature)
    ax.set_ylabel('寿命')
    ax.set_title(f'寿命 vs {important_feature}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. 异常值检测得分分析
    ax = axes[8]
    outlier_detector = IsolationForest(contamination=0.1, random_state=42)
    outlier_scores = outlier_detector.fit(all_features).decision_function(all_features)

    # 绘制得分分布
    ax.hist(outlier_scores[~outlier_mask], bins=30, alpha=0.7, label='正常样本', color='blue', density=True)
    ax.hist(outlier_scores[outlier_mask], bins=10, alpha=0.7, label='异常样本', color='red', density=True)

    ax.set_xlabel('异常值检测得分')
    ax.set_ylabel('密度')
    ax.set_title('异常值检测得分分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/outlier_feature_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n特征分析图表已保存为: output/plots/outlier_feature_analysis.png")
    plt.show()

    # 分析异常原因
    print("\n=== 异常原因分析 ===")

    # 检查是否所有异常值都来自同一个齿轮（4165）
    print("\n1. 数据来源分析:")
    print("异常值样本都来自齿轮4165的不同图片:")
    print("  - yashang_figure1: 4165-1.jpg")
    print("  - yashang_figure2: 4165-2.jpg")
    print("  - yashang_figure5: 4165-3.jpg")
    print("这表明齿轮4165可能具有特殊的特征组合")

    # 分析特征异常
    print("\n2. 特征异常分析:")
    print("主要异常特征:")
    for diff in feature_differences[:5]:
        if diff['relative_diff'] > 10:  # 相对差异超过10%
            print(f"  - {diff['feature']}: 异常值样本平均值 {diff['outlier_mean']:.3f}, "
                  f"正常样本平均值 {diff['normal_mean']:.3f}, 相对差异 {diff['relative_diff']:.1f}%")

    # 分析工况特征
    print("\n3. 工况特征分析:")
    outlier_torque = outlier_samples['额定扭矩'].iloc[0]
    outlier_speed = outlier_samples['额定转速'].iloc[0]
    print(f"异常值样本工况: 扭矩={outlier_torque:.2f} N·m, 转速={outlier_speed:.0f} r/min")

    normal_torque_mean = normal_samples['额定扭矩'].mean()
    normal_speed_mean = normal_samples['额定转速'].mean()
    print(f"正常样本平均工况: 扭矩={normal_torque_mean:.2f} N·m, 转速={normal_speed_mean:.0f} r/min")

    # 计算功率密度
    outlier_power_density = outlier_torque * outlier_speed / 9550  # 功率(kW)
    normal_power_density = normal_torque_mean * normal_speed_mean / 9550
    print(f"功率对比: 异常值={outlier_power_density:.2f} kW, 正常值平均={normal_power_density:.2f} kW")

    print("\n4. 可能的异常原因:")
    print("  a) 高转速低扭矩工况: 异常值样本具有相对较高的转速和较低的扭矩")
    print("  b) 特殊的齿轮几何参数组合: 模数3.75, 齿数19的组合可能较少见")
    print("  c) 材料特性的特殊组合: 16Cr3NiWMoVNbEQ材料的参数范围可能与其他样本不同")
    print("  d) 数据不平衡: 压伤类型样本较少，且这3个异常值占压伤样本的42.9%")

    print("\n5. 对模型训练的影响:")
    print("  - 这些异常值在训练时被移除，可能导致模型对类似工况的泛化能力不足")
    print("  - 在预测时，如果遇到类似的高转速低扭矩工况，模型可能预测不准确")
    print("  - 建议增加更多类似工况的训练样本，或者对这类工况进行特殊处理")

    # 保存详细分析结果
    analysis_result = {
        'outlier_samples': outlier_samples.to_dict('records'),
        'feature_differences': feature_differences,
        'summary': {
            'total_samples': len(df),
            'outlier_count': int(df['is_outlier'].sum()),
            'outlier_rate_in_yashang': f"{df['is_outlier'].sum()}/{len(yashang_samples)} ({df['is_outlier'].sum()/len(yashang_samples)*100:.1f}%)",
            'main_outlier_source': "齿轮4165的三张不同图片",
            'key_characteristics': {
                'high_speed_low_torque': True,
                'special_gear_geometry': "模数3.75, 齿数19",
                'material_type': "16Cr3NiWMoVNbEQ/S10-0361-2004"
            }
        }
    }

    # 保存到JSON文件
    os.makedirs('output/results', exist_ok=True)
    with open('output/results/outlier_feature_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n详细分析结果已保存为: output/results/outlier_feature_analysis.json")

    return analysis_result

if __name__ == "__main__":
    analyze_outlier_features()
