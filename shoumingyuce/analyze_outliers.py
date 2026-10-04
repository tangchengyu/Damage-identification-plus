import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_lifetime_dataset, extract_features_from_shuxing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_training_outliers():
    """
    分析训练数据中的异常值
    """
    print("=== 训练数据异常值分析 ===")

    # 定义要排除的异常值样本（与训练时保持一致）
    exclude_outliers = ['yashang_figure1', 'yashang_figure2', 'yashang_figure5']
    print(f"排除异常值样本: {exclude_outliers}")

    # 加载数据
    data_path = str(Path(__file__).resolve().parents[1] / 'data')
    dataset_items = load_lifetime_dataset(data_path, 'hongguan', exclude_outliers=exclude_outliers)

    print(f"总样本数: {len(dataset_items)}")

    # 提取特征和寿命
    all_features = []
    lifetimes = []
    sample_info = []

    for item in dataset_items:
        try:
            # 提取特征
            size_features = np.array(item['size_features'], dtype=float)
            operation_features = np.array(item['operation_features'], dtype=float)
            material_features = np.array(item['material_features'], dtype=float)

            # 合并特征
            features = np.concatenate([size_features, operation_features, material_features])

            print(f"样本 {item.get('sample_id', 'unknown')}: 特征维度 {len(features)}")
            print(f"  尺寸特征: {len(size_features)}, 工况特征: {len(operation_features)}, 材料特征: {len(material_features)}")

            # 检查特征维度（放宽条件）
            if len(features) >= 10:  # 放宽特征维度要求
                all_features.append(features)
                lifetimes.append(item['lifetime'])
                sample_info.append({
                    'sample_id': item['sample_id'],
                    'damage_type': item['damage_type'],
                    'image_path': item['image_path'],
                    'lifetime': item['lifetime']
                })
            else:
                print(f"  跳过样本，特征维度不足: {len(features)}")
        except Exception as e:
            print(f"处理样本 {item.get('sample_id', 'unknown')} 时出错: {e}")
            continue

    if len(all_features) == 0:
        print("错误：没有有效的样本数据")
        return None

    all_features = np.array(all_features)
    lifetimes = np.array(lifetimes)

    print(f"有效样本数: {len(all_features)}")
    print(f"特征维度: {all_features.shape[1]}")

    # 1. 基于特征的异常值检测
    outlier_detector = IsolationForest(contamination=0.1, random_state=42)
    feature_outlier_mask = outlier_detector.fit_predict(all_features) == 1

    # 2. 基于寿命的异常值检测（IQR方法）
    Q1 = np.percentile(lifetimes, 25)
    Q3 = np.percentile(lifetimes, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    lifetime_outlier_mask = (lifetimes >= lower_bound) & (lifetimes <= upper_bound)

    # 3. 综合异常值检测
    valid_mask = feature_outlier_mask & lifetime_outlier_mask

    print(f"\n=== 异常值检测结果 ===")
    print(f"基于特征的正常样本: {np.sum(feature_outlier_mask)}/{len(feature_outlier_mask)}")
    print(f"基于寿命的正常样本: {np.sum(lifetime_outlier_mask)}/{len(lifetime_outlier_mask)}")
    print(f"综合正常样本: {np.sum(valid_mask)}/{len(valid_mask)}")
    print(f"总异常值数量: {len(valid_mask) - np.sum(valid_mask)}")

    # 找出异常值样本
    outlier_indices = np.where(~valid_mask)[0]
    feature_outlier_indices = np.where(~feature_outlier_mask)[0]
    lifetime_outlier_indices = np.where(~lifetime_outlier_mask)[0]

    print(f"\n=== 异常值详细信息 ===")

    # 特征异常值
    print(f"\n特征异常值 ({len(feature_outlier_indices)} 个):")
    for idx in feature_outlier_indices:
        info = sample_info[idx]
        print(f"  - {info['sample_id']} ({info['damage_type']}) - 寿命: {info['lifetime']:,} - 图片: {info['image_path']}")

    # 寿命异常值
    print(f"\n寿命异常值 ({len(lifetime_outlier_indices)} 个):")
    print(f"寿命正常范围: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    for idx in lifetime_outlier_indices:
        info = sample_info[idx]
        lifetime = info['lifetime']
        if lifetime < lower_bound:
            outlier_type = "过低"
        else:
            outlier_type = "过高"
        print(f"  - {info['sample_id']} ({info['damage_type']}) - 寿命: {lifetime:,} ({outlier_type}) - 图片: {info['image_path']}")

    # 综合异常值
    print(f"\n综合异常值 ({len(outlier_indices)} 个):")
    for idx in outlier_indices:
        info = sample_info[idx]
        is_feature_outlier = idx in feature_outlier_indices
        is_lifetime_outlier = idx in lifetime_outlier_indices
        outlier_reasons = []
        if is_feature_outlier:
            outlier_reasons.append("特征异常")
        if is_lifetime_outlier:
            outlier_reasons.append("寿命异常")
        print(f"  - {info['sample_id']} ({info['damage_type']}) - 寿命: {info['lifetime']:,} - 原因: {', '.join(outlier_reasons)} - 图片: {info['image_path']}")

    # 按损伤类型统计异常值
    print(f"\n=== 按损伤类型统计异常值 ===")
    damage_types = {}
    for idx, info in enumerate(sample_info):
        damage_type = info['damage_type']
        if damage_type not in damage_types:
            damage_types[damage_type] = {'total': 0, 'outliers': 0}
        damage_types[damage_type]['total'] += 1
        if not valid_mask[idx]:
            damage_types[damage_type]['outliers'] += 1

    for damage_type, stats in damage_types.items():
        outlier_rate = stats['outliers'] / stats['total'] * 100
        print(f"{damage_type}: {stats['outliers']}/{stats['total']} ({outlier_rate:.1f}%)")

    # 可视化异常值分布
    plt.figure(figsize=(15, 10))

    # 1. 寿命分布
    plt.subplot(2, 3, 1)
    plt.hist(lifetimes[valid_mask], bins=30, alpha=0.7, label='正常样本', color='blue')
    plt.hist(lifetimes[~valid_mask], bins=30, alpha=0.7, label='异常样本', color='red')
    plt.axvline(lower_bound, color='green', linestyle='--', label=f'下界: {lower_bound:,.0f}')
    plt.axvline(upper_bound, color='green', linestyle='--', label=f'上界: {upper_bound:,.0f}')
    plt.xlabel('寿命')
    plt.ylabel('频次')
    plt.title('寿命分布')
    plt.legend()
    plt.yscale('log')

    # 2. 按损伤类型的寿命分布
    plt.subplot(2, 3, 2)
    colors = ['blue', 'orange', 'green', 'red']
    for i, (damage_type, color) in enumerate(zip(damage_types.keys(), colors)):
        type_indices = [idx for idx, info in enumerate(sample_info) if info['damage_type'] == damage_type]
        type_lifetimes = lifetimes[type_indices]
        plt.scatter(range(len(type_lifetimes)), type_lifetimes, alpha=0.6, label=damage_type, color=color)
    plt.xlabel('样本索引')
    plt.ylabel('寿命')
    plt.title('按损伤类型的寿命分布')
    plt.legend()
    plt.yscale('log')

    # 3. 特征空间可视化（PCA降维）
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)

    plt.subplot(2, 3, 3)
    plt.scatter(features_2d[valid_mask, 0], features_2d[valid_mask, 1], alpha=0.6, label='正常样本', color='blue')
    plt.scatter(features_2d[~valid_mask, 0], features_2d[~valid_mask, 1], alpha=0.8, label='异常样本', color='red', s=100)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
    plt.title('特征空间异常值分布 (PCA)')
    plt.legend()

    # 4. 异常值样本的图片路径分析
    plt.subplot(2, 3, 4)
    outlier_damage_counts = {}
    for idx in outlier_indices:
        damage_type = sample_info[idx]['damage_type']
        outlier_damage_counts[damage_type] = outlier_damage_counts.get(damage_type, 0) + 1

    if outlier_damage_counts:
        plt.pie(outlier_damage_counts.values(), labels=outlier_damage_counts.keys(), autopct='%1.1f%%')
        plt.title('异常值按损伤类型分布')

    # 5. 寿命异常值的具体分析
    plt.subplot(2, 3, 5)
    lifetime_outliers = lifetimes[~lifetime_outlier_mask]
    if len(lifetime_outliers) > 0:
        plt.hist(lifetime_outliers, bins=20, alpha=0.7, color='red')
        plt.xlabel('寿命')
        plt.ylabel('频次')
        plt.title('寿命异常值分布')
        plt.yscale('log')

    # 6. 特征异常值的分析
    plt.subplot(2, 3, 6)
    if len(feature_outlier_indices) > 0:
        outlier_scores = outlier_detector.decision_function(all_features)
        plt.hist(outlier_scores[feature_outlier_mask], bins=30, alpha=0.7, label='正常样本', color='blue')
        plt.hist(outlier_scores[~feature_outlier_mask], bins=30, alpha=0.7, label='异常样本', color='red')
        plt.xlabel('异常值得分')
        plt.ylabel('频次')
        plt.title('特征异常值得分分布')
        plt.legend()

    plt.tight_layout()

    # 保存图表
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/outlier_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n异常值分析图表已保存为: output/plots/outlier_analysis.png")
    plt.show()

    # 保存异常值详细信息
    outlier_details = {
        'summary': {
            'total_samples': len(all_features),
            'valid_samples': int(np.sum(valid_mask)),
            'outlier_samples': int(len(valid_mask) - np.sum(valid_mask)),
            'feature_outliers': int(len(feature_outlier_indices)),
            'lifetime_outliers': int(len(lifetime_outlier_indices))
        },
        'lifetime_bounds': {
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR)
        },
        'outlier_samples': []
    }

    for idx in outlier_indices:
        info = sample_info[idx]
        is_feature_outlier = idx in feature_outlier_indices
        is_lifetime_outlier = idx in lifetime_outlier_indices
        outlier_details['outlier_samples'].append({
            'sample_id': info['sample_id'],
            'damage_type': info['damage_type'],
            'image_path': info['image_path'],
            'lifetime': int(info['lifetime']),
            'is_feature_outlier': bool(is_feature_outlier),
            'is_lifetime_outlier': bool(is_lifetime_outlier)
        })

    # 保存到JSON文件
    os.makedirs('output/results', exist_ok=True)
    with open('output/results/outlier_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(outlier_details, f, indent=2, ensure_ascii=False)

    print(f"\n异常值详细信息已保存为: output/results/outlier_analysis.json")

    return outlier_details

if __name__ == "__main__":
    analyze_training_outliers()
