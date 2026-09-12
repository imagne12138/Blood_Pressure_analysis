"""
Liu2023 特征数据分析脚本
=========================
分析 169 维 PPG 特征与 BP 标签之间的关系，诊断模型训练效果差的原因。

使用方法:
    python Tests/feature_data_analysis.py

依赖: numpy, h5py, matplotlib, seaborn, scikit-learn
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import os
from pathlib import Path

# =============================================================================
# 配置
# =============================================================================
PROJ_DIR = Path(__file__).resolve().parent.parent
H5_FEATURE = PROJ_DIR / "Blood_pressure_dataset" / "liu2023_features.h5"
OUTPUT_DIR = PROJ_DIR / "Tests" / "feature_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 采样数量（全量 517k 太多，随机抽样加速分析）
SAMPLE_SIZE = 50000

print(f"读取特征文件: {H5_FEATURE}")
print(f"分析结果保存至: {OUTPUT_DIR}")


# =============================================================================
# 1. 加载数据
# =============================================================================
def load_data():
    with h5py.File(H5_FEATURE, "r") as f:
        n_total = f["features"].shape[0]
        # 随机采样索引
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, min(SAMPLE_SIZE, n_total), replace=False)
        idx.sort()

        feats = f["features"][idx].astype(np.float64)
        sbp = f["sbp"][idx].astype(np.float64).ravel()
        dbp = f["dbp"][idx].astype(np.float64).ravel()

        names_raw = f["feature_names"][:].tolist()
        feature_names = [n.decode() if isinstance(n, bytes) else n for n in names_raw]

    print(f"  总样本: {n_total}, 采样: {len(idx)}")
    print(f"  特征维度: {feats.shape[1]}")
    print(f"  SBP: mean={sbp.mean():.2f}, std={sbp.std():.2f}, range=[{sbp.min():.2f}, {sbp.max():.2f}]")
    print(f"  DBP: mean={dbp.mean():.2f}, std={dbp.std():.2f}, range=[{dbp.min():.2f}, {dbp.max():.2f}]")
    return feats, sbp, dbp, feature_names, idx


feats, sbp, dbp, feature_names, sample_idx = load_data()

# =============================================================================
# 2. 特征基本质量检查
# =============================================================================
print("\n" + "=" * 60)
print("1. 特征基本质量检查")
print("=" * 60)

# 2a. 零方差 / 近常数特征
variances = np.var(feats, axis=0)
near_zero_var = np.where(variances < 1e-6)[0]
print(f"\n  [零/近零方差特征] 数量: {len(near_zero_var)}")
if len(near_zero_var) > 0:
    for i in near_zero_var[:10]:
        print(f"    - {feature_names[i]}: var={variances[i]:.2e}")

# 2b. 含 NaN/Inf 的特征
nan_counts = np.isnan(feats).sum(axis=0)
inf_counts = np.isinf(feats).sum(axis=0)
bad_feats = np.where((nan_counts > 0) | (inf_counts > 0))[0]
print(f"\n  [含 NaN/Inf 的特征] 数量: {len(bad_feats)}")
for i in bad_feats[:5]:
    print(f"    - {feature_names[i]}: NaN={nan_counts[i]}, Inf={inf_counts[i]}")

# 2c. 特征值范围分布（看是否有极端离群值）
p1 = np.percentile(feats, 1, axis=0)
p99 = np.percentile(feats, 99, axis=0)
feat_range = p99 - p1
extreme_feats = np.where(feat_range > 1e4)[0]
print(f"\n  [极端值特征] (P1-P99 range > 1e4): {len(extreme_feats)}")
for i in extreme_feats[:10]:
    print(f"    - {feature_names[i]}: min={feats[:,i].min():.2f}, max={feats[:,i].max():.2f}, "
          f"P1={p1[i]:.2f}, P99={p99[i]:.2f}")

# 2d. 缺失率统计
missing_ratio = (nan_counts + inf_counts) / feats.shape[0]
high_missing = np.where(missing_ratio > 0.01)[0]
print(f"\n  [高缺失率特征] (>1% NaN/Inf): {len(high_missing)}")

# =============================================================================
# 3. 特征-标签相关性分析
# =============================================================================
print("\n" + "=" * 60)
print("2. 特征-标签相关性分析 (Spearman)")
print("=" * 60)

from scipy.stats import spearmanr, pearsonr

# 对每个特征计算与 SBP/DBP 的相关系数
corr_sbp = np.zeros(feats.shape[1])
corr_dbp = np.zeros(feats.shape[1])
pval_sbp = np.zeros(feats.shape[1])
pval_dbp = np.zeros(feats.shape[1])

for i in range(feats.shape[1]):
    # 跳过 NaN/Inf 样本
    mask = np.isfinite(feats[:, i])
    if mask.sum() < 100:
        corr_sbp[i] = 0
        corr_dbp[i] = 0
    else:
        corr_sbp[i], pval_sbp[i] = spearmanr(feats[mask, i], sbp[mask])
        corr_dbp[i], pval_dbp[i] = spearmanr(feats[mask, i], dbp[mask])

# 最相关的特征
top_n = 10
top_sbp = np.argsort(-np.abs(corr_sbp))[:top_n]
top_dbp = np.argsort(-np.abs(corr_dbp))[:top_n]

print(f"\n  与 SBP Spearman 相关性最高的 {top_n} 个特征:")
for i in top_sbp:
    print(f"    {feature_names[i]:30s}  r={corr_sbp[i]:+.4f}  p={pval_sbp[i]:.2e}")

print(f"\n  与 DBP Spearman 相关性最高的 {top_n} 个特征:")
for i in top_dbp:
    print(f"    {feature_names[i]:30s}  r={corr_dbp[i]:+.4f}  p={pval_dbp[i]:.2e}")

# 总体统计
sig_sbp = np.sum(np.abs(corr_sbp) > 0.05)
sig_dbp = np.sum(np.abs(corr_dbp) > 0.05)
print(f"\n  |r| > 0.05 的特征数量: SBP={sig_sbp}, DBP={sig_dbp}")
print(f"  |r| > 0.10 的特征数量: SBP={np.sum(np.abs(corr_sbp) > 0.10)}, DBP={np.sum(np.abs(corr_dbp) > 0.10)}")
print(f"  |r| > 0.15 的特征数量: SBP={np.sum(np.abs(corr_sbp) > 0.15)}, DBP={np.sum(np.abs(corr_dbp) > 0.15)}")
print(f"  最大 |r|: SBP={np.abs(corr_sbp).max():.4f}, DBP={np.abs(corr_dbp).max():.4f}")

# 相关性分布图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(corr_sbp, bins=50, color='steelblue', edgecolor='white')
axes[0].set_title(f'SBP Spearman 相关性分布 (max |r|={np.abs(corr_sbp).max():.3f})')
axes[0].set_xlabel('Spearman r')
axes[0].set_ylabel('特征数量')
axes[1].hist(corr_dbp, bins=50, color='coral', edgecolor='white')
axes[1].set_title(f'DBP Spearman 相关性分布 (max |r|={np.abs(corr_dbp).max():.3f})')
axes[1].set_xlabel('Spearman r')
axes[1].set_ylabel('特征数量')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_correlation_distribution.png', dpi=100)
plt.close()
print(f"\n  [保存] 01_correlation_distribution.png")

# =============================================================================
# 4. 单个特征预测能力
# =============================================================================
print("\n" + "=" * 60)
print("3. 单个特征的独立预测能力 (单变量回归)")
print("=" * 60)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 随机选一部分特征做单变量预测
n_test_feats = min(169, feats.shape[1])
rng = np.random.RandomState(42)
feat_subset = rng.choice(feats.shape[1], n_test_feats, replace=False)

uni_sbp_mae = []
uni_dbp_mae = []
uni_feat_names = []

for idx in feat_subset:
    mask = np.isfinite(feats[:, idx])
    x = feats[mask, idx].reshape(-1, 1)
    y_sbp = sbp[mask]
    y_dbp = dbp[mask]
    if len(x) < 1000:
        continue
    X_tr, X_va, y_sbp_tr, y_sbp_va, y_dbp_tr, y_dbp_va = train_test_split(
        x, y_sbp, y_dbp, test_size=0.2, random_state=42)

    # 线性回归 (单变量)
    lr = LinearRegression()
    lr.fit(X_tr, y_sbp_tr)
    pred = lr.predict(X_va)
    uni_sbp_mae.append(mean_absolute_error(y_sbp_va, pred))
    lr.fit(X_tr, y_dbp_tr)
    pred = lr.predict(X_va)
    uni_dbp_mae.append(mean_absolute_error(y_dbp_va, pred))
    uni_feat_names.append(feature_names[idx])

uni_sbp_mae = np.array(uni_sbp_mae)
uni_dbp_mae = np.array(uni_dbp_mae)

print(f"\n  单特征线性回归预测 SBP MAE:")
print(f"    最小: {uni_sbp_mae.min():.2f} mmHg")
print(f"    中位数: {np.median(uni_sbp_mae):.2f} mmHg")
print(f"    最大: {uni_sbp_mae.max():.2f} mmHg")

print(f"\n  单特征线性回归预测 DBP MAE:")
print(f"    最小: {uni_dbp_mae.min():.2f} mmHg")
print(f"    中位数: {np.median(uni_dbp_mae):.2f} mmHg")
print(f"    最大: {uni_dbp_mae.max():.2f} mmHg")

# 最佳单特征
best_sbp_feat = uni_feat_names[np.argmin(uni_sbp_mae)]
best_dbp_feat = uni_feat_names[np.argmin(uni_dbp_mae)]
print(f"\n  最佳单特征预测 SBP: {best_sbp_feat} (MAE={uni_sbp_mae.min():.2f} mmHg)")
print(f"  最佳单特征预测 DBP: {best_dbp_feat} (MAE={uni_dbp_mae.min():.2f} mmHg)")

# 绘制单特征 MAE 分布
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(uni_sbp_mae, bins=30, color='steelblue', edgecolor='white')
axes[0].axvline(np.median(uni_sbp_mae), color='red', ls='--', label=f'中位数={np.median(uni_sbp_mae):.1f}')
axes[0].set_title('单特征预测 SBP MAE 分布')
axes[0].set_xlabel('MAE (mmHg)')
axes[0].set_ylabel('特征数量')
axes[0].legend()
axes[1].hist(uni_dbp_mae, bins=30, color='coral', edgecolor='white')
axes[1].axvline(np.median(uni_dbp_mae), color='red', ls='--', label=f'中位数={np.median(uni_dbp_mae):.1f}')
axes[1].set_title('单特征预测 DBP MAE 分布')
axes[1].set_xlabel('MAE (mmHg)')
axes[1].set_ylabel('特征数量')
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_univariate_mae.png', dpi=100)
plt.close()
print(f"  [保存] 02_univariate_mae.png")

# =============================================================================
# 5. 特征冗余度分析
# =============================================================================
print("\n" + "=" * 60)
print("4. 特征间相关性分析 (冗余度)")
print("=" * 60)

# 计算特征间的相关系数 (采样样本)
from scipy.cluster.hierarchy import dendrogram, linkage

# 限制特征数避免内存爆炸
n_feat_sample = min(169, feats.shape[1])
feat_corr = np.corrcoef(feats[:, :n_feat_sample].T)

# 统计高度相关的特征对
upper_tri = np.triu_indices(n_feat_sample, k=1)
high_corr_pairs = np.where(np.abs(feat_corr[upper_tri]) > 0.95)[0]
print(f"\n  特征对总数: {len(upper_tri[0])}")
print(f"  高度相关对 (|r| > 0.95): {len(high_corr_pairs)}")
print(f"  高度相关对 (|r| > 0.90): {np.sum(np.abs(feat_corr[upper_tri]) > 0.90)}")
print(f"  高度相关对 (|r| > 0.80): {np.sum(np.abs(feat_corr[upper_tri]) > 0.80)}")

# 特征相关矩阵热图
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(feat_corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title(f'前 {n_feat_sample} 个特征的相关矩阵')
ax.set_xlabel('特征索引')
ax.set_ylabel('特征索引')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_feature_correlation_matrix.png', dpi=100)
plt.close()
print(f"  [保存] 03_feature_correlation_matrix.png")

# =============================================================================
# 6. 标签分布可视化
# =============================================================================
print("\n" + "=" * 60)
print("5. 标签分布分析")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(sbp, bins=80, color='steelblue', edgecolor='white', density=True)
axes[0].axvline(sbp.mean(), color='red', ls='--', label=f'mean={sbp.mean():.1f}')
axes[0].axvline(sbp.mean() - sbp.std(), color='orange', ls=':', label=f'±1std ({sbp.std():.1f})')
axes[0].axvline(sbp.mean() + sbp.std(), color='orange', ls=':')
axes[0].set_title(f'SBP 分布 (n={len(sbp)})')
axes[0].set_xlabel('SBP (mmHg)')
axes[0].set_ylabel('密度')
axes[0].legend()

axes[1].hist(dbp, bins=80, color='coral', edgecolor='white', density=True)
axes[1].axvline(dbp.mean(), color='red', ls='--', label=f'mean={dbp.mean():.1f}')
axes[1].axvline(dbp.mean() - dbp.std(), color='orange', ls=':', label=f'±1std ({dbp.std():.1f})')
axes[1].axvline(dbp.mean() + dbp.std(), color='orange', ls=':')
axes[1].set_title(f'DBP 分布 (n={len(dbp)})')
axes[1].set_xlabel('DBP (mmHg)')
axes[1].set_ylabel('密度')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_label_distribution.png', dpi=100)
plt.close()
print(f"  [保存] 04_label_distribution.png")

# =============================================================================
# 7. PCA 降维可视化
# =============================================================================
print("\n" + "=" * 60)
print("6. PCA 降维 — 特征是否包含 BP 信息")
print("=" * 60)

from sklearn.decomposition import PCA

# 处理无穷值
feats_clean = feats.copy()
feats_clean[~np.isfinite(feats_clean)] = np.nanmedian(feats_clean)

# 标准化
feats_norm = (feats_clean - feats_clean.mean(axis=0)) / (feats_clean.std(axis=0) + 1e-8)

pca = PCA(n_components=2)
coords = pca.fit_transform(feats_norm)

print(f"  PCA 前2个成分解释方差: {pca.explained_variance_ratio_.sum():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=sbp, s=2, cmap='viridis', alpha=0.5)
axes[0].set_title('PCA + SBP 着色')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(sc1, ax=axes[0], label='SBP (mmHg)')

sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=dbp, s=2, cmap='plasma', alpha=0.5)
axes[1].set_title('PCA + DBP 着色')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(sc2, ax=axes[1], label='DBP (mmHg)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_pca_visualization.png', dpi=150)
plt.close()
print(f"  [保存] 05_pca_visualization.png")

# =============================================================================
# 8. 特征归一化前后对比
# =============================================================================
print("\n" + "=" * 60)
print("7. 特征归一化前后对比 (取前10个特征)")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(16, 6))
axes = axes.ravel()
for i in range(10):
    ax = axes[i]
    ax.hist(feats_clean[:10000, i], bins=50, color='steelblue', edgecolor='white')
    ax.set_title(feature_names[i][:20], fontsize=8)
    ax.tick_params(labelsize=7)
fig.suptitle('特征值分布 (原始, 前10000样本)', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_feature_distributions.png', dpi=100)
plt.close()
print(f"  [保存] 06_feature_distributions.png")

# =============================================================================
# 汇总报告
# =============================================================================
print("\n" + "=" * 60)
print("分析汇总")
print("=" * 60)
print(f"""
1. 特征质量: {len(near_zero_var)} 个近零方差特征, {len(bad_feats)} 个含NaN/Inf特征
2. 与SBP相关性: max |r|={np.abs(corr_sbp).max():.4f}, >0.05的{sig_sbp}个, >0.10的{np.sum(np.abs(corr_sbp) > 0.10)}个
3. 与DBP相关性: max |r|={np.abs(corr_dbp).max():.4f}, >0.05的{sig_dbp}个, >0.10的{np.sum(np.abs(corr_dbp) > 0.10)}个
4. 单特征SBP MAE: 中位数={np.median(uni_sbp_mae):.1f} mmHg, 最优={uni_sbp_mae.min():.1f} mmHg
5. 单特征DBP MAE: 中位数={np.median(uni_dbp_mae):.1f} mmHg, 最优={uni_dbp_mae.min():.1f} mmHg
6. 特征间高度相关(|r|>0.95)对: {len(high_corr_pairs)} 对
7. PCA前2成分解释方差: {pca.explained_variance_ratio_.sum():.3f}
8. SBP标签范围: [{sbp.min():.0f}, {sbp.max():.0f}], std={sbp.std():.1f}
9. DBP标签范围: [{dbp.min():.0f}, {dbp.max():.0f}], std={dbp.std():.1f}
""")

print(f"\n所有图片已保存至: {OUTPUT_DIR}")
