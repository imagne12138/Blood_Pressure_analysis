"""Analyze 26-dim feature statistics."""
import h5py
import numpy as np

f = h5py.File('e:/Kaggle_projects/Blood_Pressure_analysis/Blood_pressure_dataset/ppg_features.h5', 'r')
feats = f['ppg_features'][:]
print('Shape:', feats.shape)
print('dtype:', feats.dtype)
print()

# 检查每个特征的统计量
mean = feats.mean(axis=0)
std = feats.std(axis=0)
mn = feats.min(axis=0)
mx = feats.max(axis=0)

feature_names = [
    'peak_mean','peak_std','peak_min','peak_max','peak_25%','peak_50%','peak_75%','peak_count',
    'valley_mean','valley_std','valley_min','valley_max','valley_25%','valley_50%','valley_75%','valley_count',
    'peak_int_mean','peak_int_std','peak_int_min','peak_int_max','peak_int_count',
    'valley_int_mean','valley_int_std','valley_int_min','valley_int_max','valley_int_count'
]

print(f'{"Feat":>20} {"Mean":>10} {"Std":>10} {"Min":>10} {"Max":>10} {"ZeroVar?":>8}')
print('-' * 70)
for i in range(26):
    zv = 'YES' if std[i] < 1e-8 else ''
    print(f'{feature_names[i]:>20} {mean[i]:>10.4f} {std[i]:>10.4f} {mn[i]:>10.4f} {mx[i]:>10.4f} {zv:>8}')

# 检查 count 类特征
print()
peak_counts = feats[:, 7]
valley_counts = feats[:, 15]
print(f'Peak_count unique: {len(np.unique(peak_counts))} values, range [{peak_counts.min():.0f}, {peak_counts.max():.0f}]')
print(f'Valley_count unique: {len(np.unique(valley_counts))} values, range [{valley_counts.min():.0f}, {valley_counts.max():.0f}]')

# 检查 peak_int_count 和 valley_int_count
print(f'Peak_int_count unique: {len(np.unique(feats[:, 20]))} values, range [{feats[:,20].min():.0f}, {feats[:,20].max():.0f}]')
print(f'Valley_int_count unique: {len(np.unique(feats[:, 25]))} values, range [{feats[:,25].min():.0f}, {feats[:,25].max():.0f}]')

# 检查样本相关性
corr = np.corrcoef(feats[:5000], rowvar=False)
print(f'\nMean abs corr between features: {np.mean(np.abs(corr[np.triu_indices(26, k=1)])):.4f}')
# 找出高相关特征对
high_corr = []
for i in range(26):
    for j in range(i+1, 26):
        if abs(corr[i,j]) > 0.95:
            high_corr.append((i, j, corr[i,j]))
if high_corr:
    print(f'Highly correlated pairs (|r|>0.95):')
    for i,j,r in high_corr:
        print(f'  {feature_names[i]} ~ {feature_names[j]} : r={r:.3f}')
else:
    print('No highly correlated pairs found (|r|>0.95)')

f.close()
