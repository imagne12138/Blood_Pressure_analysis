import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

CACHE = r'E:\Kaggle_projects\Blood_Pressure_analysis\cache'
FIG_DIR = os.path.join(CACHE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ===== 数据：12组实验的每折 SBP/DBP MAE（来自 docx 表格） =====
PER_FOLD = {
    1: [(18.728,7.136),(18.716,9.420),(15.227,6.844),(17.208,7.126),(18.404,8.500)],
    2: [(17.880,6.801),(17.356,8.586),(15.660,6.917),(16.251,6.408),(17.590,7.501)],
    3: [(17.322,6.520),(15.996,8.602),(13.088,6.791),(16.034,5.691),(16.469,7.159)],
    4: [(18.203,6.785),(16.827,8.592),(14.209,6.483),(16.768,7.026),(17.005,7.226)],
    5: [(17.990,6.937),(16.487,8.750),(14.343,6.343),(16.686,7.092),(17.068,7.188)],
    6: [(18.037,7.204),(17.718,8.908),(15.610,7.515),(16.042,6.617),(16.623,7.785)],
    7: [(17.349,6.536),(16.227,8.479),(13.586,6.387),(15.492,5.618),(15.741,6.766)],
    8: [(17.701,6.666),(16.672,8.786),(14.518,6.471),(16.200,6.606),(17.116,7.355)],
    9: [(17.968,7.060),(16.633,8.703),(14.226,6.260),(15.932,6.593),(17.148,7.235)],
    10: [(16.848,6.490),(16.143,8.595),(12.858,6.284),(15.763,5.944),(16.013,6.980)],
    11: [(18.907,6.692),(16.491,8.418),(13.290,6.499),(16.216,5.956),(17.016,7.208)],
    12: [(18.710,7.235),(16.930,8.603),(14.981,7.504),(16.059,6.857),(17.250,7.629)],
}

EXP_NAMES = {
    1: '1.Base(orig)', 2: '2.Base(imp)', 3: '3.Model2+WD',
    4: '4.Liu-MLP', 5: '5.Liu-MLP+BN', 6: '6.SimpleMLP',
    7: '7.Base+26', 8: '8.Base+169', 9: '9.M2+169',
    10: '10.M2+26\u2605', 11: '11.M2+Opt', 12: '12.MLP-Opt',
}


def plot_all_12_grid():
    """4x3 网格图：全部12个实验的每折 SBP+DBP 柱状图"""
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()
    x = np.arange(5)
    width = 0.35

    for idx, e in enumerate(range(1, 13)):
        ax = axes[idx]
        sbp = [d[0] for d in PER_FOLD[e]]
        dbp = [d[1] for d in PER_FOLD[e]]

        bars1 = ax.bar(x - width/2, sbp, width, label='SBP', color='#2196F3', edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, dbp, width, label='DBP', color='#FF5722', edgecolor='white', linewidth=0.5)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

        ax.axhline(np.mean(sbp), color='#2196F3', ls='--', lw=1, alpha=0.6)
        ax.axhline(np.mean(dbp), color='#FF5722', ls='--', lw=1, alpha=0.6)
        ax.set_title(f'Exp {e}: {EXP_NAMES[e]}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in range(5)], fontsize=8)
        ax.set_ylabel('MAE (mmHg)', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_ylim(0, 22)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Fold SBP & DBP MAE -- All 12 Experiments', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(FIG_DIR, 'per_fold_grouped_all.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved per_fold_grouped_all.png')


def plot_top3():
    """Top 3 模型每折 SBP+DBP 柱状图"""
    top3 = [10, 7, 3]
    labels = ['Model2+26Dim', 'Base+26Dim', 'Model2+WD']
    x = np.arange(5)
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (e, label) in enumerate(zip(top3, labels)):
        ax = axes[idx]
        sbp = [d[0] for d in PER_FOLD[e]]
        dbp = [d[1] for d in PER_FOLD[e]]

        bars1 = ax.bar(x - width/2, sbp, width, label='SBP', color='#2196F3', edgecolor='white')
        bars2 = ax.bar(x + width/2, dbp, width, label='DBP', color='#FF5722', edgecolor='white')

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{label} (SBP={np.mean(sbp):.2f})', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in range(5)], fontsize=9)
        ax.set_ylabel('MAE (mmHg)')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 22)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Fold Performance -- Top 3 Models', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(FIG_DIR, 'per_fold_grouped_top3.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved per_fold_grouped_top3.png')


if __name__ == '__main__':
    plot_all_12_grid()
    plot_top3()