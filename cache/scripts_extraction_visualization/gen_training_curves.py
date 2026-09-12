import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 把 extract_training_logs.py 的函数直接内联，不需要 import
from extract_training_logs import extract_val_mae, extract_train_loss_and_val_mae, load_all_experiments

CACHE = r'E:\Kaggle_projects\Blood_Pressure_analysis\cache'
FIG_DIR = os.path.join(CACHE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def plot_training_comparison(all_data):
    """4个代表性模型的 Val MAE 对比（5折均值±标准差）"""
    selected = {1: 'Baseline (Orginal)', 3: 'Model2+WD', 7: 'Baseline+26fusion', 10: 'Model2+26'}
    sel_colors = {1: '#E74C3C', 3: '#2196F3', 7: '#4CAF50', 10: '#9C27B0'}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for idx, (exp_num, label) in enumerate(selected.items()):
        ax = axes[idx]
        _, folds = all_data[exp_num]
        color = sel_colors[exp_num]

        max_ep = max((len(v) for v in folds.values()), default=0)
        matrix = np.full((len(folds), max_ep), np.nan)
        for f_idx, fold in enumerate(sorted(folds.keys())):
            vals = folds[fold]
            matrix[f_idx, :len(vals)] = vals

        ep_range = range(max_ep)
        mean_c = np.nanmean(matrix, axis=0)
        std_c = np.nanstd(matrix, axis=0)

        ax.plot(ep_range, mean_c, color=color, linewidth=2.5, label='Mean ± std')
        ax.fill_between(ep_range, mean_c - std_c, mean_c + std_c, color=color, alpha=0.15)

        for f_idx, fold in enumerate(sorted(folds.keys())):
            vals = folds[fold]
            ax.plot(range(len(vals)), vals, color=color, alpha=0.15, linewidth=1)

        best_ep = np.nanargmin(mean_c)
        best_v = mean_c[best_ep]
        ax.scatter(best_ep, best_v, color='gold', s=120, zorder=5, marker='*', edgecolors='black', linewidth=0.8)
        ax.annotate(f'{best_v:.3f} @ ep{best_ep}', xy=(best_ep, best_v),
                    xytext=(best_ep + max_ep * 0.02, best_v + 0.015),
                    fontsize=8, fontweight='bold', color='black',
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val MAE')
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle('Training Convergence Comparison (5-fold mean ± std)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(FIG_DIR, '06_training_curves_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved 06_training_curves_comparison.png')


def plot_train_vs_val():
    """Baseline+26融合 Fold 0: Train Loss vs Val MAE 双轴图"""
    fp = os.path.join(CACHE, 'baseline_fusion_26dim', 'log_train_fusion_baseline26_2026-06-02.txt')
    train_losses, val_maes = extract_train_loss_and_val_mae(fp, target_fold=0)

    if not train_losses or not val_maes:
        print('No data for train vs val plot')
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(range(len(train_losses)), train_losses, color='#2196F3', linewidth=2, label='Train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='#2196F3')
    ax1.tick_params(axis='y', labelcolor='#2196F3')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(range(len(val_maes)), val_maes, color='#FF5722', linewidth=2, label='Val MAE')
    ax2.set_ylabel('Val MAE', color='#FF5722')
    ax2.tick_params(axis='y', labelcolor='#FF5722')

    best_ep = np.argmin(val_maes)
    best_v = val_maes[best_ep]
    ax2.scatter(best_ep, best_v, color='gold', s=150, marker='*', edgecolors='black', zorder=5)
    ax2.annotate(f'Best MAE={best_v:.3f}', xy=(best_ep, best_v),
                 xytext=(best_ep + 2, best_v + 0.02), fontsize=9, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.title('Baseline+26fusion - Fold 0: Train Loss vs Val MAE', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '06b_train_vs_val.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved 06b_train_vs_val.png')

def plot_train_vs_val_model2_26():
    """Model2+26融合 Fold 0: Train Loss vs Val MAE 双轴图"""
    fp = os.path.join(CACHE, 'model2_fusion_26dim', 'fusion_feature_proj', 'log_train_fusion26_feat_proj_2026-06-01.txt')
    train_losses, val_maes = extract_train_loss_and_val_mae(fp, target_fold=0)

    if not train_losses or not val_maes:
        print('No data for train vs val plot')
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(range(len(train_losses)), train_losses, color='#2196F3', linewidth=2, label='Train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='#2196F3')
    ax1.tick_params(axis='y', labelcolor='#2196F3')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(range(len(val_maes)), val_maes, color='#FF5722', linewidth=2, label='Val MAE')
    ax2.set_ylabel('Val MAE', color='#FF5722')
    ax2.tick_params(axis='y', labelcolor='#FF5722')

    best_ep = np.argmin(val_maes)
    best_v = val_maes[best_ep]
    ax2.scatter(best_ep, best_v, color='gold', s=150, marker='*', edgecolors='black', zorder=5)
    ax2.annotate(f'Best MAE={best_v:.3f}', xy=(best_ep, best_v),
                 xytext=(best_ep + 2, best_v + 0.02), fontsize=9, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.title('Model2+26fusion - Fold 0: Train Loss vs Val MAE', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '06c_train_vs_val_model2_26.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('06c_train_vs_val_model2_26.png')



if __name__ == '__main__':
    all_data = load_all_experiments(CACHE)
    plot_training_comparison(all_data)
    plot_train_vs_val()
    plot_train_vs_val_model2_26()