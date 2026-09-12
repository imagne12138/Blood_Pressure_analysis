"""
生成正常范围（SBP 110-150, DBP < 90）的 Scatter / Bland-Altman / 残差直方图 / Error vs Reference 图
复用 gen_scatter_ba_plots.py 的绘图逻辑，但只使用正常范围数据。

用法:
    cd Blood_Pressure_analysis
    python cache/scripts_extraction_visualization/gen_normal_range_plots.py
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CACHE = r"E:\Kaggle_projects\Blood_Pressure_analysis\cache"
FIG_DIR = os.path.join(CACHE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载数据 ──
data = np.load(os.path.join(CACHE, "inference_results_model2_26.npz"))
pred = data["pred"]
label = data["label"]

sbp_true = label[:, 0]
dbp_true = label[:, 1]
sbp_pred = pred[:, 0]
dbp_pred = pred[:, 1]

# ── 筛选正常范围 ──
normal_mask = (sbp_true >= 110) & (sbp_true <= 150) & (dbp_true < 90)
print(f"正常范围样本: {normal_mask.sum()} / {len(normal_mask)} ({normal_mask.mean()*100:.1f}%)")

st = sbp_true[normal_mask]
sp = sbp_pred[normal_mask]
dt = dbp_true[normal_mask]
dp = dbp_pred[normal_mask]

# ── 子采样（清晰度） ──
rng = np.random.RandomState(42)
idx = rng.choice(len(st), min(3000, len(st)), replace=False)
st_s, sp_s = st[idx], sp[idx]
dt_s, dp_s = dt[idx], dp[idx]

print(f"\n正常范围统计:")
for tt, tp, yl in [(st, sp, "SBP"), (dt, dp, "DBP")]:
    mae = np.mean(np.abs(tt - tp))
    r = np.corrcoef(tt, tp)[0, 1]
    bias = np.mean(tp - tt)
    print(f"  {yl}: MAE={mae:.2f}, R={r:.3f}, Bias={bias:.2f}")

# ════════════════════════════════════════════════════
# 图 1: Scatter Plot (正常范围)
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, tt, tp, yl, cl in [
    (axes[0], st_s, sp_s, "SBP", "#2196F3"),
    (axes[1], dt_s, dp_s, "DBP", "#FF5722"),
]:
    ax.scatter(tt, tp, s=8, alpha=0.3, edgecolors="none", c=cl)
    lims = [min(tt.min(), tp.min()) - 5, max(tt.max(), tp.max()) + 5]
    ax.plot(lims, lims, "k--", lw=1.5, alpha=0.7)

    slope, intercept, r_val, _, _ = stats.linregress(tt, tp)
    x_fit = np.linspace(*lims, 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r-", lw=1.5, alpha=0.8)

    mae = np.mean(np.abs(tt - tp))
    corr = np.corrcoef(tt, tp)[0, 1]
    ax.text(0.05, 0.95, f"MAE={mae:.2f}\nR={corr:.3f}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel(f"True {yl} (mmHg)", fontsize=12)
    ax.set_ylabel(f"Predicted {yl} (mmHg)", fontsize=12)
    ax.set_title(f"{yl}: Predicted vs True (Normal Range)", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

plt.suptitle("Model2 + 26-dim Fusion: Normal Range Scatter Plots (SBP 110-150, DBP<90)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07b_scatter_normal_range.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved 07b_scatter_normal_range.png")

# ════════════════════════════════════════════════════
# 图 2: Bland-Altman (正常范围)
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, tt, tp, yl, cl in [
    (axes[0], st_s, sp_s, "SBP", "#2196F3"),
    (axes[1], dt_s, dp_s, "DBP", "#FF5722"),
]:
    mean_val = (tt + tp) / 2
    diff = tt - tp
    md = np.mean(diff)
    sd = np.std(diff)
    loa_hi = md + 1.96 * sd
    loa_lo = md - 1.96 * sd

    ax.scatter(mean_val, diff, s=8, alpha=0.3, edgecolors="none", c=cl)
    ax.axhline(md, color="k", lw=1.5, linestyle="--", label=f"Mean diff={md:.2f}")
    ax.axhline(loa_hi, color="r", lw=1, linestyle=":", label=f"+1.96SD={loa_hi:.2f}")
    ax.axhline(loa_lo, color="r", lw=1, linestyle=":", label=f"-1.96SD={loa_lo:.2f}")
    ax.legend(fontsize=9)
    ax.set_xlabel(f"Mean of True and Predicted {yl} (mmHg)", fontsize=12)
    ax.set_ylabel(f"True - Predicted {yl} (mmHg)", fontsize=12)
    ax.set_title(f"{yl}: Bland-Altman (Normal Range)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

plt.suptitle("Model2 + 26-dim Fusion: Normal Range Bland-Altman (SBP 110-150, DBP<90)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08b_bland_altman_normal_range.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved 08b_bland_altman_normal_range.png")

# ════════════════════════════════════════════════════
# 图 3: Residual Histogram (正常范围，全量数据)
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, tt, tp, yl, cl in [
    (axes[0], st, sp, "SBP", "#2196F3"),
    (axes[1], dt, dp, "DBP", "#FF5722"),
]:
    res = tt - tp
    ax.hist(res, bins=120, density=True, alpha=0.7, color=cl,
            edgecolor="white", linewidth=0.3)
    mu, sigma = np.mean(res), np.std(res)
    xs = np.linspace(res.min(), res.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), "k-", lw=2,
            label=f"N({mu:.1f},{sigma:.1f})")
    ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Prediction Error (mmHg)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{yl}: Error Distribution (Normal Range)\nµ={mu:.2f}, σ={sigma:.2f}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.suptitle("Model2 + 26-dim Fusion: Normal Range Residual Distributions",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "09b_residual_normal_range.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved 09b_residual_normal_range.png")

# ════════════════════════════════════════════════════
# 图 4: Error vs Reference (正常范围，全量数据)
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, tt, tp, yl, cl in [
    (axes[0], st, sp, "SBP", "#2196F3"),
    (axes[1], dt, dp, "DBP", "#FF5722"),
]:
    err = np.abs(tt - tp)
    ax.scatter(tt, err, s=1, alpha=0.1, edgecolors="none", c=cl)

    bins = np.linspace(tt.min(), tt.max(), 25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = [np.mean(err[(tt >= bins[i]) & (tt < bins[i+1])])
                 for i in range(len(bins) - 1)]
    ax.plot(bin_centers, bin_means, "r-o", lw=2, markersize=4, label="Binned mean")

    slope, intercept, r_val, _, _ = stats.linregress(tt, err)
    ax.plot(bins, slope * bins + intercept, "k--", lw=1.5, alpha=0.6,
            label=f"R={r_val:.3f}")

    ax.set_xlabel(f"True {yl} (mmHg)", fontsize=12)
    ax.set_ylabel("|Prediction Error| (mmHg)", fontsize=12)
    ax.set_title(f"{yl}: |Error| vs True BP (Normal Range)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.suptitle("Model2 + 26-dim Fusion: Normal Range Error vs Reference",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "10b_error_vs_reference_normal_range.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved 10b_error_vs_reference_normal_range.png")

print(f"\n✅ 全部生成完毕，文件在 {FIG_DIR}/")
print("  07b_scatter_normal_range.png")
print("  08b_bland_altman_normal_range.png")
print("  09b_residual_normal_range.png")
print("  10b_error_vs_reference_normal_range.png")
