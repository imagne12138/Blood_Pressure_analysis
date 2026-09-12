"""
只用最佳折（Fold 2）的模型和数据做推理 + 可视化
类似于论文中展示"best fold results"的做法

用法:
    cd Blood_Pressure_analysis
    python cache/scripts_extraction_visualization/gen_best_fold_plots.py
"""
import os, warnings, numpy as np
import torch, torch.nn as nn, h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")

CACHE = r"E:\Kaggle_projects\Blood_Pressure_analysis\cache"
BASE = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset"
FIG_DIR = os.path.join(CACHE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FOLD = 2  # 最佳折


class FusionModel26Infer(nn.Module):
    def __init__(self, filters=(1,32,64,128), num_layers=2, hidden_dim=128, drop_prob=0.2):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.softmax = nn.Softmax(dim=1)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(filters[i], filters[i+1], 3, 1, 1),
                          nn.BatchNorm1d(filters[i+1]), nn.ReLU(),
                          nn.Dropout(drop_prob), nn.MaxPool1d(2, 2))
            for i in range(len(filters)-1)
        ])
        self.bilstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True,
                              dropout=drop_prob, bidirectional=True)
        self.sbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Tanh())
        self.dbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Tanh())
        self.feature_projection = nn.Sequential(
            nn.Linear(26, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.1))
        fusion_dim = hidden_dim * 2 + 16
        self.linear_sbp_out = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(drop_prob), nn.Linear(hidden_dim, 1))
        self.linear_dbp_out = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(drop_prob), nn.Linear(hidden_dim, 1))

    def forward(self, x, features):
        for conv in self.convs: x = conv(x)
        lstm_out, _ = self.bilstm(x.permute(0, 2, 1))
        lstm_out = self.dropout(lstm_out)
        e_sbp = self.softmax(self.sbp_attn(lstm_out))
        e_dbp = self.softmax(self.dbp_attn(lstm_out))
        c_sbp = torch.sum(e_sbp * lstm_out, dim=1)
        c_dbp = torch.sum(e_dbp * lstm_out, dim=1)
        pf = self.feature_projection(features)
        c_sbp = torch.cat([c_sbp, pf], dim=1)
        c_dbp = torch.cat([c_dbp, pf], dim=1)
        return torch.cat([self.linear_sbp_out(c_sbp), self.linear_dbp_out(c_dbp)], dim=1)


class Fusion26Dataset(Dataset):
    def __init__(self, ppg_path, feat_path, indices_dir, train=True,
                 sbp_mean=None, sbp_std=None, dbp_mean=None, dbp_std=None,
                 feat_mean=None, feat_std=None):
        self.ppg_path = ppg_path; self.feat_path = feat_path
        self.ppg_file = None; self.feat_file = None
        self.sbp_mean = sbp_mean; self.sbp_std = sbp_std
        self.dbp_mean = dbp_mean; self.dbp_std = dbp_std
        self.feat_mean = feat_mean; self.feat_std = feat_std
        fi = np.load(indices_dir)
        self.indices = fi["train_idx"] if train else fi["val_idx"]

    def _init_file(self):
        if self.ppg_file is None:
            self.ppg_file = h5py.File(self.ppg_path, "r")
            self.ppg = self.ppg_file["ppg"]
            self.sbp = self.ppg_file["sbp"]
            self.dbp = self.ppg_file["dbp"]
        if self.feat_file is None:
            self.feat_file = h5py.File(self.feat_path, "r")
            self.feats = self.feat_file["ppg_features"]

    def __getitem__(self, index):
        self._init_file()
        idx = self.indices[index]
        ppg = self.ppg[idx]
        xp = torch.from_numpy(ppg).unsqueeze(0).float()
        fa = self.feats[idx].astype(np.float32)
        if self.feat_mean is not None:
            fa = (fa - self.feat_mean) / (self.feat_std + 1e-8)
        sb = self.sbp[idx]; db = self.dbp[idx]
        if self.sbp_mean is not None:
            sb = (sb - self.sbp_mean) / self.sbp_std
            db = (db - self.dbp_mean) / self.dbp_std
        return (xp, torch.from_numpy(fa).float()), torch.tensor([sb, db], dtype=torch.float32)

    def __len__(self):
        return len(self.indices)


def plot_fold_results(sbp_true, sbp_pred, dbp_true, dbp_pred, fold):
    """生成 4 张图：Scatter / Bland-Altman / Residual Hist / Error vs Ref"""
    # 子采样
    rng = np.random.RandomState(42)
    idx = rng.choice(len(sbp_true), min(3000, len(sbp_true)), replace=False)
    st_s, sp_s = sbp_true[idx], sbp_pred[idx]
    dt_s, dp_s = dbp_true[idx], dbp_pred[idx]

    # ── 1. Scatter ──
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
        mae = np.mean(np.abs(tt - tp)); corr = np.corrcoef(tt, tp)[0, 1]
        ax.text(0.05, 0.95, f"MAE={mae:.2f}\nR={corr:.3f}",
                transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_xlabel(f"True {yl} (mmHg)", fontsize=12)
        ax.set_ylabel(f"Predicted {yl} (mmHg)", fontsize=12)
        ax.set_title(f"{yl}: Predicted vs True (Fold {fold})", fontsize=13, fontweight="bold")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    plt.suptitle(f"Model2+26 Fusion — Best Fold (Fold {fold}) Scatter Plots",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"07c_scatter_fold{fold}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved 07c_scatter_fold{fold}.png")

    # ── 2. Bland-Altman ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tt, tp, yl, cl in [
        (axes[0], st_s, sp_s, "SBP", "#2196F3"),
        (axes[1], dt_s, dp_s, "DBP", "#FF5722"),
    ]:
        mean_val = (tt + tp) / 2; diff = tt - tp
        md = np.mean(diff); sd = np.std(diff)
        loa_hi = md + 1.96 * sd; loa_lo = md - 1.96 * sd
        ax.scatter(mean_val, diff, s=8, alpha=0.3, edgecolors="none", c=cl)
        ax.axhline(md, color="k", lw=1.5, linestyle="--", label=f"Mean diff={md:.2f}")
        ax.axhline(loa_hi, color="r", lw=1, linestyle=":", label=f"+1.96SD={loa_hi:.2f}")
        ax.axhline(loa_lo, color="r", lw=1, linestyle=":", label=f"-1.96SD={loa_lo:.2f}")
        ax.legend(fontsize=9)
        ax.set_xlabel(f"Mean of True and Predicted {yl} (mmHg)", fontsize=12)
        ax.set_ylabel(f"True - Predicted {yl} (mmHg)", fontsize=12)
        ax.set_title(f"{yl}: Bland-Altman (Fold {fold})", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
    plt.suptitle(f"Model2+26 Fusion — Best Fold (Fold {fold}) Bland-Altman",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"08c_bland_altman_fold{fold}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved 08c_bland_altman_fold{fold}.png")

    # ── 3. Residual Histogram ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tt, tp, yl, cl in [
        (axes[0], sbp_true, sbp_pred, "SBP", "#2196F3"),
        (axes[1], dbp_true, dbp_pred, "DBP", "#FF5722"),
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
        ax.set_title(f"{yl}: Error Distribution (Fold {fold})\nµ={mu:.2f}, σ={sigma:.2f}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.suptitle(f"Model2+26 Fusion — Best Fold (Fold {fold}) Residual Distributions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"09c_residual_fold{fold}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved 09c_residual_fold{fold}.png")

    # ── 4. Error vs Reference ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tt, tp, yl, cl in [
        (axes[0], sbp_true, sbp_pred, "SBP", "#2196F3"),
        (axes[1], dbp_true, dbp_pred, "DBP", "#FF5722"),
    ]:
        err = np.abs(tt - tp)
        ax.scatter(tt, err, s=1, alpha=0.1, edgecolors="none", c=cl)
        bins = np.linspace(tt.min(), tt.max(), 25)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = [np.mean(err[(tt >= bins[i]) & (tt < bins[i+1])])
                     for i in range(len(bins) - 1)]
        ax.plot(bin_centers, bin_means, "r-o", lw=2, markersize=4, label="Binned mean")
        slope, intercept, r_val, _, _ = stats.linregress(tt, err)
        ax.plot(bins, slope * bins + intercept, "k--", lw=1.5, alpha=0.6, label=f"R={r_val:.3f}")
        ax.set_xlabel(f"True {yl} (mmHg)", fontsize=12)
        ax.set_ylabel("|Prediction Error| (mmHg)", fontsize=12)
        ax.set_title(f"{yl}: |Error| vs True BP (Fold {fold})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.suptitle(f"Model2+26 Fusion — Best Fold (Fold {fold}) Error vs Reference",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"10c_error_vs_reference_fold{fold}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved 10c_error_vs_reference_fold{fold}.png")


# ── 主流程 ──
print(f"{'='*60}")
print(f"加载 Fold {FOLD} 模型 + 推理")
print(f"{'='*60}")

CKPT_DIR = os.path.join(CACHE, "model2_fusion_26dim", "fusion_feature_proj")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

ckpt = os.path.join(CKPT_DIR, f"fusion26_fold_{FOLD}_feat_proj.pkl")
ppg_path = os.path.join(BASE, "segmented_records.h5")
feat_path = os.path.join(BASE, "ppg_features.h5")
fold_dir = os.path.join(BASE, f"cv_fold_{FOLD}.npz")

# 从训练集算归一化参数
train_ds = Fusion26Dataset(ppg_path, feat_path, fold_dir, train=True)
with h5py.File(ppg_path, "r") as f:
    ti = train_ds.indices
    s = f["sbp"][ti]; d = f["dbp"][ti]
    sbp_m = float(s.mean()); sbp_s = float(s.std())
    dbp_m = float(d.mean()); dbp_s = float(d.std())
    print(f"SBP: mean={sbp_m:.1f}, std={sbp_s:.1f}")
    print(f"DBP: mean={dbp_m:.1f}, std={dbp_s:.1f}")

with h5py.File(feat_path, "r") as f:
    fe = f["ppg_features"][train_ds.indices]
    feat_m = fe.mean(axis=0).astype(np.float32)
    feat_s = fe.std(axis=0).astype(np.float32)

# 验证集
ds = Fusion26Dataset(ppg_path, feat_path, fold_dir, train=False,
                     sbp_mean=sbp_m, sbp_std=sbp_s,
                     dbp_mean=dbp_m, dbp_std=dbp_s,
                     feat_mean=feat_m, feat_std=feat_s)
loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

model = FusionModel26Infer().to(DEVICE)
state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"模型加载完成: {ckpt}")

# 推理
fp, fl = [], []
with torch.no_grad():
    for (ppg, feats), labels in loader:
        ppg, feats, labels = ppg.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
        out = model(ppg, feats)
        out[:, 0] = out[:, 0] * sbp_s + sbp_m
        out[:, 1] = out[:, 1] * dbp_s + dbp_m
        ld = labels.clone()
        ld[:, 0] = labels[:, 0] * sbp_s + sbp_m
        ld[:, 1] = labels[:, 1] * dbp_s + dbp_m
        fp.append(out.cpu().numpy())
        fl.append(ld.cpu().numpy())

pred = np.concatenate(fp, axis=0)
label = np.concatenate(fl, axis=0)

sbp_true, sbp_pred = label[:, 0], pred[:, 0]
dbp_true, dbp_pred = label[:, 1], pred[:, 1]

n = len(pred)
sbp_mae = np.mean(np.abs(sbp_true - sbp_pred))
dbp_mae = np.mean(np.abs(dbp_true - dbp_pred))
sbp_r = np.corrcoef(sbp_true, sbp_pred)[0, 1]
dbp_r = np.corrcoef(dbp_true, dbp_pred)[0, 1]
print(f"\nFold {FOLD} 验证集 ({n} samples):")
print(f"  SBP: MAE={sbp_mae:.2f}, R={sbp_r:.3f}")
print(f"  DBP: MAE={dbp_mae:.2f}, R={dbp_r:.3f}")

# 保存推理结果
np.savez(os.path.join(CACHE, f"inference_fold{FOLD}.npz"),
         pred=pred, label=label)
print(f"\n保存推理结果: inference_fold{FOLD}.npz")

# ── 画图 ──
print(f"\n生成可视化...")
plot_fold_results(sbp_true, sbp_pred, dbp_true, dbp_pred, FOLD)
print(f"\n全部完成! 文件在 {FIG_DIR}/")
