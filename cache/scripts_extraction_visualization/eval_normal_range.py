"""
对 Model2+26 融合模型的推断结果做正常范围（SBP 100-160, DBP < 100）分析。
复用 run_inference_model2_26.py 的模型和数据流程，在推理后按 BP 范围分组统计。

输出:
  1. 正常范围 vs 极端范围的对比指标
  2. 按 BP 区间细分的 MAE/Bias/RMSE
  3. 如果已有 inference_results_model2_26.npz，直接加载做分析（更快）
     否则先跑推理再分析

用法:
    python cache/scripts_extraction_visualization/eval_normal_range.py
"""
import os, warnings, numpy as np, torch, torch.nn as nn, h5py
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

CACHE = r"E:\Kaggle_projects\Blood_Pressure_analysis\cache"
BASE = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset"
FIG_DIR = os.path.join(CACHE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── 模型定义（与 run_inference_model2_26.py 一致） ──
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
        for conv in self.convs:
            x = conv(x)
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


# ── 数据集 ──
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


# ── 分析函数 ──
def print_range_analysis(pred, label, name=""):
    """按 BP 范围打印详细的误差统计"""
    print(f"\n{'='*65}")
    print(f"{name} 按 SBP 真实值分段的误差分析")
    print(f"{'='*65}")

    for bp_name, col, bins in [
        ("SBP", 0, [(0,100),(100,120),(120,140),(140,160),(160,200),(200,999)]),
        ("DBP", 1, [(0,70),(70,80),(80,100),(100,999)]),
    ]:
        print(f"\n  {bp_name}:")
        for lo, hi in bins:
            m = (label[:, col] >= lo) & (label[:, col] < hi)
            if m.sum() == 0:
                continue
            err = np.abs(pred[m, col] - label[m, col])
            bias = pred[m, col] - label[m, col]
            rmse = np.sqrt((bias ** 2).mean())
            within5 = (err < 5).mean() * 100
            within10 = (err < 10).mean() * 100
            within15 = (err < 15).mean() * 100
            print(f"    {lo:3d}-{hi:3d} mmHg: {m.sum():>6d} samples")
            print(f"      MAE={err.mean():.2f} | RMSE={rmse:.2f} | Bias={bias.mean():.2f}")
            print(f"      ±5 mmHg: {within5:.0f}%  |  ±10 mmHg: {within10:.0f}%  |  ±15 mmHg: {within15:.0f}%")


def print_summary(pred, label, name="", mask=None):
    """打印汇总指标"""
    if mask is not None:
        p = pred[mask]; l = label[mask]
    else:
        p = pred; l = label

    print(f"\n  {name} ({len(p)} samples):")
    for bp_name, col in [("SBP", 0), ("DBP", 1)]:
        err = np.abs(p[:, col] - l[:, col])
        bias = p[:, col] - l[:, col]
        mae = err.mean()
        rmse = np.sqrt((bias ** 2).mean())
        r = np.corrcoef(p[:, col], l[:, col])[0, 1]
        print(f"    {bp_name}: MAE={mae:.2f} | RMSE={rmse:.2f} | Bias={bias.mean():.2f} | R={r:.3f}")


# ── 主流程 ──
if __name__ == '__main__':
    npz_path = os.path.join(CACHE, "inference_results_model2_26.npz")

    if os.path.exists(npz_path):
        print("📂 加载已有推理结果...")
        data = np.load(npz_path)
        pred, label = data["pred"], data["label"]
        print(f"  已加载 {len(pred)} 个样本\n")
    else:
        print("🔧 未找到缓存，运行推理...")
        CKPT_DIR = os.path.join(CACHE, "model2_fusion_26dim", "fusion_feature_proj")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {DEVICE}")

        all_p, all_l = [], []
        for fold in range(5):
            ckpt = os.path.join(CKPT_DIR, f"fusion26_fold_{fold}_feat_proj.pkl")
            ppg_path = os.path.join(BASE, "segmented_records.h5")
            feat_path = os.path.join(BASE, "ppg_features.h5")
            fold_dir = os.path.join(BASE, f"cv_fold_{fold}.npz")

            # 从训练集算归一化参数
            train_ds = Fusion26Dataset(ppg_path, feat_path, fold_dir, train=True)
            with h5py.File(ppg_path, "r") as f:
                ti = train_ds.indices; s = f["sbp"][ti]; d = f["dbp"][ti]
                sbp_m = float(s.mean()); sbp_s = float(s.std())
                dbp_m = float(d.mean()); dbp_s = float(d.std())
            with h5py.File(feat_path, "r") as f:
                fe = f["ppg_features"][train_ds.indices]
                feat_m = fe.mean(axis=0).astype(np.float32)
                feat_s = fe.std(axis=0).astype(np.float32)

            ds = Fusion26Dataset(ppg_path, feat_path, fold_dir, train=False,
                                 sbp_mean=sbp_m, sbp_std=sbp_s,
                                 dbp_mean=dbp_m, dbp_std=dbp_s,
                                 feat_mean=feat_m, feat_std=feat_s)
            loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

            model = FusionModel26Infer().to(DEVICE)
            state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state); model.eval()

            fp, fl = [], []
            with torch.no_grad():
                for (ppg, feats), labels in loader:
                    ppg, feats, labels = ppg.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
                    out = model(ppg, feats)
                    # 转 mmHg
                    out[:, 0] = out[:, 0] * sbp_s + sbp_m
                    out[:, 1] = out[:, 1] * dbp_s + dbp_m
                    ld = labels.clone()
                    ld[:, 0] = labels[:, 0] * sbp_s + sbp_m
                    ld[:, 1] = labels[:, 1] * dbp_s + dbp_m
                    fp.append(out.cpu().numpy())
                    fl.append(ld.cpu().numpy())
            all_p.append(np.concatenate(fp))
            all_l.append(np.concatenate(fl))
            print(f"  Fold {fold}: {len(all_p[-1])} samples")

        pred = np.concatenate(all_p, axis=0)
        label = np.concatenate(all_l, axis=0)
        print(f"\n  总样本: {len(pred)}")

    # ── 按 BP 正常范围筛选 ──
    sbp_true = label[:, 0]
    dbp_true = label[:, 1]

    # 定义"正常范围"
    sbp_min, sbp_max = 110, 150
    dbp_max = 90

    normal_mask = (sbp_true >= sbp_min) & (sbp_true <= sbp_max) & (dbp_true <= dbp_max)
    abnormal_mask = ~normal_mask

    print(f"{'='*65}")
    print(f"正常范围定义: SBP [{sbp_min}-{sbp_max}] & DBP < {dbp_max}")
    print(f"{'='*65}")
    print(f"  正常样本: {normal_mask.sum()} ({normal_mask.mean()*100:.1f}%)")
    print(f"  异常样本: {abnormal_mask.sum()} ({abnormal_mask.mean()*100:.1f}%)")
    print()

    # ── 正常 vs 异常对比 ──
    print(f"{'='*65}")
    print("正常范围 vs 极端值 对比")
    print(f"{'='*65}")
    print_summary(pred, label, "📊 全体", mask=None)
    print_summary(pred, label, "✅ 正常范围", mask=normal_mask)
    print_summary(pred, label, "⚠️  极端值", mask=abnormal_mask)

    # ── 详细分段 ──
    print_range_analysis(pred[normal_mask], label[normal_mask], "✅ 正常范围")

    print(f"\n{'='*65}")
    print(f"极端值范围详细分析 ({abnormal_mask.sum()} samples)")
    print(f"{'='*65}")
    print_range_analysis(pred[abnormal_mask], label[abnormal_mask], "⚠️  极端值")

    # ── AAMI 标准评估 ──
    print(f"\n{'='*65}")
    print("AAMI 标准评估 (正常范围)")
    print(f"{'='*65}")
    print("AAMI SP10 标准: Bias ≤ ±5 mmHg, SD ≤ ±8 mmHg")
    for bp_name, col in [("SBP", 0), ("DBP", 1)]:
        err = pred[normal_mask, col] - label[normal_mask, col]
        bias = err.mean()
        sd = err.std()
        print(f"  {bp_name}: Bias={bias:.2f} | SD={sd:.2f} | {'✅ PASS' if abs(bias)<=5 and sd<=8 else '❌ FAIL'}")

    # ── 保存分析结果 ──
    np.savez(os.path.join(CACHE, "normal_range_analysis.npz"),
             pred=pred, label=label,
             normal_mask=normal_mask)
    print(f"\n✅ 分析结果已保存到 normal_range_analysis.npz")
