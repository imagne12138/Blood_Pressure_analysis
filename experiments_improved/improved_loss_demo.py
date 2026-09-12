"""
改进方案 Demo：加权损失 + 分层采样 + 两阶段模型

用法：
    cd Blood_Pressure_analysis
    python experiments_improved/improved_loss_demo.py

文件结构：
    experiments_improved/          ← 新实验目录
        improved_loss_demo.py      ← 本文件（加权损失demo）
        train_two_stage.py         ← 两阶段模型训练脚本（待实现）

    model/
        custom_losses.py           ← 自定义损失函数 ← 新增
        two_stage_bp.py            ← 两阶段模型定义 ← 新增

    utils/
        stratified_sampler.py      ← 分层采样器 ← 新增
"""
import os, sys, numpy as np

# ── 路径 ──
BASE = r"E:\Kaggle_projects\Blood_Pressure_analysis"
CACHE = os.path.join(BASE, "cache")
sys.path.insert(0, BASE)

# ── 结果分析 ──
print("=" * 70)
print("改进方案预览：基于当前数据的模拟分析")
print("=" * 70)

# 加载现有推理结果
data = np.load(os.path.join(CACHE, "inference_results_model2_26.npz"))
pred, label = data["pred"], data["label"]
sbp_true, dbp_true = label[:, 0], label[:, 1]
sbp_pred, dbp_pred = pred[:, 0], pred[:, 1]

sbp_err = np.abs(sbp_pred - sbp_true)
dbp_err = np.abs(dbp_pred - dbp_true)

print(f"\n当前（Model2+26融合）：")
print(f"  SBP: MAE={sbp_err.mean():.3f}, RMSE={np.sqrt((sbp_err**2).mean()):.3f}")
print(f"  DBP: MAE={dbp_err.mean():.3f}, RMSE={np.sqrt((dbp_err**2).mean()):.3f}")

# ── 方案1：极端值加权损失的预期效果分析 ──
print(f"\n{'='*70}")
print(f"方案1：加权损失预期效果分析（Weighted MAE Loss）")
print(f"{'='*70}")

sbp_mean = sbp_true.mean()
for alpha in [2.0, 3.0, 5.0]:
    deviation = np.abs(sbp_true - sbp_mean)
    max_dev = deviation.max()
    weights = 1.0 + alpha * (deviation / max_dev)
    weighted_mae = (weights * sbp_err).mean()
    print(f"  alpha={alpha:.1f}: 加权MAE={weighted_mae:.3f}  (原始MAE={sbp_err.mean():.3f})")

print(f"\n  权重范围: [{weights.min():.2f}, {weights.max():.2f}]")
print(f"  极端值(SBP>160)权重={weights[sbp_true>160].mean():.2f}")
print(f"  正常值(120-140)权重={weights[(sbp_true>=120)&(sbp_true<140)].mean():.2f}")

# ── 方案2：两阶段预期效果 ──
print(f"\n{'='*70}")
print(f"方案2：两阶段模型（分类+分区间回归）")
print(f"{'='*70}")

bins_sbp = [(0, 110), (110, 130), (130, 150), (150, 999)]
print(f"\n  SBP区间划分:")
for lo, hi in bins_sbp:
    mask = (sbp_true >= lo) & (sbp_true < hi)
    if mask.any():
        mae = np.abs(sbp_pred[mask] - sbp_true[mask]).mean()
        bias = (sbp_pred[mask] - sbp_true[mask]).mean()
        print(f"    {lo:3d}-{hi:3d}: {mask.sum():>6d} samples, MAE={mae:.2f}, Bias={bias:.2f}")

bins_dbp = [(0, 60), (60, 70), (70, 80), (80, 999)]
print(f"\n  DBP区间划分:")
for lo, hi in bins_dbp:
    mask = (dbp_true >= lo) & (dbp_true < hi)
    if mask.any():
        mae = np.abs(dbp_pred[mask] - dbp_true[mask]).mean()
        bias = (dbp_pred[mask] - dbp_true[mask]).mean()
        print(f"    {lo:3d}-{hi:3d}: {mask.sum():>6d} samples, MAE={mae:.2f}, Bias={bias:.2f}")

# ── 方案3：数据清洗 ──
print(f"\n{'='*70}")
print(f"方案3：标签离群点检测")
print(f"{'='*70}")

# 找出 SBP 和 DBP 有明确矛盾的样本
# 生理上不太可能 SBP<100 且 DBP>100
mask_abnormal = (sbp_true < 100) & (dbp_true > 100)
print(f"  矛盾标签 (SBP<100 & DBP>100): {mask_abnormal.sum()} samples")
mask_abnormal2 = (sbp_true > 180) & (dbp_true < 60)
print(f"  矛盾标签 (SBP>180 & DBP<60): {mask_abnormal2.sum()} samples")

# 毛刺检测：相邻样本 BP 突变过大（如果有序列信息）
# 这里只是示例逻辑

# ── 汇总 ──
print(f"\n{'='*70}")
print(f"推荐实施路径")
print(f"{'='*70}")
print(f"""
  第一步: 使用 WeightedMAELoss 替换当前 L1Loss（改动最小）
    文件: model/custom_losses.py → WeightedMAELoss
    修改: train_fusion_26.py 中的 criterion 即可

  第二步: 使用 StratifiedBPSampler 均衡每个 batch
    文件: utils/stratified_sampler.py
    修改: DataLoader 加入 batch_sampler

  第三步: 训练两阶段模型（效果预期最好）
    文件: model/two_stage_bp.py
    训练: experiments_improved/train_two_stage.py

  第四步: 检查并清洗标签离群点
    上述 {mask_abnormal.sum() + mask_abnormal2.sum()} 个矛盾样本建议人工核查
""")
