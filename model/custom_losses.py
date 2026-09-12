"""
自定义损失函数：加权损失、分位数损失等
用于缓解 BP 预测中的回归趋中效应（regression-to-the-mean）

用法：
    from model.custom_losses import WeightedMAELoss, QuantileLoss, FocalMAELoss
    criterion = WeightedMAELoss(bp_mean=129.0, alpha=3.0)
    loss = criterion(pred, target)
"""
import torch
import torch.nn as nn


class WeightedMAELoss(nn.Module):
    """
    加权 MAE 损失：偏离总体均值的样本给予更高权重。

    原理：SBP 在 120-140 占大多数，模型倾向于优化这个区间。
          给极端值加权可以强制模型关注尾部。

    Args:
        bp_mean: 训练集 BP 均值（SBP≈129, DBP≈67）
        alpha: 权重放大系数（建议 2.0~5.0）
        eps: 防止除零
    """
    def __init__(self, bp_mean=129.0, alpha=3.0, eps=1e-6):
        super().__init__()
        self.bp_mean = bp_mean
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        # |偏离均值| 越大 → 权重越高
        deviation = torch.abs(target - self.bp_mean)
        # 归一化权重到 [1, 1+alpha] 范围
        max_dev = torch.max(deviation) + self.eps
        weights = 1.0 + self.alpha * (deviation / max_dev)
        loss = (weights * torch.abs(pred - target)).mean()
        return loss


class PerBPWeightedMAELoss(nn.Module):
    """
    SBP/DBP 各自独立加权的 MAE 损失。
    SBP 均值~129，DBP 均值~67，各自使用自己的均值计算偏离度。
    """
    def __init__(self, sbp_mean=129.0, dbp_mean=67.0, alpha=3.0):
        super().__init__()
        self.sbp_loss = WeightedMAELoss(bp_mean=sbp_mean, alpha=alpha)
        self.dbp_loss = WeightedMAELoss(bp_mean=dbp_mean, alpha=alpha)

    def forward(self, pred, target):
        # pred/target: [B, 2] — [SBP, DBP]
        return self.sbp_loss(pred[:, 0:1], target[:, 0:1]) + \
               self.dbp_loss(pred[:, 1:2], target[:, 1:2])


class QuantileLoss(nn.Module):
    """
    分位数损失（Pinball Loss）。
    用不同分位数训练可得到预测区间，适合不确定性估计。

    Args:
        quantiles: 要预测的分位数列表，如 [0.1, 0.5, 0.9]
                   输出维度 = len(quantiles) × 2 (SBP+DBP)
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        # pred: [B, 2*Q] 每对 (SBP_q, DBP_q) 对应一个分位数
        Q = len(self.quantiles)
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            sbp_pred = pred[:, i*2]
            dbp_pred = pred[:, i*2 + 1]
            sbp_true = target[:, 0]
            dbp_true = target[:, 1]

            err_sbp = sbp_true - sbp_pred
            err_dbp = dbp_true - dbp_pred
            # pinball: q*err  if err>0  else (q-1)*err
            loss_sbp = torch.max(q * err_sbp, (q - 1) * err_sbp)
            loss_dbp = torch.max(q * err_dbp, (q - 1) * err_dbp)
            loss += loss_sbp.mean() + loss_dbp.mean()
        return loss / Q


class FocalMAELoss(nn.Module):
    """
    Focal Loss 的回归版本：对已有较大误差的样本给予更高关注。
    Loss = (|pred - true|^gamma) * |pred - true|
    当 gamma=0 时退化为 MAE。

    Args:
        gamma: 聚焦参数（建议 1.0~2.0），越大越关注大误差样本
    """
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        err = torch.abs(pred - target)
        loss = (err ** self.gamma) * err
        return loss.mean()


class LogCoshLoss(nn.Module):
    """
    Log-Cosh 损失：平滑版 MAE，对大误差惩罚比 MSE 轻，比 MAE 光滑。
    对异常值比 MSE 鲁棒，比 MAE 更容易优化（处处可导）。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        err = pred - target
        loss = torch.log(torch.cosh(err + 1e-12))
        return loss.mean()


class CompositeLoss(nn.Module):
    """
    组合损失：加权 MAE + Log-Cosh + 分位数（可选）
    可以同时优化精度和不确定性估计。

    Args:
        weights: 各损失的权重 [w_mae, w_logcosh, w_quantile]
    """
    def __init__(self, weights=[0.5, 0.5, 0.0], quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.w_mae, self.w_logcosh, self.w_quantile = weights
        self.mae = nn.L1Loss()
        self.logcosh = LogCoshLoss()
        if self.w_quantile > 0:
            self.quantile = QuantileLoss(quantiles)

    def forward(self, pred, target):
        # 若使用了分位数损失，pred 需为 [B, 2+2Q]
        loss = self.w_mae * self.mae(pred[:, :2], target) + \
               self.w_logcosh * self.logcosh(pred[:, :2], target)
        if self.w_quantile > 0 and pred.shape[1] > 2:
            loss += self.w_quantile * self.quantile(pred[:, 2:], target)
        return loss
