"""
两阶段 BP 预测模型（复用 FusionModel26 的完整 PPG 编码器）

与 FusionModel26 的关系：
  - 相同：CNN×3 + BiLSTM + 分头注意力 + 特征投影 (26→16)
  - 不同：输出层从"一个输出头"改为 N 个"分类头 + 回归头"

输入和标签全部在 归一化空间（z-score）中训练，
评测时乘 fold_std 转回 mmHg —— 与原有训练流程完全一致。

输出: sbp_pred, dbp_pred  (归一化空间)
      sbp_logits, dbp_logits  (区间分类 logits)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStageBPModel(nn.Module):
    """
    两阶段 BP 预测模型。

    第一阶段：SBP/DBP 各自分区分类
    第二阶段：每个区间独立的回归头
    最终：分类概率 × 区间预测值 加权融合

    bins 和输出都在归一化空间。
    """
    def __init__(self, filters=(1,32,64,128), num_layers=2, hidden_dim=128,
                 drop_prob=0.2, bins_sbp_norm=None, bins_dbp_norm=None):
        super().__init__()

        # 默认 4 个区间（归一化空间）
        if bins_sbp_norm is None:
            bins_sbp_norm = [(-10, -0.8), (-0.8, 0.0), (0.0, 1.0), (1.0, 10)]
        if bins_dbp_norm is None:
            bins_dbp_norm = [(-10, -0.5), (-0.5, 0.3), (0.3, 1.2), (1.2, 10)]

        self.bins_sbp_norm = bins_sbp_norm
        self.bins_dbp_norm = bins_dbp_norm
        self.num_sbp_bins = len(bins_sbp_norm)
        self.num_dbp_bins = len(bins_dbp_norm)

        # ═══════ PPG 编码器（与 FusionModel26 完全相同） ═══════
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(filters[i], filters[i+1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(filters[i+1]),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.MaxPool1d(kernel_size=2, stride=2))
            for i in range(len(filters) - 1)
        ])

        self.bilstm = nn.LSTM(
            input_size=128, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=drop_prob, bidirectional=True
        )
        self.dropout = nn.Dropout(drop_prob)

        # 分头注意力
        self.sbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Tanh())
        self.dbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Tanh())
        self.softmax = nn.Softmax(dim=1)

        # 特征投影 (26→16)
        self.feature_projection = nn.Sequential(
            nn.Linear(26, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.1)
        )

        # ═══════ 两阶段输出头 ═══════
        encoder_dim = hidden_dim * 2 + 16  # 256 context + 16 proj_feat = 272

        # 第一阶段：分类器
        self.sbp_classifier = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2), nn.ReLU(),
            nn.Dropout(drop_prob * 0.5),
            nn.Linear(encoder_dim // 2, self.num_sbp_bins),
        )
        self.dbp_classifier = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2), nn.ReLU(),
            nn.Dropout(drop_prob * 0.5),
            nn.Linear(encoder_dim // 2, self.num_dbp_bins),
        )

        # 第二阶段：各区间独立回归头（输出归一化值）
        self.sbp_regressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2), nn.ReLU(),
                nn.Dropout(drop_prob * 0.5), nn.Linear(encoder_dim // 2, 1),
            ) for _ in range(self.num_sbp_bins)
        ])
        self.dbp_regressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2), nn.ReLU(),
                nn.Dropout(drop_prob * 0.5), nn.Linear(encoder_dim // 2, 1),
            ) for _ in range(self.num_dbp_bins)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _ppg_encoder(self, x):
        """与 FusionModel26 完全相同的 PPG 编码流水线"""
        for conv in self.convs:
            x = conv(x)
        lstm_out, _ = self.bilstm(x.permute(0, 2, 1))
        lstm_out = self.dropout(lstm_out)

        e_sbp = self.softmax(self.sbp_attn(lstm_out))
        c_sbp = torch.sum(e_sbp * lstm_out, dim=1)

        e_dbp = self.softmax(self.dbp_attn(lstm_out))
        c_dbp = torch.sum(e_dbp * lstm_out, dim=1)

        return c_sbp, c_dbp

    def forward(self, x, features):
        """
        x:        [B, 1, 1024]     — PPG 原始信号
        features: [B, 26]          — 26 维统计特征
        Returns:
            sbp_pred:    [B] — SBP 预测（归一化空间）
            dbp_pred:    [B] — DBP 预测（归一化空间）
            sbp_logits:  [B, num_bins]
            dbp_logits:  [B, num_bins]
        """
        c_sbp, c_dbp = self._ppg_encoder(x)
        proj_feat = self.feature_projection(features)

        feat_sbp = torch.cat([c_sbp, proj_feat], dim=1)
        feat_dbp = torch.cat([c_dbp, proj_feat], dim=1)  # 各自注意力头

        sbp_logits = self.sbp_classifier(feat_sbp)
        dbp_logits = self.dbp_classifier(feat_dbp)
        sbp_probs = F.softmax(sbp_logits, dim=1)
        dbp_probs = F.softmax(dbp_logits, dim=1)

        sbp_bin_outs = torch.cat([r(feat_sbp) for r in self.sbp_regressors], dim=1)
        dbp_bin_outs = torch.cat([r(feat_dbp) for r in self.dbp_regressors], dim=1)

        sbp_pred = (sbp_probs * sbp_bin_outs).sum(dim=1)
        dbp_pred = (dbp_probs * dbp_bin_outs).sum(dim=1)

        return sbp_pred, dbp_pred, sbp_logits, dbp_logits


class TwoStageLoss(nn.Module):
    """
    两阶段损失（全部在归一化空间计算）

    Loss = w_cls * CE(logits, bin_labels)
         + w_reg * MAE(pred, true)
    """
    def __init__(self, bins_sbp_norm, bins_dbp_norm, w_cls=0.3, w_reg=1.0):
        super().__init__()
        self.bins_sbp_norm = bins_sbp_norm
        self.bins_dbp_norm = bins_dbp_norm
        self.w_cls = w_cls
        self.w_reg = w_reg
        self.ce = nn.CrossEntropyLoss()

    def _assign_bin(self, values, bins):
        labels = torch.zeros_like(values, dtype=torch.long)
        for i, (lo, hi) in enumerate(bins):
            labels[(values >= lo) & (values < hi)] = i
        labels[values >= bins[-1][1]] = len(bins) - 1
        return labels

    def forward(self, sbp_pred, dbp_pred, sbp_logits, dbp_logits, sbp_true, dbp_true):
        sbp_labels = self._assign_bin(sbp_true, self.bins_sbp_norm).to(sbp_logits.device)
        dbp_labels = self._assign_bin(dbp_true, self.bins_dbp_norm).to(dbp_logits.device)

        loss_cls = self.ce(sbp_logits, sbp_labels) + self.ce(dbp_logits, dbp_labels)
        loss_reg = F.l1_loss(sbp_pred, sbp_true) + F.l1_loss(dbp_pred, dbp_true)

        return self.w_cls * loss_cls + self.w_reg * loss_reg
