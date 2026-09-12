"""
门控融合模型 — context 逐维度门控特征注入

门控机制:
  gate = σ(Linear(256 → proj_dim)(context))  每个特征维度的独立开关
  gated_feat = gate × proj_feat              门控后的投影特征
  fused = concat(context, gated_feat)         融合后进入输出头

当 gate ≈ 0 时，模型退化为纯 PPG 基线，防止特征带来噪声。
"""
import torch
import torch.nn as nn


class FusionGated(nn.Module):
    """
    时序路: Conv1D(1→32→64→128) → BiLSTM(h=128,bi) → 分头注意力 → 256-dim context
    特征路: features → MLP(feature_dim → proj_dim)
    门控:   gate = σ(Linear(256 → proj_dim)(context))
    融合:   context + gated_proj → 分头输出

    forward inputs:
        x:        [B, 1, 1024]         — PPG 原始信号
        features: [B, feature_dim]     — 26 维或 169 维特征
    returns:
        [B, 2]
    """

    def __init__(self, filters, num_layers, num_directions=2, hidden_dim=128,
                 drop_prob=0.2, attn_out_dim=1, feature_dim=26, proj_dim=16):
        super(FusionGated, self).__init__()

        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.num_directions = num_directions
        self.attn_out = attn_out_dim
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.attn_out = attn_out_dim

        # ---- CNN ----
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.filters[i], self.filters[i+1], 3, 1, 1),
                nn.BatchNorm1d(self.filters[i+1]),
                nn.ReLU(), nn.Dropout(drop_prob),
                nn.MaxPool1d(2, 2))
            for i in range(len(self.filters) - 1)
        ])

        # ---- BiLSTM ----
        self.bilstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True,
                              dropout=drop_prob, bidirectional=True)

        self.sbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, self.attn_out), nn.Tanh())
        self.dbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, self.attn_out), nn.Tanh())

        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.sbp_gate = nn.Linear(hidden_dim * 2, proj_dim)
        self.dbp_gate = nn.Linear(hidden_dim * 2, proj_dim)

        fusion_dim = hidden_dim * 2 + proj_dim  # 256 + proj_dim
        self.linear_sbp_out = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, 1))
        self.linear_dbp_out = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, features):
        # ---- CNN ----
        conv_out = x
        for conv in self.convs:
            conv_out = conv(conv_out)               # [B, 128, 128]

        # ---- BiLSTM ----
        lstm_in = conv_out.permute(0, 2, 1)          # [B, 128, 128]
        lstm_out, _ = self.bilstm(lstm_in)           # [B, 128, 256]
        lstm_out = self.dropout(lstm_out)

        # ---- 注意力 ----
        e_sbp = self.sbp_attn(lstm_out)
        c_sbp = torch.sum(self.softmax(e_sbp) * lstm_out, dim=1)   # [B, 256]
        e_dbp = self.dbp_attn(lstm_out)
        c_dbp = torch.sum(self.softmax(e_dbp) * lstm_out, dim=1)   # [B, 256]

        # ---- 特征投影 + 门控 ----
        proj = self.feat_proj(features)               # [B, proj_dim]
        sbp_gate = torch.sigmoid(self.sbp_gate(c_sbp))   # [B, proj_dim]
        dbp_gate = torch.sigmoid(self.dbp_gate(c_dbp))
        sbp_gated_proj = sbp_gate * proj
        dbp_gated_proj = dbp_gate * proj                      # [B, proj_dim]

        # ---- 融合 ----
        c_sbp = torch.cat([c_sbp, sbp_gated_proj], dim=1)  # [B, 256+proj_dim]
        c_dbp = torch.cat([c_dbp, dbp_gated_proj], dim=1)  # [B, 256+proj_dim]

        # ---- 输出 ----
        sbp_out = self.linear_sbp_out(c_sbp)
        dbp_out = self.linear_dbp_out(c_dbp)
        return torch.cat([sbp_out, dbp_out], dim=1)
