"""
PPG CNN-BiLSTM-Attention + 26-dim statistical feature fusion model.
Based on Model_2_Head with separate SBP/DBP attention heads.
Fuses 26-dim peak/valley statistical features with the 256-dim context vector.
"""
import torch
import torch.nn as nn


class FusionModel26(nn.Module):
    """
    时序路: Conv1D(1→32→64→128) → BiLSTM(hidden=128, bi) → 分头注意力 → 256-dim context
    特征路: 26-dim 统计特征 (峰值/谷值/间隔统计量)
    融合:   concat(256, 26) → 各自 SBP/DBP 输出头 Linear(282→128→ReLU→Dropout→1)

    forward inputs:
        x:        [batch_size, 1, 1024]    — PPG 原始信号
        features: [batch_size, 26]         — 26 维统计特征
    returns:
        [batch_size, 2]                    — [SBP_pred, DBP_pred]
    """

    def __init__(self, filters, num_layers, num_directions=2, hidden_dim=128,
                 drop_prob=0.2, attn_out_dim=1, feature_dim=26, out_dim=2):
        super(FusionModel26, self).__init__()

        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.num_directions = num_directions
        self.attn_out = attn_out_dim
        self.output = out_dim
        self.feature_dim = feature_dim

        # ---- 1D-CNN 编码器 ----
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.filters[i],
                          out_channels=self.filters[i + 1],
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(num_features=self.filters[i + 1]),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.MaxPool1d(kernel_size=2, stride=2))
            for i in range(len(self.filters) - 1)
        ])

        # ---- BiLSTM ----
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=drop_prob,
            bidirectional=True
        )

        # ---- 分头注意力 (SBP / DBP 各自关注不同时序模式) ----
        self.sbp_attn = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_directions, self.attn_out),
            nn.Tanh()
        )
        self.dbp_attn = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_directions, self.attn_out),
            nn.Tanh()
        )

        # ---- 融合输出头: 256 (context) + 26 (features) = 282 ----
        fusion_dim = self.hidden_dim * self.num_directions + self.feature_dim

        self.linear_sbp_out = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.hidden_dim, 1)
        )
        self.linear_dbp_out = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.hidden_dim, 1)
        )

        self.feature_projection = nn.Sequential(nn.Linear(26, 16),
                                                nn.BatchNorm1d(16),
                                                nn.ReLU(),
                                                nn.Dropout(0.1),
                                                )

        # ---- 初始化权重 ----
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
        """
        x:        [B, 1, 1024]
        features: [B, 26]
        """
        # ---- CNN ----
        conv_out = x
        for conv in self.convs:
            conv_out = conv(conv_out)          # [B, 128, 128]

        # ---- BiLSTM ----
        lstm_in = conv_out.permute(0, 2, 1)    # [B, 128, 128]
        lstm_out, _ = self.bilstm(lstm_in)     # [B, 128, 256]
        lstm_out = self.dropout(lstm_out)

        # ---- SBP 注意力加权 ----
        e_sbp = self.sbp_attn(lstm_out)        # [B, 128, 1]
        alpha_sbp = self.softmax(e_sbp)        # [B, 128, 1]
        c_sbp = torch.sum(alpha_sbp * lstm_out, dim=1)  # [B, 256]

        # ---- DBP 注意力加权 ----
        e_dbp = self.dbp_attn(lstm_out)        # [B, 128, 1]
        alpha_dbp = self.softmax(e_dbp)        # [B, 128, 1]
        c_dbp = torch.sum(alpha_dbp * lstm_out, dim=1)  # [B, 256]

        # ---- 简单特征投影后concat
        proj_feature = self.feature_projection(features)

        # ---- 融合: context + features ----
        c_sbp = torch.cat([c_sbp, features], dim=1)  # [B, 282]
        c_dbp = torch.cat([c_dbp, features], dim=1)  # [B, 282]

        # ---- 融合：context + proj_features ----        
        c_sbp = torch.cat([c_sbp, proj_feature], dim=1)  # [B, 272]
        c_dbp = torch.cat([c_dbp, proj_feature], dim=1)  # [B, 272]

        # ---- 输出头 ----
        sbp_out = self.linear_sbp_out(c_sbp)   # [B, 1]
        dbp_out = self.linear_dbp_out(c_dbp)   # [B, 1]

        return torch.cat([sbp_out, dbp_out], dim=1)  # [B, 2]
