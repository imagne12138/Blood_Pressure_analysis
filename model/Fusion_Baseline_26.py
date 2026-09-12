"""
BaselineModel + 26-dim statistical feature fusion.
共享注意力 + 共享输出头，26 维峰值/谷值统计特征.
"""
import torch
import torch.nn as nn


class FusionBaseline26(nn.Module):
    """
    时序路: Conv1D(1→32→64→128) → BiLSTM(hidden=128, bi) → 共享注意力 → 256-dim context
    特征路: 26-dim 峰值/谷值统计 → MLP(26→16)
    融合:   concat(256, 16) → 共享输出头 Linear(272→128→ReLU→Dropout→2)

    forward inputs:
        x:        [batch_size, 1, 1024]  — PPG 原始信号
        features: [batch_size, 26]       — 26 维统计特征
    returns:
        [batch_size, 2]                  — [SBP_pred, DBP_pred]
    """

    def __init__(self, filters, num_layers, num_directions=2, hidden_dim=128,
                 drop_prob=0.2, attn_out_dim=1, proj_dim=16):
        super(FusionBaseline26, self).__init__()

        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.num_directions = num_directions
        self.attn_out = attn_out_dim
        self.proj_dim = proj_dim

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

        # ---- 共享注意力 (SBP/DBP 共用一个时序关注模式) ----
        self.linear_attn = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_directions, self.attn_out),
            nn.Tanh()
        )

        # ---- 特征投影 MLP: 26 → proj_dim ----
        self.feature_projection = nn.Sequential(
            nn.Linear(26, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # ---- 共享输出头 ----
        fusion_dim = self.hidden_dim * self.num_directions + self.proj_dim  # 256 + 16 = 272
        self.linear_out = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.hidden_dim, 2)
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

        # ---- 共享注意力 ----
        e = self.linear_attn(lstm_out)         # [B, 128, 1]
        alpha = self.softmax(e)                # [B, 128, 1]
        c = torch.sum(alpha * lstm_out, dim=1) # [B, 256]

        # ---- 特征投影后融合 ----
        proj_feature = self.feature_projection(features)  # [B, 16]
        c = torch.cat([c, proj_feature], dim=1)            # [B, 272]

        # ---- 共享输出 ----
        output = self.linear_out(c)            # [B, 2]

        return output
