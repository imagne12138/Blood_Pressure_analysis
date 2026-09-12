"""
Model_2 style + SBP_opt (17-dim) / DBP_opt (12-dim) 各自独立投影融合。
每个 BP 用论文筛选的最优特征子集，各自 MLP 投影后 concat 到 context。
"""
import torch
import torch.nn as nn


class FusionModelOpt(nn.Module):
    """
    SBP 路: Conv1D×3 → BiLSTM → SBP 注意力 → 256-dim context
              └── sbp_opt(17) → MLP(17→8) → concat → Linear(264→128→ReLU→Dropout→1)
    DBP 路: Conv1D×3 → BiLSTM → DBP 注意力 → 256-dim context
              └── dbp_opt(12) → MLP(12→6) → concat → Linear(262→128→ReLU→Dropout→1)

    forward inputs:
        x:        [B, 1, 1024]
        sbp_feats: [B, 17]
        dbp_feats: [B, 12]
    returns:
        [B, 2]
    """

    def __init__(self, filters, num_layers, num_directions=2, hidden_dim=128,
                 drop_prob=0.2, attn_out_dim=1):
        super(FusionModelOpt, self).__init__()

        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.num_directions = num_directions
        self.attn_out = attn_out_dim

        # ---- 1D-CNN ----
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

        # ---- 分头注意力 ----
        self.sbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, self.attn_out), nn.Tanh())
        self.dbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, self.attn_out), nn.Tanh())

        # ---- SBP opt 投影 17→8 ----
        self.sbp_proj = nn.Sequential(
            nn.Linear(17, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.1))
        self.sbp_out = nn.Sequential(
            nn.Linear(hidden_dim*2 + 8, hidden_dim),
            nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, 1))

        # ---- DBP opt 投影 12→6 ----
        self.dbp_proj = nn.Sequential(
            nn.Linear(12, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Dropout(0.1))
        self.dbp_out = nn.Sequential(
            nn.Linear(hidden_dim*2 + 6, hidden_dim),
            nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, sbp_feats, dbp_feats):
        # ---- CNN ----
        conv_out = x
        for conv in self.convs:
            conv_out = conv(conv_out)               # [B, 128, 128]

        # ---- BiLSTM ----
        lstm_in = conv_out.permute(0, 2, 1)          # [B, 128, 128]
        lstm_out, _ = self.bilstm(lstm_in)           # [B, 128, 256]
        lstm_out = self.dropout(lstm_out)

        # ---- SBP 注意力 + 特征融合 ----
        e_sbp = self.sbp_attn(lstm_out)              # [B, 128, 1]
        c_sbp = torch.sum(self.softmax(e_sbp) * lstm_out, dim=1)  # [B, 256]
        sbp_proj = self.sbp_proj(sbp_feats)          # [B, 8]
        c_sbp = torch.cat([c_sbp, sbp_proj], dim=1)  # [B, 264]
        sbp_out = self.sbp_out(c_sbp)                # [B, 1]

        # ---- DBP 注意力 + 特征融合 ----
        e_dbp = self.dbp_attn(lstm_out)              # [B, 128, 1]
        c_dbp = torch.sum(self.softmax(e_dbp) * lstm_out, dim=1)  # [B, 256]
        dbp_proj = self.dbp_proj(dbp_feats)          # [B, 6]
        c_dbp = torch.cat([c_dbp, dbp_proj], dim=1)  # [B, 262]
        dbp_out = self.dbp_out(c_dbp)                # [B, 1]

        return torch.cat([sbp_out, dbp_out], dim=1)  # [B, 2]
