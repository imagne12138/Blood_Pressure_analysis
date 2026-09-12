"""
Model_2 style + 26-dim features, concat at output head hidden layer (128-dim).
与 FusionModel26 的区别:
  - 原: features concat 到 256-dim context, 然后 Linear(282→128→1)
  - 现: context 先过 Linear(256→128), 再 concat 16-dim 投影特征, 然后 Linear(144→1)
"""
import torch
import torch.nn as nn


class FusionHeadConcat26(nn.Module):
    """
    时序路: Conv1D×3 → BiLSTM → 分头注意力 → 256-dim context
    特征投影: 26-dim → MLP(26→16)
    融合位置: 在输出头的 128-dim hidden layer, 而非 context 层

    SBP: c[256] → Linear(256→128) → ReLU → Dropout → 128h
          └── proj(16) ── concat ──→ Linear(144→1)
    DBP: 同理

    forward inputs:
        x:        [B, 1, 1024]
        features: [B, 26]
    returns:
        [B, 2]
    """

    def __init__(self, filters, num_layers, num_directions=2, hidden_dim=128,
                 drop_prob=0.2, attn_out_dim=1, proj_dim=16):
        super(FusionHeadConcat26, self).__init__()

        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.num_directions = num_directions
        self.proj_dim = proj_dim
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

        # ---- 特征投影 26→proj_dim ----
        self.feat_proj = nn.Sequential(
            nn.Linear(26, proj_dim), nn.BatchNorm1d(proj_dim),
            nn.ReLU(), nn.Dropout(0.1))

        # ---- SBP head: 先缩到128, 再融合特征后输出 ----
        self.sbp_hidden = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(drop_prob))
        self.sbp_final = nn.Linear(hidden_dim + proj_dim, 1)

        # ---- DBP head ----
        self.dbp_hidden = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(drop_prob))
        self.dbp_final = nn.Linear(hidden_dim + proj_dim, 1)

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

        # ---- 特征投影 ----
        proj = self.feat_proj(features)              # [B, 16]

        # ---- SBP: hidden → feat concat → final ----
        sbp_h = self.sbp_hidden(c_sbp)               # [B, 128]
        sbp_h = torch.cat([sbp_h, proj], dim=1)      # [B, 144]
        sbp_out = self.sbp_final(sbp_h)              # [B, 1]

        # ---- DBP ----
        dbp_h = self.dbp_hidden(c_dbp)               # [B, 128]
        dbp_h = torch.cat([dbp_h, proj], dim=1)      # [B, 144]
        dbp_out = self.dbp_final(dbp_h)              # [B, 1]

        return torch.cat([sbp_out, dbp_out], dim=1)  # [B, 2]
