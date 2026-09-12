"""Soft Ordinal Regression (CORAL) with fixed quantile bins.

Architecture:
  PPG backbone (CNNx3 + BiLSTM + Separate Attention) -> 256-dim context
  + features -> MLP(feature_dim->proj_dim) projection
  -> concat(256, proj_dim) -> CORAL head

CORAL head:
  g(x) = Linear(256+proj_dim -> 1)   shared ranker
  t_i = fixed threshold from quantile edges[1:-1]
  logit_i = g(x) - t_i
  P(y > i) = sigmoid(logit_i)
  P(y = i) = P(y > i-1) - P(y > i)
  y_pred = sum(P(y=i) * center_i)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftOrdinalFusion(nn.Module):
    def __init__(self, filters, num_layers, num_directions=2, hidden_dim=128,
                 drop_prob=0.2, attn_out_dim=1, feature_dim=26, proj_dim=16,
                 num_bins=8, sbp_edges=None, dbp_edges=None):
        super().__init__()
        self.num_bins = num_bins
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # ---- PPG backbone ----
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(filters[i], filters[i+1], 3, 1, 1),
                nn.BatchNorm1d(filters[i+1]),
                nn.ReLU(), nn.Dropout(drop_prob),
                nn.MaxPool1d(2, 2))
            for i in range(len(filters) - 1)
        ])
        self.bilstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True,
                              dropout=drop_prob, bidirectional=True)
        self.sbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, attn_out_dim), nn.Tanh())
        self.dbp_attn = nn.Sequential(nn.Linear(hidden_dim*2, attn_out_dim), nn.Tanh())
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(), nn.Dropout(0.1))

        # ---- Fixed quantile bins ----
        if sbp_edges is None:
            sbp_edges = torch.linspace(-3.0, 4.0, num_bins + 1)
        if dbp_edges is None:
            dbp_edges = torch.linspace(-3.0, 4.0, num_bins + 1)
        sbp_centers = (sbp_edges[:-1] + sbp_edges[1:]) / 2
        dbp_centers = (dbp_edges[:-1] + dbp_edges[1:]) / 2
        self.register_buffer("sbp_edges", sbp_edges)
        self.register_buffer("dbp_edges", dbp_edges)
        self.register_buffer("sbp_centers", sbp_centers)
        self.register_buffer("dbp_centers", dbp_centers)

        # ---- CORAL ranker ----
        encoder_dim = hidden_dim * 2 + proj_dim  # 256 + proj_dim
        self.sbp_ranker = nn.Linear(encoder_dim, 1)
        self.dbp_ranker = nn.Linear(encoder_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_ppg(self, x):
        for conv in self.convs:
            x = conv(x)
        lstm_out, _ = self.bilstm(x.permute(0, 2, 1))
        lstm_out = self.dropout(lstm_out)
        e_sbp = self.sbp_attn(lstm_out)
        c_sbp = torch.sum(self.softmax(e_sbp) * lstm_out, dim=1)
        e_dbp = self.dbp_attn(lstm_out)
        c_dbp = torch.sum(self.softmax(e_dbp) * lstm_out, dim=1)
        return c_sbp, c_dbp

    def forward(self, x, features):
        c_sbp, c_dbp = self._encode_ppg(x)
        proj = self.feature_projection(features)
        c_sbp = torch.cat([c_sbp, proj], dim=1)
        c_dbp = torch.cat([c_dbp, proj], dim=1)

        # CORAL ranker
        g_sbp = self.sbp_ranker(c_sbp)  # [B, 1]
        g_dbp = self.dbp_ranker(c_dbp)

        # Fixed thresholds (interior edges, K-1)
        sbp_t = self.sbp_edges[1:-1]  # [K-1]
        dbp_t = self.dbp_edges[1:-1]

        # Cumulative logits: logit_i = g(x) - t_i
        # t_i < t_{i+1} => logit_i > logit_{i+1} => P(y>i) >= P(y>i+1)
        sbp_cum_logits = g_sbp - sbp_t.unsqueeze(0)  # [B, K-1]
        dbp_cum_logits = g_dbp - dbp_t.unsqueeze(0)

        # P(y > i) = sigmoid(g(x) - t_i)
        sbp_gt = torch.sigmoid(sbp_cum_logits)
        dbp_gt = torch.sigmoid(dbp_cum_logits)

        # Convert to bin probabilities: P(y=i) = P(y>i-1) - P(y>i)
        # with P(y>-1) = 1, P(y>K-1) = 0
        one = torch.ones_like(sbp_gt[:, :1])
        zero = torch.zeros_like(sbp_gt[:, :1])

        sbp_probs = torch.cat([one, sbp_gt], dim=1) - torch.cat([sbp_gt, zero], dim=1)
        dbp_probs = torch.cat([one, dbp_gt], dim=1) - torch.cat([dbp_gt, zero], dim=1)

        # Numerical stability
        sbp_probs = sbp_probs.clamp(min=0)
        dbp_probs = dbp_probs.clamp(min=0)
        sbp_probs = sbp_probs / sbp_probs.sum(dim=1, keepdim=True)
        dbp_probs = dbp_probs / dbp_probs.sum(dim=1, keepdim=True)

        # Expected value: sum(P(y=i) * center_i)
        sbp_pred = (sbp_probs * self.sbp_centers.unsqueeze(0)).sum(dim=1)
        dbp_pred = (dbp_probs * self.dbp_centers.unsqueeze(0)).sum(dim=1)

        return torch.stack([sbp_pred, dbp_pred], dim=1)

    def get_logits(self, x, features):
        """Return ranker values for loss computation."""
        c_sbp, c_dbp = self._encode_ppg(x)
        proj = self.feature_projection(features)
        c_sbp = torch.cat([c_sbp, proj], dim=1)
        c_dbp = torch.cat([c_dbp, proj], dim=1)
        g_sbp = self.sbp_ranker(c_sbp)
        g_dbp = self.dbp_ranker(c_dbp)
        return g_sbp, g_dbp


class OrdinalLoss(nn.Module):
    """CORAL + MAE hybrid loss with fixed threshold labels."""
    def __init__(self, sbp_edges, dbp_edges, w_ord=1.0, w_reg=0.2):
        super().__init__()
        self.w_ord = w_ord
        self.w_reg = w_reg
        self.register_buffer("sbp_edges", sbp_edges)
        self.register_buffer("dbp_edges", dbp_edges)

    def forward(self, pred, target, g_sbp, g_dbp):
        """
        pred:   [B, 2] model output (normalized)
        target: [B, 2] ground truth (normalized)
        g_sbp, g_dbp: [B, 1] ranker values
        """
        sbp_true, dbp_true = target[:, 0], target[:, 1]

        # Fixed threshold labels: I(true > threshold_i)
        sbp_t = self.sbp_edges[1:-1]   # [K-1]
        dbp_t = self.dbp_edges[1:-1]

        sbp_labels = (sbp_true.unsqueeze(1) > sbp_t.unsqueeze(0)).float()
        dbp_labels = (dbp_true.unsqueeze(1) > dbp_t.unsqueeze(0)).float()

        # Cumulative logits (same as forward())
        sbp_logits = g_sbp - sbp_t.unsqueeze(0)
        dbp_logits = g_dbp - dbp_t.unsqueeze(0)

        # BCE on cumulative logits
        ord_loss = F.binary_cross_entropy_with_logits(sbp_logits, sbp_labels) + \
                   F.binary_cross_entropy_with_logits(dbp_logits, dbp_labels)

        # MAE regression loss
        reg_loss = F.l1_loss(pred[:, 0], target[:, 0]) + \
                   F.l1_loss(pred[:, 1], target[:, 1])

        return self.w_ord * ord_loss + self.w_reg * reg_loss
