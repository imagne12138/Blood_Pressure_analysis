"""
SBP_opt (17-dim) / DBP_opt (12-dim) 单独 MLP 预测。
每个 BP 只用论文筛选出的最优特征子集，各自过独立 MLP。
对比全 169 维 MLP，验证最优子集是否更高效。
"""
import torch
import torch.nn as nn


class MLP_Opt(nn.Module):
    """
    SBP: 17-dim sbp_opt → Linear(17→64→ReLU→Dropout) → Linear(64→32→ReLU→Dropout) → Linear(32→1)
    DBP: 12-dim dbp_opt → Linear(12→32→ReLU→Dropout) → Linear(32→16→ReLU→Dropout) → Linear(16→1)

    forward inputs:
        sbp_feats: [B, 17]
        dbp_feats: [B, 12]
    returns:
        [B, 2] — [SBP_pred, DBP_pred]
    """

    def __init__(self, sbp_hidden=[64, 32], dbp_hidden=[32, 16], drop_prob=0.2):
        super(MLP_Opt, self).__init__()

        # ---- SBP 分支 ----
        sbp_layers = []
        in_dim = 17
        for h in sbp_hidden:
            sbp_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(drop_prob)])
            in_dim = h
        sbp_layers.append(nn.Linear(in_dim, 1))
        self.sbp_mlp = nn.Sequential(*sbp_layers)

        # ---- DBP 分支 ----
        dbp_layers = []
        in_dim = 12
        for h in dbp_hidden:
            dbp_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(drop_prob)])
            in_dim = h
        dbp_layers.append(nn.Linear(in_dim, 1))
        self.dbp_mlp = nn.Sequential(*dbp_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sbp_feats, dbp_feats):
        sbp_out = self.sbp_mlp(sbp_feats)  # [B, 1]
        dbp_out = self.dbp_mlp(dbp_feats)  # [B, 1]
        return torch.cat([sbp_out, dbp_out], dim=1)  # [B, 2]
