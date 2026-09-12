import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    简单MLP，用作处理ppg特征向量，直接预测SBP和DBP
    效果一般不好，与时域频域融合再做输出
    """
    def __init__(self, in_feature_dim=169, hidden_1=128, hidden_2=64, out_dim=2):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_feature_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_2, out_dim)
        )

    def forward(self, x):
        return self.linear(x)

