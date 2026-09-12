import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    简单MLP，用作处理ppg特征向量，直接预测SBP和DBP
    """
    def __init__(self, in_feature_dim=26, hidden_dim=13, out_dim=2):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.linear(x)

