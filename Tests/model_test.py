from config.config import Config
from model.baseline_model import BaselineModel
from utils.data_helper import chain
from utils.create_data import LoadPPGDataset
import numpy as np
import os
import logging
import torch

cfg = Config()

logging.info("############初始化模型############")
model = BaselineModel(filters=cfg.filters, num_layers=cfg.num_layers)

print(model)
# [batch_size, feature_dim=1, window_size=1024]
x = torch.rand(4, 1, 1024)
print(x)
output = model(x)
print(output)
print(output.shape)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params}")

# 计算可训练参数量（排除被冻结的层）
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量: {trainable_params}")

# 工作目录为E:\Kaggle_projects\Blood_Pressure_analysis>
# 运行python -m Tests.data_loader_test
