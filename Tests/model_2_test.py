from config.config import Config
from model.model_2 import Model_2_Head
from utils.data_helper import chain
from utils.create_data import LoadPPGDataset
import numpy as np
import os
import logging
import torch

cfg = Config()
model = Model_2_Head(filters=cfg.filters, num_layers=cfg.num_layers)

print(model)
# [batch_size, feature_dim=1, window_size=1024]
x = torch.rand(4, 1, 1024)
print(x)
# sbp_out, dbp_out = model(x)
# print(sbp_out, dbp_out)
# print(sbp_out.shape)
out = model(x)
print(out)
print(out.shape)


total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params}")

# 计算可训练参数量（排除被冻结的层）
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量: {trainable_params}")



