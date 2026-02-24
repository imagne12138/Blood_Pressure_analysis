from config.config import Config
from model.baseline_model import BaselineModel
from utils.data_helper import chain
from utils.create_data import LoadPPGDataset
import numpy as np
import os
import logging
import torch

cfg = Config()

logging.info("############载入划分好的数据集############")
data_loader = LoadPPGDataset(batch_size=cfg.batch_size)

fold_dir = os.path.join(cfg.base_dir, "cv_fold_0.npz")
train_iter, val_iter, _, _, _, _ = data_loader.load_train_val_data(data_dir=cfg.datadir, indices_dir=fold_dir)

print(type(train_iter))
print(len(train_iter))
print(len(val_iter))

# 工作目录为E:\Kaggle_projects\Blood_Pressure_analysis>
# 运行python -m Tests.data_loader_test

