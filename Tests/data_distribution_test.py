from copy import deepcopy
from config.config import Config
from model.baseline_model import BaselineModel
from utils.data_helper import chain
from utils.create_data import LoadPPGDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import os
import logging

cfg = Config()

data_loader = LoadPPGDataset(batch_size=cfg.batch_size)

fold_dir = os.path.join(cfg.base_dir, "cv_fold_0.npz")
train_iter, val_iter = data_loader.load_train_val_data(data_dir=cfg.datadir, indices_dir=fold_dir)

print(len(train_iter))
print(len(val_iter))

train_sbp = []
for x, y in tqdm(train_iter, desc="Processing train data", colour="cyan"):
    train_sbp.extend(y[:,0].numpy())

val_sbp = []
for x, y in tqdm(val_iter, desc="Processing val data", colour="cyan"):
    val_sbp.extend(y[:,0].numpy())

print(np.mean(train_sbp), np.std(train_sbp))
print(np.mean(val_sbp), np.std(val_sbp))

plt.hist(train_sbp, bins=50, alpha=0.5, label="Train")
plt.hist(val_sbp, bins=50, alpha=0.5, label="Val")
plt.legend()
plt.savefig(r'E:\Kaggle_projects\Blood_Pressure_analysis\imgs\Comparision_of_train_val_sbp(unshuffled_fold0).jpg')
plt.show()
