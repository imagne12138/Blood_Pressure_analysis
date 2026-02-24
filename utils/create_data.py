import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py

class PPGDataset(Dataset):

    def __init__(self, data_dir, indices_dir, train=True, sbp_mean=None, sbp_std=None, dbp_mean=None, dbp_std=None):
        """
        indices读取npz文件，每一折处理一次，train和val分开读取
        训练时每一折数据分开读取
        输出x, y, 形状为[1, 1024], [2]
        """
        self.data_path = data_dir
        self.file = None # 防止频繁读取影响速度

        self.sbp_mean = sbp_mean
        self.sbp_std = sbp_std
        self.dbp_mean = dbp_mean
        self.dbp_std = dbp_std
        
        fold_indices = np.load(indices_dir)

        if train == True:
            self.indices = fold_indices["train_idx"] # numpy数组, 存所有训练或验证的窗口索引, (413840,)
        else:
            self.indices = fold_indices["val_idx"]
        # f"cv_fold_{}.npz", 分train_idx和val_idx
    
    def __getitem__(self, index):
        # num_workers = 0
        if self.file is None:
            self.file = h5py.File(self.data_path, "r")

        idx = self.indices[index] # 在训练或验证的窗口索引里取一个
        ppg = self.file["ppg"][idx] # (1024,)
        sbp = self.file["sbp"][idx] # 一个数字
        dbp = self.file["dbp"][idx] # 一个数字

        if self.sbp_mean is not None:
            sbp = (sbp - self.sbp_mean) / self.sbp_std
            dbp = (dbp - self.dbp_mean) / self.dbp_std

        x = torch.tensor(ppg, dtype=torch.float32) # (1024)
        x = x.unsqueeze(0) # (1, 1024)
        y = torch.tensor([sbp, dbp], dtype=torch.float32) # (2)
        return x, y
        
    def __len__(self):
        return len(self.indices)


class LoadPPGDataset:
    """
    按照划分的train_val数据分别加载训练和验证dataloader，
    形状为[batch_size, 1, 1024], [batch_size, 2]
    """
    
    def __init__(self, batch_size, is_sample_shuffle=True):
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    def load_train_val_data(self, data_dir, indices_dir):
        """
        data_dir: 切好窗的数据
        indices_dir: 分好的一折数据索引
        """
        train_dataset = PPGDataset(data_dir, 
                                   indices_dir,
                                   train=True)
        
        sbp_mean, sbp_std, dbp_mean, dbp_std = self.bp_statistical_values(train_dataset)
        
        train_dataset = PPGDataset(data_dir, 
                                   indices_dir, 
                                   train=True, 
                                   sbp_mean=sbp_mean, 
                                   sbp_std=sbp_std, 
                                   dbp_mean=dbp_mean, 
                                   dbp_std=dbp_std)

        val_dataset = PPGDataset(data_dir, 
                                 indices_dir, 
                                 train=False,
                                 sbp_mean=sbp_mean, 
                                 sbp_std=sbp_std, 
                                 dbp_mean=dbp_mean, 
                                 dbp_std=dbp_std)

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle) # (batch_size, 1, 1024), 送入1D卷积层
        val_iter = DataLoader(val_dataset, batch_size=self.batch_size,
                               shuffle=False)

        return train_iter, val_iter, sbp_mean, sbp_std, dbp_mean, dbp_std

    def bp_statistical_values(self, dataset):
        """
        给sbp和dbp标签分别做标准化，防过拟合
        """

        sbp = []
        dbp = []

        file = h5py.File(dataset.data_path, "r")
        indices = dataset.indices

        sbp = file["sbp"][indices]
        dbp = file["dbp"][indices]

        return (sbp.mean(), sbp.std(), dbp.mean(), dbp.std())
