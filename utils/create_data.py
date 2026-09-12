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
        self.file = None

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

    def _init_file(self):
        if self.file is None:
            self.file = h5py.File(self.data_path, "r")
            self.ppg = self.file["ppg"]
            self.sbp = self.file["sbp"]
            self.dbp = self.file["dbp"]
    
    def __getitem__(self, index):
        self._init_file()
        idx = self.indices[index] # 在训练或验证的窗口索引里取一个
        
        ppg = self.ppg[idx] # (1024,)
        sbp = self.sbp[idx] # 一个数字
        dbp = self.dbp[idx]

        if self.sbp_mean is not None:
            sbp = (sbp - self.sbp_mean) / self.sbp_std
            dbp = (dbp - self.dbp_mean) / self.dbp_std

        # x = torch.tensor(ppg, dtype=torch.float32) # (1024)
        # x = x.unsqueeze(0) # (1, 1024)
        x = torch.from_numpy(ppg).unsqueeze(0).float() # 防止每次copy memory
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
                                shuffle=self.is_sample_shuffle, 
                                num_workers=4,
                                pin_memory=True,
                                persistent_workers=True) # (batch_size, 1, 1024), 送入1D卷积层
        val_iter = DataLoader(val_dataset, batch_size=self.batch_size,
                               shuffle=False,
                               num_workers=4,
                               pin_memory=True,
                               persistent_workers=True)

        return train_iter, val_iter, sbp_mean, sbp_std, dbp_mean, dbp_std

    def bp_statistical_values(self, dataset):
        """
        给sbp和dbp标签分别做标准化，防过拟合
        """

        sbp = []
        dbp = []

        with h5py.File(dataset.data_path, "r") as file: # 关闭文件
            indices = dataset.indices

            sbp = file["sbp"][indices]
            dbp = file["dbp"][indices]

        return (sbp.mean(), sbp.std(), dbp.mean(), dbp.std())


class PPGFeatureDataset(Dataset):
    """
    以ppg_features.h5中的特征向量作为输入，
    segmented_records.h5中的sbp/dbp作为标签
    输出x, y, 形状为[26], [2]
    """

    def __init__(self, feature_path, label_path, indices_dir, train=True,
                 sbp_mean=None, sbp_std=None, dbp_mean=None, dbp_std=None,
                 feat_mean=None, feat_std=None):
        self.feature_path = feature_path
        self.label_path = label_path
        self.feature_file = None
        self.label_file = None

        self.sbp_mean = sbp_mean
        self.sbp_std = sbp_std
        self.dbp_mean = dbp_mean
        self.dbp_std = dbp_std
        self.feat_mean = feat_mean
        self.feat_std = feat_std

        fold_indices = np.load(indices_dir)
        if train:
            self.indices = fold_indices["train_idx"]
        else:
            self.indices = fold_indices["val_idx"]

    def _init_file(self):
        if self.feature_file is None:
            self.feature_file = h5py.File(self.feature_path, "r")
            self.ppg_features = self.feature_file["ppg_features"]
            self.label_file = h5py.File(self.label_path, "r")
            self.sbp = self.label_file["sbp"]
            self.dbp = self.label_file["dbp"]

    def __getitem__(self, index):
        self._init_file()
        idx = self.indices[index]

        ppg_feat = self.ppg_features[idx]  # (26,)
        if self.feat_mean is not None:
            ppg_feat = (ppg_feat - self.feat_mean) / (self.feat_std + 1e-8)

        sbp = self.sbp[idx]
        dbp = self.dbp[idx]
        if self.sbp_mean is not None:
            sbp = (sbp - self.sbp_mean) / self.sbp_std
            dbp = (dbp - self.dbp_mean) / self.dbp_std

        x = torch.from_numpy(ppg_feat).float()
        y = torch.tensor([sbp, dbp], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.indices)


class LoadPPGFeatureDataset:
    """
    加载PPG特征数据，送入MLP模型
    形状为[batch_size, 26], [batch_size, 2]
    """

    def __init__(self, batch_size, is_sample_shuffle=True):
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    def load_train_val_data(self, feature_path, label_path, indices_dir):
        train_dataset = PPGFeatureDataset(feature_path, label_path,
                                          indices_dir, train=True)

        # 从训练集计算特征归一化参数
        feat_mean, feat_std = self.feat_statistical_values(train_dataset, feature_path)

        # 从训练集计算血压标签归一化参数
        sbp_mean, sbp_std, dbp_mean, dbp_std = self.bp_statistical_values(train_dataset)

        train_dataset = PPGFeatureDataset(feature_path, label_path,
                                          indices_dir, train=True,
                                          sbp_mean=sbp_mean, sbp_std=sbp_std,
                                          dbp_mean=dbp_mean, dbp_std=dbp_std,
                                          feat_mean=feat_mean, feat_std=feat_std)

        val_dataset = PPGFeatureDataset(feature_path, label_path,
                                        indices_dir, train=False,
                                        sbp_mean=sbp_mean, sbp_std=sbp_std,
                                        dbp_mean=dbp_mean, dbp_std=dbp_std,
                                        feat_mean=feat_mean, feat_std=feat_std)

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                num_workers=4, pin_memory=True,
                                persistent_workers=True)
        val_iter = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True,
                              persistent_workers=True)

        return train_iter, val_iter, sbp_mean, sbp_std, dbp_mean, dbp_std

    def bp_statistical_values(self, dataset):
        with h5py.File(dataset.label_path, "r") as file:
            indices = dataset.indices
            sbp = file["sbp"][indices]
            dbp = file["dbp"][indices]
        return sbp.mean(), sbp.std(), dbp.mean(), dbp.std()

    def feat_statistical_values(self, dataset, feature_path):
        with h5py.File(feature_path, "r") as file:
            indices = dataset.indices
            feats = file["ppg_features"][indices]
        return feats.mean(axis=0), feats.std(axis=0)


# =============================================================================
# Liu2023 Feature Dataset (169-dim / SBP-opt / DBP-opt)
# =============================================================================

class Liu2023FeatureDataset(Dataset):
    """
    Load Liu2023 features pre-extracted by liu2023_features.extract_all_features_to_h5().

    Supports three feature modes:
      - 'full'   : all 169 features
      - 'sbp_opt': 17 SBP-optimal features  (from SBP_OPTIMAL_FEATURES)
      - 'dbp_opt': 13 DBP-optimal features  (from DBP_OPTIMAL_FEATURES)

    Output x, y shapes: [feature_dim], [2]
    """

    def __init__(self, feature_path, indices_dir, train=True,
                 feature_set='full',
                 sbp_mean=None, sbp_std=None, dbp_mean=None, dbp_std=None,
                 feat_mean=None, feat_std=None):
        self.feature_path = feature_path
        self.feature_file = None
        self.feature_set = feature_set

        self.sbp_mean = sbp_mean
        self.sbp_std = sbp_std
        self.dbp_mean = dbp_mean
        self.dbp_std = dbp_std
        self.feat_mean = feat_mean
        self.feat_std = feat_std

        fold_indices = np.load(indices_dir)
        if train:
            self.indices = fold_indices["train_idx"]
        else:
            self.indices = fold_indices["val_idx"]

    def _init_file(self):
        if self.feature_file is None:
            self.feature_file = h5py.File(self.feature_path, "r")
            # Choose feature dataset based on feature_set
            if self.feature_set == 'sbp_opt':
                self.feats_ds = self.feature_file["features_sbp"]
            elif self.feature_set == 'dbp_opt':
                self.feats_ds = self.feature_file["features_dbp"]
            else:
                self.feats_ds = self.feature_file["features"]
            self.sbp = self.feature_file["sbp"]
            self.dbp = self.feature_file["dbp"]

    def __getitem__(self, index):
        self._init_file()
        idx = self.indices[index]

        ppg_feat = self.feats_ds[idx].astype(np.float32)
        if self.feat_mean is not None:
            ppg_feat = (ppg_feat - self.feat_mean) / (self.feat_std + 1e-8)

        sbp = self.sbp[idx]
        dbp = self.dbp[idx]
        if self.sbp_mean is not None:
            sbp = (sbp - self.sbp_mean) / self.sbp_std
            dbp = (dbp - self.dbp_mean) / self.dbp_std

        x = torch.from_numpy(ppg_feat).float()
        y = torch.tensor([sbp, dbp], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.indices)


class LoadLiu2023FeatureDataset:
    """
    Dataloader wrapper for Liu2023FeatureDataset.
    Output shapes: [batch_size, feature_dim], [batch_size, 2]

    feature_set: 'full' (169), 'sbp_opt' (17), or 'dbp_opt' (13)
    """

    def __init__(self, batch_size, is_sample_shuffle=True):
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    def load_train_val_data(self, feature_path, indices_dir, feature_set='full'):
        train_dataset = Liu2023FeatureDataset(
            feature_path, indices_dir, train=True,
            feature_set=feature_set
        )

        feat_mean, feat_std = self.feat_statistical_values(train_dataset, feature_path, feature_set)
        sbp_mean, sbp_std, dbp_mean, dbp_std = self.bp_statistical_values(train_dataset)

        train_dataset = Liu2023FeatureDataset(
            feature_path, indices_dir, train=True,
            feature_set=feature_set,
            sbp_mean=sbp_mean, sbp_std=sbp_std,
            dbp_mean=dbp_mean, dbp_std=dbp_std,
            feat_mean=feat_mean, feat_std=feat_std
        )

        val_dataset = Liu2023FeatureDataset(
            feature_path, indices_dir, train=False,
            feature_set=feature_set,
            sbp_mean=sbp_mean, sbp_std=sbp_std,
            dbp_mean=dbp_mean, dbp_std=dbp_std,
            feat_mean=feat_mean, feat_std=feat_std
        )

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                num_workers=4, pin_memory=True,
                                persistent_workers=True)
        val_iter = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True,
                              persistent_workers=True)

        return train_iter, val_iter, sbp_mean, sbp_std, dbp_mean, dbp_std

    def bp_statistical_values(self, dataset):
        with h5py.File(dataset.feature_path, "r") as file:
            indices = dataset.indices
            sbp = file["sbp"][indices]
            dbp = file["dbp"][indices]
        return sbp.mean(), sbp.std(), dbp.mean(), dbp.std()

    def feat_statistical_values(self, dataset, feature_path, feature_set='full'):
        with h5py.File(feature_path, "r") as file:
            indices = dataset.indices
            if feature_set == 'sbp_opt':
                feats = file["features_sbp"][indices]
            elif feature_set == 'dbp_opt':
                feats = file["features_dbp"][indices]
            else:
                feats = file["features"][indices]
        return feats.mean(axis=0), feats.std(axis=0)


# =============================================================================
# Fusion Dataset: PPG (1×1024) + 26-dim statistical features
# =============================================================================

class Fusion26Dataset(Dataset):
    """
    Load both PPG signal and 26-dim statistical features for the fusion model.

    PPG 和 26-dim 特征来自不同 h5 文件 (segmented_records.h5 / ppg_features.h5),
    但窗口索引完全对齐 (相同的 cv_fold_*.npz 切分).

    __getitem__ returns ((ppg_tensor, feat_tensor), label_tensor)
        ppg_tensor:  [1, 1024]
        feat_tensor: [26]
        label_tensor: [2] — (SBP, DBP)
    """

    def __init__(self, ppg_path, feat_path, indices_dir, train=True,
                 sbp_mean=None, sbp_std=None, dbp_mean=None, dbp_std=None,
                 feat_mean=None, feat_std=None):
        self.ppg_path = ppg_path
        self.feat_path = feat_path
        self.ppg_file = None
        self.feat_file = None

        self.sbp_mean = sbp_mean
        self.sbp_std = sbp_std
        self.dbp_mean = dbp_mean
        self.dbp_std = dbp_std
        self.feat_mean = feat_mean
        self.feat_std = feat_std

        fold_indices = np.load(indices_dir)
        if train:
            self.indices = fold_indices["train_idx"]
        else:
            self.indices = fold_indices["val_idx"]

    def _init_file(self):
        if self.ppg_file is None:
            self.ppg_file = h5py.File(self.ppg_path, "r")
            self.ppg = self.ppg_file["ppg"]
            self.sbp = self.ppg_file["sbp"]
            self.dbp = self.ppg_file["dbp"]

            self.feat_file = h5py.File(self.feat_path, "r")
            self.feats = self.feat_file["ppg_features"]

    def __getitem__(self, index):
        self._init_file()
        idx = self.indices[index]

        # PPG signal
        ppg = self.ppg[idx]  # (1024,)
        x_ppg = torch.from_numpy(ppg).unsqueeze(0).float()  # (1, 1024)

        # 26-dim features
        feats = self.feats[idx].astype(np.float32)  # (26,)
        if self.feat_mean is not None:
            feats = (feats - self.feat_mean) / (self.feat_std + 1e-8)
        x_feat = torch.from_numpy(feats).float()  # (26,)

        # Labels
        sbp = self.sbp[idx]
        dbp = self.dbp[idx]
        if self.sbp_mean is not None:
            sbp = (sbp - self.sbp_mean) / self.sbp_std
            dbp = (dbp - self.dbp_mean) / self.dbp_std
        y = torch.tensor([sbp, dbp], dtype=torch.float32)

        return (x_ppg, x_feat), y

    def __len__(self):
        return len(self.indices)


class LoadFusion26Dataset:
    """
    DataLoader wrapper for Fusion26Dataset.

    DataLoader yields:
        ((ppg_batch, feat_batch), label_batch)
          ppg_batch:   [batch_size, 1, 1024]
          feat_batch:  [batch_size, 26]
          label_batch: [batch_size, 2]
    """

    def __init__(self, batch_size, is_sample_shuffle=True):
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    def load_train_val_data(self, ppg_path, feat_path, indices_dir):
        # First pass: compute normalization stats from training set
        train_dataset = Fusion26Dataset(ppg_path, feat_path, indices_dir, train=True)

        feat_mean, feat_std = self._feat_statistics(train_dataset, feat_path)
        sbp_mean, sbp_std, dbp_mean, dbp_std = self._bp_statistics(train_dataset)

        # Re-create with normalization
        train_dataset = Fusion26Dataset(
            ppg_path, feat_path, indices_dir, train=True,
            sbp_mean=sbp_mean, sbp_std=sbp_std,
            dbp_mean=dbp_mean, dbp_std=dbp_std,
            feat_mean=feat_mean, feat_std=feat_std
        )

        val_dataset = Fusion26Dataset(
            ppg_path, feat_path, indices_dir, train=False,
            sbp_mean=sbp_mean, sbp_std=sbp_std,
            dbp_mean=dbp_mean, dbp_std=dbp_std,
            feat_mean=feat_mean, feat_std=feat_std
        )

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                num_workers=4, pin_memory=True,
                                persistent_workers=True)
        val_iter = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True,
                              persistent_workers=True)

        return train_iter, val_iter, sbp_mean, sbp_std, dbp_mean, dbp_std

    def _feat_statistics(self, dataset, feat_path):
        with h5py.File(feat_path, "r") as f:
            indices = dataset.indices
            feats = f["ppg_features"][indices]
        return feats.mean(axis=0).astype(np.float32), feats.std(axis=0).astype(np.float32)

    def _bp_statistics(self, dataset):
        with h5py.File(dataset.ppg_path, "r") as f:
            indices = dataset.indices
            sbp = f["sbp"][indices]
            dbp = f["dbp"][indices]
        return sbp.mean(), sbp.std(), dbp.mean(), dbp.std()


# =============================================================================
# Fusion Dataset: PPG (1×1024) + 169-dim Liu2023 features
# =============================================================================

class Fusion169Dataset(Dataset):
    """
    Load both PPG signal and 169-dim Liu2023 features for the fusion model.

    PPG 来自 segmented_records.h5, 169 维特征来自 liu2023_features.h5.
    窗口索引完全对齐 (相同的 cv_fold_*.npz 切分).

    __getitem__ returns ((ppg_tensor, feat_tensor), label_tensor)
        ppg_tensor:  [1, 1024]
        feat_tensor: [169]
        label_tensor: [2] — (SBP, DBP)
    """

    def __init__(self, ppg_path, feat_path, indices_dir, train=True,
                 sbp_mean=None, sbp_std=None, dbp_mean=None, dbp_std=None,
                 feat_mean=None, feat_std=None):
        self.ppg_path = ppg_path
        self.feat_path = feat_path
        self.ppg_file = None
        self.feat_file = None

        self.sbp_mean = sbp_mean
        self.sbp_std = sbp_std
        self.dbp_mean = dbp_mean
        self.dbp_std = dbp_std
        self.feat_mean = feat_mean
        self.feat_std = feat_std

        fold_indices = np.load(indices_dir)
        if train:
            self.indices = fold_indices["train_idx"]
        else:
            self.indices = fold_indices["val_idx"]

    def _init_file(self):
        if self.ppg_file is None:
            self.ppg_file = h5py.File(self.ppg_path, "r")
            self.ppg = self.ppg_file["ppg"]
            self.sbp = self.ppg_file["sbp"]
            self.dbp = self.ppg_file["dbp"]

            self.feat_file = h5py.File(self.feat_path, "r")
            self.feats = self.feat_file["features"]

    def __getitem__(self, index):
        self._init_file()
        idx = self.indices[index]

        # PPG signal
        ppg = self.ppg[idx]  # (1024,)
        x_ppg = torch.from_numpy(ppg).unsqueeze(0).float()  # (1, 1024)

        # 169-dim features
        feats = self.feats[idx].astype(np.float32)  # (169,)
        if self.feat_mean is not None:
            feats = (feats - self.feat_mean) / (self.feat_std + 1e-8)
        x_feat = torch.from_numpy(feats).float()  # (169,)

        # Labels
        sbp = self.sbp[idx]
        dbp = self.dbp[idx]
        if self.sbp_mean is not None:
            sbp = (sbp - self.sbp_mean) / self.sbp_std
            dbp = (dbp - self.dbp_mean) / self.dbp_std
        y = torch.tensor([sbp, dbp], dtype=torch.float32)

        return (x_ppg, x_feat), y

    def __len__(self):
        return len(self.indices)


class LoadFusion169Dataset:
    """
    DataLoader wrapper for Fusion169Dataset.

    DataLoader yields:
        ((ppg_batch, feat_batch), label_batch)
          ppg_batch:   [batch_size, 1, 1024]
          feat_batch:  [batch_size, 169]
          label_batch: [batch_size, 2]
    """

    def __init__(self, batch_size, is_sample_shuffle=True):
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    def load_train_val_data(self, ppg_path, feat_path, indices_dir):
        train_dataset = Fusion169Dataset(ppg_path, feat_path, indices_dir, train=True)

        feat_mean, feat_std = self._feat_statistics(train_dataset, feat_path)
        sbp_mean, sbp_std, dbp_mean, dbp_std = self._bp_statistics(train_dataset)

        train_dataset = Fusion169Dataset(
            ppg_path, feat_path, indices_dir, train=True,
            sbp_mean=sbp_mean, sbp_std=sbp_std,
            dbp_mean=dbp_mean, dbp_std=dbp_std,
            feat_mean=feat_mean, feat_std=feat_std
        )

        val_dataset = Fusion169Dataset(
            ppg_path, feat_path, indices_dir, train=False,
            sbp_mean=sbp_mean, sbp_std=sbp_std,
            dbp_mean=dbp_mean, dbp_std=dbp_std,
            feat_mean=feat_mean, feat_std=feat_std
        )

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                num_workers=4, pin_memory=True,
                                persistent_workers=True)
        val_iter = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True,
                              persistent_workers=True)

        return train_iter, val_iter, sbp_mean, sbp_std, dbp_mean, dbp_std

    def _feat_statistics(self, dataset, feat_path):
        with h5py.File(feat_path, "r") as f:
            indices = dataset.indices
            feats = f["features"][indices]
        return feats.mean(axis=0).astype(np.float32), feats.std(axis=0).astype(np.float32)

    def _bp_statistics(self, dataset):
        with h5py.File(dataset.ppg_path, "r") as f:
            indices = dataset.indices
            sbp = f["sbp"][indices]
            dbp = f["dbp"][indices]
        return sbp.mean(), sbp.std(), dbp.mean(), dbp.std()


# =============================================================================
# Fusion Dataset: PPG + SBP_opt (17-dim) + DBP_opt (12-dim)
# =============================================================================

class FusionOptDataset(Dataset):
    """
    Load PPG + sbp_opt features (17) + dbp_opt features (12).
    __getitem__ returns ((ppg, sbp_feat, dbp_feat), label)
    """

    def __init__(self, ppg_path, feat_path, indices_dir, train=True,
                 sbp_mean=None, sbp_std=None, dbp_mean=None, dbp_std=None,
                 sbp_feat_mean=None, sbp_feat_std=None,
                 dbp_feat_mean=None, dbp_feat_std=None):
        self.ppg_path = ppg_path; self.feat_path = feat_path
        self.ppg_file = None; self.feat_file = None
        self.sbp_mean=sbp_mean; self.sbp_std=sbp_std
        self.dbp_mean=dbp_mean; self.dbp_std=dbp_std
        self.sbp_feat_mean=sbp_feat_mean; self.sbp_feat_std=sbp_feat_std
        self.dbp_feat_mean=dbp_feat_mean; self.dbp_feat_std=dbp_feat_std
        fold_indices = np.load(indices_dir)
        self.indices = fold_indices["train_idx"] if train else fold_indices["val_idx"]

    def _init_file(self):
        if self.ppg_file is None:
            self.ppg_file = h5py.File(self.ppg_path, "r")
            self.ppg = self.ppg_file["ppg"]
            self.sbp = self.ppg_file["sbp"]
            self.dbp = self.ppg_file["dbp"]
            self.feat_file = h5py.File(self.feat_path, "r")
            self.sbp_feats = self.feat_file["features_sbp"]
            self.dbp_feats = self.feat_file["features_dbp"]

    def __getitem__(self, index):
        self._init_file()
        idx = self.indices[index]
        x_ppg = torch.from_numpy(self.ppg[idx]).unsqueeze(0).float()
        sbp_f = self.sbp_feats[idx].astype(np.float32)
        if self.sbp_feat_mean is not None:
            sbp_f = (sbp_f - self.sbp_feat_mean) / (self.sbp_feat_std + 1e-8)
        dbp_f = self.dbp_feats[idx].astype(np.float32)
        if self.dbp_feat_mean is not None:
            dbp_f = (dbp_f - self.dbp_feat_mean) / (self.dbp_feat_std + 1e-8)
        sbp = self.sbp[idx]; dbp = self.dbp[idx]
        if self.sbp_mean is not None:
            sbp = (sbp - self.sbp_mean) / self.sbp_std
            dbp = (dbp - self.dbp_mean) / self.dbp_std
        return ((x_ppg, torch.from_numpy(sbp_f).float(), torch.from_numpy(dbp_f).float()),
                torch.tensor([sbp, dbp], dtype=torch.float32))

    def __len__(self):
        return len(self.indices)


class LoadFusionOptDataset:
    """DataLoader wrapper for FusionOptDataset."""
    def __init__(self, batch_size, is_sample_shuffle=True):
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    def load_train_val_data(self, ppg_path, feat_path, indices_dir):
        train_ds = FusionOptDataset(ppg_path, feat_path, indices_dir, train=True)
        sbp_fm, sbp_fs = self._feat_stats(train_ds, feat_path, "features_sbp")
        dbp_fm, dbp_fs = self._feat_stats(train_ds, feat_path, "features_dbp")
        sbp_m, sbp_s, dbp_m, dbp_s = self._bp_stats(train_ds)

        train_ds = FusionOptDataset(ppg_path, feat_path, indices_dir, train=True,
            sbp_mean=sbp_m, sbp_std=sbp_s, dbp_mean=dbp_m, dbp_std=dbp_s,
            sbp_feat_mean=sbp_fm, sbp_feat_std=sbp_fs,
            dbp_feat_mean=dbp_fm, dbp_feat_std=dbp_fs)
        val_ds = FusionOptDataset(ppg_path, feat_path, indices_dir, train=False,
            sbp_mean=sbp_m, sbp_std=sbp_s, dbp_mean=dbp_m, dbp_std=dbp_s,
            sbp_feat_mean=sbp_fm, sbp_feat_std=sbp_fs,
            dbp_feat_mean=dbp_fm, dbp_feat_std=dbp_fs)

        kw = dict(batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
        train_iter = DataLoader(train_ds, shuffle=self.is_sample_shuffle, **kw)
        val_iter = DataLoader(val_ds, shuffle=False, **kw)
        return train_iter, val_iter, sbp_m, sbp_s, dbp_m, dbp_s

    def _feat_stats(self, ds, feat_path, ds_name):
        with h5py.File(feat_path, "r") as f:
            feats = f[ds_name][ds.indices]
        return feats.mean(axis=0).astype(np.float32), feats.std(axis=0).astype(np.float32)

    def _bp_stats(self, ds):
        with h5py.File(ds.ppg_path, "r") as f:
            sbp, dbp = f["sbp"][ds.indices], f["dbp"][ds.indices]
        return sbp.mean(), sbp.std(), dbp.mean(), dbp.std()
