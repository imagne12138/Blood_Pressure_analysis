"""
分层采样器：确保每个 batch 包含足量的高/低血压样本

用法：
    from utils.stratified_sampler import StratifiedBPSampler
    sampler = StratifiedBPSampler(labels, batch_size=256)
    loader = DataLoader(ds, batch_sampler=sampler)
"""
import numpy as np
import torch
from torch.utils.data import Sampler


class StratifiedBPSampler(Sampler):
    """
    按 BP 值分层的 Batch Sampler。

    将样本按 SBP 分为 N 个区间，每个 batch 从各区间等量采样。
    确保模型在每个 batch 中都看到足量的高低血压样本。

    Args:
        sbp_labels: 所有样本的 SBP 真实值（numpy 或 list）
        batch_size: batch 大小
        bins: 分区边界，默认 [(0,100),(100,120),(120,140),(140,160),(160,999)]
        shuffle: 是否打乱
    """
    def __init__(self, sbp_labels, batch_size=256,
                 bins=None, shuffle=True):
        if bins is None:
            bins = [(0, 100), (100, 120), (120, 140), (140, 160), (160, 999)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0

        # 分配区间
        if isinstance(sbp_labels, torch.Tensor):
            sbp_labels = sbp_labels.numpy()
        sbp_labels = np.asarray(sbp_labels)

        self.bin_indices = []
        for lo, hi in bins:
            idx = np.where((sbp_labels >= lo) & (sbp_labels < hi))[0]
            self.bin_indices.append(idx)

        # 每个 batch 从各区间采样的数量
        min_bin_size = min(len(idx) for idx in self.bin_indices)
        self.samples_per_bin = max(1, batch_size // len(bins))
        # 防止某区间样本太少
        self.samples_per_bin = min(self.samples_per_bin, min_bin_size)
        self._num_batches = max(1, min_bin_size // self.samples_per_bin)

        # 打乱后用于迭代的副本
        self._reset_indices()

    def _reset_indices(self):
        self._working_indices = [
            idx.copy() for idx in self.bin_indices
        ]
        if self.shuffle:
            for arr in self._working_indices:
                np.random.RandomState(self.epoch).shuffle(arr)
        self._pos = [0] * len(self._working_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._reset_indices()

    def __iter__(self):
        self._reset_indices()
        for _ in range(self._num_batches):
            batch = []
            for bin_i in range(len(self._working_indices)):
                start = self._pos[bin_i]
                end = start + self.samples_per_bin
                idx = self._working_indices[bin_i][start:end]
                batch.extend(idx.tolist())
                self._pos[bin_i] = end
            if self.shuffle:
                np.random.RandomState(self.epoch + _).shuffle(batch)
            yield batch

    def __len__(self):
        return self._num_batches


class BalancedBPSampler(Sampler):
    """
    更简洁的版本：对极端 BP 值做上采样，然后随机采样。

    对 SBP < 100 或 SBP > 160 的样本做 3x-5x 复制，
    使得它们出现在训练中的概率更高。
    """
    def __init__(self, sbp_labels, multiplier=3.0, shuffle=True):
        if isinstance(sbp_labels, torch.Tensor):
            sbp_labels = sbp_labels.numpy()
        sbp_labels = np.asarray(sbp_labels)

        n = len(sbp_labels)
        weights = np.ones(n, dtype=np.float64)

        # 极端值加权
        extreme_mask = (sbp_labels < 100) | (sbp_labels > 160)
        weights[extreme_mask] = multiplier

        # 中等偏低
        mid_mask = (sbp_labels >= 100) & (sbp_labels < 110) | \
                   (sbp_labels >= 150) & (sbp_labels <= 160)
        weights[mid_mask] = 2.0

        self.weights = torch.from_numpy(weights)
        self.n = n
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            yield from torch.multinomial(self.weights, self.n, replacement=True).tolist()
        else:
            yield from range(self.n)

    def __len__(self):
        return self.n
