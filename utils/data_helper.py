import numpy as np
import scipy.io
from scipy.signal import detrend
from scipy.signal import find_peaks
from sklearn.model_selection import KFold

import h5py
import os
from tqdm import tqdm


def load_data(base_dir: str):
    """
    加载.mat文件并读取所有ppg和abp信号，返回ppg和abp两个列表
    
    :param base_dir: .mat文件所在路径
    :type base_dir: str
    """
    ppg = []
    abp = []

    for k in tqdm(range(1, 13), desc="Loading files", colour="green"):
        file_path = os.path.join(base_dir, f"part_{k}.mat")
        file = scipy.io.loadmat(file_path)['p'][0] # 选['p'][0]->p对应字典结构的数据value，[0]对应全部数据(1000条recoding)
        
        for i in range(len(file)):
            ppg.append(file[i][0]) # file[i][0]为list, 存的是一个recording
            abp.append(file[i][1])

    return ppg, abp


def remove_extreme_data(base_dir: str) -> None: 
    """
    读取所有.mat文件并基于条件保留所有记录时长大于8min, 
    且不含极端abp值（大于200mmHg）的recording。
    返回h5文件 （原本12个文件含12000条recording， 经过过滤剩1967条）

    :param base_dir: 数据集路径，保存路径也基于此路径
    """

    save_path = os.path.join(base_dir, "filtered_records.h5")

    fs = 125
    t = 8.192
    dt = 2.048

    filt_con = 8 * 60 * fs
    extreme_abp = 200

    ppg = []
    abp = []

    for k in tqdm(range(1, 13), desc="Removing extreme data", colour="cyan"):
        print(f"Processing file part {k} out of 12.")

        file_path = os.path.join(base_dir, f"part_{k}.mat")
        file = scipy.io.loadmat(file_path)['p'][0] # 选['p'][0]->p对应字典结构的数据value，[0]对应全部数据(1000条recoding)
        
        for i in range(len(file)):
            if len(file[i][0]) <= filt_con or max(file[i][1]) > extreme_abp:
                continue

            ppg.append(file[i][0]) # list, 存的是一个recording
            abp.append(file[i][1])
        
    with h5py.File(save_path, "w") as f:
        f.create_dataset(
            "ppg",
            data=np.array(ppg, dtype=object),
            dtype=h5py.vlen_dtype(np.float32)
        )

        f.create_dataset(
            "abp",
            data=np.array(abp, dtype=object),
            dtype=h5py.vlen_dtype(np.float32)
        )


def detrend_ppg(h5_in: str, h5_out: str) -> None:
    """
    使用scipy.signal的detrend函数简单去除每一个recording内
    ppg信号中的trend，并写入h5文件
    
    :param ppg: Description
    """
    with h5py.File(h5_in, "r") as fin, h5py.File(h5_out, "w") as fout:
        ppg_in = fin["ppg"]
        abp_in = fin["abp"]

        ppg_out = fout.create_dataset(
            "ppg",
            shape = (len(ppg_in),),
                     dtype=h5py.vlen_dtype(np.float32)
                     )
        
        abp_out = fout.create_dataset(
            "abp",
            shape=(len(abp_in),),
            dtype=h5py.vlen_dtype(np.float32)
        )

        for i in tqdm(range(len(ppg_in)), desc="Detrending data", colour="cyan"):
            ppg_out[i] = detrend(ppg_in[i]).astype(np.float32)
            abp_out[i] = abp_in[i]


def zscore_normalization(h5_in: str, h5_out: str) -> None:
    """
    用z score对每一个recording的ppg信号做归一化，
    范围缩减至0~1之间，返回h5文件
    
    :param h5_in: Description
    :param h5_out: Description
    """

    with h5py.File(h5_in, "r") as fin, h5py.File(h5_out, "w") as fout:
        ppg_in = fin["ppg"]
        abp_in = fin["abp"]

        ppg_out = fout.create_dataset(
            "ppg", 
            shape = (len(ppg_in),),
            dtype=h5py.vlen_dtype(np.float32)
        )

        abp_out = fout.create_dataset(
            "abp",
            shape = (len(abp_in),),
            dtype = h5py.vlen_dtype(np.float32)
        )

        for i in tqdm(range(len(ppg_in)), desc="Normalizing data", colour="cyan"):
            ppg_out[i] = (ppg_in[i] - ppg_in[i].mean()) / (ppg_in[i].std() + 1e-8)
            abp_out[i] = abp_in[i]


def window_seg(h5_in: str, h5_out: str) -> None:
    """
    取window size为8.192s(1024个数据点), step_size为256 (75% overlap),
    取每个window size内平均的sbp和dbp值为预测目标，处理后的ppg值为输入，
    同时记录record_id, 最终h5文件包含: ppg, sbp, dbp, record_ids
    
    :param h5_in: Description
    :type h5_in: str
    :param h5_out: Description
    :type h5_out: str
    """

    ppg_windows = []
    abp_windows = []
    sbps = []
    dbps = []
    record_ids = []
    # window_ids = []

    with h5py.File(h5_in, "r") as fin:
        ppg_in = fin["ppg"]
        abp_in = fin["abp"]

        for i in tqdm(range(len(ppg_in)), desc="Segmenting windows", colour="cyan"):
            record_len = len(ppg_in[i])
            n_windows = ((record_len - 1024) // 256) + 1

            for j in range(n_windows):
                start = j * 256
                end = j * 256 + 1024

                ppg_window = ppg_in[i][start: end] #第i个recording的ppg, 该recording下第j个window的数据
                abp_window = abp_in[i][start: end]
                sbp, dbp = window_avg_peak(abp_window=abp_window)

                if sbp is None:
                    continue

                # sbp = max(abp_in[i][start: end])
                # dbp = min(abp_in[i][start: end])

                ppg_windows.append(ppg_window.astype(np.float32))
                abp_windows.append(abp_window.astype(np.float32))
                sbps.append(sbp)
                dbps.append(dbp)
                record_ids.append(i)

        ppg = np.stack(ppg_windows)
        abp = np.stack(abp_windows)
        sbp = np.array(sbps, dtype=np.float32)
        dbp = np.array(dbps, dtype=np.float32)

    with h5py.File(h5_out, "w") as fout:
        fout.create_dataset("ppg", data=ppg, dtype=np.float32)
        fout.create_dataset("abp", data=abp, dtype=np.float32)
        fout.create_dataset("sbp", data=sbp, dtype=np.float32)
        fout.create_dataset("dbp", data=dbp, dtype=np.float32)
        fout.create_dataset("record_id", data=np.array(record_ids, dtype=np.int32))


def window_avg_peak(abp_window, fs=125):
    """
    使用scipy.signal的find_peaks方法找到每一个window_size内的所有sbp和dbp的索引
    然后分别对sbp和dbp取平均得到本window内的预测目标
    
    :param abp_window: Description
    :param fs: Description
    """

    peaks, _ = find_peaks(
        abp_window,
        distance=int(0.4 * fs), # 峰值之间最少间隔0.4s，防止抖动影响
        prominence=10
    )

    valleys, _ = find_peaks(
        -abp_window,
        distance=int(0.4 * fs),
        prominence=10
    )

    if len(peaks) == 0 or len(valleys) == 0:
        return None, None

    avg_sbp_value = np.mean(abp_window[peaks])
    avg_dbp_value = np.mean(abp_window[valleys])

    return avg_sbp_value, avg_dbp_value


def k_fold_save(k: int, data_path: str) -> None:
    """
    Docstring for k_fold_save
    
    :param data_path: Description
    :type data_path: str
    使用：
    fold = 2
    idx = np.load(f"cv_folds/fold_{fold}.npz")

    with h5py.File("segmented_records.h5", "r") as f:
        X_train = f["ppg"][idx["train_idx"]]
        y_train = f["sbp"][idx["train_idx"]]

        X_val   = f["ppg"][idx["val_idx"]]
        y_val   = f["sbp"][idx["val_idx"]]
    """
    with h5py.File(data_path, "r") as f:
        record_id = f["record_id"][:]

    unique_ids = np.unique(record_id)

    kf = KFold(n_splits=k, shuffle=False)

    for fold, (train_ids_idx, val_ids_idx) in enumerate(tqdm(kf.split(unique_ids), 
                                                             total=k, 
                                                             desc="Generating CV folds", 
                                                             colour="cyan")):
        train_ids = unique_ids[train_ids_idx]
        val_ids = unique_ids[val_ids_idx]

        train_idx = np.where(np.isin(record_id, train_ids))[0]
        val_idx = np.where(np.isin(record_id, val_ids))[0]

        np.savez(
            f"Blood_pressure_dataset/cv_fold_{fold}.npz",
            train_idx = train_idx,
            val_idx = val_idx
        )


def chain(base_dir: str, 
          h5_detrend: str, 
          h5_scaled: str, 
          h5_seg: str, 
          k: int, 
          ) -> None:
    """
    base_dir: 原始.mat文件路径
    h5_detrend: detrend后的ppg信号，h5文件路径
    h5_scaled: 归一化后的ppg信号，h5文件路径
    h5_seg: 切窗后的ppg信号，h5文件路径
    k: 交叉验证划分折数
    """
    remove_extreme_data(base_dir=base_dir)
    h5_org = os.path.join(base_dir, "filtered_records.h5")
    detrend_ppg(h5_in=h5_org, h5_out=h5_detrend)
    zscore_normalization(h5_in=h5_detrend, h5_out=h5_scaled)
    window_seg(h5_in=h5_scaled, h5_out=h5_seg)
    k_fold_save(k=k, data_path=h5_seg)


def feature_extraction_ppg(h5_in: str, h5_out: str):
    """
    人工提取PPG信号特征，包含：一个segment内ppg信号的所有峰值和谷值，峰值和谷值的均值，峰值间的平均时间间隔和谷值之间的时间间隔
    新增：统计特征向量，便于直接输入模型
    """
    fs = 125
    # 六个字段为不同特征
    ppg_peaks = []
    ppg_valleys = []
    avg_ppg_peak = []
    avg_ppg_valley = []
    avg_ppg_peak_interval = []
    avg_ppg_valley_interval = []
    record_ids = []
    feature_vectors = []  # 新增：固定长度特征向量

    with h5py.File(h5_in, "r") as fin:
        ppg_seg = fin["ppg"]
        for i in tqdm(range(len(ppg_seg)), desc="Extracting PPG features as input...", colour="cyan"):
            segment = ppg_seg[i]  # 第i个窗口的PPG信号
            peaks_ppg, _ = find_peaks(
                segment,
                distance=int(0.4 * fs), # 峰值之间最少间隔0.4s，防止抖动影响
                prominence=0.5
            )

            valleys_ppg, _ = find_peaks(
                -segment,
                distance=int(0.4 * fs),
                prominence=0.5
            )

            if len(peaks_ppg) == 0 or len(valleys_ppg) == 0:
                continue  # 跳过没有峰值或谷值的窗口

            all_ppg_peak_values = segment[peaks_ppg]
            all_ppg_valley_values = segment[valleys_ppg]

            avg_ppg_peak_value = np.mean(all_ppg_peak_values)
            avg_ppg_valley_value = np.mean(all_ppg_valley_values)

            peak_intervals_samples = np.diff(peaks_ppg)
            peak_intervals_seconds = peak_intervals_samples / fs

            valley_intervals_samples = np.diff(valleys_ppg)
            valley_intervals_seconds = valley_intervals_samples / fs

            # 每个ppg_seg取出的特征分别存入
            ppg_peaks.append(all_ppg_peak_values.astype(np.float32))
            ppg_valleys.append(all_ppg_valley_values.astype(np.float32))
            avg_ppg_peak.append(avg_ppg_peak_value)
            avg_ppg_valley.append(avg_ppg_valley_value)
            avg_ppg_peak_interval.append(peak_intervals_seconds.astype(np.float32))
            avg_ppg_valley_interval.append(valley_intervals_seconds.astype(np.float32))
            record_ids.append(i)

            # 计算统计特征向量（固定长度）
            # 1. 峰值值的统计量：均值、标准差、最小值、最大值、25%, 50%, 75%分位数、数量
            peak_stats = []
            if len(all_ppg_peak_values) > 0:
                peak_stats.extend([
                    np.mean(all_ppg_peak_values),
                    np.std(all_ppg_peak_values),
                    np.min(all_ppg_peak_values),
                    np.max(all_ppg_peak_values),
                    np.percentile(all_ppg_peak_values, 25),
                    np.percentile(all_ppg_peak_values, 50),
                    np.percentile(all_ppg_peak_values, 75),
                    len(all_ppg_peak_values)
                ])
            else:
                peak_stats.extend([0.0] * 8)

            # 2. 谷值值的统计量
            valley_stats = []
            if len(all_ppg_valley_values) > 0:
                valley_stats.extend([
                    np.mean(all_ppg_valley_values),
                    np.std(all_ppg_valley_values),
                    np.min(all_ppg_valley_values),
                    np.max(all_ppg_valley_values),
                    np.percentile(all_ppg_valley_values, 25),
                    np.percentile(all_ppg_valley_values, 50),
                    np.percentile(all_ppg_valley_values, 75),
                    len(all_ppg_valley_values)
                ])
            else:
                valley_stats.extend([0.0] * 8)

            # 3. 峰值间隔的统计量：均值、标准差、最小值、最大值、数量（间隔数）
            peak_interval_stats = []
            if len(peak_intervals_seconds) > 0:
                peak_interval_stats.extend([
                    np.mean(peak_intervals_seconds),
                    np.std(peak_intervals_seconds),
                    np.min(peak_intervals_seconds),
                    np.max(peak_intervals_seconds),
                    len(peak_intervals_seconds)
                ])
            else:
                peak_interval_stats.extend([0.0] * 5)

            # 4. 谷值间隔的统计量
            valley_interval_stats = []
            if len(valley_intervals_seconds) > 0:
                valley_interval_stats.extend([
                    np.mean(valley_intervals_seconds),
                    np.std(valley_intervals_seconds),
                    np.min(valley_intervals_seconds),
                    np.max(valley_intervals_seconds),
                    len(valley_intervals_seconds)
                ])
            else:
                valley_interval_stats.extend([0.0] * 5)

            # 拼接所有统计特征
            feature_vector = peak_stats + valley_stats + peak_interval_stats + valley_interval_stats
            feature_vectors.append(np.array(feature_vector, dtype=np.float32))

    # 转换为对象数组以存储可变长度数据
    ppg_peaks_arr = np.array(ppg_peaks, dtype=object)
    ppg_valleys_arr = np.array(ppg_valleys, dtype=object)
    avg_ppg_peak_arr = np.array(avg_ppg_peak, dtype=np.float32)
    avg_ppg_valley_arr = np.array(avg_ppg_valley, dtype=np.float32)
    avg_ppg_peak_interval_arr = np.array(avg_ppg_peak_interval, dtype=object)
    avg_ppg_valley_interval_arr = np.array(avg_ppg_valley_interval, dtype=object)
    feature_matrix = np.stack(feature_vectors) if feature_vectors else np.array([], dtype=np.float32).reshape(0, 26)

    with h5py.File(h5_out, "w") as fout:
            fout.create_dataset("ppg_peaks", data=ppg_peaks_arr, dtype=h5py.vlen_dtype(np.float32))
            fout.create_dataset("ppg_valleys", data=ppg_valleys_arr, dtype=h5py.vlen_dtype(np.float32))
            fout.create_dataset("avg_ppg_peak", data=avg_ppg_peak_arr, dtype=np.float32)
            fout.create_dataset("avg_ppg_valley", data=avg_ppg_valley_arr, dtype=np.float32)
            fout.create_dataset("avg_ppg_peak_interval", data=avg_ppg_peak_interval_arr, dtype=h5py.vlen_dtype(np.float32))
            fout.create_dataset("avg_ppg_valley_interval", data=avg_ppg_valley_interval_arr, dtype=h5py.vlen_dtype(np.float32))
            fout.create_dataset("record_id", data=np.array(record_ids, dtype=np.int32))
            fout.create_dataset("ppg_features", data=feature_matrix, dtype=np.float32)

