import os
import torch
from pathlib import Path
from utils.log_helper import logger_init
import logging

# os.listdir('E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset')

# base_dir = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset"
# org_h5data = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\filtered_records.h5"
# detrend_h5data = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\detrend_records.h5"
# scaled_h5data = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\scaled_records.h5"
# segmented_h5data = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\segmented_records.h5"

class Config():
    """
    模型配置类
    """

    def __init__(self):
        #   数据集设置相关配置
        # self.proj_base = r"E:\Kaggle_projects\Blood_Pressure_analysis"
        self.proj_base = Path(__file__).resolve().parent.parent
        # self.base_dir = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset"
        self.base_dir = self.proj_base / "Blood_pressure_dataset"
        # self.datadir = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\segmented_records.h5"
        self.datadir = self.base_dir / "segmented_records.h5"
        # self.h5_detrend = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\detrend_records.h5"
        self.h5_detrend = self.base_dir / "detrend_records.h5"
        # self.h5_scaled = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\scaled_records.h5"
        self.h5_scaled = self.base_dir / "scaled_records.h5"
        # self.h5_seg = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\segmented_records.h5"
        self.h5_seg = self.base_dir / "segmented_records.h5"
        self.k = 5
        
        #  模型相关配置
        self.batch_size = 64
        self.filters = [1, 32, 64, 128]
        self.lr = 0.001
        self.num_layers = 2
        self.num_directions = 2
        self.hidden_dim = 128


        self.dropout = 0.2
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 10e-9
        self.device = self._init_device()
        self.epochs = 500
        self.early_stopping_patience = 10
        self.model_save_dir = os.path.join(self.proj_base, 'cache')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        # 日志相关
        logger_init(log_file_name='log_train',
                    log_level=logging.INFO,
                    log_dir=self.model_save_dir)
    

    def _init_device(self):
        device = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0') 
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device