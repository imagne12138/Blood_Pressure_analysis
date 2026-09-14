# Blood Pressure Analysis

基于光电容积脉搏波（photoplethysmography，PPG）信号进行收缩压（SBP）和舒张压（DBP）估计的研究代码库。本项目比较纯 PPG 深度学习模型、手工特征模型以及 PPG–特征融合模型，并使用 5 折交叉验证评估泛化性能。

> **项目状态**：实验代码与结果整理中。详细实验记录见 [`EXPERIMENTS.md`](EXPERIMENTS.md)，结果汇总见 [`cache/实验训练结果总结.md`](cache/实验训练结果总结.md)。

## 研究概览

### 数据与预处理

- 数据来源：UCI/BPD PPG 数据集（Kachuee et al., 2015）。
- 原始数据：`Blood_pressure_dataset/part_1.mat` 至 `part_12.mat`。
- 过滤后约 **1,967 条 recording**，切分为约 **517,305 个窗口**。
- PPG 输入：单通道、长度 1024 的窗口（采样率 125 Hz，约 8.192 s）。
- 标签：每个窗口对应的 SBP 与 DBP。
- 交叉验证：预先生成的 5 折索引位于 `Blood_pressure_dataset/cv_fold_*.npz`。
- 手工特征：
  - 26 维峰值/谷值统计特征，位于 `ppg_features.h5`；
  - 169 维 Liu et al. (2023) 特征，位于 `liu2023_features.h5`（如已生成）。

### 当前实验结论

在已完成并记录的实验中，`Model2 + 26-dim feature fusion` 是综合表现最佳的方案：

| 模型 | 特征 | SBP MAE (mmHg) | DBP MAE (mmHg) |
| --- | --- | ---: | ---: |
| Model 2 + 26 维融合 | PPG + 26 维统计特征 | **15.525 ± 1.382** | **6.859 ± 0.931** |
| Baseline + 26 维融合 | PPG + 26 维统计特征 | 15.680 ± 1.230 | **6.760 ± 0.940** |
| Model 2（纯 PPG） | PPG | 15.782 ± 1.429 | 6.953 ± 0.956 |
| Model 2 + 169 维融合 | PPG + Liu2023 特征 | 16.381 ± 1.266 | 7.170 ± 0.840 |
| 改进 Liu2023 MLP | 169 维特征 | 16.515 ± 1.202 | 7.262 ± 0.800 |

详细的逐折结果、其他模型以及实验解释请以 [`EXPERIMENTS.md`](EXPERIMENTS.md) 为准。总体观察是：在本数据集和当前实验设置下，低维 26 维统计特征比 169 维特征更稳定；PPG 时序信息与统计特征具有互补性。

## 目录结构

```text
Blood_Pressure_analysis/
├── Readme.md                         # 项目说明
├── EXPERIMENTS.md                    # 实验记录与结果对比
├── requirements.txt                  # Python 依赖
├── Blood_pressure.ipynb              # 探索性分析与原型
├── Visualization.ipynb               # 可视化分析（文件名以实际文件为准）
│
├── training_scripts/                 # 训练入口
│   ├── train.py                     # Baseline / Model 2
│   ├── train_for_MLP.py             # 26 维特征 MLP
│   ├── train_for_MLP_Liu2023.py     # 169 维 Liu2023 特征 MLP
│   ├── train_for_LightGBM_Liu2023.py
│   ├── train_fusion_26.py            # Model 2 + 26 维融合
│   ├── train_fusion_169.py           # Model 2 + 169 维融合
│   ├── train_fusion_baseline.py      # Baseline + 169 维融合
│   ├── train_fusion_baseline_26.py   # Baseline + 26 维融合
│   ├── train_mlp_opt.py              # SBP/DBP 专用子集 MLP
│   ├── train_fusion_opt.py           # SBP/DBP 专用子集融合
│   ├── train_fusion_gated26.py       # Gated Fusion（26 维）
│   ├── train_fusion_gated169.py      # Gated Fusion（169 维）
│   ├── train_fusion_ordinal.py       # CORAL ordinal（26 维）
│   └── train_fusion_ordinal169.py    # CORAL ordinal（169 维）
│
├── model/                            # PyTorch 模型定义
│   ├── baseline_model.py             # 共享注意力的纯 PPG 基线
│   ├── model_2.py                    # SBP/DBP 分头注意力模型
│   ├── MLP_for_feature.py            # 26 维特征 MLP
│   ├── MLP_for_Liu2023.py            # 169 维特征 MLP
│   ├── MLP_Opt.py                    # 专用特征子集 MLP
│   ├── LightGBM_for_Liu2023.py       # LightGBM 基线
│   ├── Fusion_Model_26.py            # 分头注意力 + 26 维融合
│   ├── Fusion_Model_169.py           # 分头注意力 + 169 维融合
│   ├── Fusion_Baseline_26.py         # 共享注意力 + 26 维融合
│   ├── Fusion_Baseline.py            # 共享注意力 + 169 维融合
│   ├── Fusion_Opt.py                 # 专用子集融合
│   ├── Fusion_Gated.py               # 门控融合
│   ├── Fusion_HeadConcat.py          # Head concat 消融模型
│   ├── Fusion_Ordinal.py             # CORAL ordinal 模型
│   ├── two_stage_bp.py               # 两阶段血压估计
│   ├── custom_losses.py              # 自定义损失
│   └── custom_scheduler_for_transformer.py
│
├── utils/                            # 数据、特征与日志工具
│   ├── create_data.py                # 数据集构建与交叉验证加载
│   ├── data_helper.py                # 原始数据读取与预处理
│   ├── liu2023_features.py           # Liu2023 特征提取
│   ├── ppg_spectrogram.py            # PPG 频谱图工具
│   ├── stratified_sampler.py         # 分层采样
│   └── log_helper.py                 # 日志配置
│
├── config/config.py                  # 路径和训练超参数
├── Blood_pressure_dataset/           # 数据、HDF5 文件、样本与 CV 索引
├── cache/                            # 日志、checkpoint、图表和中间结果
├── Tests/                            # 模型、数据加载和特征验证脚本
├── experiments_improved/             # 改进损失函数实验
└── tmp/                              # 临时分析与文档生成脚本
```

## 模型演化

```text
纯 PPG
  BaselineModel
      └── Model_2_Head（SBP/DBP 分头注意力与输出）

纯手工特征
  26 维统计特征 MLP
  169 维 Liu2023 特征 MLP / LightGBM
  SBP/DBP 专用子集 MLP

PPG + 手工特征融合
  Baseline + 26/169 维特征
  Model 2 + 26/169 维特征
  Gated Fusion / Fusion Opt / CORAL Ordinal
```

核心时序编码器通常为：

```text
PPG [B, 1, 1024]
  → 3 层 Conv1D + BatchNorm + ReLU + MaxPool
  → 双向 LSTM（hidden size = 128）
  → 注意力池化（共享或 SBP/DBP 分头）
  → 回归输出
```

融合模型在注意力得到的 256 维上下文向量上加入手工特征投影，再分别预测 SBP 与 DBP。

## 快速开始

### 1. 创建环境

```bash
conda create -n Pytorch python=3.10
conda activate Pytorch
pip install -r requirements.txt
```

实际 Python、PyTorch 和 CUDA 版本应根据本机环境调整。

### 2. 准备数据

将数据文件放入：

```text
Blood_pressure_dataset/
├── part_1.mat ... part_12.mat
├── segmented_records.h5
├── ppg_features.h5
├── liu2023_features.h5       # 运行 169 维实验时需要
└── cv_fold_0.npz ... cv_fold_4.npz
```

`config/config.py` 使用项目目录的相对路径，不要求修改硬编码的绝对路径。若 HDF5 文件尚未生成，可先参考 `utils/data_helper.py` 中的数据处理函数，并使用对应测试脚本检查输出形状。

### 3. 训练

在项目根目录执行，例如：

```bash
conda run -n Pytorch python training_scripts/train.py
conda run -n Pytorch python training_scripts/train_fusion_26.py
conda run -n Pytorch python training_scripts/train_fusion_169.py
```

默认配置位于 [`config/config.py`](config/config.py)，包括：

- 5 折交叉验证；
- batch size = 128；
- learning rate = 1e-4；
- 最大训练轮数 = 500；
- early stopping patience = 20；
- 自动选择 CUDA、Apple MPS 或 CPU。

训练日志和模型 checkpoint 默认写入 `cache/`。不同实验脚本可能会在 `cache/` 下使用不同的子目录，运行前请查看脚本顶部的说明和保存路径。

### 4. 验证模型与数据

```bash
conda run -n Pytorch python Tests/model_test.py
conda run -n Pytorch python Tests/model_2_test.py
conda run -n Pytorch python Tests/check_fusion26.py
conda run -n Pytorch python Tests/check_fusion169.py
conda run -n Pytorch python Tests/data_loader_test.py
```

这些脚本主要用于确认 forward pass、输入输出维度、数据加载和特征处理正常，并不替代完整训练评估。

## 评价指标与结果解释

主要报告：

- **MAE**：SBP/DBP 的平均绝对误差，单位为 mmHg；
- **RMSE**：均方根误差；
- **Pearson r**：预测值与真实值的线性相关性；
- **Bias 与 95% limits of agreement**：用于误差一致性分析。

注意：MAE、RMSE、相关系数和 Bland–Altman 一致性界限反映的是不同性质，不能相互替代。注意力分析中，若比较注意力模式，应明确区分 pre-softmax 得分和 post-softmax 权重，并使用与分析目标一致的相似度指标。

## 注意事项

1. **模型实现与历史结果需对应。** `EXPERIMENTS.md` 记录了不同时间的实验版本；修改模型结构、特征投影或归一化方式后，应使用新的实验名和日志目录，避免覆盖旧结果。
2. **FusionModel26 的特征维度要保持一致。** 直接拼接版本与带投影版本的融合维度不同；checkpoint 加载前必须确认模型定义、投影层和保存时的 `fusion_dim` 完全一致。
3. **交叉验证应按 recording 划分。** 同一 recording 切出的窗口不能同时出现在训练集和验证集，否则会导致数据泄漏和过于乐观的结果。
4. **结果表中的平均值应由逐折结果重新计算。** 不要手工复制单折结果或用四舍五入后的数值计算均值和标准差。
5. `tmp/` 中的脚本是一次性分析和文档生成工具，不是稳定的训练 API；使用前请检查输入路径和输出文件。
6. 数据文件可能占用较大空间，不应将原始数据、HDF5 中间文件或训练 checkpoint 提交到不适合存储大文件的仓库。

## 参考资料

- Kachuee, M. et al. (2015). *Cuffless Blood Pressure Estimation Using a Smartwatch-Based Photoplethysmography Signal*. IEEE International Conference on Healthcare Informatics.
- Liu, et al. (2023). PPG-based blood pressure estimation using geometric and derivative waveform features. *Biomedical Signal Processing and Control*, 86.

## 相关文档

- [`EXPERIMENTS.md`](EXPERIMENTS.md)：实验设计、逐折指标和架构比较。
- [`cache/实验训练结果总结.md`](cache/实验训练结果总结.md)：中英文实验结果汇总。
- [`config/config.py`](config/config.py)：默认路径与训练配置。
- [`presentation/presentation_plan.md`](presentation/presentation_plan.md)：汇报/演示结构。

---

最后更新：2026-09-14
