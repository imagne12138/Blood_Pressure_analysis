# PPG 血压估计实验记录

> 项目: Blood_Pressure_analysis
> 数据集: UCI/BPD PPG 数据集 (约 1967 条 recording, 分割为 517305 个窗口)
> 基线: 纯 PPG 时序模型 (1D-CNN + BiLSTM + Attention)

---

## 目录

1. [纯 PPG 基线](#1-纯-ppg-基线)
2. [纯特征 MLP](#2-纯特征-mlp)
3. [特征融合模型](#3-特征融合模型)
4. [架构对比汇总](#4-架构对比汇总)
5. [文件索引](#5-文件索引)

---

## 1. 纯 PPG 基线

### 1.1 Model_2_Head（分头注意力 + 独立输出头）

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/model_2.py` — `Model_2_Head` |
| 训练脚本 | `train.py`（注释切换 BaselineModel / Model_2_Head） |
| 架构 | Conv1D(1→32→64→128) → BiLSTM(h=128, bi) → 分头注意力 → 独立 SBP/DBP 头 |
| 参数量 | ~764K |
| 保存路径 | `cache/model_fold_X.pkl` |
| 日志 | `cache/Using model 2 and adding weight decay.txt` |
| SBP MAE | **15.78 ± 1.43 mmHg** |
| DBP MAE | **6.95 ± 0.96 mmHg** |

### 1.2 BaselineModel（共享注意力 + 共享输出头）

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/baseline_model.py` — `BaselineModel` |
| 架构 | Conv1D(1→32→64→128) → BiLSTM(h=128, bi) → 共享注意力 → 共享输出头 |
| 参数量 | ~720K |
| 备注 | 代码与 Model_2 共用 `train.py`，注释切换；未单独记录最新日志 |

---

## 2. 纯特征 MLP

### 2.1 26 维统计特征 MLP

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/MLP_for_Liu2023.py` — `MLP`（与 169 维共用） |
| 训练脚本 | `train_for_MLP.py` |
| 特征 | 26 维峰值/谷值统计特征（`ppg_features.h5`） |
| 架构 | Linear(26→128→64→2) + BN + ReLU + Dropout(0.1) |
| 保存路径 | `cache/liu2023_mlp_fold_X.pkl`（实际是 26 维，命名沿用旧名） |
| SBP MAE | **16.81 ± 0.94 mmHg** |
| DBP MAE | **7.61 ± 0.76 mmHg** |

### 2.2 169 维 Liu2023 特征 MLP

| 项目 | 内容 |
|:---|:---|
| 训练脚本 | `train_for_MLP_Liu2023.py` |
| 特征 | 169 维全量 Liu2023 特征（`liu2023_features.h5`） |
| 架构 | Linear(169→128→64→2) + BN + ReLU + Dropout(0.1) |
| 保存路径 | `cache/liu2023_mlp_fold_X.pkl` |
| SBP MAE | **16.70 ± 1.04 mmHg** |
| DBP MAE | **7.45 ± 0.97 mmHg** |

### 2.3 169 维 LightGBM

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/LightGBM_for_Liu2023.py` — `LightGBMRegressor` |
| 训练脚本 | `train_for_LightGBM_Liu2023.py` |
| 特征 | 169 维全量 Liu2023 特征 |
| 架构 | LightGBM, num_leaves=63, max_depth=8, lr=0.05, n_estimators=1000, early_stopping=50 |
| 保存路径 | `cache/liu2023_lgbm_fold_X.pkl` |
| SBP MAE | **16.17 mmHg**（未记录标准差） |
| DBP MAE | **7.25 mmHg** |

### 2.4 最优子集 MLP（待实验）

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/MLP_Opt.py` — `MLP_Opt` |
| 训练脚本 | `train_mlp_opt.py` |
| 特征 | SBP: sbp_opt(17 维); DBP: dbp_opt(12 维)（`liu2023_features.h5` 中 `features_sbp`/`features_dbp`） |
| 架构 | SBP: Linear(17→64→32→1); DBP: Linear(12→32→16→1) |
| 参数量 | ~4K |
| 保存路径 | `cache/mlp_opt_fold_X.pkl` |
| 状态 | ⏳ 待运行 |
| 运行命令 | `conda run -n Pytorch python train_mlp_opt.py` |

---

## 3. 特征融合模型

### 3.1 分头 + 26 维直接 concat（org）

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/Fusion_Model_26.py` — `FusionModel26`（原始版本） |
| 训练脚本 | `train_fusion_26.py` |
| 特征 | 26 维统计特征，直接 concat 到 context(256→282) |
| 架构 | Conv1D×3 → BiLSTM → 分头注意力 → concat(256+26) → Linear(282→128→1) |
| 保存路径 | `cache/model_fusion_26dim/fusion_org/fusion26_fold_X.pkl` |
| 日志 | `cache/model_fusion_26dim/fusion_org/log_train_fusion26_2026-06-01.txt` |
| SBP MAE | **16.06 ± 1.29 mmHg** |
| DBP MAE | **6.94 ± 0.88 mmHg** |

### 3.2 分头 + 26 维特征投影（proj）

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/Fusion_Model_26.py` — `FusionModel26`（带投影 MLP 版本） |
| 特征 | 26 维 → MLP(26→16) → concat(256+16=272) |
| 架构 | 同上，新增 `feature_projection` 层 |
| 保存路径 | `cache/model_fusion_26dim/fusion_feature_proj/fusion26_fold_X_feat_proj.pkl` |
| 日志 | `cache/model_fusion_26dim/fusion_feature_proj/log_train_fusion26_feat_proj_2026-06-01.txt` |
| SBP MAE | **15.53 ± 1.38 mmHg** |
| DBP MAE | **6.86 ± 0.93 mmHg** |
| 最佳单折 | Fold 2 SBP **12.86 mmHg**（全实验最佳） |
| 备注 | **当前最佳融合方案** |

### 3.3 分头 + 169 维特征投影

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/Fusion_Model_169.py` — `FusionModel169` |
| 训练脚本 | `train_fusion_169.py` |
| 特征 | 169 维 → MLP(169→64) → concat(256+64=320) |
| 架构 | 同 3.2，投影维度改为 64 |
| 参数量 | ~785K |
| 保存路径 | `cache/model2_fusion_169dim/fusion169_fold_X.pkl` |
| 日志 | `cache/model2_fusion_169dim/log_train_fusion169_2026-06-02.txt` |
| SBP MAE | **16.38 ± 1.27 mmHg** |
| DBP MAE | **7.17 ± 0.84 mmHg** |

### 3.4 共享头 + 26 维投影

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/Fusion_Baseline_26.py` — `FusionBaseline26` |
| 训练脚本 | `train_fusion_baseline_26.py` |
| 架构 | Conv1D×3 → BiLSTM → 共享注意力 → concat(256+16=272) → 共享输出头 |
| 特征 | 26 维 → MLP(26→16) |
| 参数量 | ~727K |
| 保存路径 | `cache/baseline_fusion_26dim/fusion_baseline26_fold_X.pkl` |
| 日志 | `cache/baseline_fusion_26dim/log_train_fusion_baseline26_2026-06-02.txt` |
| SBP MAE | **15.68 ± 1.23 mmHg** |
| DBP MAE | **6.76 ± 0.94 mmHg** |

### 3.5 共享头 + 169 维投影

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/Fusion_Baseline.py` — `FusionBaseline169` |
| 训练脚本 | `train_fusion_baseline.py` |
| 架构 | Conv1D×3 → BiLSTM → 共享注意力 → concat(256+64=320) → 共享输出头 |
| 特征 | 169 维 → MLP(169→64) |
| 参数量 | ~744K |
| 保存路径 | `cache/baseline_fusion_169dim/fusion_baseline_fold_X.pkl` |
| 日志 | `cache/baseline_fusion_169dim/log_train_fusion_baseline_2026-06-02.txt` |
| SBP MAE | **16.44 ± 1.08 mmHg** |
| DBP MAE | **7.18 ± 0.86 mmHg** |

### 3.6 最优子集融合（待实验）

| 项目 | 内容 |
|:---|:---|
| 模型 | `model/Fusion_Opt.py` — `FusionModelOpt` |
| 训练脚本 | `train_fusion_opt.py` |
| 特征 | SBP: sbp_opt(17→8); DBP: dbp_opt(12→6); 各自独立投影后 concat |
| 架构 | Conv1D×3 → BiLSTM → 分头注意力 → 各自 concat(256+8 / 256+6) |
| 参数量 | ~760K |
| 保存路径 | `cache/fusion_opt_fold_X.pkl` |
| 状态 | ⏳ 待运行 |
| 运行命令 | `conda run -n Pytorch python train_fusion_opt.py` |

---

## 4. 架构对比汇总

### 4.1 最终结果总表

| 序号 | 方案 | 模型 | 特征 | SBP MAE | DBP MAE | 参数量 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| ① | 纯 PPG 基线 | Model_2_Head | — | 15.78 | 6.95 | 764K |
| ② | 纯特征 MLP | MLP | 26-dim | 16.81 | 7.61 | — |
| ③ | 纯特征 MLP | MLP | 169-dim | 16.70 | 7.45 | 42K |
| ④ | 纯特征 LightGBM | LightGBM | 169-dim | 16.17 | 7.25 | — |
| ⑤ | 分头 + 26 维 org | FusionModel26 | 26 concat | 16.06 | 6.94 | 764K |
| ⑥ | 分头 + 26 维 proj | FusionModel26 | 26→16 | **15.53** | **6.86** | ~765K |
| ⑦ | 分头 + 169 维 | FusionModel169 | 169→64 | 16.38 | 7.17 | 785K |
| ⑧ | 共享头 + 26 维 | FusionBaseline26 | 26→16 | 15.68 | **6.76** | 727K |
| ⑨ | 共享头 + 169 维 | FusionBaseline169 | 169→64 | 16.44 | 7.18 | 744K |
| ⑩ | 最优子集 MLP | MLP_Opt | 17/12-dim | ⏳ | ⏳ | 4K |
| ⑪ | 最优子集融合 | FusionModelOpt | 17→8 / 12→6 | ⏳ | ⏳ | 760K |

### 4.2 各折 SBP MAE 明细

| 方案 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Model_2（纯 PPG） | ~17.25 | — | ~16.32 | ~16.03 | — | **15.78** |
| 分头+26 proj | 16.85 | 16.14 | **12.86** 🏆 | 15.76 | 16.01 | **15.53** |
| 共享头+26 | 17.35 | 16.23 | 13.59 | 15.49 | 15.74 | **15.68** |
| 分头+169 | 17.97 | 16.63 | 14.23 | 15.93 | 17.15 | **16.38** |
| 共享头+169 | 17.70 | 16.67 | 14.52 | 16.20 | 17.11 | **16.44** |

### 4.3 各折 DBP MAE 明细

| 方案 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Model_2（纯 PPG） | ~6.49 | — | ~5.95 | ~5.69 | — | **6.95** |
| 分头+26 proj | 6.49 | **8.60** ❌ | 6.28 | 5.94 | 6.98 | **6.86** |
| 共享头+26 | 6.54 | **8.48** ❌ | 6.39 | **5.62** | 6.77 | **6.76** |
| 分头+169 | 7.06 | **8.70** ❌ | 6.26 | 6.59 | 7.23 | **7.17** |
| 共享头+169 | 6.67 | **8.79** ❌ | 6.47 | 6.60 | 7.36 | **7.18** |

### 4.4 关键发现

1. **26 维融合全面优于 169 维融合** — SBP 低 ~0.8 mmHg，DBP 低 ~0.3 mmHg。原因是 169 维特征与 CNN 提取的波形特征大量重叠，而 26 维峰值/谷值统计正是 CNN 不擅长的。

2. **分头 vs 共享头差异不大** — 共享头对 DBP 略好（6.76 vs 6.86），分头对 SBP 略好（15.53 vs 15.68），差距 < 0.2 mmHg。

3. **Fold 1 DBP 是所有方案的共同难题** — 8.5-8.8 mmHg，比其他折高 2+ mmHg，可能是该折 subject 的 DBP 分布特殊。

4. **Fold 2 SBP 潜力很大** — 分头+26 proj 达到 12.86 mmHg，比基线低 2.9 mmHg，说明融合理论上可以大幅提升。

5. **特征投影 MLP 有帮助** — 26 维加投影（26→16）后 SBP 从 16.06 降到 15.53。

---

## 5. 文件索引

### 模型文件 (`model/`)

| 文件 | 类 | 说明 |
|:---|:---|:---|
| `baseline_model.py` | `BaselineModel` | 共享注意力 + 共享输出头，纯 PPG 基线 |
| `model_2.py` | `Model_2_Head` | 分头注意力 + 独立输出头，纯 PPG 基线 |
| `MLP_for_Liu2023.py` | `MLP` | 通用 MLP，用于 26 维和 169 维纯特征 |
| `MLP_Opt.py` | `MLP_Opt` | **新** 最优子集 MLP（17/12 维） |
| `LightGBM_for_Liu2023.py` | `LightGBMRegressor` | LightGBM 基线 |
| `Fusion_Model_26.py` | `FusionModel26` | 分头 + 26 维特征融合 |
| `Fusion_Model_169.py` | `FusionModel169` | 分头 + 169 维特征融合 |
| `Fusion_Baseline.py` | `FusionBaseline169` | 共享头 + 169 维特征融合 |
| `Fusion_Baseline_26.py` | `FusionBaseline26` | 共享头 + 26 维特征融合 |
| `Fusion_Opt.py` | `FusionModelOpt` | **新** 最优子集 + PPG 融合 |

### 训练脚本 (根目录)

| 文件 | 对应模型 | 说明 |
|:---|:---|:---|
| `train.py` | BaselineModel / Model_2_Head | 纯 PPG 训练（注释切换） |
| `train_for_MLP.py` | MLP | 26 维特征 MLP 训练 |
| `train_for_MLP_Liu2023.py` | MLP | 169 维特征 MLP 训练 |
| `train_for_LightGBM_Liu2023.py` | LightGBMRegressor | LightGBM 训练 |
| `train_fusion_26.py` | FusionModel26 | 分头 + 26 维融合 |
| `train_fusion_169.py` | FusionModel169 | 分头 + 169 维融合 |
| `train_fusion_baseline.py` | FusionBaseline169 | 共享头 + 169 维融合 |
| `train_fusion_baseline_26.py` | FusionBaseline26 | 共享头 + 26 维融合 |
| `train_mlp_opt.py` | MLP_Opt | **新** 最优子集 MLP |
| `train_fusion_opt.py` | FusionModelOpt | **新** 最优子集融合 |

### 数据集 (`utils/create_data.py`)

| 类 | 加载内容 | 用途 |
|:---|:---|:---|
| `PPGDataset` | PPG `[1,1024]` | 纯 PPG 训练 |
| `PPGFeatureDataset` | 26 维特征 | 26 维 MLP |
| `Liu2023FeatureDataset` | 169 维（全量 / sbp_opt / dbp_opt） | 169 维 MLP |
| `Fusion26Dataset` | PPG + 26 维特征 | 26 维融合模型 |
| `Fusion169Dataset` | PPG + 169 维特征 | 169 维融合模型 |
| `FusionOptDataset` | PPG + sbp_opt(17) + dbp_opt(12) | 最优子集融合/MLP |

### 结果目录 (`cache/`)

| 目录 | 说明 |
|:---|:---|
| `model_fusion_26dim/fusion_org/` | 分头 + 26 维直接 concat |
| `model_fusion_26dim/fusion_feature_proj/` | 分头 + 26 维投影 |
| `model2_fusion_169dim/` | 分头 + 169 维投影 |
| `baseline_fusion_26dim/` | 共享头 + 26 维投影 |
| `baseline_fusion_169dim/` | 共享头 + 169 维投影 |
| `liu2023_feature_mlp/` | 169 维特征 MLP 历史日志 |
| `liu2023lgbm/` | LightGBM 历史日志 |

---

> 最后更新: 2026-06-05
