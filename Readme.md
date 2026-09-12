# Blood Pressure Analysis — Project File Structure Guide

This project investigates deep learning approaches for blood pressure (BP) estimation from photoplethysmography (PPG) signals using the MIMIC II database. It develops and evaluates a series of progressively more sophisticated model architectures, including pure-PPG CNNs, hand-crafted feature MLPs, fused multi-modal architectures, and attention-based models.

---

## File Tree

```
Blood_Pressure_analysis/
│
├── train.py                          # Main training (Baseline / Model 2)
├── train_for_MLP.py                  # Simple Feature MLP
├── train_for_MLP_Liu2023.py          # Liu2023 Feature MLP
├── train_for_LightGBM_Liu2023.py     # LightGBM on 169-dim features
├── train_fusion_26.py                # FusionModel26 (best model)
├── train_fusion_169.py               # FusionModel169
├── train_fusion_baseline.py          # Baseline + 169 fusion
├── train_fusion_baseline_26.py       # Baseline + 26 fusion
├── train_fusion_opt.py               # Fusion_Opt
├── train_fusion_gated26.py           # Gated Fusion 26
├── train_fusion_gated169.py          # Gated Fusion 169
├── train_fusion_ordinal.py           # CORAL ordinal (26-dim)
├── train_fusion_ordinal169.py        # CORAL ordinal (169-dim)
├── train_mlp_opt.py                  # MLP-Opt
├── Blood_pressure.ipynb              # EDA notebook
├── Visualisation.ipynb               # Visualisation notebook
├── requirements.txt
├── EXPERIMENTS.md
│
├── model/                            # PyTorch model definitions
│   ├── baseline_model.py             #   Shared-attention baseline
│   ├── model_2.py                    #   Separate-attention Model 2
│   ├── MLP_for_Liu2023.py            #   169-dim MLP
│   ├── MLP_for_feature.py            #   26-dim simple MLP
│   ├── MLP_Opt.py                    #   Task-specific MLPs
│   ├── LightGBM_for_Liu2023.py       #   LightGBM wrapper
│   ├── Fusion_Baseline.py            #   Baseline + 169 fusion
│   ├── Fusion_Baseline_26.py         #   Baseline + 26 fusion
│   ├── Fusion_Model_169.py           #   Model2 + 169 fusion
│   ├── Fusion_Model_26.py            #   ★ BEST: Model2 + 26 fusion
│   ├── Fusion_Opt.py                 #   Task-specific fusion
│   ├── Fusion_Gated.py               #   Gated fusion
│   ├── Fusion_HeadConcat.py          #   Head concat ablation
│   ├── Fusion_Ordinal.py             #   CORAL ordinal
│   ├── two_stage_bp.py               #   Two-stage pipeline
│   ├── custom_losses.py
│   └── custom_scheduler_for_transformer.py
│
├── utils/                            # Shared utilities
│   ├── create_data.py                #   CV dataset splitting
│   ├── data_helper.py                #   DataLoader & preprocessing
│   ├── liu2023_features.py           #   169-dim feature extraction
│   ├── ppg_spectrogram.py            #   Spectrogram generation
│   ├── log_helper.py                 #   Logging
│   └── stratified_sampler.py         #   Stratified CV sampling
│
├── config/
│   └── config.py                     # Paths & hyperparameters
│
├── Blood_pressure_dataset/           # MIMIC II PPG-BP data
│   ├── part_1.mat ... part_12.mat    #   12 MATLAB files (~12k recordings)
│   ├── Samples/                      #   Sample CSVs
│   └── shuffled_cv/                  #   Pre-computed CV indices
│
├── cache/                            #   Logs, results and drafts are stored here
│   ├── log_train_*.txt               #   Training logs
│   ├── baseline/                     #   Baseline experiment logs
│   ├── model2_fusion_26dim/          #   FusionModel26 logs
│   ├── model2_fusion_169dim/         #   FusionModel169 logs
│   ├── fusion_gated26/               #   Gated 26 logs
│   ├── fusion_gated169/              #   Gated 169 logs
│   ├── liu2023_feature_mlp/          #   Liu2023 MLP logs
│   ├── liu2023lgbm/                  #   LightGBM logs
│   ├── mlp_opt/                      #   MLP-Opt logs
│   ├── Fusion_ordinal/               #   CORAL logs
│   ├── figures/                      #   Generated figures
│   └── scripts_extraction_visualization/  # Plot scripts
│
├── Tests/                            # Unit & verification tests （Unit test, Only to confirm function working）
│   ├── model_test.py
│   ├── model_2_test.py
│   ├── check_fusion26.py
│   ├── check_fusion169.py
│   ├── data_loader_test.py
│   ├── data_distribution_test.py
│   ├── feature_data_analysis.py
│   └── ...
│
├── imgs/                             # Diagram sources
│   ├── cross_validation.drawio
│   ├── model.drawio
│   └── pipeline.drawio
│
└── experiments_improved/
    └── improved_loss_demo.py
```

The 


---

## Root Directory Files

| File                              | Purpose                                                                                                                                                                                |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train.py`                      | Main training entrypoint (Baseline / Model 2)                                                                                                                                          |
| `train_for_MLP.py`              | Training: Simple Feature MLP                                                                                                                                                           |
| `train_for_MLP_Liu2023.py`      | Training: Liu2023 Feature MLP                                                                                                                                                          |
| `train_for_LightGBM_Liu2023.py` | Training: LightGBM on Liu2023 features<br />(An exploratory experiment, not reported in final report)                                                                                  |
| `train_fusion_26.py`            | Training: FusionModel26 (Model2 + 26-dim fusion)                                                                                                                                       |
| `train_fusion_169.py`           | Training: FusionModel169 (Model2 + 169-dim fusion)                                                                                                                                     |
| `train_fusion_baseline.py`      | Training: Baseline + 169-dim fusion                                                                                                                                                    |
| `train_fusion_baseline_26.py`   | Training: Baseline + 26-dim fusion                                                                                                                                                     |
| `train_fusion_opt.py`           | Training: Fusion_Opt (task-specific features)                                                                                                                                          |
| `train_fusion_gated26.py`       | Training: Gated Fusion (26-dim)                                                                                                                                                        |
| `train_fusion_gated169.py`      | Training: Gated Fusion (169-dim)                                                                                                                                                       |
| `train_fusion_ordinal.py`       | Training: CORAL ordinal regression (26-dim)<br />(Exploratory, experiment was interrupted thus lack the result of fold 4. As an extension and mentioned in future work in chapter 8)  |
| `train_fusion_ordinal169.py`    | Training: CORAL ordinal regression (169-dim)<br />(Exploratory, experiment was interrupted thus lack the result of fold 4. As an extension and mentioned in future work in chapter 8) |
| `train_mlp_opt.py`              | Training: MLP-Opt (task-specific MLPs)                                                                                                                                                 |
| `Blood_pressure.ipynb`          | Jupyter notebook: exploratory data analysis & prototyping                                                                                                                              |
| `Visualisation.ipynb`           | Jupyter notebook: visualisation scripts                                                                                                                                                |
| `requirements.txt`              | Python package dependencies                                                                                                                                                            |
| `EXPERIMENTS.md`                | Experiment log index                                                                                                                                                                   |
|                                   |                                                                                                                                                                                        |

---

## `model/` — Model Architecture Definitions

Each file defines one PyTorch model class. Models are organised by increasing complexity.

| File                                    | Description                                                         |
| --------------------------------------- | ------------------------------------------------------------------- |
| `baseline_model.py`                   | Baseline CNN-BiLSTM with shared attention + shared output           |
| `model_2.py`                          | Model 2: separate SBP/DBP attention heads + output heads            |
| `MLP_for_Liu2023.py`                  | MLP on 169-dim Liu2023 features (no BN / with BN variants)          |
| `MLP_for_feature.py`                  | Simple MLP on 26-dim statistical features                           |
| `MLP_Opt.py`                          | Dual-branch MLP: SBP uses 17-dim, DBP uses 12-dim                   |
| `LightGBM_for_Liu2023.py`             | LightGBM wrapper for 169-dim features                               |
| `Fusion_Baseline.py`                  | Baseline + 169-dim feature fusion (shared attention)                |
| `Fusion_Baseline_26.py`               | Baseline + 26-dim feature fusion (shared attention)                 |
| `Fusion_Model_169.py`                 | Model2 + 169-dim fusion (separate attention)                        |
| `Fusion_Model_26.py`                  | **Model2 + 26-dim fusion (separate attention) — BEST MODEL** |
| `Fusion_Opt.py`                       | Model2 + task-specific optimal feature subsets                      |
| `Fusion_Gated.py`                     | Gated fusion mechanism (26-dim / 169-dim)                           |
| `Fusion_HeadConcat.py`                | Ablation: head concatenation fusion variant                         |
| `Fusion_Ordinal.py`                   | CORAL ordinal regression fusion model                               |
| `two_stage_bp.py`                     | Two-stage BP estimation pipeline                                    |
| `custom_losses.py`                    | Custom loss functions (log-cosh, etc.)                              |
| `custom_scheduler_for_transformer.py` | Learning rate schedulers                                            |

---

## `utils/` — Shared Utilities

| File                      | Purpose                                                       |
| ------------------------- | ------------------------------------------------------------- |
| `create_data.py`        | Dataset splitting & 5-fold CV generation                      |
| `data_helper.py`        | DataLoader, preprocessing, augmentation                       |
| `liu2023_features.py`   | 169-dim feature extraction (Liu et al. 2023)                  |
| `ppg_spectrogram.py`    | PPG spectrogram generation (Just prepared for possible ideas) |
| `log_helper.py`         | Logging utilities                                             |
| `stratified_sampler.py` | Stratified sampling for balanced CV                           |

---

## `config/` — Configuration

| File          | Purpose                                        |
| ------------- | ---------------------------------------------- |
| `config.py` | Dataset paths, hyperparameters, model settings |

---

## `Blood_pressure_dataset/` — Dataset Storage

Contains the MIMIC II PPG-BP dataset preprocessed by Kachuee et al. (2015).

```
Blood_pressure_dataset/
├── part_1.mat  ...  part_12.mat   # 12 MATLAB files, ~1,000 recordings each
├── Samples/                        # Pre-split sample files
└── shuffled_cv/                    # Pre-computed 5-fold CV indices
```

---

## `cache/` — Training Logs & Experiment Results

Each experiment variant has its own subdirectory containing `log_train_*.txt` with full per-epoch and per-fold metrics.

```
cache/
├── log_train_*.txt                    # Root-level training logs
├── baseline/                          # Original & improved Baseline
├── model2/                            # Model 2 (separate attention)
├── model2_fusion_26dim/               # FusionModel26 ★ best model
├── model2_fusion_169dim/              # FusionModel169
├── model2_fusion_opt/                 # Fusion_Opt (task-specific features)
├── baseline_fusion_26dim/             # Baseline + 26-dim fusion
├── baseline_fusion_169dim/            # Baseline + 169-dim fusion
├── fusion_gated26/                    # Gated fusion (26-dim)
├── fusion_gated169/                   # Gated fusion (169-dim)
├── Fusion_ordinal/                    # CORAL ordinal regression
├── liu2023_feature_mlp/               # Liu2023 MLP (2 variants)
├── liu2023lgbm/                       # LightGBM
├── mlp_opt/                           # MLP-Opt
├── 简单特征提取+mlp/                   # Simple Feature MLP
└── figures/                           # Generated plots (.png)
```

## `Tests/` — Unit & Verification Tests

Scripts to verify model forward passes, data loading, feature extraction, and dataset distribution analysis.

## `imgs/` — Diagram Sources (.drawio)

Pipeline, model architecture, and cross-validation diagrams.

## `experiments_improved/` — Experimental Loss Variants

## `tmp/` — Ad-hoc Helper Scripts

One-off scripts for docx generation, figure plotting, and model verification.

---

## Model Evolution Summary

| Stage                               | Model File                       | Description                                         |
| ----------------------------------- | -------------------------------- | --------------------------------------------------- |
| 1. Pure-PPG Baseline                | `baseline_model.py`            | CNN+BiLSTM+shared attention+shared output           |
| 2. Pure-PPG Improved                | `model_2.py`                   | Separate SBP/DBP attention + output heads           |
| 3. Feature MLP (no BN)              | `MLP_for_Liu2023.py`           | 169-dim MLP, dropout 0.3, no BN                     |
| 4. Feature MLP (BN)                 | `MLP_for_Liu2023.py`           | 169-dim MLP, BN, dropout 0.1                        |
| 5. Simple Feature MLP               | `MLP_for_feature.py`           | 26-dim MLP, minimal architecture                    |
| 6. MLP-Opt                          | `MLP_Opt.py`                   | SBP 17-dim + DBP 12-dim, separate MLPs              |
| 7. LightGBM                         | `LightGBM_for_Liu2023.py`      | Gradient boosting on 169-dim                        |
| 8. Baseline + 26 Fusion             | `Fusion_Baseline_26.py`        | Shared attention + 26-dim fusion                    |
| 9. Baseline + 169 Fusion            | `Fusion_Baseline.py`           | Shared attention + 169-dim fusion                   |
| 10. Model2 + 169 Fusion             | `Fusion_Model_169.py`          | Separate attention + 169-dim fusion                 |
| **11. Model2 + 26 Fusion ★** | **`Fusion_Model_26.py`** | **Separate attention + 26-dim fusion [BEST]** |
| 12. Fusion_Opt                      | `Fusion_Opt.py`                | Task-specific optimal feature subsets               |
| 13. Gated Fusion                    | `Fusion_Gated.py`              | Gated fusion (26/169-dim)                           |
| 14. CORAL Ordinal                   | `Fusion_Ordinal.py`            | Ordinal regression fusion                           |

---

## Key Results (Best Model: FusionModel26)

| Metric         | SBP             | DBP            |
| -------------- | --------------- | -------------- |
| MAE (mmHg)     | 15.53 ± 11.77  | 6.86 ± 6.38   |
| RMSE (mmHg)    | 19.49           | 9.37           |
| Pearson R      | 0.391           | 0.360          |
| Bias (mmHg)    | −0.11 ± 19.49 | −0.90 ± 9.32 |
| 95% LoA (mmHg) | ±38.64         | ±18.02        |

**Notes:**

- All models use 5-fold cross-validation on ~517k samples from MIMIC II.
- PPG input is 1×1024 raw signal sampled at 125 Hz.
- 26-dim features are peak/valley statistics (pulse width, amplitude, heart rate, etc.).
- 169-dim features follow the Liu et al. (2023) framework (PPG geometric + VPG/APG features).
- Reference: Liu et al., *Biomedical Signal Processing and Control*, Vol. 86, 2023.
