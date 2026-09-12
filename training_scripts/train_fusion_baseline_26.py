"""
训练脚本 — FusionBaseline26: PPG 时序路 + 26-dim 统计特征融合 (共享头)

模型:
  - FusionBaseline26 (CNN-BiLSTM-共享注意力 + 26-dim 特征 MLP 投影后 concat)
  - 投影维度: 26→16, 融合维度: 256 + 16 = 272
  - 共享输出头 (SBP/DBP 一起输出)

用于对比 FusionModel26 (分头方案)，验证共享头对 26 维融合的效果。

保存:
  - 模型: cache/fusion_baseline26_fold_X.pkl
  - 日志: log_train_fusion_baseline26

用法:
    conda run -n Pytorch python train_fusion_baseline_26.py
"""
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from config.config import Config
from model.Fusion_Baseline_26 import FusionBaseline26
from utils.create_data import LoadFusion26Dataset
from utils.log_helper import logger_init


class FusionBaseline26Config(Config):
    """Config for baseline-style 26-dim fusion."""
    def __init__(self):
        super().__init__()
        _old_paths = []
        for _h in logging.root.handlers[:]:
            if hasattr(_h, 'baseFilename'):
                _old_paths.append(_h.baseFilename)
            _h.close()
            logging.root.removeHandler(_h)
        for _p in _old_paths:
            if os.path.exists(_p):
                os.remove(_p)
        logger_init(log_file_name='log_train_fusion_baseline26',
                    log_level=logging.INFO,
                    log_dir=self.model_save_dir)
        self.proj_dim = 16
        self.h5_feature = self.base_dir / "ppg_features.h5"


def train_fold(cfg, fold):
    logging.info(f"######## Fold {fold} ########")
    logging.info("############ 载入融合数据集 (PPG + 26-dim features) ############")

    data_loader = LoadFusion26Dataset(batch_size=cfg.batch_size)
    fold_dir = os.path.join(cfg.base_dir, f"cv_fold_{fold}.npz")

    train_iter, val_iter, fold_sbp_mean, fold_sbp_std, fold_dbp_mean, fold_dbp_std = \
        data_loader.load_train_val_data(
            ppg_path=str(cfg.datadir),
            feat_path=str(cfg.h5_feature),
            indices_dir=fold_dir
        )

    logging.info("############ 初始化 FusionBaseline26 ############")
    model = FusionBaseline26(
        filters=cfg.filters,
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        drop_prob=cfg.dropout,
        proj_dim=cfg.proj_dim,
    )
    model_save_path = os.path.join(cfg.model_save_dir, f'fusion_baseline26_fold_{fold}.pkl')
    model = model.to(cfg.device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.epsilon
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6,
        min_lr=1e-6, verbose=True
    )

    best_val_loss = float("inf")
    best_metrics = None
    patience_count = 0

    for epoch in range(cfg.epochs):
        # ---- Training ----
        model.train()
        losses, sbp_losses, dbp_losses = 0, 0, 0
        mmHg_sbp_losses, mmHg_dbp_losses = 0, 0
        start_time = time.time()

        for (x_ppg, x_feat), y in tqdm(train_iter, total=len(train_iter),
                                        desc=f"Epoch {epoch} Training",
                                        colour="cyan"):
            x_ppg = x_ppg.to(cfg.device)
            x_feat = x_feat.to(cfg.device)
            y = y.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(x_ppg, x_feat)

            sbp_loss = loss_fn(outputs[:, 0], y[:, 0])
            dbp_loss = loss_fn(outputs[:, 1], y[:, 1])
            loss = sbp_loss + dbp_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                losses += loss.item()
                sbp_losses += sbp_loss.item()
                dbp_losses += dbp_loss.item()
                mmHg_sbp_losses += sbp_loss.item() * fold_sbp_std ** 2
                mmHg_dbp_losses += dbp_loss.item() * fold_dbp_std ** 2

        end_time = time.time()
        n_train = len(train_iter)
        train_loss = losses / n_train
        train_sbp_loss = sbp_losses / n_train
        train_dbp_loss = dbp_losses / n_train
        train_sbp_mmHg = mmHg_sbp_losses / n_train
        train_dbp_mmHg = mmHg_dbp_losses / n_train

        logging.info(
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
            f"SBP loss: {train_sbp_loss:.3f}, In mmHg: {train_sbp_mmHg:.3f}, "
            f"DBP loss: {train_dbp_loss:.3f}, In mmHg: {train_dbp_mmHg:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )

        # ---- Evaluation ----
        val_metrics = evaluate(cfg.device, val_iter, loss_fn, model,
                               fold_sbp_std, fold_dbp_std)
        (val_loss, val_sbp_loss, val_dbp_loss,
         val_sbp_loss_mmHg, val_dbp_loss_mmHg,
         val_mae, val_sbp_mae, val_dbp_mae,
         val_sbp_mae_mmHg, val_dbp_mae_mmHg) = val_metrics

        scheduler.step(val_loss)

        logging.info(
            f"Epoch {epoch}, Val loss: {val_loss:.3f}, "
            f"Val SBP loss: {val_sbp_loss:.3f}, In mmHg: {val_sbp_loss_mmHg:.3f}, "
            f"Val DBP loss: {val_dbp_loss:.3f}, In mmHg: {val_dbp_loss_mmHg:.3f}  "
            f"Val MAE: {val_mae:.3f}, Val SBP MAE: {val_sbp_mae:.3f}, "
            f"In mmHg: {val_sbp_mae_mmHg:.3f}, Val DBP MAE: {val_dbp_mae:.3f}, "
            f"In mmHg: {val_dbp_mae_mmHg:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            state_dict = deepcopy(model.state_dict())
            torch.save(state_dict, model_save_path)
            logging.info("Best model saved")
            patience_count = 0
        else:
            patience_count += 1

        if patience_count > cfg.early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    return best_metrics


def evaluate(device, val_iter, loss_fn, model, fold_sbp_std, fold_dbp_std):
    model.eval()
    val_loss = val_sbp_loss = val_dbp_loss = 0
    val_sbp_loss_mmHg = val_dbp_loss_mmHg = 0
    val_mae = val_sbp_mae = val_dbp_mae = 0
    val_sbp_mae_mmHg = val_dbp_mae_mmHg = 0

    with torch.no_grad():
        for (x_ppg, x_feat), y in tqdm(val_iter, total=len(val_iter),
                                        desc="Evaluating", colour="cyan"):
            x_ppg = x_ppg.to(device); x_feat = x_feat.to(device); y = y.to(device)
            outputs = model(x_ppg, x_feat)
            loss_sbp = loss_fn(outputs[:, 0], y[:, 0])
            loss_dbp = loss_fn(outputs[:, 1], y[:, 1])
            loss = loss_sbp + loss_dbp
            val_loss += loss.item()
            val_sbp_loss += loss_sbp.item()
            val_dbp_loss += loss_dbp.item()
            val_sbp_loss_mmHg += loss_sbp.item() * fold_sbp_std ** 2
            val_dbp_loss_mmHg += loss_dbp.item() * fold_dbp_std ** 2
            mae_sbp = torch.mean(torch.abs(outputs[:, 0] - y[:, 0])).item()
            mae_dbp = torch.mean(torch.abs(outputs[:, 1] - y[:, 1])).item()
            val_mae += (mae_sbp + mae_dbp) / 2
            val_sbp_mae += mae_sbp
            val_dbp_mae += mae_dbp
            val_sbp_mae_mmHg += mae_sbp * fold_sbp_std
            val_dbp_mae_mmHg += mae_dbp * fold_dbp_std

    n_val = len(val_iter)
    model.train()
    return (val_loss/n_val, val_sbp_loss/n_val, val_dbp_loss/n_val,
            val_sbp_loss_mmHg/n_val, val_dbp_loss_mmHg/n_val,
            val_mae/n_val, val_sbp_mae/n_val, val_dbp_mae/n_val,
            val_sbp_mae_mmHg/n_val, val_dbp_mae_mmHg/n_val)


if __name__ == '__main__':
    cfg = FusionBaseline26Config()

    all_fold_mae = []
    all_fold_SBP_mae = []
    all_fold_DBP_mae = []
    all_fold_SBP_mae_mmHg = []
    all_fold_DBP_mae_mmHg = []
    all_fold_loss = []
    all_fold_SBP_loss = []
    all_fold_DBP_loss = []
    all_fold_SBP_loss_mmHg = []
    all_fold_DBP_loss_mmHg = []

    for fold in range(cfg.k):
        metrics = train_fold(cfg, fold)
        (val_loss, val_sbp_loss, val_dbp_loss,
         val_sbp_loss_mmHg, val_dbp_loss_mmHg,
         val_mae, val_sbp_mae, val_dbp_mae,
         val_sbp_mae_mmHg, val_dbp_mae_mmHg) = metrics

        all_fold_mae.append(val_mae)
        all_fold_SBP_mae.append(val_sbp_mae)
        all_fold_DBP_mae.append(val_dbp_mae)
        all_fold_SBP_mae_mmHg.append(val_sbp_mae_mmHg)
        all_fold_DBP_mae_mmHg.append(val_dbp_mae_mmHg)
        all_fold_loss.append(val_loss)
        all_fold_SBP_loss.append(val_sbp_loss)
        all_fold_DBP_loss.append(val_dbp_loss)
        all_fold_SBP_loss_mmHg.append(val_sbp_loss_mmHg)
        all_fold_DBP_loss_mmHg.append(val_dbp_loss_mmHg)

        n = len(all_fold_mae)
        logging.info(f"{n} fold MAE: {np.mean(all_fold_mae):.3f} ± {np.std(all_fold_mae):.3f}")
        logging.info(f"{n} fold SBP MAE: {np.mean(all_fold_SBP_mae):.3f} ± {np.std(all_fold_SBP_mae):.3f}")
        logging.info(f"{n} fold DBP MAE: {np.mean(all_fold_DBP_mae):.3f} ± {np.std(all_fold_DBP_mae):.3f}")
        logging.info(f"{n} fold SBP MAE in mmHg: {np.mean(all_fold_SBP_mae_mmHg):.3f} ± {np.std(all_fold_SBP_mae_mmHg):.3f}")
        logging.info(f"{n} fold DBP MAE in mmHg: {np.mean(all_fold_DBP_mae_mmHg):.3f} ± {np.std(all_fold_DBP_mae_mmHg):.3f}")

    logging.info("=" * 60)
    logging.info("FINAL 5-FOLD CV RESULTS (FusionBaseline26)")
    logging.info("=" * 60)
    logging.info(f"SBP MAE = {np.mean(all_fold_SBP_mae_mmHg):.2f} ± {np.std(all_fold_SBP_mae_mmHg):.2f} mmHg")
    logging.info(f"DBP MAE = {np.mean(all_fold_DBP_mae_mmHg):.2f} ± {np.std(all_fold_DBP_mae_mmHg):.2f} mmHg")
    logging.info(f"SBP RMSE = {np.sqrt(np.mean(all_fold_SBP_loss_mmHg)):.2f} ± {np.std(np.sqrt(np.array(all_fold_SBP_loss_mmHg))):.2f} mmHg")
    logging.info(f"DBP RMSE = {np.sqrt(np.mean(all_fold_DBP_loss_mmHg)):.2f} ± {np.std(np.sqrt(np.array(all_fold_DBP_loss_mmHg))):.2f} mmHg")

    print(f"\n===== FusionBaseline26 5-Fold CV Results =====")
    print(f"SBP MAE = {np.mean(all_fold_SBP_mae_mmHg):.2f} ± {np.std(all_fold_SBP_mae_mmHg):.2f} mmHg")
    print(f"DBP MAE = {np.mean(all_fold_DBP_mae_mmHg):.2f} ± {np.std(all_fold_DBP_mae_mmHg):.2f} mmHg")
    print(f"Models saved to: {cfg.model_save_dir}/fusion_baseline26_fold_*.pkl")
