"""
Training script — Ordinal169: PPG + 169-dim features + Soft Ordinal Regression

Based on train_fusion_ordinal.py:
  - Model: SoftOrdinalFusion (169-dim input, 64-dim projection, CORAL output)
  - Loss: OrdinalLoss (CORAL + MAE hybrid)
  - Save: cache/ordinal169_fold_X.pkl
  - Log:  log_train_ordinal169
"""
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from config.config import Config
from model.Fusion_Ordinal import SoftOrdinalFusion, OrdinalLoss
import torch.nn.functional as F
from utils.create_data import LoadFusion169Dataset
from utils.log_helper import logger_init


class Ordinal169Config(Config):
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
        logger_init(log_file_name='log_train_ordinal169',
                    log_level=logging.INFO,
                    log_dir=self.model_save_dir)
        self.feature_dim = 169
        self.proj_dim = 64
        self.h5_feature = self.base_dir / "liu2023_features.h5"


def train_fold(cfg, fold):
    logging.info(f"######## Fold {fold} ########")

    data_loader = LoadFusion169Dataset(batch_size=cfg.batch_size)
    fold_dir = os.path.join(cfg.base_dir, f"cv_fold_{fold}.npz")

    train_iter, val_iter, fold_sbp_mean, fold_sbp_std, fold_dbp_mean, fold_dbp_std = \
        data_loader.load_train_val_data(
            ppg_path=str(cfg.datadir),
            feat_path=str(cfg.h5_feature),
            indices_dir=fold_dir
        )

    # ---- Compute quantile-based bin edges from training data ----
    all_y = []
    for _, y in train_iter:
        all_y.append(y)
    all_y = torch.cat(all_y, dim=0)
    sbp_norm = all_y[:, 0].numpy()
    dbp_norm = all_y[:, 1].numpy()
    quantiles = np.linspace(0, 1, 8 + 1)
    sbp_edges = torch.tensor(np.quantile(sbp_norm, quantiles), dtype=torch.float32)
    dbp_edges = torch.tensor(np.quantile(dbp_norm, quantiles), dtype=torch.float32)
    sbp_centers = (sbp_edges[:-1] + sbp_edges[1:]) / 2
    dbp_centers = (dbp_edges[:-1] + dbp_edges[1:]) / 2
    logging.info(f"SBP edges: {sbp_edges.numpy().round(3)}")
    logging.info(f"DBP edges: {dbp_edges.numpy().round(3)}")

    logging.info("############ Building SoftOrdinalFusion (169-dim) ############")
    model = SoftOrdinalFusion(
        sbp_edges=sbp_edges, dbp_edges=dbp_edges,
        filters=cfg.filters,
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        drop_prob=cfg.dropout,
        feature_dim=cfg.feature_dim,
        proj_dim=cfg.proj_dim,
        num_bins=8,
    )
    model_save_path = os.path.join(cfg.model_save_dir, f'ordinal169_fold_{fold}.pkl')
    model = model.to(cfg.device)

    loss_fn = OrdinalLoss(
        sbp_edges=sbp_edges, dbp_edges=dbp_edges,
        w_ord=1.0,
        w_reg=0.2,
    ).to(cfg.device)

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
            pred = model(x_ppg, x_feat)
            g_sbp, g_dbp = model.get_logits(x_ppg, x_feat)
            loss = loss_fn(pred, y, g_sbp, g_dbp)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                losses += loss.item()
                sbp_losses += F.l1_loss(pred[:, 0], y[:, 0]).item()
                dbp_losses += F.l1_loss(pred[:, 1], y[:, 1]).item()
                mmHg_sbp_losses += F.l1_loss(pred[:, 0], y[:, 0]).item() * fold_sbp_std
                mmHg_dbp_losses += F.l1_loss(pred[:, 1], y[:, 1]).item() * fold_dbp_std

        end_time = time.time()
        n_train = len(train_iter)
        logging.info(
            f"Epoch: {epoch}, Train loss: {losses/n_train:.3f}, "
            f"SBP MAE: {sbp_losses/n_train:.3f}, mmHg: {mmHg_sbp_losses/n_train:.3f}, "
            f"DBP MAE: {dbp_losses/n_train:.3f}, mmHg: {mmHg_dbp_losses/n_train:.3f}, "
            f"Time: {end_time - start_time:.1f}s"
        )

        val_metrics = evaluate(cfg.device, val_iter, loss_fn, model,
                               fold_sbp_std, fold_dbp_std)
        (val_loss, _, _, _, _,
         val_mae, val_sbp_mae, val_dbp_mae,
         val_sbp_mae_mmHg, val_dbp_mae_mmHg) = val_metrics

        scheduler.step(val_loss)

        logging.info(
            f"Epoch {epoch}, Val loss: {val_loss:.3f}, "
            f"Val SBP MAE: {val_sbp_mae:.3f}, mmHg: {val_sbp_mae_mmHg:.3f}, "
            f"Val DBP MAE: {val_dbp_mae:.3f}, mmHg: {val_dbp_mae_mmHg:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            state_dict = deepcopy(model.state_dict())
            torch.save(state_dict, model_save_path); logging.info("Best model saved")
            patience_count = 0
        else:
            patience_count += 1

        if patience_count > cfg.early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    return best_metrics


def evaluate(device, val_iter, loss_fn, model, fold_sbp_std, fold_dbp_std):
    model.eval()
    val_loss = 0; val_sbp_loss = 0; val_dbp_loss = 0
    val_mae = 0; val_sbp_mae = 0; val_dbp_mae = 0
    val_sbp_loss_mmHg = 0; val_dbp_loss_mmHg = 0
    val_sbp_mae_mmHg = 0; val_dbp_mae_mmHg = 0

    with torch.no_grad():
        for (x_ppg, x_feat), y in tqdm(val_iter, total=len(val_iter),
                                        desc="Evaluating", colour="cyan"):
            x_ppg = x_ppg.to(device); x_feat = x_feat.to(device)
            y = y.to(device)
            pred = model(x_ppg, x_feat)
            g_sbp, g_dbp = model.get_logits(x_ppg, x_feat)
            loss = loss_fn(pred, y, g_sbp, g_dbp)

            val_loss += loss.item()
            val_sbp_loss += F.l1_loss(pred[:, 0], y[:, 0]).item()
            val_dbp_loss += F.l1_loss(pred[:, 1], y[:, 1]).item()
            val_sbp_loss_mmHg += F.l1_loss(pred[:, 0], y[:, 0]).item() * fold_sbp_std
            val_dbp_loss_mmHg += F.l1_loss(pred[:, 1], y[:, 1]).item() * fold_dbp_std

            mae_sbp = torch.mean(torch.abs(pred[:, 0] - y[:, 0])).item()
            mae_dbp = torch.mean(torch.abs(pred[:, 1] - y[:, 1])).item()
            val_mae += (mae_sbp + mae_dbp) / 2
            val_sbp_mae += mae_sbp
            val_dbp_mae += mae_dbp
            val_sbp_mae_mmHg += mae_sbp * fold_sbp_std
            val_dbp_mae_mmHg += mae_dbp * fold_dbp_std

    n_val = len(val_iter)
    model.train()
    return (
        val_loss / n_val, val_sbp_loss / n_val, val_dbp_loss / n_val,
        val_sbp_loss_mmHg / n_val, val_dbp_loss_mmHg / n_val,
        val_mae / n_val, val_sbp_mae / n_val, val_dbp_mae / n_val,
        val_sbp_mae_mmHg / n_val, val_dbp_mae_mmHg / n_val,
    )


if __name__ == '__main__':
    cfg = Ordinal169Config()

    all_fold_mae = []
    all_fold_SBP_mae = []
    all_fold_DBP_mae = []
    all_fold_SBP_mae_mmHg = []
    all_fold_DBP_mae_mmHg = []

    for fold in range(cfg.k):
        metrics = train_fold(cfg, fold)
        (_, _, _, _, _,
         val_mae, val_sbp_mae, val_dbp_mae,
         val_sbp_mae_mmHg, val_dbp_mae_mmHg) = metrics

        all_fold_mae.append(val_mae)
        all_fold_SBP_mae.append(val_sbp_mae)
        all_fold_DBP_mae.append(val_dbp_mae)
        all_fold_SBP_mae_mmHg.append(val_sbp_mae_mmHg)
        all_fold_DBP_mae_mmHg.append(val_dbp_mae_mmHg)

        n = len(all_fold_mae)
        logging.info(f"{n} fold SBP MAE = {np.mean(all_fold_SBP_mae_mmHg):.3f} +/- {np.std(all_fold_SBP_mae_mmHg):.3f} mmHg")
        logging.info(f"{n} fold DBP MAE = {np.mean(all_fold_DBP_mae_mmHg):.3f} +/- {np.std(all_fold_DBP_mae_mmHg):.3f} mmHg")

    logging.info("=" * 60)
    logging.info("FINAL 5-FOLD RESULTS (Ordinal169)")
    logging.info("=" * 60)
    logging.info(f"SBP MAE = {np.mean(all_fold_SBP_mae_mmHg):.2f} +/- {np.std(all_fold_SBP_mae_mmHg):.2f} mmHg")
    logging.info(f"DBP MAE = {np.mean(all_fold_DBP_mae_mmHg):.2f} +/- {np.std(all_fold_DBP_mae_mmHg):.2f} mmHg")

    print(f"\n===== Ordinal169 5-Fold CV Results =====")
    print(f"SBP MAE = {np.mean(all_fold_SBP_mae_mmHg):.2f} +/- {np.std(all_fold_SBP_mae_mmHg):.2f} mmHg")
    print(f"DBP MAE = {np.mean(all_fold_DBP_mae_mmHg):.2f} +/- {np.std(all_fold_DBP_mae_mmHg):.2f} mmHg")

