"""
LightGBM 训练脚本 — 使用 Liu2023 169 维 PPG 特征预测 SBP/DBP.
对比 MLP 基线，验证树模型能否从特征中挖掘更多信号。

用法:
    conda run -n Pytorch python train_for_LightGBM_Liu2023.py
"""
import numpy as np
import h5py
import os
import logging
import time
from pathlib import Path

from config.config import Config
from model.LightGBM_for_Liu2023 import LightGBMRegressor
from utils.log_helper import logger_init
from sklearn.metrics import mean_absolute_error, mean_squared_error


class LightGBMConfig(Config):
    """Config overrides for LightGBM training."""
    def __init__(self):
        super().__init__()
        self.feature_set = 'full'
        # Log file and model save names
        logger_init(log_file_name='log_train_liu2023_lgbm',
                    log_level=logging.INFO,
                    log_dir=self.model_save_dir)


def load_fold_data(feature_path, fold_dir, feature_set='full'):
    """Load features and labels for a given fold."""
    with h5py.File(feature_path, "r") as f:
        fold = np.load(fold_dir)
        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]

        if feature_set == 'sbp_opt':
            feats = f["features_sbp"]
        elif feature_set == 'dbp_opt':
            feats = f["features_dbp"]
        else:
            feats = f["features"]

        X_train = feats[train_idx].astype(np.float32)
        X_val = feats[val_idx].astype(np.float32)

        y_train = np.column_stack([f["sbp"][train_idx], f["dbp"][train_idx]]).astype(np.float32)
        y_val = np.column_stack([f["sbp"][val_idx], f["dbp"][val_idx]]).astype(np.float32)

    return X_train, y_train, X_val, y_val


def train_fold(cfg, fold):
    logging.info(f"############ Fold {fold} ############")

    # Load data
    feature_path = cfg.base_dir / "liu2023_features.h5"
    fold_dir = os.path.join(cfg.base_dir, f"cv_fold_{fold}.npz")

    X_train, y_train, X_val, y_val = load_fold_data(
        feature_path, fold_dir, cfg.feature_set
    )
    logging.info(f"  Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")

    # Initialize model
    model = LightGBMRegressor()

    # Train
    start_time = time.time()
    model.fit(
        X_train, y_train, X_val, y_val,
        sbp_params_update=None, dbp_params_update=None
    )
    train_time = time.time() - start_time
    logging.info(f"  Training time: {train_time:.1f}s")

    # Evaluate
    y_pred = model.predict(X_val)

    sbp_mae = mean_absolute_error(y_val[:, 0], y_pred[:, 0])
    dbp_mae = mean_absolute_error(y_val[:, 1], y_pred[:, 1])
    sbp_rmse = np.sqrt(mean_squared_error(y_val[:, 0], y_pred[:, 0]))
    dbp_rmse = np.sqrt(mean_squared_error(y_val[:, 1], y_pred[:, 1]))

    logging.info(f"  SBP MAE = {sbp_mae:.2f} mmHg, RMSE = {sbp_rmse:.2f} mmHg")
    logging.info(f"  DBP MAE = {dbp_mae:.2f} mmHg, RMSE = {dbp_rmse:.2f} mmHg")

    # Save model
    model_save_path = os.path.join(cfg.model_save_dir,
                                   f'liu2023_lgbm_fold_{fold}.pkl')
    model.save(model_save_path)
    logging.info(f"  Model saved -> {model_save_path}")

    # Feature importance
    try:
        with h5py.File(feature_path, "r") as f:
            names_raw = f["feature_names"][:].tolist()
            feature_names = [n.decode() if isinstance(n, bytes) else n for n in names_raw]

        imp = model.feature_importance(feature_names)
        top_sbp = sorted(imp['sbp'].items(), key=lambda x: -x[1])[:10]
        top_dbp = sorted(imp['dbp'].items(), key=lambda x: -x[1])[:10]
        logging.info(f"  Top 10 SBP features: {[n for n, _ in top_sbp]}")
        logging.info(f"  Top 10 DBP features: {[n for n, _ in top_dbp]}")
    except Exception:
        pass

    return sbp_mae, dbp_mae, sbp_rmse, dbp_rmse, model.sbp_model.best_iteration_


if __name__ == '__main__':
    cfg = LightGBMConfig()

    all_sbp_mae, all_dbp_mae = [], []
    all_sbp_rmse, all_dbp_rmse = [], []

    for fold in range(cfg.k):
        sbp_mae, dbp_mae, sbp_rmse, dbp_rmse, best_iter = train_fold(cfg, fold)
        all_sbp_mae.append(sbp_mae)
        all_dbp_mae.append(dbp_mae)
        all_sbp_rmse.append(sbp_rmse)
        all_dbp_rmse.append(dbp_rmse)

        # Running average
        logging.info(
            f"  [{fold}/4] SBP MAE = {np.mean(all_sbp_mae):.2f} ± {np.std(all_sbp_mae):.2f}  "
            f"DBP MAE = {np.mean(all_dbp_mae):.2f} ± {np.std(all_dbp_mae):.2f}"
        )

    logging.info("=" * 60)
    logging.info("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    logging.info("=" * 60)

    logging.info(f"SBP MAE = {np.mean(all_sbp_mae):.2f} ± {np.std(all_sbp_mae):.2f} mmHg")
    logging.info(f"DBP MAE = {np.mean(all_dbp_mae):.2f} ± {np.std(all_dbp_mae):.2f} mmHg")
    logging.info(f"SBP RMSE = {np.mean(all_sbp_rmse):.2f} ± {np.std(all_sbp_rmse):.2f} mmHg")
    logging.info(f"DBP RMSE = {np.mean(all_dbp_rmse):.2f} ± {np.std(all_dbp_rmse):.2f} mmHg")

    print(f"\n===== LightGBM 5-Fold CV Results =====")
    print(f"SBP MAE = {np.mean(all_sbp_mae):.2f} ± {np.std(all_sbp_mae):.2f} mmHg")
    print(f"DBP MAE = {np.mean(all_dbp_mae):.2f} ± {np.std(all_dbp_mae):.2f} mmHg")
    print(f"SBP RMSE = {np.mean(all_sbp_rmse):.2f} ± {np.std(all_sbp_rmse):.2f} mmHg")
    print(f"DBP RMSE = {np.mean(all_dbp_rmse):.2f} ± {np.std(all_dbp_rmse):.2f} mmHg")
    print(f"Models saved to: {cfg.model_save_dir}")
