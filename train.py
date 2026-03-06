from copy import deepcopy
from config.config import Config
from model.baseline_model import BaselineModel
from model.model_2 import Model_2_Head
from utils.data_helper import chain
from utils.create_data import LoadPPGDataset
from tqdm import tqdm
import torch
import numpy as np
import time
import os
import logging


def train_model(cfg, fold):
    logging.info("############划分数据集并保存############")
    if not os.path.exists(cfg.h5_seg):
        chain(base_dir=cfg.base_dir, h5_detrend=cfg.h5_detrend, h5_scaled=cfg.h5_scaled, h5_seg=cfg.h5_seg, k=cfg.k)

    logging.info("############载入划分好的数据集############")
    data_loader = LoadPPGDataset(batch_size=cfg.batch_size)

    fold_dir = os.path.join(cfg.base_dir, f"cv_fold_{fold}.npz")
    train_iter, val_iter, fold_sbp_mean, fold_sbp_std, fold_dbp_mean, fold_dbp_std = data_loader.load_train_val_data(data_dir=cfg.datadir, indices_dir=fold_dir)

    logging.info("############初始化模型############")
    model = BaselineModel(filters=cfg.filters, num_layers=cfg.num_layers)
    # model = Model_2_Head(filters=cfg.filters, num_layers=cfg.num_layers)
    
    model_save_path = os.path.join(cfg.model_save_dir, f'model_fold_{fold}.pkl')

    model = model.to(cfg.device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=cfg.lr,
                                 betas=(cfg.beta1, cfg.beta2),
                                 eps=cfg.epsilon)
    # 加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True)
    
    best_val_loss = float("inf") # 只用sbp和dbp整体的mse作为Loss判断模型性能
    best_metrics = None
    early_stopping_patience = cfg.early_stopping_patience
    patience_count = 0
    model.train()

    for epoch in range(cfg.epochs):
        losses = 0
        sbp_losses = 0
        dbp_losses = 0
        mmHg_sbp_losses = 0
        mmHg_dbp_losses = 0

        start_time = time.time()

        for idx, (x, y) in enumerate(tqdm(train_iter, 
                                          total=len(train_iter),
                                          desc=f"Epoch {epoch} Training",
                                          colour="cyan")):
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            optimizer.zero_grad()
            outputs = model(x)
            sbp_loss = loss_fn(outputs[:, 0], y[:, 0])
            dbp_loss = loss_fn(outputs[:, 1], y[:, 1])
            loss = sbp_loss + dbp_loss
            loss.backward()
            optimizer.step()
            # 修改，断开梯度计算
            with torch.no_grad():
                losses += loss.item()
                sbp_losses += sbp_loss.item()
                dbp_losses += dbp_loss.item()
                # 反标准化mse
                mmHg_sbp_losses += sbp_loss.item() * fold_sbp_std ** 2
                mmHg_dbp_losses += dbp_loss.item() * fold_dbp_std ** 2

                # msg = f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], Train loss: {loss.item():.3f}, SBP loss: {sbp_loss.item():.3f}, DBP loss: {dbp_loss.item():.3f}"
                # logging.info(msg)

        end_time = time.time()
        train_loss = losses / len(train_iter)
        train_sbp_loss = sbp_losses / len(train_iter)
        train_dbp_loss = dbp_losses / len(train_iter)
        train_sbp_mmHg_loss = mmHg_sbp_losses / len(train_iter)
        train_dbp_mmHg_loss = mmHg_dbp_losses / len(train_iter)

        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, SBP loss: {train_sbp_loss:.3f}, In mmHg: {train_sbp_mmHg_loss:.3f}, DBP loss: {train_dbp_loss:.3f}, In mmHg:{train_dbp_mmHg_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"
        logging.info(msg)

        # Evaluate
        val_loss, val_sbp_loss, val_dbp_loss, val_sbp_loss_mmHg, val_dbp_loss_mmHg, val_mae, val_sbp_mae, val_dbp_mae, val_sbp_mae_mmHg, val_dbp_mae_mmHg = evaluate(cfg.device, val_iter, loss_fn, model, fold_sbp_std, fold_dbp_std)
        
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch}, Val loss: {val_loss:.3f}, Val SBP loss: {val_sbp_loss:.3f}, In mmHg: {val_sbp_loss_mmHg:.3f}, Val DBP loss: {val_dbp_loss:.3f}, In mmHg: {val_dbp_loss_mmHg:.3f}\
                           Val MAE: {val_mae:.3f}, Val SBP MAE: {val_sbp_mae:.3f}, In mmHg: {val_sbp_mae_mmHg:.3f}, Val DBP MAE: {val_dbp_mae:.3f}, In mmHg: {val_dbp_mae_mmHg:.3f}")
        if val_loss < best_val_loss: # 保存最佳模型参数和metrics
            best_val_loss = val_loss
            best_metrics = (val_loss, val_sbp_loss, val_dbp_loss, val_sbp_loss_mmHg, val_dbp_loss_mmHg, val_mae, val_sbp_mae, val_dbp_mae, val_sbp_mae_mmHg, val_dbp_mae_mmHg)
            state_dict = deepcopy(model.state_dict())
            torch.save(state_dict, model_save_path)
            logging.info("Best model saved")
            patience_count = 0
        else:
            patience_count += 1
        
        if patience_count > early_stopping_patience:
            logging.info("Early stopping at epoch {epoch}")
            break

    return best_metrics


def evaluate(device, val_iter, loss_fn, model, fold_sbp_std, fold_dbp_std):
    model.eval()

    val_loss = 0
    val_sbp_loss = 0
    val_dbp_loss = 0
    val_mae = 0
    val_sbp_mae = 0
    val_dbp_mae = 0
    val_sbp_loss_mmHg = 0
    val_dbp_loss_mmHg = 0
    val_sbp_mae_mmHg = 0
    val_dbp_mae_mmHg = 0

    with torch.no_grad():
        for x, y in tqdm(val_iter, 
                         total=len(val_iter), 
                         desc=f"Evalueting",
                         colour="cyan"):
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
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
            mae = (mae_sbp + mae_dbp) / 2
            val_mae += mae
            val_sbp_mae += mae_sbp
            val_dbp_mae += mae_dbp
            val_sbp_mae_mmHg += mae_sbp * fold_sbp_std
            val_dbp_mae_mmHg += mae_dbp * fold_dbp_std
    
    val_loss = val_loss / len(val_iter)
    val_sbp_loss = val_sbp_loss / len(val_iter)
    val_dbp_loss = val_dbp_loss / len(val_iter)
    val_sbp_loss_mmHg = val_sbp_loss_mmHg / len(val_iter)
    val_dbp_loss_mmHg = val_dbp_loss_mmHg / len(val_iter)

    val_mae = val_mae / len(val_iter)
    val_sbp_mae = val_sbp_mae / len(val_iter)
    val_dbp_mae = val_dbp_mae / len(val_iter)
    val_sbp_mae_mmHg = val_sbp_mae_mmHg / len(val_iter)
    val_dbp_mae_mmHg = val_dbp_mae_mmHg / len(val_iter)

    model.train()
    return val_loss, val_sbp_loss, val_dbp_loss, val_sbp_loss_mmHg, val_dbp_loss_mmHg, val_mae, val_sbp_mae, val_dbp_mae, val_sbp_mae_mmHg, val_dbp_mae_mmHg


if __name__ == '__main__':
    cfg = Config()

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
        logging.info(f"######## Fold {fold} ########")
        (val_loss, val_sbp_loss, val_dbp_loss, val_sbp_loss_mmHg, val_dbp_loss_mmHg, val_mae, val_sbp_mae, val_dbp_mae, val_sbp_mae_mmHg, val_dbp_mae_mmHg) = train_model(cfg, fold)
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

        logging.info(f"{fold} fold MAE: {np.mean(all_fold_mae):.3f} ± {np.std(all_fold_mae):.3f}")
        logging.info(f"{fold} fold SBP MAE: {np.mean(all_fold_SBP_mae):.3f} ± {np.std(all_fold_SBP_mae):.3f}")
        logging.info(f"{fold} fold DBP MAE: {np.mean(all_fold_DBP_mae):.3f} ± {np.std(all_fold_DBP_mae):.3f}")
        logging.info(f"{fold} fold SBP MAE in mmHg: {np.mean(all_fold_SBP_mae_mmHg):.3f} ± {np.std(all_fold_SBP_mae_mmHg):.3f}")
        logging.info(f"{fold} fold DBP MAE in mmHg: {np.mean(all_fold_DBP_mae_mmHg):.3f} ± {np.std(all_fold_DBP_mae_mmHg):.3f}")

        logging.info(f"{fold} fold MSE: {np.mean(all_fold_loss):.3f} ± {np.std(all_fold_loss):.3f}")
        logging.info(f"{fold} fold SBP MSE: {np.mean(all_fold_SBP_loss):.3f} ± {np.std(all_fold_SBP_loss):.3f}")
        logging.info(f"{fold} fold DBP MSE: {np.mean(all_fold_DBP_loss):.3f} ± {np.std(all_fold_DBP_loss):.3f}")
        logging.info(f"{fold} fold SBP MSE in mmHg: {np.mean(all_fold_SBP_loss_mmHg):.3f} ± {np.std(all_fold_SBP_loss_mmHg):.3f}")
        logging.info(f"{fold} fold DBP MSE in mmHg: {np.mean(all_fold_DBP_loss_mmHg):.3f} ± {np.std(all_fold_DBP_loss_mmHg):.3f}")

