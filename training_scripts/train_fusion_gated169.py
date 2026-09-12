"""
训练脚本 — FusionGated (169-dim): 门控融合 169 维特征

保存: cache/fusion_gated169_fold_X.pkl
日志: log_train_fusion_gated169

用法:
    conda run -n Pytorch python train_fusion_gated169.py
"""
import logging, os, time
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm

from config.config import Config
from model.Fusion_Gated import FusionGated
from utils.create_data import LoadFusion169Dataset
from utils.log_helper import logger_init


class Gated169Config(Config):
    def __init__(self):
        super().__init__()
        _old = []
        for h in logging.root.handlers[:]:
            if hasattr(h, 'baseFilename'): _old.append(h.baseFilename)
            h.close(); logging.root.removeHandler(h)
        for p in _old:
            if os.path.exists(p): os.remove(p)
        logger_init(log_file_name='log_train_fusion_gated169', log_level=logging.INFO, log_dir=self.model_save_dir)
        self.feature_dim = 169
        self.proj_dim = 64
        self.liu2023_h5 = self.base_dir / "liu2023_features.h5"


def train_fold(cfg, fold):
    logging.info(f"######## Fold {fold} ########")
    dl = LoadFusion169Dataset(batch_size=cfg.batch_size)
    fold_dir = os.path.join(cfg.base_dir, f"cv_fold_{fold}.npz")
    train_iter, val_iter, sbp_m, sbp_s, dbp_m, dbp_s = dl.load_train_val_data(
        ppg_path=str(cfg.datadir), feat_path=str(cfg.liu2023_h5), indices_dir=fold_dir)

    model = FusionGated(filters=cfg.filters, num_layers=cfg.num_layers, drop_prob=cfg.dropout,
                        feature_dim=cfg.feature_dim, proj_dim=cfg.proj_dim)
    save_path = os.path.join(cfg.model_save_dir, f'fusion_gated169_fold_{fold}.pkl')
    model = model.to(cfg.device)

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                           betas=(cfg.beta1, cfg.beta2), eps=cfg.epsilon)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    best_loss = float("inf"); pcount = 0
    for ep in range(cfg.epochs):
        model.train()
        losses = sl = dl = 0; msl = mdl = 0
        t0 = time.time()
        for (xp, xf), y in tqdm(train_iter, total=len(train_iter), desc=f"Epoch {ep}", colour="cyan"):
            xp = xp.to(cfg.device); xf = xf.to(cfg.device); y = y.to(cfg.device)
            opt.zero_grad()
            o = model(xp, xf)
            ls = loss_fn(o[:,0], y[:,0]); ld = loss_fn(o[:,1], y[:,1]); l = ls + ld
            l.backward(); opt.step()
            with torch.no_grad():
                losses += l.item(); sl += ls.item(); dl += ld.item()
                msl += ls.item() * sbp_s**2; mdl += ld.item() * dbp_s**2
        n = len(train_iter)
        logging.info(f"Epoch: {ep}, Train loss: {losses/n:.3f}, SBP mmHg: {msl/n:.3f}, DBP mmHg: {mdl/n:.3f}, Time: {time.time()-t0:.1f}s")

        metrics = evaluate(cfg.device, val_iter, loss_fn, model, sbp_s, dbp_s, fold=fold)
        vl, vsl, vdl, vslm, vdlm, vmae, vsmae, vdmae, vsmae_m, vdmae_m = metrics
        sched.step(vl)
        logging.info(f"Epoch {ep}, Val loss: {vl:.3f}, SBP mmHg: {vslm:.3f}, DBP mmHg: {vdlm:.3f}  "
                     f"MAE: {vmae:.3f}, SBP MAE: {vsmae:.3f}, mmHg: {vsmae_m:.3f}, DBP MAE: {vdmae:.3f}, mmHg: {vdmae_m:.3f}")
        if best_loss > vl:
            best_loss = vl
            best_metrics = metrics
            torch.save(deepcopy(model.state_dict()), save_path); logging.info("Best model saved"); pcount = 0
        else:
            pcount += 1
        if pcount > cfg.early_stopping_patience:
            logging.info(f"Early stopping at epoch {ep}"); break
    return best_metrics


def evaluate(dev, val_iter, loss_fn, model, s_std, d_std, fold=None):
    model.eval()
    v = dict.fromkeys(['vl','vsl','vdl','vslm','vdlm','vma','vsma','vdma','vsma_m','vdma_m'], 0)
    gate_sbp_list, gate_dbp_list = [], []
    with torch.no_grad():
        for (xp, xf), y in val_iter:
            xp=xp.to(dev); xf=xf.to(dev); y=y.to(dev)
            o = model(xp, xf)
            ls = loss_fn(o[:,0],y[:,0]); ld = loss_fn(o[:,1],y[:,1])
            v['vl'] += (ls+ld).item(); v['vsl'] += ls.item(); v['vdl'] += ld.item()
            v['vslm'] += ls.item()*s_std**2; v['vdlm'] += ld.item()*d_std**2
            ms = torch.mean(torch.abs(o[:,0]-y[:,0])).item()
            md = torch.mean(torch.abs(o[:,1]-y[:,1])).item()
            v['vma'] += (ms+md)/2; v['vsma'] += ms; v['vdma'] += md
            v['vsma_m'] += ms*s_std; v['vdma_m'] += md*d_std
            gate_sbp_list.append(model.sbp_gate.weight.data.cpu())
            gate_dbp_list.append(model.dbp_gate.weight.data.cpu())
    n = len(val_iter)
    if fold is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        w_sbp = model.sbp_gate.weight.data.cpu().numpy()
        w_dbp = model.dbp_gate.weight.data.cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(range(model.proj_dim), w_sbp.mean(axis=1))
        axes[0].set_title(f'SBP Gate — mean weight per proj dim (Fold {fold})')
        axes[0].set_xlabel('Projection dimension'); axes[0].set_ylabel('Mean |weight|')
        axes[1].bar(range(model.proj_dim), w_dbp.mean(axis=1))
        axes[1].set_title(f'DBP Gate — mean weight per proj dim (Fold {fold})')
        axes[1].set_xlabel('Projection dimension'); axes[1].set_ylabel('Mean |weight|')
        plt.tight_layout()
        os.makedirs(r'E:\Kaggle_projects\Blood_Pressure_analysis\cache\figures', exist_ok=True)
        plt.savefig(fr'E:\Kaggle_projects\Blood_Pressure_analysis\cache\figures\gate_pattern169_fold{fold}.png', dpi=150)
        plt.close()
        w_sbp_flat = w_sbp.ravel(); w_dbp_flat = w_dbp.ravel()
        cos_sim = np.dot(w_sbp_flat, w_dbp_flat) / (np.linalg.norm(w_sbp_flat) * np.linalg.norm(w_dbp_flat) + 1e-8)
        logging.info(f"Fold {fold} Gate cosine similarity: {cos_sim:.4f}")
    model.train()
    return tuple(v[k]/n for k in ['vl','vsl','vdl','vslm','vdlm','vma','vsma','vdma','vsma_m','vdma_m'])
if __name__ == '__main__':
    cfg = Gated169Config()
    maes = {k:[] for k in ['m','sm','dm','sm_m','dm_m']}
    for fold in range(cfg.k):
        m = train_fold(cfg, fold)
        (_, _, _, _, _, vmae, vsmae, vdmae, vsmae_m, vdmae_m) = m
        maes['m'].append(vmae); maes['sm'].append(vsmae); maes['dm'].append(vdmae)
        maes['sm_m'].append(vsmae_m); maes['dm_m'].append(vdmae_m)
        n = len(maes['m'])
        logging.info(f"{n} fold SBP MAE = {np.mean(maes['sm_m']):.3f} ± {np.std(maes['sm_m']):.3f} mmHg")
        logging.info(f"{n} fold DBP MAE = {np.mean(maes['dm_m']):.3f} ± {np.std(maes['dm_m']):.3f} mmHg")

    logging.info("="*60)
    logging.info("FINAL 5-FOLD CV (FusionGated169)")
    logging.info(f"SBP MAE = {np.mean(maes['sm_m']):.2f} ± {np.std(maes['sm_m']):.2f} mmHg")
    logging.info(f"DBP MAE = {np.mean(maes['dm_m']):.2f} ± {np.std(maes['dm_m']):.2f} mmHg")
    print(f"\n===== FusionGated169 5-Fold CV =====")
    print(f"SBP MAE = {np.mean(maes['sm_m']):.2f} ± {np.std(maes['sm_m']):.2f} mmHg")
    print(f"DBP MAE = {np.mean(maes['dm_m']):.2f} ± {np.std(maes['dm_m']):.2f} mmHg")

