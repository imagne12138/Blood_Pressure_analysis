"""
训练脚本 — MLP_Opt: SBP_opt(17-dim) + DBP_opt(12-dim) 单独 MLP 预测

验证论文筛选的最优特征子集单独做 MLP 的效果:
  - 对比 169 维全量 MLP (16.70/7.45)
  - 看精选特征是否更高效

保存: cache/mlp_opt_fold_X.pkl
日志: log_train_mlp_opt

用法:
    conda run -n Pytorch python train_mlp_opt.py
"""
import logging, os, time
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm

from config.config import Config
from model.MLP_Opt import MLP_Opt
from utils.create_data import LoadFusionOptDataset
from utils.log_helper import logger_init


class MLPOptConfig(Config):
    def __init__(self):
        super().__init__()
        _old = []
        for h in logging.root.handlers[:]:
            if hasattr(h, 'baseFilename'): _old.append(h.baseFilename)
            h.close(); logging.root.removeHandler(h)
        for p in _old:
            if os.path.exists(p): os.remove(p)
        logger_init(log_file_name='log_train_mlp_opt', log_level=logging.INFO, log_dir=self.model_save_dir)
        self.liu2023_h5 = self.base_dir / "liu2023_features.h5"


def train_fold(cfg, fold):
    logging.info(f"######## Fold {fold} ########")
    dl = LoadFusionOptDataset(batch_size=cfg.batch_size)
    fold_dir = os.path.join(cfg.base_dir, f"cv_fold_{fold}.npz")
    train_iter, val_iter, sbp_m, sbp_s, dbp_m, dbp_s = dl.load_train_val_data(
        ppg_path=str(cfg.datadir), feat_path=str(cfg.liu2023_h5), indices_dir=fold_dir)

    model = MLP_Opt(drop_prob=cfg.dropout)
    save_path = os.path.join(cfg.model_save_dir, f'mlp_opt_fold_{fold}.pkl')
    model = model.to(cfg.device)

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                           betas=(cfg.beta1, cfg.beta2), eps=cfg.epsilon)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    best_loss = float("inf"); best_metrics = None; pcount = 0
    for ep in range(cfg.epochs):
        model.train()
        losses = sl = dl = 0; msl = mdl = 0
        t0 = time.time()
        for (_, x_s, x_d), y in tqdm(train_iter, total=len(train_iter), desc=f"Epoch {ep}", colour="cyan"):
            x_s = x_s.to(cfg.device); x_d = x_d.to(cfg.device); y = y.to(cfg.device)
            opt.zero_grad()
            o = model(x_s, x_d)
            l_s = loss_fn(o[:,0], y[:,0]); l_d = loss_fn(o[:,1], y[:,1]); l = l_s + l_d
            l.backward(); opt.step()
            with torch.no_grad():
                losses += l.item(); sl += l_s.item(); dl += l_d.item()
                msl += l_s.item() * sbp_s**2; mdl += l_d.item() * dbp_s**2
        n = len(train_iter)
        logging.info(f"Epoch: {ep}, Train loss: {losses/n:.3f}, SBP mmHg: {msl/n:.3f}, DBP mmHg: {mdl/n:.3f}, Time: {time.time()-t0:.1f}s")

        metrics = evaluate(cfg.device, val_iter, loss_fn, model, sbp_s, dbp_s)
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


def evaluate(dev, val_iter, loss_fn, model, s_std, d_std):
    model.eval()
    v = dict.fromkeys(['vl','vsl','vdl','vslm','vdlm','vma','vsma','vdma','vsma_m','vdma_m'], 0)
    with torch.no_grad():
        for (_, x_s, x_d), y in val_iter:
            x_s=x_s.to(dev); x_d=x_d.to(dev); y=y.to(dev)
            o = model(x_s, x_d)
            ls = loss_fn(o[:,0],y[:,0]); ld = loss_fn(o[:,1],y[:,1])
            v['vl'] += (ls+ld).item(); v['vsl'] += ls.item(); v['vdl'] += ld.item()
            v['vslm'] += ls.item()*s_std**2; v['vdlm'] += ld.item()*d_std**2
            m_s = torch.mean(torch.abs(o[:,0]-y[:,0])).item()
            m_d = torch.mean(torch.abs(o[:,1]-y[:,1])).item()
            v['vma'] += (m_s+m_d)/2; v['vsma'] += m_s; v['vdma'] += m_d
            v['vsma_m'] += m_s*s_std; v['vdma_m'] += m_d*d_std
    n = len(val_iter); model.train()
    return tuple(v[k]/n for k in ['vl','vsl','vdl','vslm','vdlm','vma','vsma','vdma','vsma_m','vdma_m'])


if __name__ == '__main__':
    cfg = MLPOptConfig()
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
    logging.info("FINAL 5-FOLD CV (MLP_Opt)")
    logging.info(f"SBP MAE = {np.mean(maes['sm_m']):.2f} ± {np.std(maes['sm_m']):.2f} mmHg")
    logging.info(f"DBP MAE = {np.mean(maes['dm_m']):.2f} ± {np.std(maes['dm_m']):.2f} mmHg")
    print(f"\n===== MLP_Opt 5-Fold CV =====")
    print(f"SBP MAE = {np.mean(maes['sm_m']):.2f} ± {np.std(maes['sm_m']):.2f} mmHg")
    print(f"DBP MAE = {np.mean(maes['dm_m']):.2f} ± {np.std(maes['dm_m']):.2f} mmHg")
