import re
import os
from collections import defaultdict


def extract_val_mae(fp: str) -> dict[int, list[float]]:
    """
    从训练日志中提取每折的 Val MAE 序列。
    支持两种格式：
      Format A: Val MAE: X, Val SBP MAE: Y, ...
      Format B: MAE: X, SBP MAE: Y, ...
    
    返回: {fold_num: [val_mae_epoch0, val_mae_epoch1, ...]}
    """
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    fold_starts = []
    for m in re.finditer(r'[#\*]+ Fold (\d+) [#\*]+', content):
        fold_starts.append((int(m.group(1)), m.end()))

    if not fold_starts:
        fold_starts = [(0, 0)]

    fold_data = {}
    for fold_num, start_pos in fold_starts:
        end_pos = len(content)
        for fn2, sp2 in fold_starts:
            if sp2 > start_pos:
                end_pos = sp2 - 1
                break

        section = content[start_pos:end_pos]
        vals = []
        for line in section.split('\n'):
            m = re.search(r'Val MAE:\s*([\d.]+)', line)
            if m:
                vals.append(float(m.group(1)))
                continue
            m = re.search(r'(?<!Val )MAE:\s*([\d.]+),\s*SBP MAE:', line)
            if m:
                vals.append(float(m.group(1)))

        if vals:
            if fold_num not in fold_data or len(vals) > len(fold_data[fold_num]):
                fold_data[fold_num] = vals

    return fold_data


def extract_train_loss_and_val_mae(fp: str, target_fold: int = 0):
    """
    提取指定折的逐 epoch Train loss 和 Val MAE。
    返回: (train_losses: list[float], val_maes: list[float])
    """
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    fold_markers = [(int(m.group(1)), m.end())
                    for m in re.finditer(r'[#\*]+ Fold (\d+) [#\*]+', content)]

    # 找到目标折的最后一次出现（可能是重跑后的完整结果）
    target_starts = [sp for fn, sp in fold_markers if fn == target_fold]
    if not target_starts:
        return [], []

    start_pos = target_starts[-1]
    end_pos = len(content)
    for fn, sp in fold_markers:
        if sp > start_pos:
            end_pos = sp - 1
            break

    section = content[start_pos:end_pos]

    train_losses = []
    val_maes = []

    for line in section.split('\n'):
        m = re.search(r'Epoch: (\d+), Train loss: ([\d.]+)', line)
        if m:
            train_losses.append(float(m.group(2)))
        m = re.search(r'Val MAE:\s*([\d.]+)', line)
        if m:
            val_maes.append(float(m.group(1)))

    return train_losses, val_maes


def load_all_experiments(cache_root: str) -> dict:
    """
    加载所有实验的 Val MAE 数据。
    返回: {exp_num: (label, {fold: [val_maes]})}
    """
    experiments = {
        1: ('原始Baseline', [os.path.join('baseline', 'Training logs for original baseline model.txt')]),
        2: ('Baseline改进', [os.path.join('baseline', 'Baseline model adding dropout in cnn layers and simplifying attention layer.txt')]),
        3: ('Model2+WD', [os.path.join('model2', 'Using model 2 and adding weight decay.txt')]),
        4: ('Liu2023 MLP', [os.path.join('liu2023_feature_mlp',
            'log_train_2026-05-26_liu2023_features_hidden12_64_32_dropout0.3_无BN.txt')]),
        5: ('Liu2023改进MLP', [os.path.join('liu2023_feature_mlp',
            'log_train_2026-05-26_improved_MLP (hidden1, 2扩张，加BN，降dropout至0.1).txt')]),
        6: ('简单特征MLP', [os.path.join('简单特征提取+mlp',
            'log_train_2026-05-31.txt')]),
        7: ('Baseline+26融合', [os.path.join('baseline_fusion_26dim',
            'log_train_fusion_baseline26_2026-06-02.txt')]),
        8: ('Baseline+169融合', [os.path.join('baseline_fusion_169dim',
            'log_train_fusion_baseline_2026-06-02.txt')]),
        9: ('Model2+169融合', [os.path.join('model2_fusion_169dim',
            'log_train_fusion169_2026-06-02.txt')]),
        10: ('Model2+26融合', [os.path.join('model2_fusion_26dim', 'fusion_feature_proj', 
             'log_train_fusion26_feat_proj_2026-06-01.txt')]),
        11: ('Model2+Opt', [os.path.join('model2_fusion_opt',
             'log_train_fusion_opt_2026-06-06.txt')]),
        12: ('MLP-Opt', [os.path.join('mlp_opt',
             'log_train_mlp_opt_2026-06-06.txt')]),
    }

    all_data = {}
    for exp_num, (label, fnames) in experiments.items():
        all_folds = {}
        for fn_rel in fnames:
            fp = os.path.join(cache_root, fn_rel)
            if os.path.exists(fp) and os.path.getsize(fp) > 0:
                folds = extract_val_mae(fp)
                for fold, vals in folds.items():
                    if fold not in all_folds or len(vals) > len(all_folds[fold]):
                        all_folds[fold] = vals
        all_data[exp_num] = (label, all_folds)
    return all_data


if __name__ == '__main__':
    cache_dir = r'E:\Kaggle_projects\Blood_Pressure_analysis\cache'
    data = load_all_experiments(cache_dir)
    for exp_num in sorted(data.keys()):
        label, folds = data[exp_num]
        n_folds = len(folds)
        epochs = [len(v) for v in folds.values()]
        print(f'Exp{exp_num:2d} {label:25s}: {n_folds} folds, {epochs}')

