"""Verify PermEn: compare CPU vs GPU vs extracted H5 values."""
import numpy as np
import h5py

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Load real ppg windows
f_src = h5py.File('Blood_pressure_dataset/segmented_records.h5', 'r')
f_feat = h5py.File('Blood_pressure_dataset/liu2023_features.h5', 'r')

names = [n.decode() if isinstance(n, bytes) else n for n in f_feat['feature_names'][:]]
perm_idx = names.index('PermEn')


def _perm_en_cpu(signal, m=3):
    N = len(signal)
    patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
    perm_keys = [tuple(np.argsort(p)) for p in patterns]
    _, counts = np.unique(perm_keys, return_counts=True)
    probs = counts / len(perm_keys)
    return -np.sum(probs * np.log2(probs + 1e-10))


# Filter (same as pipeline)
from scipy.signal import firwin, lfilter
taps = firwin(51, [0.5, 8.0], pass_zero=False, fs=125, window='hamming')

print("Window | CPU (filtered) |  H5 extracted  | Match")
print("-" * 55)

all_ok = True
for wi in range(20):
    ppg = f_src['ppg'][wi].astype(np.float64)
    ppg_filt = lfilter(taps, 1.0, ppg)
    cpu_val = _perm_en_cpu(ppg_filt)
    h5_val = float(f_feat['features'][wi, perm_idx])
    match = abs(cpu_val - h5_val) < 1e-4
    if not match:
        all_ok = False
    print(f"  {wi:3d}   |  {cpu_val:.6f}   |  {h5_val:.10f}  | {'OK' if match else 'MISMATCH'}")

print(f"\nAll matched: {all_ok}")

# Also test GPU if available
if HAS_TORCH and torch.cuda.is_available():
    print("\n--- GPU vs CPU comparison ---")
    device = 'cuda'
    for wi in range(5):
        ppg = f_src['ppg'][wi].astype(np.float64)
        cpu_val = _perm_en_cpu(ppg)

        sig = torch.as_tensor(ppg, dtype=torch.float64, device=device)
        N, m = len(ppg), 3
        rows = torch.arange(N - m + 1, device=device)
        cols = torch.arange(m, device=device)
        patterns = sig[rows[:, None] + cols[None, :]]
        perm = torch.argsort(patterns, dim=1)
        perm_np = perm.cpu().numpy()
        perm_keys = [tuple(p) for p in perm_np]
        _, counts = np.unique(perm_keys, return_counts=True)
        probs = counts / len(perm_keys)
        gpu_val = -np.sum(probs * np.log2(probs + 1e-10))

        match = abs(cpu_val - gpu_val) < 1e-10
        print(f"  window {wi}: CPU={cpu_val:.6f}  GPU={gpu_val:.6f}  {'OK' if match else 'MISMATCH'}")

f_src.close()
f_feat.close()
