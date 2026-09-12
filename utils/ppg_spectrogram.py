"""
PPG → Spectrogram conversion
==============================
Based on Ring-BP (Liu Bin 2024) approach:
  - STFT-based time-frequency representation of PPG signals
  - Used as input to MobileNet-V3-Small for frequency-domain feature extraction
  - Combined with time-domain EfficientNet branch via cross-attention

Usage:
    from utils.ppg_spectrogram import (
        ppg_to_spectrogram,
        extract_spectrograms_to_h5,
        PPGSpectrogramDataset,
        LoadPPGSpectrogramDataset,
    )

    # Single segment
    spec = ppg_to_spectrogram(ppg_segment, fs=125)
    print(spec.shape)  # (n_freq_bins, n_time_frames)

    # Batch extraction
    extract_spectrograms_to_h5(
        h5_in="Blood_pressure_dataset/segmented_records.h5",
        h5_out="Blood_pressure_dataset/ppg_spectrograms.h5",
        fs=125
    )

    # PyTorch DataLoader
    loader = LoadPPGSpectrogramDataset(batch_size=64)
    train_iter, val_iter, sbp_mean, sbp_std, dbp_mean, dbp_std = \
        loader.load_train_val_data(
            spec_path="Blood_pressure_dataset/ppg_spectrograms.h5",
            indices_dir="Blood_pressure_dataset/cv_fold_0.npz",
            target_size=(128, 128)  # resize for CNN input
        )
"""

import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import zoom
import h5py
from tqdm import tqdm
import warnings

# Optional PyTorch import — Dataset/DataLoader classes only needed during training.
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = object


# =============================================================================
# 1.  PPG → Spectrogram conversion
# =============================================================================

def ppg_to_spectrogram(ppg, fs=125, nperseg=128, noverlap=None,
                        freq_range=(0.5, 8.0), log_scale=True):
    """
    Convert a PPG segment to a log-power spectrogram (STFT).

    Parameters
    ----------
    ppg : 1-D array
        PPG signal segment.
    fs : int
        Sampling rate in Hz. Default 125.
    nperseg : int
        Length of each STFT window (in samples). Default 128 (~1.02s at 125Hz).
    noverlap : int or None
        Overlap between windows. Default None → 75% overlap (noverlap = nperseg * 3 // 4).
    freq_range : tuple
        Frequency range to keep (lowcut, highcut) in Hz. Default (0.5, 8.0).
    log_scale : bool
        If True, return 10*log10(power). Default True.

    Returns
    -------
    Sxx : 2-D ndarray (n_freq_bins, n_time_frames)
        Spectrogram (log-power if log_scale=True).
    freqs : 1-D ndarray
        Frequency bins (Hz) corresponding to the rows of Sxx.
    times : 1-D ndarray
        Time bins (s) corresponding to the columns of Sxx.
    """
    if noverlap is None:
        noverlap = nperseg * 3 // 4  # 75% overlap

    f, t, Sxx = spectrogram(ppg, fs=fs, window='hann',
                            nperseg=nperseg, noverlap=noverlap,
                            scaling='density', mode='magnitude')

    # Crop to PPG-relevant frequency band
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    Sxx = Sxx[freq_mask, :]
    freqs = f[freq_mask]

    if log_scale:
        Sxx = 10 * np.log10(Sxx + 1e-10)

    return Sxx, freqs, t


def resize_spectrogram(spec, target_size):
    """
    Resize a 2-D spectrogram to a fixed (H, W) using bilinear interpolation.

    Parameters
    ----------
    spec : 2-D ndarray (H_in, W_in)
    target_size : tuple (H_out, W_out)
        Desired output size (rows, columns).

    Returns
    -------
    spec_resized : 2-D ndarray (H_out, W_out)
    """
    if spec.ndim != 2:
        raise ValueError(f"Expected 2-D input, got shape {spec.shape}")

    h_in, w_in = spec.shape
    h_out, w_out = target_size

    zoom_factors = (h_out / h_in, w_out / w_in)
    spec_resized = zoom(spec, zoom_factors, order=1)  # bilinear

    return spec_resized


def normalize_spectrogram(spec, method='minmax'):
    """
    Normalize spectrogram to [0, 1] or z-score.

    Parameters
    ----------
    spec : 2-D ndarray
    method : str
        'minmax' → scale to [0, 1]; 'zscore' → zero mean, unit variance.

    Returns
    -------
    spec_norm : 2-D ndarray, same shape as input.
    """
    if method == 'minmax':
        s_min, s_max = spec.min(), spec.max()
        if s_max - s_min < 1e-10:
            return np.zeros_like(spec)
        return (spec - s_min) / (s_max - s_min)
    elif method == 'zscore':
        s_mean, s_std = spec.mean(), spec.std()
        if s_std < 1e-10:
            return np.zeros_like(spec)
        return (spec - s_mean) / s_std
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# 2.  Batch conversion → H5
# =============================================================================

def extract_spectrograms_to_h5(h5_in, h5_out, fs=125, nperseg=128,
                                noverlap=None, freq_range=(0.5, 8.0),
                                target_size=None, log_scale=True,
                                norm_method='minmax'):
    """
    Batch-convert PPG segments to spectrograms and save to h5.

    Reads segmented_records.h5 (ppg, sbp, dbp, record_id), converts each
    window to a spectrogram, and writes:
      - 'spectrograms' : (N, H, W) float32 — raw spectrograms
      - 'sbp' / 'dbp' / 'record_id' — copied from input

    If target_size is set, adds:
      - 'spectrograms_resized' : (N, H_out, W_out) float32

    Parameters
    ----------
    h5_in : str           Path to segmented_records.h5
    h5_out : str          Output h5 path.
    fs : int              Sampling rate in Hz.
    nperseg : int         STFT window length.
    noverlap : int or None  STFT overlap.
    freq_range : tuple    Frequency range (low, high) in Hz.
    target_size : tuple or None  If set, resize all spectrograms to (H, W).
    log_scale : bool      Convert power to dB scale.
    norm_method : str or None  'minmax' or 'zscore' or None.
    """
    with h5py.File(h5_in, "r") as fin:
        ppg_in = fin["ppg"]
        sbp_in = fin["sbp"]
        dbp_in = fin["dbp"]
        rid_in = fin["record_id"]
        n = len(ppg_in)

        # First window: determine spectrogram shape
        spec0, freqs, times = ppg_to_spectrogram(
            ppg_in[0].astype(np.float64),
            fs=fs, nperseg=nperseg, noverlap=noverlap,
            freq_range=freq_range, log_scale=log_scale
        )
        h_in, w_in = spec0.shape

        all_specs = np.zeros((n, h_in, w_in), dtype=np.float32)
        all_specs[0] = normalize_spectrogram(spec0, method=norm_method) \
            if norm_method else spec0.astype(np.float32)

        for i in tqdm(range(1, n), desc="Converting to spectrograms", colour="cyan"):
            spec, _, _ = ppg_to_spectrogram(
                ppg_in[i].astype(np.float64),
                fs=fs, nperseg=nperseg, noverlap=noverlap,
                freq_range=freq_range, log_scale=log_scale
            )
            spec_norm = normalize_spectrogram(spec, method=norm_method) \
                if norm_method else spec.astype(np.float32)
            all_specs[i] = spec_norm

    print(f"Spectrogram shape per window: ({h_in}, {w_in})")
    print(f"  Frequency range kept: {freq_range[0]}-{freq_range[1]} Hz "
          f"({h_in} bins)")
    print(f"  Time frames: {w_in} (window={nperseg} samples, "
          f"overlap={noverlap or nperseg*3//4})")

    with h5py.File(h5_out, "w") as fout:
        fout.create_dataset("spectrograms", data=all_specs, dtype=np.float32)
        fout.create_dataset("params_fs", data=fs)
        fout.create_dataset("params_nperseg", data=nperseg)
        fout.create_dataset("params_noverlap",
                            data=noverlap if noverlap is not None else nperseg * 3 // 4)
        fout.create_dataset("params_freq_low", data=freq_range[0])
        fout.create_dataset("params_freq_high", data=freq_range[1])
        fout.create_dataset("params_log_scale", data=int(log_scale))
        fout.create_dataset("params_norm_method",
                            data=norm_method if norm_method else 'none')

        # Resized version if requested
        if target_size is not None:
            resized = np.zeros((n, target_size[0], target_size[1]),
                               dtype=np.float32)
            for i in range(n):
                resized[i] = resize_spectrogram(all_specs[i], target_size)
            fout.create_dataset("spectrograms_resized",
                                data=resized, dtype=np.float32)
            print(f"Resized spectrograms: ({target_size[0]}, {target_size[1]})")

        with h5py.File(h5_in, "r") as fin:
            fout.create_dataset("sbp", data=fin["sbp"][:], dtype=np.float32)
            fout.create_dataset("dbp", data=fin["dbp"][:], dtype=np.float32)
            fout.create_dataset("record_id", data=fin["record_id"][:],
                                dtype=np.int32)

    print(f"Saved {n} spectrograms → {h5_out}")


# =============================================================================
# 3.  PyTorch Dataset / DataLoader
# =============================================================================

class PPGSpectrogramDataset(Dataset):
    """
    PyTorch Dataset that loads spectrograms + SBP/DBP labels.

    Supports three modes via use_resized:
      - use_resized=False : returns raw spectrogram (H, W)
      - use_resized=True  : returns resized spectrogram (H_out, W_out)
      - add_channel_dim=True (default): adds channel dim → (1, H, W)
        suitable for 2D CNN (Conv2d expects N, C, H, W).

    Output shapes:
      x: (1, H, W) — spectrogram with channel dim for Conv2d
      y: (2,)      — [SBP, DBP]
    """

    def __init__(self, spec_path, indices_dir, train=True,
                 use_resized=False, add_channel_dim=True,
                 sbp_mean=None, sbp_std=None, dbp_mean=None, dbp_std=None,
                 spec_mean=None, spec_std=None):
        self.spec_path = spec_path
        self.spec_file = None
        self.use_resized = use_resized
        self.add_channel_dim = add_channel_dim

        self.sbp_mean = sbp_mean
        self.sbp_std = sbp_std
        self.dbp_mean = dbp_mean
        self.dbp_std = dbp_std
        self.spec_mean = spec_mean
        self.spec_std = spec_std

        fold_indices = np.load(indices_dir)
        if train:
            self.indices = fold_indices["train_idx"]
        else:
            self.indices = fold_indices["val_idx"]

    def _init_file(self):
        if self.spec_file is None:
            self.spec_file = h5py.File(self.spec_path, "r")
            ds_name = "spectrograms_resized" if self.use_resized \
                else "spectrograms"
            self.spec_ds = self.spec_file[ds_name]
            self.sbp = self.spec_file["sbp"]
            self.dbp = self.spec_file["dbp"]

    def __getitem__(self, index):
        self._init_file()
        idx = self.indices[index]

        spec = self.spec_ds[idx].astype(np.float32)  # (H, W)
        if self.spec_mean is not None:
            spec = (spec - self.spec_mean) / (self.spec_std + 1e-8)

        if self.add_channel_dim:
            spec = np.expand_dims(spec, axis=0)  # (1, H, W)

        sbp = self.sbp[idx]
        dbp = self.dbp[idx]
        if self.sbp_mean is not None:
            sbp = (sbp - self.sbp_mean) / self.sbp_std
            dbp = (dbp - self.dbp_mean) / self.dbp_std

        x = torch.from_numpy(spec).float()
        y = torch.tensor([sbp, dbp], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.indices)


class LoadPPGSpectrogramDataset:
    """
    Dataloader wrapper for PPGSpectrogramDataset.

    Output shapes:
      x: (batch_size, 1, H, W) — spectrogram for Conv2d
      y: (batch_size, 2)       — [SBP, DBP]

    use_resized: True → use pre-resized spectrograms from h5
                  False → use raw spectrograms and resize on-the-fly
    """

    def __init__(self, batch_size, is_sample_shuffle=True):
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    def load_train_val_data(self, spec_path, indices_dir,
                            use_resized=False, target_size=None):
        train_dataset = PPGSpectrogramDataset(
            spec_path, indices_dir, train=True,
            use_resized=use_resized
        )

        spec_mean, spec_std = self.spec_statistical_values(
            train_dataset, spec_path, use_resized
        )
        sbp_mean, sbp_std, dbp_mean, dbp_std = self.bp_statistical_values(
            train_dataset
        )

        train_dataset = PPGSpectrogramDataset(
            spec_path, indices_dir, train=True,
            use_resized=use_resized,
            sbp_mean=sbp_mean, sbp_std=sbp_std,
            dbp_mean=dbp_mean, dbp_std=dbp_std,
            spec_mean=spec_mean, spec_std=spec_std
        )

        val_dataset = PPGSpectrogramDataset(
            spec_path, indices_dir, train=False,
            use_resized=use_resized,
            sbp_mean=sbp_mean, sbp_std=sbp_std,
            dbp_mean=dbp_mean, dbp_std=dbp_std,
            spec_mean=spec_mean, spec_std=spec_std
        )

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                num_workers=4, pin_memory=True,
                                persistent_workers=True)
        val_iter = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True,
                              persistent_workers=True)

        return train_iter, val_iter, sbp_mean, sbp_std, dbp_mean, dbp_std

    def bp_statistical_values(self, dataset):
        with h5py.File(dataset.spec_path, "r") as f:
            indices = dataset.indices
            sbp = f["sbp"][indices]
            dbp = f["dbp"][indices]
        return sbp.mean(), sbp.std(), dbp.mean(), dbp.std()

    def spec_statistical_values(self, dataset, spec_path, use_resized=False):
        with h5py.File(spec_path, "r") as f:
            indices = dataset.indices
            ds_name = "spectrograms_resized" if use_resized else "spectrograms"
            specs = f[ds_name][indices]
        return specs.mean(), specs.std()


# =============================================================================
# 4.  Visualisation helper
# =============================================================================

def plot_spectrogram(spec, fs=125, nperseg=128, noverlap=None,
                      freq_range=(0.5, 8.0), ax=None, title=None):
    """
    Plot a single spectrogram with proper axis labels.

    Parameters
    ----------
    spec : 2-D ndarray  (n_freq_bins, n_time_frames)
    fs, nperseg, noverlap, freq_range : STFT parameters (for axis labels).
    ax : matplotlib Axes or None
    title : str or None
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 4))

    if noverlap is None:
        noverlap = nperseg * 3 // 4
    hop = nperseg - noverlap

    # Estimate time and frequency axes
    n_frames = spec.shape[1]
    times = np.arange(n_frames) * hop / fs
    freqs = np.linspace(freq_range[0], freq_range[1], spec.shape[0])

    im = ax.pcolormesh(times, freqs, spec, shading='gouraud', cmap='viridis')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title or f'PPG Spectrogram ({spec.shape[0]}×{spec.shape[1]})')
    plt.colorbar(im, ax=ax, label='dB')

    return ax


# =============================================================================
# 5.  Self-test
# =============================================================================

if __name__ == '__main__':
    import os
    import tempfile

    print("=" * 60)
    print("1. Single PPG → Spectrogram test")
    print("=" * 60)

    # Generate synthetic PPG (8.192s at 125Hz = 1024 samples)
    fs = 125
    n_samples = 1024
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)
    hr = 72
    ppg_synth = np.sin(2 * np.pi * hr / 60 * t) * 0.5 + 0.5
    ppg_synth += 0.05 * np.random.randn(n_samples)

    spec, freqs, times = ppg_to_spectrogram(ppg_synth, fs=fs)
    print(f"Input PPG: {n_samples} samples @ {fs}Hz ({n_samples/fs:.2f}s)")
    print(f"Spectrogram shape: {spec.shape}  "
          f"(freq_bins={spec.shape[0]}, time_frames={spec.shape[1]})")
    print(f"Frequency range: {freqs[0]:.2f} – {freqs[-1]:.2f} Hz")
    print(f"Time range: {times[0]:.2f} – {times[-1]:.2f} s")

    # Resize test
    spec_resized = resize_spectrogram(spec, (64, 64))
    print(f"Resized spectrogram: {spec_resized.shape}")

    # Normalize test
    spec_norm = normalize_spectrogram(spec, method='minmax')
    print(f"Normalized spectrogram: min={spec_norm.min():.3f}, "
          f"max={spec_norm.max():.3f}")

    print("\n" + "=" * 60)
    print("2. Batch extraction to h5 test")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    demo_h5_in = os.path.join(tmpdir, "segmented_demo.h5")
    demo_h5_out = os.path.join(tmpdir, "ppg_spectrograms.h5")

    n_windows = 20
    dummy_ppg = np.zeros((n_windows, 1024), dtype=np.float32)
    dummy_sbp = np.random.uniform(110, 150, n_windows).astype(np.float32)
    dummy_dbp = np.random.uniform(60, 90, n_windows).astype(np.float32)
    dummy_rid = np.zeros(n_windows, dtype=np.int32)

    for i in range(n_windows):
        phase = np.linspace(0, 4 * np.pi, 1024)
        dummy_ppg[i] = (np.sin(phase + i * 0.3) * 0.5 + 0.5
                        + 0.03 * np.random.randn(1024))

    with h5py.File(demo_h5_in, "w") as f:
        f.create_dataset("ppg", data=dummy_ppg, dtype=np.float32)
        f.create_dataset("sbp", data=dummy_sbp, dtype=np.float32)
        f.create_dataset("dbp", data=dummy_dbp, dtype=np.float32)
        f.create_dataset("record_id", data=dummy_rid, dtype=np.int32)

    print(f"Created {n_windows} demo windows → {demo_h5_in}")

    extract_spectrograms_to_h5(
        demo_h5_in, demo_h5_out, fs=fs,
        target_size=(128, 128)
    )

    with h5py.File(demo_h5_out, "r") as f:
        print(f"\nOutput verification:")
        print(f"  spectrograms         shape: {f['spectrograms'].shape}")
        print(f"  spectrograms_resized shape: {f['spectrograms_resized'].shape}")
        print(f"  sbp                  shape: {f['sbp'].shape}")

    import shutil
    shutil.rmtree(tmpdir)
    print(f"\nCleaned up: {tmpdir}")
    print("\nAll tests passed!")
