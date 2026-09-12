"""
少量样本的PPG→Spectrogram可视化测试
=====================================
先使用合成PPG演示，如果segmented_records.h5存在则读取真实数据。
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from utils.ppg_spectrogram import (
    ppg_to_spectrogram,
    normalize_spectrogram,
    plot_spectrogram,
)

# ====== 配置参数 ======
FS = 125
NPERSEG = 128
N_OVERLAP = 96     # 75% overlap
FREQ_RANGE = (0.5, 8.0)
N_SAMPLES = 6      # 可视化前6个窗口

# ====== 尝试加载真实数据 ======
h5_path = "Blood_pressure_dataset/segmented_records.h5"
if os.path.exists(h5_path):
    print("加载真实数据集...")
    with h5py.File(h5_path, "r") as f:
        ppg_data = f["ppg"][:N_SAMPLES]
        sbp_data = f["sbp"][:N_SAMPLES]
        dbp_data = f["dbp"][:N_SAMPLES]
    source = "真实数据"
else:
    print("真实数据集不存在，使用合成PPG信号...")
    # 生成N_SAMPLES个合成PPG窗口(1024点)
    n = 1024
    t = np.linspace(0, n / FS, n, endpoint=False)
    ppg_data = []
    sbp_data = []
    dbp_data = []
    rng = np.random.RandomState(42)
    for i in range(N_SAMPLES):
        hr = 60 + rng.randint(-10, 15)   # 50-75 bpm
        amp = 0.8 + rng.rand() * 0.4      # 0.8-1.2
        noise = 0.03 * rng.randn(n)
        signal = np.sin(2 * np.pi * hr / 60 * t + i * 0.7) * amp * 0.5 + 0.5
        signal += noise
        ppg_data.append(signal)
        sbp_data.append(120 + rng.randn() * 5)
        dbp_data.append(75 + rng.randn() * 3)
    source = "合成数据"

# ====== 转换所有样本为频谱图 ======
specs = []
for i, ppg in enumerate(ppg_data):
    spec, freqs, times = ppg_to_spectrogram(
        ppg, fs=FS, nperseg=NPERSEG,
        noverlap=N_OVERLAP, freq_range=FREQ_RANGE,
        log_scale=True
    )
    spec_norm = normalize_spectrogram(spec, method='minmax')
    specs.append(spec_norm)
    print(f"样本{i+1}: PPG心率先验≈?, "
          f"频谱图 {spec_norm.shape[0]}频段×{spec_norm.shape[1]}时间帧, "
          f"SBP={sbp_data[i]:.0f}, DBP={dbp_data[i]:.0f}")

print(f"\n频谱图尺寸: {specs[0].shape}")
print(f"频率范围: {freqs[0]:.2f}–{freqs[-1]:.2f} Hz")
print(f"时间范围: {times[0]:.3f}–{times[-1]:.3f} s")

# ====== 可视化: PPG + 频谱图 并排 ======
fig, axes = plt.subplots(N_SAMPLES, 2, figsize=(14, 2.5 * N_SAMPLES))

for i in range(N_SAMPLES):
    t_ppg = np.arange(len(ppg_data[i])) / FS

    # 左列: 原始PPG
    ax_ppg = axes[i, 0] if N_SAMPLES > 1 else axes[0]
    ax_ppg.plot(t_ppg, ppg_data[i], 'b-', linewidth=0.6)
    ax_ppg.set_ylabel('Amplitude')
    ax_ppg.set_title(f'Sample {i+1} — PPG raw\n'
                     f'SBP={sbp_data[i]:.1f}  DBP={dbp_data[i]:.1f}',
                     fontsize=9)

    # 右列: 频谱图
    ax_spec = axes[i, 1] if N_SAMPLES > 1 else axes[1]
    # 用pcolormesh画频谱图
    im = ax_spec.pcolormesh(times, freqs, specs[i],
                            shading='gouraud', cmap='viridis')
    ax_spec.set_xlabel('Time (s)')
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_title(f'Spectrogram ({specs[i].shape[0]}×{specs[i].shape[1]})',
                      fontsize=9)

    if i == 0:
        axes[i, 0].set_xlabel('Time (s)') if N_SAMPLES > 1 else None

plt.tight_layout()
plt.savefig('ppg_spectrogram_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n已保存: ppg_spectrogram_demo.png")

# ====== 平均频谱图 ======
mean_spec = np.mean(specs, axis=0)
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
im2 = ax2.pcolormesh(times, freqs, mean_spec,
                     shading='gouraud', cmap='viridis')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title(f'Mean Spectrogram ({source})')
plt.colorbar(im2, ax=ax2, label='Normalized power')
plt.tight_layout()
plt.savefig('ppg_spectrogram_mean.png', dpi=150, bbox_inches='tight')
plt.show()
print("已保存: ppg_spectrogram_mean.png")
