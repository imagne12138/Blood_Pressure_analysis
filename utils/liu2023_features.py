"""
Liu2023 PPG Feature Extraction
===============================
Full implementation of the 172-feature extraction pipeline from:
  Liu et al. "A novel interpretable feature set optimization method in
  blood pressure estimation using photoplethysmography signals" (2023)

Feature dimensions (10):
  1. Temporal (60)     - PPG/VPG/APG feature point time spans
  2. Amplitude (45)    - Feature point amplitudes, differences, ratios
  3. Pulse Width (7)   - Widths at various % of pulse height
  4. Pulse Area (8)    - Areas enclosed by feature points
  5. Angle/Slope (18)  - Slopes/angles between feature points
  6. Entropy (4)       - Approximate, sample, fuzzy, permutation entropy
  7. Frequency (5)     - FFT-based frequency domain features
  8. Statistical (14)  - Mean, std, skewness, kurtosis, etc.
  9. Mixed (5)         - Enhancement index, stiffness index, etc.
 10. Clinical (6)      - Gender, age, height, weight, BMI, HR

Usage:
    from liu2023_features import extract_all_features

    ppg_signal = ...            # 1-D array, any sampling rate
    features = extract_all_features(ppg_signal, fs=1000)

    # Or with clinical info:
    features = extract_all_features(ppg_signal, fs=1000,
                                    clinical=dict(age=57, bmi=23.1, ...))
"""

import numpy as np
from scipy.signal import firwin, lfilter, find_peaks
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from collections import OrderedDict
import warnings
import os
import h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# =============================================================================
# 1.  PREPROCESSING — FIR bandpass filter (0.5-8 Hz, 50th order, Hamming)
# =============================================================================

def design_fir_filter(fs, lowcut=0.5, highcut=8.0, numtaps=50):
    """50th-order FIR bandpass filter with Hamming window (same as paper)."""
    if numtaps % 2 == 0:
        numtaps += 1  # firwin requires odd length
    taps = firwin(numtaps, [lowcut, highcut], window='hamming',
                  pass_zero='bandpass', fs=fs)
    return taps


def apply_fir_filter(signal, taps):
    """Apply FIR filter to signal, compensating for delay."""
    # Use filtfilt-like behavior: apply filter twice for zero-phase
    # but paper uses standard FIR so we use lfilter
    filtered = lfilter(taps, 1.0, signal)
    return filtered


# =============================================================================
# 2.  FEATURE POINT DETECTION
#     PPG:  O (onset), S (systolic peak), N (dicrotic notch), D (diastolic peak)
#     VPG:  w (main peak), y (trough), z (sub-peak)
#     APG:  a, b, c, d, e waves
# =============================================================================

def _detect_cycles_ppg(ppg, fs):
    """
    Detect individual cardiac cycles in PPG and return indices of
    O, S, N, D for each valid cycle.

    Strategy:
      1. Find systolic peaks (S) via find_peaks with adaptive distance.
      2. For each S, find the preceding valley → O (onset).
      3. Between S and next S, find:
         - The next significant peak after S  → D (diastolic peak)
         - The minimum between S and D        → N (dicrotic notch)

    Returns list of dicts [{'O':i, 'S':i, 'N':i, 'D':i}, ...]
    """
    min_distance = int(0.4 * fs)   # at least 400ms between beats
    peaks, props = find_peaks(ppg, distance=min_distance,
                              prominence=0.05 * (np.max(ppg) - np.min(ppg)))

    cycles = []
    for i in range(len(peaks)):
        s_idx = peaks[i]

        # Onset O = preceding valley before S
        # Search from previous S (or start) to current S
        prev_s = peaks[i - 1] if i > 0 else max(0, s_idx - int(1.5 * fs))
        o_idx = np.argmin(ppg[prev_s:s_idx]) + prev_s

        # End bound: next S or end of signal
        next_s = peaks[i + 1] if i + 1 < len(peaks) else len(ppg) - 1

        # Diastolic peak D: the highest peak between current S and next S
        # but at least 200ms after S (refractory)
        search_start = s_idx + int(0.2 * fs)
        search_end = next_s
        if search_start >= search_end:
            continue

        segment = ppg[search_start:search_end]
        if len(segment) < 3:
            d_idx = search_start + len(segment) // 2
        else:
            d_peaks, _ = find_peaks(segment, distance=int(0.1 * fs))
            if len(d_peaks) > 0:
                d_idx = search_start + d_peaks[0]
            else:
                d_idx = search_start + np.argmax(segment)

        # Dicrotic notch N = minimum between S and D
        nd_region = ppg[s_idx:d_idx]
        if len(nd_region) > 2:
            # The notch is the inflection point — often a local min
            # but could be just a shoulder. Find global min in S→D.
            n_candidates, _ = find_peaks(-ppg[s_idx:d_idx],
                                         distance=int(0.02 * fs))
            if len(n_candidates) > 0:
                n_idx = s_idx + n_candidates[-1]  # last min before D
            else:
                n_idx = s_idx + np.argmin(nd_region)
        else:
            n_idx = s_idx + (d_idx - s_idx) // 2

        if n_idx >= d_idx:
            n_idx = s_idx + (d_idx - s_idx) // 2

        cycles.append({'O': int(o_idx), 'S': int(s_idx),
                       'N': int(n_idx), 'D': int(d_idx)})

    return cycles


def _compute_vpg(ppg):
    """First derivative of PPG → VPG (velocity photoplethysmography)."""
    return np.gradient(ppg)


def _compute_apg(vpg):
    """Second derivative of PPG → APG (acceleration photoplethysmography)."""
    return np.gradient(vpg)


def _detect_vpg_points(vpg, cycle):
    """
    Detect VPG feature points within one cardiac cycle:
      w — main peak (maximum)
      y — trough (minimum between w and z)
      z — sub-peak (peak after y)
    """
    s, d = cycle['S'], cycle['D']
    segment = vpg[s:d]
    if len(segment) < 5:
        return {'w': s, 'y': s + (d - s) // 2, 'z': d}

    # w = global max in S→D
    w_rel = np.argmax(segment)
    w_idx = s + w_rel

    # After w, find the trough y
    post_w = segment[w_rel:]
    if len(post_w) > 3:
        y_rel = np.argmin(post_w) + w_rel
    else:
        y_rel = w_rel + len(post_w) // 2
    y_idx = s + y_rel

    # After y, find the sub-peak z
    post_y = segment[y_rel:]
    if len(post_y) > 3:
        z_peaks, _ = find_peaks(post_y, distance=2)
        if len(z_peaks) > 0:
            z_rel = z_peaks[0] + y_rel
        else:
            z_rel = y_rel + np.argmax(post_y)
    else:
        z_rel = y_rel
    z_idx = s + z_rel

    return {'w': int(w_idx), 'y': int(y_idx), 'z': int(z_idx)}


def _detect_apg_points(apg, cycle):
    """
    Detect APG feature points within one cardiac cycle:
      a — early systolic positive wave (first prominent peak)
      b — early systolic negative wave (trough after a)
      c — late systolic re-increasing wave (peak after b)
      d — late systolic decreasing wave (trough after c)
      e — early diastolic positive wave / dicrotic notch (peak after d)
    """
    s, d = cycle['S'], cycle['D']
    segment = apg[s:d]
    if len(segment) < 5:
        mid = s + (d - s) // 2
        return {'a': s, 'b': s + 1, 'c': mid, 'd': mid + 1, 'e': d}

    segment_len = len(segment)
    third = segment_len // 3

    # a = first major peak (within first third)
    region_a = segment[:third] if third > 5 else segment
    a_rel = np.argmax(region_a)
    a_idx = s + a_rel

    # b = first trough after a
    post_a = segment[a_rel:]
    if len(post_a) > 3:
        b_rel = np.argmin(post_a[:len(post_a)//2]) + a_rel
    else:
        b_rel = a_rel + 1
    b_idx = s + b_rel

    # c = next peak after b
    post_b = segment[b_rel:]
    if len(post_b) > 3:
        c_peaks, _ = find_peaks(post_b, distance=2)
        if len(c_peaks) > 0:
            c_rel = c_peaks[0] + b_rel
        else:
            c_rel = b_rel + np.argmax(post_b)
    else:
        c_rel = b_rel
    c_idx = s + c_rel

    # d = trough after c
    post_c = segment[c_rel:]
    if len(post_c) > 3:
        d_rel_min = np.argmin(post_c[:len(post_c)//2]) + c_rel
    else:
        d_rel_min = c_rel + 1
    d_idx = s + d_rel_min

    # e = peak after d (diastolic, should be near the dicrotic notch)
    post_d = segment[d_rel_min:]
    if len(post_d) > 3:
        e_peaks, _ = find_peaks(post_d, distance=2)
        if len(e_peaks) > 0:
            e_rel_local = e_peaks[0] + d_rel_min
        else:
            e_rel_local = d_rel_min + np.argmax(post_d)
    else:
        e_rel_local = d_rel_min
    e_idx = s + e_rel_local

    return {'a': int(a_idx), 'b': int(b_idx), 'c': int(c_idx),
            'd': int(d_idx), 'e': int(e_idx)}


def extract_feature_points(ppg, fs):
    """
    Full feature point detection returning per-cycle indices for
    PPG, VPG, and APG.
    """
    vpg = _compute_vpg(ppg)
    apg = _compute_apg(vpg)

    cycles = _detect_cycles_ppg(ppg, fs)

    ppg_points = []
    vpg_points = []
    apg_points = []

    for cycle in cycles:
        ppg_points.append(cycle)
        vpg_points.append(_detect_vpg_points(vpg, cycle))
        apg_points.append(_detect_apg_points(apg, cycle))

    return ppg_points, vpg_points, apg_points, vpg, apg


# =============================================================================
# 3.  FEATURE EXTRACTION FUNCTIONS
# =============================================================================

def _norm_amplitude(signal, amp_ref=None):
    """Normalize amplitude: amplitude / peak-to-peak range."""
    if amp_ref is None:
        amp_ref = np.max(signal) - np.min(signal)
    if amp_ref == 0:
        amp_ref = 1.0
    return amp_ref


def extract_temporal_features(ppg, ppg_pts, vpg_pts, apg_pts, fs):
    """
    60 temporal features:
      - PPG (22): time spans between O, S, N, D and mapping to APG/VPG
      - VPG (27): time spans between w, y, z and mapping to APG
      - APG (11): time spans between a, b, c, d, e

    Returns dict of 60 feature_name → value (averaged across cycles).
    """
    feats = OrderedDict()

    if len(ppg_pts) == 0:
        return {f'T_{i}': 0.0 for i in range(60)}

    def _T(p1, p2):
        return (p2 - p1) / fs * 1000  # convert to ms

    # ---- PPG temporal (22) ----
    ppg_t_names = [
        ('OS', 'O', 'S'), ('SD', 'S', 'D'), ('ND', 'N', 'D'),
        ('SN', 'S', 'N'), ('OD', 'O', 'D'), ('ON', 'O', 'N'),
        ('Oa', 'O', 'a'), ('Ob', 'O', 'b'), ('Oc', 'O', 'c'),
        ('Od', 'O', 'd'), ('Oe', 'O', 'e'),
        ('Sa', 'S', 'a'), ('Sb', 'S', 'b'), ('Sc', 'S', 'c'),
        ('Sd', 'S', 'd'), ('Se', 'S', 'e'),
        ('Na', 'N', 'a'), ('Nb', 'N', 'b'), ('Nc', 'N', 'c'),
        ('Nd', 'N', 'd'), ('Ne', 'N', 'e'),
        ('OO', 'O', 'O'),  # placeholder for CT/period
    ]
    ppg_t_vals = []
    for name, p1, p2 in ppg_t_names:
        vals = []
        for c_idx in range(len(ppg_pts)):
            pp = ppg_pts[c_idx]
            ap = apg_pts[c_idx] if c_idx < len(apg_pts) else pp
            # Resolve point index
            def _idx(pt_dict, key):
                if key in pt_dict:
                    return pt_dict[key]
                return pt_dict.get(key.upper(), pt_dict.get(pt_dict.keys()[0]))
            i1 = pp.get(p1.upper(), pp.get(p1))
            i2 = ap.get(p2.upper(), ap.get(p2)) if p2 in 'abcde' else pp.get(p2.upper(), pp.get(p2))
            if i1 is not None and i2 is not None:
                vals.append(abs(_T(i1, i2)))
        ppg_t_vals.append(np.mean(vals) if vals else 0.0)

    # Rename and assign PPG temporal
    pg_map = {0: 'T_OS', 1: 'T_SD', 2: 'T_ND', 3: 'T_SN', 4: 'T_OD',
              5: 'T_ON', 6: 'T_Oa', 7: 'T_Ob', 8: 'T_Oc', 9: 'T_Od',
              10: 'T_Oe', 11: 'T_Sa', 12: 'T_Sb', 13: 'T_Sc', 14: 'T_Sd',
              15: 'T_Se', 16: 'T_Na', 17: 'T_Nb', 18: 'T_Nc', 19: 'T_Nd',
              20: 'T_Ne', 21: 'CT'}
    for k, v in pg_map.items():
        feats[v] = ppg_t_vals[k]

    # Heart period (mean inter-beat interval)
    if len(ppg_pts) > 1:
        s_peaks = [p['S'] for p in ppg_pts]
        ibi = np.diff(s_peaks) / fs * 1000
        feats['CT'] = np.mean(ibi)
        feats['T_ww'] = np.mean(ibi)   # same as heart period
    else:
        feats['CT'] = ppg_t_vals[21]
        feats['T_ww'] = ppg_t_vals[21]

    # ---- VPG temporal (27) ----
    vpg_t_names = [
        ('wy', 'w', 'y'), ('yz', 'y', 'z'), ('wz', 'w', 'z'),
        ('wa', 'w', 'a'), ('wb', 'w', 'b'), ('wc', 'w', 'c'),
        ('wd', 'w', 'd'), ('we', 'w', 'e'),
        ('ya', 'y', 'a'), ('yb', 'y', 'b'), ('yc', 'y', 'c'),
        ('yd', 'y', 'd'), ('ye', 'y', 'e'),
        ('za', 'z', 'a'), ('zb', 'z', 'b'), ('zc', 'z', 'c'),
        ('zd', 'z', 'd'), ('ze', 'z', 'e'),
        ('aw', 'a', 'w'), ('bw', 'b', 'w'), ('cw', 'c', 'w'),
        ('dw', 'd', 'w'), ('ew', 'e', 'w'),
        ('oa/ww', None, None), ('oc/ww', None, None),
        ('oe/ww', None, None), ('ob/ww', None, None),
    ]
    for name, p1, p2 in vpg_t_names:
        if name.endswith('/ww') or name == 'oa/ww':
            # Ratio features
            if name == 'oa/ww':
                # T_Oa / T_ww
                val = feats.get('T_Oa', 0) / max(feats.get('T_ww', 1), 1)
            elif name == 'oc/ww':
                val = feats.get('T_Oc', 0) / max(feats.get('T_ww', 1), 1)
            elif name == 'oe/ww':
                val = feats.get('T_Oe', 0) / max(feats.get('T_ww', 1), 1)
            elif name == 'ob/ww':
                val = feats.get('T_Ob', 0) / max(feats.get('T_ww', 1), 1)
            else:
                val = 0.0
            feats[f'T_{name}'] = val
            continue

        vals = []
        for c_idx in range(len(vpg_pts)):
            vp = vpg_pts[c_idx]
            ap = apg_pts[c_idx] if c_idx < len(apg_pts) else {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
            d1 = vp.get(p1, vp.get(p1.upper()))
            d2 = ap.get(p2, ap.get(p2.upper())) if p2 in 'abcde' else vp.get(p2, vp.get(p2.upper()))
            # Fall back
            if d1 is None or d2 is None:
                continue
            vals.append(abs(_T(d1, d2)))
        feats[f'T_{name}'] = np.mean(vals) if vals else 0.0

    # ---- APG temporal (11) ----
    apg_t_pairs = [('ab', 'a', 'b'), ('ac', 'a', 'c'), ('ad', 'a', 'd'),
                   ('ae', 'a', 'e'), ('bc', 'b', 'c'), ('bd', 'b', 'd'),
                   ('be', 'b', 'e'), ('cd', 'c', 'd'), ('ce', 'c', 'e'),
                   ('de', 'd', 'e'), ('ad', 'a', 'd')]
    for name, p1, p2 in apg_t_pairs:
        vals = []
        for ap in apg_pts:
            i1, i2 = ap.get(p1), ap.get(p2)
            if i1 is not None and i2 is not None:
                vals.append(abs(_T(i1, i2)))
        feats[f'T_{name}'] = np.mean(vals) if vals else 0.0

    return feats


def extract_amplitude_features(ppg, ppg_pts, vpg, vpg_pts, apg, apg_pts, fs):
    """
    45 amplitude features:
      - Feature point amplitudes
      - Differences between amplitudes
      - Ratios of amplitudes
    """
    feats = OrderedDict()

    if len(ppg_pts) == 0:
        return {f'A_{i}': 0.0 for i in range(45)}

    # Get amplitude values at each point
    def _A(signal, idx):
        return signal[idx]

    amp_ref = np.max(ppg) - np.min(ppg)
    if amp_ref == 0:
        amp_ref = 1.0

    # ---- PPG amplitude (15ish) ----
    ppg_amp_names = ['A_S', 'A_N', 'A_D', 'A_O',
                     'A_SD', 'A_ND', 'A_SN', 'A_ON', 'A_OS', 'A_OD']
    ppg_amp_vals = []
    for name in ppg_amp_names:
        vals = []
        for p in ppg_pts:
            if name == 'A_SD':
                v = abs(_A(ppg, p['S']) - _A(ppg, p['D']))
            elif name == 'A_ND':
                v = abs(_A(ppg, p['N']) - _A(ppg, p['D']))
            elif name == 'A_SN':
                v = abs(_A(ppg, p['S']) - _A(ppg, p['N']))
            elif name == 'A_ON':
                v = abs(_A(ppg, p['O']) - _A(ppg, p['N']))
            elif name == 'A_OS':
                v = abs(_A(ppg, p['O']) - _A(ppg, p['S']))
            elif name == 'A_OD':
                v = abs(_A(ppg, p['O']) - _A(ppg, p['D']))
            else:
                pt = p.get(name[2])  # A_S → 'S'
                if pt is None:
                    continue
                v = _A(ppg, pt)
            vals.append(v)
        ppg_amp_vals.append(np.mean(vals) if vals else 0.0)

    for i, name in enumerate(ppg_amp_names):
        feats[name] = ppg_amp_vals[i]

    # ---- VPG amplitude ----
    vpg_amp_items = [
        ('A_w', 'w'), ('A_y', 'y'), ('A_z', 'z'),
        ('A_wy', 'w', 'y', 'diff'),
        ('A_wz', 'w', 'z', 'diff'),
        ('A_yz', 'y', 'z', 'diff'),
    ]
    for item in vpg_amp_items:
        name = item[0]
        vals = []
        for vp in vpg_pts:
            i1 = vp[item[1]]
            if len(item) == 2:
                v = _A(vpg, i1)
            elif item[3] == 'diff':
                i2 = vp[item[2]]
                v = abs(_A(vpg, i1) - _A(vpg, i2))
            vals.append(v)
        feats[name] = np.mean(vals) if vals else 0.0

    # VPG ratios
    vpg_ratio_items = [
        ('A_y/w', 'y', 'w'), ('A_z/w', 'z', 'w'), ('A_y/z', 'y', 'z')
    ]
    for name, p1, p2 in vpg_ratio_items:
        vals = []
        for vp in vpg_pts:
            v1, v2 = abs(_A(vpg, vp[p1])), abs(_A(vpg, vp[p2]))
            if v2 != 0:
                vals.append(v1 / v2)
        feats[name] = np.mean(vals) if vals else 0.0

    # ---- APG amplitude ----
    apg_amp_items = [
        ('A_a', 'a'), ('A_b', 'b'), ('A_c', 'c'), ('A_d', 'd'), ('A_e', 'e'),
    ]
    for name, pt in apg_amp_items:
        vals = [abs(_A(apg, ap[pt])) for ap in apg_pts if pt in ap]
        feats[name] = np.mean(vals) if vals else 0.0

    # APG ratios
    apg_ratio_items = [
        ('A_b/a', 'b', 'a'), ('A_c/a', 'c', 'a'),
        ('A_d/a', 'd', 'a'), ('A_e/a', 'e', 'a'),
    ]
    for name, p1, p2 in apg_ratio_items:
        vals = []
        for ap in apg_pts:
            v1, v2 = abs(_A(apg, ap[p1])), abs(_A(apg, ap[p2]))
            if v2 != 0:
                vals.append(v1 / v2)
        feats[name] = np.mean(vals) if vals else 0.0

    # APG area feature: A_bcde/a (area of waveform enclosed by b-c-d-e / amp of a)
    vals_bcde = []
    for ap in apg_pts:
        if all(k in ap for k in ['b', 'c', 'd', 'e']):
            idx_b, idx_c, idx_d, idx_e = ap['b'], ap['c'], ap['d'], ap['e']
            segment = apg[idx_b:idx_e+1]
            area = np.trapz(np.abs(segment)) if len(segment) > 1 else 0
            amp_a = abs(_A(apg, ap['a']))
            vals_bcde.append(area / max(amp_a, 1e-10))
    feats['A_bcde/a'] = np.mean(vals_bcde) if vals_bcde else 0.0

    return feats


def extract_pulse_width_features(ppg, ppg_pts, fs):
    """
    7 pulse width features:
      Width at various percentages (25%, 33%, 50%, 66%, 75%)
      of the maximum pulse width (from systole to diastole).
    """
    feats = OrderedDict()
    if len(ppg_pts) == 0:
        return {f'Width_{i}': 0.0 for i in range(7)}

    percentages = [25, 33, 50, 66, 75]
    width_vals = {p: [] for p in percentages}

    for p in ppg_pts:
        o, s, n, d = p['O'], p['S'], p['N'], p['D']
        pulse = ppg[o:d+1]
        if len(pulse) < 3:
            continue

        # Max pulse amplitude
        amp_max = ppg[s] - ppg[o]
        if amp_max <= 0:
            continue

        for pct in percentages:
            threshold = ppg[o] + amp_max * pct / 100.0
            # Find crossing points on rising and falling edges
            rising = pulse[:s-o] if s-o > 0 else pulse
            falling = pulse[s-o:] if s-o < len(pulse) else pulse

            # Rising edge crossing
            cross_up = None
            for j in range(len(rising) - 1):
                if rising[j] <= threshold <= rising[j+1]:
                    cross_up = j
                    break
            # Falling edge crossing
            cross_down = None
            for j in range(len(falling) - 1):
                if falling[j] >= threshold >= falling[j+1]:
                    cross_down = j + (s - o)
                    break

            if cross_up is not None and cross_down is not None:
                width_samples = cross_down - cross_up
                width_vals[pct].append(width_samples / fs * 1000)  # ms

    for pct in percentages:
        feats[f'Width_{pct}pct'] = np.mean(width_vals[pct]) if width_vals[pct] else 0.0

    # Max pulse width (full width at baseline)
    max_widths = [(p['D'] - p['O']) / fs * 1000 for p in ppg_pts]
    feats['Width_max'] = np.mean(max_widths) if max_widths else 0.0

    return feats


def extract_pulse_area_features(ppg, ppg_pts, fs):
    """
    8 pulse area features:
      Areas enclosed by PPG feature points.
    """
    feats = OrderedDict()
    if len(ppg_pts) == 0:
        return {f'Area_{i}': 0.0 for i in range(8)}

    area_items = [
        ('OS', 'O', 'S'), ('SD', 'S', 'D'), ('ND', 'N', 'D'),
        ('SN', 'S', 'N'), ('ON', 'O', 'N'), ('OD', 'O', 'D'),
        ('OSN', 'O', 'S', 'N'), ('OSD', 'O', 'S', 'D'),
    ]
    for item in area_items:
        name = item[0]
        vals = []
        for p in ppg_pts:
            indices = [p[key] for key in item[1:] if key in p]
            if len(indices) < 2:
                continue
            start, end = indices[0], indices[-1]
            segment = ppg[start:end+1]
            area = np.trapz(segment - np.min(segment)) if len(segment) > 1 else 0
            vals.append(area)
        feats[f'Area_{name}'] = np.mean(vals) if vals else 0.0

    return feats


def extract_angle_features(ppg, ppg_pts, vpg, vpg_pts, apg, apg_pts, fs):
    """
    18 angle/slope features:
      Slopes/angles between feature points in PPG, VPG, and APG.
    """
    feats = OrderedDict()
    if len(ppg_pts) == 0:
        return {f'Angle_{i}': 0.0 for i in range(18)}

    def _angle(signal, i1, i2):
        """Angle in degrees between two points on a signal."""
        dx = (i2 - i1) / fs
        dy = signal[i2] - signal[i1]
        if abs(dx) < 1e-10:
            return 90.0 if dy > 0 else -90.0
        return np.degrees(np.arctan2(dy, dx))

    # PPG angles
    ppg_angle_items = ['SO', 'SD', 'ND', 'SN', 'OD', 'ON']
    for item in ppg_angle_items:
        vals = []
        for p in ppg_pts:
            i1, i2 = p[item[0]], p[item[1]]
            vals.append(_angle(ppg, i1, i2))
        feats[f'Angle_{item}'] = np.mean(vals) if vals else 0.0

    # VPG angles
    vpg_angle_items = ['wy', 'wz', 'yz', 'yw', 'zw', 'zy']
    for item in vpg_angle_items:
        vals = []
        for vp in vpg_pts:
            i1, i2 = vp[item[0]], vp[item[1]]
            vals.append(_angle(vpg, i1, i2))
        feats[f'Angle_{item}'] = np.mean(vals) if vals else 0.0

    # APG angles
    apg_angle_items = ['ab', 'bc', 'cd', 'de', 'ae', 'ad']
    for item in apg_angle_items:
        vals = []
        for ap in apg_pts:
            i1, i2 = ap[item[0]], ap[item[1]]
            vals.append(_angle(apg, i1, i2))
        feats[f'Angle_{item}'] = np.mean(vals) if vals else 0.0

    return feats


# =============================================================================
# GPU-accelerated entropy helpers (PyTorch with CPU fallback)
# =============================================================================
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available() and os.environ.get('LIU2023_NOGPU') != '1'
except Exception:
    _HAS_CUDA = False


def _ap_en_gpu(signal, m=2, r=None):
    """Approximate entropy with GPU acceleration via PyTorch."""
    device = 'cuda' if _HAS_CUDA else 'cpu'
    sig = torch.as_tensor(signal, dtype=torch.float64, device=device)
    N = len(signal)
    r_val = 0.15 * float(np.std(signal)) if r is None else r

    def _phi(m_val):
        rows = torch.arange(N - m_val + 1, device=device)[:, None]
        pat = sig[rows + torch.arange(m_val, device=device)]
        dists = torch.max(torch.abs(pat[:, None] - pat), dim=2).values
        count = torch.sum(dists <= r_val, dim=1)
        return float(torch.mean(
            torch.log(count.to(torch.float64) / (N - m_val + 1))
        ).item())

    if N - m + 1 <= 0:
        return 0.0
    return abs(_phi(m) - _phi(m + 1))


def _samp_en_gpu(signal, m=2, r=None):
    """Sample entropy with GPU acceleration via PyTorch."""
    device = 'cuda' if _HAS_CUDA else 'cpu'
    sig = torch.as_tensor(signal, dtype=torch.float64, device=device)
    N = len(signal)
    r_val = 0.15 * float(np.std(signal)) if r is None else r

    def _count(m_val):
        rows = torch.arange(N - m_val, device=device)[:, None]
        pat = sig[rows + torch.arange(m_val, device=device)]
        dists = torch.max(torch.abs(pat[:, None] - pat), dim=2).values
        mask = ~torch.eye(len(pat), dtype=torch.bool, device=device)
        return int(torch.sum(dists[mask] <= r_val).item())

    A = _count(m + 1)
    B = _count(m)
    if B == 0:
        return 0.0
    import math
    return -math.log(A / B)


def _fuzzy_en_gpu(signal, m=2, r=None, n_exp=2):
    """Fuzzy entropy with GPU acceleration via PyTorch."""
    device = 'cuda' if _HAS_CUDA else 'cpu'
    sig = torch.as_tensor(signal, dtype=torch.float64, device=device)
    N = len(signal)
    r_val = 0.15 * float(np.std(signal)) if r is None else r

    rows_m = torch.arange(N - m + 1, device=device)[:, None]
    pat_m = sig[rows_m + torch.arange(m, device=device)]
    pat_m = pat_m - torch.mean(pat_m, dim=1, keepdim=True)

    rows_m1 = torch.arange(N - m, device=device)[:, None]
    pat_m1 = sig[rows_m1 + torch.arange(m + 1, device=device)]
    pat_m1 = pat_m1 - torch.mean(pat_m1, dim=1, keepdim=True)

    def _fuzzy_phi(pat_set):
        dists = torch.max(torch.abs(pat_set[:, None] - pat_set), dim=2).values
        fuzzy_sim = torch.exp(-(dists ** n_exp) / r_val)
        mask = ~torch.eye(len(pat_set), dtype=torch.bool, device=device)
        return float(torch.mean(fuzzy_sim[mask]).item())

    phi_m = _fuzzy_phi(pat_m) if len(pat_m) > 0 else 0.0
    phi_m1 = _fuzzy_phi(pat_m1) if len(pat_m1) > 0 else 0.0
    if phi_m == 0:
        return 0.0
    import math
    return -math.log(phi_m1 / phi_m)


def _perm_en_gpu(signal, m=3):
    """Permutation entropy with GPU acceleration via PyTorch."""
    device = 'cuda' if _HAS_CUDA else 'cpu'
    N = len(signal)
    if N - m + 1 <= 0:
        return 0.0
    sig = torch.as_tensor(signal, dtype=torch.float64, device=device)
    # Sliding-window patterns on GPU
    idx = torch.arange(m, device=device)[None, :] + torch.arange(N - m + 1, device=device)[:, None]
    patterns = sig[idx]
    perm = torch.argsort(patterns, dim=1)  # (N-m+1, m)
    # Counting — move to CPU, m is tiny
    perm_np = perm.cpu().numpy()
    perm_keys = [tuple(p) for p in perm_np]
    _, counts = np.unique(perm_keys, return_counts=True, axis=0)
    probs = counts / len(perm_keys)
    return -np.sum(probs * np.log2(probs + 1e-10))


def extract_entropy_features(ppg, fs):
    """
    4 entropy features: Approximate, Sample, Fuzzy, Permutation entropy.
    """
    feats = OrderedDict()
    n = len(ppg)

    if n < 100:
        return {'ApEn': 0.0, 'SampEn': 0.0, 'FuzzyEn': 0.0, 'PermEn': 0.0}

    # ---- Permutation entropy (shared by both GPU and CPU paths) ----
    def _perm_en(signal, m=3):
        N = len(signal)
        patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
        perm_keys = [tuple(np.argsort(p)) for p in patterns]
        unique, counts = np.unique(perm_keys, return_counts=True, axis=0)
        probs = counts / len(perm_keys)
        return -np.sum(probs * np.log2(probs + 1e-10))

    # GPU-accelerated path (PyTorch + CUDA, disabled in multiprocessing workers)
    if _HAS_CUDA and os.environ.get('LIU2023_NOGPU') != '1':
        try:
            r = 0.15 * np.std(ppg)
            feats['ApEn'] = _ap_en_gpu(ppg, r=r)
            feats['SampEn'] = _samp_en_gpu(ppg, r=r)
            feats['FuzzyEn'] = _fuzzy_en_gpu(ppg, r=r)
            feats['PermEn'] = _perm_en(ppg)
            return feats
        except Exception as _e:
            import warnings
            warnings.warn(f"GPU entropy failed ({_e}), falling back to CPU")

    # ---- Original CPU implementation ----
    m = 2
    r = 0.15 * np.std(ppg)

    def _ap_en(signal, m=m, r=r):
        N = len(signal)
        def _phi(m_val):
            patterns = np.array([signal[i:i+m_val] for i in range(N - m_val + 1)])
            count = np.sum(np.max(np.abs(patterns[:, np.newaxis] - patterns), axis=2) <= r, axis=1)
            return np.mean(np.log(count / (N - m_val + 1)))
        if N - m + 1 <= 0:
            return 0.0
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        return abs(phi_m - phi_m1)

    def _samp_en(signal, m=m, r=r):
        N = len(signal)
        def _count_matches(m_val):
            patterns = np.array([signal[i:i+m_val] for i in range(N - m_val)])
            dists = np.max(np.abs(patterns[:, np.newaxis] - patterns), axis=2)
            mask = ~np.eye(len(patterns), dtype=bool)
            return np.sum(dists[mask] <= r)
        A = _count_matches(m + 1)
        B = _count_matches(m)
        if B == 0:
            return 0.0
        return -np.log(A / B)

    def _fuzzy_en(signal, m=m, r=r, n_exp=2):
        N = len(signal)
        patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
        local_mean = np.mean(patterns, axis=1, keepdims=True)
        patterns = patterns - local_mean

        patterns_m1 = np.array([signal[i:i+m+1] for i in range(N - m)])
        local_mean_m1 = np.mean(patterns_m1, axis=1, keepdims=True)
        patterns_m1 = patterns_m1 - local_mean_m1

        def _fuzzy_phi(pat_set):
            dists = np.max(np.abs(pat_set[:, np.newaxis] - pat_set), axis=2)
            fuzzy_sim = np.exp(-(dists ** n_exp) / r)
            mask = ~np.eye(len(pat_set), dtype=bool)
            return np.mean(fuzzy_sim[mask])

        phi_m = _fuzzy_phi(patterns) if len(patterns) > 0 else 0
        phi_m1 = _fuzzy_phi(patterns_m1) if len(patterns_m1) > 0 else 0
        if phi_m == 0:
            return 0.0
        return -np.log(phi_m1 / phi_m)

    # Compute (with safety for short signals)
    try:
        feats['ApEn'] = _ap_en(ppg)
    except Exception:
        feats['ApEn'] = 0.0

    try:
        feats['SampEn'] = _samp_en(ppg)
    except Exception:
        feats['SampEn'] = 0.0

    try:
        feats['FuzzyEn'] = _fuzzy_en(ppg)
    except Exception:
        feats['FuzzyEn'] = 0.0

    try:
        feats['PermEn'] = _perm_en(ppg)
    except Exception:
        feats['PermEn'] = 0.0

    return feats


def extract_frequency_features(ppg, fs):
    """
    5 frequency domain features:
      - Peak frequency (max magnitude)
      - Mean frequency (spectral centroid)
      - Spectral entropy
      - Power in low-frequency band (0.5-2 Hz)
      - Power in high-frequency band (2-8 Hz)
    """
    feats = OrderedDict()
    n = len(ppg)
    if n < 10:
        return {'F_peak': 0.0, 'F_mean': 0.0, 'F_entropy': 0.0,
                'F_power_LF': 0.0, 'F_power_HF': 0.0}

    yf = fft(ppg)
    xf = fftfreq(n, 1/fs)
    mag = np.abs(yf[:n//2])
    freq = xf[:n//2]

    # Only consider 0.5-8 Hz band (per paper's filter range)
    band_mask = (freq >= 0.5) & (freq <= 8.0)
    freq_band = freq[band_mask]
    mag_band = mag[band_mask]

    if len(mag_band) == 0:
        return {'F_peak': 0.0, 'F_mean': 0.0, 'F_entropy': 0.0,
                'F_power_LF': 0.0, 'F_power_HF': 0.0}

    # Peak frequency
    feats['F_peak'] = freq_band[np.argmax(mag_band)]

    # Mean frequency (spectral centroid)
    feats['F_mean'] = np.sum(freq_band * mag_band) / max(np.sum(mag_band), 1e-10)

    # Spectral entropy
    power_band = mag_band ** 2
    psd = power_band / max(np.sum(power_band), 1e-10)
    feats['F_entropy'] = -np.sum(psd * np.log2(psd + 1e-10))

    # Power in LF (0.5-2 Hz) and HF (2-8 Hz)
    lf_mask = freq_band <= 2.0
    hf_mask = freq_band > 2.0
    feats['F_power_LF'] = np.sum(power_band[lf_mask]) if np.any(lf_mask) else 0.0
    feats['F_power_HF'] = np.sum(power_band[hf_mask]) if np.any(hf_mask) else 0.0

    return feats


def extract_statistical_features(ppg, vpg):
    """
    14 statistical features (general features):
      Mean, std, variance, skewness, kurtosis, min, max, range,
      median, 25th/75th percentile, interquartile range, RMS, crest factor.

    Applied to both PPG and VPG as in the paper.
    """
    feats = OrderedDict()

    for prefix, sig in [('G_ppg', ppg), ('G_vpg', vpg)]:
        if len(sig) == 0:
            for suffix in ['mean', 'std', 'var', 'skew', 'kurt', 'min', 'max',
                           'range', 'med', 'p25', 'p75', 'iqr', 'rms', 'crest']:
                feats[f'{prefix}_{suffix}'] = 0.0
            continue

        feats[f'{prefix}_mean'] = np.mean(sig)
        feats[f'{prefix}_std'] = np.std(sig)
        feats[f'{prefix}_var'] = np.var(sig)
        feats[f'{prefix}_skew'] = skew(sig)
        feats[f'{prefix}_kurt'] = kurtosis(sig)
        feats[f'{prefix}_min'] = np.min(sig)
        feats[f'{prefix}_max'] = np.max(sig)
        feats[f'{prefix}_range'] = np.max(sig) - np.min(sig)
        feats[f'{prefix}_med'] = np.median(sig)
        feats[f'{prefix}_p25'] = np.percentile(sig, 25)
        feats[f'{prefix}_p75'] = np.percentile(sig, 75)
        feats[f'{prefix}_iqr'] = feats[f'{prefix}_p75'] - feats[f'{prefix}_p25']
        feats[f'{prefix}_rms'] = np.sqrt(np.mean(sig ** 2))
        # Crest factor = peak / RMS
        peak_val = max(abs(sig))
        rms_val = feats[f'{prefix}_rms']
        feats[f'{prefix}_crest'] = peak_val / max(rms_val, 1e-10)

    return feats


def extract_mixed_features(ppg, ppg_pts, fs):
    """
    5 mixed features known to correlate with BP:
      - SI: Stiffness Index (height / △T)
      - RI: Reflection Index
      - EI: Enhancement Index
      - IPA: Inflection Point Area
      - LASI: Large Artery Stiffness Index
    """
    feats = OrderedDict()
    if len(ppg_pts) == 0:
        return {'SI': 0.0, 'RI': 0.0, 'EI': 0.0, 'IPA': 0.0, 'LASI': 0.0}

    # SI (Stiffness Index) = height / △T
    # △T = T_SD (time from systolic peak to diastolic peak)
    sd_vals = []
    for p in ppg_pts:
        t_sd = (p['D'] - p['S']) / fs
        if t_sd > 0:
            sd_vals.append(t_sd)
    delta_t = np.mean(sd_vals) if sd_vals else 1.0
    # Height = subject height in cm (not available from signal alone, use default)
    # The paper uses clinical height; here we use a proxy from signal amplitude
    height_cm = 170.0  # default fallback
    feats['SI'] = height_cm / max(delta_t, 0.001)

    # RI (Reflection Index) = (S - N) / S * 100
    ri_vals = []
    for p in ppg_pts:
        s_val = ppg[p['S']]
        n_val = ppg[p['N']]
        if s_val != 0:
            ri_vals.append((s_val - n_val) / s_val * 100)
    feats['RI'] = np.mean(ri_vals) if ri_vals else 0.0

    # EI (Enhancement Index) = D / S
    ei_vals = []
    for p in ppg_pts:
        s_val, d_val = ppg[p['S']], ppg[p['D']]
        if s_val != 0:
            ei_vals.append(d_val / s_val)
    feats['EI'] = np.mean(ei_vals) if ei_vals else 0.0

    # IPA (Inflection Point Area) = area ratio
    ipa_vals = []
    for p in ppg_pts:
        o, s, n, d = p['O'], p['S'], p['N'], p['D']
        total_area = np.trapz(ppg[o:d+1] - ppg[o]) if d > o else 0
        systolic_area = np.trapz(ppg[o:n+1] - ppg[o]) if n > o else 0
        if total_area != 0:
            ipa_vals.append(systolic_area / total_area * 100)
    feats['IPA'] = np.mean(ipa_vals) if ipa_vals else 0.0

    # LASI (Large Artery Stiffness Index) = 1 / T_SD
    # Actually often LASI = height / T_SD (similar to SI)
    feats['LASI'] = 1.0 / max(delta_t, 0.001)

    return feats


def extract_clinical_features(clinical_info=None):
    """
    6 clinical features:
      Gender, Age, Height, Weight, BMI, HR

    clinical_info: dict with keys 'gender', 'age', 'height', 'weight', 'bmi', 'hr'
    """
    feats = OrderedDict()
    defaults = {'gender': 0, 'age': 0, 'height': 0, 'weight': 0, 'bmi': 0, 'hr': 0}
    if clinical_info is not None:
        defaults.update(clinical_info)

    feats['Gender'] = defaults['gender']
    feats['Age'] = defaults['age']
    feats['Height'] = defaults['height']
    feats['Weight'] = defaults['weight']
    feats['BMI'] = defaults['bmi']
    feats['HR'] = defaults['hr']

    return feats


# =============================================================================
# 4.  MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_all_features(ppg_signal, fs=1000, clinical_info=None,
                         apply_filter=True, precomputed_entropy=None):
    """
    Full 172-feature extraction following Liu2023 pipeline.

    Parameters
    ----------
    ppg_signal : 1-D array
        Raw PPG signal.
    fs : int
        Sampling rate in Hz. Paper uses 1000; existing pipeline uses 125.
    clinical_info : dict, optional
        Clinical info: {'gender':0/1, 'age':float, 'height':cm,
                        'weight':kg, 'bmi':float, 'hr':bpm}
    apply_filter : bool
        Whether to apply FIR bandpass filter (0.5-8 Hz). Default True.

    Returns
    -------
    features : OrderedDict of 166 PPG-derived features (+6 clinical if provided)
    meta : dict with feature point locations for debugging
    """
    features = OrderedDict()
    meta = {}

    # ---- Step 1: Filtering ----
    if apply_filter:
        taps = design_fir_filter(fs)
        ppg_filt = apply_fir_filter(ppg_signal, taps)
    else:
        ppg_filt = ppg_signal.copy()

    # ---- Step 2: Feature point detection ----
    ppg_pts, vpg_pts, apg_pts, vpg, apg = extract_feature_points(ppg_filt, fs)
    meta['ppg_points'] = ppg_pts
    meta['vpg_points'] = vpg_pts
    meta['apg_points'] = apg_pts
    meta['n_cycles'] = len(ppg_pts)

    # ---- Step 3: Feature extraction ----
    # (1) Temporal (60)
    feats_t = extract_temporal_features(ppg_filt, ppg_pts, vpg_pts, apg_pts, fs)
    features.update(feats_t)

    # (2) Amplitude (45)
    feats_a = extract_amplitude_features(ppg_filt, ppg_pts, vpg, vpg_pts, apg, apg_pts, fs)
    features.update(feats_a)

    # (3) Pulse Width (7)
    feats_w = extract_pulse_width_features(ppg_filt, ppg_pts, fs)
    features.update(feats_w)

    # (4) Pulse Area (8)
    feats_ar = extract_pulse_area_features(ppg_filt, ppg_pts, fs)
    features.update(feats_ar)

    # (5) Angle/Slope (18)
    feats_ang = extract_angle_features(ppg_filt, ppg_pts, vpg, vpg_pts, apg, apg_pts, fs)
    features.update(feats_ang)

    # (6) Entropy (4)
    if precomputed_entropy is not None:
        features.update(precomputed_entropy)
    else:
        feats_en = extract_entropy_features(ppg_filt, fs)
        features.update(feats_en)

    # (7) Frequency Domain (5)
    feats_f = extract_frequency_features(ppg_filt, fs)
    features.update(feats_f)

    # (8) Statistical (14) — on both PPG and VPG → 28
    feats_st = extract_statistical_features(ppg_filt, vpg)
    features.update(feats_st)

    # (9) Mixed (5)
    feats_m = extract_mixed_features(ppg_filt, ppg_pts, fs)
    features.update(feats_m)

    # (10) Clinical (6)
    feats_c = extract_clinical_features(clinical_info)
    features.update(feats_c)

    meta['n_features'] = len(features)
    return features, meta


def extract_all_features_batch(ppg_signals, fs=1000, clinical_infos=None,
                               apply_filter=True):
    """
    Batch feature extraction for multiple PPG segments.
    Averages features across segments (per paper's methodology).

    Parameters
    ----------
    ppg_signals : list of 1-D arrays
        Multiple PPG segments from the same subject.
    fs, clinical_infos, apply_filter : same as extract_all_features
        clinical_infos should be a single dict applied to all segments.

    Returns
    -------
    avg_features : OrderedDict of averaged features
    all_features : list of per-segment feature dicts
    """
    all_features = []
    for i, sig in enumerate(ppg_signals):
        feats, _ = extract_all_features(sig, fs=fs, clinical_info=clinical_infos,
                                        apply_filter=apply_filter)
        all_features.append(feats)

    # Average across segments
    avg_features = OrderedDict()
    for key in all_features[0].keys():
        vals = [f[key] for f in all_features]
        avg_features[key] = np.mean(vals)

    return avg_features, all_features


# =============================================================================
# 5.  QUALITY CHECK FUNCTIONS
# =============================================================================

def extraction_quality_report(features):
    """
    Print per-category zero-value ratio to quickly identify
    which feature groups failed extraction.

    Parameters
    ----------
    features : OrderedDict or dict
        Feature dict returned by extract_all_features.
    """
    groups = OrderedDict([
        ('Temporal',   [k for k in features if k.startswith('T_') or k == 'CT']),
        ('Amplitude',  [k for k in features if k.startswith('A_')]),
        ('Width',      [k for k in features if k.startswith('Width')]),
        ('Area',       [k for k in features if k.startswith('Area')]),
        ('Angle',      [k for k in features if k.startswith('Angle')]),
        ('Entropy',    ['ApEn', 'SampEn', 'FuzzyEn', 'PermEn']),
        ('Frequency',  [k for k in features if k.startswith('F_')]),
        ('Statistical',[k for k in features if k.startswith('G_')]),
        ('Mixed',      ['SI', 'RI', 'EI', 'IPA', 'LASI']),
        ('Clinical',   ['Gender', 'Age', 'Height', 'Weight', 'BMI', 'HR']),
    ])
    print(f"{'Category':<15} {'Total':>5} {'Zero':>5} {'Zero%':>8}")
    print("-" * 35)
    all_zero = 0
    total = 0
    for group, keys in groups.items():
        zero_count = sum(1 for k in keys if abs(features.get(k, 0)) < 1e-10)
        all_zero += zero_count
        total += len(keys)
        print(f"{group:<15} {len(keys):>5} {zero_count:>5} {zero_count/len(keys)*100:>7.1f}%")
    print("-" * 35)
    print(f"{'TOTAL':<15} {total:>5} {all_zero:>5} {all_zero/total*100:>7.1f}%")


def plot_feature_points(ppg, vpg, apg, ppg_pts, vpg_pts, apg_pts, fs=125):
    """
    Plot the first 2 cardiac cycles with detected feature points
    on PPG, VPG, and APG for visual quality inspection.

    Parameters
    ----------
    ppg, vpg, apg : 1-D arrays
    ppg_pts, vpg_pts, apg_pts : list of dict
        Feature point dicts from extract_feature_points / meta.
    fs : int
        Sampling rate in Hz.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    t = np.arange(len(ppg)) / fs

    # PPG
    ax = axes[0]
    ax.plot(t, ppg, 'b-', label='PPG', linewidth=0.8)
    for p in ppg_pts[:2]:
        for key, color in [('O', 'g'), ('S', 'r'), ('N', 'orange'), ('D', 'm')]:
            if p.get(key) is not None:
                ax.axvline(t[p[key]], color=color, linestyle='--', alpha=0.5)
    ax.set_ylabel('PPG')
    ax.legend(['PPG', 'O', 'S', 'N', 'D'])

    # VPG
    ax = axes[1]
    ax.plot(t, vpg, 'r-', label='VPG', linewidth=0.8)
    for vp in vpg_pts[:2]:
        for key, color in [('w', 'r'), ('y', 'g'), ('z', 'orange')]:
            if vp.get(key) is not None:
                ax.axvline(t[vp[key]], color=color, linestyle='--', alpha=0.5)
    ax.set_ylabel('VPG')
    ax.legend(['VPG', 'w', 'y', 'z'])

    # APG
    ax = axes[2]
    ax.plot(t, apg, 'g-', label='APG', linewidth=0.8)
    for ap in apg_pts[:2]:
        for key, color in [('a', 'r'), ('b', 'g'), ('c', 'orange'),
                           ('d', 'm'), ('e', 'b')]:
            if ap.get(key) is not None:
                ax.axvline(t[ap[key]], color=color, linestyle='--', alpha=0.5)
    ax.set_ylabel('APG')
    ax.legend(['APG', 'a', 'b', 'c', 'd', 'e'])

    ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6.  BATCH EXTRACTION -> H5 (for use with create_data.py)
# =============================================================================

FEATURE_NAMES = None


def _get_feature_names(fs=125, apply_filter=True):
    """Extract a single dummy PPG to get the ordered list of feature names."""
    dummy = np.zeros(1024, dtype=np.float32)
    dummy[100] = 0.1
    for i in range(3):
        start = 300 + i * 300
        end = min(start + 100, 1024)
        if end > start:
            dummy[start:end] += np.hanning(end - start) * 0.5
    feats, _ = extract_all_features(dummy, fs=fs, apply_filter=apply_filter)
    return list(feats.keys())


def _process_window(args):
    """Process a single window in a worker process (module-level for pickling)."""
    if len(args) == 7:
        i, window, fs, clinical_info, apply_filter, names, precomputed_entropy = args
    else:
        i, window, fs, clinical_info, apply_filter, names = args
        precomputed_entropy = None
    feats, _ = extract_all_features(window, fs=fs, clinical_info=clinical_info,
                                    apply_filter=apply_filter,
                                    precomputed_entropy=precomputed_entropy)
    return i, np.array([feats.get(k, 0.0) for k in names], dtype=np.float32)


def extract_all_features_to_h5(h5_in, h5_out, fs=125, apply_filter=True,
                                clinical_info=None, verbose=True, n_jobs=None):
    """
    Batch-extract Liu2023 features from segmented_records.h5 and save to h5.

    Reads ppg windows, extracts 169-dim features per window, and writes:
      - 'features'      : (N, 169) float32 feature matrix
      - 'features_sbp'  : (N, 17)  float32 SBP optimal subset
      - 'features_dbp'  : (N, 13)  float32 DBP optimal subset
      - 'feature_names' : string dataset of all feature names
      - 'sbp' / 'dbp' / 'record_id' : copied from input h5

    Parameters
    ----------
    h5_in : str          Path to segmented_records.h5
    h5_out : str         Output h5 path.
    fs : int             Sampling rate in Hz. Default 125.
    apply_filter : bool  Whether to apply FIR bandpass filter. Default True.
    clinical_info : dict or None  Passed to extract_all_features.
    verbose : bool       Print quality report for first window.
    n_jobs : int or None Number of parallel workers. None = use all CPUs.
    """
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        FEATURE_NAMES = _get_feature_names(fs=fs, apply_filter=apply_filter)

    # Build index maps for optimal subsets
    sbp_idx = [FEATURE_NAMES.index(name) for name in SBP_OPTIMAL_FEATURES
               if name in FEATURE_NAMES]
    dbp_idx = [FEATURE_NAMES.index(name) for name in DBP_OPTIMAL_FEATURES
               if name in FEATURE_NAMES]

    with h5py.File(h5_in, "r") as fin:
        ppg_in = fin["ppg"]
        n = len(ppg_in)

        all_feats = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

        # Quality check on first window (always sequential)
        if verbose and n > 0:
            window0 = ppg_in[0].astype(np.float64)
            feats0, meta0 = extract_all_features(
                window0, fs=fs, clinical_info=clinical_info,
                apply_filter=apply_filter
            )
            all_feats[0] = np.array([feats0.get(k, 0.0) for k in FEATURE_NAMES],
                                    dtype=np.float32)
            print(f"\nFirst window quality check (n_cycles={meta0['n_cycles']}):")
            extraction_quality_report(feats0)

        start = 1 if (verbose and n > 0) else 0
        remaining = n - start

        if remaining == 0:
            pass
        elif n_jobs == 1 or remaining < 100:
            # Sequential
            for i in tqdm(range(start, n), desc="Extracting Liu2023 features",
                          colour="cyan"):
                window = ppg_in[i].astype(np.float64)
                feats, _ = extract_all_features(
                    window, fs=fs, clinical_info=clinical_info,
                    apply_filter=apply_filter
                )
                all_feats[i] = np.array([feats.get(k, 0.0) for k in FEATURE_NAMES],
                                        dtype=np.float32)
        else:
            # Parallel: disable CUDA in workers to avoid context conflicts
            os.environ['LIU2023_NOGPU'] = '1'
            n_jobs = n_jobs or min(cpu_count(), 16)
            n_jobs = max(1, min(n_jobs, cpu_count()))  # sanitise

            # Pre-compute entropy on GPU in the main process
            entropy_precomputed = None
            if _HAS_CUDA:
                if verbose:
                    print(f"\nPre-computing 4 entropy features for {remaining} "
                          f"windows on GPU ({torch.cuda.get_device_name(0)})...")
                entropy_precomputed = []
                pbar = tqdm(range(start, n), desc="GPU entropy pre-compute",
                            colour="green", unit="win")
                for idx in pbar:
                    w = ppg_in[idx].astype(np.float64)
                    r = 0.15 * float(np.std(w))
                    ent = OrderedDict()
                    try:
                        ent['ApEn'] = _ap_en_gpu(w, r=r)
                        ent['SampEn'] = _samp_en_gpu(w, r=r)
                        ent['FuzzyEn'] = _fuzzy_en_gpu(w, r=r)
                        ent['PermEn'] = _perm_en_gpu(w)
                    except Exception as e:
                        warnings.warn(f"GPU entropy failed on window {idx}: {e}, using CPU")
                        ent = extract_entropy_features(w, fs)
                    entropy_precomputed.append(ent)

            def _gen_tasks():
                """Generator yielding one window at a time (avoids memory blowup)."""
                for j in range(start, n):
                    ent = entropy_precomputed[j - start] if entropy_precomputed is not None else None
                    yield (j, ppg_in[j].astype(np.float64), fs, clinical_info,
                           apply_filter, FEATURE_NAMES, ent)

            with Pool(n_jobs) as pool:
                for i, feat_vec in tqdm(
                    pool.imap_unordered(_process_window, _gen_tasks(), chunksize=32),
                    total=remaining, desc="Extracting Liu2023 features",
                    colour="cyan"
                ):
                    all_feats[i] = feat_vec

    with h5py.File(h5_out, "w") as fout:
        fout.create_dataset("features", data=all_feats, dtype=np.float32)
        fout.create_dataset("features_sbp",
                            data=all_feats[:, sbp_idx], dtype=np.float32)
        fout.create_dataset("features_dbp",
                            data=all_feats[:, dbp_idx], dtype=np.float32)

        dt_str = h5py.string_dtype()
        fout.create_dataset("feature_names", data=FEATURE_NAMES, dtype=dt_str)
        fout.create_dataset("feature_names_sbp",
                            data=[FEATURE_NAMES[i] for i in sbp_idx], dtype=dt_str)
        fout.create_dataset("feature_names_dbp",
                            data=[FEATURE_NAMES[i] for i in dbp_idx], dtype=dt_str)

        with h5py.File(h5_in, "r") as fin:
            fout.create_dataset("sbp", data=fin["sbp"][:], dtype=np.float32)
            fout.create_dataset("dbp", data=fin["dbp"][:], dtype=np.float32)
            fout.create_dataset("record_id", data=fin["record_id"][:],
                                dtype=np.int32)

    print(f"\nSaved {n} samples x {len(FEATURE_NAMES)} features -> {h5_out}")
    print(f"  SBP optimal subset: {len(sbp_idx)} features")
    print(f"  DBP optimal subset: {len(dbp_idx)} features")


# =============================================================================
# 7.  OPTIMAL FEATURE SUBSETS (from the paper's Table 3)
# =============================================================================

SBP_OPTIMAL_FEATURES = [
    'Angle_ND', 'Angle_SD', 'A_bcde/a', 'T_ND', 'F_peak',
    'CT', 'T_oc/ww', 'T_wc', 'G_ppg_mean', 'T_oa/ww', 'SI',
    'A_y/z', 'T_Oa', 'A_wz', 'G_vpg_mean', 'T_Sd', 'HR'
]

DBP_OPTIMAL_FEATURES = [
    'T_wc', 'T_ND', 'Angle_zy', 'Angle_de', 'T_SD', 'HR',
    'Angle_yw', 'T_oe/ww', 'T_SN', 'A_y/z', 'A_wz', 'T_ze'
]

# Note on paper-to-code name mappings:
#   Angle_ND  ← Angle_ND       T_SD   ← T_SD (△T)
#   F_peak    ← E_F            CT     ← T_OS
#   G_ppg_mean ← G_mf          SI     ← SI
#   G_vpg_mean ← G_vpgave
#   Angle_de  ← Angle_ed (paper's ed = d→e direction, same as Angle_de)
#
# Note: BMI, Age, Weight (clinical features) removed from both subsets
#       since source dataset only contains PPG+ABP signals.
#       A_ce, Age, BMI, Weight removed from DBP list because A_ce is
#       not implemented in this codebase and the others are clinical.
#       SBP: 20→17, DBP: 16→12 features.
#       HR retained as it can be derived from PPG cycle intervals.


if __name__ == '__main__':
    # ========== 1. Single-segment test ==========
    # print("=" * 60)
    # print("1. Single PPG segment feature extraction test")
    # print("=" * 60)
    # fs = 1000
    # t = np.linspace(0, 2.1, int(2.1 * fs), endpoint=False)
    # # Create synthetic PPG-like signal
    # hr = 72
    # beats = np.sin(2 * np.pi * hr / 60 * t)
    # ppg_synth = beats * 0.5 + 0.5
    # ppg_synth += 0.1 * np.random.randn(len(ppg_synth))

    # feats, meta = extract_all_features(ppg_synth, fs=fs)
    # print(f"Extracted {meta['n_features']} features")
    # print(f"Detected {meta['n_cycles']} cardiac cycles")
    # extraction_quality_report(feats)
    # print()
    # for k, v in list(feats.items())[:10]:
    #     print(f"  {k}: {v:.4f}")
    # print("  ...")
    # print(f"\nSBP optimal subset ({len(SBP_OPTIMAL_FEATURES)} features):")
    # for name in SBP_OPTIMAL_FEATURES:
    #     val = feats.get(name, 'N/A')
    #     print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")

    # print(f"\nDBP optimal subset ({len(DBP_OPTIMAL_FEATURES)} features):")
    # for name in DBP_OPTIMAL_FEATURES:
    #     val = feats.get(name, 'N/A')
    #     print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")

    # # ========== 2. Batch extraction → h5 test (optional) ==========
    # print("\n" + "=" * 60)
    # print("2. Batch extraction to h5 (synthetic data)")
    # print("=" * 60)

    # # Create a synthetic segmented_records.h5 for testing
    # import os
    # import tempfile
    # tmpdir = tempfile.mkdtemp()
    # demo_h5_in = os.path.join(tmpdir, "segmented_demo.h5")
    # demo_h5_out = os.path.join(tmpdir, "liu2023_demo.h5")

    # # Generate 50 synthetic PPG windows (1024 samples each) + dummy sbp/dbp
    # n_windows = 50
    # dummy_ppg = np.zeros((n_windows, 1024), dtype=np.float32)
    # dummy_sbp = np.random.uniform(110, 150, n_windows).astype(np.float32)
    # dummy_dbp = np.random.uniform(60, 90, n_windows).astype(np.float32)
    # dummy_rid = np.zeros(n_windows, dtype=np.int32)

    # for i in range(n_windows):
    #     phase = np.linspace(0, 4 * np.pi, 1024)
    #     dummy_ppg[i] = (np.sin(phase + i * 0.5) * 0.5 + 0.5
    #                     + 0.05 * np.random.randn(1024))

    # with h5py.File(demo_h5_in, "w") as f:
    #     f.create_dataset("ppg", data=dummy_ppg, dtype=np.float32)
    #     f.create_dataset("sbp", data=dummy_sbp, dtype=np.float32)
    #     f.create_dataset("dbp", data=dummy_dbp, dtype=np.float32)
    #     f.create_dataset("record_id", data=dummy_rid, dtype=np.int32)

    # print(f"Created synthetic segmented data: {n_windows} windows → {demo_h5_in}")

    # extract_all_features_to_h5(demo_h5_in, demo_h5_out, fs=125,
    #                             clinical_info=None, verbose=True)

    # # Verify output
    # with h5py.File(demo_h5_out, "r") as f:
    #     print(f"\nOutput verification:")
    #     print(f"  features      shape: {f['features'].shape}")
    #     print(f"  features_sbp  shape: {f['features_sbp'].shape}")
    #     print(f"  features_dbp  shape: {f['features_dbp'].shape}")
    #     print(f"  sbp           shape: {f['sbp'].shape}")
    #     print(f"  feature_names count: {len(f['feature_names'])}")
    #     print(f"  feature_names_sbp  : {list(f['feature_names_sbp'][:])}")
    #     print(f"  feature_names_dbp  : {list(f['feature_names_dbp'][:])}")

    # # Cleanup temp files
    # import shutil
    # shutil.rmtree(tmpdir)
    # print(f"\nCleaned up temp directory: {tmpdir}")
    # print("\nAll tests passed!")

    extract_all_features_to_h5(
        h5_in = r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\segmented_records.h5",
        h5_out=r"E:\Kaggle_projects\Blood_Pressure_analysis\Blood_pressure_dataset\liu2023_features.h5",
        fs=125
    )