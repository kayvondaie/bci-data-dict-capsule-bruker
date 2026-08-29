"""
bonsai_npy_threshold_calculator_multiCN.py

Multi-CN analogue of bonsai_npy_threshold_calculator.py for the
geometric-mean ensemble BCI experiment (N ≤ 10 conditioned neurons).

Key differences from the single-CN version
-------------------------------------------
• CN fluorescence is loaded from suite2p F.npy (neuropil-corrected),
  not from the ScanImage integration ROI CSV.
• Thresholds (2×N per trial) come from *_threshold_M.mat files written
  at each trial start by BCI_analog_display_geomean.m.  Because suite2p
  F values are in different units than ScanImage integration values, the
  .mat thresholds are used only to reconstruct epoch boundaries and the
  cumulative difficulty multiplier M.  Per-CN normalization is recomputed
  from the suite2p spontaneous data (spont_pre, matching MATLAB's
  5th/80th-percentile approach).  Fallback: first-epoch baseline if no
  spont_pre data is present.
• cn_norm (N × frames) and combo (geometric mean, 1 × frames) are
  computed in Python from the above.
• 6-panel figure targets ensemble analysis.

CN identification
-----------------
data['conditioned_neuron'] is inspected first.  For multi-CN experiments
ddc.main() may return only the *primary* (first) CN index there; if so,
pass cn_s2p_indices explicitly after running the inspection cell in the
notebook to determine the correct suite2p ROI indices.

Usage
-----
    from bonsai_npy_threshold_calculator_multiCN import run

    # All trials, auto-detect CN indices:
    figs = run(folder, data)

    # Skip first 40 trials, supply known s2p ROI indices manually:
    figs = run(folder, data, trial_start=40,
               cn_s2p_indices=[12, 7, 45, 3, 88, 21, 56, 30, 14, 62])

Returns a list containing one matplotlib Figure (6-panel QC).
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

# ── Module-level defaults (overridden by run() kwargs) ───────────────────────
TRIAL_START     = 0     # first trial to include; 40 skips early PMT artefacts
N_CNS           = 10    # expected number of conditioned neurons
CN_SMOOTH       = 5     # heatmap smoothing window (trials); 0 = off
NEUROPIL_COEFF  = 0.7   # standard suite2p neuropil correction coefficient


# ─── Internal helpers ────────────────────────────────────────────────────────

def _load_mat_file(path):
    """scipy first (v5/v7), h5py fallback for v7.3 HDF5."""
    try:
        return scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except Exception:
        import h5py
        return h5py.File(str(path), 'r')


def _mat_array(mat, key, dtype=float):
    val = mat[key]
    if hasattr(val, '__getitem__') and not isinstance(val, np.ndarray):
        val = val[()]          # h5py Dataset → numpy
    return np.asarray(val, dtype=dtype)


def _load_threshold_history(folder):
    """
    Find *_threshold_N.mat files (written per trial by BCI_analog_display_geomean.m).

    Returns
    -------
    history       : list of dicts sorted by trial index
                    {'trial': int, 'threshold': ndarray(2, N), 'rois': ndarray(N,)}
    selected_rois : 1-based ScanImage ROI index array from the last file, or None
    """
    search_dir = Path(folder)
    thresh_files = sorted(search_dir.glob('*_threshold_[0-9]*.mat'))
    if not thresh_files:
        thresh_files = sorted(search_dir.parent.glob('*_threshold_[0-9]*.mat'))
    if not thresh_files:
        raise FileNotFoundError(
            f'No *_threshold_N.mat files in {folder}.\n'
            'These are written per-trial by BCI_analog_display_geomean.m.'
        )

    history = []
    for fp in thresh_files:
        m = re.search(r'threshold_(\d+)\.mat$', fp.name)
        if not m:
            continue
        trial_idx = int(m.group(1))
        try:
            mat   = _load_mat_file(fp)
            thresh = _mat_array(mat, 'BCI_threshold')               # (2, N)
            try:
                rois = np.asarray(_mat_array(mat, 'selected_rois'), dtype=int).flatten()
            except Exception:
                rois = None
            history.append({'trial': trial_idx, 'threshold': thresh, 'rois': rois})
        except Exception as e:
            print(f'  WARNING: skipping {fp.name}: {e}')

    history.sort(key=lambda x: x['trial'])
    selected_rois = history[-1]['rois'] if history else None

    # BCI_analog_display_geomean.m saves a file at EVERY trial start, so most
    # consecutive entries have identical upper thresholds.  Keep only the first
    # entry of each run of identical high rows — these are the actual M changes.
    deduped = []
    for h in history:
        if not deduped or not np.allclose(h['threshold'][1], deduped[-1]['threshold'][1]):
            deduped.append(h)

    n_raw = len(history)
    history = deduped
    print(f'  Threshold files: {n_raw} total → {len(history)} distinct M value(s)')
    return history, selected_rois


def _load_s2p_fluorescence(folder, cn_s2p_indices, neuropil_coeff=NEUROPIL_COEFF):
    """
    Load neuropil-corrected fluorescence for the specified suite2p ROI indices.

    Parameters
    ----------
    folder          : session pophys folder (contains suite2p_BCI/plane0/)
    cn_s2p_indices  : (N,) int array — 0-based suite2p ROI indices
    neuropil_coeff  : coefficient for Fneu subtraction (default 0.7)

    Returns
    -------
    Fc : (N, n_frames) neuropil-corrected fluorescence
    """
    plane0 = Path(folder) / 'suite2p_BCI' / 'plane0'
    F    = np.load(str(plane0 / 'F.npy'),    mmap_mode='r')   # (n_ROIs, n_frames)
    Fneu = np.load(str(plane0 / 'Fneu.npy'), mmap_mode='r')   # (n_ROIs, n_frames)
    idx  = np.asarray(cn_s2p_indices, dtype=int)
    Fc   = F[idx, :].astype(float) - neuropil_coeff * Fneu[idx, :].astype(float)
    return Fc


def _identify_cn_indices(data, n_cns):
    """
    Attempt to read suite2p ROI indices for all N CNs from the data dict.

    ddc.main() stores the primary CN in data['conditioned_neuron'].
    For multi-CN sessions it may contain a list / nested structure with all N.

    Returns (indices, reliable) where reliable=False means the caller should
    verify and possibly supply cn_s2p_indices manually.
    """
    raw = data.get('conditioned_neuron', None)
    if raw is None:
        return None, False

    # Flatten any nested MATLAB cell-array structure
    try:
        arr = np.asarray(raw).flatten()
        indices = np.array([int(np.asarray(x).flat[0]) for x in arr])
    except Exception:
        return None, False

    if len(indices) >= n_cns:
        return indices[:n_cns], True
    elif len(indices) == 1:
        # Only primary CN — multi-CN indices not stored in data dict
        print(f'  NOTE: data["conditioned_neuron"] has 1 entry ({indices[0]}); '
              f'expected {n_cns}.\n'
              '  Supply cn_s2p_indices=[...] manually after running the '
              'inspection cell.')
        return None, False
    else:
        # Partial list — use what we have and warn
        print(f'  WARNING: data["conditioned_neuron"] has {len(indices)} entries, '
              f'expected {n_cns}. Using all available.')
        return indices, False


def _compute_normalization(Fc_cn, folder, cn_s2p_indices, n_cns):
    """
    Compute per-CN normalisation bounds (low_i, high_i) from suite2p data.

    Preferred: spontaneous pre-session data (suite2p_spont_pre/plane0/F.npy),
    matching MATLAB's 5th / 80th percentile approach in spont_thresholds_geomean.m.
    Fallback: first 500 frames of the BCI epoch itself.

    Returns
    -------
    low  : (n_cns,) — 5th  percentile (noise floor)
    high : (n_cns,) — 80th percentile (reference upper bound)
    """
    spont_F_path = Path(folder) / 'suite2p_spont_pre' / 'plane0' / 'F.npy'
    spont_Fn_path = Path(folder) / 'suite2p_spont_pre' / 'plane0' / 'Fneu.npy'

    if spont_F_path.exists() and spont_Fn_path.exists():
        idx = np.asarray(cn_s2p_indices, dtype=int)
        F_sp    = np.load(str(spont_F_path),  mmap_mode='r')[idx, :].astype(float)
        Fneu_sp = np.load(str(spont_Fn_path), mmap_mode='r')[idx, :].astype(float)
        Fc_sp   = F_sp - NEUROPIL_COEFF * Fneu_sp
        low  = np.percentile(Fc_sp, 5,  axis=1)
        high = np.percentile(Fc_sp, 80, axis=1)
        print('  Normalisation: using suite2p spont_pre (5th / 80th pct).')
    else:
        baseline_frames = min(500, Fc_cn.shape[1])
        low  = np.percentile(Fc_cn[:, :baseline_frames], 5,  axis=1)
        high = np.percentile(Fc_cn[:, :baseline_frames], 80, axis=1)
        print('  Normalisation: no spont_pre found — using first '
              f'{baseline_frames} BCI frames as baseline.')

    return low, high


def _epoch_bounds(history, n_trials):
    """Convert threshold-change trial indices to (start, end) inclusive pairs."""
    if not history:
        return [(0, n_trials - 1)]
    change_trials = sorted({h['trial'] for h in history})
    bounds = []
    for i, t in enumerate(change_trials):
        end = change_trials[i + 1] - 1 if i + 1 < len(change_trials) else n_trials - 1
        bounds.append((t, end))
    return bounds


# ─── Main entry point ────────────────────────────────────────────────────────

def run(folder, data,
        trial_start=TRIAL_START,
        n_cns=None,
        cn_smooth=CN_SMOOTH,
        cn_s2p_indices=None):
    """
    Produce the 6-panel multi-CN BCI QC figure.

    Parameters
    ----------
    folder          : str — session pophys directory (same as ddc.main argument)
    data            : dict — output of data_dict_create_module_bruker.main()
    trial_start     : int — first trial to include (0 = all; 40 = skip early artefacts)
    n_cns           : int — expected number of conditioned neurons (default 10)
    cn_smooth       : int — heatmap smoothing in trials (0 = off)
    cn_s2p_indices  : list[int] or None
        0-based suite2p ROI indices for the N selected CNs, in the same order as
        selected_rois from the threshold .mat files.  Pass this explicitly once
        you have confirmed the correct indices from the inspection cell.
        If None, the function attempts to read them from data['conditioned_neuron'].

    Returns
    -------
    list containing one matplotlib.figure.Figure
    """
    folder = folder.rstrip('/\\')
    figs   = []

    # ── 1. ops (frames per trial) ─────────────────────────────────────────
    ops = np.load(
        os.path.join(folder, 'suite2p_BCI', 'plane0', 'ops.npy'),
        allow_pickle=True
    ).tolist()
    len_files = np.asarray(ops['frames_per_file'], dtype=int)

    # ── 2. TTR (identical to single-CN script) ────────────────────────────
    rt = np.array([x[0] if len(x) > 0 else np.nan
                   for x in data['threshold_crossing_time']])
    st = np.array([x[0] if len(x) > 0 else np.nan
                   for x in data['SI_start_times']])
    ttr_all = rt - st
    rew_all  = ~np.isnan(ttr_all)

    n_trials_total = min(len(len_files), len(ttr_all))
    len_files      = len_files[:n_trials_total]
    ttr_all        = ttr_all[:n_trials_total]
    rew_all        = rew_all[:n_trials_total]

    if trial_start >= n_trials_total:
        raise ValueError(f'trial_start={trial_start} >= n_trials_total={n_trials_total}')

    # ── 3. Threshold history ──────────────────────────────────────────────
    thresh_history, selected_rois = _load_threshold_history(folder)
    n_detected = len(selected_rois) if selected_rois is not None else N_CNS
    if n_cns is None:
        n_cns = n_detected
    else:
        n_cns = min(n_cns, n_detected)
    print(f'  {n_cns} CNs (auto-detected)  |  {len(thresh_history)} threshold updates  |  '
          f'{n_trials_total} trials  |  trial_start={trial_start}')

    # ── 4. Identify suite2p ROI indices for selected CNs ──────────────────
    if cn_s2p_indices is not None:
        cn_idx = np.asarray(cn_s2p_indices, dtype=int)[:n_cns]
        print(f'  CN s2p indices (manual): {cn_idx}')
    else:
        cn_idx, reliable = _identify_cn_indices(data, n_cns)
        if cn_idx is None:
            raise RuntimeError(
                'Cannot determine suite2p ROI indices for the selected CNs.\n'
                'Run the inspection cell in the notebook, then pass '
                'cn_s2p_indices=[...] explicitly to run().'
            )
        flag = '' if reliable else '  ← verify with inspection cell'
        print(f'  CN s2p indices (auto): {cn_idx}{flag}')

    # Lock n_cns to however many indices we actually have — guards against
    # data['conditioned_neuron'] returning fewer entries than selected_rois.
    n_cns = len(cn_idx)

    # ── 5. Load suite2p F, apply neuropil correction ──────────────────────
    Fc = _load_s2p_fluorescence(folder, cn_idx)   # (n_cns, n_frames)
    n_frames = Fc.shape[1]

    # ── 6. Per-CN normalisation (5th / 80th percentile, spont or fallback) ─
    low, high = _compute_normalization(Fc, folder, cn_idx, n_cns)

    # Broadcast to per-frame arrays using threshold history for epoch-level
    # updates of high (MATLAB raises high_i mid-session via multiplier M).
    # For the analysis we use a single fixed normalisation; epoch-level
    # difficulty is captured separately via the cumulative-M panel.
    denom   = np.where(np.abs(high - low) < 1e-6, 1e-6, high - low)
    cn_norm = np.clip(
        (Fc - low[:, np.newaxis]) / denom[:, np.newaxis], 0.0, 1.0
    )   # (n_cns, n_frames)

    # Geometric mean via log-space mean
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        combo = np.exp(np.nanmean(np.log(np.clip(cn_norm, 1e-9, 1.0)), axis=0))

    # ── 7. Per-trial mean activity (for heatmap) ──────────────────────────
    frame_starts = np.concatenate([[0], np.cumsum(len_files[:-1])])
    cn_trial_mean = np.zeros((n_cns, n_trials_total))
    for t in range(n_trials_total):
        fs = int(frame_starts[t])
        fe = min(fs + int(len_files[t]), n_frames)
        cn_trial_mean[:, t] = cn_norm[:, fs:fe].mean(axis=1)

    if cn_smooth > 1:
        cn_trial_mean = uniform_filter1d(
            cn_trial_mean.astype(float), size=cn_smooth, axis=-1, mode='nearest'
        )

    # ── 8. Slice to analysis window ───────────────────────────────────────
    ttr        = ttr_all[trial_start:]
    rew        = rew_all[trial_start:]
    trial_nums = np.arange(trial_start, n_trials_total)

    # ── 9. Epoch structure ────────────────────────────────────────────────
    all_bounds = _epoch_bounds(thresh_history, n_trials_total)
    epoch_bounds_plot = [
        (max(s, trial_start), e)
        for s, e in all_bounds if e >= trial_start
    ]
    n_epochs = len(epoch_bounds_plot)
    colors   = [plt.cm.tab10(i / 10) for i in range(n_epochs)]

    hit_rate_per_epoch = []
    for s, e in epoch_bounds_plot:
        sl    = slice(s - trial_start, e - trial_start + 1)
        n_hit = int(np.sum(rew[sl]))
        n_tr  = e - s + 1
        hit_rate_per_epoch.append(n_hit / n_tr if n_tr > 0 else np.nan)

    # ── 10. Peri-reward ensemble signal ───────────────────────────────────
    frame_rate = float(ops.get('fs', 60.0))   # fall back to 60 Hz if missing
    pre_fr  = int(2.0 * frame_rate)
    post_fr = int(4.0 * frame_rate)
    win_len = pre_fr + post_fr
    peri_t  = np.arange(-pre_fr, post_fr) / frame_rate

    peri_by_epoch = []
    for s, e in epoch_bounds_plot:
        traces = []
        for t_abs in range(s, min(e + 1, n_trials_total)):
            t_rel = t_abs - trial_start
            if t_rel < 0 or t_rel >= len(ttr) or np.isnan(ttr[t_rel]):
                continue
            reward_fr = int(frame_starts[t_abs]) + int(round(ttr[t_rel] * frame_rate))
            r0, r1    = reward_fr - pre_fr, reward_fr + post_fr
            if r0 < 0 or r1 > n_frames:
                continue
            traces.append(combo[r0:r1])
        if traces:
            arr = np.vstack(traces)
            peri_by_epoch.append({'mean': arr.mean(0),
                                   'sem':  arr.std(0) / np.sqrt(len(arr)),
                                   'n':    len(arr)})
        else:
            peri_by_epoch.append({'mean': np.full(win_len, np.nan),
                                   'sem':  np.full(win_len, np.nan),
                                   'n': 0})

    # ── 11. Cumulative difficulty multiplier M ────────────────────────────
    if thresh_history:
        low0  = thresh_history[0]['threshold'][0, :n_cns]
        high0 = thresh_history[0]['threshold'][1, :n_cns]
        dM    = np.where(np.abs(high0 - low0) < 1e-6, 1e-6, high0 - low0)
        thresh_trials_arr = np.array([h['trial'] for h in thresh_history])
        cum_M = np.array([
            float(np.mean((h['threshold'][1, :n_cns] - low0) / dM))
            for h in thresh_history
        ])
    else:
        thresh_trials_arr = np.array([])
        cum_M             = np.array([])

    ep_centers  = np.array([(s + e) / 2.0 for s, e in epoch_bounds_plot])
    ep_mean_ttr = np.array([
        float(np.nanmean(ttr[max(0, s - trial_start): e - trial_start + 1]))
        for s, e in epoch_bounds_plot
    ])

    # ── 12. CN correlation matrix (analysis window) ───────────────────────
    start_fr    = int(frame_starts[trial_start])
    corr_mat    = np.corrcoef(cn_norm[:, start_fr:])
    sort_idx    = np.argsort(cn_trial_mean[:, trial_start:].mean(axis=1))[::-1]
    corr_sorted = corr_mat[np.ix_(sort_idx, sort_idx)]

    # Tick labels: prefer "s2p_<roi_idx>" for clarity
    cn_labels = [f's2p {cn_idx[sort_idx[i]]}' for i in range(n_cns)]

    # ─── Figure ──────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8})
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    def _clean(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Panel 1 — TTR Trajectory
    for ep_idx, (s, e) in enumerate(epoch_bounds_plot):
        i0, i1 = s - trial_start, e - trial_start + 1
        mask = ~np.isnan(ttr[i0:i1])
        ax1.scatter(trial_nums[i0:i1][mask], ttr[i0:i1][mask],
                    s=12, color=colors[ep_idx], alpha=0.75, zorder=2,
                    label=f'Ep {ep_idx+1}')
    for t_th in thresh_trials_arr[thresh_trials_arr >= trial_start]:
        ax1.axvline(t_th, color='gray', lw=0.7, ls='--', alpha=0.5, zorder=1)
    suffix = f'  [trial ≥ {trial_start}]' if trial_start > 0 else ''
    ax1.set_title(f'TTR Trajectory{suffix}')
    ax1.set_xlabel('Trial');  ax1.set_ylabel('TTR (s)')
    ax1.set_xlim(trial_start - 1, n_trials_total)
    if n_epochs <= 7:
        ax1.legend(fontsize=6, loc='upper right', markerscale=1.2, framealpha=0.7)
    _clean(ax1)

    # Panel 2 — Per-CN Activity Heatmap
    hm = cn_trial_mean[sort_idx, trial_start:]
    im2 = ax2.imshow(hm, aspect='auto', cmap='hot',
                     vmin=0, vmax=np.nanpercentile(hm, 98), interpolation='nearest')
    for t_th in thresh_trials_arr[thresh_trials_arr >= trial_start]:
        ax2.axvline(t_th - trial_start, color='cyan', lw=0.8, alpha=0.7)
    ax2.set_xlabel('Trial (from trial_start)')
    ax2.set_ylabel('CN  (sorted by activity)')
    ax2.set_yticks(range(n_cns));  ax2.set_yticklabels(cn_labels, fontsize=6)
    ax2.set_title('Per-CN Normalized Activity')
    plt.colorbar(im2, ax=ax2, label='Mean $n_i$', fraction=0.046, pad=0.04)

    # Panel 3 — Hit Rate per Epoch
    ep_x = np.arange(n_epochs)
    ax3.bar(ep_x, [r * 100 for r in hit_rate_per_epoch],
            color=colors[:n_epochs], edgecolor='k', linewidth=0.5)
    ax3.axhline(100, color='gray', ls='--', lw=0.8, alpha=0.6)
    for x, hr, (s, e) in zip(ep_x, hit_rate_per_epoch, epoch_bounds_plot):
        if not np.isnan(hr):
            ax3.text(x, hr * 100 + 1.5, f'{hr*100:.0f}%\n(n={e-s+1})',
                     ha='center', va='bottom', fontsize=6)
    ax3.set_xticks(ep_x)
    ax3.set_xticklabels([f'Ep {i+1}' for i in range(n_epochs)], fontsize=7)
    ax3.set_xlabel('Epoch');  ax3.set_ylabel('Hit Rate (%)')
    ax3.set_ylim(0, 120);  ax3.set_title('Hit Rate by Epoch')
    _clean(ax3)

    # Panel 4 — Peri-Reward Ensemble Signal
    ax4.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5, label='Reward')
    for ep_idx, d in enumerate(peri_by_epoch):
        if d['n'] == 0:
            continue
        mu, se = d['mean'], d['sem']
        ax4.plot(peri_t, mu, color=colors[ep_idx], lw=1.5,
                 label=f'Ep {ep_idx+1}  (n={d["n"]})')
        ax4.fill_between(peri_t, mu - se, mu + se, color=colors[ep_idx], alpha=0.18)
    ax4.set_xlabel('Time relative to reward (s)')
    ax4.set_ylabel('Geometric mean (combo)')
    ax4.set_title('Peri-Reward Ensemble Signal')
    ax4.legend(fontsize=6, loc='upper left', framealpha=0.7)
    _clean(ax4)

    # Panel 5 — Difficulty History + Mean TTR
    if len(cum_M) > 0:
        vis = thresh_trials_arr >= trial_start
        ax5.step(thresh_trials_arr[vis], cum_M[vis], where='post', color='k', lw=1.5)
        ax5.scatter(thresh_trials_arr[vis], cum_M[vis], color='k', s=22, zorder=3)
        ax5.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax5.set_xlabel('Trial of threshold update')
        ax5.set_ylabel('Cumulative multiplier M')
        ax5.set_title('Difficulty History  +  Mean TTR')
        _clean(ax5)
        ax5b = ax5.twinx()
        ax5b.step(thresh_trials_arr[vis], ep_mean_ttr, where='post',
                  color='royalblue', lw=1.5, alpha=0.85)
        ax5b.set_ylabel('Mean TTR / epoch (s)', color='royalblue')
        ax5b.tick_params(axis='y', colors='royalblue')
        ax5b.spines['right'].set_color('royalblue')
        ax5b.spines['top'].set_visible(False)
    else:
        ax5.text(0.5, 0.5, 'No threshold history', ha='center', va='center',
                 transform=ax5.transAxes, color='gray')
        ax5.set_title('Difficulty History  +  Mean TTR');  _clean(ax5)

    # Panel 6 — CN Co-Activation Correlation
    im6 = ax6.imshow(corr_sorted, cmap='RdBu_r', vmin=-1, vmax=1,
                     interpolation='nearest', aspect='equal')
    ax6.set_xticks(range(n_cns));  ax6.set_xticklabels(cn_labels, fontsize=6, rotation=45, ha='right')
    ax6.set_yticks(range(n_cns));  ax6.set_yticklabels(cn_labels, fontsize=6)
    plt.colorbar(im6, ax=ax6, label='Pearson r', fraction=0.046, pad=0.04)
    ax6.set_title('CN Co-Activation Correlation')
    for i in range(n_cns):
        for j in range(i):
            r = corr_sorted[i, j]
            ax6.text(j, i, f'{r:.2f}', ha='center', va='center', fontsize=4,
                     color='white' if abs(r) > 0.6 else 'black')

    fig.suptitle(
        f"Multi-CN BCI QC  |  {data.get('mouse','?')}  {data.get('session','?')}  |  "
        f"{n_cns} CNs  |  trial_start={trial_start}",
        fontsize=11, fontweight='bold'
    )
    figs.append(fig)
    return figs


if __name__ == '__main__':
    import sys
    print('Import this module and call run(folder, data).')
    sys.exit(1)
