"""Supra-threshold CN drive vs mean CN activity — Ctrl vs LC-KO.

Hypothesis (see chat): mice can beat the expected hit rate with NO change in
MEAN CN activity, because the task integrates a *rectified, gain-scaled* drive
    speed(t) = clip( (CN(t) - lower) / (upper - lower), 0, 1 )
and (CN - lower)_+ is convex — a burstier signal at the same mean yields more
supra-threshold drive. So mean(CN) can be flat while the drive that actually
moves the port rises (Ctrl) or falls (LC-KO).

Signal: roi_csv[:, cn_csv_index+2] — the real-time readout the port saw, in the
SAME units as BCI_thresholds (verified: CN mean ~767 straddles lower ~648).
data['F']/df_closedloop are std-normalized dF/F (wrong scale for rectification).

Window: fixed [go cue, go cue + W] on ALL trials (non-circular, no hit-selection).
go cue time = trial_start[i] (verified == SI_start - SI_start[0], same clock as
roi_csv col0). W default 3 s (≈ before the median crossing of ~4 s).

Per epoch we compute mean(CN), mean((CN-lower)_+), mean(drive); then the ratio
later-epoch / first-epoch (the same baseline the expected-hit-rate null uses).
Prediction: the drive / supra-threshold ratios separate the groups; mean(CN) does not.

Run:  python cn_supra_drive_analysis.py
Out:  /results/figures/cn_supra_drive.png  (+ printed stats)
"""

import os
import re
import glob
import pickle
import warnings

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

GROUPS = {
    "Ctrl":  ["820614", "824946", "820615", "855519"],
    "LC-KO": ["857094", "857095", "849680"],
}
GROUP_COLOR = {"Ctrl": "steelblue", "LC-KO": "tomato"}
SUBJ2GROUP = {s: g for g, subs in GROUPS.items() for s in subs}

W_SEC      = 3.0     # fixed post-cue window (seconds)
HIT_WINDOW = 10
SCRATCH    = "/scratch"


def _unpickle(v):
    return pickle.loads(np.asarray(v).tobytes())


def _first(x):
    a = np.asarray(x).ravel()
    return float(a[0]) if a.size > 0 else np.nan


def load_session(path):
    with h5py.File(path, "r") as f:
        dt      = float(np.array(f["dt_si"]))
        cn_csv  = int(np.array(f["cn_csv_index"])[0])
        thr     = np.array(f["BCI_thresholds"])           # (2, ntrials)
        roi     = np.array(f["roi_csv"])                  # (nframes, ncols) continuous
        ts      = np.array(f["trial_start"], dtype=float) # go-cue times (roi clock)
        tc      = _unpickle(f["threshold_crossing_time"][()])
        si      = _unpickle(f["SI_start_times"][()])
    return dict(dt=dt, cn_csv=cn_csv, thr=thr, roi=roi, ts=ts, tc=tc, si=si)


def analyze_session(d, w_sec=W_SEC):
    thr   = d["thr"]
    roit  = d["roi"][:, 0]                      # time (s), continuous
    roicn = d["roi"][:, d["cn_csv"] + 2]        # CN readout, threshold units
    ts    = d["ts"]
    n     = thr.shape[1]
    n     = min(n, len(ts))

    rt  = np.array([_first(d["tc"][i]) - _first(d["si"][i]) for i in range(n)])
    hit = (~np.isnan(rt)).astype(float)

    # epochs from UPPER-threshold switches
    k_upper  = np.diff(thr[1, :n])
    switches = np.concatenate(([0], np.where((k_upper != 0) & (~np.isnan(k_upper)))[0]))
    upr = thr[1, switches + 1]
    lwr = float(thr[0, switches[0] + 1])
    first_end = int(switches[1]) if len(switches) > 1 else n
    rt_e1 = rt[0:first_end]

    expected_rate = np.full(n, np.nan)
    for k in range(len(switches)):
        s = int(switches[k]); e = int(switches[k + 1]) if k + 1 < len(switches) else n
        alpha = (upr[0] - lwr) / (upr[k] - lwr) if (upr[k] - lwr) != 0 else 1.0
        expected_rate[s:e] = float(np.nanmean(rt_e1 / alpha < 10))
    actual_hr = np.convolve(hit, np.ones(HIT_WINDOW) / HIT_WINDOW, mode="same")
    hit_diff  = actual_hr - expected_rate

    # per-trial drive quantities over fixed window [go, go+W], ALL trials
    mean_cn = np.full(n, np.nan)
    rect    = np.full(n, np.nan)     # mean (CN - lower)_+
    drive   = np.full(n, np.nan)     # mean clip((CN-lower)/(upper-lower),0,1)
    for i in range(n):
        lo = thr[0, i]; hi = thr[1, i]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        a = np.searchsorted(roit, ts[i])
        b = np.searchsorted(roit, ts[i] + w_sec)
        if b <= a:
            continue
        seg = roicn[a:b]
        mean_cn[i] = np.mean(seg)
        rect[i]    = np.mean(np.maximum(seg - lo, 0.0))
        drive[i]   = np.mean(np.clip((seg - lo) / (hi - lo), 0.0, 1.0))

    # per-epoch means
    ep_edges = list(switches) + [n]
    epochs = []
    for k in range(len(ep_edges) - 1):
        s, e = int(ep_edges[k]), int(ep_edges[k + 1])
        epochs.append(dict(
            n_trials=e - s,
            mean_cn=np.nanmean(mean_cn[s:e]),
            rect=np.nanmean(rect[s:e]),
            drive=np.nanmean(drive[s:e]),
            actual_hr=float(np.nanmean(hit[s:e])),
            expected_hr=float(np.nanmean(expected_rate[s:e])),
        ))

    return dict(hit_diff=hit_diff, epochs=epochs)


def later_over_first(epochs, key):
    if len(epochs) < 2:
        return np.nan
    base = epochs[0][key]
    if not np.isfinite(base) or abs(base) < 1e-9:
        return np.nan
    vals, wts = [], []
    for e in epochs[1:]:
        if np.isfinite(e[key]):
            vals.append(e[key] / base); wts.append(e["n_trials"])
    if not vals:
        return np.nan
    return float(np.average(vals, weights=wts))


def discover():
    best = {}
    for p in sorted(glob.glob(os.path.join(SCRATCH, "*", "pophys", "*_BCI.h5"))):
        m = re.search(r"(\d{6})_(\d{4}-\d{2}-\d{2})_(bci\d*)_BCI", os.path.basename(p))
        if not m:
            continue
        sub, date = m.group(1), m.group(2)
        if sub not in SUBJ2GROUP:
            continue
        try:
            with h5py.File(p, "r") as f:
                nt = f["F"].shape[2]
        except Exception:
            continue
        key = (sub, date)
        if key not in best or nt > best[key][0]:
            best[key] = (nt, p)
    return best


def main():
    sessions = discover()
    print(f"Discovered {len(sessions)} sessions.  Window = {W_SEC}s post-cue.")

    scal = {g: dict(hit_diff=[], r_cn=[], r_rect=[], r_drive=[]) for g in GROUPS}
    for (sub, date), (nt, path) in sorted(sessions.items()):
        g = SUBJ2GROUP[sub]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = analyze_session(load_session(path))
        except Exception as e:
            print(f"  FAIL {sub} {date}: {e}"); continue
        scal[g]["hit_diff"].append(np.nanmean(r["hit_diff"]))
        scal[g]["r_cn"].append(later_over_first(r["epochs"], "mean_cn"))
        scal[g]["r_rect"].append(later_over_first(r["epochs"], "rect"))
        scal[g]["r_drive"].append(later_over_first(r["epochs"], "drive"))

    for g in GROUPS:
        print(f"  {g}: {len(scal[g]['hit_diff'])} sessions")

    print("\n── later-epoch / first-epoch ratio (null=1.0)  +  behavior ──")

    def report(name, key):
        a = np.array(scal["Ctrl"][key], float); b = np.array(scal["LC-KO"][key], float)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan
        print(f"  {name:34s} Ctrl={np.mean(a):+.4f}±{stats.sem(a):.4f}  "
              f"LC-KO={np.mean(b):+.4f}±{stats.sem(b):.4f}  p={p:.4g}")
        return p

    report("hit_diff (behavior)",            "hit_diff")
    report("ratio mean(CN)        [old]",    "r_cn")
    report("ratio (CN-lower)+     [supra]",  "r_rect")
    report("ratio drive=clip()    [speed]",  "r_drive")

    # figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f"Supra-threshold CN drive vs mean CN  (fixed {W_SEC}s post-cue window)",
                 fontsize=13, fontweight="bold")
    panels = [("hit_diff", "Behavior: actual−expected HR", "hit_diff", 0.0),
              ("r_cn",    "ratio mean(CN)  [old]",          "later/epoch1", 1.0),
              ("r_rect",  "ratio (CN−lower)₊  [supra]",     "later/epoch1", 1.0),
              ("r_drive", "ratio drive  [port speed]",      "later/epoch1", 1.0)]
    np.random.seed(0)
    for ax, (key, title, ylab, h) in zip(axes, panels):
        for xi, g in enumerate(GROUPS):
            v = np.array(scal[g][key], float); v = v[np.isfinite(v)]
            jit = (np.random.rand(len(v)) - 0.5) * 0.25
            ax.scatter(xi + jit, v, s=22, color=GROUP_COLOR[g], alpha=0.6, linewidths=0)
            ax.plot([xi - 0.18, xi + 0.18], [np.mean(v)] * 2, color="k", lw=2.2)
        a = np.array(scal["Ctrl"][key], float); b = np.array(scal["LC-KO"][key], float)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan
        ax.axhline(h, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(list(GROUPS))
        ax.set_title(f"{title}\np={p:.3g}", fontsize=10)
        ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)

    out = "/results/figures/cn_supra_drive.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
