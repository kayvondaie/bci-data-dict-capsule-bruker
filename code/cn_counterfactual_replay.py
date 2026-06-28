"""Counterfactual replay: last-epoch CN through epoch-1 thresholds.

Idea (chat): both groups end the session with ~equal TTR and ~equal slack, but
Ctrl were ratcheted to ~3x threshold and LC-KO to ~2x. To ask who actually
ADAPTED more — independent of where the staircase placed them — rewind the
thresholds: take the CN each animal produced in its LAST (fully-adapted) epoch
and replay it through that session's EPOCH-1 thresholds. The more-adapted animal
should drive the easy threshold harder (shorter counterfactual TTR, higher
counterfactual hit rate).

Mechanics. Port speed = clip((CN-lower)/(upper-lower),0,1)*vmax; reward when the
integral reaches distance D, i.e. when
    ∫ clip((CN-lower)/(upper-lower),0,1) dt = D/vmax = K'   (a per-rig constant).
So: estimate K' per session from its hit trials (clipped drive integral to the
real crossing), then for each last-epoch trial integrate the clipped drive under
the EPOCH-1 (lower, upper_1) and find the counterfactual crossing time (cap 10 s).

Readouts per session:
  cf_TTR     median counterfactual TTR of last-epoch CN under epoch-1 thresholds
  cf_hit     counterfactual hit rate (crossed within 10 s) under epoch-1 thresholds
  speedup    actual epoch-1 median TTR / cf_TTR   (within-session, unit-free)
The within-session `speedup` cancels per-animal fluorescence scale.

Signal: roi_csv[:, cn_csv+2] (threshold units). Window from go cue = trial_start[i].
Run:  python cn_counterfactual_replay.py
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

MAX_T   = 10.0     # response-window cap (s) for counterfactual crossing
SCRATCH = "/scratch"


def _unpickle(v):
    return pickle.loads(np.asarray(v).tobytes())


def _first(x):
    a = np.asarray(x).ravel()
    return float(a[0]) if a.size > 0 else np.nan


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


def clipped_drive(seg, lo, hi):
    return np.clip((seg - lo) / (hi - lo), 0.0, 1.0)


def analyze(path):
    with h5py.File(path, "r") as f:
        thr    = np.array(f["BCI_thresholds"])
        roi    = np.array(f["roi_csv"])
        cn_csv = int(np.array(f["cn_csv_index"])[0])
        ts     = np.array(f["trial_start"], dtype=float)
        tc     = _unpickle(f["threshold_crossing_time"][()])
        si     = _unpickle(f["SI_start_times"][()])
    roit  = roi[:, 0]
    roicn = roi[:, cn_csv + 2]
    dtr   = float(np.median(np.diff(roit)))
    n     = min(thr.shape[1], len(ts))
    rt    = np.array([_first(tc[i]) - _first(si[i]) for i in range(n)])

    ku = np.diff(thr[1, :n])
    sw = np.concatenate(([0], np.where((ku != 0) & (~np.isnan(ku)))[0]))
    if len(sw) < 2:
        return None
    upper1 = float(thr[1, sw[0] + 1])
    lower  = float(thr[0, sw[0] + 1])
    fe     = int(sw[1])                      # end of epoch 1
    last_s = int(sw[-1])                     # start of last epoch

    def seg_of(i, t_end):
        a = np.searchsorted(roit, ts[i])
        b = np.searchsorted(roit, ts[i] + t_end)
        return roicn[a:b] if b > a else None

    # ── estimate K' = clipped drive integral to crossing, over hit trials ──────
    Kp = []
    for i in range(n):
        lo, hi = thr[0, i], thr[1, i]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo or np.isnan(rt[i]):
            continue
        s = seg_of(i, rt[i])
        if s is None or len(s) < 3:
            continue
        Kp.append(np.sum(clipped_drive(s, lo, hi)) * dtr)
    if len(Kp) < 5:
        return None
    Kp = float(np.median(Kp))

    # ── actual epoch-1 TTR (reference) ────────────────────────────────────────
    ttr_e1 = np.nanmedian(rt[0:fe])

    # ── replay last-epoch CN under epoch-1 thresholds ─────────────────────────
    cf_ttr = []
    cf_hit = []
    for i in range(last_s, n):
        s = seg_of(i, MAX_T)
        if s is None or len(s) < 3:
            continue
        cum = np.cumsum(clipped_drive(s, lower, upper1)) * dtr
        idx = np.searchsorted(cum, Kp)
        if idx < len(cum):
            cf_ttr.append((idx + 1) * dtr)
            cf_hit.append(1.0)
        else:
            cf_hit.append(0.0)
    if not cf_hit:
        return None

    cf_ttr_med = float(np.median(cf_ttr)) if cf_ttr else np.nan
    return dict(
        cf_ttr=cf_ttr_med,
        cf_hit=float(np.mean(cf_hit)),
        ttr_e1=float(ttr_e1),
        speedup=(ttr_e1 / cf_ttr_med) if (cf_ttr and cf_ttr_med > 0) else np.nan,
    )


def main():
    sessions = discover()
    print(f"Discovered {len(sessions)} sessions.")
    scal = {g: dict(cf_ttr=[], cf_hit=[], speedup=[], ttr_e1=[]) for g in GROUPS}
    for (sub, date), (nt, path) in sorted(sessions.items()):
        g = SUBJ2GROUP[sub]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = analyze(path)
        except Exception as e:
            print(f"  FAIL {sub} {date}: {e}"); continue
        if r is None:
            continue
        for k in scal[g]:
            scal[g][k].append(r[k])

    for g in GROUPS:
        print(f"  {g}: {len(scal[g]['cf_ttr'])} sessions (>=2 epochs)")

    print("\n── last-epoch CN replayed through epoch-1 thresholds ──")

    def report(name, key, better):
        a = np.array(scal["Ctrl"][key], float); b = np.array(scal["LC-KO"][key], float)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan
        print(f"  {name:34s} Ctrl={np.mean(a):+.3f}±{stats.sem(a):.3f}  "
              f"LC-KO={np.mean(b):+.3f}±{stats.sem(b):.3f}  p={p:.4g}  ({better})")

    report("actual epoch-1 TTR (s)",        "ttr_e1",  "context")
    report("counterfactual TTR (s)",        "cf_ttr",  "lower = drives harder")
    report("counterfactual hit rate",       "cf_hit",  "higher = better")
    report("speedup = TTR_e1 / cf_TTR",     "speedup", "higher = adapted more")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Last-epoch CN replayed through epoch-1 thresholds (who adapted more?)",
                 fontsize=12, fontweight="bold")
    np.random.seed(0)
    for ax, (key, title, h) in zip(axes,
            [("cf_ttr", "counterfactual TTR (s)\nlower = drives harder", None),
             ("cf_hit", "counterfactual hit rate", None),
             ("speedup", "speedup TTR_e1 / cf_TTR\nhigher = adapted more", 1.0)]):
        for xi, g in enumerate(GROUPS):
            v = np.array(scal[g][key], float); v = v[np.isfinite(v)]
            jit = (np.random.rand(len(v)) - 0.5) * 0.25
            ax.scatter(xi + jit, v, s=24, color=GROUP_COLOR[g], alpha=0.6, linewidths=0)
            ax.plot([xi - 0.18, xi + 0.18], [np.mean(v)] * 2, color="k", lw=2.2)
        a = np.array(scal["Ctrl"][key], float); b = np.array(scal["LC-KO"][key], float)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan
        if h is not None:
            ax.axhline(h, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(list(GROUPS))
        ax.set_title(f"{title}\np={p:.3g}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    out = "/results/figures/cn_counterfactual_replay.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
