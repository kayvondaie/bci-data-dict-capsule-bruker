"""CN activity from trial-start to threshold-crossing — group comparison.

Motivation
----------
Behaviorally, Ctrl and LC-KO mice differ sharply in *actual − expected hit rate*
(Ctrl beat the gain-cut null; LC-KO collapse toward it). But the CN metrics in
bci_qc_summary_multigroup.ipynb use the WHOLE-TRIAL mean of F, which (a) includes
the 2 s pre-cue baseline and post-reward frames, and (b) is gain-blind. This
script restricts the CN average to the behaviorally relevant window — trial start
(go cue) to the threshold crossing — and asks whether THAT separates the groups.

Window (per trial i, on data['F'][:, cn, i], axis-0 = frames):
    two         = int(round(2.0 / dt_si))                  # go cue = 2 s in
    rt_i        = threshold_crossing_time[i] - SI_start_times[i]   # within-trial, sec
    cross_frame = two + int(round(rt_i / dt_si))
    window      = F[two:cross_frame, cn, i]                # start -> crossing (hits only)

Data source: prebuilt per-session `data` dicts saved as h5 in /scratch (fast;
no ddc.main rerun). Each h5 is one (subject, date, stem) session.

Run:  python cn_to_crossing_analysis.py
Out:  /results/figures/cn_to_crossing.png  (+ printed stats)
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

# ── Groups ────────────────────────────────────────────────────────────────────
GROUPS = {
    "Ctrl":  ["820614", "824946", "820615", "855519"],
    "LC-KO": ["857094", "857095", "849680"],
}
GROUP_COLOR = {"Ctrl": "steelblue", "LC-KO": "tomato"}
SUBJ2GROUP = {s: g for g, subs in GROUPS.items() for s in subs}

HIT_WINDOW = 10          # trials, moving-avg window for actual hit rate
SCRATCH    = "/scratch"


# ── h5 → data dict ─────────────────────────────────────────────────────────────
def _unpickle(v):
    """h5 void/bytes field -> python object (lists of per-trial arrays)."""
    return pickle.loads(np.asarray(v).tobytes())


def load_session(path):
    """Load only what we need from a session h5 (avoids loading full F)."""
    with h5py.File(path, "r") as f:
        dt   = float(np.array(f["dt_si"]))
        cn   = int(np.array(f["conditioned_neuron"])[0][0])
        thr  = np.array(f["BCI_thresholds"])                 # (2, ntrials)
        F_cn = f["F"][:, cn, :]                              # (frames, ntrials) — CN only
        tc   = _unpickle(f["threshold_crossing_time"][()])
        si   = _unpickle(f["SI_start_times"][()])
    return dict(dt=dt, cn=cn, thr=thr, F_cn=F_cn, tc=tc, si=si)


def _first(x):
    a = np.asarray(x).ravel()
    return float(a[0]) if a.size > 0 else np.nan


# ── Per-session metrics ─────────────────────────────────────────────────────────
def analyze_session(d):
    dt   = d["dt"]
    thr  = d["thr"]
    F_cn = d["F_cn"]                       # (frames, ntrials)
    n    = F_cn.shape[1]
    two  = int(round(2.0 / dt))

    # within-trial crossing time (sec); NaN on miss
    rt = np.array([_first(d["tc"][i]) - _first(d["si"][i]) for i in range(n)])
    hit = (~np.isnan(rt)).astype(float)

    # ── behavioral null: expected hit rate per epoch (gain-cut model) ──────────
    k_upper  = np.diff(thr[1, :])
    switches = np.concatenate(([0], np.where((k_upper != 0) & (~np.isnan(k_upper)))[0]))
    upr = thr[1, switches + 1]
    lwr = float(thr[0, switches[0] + 1])
    first_end = int(switches[1]) if len(switches) > 1 else n
    rt_e1 = rt[0:first_end]

    expected_rate = np.full(n, np.nan)
    alphas = np.full(len(switches), np.nan)
    for k in range(len(switches)):
        s = int(switches[k]); e = int(switches[k + 1]) if k + 1 < len(switches) else n
        alpha = (upr[0] - lwr) / (upr[k] - lwr) if (upr[k] - lwr) != 0 else 1.0
        alphas[k] = alpha
        expected_rate[s:e] = float(np.nanmean(rt_e1 / alpha < 10))

    actual_hr = np.convolve(hit, np.ones(HIT_WINDOW) / HIT_WINDOW, mode="same")
    hit_diff  = actual_hr - expected_rate

    # ── CN activity: trial start -> crossing (hits only) ──────────────────────
    cn_to_cross = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(rt[i]):
            continue
        cf = two + int(round(rt[i] / dt))
        cf = min(cf, F_cn.shape[0])
        if cf > two:
            cn_to_cross[i] = np.nanmean(F_cn[two:cf, i])

    # ── old metric for contrast: whole-trial mean F ───────────────────────────
    cn_whole = np.nanmean(F_cn, axis=0)

    # ── per-epoch summaries ───────────────────────────────────────────────────
    ep_edges = list(switches) + [n]
    epochs = []
    for k in range(len(ep_edges) - 1):
        s, e = int(ep_edges[k]), int(ep_edges[k + 1])
        epochs.append(dict(
            idx=k, alpha=alphas[k],
            actual_hr=float(np.nanmean(hit[s:e])),
            expected_hr=float(np.nanmean(expected_rate[s:e])),
            cn_to_cross=float(np.nanmean(cn_to_cross[s:e])),
            cn_whole=float(np.nanmean(cn_whole[s:e])),
            n_trials=e - s,
        ))

    return dict(
        n=n, switches=switches, rt=rt, hit=hit,
        expected_rate=expected_rate, hit_diff=hit_diff,
        cn_to_cross=cn_to_cross, cn_whole=cn_whole, epochs=epochs,
    )


# ── Discover sessions (dedupe to best stem per subject/date) ─────────────────────
def discover():
    best = {}   # (subject,date) -> (ntrials, path, subject)
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
            best[key] = (nt, p, sub)
    return best


def epoch_ratios(epochs, key="cn_to_cross"):
    """Per-epoch activity normalized to the FIRST epoch (the same baseline the
    expected-hit-rate null uses). Returns list of dicts per non-first epoch with
    the actual activity ratio and the ratio REQUIRED to fully offset the gain cut
    (1/alpha). ratio≈1 → no change from epoch 1 (matches expected); ratio≈1/alpha
    → full compensation (hit rate maintained)."""
    if len(epochs) < 2:
        return []
    base = epochs[0][key]
    if not np.isfinite(base) or abs(base) < 1e-9:
        return []
    out = []
    for e in epochs[1:]:
        if not np.isfinite(e[key]):
            continue
        out.append(dict(
            ratio=e[key] / base,                 # actual activity relative to epoch 1
            required=1.0 / e["alpha"] if e["alpha"] not in (0, None) else np.nan,
            beat=e["actual_hr"] - e["expected_hr"],
            n_trials=e["n_trials"],
        ))
    return out


def main():
    sessions = discover()
    print(f"Discovered {len(sessions)} sessions (deduped to best stem).")

    recs = {g: [] for g in GROUPS}
    for (sub, date), (nt, path, _) in sorted(sessions.items()):
        g = SUBJ2GROUP[sub]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = analyze_session(load_session(path))
            r["label"] = f"{sub} {date}"; r["subject"] = sub
            recs[g].append(r)
        except Exception as e:
            print(f"  FAIL {sub} {date}: {e}")

    for g in GROUPS:
        print(f"  {g}: {len(recs[g])} sessions")

    # ── session-level scalars ─────────────────────────────────────────────────
    # Per session: mean activity ratio over all later epochs (trial-weighted),
    # for both the start->cross window (new) and whole-trial (old) measures.
    scal = {g: dict(hit_diff=[], ratio_cross=[], ratio_whole=[]) for g in GROUPS}
    pooled = {g: dict(ratio=[], required=[], beat=[]) for g in GROUPS}  # per (session,epoch)
    for g in GROUPS:
        for r in recs[g]:
            scal[g]["hit_diff"].append(np.nanmean(r["hit_diff"]))
            rc = epoch_ratios(r["epochs"], "cn_to_cross")
            rw = epoch_ratios(r["epochs"], "cn_whole")
            if rc:
                w = np.array([e["n_trials"] for e in rc], float)
                scal[g]["ratio_cross"].append(np.average([e["ratio"] for e in rc], weights=w))
                for e in rc:
                    pooled[g]["ratio"].append(e["ratio"])
                    pooled[g]["required"].append(e["required"])
                    pooled[g]["beat"].append(e["beat"])
            else:
                scal[g]["ratio_cross"].append(np.nan)
            if rw:
                w = np.array([e["n_trials"] for e in rw], float)
                scal[g]["ratio_whole"].append(np.average([e["ratio"] for e in rw], weights=w))
            else:
                scal[g]["ratio_whole"].append(np.nan)

    print("\n── Session-level scalars (mean ± sem) ──")
    print("  (activity ratio = later-epoch CN / first-epoch CN; null=1.0, full comp=1/alpha)")

    def report(name, key):
        a = np.array(scal["Ctrl"][key]); b = np.array(scal["LC-KO"][key])
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan
        print(f"  {name:30s} Ctrl={np.mean(a):+.4f}±{stats.sem(a):.4f} (n={len(a)})  "
              f"LC-KO={np.mean(b):+.4f}±{stats.sem(b):.4f} (n={len(b)})  p={p:.4g}")

    report("hit_diff (behavior)",          "hit_diff")
    report("activity ratio start→cross",   "ratio_cross")
    report("activity ratio whole-trial",   "ratio_whole")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("CN activity relative to first epoch (neural analog of expected hit rate)",
                 fontsize=13, fontweight="bold")

    def beeswarm(ax, key, title, ylabel, hline):
        for xi, g in enumerate(GROUPS):
            v = np.array(scal[g][key]); v = v[~np.isnan(v)]
            jit = (np.random.rand(len(v)) - 0.5) * 0.25
            ax.scatter(xi + jit, v, s=24, color=GROUP_COLOR[g], alpha=0.6, linewidths=0)
            ax.plot([xi - 0.18, xi + 0.18], [np.mean(v)] * 2, color="k", lw=2.2)
        a = np.array(scal["Ctrl"][key]); b = np.array(scal["LC-KO"][key])
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan
        ax.set_xticks([0, 1]); ax.set_xticklabels(list(GROUPS))
        if hline is not None:
            ax.axhline(hline, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"{title}\nMann-Whitney p={p:.3g}", fontsize=10)
        ax.set_ylabel(ylabel); ax.spines[["top", "right"]].set_visible(False)

    beeswarm(axes[0, 0], "hit_diff",    "Behavior: actual − expected hit rate", "hit_diff", 0.0)
    beeswarm(axes[0, 1], "ratio_cross", "NEW: CN activity ratio (later/epoch1)\nstart→cross window",
             "activity ratio", 1.0)

    # Panel C: actual activity ratio vs required (1/alpha), per (session,epoch)
    ax = axes[1, 0]
    allr = []
    for g in GROUPS:
        x = np.array(pooled[g]["required"]); y = np.array(pooled[g]["ratio"])
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=18, color=GROUP_COLOR[g], alpha=0.45, linewidths=0, label=g)
        allr.append((x[m], y[m]))
    lim = [0.8, max(2.0, np.nanmax([np.nanmax(x) if len(x) else 1 for x, _ in allr]) * 1.05)]
    ax.plot(lim, lim, "k-", lw=1, alpha=0.6, label="full comp (y=1/α)")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6, label="no change (null)")
    ax.set_xlim(lim); ax.set_xlabel("required ratio 1/α (gain cut)")
    ax.set_ylabel("actual activity ratio (later/epoch1)")
    ax.set_title("Does CN activity track the gain cut?", fontsize=10)
    ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)

    # Panel D: per-session neural ratio vs behavioral hit_diff
    ax = axes[1, 1]
    for g in GROUPS:
        x = np.array(scal[g]["ratio_cross"]); y = np.array(scal[g]["hit_diff"])
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=28, color=GROUP_COLOR[g], alpha=0.65, linewidths=0, label=g)
    ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(1.0, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("CN activity ratio (later/epoch1)"); ax.set_ylabel("hit_diff (behavior)")
    ax.set_title("Neural ratio vs behavior", fontsize=10)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    out = "/results/figures/cn_ratio_to_epoch1.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
