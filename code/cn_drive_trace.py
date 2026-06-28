"""Trial-resolved CN-drive adaptation, group-pooled — analog of the behavioral
'actual - expected hit rate vs trial' panel.

Left panel  : hit_diff = MA(hit,10) - expected_rate            (behavior, as in
              bci_qc_summary_multigroup.ipynb Panel 1)
Right panel : drive_diff = MA(d,10) - d_epoch1_baseline, where
              d_i = mean_[go,go+W] (CN-lower)_+/(upper1-lower)  evaluated against
              the FIXED epoch-1 threshold. Rises above 0 = CN drives the original
              bar harder than at the start (no-adaptation null = flat 0).

Both pooled across sessions by trial index (mean +/- SEM), cut where < MIN_SESS
sessions contribute. roi_csv used for CN (threshold units).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cn_counterfactual_replay as M
import cn_counterfactual_engaged as E

plt.rcParams.update({
    "font.family": "Arial", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
B_BOOT = 10000

W        = 5.0
MA       = 10
MIN_SESS = 4
DROP_FIRST = 1   # first trial is an anomalous high-drive "freebie" (~always a hit)
BASE_N   = 20    # baseline = mean over first BASE_N trials (more stable than the
                 # ~11-trial epoch 1); set to None to use the first epoch instead
GC = {"Ctrl": "steelblue", "LC-KO": "tomato"}


def mov(x, k):
    """Edge-normalized, NaN-aware moving average: divides by the number of valid
    terms actually in the window (not k), so the endpoints aren't dragged toward
    zero by the implicit zero-padding of a plain 'same' convolution."""
    x = np.asarray(x, float)
    v = (~np.isnan(x)).astype(float)
    xf = np.where(v > 0, x, 0.0)
    num = np.convolve(xf, np.ones(k), mode="same")
    den = np.convolve(v,  np.ones(k), mode="same")
    return num / np.where(den > 0, den, np.nan)


def session_traces(a):
    roit = a["roit"]; roicn = a["roicn"]
    # drop the anomalous first trial(s)
    thr = a["thr"][:, DROP_FIRST:]
    ts  = a["ts"][DROP_FIRST:]
    rt  = a["rt"][DROP_FIRST:]
    n   = a["n"] - DROP_FIRST
    ku = np.diff(thr[1, :n])
    sw = np.concatenate(([0], np.where((ku != 0) & (~np.isnan(ku)))[0]))
    if len(sw) < 2:
        return None
    upper1 = float(thr[1, sw[0] + 1]); lower = float(thr[0, sw[0] + 1])
    if upper1 <= lower:
        return None
    fe = int(sw[1])

    # behavioral hit_diff
    hit = (~np.isnan(rt)).astype(float)
    upr = thr[1, sw + 1]; lwr = lower
    exp = np.full(n, np.nan)
    for k in range(len(sw)):
        s = int(sw[k]); e = int(sw[k + 1]) if k + 1 < len(sw) else n
        al = (upr[0] - lwr) / (upr[k] - lwr) if (upr[k] - lwr) != 0 else 1.0
        exp[s:e] = float(np.nanmean(rt[0:fe] / al < 10))
    hit_diff = mov(hit, MA) - exp

    # per-trial drive index vs FIXED epoch-1 threshold
    d = np.full(n, np.nan)
    for i in range(n):
        x = np.searchsorted(roit, ts[i]); y = np.searchsorted(roit, ts[i] + W)
        if y - x >= 3:
            d[i] = np.mean(np.maximum(roicn[x:y] - lower, 0.0) / (upper1 - lower))
    # baseline on the SMOOTHED series (epoch 1 ~11 trials ≈ MA window, so raw-vs-
    # smoothed mismatch otherwise offsets short epochs). Default window = first
    # BASE_N trials (more stable than the short epoch 1); fall back to epoch 1.
    sd = mov(d, MA)
    nb = min(BASE_N, n) if BASE_N else fe
    base = np.nanmean(sd[0:nb])
    drive_diff = sd - base
    return hit_diff, drive_diff


def pool(traces):
    L = max(len(t) for t in traces)
    M_ = np.full((len(traces), L), np.nan)
    for i, t in enumerate(traces):
        M_[i, :len(t)] = t
    n = np.sum(~np.isnan(M_), axis=0)
    mean = np.nanmean(M_, axis=0)
    sem = np.nanstd(M_, axis=0) / np.sqrt(np.maximum(n, 1))
    return mean, sem, n


def _pad(traces, L):
    M_ = np.full((len(traces), L), np.nan)
    for i, t in enumerate(traces):
        k = min(len(t), L)
        M_[i, :k] = t[:k]
    return M_


def flat_pointwise(matC, matL, rng):
    """Per-trial flat (session-level) bootstrap of the group difference; returns
    2.5/97.5 percentile arrays of (Ctrl - LC-KO) at each trial."""
    L = matC.shape[1]
    diff = np.empty((B_BOOT, L))
    for i in range(B_BOOT):
        rc = matC[rng.integers(0, matC.shape[0], matC.shape[0])]
        rl = matL[rng.integers(0, matL.shape[0], matL.shape[0])]
        diff[i] = np.nanmean(rc, axis=0) - np.nanmean(rl, axis=0)
    return np.nanpercentile(diff, 2.5, axis=0), np.nanpercentile(diff, 97.5, axis=0)


def flat_overall(valsC, valsL, rng):
    """Flat bootstrap of the group difference in a per-session summary scalar."""
    valsC = np.asarray(valsC, float); valsL = np.asarray(valsL, float)
    valsC = valsC[np.isfinite(valsC)]; valsL = valsL[np.isfinite(valsL)]
    bd = np.empty(B_BOOT)
    for i in range(B_BOOT):
        bd[i] = (np.mean(valsC[rng.integers(0, len(valsC), len(valsC))]) -
                 np.mean(valsL[rng.integers(0, len(valsL), len(valsL))]))
    p = min(2 * min(np.mean(bd >= 0), np.mean(bd <= 0)), 1.0)
    return (np.mean(valsC) - np.mean(valsL),
            np.percentile(bd, 2.5), np.percentile(bd, 97.5), p)


def main():
    sess = M.discover()
    print(f"Loading {len(sess)} sessions ...")
    recs = {g: {"hit": [], "drv": []} for g in M.GROUPS}
    for (sub, date), (nt, path) in sorted(sess.items()):
        g = M.SUBJ2GROUP[sub]
        try:
            tr = session_traces(E.load_arrays(path))
        except Exception:
            tr = None
        if tr is None:
            continue
        recs[g]["hit"].append(tr[0]); recs[g]["drv"].append(tr[1])

    def late_third(t):
        t = np.asarray(t, float)
        return np.nanmean(t[len(t) * 2 // 3:])    # summary = adapted (late) portion

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
    panels = [("hit", "Actual − expected hit rate", "hit_diff"),
              ("drv", "CN activity − baseline", "CN activity")]
    for ax, (key, title, ylab) in zip(axes, panels):
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
        cuts = {}
        for g in M.GROUPS:
            traces = recs[g][key]
            if not traces:
                continue
            mean, sem, n = pool(traces)
            cut = 0
            for i in range(len(n)):
                if n[i] >= MIN_SESS:
                    cut = i
                else:
                    break
            cuts[g] = cut
            x = np.arange(cut + 1)
            ax.fill_between(x, (mean - sem)[:cut + 1], (mean + sem)[:cut + 1],
                            color=GC[g], alpha=0.15, lw=0)
            ax.plot(x, mean[:cut + 1], color=GC[g], lw=1.0,
                    label=f"{g} (n={len(traces)})")

        # ── stat: flat session-level bootstrap on the late-third summary ─────
        rng = np.random.default_rng(0)
        d, dlo, dhi, p = flat_overall([late_third(t) for t in recs["Ctrl"][key]],
                                      [late_third(t) for t in recs["LC-KO"][key]], rng)
        ax.text(0.03, 0.97, f"p = {p:.3f}", transform=ax.transAxes,
                ha="left", va="top")

        ax.set_xlabel("Trial #"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.margins(x=0.02)
        # legend in the empty upper-left zone (traces hug 0 there early), below p
        leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.88),
                        handlelength=0, handletextpad=0, frameon=False,
                        borderpad=0.1, labelspacing=0.2, fontsize=6)
        for txt in leg.get_texts():                      # colored text, no line
            if txt.get_text().startswith("Ctrl"):
                txt.set_color(GC["Ctrl"])
            elif txt.get_text().startswith("LC-KO"):
                txt.set_color(GC["LC-KO"])
        ax.spines[["top", "right"]].set_visible(False)

    out = "/results/figures/cn_drive_trace.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(out, dpi=300)                 # no bbox_inches='tight' → exact 3.5×1.75 in
    fig.savefig(out.replace(".png", ".pdf"))  # vector copy
    w, h = fig.get_size_inches()
    print(f"Saved → {out}  ({w}×{h} in)")


if __name__ == "__main__":
    main()
