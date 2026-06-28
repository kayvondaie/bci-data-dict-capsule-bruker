"""Trial-resolved CN-drive adaptation, group-pooled — analog of the behavioral
'actual - expected hit rate vs trial' panel.

Left panel  : hit_diff = MA(hit,10) - expected_rate            (behavior, as in
              bci_qc_summary_multigroup.ipynb Panel 1)
Right panel : drive_diff = MA(d,10) - d_epoch1_baseline, where
              d_i = mean_[go,crossing] (CN-lower)_+/(upper1-lower)  evaluated against
              the FIXED epoch-1 threshold (window runs from the go cue to that
              trial's threshold crossing; misses use the ~10 s timeout).
              Rises above 0 = CN drives the original
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

MISS_W   = 10.0  # window end for missed trials (no crossing): ~the trial timeout
MA       = 10
NGRID    = 100   # number of points on the fractional-progress grid (norm version)
CRIT     = 0.5   # disengagement criterion for quit_point: trim each session at the
                 # last trial whose 10-trial MA hit rate held >= CRIT (drops the
                 # consistent end-of-session collapse from both trace and stat)
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


def session_traces(a, end_src="cross", pad=0.0, hits_only=False):
    """end_src: 'cross' → window ends at threshold crossing (rt); 'reward' → ends
    at reward delivery (rwt = crossing + lick latency). pad (s) extends the window
    past that endpoint on hit trials (still capped at the next trial start).
    hits_only=True → drive computed on crossing trials only (misses → NaN), so
    hit COUNT can't enter the metric. quit_point/engagement and behavioral
    hit_diff always use rt regardless."""
    if a.get("bnd") is None:
        return None                           # need frames_per_file for frame-accurate windows
    roit_i = a["roit_i"]; roicn_i = a["roicn_i"]; bnd = a["bnd"]
    # drop the anomalous first trial(s)
    thr = a["thr"][:, DROP_FIRST:]
    ts  = a["ts"][DROP_FIRST:]
    rt  = a["rt"][DROP_FIRST:]
    endt = (a["rwt"] if end_src == "reward" else a["rt"])[DROP_FIRST:]
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

    # per-trial drive index vs FIXED epoch-1 threshold.
    # Frame-accurate per trial (roi_csv col0 is imaging-frame time, NOT wall clock):
    # slice trial i's frames by cumsum(frames_per_file) on the gap-filled grid,
    # re-zero within-trial imaging time, and cut at the threshold crossing
    # (end_src='cross') or reward (end_src='reward', + pad). Misses → ~10 s timeout.
    d = np.full(n, np.nan)
    for i in range(n):
        if hits_only and not np.isfinite(endt[i]):
            continue                          # miss → leave d[i] = NaN
        gi = i + DROP_FIRST                   # original trial index into bnd/frames
        if gi + 1 >= len(bnd):
            continue
        ind = np.arange(bnd[gi], min(bnd[gi + 1], len(roicn_i)))
        if len(ind) < 3:
            continue
        tw = roit_i[ind] - roit_i[ind[0]]     # within-trial imaging time (== wall, no ITI)
        lim = (endt[i] + pad) if np.isfinite(endt[i]) else MISS_W
        m = tw < lim
        if m.sum() >= 3:
            d[i] = np.mean(np.maximum(roicn_i[ind][m] - lower, 0.0) / (upper1 - lower))
    # baseline on the SMOOTHED series (epoch 1 ~11 trials ≈ MA window, so raw-vs-
    # smoothed mismatch otherwise offsets short epochs). Default window = first
    # BASE_N trials (more stable than the short epoch 1); fall back to epoch 1.
    sd = mov(d, MA)
    nb = min(BASE_N, n) if BASE_N else fe
    base = np.nanmean(sd[0:nb])
    drive_diff = sd - base
    return hit_diff, drive_diff, rt[:n]


def to_progress(trace, ngrid=NGRID):
    """Resample a per-session trace onto a common 0..1 fractional-progress grid
    (NaN-aware). Each session then spans the full axis, so pooling weights every
    session equally at every point regardless of its trial count."""
    t = np.asarray(trace, float)
    m = np.isfinite(t)
    if m.sum() < 2:
        return np.full(ngrid, np.nan)
    xo = np.linspace(0.0, 1.0, len(t))
    xg = np.linspace(0.0, 1.0, ngrid)
    return np.interp(xg, xo[m], t[m])


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


def late_third(t):
    t = np.asarray(t, float)
    return np.nanmean(t[len(t) * 2 // 3:])    # summary = adapted (late) portion


def make_figure(recs, out, norm):
    """Build the two-panel figure. norm=True → x-axis is fractional session
    progress (0-100%, each session resampled); norm=False → absolute trial #."""
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
    panels = [("hit", "Actual − expected hit rate", "hit_diff"),
              ("drv", "CN activity − baseline", "CN activity")]
    for ax, (key, title, ylab) in zip(axes, panels):
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
        for g in M.GROUPS:
            traces = recs[g][key]
            if not traces:
                continue
            if norm:
                traces = [to_progress(t) for t in traces]
            mean, sem, n = pool(traces)
            cut = 0
            for i in range(len(n)):
                if n[i] >= MIN_SESS:
                    cut = i
                else:
                    break
            x = np.arange(cut + 1)
            if norm:
                x = x / (NGRID - 1) * 100.0
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

        ax.set_xlabel("Session progress (%)" if norm else "Trial #")
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.margins(x=0.02)
        # legend in the empty upper-left zone (traces hug 0 there early), below p
        leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.86),
                        handlelength=0, handletextpad=0, frameon=False,
                        borderpad=0.1, labelspacing=0.2)
        for txt in leg.get_texts():                      # colored text, no line
            if txt.get_text().startswith("Ctrl"):
                txt.set_color(GC["Ctrl"])
            elif txt.get_text().startswith("LC-KO"):
                txt.set_color(GC["LC-KO"])
        ax.spines[["top", "right"]].set_visible(False)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(out, dpi=300)                 # no bbox_inches='tight' → exact 3.5×1.75 in
    fig.savefig(out.replace(".png", ".pdf"))  # vector copy
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"Saved → {out}  ({w}×{h} in)")


# window-endpoint modes: (filename suffix, end_src, post-endpoint pad in s)
MODES = [("",           "cross",  0.0),   # go cue → threshold crossing
         ("_reward",    "reward", 0.0),   # go cue → reward delivery
         ("_reward1s",  "reward", 1.0),   # go cue → reward + 1 s (GCaMP tail)
         ("_reward2s",  "reward", 2.0)]   # go cue → reward + 2 s


def main():
    sess = M.discover()
    print(f"Loading {len(sess)} sessions ...")
    recs = {sfx: {g: {"hit": [], "drv": []} for g in M.GROUPS}
            for sfx, _, _ in MODES}
    for (sub, date), (nt, path) in sorted(sess.items()):
        g = M.SUBJ2GROUP[sub]
        try:
            a = E.load_arrays(path)
        except Exception:
            continue
        for sfx, src, pad in MODES:
            try:
                tr = session_traces(a, end_src=src, pad=pad)
            except Exception:
                tr = None
            if tr is None:
                continue
            hit, drv, rt = tr
            q = E.quit_point(rt, CRIT)        # trim end-of-session disengagement
            recs[sfx][g]["hit"].append(hit[:q]); recs[sfx][g]["drv"].append(drv[:q])

    # each mode × {absolute trial #, progress-normalized}
    base = "/results/figures/cn_drive_trace"
    for sfx, _, _ in MODES:
        make_figure(recs[sfx], f"{base}{sfx}.png",          norm=False)
        make_figure(recs[sfx], f"{base}{sfx}_progress.png", norm=True)


if __name__ == "__main__":
    main()
