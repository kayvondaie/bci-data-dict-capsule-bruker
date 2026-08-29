"""Explore stronger estimators of the adapted-CN driving difference.

Same construct as cn_counterfactual_replay (adapted last-epoch CN evaluated on
the animal's own epoch-1 thresholds), two refinements:
  (1) cf_drive_index: un-clipped mean (CN-lower)_+/(upper1-lower) over a fixed
      post-cue window — non-saturating, so it doesn't compress strong adaptation
      the way cf_TTR (which floors) does. Higher = drives harder.
  (2) longitudinal: per-animal slope of cf_drive_index across training days.

Engaged-trimmed (CRIT=0.5) to remove end-of-session disengagement.
Hierarchical bootstrap (animals->sessions) for the group contrast.

NOTE: this is exploratory on a fixed 4v3 cohort — treat agreement across
estimators (all favoring Ctrl) as the signal, not any single p.
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import cn_counterfactual_replay as M
import cn_counterfactual_engaged as E

B = 20000
SEED = 0
CRIT = 0.5
W = 5.0


def cf_drive_index(a, crit=CRIT, w=W):
    n = E.quit_point(a["rt"], crit)
    if n < 20 or a.get("bnd") is None:
        return None
    thr = a["thr"]; roit_i = a["roit_i"]; roicn_i = a["roicn_i"]; bnd = a["bnd"]
    ku = np.diff(thr[1, :n])
    sw = np.concatenate(([0], np.where((ku != 0) & (~np.isnan(ku)))[0]))
    if len(sw) < 2:
        return None
    upper1 = float(thr[1, sw[0] + 1]); lower = float(thr[0, sw[0] + 1])
    if upper1 <= lower:
        return None
    last_s = int(sw[-1])
    vals = []
    for i in range(last_s, n):
        seg = M.seg_frames(roit_i, roicn_i, bnd, i, w)   # frame-accurate [go cue, go cue+w]
        if seg is None or len(seg) < 3:
            continue
        vals.append(np.mean(np.maximum(seg - lower, 0.0) / (upper1 - lower)))
    return float(np.median(vals)) if vals else None


def boot_group(animals, rng):
    keys = list(animals); out = []
    for ci in rng.integers(0, len(keys), len(keys)):
        s = animals[keys[ci]]
        out.append(s[rng.integers(0, len(s), len(s))])
    return np.concatenate(out)


def hboot(data, higher_better):
    oc = np.mean(np.concatenate(list(data["Ctrl"].values())))
    ol = np.mean(np.concatenate(list(data["LC-KO"].values())))
    rng = np.random.default_rng(SEED)
    bd = np.empty(B)
    for i in range(B):
        bd[i] = np.mean(boot_group(data["Ctrl"], rng)) - np.mean(boot_group(data["LC-KO"], rng))
    lo, hi = np.percentile(bd, [2.5, 97.5])
    p = min(2 * min(np.mean(bd >= 0), np.mean(bd <= 0)), 1.0)
    favor = np.mean(bd > 0) if higher_better else np.mean(bd < 0)
    return oc, ol, oc - ol, lo, hi, p, favor


def main():
    sess = M.discover()
    print(f"Loading {len(sess)} sessions ...")
    # collect per (animal,date) so we can do longitudinal too
    rec = defaultdict(list)   # animal -> [(date, drive_index)]
    grp = {}
    for (sub, date), (nt, path) in sorted(sess.items()):
        try:
            a = E.load_arrays(path)
            di = cf_drive_index(a)
        except Exception:
            di = None
        if di is None or not np.isfinite(di):
            continue
        rec[sub].append((date, di)); grp[sub] = M.SUBJ2GROUP[sub]

    # ── (1) group contrast on drive index ─────────────────────────────────────
    data = {g: {} for g in M.GROUPS}
    for sub, lst in rec.items():
        data[grp[sub]][sub] = np.array([v for _, v in lst])
    oc, ol, d, lo, hi, p, favor = hboot(data, higher_better=True)
    print(f"\n(1) Counterfactual DRIVE INDEX (non-saturating; higher=drives harder), CRIT={CRIT}, W={W}s")
    print(f"    Ctrl={oc:.3f}  LC-KO={ol:.3f}  diff={d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  "
          f"p={p:.4f}  P(Ctrl harder)={favor:.3f}")

    # ── (2) longitudinal: per-animal slope across training days ───────────────
    print("\n(2) Longitudinal — drive index across training days (per animal)")
    slopes = {g: {} for g in M.GROUPS}
    for sub in sorted(rec, key=lambda s: (grp[s], s)):
        lst = sorted(rec[sub])           # by date
        x = np.arange(len(lst), dtype=float)
        y = np.array([v for _, v in lst])
        if len(x) >= 3 and np.std(x) > 0:
            sl = stats.linregress(x, y).slope
            slopes[grp[sub]][sub] = sl
        else:
            sl = np.nan
        print(f"    {sub} [{grp[sub]:5s}]  n={len(lst):2d}  drive idx {y[0]:.2f}->{y[-1]:.2f}  slope/session={sl:+.4f}")

    cs = np.array(list(slopes["Ctrl"].values())); ls = np.array(list(slopes["LC-KO"].values()))
    cs = cs[np.isfinite(cs)]; ls = ls[np.isfinite(ls)]
    print(f"\n    Ctrl slopes:  {[round(float(x),4) for x in cs]}")
    print(f"    LC-KO slopes: {[round(float(x),4) for x in ls]}")
    if len(cs) and len(ls):
        try:
            _, pp = stats.mannwhitneyu(cs, ls, alternative="two-sided")
        except Exception:
            pp = np.nan
        print(f"    per-animal slope: Ctrl mean={np.mean(cs):+.4f}  LC-KO mean={np.mean(ls):+.4f}  MWU p={pp:.3f}")


if __name__ == "__main__":
    main()
