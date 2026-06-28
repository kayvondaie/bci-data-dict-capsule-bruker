"""Counterfactual replay with end-of-session disengagement trimmed.

Mice tend to quit/relax near the end; since the replay uses LAST-epoch CN, that
terminal decline biases the adapted-CN estimate. Here we detect each session's
disengagement point and drop the trailing trials, then re-run the counterfactual
TTR (last-epoch CN through epoch-1 thresholds) + hierarchical bootstrap.

Disengagement rule (low researcher DOF, tested at several criteria):
  rolling hit rate (window 10); walk from the end and drop the final contiguous
  run of trials whose rolling HR < CRIT. CRIT=None means no trimming (baseline).

Reuses loaders/helpers from cn_counterfactual_replay.
"""

import numpy as np
import h5py
from collections import defaultdict
import cn_counterfactual_replay as M

B    = 20000
SEED = 0
CRITS = [None, 0.4, 0.5, 0.6]
W_HR = 10


def load_arrays(path):
    with h5py.File(path, "r") as f:
        thr    = np.array(f["BCI_thresholds"])
        roi    = np.array(f["roi_csv"])
        cn_csv = int(np.array(f["cn_csv_index"])[0])
        ts     = np.array(f["trial_start"], dtype=float)
        tc     = M._unpickle(f["threshold_crossing_time"][()])
        si     = M._unpickle(f["SI_start_times"][()])
    roit  = roi[:, 0]
    roicn = roi[:, cn_csv + 2]
    dtr   = float(np.median(np.diff(roit)))
    n     = min(thr.shape[1], len(ts))
    rt    = np.array([M._first(tc[i]) - M._first(si[i]) for i in range(n)])
    return dict(thr=thr, roit=roit, roicn=roicn, dtr=dtr, ts=ts, rt=rt, n=n)


def quit_point(rt, crit):
    """Index to truncate at (keep [0, q)). crit=None -> full length."""
    n = len(rt)
    if crit is None:
        return n
    hit = (~np.isnan(rt)).astype(float)
    rhr = np.convolve(hit, np.ones(W_HR) / W_HR, mode="same")
    eng = rhr >= crit
    i = n - 1
    while i >= 0 and not eng[i]:
        i -= 1
    return i + 1


def cf_ttr(a, crit):
    n = quit_point(a["rt"], crit)
    if n < 20:
        return None, 0
    thr = a["thr"]; rt = a["rt"][:n]
    ts = a["ts"]; roit = a["roit"]; roicn = a["roicn"]; dtr = a["dtr"]
    ku = np.diff(thr[1, :n])
    sw = np.concatenate(([0], np.where((ku != 0) & (~np.isnan(ku)))[0]))
    if len(sw) < 2:
        return None, a["n"] - n
    upper1 = float(thr[1, sw[0] + 1]); lower = float(thr[0, sw[0] + 1])
    last_s = int(sw[-1])

    def seg_of(i, t_end):
        x = np.searchsorted(roit, ts[i]); y = np.searchsorted(roit, ts[i] + t_end)
        return roicn[x:y] if y > x else None

    Kp = []
    for i in range(n):
        lo, hi = thr[0, i], thr[1, i]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo or np.isnan(rt[i]):
            continue
        s = seg_of(i, rt[i])
        if s is not None and len(s) >= 3:
            Kp.append(np.sum(M.clipped_drive(s, lo, hi)) * dtr)
    if len(Kp) < 5:
        return None, a["n"] - n
    Kp = float(np.median(Kp))

    tts = []
    for i in range(last_s, n):
        s = seg_of(i, M.MAX_T)
        if s is None or len(s) < 3:
            continue
        cum = np.cumsum(M.clipped_drive(s, lower, upper1)) * dtr
        idx = np.searchsorted(cum, Kp)
        if idx < len(cum):
            tts.append((idx + 1) * dtr)
    if not tts:
        return None, a["n"] - n
    return float(np.median(tts)), a["n"] - n


def boot_group(animals, rng):
    keys = list(animals)
    out = []
    for ci in rng.integers(0, len(keys), len(keys)):
        s = animals[keys[ci]]
        out.append(s[rng.integers(0, len(s), len(s))])
    return np.concatenate(out)


def main():
    sess = M.discover()
    print(f"Loading {len(sess)} sessions ...")
    arrs = {}
    for (sub, date), (nt, path) in sorted(sess.items()):
        try:
            arrs[(sub, date)] = load_arrays(path)
        except Exception as e:
            print(f"  FAIL load {sub} {date}: {e}")

    for crit in CRITS:
        by = {g: defaultdict(list) for g in M.GROUPS}
        trimmed = []
        for (sub, date), a in arrs.items():
            g = M.SUBJ2GROUP[sub]
            try:
                v, ntrim = cf_ttr(a, crit)
            except Exception:
                v, ntrim = None, 0
            if v is not None and np.isfinite(v):
                by[g][sub].append(v); trimmed.append(ntrim)
        data = {g: {x: np.array(y) for x, y in d.items()} for g, d in by.items()}

        oc = np.mean(np.concatenate(list(data["Ctrl"].values())))
        ol = np.mean(np.concatenate(list(data["LC-KO"].values())))
        rng = np.random.default_rng(SEED)
        bd = np.empty(B)
        for i in range(B):
            bd[i] = np.mean(boot_group(data["Ctrl"], rng)) - np.mean(boot_group(data["LC-KO"], rng))
        lo, hi = np.percentile(bd, [2.5, 97.5])
        p = min(2 * min(np.mean(bd >= 0), np.mean(bd <= 0)), 1.0)
        nse = sum(len(v) for v in data["Ctrl"].values()) + sum(len(v) for v in data["LC-KO"].values())
        tag = "no trim" if crit is None else f"CRIT={crit}"
        print(f"\n[{tag}]  median trials trimmed/session = {np.median(trimmed):.0f}  "
              f"(sessions used: {nse})")
        print(f"   Ctrl cf_TTR={oc:.3f}  LC-KO cf_TTR={ol:.3f}  "
              f"diff={oc-ol:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  p={p:.4f}  "
              f"P(Ctrl harder)={np.mean(bd<0):.3f}")


if __name__ == "__main__":
    main()
