"""Hierarchical bootstrap on counterfactual TTR (Ctrl vs LC-KO).

Respects the session-in-animal nesting: each replicate resamples ANIMALS with
replacement (within group), then SESSIONS with replacement within each chosen
animal, then computes the group statistic on the pooled resampled sessions.
This propagates between-animal variability into the CI (per-session tests don't;
per-animal means are too noisy with 3-4 animals).

Statistic: pooled-session mean counterfactual TTR per group, and the group diff
(Ctrl - LC-KO). cf_TTR = last-epoch CN replayed through epoch-1 thresholds
(see cn_counterfactual_replay.py). Lower cf_TTR = drives the easy bar harder.
"""

import numpy as np
from collections import defaultdict
import cn_counterfactual_replay as M

B    = 20000
SEED = 0


def collect():
    sess = M.discover()
    by = {g: defaultdict(list) for g in M.GROUPS}   # group -> animal -> [cf_ttr,...]
    for (sub, date), (nt, path) in sorted(sess.items()):
        g = M.SUBJ2GROUP[sub]
        try:
            r = M.analyze(path)
        except Exception:
            r = None
        if r is None or not np.isfinite(r["cf_ttr"]):
            continue
        by[g][sub].append(r["cf_ttr"])
    return {g: {a: np.array(v) for a, v in d.items()} for g, d in by.items()}


def boot_group(animals, rng):
    """One hierarchical resample of a group -> pooled session values."""
    keys = list(animals)
    chosen = rng.integers(0, len(keys), len(keys))
    out = []
    for ci in chosen:
        s = animals[keys[ci]]
        out.append(s[rng.integers(0, len(s), len(s))])
    return np.concatenate(out)


def main():
    data = collect()
    for g in M.GROUPS:
        n_an = len(data[g]); n_se = sum(len(v) for v in data[g].values())
        print(f"  {g}: {n_an} animals, {n_se} sessions")

    obs_ctrl = np.mean(np.concatenate(list(data["Ctrl"].values())))
    obs_lcko = np.mean(np.concatenate(list(data["LC-KO"].values())))
    obs_diff = obs_ctrl - obs_lcko

    rng = np.random.default_rng(SEED)
    bc = np.empty(B); bl = np.empty(B); bd = np.empty(B)
    for i in range(B):
        c = np.mean(boot_group(data["Ctrl"], rng))
        l = np.mean(boot_group(data["LC-KO"], rng))
        bc[i] = c; bl[i] = l; bd[i] = c - l

    def ci(x):
        return np.percentile(x, [2.5, 97.5])

    # two-sided bootstrap p for diff != 0
    p = 2 * min(np.mean(bd >= 0), np.mean(bd <= 0))
    p = min(p, 1.0)

    print(f"\nCounterfactual TTR (s) — hierarchical bootstrap (B={B})")
    print(f"  Ctrl  mean = {obs_ctrl:.3f}   95% CI [{ci(bc)[0]:.3f}, {ci(bc)[1]:.3f}]")
    print(f"  LC-KO mean = {obs_lcko:.3f}   95% CI [{ci(bl)[0]:.3f}, {ci(bl)[1]:.3f}]")
    print(f"  diff (Ctrl-LCKO) = {obs_diff:+.3f}   95% CI [{ci(bd)[0]:+.3f}, {ci(bd)[1]:+.3f}]")
    print(f"  two-sided bootstrap p = {p:.4f}")
    print(f"  P(Ctrl drives harder, diff<0) = {np.mean(bd < 0):.4f}")


if __name__ == "__main__":
    main()
