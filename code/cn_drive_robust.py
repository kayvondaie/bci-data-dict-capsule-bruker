"""Robustness of the counterfactual drive-index group effect.

Checks whether (1) it survives equal-animal-weighting (mean of per-animal means,
not session-pooled), and (2) leave-one-animal-out — is it carried by one animal?
"""
import numpy as np
from collections import defaultdict
from scipy import stats
import cn_counterfactual_replay as M
import cn_counterfactual_engaged as E
from cn_explore_power import cf_drive_index

B = 20000


def boot_group(animals, rng):
    keys = list(animals); out = []
    for ci in rng.integers(0, len(keys), len(keys)):
        s = animals[keys[ci]]
        out.append(s[rng.integers(0, len(s), len(s))])
    return np.concatenate(out)


def pooled_p(data):
    rng = np.random.default_rng(0); bd = np.empty(B)
    for i in range(B):
        bd[i] = np.mean(boot_group(data["Ctrl"], rng)) - np.mean(boot_group(data["LC-KO"], rng))
    return (np.mean(np.concatenate(list(data["Ctrl"].values()))),
            np.mean(np.concatenate(list(data["LC-KO"].values()))),
            min(2 * min(np.mean(bd >= 0), np.mean(bd <= 0)), 1.0),
            np.mean(bd > 0))


def main():
    sess = M.discover()
    print(f"Loading {len(sess)} sessions ...")
    by = {g: {} for g in M.GROUPS}; per = {}
    for (sub, date), (nt, path) in sorted(sess.items()):
        try:
            di = cf_drive_index(E.load_arrays(path))
        except Exception:
            di = None
        if di is None or not np.isfinite(di):
            continue
        per.setdefault(sub, []).append(di)
    for sub, v in per.items():
        by[M.SUBJ2GROUP[sub]][sub] = np.array(v)

    print("\nPer-animal mean drive index:")
    cam, lam = [], []
    for g in M.GROUPS:
        for sub in sorted(by[g]):
            m = np.mean(by[g][sub])
            (cam if g == "Ctrl" else lam).append(m)
            print(f"  {sub} [{g:5s}] n={len(by[g][sub]):2d}  mean={m:.4f}")

    print(f"\nEqual-animal-weight (mean of animal means):")
    print(f"  Ctrl={np.mean(cam):.4f}  LC-KO={np.mean(lam):.4f}")
    try:
        _, pmw = stats.mannwhitneyu(cam, lam, alternative="two-sided")
    except Exception:
        pmw = np.nan
    print(f"  per-animal MWU p={pmw:.3f}  (n={len(cam)} vs {len(lam)})")

    oc, ol, p, fav = pooled_p(by)
    print(f"\nFull hierarchical bootstrap (session-pooled): Ctrl={oc:.4f} LC-KO={ol:.4f} p={p:.4f} P(Ctrl>)={fav:.3f}")

    print("\nLeave-one-CTRL-animal-out (hierarchical bootstrap):")
    for drop in sorted(by["Ctrl"]):
        sub_data = {"Ctrl": {k: v for k, v in by["Ctrl"].items() if k != drop}, "LC-KO": by["LC-KO"]}
        oc, ol, p, fav = pooled_p(sub_data)
        print(f"  drop {drop}: Ctrl={oc:.4f} LC-KO={ol:.4f}  p={p:.4f}  P(Ctrl>)={fav:.3f}")

    print("\nLeave-one-LCKO-animal-out:")
    for drop in sorted(by["LC-KO"]):
        sub_data = {"Ctrl": by["Ctrl"], "LC-KO": {k: v for k, v in by["LC-KO"].items() if k != drop}}
        oc, ol, p, fav = pooled_p(sub_data)
        print(f"  drop {drop}: Ctrl={oc:.4f} LC-KO={ol:.4f}  p={p:.4f}  P(Ctrl>)={fav:.3f}")


if __name__ == "__main__":
    main()
