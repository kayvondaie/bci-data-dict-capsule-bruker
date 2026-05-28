"""Batch-process all attached sessions in this workstation.

Shift+Enter through cells. CELL 2 discovers every (raw, processed)
pair currently mounted at /data/; CELL 3 loops through them and writes
per-session outputs to /results/<subject>_<date>_<stem>/.

When you stop the workstation, choose "save results" to capture
/results/ as a CO data asset with all sessions inside.

To process only a subset, edit `TARGETS` in CELL 2 to a hand-picked list.
"""

# %% CELL 1 — Imports + helper for processing one session
import os, json, subprocess, re, sys, logging, traceback, shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, "/code")
import extract_scanimage_metadata
import data_dict_create_module_bruker
import bonsai_npy_threshold_calculator
ddc = data_dict_create_module_bruker

%matplotlib inline
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt


def find_pair_in_data(subject: str, date: str):
    """Return (raw_path, proc_path, proc_root) for the given subject+date in /data/.
    Raises if not found or ambiguous."""
    asset_prefix = f"single-plane-ophys_{subject}_{date}_"
    attached = sorted(Path("/data").iterdir())
    raws = [p for p in attached if p.name.startswith(asset_prefix) and "_processed_" not in p.name]
    procs = [p for p in attached if p not in raws and subject in p.name and date in p.name]

    if not raws:
        raise RuntimeError(f"No raw asset for {subject}/{date}. Attached: {[p.name for p in attached]}")
    if len(raws) > 1:
        raise RuntimeError(f"Multiple raw assets for {subject}/{date}: {[r.name for r in raws]}")
    raw = raws[0]

    if not procs:
        raise RuntimeError(f"No processed asset for {subject}/{date}")

    def _ts(p):
        m = re.search(r"_processed_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", p.name)
        return m.group(1) if m else ""
    procs_sorted = sorted(procs, key=_ts, reverse=True)
    proc = procs_sorted[0]

    if (proc / "extraction").is_dir():
        proc_root = proc
    else:
        candidates = [c for c in proc.iterdir() if c.is_dir() and (c / "extraction").is_dir()]
        if not candidates:
            raise RuntimeError(f"No extraction/ inside {proc.name}")
        proc_root = candidates[0]
    return raw, proc, proc_root


def process_session(subject: str, date: str, target_stem: str):
    """Build workspace + run ddc + run bonsai for one session.
    Saves to /results/<subject>_<date>_<stem>/. Returns the path to outputs."""
    session_tag = f"{subject}_{date}_{target_stem}"
    workspace = Path(f"/scratch/{session_tag}")
    pophys = workspace / "pophys"
    results = Path("/results") / session_tag
    figures_dir = results / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    raw, proc, proc_root = find_pair_in_data(subject, date)
    extraction = proc_root / "extraction"
    mc = proc_root / "motion_correction"
    print(f"  raw:  {raw.name}")
    print(f"  proc: {proc.name}")

    # Resolve target epoch with fallback to base stem (e.g. bci2 -> bci)
    with open(mc / "epoch_locations.json") as f:
        epoch_locations = json.load(f)
    with open(mc / "trial_locations.json") as f:
        trial_locations = json.load(f)

    if target_stem not in epoch_locations:
        base = target_stem.rstrip("0123456789") or target_stem
        candidates = sorted(k for k in epoch_locations if base in k)
        if not candidates:
            raise RuntimeError(f"Epoch {target_stem!r} not found. Available: {list(epoch_locations)}")
        resolved = base if base in epoch_locations else candidates[0]
        print(f"  NOTE: {target_stem!r} not in epochs; using {resolved!r}")
        target_stem = resolved

    target_start, target_end = epoch_locations[target_stem]

    # Build workspace
    subprocess.run(["rm", "-rf", str(workspace)], check=False)
    pophys.mkdir(parents=True)
    (workspace / "behavior").symlink_to(raw / "behavior")

    target_tifs = sorted(
        name for name, (s, e) in trial_locations.items()
        if name.startswith(f"{target_stem}_") and s >= target_start and e <= target_end
    )
    frames_per_file = [trial_locations[n][1] - trial_locations[n][0] + 1 for n in target_tifs]
    print(f"  {len(target_tifs)} TIFFs ({sum(frames_per_file)} frames)")

    # No need to symlink TIFFs — nothing in ddc.main / bonsai reads them
    # from the workspace. We extract siHeader directly from the raw asset.
    # Sidecar files (csv, mat) ARE used by ddc.main, so symlink those.
    for f in (raw / "pophys").iterdir():
        if f.is_file() and f.suffix != ".tif" and f.name.startswith(f"{target_stem}_"):
            tgt = pophys / f.name
            if not tgt.exists():
                tgt.symlink_to(f)

    bci_dir = pophys / "suite2p_BCI" / "plane0"
    bci_dir.mkdir(parents=True)
    frame_slice = slice(target_start, target_end + 1)
    for fname in ["F", "Fneu", "spks"]:
        src_path = extraction / f"{fname}.npy"
        if src_path.exists():
            arr = np.load(src_path)
            np.save(bci_dir / f"{fname}.npy", arr[:, frame_slice])
            del arr

    for fname in ["stat.npy", "iscell.npy"]:
        src = extraction / fname
        if src.exists():
            (bci_dir / fname).symlink_to(src)

    ops_path = extraction / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).tolist()
    ops["frames_per_file"] = frames_per_file
    np.save(bci_dir / "ops.npy", ops)

    # Read siHeader directly from the raw asset — no need to symlink the TIFF.
    first_tif = raw / "pophys" / target_tifs[0]
    siHeader = extract_scanimage_metadata.extract_scanimage_metadata(str(first_tif))
    siHeader["siBase"] = {0: target_stem, 1: "", 2: "spont_pre"}
    siHeader["savefolders"] = {0: target_stem, 1: "spont", 2: "spont_post", 3: "spont_pre", 4: "spont_post"}
    np.save(bci_dir / "siHeader.npy", siHeader)

    if "spont_pre" in epoch_locations:
        s_start, s_end = epoch_locations["spont_pre"]
        spont_dir = pophys / "suite2p_spont_pre" / "plane0"
        spont_dir.mkdir(parents=True)
        s_slice = slice(s_start, s_end + 1)
        for fname in ["F", "Fneu", "spks"]:
            src_path = extraction / f"{fname}.npy"
            if src_path.exists():
                arr = np.load(src_path)
                np.save(spont_dir / f"{fname}.npy", arr[:, s_slice])
                del arr
        for fname in ["stat.npy", "iscell.npy"]:
            src = extraction / fname
            if src.exists():
                (spont_dir / fname).symlink_to(src)

    # Run ddc.main
    folder = str(pophys) + "/"
    data = ddc.main(folder)
    m = re.match(r"single-plane-ophys_(\d+)_(\d{4}-\d{2}-\d{2})_", raw.name)
    if m:
        data["mouse"] = m.group(1)
        data["session"] = m.group(2)

    # Run bonsai
    try:
        figs = bonsai_npy_threshold_calculator.run(folder, data)
    except Exception as e:
        print(f"  bonsai failed: {e}")
        figs = []

    # Copy ddc.main's HDF5 output (cross-language readable) to results.
    # ddc names it data_main_<slugified-folder>_BCI.h5; glob for it.
    import shutil
    h5_candidates = list(pophys.glob("data_main_*_BCI.h5"))
    if h5_candidates:
        shutil.copy(h5_candidates[0], results / "data_dict.h5")
    else:
        print(f"  WARNING: no data_main_*_BCI.h5 found in {pophys}")

    for i, fig in enumerate(figs):
        fig.savefig(figures_dir / f"{target_stem}_fig_{i:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    with open(results / "run_log.txt", "w") as f:
        f.write(f"subject={subject}\ndate={date}\ntarget_stem={target_stem}\n")
        f.write(f"raw={raw.name}\nproc={proc.name}\n")
        f.write(f"n_figures={len(figs)}\n")
    return results, len(figs)


# %% CELL 2 — Discover all attached (raw, processed) pairs and pick TARGETS
# Edit TARGETS below to override the auto-discovered list (e.g. to process
# just one session, or to set a non-default TARGET_STEM per session).

DEFAULT_TARGET_STEM = "bci"

attached = sorted(Path("/data").iterdir())
print(f"Attached assets in /data ({len(attached)}):")
for p in attached:
    print(f"  {p.name}")

# Auto-discover (raw, proc) pairs
raws_all = [p for p in attached if p.name.startswith("single-plane-ophys_") and "_processed_" not in p.name]
auto_targets: list[tuple[str, str, str]] = []
for raw in raws_all:
    m = re.match(r"single-plane-ophys_(\d+)_(\d{4}-\d{2}-\d{2})_", raw.name)
    if not m:
        continue
    subject, date = m.group(1), m.group(2)
    # Confirm a matching processed asset is also attached
    has_proc = any(
        p.name.startswith(raw.name + "_processed_") or
        (subject in p.name and date in p.name and "_processed_" not in p.name and p != raw)
        for p in attached
    )
    if has_proc:
        auto_targets.append((subject, date, DEFAULT_TARGET_STEM))

# === Edit here to override ===
TARGETS = auto_targets   # default: process everything attached
# Example overrides:
#   TARGETS = [("850378", "2026-05-27", "bci")]
#   TARGETS = [("850378", "2026-05-27", "bci"), ("824468", "2026-05-27", "bci2")]
# ==============================

print(f"\nWill process {len(TARGETS)} session(s):")
for sub, dt, stem in TARGETS:
    print(f"  {sub}  {dt}  stem={stem}")


# %% CELL 3 — Process all TARGETS, save to /results/<session>/
import time

completed: list[dict] = []
failed: list[dict] = []
t0 = time.time()

for subject, date, stem in TARGETS:
    print(f"\n{'='*70}")
    print(f"  {subject}  {date}  {stem}")
    print(f"{'='*70}")
    t_session = time.time()
    try:
        results_dir, n_figs = process_session(subject, date, stem)
        dt = time.time() - t_session
        completed.append({"subject": subject, "date": date, "stem": stem,
                          "n_figures": n_figs, "results": str(results_dir),
                          "elapsed_sec": round(dt, 1)})
        print(f"  -> {results_dir}  ({n_figs} figs, {dt:.0f} sec)")
    except Exception as e:
        dt = time.time() - t_session
        failed.append({"subject": subject, "date": date, "stem": stem,
                       "error": str(e), "elapsed_sec": round(dt, 1)})
        print(f"  FAILED: {e}")
        traceback.print_exc()

t_total = time.time() - t0
print(f"\n{'='*70}")
print(f"Done in {t_total:.0f} sec.")
print(f"  completed: {len(completed)}")
print(f"  failed:    {len(failed)}")
print(f"{'='*70}")
if failed:
    print("Failures:")
    for f in failed:
        print(f"  {f['subject']} {f['date']} {f['stem']}: {f['error']}")

# Save a batch summary alongside per-session outputs
with open(Path("/results") / "batch_summary.json", "w") as f:
    json.dump({"completed": completed, "failed": failed,
               "total_sec": round(t_total, 1)}, f, indent=2)


# %% CELL 4 — Quick peek at any session's figures (set INSPECT to a session)
INSPECT = TARGETS[0] if TARGETS else None
if INSPECT:
    subject, date, stem = INSPECT
    session_tag = f"{subject}_{date}_{stem}"
    fig_dir = Path("/results") / session_tag / "figures"
    if fig_dir.is_dir():
        from matplotlib.image import imread
        pngs = sorted(fig_dir.glob("*.png"))
        print(f"{session_tag}: {len(pngs)} figures")
        for png in pngs:
            img = imread(str(png))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(png.name)
            plt.show()
    else:
        print(f"No figures dir at {fig_dir}")
