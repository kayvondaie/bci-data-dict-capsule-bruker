"""Batch-process all attached sessions in this workstation.

Shift+Enter through cells. CELL 2 discovers every (raw, processed)
pair currently mounted at /data/; CELL 3 loops through them and writes
per-session outputs to /results/<subject>_<date>_<stem>/.

When you stop the workstation, choose "save results" to capture
/results/ as a CO data asset with all sessions inside.

To process only a subset, edit `TARGETS` in CELL 2 to a hand-picked list.
"""

# %% CELL 1 — Imports + helper for processing one session
import os, json, subprocess, re, sys, logging, traceback
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


_CHAN_COLOR_MAP = {"chan1": "green", "chan2": "red"}


def process_session(
    subject: str,
    date: str,
    target_stem: str,
    chan_override: str = None,
    color: str = "",
):
    """Build workspace + run ddc + run bonsai for one session.

    For dual-channel processed assets (top-level `chan1/`, `chan2/` subdirs
    from the dual-channel pipeline), this function recursively calls itself
    once per channel with `chan_override` set. Each recursive call narrows
    proc_root to that channel's subtree and suffixes session_tag with the
    color name so the two runs don't collide.

    Outputs:
      /results/figures/<session_tag>_fig_00.png      (ephemeral; captured if
                                                      you save results as asset)
      /scratch/learning_pngs/<session_tag>_fig_00.png (persistent across
                                                      workstation sessions on
                                                      this capsule — used to
                                                      skip already-processed
                                                      sessions on rerun)
    The ddc.main .h5 stays at /scratch/<session_tag>/pophys/data_main_*_BCI.h5
    """
    # --- Dual-channel dispatch ---
    if chan_override is None:
        raw, proc_top, _ = find_pair_in_data(subject, date)
        chan_subdirs = sorted(
            p for p in proc_top.iterdir()
            if p.is_dir() and re.match(r"^chan\d+$", p.name)
        )
        if chan_subdirs:
            print(f"  dual-channel processed asset detected: "
                  f"{[c.name for c in chan_subdirs]}; running per channel")
            all_figs = []
            for chan_dir in chan_subdirs:
                chan_color = _CHAN_COLOR_MAP.get(chan_dir.name, chan_dir.name)
                figs_chan = process_session(
                    subject, date, target_stem,
                    chan_override=chan_dir.name, color=chan_color,
                )
                all_figs.extend(figs_chan or [])
            return all_figs

    color_suffix = f"_{color}" if color else ""
    session_tag = f"{subject}_{date}_{target_stem}{color_suffix}"
    workspace = Path(f"/scratch/{session_tag}")
    pophys = workspace / "pophys"
    results_root = Path("/results")
    figures_dir = results_root / "figures"
    persistent_pngs_dir = Path("/scratch/learning_pngs")
    results_root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    persistent_pngs_dir.mkdir(exist_ok=True)

    raw, proc, proc_root = find_pair_in_data(subject, date)
    # If invoked for a specific channel, narrow proc_root into that chan's subtree.
    if chan_override is not None:
        chan_root = proc / chan_override
        if not chan_root.is_dir():
            raise RuntimeError(f"chan_override={chan_override} not found under {proc}")
        if (chan_root / "extraction").is_dir():
            proc_root = chan_root
        else:
            candidates = [c for c in chan_root.iterdir()
                          if c.is_dir() and (c / "extraction").is_dir()]
            if not candidates:
                raise RuntimeError(f"No extraction/ inside {chan_root}")
            proc_root = candidates[0]
    extraction = proc_root / "extraction"
    mc = proc_root / "motion_correction"
    print(f"  raw:  {raw.name}")
    print(f"  proc: {proc.name}"
          + (f"  (channel {chan_override} / {color})" if chan_override else ""))

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

    # Build suite2p_photostim[N]/ slices for any photostim epochs.
    # data_dict_create_module_bruker.py:178 needs these to populate
    # data['photostim'] via create_photostim_Fstim().
    session_json_fp = raw / "session.json"
    if session_json_fp.exists():
        with open(session_json_fp) as jf:
            session_meta = json.load(jf)
        photostim_stems = [
            ep["output_parameters"]["tiff_stem"]
            for ep in session_meta.get("stimulus_epochs", [])
            if "photostim" in ep.get("stimulus_name", "").lower()
        ]
        for i, ps_stem in enumerate(photostim_stems):
            if ps_stem not in epoch_locations:
                continue
            ps_start, ps_end = epoch_locations[ps_stem]
            suffix = "" if i == 0 else str(i + 1)
            ps_dir = pophys / f"suite2p_photostim{suffix}" / "plane0"
            ps_dir.mkdir(parents=True, exist_ok=True)
            ps_slice = slice(ps_start, ps_end + 1)
            for fname in ["F", "Fneu"]:
                src_path = extraction / f"{fname}.npy"
                if src_path.exists():
                    arr = np.load(src_path)
                    np.save(ps_dir / f"{fname}.npy", arr[:, ps_slice])
                    del arr
            for fname in ["stat.npy", "iscell.npy", "ops.npy"]:
                src = extraction / fname
                if src.exists() and not (ps_dir / fname).exists():
                    (ps_dir / fname).symlink_to(src)
            ps_tifs = sorted(
                n for n, (s, e) in trial_locations.items()
                if n.startswith(f"{ps_stem}_") and s >= ps_start and e <= ps_end
            )
            if ps_tifs:
                ps_first_tif = raw / "pophys" / ps_tifs[0]
                ps_siHeader = extract_scanimage_metadata.extract_scanimage_metadata(
                    str(ps_first_tif)
                )
                ps_siHeader["siBase"] = {0: target_stem, 1: "", 2: ps_stem}
                ps_siHeader["savefolders"] = {
                    0: target_stem, 1: "spont", 2: f"photostim{suffix}"
                }
                np.save(ps_dir / "siHeader.npy", ps_siHeader)
            print(
                f"  Built suite2p_photostim{suffix}/ from epoch {ps_stem!r} "
                f"({ps_end - ps_start + 1} frames)"
            )

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

    # H5 stays in /scratch/<session>/pophys/data_main_*_BCI.h5 (ddc.main
    # already wrote it there). Not copied to /results/.

    # Only the first figure is informative — save fig 00 to both targets.
    saved = 0
    if figs:
        out_results = figures_dir / f"{session_tag}_fig_00.png"
        out_persist = persistent_pngs_dir / f"{session_tag}_fig_00.png"
        figs[0].savefig(out_results, dpi=150, bbox_inches="tight")
        figs[0].savefig(out_persist, dpi=150, bbox_inches="tight")
        saved = 1
    # Close all (including any we didn't save) to free memory.
    for fig in figs:
        plt.close(fig)

    return persistent_pngs_dir / f"{session_tag}_fig_00.png", saved


# %% CELL 1.5 — Self-attach recent BCI sessions via CO API
# Auto-attaches today's (and recent) sessions to THIS workstation so CELL 2
# auto-discovery sees them. No need to visit the orchestrator capsule.
#
# Requires CODEOCEAN_TOKEN env var (set in the env editor — should already
# be there from setup). Skips silently if missing.
#
# Set HOURS_BACK or skip this cell to control how far back to look.

HOURS_BACK = 90

_token = os.environ.get("CODEOCEAN_TOKEN")
_capsule_id = os.environ.get("CO_CAPSULE_ID")
if not _token or not _capsule_id:
    print("Skipping self-attach — CODEOCEAN_TOKEN or CO_CAPSULE_ID not set.")
else:
    import time as _time
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    from codeocean import CodeOcean as _CO
    from codeocean.data_asset import DataAssetSearchParams as _SP
    from codeocean.capsule import DataAssetAttachParams as _AP

    _c = _CO(domain="https://codeocean.allenneuraldynamics.org", token=_token)

    # Find my own running computation (this workstation)
    _comps = _c.capsules.list_computations(capsule_id=_capsule_id)
    _running = [x for x in _comps if str(getattr(x, "state", "")).lower().endswith("running")]
    if not _running:
        print("Skipping self-attach — no running computation found (am I in a workstation?).")
    else:
        _my_comp = _running[0].id
        _cutoff = int((_dt.now(_tz.utc) - _td(hours=HOURS_BACK)).timestamp())

        _sr = _c.data_assets.search_data_assets(_SP(
            query="name:single-plane-ophys_8", limit=500,
        ))
        _raws = [a for a in _sr.results
                 if a.name.startswith("single-plane-ophys_")
                 and "_processed_" not in a.name
                 and a.created >= _cutoff]
        _procs = [a for a in _sr.results if "_processed_" in a.name]

        # For each recent raw, find its matching processed (most recent if multiple).
        # Collect both raw + proc assets that need attaching independently — if
        # the raw is already mounted but its processed isn't (or vice versa),
        # we still attach the missing one. Otherwise CELL 2 silently drops the
        # session because has_proc=False.
        _attached_now = {p.name for p in Path("/data").iterdir()}
        _to_attach = []
        _pairs_found = 0
        for raw in _raws:
            matching = [p for p in _procs if p.name.startswith(raw.name + "_processed_")]
            if not matching:
                continue
            matching.sort(key=lambda p: p.created, reverse=True)
            proc = matching[0]
            _pairs_found += 1
            if raw.name not in _attached_now:
                _to_attach.append(raw)
            if proc.name not in _attached_now:
                _to_attach.append(proc)

        print(f"Found {_pairs_found} recent pair(s); {len(_to_attach)} asset(s) need attaching:")
        for a in _to_attach:
            print(f"  {a.name}")

        if _to_attach:
            _params = [_AP(id=a.id, mount=a.name) for a in _to_attach]
            try:
                _c.computations.attach_data_assets(
                    computation_id=_my_comp, attach_params=_params,
                )
                print(f"\nLive-attached {len(_params)} assets. Waiting for /data/ to update...")
                # Quick poll until everything shows up
                expected = {a.name for a in _to_attach}
                for delay in [2, 4, 8, 15]:
                    _time.sleep(delay)
                    now = {p.name for p in Path("/data").iterdir()}
                    missing = expected - now
                    if not missing:
                        print(f"  all mounted after {delay}s")
                        break
                    else:
                        print(f"  after {delay}s: still missing {len(missing)}")
                else:
                    print(f"  WARNING: {len(missing)} mounts didn't appear: {sorted(missing)}")
            except Exception as e:
                _msg = str(e).lower()
                if "already attached" in _msg:
                    print(f"  Some assets already attached: {e}")
                else:
                    print(f"  Attach failed: {e}")


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


# %% CELL 3 — Process all TARGETS, save figures
# Skip sessions whose PNG already exists in /scratch/learning_pngs/ (persistent
# across workstation sessions on this capsule). Re-running this cell is
# idempotent: previously-processed sessions are skipped.
# Set FORCE = True to re-run sessions that already have a PNG.

import time

FORCE = False

completed: list[dict] = []
skipped: list[dict] = []
failed: list[dict] = []
t0 = time.time()
figures_dir = Path("/results") / "figures"
persistent_pngs_dir = Path("/scratch/learning_pngs")
figures_dir.mkdir(parents=True, exist_ok=True)
persistent_pngs_dir.mkdir(parents=True, exist_ok=True)

for subject, date, stem in TARGETS:
    session_tag = f"{subject}_{date}_{stem}"
    persisted_png = persistent_pngs_dir / f"{session_tag}_fig_00.png"

    if persisted_png.exists() and not FORCE:
        # Also mirror to /results/figures/ in case this is a fresh workstation
        # and /results/ doesn't have it yet (so the saved-asset still includes
        # historical sessions).
        results_png = figures_dir / f"{session_tag}_fig_00.png"
        if not results_png.exists():
            import shutil
            shutil.copy(persisted_png, results_png)
        print(f"  SKIP  {subject}  {date}  {stem}  (PNG in /scratch/learning_pngs)")
        skipped.append({"subject": subject, "date": date, "stem": stem,
                        "png": str(persisted_png)})
        continue

    print(f"\n{'='*70}")
    print(f"  {subject}  {date}  {stem}")
    print(f"{'='*70}")
    t_session = time.time()
    try:
        results_path, n_figs = process_session(subject, date, stem)
        dt = time.time() - t_session
        completed.append({"subject": subject, "date": date, "stem": stem,
                          "n_figures": n_figs, "png": str(results_path),
                          "elapsed_sec": round(dt, 1)})
        print(f"  -> {results_path}  ({n_figs} figs, {dt:.0f} sec)")
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
print(f"  skipped:   {len(skipped)}  (PNG already existed; set FORCE=True to re-run)")
print(f"  failed:    {len(failed)}")
print(f"{'='*70}")
if failed:
    print("Failures:")
    for f in failed:
        print(f"  {f['subject']} {f['date']} {f['stem']}: {f['error']}")

# Save a batch summary alongside per-session outputs
with open(Path("/results") / "batch_summary.json", "w") as f:
    json.dump({"completed": completed, "skipped": skipped, "failed": failed,
               "total_sec": round(t_total, 1)}, f, indent=2)


# %% CELL 4 — Quick peek at any session's figures (set INSPECT to a session)
INSPECT = TARGETS[0] if TARGETS else None
if INSPECT:
    subject, date, stem = INSPECT
    session_tag = f"{subject}_{date}_{stem}"
    fig_dir = Path("/results") / "figures"
    pngs = sorted(fig_dir.glob(f"{session_tag}_fig_*.png"))
    print(f"{session_tag}: {len(pngs)} figures in {fig_dir}")
    for png in pngs:
        from matplotlib.image import imread
        img = imread(str(png))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(png.name)
        plt.show()

# %%
