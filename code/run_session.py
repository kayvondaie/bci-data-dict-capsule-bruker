"""Non-interactive entry point for a single BCI session's data-dict analysis.

Invoked by `code/run` when the capsule is launched as a Reproducible Run
(e.g. from the bci_analysis orchestrator capsule). Mirrors the logic of
`explore.py` CELLs 1-3 but reads SUBJECT/DATE/TARGET_STEM from env vars
and writes everything to /results/ instead of showing plots.

Environment variables (set by the orchestrator's run_capsule call):
    SUBJECT       e.g. "850378"
    DATE          e.g. "2026-05-26"
    TARGET_STEM   e.g. "bci" or "bci2"; default "bci"

Inputs (mounted by Reproducible Run):
    /data/<raw_asset>/         raw session data (with pophys/, behavior/)
    /data/<processed_asset>/   kd pipeline output (extraction/, motion_correction/)

Outputs:
    /results/data_dict.pkl                    pickled data dict from ddc.main
    /results/figures/<target>_fig_<n>.png     one PNG per bonsai figure
    /results/run_log.txt                      brief summary
"""

# Headless plotting — must come before any pyplot import.
import matplotlib
matplotlib.use("Agg")

import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")

import extract_scanimage_metadata
import data_dict_create_module_bruker
import bonsai_npy_threshold_calculator
ddc = data_dict_create_module_bruker

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config from env vars
# ---------------------------------------------------------------------------
SUBJECT = os.environ.get("SUBJECT")
DATE = os.environ.get("DATE")
TARGET_STEM = os.environ.get("TARGET_STEM", "bci")
# COLOR_SUFFIX / CHANNEL are set by the dual-channel dispatcher below when
# it re-invokes this script per channel. Empty for single-channel runs.
COLOR_SUFFIX = os.environ.get("COLOR_SUFFIX", "")

if not SUBJECT or not DATE:
    print("ERROR: SUBJECT and DATE env vars must be set.", file=sys.stderr)
    print("       Set them via the orchestrator's run_capsule(parameters=...) call.",
          file=sys.stderr)
    sys.exit(2)

print("=" * 70)
print(f"BCI data-dict session run")
print(f"  SUBJECT     = {SUBJECT}")
print(f"  DATE        = {DATE}")
print(f"  TARGET_STEM = {TARGET_STEM}")
if COLOR_SUFFIX:
    print(f"  COLOR       = {COLOR_SUFFIX.lstrip('_')}")
print("=" * 70)

# Workspace stays per-session for isolation, but outputs land flat in
# /results/ with session-tagged filenames so all sessions in a batch
# share one figures/ folder. Per-channel runs get a color-suffixed tag so
# two subprocesses don't collide on scratch dir or output filenames.
SESSION_TAG = f"{SUBJECT}_{DATE}_{TARGET_STEM}{COLOR_SUFFIX}"
WORKSPACE = Path(f"/scratch/{SESSION_TAG}")
POPHYS = WORKSPACE / "pophys"
RESULTS_ROOT = Path("/results")
FIGURES_DIR = RESULTS_ROOT / "figures"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Find raw + processed assets in /data/
# ---------------------------------------------------------------------------
asset_prefix = f"single-plane-ophys_{SUBJECT}_{DATE}_"
attached = sorted(Path("/data").iterdir())

raws = [
    p for p in attached
    if p.name.startswith(asset_prefix) and "_processed_" not in p.name
]
# Tolerate manually-named processed assets that don't start with single-plane-ophys_
procs = [
    p for p in attached
    if p not in raws and SUBJECT in p.name and DATE in p.name
]

if not raws:
    print(f"ERROR: no RAW asset attached matching {asset_prefix}", file=sys.stderr)
    print(f"  Attached: {[p.name for p in attached]}", file=sys.stderr)
    sys.exit(3)
if len(raws) > 1:
    print(f"ERROR: multiple raw assets match — attach only one:", file=sys.stderr)
    for r in raws:
        print(f"  {r.name}", file=sys.stderr)
    sys.exit(3)
raw = raws[0]

if not procs:
    print(f"ERROR: no PROCESSED asset attached for {SUBJECT}/{DATE}", file=sys.stderr)
    print(f"  Attached: {[p.name for p in attached]}", file=sys.stderr)
    sys.exit(3)


def _proc_ts(p):
    m = re.search(r"_processed_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", p.name)
    return m.group(1) if m else ""


procs_sorted = sorted(procs, key=_proc_ts, reverse=True)
proc = procs_sorted[0]
if len(procs_sorted) > 1:
    print(f"Multiple processed assets matched; using most recent:")
    print(f"  using:   {proc.name}")
    for older in procs_sorted[1:]:
        print(f"  ignored: {older.name}")

# === DUAL-CHANNEL DISPATCHER ==========================================
# For dual-channel sessions the processed asset has `chan1/`, `chan2/`
# subdirectories at the top level (from the dual-channel motion-correction
# and extraction dispatchers upstream). If we see those AND we're not
# already inside a per-channel subprocess, dispatch one subprocess per
# channel. Each subprocess sets CHANNEL=chan1 and COLOR_SUFFIX=_green (or
# chan2/_red), and its outputs land as `<session_tag>_green.h5` /
# `<session_tag>_red.h5` in /results/.
#
# Channel-to-color mapping (rig convention, per Kayvon):
#   chan1 = green (GCaMP in neuromodulatory axons)
#   chan2 = red   (BCI cell bodies in M1)
_COLOR_MAP = {"chan1": "green", "chan2": "red"}
_channel = os.environ.get("CHANNEL")  # set by parent when dispatching

_chan_subdirs = sorted(
    p for p in proc.iterdir() if p.is_dir() and re.match(r"^chan\d+$", p.name)
) if proc.is_dir() else []

if _chan_subdirs and _channel is None:
    print(f"Dual-channel processed asset detected: "
          f"{[c.name for c in _chan_subdirs]}; running once per channel.")
    _rc_max = 0
    for _chan_dir in _chan_subdirs:
        _color = _COLOR_MAP.get(_chan_dir.name, _chan_dir.name)
        _env = os.environ.copy()
        _env["CHANNEL"] = _chan_dir.name
        _env["COLOR_SUFFIX"] = f"_{_color}"
        _cmd = [sys.executable, __file__]
        print(f"\n=== Dispatching {_chan_dir.name} ({_color}) ===")
        _rc = subprocess.call(_cmd, env=_env)
        print(f"=== {_chan_dir.name} ({_color}) exit code = {_rc} ===\n")
        _rc_max = max(_rc_max, _rc)
    sys.exit(_rc_max)

# If CHANNEL was set (we're inside a per-channel subprocess), narrow proc
# down to that channel's subtree BEFORE the usual extraction/motion_correction
# discovery. Everything below this line then runs unchanged, just against
# the chan-specific data.
if _channel is not None:
    _chan_root = proc / _channel
    if not _chan_root.is_dir():
        print(f"ERROR: CHANNEL={_channel} but {_chan_root} not found",
              file=sys.stderr)
        sys.exit(3)
    print(f"Per-channel subprocess: narrowing proc {proc.name} -> {_channel}")
    proc = _chan_root  # everything downstream treats this as `proc`

# Newer assets nest extraction/ under a subfolder; older are flat.
if (proc / "extraction").is_dir():
    proc_root = proc
else:
    candidates = [c for c in proc.iterdir() if c.is_dir() and (c / "extraction").is_dir()]
    if not candidates:
        print(f"ERROR: no extraction/ in {proc.name}", file=sys.stderr)
        sys.exit(3)
    proc_root = candidates[0]

extraction = proc_root / "extraction"
mc = proc_root / "motion_correction"

print(f"raw:        {raw.name}")
print(f"processed:  {proc.name}"
      + (f"  (channel {_channel} / {COLOR_SUFFIX.lstrip('_')})" if _channel else ""))
print(f"  data at:  {proc_root}")


# ---------------------------------------------------------------------------
# Resolve target epoch (with fallback)
# ---------------------------------------------------------------------------
with open(mc / "epoch_locations.json") as f:
    epoch_locations = json.load(f)
with open(mc / "trial_locations.json") as f:
    trial_locations = json.load(f)

if TARGET_STEM not in epoch_locations:
    base = TARGET_STEM.rstrip("0123456789") or TARGET_STEM
    candidates = sorted(k for k in epoch_locations if base in k)
    if not candidates:
        print(f"ERROR: epoch {TARGET_STEM!r} not found and no fallback containing {base!r}",
              file=sys.stderr)
        print(f"  Available: {list(epoch_locations)}", file=sys.stderr)
        sys.exit(4)
    resolved = base if base in epoch_locations else candidates[0]
    print(f"NOTE: {TARGET_STEM!r} not in epoch_locations; using {resolved!r}")
    TARGET_STEM = resolved

target_start, target_end = epoch_locations[TARGET_STEM]
target_n_frames = target_end - target_start + 1
print(f"epoch[{TARGET_STEM!r}] = [{target_start}, {target_end}]  ({target_n_frames} frames)")


# ---------------------------------------------------------------------------
# Build workspace at /scratch/session/
# ---------------------------------------------------------------------------
subprocess.run(["rm", "-rf", str(WORKSPACE)], check=False)
POPHYS.mkdir(parents=True)
(WORKSPACE / "behavior").symlink_to(raw / "behavior")

target_tifs = sorted(
    name for name, (s, e) in trial_locations.items()
    if name.startswith(f"{TARGET_STEM}_") and s >= target_start and e <= target_end
)
frames_per_file = [trial_locations[n][1] - trial_locations[n][0] + 1 for n in target_tifs]
print(f"{len(target_tifs)} TIFFs in {TARGET_STEM} epoch ({sum(frames_per_file)} frames)")

# No need to symlink TIFFs — nothing reads them from the workspace.
# siHeader is extracted directly from the raw asset below.
# Sidecar files (csv, mat) ARE used by ddc.main, so symlink those.
for f in (raw / "pophys").iterdir():
    if f.is_file() and f.suffix != ".tif" and f.name.startswith(f"{TARGET_STEM}_"):
        tgt = POPHYS / f.name
        if not tgt.exists():
            tgt.symlink_to(f)

bci_dir = POPHYS / "suite2p_BCI" / "plane0"
bci_dir.mkdir(parents=True)
frame_slice = slice(target_start, target_end + 1)
for fname in ["F", "Fneu", "spks"]:
    src_path = extraction / f"{fname}.npy"
    if src_path.exists():
        arr = np.load(src_path)
        sliced = arr[:, frame_slice]
        np.save(bci_dir / f"{fname}.npy", sliced)
        print(f"  Sliced {fname}.npy: {arr.shape} -> {sliced.shape}")
        del arr

for fname in ["stat.npy", "iscell.npy"]:
    src = extraction / fname
    if src.exists():
        (bci_dir / fname).symlink_to(src)

ops_path = extraction / "ops.npy"
ops = np.load(ops_path, allow_pickle=True).tolist()
ops["frames_per_file"] = frames_per_file
np.save(bci_dir / "ops.npy", ops)

# Read siHeader directly from the raw asset.
first_tif = raw / "pophys" / target_tifs[0]
siHeader = extract_scanimage_metadata.extract_scanimage_metadata(str(first_tif))
siHeader["siBase"] = {0: TARGET_STEM, 1: "", 2: "spont_pre"}
siHeader["savefolders"] = {0: TARGET_STEM, 1: "spont", 2: "spont_post", 3: "spont_pre", 4: "spont_post"}
np.save(bci_dir / "siHeader.npy", siHeader)

if "spont_pre" in epoch_locations:
    s_start, s_end = epoch_locations["spont_pre"]
    spont_dir = POPHYS / "suite2p_spont_pre" / "plane0"
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

# Build suite2p_photostim[N]/ slices for any photostim epochs in this session.
# data_dict_create_module_bruker.py:178 looks for suite2p_photostim/ to
# populate data['photostim'] via create_photostim_Fstim(); without these
# directories the photostim block silently produces nothing.
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
            print(f"NOTE: photostim epoch {ps_stem!r} not in epoch_locations, skipping.")
            continue
        ps_start, ps_end = epoch_locations[ps_stem]
        suffix = "" if i == 0 else str(i + 1)
        ps_dir = POPHYS / f"suite2p_photostim{suffix}" / "plane0"
        ps_dir.mkdir(parents=True, exist_ok=True)
        ps_slice = slice(ps_start, ps_end + 1)
        for fname in ["F", "Fneu"]:
            src_path = extraction / f"{fname}.npy"
            if src_path.exists():
                arr = np.load(src_path)
                np.save(ps_dir / f"{fname}.npy", arr[:, ps_slice])
                del arr
        # ROIs/ops are shared across all epochs (segmented once, non-photostim only).
        for fname in ["stat.npy", "iscell.npy", "ops.npy"]:
            src = extraction / fname
            if src.exists() and not (ps_dir / fname).exists():
                (ps_dir / fname).symlink_to(src)
        # siHeader from the photostim TIFF (carries stimGroups/SLM metadata that
        # create_photostim_Fstim reads to build stimPosition, stimDist, etc.).
        ps_tifs = sorted(
            n for n, (s, e) in trial_locations.items()
            if n.startswith(f"{ps_stem}_") and s >= ps_start and e <= ps_end
        )
        if ps_tifs:
            ps_first_tif = raw / "pophys" / ps_tifs[0]
            ps_siHeader = extract_scanimage_metadata.extract_scanimage_metadata(
                str(ps_first_tif)
            )
            ps_siHeader["siBase"] = {0: TARGET_STEM, 1: "", 2: ps_stem}
            ps_siHeader["savefolders"] = {
                0: TARGET_STEM, 1: "spont", 2: f"photostim{suffix}"
            }
            np.save(ps_dir / "siHeader.npy", ps_siHeader)
        print(
            f"Built suite2p_photostim{suffix}/ from epoch {ps_stem!r} "
            f"({ps_end - ps_start + 1} frames)"
        )

print(f"Workspace ready at {WORKSPACE}")


# ---------------------------------------------------------------------------
# Run ddc.main + bonsai
# ---------------------------------------------------------------------------
folder = str(POPHYS) + "/"
print(f"\nRunning ddc.main(folder={folder!r}) ...")
data = ddc.main(folder)

# Override mouse/session — ddc's path parsing assigns nonsense for /scratch/session/
m = re.match(r"single-plane-ophys_(\d+)_(\d{4}-\d{2}-\d{2})_", raw.name)
if m:
    data["mouse"] = m.group(1)
    data["session"] = m.group(2)

print(f"ddc.main returned dict with {len(data)} keys")


print(f"\nRunning bonsai_npy_threshold_calculator.run(...) ...")
try:
    figs = bonsai_npy_threshold_calculator.run(folder, data)
    print(f"  produced {len(figs)} figures")
except Exception as e:
    print(f"  bonsai run failed: {e}", file=sys.stderr)
    traceback.print_exc()
    figs = []


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
# Copy ddc.main's HDF5 output to /results/<session_tag>.h5
import shutil
h5_candidates = list(POPHYS.glob("data_main_*_BCI.h5"))
h5_out = RESULTS_ROOT / f"{SESSION_TAG}.h5"
if h5_candidates:
    shutil.copy(h5_candidates[0], h5_out)
    print(f"\nWrote {h5_out}")
else:
    print(f"\nWARNING: no data_main_*_BCI.h5 found in {POPHYS}")

for i, fig in enumerate(figs):
    out = FIGURES_DIR / f"{SESSION_TAG}_fig_{i:02d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

print(f"\nDone.")
