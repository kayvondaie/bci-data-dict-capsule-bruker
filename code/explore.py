# %% CELL 1 — Config, imports, asset discovery, workspace build
import os, json, subprocess, re, sys, logging
import numpy as np
from pathlib import Path

sys.path.insert(0, "/code")
import extract_scanimage_metadata
import data_dict_create_module_bruker
ddc = data_dict_create_module_bruker

%matplotlib inline
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ====== FILL THESE IN ======
SUBJECT = "820614"
DATE = "2026-05-28"
TARGET_STEM = "bci"   # which epoch to extract (bci / bci2 / spont_pre / ...)
# ===========================

WORKSPACE = Path("/scratch/session")
POPHYS = WORKSPACE / "pophys"

# --- Find raw asset ---
asset_prefix = f"single-plane-ophys_{SUBJECT}_{DATE}_"
attached = sorted(Path("/data").iterdir())
raws = [p for p in attached if p.name.startswith(asset_prefix) and "_processed_" not in p.name]
# Accept manually-named processed assets that don't follow the auto-capture
# convention (e.g. "850378_2026-05-26_11-35-43" missing _processed_).
procs = [
    p for p in attached
    if p not in raws and SUBJECT in p.name and DATE in p.name
]

if not raws:
    raise RuntimeError(
        f"No RAW asset attached matching subject={SUBJECT} date={DATE}.\n"
        f"Expected something starting with: {asset_prefix}\n"
        f"Currently attached:\n  " + "\n  ".join(p.name for p in attached)
    )
if len(raws) > 1:
    raise RuntimeError(
        f"Multiple raw assets match subject={SUBJECT} date={DATE}, can't pick:\n  "
        + "\n  ".join(p.name for p in raws) + "\nDetach all but one."
    )
raw = raws[0]

if not procs:
    raise RuntimeError(
        f"No PROCESSED asset attached matching subject={SUBJECT} date={DATE}.\n"
        f"Currently attached:\n  " + "\n  ".join(p.name for p in attached)
    )

def _proc_ts(p):
    m = re.search(r"_processed_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", p.name)
    return m.group(1) if m else ""

procs_sorted = sorted(procs, key=_proc_ts, reverse=True)
proc = procs_sorted[0]
if len(procs_sorted) > 1:
    print(f"Multiple processed assets matched; using most recent ({_proc_ts(proc)}):")
    print(f"  using:   {proc.name}")
    for older in procs_sorted[1:]:
        print(f"  ignored: {older.name}")

# Detect nested folder (newer assets) vs flat (older assets)
if (proc / "extraction").is_dir():
    proc_root = proc
else:
    candidates = [c for c in proc.iterdir() if c.is_dir() and (c / "extraction").is_dir()]
    if not candidates:
        raise RuntimeError(
            f"Couldn't find extraction/ inside {proc.name}. Contents:\n  "
            + "\n  ".join(p.name for p in proc.iterdir())
        )
    proc_root = candidates[0]

extraction = proc_root / "extraction"
mc = proc_root / "motion_correction"

print(f"raw:        {raw.name}")
print(f"processed:  {proc.name}")
print(f"  data at:  {proc_root}")
print(f"target:     epoch '{TARGET_STEM}'")

# --- Validate target epoch ---
with open(mc / "epoch_locations.json") as f:
    epoch_locations = json.load(f)
with open(mc / "trial_locations.json") as f:
    trial_locations = json.load(f)

if TARGET_STEM not in epoch_locations:
    raise RuntimeError(
        f"Epoch {TARGET_STEM!r} not in this processed asset's epoch_locations.\n"
        f"Available: {list(epoch_locations)}"
    )

target_start, target_end = epoch_locations[TARGET_STEM]
target_n_frames = target_end - target_start + 1
print(f"epoch_locations[{TARGET_STEM!r}] = [{target_start}, {target_end}]  ({target_n_frames} frames)")

# --- Build workspace ---
subprocess.run(["rm", "-rf", str(WORKSPACE)], check=False)
POPHYS.mkdir(parents=True)
(WORKSPACE / "behavior").symlink_to(raw / "behavior")

target_tifs = sorted(
    name for name, (s, e) in trial_locations.items()
    if name.startswith(f"{TARGET_STEM}_") and s >= target_start and e <= target_end
)
frames_per_file = [trial_locations[n][1] - trial_locations[n][0] + 1 for n in target_tifs]
print(f"{len(target_tifs)} TIFFs in {TARGET_STEM} epoch ({sum(frames_per_file)} frames)")

for tname in target_tifs:
    src = raw / "pophys" / tname
    if src.is_file():
        (POPHYS / tname).symlink_to(src)

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

for fname in ["stat.npy", "iscell.npy"]:
    src = extraction / fname
    if src.exists():
        (bci_dir / fname).symlink_to(src)

ops_path = extraction / "ops.npy"
ops = np.load(ops_path, allow_pickle=True).tolist()
ops["frames_per_file"] = frames_per_file
np.save(bci_dir / "ops.npy", ops)

first_tif = sorted(POPHYS.glob(f"{TARGET_STEM}_*.tif"))[0]
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
    for fname in ["stat.npy", "iscell.npy"]:
        src = extraction / fname
        if src.exists():
            (spont_dir / fname).symlink_to(src)

print(f"\n✓ Workspace ready at {WORKSPACE}")


# %% CELL 2 — Run ddc.main with the workspace as folder
import importlib
importlib.reload(data_dict_create_module_bruker)
ddc = data_dict_create_module_bruker

folder = str(POPHYS) + "/"
print(f"Calling ddc.main(folder={folder!r}) ...")
data = ddc.main(folder)

# Override mouse/session — ddc parses them from path, which is useless for our workspace
m = re.match(r"single-plane-ophys_(\d+)_(\d{4}-\d{2}-\d{2})_", raw.name)
if m:
    data["mouse"] = m.group(1)
    data["session"] = m.group(2)
    print(f"Overrode mouse={data['mouse']}, session={data['session']}")

print(f"\nReturned dict keys ({len(data)}):")
for k in data:
    v = data[k]
    if hasattr(v, "shape"):
        print(f"  {k}: array {v.shape} {v.dtype}")
    elif isinstance(v, (list, tuple)):
        print(f"  {k}: {type(v).__name__} length {len(v)}")
    else:
        s = repr(v)
        print(f"  {k}: {type(v).__name__} = {s[:80]}{'...' if len(s) > 80 else ''}")


# %% CELL 3 — Run bonsai threshold analysis
import importlib
import bonsai_npy_threshold_calculator
importlib.reload(bonsai_npy_threshold_calculator)

figs = bonsai_npy_threshold_calculator.run(folder, data)
print(f"Generated {len(figs)} figures.")
import matplotlib.pyplot as plt
plt.show()
# %%
