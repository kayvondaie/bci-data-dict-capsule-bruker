# bci-data-dict-capsule

Code Ocean capsule that reproduces basic BCI analysis using the AIND
processed data assets. Wraps your local Bergamo/Bruker data-dict creation
and Bonsai NPY threshold calculation in a CO-runnable form.

## What's in `code/`

| File | Purpose |
|------|---------|
| `data_dict_create_module_bruker.py` | Builds the BCI analysis data dictionary from session data |
| `extract_scanimage_metadata.py` | Reads ScanImage TIFF metadata; called by the above |
| `folder_props_fun.py` | Helper for enumerating session folder layout |
| `bonsai_npy_threshold_calculator.py` | Computes Bonsai thresholds from NPY traces |
| `run` | Entry point invoked when the capsule runs |

## Environment

`environment/Dockerfile` installs the pip deps:

- numpy, scipy, pandas, h5py, matplotlib
- scanimage-tiff-reader, tifffile

Plus a placeholder `git-askpass` file to satisfy the upstream-CO Dockerfile
convention (same trick we used for motion-correction-kd / extraction-kd).

## Status: WORK IN PROGRESS

The `code/run` script currently just imports the modules to confirm the
environment builds correctly. **It does not yet do real analysis.**

To make this capsule useful, the `run` script needs to be fleshed out with
calls to `data_dict_create_module_bruker.main(...)` (or whatever the right
entry point is) that consume `/data/<attached-asset>/` and write outputs
to `/results/`.

## Data inputs

The processed data asset from `aind-single-plane-ophys-pipeline-kd` contains:

- `*_extraction.h5` — combined traces / ROIs
- `suite2p_bci/plane0/{F,Fneu,stat,ops,iscell,spks}.npy` — BCI epoch
- `suite2p_spont_pre/plane0/{F,Fneu,stat,ops,iscell,spks}.npy` — spont epoch
- `session.json`, `rig.json`, `data_description.json`, etc.
- **No raw `.tif` files** — those got consumed by the bergamo_stitcher

If `data_dict_create_module_bruker` needs raw TIFF headers, also attach the
raw asset (`single-plane-ophys_<subject>_<date>`) so its `pophys/*.tif`
files are available at `/data/<raw-asset>/pophys/`.

## Iteration workflow

This repo is intended to be a CO capsule's source. To iterate:

1. Edit `.py` files on the share at
   `\\allen\aind\scratch\BCI\data_uploader_gui\bci-data-dict-capsule\`
2. Commit + push to GitHub
3. Pull from GitHub in the CO capsule (Reproducibility panel → Pull)
4. Reproducible Run

Same pattern we used for the motion-correction and extraction capsule forks.
