"""Find recently-uploaded BCI sessions and attach them to this capsule.

Shift+Enter each cell. Edit HOURS / DRY_RUN at the top to taste.
After attaching, stop + relaunch the cloud workstation to see them in /data.
"""

# %% CELL 1 — Config + imports
HOURS = 30        # look back this many hours from now
DRY_RUN = False    # True = preview only, False = actually attach

import os
import sys
from datetime import datetime, timedelta, timezone

from codeocean import CodeOcean
from codeocean.data_asset import DataAssetSearchParams
from codeocean.capsule import DataAssetAttachParams

CAPSULE_ID = os.environ["CO_CAPSULE_ID"]
TOKEN = os.environ.get("CODEOCEAN_TOKEN")
DOMAIN = "https://codeocean.allenneuraldynamics.org"

if not TOKEN:
    raise RuntimeError("CODEOCEAN_TOKEN env var not set — add it as a Secret on this capsule.")

client = CodeOcean(domain=DOMAIN, token=TOKEN)
print(f"Capsule: {CAPSULE_ID}")
print(f"Looking back {HOURS} hours, dry_run={DRY_RUN}")


# %% CELL 2 — Search for recently-uploaded BCI assets
cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS)
cutoff_unix = int(cutoff.timestamp())
print(f"Cutoff: {cutoff.isoformat()} UTC")

results = client.data_assets.search_data_assets(DataAssetSearchParams(
    query="single-plane-ophys_8",   # all our subject IDs start with 8
    limit=200,
))

matched = [
    a for a in results.results
    if a.created >= cutoff_unix
    and a.name.startswith("single-plane-ophys_")
]
print(f"\nFound {len(matched)} matching assets in last {HOURS} hr:")
for a in matched:
    age_hr = (datetime.now(timezone.utc).timestamp() - a.created) / 3600
    print(f"  [{age_hr:5.1f}h] {a.name}")


# %% CELL 3 — Attach (set DRY_RUN=False above first!)
if DRY_RUN:
    print("[DRY_RUN] not attaching. Set DRY_RUN=False in CELL 1 and re-run CELLs 1+3.")
elif not matched:
    print("Nothing to attach.")
else:
    print(f"Attaching {len(matched)} assets to capsule {CAPSULE_ID}...")
    n_ok, n_skipped, n_failed = 0, 0, 0
    for a in matched:
        try:
            client.capsules.attach_data_assets(
                capsule_id=CAPSULE_ID,
                attach_params=[DataAssetAttachParams(id=a.id, mount=a.name)],
            )
            print(f"  OK     {a.name}")
            n_ok += 1
        except Exception as e:
            msg = str(e).lower()
            if "already" in msg or "exists" in msg or "duplicate" in msg:
                print(f"  SKIP   {a.name}  (already attached)")
                n_skipped += 1
            else:
                print(f"  FAIL   {a.name}  ({e})")
                n_failed += 1
    print(f"\nDone: {n_ok} attached, {n_skipped} skipped, {n_failed} failed.")
    print("Restart the cloud workstation to see new attachments in /data.")
# %%
