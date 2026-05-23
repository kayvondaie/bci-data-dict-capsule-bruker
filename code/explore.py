# %% Imports
import sys
sys.path.insert(0, "/code")
import data_dict_create_module_bruker
import bonsai_npy_threshold_calculator
import extract_scanimage_metadata
import folder_props_fun
print("All modules imported.")

# %% What's in /data?
import os
from pathlib import Path
for asset in os.listdir("/data"):
    print(f"\n=== /data/{asset} ===")
    for f in sorted(os.listdir(f"/data/{asset}"))[:20]:
        print(f"  {f}")