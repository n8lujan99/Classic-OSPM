from copy import deepcopy
from pathlib import Path

from Data.Galaxy_Profiles.Segue1.Segue1_OSPM_Config import CONFIG as SEGUE1_FULL_CONFIG


SEGUE1_ROOT = Path(__file__).resolve().parents[1] / "Segue1"

CONFIG = deepcopy(SEGUE1_FULL_CONFIG)
CONFIG["GALAXY"] = "Segue1Matched"
CONFIG["COMPARISON_TAG"] = "matched_bins"

CONFIG["SURFACE_BRIGHTNESS_CSV"] = str(
    SEGUE1_ROOT / "segue1_NO09_surface_brightness_on_simon_bins_16.csv"
)
CONFIG["STELLAR_MODEL"]["grid_csv"] = str(
    SEGUE1_ROOT / "segue1_NO09_axisymmetric_light_grid.csv"
)

CONFIG["CSV_PATH"] = str(
    SEGUE1_ROOT
    / "default"
    / "hpc_compare"
    / "matched_bins"
    / "daemon_deck_karl_segue1_matched_bins_vcirc.csv"
)

print("[CONFIG] COMPARISON_TAG =", CONFIG["COMPARISON_TAG"])
print("[CONFIG] MATCHED SURFACE_BRIGHTNESS_CSV =", CONFIG["SURFACE_BRIGHTNESS_CSV"])
print("[CONFIG] MATCHED GRID_CSV =", CONFIG["STELLAR_MODEL"]["grid_csv"])
print("[CONFIG] CSV_PATH =", CONFIG["CSV_PATH"])
