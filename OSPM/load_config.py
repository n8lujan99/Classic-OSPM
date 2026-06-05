from pathlib import Path
from importlib import import_module
from .AI_defaults import CONFIG as AI_DEFAULTS

# --------------------------------------------------
# Galaxy authority (single source of truth)
# --------------------------------------------------

OSPM_ROOT = Path(__file__).resolve().parents[1]
WHICH_FILE = OSPM_ROOT / "which_galaxy"

def _get_galaxy_name():
    if not WHICH_FILE.exists():
        raise RuntimeError("which_galaxy missing at repo root")
    name = WHICH_FILE.read_text().strip()
    if not name:
        raise RuntimeError("which_galaxy is empty")
    return name

def get_profile_root():
    gal = _get_galaxy_name()
    return OSPM_ROOT / "Data" / "Gal_Profiles" / gal

# --------------------------------------------------
# Config loader
# --------------------------------------------------

def load_config():
    galaxy = _get_galaxy_name()

    try:
        mod = import_module(
            f"Data.Gal_Profiles.{galaxy}.{galaxy}_OSPM_Config"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load config for {galaxy}") from e

    cfg = {**AI_DEFAULTS, **mod.CONFIG}

    # Declare identity explicitly
    cfg["GALAXY"] = galaxy

    required = [
        "GALAXY",
        "MODE",
        "HALO_TYPE",
        "MIN_DISTANCE",
        "MAX_DISTANCE",
        "NORBIT",
        "BATCH_SIZE",
        "MAX_RUNS",
    ]

    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"CONFIG missing required keys: {missing}")

    return cfg
