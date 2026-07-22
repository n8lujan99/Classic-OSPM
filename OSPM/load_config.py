from pathlib import Path
from importlib import import_module
import os
from .AI_defaults import CONFIG as AI_DEFAULTS

# --------------------------------------------------
# Galaxy authority (single source of truth)
# --------------------------------------------------

OSPM_ROOT = Path(__file__).resolve().parents[1]
WHICH_FILE = OSPM_ROOT / "which_galaxy"


def _get_galaxy_name():
    env_name = os.environ.get("OSPM_GALAXY", "").strip()
    if env_name:
        return env_name
    if not WHICH_FILE.exists():
        raise RuntimeError("which_galaxy missing at repo root")
    name = WHICH_FILE.read_text().strip()
    if not name:
        raise RuntimeError("which_galaxy is empty")
    return name


def get_profile_root():
    gal = _get_galaxy_name()
    return OSPM_ROOT / "Data" / "Galaxy_Profiles" / gal


# --------------------------------------------------
# Config loader
# --------------------------------------------------

def load_config():
    galaxy = _get_galaxy_name()

    module_name = f"Data.Galaxy_Profiles.{galaxy}.{galaxy}_OSPM_Config"

    try:
        mod = import_module(module_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load config for {galaxy} using {module_name}") from e

    cfg = {**AI_DEFAULTS, **mod.CONFIG}

    # Declare identity explicitly
    cfg["GALAXY"] = galaxy
    cfg["HALO_TYPE"] = str(cfg.get("HALO_TYPE", "none")).strip().lower()
    cfg["HALO_PARAMETERIZATION"] = str(
        cfg.get("HALO_PARAMETERIZATION", "rho_rs")
    ).strip().lower()

    if cfg["HALO_PARAMETERIZATION"] in ("", "default"):
        cfg["HALO_PARAMETERIZATION"] = "rho_rs"

    allowed_halo_parameterizations = ("rho_rs", "vcirc_rs", "v0_rc")
    if cfg["HALO_PARAMETERIZATION"] not in allowed_halo_parameterizations:
        raise ValueError(
            "HALO_PARAMETERIZATION must be 'rho_rs', 'vcirc_rs', or 'v0_rc', "
            f"got {cfg['HALO_PARAMETERIZATION']!r}"
        )

    halo_type = cfg["HALO_TYPE"]
    halo_parameterization = cfg["HALO_PARAMETERIZATION"]

    if halo_parameterization == "v0_rc" and halo_type != "nonsingular_isothermal":
        raise ValueError(
            "HALO_PARAMETERIZATION='v0_rc' requires "
            "HALO_TYPE='nonsingular_isothermal', "
            f"got HALO_TYPE={halo_type!r}"
        )

    if halo_type == "nonsingular_isothermal" and halo_parameterization != "v0_rc":
        raise ValueError(
            "HALO_TYPE='nonsingular_isothermal' requires "
            "HALO_PARAMETERIZATION='v0_rc', "
            f"got HALO_PARAMETERIZATION={halo_parameterization!r}"
        )

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

    print("[CONFIG LOAD] HALO_PARAMETERIZATION =", cfg["HALO_PARAMETERIZATION"])
    print("[CONFIG LOAD] PARAMETER_NAMES =", cfg.get("PARAMETER_NAMES"))
    print("[CONFIG LOAD] THETA_BOUNDS =", cfg.get("THETA_BOUNDS"))

    return cfg
