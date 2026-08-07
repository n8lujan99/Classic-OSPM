# OSPM_Config_Center — Draco
# Karl-style Draco config.
#
# Observational inputs:
#   1. Odenkirchen et al. 2001 surface-brightness tracer profile
#   2. Walker et al. 2023 stellar LOS velocity sample
#
# The 3D light grid is derived from the surface-brightness profile.
# It is not an additional observational data set.
#
# Shared solver, orbit-library, AI, deck, and runtime defaults are supplied by
# OSPM/load_config.py. This file contains only Draco-specific authority.

from pathlib import Path
from Data.Data_Prep.Data_Paths import build_data_paths
LOCAL_DEBUG = False
PROFILE_ROOT = Path(__file__).resolve().parent
if not PROFILE_ROOT.exists():
    raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")

INITIAL_THETA = [100.0, 1800.0, 9.0e5, 1.0]
FIXED_THETA = INITIAL_THETA.copy() if LOCAL_DEBUG else None

# Alternative local-debug tests:
# FIXED_THETA = [100.0, 1800.0, 0.0, 1.0]       # No black hole
# FIXED_THETA = [100.0, 1800.0, 3.0e6, 1.0]     # Larger black hole
# FIXED_THETA = [100.0, 300.0, 9.0e5, 1.0]      # Compact core
# FIXED_THETA = [100.0, 20000.0, 9.0e5, 1.0]    # Extended core

CONFIG = {
    # =========================================================
    # Local debug control
    # =========================================================
    "LOCAL_DEBUG": LOCAL_DEBUG,
    "FIXED_THETA": FIXED_THETA,
    # Draco keeps the full orbit library during local debug.
    "NORBIT": 10000,
    # =========================================================
    # Halo model and parameterization
    # =========================================================
    "HALO_TYPE": "nonsingular_isothermal",
    "HALO_PARAMETERIZATION": "v0_rc",
    "PARAMETER_NAMES": ["v0", "r_c", "MBH", "ML"],
    "INITIAL_THETA": INITIAL_THETA,
    "THETA_BOUNDS": [
        (0.0, 200.0),       # v0, km/s
        (100.0, 1.0e6),     # r_c, pc
        (0.0, 5.0e6),       # MBH, Msun
        (0.2, 20.0),        # M/L
    ],
    # =========================================================
    # Galaxy geometry
    # =========================================================
    "RA0_DEG": 260.0517,
    "DEC0_DEG": 57.9153,
    "DISTANCE_PC": 76000.0,
    "PA_DEG": 90.0,
    "AXIS_RATIO_Q": 0.70,
    "R_HALF_LIGHT_PC": 221.0,
    "R_MAX_STARS_PC": 1500.0,
    "INCLINATION_DEG": 78.0,
    # Systemic velocity from draco_walker2023.csv preparation.
    "V_SYS_KMS": -291.68214888089926,
    # =========================================================
    # Stellar tracer and light model
    # =========================================================
    "STELLAR_MODEL": {
        "type": "karl_light_grid",
        "grid_csv": str(PROFILE_ROOT/"draco_oden_kirchen2001_axisymmetric_light_grid_full.csv"),
        "Ltot": 2.7e5,
        "geometry": "axisymmetric_density_grid",
        "q_axis_ratio": 0.69,
        "R_cyl_col": "R_cyl_pc",
        "z_col": "z_pc",
        "nu_col": "nu_Lsun_pc3",
        "volume_col": "cell_volume_pc3",
        "luminosity_col": "cell_luminosity_Lsun",
        "force_softening_pc": 0.2,
        "force_nR": 96,
        "force_nZ": 96,
        "force_nphi": 32,
        "source": "Odenkirchen2001",
    },
    # =========================================================
    # Data harvesting and quality
    # =========================================================
    "RADIUS_DEG": 0.6,
    "RUWE_MAX": 1.4,
    "PAR_SNR_MIN": 5.0,
    # =========================================================
    # Data-column authority
    # =========================================================
    "STAR_R_COL": "r_pc",
    "STAR_V_COL": "vlos",
    "STAR_VERR_COL": "vlos_err",
    "RA_COL": "ra",
    "DEC_COL": "dec",
    "VLOS_COL": "vlos",
    # =========================================================
    # Observed products
    # =========================================================
    "SURFACE_BRIGHTNESS_CSV": str(PROFILE_ROOT/"draco_oden_kirchen2001_surface_brightness_profile.csv"),
    "KINEMATIC_BINS_CSV": str(PROFILE_ROOT/"draco_walker2023_kinematic_bins_20.csv"),
    "DATA_CSV": str(PROFILE_ROOT/"draco_walker2023.csv"),
    # =========================================================
    # Draco-specific numerical overrides
    # =========================================================
    "OBSERVABLES": {"KARL_MAXITER": 4000}, # this is different than the load_config.py default of 2000, which was too low for Draco
    "CHUNK_SIZE": 40,
    "CSV_FLUSH_INTERVAL": 10,
    "EVAL_TIMEOUT_S": 1200.0,
    "PEN_SPHERE_STRENGTH": 200,
    "MIN_DISTANCE": 1e-6,
    "MAX_DISTANCE": 5e3,
    "POTENTIAL_EXTENT": 10.0,
    # =========================================================
    # Paths and run identity
    # =========================================================
    **build_data_paths(PROFILE_ROOT),
    "DATA_CSV": str(PROFILE_ROOT/"draco_walker2023.csv"),
    "CSV_PATH": str(PROFILE_ROOT/"default"/"draco-m-f-i-n-chi.csv"),
}


"""
16JUL2026 Draco full_light reset

MODE              = karl
stellar model     = karl_light_grid
light inputs      = full Odenkirchen profile
kinematic inputs  = Walker binned
comparison tag    = full_light

The previous Draco runs used the matched-bin light products, so they are not
directly comparable to this setup. This run switches both the stellar light
grid and surface-brightness constraints to the full Odenkirchen profile while
keeping the Walker kinematic bins. Shared daemon settings are now supplied by
OSPM/load_config.py; remaining differences are intentionally Draco-specific.

draco_nonsingular_isothermal_full_light

draco_multi_full_iso_chi.csv
28JUL2026
"""