# OSPM_Config_Center — Segue1
# Karl-style Segue 1 config.
# Observational inputs:
#   1. Niederste-Ostholt et al. 2009 Fig. 7 digitized number-count tracer profile
#   2. Simon stellar LOS velocity sample
# The 3D light grid is derived from the number-count surface-density profile.
# It is not an additional observational data set.
# Shared solver, orbit-library, AI, deck, and runtime defaults are supplied by
# OSPM/load_config.py. This file contains only Segue 1-specific authority.

from pathlib import Path
from Data.Data_Prep.Data_Paths import build_data_paths
LOCAL_DEBUG = False  # True for local debugging, False for production runs
PROFILE_ROOT = Path(__file__).resolve().parent
if not PROFILE_ROOT.exists(): raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")

INITIAL_THETA = [18, 300, 2.4e6, 3]
FIXED_THETA = INITIAL_THETA.copy() if LOCAL_DEBUG else None

CONFIG = {
    # Local debug control
    "LOCAL_DEBUG": LOCAL_DEBUG, "FIXED_THETA": FIXED_THETA,
    # Halo model and parameterization
    "HALO_TYPE": "nonsingular_isothermal", "HALO_PARAMETERIZATION": "v0_rc", 
    "PARAMETER_NAMES": ["v0", "r_c", "MBH", "ML"], "INITIAL_THETA": INITIAL_THETA,
    # Restricted after the wide-core run allowed nearly halo-free, black-hole-dominated solutions.
    "THETA_BOUNDS": [
        (0.0, 25.0),        # v0, km/s; 2025 paper range
        (1.0, 500),         # r_c, pc; 2025 paper range
        (0.0, 4e6),         # MBH, Msun; extended above paper's 1.5e6 ceiling because χ² is still improving toward higher MBH in this code
        (0.2, 5.0),         # M/L; 2025 paper range
    ],
    # Galaxy geometry
    "RA0_DEG": 151.7667,
    "DEC0_DEG": 16.0819,
    "DISTANCE_PC": 23000.0,
    "PA_DEG": 90.0,
    "AXIS_RATIO_Q": 1.0,
    "R_HALF_LIGHT_PC": 29.4,
    "R_MAX_STARS_PC": 120.0,
    "INCLINATION_DEG": 90.0,
    "V_SYS_KMS": 208.419339,  # Systemic velocity from Segue1_Simon_stars_v2.csv preparation.
    # Stellar tracer and light model
    "TRACER_CONSTRAINT_MODE": "density_3d",  # Use "projected_light" to restore the current baseline.
    "STELLAR_MODEL": {
        "type": "karl_light_grid",
        "grid_csv": str(PROFILE_ROOT / "segue1_NO09_axisymmetric_light_grid_full.csv"),
        "tracer_grid_csv": str(PROFILE_ROOT / "segue1_NO09_axisymmetric_light_grid_abel_full.csv"),
        "Ltot": 340.0,
        "geometry": "axisymmetric_density_grid",
        "q_axis_ratio": 1.0,
        "R_cyl_col": "R_cyl_pc",
        "z_col": "z_pc",
        "nu_col": "nu_Lsun_pc3",
        "volume_col": "cell_volume_pc3",
        "luminosity_col": "cell_luminosity_Lsun",
        "force_softening_pc": 0.2,
        "force_nR": 96,
        "force_nZ": 96,
        "force_nphi": 32,
        "source": "Niederste-Ostholt2009_Fig7_digitized",
    },
    # Data harvesting and quality
    "RADIUS_DEG": 0.6,
    "RUWE_MAX": 1.4,
    "PAR_SNR_MIN": 5.0,
    # Observed products
    "SURFACE_BRIGHTNESS_CSV": str(PROFILE_ROOT / "segue1_NO09_surface_brightness_full.csv"),
    "KINEMATIC_BINS_CSV": str(PROFILE_ROOT / "segue1_simon_kinematic_bins_16.csv"),
    "DATA_CSV": str(PROFILE_ROOT / "Segue1_Simon_stars_v2.csv"),
    # Galaxy-scale numerical domain
    "MAX_DISTANCE": 2e3,
    "MBH_LOG_FLOOR": 1.0e3,
    "MBH_ZERO_FRACTION": 0.10,
    # Paths and run identity
    **build_data_paths(PROFILE_ROOT),
    "DATA_CSV": str(PROFILE_ROOT/"Segue1_Simon_stars_v2.csv"),
    "CSV_PATH": str(PROFILE_ROOT/"default"/"segue1-paper-bounds-expanded-mbh-phasevolume-try7-density3d-abel.csv"),
}