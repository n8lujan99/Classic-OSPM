# OSPM_Config_Center — Carina
# Karl-style Carina config.
# Observational inputs will be supplied once the Carina tracer and LOSVD products are finalized.
# The 3D light grid will be derived from the observed surface-density profile.
# It is not an additional observational data set.
# Shared solver, orbit-library, AI, deck, and runtime defaults are supplied by
# OSPM/load_config.py. This file contains only Carina-specific authority.

from pathlib import Path
from Data.Data_Prep.Data_Paths import build_data_paths

LOCAL_DEBUG = False  # True for local debugging, False for production runs

PROFILE_ROOT = Path(__file__).resolve().parent
if not PROFILE_ROOT.exists(): raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")

INITIAL_THETA = [100.0, 1800.0, 9.0e5, 1.0]
FIXED_THETA = INITIAL_THETA.copy() if LOCAL_DEBUG else None

CONFIG = {
    # Local debug control
    "LOCAL_DEBUG": LOCAL_DEBUG, "FIXED_THETA": FIXED_THETA,

    # Halo model and parameterization
    "HALO_TYPE": "nonsingular_isothermal", "HALO_PARAMETERIZATION": "v0_rc",
    "PARAMETER_NAMES": ["v0", "r_c", "MBH", "ML"], "INITIAL_THETA": INITIAL_THETA,
    "THETA_BOUNDS": [
        (0.0, 200.0),       # v0, km/s
        (1.0, 5000.0),      # r_c, pc
        (0.0, 5.0e6),       # MBH, Msun
        (0.2, 20.0),        # M/L
    ],

    # Galaxy geometry
    "RA0_DEG": 100.4029,
    "DEC0_DEG": -50.9661,
    "DISTANCE_PC": 105000.0,
    "PA_DEG": 65.0,
    "AXIS_RATIO_Q": 1.0,
    "R_HALF_LIGHT_PC": 250.0,
    "R_MAX_STARS_PC": 1250.0,
    "INCLINATION_DEG": 90.0,
    "V_SYS_KMS": None,  # TODO: set from prepared Carina velocity catalog.

    # Stellar tracer and light model
    "TRACER_CONSTRAINT_MODE": "density_3d",
    "STELLAR_MODEL": {
        "type": "karl_light_grid",
        "grid_csv": str(PROFILE_ROOT / "carina_axisymmetric_light_grid_full.csv"),
        "tracer_grid_csv": str(PROFILE_ROOT / "carina_axisymmetric_light_grid_abel_full.csv"),
        "Ltot": 2.7e5,  # TODO: replace with adopted Carina luminosity.
        "geometry": "axisymmetric_density_grid",
        "q_axis_ratio": 1.0,  # TODO: replace with adopted intrinsic tracer flattening.
        "R_cyl_col": "R_cyl_pc",
        "z_col": "z_pc",
        "nu_col": "nu_Lsun_pc3",
        "volume_col": "cell_volume_pc3",
        "luminosity_col": "cell_luminosity_Lsun",
        "force_softening_pc": 0.2,
        "force_nR": 96,
        "force_nZ": 96,
        "force_nphi": 32,
        "source": "Carina_surface_brightness_TBD",
    },

    # Data harvesting and quality
    "RADIUS_DEG": 0.6,
    "RUWE_MAX": 1.4,
    "PAR_SNR_MIN": 5.0,

    # Data-column authority
    "STAR_R_COL": "r_pc",
    "STAR_V_COL": "vlos",
    "STAR_VERR_COL": "vlos_err",
    "RA_COL": "ra",
    "DEC_COL": "dec",
    "VLOS_COL": "vlos",

    # Observed products
    "SURFACE_BRIGHTNESS_CSV": str(PROFILE_ROOT / "carina_surface_brightness_profile.csv"),
    "KINEMATIC_BINS_CSV": str(PROFILE_ROOT / "carina_kinematic_bins.csv"),
    "DATA_CSV": str(PROFILE_ROOT / "carina_stars.csv"),

    # Galaxy-scale numerical domain
    "MAX_DISTANCE": 5e3,
    "MBH_LOG_FLOOR": 1.0e3,
    "MBH_ZERO_FRACTION": 0.10,

    # Paths and run identity
    **build_data_paths(PROFILE_ROOT),
    "CSV_PATH": str(PROFILE_ROOT / "default" / "carina-try1-density3d-abel.csv"),
}