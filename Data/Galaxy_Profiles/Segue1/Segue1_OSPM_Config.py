# OSPM_Config_Center — Segue1
# Karl-style Segue 1 config.
#
# Observational inputs:
#   1. Niederste-Ostholt et al. 2009 Fig. 7 digitized number-count tracer profile
#   2. Simon stellar LOS velocity sample
#
# The 3D light grid is derived from the number-count surface-density profile.
# It is not an additional observational data set.
#
# Shared solver, orbit-library, AI, deck, and runtime defaults are supplied by
# OSPM/load_config.py. This file contains only Segue 1-specific authority.

from pathlib import Path
from Data.Data_Prep.Data_Paths import build_data_paths

LOCAL_DEBUG = False # True for local debugging, False for production runs

PROFILE_ROOT = Path(__file__).resolve().parent
if not PROFILE_ROOT.exists():
    raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")

INITIAL_THETA = [21.0, 100.0, 4.5e5, 0.3]
FIXED_THETA = INITIAL_THETA.copy() if LOCAL_DEBUG else None

CONFIG = {
    # =========================================================
    # Local debug control
    # =========================================================
    "LOCAL_DEBUG": LOCAL_DEBUG,
    "FIXED_THETA": FIXED_THETA,

    # =========================================================
    # Halo model and parameterization
    # =========================================================
    "HALO_TYPE": "nonsingular_isothermal",
    "HALO_PARAMETERIZATION": "v0_rc",
    "PARAMETER_NAMES": ["v0", "r_c", "MBH", "ML"],
    "INITIAL_THETA": INITIAL_THETA,

    # Restricted after the wide-core run allowed nearly halo-free,
    # black-hole-dominated solutions.
    "THETA_BOUNDS": [
        (0.0, 30.0),        # v0, km/s; 2025 paper range
        (1.0, 10000.0),     # r_c, pc; 2025 paper range
        (0.0, 2.5e6),       # MBH, Msun; extended above paper's 1.5e6 ceiling because χ² is still improving toward higher MBH in this code
        (0.2, 1.6),         # M/L; 2025 paper range
    ],

    # =========================================================
    # Galaxy geometry
    # =========================================================
    "RA0_DEG": 151.7667,
    "DEC0_DEG": 16.0819,
    "DISTANCE_PC": 23000.0,
    "PA_DEG": 90.0,
    "AXIS_RATIO_Q": 1.0,
    "R_HALF_LIGHT_PC": 29.4,
    "R_MAX_STARS_PC": 120.0,
    "INCLINATION_DEG": 90.0,

    # Systemic velocity from Segue1_Simon_stars_v2.csv preparation.
    "V_SYS_KMS": 208.419339,

    # =========================================================
    # Stellar tracer and light model
    # =========================================================
    "STELLAR_MODEL": {
        "type": "karl_light_grid",
        "grid_csv": str(PROFILE_ROOT / "segue1_NO09_axisymmetric_light_grid_full.csv"),
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

    # =========================================================
    # Data harvesting and quality
    # =========================================================
    "RADIUS_DEG": 0.6,
    "RUWE_MAX": 1.4,
    "PAR_SNR_MIN": 5.0,

    # =========================================================
    # Observed products
    # =========================================================
    "SURFACE_BRIGHTNESS_CSV": str(PROFILE_ROOT / "segue1_NO09_surface_brightness_full.csv"),
    "KINEMATIC_BINS_CSV": str(PROFILE_ROOT / "segue1_simon_kinematic_bins_16.csv"),
    "DATA_CSV": str(PROFILE_ROOT / "Segue1_Simon_stars_v2.csv"),

    # =========================================================
    # Galaxy-scale numerical domain
    # =========================================================
    "MAX_DISTANCE": 2e3,

    # =========================================================
    # Paths and run identity
    # =========================================================
    **build_data_paths(PROFILE_ROOT),
    "DATA_CSV": str(PROFILE_ROOT/"Segue1_Simon_stars_v2.csv"),
    "CSV_PATH": str(PROFILE_ROOT/"default"/"segue1-paper-bounds-expanded-mbh-phasevolume.csv"),
}

# Just some notes on the Segue 1 runs and analysis.
# These are not used by the code, but are here for reference.

"""
16JUL2026 run full_light
MODE              = karl
stellar model     = karl_light_grid
light inputs      = full
kinematic inputs  = binned
comparison tag    = full_light

17JUL2026 analysis of 16JUL2026 run full_light
Run shows that we are not giving v_circ a large enough range and it is running
into a wall. At the same time, MBH is beginning to become distinguished.
M/L is degenerate as expected, and r_s also appears degenerate, so no changes
were made to those two. Expanded the v_circ range from 0–30 to 0–80 km/s.

17JUL2026 run full_light
Extended the 16JUL2026 full_light run to 300,000 runs with v_circ expanded
to 0–80 km/s.

22JUL2026 analysis of 17JUL2026 run full_light
Finished the current full_light runs for Segue 1 and Draco using the NFW halo.
Segue 1 roughly recovered the previously identified SMBH result, although the
final constraint depended on the completed landscape. Draco began forming an
SMBH basin, but the result remained uncertain.

22JUL2026 run nonsingular_isothermal full_light
MODE                   = karl
stellar model          = karl_light_grid
light inputs           = full
kinematic inputs       = binned
halo type              = nonsingular_isothermal
halo parameterization  = v0_rc
halo parameters        = v0, r_c
comparison tag         = nonsingular_isothermal_full_light

Started cored-halo runs for Segue 1 and Draco to test whether the emerging SMBH
results survive a change in halo profile.

05AUG2026 analysis of nonsingular_isothermal full_light
The Segue 1 run preferred an SMBH near 1.8–2.0 million Msun while placing the
halo core radius at tens of kiloparsecs. That parameter-space corner made the
halo effectively negligible across the observed galaxy. Restricted r_c and M/L
to test whether the SMBH still grows once that nearly halo-free escape is closed.
"""