# OSPM_Config_Center — Draco
# Galaxy-specific configuration only.
# Shared solver, orbit-library, AI, deck, and runtime defaults come from OSPM/load_config.py.

from pathlib import Path
from Data.Data_Prep.Data_Paths import build_data_paths

LOCAL_DEBUG = True

PROFILE_ROOT = Path(__file__).resolve().parent
if not PROFILE_ROOT.exists():
    raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")

INITIAL_THETA = [100.0, 1800.0, 9.0e5, 1.0]
FIXED_THETA = INITIAL_THETA.copy() if LOCAL_DEBUG else None

CONFIG = {
    "LOCAL_DEBUG": LOCAL_DEBUG, "FIXED_THETA": FIXED_THETA,
    # Halo and search space
    "HALO_TYPE": "nonsingular_isothermal",
    "HALO_PARAMETERIZATION": "v0_rc",
    "PARAMETER_NAMES": ["v0", "r_c", "MBH", "ML"],
    "INITIAL_THETA": INITIAL_THETA,
    "THETA_BOUNDS": [
        (0.0, 200.0),
        (1.0, 5000.0),
        (0.0, 5.0e6),
        (0.2, 20.0),
    ],
    # Galaxy geometry
    "RA0_DEG": 260.0517,
    "DEC0_DEG": 57.9153,
    "DISTANCE_PC": 76000.0,
    "PA_DEG": 90.0,
    "AXIS_RATIO_Q": 0.70,
    "R_HALF_LIGHT_PC": 221.0,
    "R_MAX_STARS_PC": 1500.0,
    "INCLINATION_DEG": 78.0,
    "V_SYS_KMS": -291.68214888089926,
    # Intrinsic tracer constraint.
    # Keep the old grid for the stellar force so this isolates the tracer change,
    # exactly as in the Segue 1 density_3d test.
    "TRACER_CONSTRAINT_MODE": "density_3d",
    "STELLAR_MODEL": {
        "type": "karl_light_grid",
        "grid_csv": str(PROFILE_ROOT / "draco_oden_kirchen2001_axisymmetric_light_grid_full.csv"),
        "tracer_grid_csv": str(PROFILE_ROOT / "draco_oden_kirchen2001_axisymmetric_light_grid_abel_full.csv"),
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
    # Paths
    **build_data_paths(PROFILE_ROOT),

    # Draco data contract
    "STAR_R_COL": "r_pc",
    "STAR_V_COL": "vlos",
    "STAR_VERR_COL": "vlos_err",
    "RA_COL": "ra",
    "DEC_COL": "dec",
    "VLOS_COL": "vlos",

    # Observed products
    "SURFACE_BRIGHTNESS_CSV": str(PROFILE_ROOT / "draco_oden_kirchen2001_surface_brightness_profile.csv"),
    "KINEMATIC_BINS_CSV": str(PROFILE_ROOT / "draco_walker2023_kinematic_bins_20.csv"),
    "DATA_CSV": str(PROFILE_ROOT / "draco_walker2023.csv"),

    # Draco needs the longer weight solve.
    "OBSERVABLES": {"KARL_MAXITER": 4000},

    # Draco numerical domain
    "MIN_DISTANCE": 1e-6,
    "MAX_DISTANCE": 5e3,
    "POTENTIAL_EXTENT": 10.0,

    # Draco search behavior
    "MBH_LOG_FLOOR": 1.0e3,
    "MBH_ZERO_FRACTION": 0.10,

    # Draco-specific runtime overrides
    "CHUNK_SIZE": 40,
    "CSV_FLUSH_INTERVAL": 10,
    "EVAL_TIMEOUT_S": 1200.0,
    "PEN_SPHERE_STRENGTH": 200,

    # Run identity
    "CSV_PATH": str(PROFILE_ROOT / "default" / "draco-try2-density3d-abel.csv"),
    }