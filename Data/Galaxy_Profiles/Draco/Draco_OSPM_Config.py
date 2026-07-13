# OSPM_Config_Center — Draco
# Local development config for Karl-style Draco OSPM.

from pathlib import Path
import os
import multiprocessing as mp
from Data.Data_Prep.Data_Paths import build_data_paths

Galaxy = "Draco"
LOCAL_DEBUG = False

PROFILE_ROOT = Path(__file__).resolve().parent
if not PROFILE_ROOT.exists():
    raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")

def detect_workers():
    slurm = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm)
    return mp.cpu_count()

WORKERS = detect_workers()

NORBIT = 1000 if LOCAL_DEBUG else 10000
BATCH_SIZE = 1 if LOCAL_DEBUG else 120
MIN_BATCH_SIZE = 1 if LOCAL_DEBUG else 120
MAX_BATCH_SIZE = 1 if LOCAL_DEBUG else 360
CHUNK_SIZE = 1 if LOCAL_DEBUG else 60
LOG_INTERVAL = 1 if LOCAL_DEBUG else 10
PROF_EVERY = 1 if LOCAL_DEBUG else 20
EVAL_TIMEOUT_S = 200.0 if LOCAL_DEBUG else 600.0
MAX_RUNS = 1 if LOCAL_DEBUG else 100000

if NORBIT % 2 != 0:
    raise ValueError( f"Karl paired-orbit Spherical path requires even NORBIT because NORBIT is the final A-matrix column count; got {NORBIT}" )

CONFIG = {
    "N_WORKERS": WORKERS,

    "MODE":        "karl",
    "GALAXY":      Galaxy,
    "HALO_TYPE":   "nfw",
    "HALO_PARAMETERIZATION": "vcirc_rs",
    "RA0_DEG":          260.0517,
    "DEC0_DEG":         57.9153,
    "DISTANCE_PC":      76000.0,
    "PA_DEG":           90.0,
    "AXIS_RATIO_Q":     0.70,
    "R_HALF_LIGHT_PC":  221.0,
    "R_MAX_STARS_PC":   1500.0,
    "VLOS_COL":        "vlos",
    "V_SYS_KMS":       -291.68214888089926, 

    "STELLAR_MODEL": {
        "type": "karl_light_grid",
        "grid_csv": str(PROFILE_ROOT / "draco_oden_kirchen2001_axisymmetric_light_grid.csv"),
        "Ltot": 2.7e5,
        "geometry": "axisymmetric_density_grid", # only other option is "spherical_enclosed_light_grid" axi is for flat
        "q_axis_ratio": 0.69,
        "R_cyl_col": "R_cyl_pc",
        "z_col": "z_pc",
        "nu_col": "nu_Lsun_pc3",
        "volume_col": "cell_volume_pc3",
        "luminosity_col": "cell_luminosity_Lsun",
        "force_softening_pc": 0.5,
        "force_nR": 96,
        "force_nZ": 96,
        "force_nphi": 32,
        "source": "Odenkirchen2001",
    },

    "INCLINATION_DEG":  78.0,
    "RADIUS_DEG":   0.6,
    "RUWE_MAX":     1.4,
    "PAR_SNR_MIN":  5.0,
    "STAR_R_COL":      "r_pc",
    "STAR_V_COL":      "vlos",
    "STAR_VERR_COL":   "vlos_err",
    "RA_COL":          "ra",
    "DEC_COL":         "dec",

    "SURFACE_BRIGHTNESS_CSV": str(PROFILE_ROOT / "draco_oden_kirchen2001_surface_brightness_on_walker_bins_20.csv"),
    "KINEMATIC_BINS_CSV":     str(PROFILE_ROOT / "draco_walker2023_kinematic_bins_20.csv"),

    "NORBIT": NORBIT,

    "OBSERVABLES": {
        "NVBIN": 21,
        "MIN_STARS_PER_BIN": 20,
        "LAMBDA_LIGHT": 0.3,
        "NTHETA_LAUNCH": 9,

        # Live Karl weight/scoring path.
        "WEIGHT_MODE": "entropy",
        "WEIGHT_SOLVER": "expanded_cm",
        "LOSVD_SCORE_MODE": "standard",
        "KARL_ALPHAT": 1.0,
        "KARL_MAXITER": 60,
        "ENTROPY_FLOOR": 1e-12,

        # Halo flattening used by the halo force path.
        # Stellar flattening stays in STELLAR_MODEL["q_axis_ratio"].
        "HALO_Q_AXIS_RATIO": 1.0,
    },

    "PARAMETER_NAMES": ["vcirc", "r_s", "MBH", "ML"],
    "INITIAL_THETA": [183.9114355853, 1800.0, 9e5, 1.0],
    "THETA_BOUNDS": [(0.0, 250.0), (100.0, 10000.0), (0.0, 1e6), (0.2, 2.0)],
    "PEN_SPHERE_STRENGTH": 200,
    "PEN_SPHERE_POWER":    2.0,
    "PEN_SLOPE_STRENGTH":  5000,
    "MIN_DISTANCE":             1e-6,
    "MAX_DISTANCE":             5e3,
    "R_GRID_POINTS":            256,
    "POTENTIAL_EXTENT":         10.0,
    "BH_MIN_RADIUS_MULTIPLIER": 2.0,

    "REQUIRE_COLUMNS": [ "vcirc", "r_s", "MBH", "ML", "chi2", "reward", "status", "proposal_id", "refine_passes", "chi2_losvd", "chi2_light", 
        "chi2_total", "chi2_inner", "chi2_outer", "N_inner", "N_outer", "N_nonzero_weights", "effective_N_orbits", "max_weight_fraction", "halo_type",
        # Runtime contract diagnostics:
        "weight_mode", "weight_solver_mode", "losvd_score_mode", "alphat", "halo_q_axis_ratio", "karl_halo_params_active",
    ],

    "ALLOWED_STATUSES": [
        "todo", "seed", "pass", "orbit_fail", "numeric_fail", "unknown_fail", "timeout", "forbidden", "pass_full", "pass_bh_only", "pass_halo_only",
        "pass_bh_up", "pass_bh_down", "pass_halo_up", "pass_halo_down", "pass_ml_up", "pass_ml_down", "orbit_fail_full", "orbit_fail_bh_only", "orbit_fail_halo_only",
        "numeric_fail_full", "numeric_fail_bh_only", "numeric_fail_halo_only", "timeout_full", "timeout_bh_only", "timeout_halo_only", "timeout_bh_up", "timeout_bh_down",
        "timeout_halo_up", "timeout_halo_down", "timeout_ml_up", "timeout_ml_down", "unknown_fail_full", "unknown_fail_bh_only", "unknown_fail_halo_only", "unknown_fail_bh_up", "unknown_fail_bh_down",
        "unknown_fail_halo_up", "unknown_fail_halo_down", "unknown_fail_ml_up", "unknown_fail_ml_down", "numeric_fail_bh_up", "numeric_fail_bh_down", "numeric_fail_halo_up", "numeric_fail_halo_down",
        "numeric_fail_ml_up", "numeric_fail_ml_down", "orbit_fail_bh_up", "orbit_fail_bh_down", "orbit_fail_halo_up", "orbit_fail_halo_down", "orbit_fail_ml_up", "orbit_fail_ml_down",
    ],
    "FILL_DEFAULT_STATUS": "todo",

    "BATCH_SIZE":          BATCH_SIZE,
    "MIN_BATCH_SIZE":      MIN_BATCH_SIZE,
    "MAX_BATCH_SIZE":      MAX_BATCH_SIZE,
    "CHUNK_SIZE":          CHUNK_SIZE,
    "_PRINT_EVERY":        10,
    "_print_counter":      0,

    "AI_START_AFTER":        100, # normally closer to 500 for initial seeding but reduced to 100 for local debug
    "MIN_TRAIN_POINTS":      300,
    "TRAIN_WINDOW":          3000,
    "AI_NOISE_INIT":         0.30,
    "AI_NOISE_MIN":          0.02,
    "AI_NOISE_TAU":          8000,
    "AI_MIN_DISTINCT_PASS":  500,
    "RESET_INTERVAL":        10000,
    "AI_DEBUG_EVERY":        200,
    "AI_SNAPSHOT_EVERY":     2000,
    "FLAT_WINDOW":           300,
    "FLAT_THRESHOLD":        1e-6,
    "FLAT_PATIENCE":         5,
    "AI_RESET_ON_FLAT":      False,

    "MAX_RUNS":              MAX_RUNS,
    "STOP_NO_IMPROVEMENT":   5000,
    "IMPROVEMENT_EPSILON":   1e-6,
    "LOG_INTERVAL":          LOG_INTERVAL,
    "PROF_EVERY":            PROF_EVERY,
    "EVAL_TIMEOUT_S":        EVAL_TIMEOUT_S,

    "G":    6.67430e-11,
    "Msun": 1.98847e30,

    **build_data_paths(PROFILE_ROOT),
    "DATA_CSV": str(PROFILE_ROOT / "draco_walker2023.csv"),
    "CSV_PATH": str(PROFILE_ROOT / "default" / "daemon_deck_karl_draco_vcirc.csv"),
}

print("[CONFIG] CSV_PATH =", CONFIG["CSV_PATH"])
print("[CONFIG] LOCAL_DEBUG =", LOCAL_DEBUG)
print("[CONFIG] NORBIT =", CONFIG["NORBIT"])
print("[CONFIG] MAX_RUNS =", CONFIG["MAX_RUNS"])
print("[CONFIG] BATCH_SIZE =", CONFIG["BATCH_SIZE"])
print("[CONFIG] CHUNK_SIZE =", CONFIG["CHUNK_SIZE"])
print("[CONFIG] HALO_PARAMETERIZATION =", CONFIG["HALO_PARAMETERIZATION"])
print("[CONFIG] PARAMETER_NAMES =", CONFIG["PARAMETER_NAMES"])
print("[CONFIG] THETA_BOUNDS =", CONFIG["THETA_BOUNDS"])
print("[CONFIG] STELLAR_GEOMETRY =", CONFIG["STELLAR_MODEL"]["geometry"])
print("[CONFIG] NTHETA_LAUNCH =", CONFIG["OBSERVABLES"]["NTHETA_LAUNCH"])
print("[CONFIG] WEIGHT_MODE =", CONFIG["OBSERVABLES"]["WEIGHT_MODE"])
print("[CONFIG] WEIGHT_SOLVER =", CONFIG["OBSERVABLES"]["WEIGHT_SOLVER"])
print("[CONFIG] LOSVD_SCORE_MODE =", CONFIG["OBSERVABLES"]["LOSVD_SCORE_MODE"])
