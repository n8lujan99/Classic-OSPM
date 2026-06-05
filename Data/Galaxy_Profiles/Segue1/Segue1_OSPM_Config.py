# OSPM_Config_Center — Segue1
# Only place that should have Galaxy-specific configuration variables
# so all other modules remain galaxy-agnostic

from pathlib import Path
import pathlib
import os
import multiprocessing as mp
from Data.Data_Prep.Data_Paths import build_data_paths

Galaxy = "Segue1"

# Run mode toggle
LOCAL_DEBUG = False   # True = local responsive run, False = full run / HPC-style

# Profile root (this directory contains default/, data.csv, etc.)
PROFILE_ROOT = Path(__file__).resolve().parent
if not PROFILE_ROOT.exists():
    raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")


def detect_workers():
    slurm = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm)
    return mp.cpu_count()

WORKERS = detect_workers()

# Mode-dependent knobs
NORBIT = 1000 if LOCAL_DEBUG else 2500
BATCH_SIZE = 40 if LOCAL_DEBUG else 90
MIN_BATCH_SIZE = 40 if LOCAL_DEBUG else 90
MAX_BATCH_SIZE = 120 if LOCAL_DEBUG else 270
CHUNK_SIZE = 30 if LOCAL_DEBUG else 90
LOG_INTERVAL = 1 if LOCAL_DEBUG else 10
PROF_EVERY = 2 if LOCAL_DEBUG else 20
EVAL_TIMEOUT_S = 20.0 if LOCAL_DEBUG else 600.0

CONFIG = {
    # =========================================================
    # Parallelization
    # =========================================================
    "N_WORKERS": WORKERS,

    # =========================================================
    # Identity
    # =========================================================
    "MODE":      "stellar",
    "GALAXY":    Galaxy,
    "HALO_TYPE": "nfw",

    # =========================================================
    # Galaxy geometry (declared, never fitted)
    # =========================================================
    "RA0_DEG":         151.7667,
    "DEC0_DEG":        16.0819,
    "DISTANCE_PC":     23000.0,
    "PA_DEG":          None,
    "AXIS_RATIO_Q":    1.0,
    "R_HALF_LIGHT_PC": 29.0,
    "R_MAX_STARS_PC":  120.0,
    "INCLINATION_DEG": 90.0,

    # Fixed stellar light model (Segue 1)
    "STELLAR_MODEL": {
        "type": "plummer",
        "Ltot": 340.0,
        "a_pc": 22.4,
    },

    # =========================================================
    # Data harvesting & quality
    # =========================================================
    "RADIUS_DEG":  0.6,
    "RUWE_MAX":    1.4,
    "PAR_SNR_MIN": 5.0,

    # Column authority (Segue 1 conventions)
    "STAR_R_COL":    "r_pc",
    "STAR_V_COL":    "vlos",
    "STAR_VERR_COL": "vlos_err",
    "RA_COL":        "ra",
    "DEC_COL":       "dec",
    "VLOS_COL":      "vlos",

    # =========================================================
    # OSPM numerical setup
    # =========================================================
    "NORBIT": NORBIT,
    "BINNING": {
        "MIN_BINS":         4,
        "N_TARGET_CIRC":    5,
        "MIN_PER_BIN_CIRC": 3,
    },
    "OBSERVABLES": {
        "NBINS_OCC":  6,
        "LAMBDA_OCC": 0.3,
    },

    # =========================================================
    # Parameter space
    # =========================================================
    "PARAMETER_NAMES": ["rho_s", "r_s", "MBH", "ML"],
    "INITIAL_THETA":   [0.1, 300.0, 5e5, 2.0],
    "THETA_BOUNDS": [
        (0.0, 5.0), # rho_s must be lower than 6 to avoid over massive halos that could look like smbhs
        (100, 5000.0), # r_s in pc 100 pc = 0.1 kpc the min used in lujan 2025
        (0.0, 2e6), # MBH solar masses 0 to 2 million
        (0.2, 1.6), # ML solar mass / solar luminosity 0.2 to 2 used in lujan 2025
    ],

    # Penalties
    "PEN_SPHERE_STRENGTH": 2500,
    "PEN_SPHERE_POWER":    2.0,
    "PEN_SLOPE_STRENGTH":  5000,

    # =========================================================
    # Physical domain (solver only)
    # =========================================================
    "MIN_DISTANCE":             5e-4,
    "MAX_DISTANCE":             2e3,
    "R_GRID_POINTS":            256,
    "POTENTIAL_EXTENT":         6.0,
    "BH_MIN_RADIUS_MULTIPLIER": 2.0,

    # =========================================================
    # Deck semantics
    # =========================================================
    "REQUIRE_COLUMNS": ["rho_s", "r_s", "MBH", "ML", "chi2", "reward", "status", "proposal_id"],
    "ALLOWED_STATUSES": ["todo", "seed", "pass", "orbit_fail", "numeric_fail", "unknown_fail", "forbidden"],
    "FILL_DEFAULT_STATUS": "todo",

    # =========================================================
    # Sampling & control
    # =========================================================
    "BATCH_SIZE":     BATCH_SIZE,
    "MIN_BATCH_SIZE": MIN_BATCH_SIZE,
    "MAX_BATCH_SIZE": MAX_BATCH_SIZE,
    "CHUNK_SIZE":     CHUNK_SIZE,
    "_PRINT_EVERY":   10,
    "_print_counter": 1,

    # =========================================================
    # AI / learning
    # =========================================================
    "AI_START_AFTER":       300,
    "MIN_TRAIN_POINTS":     300,
    "TRAIN_WINDOW":         500,
    "AI_NOISE_INIT":        0.30,
    "AI_NOISE_MIN":         0.02,
    "AI_NOISE_TAU":         5000,
    "AI_MIN_DISTINCT_PASS": 800,
    "RESET_INTERVAL":       10000,
    "AI_DEBUG_EVERY":       200,
    "AI_SNAPSHOT_EVERY":    2000,
    "FLAT_WINDOW":          200,
    "FLAT_THRESHOLD":       1e-6,
    "FLAT_PATIENCE":        10,
    "AI_RESET_ON_FLAT":     True,

    # =========================================================
    # Termination
    # =========================================================
    "MAX_RUNS":            100000,
    "STOP_NO_IMPROVEMENT": 1000,
    "IMPROVEMENT_EPSILON": 1e-6,
    "LOG_INTERVAL":        LOG_INTERVAL,
    "PROF_EVERY":          PROF_EVERY,
    "EVAL_TIMEOUT_S":      EVAL_TIMEOUT_S,

    # =========================================================
    # Paths (authoritative)
    # =========================================================
    **build_data_paths(PROFILE_ROOT),
    "CSV_PATH": str(PROFILE_ROOT / "default" / "daemon_deck_oldbounds.csv"),
}

print("[CONFIG] CSV_PATH =", CONFIG["CSV_PATH"])
print(f"[CONFIG] LOCAL_DEBUG={LOCAL_DEBUG} | NORBIT={CONFIG['NORBIT']} | BATCH={CONFIG['BATCH_SIZE']} | CHUNK={CONFIG['CHUNK_SIZE']}")
