# OSPM_Config_Center — Segue1
# Karl-style Segue 1 config.
# Observational inputs:
#   1. Niederste-Ostholt et al. 2009 Fig. 7 digitized number-count tracer profile
#   2. Simon stellar LOS velocity sample
#
# The 3D light grid is derived from the number-count surface-density profile.
# It is not an additional observational data set.

from pathlib import Path
import os
import multiprocessing as mp
from Data.Data_Prep.Data_Paths import build_data_paths

Galaxy = "Segue1"
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
MAX_RUNS = 1 if LOCAL_DEBUG else 300000
FIXED_THETA = [21.0, 1000.0, 4.5e5, 0.3] if LOCAL_DEBUG else None
EVAL_VARIANTS = ["full"] if LOCAL_DEBUG else None

if NORBIT % 2 != 0:
    raise ValueError( f"Karl paired-orbit path requires even NORBIT; got {NORBIT}" )

CONFIG = {
    # =========================================================
    # Parallelization
    # =========================================================
    "N_WORKERS": WORKERS,

    # =========================================================
    # Identity
    # =========================================================
    "MODE":      "karl",
    "GALAXY":    Galaxy,
    "HALO_TYPE": "nonsingular_isothermal", #  a few options "nonsingular_isothermal" and "NFW"
    "HALO_PARAMETERIZATION": "v0_rc", # two options: "v0_rc" or "vcirc_rs" which are cored and nfw respectively 

    # =========================================================
    # Galaxy geometry
    # =========================================================
    "RA0_DEG":         151.7667,
    "DEC0_DEG":        16.0819,
    "DISTANCE_PC":     23000.0,
    "PA_DEG":          90.0,
    "AXIS_RATIO_Q":    1.0,
    "R_HALF_LIGHT_PC": 29.4,
    "R_MAX_STARS_PC":  120.0,
    "INCLINATION_DEG": 90.0,

    # Systemic velocity from Segue1_Simon_stars_v2.csv preparation.
    "V_SYS_KMS":       208.419339,

    # =========================================================
    # Stellar tracer/light model
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
    "RADIUS_DEG":  0.6,
    "RUWE_MAX":    1.4,
    "PAR_SNR_MIN": 5.0,

    # =========================================================
    # Column authority
    # =========================================================
    "STAR_R_COL":      "R_pc",
    "STAR_V_COL":      "vlos_kms",
    "STAR_VERR_COL":   "verr_kms",
    "RA_COL":          "ra_deg",
    "DEC_COL":         "dec_deg",
    "VLOS_COL":        "vlos_kms",

    # =========================================================
    # Draco-style observed products
    # =========================================================
    "SURFACE_BRIGHTNESS_CSV": str( PROFILE_ROOT / "segue1_NO09_surface_brightness_full.csv" ),
    "KINEMATIC_BINS_CSV": str( PROFILE_ROOT / "segue1_simon_kinematic_bins_16.csv" ),

    # =========================================================
    # OSPM numerical setup
    # =========================================================
    "NORBIT": NORBIT,

    "OBSERVABLES": {
        "NVBIN": 21,
        "NTHETA_LAUNCH": 9,

        # Orbit-library coverage and scheduler
        "ORBIT_FILL_PCT": 0.85,
        "ORBIT_REGIONAL_FLOOR": 0.80,
        "ORBIT_MAX_REGIONAL_GAP": 0.10,
        "ORBIT_SHELL_BANDS": 8,
        "ORBIT_COVERAGE_CHECK_EVERY": 50,

        # Warning classification; warned libraries still reach weights
        "ORBIT_WARN_FILL_PCT": 0.95,
        "ORBIT_WARN_SUCCESS_PCT": 0.99,
        "ORBIT_WARN_REGIONAL_FLOOR": 0.80,
        "ORBIT_WARN_MAX_REGIONAL_GAP": 0.15,

        # 0 automatically reserves about one-third of Julia threads as helpers
        "MODEL_OWNER_LIMIT": 0,

        # Expanded-CM solver and hard light-constraint convergence.
        "KARL_ALPHAT": 1.0,
        "KARL_LIGHT_REL_TOL": 0.01,
        "KARL_DELTA_CHI2_ITER_TOL": 0.3,
        "KARL_MAXITER": 1000,
        "ENTROPY_FLOOR": 1e-12,

        # Halo flattening used by the halo force path.
        # Stellar flattening stays in STELLAR_MODEL["q_axis_ratio"].
        "HALO_Q_AXIS_RATIO": 1.0,
    },

    # =========================================================
    # Parameter space
    # =========================================================
    "PARAMETER_NAMES": ["v0", "r_c", "MBH", "ML"],
    "INITIAL_THETA": [21.0, 1000.0, 4.5e5, 0.3],
    "FIXED_THETA": FIXED_THETA,
    "EVAL_VARIANTS": EVAL_VARIANTS,
    "THETA_BOUNDS": [
        (0.0, 35.0),         # v0, km/s
        (100.0, 100000.0),   # r_c, pc
        (0.0, 2e6),          # MBH, Msun
        (0.2, 20.0),         # M/L
    ],

    # =========================================================
    # Penalties
    # =========================================================
    "PEN_SPHERE_STRENGTH": 2500,
    "PEN_SPHERE_POWER":    2.0,
    "PEN_SLOPE_STRENGTH":  5000,

    # =========================================================
    # Physical domain
    # =========================================================
    "MIN_DISTANCE":             5e-4,
    "MAX_DISTANCE":             2e3,
    "R_GRID_POINTS":            256,
    "POTENTIAL_EXTENT":         6.0,
    "BH_MIN_RADIUS_MULTIPLIER": 2.0,

    # =========================================================
    # Deck semantics
    # =========================================================
    "REQUIRE_COLUMNS": [
        "v0", "r_c", "MBH", "ML", "chi2", "reward", "status", "proposal_id",
        # Scientific score and direct solver diagnostics:
        "chi2_losvd",
        "delta_chi2_iteration",
        "max_light_relative_residual",
        "light_constraint_ok",
        "solver_converged",
        "solver_iterations",
        "solver_failure_reason",
        "julia_status_code",
        # Radial and orbit-weight diagnostics:
        "chi2_inner",
        "chi2_outer",
        "N_inner",
        "N_outer",
        "N_nonzero_weights",
        "effective_N_orbits",
        "max_weight_fraction",
        # Runtime contract diagnostics:
        "halo_type",
        "alphat",
        "light_rel_tol",
        "delta_chi2_iter_tol",
        "halo_q_axis_ratio",
        "karl_halo_params_active",
        # Orbit-library coverage diagnostics:
        "coverage_status",
        "coverage_strict",
        "coverage_issue_region",
        "coverage_issue_axis",
        "coverage_issue_shell_bands",
        "coverage_reasons",
        "coverage_fraction",
        "coverage_attempted_fraction",
        "coverage_success_fraction",
        "coverage_shell_min",
        "coverage_lfrac_min",
        "coverage_theta_min",
        "coverage_shell_gap",
        "coverage_lfrac_gap",
        "coverage_theta_gap",
        "coverage_joint_holes",
        "coverage_deadline_hit",
        "successful_base_orbits",
        "planned_base_orbits",
    ],

    "ALLOWED_STATUSES": [
        "todo", "seed", "pass", "orbit_fail", "numeric_fail", "unknown_fail", "timeout", "forbidden", "pass_full", "pass_bh_only", "pass_halo_only",
        "pass_bh_up", "pass_bh_down", "pass_halo_up", "pass_halo_down", "pass_ml_up", "pass_ml_down", "orbit_fail_full", "orbit_fail_bh_only", "orbit_fail_halo_only",
        "numeric_fail_full", "numeric_fail_bh_only", "numeric_fail_halo_only", "timeout_full", "timeout_bh_only", "timeout_halo_only", "timeout_bh_up", "timeout_bh_down",
        "timeout_halo_up", "timeout_halo_down", "timeout_ml_up", "timeout_ml_down", "unknown_fail_full", "unknown_fail_bh_only", "unknown_fail_halo_only", "unknown_fail_bh_up", "unknown_fail_bh_down",
        "unknown_fail_halo_up", "unknown_fail_halo_down", "unknown_fail_ml_up", "unknown_fail_ml_down", "numeric_fail_bh_up", "numeric_fail_bh_down", "numeric_fail_halo_up", "numeric_fail_halo_down",
        "numeric_fail_ml_up", "numeric_fail_ml_down", "orbit_fail_bh_up", "orbit_fail_bh_down", "orbit_fail_halo_up", "orbit_fail_halo_down", "orbit_fail_ml_up", "orbit_fail_ml_down",
        "solver_failed_full", "solver_failed_bh_only", "solver_failed_halo_only", "solver_failed_bh_up", "solver_failed_bh_down", "solver_failed_halo_up", "solver_failed_halo_down",
        "solver_failed_ml_up", "solver_failed_ml_down", "physics_exception_full", "physics_exception_bh_only", "physics_exception_halo_only", "physics_exception_bh_up",
        "physics_exception_bh_down", "physics_exception_halo_up", "physics_exception_halo_down", "physics_exception_ml_up", "physics_exception_ml_down",
    ],

    "FILL_DEFAULT_STATUS": "todo",

    # =========================================================
    # Sampling and control
    # =========================================================
    "BATCH_SIZE":          BATCH_SIZE,
    "MIN_BATCH_SIZE":      MIN_BATCH_SIZE,
    "MAX_BATCH_SIZE":      MAX_BATCH_SIZE,
    "CHUNK_SIZE":          CHUNK_SIZE,
    "_PRINT_EVERY":        10,
    "_print_counter":      1,

    # =========================================================
    # AI / learning
    # =========================================================
    "AI_START_AFTER":       500,
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
    "MAX_RUNS":            MAX_RUNS,
    "STOP_NO_IMPROVEMENT": 2000,
    "IMPROVEMENT_EPSILON": 1e-6,
    "LOG_INTERVAL":        LOG_INTERVAL,
    "PROF_EVERY":          PROF_EVERY,
    "EVAL_TIMEOUT_S":      EVAL_TIMEOUT_S,

    # =========================================================
    # Physical constants
    # =========================================================
    "G":    6.67430e-11,
    "Msun": 1.98847e30,

    # =========================================================
    # Paths
    # =========================================================
    **build_data_paths(PROFILE_ROOT),
    "DATA_CSV": str(PROFILE_ROOT / "Segue1_Simon_stars_v2.csv"),
    "COMPARISON_TAG": "nonsingular_isothermal_full_light",
    "CSV_PATH": str(PROFILE_ROOT / "default" / "segue1-m-f-i-n-chi.csv"),
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
print("[CONFIG] FIXED_THETA =", CONFIG["FIXED_THETA"])
print("[CONFIG] EVAL_VARIANTS =", CONFIG["EVAL_VARIANTS"])
print("[CONFIG] STELLAR_GEOMETRY =", CONFIG["STELLAR_MODEL"]["geometry"])
print("[CONFIG] NTHETA_LAUNCH =", CONFIG["OBSERVABLES"]["NTHETA_LAUNCH"])
print("[CONFIG] KARL_ALPHAT =", CONFIG["OBSERVABLES"]["KARL_ALPHAT"])
print("[CONFIG] KARL_LIGHT_REL_TOL =", CONFIG["OBSERVABLES"]["KARL_LIGHT_REL_TOL"])
print("[CONFIG] KARL_DELTA_CHI2_ITER_TOL =", CONFIG["OBSERVABLES"]["KARL_DELTA_CHI2_ITER_TOL"])


"""
16JUL2026 run full_light
MODE              = karl
stellar model     = karl_light_grid
light inputs      = full
kinematic inputs  = binned
comparison tag    = full_light

17JUL2026 analysis of 16JUL2026 run full_light
Run shows that we are not giving v_circ a large enough range and its running into a wall
while at the same time the mbh is begining to become distinguished the M/L is degenerate like expected and the r_s 
looks like its going degenearate as weell so no changes for those two, and mbh should still be in proper range. 
Going to expand the v_circ range from 0-30 to 0-80 and see if that helps.

17JUL2026 run full_light
Extending the 16JUL2026 full_light run to 300,000 runs and expanding the v_circ range to 0-80.

22JUL2026 analysis of 17JUL2026 run full_light
Finishing the current full_light runs for both Segue 1 and Draco using the NFW halo.
Segue 1 roughly recovers the previously identified SMBH result, although the final constraint
will depend on the completed landscape. Draco is beginning to form an SMBH basin, but the
result is still uncertain and is not yet strong enough to claim a recovered SMBH.

22JUL2026 run nonsingular_isothermal full_light
MODE                   = karl
stellar model          = karl_light_grid
light inputs           = full
kinematic inputs       = binned
halo type              = nonsingular_isothermal
halo parameterization  = v0_rc
halo parameters        = v0, r_c
comparison tag         = nonsingular_isothermal_full_light

Starting cored-halo runs for Segue 1 and Draco to test whether the emerging SMBH results
survive a change in halo profile. For Segue 1, this will test whether the roughly recovered
SMBH remains present with a cored halo. For Draco, this will show whether the cored halo
sharpens, weakens, or removes the currently uncertain SMBH basin.

"""