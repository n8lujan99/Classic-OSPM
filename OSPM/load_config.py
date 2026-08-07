# This holds the configuration defaults and the logic to load the galaxy-specific configuration. 
# It also defines the shared contract for the deck results and validation of the configuration.
# The individual galaxy configs contain only the galaxy-specific parameters and overrides, 
# and are loaded dynamically based on the galaxy name.
# !WARNING!: Changing this file will affect all galaxies, so be careful with edits here.

from pathlib import Path
from importlib import import_module
import copy
import multiprocessing as mp
import os
from .AI_defaults import CONFIG as AI_DEFAULTS

# --------------------------------------------------
# Repository and galaxy authority
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
    return OSPM_ROOT / "Data" / "Galaxy_Profiles" / _get_galaxy_name()

def detect_workers():
    slurm = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm)
    return mp.cpu_count()

# --------------------------------------------------
# Shared deck contract
# --------------------------------------------------

DECK_RESULT_COLUMNS = [
    "chi2", "reward", "status", "proposal_id",
    # Scientific score and solver diagnostics
    "chi2_losvd", "delta_chi2_iteration",
    "max_light_relative_residual", "max_light_sigma_residual", "light_constraint_ok",
    "solver_converged", "solver_iterations", "solver_failure_reason", "julia_status_code",
    # Radial and orbit-weight diagnostics
    "chi2_inner", "chi2_outer", "N_inner", "N_outer", "N_nonzero_weights", "effective_N_orbits", "max_weight_fraction",
    # Runtime contract diagnostics
    "halo_type", "alphat", "light_rel_tol", "light_sigma_tol",
    "delta_chi2_iter_tol", "halo_q_axis_ratio", "karl_halo_params_active",
    # Orbit-library coverage diagnostics
    "coverage_status", "coverage_strict", "coverage_issue_region", "coverage_issue_axis", "coverage_issue_shell_bands",
    "coverage_reasons", "coverage_fraction", "coverage_attempted_fraction", "coverage_success_fraction", "coverage_shell_min",
    "coverage_lfrac_min", "coverage_theta_min", "coverage_shell_gap", "coverage_lfrac_gap", "coverage_theta_gap", "coverage_joint_holes",
    "coverage_deadline_hit", "successful_base_orbits", "planned_base_orbits",
    # Karl phase-volume diagnostics
    "phase_volume_valid", "phase_volume_convention", "phase_volume_normalization",
    "phase_volume_launches_recorded", "phase_volume_sos_recorded",
    "phase_volume_valid_base_orbits", "phase_volume_invalid_recorded_orbits",
    "phase_volume_nested_groups", "phase_volume_duplicate_area_clusters",
    "phase_volume_duplicate_area_orbits", "raw_phase_volume_min",
    "raw_phase_volume_max", "raw_phase_volume_dynamic_range",
    "normalized_phase_volume_min", "normalized_phase_volume_max",
    "wphase_min", "wphase_max", "wphase_dynamic_range",
    "wphase_pair_max_relative_mismatch",
]


ALLOWED_STATUSES = [
    "todo", "seed", "pass", "orbit_fail", "numeric_fail", "unknown_fail", "timeout", "forbidden",
    "pass_full", "pass_bh_only", "pass_halo_only", "pass_bh_up","pass_bh_down", "pass_halo_up", "pass_halo_down", "pass_ml_up", "pass_ml_down",
    "orbit_fail_full", "orbit_fail_bh_only", "orbit_fail_halo_only", "orbit_fail_bh_up", "orbit_fail_bh_down", "orbit_fail_halo_up", "orbit_fail_halo_down", "orbit_fail_ml_up", "orbit_fail_ml_down",
    "numeric_fail_full", "numeric_fail_bh_only", "numeric_fail_halo_only", "numeric_fail_bh_up", "numeric_fail_bh_down", "numeric_fail_halo_up", "numeric_fail_halo_down", "numeric_fail_ml_up", "numeric_fail_ml_down",
    "timeout_full", "timeout_bh_only", "timeout_halo_only", "timeout_bh_up", "timeout_bh_down", "timeout_halo_up", "timeout_halo_down", "timeout_ml_up", "timeout_ml_down",
    "unknown_fail_full", "unknown_fail_bh_only", "unknown_fail_halo_only", "unknown_fail_bh_up", "unknown_fail_bh_down", "unknown_fail_halo_up", "unknown_fail_halo_down", "unknown_fail_ml_up", "unknown_fail_ml_down",
    "solver_failed_full", "solver_failed_bh_only", "solver_failed_halo_only", "solver_failed_bh_up", "solver_failed_bh_down", "solver_failed_halo_up", "solver_failed_halo_down", "solver_failed_ml_up", "solver_failed_ml_down",
    "physics_exception_full", "physics_exception_bh_only", "physics_exception_halo_only", "physics_exception_bh_up", "physics_exception_bh_down", "physics_exception_halo_up", "physics_exception_halo_down", "physics_exception_ml_up", "physics_exception_ml_down",
]


# --------------------------------------------------
# Shared defaults
# --------------------------------------------------

def _build_general_defaults(local_debug):
    # Local debug reduces the number of evaluated potentials, not the physical
    # orbit grid. Karl phase volumes require the complete (E, Lz, I3) library.
    norbit = 10000
    if norbit % 2 != 0:
        raise ValueError(f"Karl paired-orbit path requires even NORBIT; got {norbit}")
    return {
        # Core runtime
        "MODE": "karl","LOCAL_DEBUG": local_debug,"N_WORKERS": detect_workers(),"NORBIT": norbit,
        # Standard data-column contract
        "STAR_R_COL": "R_pc", "STAR_V_COL": "vlos_kms", "STAR_VERR_COL": "verr_kms", "RA_COL": "ra_deg", "DEC_COL": "dec_deg", "VLOS_COL": "vlos_kms",
        # Observable and solver policy
        "OBSERVABLES": { "NVBIN": 21, "NTHETA_LAUNCH": 9,
            "ORBIT_FILL_PCT": 0.85,
            "ORBIT_REGIONAL_FLOOR": 0.80,
            "ORBIT_MAX_REGIONAL_GAP": 0.10,
            "ORBIT_SHELL_BANDS": 8,
            "ORBIT_COVERAGE_CHECK_EVERY": 50,
            "ORBIT_WARN_FILL_PCT": 0.95,
            "ORBIT_WARN_SUCCESS_PCT": 0.99,
            "ORBIT_WARN_REGIONAL_FLOOR": 0.80,
            "ORBIT_WARN_MAX_REGIONAL_GAP": 0.15,
            "MODEL_OWNER_LIMIT": 15,
            "THREADS_PER_MODEL": 3,
            "KARL_ALPHAT": 1.0,
            "KARL_LIGHT_REL_TOL": 0.01,
            "KARL_LIGHT_SIGMA_TOL": 2.0,
            "KARL_DELTA_CHI2_ITER_TOL": 0.3,
            "KARL_MAXITER": 1000,
            "ENTROPY_FLOOR": 1e-12,
            "HALO_Q_AXIS_RATIO": 1.0,
        },

        # Shared penalty policy
        "PEN_SPHERE_STRENGTH": 2500,
        "PEN_SPHERE_POWER": 2.0,
        "PEN_SLOPE_STRENGTH": 5000,
        # Shared numerical-domain controls
        "MIN_DISTANCE": 5e-4,
        "R_GRID_POINTS": 256,
        "POTENTIAL_EXTENT": 6.0,
        "BH_MIN_RADIUS_MULTIPLIER": 2.0,
        # Deck behavior
        "ALLOWED_STATUSES": list(ALLOWED_STATUSES),
        "FILL_DEFAULT_STATUS": "todo",
        # Sampling
        "BATCH_SIZE": 1 if local_debug else 120,
        "MIN_BATCH_SIZE": 1 if local_debug else 120,
        "MAX_BATCH_SIZE": 1 if local_debug else 360,
        "CHUNK_SIZE": 1 if local_debug else 20,
        "_PRINT_EVERY": 10,
        "_print_counter": 1,
        # AI and learning
        "AI_START_AFTER": 500,
        "MIN_TRAIN_POINTS": 300,
        "TRAIN_WINDOW": 500,
        "AI_NOISE_INIT": 0.30,
        "AI_NOISE_MIN": 0.02,
        "AI_NOISE_TAU": 5000,
        "AI_MIN_DISTINCT_PASS": 800,
        "RESET_INTERVAL": 10000,
        "AI_DEBUG_EVERY": 200,
        "AI_SNAPSHOT_EVERY": 2000,
        "FLAT_WINDOW": 200,
        "FLAT_THRESHOLD": 1e-6,
        "FLAT_PATIENCE": 10,
        "AI_RESET_ON_FLAT": True,
        # Termination and runtime
        "MAX_RUNS": 1 if local_debug else 300000,
        "STOP_NO_IMPROVEMENT": 2000,
        "IMPROVEMENT_EPSILON": 1e-6,
        "LOG_INTERVAL": 1 if local_debug else 10,
        "PROF_EVERY": 1 if local_debug else 20,
        "EVAL_TIMEOUT_S": 200.0 if local_debug else 600.0,
        # Debug evaluation policy
        "EVAL_VARIANTS": ["full"] if local_debug else None,
        # Physical constants
        "G": 6.67430e-11,
        "Msun": 1.98847e30,
    }


# --------------------------------------------------
# Merge and validation helpers
# --------------------------------------------------
def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (key in result and isinstance(result[key], dict) and isinstance(value, dict)):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def _build_required_columns(parameter_names):
    names = list(parameter_names)
    if len(names) != 4:
        raise ValueError("PARAMETER_NAMES must contain four values:" "[halo amplitude, halo scale, MBH, ML]")
    return names + list(DECK_RESULT_COLUMNS)

def _validate_halo_contract(cfg):
    cfg["HALO_TYPE"] = str(cfg.get("HALO_TYPE", "none")).strip().lower()
    cfg["HALO_PARAMETERIZATION"] = str( cfg.get("HALO_PARAMETERIZATION", "rho_rs")).strip().lower()
    if cfg["HALO_PARAMETERIZATION"] in ("", "default"):
        cfg["HALO_PARAMETERIZATION"] = "rho_rs"
    allowed = ("rho_rs", "vcirc_rs", "v0_rc")
    if cfg["HALO_PARAMETERIZATION"] not in allowed:
        raise ValueError( "HALO_PARAMETERIZATION must be 'rho_rs', " "'vcirc_rs', or 'v0_rc'; got " f"{cfg['HALO_PARAMETERIZATION']!r}")
    halo_type = cfg["HALO_TYPE"]
    parameterization = cfg["HALO_PARAMETERIZATION"]
    if (parameterization == "v0_rc" and halo_type != "nonsingular_isothermal"):
        raise ValueError("HALO_PARAMETERIZATION='v0_rc' requires " "HALO_TYPE='nonsingular_isothermal'; got " f"HALO_TYPE={halo_type!r}")
    if (halo_type == "nonsingular_isothermal" and parameterization != "v0_rc"):
        raise ValueError("HALO_TYPE='nonsingular_isothermal' requires " "HALO_PARAMETERIZATION='v0_rc'; got " f"HALO_PARAMETERIZATION={parameterization!r}")

def _validate_parameter_contract(cfg):
    names = list(cfg["PARAMETER_NAMES"])
    initial = list(cfg["INITIAL_THETA"])
    bounds = list(cfg["THETA_BOUNDS"])
    if not (len(names) == len(initial) == len(bounds) == 4):
        raise ValueError("PARAMETER_NAMES, INITIAL_THETA, and THETA_BOUNDS " "must each contain four entries")
    expected_names = {"rho_rs": ["rho_s", "r_s", "MBH", "ML"], "vcirc_rs": ["vcirc", "r_s", "MBH", "ML"], "v0_rc": ["v0", "r_c", "MBH", "ML"]}[cfg["HALO_PARAMETERIZATION"]]
    if names != expected_names:
        raise ValueError( f"HALO_PARAMETERIZATION={cfg['HALO_PARAMETERIZATION']!r} " f"expects PARAMETER_NAMES={expected_names!r}; " f"got {names!r}")
    for index, ((lo, hi), value) in enumerate(zip(bounds, initial)):
        if lo > hi:
            raise ValueError(f"Invalid THETA_BOUNDS[{index}]: lower bound {lo} " f"is greater than upper bound {hi}")
        if not lo <= value <= hi:
            raise ValueError(f"INITIAL_THETA[{index}]={value} is outside " f"THETA_BOUNDS[{index}]={bounds[index]}")
    fixed = cfg.get("FIXED_THETA")
    if fixed is not None:
        if len(fixed) != 4:
            raise ValueError("FIXED_THETA must contain four entries or be None")
        for index, ((lo, hi), value) in enumerate(zip(bounds, fixed)):
            if not lo <= value <= hi:
                raise ValueError( f"FIXED_THETA[{index}]={value} is outside " f"THETA_BOUNDS[{index}]={bounds[index]}")
    cfg["REQUIRE_COLUMNS"] = _build_required_columns(names)

# --------------------------------------------------
# Config loader
# --------------------------------------------------
def load_config():
    galaxy = _get_galaxy_name()
    module_name = (f"Data.Galaxy_Profiles.{galaxy}.{galaxy}_OSPM_Config")
    try:
        mod = import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load config for {galaxy} using {module_name}") from exc
    galaxy_config = dict(mod.CONFIG)
    local_debug = bool(galaxy_config.get("LOCAL_DEBUG", getattr(mod, "LOCAL_DEBUG", False)))
    cfg = _build_general_defaults(local_debug)
    cfg = _deep_merge(cfg, AI_DEFAULTS)
    cfg = _deep_merge(cfg, galaxy_config)
    cfg["GALAXY"] = galaxy
    cfg["LOCAL_DEBUG"] = local_debug
    required = [ "GALAXY", "MODE", "HALO_TYPE", "HALO_PARAMETERIZATION", "PARAMETER_NAMES", "INITIAL_THETA", "THETA_BOUNDS",
        "FIXED_THETA", "STELLAR_MODEL", "MIN_DISTANCE", "MAX_DISTANCE", "NORBIT", "BATCH_SIZE", "MAX_RUNS", "SURFACE_BRIGHTNESS_CSV",
        "KINEMATIC_BINS_CSV", "DATA_CSV", "CSV_PATH"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"CONFIG missing required keys: {missing}")
    _validate_halo_contract(cfg)
    _validate_parameter_contract(cfg)
    if cfg["NORBIT"] % 2 != 0:
        raise ValueError("Karl paired-orbit path requires even NORBIT; " f"got {cfg['NORBIT']}")
    print("[CONFIG LOAD] GALAXY =", cfg["GALAXY"])
    print("[CONFIG LOAD] LOCAL_DEBUG =", cfg["LOCAL_DEBUG"])
    print("[CONFIG LOAD] HALO_TYPE =", cfg["HALO_TYPE"])
    print("[CONFIG LOAD] HALO_PARAMETERIZATION =", cfg["HALO_PARAMETERIZATION"])
    print("[CONFIG LOAD] PARAMETER_NAMES =", cfg["PARAMETER_NAMES"])
    print("[CONFIG LOAD] THETA_BOUNDS =", cfg["THETA_BOUNDS"])
    print("[CONFIG LOAD] CSV_PATH =", cfg["CSV_PATH"])
    return cfg