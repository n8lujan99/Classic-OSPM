# OSPM_Deterministic.py
#
# Deterministic OSPM model-suite runner.
#
# This path intentionally bypasses the AI proposal system.
# The model suite is defined directly in this file for now.
#
# Current experiment:
#   v0  = 23 km/s
#   r_c = 100 kpc = 100000 pc
#   MBH = 0 -> 1e5 Msun, 50 points
#   M/L = 0.1 -> 5.0, 50 points
#
# Total:
#   50 x 50 = 2500 models
#
# Parallelism remains inside the existing Julia batch scheduler.

import os
import time

import numpy as np
import pandas as pd

from OSPM.Physics import OSPM_Physics as P


# =============================================================================
# DETERMINISTIC SUITE DEFINITION
# =============================================================================

FIXED_V0_KMS = 23.0
FIXED_RC_PC = 100000.0

MBH_MIN = 0.0
MBH_MAX = 1.0e5
MBH_STEPS = 50

ML_MIN = 0.1
ML_MAX = 5.0
ML_STEPS = 50

OUTPUT_CSV_NAME = "draco_karl_deterministic_v0_23_rc_100kpc_bh_ml_50x50.csv"


# =============================================================================
# JULIA RESULT CONTRACT
# =============================================================================

# These names follow the current evaluate_batch_theta result ordering.
# If Julia later appends more diagnostics, they are retained automatically
# as julia_result_N columns rather than discarded.
JULIA_RESULT_NAMES = [
    "julia_status_code",
    "fit_statistic_value",
    "chi2_inner",
    "chi2_outer",
    "delta_fit_statistic_iteration",
    "max_light_relative_residual",
    "max_light_sigma_residual",
    "light_constraint_ok",
    "solver_converged",
    "solver_iterations",
    "solver_failure_reason",
    "N_inner",
    "N_outer",
    "N_nonzero_weights",
    "effective_N_orbits",
    "max_weight_fraction",
    "coverage_status",
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
    "phase_volume_valid",
    "phase_volume_convention",
    "phase_volume_normalization",
    "phase_volume_launches_recorded",
    "phase_volume_sos_recorded",
    "phase_volume_valid_base_orbits",
    "phase_volume_invalid_recorded_orbits",
    "phase_volume_nested_groups",
    "phase_volume_duplicate_area_clusters",
    "phase_volume_duplicate_area_orbits",
    "raw_phase_volume_min",
    "raw_phase_volume_max",
    "raw_phase_volume_dynamic_range",
    "normalized_phase_volume_min",
    "normalized_phase_volume_max",
    "wphase_min",
    "wphase_max",
    "wphase_dynamic_range",
    "wphase_pair_max_relative_mismatch",
]


# =============================================================================
# HELPERS
# =============================================================================

def _config_option(config, *names, default=None):
    observable_config = config.get("OBSERVABLES", {}) or {}

    for source in (observable_config, config):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]

    return default


def _scalar_at(value, index):
    arr = np.asarray(value).ravel()

    if index >= arr.size:
        raise IndexError(
            f"Julia result array has size {arr.size}, "
            f"cannot read model index {index}"
        )

    out = arr[index]

    if isinstance(out, np.generic):
        out = out.item()

    return out


def _status_from_result(code, score, losvd_fit_statistic):
    code = int(code)
    score = float(score)

    fit = str(losvd_fit_statistic).strip().lower()

    if fit == "multinomial":
        finite_score = np.isfinite(score) and score >= 0.0
    else:
        finite_score = np.isfinite(score) and score > 1.0e-12

    if code == 0:
        return "pass" if finite_score else "numeric_fail"

    return {
        1: "orbit_fail",
        2: "solver_failed",
        3: "physics_exception",
        4: "timeout",
    }.get(code, "unknown_fail")


def _output_path(config):
    configured_csv = config.get("CSV_PATH", None)

    if configured_csv is not None:
        output_dir = os.path.dirname(os.path.abspath(str(configured_csv)))
    else:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, OUTPUT_CSV_NAME)


# =============================================================================
# KARL 50 x 50 MODEL GRID
# =============================================================================

def generate_theta_set():
    mbh_values = np.linspace(MBH_MIN, MBH_MAX, MBH_STEPS, dtype=float)
    ml_values = np.linspace(ML_MIN, ML_MAX, ML_STEPS, dtype=float)

    models = []
    model_id = 1

    for MBH in mbh_values:
        for ML in ml_values:
            models.append({
                "model_id": model_id,
                "v0": FIXED_V0_KMS,
                "r_c": FIXED_RC_PC,
                "MBH": float(MBH),
                "ML": float(ML),
            })

            model_id += 1

    expected = MBH_STEPS * ML_STEPS

    if len(models) != expected:
        raise RuntimeError(
            f"Deterministic suite generation failed: "
            f"expected {expected} models, got {len(models)}"
        )

    print(
        "[DETERMINISTIC GRID]"
        f" models={len(models)}"
        f" v0={FIXED_V0_KMS} km/s"
        f" r_c={FIXED_RC_PC} pc"
        f" MBH=[{MBH_MIN}, {MBH_MAX}] steps={MBH_STEPS}"
        f" ML=[{ML_MIN}, {ML_MAX}] steps={ML_STEPS}",
        flush=True,
    )

    print(
        "[DETERMINISTIC GRID]"
        f" MBH_step={(MBH_MAX - MBH_MIN) / (MBH_STEPS - 1):.12g}"
        f" ML_step={(ML_MAX - ML_MIN) / (ML_STEPS - 1):.12g}",
        flush=True,
    )

    return models


# =============================================================================
# RESTART / OUTPUT
# =============================================================================

def _completed_model_ids(output_path):
    if not os.path.exists(output_path):
        return set()

    try:
        df = pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        return set()

    if "model_id" not in df.columns:
        raise RuntimeError(
            f"Existing deterministic output is missing model_id: {output_path}"
        )

    completed = pd.to_numeric(df["model_id"], errors="coerce")
    completed = completed[np.isfinite(completed)]

    return set(completed.astype(int).tolist())


def _append_rows(output_path, rows):
    if not rows:
        return

    frame = pd.DataFrame(rows)

    write_header = not os.path.exists(output_path)

    frame.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
    )


# =============================================================================
# DETERMINISTIC RUNNER
# =============================================================================

def run_deterministic(config, physics_engine):
    if physics_engine is None:
        raise RuntimeError("physics_engine is required")

    parameter_names = list(config.get("PARAMETER_NAMES", []))

    expected_parameter_names = ["v0", "r_c", "MBH", "ML"]

    if parameter_names != expected_parameter_names:
        raise RuntimeError(
            "Current deterministic suite is specifically defined for "
            f"PARAMETER_NAMES={expected_parameter_names}; "
            f"received {parameter_names}"
        )

    halo_parameterization = str(
        config.get("HALO_PARAMETERIZATION", "")
    ).strip().lower()

    if halo_parameterization != "v0_rc":
        raise RuntimeError(
            "Current deterministic suite requires "
            "HALO_PARAMETERIZATION='v0_rc'"
        )

    obs = getattr(physics_engine, "__wrapped_obs__", None)
    halo_type = getattr(physics_engine, "__halo_type__", None)
    surface_brightness_profile = getattr(
        physics_engine,
        "__surface_brightness_profile__",
        None,
    )

    if obs is None:
        raise RuntimeError(
            "Wrapped physics engine is missing __wrapped_obs__"
        )

    if halo_type is None:
        raise RuntimeError(
            "Wrapped physics engine is missing __halo_type__"
        )

    if surface_brightness_profile is None:
        raise RuntimeError(
            "Wrapped physics engine is missing "
            "__surface_brightness_profile__"
        )
    Norbit = int(getattr(obs, "Norbit"))

    losvd_fit_statistic = str(
        _config_option(
            config,
            "LOSVD_FIT_STATISTIC",
            "losvd_fit_statistic",
            default="legacy_chi2",
        )
    ).strip().lower()

    losvd_conditioning = str(
        _config_option(
            config,
            "LOSVD_CONDITIONING",
            "losvd_conditioning",
            default="vlos_cut",
        )
    ).strip().lower()

    models = generate_theta_set()

    batch_size = int(config["CHUNK_SIZE"])
    if batch_size <= 0:
        raise ValueError("CHUNK_SIZE must be positive")

    output_path = _output_path(config)
    completed_ids = _completed_model_ids(output_path)

    pending_models = [
        model
        for model in models
        if int(model["model_id"]) not in completed_ids
    ]

    print(
        "[DETERMINISTIC]"
        f" total_models={len(models)}"
        f" already_completed={len(completed_ids)}"
        f" pending={len(pending_models)}"
        f" batch_size={batch_size}",
        flush=True,
    )

    print(
        f"[DETERMINISTIC] output={output_path}",
        flush=True,
    )

    if not pending_models:
        print(
            "[DETERMINISTIC] All requested models are already complete.",
            flush=True,
        )

        return {
            "mode": "deterministic",
            "requested_models": len(models),
            "completed_models": len(models),
            "output_csv": output_path,
        }

    run_id = config.get("RUN_ID", "")
    worker_id = int(config.get("WORKER_ID", 0))

    run_t0 = time.perf_counter()

    for batch_start in range(
        0,
        len(pending_models),
        batch_size,
    ):
        batch_models = pending_models[
            batch_start:batch_start + batch_size
        ]

        theta_rows = np.asarray(
            [
                [
                    model["v0"],
                    model["r_c"],
                    model["MBH"],
                    model["ML"],
                ]
                for model in batch_models
            ],
            dtype=float,
        )

        # evaluate_batch_theta_julia expects:
        #
        #     shape = (4, nbatch)
        #
        theta_matrix = theta_rows.T

        first_model_id = int(batch_models[0]["model_id"])
        last_model_id = int(batch_models[-1]["model_id"])

        print(
            "[DETERMINISTIC BATCH]"
            f" models={first_model_id}-{last_model_id}"
            f" count={len(batch_models)}",
            flush=True,
        )

        batch_t0 = time.perf_counter()

        batch_result = P.evaluate_batch_theta_julia(
            thetas=theta_matrix,
            obs=obs,
            halo_type=halo_type,
            surface_brightness_profile=surface_brightness_profile,
            Norbit=Norbit,
            config=config,
        )

        batch_elapsed = time.perf_counter() - batch_t0

        if not isinstance(batch_result, tuple):
            raise RuntimeError(
                "evaluate_batch_theta_julia did not return a tuple"
            )

        if len(batch_result) < 2:
            raise RuntimeError(
                "evaluate_batch_theta_julia returned fewer than "
                "the required status and fit-statistic arrays"
            )

        for result_index, result_array in enumerate(batch_result):
            arr = np.asarray(result_array).ravel()

            if arr.size != len(batch_models):
                raise RuntimeError(
                    "Julia batch result length mismatch: "
                    f"result_index={result_index} "
                    f"expected={len(batch_models)} "
                    f"received={arr.size}"
                )

        rows = []

        for j, model in enumerate(batch_models):
            code = int(_scalar_at(batch_result[0], j))
            score = float(_scalar_at(batch_result[1], j))

            status = _status_from_result(
                code,
                score,
                losvd_fit_statistic,
            )

            row = {
                "model_id": int(model["model_id"]),
                "v0": float(model["v0"]),
                "r_c": float(model["r_c"]),
                "MBH": float(model["MBH"]),
                "ML": float(model["ML"]),
                "chi2": score,
                "chi2_losvd": score,
                "fit_statistic_value": score,
                "losvd_fit_statistic": losvd_fit_statistic,
                "losvd_conditioning": losvd_conditioning,
                "status": f"{status}_full",
                "julia_status_code": code,
                "run_id": run_id,
                "worker_id": worker_id,
            }

            # Preserve the remaining Julia diagnostics.
            #
            # Known outputs get descriptive names.
            # Anything added later by Julia is still retained.
            for result_index in range(2, len(batch_result)):
                if result_index < len(JULIA_RESULT_NAMES):
                    name = JULIA_RESULT_NAMES[result_index]
                else:
                    name = f"julia_result_{result_index}"

                row[name] = _scalar_at(
                    batch_result[result_index],
                    j,
                )

            rows.append(row)

        _append_rows(output_path, rows)

        finished = len(completed_ids) + batch_start + len(batch_models)

        print(
            "[DETERMINISTIC BATCH COMPLETE]"
            f" models={first_model_id}-{last_model_id}"
            f" elapsed_s={batch_elapsed:.3f}"
            f" completed={finished}/{len(models)}",
            flush=True,
        )

    total_elapsed = time.perf_counter() - run_t0

    print(
        "[DETERMINISTIC COMPLETE]"
        f" models={len(models)}"
        f" elapsed_s={total_elapsed:.3f}"
        f" output={output_path}",
        flush=True,
    )

    return {
        "mode": "deterministic",
        "requested_models": len(models),
        "completed_models": len(models),
        "output_csv": output_path,
        "elapsed_s": total_elapsed,
    }
