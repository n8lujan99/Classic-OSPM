import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("dark_background")


# --------------------------------------------------
# Resolve repository and galaxy
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
WHICH_GALAXY = REPO_ROOT / "which_galaxy"

if WHICH_GALAXY.exists():
    GALAXY = WHICH_GALAXY.read_text().strip() or "unknown"
else:
    GALAXY = "unknown"

DEFAULT_PROFILE_DIR = (
    REPO_ROOT
    / "Data"
    / "Galaxy_Profiles"
    / GALAXY
    / "default"
)


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv_positional",
        nargs="?",
        default=None,
        help="Optional CSV path.",
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="CSV path, absolute or relative to the repository root.",
    )

    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Profile directory containing the model CSV.",
    )

    parser.add_argument(
        "--pattern",
        default="*.csv",
        help=(
            "Glob pattern used when automatically selecting a CSV. "
            "Non-model CSV files are ignored automatically."
        ),
    )

    parser.add_argument(
        "--zoom-dchi",
        type=float,
        default=50.0,
        help="Keep models with chi2 less than best chi2 plus this value.",
    )

    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help=(
            "Include every row with finite model parameters and objective, "
            "including failed/diagnostic statuses."
        ),
    )

    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Plot only status == pass_full instead of all successful pass_* variants.",
    )

    return parser.parse_args()


# --------------------------------------------------
# Path helpers
# --------------------------------------------------

def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def timestamp_from_filename(path):
    match = re.search(
        r"(\d{8})_(\d{6})",
        os.path.basename(path),
    )

    if match is None:
        return None

    return int(match.group(1) + match.group(2))


def looks_like_model_csv(path):
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False

    basic = {"MBH", "ML", "status"}
    if not basic.issubset(columns):
        return False

    has_objective = any(
        column in columns
        for column in ("chi2", "chi_total", "chi2_total", "chi2_losvd")
    )

    has_halo_amplitude = any(
        column in columns
        for column in ("v0", "vcirc", "rho_s")
    )

    has_halo_radius = any(
        column in columns
        for column in ("r_c", "r_s")
    )

    return has_objective and has_halo_amplitude and has_halo_radius


def find_latest_csv(profile_dir, pattern):
    profile_dir = Path(profile_dir)

    if not profile_dir.exists():
        raise FileNotFoundError(
            f"Profile directory not found:\n  {profile_dir}"
        )

    candidates = [
        Path(path)
        for path in glob.glob(str(profile_dir / pattern))
        if looks_like_model_csv(path)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No OSPM model CSV matching {pattern!r} found in:\n  {profile_dir}"
        )

    timestamped = [
        (timestamp_from_filename(path), path)
        for path in candidates
    ]

    timestamped = [
        item
        for item in timestamped
        if item[0] is not None
    ]

    if timestamped:
        return max(timestamped, key=lambda item: item[0])[1]

    return max(candidates, key=lambda path: path.stat().st_mtime)


def infer_galaxy_from_csv(csv_file):
    csv_file = Path(csv_file).resolve()

    try:
        relative = csv_file.relative_to(
            REPO_ROOT / "Data" / "Galaxy_Profiles"
        )
        return relative.parts[0]

    except Exception:
        return GALAXY


# --------------------------------------------------
# Data helpers
# --------------------------------------------------

def first_existing(frame, *columns):
    for column in columns:
        if column in frame.columns:
            return column

    return None


def require_columns(frame, columns):
    missing = [column for column in columns if column not in frame.columns]

    if missing:
        raise KeyError(
            "Missing required column(s): "
            + ", ".join(missing)
            + "\nAvailable columns:\n  "
            + "\n  ".join(frame.columns)
        )


def numeric_clean(frame, columns):
    frame = frame.copy()

    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    return frame.replace([np.inf, -np.inf], np.nan)


def finite_positive_floor(values):
    values = pd.to_numeric(values, errors="coerce")
    positive = values[np.isfinite(values) & (values > 0)]

    if len(positive) == 0:
        return 1.0

    return positive.min() / 3.0


def filter_model_rows(frame, include_diagnostics, full_only):
    if include_diagnostics or "status" not in frame.columns:
        return frame.copy()

    status = frame["status"].astype(str)

    if full_only:
        keep = status == "pass_full"
    else:
        keep = status.str.startswith("pass")

    return frame[keep].copy()


def split_black_hole_models(frame, mbh_floor):
    no_black_hole = frame["MBH"] <= 0

    no_bh = frame[no_black_hole].copy()
    with_bh = frame[~no_black_hole].copy()

    no_bh["MBH_plot"] = mbh_floor
    with_bh["MBH_plot"] = with_bh["MBH"]

    return no_bh, with_bh


def identify_halo_columns(frame):
    if "v0" in frame.columns:
        halo_column = "v0"
        halo_label = r"$v_0\;({\rm km\,s^{-1}})$"
        halo_log = False

    elif "vcirc" in frame.columns:
        halo_column = "vcirc"
        halo_label = r"$v_{\rm circ}\;({\rm km\,s^{-1}})$"
        halo_log = False

    elif "rho_s" in frame.columns:
        halo_column = "rho_s"
        halo_label = r"$\rho_s$"
        halo_log = True

    else:
        raise KeyError(
            "Missing halo-amplitude column. Expected v0, vcirc, or rho_s."
        )

    if "r_c" in frame.columns:
        radius_column = "r_c"
        radius_label = r"$r_c\;({\rm pc})$"

    elif "r_s" in frame.columns:
        radius_column = "r_s"
        radius_label = r"$r_s\;({\rm pc})$"

    else:
        raise KeyError(
            "Missing halo-radius column. Expected r_c or r_s."
        )

    return (
        halo_column,
        halo_label,
        halo_log,
        radius_column,
        radius_label,
    )


def add_ratio_column(frame, numerator, denominator, output):
    if numerator not in frame.columns or denominator not in frame.columns:
        return

    denom = pd.to_numeric(frame[denominator], errors="coerce")
    numer = pd.to_numeric(frame[numerator], errors="coerce")

    valid = np.isfinite(numer) & np.isfinite(denom) & (denom != 0)
    frame[output] = np.nan
    frame.loc[valid, output] = numer[valid] / denom[valid]


# --------------------------------------------------
# Plot helpers
# --------------------------------------------------

def set_window_title(figure, title):
    manager = getattr(figure.canvas, "manager", None)

    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def add_mbh_zero_marker(axis, mbh_floor):
    axis.axvline(
        mbh_floor,
        color="#4FA3FF",
        alpha=0.45,
        linestyle="--",
    )

    axis.text(
        mbh_floor,
        0.02,
        r" $M_{\rm BH}=0$",
        color="#4FA3FF",
        fontsize=8,
        rotation=90,
        va="bottom",
        transform=axis.get_xaxis_transform(),
    )


def scatter_split(
    axis,
    black_hole_models,
    no_black_hole_models,
    x_column,
    y_column,
    black_hole_size=10,
    no_black_hole_size=16,
    label=True,
):
    axis.scatter(
        black_hole_models[x_column],
        black_hole_models[y_column],
        s=black_hole_size,
        c="#39EB33",
        edgecolors="none",
        rasterized=True,
        label="Black-hole models" if label else None,
    )

    axis.scatter(
        no_black_hole_models[x_column],
        no_black_hole_models[y_column],
        s=no_black_hole_size,
        c="#4FA3FF",
        edgecolors="none",
        rasterized=True,
        label=r"$M_{\rm BH}=0$" if label else None,
    )


def plot_parameter_landscape(
    frame,
    objective_column,
    objective_label,
    parameters,
    mbh_floor,
    galaxy_label,
    csv_file,
    title_suffix,
):
    no_bh, with_bh = split_black_hole_models(frame, mbh_floor)

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(24, 5.2),
        sharey=True,
    )

    for axis, (
        plot_column,
        axis_label,
        physical_column,
        use_log_scale,
    ) in zip(axes, parameters):
        scatter_split(
            axis,
            with_bh,
            no_bh,
            plot_column,
            objective_column,
        )

        if use_log_scale:
            axis.set_xscale("log")

        axis.set_xlabel(axis_label)
        axis.set_ylabel(objective_label)
        axis.grid(alpha=0.18, linewidth=0.6)

        if physical_column == "MBH":
            add_mbh_zero_marker(axis, mbh_floor)

    axes[0].legend(loc="best", fontsize=8)

    figure.suptitle(
        f"{galaxy_label} OSPM {title_suffix}\n"
        f"source: {csv_file.name}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    set_window_title(figure, f"OSPM {title_suffix}")


def plot_metric_strip(
    frame,
    metric_panels,
    x_column,
    x_label,
    galaxy_label,
    csv_file,
    window_title,
    x_log=False,
    mbh_floor=None,
    unity_line_columns=None,
):
    panels = [
        panel
        for panel in metric_panels
        if panel[0] in frame.columns
        and frame[panel[0]].notna().any()
    ]

    if not panels:
        return None

    unity_line_columns = set(unity_line_columns or [])

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 5.1),
        squeeze=False,
    )
    axes = axes[0]

    if x_column == "MBH_plot":
        no_bh, with_bh = split_black_hole_models(frame, mbh_floor)

    for axis, (column, label, title, y_log) in zip(axes, panels):
        if x_column == "MBH_plot":
            scatter_split(
                axis,
                with_bh,
                no_bh,
                x_column,
                column,
                black_hole_size=12,
                no_black_hole_size=18,
                label=False,
            )
            add_mbh_zero_marker(axis, mbh_floor)

        else:
            axis.plot(
                frame[x_column],
                frame[column],
                linestyle="none",
                marker="o",
                markersize=2.5,
                alpha=0.55,
            )

        if x_log:
            axis.set_xscale("log")

        if y_log:
            positive = pd.to_numeric(frame[column], errors="coerce") > 0
            if positive.any():
                axis.set_yscale("log")

        if column in unity_line_columns:
            axis.axhline(1.0, linestyle="--", alpha=0.55, linewidth=1.0)

        axis.set_xlabel(x_label)
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.grid(alpha=0.18, linewidth=0.6)

    figure.suptitle(
        f"{galaxy_label} OSPM {window_title}\n"
        f"source: {csv_file.name}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    set_window_title(figure, f"OSPM {window_title}")
    return figure


# --------------------------------------------------
# Main analysis
# --------------------------------------------------

def main():
    args = parse_args()

    if args.csv:
        csv_file = resolve_path(args.csv)

    elif args.csv_positional:
        csv_file = resolve_path(args.csv_positional)

    else:
        profile_dir = (
            resolve_path(args.profile_dir)
            if args.profile_dir
            else DEFAULT_PROFILE_DIR
        )
        csv_file = find_latest_csv(profile_dir, args.pattern)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n  {csv_file}"
        )

    galaxy_label = infer_galaxy_from_csv(csv_file)
    all_df = pd.read_csv(csv_file)
    all_df["csv_row"] = np.arange(1, len(all_df) + 1)

    (
        halo_param_col,
        halo_param_label,
        halo_log_scale,
        radius_param_col,
        radius_param_label,
    ) = identify_halo_columns(all_df)

    aliases = {
        "objective": (
            "chi2",
            "chi_total",
            "chi2_total",
            "chi2_losvd",
        ),
        "losvd": (
            "chi2_losvd",
            "chi_losvd_score",
            "chi2",
        ),
        "inner": ("chi2_inner", "chi_inner"),
        "outer": ("chi2_outer", "chi_outer"),
        "nonzero": ("N_nonzero_weights", "N_nonzero"),
        "neff": ("effective_N_orbits", "Neff"),
        "objective_value": ("reward", "profit"),
    }

    columns = {
        name: first_existing(all_df, *choices)
        for name, choices in aliases.items()
    }

    objective_column = columns["objective"]

    if objective_column is None:
        raise KeyError(
            "No objective column found. Expected chi2, chi_total, "
            "chi2_total, or chi2_losvd."
        )

    required_columns = [
        objective_column,
        "MBH",
        halo_param_col,
        radius_param_col,
        "ML",
    ]
    require_columns(all_df, required_columns)

    known_numeric_columns = {
        *required_columns,
        "csv_row",
        "chi2",
        "reward",
        "proposal_id",
        "chi2_losvd",
        "delta_chi2_iteration",
        "max_light_relative_residual",
        "max_light_sigma_residual",
        "light_constraint_ok",
        "solver_converged",
        "solver_iterations",
        "julia_status_code",
        "chi2_inner",
        "chi2_outer",
        "N_inner",
        "N_outer",
        "N_nonzero_weights",
        "effective_N_orbits",
        "max_weight_fraction",
        "alphat",
        "light_rel_tol",
        "light_sigma_tol",
        "delta_chi2_iter_tol",
        "halo_q_axis_ratio",
        "karl_halo_params_active",
        "coverage_strict",
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
        # Compatibility with older decks:
        "chi_total",
        "chi2_total",
        "chi_losvd_score",
        "chi2_light",
        "entropy",
        "refine_passes",
    }

    df = numeric_clean(all_df, known_numeric_columns)
    df = df.dropna(subset=required_columns)
    df = filter_model_rows(
        df,
        include_diagnostics=args.include_diagnostics,
        full_only=args.full_only,
    )

    if len(df) == 0:
        if args.include_diagnostics:
            mode = "all finite diagnostic rows"
        elif args.full_only:
            mode = "status == pass_full"
        else:
            mode = "all successful pass* statuses"

        raise ValueError(
            f"No usable rows remain after cleaning with {mode}."
        )

    df = df.copy()
    df["run_index"] = np.arange(1, len(df) + 1)

    add_ratio_column(
        df,
        "delta_chi2_iteration",
        "delta_chi2_iter_tol",
        "delta_chi2_over_tol",
    )
    add_ratio_column(
        df,
        "max_light_sigma_residual",
        "light_sigma_tol",
        "light_sigma_over_tol",
    )
    add_ratio_column(
        df,
        "phase_volume_valid_base_orbits",
        "successful_base_orbits",
        "phase_volume_valid_fraction",
    )

    best_index = df[objective_column].idxmin()
    best_row = df.loc[best_index]
    best_value = float(best_row[objective_column])
    zoom = df[
        df[objective_column] <= best_value + args.zoom_dchi
    ].copy()

    mbh_floor = finite_positive_floor(df["MBH"])

    for frame in (df, zoom):
        frame["MBH_plot"] = frame["MBH"]
        frame.loc[frame["MBH"] <= 0, "MBH_plot"] = mbh_floor

    if objective_column == "chi2":
        objective_label = r"$\chi^2$"
    elif objective_column in {"chi_total", "chi2_total"}:
        objective_label = r"$\chi^2_{\rm total}$"
    else:
        objective_label = objective_column

    parameters = [
        (
            "MBH_plot",
            r"$M_{\rm BH}\;(M_\odot)$",
            "MBH",
            True,
        ),
        (
            halo_param_col,
            halo_param_label,
            halo_param_col,
            halo_log_scale,
        ),
        (
            radius_param_col,
            radius_param_label,
            radius_param_col,
            True,
        ),
        (
            "ML",
            r"$M/L$",
            "ML",
            False,
        ),
    ]

    print(f"[PLOT] galaxy          = {galaxy_label}")
    print(f"[PLOT] using           = {csv_file}")
    print(f"[INFO] all CSV rows    = {len(all_df)}")
    print(f"[INFO] plotted rows    = {len(df)}")
    print(f"[INFO] objective       = {objective_column}")
    print(f"[INFO] best objective  = {best_value}")
    print(f"[INFO] zoom rows       = {len(zoom)}")

    if "status" in all_df.columns:
        print("\nSTATUS COUNTS")
        print(all_df["status"].astype(str).value_counts().to_string())

    if "coverage_status" in df.columns:
        coverage_counts = df["coverage_status"].dropna().astype(str).value_counts()
        if len(coverage_counts) > 0:
            print("\nPLOTTED COVERAGE STATUS COUNTS")
            print(coverage_counts.to_string())

    print("\nBEST MODEL")
    best_columns = [
        "csv_row",
        "status" if "status" in best_row.index else None,
        halo_param_col,
        radius_param_col,
        "MBH",
        "ML",
        objective_column,
    ]

    diagnostic_print_order = [
        columns["losvd"],
        "delta_chi2_iteration",
        "delta_chi2_iter_tol",
        "max_light_relative_residual",
        "light_rel_tol",
        "max_light_sigma_residual",
        "light_sigma_tol",
        "light_constraint_ok",
        "solver_converged",
        "solver_iterations",
        "solver_failure_reason",
        "julia_status_code",
        columns["inner"],
        columns["outer"],
        columns["nonzero"],
        columns["neff"],
        "max_weight_fraction",
        "coverage_status",
        "coverage_fraction",
        "coverage_success_fraction",
        "coverage_shell_min",
        "coverage_lfrac_min",
        "coverage_theta_min",
        "coverage_joint_holes",
        "successful_base_orbits",
        "planned_base_orbits",
        "phase_volume_valid",
        "phase_volume_convention",
        "phase_volume_normalization",
        "phase_volume_valid_base_orbits",
        "phase_volume_invalid_recorded_orbits",
        "phase_volume_nested_groups",
        "phase_volume_duplicate_area_clusters",
        "phase_volume_duplicate_area_orbits",
        "raw_phase_volume_dynamic_range",
        "normalized_phase_volume_min",
        "normalized_phase_volume_max",
        "wphase_dynamic_range",
        "wphase_pair_max_relative_mismatch",
        columns["objective_value"],
    ]

    best_columns = [
        column
        for column in best_columns
        if column is not None
    ]

    for column in diagnostic_print_order:
        if (
            column
            and column in best_row.index
            and column not in best_columns
        ):
            best_columns.append(column)

    print(best_row[best_columns].to_string())

    plot_parameter_landscape(
        df,
        objective_column,
        objective_label,
        parameters,
        mbh_floor,
        galaxy_label,
        csv_file,
        "full objective landscape",
    )

    plot_parameter_landscape(
        zoom,
        objective_column,
        objective_label,
        parameters,
        mbh_floor,
        galaxy_label,
        csv_file,
        rf"zoomed objective landscape, $\Delta\chi^2\leq {args.zoom_dchi:g}$",
    )

    fit_panels = []

    for column, label, title in [
        (columns["losvd"], r"$\chi^2_{\rm LOSVD}$", "LOSVD fit"),
        (columns["inner"], r"$\chi^2_{\rm inner}$", "Inner fit"),
        (columns["outer"], r"$\chi^2_{\rm outer}$", "Outer fit"),
        (objective_column, objective_label, "Scientific objective"),
    ]:
        if column:
            fit_panels.append((column, label, title, False))

    plot_metric_strip(
        df,
        fit_panels,
        "MBH_plot",
        r"$M_{\rm BH}\;(M_\odot)$",
        galaxy_label,
        csv_file,
        "scientific fit diagnostics",
        x_log=True,
        mbh_floor=mbh_floor,
    )

    convergence_panels = [
        (
            "delta_chi2_over_tol",
            r"$\Delta\chi^2_{\rm iter}/{\rm tol}$",
            "Iteration convergence / tolerance",
            True,
        ),
        (
            "max_light_relative_residual",
            "maximum relative light residual",
            "Relative light residual",
            True,
        ),
        (
            "light_sigma_over_tol",
            r"max light $|\Delta|/\sigma$ / tolerance",
            "2σ light check / tolerance",
            True,
        ),
        (
            "solver_iterations",
            "solver iterations",
            "Weight-solver iterations",
            False,
        ),
    ]

    plot_metric_strip(
        df,
        convergence_panels,
        "MBH_plot",
        r"$M_{\rm BH}\;(M_\odot)$",
        galaxy_label,
        csv_file,
        "Karl convergence and light constraints",
        x_log=True,
        mbh_floor=mbh_floor,
        unity_line_columns={
            "delta_chi2_over_tol",
            "light_sigma_over_tol",
        },
    )

    orbit_panels = []

    for column, label, title, y_log in [
        (columns["nonzero"], "active orbit weights", "Active orbit count", False),
        (columns["neff"], r"$N_{\rm eff}$", "Effective orbit count", True),
        (
            "max_weight_fraction",
            "maximum weight fraction",
            "Largest orbit share",
            True,
        ),
        ("alphat", r"$\alpha_t$", "Entropy multiplier", True),
        ("N_inner", "inner constraints", "Inner constraint count", False),
        ("N_outer", "outer constraints", "Outer constraint count", False),
        (columns["objective_value"], "reward", "Daemon reward", False),
    ]:
        if column:
            orbit_panels.append((column, label, title, y_log))

    plot_metric_strip(
        df,
        orbit_panels,
        "MBH_plot",
        r"$M_{\rm BH}\;(M_\odot)$",
        galaxy_label,
        csv_file,
        "orbit weights and objective diagnostics",
        x_log=True,
        mbh_floor=mbh_floor,
    )

    coverage_panels = [
        (
            "coverage_success_fraction",
            "successful / planned orbits",
            "Orbit-library success fraction",
            False,
        ),
        (
            "coverage_shell_min",
            "minimum shell coverage",
            "Shell coverage minimum",
            False,
        ),
        (
            "coverage_lfrac_min",
            "minimum L-fraction coverage",
            "L-fraction coverage minimum",
            False,
        ),
        (
            "coverage_theta_min",
            "minimum third-integral coverage",
            "Third-integral coverage minimum",
            False,
        ),
        (
            "coverage_joint_holes",
            "empty joint cells",
            "Joint coverage holes",
            False,
        ),
        (
            "successful_base_orbits",
            "successful base orbits",
            "Successful base orbits",
            False,
        ),
    ]

    plot_metric_strip(
        df,
        coverage_panels,
        "MBH_plot",
        r"$M_{\rm BH}\;(M_\odot)$",
        galaxy_label,
        csv_file,
        "orbit-library coverage diagnostics",
        x_log=True,
        mbh_floor=mbh_floor,
    )

    phase_volume_panels = [
        (
            "phase_volume_valid_fraction",
            "phase-volume valid / successful",
            "Phase-volume completeness",
            False,
        ),
        (
            "phase_volume_invalid_recorded_orbits",
            "invalid recorded orbits",
            "Invalid phase-volume records",
            False,
        ),
        (
            "phase_volume_duplicate_area_orbits",
            "duplicate-area orbits",
            "Duplicate phase-volume areas",
            False,
        ),
        (
            "raw_phase_volume_dynamic_range",
            "raw phase-volume dynamic range",
            "Raw phase-volume range",
            True,
        ),
        (
            "wphase_dynamic_range",
            "weight-phase dynamic range",
            "Weight phase-volume range",
            True,
        ),
        (
            "wphase_pair_max_relative_mismatch",
            "max pair relative mismatch",
            "Paired phase-volume mismatch",
            False,
        ),
    ]

    plot_metric_strip(
        df,
        phase_volume_panels,
        "MBH_plot",
        r"$M_{\rm BH}\;(M_\odot)$",
        galaxy_label,
        csv_file,
        "Karl phase-volume diagnostics",
        x_log=True,
        mbh_floor=mbh_floor,
    )

    phase_scale_panels = [
        (
            "raw_phase_volume_min",
            "raw phase-volume minimum",
            "Raw phase-volume minimum",
            True,
        ),
        (
            "raw_phase_volume_max",
            "raw phase-volume maximum",
            "Raw phase-volume maximum",
            True,
        ),
        (
            "normalized_phase_volume_min",
            "normalized phase-volume minimum",
            "Normalized minimum",
            True,
        ),
        (
            "normalized_phase_volume_max",
            "normalized phase-volume maximum",
            "Normalized maximum",
            True,
        ),
        (
            "wphase_min",
            "wphase minimum",
            "Inverse phase-volume weight minimum",
            True,
        ),
        (
            "wphase_max",
            "wphase maximum",
            "Inverse phase-volume weight maximum",
            True,
        ),
    ]

    plot_metric_strip(
        df,
        phase_scale_panels,
        "MBH_plot",
        r"$M_{\rm BH}\;(M_\odot)$",
        galaxy_label,
        csv_file,
        "phase-volume scale diagnostics",
        x_log=True,
        mbh_floor=mbh_floor,
    )

    print("\n[SHOW] Opening interactive Matplotlib windows. No files are written.")
    plt.show()


if __name__ == "__main__":
    main()