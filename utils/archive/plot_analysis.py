import argparse
import glob
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("dark_background")

# Short, wide defaults so dense panel strips stay readable.
LANDSCAPE_FIG_HEIGHT = 3.9
LANDSCAPE_FIG_WIDTH_PER_PANEL = 6.2
METRIC_FIG_HEIGHT = 3.7
METRIC_FIG_WIDTH_PER_PANEL = 6.0

# --------------------------------------------------
# Resolve repository and galaxy
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
WHICH_GALAXY = REPO_ROOT / "which_galaxy"
if WHICH_GALAXY.exists():
    GALAXY = WHICH_GALAXY.read_text().strip() or "unknown"
else:
    GALAXY = "unknown"
DEFAULT_PROFILE_DIR = ( REPO_ROOT/"Data"/"Galaxy_Profiles"/GALAXY/"default")

# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "csv_positional", nargs="?", default=None, help="Optional CSV path.")
    parser.add_argument( "--csv", default=None, help="CSV path, absolute or relative to the repository root.")
    parser.add_argument( "--profile-dir", default=None, help="Profile directory containing the model CSV.")
    parser.add_argument( "--pattern", default="*nonsingular_isothermal_full_light.csv", help="Glob pattern used when automatically selecting a CSV.")
    parser.add_argument( "--zoom-dchi", type=float, default=50.0, help="Keep models with objective less than best objective plus this value.")
    parser.add_argument( "--include-diagnostics", action="store_true", help="Include all finite statuses instead of pass_full only.")
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
    match = re.search( r"(\d{8})_(\d{6})", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1) + match.group(2))

def find_latest_csv(profile_dir, pattern):
    profile_dir = Path(profile_dir)
    if not profile_dir.exists():
        raise FileNotFoundError(f"Profile directory not found:\n  {profile_dir}")
    candidates = glob.glob(str(profile_dir / pattern))
    if not candidates:
        raise FileNotFoundError(f"No CSV matching {pattern!r} found in:\n  {profile_dir}")
    timestamped = [(timestamp_from_filename(path), path) for path in candidates]
    timestamped = [item for item in timestamped if item[0] is not None]
    if timestamped:
        return Path(max(timestamped, key=lambda item: item[0])[1])
    return Path(max(candidates, key=os.path.getmtime))


def infer_galaxy_from_csv(csv_file):
    csv_file = Path(csv_file).resolve()
    try:
        relative = csv_file.relative_to(REPO_ROOT / "Data" / "Galaxy_Profiles")
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
        raise KeyError( "Missing required column(s): " + ", ".join(missing) + "\nAvailable columns:\n  " + "\n  ".join(frame.columns))

def numeric_clean(frame, columns):
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric( frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)

def finite_positive_floor(values):
    values = pd.to_numeric(values, errors="coerce")
    positive = values[np.isfinite(values) & (values > 0)]
    if len(positive) == 0:
        return 1.0
    return positive.min() / 3.0

def filter_model_rows(frame, include_diagnostics):
    if "status" not in frame.columns:
        return frame.copy()
    status = frame["status"].astype(str)
    return frame[status.str.startswith("pass_")].copy()

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
        raise KeyError( "Missing halo-amplitude column. Expected v0, vcirc, or rho_s.")
    if "r_c" in frame.columns:
        radius_column = "r_c"
        radius_label = r"$r_c\;({\rm pc})$"
    elif "r_s" in frame.columns:
        radius_column = "r_s"
        radius_label = r"$r_s\;({\rm pc})$"
    else:
        raise KeyError("Missing halo-radius column. Expected r_c or r_s.")
    return ( halo_column, halo_label, halo_log, radius_column, radius_label)


# --------------------------------------------------
# Plot helpers
# --------------------------------------------------

def set_window_title(figure, title):
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)

def add_mbh_zero_marker(axis, mbh_floor):
    axis.axvline(mbh_floor, color="#4FA3FF", alpha=0.45, linestyle="--")
    axis.text(mbh_floor, 0.02, r" $M_{\rm BH}=0$", color="#4FA3FF", fontsize=8, rotation=90, va="bottom", transform=axis.get_xaxis_transform())

def scatter_split(axis, black_hole_models, no_black_hole_models, x_column, y_column, black_hole_size=18, no_black_hole_size=28, label=True):
    axis.scatter(black_hole_models[x_column], black_hole_models[y_column], s=black_hole_size, c="#39EB33", edgecolors="none", rasterized=True, label="Black-hole models" if label else None)
    axis.scatter(no_black_hole_models[x_column], no_black_hole_models[y_column], s=no_black_hole_size, c="#4FA3FF", edgecolors="none", rasterized=True, label=r"$M_{\rm BH}=0$" if label else None)

def plot_parameter_landscape(frame, objective_column, objective_label, parameters, mbh_floor, galaxy_label, csv_file, title_suffix):
    no_bh, with_bh = split_black_hole_models(frame, mbh_floor)
    figure, axes = plt.subplots( 1, 4, figsize=(LANDSCAPE_FIG_WIDTH_PER_PANEL * len(parameters), LANDSCAPE_FIG_HEIGHT), sharey=True)
    for axis, ( plot_column, axis_label, physical_column, use_log_scale) in zip(axes, parameters):
        scatter_split( axis, with_bh, no_bh, plot_column, objective_column)
        if use_log_scale:
            axis.set_xscale("log")
        axis.set_xlabel(axis_label)
        axis.set_ylabel(objective_label)
        axis.grid(alpha=0.18, linewidth=0.6)
        if physical_column == "MBH":
            add_mbh_zero_marker(axis, mbh_floor)
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle( f"{galaxy_label} OSPM {title_suffix}\n" f"source: {csv_file.name}", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    set_window_title(figure, f"OSPM {title_suffix}")

def plot_metric_strip(frame, metric_panels, x_column, x_label, galaxy_label, csv_file, window_title, x_log=False, mbh_floor=None):
    panels = [panel for panel in metric_panels if panel[0] in frame.columns and frame[panel[0]].notna().any()]
    if not panels:
        return None
    figure, axes = plt.subplots( 1, len(panels), figsize=( METRIC_FIG_WIDTH_PER_PANEL * len(panels), METRIC_FIG_HEIGHT), squeeze=False)
    axes = axes[0]

    if x_column == "MBH_plot":
        no_bh, with_bh = split_black_hole_models(frame, mbh_floor)

    for axis, (column, label, title, y_log) in zip(axes, panels):
        if x_column == "MBH_plot":
            scatter_split( axis, with_bh, no_bh, x_column, column, black_hole_size=20, no_black_hole_size=32, label=False)
            add_mbh_zero_marker(axis, mbh_floor)
        else:
            axis.plot(frame[x_column], frame[column], linestyle="none", marker="o", markersize=4.2, alpha=0.75)
        if x_log:
            axis.set_xscale("log")
        if y_log:
            positive = frame[column] > 0
            if positive.any():
                axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.grid(alpha=0.18, linewidth=0.6)

    figure.suptitle(f"{galaxy_label} OSPM {window_title}\n" f"source: {csv_file.name}", fontsize=13)
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
            if args.profile_dir else DEFAULT_PROFILE_DIR)
        csv_file = find_latest_csv(profile_dir, args.pattern)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found:\n  {csv_file}")
    galaxy_label = infer_galaxy_from_csv(csv_file)
    all_df = pd.read_csv(csv_file)
    (halo_param_col, halo_param_label, halo_log_scale, radius_param_col, radius_param_label,) = identify_halo_columns(all_df)

    aliases = { 
        "objective": ( "chi_total", "chi2_total", "chi2", "chi_losvd_score", "chi2_losvd"),
        "losvd_score": ( "chi_losvd_score", "chi2_losvd", "chi2",),
        "losvd_solver": ("chi_losvd_solver",),
        "light_score": ("chi_light_score", "chi2_light"),
        "light_solver": ("chi_light_solver",),
        "inner": ("chi2_inner", "chi_inner"),
        "outer": ("chi2_outer", "chi_outer"),
        "occupancy": ("chi2_occ", "chi_occ"),
        "nonzero": ("N_nonzero", "N_nonzero_weights"),
        "neff": ("Neff", "effective_N_orbits"),
        "objective_value": ("profit", "reward"),
    }

    columns = {name: first_existing(all_df, *choices) for name, choices in aliases.items()}
    objective_column = columns["objective"]
    if objective_column is None:
        raise KeyError("No objective column found. Expected chi_total, chi2_total, chi2, " "chi_losvd_score, or chi2_losvd.")

    required_columns = [objective_column, "MBH", halo_param_col, radius_param_col, "ML"]
    require_columns(all_df, required_columns)
    known_numeric_columns = { *required_columns, "chi_losvd_score", "chi_losvd_solver", "chi_light_score", "chi_light_solver", "chi_total", "chi_slack",
        "chi_slack_over_alphat", "slack_to_losvd", "chi2", "chi2_losvd", "chi2_light", "chi2_total", "chi2_inner", "chi2_outer", "chi2_occ", "entropy",
        "profit", "reward", "alphat", "slack_l2", "slack_max_abs", "rcond_est", "max_abs_dw", "N_slack", "N_nonzero", "N_nonzero_weights", "Neff",
        "effective_N_orbits", "max_weight_fraction", "N_inner", "N_outer", "refine_passes", "proposal_id"}

    df = numeric_clean(all_df, known_numeric_columns)
    df = df.dropna(subset=required_columns)
    df = filter_model_rows(df, args.include_diagnostics)

    if len(df) == 0:
        mode = ("all finite statuses" if args.include_diagnostics else "status == pass_full")
        raise ValueError(f"No usable rows remain after cleaning with {mode}.")
    df = df.copy()
    df["run_index"] = np.arange(1, len(df) + 1)
    losvd_score = columns["losvd_score"]
    losvd_solver = columns["losvd_solver"]
    if losvd_score and losvd_solver:
        df["chi_losvd_score_minus_solver"] = (df[losvd_score] - df[losvd_solver])
    light_score = columns["light_score"]
    light_solver = columns["light_solver"]
    if light_score and light_solver:
        df["chi_light_score_minus_solver"] = (df[light_score] - df[light_solver])
    best_index = df[objective_column].idxmin()
    best_row = df.loc[best_index]
    best_value = float(best_row[objective_column])
    zoom = df[ df[objective_column] <= best_value + args.zoom_dchi].copy()
    mbh_floor = finite_positive_floor(df["MBH"])

    for frame in (df, zoom):
        frame["MBH_plot"] = frame["MBH"]
        frame.loc[frame["MBH"] <= 0, "MBH_plot"] = mbh_floor
    objective_label = (r"$\chi^2_{\rm total}$" if objective_column in {"chi_total", "chi2_total"} else objective_column)
    parameters = [ ( "MBH_plot", r"$M_{\rm BH}\;(M_\odot)$", "MBH", True), ( halo_param_col, halo_param_label, halo_param_col, halo_log_scale), ( radius_param_col, radius_param_label, radius_param_col, True), ( "ML", r"$M/L$", "ML", False)]
    print(f"[PLOT] galaxy          = {galaxy_label}")
    print(f"[PLOT] using           = {csv_file}")
    print(f"[INFO] all CSV rows    = {len(all_df)}")
    print(f"[INFO] plotted rows    = {len(df)}")
    print(f"[INFO] objective       = {objective_column}")
    print(f"[INFO] best objective  = {best_value}")
    print(f"[INFO] zoom rows       = {len(zoom)}")
    print("\nBEST MODEL")
    best_columns = [ halo_param_col, radius_param_col, "MBH", "ML", objective_column]
    diagnostic_print_order = [ columns["losvd_score"], columns["losvd_solver"], columns["light_score"], columns["light_solver"], "chi_slack", "chi_slack_over_alphat",
        "slack_to_losvd", columns["inner"], columns["outer"], "rcond_est", "max_abs_dw", columns["nonzero"], columns["neff"], "max_weight_fraction", "entropy", columns["objective_value"]]
    for column in diagnostic_print_order:
        if (column and column in best_row.index and column not in best_columns): best_columns.append(column)
    print(best_row[best_columns].to_string())
    plot_parameter_landscape( df, objective_column, objective_label, parameters, mbh_floor, galaxy_label, csv_file, "full objective landscape")
    plot_parameter_landscape( zoom, objective_column, objective_label, parameters, mbh_floor, galaxy_label, csv_file, rf"zoomed objective landscape, $\Delta\chi^2\leq {args.zoom_dchi:g}$")

    chi_setup_panels = []

    for column, label, title in [
        (columns["losvd_score"], r"$\chi^2_{\rm LOSVD,score}$", "LOSVD score"),
        (columns["losvd_solver"], r"$\chi^2_{\rm LOSVD,solver}$", "LOSVD solver"),
        (columns["light_score"], r"$\chi^2_{\rm light,score}$", "Light score"),
        (columns["light_solver"], r"$\chi^2_{\rm light,solver}$", "Light solver"),
        (objective_column, objective_label, "Combined objective"),
        ( "chi_losvd_score_minus_solver", r"$\chi^2_{\rm LOSVD,score}-\chi^2_{\rm LOSVD,solver}$", "LOSVD score − solver"),
        ( "chi_light_score_minus_solver", r"$\chi^2_{\rm light,score}-\chi^2_{\rm light,solver}$", "Light score − solver"),
    ]:
        if column:
            chi_setup_panels.append((column, label, title, False))
    plot_metric_strip( df, chi_setup_panels, "run_index", "plotted model row", galaxy_label, csv_file, "χ² score and solver setup")
    slack_region_panels = [
        ("chi_slack", r"$\chi^2_{\rm slack}$", "Slack penalty", True),
        ("chi_slack_over_alphat", r"$\chi^2_{\rm slack}/\alpha_t$", "Slack / alphat", True),
        ("slack_to_losvd", "slack / LOSVD", "Slack-to-LOSVD ratio",True)]
    for column, label, title in [
        (columns["inner"], r"$\chi^2_{\rm inner}$", "Inner fit"),
        (columns["outer"], r"$\chi^2_{\rm outer}$", "Outer fit"),
        (columns["occupancy"], r"$\chi^2_{\rm occ}$", "Occupancy fit"),
    ]:
        if column:
            slack_region_panels.append((column, label, title, False))
    plot_metric_strip( df, slack_region_panels, "MBH_plot", r"$M_{\rm BH}\;(M_\odot)$", galaxy_label, csv_file, "slack and regional fit diagnostics", x_log=True, mbh_floor=mbh_floor)
    solver_panels = [
        ("rcond_est", "estimated rcond", "Conditioning", True),
        ("max_abs_dw", r"$\max|\Delta w|$", "Largest weight step", True),
        ("slack_l2", r"$\|s\|_2$", "Slack L2 norm", True),
        ("slack_max_abs", r"$\max|s|$", "Largest slack", True),
        ("N_slack", "slack variables", "Slack count", False),
    ]
    plot_metric_strip( df, solver_panels, "run_index", "plotted model row", galaxy_label, csv_file, "solver health")
    orbit_panels = []
    for column, label, title, y_log in [
        (columns["nonzero"], "active orbit weights", "Active orbit count", False),
        (columns["neff"], r"$N_{\rm eff}$", "Effective orbit count", True),
        ( "max_weight_fraction", "maximum weight fraction", "Largest orbit share", True),
        ("entropy", "entropy", "Entropy", False),
        (columns["objective_value"], "profit / reward", "Objective value", False),
        ("alphat", r"$\alpha_t$", "Entropy multiplier", True),
        ("N_inner", "inner constraints", "Inner constraint count", False),
        ("N_outer", "outer constraints", "Outer constraint count", False),
    ]:
        if column:
            orbit_panels.append((column, label, title, y_log))
    plot_metric_strip( df, orbit_panels, "MBH_plot", r"$M_{\rm BH}\;(M_\odot)$", galaxy_label, csv_file, "orbit weights and objective diagnostics", x_log=True, mbh_floor=mbh_floor)
    print("\n[SHOW] Opening interactive Matplotlib windows. No files are written.")
    plt.show()
if __name__ == "__main__":
    main()