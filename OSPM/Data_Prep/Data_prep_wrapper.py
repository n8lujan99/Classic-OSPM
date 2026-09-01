"""
Data_prep_wrapper.py

Run the config-driven OSPM galaxy data-prep pipeline.

Required observational inputs:
    CONFIG["DATA_PREP"]["surface_brightness"]["input_csv"]
    CONFIG["DATA_CSV"]

Derived products:
    CONFIG["SURFACE_BRIGHTNESS_CSV"]
    CONFIG["KINEMATIC_BINS_CSV"]
    <Galaxy>_abel_deprojection.csv
    STELLAR_MODEL["tracer_grid_csv"]
    STELLAR_MODEL["grid_csv"]

Pipeline:
    Surface_Brightness.py
        -> Observable_mapping.py build-losvd-bins
        -> Abel_deprojection.py
        -> Observable_mapping.py build-axisymmetric-light-grid
        -> Observable_mapping.py build-stellar-force-grid
        -> Plotting/plot_galaxy_observables.py

Galaxy-specific scientific choices come from the galaxy config.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# SHARED PIPELINE DEFAULTS
# ============================================================================

ABEL_N_GRID = 256
N_THETA = 64
CENTRAL_SHELLS = 32
MAKE_DIAGNOSTIC_PLOTS = True
CONFIRM_BEFORE_WRITE = True

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
DATA_PREP_DIR = REPO_ROOT / "OSPM" / "Data_Prep"
PLOTTING_DIR = REPO_ROOT / "Plotting"


# ============================================================================
# CONFIG / PATHS
# ============================================================================

def profile_root(galaxy: str) -> Path:
    return REPO_ROOT / "Data" / "Galaxy_Profiles" / galaxy

def config_path(galaxy: str) -> Path:
    return profile_root(galaxy) / f"{galaxy}_OSPM_Config.py"

def load_config(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Galaxy config does not exist: {path}")
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location("_ospm_data_prep_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "CONFIG", None)
    if not isinstance(cfg, dict):
        raise TypeError(f"{path} must define CONFIG as a dict")
    return cfg

def require_config(cfg: dict, key: str):
    if key not in cfg:
        raise KeyError(f"Galaxy config is missing required key: {key}")
    return cfg[key]

def path_from_config(value) -> Path:
    return Path(str(value)).expanduser().resolve()

def resolved_paths(galaxy: str, cfg: dict) -> dict[str, Path]:
    root = profile_root(galaxy)
    stellar = require_config(cfg, "STELLAR_MODEL")
    if not isinstance(stellar, dict):
        raise TypeError("CONFIG['STELLAR_MODEL'] must be a dict")
    return {
        "root": root,
        "config": config_path(galaxy),
        "prepared_sb": path_from_config(require_config(cfg, "SURFACE_BRIGHTNESS_CSV")),
        "kinematic_bins": path_from_config(require_config(cfg, "KINEMATIC_BINS_CSV")),
        "plots": root / "plots",
        "abel": root / "plots" / f"{galaxy}_abel_deprojection.csv",
        "tracer": path_from_config(stellar["tracer_grid_csv"]),
        "force": path_from_config(stellar["grid_csv"]),
        "manifest": root / f"{galaxy}_data_prep_manifest.txt",
    }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_losvd_stars(cfg: dict):
    data_path = path_from_config(require_config(cfg, "DATA_CSV"))
    if not data_path.is_file():
        raise FileNotFoundError(f"LOSVD stellar-data file does not exist: {data_path}")
    stars = pd.read_csv(data_path)
    r_col = str(require_config(cfg, "STAR_R_COL"))
    v_col = str(require_config(cfg, "STAR_V_COL"))
    verr_col = str(require_config(cfg, "STAR_VERR_COL"))
    required = {r_col, v_col, verr_col}
    missing = required - set(stars.columns)
    if missing:
        raise KeyError(f"LOSVD stellar-data file missing columns: {sorted(missing)}")
    r = pd.to_numeric(stars[r_col], errors="coerce").to_numpy(float)
    v = pd.to_numeric(stars[v_col], errors="coerce").to_numpy(float)
    verr = pd.to_numeric(stars[verr_col], errors="coerce").to_numpy(float)
    good = np.isfinite(r) & np.isfinite(v) & np.isfinite(verr) & (r >= 0.0) & (verr > 0.0)
    if np.count_nonzero(good) < 2:
        raise ValueError("LOSVD input has fewer than two usable stars")
    vsys = float(require_config(cfg, "V_SYS_KMS"))
    vrel = v[good] - vsys
    return {
        "path": data_path,
        "n": int(np.count_nonzero(good)),
        "r_min": float(np.min(r[good])),
        "r_max": float(np.max(r[good])),
        "v_min": float(np.min(vrel)),
        "v_max": float(np.max(vrel)),
        "verr_min": float(np.min(verr[good])),
        "verr_max": float(np.max(verr[good])),
    }

def validate_generated_kinematic_bins(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Generated LOSVD/kinematic-bin file does not exist: {path}")
    bins = pd.read_csv(path)
    required = {"bin_id", "R_inner_pc", "R_outer_pc", "R_mid_pc", "N_vlos"}
    missing = required - set(bins.columns)
    if missing:
        raise KeyError(f"Generated LOSVD/kinematic-bin file missing columns: {sorted(missing)}")
    if len(bins) < 1:
        raise ValueError("Generated LOSVD/kinematic-bin file is empty")
    rin = pd.to_numeric(bins["R_inner_pc"], errors="coerce").to_numpy(float)
    rout = pd.to_numeric(bins["R_outer_pc"], errors="coerce").to_numpy(float)
    n_vlos = pd.to_numeric(bins["N_vlos"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(rin)) or not np.all(np.isfinite(rout)):
        raise ValueError("Generated LOSVD bin edges contain non-finite values")
    if np.any(rout <= rin):
        raise ValueError("Generated LOSVD bins require R_outer_pc > R_inner_pc")
    if not np.all(np.isfinite(n_vlos)) or np.any(n_vlos <= 0.0):
        raise ValueError("Generated LOSVD bins require positive finite N_vlos")
    if len(bins) > 1:
        scale = max(1.0, float(np.max(np.abs(np.r_[rin, rout]))))
        if np.any(np.abs(rin[1:] - rout[:-1]) > 1e-10 * scale):
            raise ValueError("Generated LOSVD radial bins must be contiguous")
    return bins

def validate_data_prep_config(cfg: dict):
    data_prep = require_config(cfg, "DATA_PREP")
    if not isinstance(data_prep, dict):
        raise TypeError("CONFIG['DATA_PREP'] must be a dict")
    surface_brightness = data_prep.get("surface_brightness")
    if not isinstance(surface_brightness, dict):
        raise TypeError("CONFIG['DATA_PREP']['surface_brightness'] must be a dict")
    required_sb = {"input_csv", "input_type", "source", "preferred_profile", "radius_type", "background", "background_err"}
    missing_sb = required_sb - set(surface_brightness)
    if missing_sb:
        raise KeyError(f"CONFIG['DATA_PREP']['surface_brightness'] missing keys: {sorted(missing_sb)}")
    surface_input = path_from_config(surface_brightness["input_csv"])
    if not surface_input.is_file():
        raise FileNotFoundError(f"Surface-brightness observational input does not exist: {surface_input}")

    losvd_bins = data_prep.get("losvd_bins")
    if not isinstance(losvd_bins, dict):
        raise TypeError("CONFIG['DATA_PREP']['losvd_bins'] must be a dict")
    mode = str(losvd_bins.get("mode", "")).strip().lower()
    if mode == "min_count":
        if "min_stars" not in losvd_bins:
            raise KeyError("DATA_PREP losvd_bins mode=min_count requires min_stars")
        min_stars = int(losvd_bins["min_stars"])
        if min_stars < 1:
            raise ValueError("DATA_PREP losvd_bins min_stars must be at least 1")
        losvd_controls = {"mode": mode, "min_stars": min_stars, "drop_partial": bool(losvd_bins.get("drop_partial", False)), "n_bins": None}
    elif mode == "equal_count":
        if "n_bins" not in losvd_bins:
            raise KeyError("DATA_PREP losvd_bins mode=equal_count requires n_bins")
        n_bins = int(losvd_bins["n_bins"])
        if n_bins < 1:
            raise ValueError("DATA_PREP losvd_bins n_bins must be at least 1")
        losvd_controls = {"mode": mode, "min_stars": None, "drop_partial": False, "n_bins": n_bins}
    else:
        raise ValueError("DATA_PREP losvd_bins mode must be 'min_count' or 'equal_count'")

    required_abel = {"abel_smoothing_target", "abel_outer_transition_sigma", "abel_outer_tail_points"}
    missing_abel_keys = required_abel - set(data_prep)
    if missing_abel_keys:
        raise KeyError(f"CONFIG['DATA_PREP'] missing keys: {sorted(missing_abel_keys)}")

    controls = {
        "surface_brightness": surface_brightness,
        "surface_brightness_input": surface_input,
        "losvd_bins": losvd_controls,
        "smoothing_target": data_prep["abel_smoothing_target"],
        "outer_transition_sigma": data_prep["abel_outer_transition_sigma"],
        "outer_tail_points": data_prep["abel_outer_tail_points"],
    }
    missing_values = []
    if controls["smoothing_target"] is None:
        missing_values.append("abel_smoothing_target")
    elif not math.isfinite(float(controls["smoothing_target"])) or float(controls["smoothing_target"]) < 0.0:
        raise ValueError("DATA_PREP abel_smoothing_target must be finite and non-negative")
    if controls["outer_transition_sigma"] is None:
        missing_values.append("abel_outer_transition_sigma")
    elif not math.isfinite(float(controls["outer_transition_sigma"])) or float(controls["outer_transition_sigma"]) <= 0.0:
        raise ValueError("DATA_PREP abel_outer_transition_sigma must be positive and finite")
    if controls["outer_tail_points"] is None:
        missing_values.append("abel_outer_tail_points")
    elif int(controls["outer_tail_points"]) < 3:
        raise ValueError("DATA_PREP abel_outer_tail_points must be at least 3")
    controls["missing"] = missing_values
    return controls

def validate_config_contract(cfg: dict):
    distance_pc = float(require_config(cfg, "DISTANCE_PC"))
    q = float(require_config(cfg, "AXIS_RATIO_Q"))
    if not math.isfinite(distance_pc) or distance_pc <= 0.0:
        raise ValueError("CONFIG['DISTANCE_PC'] must be positive and finite")
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("CONFIG['AXIS_RATIO_Q'] must be positive and finite")
    if str(require_config(cfg, "TRACER_CONSTRAINT_MODE")).strip().lower() != "density_3d":
        raise ValueError("This wrapper expects TRACER_CONSTRAINT_MODE='density_3d'")
    if str(require_config(cfg, "LOSVD_TARGET_MODE")).strip().lower() != "karl_resolved_stars":
        raise ValueError("This wrapper expects LOSVD_TARGET_MODE='karl_resolved_stars'")
    stellar = require_config(cfg, "STELLAR_MODEL")
    if not isinstance(stellar, dict):
        raise TypeError("CONFIG['STELLAR_MODEL'] must be a dict")
    ltot = float(stellar["Ltot"])
    q_stellar = float(stellar["q_axis_ratio"])
    if not math.isfinite(ltot) or ltot <= 0.0:
        raise ValueError("STELLAR_MODEL['Ltot'] must be positive and finite")
    if not np.isclose(q_stellar, q, rtol=1e-12, atol=1e-12):
        raise ValueError(f"STELLAR_MODEL q={q_stellar} disagrees with CONFIG AXIS_RATIO_Q={q}")
    return {"distance_pc": distance_pc, "q": q, "ltot": ltot}

def check_kde_support(cfg: dict, losvd: dict):
    lo = cfg.get("KARL_RESOLVED_VMIN_KMS")
    hi = cfg.get("KARL_RESOLVED_VMAX_KMS")
    if lo is None or hi is None:
        return (
            "WARNING: config has no KARL_RESOLVED_VMIN_KMS/KARL_RESOLVED_VMAX_KMS. "
            f"Observed systemic-centered range is [{losvd['v_min']:.12g}, {losvd['v_max']:.12g}] km/s."
        )
    lo = float(lo)
    hi = float(hi)
    if lo > losvd["v_min"] or hi < losvd["v_max"]:
        return (
            "WARNING: configured resolved-star KDE support does not contain the full observed velocity range.\n"
            f"  configured = [{lo:.12g}, {hi:.12g}] km/s\n"
            f"  observed   = [{losvd['v_min']:.12g}, {losvd['v_max']:.12g}] km/s\n"
            "Review KARL_RESOLVED_VMIN_KMS/KARL_RESOLVED_VMAX_KMS before the integration run."
        )
    return (
        "Resolved-star KDE support contains the observed systemic-centered velocity range:\n"
        f"  configured = [{lo:.12g}, {hi:.12g}] km/s\n"
        f"  observed   = [{losvd['v_min']:.12g}, {losvd['v_max']:.12g}] km/s"
    )


# ============================================================================
# COMMAND BUILDING / EXECUTION
# ============================================================================

def fmt_command(cmd: list[str]) -> str:
    parts = []
    for value in cmd:
        text = str(value)
        if any(ch.isspace() for ch in text):
            parts.append(repr(text))
        else:
            parts.append(text)
    return " ".join(parts)

def run_command(label: str, cmd: list[str]):
    print()
    print("=" * 78)
    print(label)
    print("=" * 78)
    print(fmt_command(cmd))
    print()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

def build_losvd_bin_command(cfg: dict, paths: dict[str, Path], prep: dict) -> list[str]:
    controls = prep["losvd_bins"]
    cmd = [
        sys.executable,
        str(DATA_PREP_DIR / "Observable_mapping.py"),
        "build-losvd-bins",
        "--stars", str(path_from_config(require_config(cfg, "DATA_CSV"))),
        "--out", str(paths["kinematic_bins"]),
        "--radius-col", str(require_config(cfg, "STAR_R_COL")),
        "--velocity-col", str(require_config(cfg, "STAR_V_COL")),
        "--velocity-error-col", str(require_config(cfg, "STAR_VERR_COL")),
        "--mode", controls["mode"],
    ]
    if controls["mode"] == "min_count":
        cmd.extend(["--min-stars", str(controls["min_stars"])])
        if controls["drop_partial"]:
            cmd.append("--drop-partial-bins")
    else:
        cmd.extend(["--n-bins", str(controls["n_bins"])])
    return cmd

def build_commands(galaxy: str, cfg: dict, paths: dict[str, Path], contract: dict, prep: dict, make_plots: bool):
    surface_cmd = [sys.executable, str(DATA_PREP_DIR / "Surface_Brightness.py"), "--galaxy", galaxy]
    if not make_plots:
        surface_cmd.append("--no-plots")
    commands = [
        ("PREPARE SURFACE BRIGHTNESS", surface_cmd),
        ("BUILD LOSVD RADIAL BINS", build_losvd_bin_command(cfg, paths, prep)),
    ]
    if prep["missing"]:
        return commands
    commands.extend([
        (
            "ABEL DEPROJECTION",
            [
                sys.executable, str(DATA_PREP_DIR / "Abel_deprojection.py"),
                "--galaxy", galaxy,
                "--surface-brightness", str(paths["prepared_sb"]),
                "--outdir", str(paths["plots"]),
                "--n-grid", str(ABEL_N_GRID),
                "--smoothing-target", f"{float(prep['smoothing_target']):.17g}",
                "--outer-transition-sigma", f"{float(prep['outer_transition_sigma']):.17g}",
                "--outer-tail-points", str(int(prep["outer_tail_points"])),
            ],
        ),
        (
            "BUILD TRACER GRID",
            [
                sys.executable, str(DATA_PREP_DIR / "Observable_mapping.py"), "build-axisymmetric-light-grid",
                "--surface-brightness", str(paths["prepared_sb"]),
                "--density-profile", str(paths["abel"]),
                "--out", str(paths["tracer"]),
                "--ltot", f"{contract['ltot']:.17g}",
                "--q-axis-ratio", f"{contract['q']:.17g}",
                "--n-theta", str(N_THETA),
            ],
        ),
        (
            "BUILD STELLAR FORCE GRID",
            [
                sys.executable, str(DATA_PREP_DIR / "Observable_mapping.py"), "build-stellar-force-grid",
                "--surface-brightness", str(paths["prepared_sb"]),
                "--density-profile", str(paths["abel"]),
                "--out", str(paths["force"]),
                "--ltot", f"{contract['ltot']:.17g}",
                "--q-axis-ratio", f"{contract['q']:.17g}",
                "--n-theta", str(N_THETA),
                "--central-shells", str(CENTRAL_SHELLS),
            ],
        ),
    ])
    if make_plots:
        commands.append((
            "BUILD GALAXY DIAGNOSTIC PLOTS",
            [sys.executable, str(PLOTTING_DIR / "plot_galaxy_observables.py"), "--galaxy", galaxy],
        ))
    return commands


# ============================================================================
# REPORTING
# ============================================================================

def write_manifest(galaxy: str, paths: dict[str, Path], contract: dict, losvd: dict, prep: dict, commands: list[tuple[str, list[str]]]):
    sb = prep["surface_brightness"]
    bins = prep["losvd_bins"]
    lines = [
        "OSPM GALAXY DATA-PREP MANIFEST",
        "==============================",
        f"Galaxy: {galaxy}",
        f"Config: {paths['config']}",
        "",
        "OBSERVATIONAL INPUTS",
        "--------------------",
        f"Surface brightness: {prep['surface_brightness_input']}",
        f"LOSVD stellar data: {losvd['path']}",
        "",
        "SURFACE-BRIGHTNESS PREP",
        "-----------------------",
        f"Input type:        {sb['input_type']}",
        f"Source:            {sb['source']}",
        f"Preferred profile: {sb['preferred_profile']}",
        f"Radius type:       {sb['radius_type']}",
        f"Background:        {sb['background']}",
        f"Background error:  {sb['background_err']}",
        "",
        "LOSVD RADIAL BINNING",
        "--------------------",
        f"Mode:              {bins['mode']}",
        f"Minimum stars:     {bins['min_stars']}",
        f"Equal-count bins:  {bins['n_bins']}",
        f"Drop partial:      {bins['drop_partial']}",
        f"Output:            {paths['kinematic_bins']}",
        "",
        "GALAXY VALUES FROM CONFIG",
        "-------------------------",
        f"Distance [pc]: {contract['distance_pc']:.17g}",
        f"q:             {contract['q']:.17g}",
        f"Ltot [Lsun]:   {contract['ltot']:.17g}",
        "",
        "ABEL CHOICES FROM CONFIG['DATA_PREP']",
        "---------------------------------------",
        f"Smoothing target:       {prep['smoothing_target']}",
        f"Outer transition sigma: {prep['outer_transition_sigma']}",
        f"Outer tail points:      {prep['outer_tail_points']}",
        f"Abel radial grid:       {ABEL_N_GRID}",
        "",
        "SHARED GRID DEFAULTS",
        "--------------------",
        f"N theta:                  {N_THETA}",
        f"Unresolved-center shells: {CENTRAL_SHELLS}",
        "",
        "LOSVD SUMMARY",
        "-------------",
        f"Usable stars:            {losvd['n']}",
        f"Projected R range [pc]:  {losvd['r_min']:.17g} -> {losvd['r_max']:.17g}",
        f"Relative v range [km/s]: {losvd['v_min']:.17g} -> {losvd['v_max']:.17g}",
        f"Velocity-error range:    {losvd['verr_min']:.17g} -> {losvd['verr_max']:.17g}",
        "",
        "GENERATED PRODUCTS",
        "------------------",
        f"Prepared surface brightness: {paths['prepared_sb']}",
        f"LOSVD radial bins:           {paths['kinematic_bins']}",
        f"Abel density:                {paths['abel']}",
        f"Tracer density grid:         {paths['tracer']}",
        f"Stellar force grid:          {paths['force']}",
        "",
        "COMMANDS",
        "--------",
    ]
    for label, cmd in commands:
        lines.append(label)
        lines.append(fmt_command(cmd))
        lines.append("")
    paths["manifest"].write_text("\n".join(lines) + "\n")
    print(f"Manifest: {paths['manifest']}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run the config-driven OSPM galaxy data-prep pipeline.")
    parser.add_argument("--galaxy", required=True, help="Galaxy profile directory/config name, for example Draco or Segue1.")
    parser.add_argument("--check", action="store_true", help="Validate observational inputs/config and print the plan without writing products.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt.")
    parser.add_argument("--no-plots", action="store_true", help="Generate data products but skip surface-brightness and final diagnostic plots.")
    return parser.parse_args()

def main():
    args = parse_args()
    galaxy = str(args.galaxy).strip()
    if not galaxy:
        raise ValueError("--galaxy cannot be empty")
    root = profile_root(galaxy)
    if not root.is_dir():
        raise FileNotFoundError(f"Galaxy profile directory does not exist: {root}")

    cfg = load_config(config_path(galaxy))
    paths = resolved_paths(galaxy, cfg)
    contract = validate_config_contract(cfg)
    prep = validate_data_prep_config(cfg)
    losvd = validate_losvd_stars(cfg)

    print()
    print("=" * 78)
    print(f"OSPM DATA PREP: {galaxy}")
    print("=" * 78)
    print(f"Repo root:             {REPO_ROOT}")
    print(f"Galaxy root:           {paths['root']}")
    print(f"Config:                {paths['config']}")
    print(f"Surface-brightness in: {prep['surface_brightness_input']}")
    print(f"Surface input type:    {prep['surface_brightness']['input_type']}")
    print(f"LOSVD stars:           {losvd['path']}")
    print(f"LOSVD bins out:        {paths['kinematic_bins']}")
    print(f"Distance [pc]:         {contract['distance_pc']}")
    print(f"q:                     {contract['q']}")
    print(f"Ltot [Lsun]:           {contract['ltot']}")
    print(f"LOSVD usable stars:    {losvd['n']}")
    print(f"LOSVD vrel range:      {losvd['v_min']:.6f} -> {losvd['v_max']:.6f} km/s")
    print()
    print(check_kde_support(cfg, losvd))

    print()
    print("LOSVD BINNING")
    print("-------------")
    print(f"mode:          {prep['losvd_bins']['mode']}")
    print(f"min_stars:     {prep['losvd_bins']['min_stars']}")
    print(f"n_bins:        {prep['losvd_bins']['n_bins']}")
    print(f"drop_partial:  {prep['losvd_bins']['drop_partial']}")

    print()
    print("ABEL DATA_PREP")
    print("--------------")
    print(f"abel_smoothing_target:       {prep['smoothing_target']}")
    print(f"abel_outer_transition_sigma: {prep['outer_transition_sigma']}")
    print(f"abel_outer_tail_points:      {prep['outer_tail_points']}")

    make_plots = MAKE_DIAGNOSTIC_PLOTS and not args.no_plots
    commands = build_commands(galaxy, cfg, paths, contract, prep, make_plots)

    print()
    print("PLANNED OUTPUTS")
    print("---------------")
    print(paths["prepared_sb"])
    print(paths["kinematic_bins"])
    if not prep["missing"]:
        print(paths["abel"])
        print(paths["tracer"])
        print(paths["force"])

    print()
    print("PLANNED STAGES")
    print("--------------")
    for label, cmd in commands:
        print(label)
        print("  " + fmt_command(cmd))

    if prep["missing"]:
        print()
        print("=" * 78)
        print("ABEL SETTINGS REQUIRED BEFORE THE FULL PIPELINE CAN RUN")
        print("=" * 78)
        for name in prep["missing"]:
            print(f"  CONFIG['DATA_PREP']['{name}'] = None")
        print()
        print("Surface brightness and LOSVD radial bins can still be generated, but the wrapper will stop before Abel until those choices are set.")

    if args.check:
        print()
        print("CHECK ONLY: no files were written and no data-prep commands were run.")
        return

    if CONFIRM_BEFORE_WRITE and not args.yes:
        print()
        answer = input(f"Run the listed data-prep stages for {galaxy}? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled. No data-prep stages were run.")
            return

    run_command(*commands[0])
    run_command(*commands[1])

    generated_bins = validate_generated_kinematic_bins(paths["kinematic_bins"])
    print()
    print("GENERATED LOSVD BINS")
    print("--------------------")
    print(f"N bins:      {len(generated_bins)}")
    print(f"N_vlos:      {generated_bins['N_vlos'].astype(int).tolist()}")
    print(f"N_vlos sum:  {int(generated_bins['N_vlos'].sum())}")

    if prep["missing"]:
        print()
        print("Surface-brightness and LOSVD-bin preparation finished.")
        print("Full pipeline stopped intentionally because DATA_PREP Abel choices are unset.")
        return

    for label, cmd in commands[2:]:
        run_command(label, cmd)

    write_manifest(galaxy, paths, contract, losvd, prep, commands)

    print()
    print("=" * 78)
    print(f"{galaxy} DATA PREP COMPLETE")
    print("=" * 78)
    print("The observational surface-brightness and stellar data were preserved.")
    print("All configured derived prep products were regenerated through the shared pipeline.")

if __name__ == "__main__":
    main()
