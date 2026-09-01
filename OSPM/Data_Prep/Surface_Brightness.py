"""
surface_brightness.py

Generic OSPM surface-brightness / projected-density preparation.

This replaces galaxy-specific surface-brightness preparation scripts. The
observational measurements remain in external CSV files; galaxy-specific
interpretation and provenance live in CONFIG["DATA_PREP"]["surface_brightness"].

The script currently supports the two observational input forms used by Draco
and Segue 1:

    surface_density_annuli
        Explicit annulus edges/midpoints plus Sigma and Sigma uncertainty.

    fixed_annulus_counts
        Digitized or tabulated count measurements at fixed-annulus centers.
        Exact contiguous annulus edges are reconstructed from the configured
        annulus width, then counts are converted to projected density.

Galaxy geometry and distance are not duplicated in DATA_PREP. They come from:

    CONFIG["DISTANCE_PC"]
    CONFIG["AXIS_RATIO_Q"]

The prepared output path comes from:

    CONFIG["SURFACE_BRIGHTNESS_CSV"]

No galaxy measurements or galaxy-specific constants are embedded in this file.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REVIEW_LOW_SN_SIGMA = 2.0
REVIEW_RISE_SIGMA = 2.0
REVIEW_OUTER_DETECTION_SIGMA = 3.0
RADIAL_EDGE_RTOL = 1e-10
RADIAL_EDGE_ATOL = 1e-12


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]


# ============================================================================
# CONFIG
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

    spec = importlib.util.spec_from_file_location("_ospm_surface_brightness_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import config: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = getattr(module, "CONFIG", None)
    if not isinstance(cfg, dict):
        raise TypeError(f"{path} must define CONFIG as a dict")

    return cfg


def require(mapping: dict, key: str, context: str):
    if key not in mapping:
        raise KeyError(f"{context} is missing required key: {key}")
    return mapping[key]


def load_surface_brightness_settings(cfg: dict) -> dict:
    data_prep = require(cfg, "DATA_PREP", "CONFIG")
    if not isinstance(data_prep, dict):
        raise TypeError("CONFIG['DATA_PREP'] must be a dict")

    sb = require(data_prep, "surface_brightness", "CONFIG['DATA_PREP']")
    if not isinstance(sb, dict):
        raise TypeError("CONFIG['DATA_PREP']['surface_brightness'] must be a dict")

    settings = dict(sb)

    for key in (
        "input_csv",
        "input_type",
        "source",
        "preferred_profile",
        "radius_type",
        "background",
        "background_err",
    ):
        require(settings, key, "CONFIG['DATA_PREP']['surface_brightness']")

    settings["input_type"] = str(settings["input_type"]).strip().lower()
    settings["source"] = str(settings["source"]).strip()
    settings["preferred_profile"] = str(settings["preferred_profile"]).strip()
    settings["radius_type"] = str(settings["radius_type"]).strip()
    settings["background"] = float(settings["background"])
    settings["background_err"] = float(settings["background_err"])
    settings["sigma_units"] = str(settings.get("sigma_units", "stars_per_arcmin2")).strip()
    settings["note"] = settings.get("note")

    if settings["input_type"] not in {"surface_density_annuli", "fixed_annulus_counts"}:
        raise ValueError(
            "surface_brightness input_type must be 'surface_density_annuli' "
            "or 'fixed_annulus_counts'"
        )

    if not settings["source"]:
        raise ValueError("surface_brightness source cannot be empty")
    if not settings["preferred_profile"]:
        raise ValueError("surface_brightness preferred_profile cannot be empty")
    if not settings["radius_type"]:
        raise ValueError("surface_brightness radius_type cannot be empty")
    if not math.isfinite(settings["background"]):
        raise ValueError("surface_brightness background must be finite")
    if not math.isfinite(settings["background_err"]) or settings["background_err"] < 0.0:
        raise ValueError("surface_brightness background_err must be finite and non-negative")

    return settings


# ============================================================================
# UNIT HELPERS
# ============================================================================

def angle_to_arcmin(value, units: str):
    units = str(units).strip().lower()

    if units in {"arcmin", "arcminute", "arcminutes"}:
        return np.asarray(value, dtype=float)
    if units in {"deg", "degree", "degrees"}:
        return 60.0 * np.asarray(value, dtype=float)
    if units in {"arcsec", "arcsecond", "arcseconds"}:
        return np.asarray(value, dtype=float) / 60.0

    raise ValueError(f"Unsupported angular radius units: {units}")


def scalar_angle_to_arcmin(value: float, units: str) -> float:
    return float(angle_to_arcmin([value], units)[0])


# ============================================================================
# OBSERVATIONAL INPUT NORMALIZATION
# ============================================================================

def numeric_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        raise KeyError(f"Surface-brightness input missing column: {name}")

    values = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(values)):
        bad = np.where(~np.isfinite(values))[0]
        raise ValueError(f"Column {name!r} contains non-finite values at rows {bad.tolist()}")

    return values


def normalize_surface_density_annuli(raw: pd.DataFrame, settings: dict) -> pd.DataFrame:
    rin_col = str(settings.get("rin_col", "rin_arcmin"))
    rout_col = str(settings.get("rout_col", "rout_arcmin"))
    rmid_col = str(settings.get("rmid_col", "rm_arcmin"))
    sigma_col = str(settings.get("sigma_col", "Sigma"))
    sigma_err_col = str(settings.get("sigma_err_col", "Sigma_err"))
    radius_units = str(settings.get("radius_units", "arcmin"))

    rin = angle_to_arcmin(numeric_column(raw, rin_col), radius_units)
    rout = angle_to_arcmin(numeric_column(raw, rout_col), radius_units)
    rmid = angle_to_arcmin(numeric_column(raw, rmid_col), radius_units)
    sigma = numeric_column(raw, sigma_col)
    sigma_err = numeric_column(raw, sigma_err_col)

    if np.any(rin < 0.0):
        raise ValueError("Surface-density annulus inner radii must be non-negative")
    if np.any(rout <= rin):
        raise ValueError("Surface-density annulus outer radii must exceed inner radii")
    if np.any((rmid <= rin) | (rmid >= rout)):
        raise ValueError("Surface-density annulus midpoint must lie inside its annulus")
    if np.any(sigma_err < 0.0):
        raise ValueError("Surface-density uncertainties must be non-negative")

    out = pd.DataFrame({
        "source_row": np.arange(len(raw), dtype=int),
        "rin_arcmin": rin,
        "rout_arcmin": rout,
        "rm_arcmin": rmid,
        "Sigma_source_arcmin2": sigma,
        "Sigma_source_err_arcmin2": sigma_err,
    })

    return out


def normalize_fixed_annulus_counts(raw: pd.DataFrame, settings: dict, q_axis_ratio: float) -> pd.DataFrame:
    center_col = str(settings.get("center_col", "digitized_center_deg"))
    count_col = str(settings.get("count_col", "count"))
    count_err_col = settings.get("count_err_col")
    center_units = str(settings.get("center_units", "deg"))

    annulus_width = float(require(
        settings,
        "annulus_width",
        "CONFIG['DATA_PREP']['surface_brightness']",
    ))
    annulus_width_units = str(settings.get("annulus_width_units", center_units))
    round_counts = bool(settings.get("round_counts", False))
    count_error_model = str(settings.get("count_error_model", "poisson")).strip().lower()

    if not math.isfinite(annulus_width) or annulus_width <= 0.0:
        raise ValueError("surface_brightness annulus_width must be positive and finite")

    center_arcmin = angle_to_arcmin(numeric_column(raw, center_col), center_units)
    width_arcmin = scalar_angle_to_arcmin(annulus_width, annulus_width_units)

    raw_counts = numeric_column(raw, count_col)
    if np.any(raw_counts <= 0.0):
        raise ValueError("Fixed-annulus counts must be positive")

    if round_counts:
        counts = np.rint(raw_counts).astype(int).astype(float)
        if np.any(counts <= 0.0):
            raise ValueError("Rounded fixed-annulus counts must be positive")
    else:
        counts = raw_counts.astype(float)

    if count_err_col is not None:
        count_err = numeric_column(raw, str(count_err_col))
        if np.any(count_err < 0.0):
            raise ValueError("Count uncertainties must be non-negative")
        count_error_source = f"column:{count_err_col}"
    else:
        if count_error_model != "poisson":
            raise ValueError(
                "Without count_err_col, fixed_annulus_counts currently supports "
                "count_error_model='poisson' only"
            )
        count_err = np.sqrt(counts)
        count_error_source = "poisson_sqrt_count"

    order = np.argsort(center_arcmin)
    center_arcmin = center_arcmin[order]
    raw_counts = raw_counts[order]
    counts = counts[order]
    count_err = count_err[order]
    source_rows = np.arange(len(raw), dtype=int)[order]

    bin_id = np.rint(center_arcmin / width_arcmin - 0.5).astype(int)
    if np.any(bin_id < 0):
        raise ValueError("At least one fixed-annulus center maps to a negative annulus index")

    expected_center_arcmin = (bin_id + 0.5) * width_arcmin
    center_offset_arcmin = center_arcmin - expected_center_arcmin

    if np.any(np.abs(center_offset_arcmin) >= 0.5 * width_arcmin):
        bad = np.where(np.abs(center_offset_arcmin) >= 0.5 * width_arcmin)[0]
        raise ValueError(
            "At least one measured center cannot be assigned uniquely to the "
            f"configured fixed annuli; bad rows={bad.tolist()}"
        )

    if len(np.unique(bin_id)) != len(bin_id):
        raise ValueError("Multiple count measurements map to the same fixed annulus")

    expected_ids = np.arange(bin_id[0], bin_id[-1] + 1, dtype=int)
    if not np.array_equal(bin_id, expected_ids):
        raise ValueError(
            "Fixed-annulus measurements do not form one contiguous annulus sequence"
        )

    if bin_id[0] != 0:
        raise ValueError(
            "Fixed-annulus profile must begin with the central annulus; "
            f"first mapped bin is {bin_id[0]}"
        )

    rin_arcmin = bin_id * width_arcmin
    rout_arcmin = (bin_id + 1) * width_arcmin
    rmid_arcmin = (bin_id + 0.5) * width_arcmin

    area_arcmin2 = math.pi * q_axis_ratio * (rout_arcmin**2 - rin_arcmin**2)
    sigma = counts / area_arcmin2
    sigma_err = count_err / area_arcmin2

    out = pd.DataFrame({
        "source_row": source_rows,
        "bin_id": bin_id,
        "measured_center_arcmin": center_arcmin,
        "expected_center_arcmin": expected_center_arcmin,
        "center_offset_arcmin": center_offset_arcmin,
        "count_input": raw_counts,
        "count": counts,
        "count_err": count_err,
        "count_error_source": count_error_source,
        "rin_arcmin": rin_arcmin,
        "rout_arcmin": rout_arcmin,
        "rm_arcmin": rmid_arcmin,
        "Sigma_source_arcmin2": sigma,
        "Sigma_source_err_arcmin2": sigma_err,
    })

    out.attrs["annulus_width_arcmin"] = width_arcmin
    out.attrs["max_center_offset_arcmin"] = float(np.max(np.abs(center_offset_arcmin)))

    return out


def normalize_observational_input(raw: pd.DataFrame, settings: dict, q_axis_ratio: float) -> pd.DataFrame:
    if len(raw) == 0:
        raise ValueError("Surface-brightness input CSV contains no data rows")

    if settings["input_type"] == "surface_density_annuli":
        return normalize_surface_density_annuli(raw, settings)

    if settings["input_type"] == "fixed_annulus_counts":
        return normalize_fixed_annulus_counts(raw, settings, q_axis_ratio)

    raise RuntimeError(f"Unhandled surface-brightness input type: {settings['input_type']}")


# ============================================================================
# STANDARD OSPM PROFILE
# ============================================================================

def build_profile(normalized: pd.DataFrame, *, galaxy: str, settings: dict, pc_per_arcmin: float, q_axis_ratio: float):
    if not math.isfinite(pc_per_arcmin) or pc_per_arcmin <= 0.0:
        raise ValueError("pc_per_arcmin must be positive and finite")
    if not math.isfinite(q_axis_ratio) or q_axis_ratio <= 0.0:
        raise ValueError("q_axis_ratio must be positive and finite")

    background = settings["background"]
    background_err = settings["background_err"]
    ellipticity = 1.0 - q_axis_ratio

    rin = normalized["rin_arcmin"].to_numpy(float)
    rout = normalized["rout_arcmin"].to_numpy(float)
    rmid = normalized["rm_arcmin"].to_numpy(float)
    sigma_source = normalized["Sigma_source_arcmin2"].to_numpy(float)
    sigma_source_err = normalized["Sigma_source_err_arcmin2"].to_numpy(float)

    order = np.argsort(rmid, kind="stable")
    normalized = normalized.iloc[order].reset_index(drop=True)

    rin = normalized["rin_arcmin"].to_numpy(float)
    rout = normalized["rout_arcmin"].to_numpy(float)
    rmid = normalized["rm_arcmin"].to_numpy(float)
    sigma_source = normalized["Sigma_source_arcmin2"].to_numpy(float)
    sigma_source_err = normalized["Sigma_source_err_arcmin2"].to_numpy(float)

    if np.any(np.diff(rmid) <= 0.0):
        raise ValueError("Prepared surface-brightness radii must be strictly increasing")

    if not np.isclose(rin[0], 0.0, rtol=0.0, atol=RADIAL_EDGE_ATOL):
        raise ValueError(
            "The first surface-brightness annulus must begin at R=0 so the "
            "unresolved-center light budget is defined"
        )

    gap_count = 0
    overlap_count = 0
    for i in range(1, len(normalized)):
        if np.isclose(rin[i], rout[i - 1], rtol=RADIAL_EDGE_RTOL, atol=RADIAL_EDGE_ATOL):
            continue
        if rin[i] > rout[i - 1]:
            gap_count += 1
        else:
            overlap_count += 1

    sigma = sigma_source - background
    sigma_err = np.hypot(sigma_source_err, background_err)

    area_arcmin2 = math.pi * q_axis_ratio * (rout**2 - rin**2)
    area_pc2 = area_arcmin2 * pc_per_arcmin**2
    light_raw = sigma * area_arcmin2
    light_raw_err = sigma_err * area_arcmin2

    sigma_significance = np.divide(
        sigma,
        sigma_err,
        out=np.full_like(sigma, np.inf, dtype=float),
        where=sigma_err > 0.0,
    )

    profile = normalized.copy()

    profile.insert(0, "galaxy", galaxy)
    profile.insert(1, "source", settings["source"])
    profile.insert(2, "preferred_profile", settings["preferred_profile"])
    profile.insert(3, "radius_type", settings["radius_type"])
    profile.insert(4, "input_type", settings["input_type"])

    profile["R_inner_pc"] = rin * pc_per_arcmin
    profile["R_outer_pc"] = rout * pc_per_arcmin
    profile["R_pc"] = rmid * pc_per_arcmin

    profile["Sigma_background_arcmin2"] = background
    profile["Sigma_background_err_arcmin2"] = background_err

    profile["Sigma"] = sigma
    profile["Sigma_err"] = sigma_err
    profile["Sigma_pc2"] = sigma / pc_per_arcmin**2
    profile["Sigma_err_pc2"] = sigma_err / pc_per_arcmin**2
    profile["Sigma_significance"] = sigma_significance

    profile["light_raw"] = light_raw
    profile["light_raw_err"] = light_raw_err

    profile["ellipticity"] = ellipticity
    profile["q_axis_ratio"] = q_axis_ratio
    profile["area_arcmin2"] = area_arcmin2
    profile["area_pc2"] = area_pc2

    profile["Sigma_units"] = settings["sigma_units"]
    profile["R_units"] = "pc"
    profile["area_model"] = "elliptical_annulus_pi_q_delta_a2"
    profile["pc_per_arcmin_assumed"] = pc_per_arcmin

    note = settings["note"]
    if note is None:
        note = (
            f"Generic OSPM surface-brightness preparation for {galaxy}. "
            f"Input type={settings['input_type']}. Background={background:.12g} +/- "
            f"{background_err:.12g} in the source projected-density units. "
            "Background uncertainty is combined in quadrature with the source uncertainty."
        )
    profile["note"] = str(note)

    profile.attrs["radial_gap_count"] = gap_count
    profile.attrs["radial_overlap_count"] = overlap_count

    return profile


def diagnose_profile(profile: pd.DataFrame):
    flags = [[] for _ in range(len(profile))]
    summary = []

    R = profile["R_pc"].to_numpy(float)
    rin = profile["rin_arcmin"].to_numpy(float)
    rout = profile["rout_arcmin"].to_numpy(float)
    sigma = profile["Sigma"].to_numpy(float)
    sigma_err = profile["Sigma_err"].to_numpy(float)
    significance = profile["Sigma_significance"].to_numpy(float)

    nonpositive = np.where(sigma <= 0.0)[0]
    for i in nonpositive:
        flags[i].append("NONPOSITIVE_CORRECTED_SIGMA")
    if len(nonpositive):
        summary.append(
            f"{len(nonpositive)} bin(s) have Sigma <= 0 after background subtraction. "
            "These measurements are retained, but the profile is not ready for "
            "positive-density normalization or Abel input."
        )

    low_sn = np.where((sigma > 0.0) & (significance < REVIEW_LOW_SN_SIGMA))[0]
    for i in low_sn:
        flags[i].append("LOW_SIGNAL_TO_NOISE")
    if len(low_sn):
        summary.append(
            f"{len(low_sn)} positive bin(s) are less than "
            f"{REVIEW_LOW_SN_SIGMA:.1f} sigma above the adopted background."
        )

    rise_significance = np.full(len(profile), np.nan)
    for i in range(1, len(profile)):
        denom = math.hypot(sigma_err[i - 1], sigma_err[i])
        rise_significance[i] = (
            (sigma[i] - sigma[i - 1]) / denom if denom > 0.0 else np.nan
        )

        if np.isfinite(rise_significance[i]) and rise_significance[i] >= REVIEW_RISE_SIGMA:
            flags[i].append("SIGNIFICANT_OUTWARD_RISE")
            summary.append(
                f"Outward rise between R={R[i - 1]:.6g} and {R[i]:.6g} pc is "
                f"{rise_significance[i]:.2f} sigma."
            )
        elif sigma[i] > sigma[i - 1]:
            flags[i].append("OUTWARD_RISE")

    gap_count = 0
    overlap_count = 0

    for i in range(1, len(profile)):
        if np.isclose(rin[i], rout[i - 1], rtol=RADIAL_EDGE_RTOL, atol=RADIAL_EDGE_ATOL):
            continue
        if rin[i] > rout[i - 1]:
            gap_count += 1
            flags[i].append("RADIAL_GAP_BEFORE_BIN")
        else:
            overlap_count += 1
            flags[i].append("RADIAL_OVERLAP_BEFORE_BIN")

    if gap_count:
        summary.append(f"{gap_count} gap(s) exist between adjacent radial annuli.")
    if overlap_count:
        summary.append(f"{overlap_count} overlap(s) exist between adjacent radial annuli.")

    outer_significance = float(significance[-1])
    if np.isfinite(outer_significance) and outer_significance >= REVIEW_OUTER_DETECTION_SIGMA:
        flags[-1].append("OUTER_PROFILE_STILL_DETECTED")
        summary.append(
            f"The outermost bin remains {outer_significance:.2f} sigma above the "
            "adopted background. The measured profile may end before the tracer "
            "distribution has reached the background."
        )

    profile["outward_rise_significance"] = rise_significance
    profile["review_flags"] = [";".join(x) for x in flags]
    profile["bin_review_required"] = [bool(x) for x in flags]

    review_required = bool(summary)
    profile["profile_review_status"] = "REVIEW_REQUIRED" if review_required else "PASS"
    profile["profile_review_required"] = review_required

    return profile, summary


def normalize_light(profile: pd.DataFrame):
    if np.any(profile["Sigma"].to_numpy(float) <= 0.0):
        profile["light_frac"] = np.nan
        profile["light_frac_err_approx"] = np.nan
        return profile, False

    light_total = float(profile["light_raw"].sum())
    if not math.isfinite(light_total) or light_total <= 0.0:
        raise ValueError("Corrected total light/tracer proxy is not positive")

    profile["light_frac"] = profile["light_raw"] / light_total
    profile["light_frac_err_approx"] = profile["light_raw_err"] / light_total

    light_frac_sum = float(profile["light_frac"].sum())
    if not math.isclose(light_frac_sum, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"light_frac does not sum to unity: {light_frac_sum:.16g}")

    return profile, True


# ============================================================================
# REPORT / PLOTS
# ============================================================================

def write_review_report(profile: pd.DataFrame, summary: list[str], report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    status = str(profile["profile_review_status"].iloc[0])

    lines = [
        "SURFACE-BRIGHTNESS PREPARATION REVIEW",
        "=====================================",
        f"Galaxy: {profile['galaxy'].iloc[0]}",
        f"Source: {profile['source'].iloc[0]}",
        f"Profile: {profile['preferred_profile'].iloc[0]}",
        f"Input type: {profile['input_type'].iloc[0]}",
        f"Status: {status}",
        "",
    ]

    if summary:
        lines.append("REVIEW REASONS")
        lines.append("--------------")
        for item in summary:
            lines.append(f"- {item}")
    else:
        lines.append("No automatic review conditions were triggered.")

    flagged = profile.loc[profile["bin_review_required"]]
    if len(flagged):
        lines.extend(["", "FLAGGED BINS", "------------"])
        for _, row in flagged.iterrows():
            lines.append(
                f"R={row['R_pc']:.6g} pc  Sigma={row['Sigma']:.6g} +/- "
                f"{row['Sigma_err']:.6g}  significance={row['Sigma_significance']:.3f}  "
                f"flags={row['review_flags']}"
            )

    lines.extend([
        "",
        "No suspicious measurements were modified automatically.",
        "A REVIEW_REQUIRED status means the profile should be inspected before "
        "downstream modeling or deprojection.",
    ])

    report_path.write_text("\n".join(lines) + "\n")


def make_plots(profile: pd.DataFrame, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)

    correction_plot = outdir / f"{stem}_background_correction.png"
    abel_input_plot = outdir / f"{stem}_abel_input.png"

    R_pc = profile["R_pc"].to_numpy(float)
    sigma_source = profile["Sigma_source_arcmin2"].to_numpy(float)
    sigma_source_err = profile["Sigma_source_err_arcmin2"].to_numpy(float)
    sigma = profile["Sigma"].to_numpy(float)
    sigma_err = profile["Sigma_err"].to_numpy(float)

    background = float(profile["Sigma_background_arcmin2"].iloc[0])
    galaxy = str(profile["galaxy"].iloc[0])
    source = str(profile["source"].iloc[0])
    preferred_profile = str(profile["preferred_profile"].iloc[0])
    sigma_units = str(profile["Sigma_units"].iloc[0])
    review_required = bool(profile["profile_review_required"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        R_pc,
        sigma_source,
        yerr=sigma_source_err,
        fmt="o",
        capsize=3,
        label=f"{source} {preferred_profile}".strip(),
    )
    ax.errorbar(
        R_pc,
        sigma,
        yerr=sigma_err,
        fmt="o",
        capsize=3,
        label="Prepared profile",
    )

    if background > 0.0:
        ax.axhline(
            background,
            linestyle="--",
            linewidth=1.2,
            label="Adopted fitted background",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Projected radius R [pc]")
    ax.set_ylabel(f"Projected density [{sigma_units}]")
    ax.set_title(f"{galaxy} Surface-Brightness Preparation")
    ax.legend()
    ax.grid(alpha=0.25)

    if review_required:
        ax.text(
            0.02,
            0.02,
            "REVIEW REQUIRED",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(correction_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)

    positive = sigma > 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        R_pc[positive],
        sigma[positive],
        yerr=sigma_err[positive],
        fmt="o",
        capsize=3,
        label="Positive Abel-input measurements",
    )

    if np.any(~positive):
        ax.errorbar(
            R_pc[~positive],
            np.maximum(sigma_err[~positive], np.finfo(float).tiny),
            yerr=sigma_err[~positive],
            fmt="x",
            capsize=3,
            label="Non-positive corrected measurement",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Projected radius R [pc]")
    ax.set_ylabel(f"Projected density [{sigma_units}]")
    ax.set_title(f"{galaxy} Prepared Surface-Brightness Profile")
    ax.legend()
    ax.grid(alpha=0.25)

    if review_required:
        ax.text(
            0.02,
            0.02,
            "REVIEW REQUIRED BEFORE ABEL USE",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(abel_input_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return correction_plot, abel_input_plot


# ============================================================================
# MAIN
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare a galaxy surface-brightness profile for OSPM from its config and observational CSV."
    )
    parser.add_argument("--galaxy", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and summarize the observational input without writing outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write the prepared profile but skip surface-brightness diagnostic plots.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    galaxy = str(args.galaxy).strip()

    if not galaxy:
        raise ValueError("--galaxy cannot be empty")

    cfg = load_config(config_path(galaxy))
    settings = load_surface_brightness_settings(cfg)

    distance_pc = float(require(cfg, "DISTANCE_PC", "CONFIG"))
    q_axis_ratio = float(require(cfg, "AXIS_RATIO_Q", "CONFIG"))
    output_path = Path(str(require(cfg, "SURFACE_BRIGHTNESS_CSV", "CONFIG"))).expanduser().resolve()
    input_path = Path(str(settings["input_csv"])).expanduser().resolve()

    if not math.isfinite(distance_pc) or distance_pc <= 0.0:
        raise ValueError("CONFIG['DISTANCE_PC'] must be positive and finite")
    if not math.isfinite(q_axis_ratio) or q_axis_ratio <= 0.0:
        raise ValueError("CONFIG['AXIS_RATIO_Q'] must be positive and finite")
    if not input_path.is_file():
        raise FileNotFoundError(f"Surface-brightness observational input does not exist: {input_path}")

    pc_per_arcmin = distance_pc * math.pi / (180.0 * 60.0)

    raw = pd.read_csv(input_path)
    normalized = normalize_observational_input(raw, settings, q_axis_ratio)
    profile = build_profile(
        normalized,
        galaxy=galaxy,
        settings=settings,
        pc_per_arcmin=pc_per_arcmin,
        q_axis_ratio=q_axis_ratio,
    )
    profile, review_summary = diagnose_profile(profile)
    profile, normalized_light = normalize_light(profile)

    print()
    print("=" * 76)
    print(f"OSPM SURFACE BRIGHTNESS: {galaxy}")
    print("=" * 76)
    print(f"Input:              {input_path}")
    print(f"Input type:         {settings['input_type']}")
    print(f"Source:             {settings['source']}")
    print(f"Preferred profile:  {settings['preferred_profile']}")
    print(f"Radius type:        {settings['radius_type']}")
    print(f"Rows:               {len(profile)}")
    print(f"Distance [pc]:      {distance_pc:.12g}")
    print(f"pc per arcmin:      {pc_per_arcmin:.12g}")
    print(f"q_axis_ratio:       {q_axis_ratio:.12g}")
    print(
        f"Background:         {settings['background']:.12g} +/- "
        f"{settings['background_err']:.12g} [{settings['sigma_units']}]"
    )

    if settings["input_type"] == "fixed_annulus_counts":
        width = normalized.attrs.get("annulus_width_arcmin")
        offset = normalized.attrs.get("max_center_offset_arcmin")
        if width is not None:
            print(f"Annulus width:      {width:.12g} arcmin")
        if offset is not None:
            print(f"Max center offset:  {offset:.12g} arcmin")

    print(
        f"Outermost Sigma:    {profile['Sigma'].iloc[-1]:.12g} +/- "
        f"{profile['Sigma_err'].iloc[-1]:.12g}"
    )
    print(f"Profile status:     {profile['profile_review_status'].iloc[0]}")

    if normalized_light:
        print(f"light_frac sum:     {profile['light_frac'].sum():.12f}")
    else:
        print("light_frac:         not generated because corrected Sigma is non-positive")

    if review_summary:
        print()
        for item in review_summary:
            print(f"WARNING: {item}")

    if args.check:
        print()
        print("CHECK ONLY: no files were written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path, index=False)

    report_path = output_path.with_suffix(".review.txt")
    write_review_report(profile, review_summary, report_path)

    correction_plot = None
    abel_input_plot = None

    if not args.no_plots:
        plots_dir = profile_root(galaxy) / "plots"
        stem = f"{galaxy}_surface_brightness"
        correction_plot, abel_input_plot = make_plots(profile, plots_dir, stem)

    print()
    print(output_path)
    print(f"Review report: {report_path}")

    if correction_plot is not None:
        print(correction_plot)
    if abel_input_plot is not None:
        print(abel_input_plot)


if __name__ == "__main__":
    main()
