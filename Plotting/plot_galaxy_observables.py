#!/usr/bin/env python3
#######################################################################################################################
#
#
#
#######################################################################################################################

from pathlib import Path
import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr

def resolve_repo_root():
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [cwd, here, *cwd.parents, *here.parents]
    for base in candidates:
        if (base / "OSPM" / "load_config.py").exists() and (base / "Data" / "Galaxy_Profiles").is_dir():
            return base
    raise FileNotFoundError("Could not find the OSPM repository root. Expected OSPM/load_config.py and Data/Galaxy_Profiles/. Run this from inside the OSPM repository.")

def load_galaxy_config(galaxy_override=None):
    repo_root = resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if galaxy_override:
        os.environ["OSPM_GALAXY"] = galaxy_override.strip()
    from OSPM.load_config import get_profile_root, load_config
    config = load_config()
    profile_root = Path(get_profile_root()).resolve()
    return config, profile_root, repo_root

def resolve_config_path(value, profile_root, repo_root):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    repo_candidate = (repo_root / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (profile_root / path).resolve()

def observable_option(config, *names, default=None):
    obs_cfg = config.get("OBSERVABLES", {})
    if not isinstance(obs_cfg, dict):
        obs_cfg = {}
    for name in names:
        if name in obs_cfg and obs_cfg[name] is not None:
            return obs_cfg[name]
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return default

def active_tracer_path(config, profile_root, repo_root):
    stellar_model = config.get("STELLAR_MODEL", {}) or {}
    tracer_mode = str(config.get("TRACER_CONSTRAINT_MODE", "projected_light")).strip().lower()
    if tracer_mode == "density_3d" and stellar_model.get("tracer_grid_csv"):
        return resolve_config_path(stellar_model["tracer_grid_csv"], profile_root, repo_root)
    if stellar_model.get("grid_csv"):
        return resolve_config_path(stellar_model["grid_csv"], profile_root, repo_root)
    raise KeyError("CONFIG['STELLAR_MODEL'] must provide grid_csv or tracer_grid_csv")

def safe_prefix(galaxy):
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(galaxy).strip())
    return prefix or "galaxy"

def as_bool_array(series):
    if series.dtype == bool:
        return series.to_numpy(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"}).to_numpy(bool)

def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def first_col(df, names, required=True):
    for name in names:
        if name and name in df.columns:
            return name
    if required:
        raise KeyError(f"None of these columns found: {names}\nAvailable columns: {list(df.columns)}")
    return None

def resolve_v_sys(path, configured_v_sys):
    if configured_v_sys is not None and np.isfinite(float(configured_v_sys)):
        return float(configured_v_sys)
    df = pd.read_csv(path, nrows=1000)
    if "v_sys_kms" in df.columns:
        values = pd.to_numeric(df["v_sys_kms"], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values):
            return float(np.median(values))
    raise ValueError("V_SYS_KMS is not defined in the active galaxy config and the star table does not provide a finite v_sys_kms column.")

def load_stars(path, v_sys_kms, r_col, v_col, verr_col):
    df = pd.read_csv(path)
    required = {r_col, v_col, verr_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} is missing configured star columns: {sorted(missing)}")
    r = pd.to_numeric(df[r_col], errors="coerce").to_numpy(float)
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(float)
    ve = pd.to_numeric(df[verr_col], errors="coerce").to_numpy(float)
    if "has_vlos" in df.columns:
        has_vlos = as_bool_array(df["has_vlos"])
    else:
        has_vlos = np.ones(len(df), dtype=bool)
    valid = has_vlos & np.isfinite(r) & np.isfinite(v) & np.isfinite(ve) & (r >= 0.0) & (ve > 0.0)
    stars = df.loc[valid].copy()
    stars["_r_pc"] = r[valid]
    stars["_vlos_kms"] = v[valid]
    stars["_vlos_err_kms"] = ve[valid]
    stars["_vlos_rel_kms"] = stars["_vlos_kms"].to_numpy(float) - float(v_sys_kms)
    stars = stars.sort_values("_r_pc", kind="mergesort").reset_index(drop=True)
    return stars

def load_losvd_bins(path, stars):
    bins = pd.read_csv(path)
    required = {"R_inner_pc", "R_outer_pc"}
    missing = required - set(bins.columns)
    if missing:
        raise KeyError(f"{path.name} is missing required columns: {sorted(missing)}")
    bins = bins.copy().reset_index(drop=True)
    if "bin_id" not in bins.columns:
        bins.insert(0, "bin_id", np.arange(len(bins), dtype=int))
    for col in ("R_inner_pc", "R_outer_pc"):
        bins[col] = pd.to_numeric(bins[col], errors="coerce").astype(float)
    if "R_mid_pc" not in bins.columns:
        bins["R_mid_pc"] = 0.5 * (bins["R_inner_pc"] + bins["R_outer_pc"])
    else:
        bins["R_mid_pc"] = pd.to_numeric(bins["R_mid_pc"], errors="coerce")
    if bins[["R_inner_pc", "R_outer_pc"]].isna().any().any():
        raise ValueError(f"{path.name} contains non-numeric radial edges")
    edges = np.r_[float(bins["R_inner_pc"].iloc[0]), bins["R_outer_pc"].to_numpy(float)]
    edges[0] = 0.0
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{path.name} radial edges are not strictly increasing")
    rmax = float(stars["_r_pc"].max())
    if edges[-1] <= rmax:
        edges[-1] = np.nextafter(rmax, np.inf)
        bins.loc[len(bins) - 1, "R_outer_pc"] = edges[-1]
        bins.loc[len(bins) - 1, "R_mid_pc"] = 0.5 * (float(bins.loc[len(bins) - 1, "R_inner_pc"]) + float(bins.loc[len(bins) - 1, "R_outer_pc"]))
    return bins, edges

def assign_bins(r, edges):
    idx = np.searchsorted(edges, r, side="right") - 1
    idx[(idx < 0) | (idx >= len(edges) - 1)] = -1
    return idx

def profile_nll_sigma(sigma, v, verr):
    sigma = max(float(sigma), 0.0)
    var = sigma * sigma + verr * verr
    w = 1.0 / var
    mu = float(np.sum(w * v) / np.sum(w))
    resid = v - mu
    nll = 0.5 * np.sum(np.log(var) + resid * resid / var)
    return float(nll), mu

def intrinsic_dispersion_mle(v, verr):
    v = np.asarray(v, float)
    verr = np.asarray(verr, float)
    good = np.isfinite(v) & np.isfinite(verr) & (verr > 0.0)
    v = v[good]
    verr = verr[good]
    if len(v) < 2:
        return math.nan, math.nan, math.nan, math.nan
    raw_std = float(np.std(v, ddof=1))
    scale = max(raw_std, float(np.median(verr)), 1.0)
    upper = max(50.0, 8.0 * scale)
    result = minimize_scalar(lambda s: profile_nll_sigma(s, v, verr)[0], bounds=(0.0, upper), method="bounded", options={"xatol": 1e-8})
    sigma_hat = max(float(result.x), 0.0)
    nll_min, mu_hat = profile_nll_sigma(sigma_hat, v, verr)
    target = nll_min + 0.5
    nll_zero = profile_nll_sigma(0.0, v, verr)[0]
    if sigma_hat <= 1e-10 or nll_zero <= target:
        sigma_lo = 0.0
    else:
        sigma_lo = brentq(lambda s: profile_nll_sigma(s, v, verr)[0] - target, 0.0, sigma_hat)
    hi = max(1.0, 1.5 * sigma_hat)
    while profile_nll_sigma(hi, v, verr)[0] < target and hi < 1000.0:
        hi *= 2.0
    if hi >= 1000.0 and profile_nll_sigma(hi, v, verr)[0] < target:
        sigma_hi = math.nan
    else:
        sigma_hi = brentq(lambda s: profile_nll_sigma(s, v, verr)[0] - target, sigma_hat, hi)
    return sigma_hat, sigma_lo, sigma_hi, mu_hat

def velocity_edges_auto(stars, nvbin):
    v = stars["_vlos_rel_kms"].to_numpy(float)
    ve = stars["_vlos_err_kms"].to_numpy(float)
    good_err = ve[np.isfinite(ve) & (ve > 0.0)]
    pad = 3.0 * float(np.median(good_err))
    return np.linspace(float(np.min(v) - pad), float(np.max(v) + pad), int(nvbin) + 1)

def gaussian_losvd(v, verr, velocity_edges):
    v = np.asarray(v, float)
    verr = np.asarray(verr, float)
    counts = np.zeros(len(velocity_edges) - 1, dtype=float)

    for vi, ei in zip(v, verr):
        z = (velocity_edges - vi) / ei
        p = np.diff(ndtr(z))
        psum = float(np.sum(p))
        if psum > 0.0 and np.isfinite(psum):
            counts += p / psum
        else:
            j = np.searchsorted(velocity_edges, vi, side="right") - 1
            if 0 <= j < len(counts):
                counts[j] += 1.0

    if len(v) > 0:
        counts /= float(len(v))
    return counts

def plot_surface_brightness(path, outdir, galaxy, prefix):
    df = pd.read_csv(path)
    rcol = first_col(df, ["R_pc", "rm_pc", "R_mid_pc", "r_pc", "radius_pc"])
    scol = first_col(df, ["Sigma", "S2_sigma_arcmin2", "Sigma_Lsun_pc2", "surface_brightness"])
    ecol = first_col(df, ["Sigma_err", "S2_sigma_err_arcmin2", "Sigma_err_Lsun_pc2", "surface_brightness_err"], required=False)
    work = pd.DataFrame({"R": pd.to_numeric(df[rcol], errors="coerce"), "Sigma": pd.to_numeric(df[scol], errors="coerce")})
    if ecol is not None:
        work["Sigma_err"] = pd.to_numeric(df[ecol], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["R", "Sigma"])
    work = work[(work["R"] > 0.0) & (work["Sigma"] > 0.0)]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if ecol is not None:
        err = work["Sigma_err"].to_numpy(float)
        gooderr = np.isfinite(err) & (err >= 0.0)
        ax.errorbar(work.loc[gooderr, "R"], work.loc[gooderr, "Sigma"], yerr=work.loc[gooderr, "Sigma_err"], fmt="o", capsize=3)
        if np.any(~gooderr):
            ax.plot(work.loc[~gooderr, "R"], work.loc[~gooderr, "Sigma"], "o")
    else:
        ax.plot(work["R"], work["Sigma"], "o-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Projected radius R [pc]")
    if "arcmin" in scol.lower() or scol == "Sigma":
        ax.set_ylabel(r"Surface density $\Sigma$ [stars arcmin$^{-2}$]")
    elif "lsun" in scol.lower():
        ax.set_ylabel(r"Surface brightness $\Sigma_L$ [$L_\odot$ pc$^{-2}$]")
    else:
        ax.set_ylabel(r"Surface brightness / density $\Sigma$")
    ax.set_title(f"{galaxy} 2D Surface-Brightness Profile")
    ax.grid(alpha=0.25)
    savefig(fig, outdir / f"01_{prefix}_surface_brightness_2d.png")

def load_tracer_density(path, stellar_model):
    df = pd.read_csv(path)
    geometry = str(stellar_model.get("geometry", "")).strip().lower()
    nucol = first_col(df, [stellar_model.get("nu_col"),"nu_Lsun_pc3", "nu_stars_pc3", "number_density_stars_pc3", "density", "rho"])
    volcol = first_col(df, [stellar_model.get("volume_col"), "cell_volume_pc3", "volume_pc3"], required=False)
    radius_candidates = [stellar_model.get("radius_col"), "m_pc", "r_pc", "radius_pc"]
    mcol = first_col(df, radius_candidates, required=False)
    Rcol = first_col(df, [stellar_model.get("R_cyl_col"), "R_cyl_pc", "R_pc"], required=False)
    zcol = first_col(df, [stellar_model.get("z_col"), "z_pc", "z"], required=False)
    if mcol is None and Rcol is not None and zcol is not None:
        df = df.copy()
        df["_m_plot_pc"] = np.sqrt(pd.to_numeric(df[Rcol], errors="coerce") ** 2 + pd.to_numeric(df[zcol], errors="coerce") ** 2)
        mcol = "_m_plot_pc"
    if mcol is None:
        raise KeyError(f"Could not determine an intrinsic radius column for {path.name}. " f"Geometry={geometry!r}; available columns={list(df.columns)}")

    return df, geometry, Rcol, zcol, mcol, nucol, volcol

def plot_tracer_density_radial(df, mcol, nucol, volcol, force_path, stellar_model, outdir, galaxy, prefix):
    cols = [mcol, nucol] + ([volcol] if volcol is not None else [])
    work = df[cols].copy().replace([np.inf, -np.inf], np.nan).dropna(subset=[mcol, nucol])
    work[mcol] = pd.to_numeric(work[mcol], errors="coerce")
    work[nucol] = pd.to_numeric(work[nucol], errors="coerce")
    work = work.dropna(subset=[mcol, nucol])
    work = work[(work[mcol] > 0.0) & (work[nucol] > 0.0)]
    work["_rkey"] = work[mcol].round(10)
    rows = []

    for _, group in work.groupby("_rkey", sort=True):
        radius = float(group[mcol].mean())
        y = group[nucol].to_numpy(float)
        if volcol is not None:
            w = pd.to_numeric(group[volcol], errors="coerce").to_numpy(float)
            good = np.isfinite(w) & (w > 0.0) & np.isfinite(y)
            density = float(np.average(y[good], weights=w[good])) if np.any(good) else float(np.nanmedian(y))
        else:
            density = float(np.nanmedian(y))
        rows.append((radius, density))

    tracer = pd.DataFrame(rows, columns=["radius_pc", "tracer_density"])
    tracer = tracer[np.isfinite(tracer["tracer_density"]) & (tracer["tracer_density"] > 0.0)].sort_values("radius_pc")
    force, _ = load_radial_density_profile(force_path, stellar_model)
    force_raw = pd.read_csv(force_path)
    support_col = first_col(force_raw, ["abel_support_min_pc"], required=False)

    if support_col is not None:
        support = pd.to_numeric(force_raw[support_col], errors="coerce").dropna()
        support_min = float(support.iloc[0]) if len(support) else math.nan
    else:
        unresolved = force_raw.loc[
            force_raw.get("density_source", pd.Series(index=force_raw.index, dtype=str),).astype(str).eq("unresolved_center_force_model")]
        support_min = (
            float(pd.to_numeric(unresolved["R_outer_pc"], errors="coerce").max())
            if len(unresolved) and "R_outer_pc" in unresolved.columns
            else math.nan
        )
    if np.isfinite(support_min):
        resolved = tracer[tracer["radius_pc"] >= support_min].copy()
        inner = force[force["radius_pc"] <= support_min].copy()
    else:
        resolved = tracer.copy()
        inner = force.iloc[0:0].copy()
    radial_out = resolved.rename(columns={"tracer_density": "density"}).copy()
    radial_out["density_source"] = "resolved_abel"
    if len(inner):
        inner_out = inner.copy()
        inner_out["density_source"] = "karl_inner_continuation"
        radial_out = pd.concat([inner_out, radial_out], ignore_index=True)
    radial_out = radial_out.sort_values("radius_pc")
    radial_out.to_csv(outdir / f"{prefix}_tracer_density_3d_radial_profile.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if len(inner):
        ax.plot(inner["radius_pc"], inner["density"], "o--", markersize=3, linewidth=1.5, label="Karl-style inner continuation",)
    ax.plot(resolved["radius_pc"], resolved["tracer_density"],"o-", markersize=3, linewidth=1.5, label="Resolved Abel deprojection")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Intrinsic radius $m$ [pc]")

    if "lsun" in nucol.lower():
        ax.set_ylabel(r"3D tracer density $\nu_L$ [$L_\odot$ pc$^{-3}$]")
    else:
        ax.set_ylabel(r"3D tracer density $\nu$")
    if np.isfinite(support_min):
        if len(inner):
            xmin = float(inner["radius_pc"].min())
        else:
            xmin = float(resolved["radius_pc"].min())
        ax.axvspan(xmin, support_min, alpha=0.08, label="Unresolved inner region")
        ax.axvline(support_min, linestyle=":", linewidth=1.2, label=f"Abel support minimum = {support_min:.3f} pc")

    ax.set_title(f"{galaxy} 3D Tracer-Density Profile")
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(fig, outdir / f"02_{prefix}_tracer_density_3d_profile.png")

def plot_tracer_density_map(df, Rcol, zcol, nucol, outdir, galaxy, prefix):
    d = df[[Rcol, zcol, nucol]].copy()
    for col in (Rcol, zcol, nucol):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d[Rcol] >= 0.0) & (d[nucol] > 0.0)]
    if d.empty:
        raise ValueError("No positive finite 3D tracer-density cells found")
    R = d[Rcol].to_numpy(float)
    z = d[zcol].to_numpy(float)
    lognu = np.log10(d[nucol].to_numpy(float))
    positive_R = R > 0.0
    R_plot = np.concatenate([-R[positive_R], R])
    z_plot = np.concatenate([z[positive_R], z])
    lognu_plot = np.concatenate([lognu[positive_R], lognu])
    full_R = float(np.max(R))
    full_z = float(np.max(np.abs(z)))
    pad = 1.05
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")
    ax.set_facecolor("black")
    smooth = ax.tricontourf(R_plot, z_plot, lognu_plot, levels=256, cmap="inferno", vmin=float(np.min(lognu)), vmax=float(np.max(lognu)))
    ax.set_xlim(-pad * full_R, pad * full_R)
    ax.set_ylim(-pad * full_z, pad * full_z)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Cylindrical radius $R$ [pc]", color="white")
    ax.set_ylabel(r"$z$ [pc]", color="white")
    ax.set_title(f"{galaxy} Deprojected 3D Tracer Luminosity Density", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")
    cbar = fig.colorbar(smooth, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")
    if "lsun" in nucol.lower():
        cbar.set_label(r"$\log_{10}\nu_L$ [$L_\odot$ pc$^{-3}$]", color="white")
    else:
        cbar.set_label(r"$\log_{10}\nu$", color="white")

    fig.savefig(outdir / f"03_{prefix}_tracer_density_3d_map.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def load_radial_density_profile(path, stellar_model):
    df, geometry, Rcol, zcol, mcol, nucol, volcol = load_tracer_density(path, stellar_model)
    cols = [mcol, nucol] + ([volcol] if volcol is not None else [])
    work = df[cols].copy().replace([np.inf, -np.inf], np.nan).dropna(subset=[mcol, nucol])
    work[mcol] = pd.to_numeric(work[mcol], errors="coerce")
    work[nucol] = pd.to_numeric(work[nucol], errors="coerce")
    work = work.dropna(subset=[mcol, nucol])
    work = work[(work[mcol] > 0.0) & (work[nucol] > 0.0)]
    work["_rkey"] = work[mcol].round(10)

    rows = []
    for _, group in work.groupby("_rkey", sort=True):
        radius = float(group[mcol].mean())
        y = group[nucol].to_numpy(float)
        if volcol is not None:
            w = pd.to_numeric(group[volcol], errors="coerce").to_numpy(float)
            good = np.isfinite(w) & (w > 0.0) & np.isfinite(y)
            density = float(np.average(y[good], weights=w[good])) if np.any(good) else float(np.nanmedian(y))
        else:
            density = float(np.nanmedian(y))
        rows.append((radius, density))

    radial = pd.DataFrame(rows, columns=["radius_pc", "density"])
    return radial[np.isfinite(radial["density"]) & (radial["density"] > 0.0)].sort_values("radius_pc"), nucol

def plot_stellar_force_density_profile(path, stellar_model, outdir, galaxy, prefix):
    radial, nucol = load_radial_density_profile(path, stellar_model)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(radial["radius_pc"], radial["density"], "o-", markersize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Intrinsic radius $m$ [pc]")
    if "lsun" in nucol.lower():
        ax.set_ylabel(r"Stellar luminosity density $\nu_L$ [$L_\odot$ pc$^{-3}$]")
    else:
        ax.set_ylabel(r"Stellar density $\nu$")
    ax.set_title(f"{galaxy} Stellar Force-Density Profile")
    ax.grid(alpha=0.25)
    savefig(fig, outdir / f"08_{prefix}_stellar_force_density_profile.png")

def plot_tracer_vs_stellar_force_inner(tracer_path, force_path, stellar_model, outdir, galaxy, prefix):
    tracer, _ = load_radial_density_profile(tracer_path, stellar_model)
    force, _ = load_radial_density_profile(force_path, stellar_model)
    force_raw = pd.read_csv(force_path)

    support_col = first_col(force_raw, ["abel_support_min_pc"], required=False)
    if support_col is not None:
        support = pd.to_numeric(force_raw[support_col], errors="coerce").dropna()
        support_min = float(support.iloc[0]) if len(support) else math.nan
    else:
        unresolved = force_raw.loc[force_raw.get("density_source", pd.Series(index=force_raw.index, dtype=str)).astype(str).eq("unresolved_center_force_model")]
        support_min = float(pd.to_numeric(unresolved["R_outer_pc"], errors="coerce").max()) if len(unresolved) and "R_outer_pc" in unresolved.columns else math.nan

    rmax = 2.5 * support_min if np.isfinite(support_min) and support_min > 0.0 else min(float(tracer["radius_pc"].max()), float(force["radius_pc"].max()))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(tracer["radius_pc"], tracer["density"], "o-", markersize=4, label="Tracer constraint grid")
    ax.plot(force["radius_pc"], force["density"], "o-", markersize=3, label="Stellar force grid")
    if np.isfinite(support_min):
        ax.axvline(support_min, linestyle="--", linewidth=1.0, label=f"Abel support minimum = {support_min:.3f} pc")
    ax.set_xlim(left=max(min(float(tracer["radius_pc"].min()), float(force["radius_pc"].min())) * 0.8, 1.0e-4), right=rmax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Intrinsic radius $m$ [pc]")
    ax.set_ylabel(r"3D luminosity density $\nu_L$ [$L_\odot$ pc$^{-3}$]")
    ax.set_title(f"{galaxy} Inner Tracer Grid vs Stellar Force Grid")
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(fig, outdir / f"09_{prefix}_tracer_vs_stellar_force_inner_profile.png")

def _plot_jl_vector_f64(vec, Main, name="vec"):
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if not np.isfinite(arr).all():
        bad = arr[~np.isfinite(arr)]
        raise ValueError(f"{name} has non-finite values: {bad[:10]}")
    Main._plot_tmp_vector_f64 = arr.tolist()
    return Main.seval("Float64[y for y in _plot_tmp_vector_f64]")

def _plot_jl_vector_bool(vec, Main):
    arr = np.asarray(vec, dtype=bool).ravel()
    Main._plot_tmp_vector_bool = arr.tolist()
    return Main.seval("Bool[y for y in _plot_tmp_vector_bool]")

def _plot_jl_surface_brightness_profile(profile, Main):
    Main._plot_sb_R_pc = np.asarray(profile["R_pc"], dtype=np.float64).ravel().tolist()
    Main._plot_sb_R_inner_pc = np.asarray(profile["R_inner_pc"], dtype=np.float64).ravel().tolist()
    Main._plot_sb_R_outer_pc = np.asarray(profile["R_outer_pc"], dtype=np.float64).ravel().tolist()
    Main._plot_sb_light_frac = np.asarray(profile["light_frac"], dtype=np.float64).ravel().tolist()
    Main._plot_sb_Sigma = np.asarray(profile["Sigma"], dtype=np.float64).ravel().tolist()
    Main._plot_sb_Sigma_err = np.asarray(profile["Sigma_err"], dtype=np.float64).ravel().tolist()
    return Main.seval("Dict{Symbol,Any}(:R_pc => Float64[x for x in _plot_sb_R_pc], :R_inner_pc => Float64[x for x in _plot_sb_R_inner_pc], :R_outer_pc => Float64[x for x in _plot_sb_R_outer_pc], :light_frac => Float64[x for x in _plot_sb_light_frac], :Sigma => Float64[x for x in _plot_sb_Sigma], :Sigma_err => Float64[x for x in _plot_sb_Sigma_err])")

def _plot_jl_array(value, dtype=float):
    return np.asarray([dtype(x) for x in value], dtype=dtype)

def load_julia_karl_resolved_targets(config, stars_path, sb_path, losvd_bins_path, force_path, tracer_path, v_sys_override=None, nvbin_override=None):
    plot_config = dict(config)
    plot_config["DATA_CSV"] = str(stars_path)
    plot_config["SURFACE_BRIGHTNESS_CSV"] = str(sb_path)
    plot_config["KINEMATIC_BINS_CSV"] = str(losvd_bins_path)

    stellar_model = dict(plot_config.get("STELLAR_MODEL", {}) or {})
    if force_path is not None:
        stellar_model["grid_csv"] = str(force_path)
    if tracer_path is not None and "tracer_grid_csv" in stellar_model:
        stellar_model["tracer_grid_csv"] = str(tracer_path)
    plot_config["STELLAR_MODEL"] = stellar_model

    if v_sys_override is not None:
        plot_config["V_SYS_KMS"] = float(v_sys_override)
    if nvbin_override is not None:
        obs_cfg = dict(plot_config.get("OBSERVABLES", {}) or {})
        obs_cfg["NVBIN"] = int(nvbin_override)
        plot_config["OBSERVABLES"] = obs_cfg

    mode = str(observable_option(plot_config, "LOSVD_TARGET_MODE", "losvd_target_mode", default="current")).strip().lower()
    if mode != "karl_resolved_stars":
        raise ValueError(f"Exact Julia resolved-star target requested while LOSVD_TARGET_MODE={mode!r}")

    os.environ.setdefault("OSPM_USE_JULIA", "1")
    from OSPM.Controllers.OSPM_RUN import build_observables
    from OSPM.Physics import OSPM_Physics as physics_bridge

    obs = build_observables(plot_config)
    physics_bridge._jl_init()
    from juliacall import Main

    nvbin = int(observable_option(plot_config, "NVBIN", "Nvbin", "nvbin", default=21))
    velocity_edges = observable_option(plot_config, "VELOCITY_EDGES", "velocity_edges", default=None)
    kinematic_edges_pc = observable_option(plot_config, "KINEMATIC_BIN_EDGES_PC", "kinematic_bin_edges_pc", default=obs.kinematic_bin_edges_pc)
    light_edges_pc = observable_option(plot_config, "LIGHT_BIN_EDGES_PC", "light_bin_edges_pc", default=obs.light_bin_edges_pc)
    surface_brightness_profile = observable_option(plot_config, "SURFACE_BRIGHTNESS_PROFILE", "surface_brightness_profile", default=obs.surface_brightness_profile)
    kde_grid = int(observable_option(plot_config, "KARL_RESOLVED_KDE_GRID", "karl_resolved_kde_grid", default=17))
    kde_width_bins = float(observable_option(plot_config, "KARL_RESOLVED_KDE_WIDTH_BINS", "karl_resolved_kde_width_bins", default=3.0))
    vmin_kms = float(observable_option(plot_config, "KARL_RESOLVED_VMIN_KMS", "karl_resolved_vmin_kms", default=-25.0))
    vmax_kms = float(observable_option(plot_config, "KARL_RESOLVED_VMAX_KMS", "karl_resolved_vmax_kms", default=25.0))
    bootstraps = int(observable_option(plot_config, "KARL_RESOLVED_BOOTSTRAPS", "karl_resolved_bootstraps", default=300))
    envelope_floor = float(observable_option(plot_config, "KARL_RESOLVED_ENVELOPE_FLOOR", "karl_resolved_envelope_floor", default=0.003))

    pc_m = 3.0856775814913673e16
    Main._plot_R_star_jl = _plot_jl_vector_f64(obs.R_star_m, Main, name="R_star_m")
    Main._plot_valid_vlos_jl = _plot_jl_vector_bool(obs.valid_vlos, Main)
    Main._plot_v_star_jl = _plot_jl_vector_f64(obs.v_star_mps, Main, name="v_star_mps")
    Main._plot_verr_star_jl = _plot_jl_vector_f64(obs.verr_star_mps, Main, name="verr_star_mps")
    Main._plot_kinematic_edges_input_jl = _plot_jl_vector_f64(np.asarray(kinematic_edges_pc, float) * pc_m, Main, name="kinematic_bin_edges")
    Main._plot_light_edges_input_jl = _plot_jl_vector_f64(np.asarray(light_edges_pc, float) * pc_m, Main, name="light_bin_edges")
    Main._plot_surface_brightness_jl = _plot_jl_surface_brightness_profile(surface_brightness_profile, Main)
    if velocity_edges is None:
        Main._plot_velocity_edges_input_jl = Main.seval("nothing")
    else:
        Main._plot_velocity_edges_input_jl = _plot_jl_vector_f64(velocity_edges, Main, name="velocity_edges")

    Main.seval(f"_plot_nvbin_jl = {nvbin}")
    Main.seval(f"_plot_kde_grid_jl = {kde_grid}")
    Main.seval(f"_plot_kde_width_bins_jl = {kde_width_bins!r}")
    Main.seval(f"_plot_vmin_kms_jl = {vmin_kms!r}")
    Main.seval(f"_plot_vmax_kms_jl = {vmax_kms!r}")
    Main.seval(f"_plot_bootstraps_jl = {bootstraps}")
    Main.seval(f"_plot_envelope_floor_jl = {envelope_floor!r}")

    result = Main.seval("""
_plot_spatial_edges_jl = OSPMPhysicsSpherical.resolve_karl_spatial_edges(_plot_kinematic_edges_input_jl)
_plot_light_edges_jl = OSPMPhysicsSpherical.resolve_karl_light_edges(_plot_light_edges_input_jl)
_plot_vlos_idx_jl = findall(_plot_valid_vlos_jl)
_plot_velocity_edges_jl = _plot_velocity_edges_input_jl === nothing ? OSPMPhysicsSpherical.build_velocity_edges_auto(_plot_v_star_jl[_plot_vlos_idx_jl], _plot_verr_star_jl[_plot_vlos_idx_jl]; Nvbin=_plot_nvbin_jl) : Float64.(_plot_velocity_edges_input_jl)
_plot_targets_jl = OSPMPhysicsSpherical.observed_targets_karl(_plot_R_star_jl, _plot_valid_vlos_jl, _plot_v_star_jl, _plot_verr_star_jl, _plot_spatial_edges_jl, _plot_velocity_edges_jl; surface_brightness_profile=_plot_surface_brightness_jl, light_edges=_plot_light_edges_jl, target_mode=:karl_resolved_stars, karl_resolved_kde_grid=_plot_kde_grid_jl, karl_resolved_kde_width_bins=_plot_kde_width_bins_jl, karl_resolved_vmin_kms=_plot_vmin_kms_jl, karl_resolved_vmax_kms=_plot_vmax_kms_jl, karl_resolved_bootstraps=_plot_bootstraps_jl, karl_resolved_envelope_floor=_plot_envelope_floor_jl)
(_plot_targets_jl[1], _plot_targets_jl[2], _plot_targets_jl[5], _plot_spatial_edges_jl, _plot_velocity_edges_jl)
""")

    losvd_target_jl, losvd_sigma_jl, counts_by_spatial_jl, spatial_edges_jl, velocity_edges_jl = result
    losvd_target = _plot_jl_array(losvd_target_jl, float)
    losvd_sigma = _plot_jl_array(losvd_sigma_jl, float)
    counts_by_spatial = _plot_jl_array(counts_by_spatial_jl, float)
    spatial_edges_pc = _plot_jl_array(spatial_edges_jl, float) / pc_m
    velocity_edges_kms = _plot_jl_array(velocity_edges_jl, float) / 1.0e3

    nspatial = len(spatial_edges_pc) - 1
    nvbin_eff = len(velocity_edges_kms) - 1
    if len(losvd_target) != nspatial * nvbin_eff or len(losvd_sigma) != nspatial * nvbin_eff:
        raise ValueError("Julia karl_resolved_stars target length does not match the returned spatial/velocity grids")
    if len(counts_by_spatial) != nspatial:
        raise ValueError("Julia karl_resolved_stars star-count length does not match the returned spatial grid")

    return {
        "losvd_target": losvd_target.reshape(nspatial, nvbin_eff),
        "losvd_sigma": losvd_sigma.reshape(nspatial, nvbin_eff),
        "counts_by_spatial": counts_by_spatial,
        "spatial_edges_pc": spatial_edges_pc,
        "velocity_edges_kms": velocity_edges_kms,
        "requested_nvbin": nvbin,
        "used_nvbin": nvbin_eff,
        "kde_grid": kde_grid,
        "kde_width_bins": kde_width_bins,
        "vmin_kms": vmin_kms,
        "vmax_kms": vmax_kms,
        "bootstraps": bootstraps,
        "envelope_floor": envelope_floor,
    }

def plot_julia_karl_resolved_losvd_panels(targets, outdir, galaxy, prefix):
    matrix = np.asarray(targets["losvd_target"], float)
    sigma = np.asarray(targets["losvd_sigma"], float)
    counts = np.asarray(targets["counts_by_spatial"], float)
    spatial_edges = np.asarray(targets["spatial_edges_pc"], float)
    velocity_edges = np.asarray(targets["velocity_edges_kms"], float)
    centers = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])

    n = matrix.shape[0]
    ncols = 4 if n <= 16 else 5
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25 * ncols, 2.45 * nrows), sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ib in range(n):
        ax = axes[ib]
        target = matrix[ib]
        target_sigma = sigma[ib]
        valid_sigma = np.isfinite(target_sigma) & (target_sigma >= 0.0) & (target_sigma != -666.0)
        ax.step(centers, target, where="mid", linewidth=1.4, label="Julia target")
        if np.any(valid_sigma):
            lower = np.maximum(target - np.where(valid_sigma, target_sigma, 0.0), 0.0)
            upper = target + np.where(valid_sigma, target_sigma, 0.0)
            lower[~valid_sigma] = np.nan
            upper[~valid_sigma] = np.nan
            ax.fill_between(centers, lower, upper, where=valid_sigma, step="mid", alpha=0.2, label="Julia 1σ envelope")
        rmid = 0.5 * (spatial_edges[ib] + spatial_edges[ib + 1])
        ax.set_title(f"R={rmid:.0f} pc, N={int(round(counts[ib]))}", fontsize=9)
        ax.grid(alpha=0.2)

    for ax in axes[n:]:
        ax.axis("off")

    for ib, ax in enumerate(axes[:n]):
        if ib // ncols == nrows - 1:
            ax.set_xlabel(r"$v_{\rm los}-v_{\rm sys}$ [km s$^{-1}$]")
        if ib % ncols == 0:
            ax.set_ylabel("LOSVD target (total projected-light fraction)")

    if n > 0:
        axes[0].legend(fontsize=8)

    fig.suptitle(f"{galaxy} Julia karl_resolved_stars LOSVD Targets Used by OSPM ({n} apertures × {matrix.shape[1]} velocity bins)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = outdir / f"10_{prefix}_julia_karl_resolved_stars_losvd_panels.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{galaxy}] Julia LOSVD target = {path}")


def plot_observed_vs_julia_karl_resolved_probability(stars, targets, outdir, galaxy, prefix):
    matrix = np.asarray(targets["losvd_target"], float)
    sigma = np.asarray(targets["losvd_sigma"], float)
    counts = np.asarray(targets["counts_by_spatial"], float)
    spatial_edges = np.asarray(targets["spatial_edges_pc"], float)
    velocity_edges = np.asarray(targets["velocity_edges_kms"], float)
    centers = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])

    n = matrix.shape[0]
    if len(spatial_edges) != n + 1:
        raise ValueError("Julia resolved-star spatial grid does not match the LOSVD target matrix")
    if matrix.shape != sigma.shape:
        raise ValueError("Julia resolved-star LOSVD target and sigma matrices do not match")

    membership = assign_bins(stars["_r_pc"].to_numpy(float), spatial_edges)

    ncols = 4 if n <= 16 else 5
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.25 * ncols, 2.45 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    support_lo = float(targets["vmin_kms"])
    support_hi = float(targets["vmax_kms"])

    for ib in range(n):
        ax = axes[ib]
        sub = stars.loc[membership == ib]
        v = sub["_vlos_rel_kms"].to_numpy(float)
        ve = sub["_vlos_err_kms"].to_numpy(float)

        expected_count = int(round(counts[ib]))
        if len(sub) != expected_count:
            raise ValueError(
                f"Julia aperture {ib} contains {expected_count} stars, "
                f"but the plotting-side aperture assignment contains {len(sub)}"
            )

        target = np.asarray(matrix[ib], float)
        target_sigma = np.asarray(sigma[ib], float)

        support = (
            np.isfinite(target)
            & np.isfinite(target_sigma)
            & (target_sigma >= 0.0)
            & (target_sigma != -666.0)
        )
        if not np.any(support):
            raise ValueError(f"Julia aperture {ib} has no supported LOSVD bins")
        julia_sum = float(np.sum(target[support]))
        if not np.isfinite(julia_sum) or julia_sum <= 0.0:
            raise ValueError(f"Julia aperture {ib} has invalid supported LOSVD normalization={julia_sum}")
        julia_probability = target[support] / julia_sum
        julia_sigma_probability = target_sigma[support] / julia_sum
        observed = gaussian_losvd(v, ve, velocity_edges)
        observed_sum = float(np.sum(observed[support]))
        if not np.isfinite(observed_sum) or observed_sum <= 0.0:
            raise ValueError(f"Observed aperture {ib} has invalid supported LOSVD normalization={observed_sum}")
        observed_probability = observed[support] / observed_sum
        support_centers = centers[support]
        lower = np.maximum(julia_probability - julia_sigma_probability, 0.0)
        upper = julia_probability + julia_sigma_probability
        ax.step(support_centers, observed_probability, where="mid", linewidth=1.2, label="Observed stars")
        julia_line = ax.step(support_centers, julia_probability, where="mid", linewidth=1.5, label="Julia Karl-resolved")[0]
        ax.fill_between(support_centers, lower, upper, step="mid", alpha=0.18, color=julia_line.get_color(), label="Julia 1σ envelope")
        if len(v):
            ax.plot(v, np.full(len(v), 0.025), "|", transform=ax.get_xaxis_transform(), markersize=5, alpha=0.55, label="Individual stars" if ib == 0 else None)
        rmed = float(np.median(sub["_r_pc"])) if len(sub) else math.nan
        ax.set_title(f"Rmed={rmed:.0f} pc, N={len(sub)}", fontsize=9)
        ax.set_xlim(support_lo, support_hi)
        ax.grid(alpha=0.2)
    for ax in axes[n:]:
        ax.axis("off")
    for ib, ax in enumerate(axes[:n]):
        if ib // ncols == nrows - 1:
            ax.set_xlabel(r"$v_{\rm los}-v_{\rm sys}$ [km s$^{-1}$]")
        if ib % ncols == 0:
            ax.set_ylabel("Per-aperture probability")
    if n > 0:
        axes[0].legend(fontsize=7)
    fig.suptitle(f"{galaxy} Observed vs Julia karl_resolved_stars LOSVD Shape " f"({n} apertures; same stars, same Julia velocity bins)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = outdir / f"11_{prefix}_observed_vs_julia_karl_resolved_probability_losvd_panels.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{galaxy}] Julia probability comparison = {path}")

def load_karl_observables(path):
    df = pd.read_csv(path, low_memory=False)
    if "row_type" not in df.columns:
        raise KeyError(f"{path.name} is missing row_type")
    aperture = df.loc[df["row_type"].astype(str).eq("aperture")].copy()
    losvd = df.loc[df["row_type"].astype(str).eq("losvd_bin")].copy()
    spatial = df.loc[df["row_type"].astype(str).eq("spatial_cell")].copy()
    if aperture.empty:
        raise ValueError(f"{path.name} contains no aperture rows")
    if losvd.empty:
        raise ValueError(f"{path.name} contains no losvd_bin rows")
    if spatial.empty:
        raise ValueError(f"{path.name} contains no spatial_cell rows")
    return aperture, losvd, spatial

def karl_aperture_radial_ranges(aperture, spatial):
    ap = aperture.copy()
    sp = spatial.copy()
    for col in ["aperture_id", "ir_start", "ir_end"]:
        ap[col] = pd.to_numeric(ap[col], errors="coerce")
    for col in ["ir", "r_inner_pc", "r_outer_pc"]:
        sp[col] = pd.to_numeric(sp[col], errors="coerce")
    ap = ap.dropna(subset=["aperture_id", "ir_start", "ir_end"])
    sp = sp.dropna(subset=["ir", "r_inner_pc", "r_outer_pc"])
    ranges = {}
    for _, row in ap.iterrows():
        apid = int(row["aperture_id"])
        ir_start = int(row["ir_start"])
        ir_end = int(row["ir_end"])
        cells = sp[(sp["ir"] >= ir_start) & (sp["ir"] <= ir_end)]
        if cells.empty:
            ranges[apid] = (math.nan, math.nan)
        else:
            ranges[apid] = (float(cells["r_inner_pc"].min()), float(cells["r_outer_pc"].max()))
    return ranges

def plot_karl_dispersion(path, outdir, galaxy, prefix):
    aperture, _, spatial = load_karl_observables(path)
    ranges = karl_aperture_radial_ranges(aperture, spatial)
    for col in ["aperture_id", "aperture_sigma_kms", "aperture_sigma_lo_kms", "aperture_sigma_hi_kms"]:
        aperture[col] = pd.to_numeric(aperture[col], errors="coerce")
    aperture = aperture.dropna(subset=["aperture_id", "aperture_sigma_kms", "aperture_sigma_lo_kms", "aperture_sigma_hi_kms"]).sort_values("aperture_id")
    x, xlo, xhi, y, ylo, yhi = [], [], [], [], [], []
    for _, row in aperture.iterrows():
        apid = int(row["aperture_id"])
        rin, rout = ranges.get(apid, (math.nan, math.nan))
        if not (np.isfinite(rin) and np.isfinite(rout) and rout > rin):
            continue
        rmid = 0.5 * (rin + rout)
        sigma = float(row["aperture_sigma_kms"])
        x.append(rmid)
        xlo.append(rmid - rin)
        xhi.append(rout - rmid)
        y.append(sigma)
        ylo.append(sigma - float(row["aperture_sigma_lo_kms"]))
        yhi.append(float(row["aperture_sigma_hi_kms"]) - sigma)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(np.asarray(x), np.asarray(y), xerr=np.vstack([xlo, xhi]), yerr=np.vstack([ylo, yhi]), fmt="o", capsize=3)
    ax.set_xlabel("Karl-observables radial aperture [pc]")
    ax.set_ylabel(r"Intrinsic $\sigma_{\rm los}$ [km s$^{-1}$]")
    ax.set_title(f"{galaxy} Velocity-Dispersion Profile from Karl Observables CSV")
    ax.grid(alpha=0.25)
    savefig(fig, outdir / f"10_{prefix}_karl_velocity_dispersion_profile.png")

def karl_losvd_matrix(path):
    aperture, losvd, spatial = load_karl_observables(path)
    ranges = karl_aperture_radial_ranges(aperture, spatial)
    losvd["aperture_id"] = pd.to_numeric(losvd["aperture_id"], errors="coerce")
    losvd["velocity_bin"] = pd.to_numeric(losvd["velocity_bin"], errors="coerce")
    losvd["velocity_low_kms"] = pd.to_numeric(losvd["velocity_low_kms"], errors="coerce")
    losvd["velocity_center_kms"] = pd.to_numeric(losvd["velocity_center_kms"], errors="coerce")
    losvd["velocity_high_kms"] = pd.to_numeric(losvd["velocity_high_kms"], errors="coerce")
    losvd["losvd_target"] = pd.to_numeric(losvd["losvd_target"], errors="coerce")
    losvd = losvd.dropna(subset=["aperture_id", "velocity_bin", "velocity_low_kms", "velocity_center_kms", "velocity_high_kms", "losvd_target"])
    losvd["aperture_id"] = losvd["aperture_id"].astype(int)
    losvd["velocity_bin"] = losvd["velocity_bin"].astype(int)
    aperture_ids = sorted(losvd["aperture_id"].unique())
    velocity_bins = sorted(losvd["velocity_bin"].unique())
    matrix = np.full((len(aperture_ids), len(velocity_bins)), np.nan)
    for ia, apid in enumerate(aperture_ids):
        sub = losvd.loc[losvd["aperture_id"].eq(apid)].set_index("velocity_bin")
        for iv, vbin in enumerate(velocity_bins):
            if vbin in sub.index:
                matrix[ia, iv] = float(sub.loc[vbin, "losvd_target"])
    first_ap = losvd.loc[losvd["aperture_id"].eq(aperture_ids[0])].sort_values("velocity_bin")
    velocity_edges = np.r_[first_ap["velocity_low_kms"].to_numpy(float)[0], first_ap["velocity_high_kms"].to_numpy(float)]
    centers = first_ap["velocity_center_kms"].to_numpy(float)
    aperture["aperture_id"] = pd.to_numeric(aperture["aperture_id"], errors="coerce")
    aperture["star_count"] = pd.to_numeric(aperture["star_count"], errors="coerce")
    aperture["aperture_light"] = pd.to_numeric(aperture["aperture_light"], errors="coerce")
    aperture = aperture.dropna(subset=["aperture_id"]).copy()
    aperture["aperture_id"] = aperture["aperture_id"].astype(int)
    aperture = aperture.set_index("aperture_id")

    return aperture_ids, velocity_bins, velocity_edges, centers, matrix, aperture, ranges

def plot_karl_losvd_heatmap(path, outdir, galaxy, prefix):
    aperture_ids, velocity_bins, velocity_edges, centers, matrix, aperture, ranges = karl_losvd_matrix(path)
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(matrix, origin="lower", aspect="auto", extent=[velocity_edges[0], velocity_edges[-1], -0.5, len(aperture_ids) - 0.5])
    ax.set_xlabel(r"$v_{\rm los}-v_{\rm sys}$ [km s$^{-1}$]")
    ax.set_ylabel("Karl LOSVD radial aperture")
    ax.set_title(f"{galaxy} Karl Observables LOSVD Targets")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("LOSVD target")
    yticks = np.arange(len(aperture_ids))
    if len(yticks) > 20:
        yticks = yticks[::2]
    labels = []
    for i in yticks:
        apid = aperture_ids[i]
        rin, rout = ranges.get(apid, (math.nan, math.nan))
        rmid = 0.5 * (rin + rout) if np.isfinite(rin) and np.isfinite(rout) else math.nan
        labels.append(f"{apid}: {rmid:.0f} pc" if np.isfinite(rmid) else f"{apid}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    savefig(fig, outdir / f"11_{prefix}_karl_losvd_heatmap.png")

def plot_karl_losvd_panels(path, outdir, galaxy, prefix):
    aperture_ids, velocity_bins, velocity_edges, centers, matrix, aperture, ranges = karl_losvd_matrix(path)
    n = len(aperture_ids)
    ncols = 4 if n <= 16 else 5
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25 * ncols, 2.35 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ia, apid in enumerate(aperture_ids):
        ax = axes[ia]
        ax.step(centers, matrix[ia], where="mid")
        row = aperture.loc[apid]
        rin, rout = ranges.get(apid, (math.nan, math.nan))
        rmid = 0.5 * (rin + rout) if np.isfinite(rin) and np.isfinite(rout) else math.nan
        nstar = int(row["star_count"]) if np.isfinite(float(row["star_count"])) else 0
        title = f"R={rmid:.0f} pc, N={nstar}" if np.isfinite(rmid) else f"Aperture {apid}, N={nstar}"
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes[n:]:
        ax.axis("off")
    for ia, ax in enumerate(axes[:n]):
        if ia // ncols == nrows - 1:
            ax.set_xlabel(r"$v_{\rm los}-v_{\rm sys}$")
        if ia % ncols == 0:
            ax.set_ylabel("LOSVD target")
    fig.suptitle(f"{galaxy} Karl Observables LOSVDs ({n} spatial apertures × {len(velocity_bins)} velocity bins)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / f"12_{prefix}_karl_losvd_panels.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def analyze_losvd(stars, bins, edges, nvbin):
    membership = assign_bins(stars["_r_pc"].to_numpy(float), edges)
    velocity_edges = velocity_edges_auto(stars, nvbin)
    rows = []
    losvds = []
    for ib in range(len(edges) - 1):
        mask = membership == ib
        sub = stars.loc[mask]
        v = sub["_vlos_rel_kms"].to_numpy(float)
        ve = sub["_vlos_err_kms"].to_numpy(float)
        sigma, sigma_lo, sigma_hi, mu = intrinsic_dispersion_mle(v, ve)
        losvd = gaussian_losvd(v, ve, velocity_edges)
        losvds.append(losvd)
        rin = float(edges[ib])
        rout = float(edges[ib + 1])
        rmed = float(np.median(sub["_r_pc"])) if len(sub) else math.nan
        rows.append({
            "bin_id": int(bins.iloc[ib]["bin_id"]) if ib < len(bins) else ib,
            "R_inner_pc": rin,
            "R_outer_pc": rout,
            "R_mid_pc": 0.5 * (rin + rout),
            "R_median_pc": rmed,
            "N_vlos": int(len(sub)),
            "mean_vlos_rel_kms": mu,
            "sigma_los_kms": sigma,
            "sigma_los_lo68_kms": sigma_lo,
            "sigma_los_hi68_kms": sigma_hi,
        })
    profile = pd.DataFrame(rows)
    return profile, np.asarray(losvds), velocity_edges, membership

def plot_raw_vlos(stars, edges, outdir, galaxy, prefix):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(stars["_r_pc"], stars["_vlos_rel_kms"], yerr=stars["_vlos_err_kms"], fmt=".", markersize=4, linewidth=0.5, alpha=0.55)
    for edge in edges[1:-1]:
        ax.axvline(edge, linewidth=0.6, alpha=0.25)
    ax.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Projected radius R [pc]")
    ax.set_ylabel(r"$v_{\rm los}-v_{\rm sys}$ [km s$^{-1}$]")
    ax.set_title(f"{galaxy} LOS Velocities and LOSVD Radial Apertures")
    ax.grid(alpha=0.2)
    savefig(fig, outdir / f"04_{prefix}_los_velocities.png")

def plot_dispersion(profile, outdir, galaxy, prefix):
    good = (
        np.isfinite(profile["R_median_pc"]) &
        np.isfinite(profile["sigma_los_kms"]) &
        np.isfinite(profile["sigma_los_lo68_kms"]) &
        np.isfinite(profile["sigma_los_hi68_kms"]) &
        (profile["N_vlos"] >= 2)
    )

    p = profile.loc[good].copy()
    x = p["R_median_pc"].to_numpy(float)
    y = p["sigma_los_kms"].to_numpy(float)
    xlo = x - p["R_inner_pc"].to_numpy(float)
    xhi = p["R_outer_pc"].to_numpy(float) - x
    ylo = y - p["sigma_los_lo68_kms"].to_numpy(float)
    yhi = p["sigma_los_hi68_kms"].to_numpy(float) - y
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(x, y, xerr=np.vstack([xlo, xhi]), yerr=np.vstack([ylo, yhi]), fmt="o", capsize=3)
    ax.set_xlabel("Median projected radius in LOSVD aperture [pc]")
    ax.set_ylabel(r"Intrinsic $\sigma_{\rm los}$ [km s$^{-1}$]")
    ax.set_title(f"{galaxy} Velocity-Dispersion Profile from LOSVD Apertures")
    ax.grid(alpha=0.25)
    savefig(fig, outdir / f"05_{prefix}_velocity_dispersion_profile.png")

def plot_losvd_heatmap(profile, losvds, velocity_edges, outdir, galaxy, prefix):
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(losvds, origin="lower", aspect="auto", extent=[velocity_edges[0], velocity_edges[-1], -0.5, len(profile) - 0.5])
    ax.set_xlabel(r"$v_{\rm los}-v_{\rm sys}$ [km s$^{-1}$]")
    ax.set_ylabel("LOSVD radial aperture")
    ax.set_title(f"{galaxy} Observed LOSVDs")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("LOSVD probability")
    yticks = np.arange(len(profile))
    if len(profile) > 20:
        yticks = yticks[::2]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(profile.iloc[i]['bin_id'])}: {profile.iloc[i]['R_median_pc']:.0f} pc" for i in yticks])
    savefig(fig, outdir / f"06_{prefix}_losvd_heatmap.png")

def plot_losvd_panels(profile, losvds, velocity_edges, outdir, galaxy, prefix):
    n = len(profile)
    ncols = 4 if n <= 16 else 5
    nrows = int(math.ceil(n / ncols))
    mids = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25 * ncols, 2.35 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ib in range(n):
        ax = axes[ib]
        ax.step(mids, losvds[ib], where="mid")
        row = profile.iloc[ib]
        ax.set_title(f"R={row.R_median_pc:.0f} pc, N={int(row.N_vlos)}", fontsize=9)
        ax.grid(alpha=0.2)

    for ax in axes[n:]:
        ax.axis("off")

    for ib, ax in enumerate(axes[:n]):
        if ib // ncols == nrows - 1:
            ax.set_xlabel(r"$v_{\rm los}-v_{\rm sys}$")
        if ib % ncols == 0:
            ax.set_ylabel("Probability")

    fig.suptitle(f"{galaxy} Observed LOSVDs ({len(profile)} spatial apertures × {len(velocity_edges) - 1} velocity bins)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / f"07_{prefix}_losvd_panels.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_observed_vs_karl_losvd(profile, observed_losvds, observed_velocity_edges, karl_path, outdir, galaxy, prefix):
    aperture_ids, velocity_bins, karl_velocity_edges, karl_centers, karl_matrix, aperture, ranges = karl_losvd_matrix(karl_path)

    n = min(len(profile), len(aperture_ids))
    ncols = 4 if n <= 16 else 5
    nrows = int(math.ceil(n / ncols))
    observed_centers = 0.5 * (observed_velocity_edges[:-1] + observed_velocity_edges[1:])

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25 * ncols, 2.45 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ia in range(n):
        ax = axes[ia]
        apid = aperture_ids[ia]
        row = aperture.loc[apid]

        aperture_light = float(row["aperture_light"])
        if not np.isfinite(aperture_light) or aperture_light <= 0.0:
            raise ValueError(f"Karl aperture {apid} has invalid aperture_light={aperture_light}")

        karl_probability = np.asarray(karl_matrix[ia], float) / aperture_light
        karl_sum = float(np.nansum(karl_probability))
        if not np.isfinite(karl_sum) or karl_sum <= 0.0:
            raise ValueError(f"Karl aperture {apid} has invalid normalized LOSVD sum={karl_sum}")
        karl_probability /= karl_sum

        observed = np.asarray(observed_losvds[ia], float)
        observed_sum = float(np.nansum(observed))
        if np.isfinite(observed_sum) and observed_sum > 0.0:
            observed = observed / observed_sum

        ax.step(observed_centers, observed, where="mid", linewidth=1.2, label="Observed stars")
        ax.step(karl_centers, karl_probability, where="mid", linewidth=1.5, label="Karl Gaussian target")

        rin, rout = ranges.get(apid, (math.nan, math.nan))
        if np.isfinite(rin) and np.isfinite(rout):
            rmid = 0.5 * (rin + rout)
        else:
            rmid = float(profile.iloc[ia]["R_median_pc"])
        nstar = int(row["star_count"]) if np.isfinite(float(row["star_count"])) else int(profile.iloc[ia]["N_vlos"])
        ax.set_title(f"R={rmid:.0f} pc, N={nstar}", fontsize=9)
        ax.grid(alpha=0.2)

    for ax in axes[n:]:
        ax.axis("off")

    for ia, ax in enumerate(axes[:n]):
        if ia // ncols == nrows - 1:
            ax.set_xlabel(r"$v_{\rm los}-v_{\rm sys}$")
        if ia % ncols == 0:
            ax.set_ylabel("Per-aperture probability")

    if n > 0:
        axes[0].legend(fontsize=8)

    fig.suptitle(f"{galaxy} Observed LOSVDs vs Karl Gaussian Targets")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / f"13_{prefix}_observed_vs_karl_losvd_panels.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot the observational inputs for the galaxy selected by OSPM/load_config.py.")
    parser.add_argument("--galaxy", default=None, help="Optional override for which_galaxy / OSPM_GALAXY.")
    parser.add_argument("--profile-root", type=Path, default=None, help="Optional root used to resolve relative config paths and the default plots directory.")
    parser.add_argument("--stars", type=Path, default=None, help="Override CONFIG['DATA_CSV'].")
    parser.add_argument("--surface-brightness", type=Path, default=None, help="Override CONFIG['SURFACE_BRIGHTNESS_CSV'].")
    parser.add_argument("--losvd-bins", type=Path, default=None, help="Override CONFIG['KINEMATIC_BINS_CSV'].")
    parser.add_argument("--tracer-density", type=Path, default=None, help="Override the active STELLAR_MODEL tracer grid.")
    parser.add_argument("--stellar-force-grid", type=Path, default=None, help="Override CONFIG['STELLAR_MODEL']['grid_csv'].")
    parser.add_argument("--karl-observables", type=Path, default=None, help="Override CONFIG['KARL_OBSERVABLES_CSV'].")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--v-sys", type=float, default=None, help="Override CONFIG['V_SYS_KMS'].")
    parser.add_argument("--nvbin", type=int, default=None, help="Override CONFIG['OBSERVABLES']['NVBIN'].")
    args = parser.parse_args()
    config, config_profile_root, repo_root = load_galaxy_config(args.galaxy)
    galaxy = str(config["GALAXY"])
    prefix = safe_prefix(galaxy)
    profile_root = args.profile_root.resolve() if args.profile_root is not None else config_profile_root
    stars_path = args.stars.resolve() if args.stars is not None else resolve_config_path(config["DATA_CSV"], profile_root, repo_root)
    sb_path = args.surface_brightness.resolve() if args.surface_brightness is not None else resolve_config_path(config["SURFACE_BRIGHTNESS_CSV"], profile_root, repo_root)
    losvd_bins_path = args.losvd_bins.resolve() if args.losvd_bins is not None else resolve_config_path(config["KINEMATIC_BINS_CSV"], profile_root, repo_root)
    tracer_path = args.tracer_density.resolve() if args.tracer_density is not None else active_tracer_path(config, profile_root, repo_root)
    stellar_model = config.get("STELLAR_MODEL", {}) or {}
    force_path = args.stellar_force_grid.resolve() if args.stellar_force_grid is not None else resolve_config_path(stellar_model["grid_csv"], profile_root, repo_root)
    losvd_target_mode = str(observable_option(config, "LOSVD_TARGET_MODE", "losvd_target_mode", default="current")).strip().lower()
    if losvd_target_mode == "karl_mode0_observables":
        karl_path = args.karl_observables.resolve() if args.karl_observables is not None else resolve_config_path(config["KARL_OBSERVABLES_CSV"], profile_root, repo_root)
    else:
        karl_path = None
    r_col = str(config.get("STAR_R_COL", "R_pc"))
    v_col = str(config.get("STAR_V_COL", config.get("VLOS_COL", "vlos_kms")))
    verr_col = str(config.get("STAR_VERR_COL", "verr_kms"))
    v_sys = float(args.v_sys) if args.v_sys is not None else resolve_v_sys(stars_path, config.get("V_SYS_KMS"))
    nvbin = int(args.nvbin) if args.nvbin is not None else int(config.get("OBSERVABLES", {}).get("NVBIN", 21))
    if nvbin < 1:
        raise ValueError(f"NVBIN must be positive; got {nvbin}")

    outdir = args.outdir.resolve() if args.outdir is not None else profile_root / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    required_paths = [stars_path, sb_path, losvd_bins_path, tracer_path, force_path]

    if karl_path is not None:
        required_paths.append(karl_path)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n" + "\n".join(missing))

    print(f"[{galaxy}] profile root       = {profile_root}")
    print(f"[{galaxy}] stars              = {stars_path}")
    print(f"[{galaxy}] surface brightness = {sb_path}")
    print(f"[{galaxy}] LOSVD bins         = {losvd_bins_path}")
    print(f"[{galaxy}] 3D tracer density  = {tracer_path}")
    print(f"[{galaxy}] stellar force grid = {force_path}")
    print(f"[{galaxy}] LOSVD target mode  = {losvd_target_mode}")
    if karl_path is not None:
        print(f"[{galaxy}] Karl observables   = {karl_path}")
    elif losvd_target_mode == "karl_resolved_stars":
        print(f"[{galaxy}] Karl observables   = bypassed; resolved-star target is built in Julia")
    print(f"[{galaxy}] tracer mode        = {config.get('TRACER_CONSTRAINT_MODE', 'projected_light')}")
    print(f"[{galaxy}] star columns       = R:{r_col} v:{v_col} verr:{verr_col}")
    print(f"[{galaxy}] v_sys [km/s]       = {v_sys}")
    print(f"[{galaxy}] velocity bins      = {nvbin}")
    print(f"[{galaxy}] plots              = {outdir}")

    stars = load_stars(stars_path, v_sys, r_col, v_col, verr_col)
    if stars.empty:
        raise ValueError(f"{stars_path.name} contains no valid LOS velocity stars after configured-column filtering")
    bins, edges = load_losvd_bins(losvd_bins_path, stars)
    plot_surface_brightness(sb_path, outdir, galaxy, prefix)
    tracer, geometry, Rcol, zcol, mcol, nucol, volcol = load_tracer_density(tracer_path, stellar_model)
    plot_tracer_density_radial(tracer, mcol, nucol, volcol, force_path, stellar_model, outdir, galaxy, prefix)
    if Rcol is not None and zcol is not None:
        plot_tracer_density_map(tracer, Rcol, zcol, nucol, outdir, galaxy, prefix)
    else:
        print(f"[{galaxy}] 3D map skipped      = geometry {geometry!r} has no configured R_cyl/z columns")
    profile, losvds, velocity_edges, membership = analyze_losvd(stars, bins, edges, nvbin)
    profile.to_csv(outdir / f"{prefix}_velocity_dispersion_profile.csv", index=False)
    plot_raw_vlos(stars, edges, outdir, galaxy, prefix)
    plot_dispersion(profile, outdir, galaxy, prefix)
    plot_losvd_heatmap(profile, losvds, velocity_edges, outdir, galaxy, prefix)
    plot_losvd_panels(profile, losvds, velocity_edges, outdir, galaxy, prefix)
    plot_stellar_force_density_profile(force_path, stellar_model, outdir, galaxy, prefix)
    plot_tracer_vs_stellar_force_inner(tracer_path, force_path, stellar_model, outdir, galaxy, prefix)

    if losvd_target_mode == "karl_resolved_stars":
        julia_targets = load_julia_karl_resolved_targets(config, stars_path, sb_path, losvd_bins_path, force_path, tracer_path, v_sys_override=args.v_sys, nvbin_override=args.nvbin)
        plot_julia_karl_resolved_losvd_panels(julia_targets, outdir, galaxy, prefix)
        plot_observed_vs_julia_karl_resolved_probability(stars, julia_targets, outdir, galaxy, prefix)

    elif losvd_target_mode == "karl_mode0_observables":
        plot_karl_dispersion(karl_path, outdir, galaxy, prefix)
        plot_karl_losvd_heatmap(karl_path, outdir, galaxy, prefix)
        plot_karl_losvd_panels(karl_path, outdir, galaxy, prefix)
        plot_observed_vs_karl_losvd(profile, losvds, velocity_edges, karl_path, outdir, galaxy, prefix)

    n_assigned = int(np.sum(membership >= 0))
    n_unassigned = int(np.sum(membership < 0))
    final_n = int(profile.iloc[-1]["N_vlos"]) if len(profile) else 0

    print()
    print(f"[{galaxy}] valid LOS stars    = {len(stars)}")
    print(f"[{galaxy}] LOSVD apertures    = {len(profile)}")
    print(f"[{galaxy}] velocity bins      = {nvbin}")
    print(f"[{galaxy}] assigned stars     = {n_assigned}")
    print(f"[{galaxy}] unassigned stars   = {n_unassigned}")
    print(f"[{galaxy}] final aperture N   = {final_n}")
    print()
    print("Outputs:")
    for path in sorted(outdir.iterdir()):
        if prefix in path.name:
            print(f"  {path.name}")

if __name__ == "__main__":
    main()
