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

    raise FileNotFoundError(
        "Could not find the OSPM repository root. Expected OSPM/load_config.py "
        "and Data/Galaxy_Profiles/. Run this from inside the OSPM repository."
    )

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

    raise ValueError(
        "V_SYS_KMS is not defined in the active galaxy config and the star table "
        "does not provide a finite v_sys_kms column."
    )

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
        bins.loc[len(bins) - 1, "R_mid_pc"] = 0.5 * (
            float(bins.loc[len(bins) - 1, "R_inner_pc"]) +
            float(bins.loc[len(bins) - 1, "R_outer_pc"])
        )

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

    result = minimize_scalar(
        lambda s: profile_nll_sigma(s, v, verr)[0],
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-8},
    )
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

    work = pd.DataFrame({
        "R": pd.to_numeric(df[rcol], errors="coerce"),
        "Sigma": pd.to_numeric(df[scol], errors="coerce"),
    })
    if ecol is not None:
        work["Sigma_err"] = pd.to_numeric(df[ecol], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["R", "Sigma"])
    work = work[(work["R"] > 0.0) & (work["Sigma"] > 0.0)]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if ecol is not None:
        err = work["Sigma_err"].to_numpy(float)
        gooderr = np.isfinite(err) & (err >= 0.0)
        ax.errorbar(
            work.loc[gooderr, "R"],
            work.loc[gooderr, "Sigma"],
            yerr=work.loc[gooderr, "Sigma_err"],
            fmt="o",
            capsize=3,
        )
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

    nucol = first_col(df, [
        stellar_model.get("nu_col"),
        "nu_Lsun_pc3", "nu_stars_pc3", "number_density_stars_pc3", "density", "rho",
    ])
    volcol = first_col(df, [stellar_model.get("volume_col"), "cell_volume_pc3", "volume_pc3"], required=False)

    radius_candidates = [stellar_model.get("radius_col"), "m_pc", "r_pc", "radius_pc"]
    mcol = first_col(df, radius_candidates, required=False)

    Rcol = first_col(df, [stellar_model.get("R_cyl_col"), "R_cyl_pc", "R_pc"], required=False)
    zcol = first_col(df, [stellar_model.get("z_col"), "z_pc", "z"], required=False)

    if mcol is None and Rcol is not None and zcol is not None:
        df = df.copy()
        df["_m_plot_pc"] = np.sqrt(
            pd.to_numeric(df[Rcol], errors="coerce") ** 2 +
            pd.to_numeric(df[zcol], errors="coerce") ** 2
        )
        mcol = "_m_plot_pc"

    if mcol is None:
        raise KeyError(
            f"Could not determine an intrinsic radius column for {path.name}. "
            f"Geometry={geometry!r}; available columns={list(df.columns)}"
        )

    return df, geometry, Rcol, zcol, mcol, nucol, volcol

def plot_tracer_density_radial(df, mcol, nucol, volcol, outdir, galaxy, prefix):
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

    radial = pd.DataFrame(rows, columns=["radius_pc", "tracer_density"])
    radial = radial[np.isfinite(radial["tracer_density"]) & (radial["tracer_density"] > 0.0)].sort_values("radius_pc")
    radial.to_csv(outdir / f"{prefix}_tracer_density_3d_radial_profile.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(radial["radius_pc"], radial["tracer_density"], "o-", markersize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Intrinsic radius $m$ [pc]")

    if "lsun" in nucol.lower():
        ax.set_ylabel(r"3D tracer density $\nu_L$ [$L_\odot$ pc$^{-3}$]")
    else:
        ax.set_ylabel(r"3D tracer density $\nu$")

    ax.set_title(f"{galaxy} 3D Tracer-Density Profile")
    ax.grid(alpha=0.25)
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

    smooth = ax.tricontourf(
        R_plot,
        z_plot,
        lognu_plot,
        levels=256,
        cmap="inferno",
        vmin=float(np.min(lognu)),
        vmax=float(np.max(lognu)),
    )

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

    fig.savefig(
        outdir / f"03_{prefix}_tracer_density_3d_map.png",
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
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
    ax.errorbar(
        stars["_r_pc"],
        stars["_vlos_rel_kms"],
        yerr=stars["_vlos_err_kms"],
        fmt=".",
        markersize=4,
        linewidth=0.5,
        alpha=0.55)

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
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([xlo, xhi]),
        yerr=np.vstack([ylo, yhi]),
        fmt="o",
        capsize=3,
    )
    ax.set_xlabel("Median projected radius in LOSVD aperture [pc]")
    ax.set_ylabel(r"Intrinsic $\sigma_{\rm los}$ [km s$^{-1}$]")
    ax.set_title(f"{galaxy} Velocity-Dispersion Profile from LOSVD Apertures")
    ax.grid(alpha=0.25)
    savefig(fig, outdir / f"05_{prefix}_velocity_dispersion_profile.png")

def plot_losvd_heatmap(profile, losvds, velocity_edges, outdir, galaxy, prefix):
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(
        losvds,
        origin="lower",
        aspect="auto",
        extent=[velocity_edges[0], velocity_edges[-1], -0.5, len(profile) - 0.5],
    )
    ax.set_xlabel(r"$v_{\rm los}-v_{\rm sys}$ [km s$^{-1}$]")
    ax.set_ylabel("LOSVD radial aperture")
    ax.set_title(f"{galaxy} Observed LOSVDs")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("LOSVD probability")

    yticks = np.arange(len(profile))
    if len(profile) > 20:
        yticks = yticks[::2]
    ax.set_yticks(yticks)
    ax.set_yticklabels([
        f"{int(profile.iloc[i]['bin_id'])}: {profile.iloc[i]['R_median_pc']:.0f} pc"
        for i in yticks
    ])

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

def main():
    parser = argparse.ArgumentParser(
        description="Plot the observational inputs for the galaxy selected by OSPM/load_config.py."
    )
    parser.add_argument("--galaxy", default=None, help="Optional override for which_galaxy / OSPM_GALAXY.")
    parser.add_argument("--profile-root", type=Path, default=None, help="Optional root used to resolve relative config paths and the default plots directory.")
    parser.add_argument("--stars", type=Path, default=None, help="Override CONFIG['DATA_CSV'].")
    parser.add_argument("--surface-brightness", type=Path, default=None, help="Override CONFIG['SURFACE_BRIGHTNESS_CSV'].")
    parser.add_argument("--losvd-bins", type=Path, default=None, help="Override CONFIG['KINEMATIC_BINS_CSV'].")
    parser.add_argument("--tracer-density", type=Path, default=None, help="Override the active STELLAR_MODEL tracer grid.")
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
    r_col = str(config.get("STAR_R_COL", "R_pc"))
    v_col = str(config.get("STAR_V_COL", config.get("VLOS_COL", "vlos_kms")))
    verr_col = str(config.get("STAR_VERR_COL", "verr_kms"))
    v_sys = float(args.v_sys) if args.v_sys is not None else resolve_v_sys(stars_path, config.get("V_SYS_KMS"))
    nvbin = int(args.nvbin) if args.nvbin is not None else int(config.get("OBSERVABLES", {}).get("NVBIN", 21))

    if nvbin < 1:
        raise ValueError(f"NVBIN must be positive; got {nvbin}")

    outdir = args.outdir.resolve() if args.outdir is not None else profile_root / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    required_paths = [stars_path, sb_path, losvd_bins_path, tracer_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n" + "\n".join(missing))

    print(f"[{galaxy}] profile root       = {profile_root}")
    print(f"[{galaxy}] stars              = {stars_path}")
    print(f"[{galaxy}] surface brightness = {sb_path}")
    print(f"[{galaxy}] LOSVD bins         = {losvd_bins_path}")
    print(f"[{galaxy}] 3D tracer density  = {tracer_path}")
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
    plot_tracer_density_radial(tracer, mcol, nucol, volcol, outdir, galaxy, prefix)

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