#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}

def first_finite(series, default=np.nan):
    arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    return float(arr[0]) if arr.size else default

def interp_loglog(x_old, y_old, x_new):
    x_old = np.asarray(x_old, float)
    y_old = np.asarray(y_old, float)
    x_new = np.asarray(x_new, float)
    good = np.isfinite(x_old) & np.isfinite(y_old) & (x_old > 0) & (y_old > 0)
    if np.count_nonzero(good) < 2:
        raise ValueError("Need at least two positive finite points for log-log interpolation.")
    lx = np.log(x_old[good])
    ly = np.log(y_old[good])
    order = np.argsort(lx)
    lx = lx[order]
    ly = ly[order]
    lx_new = np.log(np.maximum(x_new, np.exp(lx[0])))
    ly_new = np.interp(lx_new, lx, ly, left=ly[0], right=ly[-1])
    return np.exp(ly_new)

def interp_linear(x_old, y_old, x_new):
    x_old = np.asarray(x_old, float)
    y_old = np.asarray(y_old, float)
    x_new = np.asarray(x_new, float)
    good = np.isfinite(x_old) & np.isfinite(y_old)
    if np.count_nonzero(good) < 2:
        raise ValueError("Need at least two finite points for linear interpolation.")
    order = np.argsort(x_old[good])
    return np.interp( x_new, x_old[good][order], y_old[good][order], left=y_old[good][order][0], right=y_old[good][order][-1] )

def make_min_count_bins(r_pc, min_per_bin=20, drop_partial=False):
    r = np.sort(np.asarray(r_pc, float))
    r = r[np.isfinite(r)]
    if len(r) == 0:
        raise ValueError("No finite radii for kinematic bins.")
    edges = [r[0]]
    rows = []
    i = 0
    bin_id = 0
    while i < len(r):
        j = min(i + min_per_bin, len(r))
        nbin = j - i
        if drop_partial and nbin < min_per_bin:
            break
        if nbin <= 0:
            break
        if j < len(r):
            edge_out = 0.5 * (r[j - 1] + r[j])
        else:
            edge_out = r[j - 1]
        if edge_out <= edges[-1]:
            edge_out = edges[-1] + max(abs(edges[-1]), 1.0) * 1e-9
        edges.append(edge_out)
        rows.append({"bin_id": bin_id, "R_inner_pc": edges[-2], "R_outer_pc": edges[-1], "R_mid_pc": 0.5 * (edges[-2] + edges[-1]), "N_vlos": nbin })
        i = j
        bin_id += 1
    return np.asarray(edges, float), pd.DataFrame(rows)


def plot_ellipse(ax, radius_pc, q, **kwargs):
    t = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(radius_pc * np.cos(t), q * radius_pc * np.sin(t), **kwargs)


def warn_if_extrapolating(R_old, R_new, label="R_pc"):
    R_old = np.asarray(R_old, float)
    R_new = np.asarray(R_new, float)
    old_good = R_old[np.isfinite(R_old)]
    new_good = R_new[np.isfinite(R_new)]
    if old_good.size == 0 or new_good.size == 0:
        return
    lo = np.nanmin(old_good)
    hi = np.nanmax(old_good)
    below = int(np.count_nonzero(new_good < lo))
    above = int(np.count_nonzero(new_good > hi))
    if below or above:
        print(
            f"WARNING: {below + above} target {label} values fall outside the source surface-brightness range "
            f"[{lo:.6g}, {hi:.6g}]. Edge extrapolation will be used."
        )


def validate_surface_brightness_bins(out, bins):
    required = {"R_inner_pc", "R_outer_pc", "R_pc", "light_frac", "area_pc2"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"rebinned surface-brightness output missing columns: {sorted(missing)}")
    if len(out) != len(bins):
        raise ValueError(f"rebinned light rows do not match target bins: {len(out)} vs {len(bins)}")
    if not np.all(np.isfinite(out["R_inner_pc"])) or not np.all(np.isfinite(out["R_outer_pc"])):
        raise ValueError("rebinned surface-brightness bin edges contain non-finite values")
    if not np.all(out["R_outer_pc"].to_numpy(float) > out["R_inner_pc"].to_numpy(float)):
        raise ValueError("rebinned surface-brightness bins require R_outer_pc > R_inner_pc")
    light = out["light_frac"].to_numpy(float)
    if not np.all(np.isfinite(light)) or np.any(light < 0.0):
        raise ValueError("light_frac must be finite and non-negative")
    lsum = float(np.sum(light))
    if not np.isclose(lsum, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"light_frac must sum to 1; got {lsum:.16g}")
    area = out["area_pc2"].to_numpy(float)
    if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("area_pc2 must be finite and positive")
    return True


def validate_light_grid(out, ltot):
    required = {"r_pc", "R_inner_pc", "R_outer_pc", "theta_rad", "nu_Lsun_pc3", "cell_luminosity_Lsun", "light_frac", "Lenc_frac", "geometry"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"spherical light grid missing columns: {sorted(missing)}")
    r = out["r_pc"].to_numpy(float)
    lum = out["cell_luminosity_Lsun"].to_numpy(float)
    lenc = out["Lenc_frac"].to_numpy(float)
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("spherical light grid radii must be positive and finite")
    if not np.all(np.isfinite(lum)) or np.any(lum < 0.0):
        raise ValueError("spherical light grid cell luminosities must be finite and non-negative")
    Lsum = float(np.sum(lum))
    if not np.isclose(Lsum, float(ltot), rtol=1e-8, atol=max(1e-8, 1e-10 * abs(float(ltot)))):
        raise ValueError(f"spherical light grid luminosity sum {Lsum:.16g} does not match Ltot {float(ltot):.16g}")
    unique_lenc = pd.Series(lenc[np.isfinite(lenc)]).drop_duplicates().to_numpy(float)
    if unique_lenc.size == 0 or np.any(np.diff(unique_lenc) < -1e-12):
        raise ValueError("Lenc_frac must be monotonic non-decreasing")
    if not np.isclose(np.nanmax(lenc), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError(f"Lenc_frac should end at 1; got {np.nanmax(lenc):.16g}")I 
    return True


def validate_axisymmetric_grid(out, ltot):
    required = {
        "R_cyl_pc", "z_pc", "r_pc", "m_pc", "theta_rad", "q_axis_ratio",
        "nu_Lsun_pc3", "cell_volume_pc3", "cell_luminosity_Lsun",
        "geometry", "flattened_geometry", "density_coordinate",
    }
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"axisymmetric light grid missing columns: {sorted(missing)}")

    if not (out["geometry"].astype(str) == "axisymmetric_density_grid").all():
        raise ValueError("axisymmetric grid geometry must be axisymmetric_density_grid")

    q = out["q_axis_ratio"].to_numpy(float)
    R = out["R_cyl_pc"].to_numpy(float)
    z = out["z_pc"].to_numpy(float)
    vol = out["cell_volume_pc3"].to_numpy(float)
    lum = out["cell_luminosity_Lsun"].to_numpy(float)
    nu = out["nu_Lsun_pc3"].to_numpy(float)

    if not np.all(np.isfinite(q)) or np.any(q <= 0.0):
        raise ValueError("q_axis_ratio must be positive and finite")
    if not np.all(np.isfinite(R)) or np.any(R < 0.0):
        raise ValueError("R_cyl_pc must be finite and non-negative")
    if not np.all(np.isfinite(z)) or not (np.nanmin(z) < 0.0 and np.nanmax(z) > 0.0):
        raise ValueError("z_pc must be finite and span both sides of the midplane")
    if not np.all(np.isfinite(vol)) or np.any(vol <= 0.0):
        raise ValueError("cell_volume_pc3 must be finite and positive")
    if not np.all(np.isfinite(lum)) or np.any(lum < 0.0):
        raise ValueError("cell_luminosity_Lsun must be finite and non-negative")
    if not np.all(np.isfinite(nu)) or np.any(nu < 0.0):
        raise ValueError("nu_Lsun_pc3 must be finite and non-negative")

    Lsum = float(np.sum(lum))
    if not np.isclose(Lsum, float(ltot), rtol=1e-8, atol=max(1e-8, 1e-10 * abs(float(ltot)))):
        raise ValueError(f"axisymmetric light grid luminosity sum {Lsum:.16g} does not match Ltot {float(ltot):.16g}")

    return True


def cmd_plot_bins(args):
    sb_path = Path(args.surface_brightness)
    star_path = Path(args.stars)
    out_path = Path(args.out)

    sb = pd.read_csv(sb_path)
    stars = pd.read_csv(star_path)

    required_sb = {"R_inner_pc", "R_outer_pc", "q_axis_ratio"}
    required_stars = {"x_pc", "y_pc", "r_pc", "has_vlos"}

    missing_sb = required_sb - set(sb.columns)
    missing_stars = required_stars - set(stars.columns)

    if missing_sb:
        raise KeyError(f"Surface brightness file missing columns: {sorted(missing_sb)}")
    if missing_stars:
        raise KeyError(f"Star file missing columns: {sorted(missing_stars)}")

    q = float(sb["q_axis_ratio"].dropna().iloc[0])

    has_vlos = stars["has_vlos"].map(as_bool).to_numpy()
    vstars = stars.loc[has_vlos].copy()

    if len(vstars) == 0:
        raise ValueError("No stars with has_vlos=True.")

    x = vstars["x_pc"].to_numpy(float)
    y = vstars["y_pc"].to_numpy(float)
    r = vstars["r_pc"].to_numpy(float)

    kin_edges, kin_bins = make_min_count_bins( r, min_per_bin=args.min_stars, drop_partial=args.drop_partial_bins)

    r_keep_max = kin_edges[-1]
    keep = r <= r_keep_max

    x_plot = x[keep]
    y_plot = y[keep]
    r_plot = r[keep]

    sb_edges = np.unique(np.r_[sb["R_inner_pc"].to_numpy(float), sb["R_outer_pc"].to_numpy(float)])
    sb_edges = sb_edges[np.isfinite(sb_edges)]

    if args.bins_out is not None:
        bins_out = Path(args.bins_out)
        bins_out.parent.mkdir(parents=True, exist_ok=True)
        kin_bins.to_csv(bins_out, index=False)
        print(f"Saved bins: {bins_out}")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x_plot, y_plot, s=12, alpha=0.75, label="vlos stars in full bins")

    for k, rr in enumerate(kin_edges):
        plot_ellipse( ax, rr, q, linewidth=1.8, linestyle="-", label=f"kinematic bins ({args.min_stars} stars)" if k == 0 else None)

    for k, rr in enumerate(sb_edges):
        plot_ellipse( ax, rr, q, linewidth=1.1, linestyle="--", label="surface-brightness bins" if k == 0 else None)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [pc]")
    ax.set_ylabel("y [pc]")
    ax.set_title(args.title or "Projected stars with kinematic and surface-brightness bins")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)

    print(f"Saved: {out_path}")
    print(f"N vlos stars total: {len(vstars)}")
    print(f"N vlos stars plotted: {len(r_plot)}")
    print(f"N kinematic bins: {len(kin_bins)}")
    print(f"N surface-brightness bins: {len(sb_edges) - 1}")


def cmd_rebin_sb(args):
    sb_path = Path(args.surface_brightness)
    bins_path = Path(args.target_bins)
    out_path = Path(args.out)

    sb = pd.read_csv(sb_path)
    bins = pd.read_csv(bins_path)

    required_sb = {"R_pc", "Sigma"}
    required_bins = {"R_inner_pc", "R_outer_pc"}

    missing_sb = required_sb - set(sb.columns)
    missing_bins = required_bins - set(bins.columns)

    if missing_sb:
        raise KeyError(f"surface brightness file missing columns: {sorted(missing_sb)}")
    if missing_bins:
        raise KeyError(f"target bin file missing columns: {sorted(missing_bins)}")

    R_inner = bins["R_inner_pc"].to_numpy(float)
    R_outer = bins["R_outer_pc"].to_numpy(float)

    if "R_mid_pc" in bins.columns:
        R_mid = bins["R_mid_pc"].to_numpy(float)
    else:
        R_mid = 0.5 * (R_inner + R_outer)

    if not np.all(np.isfinite(R_inner)) or not np.all(np.isfinite(R_outer)):
        raise ValueError("target bin edges contain non-finite values")
    if not np.all(R_outer > R_inner):
        raise ValueError("target bins require R_outer_pc > R_inner_pc for every row")

    R_old = sb["R_pc"].to_numpy(float)
    Sigma_old = sb["Sigma"].to_numpy(float)

    warn_if_extrapolating(R_old, R_mid, label="R_mid_pc")

    if args.method == "loglog":
        Sigma_new = interp_loglog(R_old, Sigma_old, R_mid)
    elif args.method == "linear":
        Sigma_new = interp_linear(R_old, Sigma_old, R_mid)
    else:
        raise ValueError("method must be 'loglog' or 'linear'")

    if "Sigma_err" in sb.columns:
        Sigma_err_new = interp_linear(R_old, sb["Sigma_err"].to_numpy(float), R_mid)
    else:
        Sigma_err_new = np.full_like(Sigma_new, np.nan)

    q = first_finite(sb["q_axis_ratio"], 1.0) if "q_axis_ratio" in sb.columns else 1.0
    ellipticity = 1.0 - q if np.isfinite(q) else np.nan

    area_pc2 = np.pi * q * (R_outer**2 - R_inner**2)
    light = Sigma_new * area_pc2

    good_light = np.isfinite(light) & (light >= 0.0)
    if not np.any(good_light):
        raise ValueError("rebinned light is zero or non-finite everywhere")

    light = np.where(good_light, light, 0.0)
    light_sum = light.sum()

    if not np.isfinite(light_sum) or light_sum <= 0.0:
        raise ValueError("rebinned light sum is not positive")

    light_frac = light / light_sum

    out = pd.DataFrame({
        "R_inner_pc": R_inner,
        "R_outer_pc": R_outer,
        "R_pc": R_mid,
        "Sigma": Sigma_new,
        "Sigma_err": Sigma_err_new,
        "light_raw": light,
        "light_frac": light_frac,
        "q_axis_ratio": q,
        "ellipticity": ellipticity,
        "area_pc2": area_pc2,
        "source_surface_brightness_csv": str(sb_path),
        "source_target_bins_csv": str(bins_path),
        "rebin_method": args.method,
    })

    for col in ["galaxy", "source", "preferred_profile", "radius_type", "pc_per_arcmin_assumed", "note"]:
        if col in sb.columns:
            out[col] = sb[col].iloc[0]

    if "bin_id" in bins.columns:
        out.insert(0, "bin_id", bins["bin_id"].to_numpy(int))

    if "N_vlos" in bins.columns:
        out["N_vlos"] = bins["N_vlos"].to_numpy(int)

    validate_surface_brightness_bins(out, bins)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"N target bins: {len(bins)}")
    print(f"N output bins: {len(out)}")
    print(f"light_frac sum: {out['light_frac'].sum():.16f}")
    print(f"q_axis_ratio: {q}")
    print(f"method: {args.method}")
    print("validation: passed")


def smooth_log_profile(R_pc, Sigma, n_grid=512):
    R_pc = np.asarray(R_pc, float)
    Sigma = np.asarray(Sigma, float)

    good = np.isfinite(R_pc) & np.isfinite(Sigma) & (R_pc > 0) & (Sigma > 0)
    R = R_pc[good]
    S = Sigma[good]

    if len(R) < 4:
        raise ValueError("Need at least four finite positive surface-brightness points.")

    order = np.argsort(R)
    R = R[order]
    S = S[order]

    rmin = max(R[0] * 0.5, 1e-3)
    rmax = R[-1] * 2.0

    Rg = np.geomspace(rmin, rmax, n_grid)
    Sg = interp_loglog(R, S, Rg)

    return Rg, Sg


def abel_deproject_spherical(R_grid_pc, Sigma_grid):
    R = np.asarray(R_grid_pc, float)
    S = np.asarray(Sigma_grid, float)

    if not np.all(np.diff(R) > 0):
        raise ValueError("R_grid_pc must be strictly increasing.")

    dSdR = np.gradient(S, R)
    nu = np.zeros_like(R)

    for i, r in enumerate(R):
        Rp = R[i:].copy()
        dS = dSdR[i:].copy()

        if len(Rp) < 2:
            nu[i] = nu[i - 1] if i > 0 else 0.0
            continue

        denom = np.sqrt(np.maximum(Rp**2 - r**2, 1e-30))
        integrand = -dS / denom
        val = np.trapezoid(integrand, Rp) / np.pi
        nu[i] = max(val, 0.0) if np.isfinite(val) else 0.0

    good = np.isfinite(nu) & (nu > 0)
    if np.count_nonzero(good) < 3:
        raise ValueError("Abel deprojection produced too few positive density points.")

    nu_clean = np.interp( np.log(R), np.log(R[good]), np.log(nu[good]), left=np.log(nu[good][0]), right=np.log(nu[good][-1]))

    return np.exp(nu_clean)


def cumulative_luminosity_from_nu(r_pc, nu):
    r = np.asarray(r_pc, float)
    n = np.asarray(nu, float)

    Lenc = np.zeros_like(r)

    for i in range(1, len(r)):
        dr = r[i] - r[i - 1]
        shell0 = 4.0 * np.pi * r[i - 1] ** 2 * n[i - 1]
        shell1 = 4.0 * np.pi * r[i] ** 2 * n[i]
        Lenc[i] = Lenc[i - 1] + 0.5 * dr * (shell0 + shell1)

    return Lenc


def shell_grid_from_surface_brightness(sb, ltot):
    using_shell_bins = {"R_inner_pc", "R_outer_pc", "light_frac"}.issubset(sb.columns)

    if using_shell_bins:
        R_inner = sb["R_inner_pc"].to_numpy(float)
        R_outer = sb["R_outer_pc"].to_numpy(float)

        if "R_pc" in sb.columns:
            Rg = sb["R_pc"].to_numpy(float)
        else:
            Rg = 0.5 * (R_inner + R_outer)

        light_frac = sb["light_frac"].to_numpy(float)

        good = (
            np.isfinite(R_inner)
            & np.isfinite(R_outer)
            & np.isfinite(Rg)
            & np.isfinite(light_frac)
            & (R_outer > R_inner)
            & (light_frac >= 0.0)
        )

        R_inner = R_inner[good]
        R_outer = R_outer[good]
        Rg = Rg[good]
        light_frac = light_frac[good]

        if len(Rg) == 0:
            raise ValueError("No valid shell bins found in surface-brightness file.")

        if not np.any(light_frac > 0.0):
            raise ValueError("light_frac is zero everywhere in surface-brightness file.")

        light_frac = light_frac / light_frac.sum()

        shell_volume = (4.0 * np.pi / 3.0) * (R_outer**3 - R_inner**3)
        L_shell = float(ltot) * light_frac
        nu = L_shell / shell_volume
        Lenc_frac = np.cumsum(light_frac)

        if "Sigma" in sb.columns:
            Sg = sb.loc[good, "Sigma"].to_numpy(float)
        else:
            Sg = np.full_like(Rg, np.nan)

        return R_inner, R_outer, Rg, Sg, nu, L_shell, light_frac, Lenc_frac, "spherical_enclosed_light_grid"

    return None


def abel_grid_from_surface_brightness(sb, ltot, radius_col="R_pc", sigma_col="Sigma", n_radial=256):
    for col in (radius_col, sigma_col):
        if col not in sb.columns:
            raise KeyError(f"surface-brightness CSV missing column: {col}")

    R_raw = sb[radius_col].to_numpy(float)
    Sigma_raw = sb[sigma_col].to_numpy(float)

    Rg, Sg = smooth_log_profile(R_raw, Sigma_raw, n_grid=n_radial)
    nu_raw = abel_deproject_spherical(Rg, Sg)

    Lenc_raw = cumulative_luminosity_from_nu(Rg, nu_raw)
    Ltot_raw = Lenc_raw[-1]

    if not np.isfinite(Ltot_raw) or Ltot_raw <= 0:
        raise ValueError("Cumulative luminosity from deprojected profile is not positive.")

    scale = float(ltot) / Ltot_raw
    nu = nu_raw * scale
    Lenc = Lenc_raw * scale
    Lenc_frac = Lenc / Lenc[-1]

    R_inner = np.r_[Rg[0], 0.5 * (Rg[1:] + Rg[:-1])]
    R_outer = np.r_[0.5 * (Rg[1:] + Rg[:-1]), Rg[-1]]

    L_shell = np.diff(np.r_[0.0, Lenc])
    light_frac = L_shell / np.sum(L_shell)

    return R_inner, R_outer, Rg, Sg, nu, L_shell, light_frac, Lenc_frac, "spherical_abel_enclosed_light_grid"


def cmd_build_light_grid(args):
    sb_path = Path(args.surface_brightness)
    out_path = Path(args.out)

    sb = pd.read_csv(sb_path)

    shell = shell_grid_from_surface_brightness(sb, args.ltot)

    if shell is None:
        shell = abel_grid_from_surface_brightness(sb, args.ltot, radius_col=args.radius_col, sigma_col=args.sigma_col, n_radial=args.n_radial)

    R_inner, R_outer, Rg, Sg, nu, L_shell, light_frac, Lenc_frac, grid_geometry = shell

    theta = np.linspace(0.0, 0.5 * np.pi, args.n_theta)

    rows = []
    for r, rin, rout, sig, nval, lraw, lfrac, lenc in zip(Rg, R_inner, R_outer, Sg, nu, L_shell, light_frac, Lenc_frac):
        for th in theta:
            rows.append({
                "r_pc": r,
                "R_inner_pc": rin,
                "R_outer_pc": rout,
                "theta_rad": th,
                "nu_Lsun_pc3": nval,
                "Sigma_Lsun_pc2": sig,
                "cell_luminosity_Lsun": lraw / len(theta),
                "light_frac": lfrac,
                "Lenc_frac": lenc,
                "geometry": grid_geometry,
                "force_model": "spherical_enclosed_light",
                "flattened_geometry": "metadata_only",
                "density_coordinate": "r_pc",
                "source_surface_brightness_csv": str(sb_path),
            })

    out = pd.DataFrame(rows)

    for col in ["galaxy", "source", "preferred_profile", "radius_type", "q_axis_ratio", "ellipticity"]:
        if col in sb.columns:
            out[col] = sb[col].iloc[0]

    out["Ltot_Lsun"] = float(args.ltot)
    out["n_radial"] = int(len(Rg))
    out["n_theta"] = int(args.n_theta)

    validate_light_grid(out, args.ltot)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"N rows: {len(out)}")
    print(f"N radial: {len(Rg)}")
    print(f"N theta: {args.n_theta}")
    print(f"geometry: {grid_geometry}")
    print(f"Ltot target [Lsun]: {args.ltot}")
    print(f"Lenc_frac min/max: {out['Lenc_frac'].min()} {out['Lenc_frac'].max()}")
    print(f"nu_Lsun_pc3 min/max: {out['nu_Lsun_pc3'].min()} {out['nu_Lsun_pc3'].max()}")
    print("validation: passed")


def cmd_build_axisymmetric_light_grid(args):
    sb_path = Path(args.surface_brightness)
    out_path = Path(args.out)

    sb = pd.read_csv(sb_path)

    shell = shell_grid_from_surface_brightness(sb, args.ltot)

    if shell is None:
        shell = abel_grid_from_surface_brightness( sb, args.ltot, radius_col=args.radius_col, sigma_col=args.sigma_col, n_radial=args.n_radial)

    R_inner, R_outer, Rg, Sg, nu_sph, L_shell, light_frac, Lenc_frac, source_geometry = shell

    if args.q_axis_ratio is not None:
        q = float(args.q_axis_ratio)
    elif "q_axis_ratio" in sb.columns:
        q = first_finite(sb["q_axis_ratio"], 1.0)
    else:
        q = 1.0

    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("q_axis_ratio must be positive and finite")

    theta_edges = np.linspace(0.0, np.pi, args.n_theta + 1)
    phi_full = 2.0 * np.pi

    rows = []

    for ir, (rin, rout, rmid, sig, Lsh, lfrac, lenc) in enumerate( zip(R_inner, R_outer, Rg, Sg, L_shell, light_frac, Lenc_frac)):
        if not (np.isfinite(rin) and np.isfinite(rout) and rout > rin and np.isfinite(Lsh) and Lsh >= 0.0):
            continue

        shell_volume_oblate = (4.0 * np.pi / 3.0) * q * (rout**3 - rin**3)

        for it in range(args.n_theta):
            th0 = theta_edges[it]
            th1 = theta_edges[it + 1]
            th = 0.5 * (th0 + th1)

            cos0 = np.cos(th0)
            cos1 = np.cos(th1)
            theta_fraction = abs(cos0 - cos1) / 2.0

            cell_volume = shell_volume_oblate * theta_fraction
            cell_luminosity = Lsh * theta_fraction

            m_pc = rmid
            R_cyl = m_pc * np.sin(th)
            z = q * m_pc * np.cos(th)
            r_spherical = np.sqrt(R_cyl * R_cyl + z * z)

            nu_cell = cell_luminosity / cell_volume if cell_volume > 0.0 else 0.0

            rows.append({
                "shell_id": ir,
                "theta_id": it,

                "R_cyl_pc": R_cyl,
                "z_pc": z,
                "r_pc": r_spherical,
                "m_pc": m_pc,
                "theta_rad": th,
                "theta_inner_rad": th0,
                "theta_outer_rad": th1,

                "R_inner_pc": rin,
                "R_outer_pc": rout,
                "Sigma_Lsun_pc2": sig,

                "q_axis_ratio": q,
                "nu_Lsun_pc3": nu_cell,
                "cell_volume_pc3": cell_volume,
                "cell_luminosity_Lsun": cell_luminosity,

                "shell_luminosity_Lsun": Lsh,
                "light_frac": lfrac,
                "Lenc_frac": lenc,

                "geometry": "axisymmetric_density_grid",
                "flattened_geometry": "oblate_homeoid",
                "density_coordinate": "m_pc",
                "coordinate_model": "R_cyl=m*sin(theta), z=q*m*cos(theta)",
                "source_grid_geometry": source_geometry,
                "source_surface_brightness_csv": str(sb_path),
            })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        raise ValueError("Axisymmetric light grid produced zero rows")

    Lsum = out["cell_luminosity_Lsun"].sum()

    if not np.isfinite(Lsum) or Lsum <= 0.0:
        raise ValueError("Axisymmetric light grid luminosity sum is not positive")

    out["cell_luminosity_Lsun"] *= float(args.ltot) / Lsum
    out["nu_Lsun_pc3"] = out["cell_luminosity_Lsun"] / out["cell_volume_pc3"]

    for col in ["galaxy", "source", "preferred_profile", "radius_type", "ellipticity"]:
        if col in sb.columns:
            out[col] = sb[col].iloc[0]

    out["Ltot_Lsun"] = float(args.ltot)
    out["n_radial"] = int(len(Rg))
    out["n_theta"] = int(args.n_theta)
    out["theta_range"] = "0_to_pi_full_meridional_plane"
    out["volume_model"] = "oblate_homeoid_shell_fraction"
    out["force_ready"] = True

    validate_axisymmetric_grid(out, args.ltot)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"N rows: {len(out)}")
    print(f"N radial: {len(Rg)}")
    print(f"N theta: {args.n_theta}")
    print("geometry: axisymmetric_density_grid")
    print(f"source geometry: {source_geometry}")
    print(f"q_axis_ratio: {q}")
    print(f"Ltot target [Lsun]: {args.ltot}")
    print(f"Lsum grid [Lsun]: {out['cell_luminosity_Lsun'].sum()}")
    print(f"nu_Lsun_pc3 min/max: {out['nu_Lsun_pc3'].min()} {out['nu_Lsun_pc3'].max()}")
    print(f"z_pc min/max: {out['z_pc'].min()} {out['z_pc'].max()}")
    print(f"R_cyl_pc min/max: {out['R_cyl_pc'].min()} {out['R_cyl_pc'].max()}")
    print("validation: passed")



# ============================================================
# KARL GDEN-LIKE PRODUCTS
# ============================================================

def _require_columns(df, required, label):
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"{label} missing columns: {sorted(missing)}")


def _halo_density_msun_pc3_from_args(R_cyl_pc, z_pc, args):
    R = np.asarray(R_cyl_pc, float)
    z = np.asarray(z_pc, float)
    q = max(abs(float(args.halo_q_axis_ratio)), 1e-8)

    if args.halo_type == "none":
        return np.zeros_like(R, dtype=float)

    m = np.sqrt(R * R + (z / q) ** 2)
    m = np.maximum(m, 1e-12)

    if args.halo_type == "nfw":
        rho_s = float(args.halo_rho_s_msun_pc3)
        rs = max(float(args.halo_rs_pc), 1e-12)
        x = m / rs
        return rho_s / (x * (1.0 + x) ** 2 + 1e-30)

    if args.halo_type == "karl_nfw":
        c = max(float(args.cnfw), 1e-12)
        rs = max(float(args.rsnfw_pc), 1e-12)
        hparam = float(args.hubble_km_s_mpc) / 100.0
        rhocrit = 2.7754996776e-7 * hparam**2
        xd = 200.0 / 3.0 * c**3 / (np.log(1.0 + c) - c / (1.0 + c))
        x = m / rs
        return rhocrit * xd / (x * (1.0 + x) ** 2 + 1e-30)

    if args.halo_type == "karl_gamma":
        gamma = float(args.gamma)
        mscale = max(float(args.rsgamma_pc), 1e-12)
        xmgamma = float(args.xmgamma_msun)

        if gamma != 0.0:
            return (
                xmgamma
                / (4.0 * np.pi)
                * (3.0 - gamma)
                * mscale
                / (m**gamma * (mscale + m) ** (4.0 - gamma) + 1e-30)
            )

        return (
            3.0
            * xmgamma
            / (4.0 * np.pi * mscale**3)
            * (1.0 + (m / mscale) ** 2) ** (-2.5)
        )

    if args.halo_type == "karl_isothermal":
        qdm = q
        rc = max(float(args.rc_pc), 1e-12)
        v0 = float(args.v0_km_s)
        dis = max(float(args.dis), 1e-12)

        xR = R
        xZ = z

        rho = 0.78722918 / (dis * dis)
        rho *= v0 * v0 / (qdm * qdm)

        num = (
            (2.0 * qdm * qdm + 1.0) * rc * rc
            + xR * xR
            + 2.0 * (1.0 - 0.5 / (qdm * qdm)) * xZ * xZ
        )
        den = (rc * rc + xR * xR + xZ * xZ / (qdm * qdm)) ** 2

        return rho * num / np.maximum(den, 1e-30)

    raise ValueError(f"Unknown halo_type: {args.halo_type}")


def validate_gden_products(out, args):
    required = {
        "shell_id", "theta_id", "R_cyl_pc", "z_pc", "r_pc", "m_pc",
        "theta_rad", "cell_volume_pc3", "dL_Lsun", "dM_star_Msun",
        "dMhalo_Msun", "ratML_local", "rho_halo_Msun_pc3",
        "stellar_ML", "halo_type", "geometry", "product_type",
    }
    _require_columns(out, required, "gden-like products")

    vol = out["cell_volume_pc3"].to_numpy(float)
    dL = out["dL_Lsun"].to_numpy(float)
    dM = out["dM_star_Msun"].to_numpy(float)
    dMh = out["dMhalo_Msun"].to_numpy(float)

    if not np.all(np.isfinite(vol)) or np.any(vol <= 0.0):
        raise ValueError("gden products require positive finite cell_volume_pc3")
    if not np.all(np.isfinite(dL)) or np.any(dL < 0.0):
        raise ValueError("gden products require non-negative finite dL_Lsun")
    if not np.all(np.isfinite(dM)) or np.any(dM < 0.0):
        raise ValueError("gden products require non-negative finite dM_star_Msun")
    if not np.all(np.isfinite(dMh)) or np.any(dMh < 0.0):
        raise ValueError("gden products require non-negative finite dMhalo_Msun")

    lsum = float(dL.sum())
    if not np.isclose(lsum, float(args.ltot), rtol=1e-8, atol=max(1e-8, 1e-10 * abs(float(args.ltot)))):
        raise ValueError(f"gden dL sum {lsum:.16g} does not match Ltot {float(args.ltot):.16g}")

    ml = float(args.ml)
    msum = float(dM.sum())
    expected_mstar = ml * float(args.ltot)

    if not np.isclose(msum, expected_mstar, rtol=1e-8, atol=max(1e-8, 1e-10 * abs(expected_mstar))):
        raise ValueError(f"gden stellar mass sum {msum:.16g} does not match ML*Ltot {expected_mstar:.16g}")

    return True


def cmd_build_gden_products(args):
    grid_path = Path(args.light_grid)
    out_path = Path(args.out)

    grid = pd.read_csv(grid_path)

    required = {
        "R_cyl_pc", "z_pc", "r_pc", "m_pc", "theta_rad",
        "theta_inner_rad", "theta_outer_rad",
        "cell_volume_pc3", "cell_luminosity_Lsun",
        "q_axis_ratio", "geometry",
    }
    _require_columns(grid, required, "axisymmetric light grid")

    if not (grid["geometry"].astype(str) == "axisymmetric_density_grid").all():
        raise ValueError("build-gden-products requires an axisymmetric_density_grid light grid")

    out = grid.copy()

    dL = out["cell_luminosity_Lsun"].to_numpy(float)
    vol = out["cell_volume_pc3"].to_numpy(float)
    R = out["R_cyl_pc"].to_numpy(float)
    z = out["z_pc"].to_numpy(float)

    Lsum = float(np.nansum(dL))
    if not np.isfinite(Lsum) or Lsum <= 0.0:
        raise ValueError("light grid luminosity sum is not positive")

    dL = np.where(np.isfinite(dL) & (dL >= 0.0), dL, 0.0)
    dL *= float(args.ltot) / dL.sum()

    dM_star = float(args.ml) * dL

    rho_halo = _halo_density_msun_pc3_from_args(R, z, args)
    rho_halo = np.where(np.isfinite(rho_halo) & (rho_halo >= 0.0), rho_halo, 0.0)
    dMhalo = rho_halo * vol

    ratML = np.full_like(dL, np.nan, dtype=float)
    good_light = dL > 0.0
    ratML[good_light] = (dM_star[good_light] + dMhalo[good_light]) / dL[good_light]

    out["dL_Lsun"] = dL
    out["dM_star_Msun"] = dM_star
    out["rho_halo_Msun_pc3"] = rho_halo
    out["dMhalo_Msun"] = dMhalo
    out["ratML_local"] = ratML
    out["stellar_ML"] = float(args.ml)
    out["halo_type"] = args.halo_type
    out["halo_q_axis_ratio"] = float(args.halo_q_axis_ratio)
    out["product_type"] = "karl_gden_like"
    out["source_light_grid_csv"] = str(grid_path)
    out["Ltot_Lsun"] = float(args.ltot)
    out["total_dL_Lsun"] = float(dL.sum())
    out["total_dM_star_Msun"] = float(dM_star.sum())
    out["total_dMhalo_Msun"] = float(dMhalo.sum())
    out["gden_norm_light"] = float(args.ltot) / Lsum
    out["gden_note"] = (
        "Python gden-like product: dL, dM_star, dMhalo, and local ratML "
        "computed from axisymmetric light-grid cells before Julia orbit evaluation."
    )

    validate_gden_products(out, args)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if args.norm_out is not None:
        norm_path = Path(args.norm_out)
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        norm = pd.DataFrame([{
            "source_light_grid_csv": str(grid_path),
            "gden_products_csv": str(out_path),
            "Ltot_Lsun": float(args.ltot),
            "stellar_ML": float(args.ml),
            "halo_type": args.halo_type,
            "halo_q_axis_ratio": float(args.halo_q_axis_ratio),
            "total_dL_Lsun": float(dL.sum()),
            "total_dM_star_Msun": float(dM_star.sum()),
            "total_dMhalo_Msun": float(dMhalo.sum()),
            "gden_norm_light": float(args.ltot) / Lsum,
        }])
        norm.to_csv(norm_path, index=False)
        print(f"Saved norm: {norm_path}")

    print(f"Saved: {out_path}")
    print(f"N rows: {len(out)}")
    print(f"halo_type: {args.halo_type}")
    print(f"Ltot target [Lsun]: {args.ltot}")
    print(f"stellar M/L: {args.ml}")
    print(f"total dL [Lsun]: {out['dL_Lsun'].sum():.12g}")
    print(f"total dM_star [Msun]: {out['dM_star_Msun'].sum():.12g}")
    print(f"total dMhalo [Msun]: {out['dMhalo_Msun'].sum():.12g}")
    print("validation: passed")


def build_parser():
    p = argparse.ArgumentParser(description="OSPM observable mapping utilities.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plot = sub.add_parser("plot-bins", help="Plot projected stars, kinematic bins, and surface-brightness bins.")
    p_plot.add_argument("--surface-brightness", required=True)
    p_plot.add_argument("--stars", required=True)
    p_plot.add_argument("--out", default="observable_bins.png")
    p_plot.add_argument("--min-stars", type=int, default=20)
    p_plot.add_argument("--drop-partial-bins", action="store_true")
    p_plot.add_argument("--bins-out", default=None)
    p_plot.add_argument("--title", default=None)
    p_plot.set_defaults(func=cmd_plot_bins)

    p_rebin = sub.add_parser("rebin-sb", help="Rebin a surface-brightness profile onto target radial bins.")
    p_rebin.add_argument("--surface-brightness", required=True)
    p_rebin.add_argument("--target-bins", required=True)
    p_rebin.add_argument("--out", required=True)
    p_rebin.add_argument("--method", default="loglog", choices=["loglog", "linear"])
    p_rebin.set_defaults(func=cmd_rebin_sb)

    p_grid = sub.add_parser("build-light-grid", help="Build a spherical Karl-style luminosity-density grid.")
    p_grid.add_argument("--surface-brightness", required=True)
    p_grid.add_argument("--out", required=True)
    p_grid.add_argument("--radius-col", default="R_pc")
    p_grid.add_argument("--sigma-col", default="Sigma")
    p_grid.add_argument("--ltot", type=float, default=2.7e5)
    p_grid.add_argument("--n-radial", type=int, default=256)
    p_grid.add_argument("--n-theta", type=int, default=32)
    p_grid.set_defaults(func=cmd_build_light_grid)

    p_axis = sub.add_parser("build-axisymmetric-light-grid", help="Build a force-ready axisymmetric stellar density grid.")
    p_axis.add_argument("--surface-brightness", required=True)
    p_axis.add_argument("--out", required=True)
    p_axis.add_argument("--radius-col", default="R_pc")
    p_axis.add_argument("--sigma-col", default="Sigma")
    p_axis.add_argument("--ltot", type=float, default=2.7e5)
    p_axis.add_argument("--n-radial", type=int, default=256)
    p_axis.add_argument("--n-theta", type=int, default=64)
    p_axis.add_argument("--q-axis-ratio", type=float, default=None)
    p_axis.set_defaults(func=cmd_build_axisymmetric_light_grid)


    p_gden = sub.add_parser("build-gden-products", help="Build Karl gden-like mass/light products from an axisymmetric light grid.")
    p_gden.add_argument("--light-grid", required=True)
    p_gden.add_argument("--out", required=True)
    p_gden.add_argument("--norm-out", default=None)
    p_gden.add_argument("--ltot", type=float, default=2.7e5)
    p_gden.add_argument("--ml", type=float, default=1.0)
    p_gden.add_argument("--halo-type", default="none", choices=["none", "nfw", "karl_nfw", "karl_gamma", "karl_isothermal"])
    p_gden.add_argument("--halo-q-axis-ratio", type=float, default=1.0)

    # Modern NFW-like density: rho_s / (x(1+x)^2)
    p_gden.add_argument("--halo-rho-s-msun-pc3", type=float, default=0.0)
    p_gden.add_argument("--halo-rs-pc", type=float, default=1000.0)

    # Karl ihalo=2 NFW concentration branch
    p_gden.add_argument("--cnfw", type=float, default=1.0)
    p_gden.add_argument("--rsnfw-pc", type=float, default=1000.0)
    p_gden.add_argument("--hubble-km-s-mpc", type=float, default=70.0)

    # Karl ihalo=1 gamma / Plummer branch
    p_gden.add_argument("--gamma", type=float, default=1.0)
    p_gden.add_argument("--xmgamma-msun", type=float, default=0.0)
    p_gden.add_argument("--rsgamma-pc", type=float, default=1000.0)

    # Karl ihalo=3 non-singular isothermal spheroid branch
    p_gden.add_argument("--v0-km-s", type=float, default=0.0)
    p_gden.add_argument("--rc-pc", type=float, default=1000.0)
    p_gden.add_argument("--dis", type=float, default=1.0)

    p_gden.set_defaults(func=cmd_build_gden_products)


    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()