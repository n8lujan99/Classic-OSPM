#!/usr/bin/env python3
"""
ROLE: OSPM Data Prep / pre-staging.
This script converts adopted galaxy observations into static OSPM input
products before any theta-dependent model evaluation begins.
It is responsible for preparing products such as:
    - kinematic radial bins
    - rebinned surface-brightness profiles
    - stellar luminosity / force grids
    - tracer-density grids
These products should change only when the adopted galaxy data, geometry, or
preprocessing choices change.
This script is NOT part of the per-theta OSPM solve. Halo parameters, MBH,
stellar M/L, orbit integration, orbit weights, and chi2 belong to the runtime
physics/model layer.
The active galaxy-specific filenames and parameters should come from the
galaxy setup/config rather than being hard-coded here.
"""
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
    groups = [r[i:i + min_per_bin] for i in range(0, len(r), min_per_bin)]
    if len(groups) > 1 and len(groups[-1]) < min_per_bin:
        if drop_partial:
            groups.pop()
        else:
            groups[-2] = np.concatenate((groups[-2], groups[-1]))
            groups.pop()
    if not groups:
        raise ValueError("No complete kinematic bins.")
    edges = np.empty(len(groups) + 1, dtype=float)
    edges[0] = 0.0
    for i in range(1, len(groups)):
        edges[i] = 0.5 * (groups[i - 1][-1] + groups[i][0])
    edges[-1] = np.nextafter(groups[-1][-1], np.inf)
    rows = []
    for i, group in enumerate(groups):
        rows.append({
            "bin_id": i,
            "R_inner_pc": edges[i],
            "R_outer_pc": edges[i + 1],
            "R_mid_pc": 0.5 * (edges[i] + edges[i + 1]),
            "N_vlos": len(group),
        })

    return edges, pd.DataFrame(rows)

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
        print(f"WARNING: {below + above} target {label} values fall outside the source surface-brightness range " f"[{lo:.6g}, {hi:.6g}]. Edge extrapolation will be used.")

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

def validate_axisymmetric_grid(out, ltot):
    required = {
        "shell_id", "theta_id",
        "R_cyl_pc", "z_pc", "r_pc", "m_pc",
        "theta_rad", "theta_inner_rad", "theta_outer_rad",
        "R_inner_pc", "R_outer_pc",
        "q_axis_ratio", "nu_pc3", "nu_Lsun_pc3",
        "cell_volume_pc3", "cell_luminosity_Lsun",
        "shell_luminosity_Lsun", "light_frac", "Lenc_frac",
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
    nu = out["nu_pc3"].to_numpy(float)
    nu_lsun = out["nu_Lsun_pc3"].to_numpy(float)
    vol = out["cell_volume_pc3"].to_numpy(float)
    lum = out["cell_luminosity_Lsun"].to_numpy(float)
    if not np.all(np.isfinite(q)) or np.any(q <= 0.0):
        raise ValueError("q_axis_ratio must be positive and finite")
    if not np.all(np.isfinite(R)) or np.any(R < 0.0):
        raise ValueError("R_cyl_pc must be finite and non-negative")
    if not np.all(np.isfinite(z)) or not (np.nanmin(z) < 0.0 and np.nanmax(z) > 0.0):
        raise ValueError("z_pc must be finite and span both sides of the midplane")
    if not np.all(np.isfinite(nu)) or np.any(nu <= 0.0):
        raise ValueError("nu_pc3 must be positive and finite")
    if not np.all(np.isfinite(nu_lsun)) or np.any(nu_lsun <= 0.0):
        raise ValueError("nu_Lsun_pc3 must be positive and finite")
    if not np.all(np.isfinite(vol)) or np.any(vol <= 0.0):
        raise ValueError("cell_volume_pc3 must be positive and finite")
    if not np.all(np.isfinite(lum)) or np.any(lum < 0.0):
        raise ValueError("cell_luminosity_Lsun must be finite and non-negative")
    Lsum = float(np.sum(lum))
    if not np.isclose(Lsum, float(ltot), rtol=1e-8, atol=max(1e-8, 1e-10 * abs(float(ltot)))):
        raise ValueError(f"axisymmetric light grid luminosity sum {Lsum:.16g} does not match Ltot {float(ltot):.16g}")
    shell = out.sort_values(["shell_id", "theta_id"]).groupby("shell_id", sort=True).first()
    if not np.isclose(shell["light_frac"].sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("shell light_frac must sum to 1")
    if not np.isclose(shell["Lenc_frac"].iloc[-1], 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("final Lenc_frac must equal 1")
    if np.any(np.diff(shell["Lenc_frac"].to_numpy(float)) < -1e-12):
        raise ValueError("Lenc_frac must be monotonic non-decreasing")
    shell_from_cells = out.groupby("shell_id", sort=True)["cell_luminosity_Lsun"].sum().to_numpy(float)
    shell_expected = shell["shell_luminosity_Lsun"].to_numpy(float)
    if not np.allclose(shell_from_cells, shell_expected, rtol=1e-12, atol=1e-12):
        raise ValueError("cell luminosities do not sum to their shell luminosities")
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

##################################################################################################################################################
##################################################################################################################################################

def cmd_build_axisymmetric_light_grid(args):
    sb_path = Path(args.surface_brightness)
    density_path = Path(args.density_profile)
    out_path = Path(args.out)
    sb = pd.read_csv(sb_path)
    density = pd.read_csv(density_path)

    required_density = {"row_type", "r_pc", "nu_pc3", "abel_support_min_pc", "unresolved_center_raw_light"}
    missing_density = required_density - set(density.columns)
    if missing_density:
        raise KeyError(f"prepared density profile missing columns: {sorted(missing_density)}")

    unresolved_rows = density.loc[density["row_type"].astype(str) == "unresolved_center"].copy()
    resolved = density.loc[density["row_type"].astype(str) == "resolved_abel"].copy()

    if len(unresolved_rows) != 1:
        raise ValueError(f"prepared density profile requires exactly one unresolved_center row; found {len(unresolved_rows)}")
    if len(resolved) < 2:
        raise ValueError("prepared density profile requires at least two resolved_abel rows")

    unresolved = unresolved_rows.iloc[0]
    support_min_pc = float(unresolved["abel_support_min_pc"])
    unresolved_raw_light = float(unresolved["unresolved_center_raw_light"])

    if not np.isfinite(support_min_pc) or support_min_pc <= 0.0:
        raise ValueError("abel_support_min_pc must be positive and finite")
    if not np.isfinite(unresolved_raw_light) or unresolved_raw_light <= 0.0:
        raise ValueError("unresolved_center_raw_light must be positive and finite")

    m_grid = resolved["r_pc"].to_numpy(float)
    nu_grid = resolved["nu_pc3"].to_numpy(float)

    if not np.all(np.isfinite(m_grid)) or np.any(m_grid <= 0.0):
        raise ValueError("resolved Abel density radii must be positive and finite")
    if not np.all(np.diff(m_grid) > 0.0):
        raise ValueError("resolved Abel density radii must be strictly increasing")
    if not np.all(np.isfinite(nu_grid)) or np.any(nu_grid <= 0.0):
        raise ValueError("resolved Abel density values must be positive and finite")
    if not np.isclose(m_grid[0], support_min_pc, rtol=1e-10, atol=1e-10):
        raise ValueError(f"first resolved Abel radius {m_grid[0]:.16g} does not match abel_support_min_pc {support_min_pc:.16g}")
    if args.n_radial is not None and int(args.n_radial) != len(m_grid):
        raise ValueError(f"prepared density has {len(m_grid)} resolved Abel radial points but --n-radial={args.n_radial}")

    if args.q_axis_ratio is not None:
        q = float(args.q_axis_ratio)
    elif "q_axis_ratio" in sb.columns:
        q = first_finite(sb["q_axis_ratio"], 1.0)
    else:
        q = 1.0
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("q_axis_ratio must be positive and finite")

    resolved_edges = np.empty(len(m_grid) + 1, dtype=float)
    resolved_edges[0] = support_min_pc
    resolved_edges[-1] = m_grid[-1]
    resolved_edges[1:-1] = 0.5 * (m_grid[:-1] + m_grid[1:])

    resolved_R_inner = resolved_edges[:-1]
    resolved_R_outer = resolved_edges[1:]
    resolved_spherical_volume = (4.0 * np.pi / 3.0) * (resolved_R_outer**3 - resolved_R_inner**3)

    if not np.all(np.isfinite(resolved_spherical_volume)) or np.any(resolved_spherical_volume <= 0.0):
        raise ValueError("resolved Abel density produced invalid radial shell volumes")

    resolved_raw_shell_light = q * nu_grid * resolved_spherical_volume
    central_spherical_volume = (4.0 * np.pi / 3.0) * support_min_pc**3
    central_nu_pc3 = unresolved_raw_light / (q * central_spherical_volume)

    if not np.isfinite(central_nu_pc3) or central_nu_pc3 <= 0.0:
        raise ValueError("unresolved central light produced a non-positive bookkeeping density")

    raw_shell_light = np.r_[unresolved_raw_light, resolved_raw_shell_light]
    raw_total = float(np.sum(raw_shell_light))
    if not np.isfinite(raw_total) or raw_total <= 0.0:
        raise ValueError("prepared density plus unresolved center integrates to non-positive light")

    luminosity_scale = float(args.ltot) / raw_total
    L_shell = raw_shell_light * luminosity_scale
    light_frac = L_shell / np.sum(L_shell)
    Lenc_frac = np.cumsum(light_frac)

    shell_m_pc = np.r_[0.5 * support_min_pc, m_grid]
    shell_nu_pc3 = np.r_[central_nu_pc3, nu_grid]
    shell_R_inner = np.r_[0.0, resolved_R_inner]
    shell_R_outer = np.r_[support_min_pc, resolved_R_outer]
    shell_spherical_volume = np.r_[central_spherical_volume, resolved_spherical_volume]
    shell_source = np.r_[np.array(["unresolved_center"], dtype=object), np.full(len(m_grid), "resolved_abel", dtype=object)]

    theta_edges = np.linspace(0.0, np.pi, args.n_theta + 1)
    rows = []

    for ir, (m_pc, nu_pc3) in enumerate(zip(shell_m_pc, shell_nu_pc3)):
        shell_volume_oblate = q * shell_spherical_volume[ir]

        for it in range(args.n_theta):
            th0 = theta_edges[it]
            th1 = theta_edges[it + 1]
            th = 0.5 * (th0 + th1)
            theta_fraction = abs(np.cos(th0) - np.cos(th1)) / 2.0
            cell_volume = shell_volume_oblate * theta_fraction
            cell_luminosity = L_shell[ir] * theta_fraction

            R_cyl = m_pc * np.sin(th)
            z = q * m_pc * np.cos(th)
            r_spherical = np.sqrt(R_cyl * R_cyl + z * z)
            nu_lsun_pc3 = cell_luminosity / cell_volume

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
                "R_inner_pc": shell_R_inner[ir],
                "R_outer_pc": shell_R_outer[ir],
                "q_axis_ratio": q,
                "nu_pc3": nu_pc3,
                "nu_Lsun_pc3": nu_lsun_pc3,
                "cell_volume_pc3": cell_volume,
                "cell_luminosity_Lsun": cell_luminosity,
                "shell_luminosity_Lsun": L_shell[ir],
                "light_frac": light_frac[ir],
                "Lenc_frac": Lenc_frac[ir],
                "density_source": shell_source[ir],
                "geometry": "axisymmetric_density_grid",
                "flattened_geometry": "oblate_homeoid",
                "density_coordinate": "m_pc",
                "coordinate_model": "R_cyl=m*sin(theta), z=q*m*cos(theta)",
                "source_grid_geometry": "prepared_radial_density_grid",
                "source_surface_brightness_csv": str(sb_path),
                "source_density_profile_csv": str(density_path),
            })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("Axisymmetric light grid produced zero rows")
    for col in ["galaxy", "source", "preferred_profile", "radius_type", "ellipticity"]:
        if col in sb.columns:
            out[col] = sb[col].iloc[0]

    out["Ltot_Lsun"] = float(args.ltot)
    out["n_radial"] = int(len(shell_m_pc))
    out["n_resolved_abel"] = int(len(m_grid))
    out["n_theta"] = int(args.n_theta)
    out["abel_support_min_pc"] = support_min_pc
    out["unresolved_center_raw_light"] = unresolved_raw_light
    out["unresolved_center_Lsun"] = float(L_shell[0])
    out["theta_range"] = "0_to_pi_full_meridional_plane"
    out["volume_model"] = "oblate_homeoid_shell_fraction"

    validate_axisymmetric_grid(out, args.ltot)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"N rows: {len(out)}")
    print(f"N radial shells: {len(shell_m_pc)}")
    print(f"N resolved Abel radial points: {len(m_grid)}")
    print(f"N theta: {args.n_theta}")
    print("geometry: axisymmetric_density_grid")
    print("source geometry: prepared_radial_density_grid")
    print(f"q_axis_ratio: {q}")
    print(f"Abel support minimum [pc]: {support_min_pc}")
    print(f"unresolved center raw light: {unresolved_raw_light}")
    print(f"unresolved center mapped light [Lsun]: {L_shell[0]}")
    print(f"Ltot target [Lsun]: {args.ltot}")
    print(f"Lsum grid [Lsun]: {out['cell_luminosity_Lsun'].sum()}")
    print(f"nu_pc3 min/max: {out['nu_pc3'].min()} {out['nu_pc3'].max()}")
    print(f"nu_Lsun_pc3 min/max: {out['nu_Lsun_pc3'].min()} {out['nu_Lsun_pc3'].max()}")
    print(f"z_pc min/max: {out['z_pc'].min()} {out['z_pc'].max()}")
    print(f"R_cyl_pc min/max: {out['R_cyl_pc'].min()} {out['R_cyl_pc'].max()}")
    print("validation: passed")


def cmd_build_stellar_force_grid(args):
    sb_path = Path(args.surface_brightness)
    density_path = Path(args.density_profile)
    out_path = Path(args.out)
    sb = pd.read_csv(sb_path)
    density = pd.read_csv(density_path)

    required_density = {
        "row_type", "r_pc", "nu_pc3",
        "abel_support_min_pc", "unresolved_center_raw_light",
        "inner_continuation_model", "inner_cslope",
        "inner_boundary_nu_pc3", "inner_continuation_raw_light",
        "inner_continuation_light_residual",
    }
    missing_density = required_density - set(density.columns)
    if missing_density:
        raise KeyError(f"prepared density profile missing columns: {sorted(missing_density)}")

    unresolved_rows = density.loc[density["row_type"].astype(str) == "unresolved_center"].copy()
    resolved = density.loc[density["row_type"].astype(str) == "resolved_abel"].copy()

    if len(unresolved_rows) != 1:
        raise ValueError(f"prepared density profile requires exactly one unresolved_center row; found {len(unresolved_rows)}")
    if len(resolved) < 2:
        raise ValueError("prepared density profile requires at least two resolved_abel rows")

    unresolved = unresolved_rows.iloc[0]
    support_min_pc = float(unresolved["abel_support_min_pc"])
    unresolved_raw_light = float(unresolved["unresolved_center_raw_light"])
    inner_model = str(unresolved["inner_continuation_model"]).strip().lower()
    inner_cslope = float(unresolved["inner_cslope"])
    inner_boundary_nu_pc3 = float(unresolved["inner_boundary_nu_pc3"])
    inner_continuation_raw_light = float(unresolved["inner_continuation_raw_light"])
    inner_continuation_light_residual = float(unresolved["inner_continuation_light_residual"])

    if not np.isfinite(support_min_pc) or support_min_pc <= 0.0:
        raise ValueError("abel_support_min_pc must be positive and finite")
    if not np.isfinite(unresolved_raw_light) or unresolved_raw_light <= 0.0:
        raise ValueError("unresolved_center_raw_light must be positive and finite")
    if inner_model != "karl_powerlaw":
        raise ValueError(f"stellar force grid requires inner_continuation_model=karl_powerlaw; got {inner_model}")
    if not np.isfinite(inner_cslope) or inner_cslope <= -3.0:
        raise ValueError("inner_cslope must be finite and greater than -3")
    if not np.isfinite(inner_boundary_nu_pc3) or inner_boundary_nu_pc3 <= 0.0:
        raise ValueError("inner_boundary_nu_pc3 must be positive and finite")
    if not np.isfinite(inner_continuation_raw_light) or inner_continuation_raw_light <= 0.0:
        raise ValueError("inner_continuation_raw_light must be positive and finite")
    if not np.isfinite(inner_continuation_light_residual):
        raise ValueError("inner_continuation_light_residual must be finite")
    if not np.isclose(inner_continuation_raw_light, unresolved_raw_light, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "Karl inner continuation light does not match unresolved central light: "
            f"{inner_continuation_raw_light:.16g} vs {unresolved_raw_light:.16g}"
        )
    if not np.isclose(inner_continuation_light_residual, 0.0, rtol=0.0, atol=1e-10 * unresolved_raw_light):
        raise ValueError(
            "Karl inner continuation light residual is not numerically zero: "
            f"{inner_continuation_light_residual:.16g}"
        )

    m_grid = resolved["r_pc"].to_numpy(float)
    nu_grid = resolved["nu_pc3"].to_numpy(float)

    if not np.all(np.isfinite(m_grid)) or np.any(m_grid <= 0.0):
        raise ValueError("resolved Abel density radii must be positive and finite")
    if not np.all(np.diff(m_grid) > 0.0):
        raise ValueError("resolved Abel density radii must be strictly increasing")
    if not np.all(np.isfinite(nu_grid)) or np.any(nu_grid <= 0.0):
        raise ValueError("resolved Abel density values must be positive and finite")
    if not np.isclose(m_grid[0], support_min_pc, rtol=1e-10, atol=1e-10):
        raise ValueError(f"first resolved Abel radius {m_grid[0]:.16g} does not match abel_support_min_pc {support_min_pc:.16g}")
    if args.n_radial is not None and int(args.n_radial) != len(m_grid):
        raise ValueError(f"prepared density has {len(m_grid)} resolved Abel radial points but --n-radial={args.n_radial}")

    if args.q_axis_ratio is not None:
        q = float(args.q_axis_ratio)
    elif "q_axis_ratio" in sb.columns:
        q = first_finite(sb["q_axis_ratio"], 1.0)
    else:
        q = 1.0
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("q_axis_ratio must be positive and finite")

    central_n_shells = int(args.central_shells)
    if central_n_shells < 1:
        raise ValueError("--central-shells must be at least 1")

    if not np.isclose(nu_grid[0], inner_boundary_nu_pc3, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "first resolved Abel density does not match inner_boundary_nu_pc3: "
            f"{nu_grid[0]:.16g} vs {inner_boundary_nu_pc3:.16g}"
        )

    resolved_edges = np.empty(len(m_grid) + 1, dtype=float)
    resolved_edges[0] = support_min_pc
    resolved_edges[-1] = m_grid[-1]
    resolved_edges[1:-1] = 0.5 * (m_grid[:-1] + m_grid[1:])
    resolved_R_inner = resolved_edges[:-1]
    resolved_R_outer = resolved_edges[1:]
    resolved_spherical_volume = (4.0 * np.pi / 3.0) * (resolved_R_outer**3 - resolved_R_inner**3)
    if not np.all(np.isfinite(resolved_spherical_volume)) or np.any(resolved_spherical_volume <= 0.0):
        raise ValueError("resolved Abel density produced invalid radial shell volumes")
    resolved_raw_shell_light = q * nu_grid * resolved_spherical_volume

    central_edges = np.linspace(0.0, support_min_pc, central_n_shells + 1)
    central_R_inner = central_edges[:-1]
    central_R_outer = central_edges[1:]
    central_spherical_volume = (4.0 * np.pi / 3.0) * (central_R_outer**3 - central_R_inner**3)
    central_oblate_volume = q * central_spherical_volume

    exponent = 3.0 + inner_cslope
    central_integrals = central_R_outer**exponent - central_R_inner**exponent
    if not np.all(np.isfinite(central_integrals)) or np.any(central_integrals <= 0.0):
        raise ValueError("Karl unresolved-center continuation produced invalid radial integrals")

    central_prefactor = 4.0 * np.pi * q * inner_boundary_nu_pc3 * support_min_pc**(-inner_cslope) / exponent
    central_raw_shell_light = central_prefactor * central_integrals

    central_light_sum = float(np.sum(central_raw_shell_light))
    if not np.isclose(central_light_sum, unresolved_raw_light, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "mapped Karl inner continuation does not reproduce unresolved central light: "
            f"{central_light_sum:.16g} vs {unresolved_raw_light:.16g}"
        )

    central_nu_pc3 = central_raw_shell_light / central_oblate_volume

    mean_num_exp = 4.0 + inner_cslope
    central_m_pc = (
        exponent / mean_num_exp
        * (central_R_outer**mean_num_exp - central_R_inner**mean_num_exp)
        / central_integrals
    )

    if not np.all(np.isfinite(central_nu_pc3)) or np.any(central_nu_pc3 <= 0.0):
        raise ValueError("Karl unresolved-center continuation produced non-positive shell-average densities")
    if not np.all(np.isfinite(central_m_pc)) or np.any(central_m_pc <= 0.0):
        raise ValueError("Karl unresolved-center continuation produced invalid representative radii")
    raw_shell_light = np.r_[central_raw_shell_light, resolved_raw_shell_light]
    raw_total = float(np.sum(raw_shell_light))
    if not np.isfinite(raw_total) or raw_total <= 0.0:
        raise ValueError("prepared density plus unresolved center integrates to non-positive light")
    luminosity_scale = float(args.ltot) / raw_total
    L_shell = raw_shell_light * luminosity_scale
    light_frac = L_shell / np.sum(L_shell)
    Lenc_frac = np.cumsum(light_frac)
    shell_m_pc = np.r_[central_m_pc, m_grid]
    shell_nu_pc3 = np.r_[central_nu_pc3, nu_grid]
    shell_R_inner = np.r_[central_R_inner, resolved_R_inner]
    shell_R_outer = np.r_[central_R_outer, resolved_R_outer]
    shell_spherical_volume = np.r_[central_spherical_volume, resolved_spherical_volume]
    shell_source = np.r_[np.full(central_n_shells, "unresolved_center_force_model", dtype=object), np.full(len(m_grid), "resolved_abel", dtype=object)]
    theta_edges = np.linspace(0.0, np.pi, args.n_theta + 1)
    rows = []

    for ir, (m_pc, nu_pc3) in enumerate(zip(shell_m_pc, shell_nu_pc3)):
        shell_volume_oblate = q * shell_spherical_volume[ir]

        for it in range(args.n_theta):
            th0 = theta_edges[it]
            th1 = theta_edges[it + 1]
            th = 0.5 * (th0 + th1)
            theta_fraction = abs(np.cos(th0) - np.cos(th1)) / 2.0
            cell_volume = shell_volume_oblate * theta_fraction
            cell_luminosity = L_shell[ir] * theta_fraction
            R_cyl = m_pc * np.sin(th)
            z = q * m_pc * np.cos(th)
            r_spherical = np.sqrt(R_cyl * R_cyl + z * z)
            nu_lsun_pc3 = cell_luminosity / cell_volume

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
                "R_inner_pc": shell_R_inner[ir],
                "R_outer_pc": shell_R_outer[ir],
                "q_axis_ratio": q,
                "nu_pc3": nu_pc3,
                "nu_Lsun_pc3": nu_lsun_pc3,
                "cell_volume_pc3": cell_volume,
                "cell_luminosity_Lsun": cell_luminosity,
                "shell_luminosity_Lsun": L_shell[ir],
                "light_frac": light_frac[ir],
                "Lenc_frac": Lenc_frac[ir],
                "density_source": shell_source[ir],
                "geometry": "axisymmetric_density_grid",
                "flattened_geometry": "oblate_homeoid",
                "density_coordinate": "m_pc",
                "coordinate_model": "R_cyl=m*sin(theta), z=q*m*cos(theta)",
                "source_grid_geometry": "prepared_radial_density_grid",
                "source_surface_brightness_csv": str(sb_path),
                "source_density_profile_csv": str(density_path),
            })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("Stellar force grid produced zero rows")

    for col in ["galaxy", "source", "preferred_profile", "radius_type", "ellipticity"]:
        if col in sb.columns:
            out[col] = sb[col].iloc[0]

    out["Ltot_Lsun"] = float(args.ltot)
    out["n_radial"] = int(len(shell_m_pc))
    out["n_resolved_abel"] = int(len(m_grid))
    out["n_theta"] = int(args.n_theta)
    out["abel_support_min_pc"] = support_min_pc
    out["unresolved_center_raw_light"] = unresolved_raw_light
    out["unresolved_center_Lsun"] = float(np.sum(L_shell[:central_n_shells]))
    out["unresolved_center_model"] = inner_model
    out["inner_cslope"] = inner_cslope
    out["inner_boundary_nu_pc3"] = inner_boundary_nu_pc3
    out["inner_continuation_raw_light"] = inner_continuation_raw_light
    out["inner_continuation_light_residual"] = inner_continuation_light_residual
    out["unresolved_center_n_shells"] = central_n_shells
    out["product_type"] = "stellar_force_grid"
    out["theta_range"] = "0_to_pi_full_meridional_plane"
    out["volume_model"] = "oblate_homeoid_shell_fraction"
    validate_axisymmetric_grid(out, args.ltot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"N rows: {len(out)}")
    print(f"N radial shells: {len(shell_m_pc)}")
    print(f"N unresolved-center shells: {central_n_shells}")
    print(f"N resolved Abel radial points: {len(m_grid)}")
    print(f"N theta: {args.n_theta}")
    print("product type: stellar_force_grid")
    print("geometry: axisymmetric_density_grid")
    print("source geometry: prepared_radial_density_grid")
    print(f"q_axis_ratio: {q}")
    print(f"Abel support minimum [pc]: {support_min_pc}")
    print(f"unresolved center raw light: {unresolved_raw_light}")
    print(f"unresolved center mapped light [Lsun]: {np.sum(L_shell[:central_n_shells])}")
    print(f"unresolved center model: {inner_model}")
    print(f"Karl inner cslope: {inner_cslope}")
    print(f"inner boundary nu [stars pc^-3]: {inner_boundary_nu_pc3}")
    print(f"inner continuation raw light: {inner_continuation_raw_light}")
    print(f"inner continuation light residual: {inner_continuation_light_residual}")
    print(f"Ltot target [Lsun]: {args.ltot}")
    print(f"Lsum grid [Lsun]: {out['cell_luminosity_Lsun'].sum()}")
    print(f"nu_pc3 min/max: {out['nu_pc3'].min()} {out['nu_pc3'].max()}")
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
            return (xmgamma / (4.0 * np.pi) * (3.0 - gamma) * mscale / (m**gamma * (mscale + m) ** (4.0 - gamma) + 1e-30))
        return (3.0 * xmgamma / (4.0 * np.pi * mscale**3) * (1.0 + (m / mscale) ** 2) ** (-2.5))

    if args.halo_type == "karl_isothermal":
        qdm = q
        rc = max(float(args.rc_pc), 1e-12)
        v0 = float(args.v0_km_s)
        dis = max(float(args.dis), 1e-12)
        xR = R
        xZ = z
        rho = 0.78722918 / (dis * dis)
        rho *= v0 * v0 / (qdm * qdm)
        num = ((2.0 * qdm * qdm + 1.0) * rc * rc + xR * xR + 2.0 * (1.0 - 0.5 / (qdm * qdm)) * xZ * xZ)
        den = (rc * rc + xR * xR + xZ * xZ / (qdm * qdm)) ** 2
        return rho * num / np.maximum(den, 1e-30)
    raise ValueError(f"Unknown halo_type: {args.halo_type}")


def validate_gden_products(out, args):
    required = {"shell_id", "theta_id", "R_cyl_pc", "z_pc", "r_pc", "m_pc","theta_rad", "cell_volume_pc3", "dL_Lsun", "dM_star_Msun",
        "dMhalo_Msun", "ratML_local", "rho_halo_Msun_pc3", "stellar_ML", "halo_type", "geometry", "product_type"}
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

    p_axis = sub.add_parser("build-axisymmetric-light-grid", help="Map a prepared radial density profile onto the axisymmetric stellar geometry.")
    p_axis.add_argument("--surface-brightness", required=True)
    p_axis.add_argument("--density-profile", required=True)
    p_axis.add_argument("--out", required=True)
    p_axis.add_argument("--ltot", type=float, default=2.7e5)
    p_axis.add_argument("--n-radial", type=int, default=None)
    p_axis.add_argument("--n-theta", type=int, default=64)
    p_axis.add_argument("--q-axis-ratio", type=float, default=None)
    p_axis.set_defaults(func=cmd_build_axisymmetric_light_grid)

    p_force = sub.add_parser("build-stellar-force-grid", help="Build the stellar-force grid from the prepared Abel density and its Karl-style unresolved-center continuation.")
    p_force.add_argument("--surface-brightness", required=True)
    p_force.add_argument("--density-profile", required=True)
    p_force.add_argument("--out", required=True)
    p_force.add_argument("--ltot", type=float, default=2.7e5)
    p_force.add_argument("--n-radial", type=int, default=None)
    p_force.add_argument("--n-theta", type=int, default=64)
    p_force.add_argument("--q-axis-ratio", type=float, default=None)
    p_force.add_argument("--central-shells", type=int, default=32)
    p_force.set_defaults(func=cmd_build_stellar_force_grid)


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
