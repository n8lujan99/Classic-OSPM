#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _read_config():
    from Data.Galaxy_Profiles.Draco.Draco_OSPM_Config import CONFIG
    return CONFIG


def check_files_and_rows(config):
    print("\n[CHECK] Config and data products")

    paths = {
        "DATA_CSV": config["DATA_CSV"],
        "SURFACE_BRIGHTNESS_CSV": config["SURFACE_BRIGHTNESS_CSV"],
        "KINEMATIC_BINS_CSV": config["KINEMATIC_BINS_CSV"],
        "STELLAR_GRID": config["STELLAR_MODEL"]["grid_csv"],
    }

    for label, path in paths.items():
        p = Path(path)
        print(f"{label:24s}", "OK" if p.exists() else "MISS", p)
        if not p.exists():
            raise FileNotFoundError(path)

    sb = pd.read_csv(config["SURFACE_BRIGHTNESS_CSV"])
    kb = pd.read_csv(config["KINEMATIC_BINS_CSV"])
    grid = pd.read_csv(config["STELLAR_MODEL"]["grid_csv"])

    if len(sb) != len(kb):
        raise RuntimeError(
            f"surface-brightness rows ({len(sb)}) do not match kinematic-bin rows ({len(kb)}). "
            "Run observable_mapping.py rebin-sb onto the kinematic bins."
        )

    if "light_frac" not in sb.columns:
        raise RuntimeError("surface-brightness CSV is missing light_frac")

    lsum = float(sb["light_frac"].sum())
    if not np.isfinite(lsum) or not np.isclose(lsum, 1.0, rtol=0.0, atol=1e-8):
        raise RuntimeError(f"surface-brightness light_frac sum is {lsum}, expected 1")

    print("surface brightness rows:", len(sb))
    print("kinematic bin rows:    ", len(kb))
    print("light_frac sum:        ", f"{lsum:.16f}")

    if "geometry" in grid.columns:
        print("grid geometry values:  ", list(grid["geometry"].dropna().unique()))
    else:
        print("grid geometry values:   NO geometry column")

    if "Lenc_frac" in grid.columns:
        print("Lenc_frac max:         ", f"{grid['Lenc_frac'].max():.16f}")

    if {"cell_luminosity_Lsun", "cell_volume_pc3", "R_cyl_pc", "z_pc"}.issubset(grid.columns):
        print("axisymmetric Lsum:     ", f"{grid['cell_luminosity_Lsun'].sum():.8f}")
        print("axisymmetric z range:  ", float(grid["z_pc"].min()), float(grid["z_pc"].max()))
        print("axisymmetric R min:    ", float(grid["R_cyl_pc"].min()))
        print("axisymmetric volume min:", float(grid["cell_volume_pc3"].min()))

    return sb, kb, grid


def check_active_axisymmetric_grid(config):
    print("\n[CHECK] Active stellar grid contract")

    sm = config["STELLAR_MODEL"]
    geometry = str(sm.get("geometry", "")).strip().lower()
    grid = pd.read_csv(sm["grid_csv"])

    print("config geometry:", geometry)

    if geometry == "axisymmetric_density_grid":
        required = [
            sm.get("R_cyl_col", "R_cyl_pc"),
            sm.get("z_col", "z_pc"),
            sm.get("nu_col", "nu_Lsun_pc3"),
            sm.get("volume_col", "cell_volume_pc3"),
            sm.get("luminosity_col", "cell_luminosity_Lsun"),
            "q_axis_ratio",
        ]
        missing = [c for c in required if c not in grid.columns]
        if missing:
            raise RuntimeError(f"axisymmetric grid is missing required columns: {missing}")

        if "geometry" in grid.columns:
            values = {str(x).strip().lower() for x in grid["geometry"].dropna().unique()}
            if "axisymmetric_density_grid" not in values:
                raise RuntimeError(f"config says axisymmetric_density_grid but grid geometry values are {values}")

        lsum = float(grid[sm.get("luminosity_col", "cell_luminosity_Lsun")].sum())
        ltot = float(sm["Ltot"])
        if not np.isclose(lsum, ltot, rtol=1e-8, atol=max(1e-8, 1e-8 * ltot)):
            raise RuntimeError(f"axisymmetric grid luminosity sum {lsum} does not match Ltot {ltot}")

        if float(grid[sm.get("volume_col", "cell_volume_pc3")].min()) <= 0:
            raise RuntimeError("axisymmetric grid has non-positive cell volume")

        if float(grid[sm.get("R_cyl_col", "R_cyl_pc")].min()) < -1e-12:
            raise RuntimeError("axisymmetric grid has negative R_cyl")

        z = grid[sm.get("z_col", "z_pc")]
        if not (float(z.min()) < 0.0 and float(z.max()) > 0.0):
            raise RuntimeError("axisymmetric grid z_pc must span both sides of the midplane")

        print("axisymmetric grid contract: PASS")
    else:
        print("axisymmetric grid contract: SKIP because active geometry is not axisymmetric_density_grid")



def _axisymmetric_model_from_grid(config, grid_path):
    sm = dict(config["STELLAR_MODEL"])

    sm["grid_csv"] = str(grid_path)
    sm["geometry"] = "axisymmetric_density_grid"
    sm["q_axis_ratio"] = 1.0

    sm["R_cyl_col"] = sm.get("R_cyl_col", "R_cyl_pc")
    sm["z_col"] = sm.get("z_col", "z_pc")
    sm["nu_col"] = sm.get("nu_col", "nu_Lsun_pc3")
    sm["volume_col"] = sm.get("volume_col", "cell_volume_pc3")
    sm["luminosity_col"] = sm.get("luminosity_col", "cell_luminosity_Lsun")

    return sm


def check_axisymmetric_grid_file(grid_path, *, q_expected=None):
    p = Path(grid_path)
    if not p.exists():
        raise FileNotFoundError(grid_path)

    grid = pd.read_csv(p)

    required = {
        "R_cyl_pc",
        "z_pc",
        "nu_Lsun_pc3",
        "cell_volume_pc3",
        "cell_luminosity_Lsun",
        "q_axis_ratio",
        "geometry",
    }
    missing = required - set(grid.columns)
    if missing:
        raise RuntimeError(f"axisymmetric comparison grid missing columns: {sorted(missing)}")

    values = {str(x).strip().lower() for x in grid["geometry"].dropna().unique()}
    if "axisymmetric_density_grid" not in values:
        raise RuntimeError(f"axisymmetric comparison grid geometry values are {values}")

    if q_expected is not None:
        q = float(grid["q_axis_ratio"].dropna().iloc[0])
        if not np.isclose(q, float(q_expected), rtol=0.0, atol=1e-8):
            raise RuntimeError(f"axisymmetric comparison grid q_axis_ratio={q}, expected {q_expected}")

    if float(grid["cell_volume_pc3"].min()) <= 0.0:
        raise RuntimeError("axisymmetric comparison grid has non-positive cell volume")

    if float(grid["cell_luminosity_Lsun"].sum()) <= 0.0:
        raise RuntimeError("axisymmetric comparison grid luminosity sum is non-positive")

    z = grid["z_pc"]
    if not (float(z.min()) < 0.0 and float(z.max()) > 0.0):
        raise RuntimeError("axisymmetric comparison grid z_pc must span both sides of the midplane")

    return grid


def check_q1_axisymmetric_recovery(config, axisym_q1_grid, *, tolerance=0.25):
    print("\n[CHECK] q=1 axisymmetric stellar force recovery")

    from OSPM.Physics.OSPM_Physics import force_at_rtheta_julia, pc

    check_axisymmetric_grid_file(axisym_q1_grid, q_expected=1.0)

    spherical_model = dict(config["STELLAR_MODEL"])
    axisym_model = _axisymmetric_model_from_grid(config, axisym_q1_grid)

    theta = [0.0, 1800.0, 0.0, 1.0]
    radii_pc = (10.0, 30.0, 100.0, 300.0)

    worst = 0.0

    for r_pc in radii_pc:
        sph = force_at_rtheta_julia(
            r_m=r_pc * pc,
            theta_rad=np.pi / 2.0,
            theta=theta,
            halo_type="none",
            stellar_model=spherical_model,
        )
        ax = force_at_rtheta_julia(
            r_m=r_pc * pc,
            theta_rad=np.pi / 2.0,
            theta=theta,
            halo_type="none",
            stellar_model=axisym_model,
        )

        if not all(np.isfinite(x) for x in (sph["FR"], sph["FZ"], ax["FR"], ax["FZ"])):
            raise RuntimeError("q=1 comparison produced non-finite force values")

        denom = max(abs(sph["FR"]), 1e-30)
        frac = abs(ax["FR"] - sph["FR"]) / denom
        worst = max(worst, frac)

        print(
            f"r={r_pc:7.1f} pc | "
            f"spherical FR={sph['FR']:+.6e} | "
            f"axisym-q1 FR={ax['FR']:+.6e} | "
            f"frac_diff={frac:.4f} | "
            f"axisym-q1 FZ={ax['FZ']:+.6e}"
        )

        if sph["FR"] >= 0.0:
            raise RuntimeError("spherical comparison force should point inward")

        if ax["FR"] >= 0.0:
            raise RuntimeError("q=1 axisymmetric comparison force should point inward")

        if abs(ax["FZ"]) > max(1e-20, 1e-6 * abs(ax["FR"])):
            raise RuntimeError("q=1 axisymmetric midplane FZ should be approximately zero")

        if frac > tolerance:
            raise RuntimeError(
                f"q=1 axisymmetric force differs from spherical force by {frac:.4f}, "
                f"above tolerance {tolerance:.4f}"
            )

    print(f"q=1 recovery: PASS | worst fractional difference = {worst:.4f}")

def check_force(config):
    print("\n[CHECK] Julia force probe")

    from OSPM.Physics.OSPM_Physics import force_at_rtheta_julia, pc

    sm = config["STELLAR_MODEL"]
    theta = [0.0, 1800.0, 0.0, 1.0]

    for r_pc in (10.0, 30.0, 100.0, 300.0):
        mid = force_at_rtheta_julia(
            r_m=r_pc * pc,
            theta_rad=np.pi / 2.0,
            theta=theta,
            halo_type="none",
            stellar_model=sm,
        )
        above = force_at_rtheta_julia(
            r_m=r_pc * pc,
            theta_rad=np.pi / 3.0,
            theta=theta,
            halo_type="none",
            stellar_model=sm,
        )

        print(
            f"r={r_pc:7.1f} pc | "
            f"mid FR={mid['FR']:+.6e} FZ={mid['FZ']:+.6e} | "
            f"above FR={above['FR']:+.6e} FZ={above['FZ']:+.6e}"
        )

        if not (np.isfinite(mid["FR"]) and np.isfinite(mid["FZ"]) and np.isfinite(above["FR"]) and np.isfinite(above["FZ"])):
            raise RuntimeError("force probe produced non-finite values")

        if mid["FR"] >= 0:
            raise RuntimeError("midplane FR should point inward and be negative")

        if abs(mid["FZ"]) > max(1e-20, 1e-6 * abs(mid["FR"])):
            raise RuntimeError("midplane FZ should be approximately zero")

        if above["FR"] >= 0:
            raise RuntimeError("above-plane FR should point inward and be negative")

        if above["FZ"] >= 0:
            raise RuntimeError("above-plane FZ should point toward the midplane and be negative for z > 0")

    print("force probe: PASS")


def main():
    parser = argparse.ArgumentParser(description="Check OSPM Draco surface-brightness, grid, and optional force readiness.")
    parser.add_argument("--force", action="store_true", help="Also run Julia force_at_rtheta probes. Requires OSPM_USE_JULIA=1.")
    parser.add_argument("--axisym-q1-grid", default=None, help="Optional q=1 axisymmetric density-grid CSV to compare against the spherical stellar force.")
    parser.add_argument("--q1-tolerance", type=float, default=0.25, help="Maximum allowed fractional FR difference for the q=1 axisymmetric recovery check.")
    args = parser.parse_args()

    config = _read_config()
    check_files_and_rows(config)
    check_active_axisymmetric_grid(config)

    if args.force:
        check_force(config)

    if args.axisym_q1_grid is not None:
        check_q1_axisymmetric_recovery(config, args.axisym_q1_grid, tolerance=float(args.q1_tolerance))

    print("\nAll requested checks passed.")


if __name__ == "__main__":
    main()
