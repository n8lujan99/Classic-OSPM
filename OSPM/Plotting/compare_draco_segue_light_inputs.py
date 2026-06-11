import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path("Data/Galaxy_Profiles")

FILES = {
    "draco_sb": ROOT / "Draco/draco_oden_kirchen2001_surface_brightness_on_walker_bins_20.csv",
    "segue_sb": ROOT / "Segue1/segue1_NO09_surface_brightness_on_simon_bins_16.csv",
    "draco_bins": ROOT / "Draco/draco_walker2023_kinematic_bins_20.csv",
    "segue_bins": ROOT / "Segue1/segue1_simon_kinematic_bins_16.csv",
    "draco_grid": ROOT / "Draco/draco_oden_kirchen2001_axisymmetric_light_grid.csv",
    "segue_grid": ROOT / "Segue1/segue1_NO09_axisymmetric_light_grid.csv",
}

RHALF = {
    "Draco": 221.0,
    "Segue1": 29.4,
}


def check_columns(name_a, df_a, name_b, df_b):
    cols_a = list(df_a.columns)
    cols_b = list(df_b.columns)

    print(f"\nCOLUMN CHECK: {name_a} vs {name_b}")
    print("-" * 80)
    print("same columns:", cols_a == cols_b)

    missing_from_b = [c for c in cols_a if c not in cols_b]
    extra_in_b = [c for c in cols_b if c not in cols_a]

    print("missing from second:", missing_from_b)
    print("extra in second:", extra_in_b)


def summarize_sb(label, df):
    print(f"\nSB SUMMARY: {label}")
    print("-" * 80)
    print("shape:", df.shape)
    print("R range pc:", float(df["R_inner_pc"].min()), "to", float(df["R_outer_pc"].max()))
    print("Sigma range:", float(df["Sigma"].min()), "to", float(df["Sigma"].max()))
    print("light_frac sum:", float(df["light_frac"].sum()))
    print("light_frac min/max:", float(df["light_frac"].min()), float(df["light_frac"].max()))
    print("any negative Sigma:", bool((df["Sigma"] < 0).any()))
    print("any negative light_frac:", bool((df["light_frac"] < 0).any()))
    print(df[["bin_id", "R_inner_pc", "R_outer_pc", "R_pc", "Sigma", "light_frac", "N_vlos"]].head(10).to_string(index=False))


def summarize_grid(label, df):
    print(f"\nGRID SUMMARY: {label}")
    print("-" * 80)
    print("shape:", df.shape)
    print("n_radial unique:", sorted(df["n_radial"].unique()))
    print("n_theta unique:", sorted(df["n_theta"].unique()))
    print("q unique:", sorted(df["q_axis_ratio"].unique()))
    print("Ltot unique:", sorted(df["Ltot_Lsun"].unique()))
    print("cell luminosity sum:", float(df["cell_luminosity_Lsun"].sum()))
    print("nu range:", float(df["nu_Lsun_pc3"].min()), "to", float(df["nu_Lsun_pc3"].max()))
    print("force_ready all true:", bool(df["force_ready"].all()))
    print("any negative density:", bool((df["nu_Lsun_pc3"] < 0).any()))
    print("Lenc monotonic by shell:")

    shell = (
        df.groupby("shell_id")
        .agg(
            R_inner_pc=("R_inner_pc", "first"),
            R_outer_pc=("R_outer_pc", "first"),
            shell_luminosity_Lsun=("shell_luminosity_Lsun", "first"),
            light_frac=("light_frac", "first"),
            Lenc_frac=("Lenc_frac", "first"),
        )
        .reset_index()
    )

    print(shell.head(12).to_string(index=False))
    print("final Lenc_frac:", float(shell["Lenc_frac"].iloc[-1]))


def make_plots(draco_sb, segue_sb, draco_grid, segue_grid):
    outdir = Path("OSPM/Plotting")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Surface density shape, normalized radius.
    plt.figure(figsize=(7, 5))
    plt.loglog(
        draco_sb["R_pc"] / RHALF["Draco"],
        draco_sb["Sigma"] / draco_sb["Sigma"].max(),
        marker="o",
        linestyle="-",
        label="Draco"
    )
    plt.loglog(
        segue_sb["R_pc"] / RHALF["Segue1"],
        segue_sb["Sigma"] / segue_sb["Sigma"].max(),
        marker="s",
        linestyle="-",
        label="Segue 1"
    )
    plt.xlabel("R / R_half")
    plt.ylabel("Sigma / max(Sigma)")
    plt.title("Projected tracer surface-density shape")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "draco_vs_segue_projected_sigma_shape.png", dpi=200)
    plt.close()

    # 2. Cumulative aperture light.
    d = draco_sb.copy()
    s = segue_sb.copy()

    d["cum_light"] = d["light_frac"].cumsum()
    s["cum_light"] = s["light_frac"].cumsum()

    plt.figure(figsize=(7, 5))
    plt.plot(
        d["R_outer_pc"] / RHALF["Draco"],
        d["cum_light"],
        marker="o",
        linestyle="-",
        label="Draco"
    )
    plt.plot(
        s["R_outer_pc"] / RHALF["Segue1"],
        s["cum_light"],
        marker="s",
        linestyle="-",
        label="Segue 1"
    )
    plt.xlabel("Outer radius / R_half")
    plt.ylabel("Cumulative light fraction")
    plt.title("Cumulative tracer light in projected apertures")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "draco_vs_segue_cumulative_light.png", dpi=200)
    plt.close()

    # 3. 3D density shell shape.
    d_shell = (
        draco_grid.groupby("shell_id")
        .agg(m_pc=("m_pc", "first"), nu=("nu_Lsun_pc3", "first"))
        .reset_index()
    )
    s_shell = (
        segue_grid.groupby("shell_id")
        .agg(m_pc=("m_pc", "first"), nu=("nu_Lsun_pc3", "first"))
        .reset_index()
    )

    plt.figure(figsize=(7, 5))
    plt.loglog(
        d_shell["m_pc"] / RHALF["Draco"],
        d_shell["nu"] / d_shell["nu"].max(),
        marker="o",
        linestyle="-",
        label="Draco"
    )
    plt.loglog(
        s_shell["m_pc"] / RHALF["Segue1"],
        s_shell["nu"] / s_shell["nu"].max(),
        marker="s",
        linestyle="-",
        label="Segue 1"
    )
    plt.xlabel("m / R_half")
    plt.ylabel("nu / max(nu)")
    plt.title("3D tracer-density grid shape")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "draco_vs_segue_3d_density_shape.png", dpi=200)
    plt.close()

    print("\nSaved plots:")
    print(outdir / "draco_vs_segue_projected_sigma_shape.png")
    print(outdir / "draco_vs_segue_cumulative_light.png")
    print(outdir / "draco_vs_segue_3d_density_shape.png")


def main():
    data = {k: pd.read_csv(v) for k, v in FILES.items()}

    print("\nFILE SHAPES")
    print("=" * 80)
    for k, df in data.items():
        print(f"{k:12s}", df.shape, FILES[k])

    check_columns("Draco SB-on-bins", data["draco_sb"], "Segue SB-on-bins", data["segue_sb"])
    check_columns("Draco kinematic bins", data["draco_bins"], "Segue kinematic bins", data["segue_bins"])
    check_columns("Draco light grid", data["draco_grid"], "Segue light grid", data["segue_grid"])

    summarize_sb("Draco", data["draco_sb"])
    summarize_sb("Segue1", data["segue_sb"])

    summarize_grid("Draco", data["draco_grid"])
    summarize_grid("Segue1", data["segue_grid"])

    make_plots(
        data["draco_sb"],
        data["segue_sb"],
        data["draco_grid"],
        data["segue_grid"],
    )


if __name__ == "__main__":
    main()
