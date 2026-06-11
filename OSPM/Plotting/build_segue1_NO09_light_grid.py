import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path("Data/Galaxy_Profiles/Segue1")

SB_ON_BINS = ROOT / "segue1_NO09_surface_brightness_on_simon_bins_16.csv"
OUT_GRID = ROOT / "segue1_NO09_axisymmetric_light_grid.csv"

GALAXY = "Segue1"
SOURCE = "Niederste-Ostholt_et_al_2009_Fig7_digitized"

# Keep this tied to the current Segue config for now.
# This is just the tracer luminosity normalization.
# The shape comes from the NO09 number-count profile.
LTOT_LSUN = 340.0

Q_AXIS_RATIO = 1.0
ELLIPTICITY = 0.0
N_THETA = 64


def main():
    sb = pd.read_csv(SB_ON_BINS)

    required = [
        "bin_id",
        "R_inner_pc",
        "R_outer_pc",
        "R_pc",
        "Sigma",
        "light_raw",
        "light_frac",
    ]

    missing = [c for c in required if c not in sb.columns]
    if missing:
        raise ValueError(f"Missing columns in {SB_ON_BINS}: {missing}")

    # Convert the number-count surface density into an arbitrary luminosity
    # surface density with the same total Ltot.
    #
    # This keeps the tracer shape from NO09 but gives the Karl grid Lsun units.
    total_raw = sb["light_raw"].sum()
    if total_raw <= 0:
        raise ValueError("Total raw light/tracer weight is not positive.")

    sigma_scale = LTOT_LSUN / total_raw

    theta_edges = np.linspace(0.0, np.pi, N_THETA + 1)

    rows = []

    cumulative_light_frac = 0.0

    for shell_id, row in sb.iterrows():
        rin = float(row["R_inner_pc"])
        rout = float(row["R_outer_pc"])
        m_mid = 0.5 * (rin + rout)

        light_frac = float(row["light_frac"])
        cumulative_light_frac += light_frac

        shell_luminosity = LTOT_LSUN * light_frac

        shell_volume = (4.0 * np.pi / 3.0) * Q_AXIS_RATIO * (rout**3 - rin**3)

        if shell_volume <= 0:
            raise ValueError(f"Bad shell volume for shell {shell_id}: {shell_volume}")

        nu = shell_luminosity / shell_volume

        sigma_lsun_pc2 = float(row["Sigma"]) * sigma_scale

        for theta_id in range(N_THETA):
            th_in = theta_edges[theta_id]
            th_out = theta_edges[theta_id + 1]
            th_mid = 0.5 * (th_in + th_out)

            R_cyl = m_mid * np.sin(th_mid)
            z = Q_AXIS_RATIO * m_mid * np.cos(th_mid)

            r_sph = np.sqrt(R_cyl**2 + z**2)

            # Axisymmetric oblate-homeoid volume element integrated over phi.
            # For q=1 this is just the spherical-shell wedge volume.
            cell_volume = (
                2.0
                * np.pi
                * Q_AXIS_RATIO
                / 3.0
                * (rout**3 - rin**3)
                * (np.cos(th_in) - np.cos(th_out))
            )

            cell_luminosity = nu * cell_volume

            rows.append(
                {
                    "shell_id": int(shell_id),
                    "theta_id": int(theta_id),
                    "R_cyl_pc": R_cyl,
                    "z_pc": z,
                    "r_pc": r_sph,
                    "m_pc": m_mid,
                    "theta_rad": th_mid,
                    "theta_inner_rad": th_in,
                    "theta_outer_rad": th_out,
                    "R_inner_pc": rin,
                    "R_outer_pc": rout,
                    "Sigma_Lsun_pc2": sigma_lsun_pc2,
                    "q_axis_ratio": Q_AXIS_RATIO,
                    "nu_Lsun_pc3": nu,
                    "cell_volume_pc3": cell_volume,
                    "cell_luminosity_Lsun": cell_luminosity,
                    "shell_luminosity_Lsun": shell_luminosity,
                    "light_frac": light_frac,
                    "Lenc_frac": cumulative_light_frac,
                    "geometry": "axisymmetric_density_grid",
                    "flattened_geometry": "spherical",
                    "density_coordinate": "m_pc",
                    "coordinate_model": "R_cyl=m*sin(theta), z=q*m*cos(theta)",
                    "source_grid_geometry": "spherical_enclosed_light_grid",
                    "source_surface_brightness_csv": str(SB_ON_BINS),
                    "galaxy": GALAXY,
                    "source": SOURCE,
                    "preferred_profile": "CMD_mask_number_counts",
                    "radius_type": "projected_circular_radius",
                    "ellipticity": ELLIPTICITY,
                    "Ltot_Lsun": LTOT_LSUN,
                    "n_radial": len(sb),
                    "n_theta": N_THETA,
                    "theta_range": "0_to_pi_full_meridional_plane",
                    "volume_model": "spherical_shell_fraction",
                    "force_ready": True,
                }
            )

    grid = pd.DataFrame(rows)

    grid.to_csv(OUT_GRID, index=False)

    print(f"Saved {OUT_GRID}")
    print("shape:", grid.shape)
    print("total cell luminosity:", grid["cell_luminosity_Lsun"].sum())
    print("target Ltot:", LTOT_LSUN)
    print()
    print(grid.head().to_string(index=False))


if __name__ == "__main__":
    main()
