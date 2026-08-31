#!/usr/bin/env python3
"""
Build the adopted Draco surface-brightness profile used by OSPM.

Source:
    Odenkirchen et al. (2001), Table 3, S2 profile.

Production treatment:
    Sigma_Draco = Sigma_Table3 - Sigma_background

with

    Sigma_background = 0.0760 +/- 0.0014 stars arcmin^-2.

The quoted Table 3 uncertainty and fitted-background uncertainty are
combined in quadrature for the per-bin production uncertainty.

The raw Table 3 measurements are retained in the output for provenance.
The generic Sigma/Sigma_err/light_frac columns contain the corrected S2
profile used downstream.
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# ADOPTED DRACO PHOTOMETRY
# ============================================================

GALAXY = "Draco"
SOURCE = "Odenkirchen_et_al_2001_Table_3"
PREFERRED_PROFILE = "S2"
RADIUS_TYPE = "elliptical_major_axis"

PC_PER_ARCMIN = 221.0 / 10.0
ELLIPTICITY = 0.31
Q_AXIS_RATIO = 1.0 - ELLIPTICITY

S2_BACKGROUND_ARCMIN2 = 0.0760
S2_BACKGROUND_ERR_ARCMIN2 = 0.0014

# rin_arcmin, rout_arcmin, rm_arcmin, S2, S2_err
TABLE3_S2 = [
    (0.0, 1.0, 0.71, 7.181, 1.795),
    (1.0, 2.0, 1.58, 5.087, 0.872),
    (2.0, 3.0, 2.55, 4.937, 0.666),
    (3.0, 4.0, 3.54, 4.937, 0.563),
    (4.0, 5.0, 4.53, 4.388, 0.468),
    (5.0, 6.0, 5.52, 3.142, 0.358),
    (6.0, 7.0, 6.52, 3.452, 0.345),
    (7.0, 8.0, 7.52, 2.723, 0.285),
    (8.0, 10.0, 9.06, 2.132, 0.163),
    (10.0, 12.0, 11.05, 1.642, 0.129),
    (12.0, 14.0, 13.04, 0.941, 0.090),
    (14.0, 16.0, 15.03, 0.778, 0.076),
    (16.0, 18.0, 17.03, 0.442, 0.054),
    (18.0, 22.0, 20.10, 0.278, 0.028),
    (22.0, 28.0, 25.18, 0.180, 0.016),
    (28.0, 34.0, 31.14, 0.110, 0.012),
    (34.0, 40.0, 37.12, 0.088, 0.009),
    (40.0, 60.0, 50.99, 0.078, 0.004),
]


PROFILE_ROOT = Path(__file__).resolve().parent
OUTPATH = PROFILE_ROOT / "Draco_surface_brightness.csv"
PLOT_DIR = PROFILE_ROOT / "plots"
CORRECTION_PLOT = PLOT_DIR / "Draco_surface_brightness_background_correction.png"
ABEL_INPUT_PLOT = PLOT_DIR / "Draco_surface_brightness_abel_input.png"


def build_profile():
    processed = []

    for rin, rout, rmid, sigma_table3, sigma_table3_err in TABLE3_S2:
        if rout <= rin:
            raise ValueError(f"Bad radial bin: rin={rin}, rout={rout}")

        sigma = sigma_table3 - S2_BACKGROUND_ARCMIN2
        sigma_err = math.hypot(sigma_table3_err, S2_BACKGROUND_ERR_ARCMIN2)

        if not math.isfinite(sigma) or not math.isfinite(sigma_err):
            raise ValueError(f"Non-finite corrected profile value at R={rmid} arcmin")
        if sigma <= 0.0:
            raise ValueError(
                f"Background subtraction produced non-positive Sigma at "
                f"R={rmid} arcmin: {sigma:.12g}"
            )

        area_arcmin2 = math.pi * Q_AXIS_RATIO * (rout**2 - rin**2)
        area_pc2 = area_arcmin2 * PC_PER_ARCMIN**2

        light_raw = sigma * area_arcmin2
        light_raw_err = sigma_err * area_arcmin2

        processed.append({
            "galaxy": GALAXY,
            "source": SOURCE,
            "preferred_profile": PREFERRED_PROFILE,
            "radius_type": RADIUS_TYPE,

            "rin_arcmin": rin,
            "rout_arcmin": rout,
            "rm_arcmin": rmid,

            "R_inner_pc": rin * PC_PER_ARCMIN,
            "R_outer_pc": rout * PC_PER_ARCMIN,
            "R_pc": rmid * PC_PER_ARCMIN,

            "Sigma_table3_arcmin2": sigma_table3,
            "Sigma_table3_err_arcmin2": sigma_table3_err,
            "Sigma_background_arcmin2": S2_BACKGROUND_ARCMIN2,
            "Sigma_background_err_arcmin2": S2_BACKGROUND_ERR_ARCMIN2,

            "Sigma": sigma,
            "Sigma_err": sigma_err,
            "Sigma_pc2": sigma / PC_PER_ARCMIN**2,
            "Sigma_err_pc2": sigma_err / PC_PER_ARCMIN**2,

            "ellipticity": ELLIPTICITY,
            "q_axis_ratio": Q_AXIS_RATIO,

            "area_arcmin2": area_arcmin2,
            "area_pc2": area_pc2,

            "light_raw": light_raw,
            "light_raw_err": light_raw_err,

            "Sigma_units": "stars_per_arcmin2",
            "R_units": "pc",
            "area_model": "elliptical_annulus_pi_q_delta_a2",
            "pc_per_arcmin_assumed": PC_PER_ARCMIN,
            "note": (
                "Production Draco light profile: Odenkirchen et al. 2001 "
                "Table 3 S2 with fitted S2 background 0.0760 +/- 0.0014 "
                "stars arcmin^-2 subtracted before normalization. "
                "Sigma_err combines the Table 3 statistical uncertainty "
                "and fitted-background uncertainty in quadrature."
            ),
        })

    light_total = sum(row["light_raw"] for row in processed)

    if not math.isfinite(light_total) or light_total <= 0.0:
        raise ValueError("Corrected Draco light proxy is not positive")

    for row in processed:
        row["light_frac"] = row["light_raw"] / light_total
        row["light_frac_err_approx"] = row["light_raw_err"] / light_total

    light_frac_sum = sum(row["light_frac"] for row in processed)

    if not math.isclose(light_frac_sum, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"light_frac does not sum to unity: {light_frac_sum:.16g}")

    return processed


def write_profile(processed):
    fieldnames = [
        "galaxy",
        "source",
        "preferred_profile",
        "radius_type",

        "rin_arcmin",
        "rout_arcmin",
        "rm_arcmin",

        "R_inner_pc",
        "R_outer_pc",
        "R_pc",

        "Sigma_table3_arcmin2",
        "Sigma_table3_err_arcmin2",
        "Sigma_background_arcmin2",
        "Sigma_background_err_arcmin2",

        "Sigma",
        "Sigma_err",
        "Sigma_pc2",
        "Sigma_err_pc2",

        "light_raw",
        "light_raw_err",
        "light_frac",
        "light_frac_err_approx",

        "ellipticity",
        "q_axis_ratio",
        "area_arcmin2",
        "area_pc2",

        "Sigma_units",
        "R_units",
        "area_model",
        "pc_per_arcmin_assumed",
        "note",
    ]

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed)


def make_plots(processed):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    R_pc = [row["R_pc"] for row in processed]
    sigma_raw = [row["Sigma_table3_arcmin2"] for row in processed]
    sigma_raw_err = [row["Sigma_table3_err_arcmin2"] for row in processed]
    sigma = [row["Sigma"] for row in processed]
    sigma_err = [row["Sigma_err"] for row in processed]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        R_pc,
        sigma_raw,
        yerr=sigma_raw_err,
        fmt="o",
        capsize=3,
        label="Odenkirchen Table 3 S2",
    )

    ax.errorbar(
        R_pc,
        sigma,
        yerr=sigma_err,
        fmt="o",
        capsize=3,
        label="Background-subtracted production profile",
    )

    ax.axhline(
        S2_BACKGROUND_ARCMIN2,
        linestyle="--",
        linewidth=1.2,
        label="Fitted S2 background",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Projected radius R [pc]")
    ax.set_ylabel(r"$\Sigma$ [stars arcmin$^{-2}$]")
    ax.set_title("Draco Surface-Brightness Background Correction")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(CORRECTION_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        R_pc,
        sigma,
        yerr=sigma_err,
        fmt="o",
        capsize=3,
        label="Abel input",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Projected radius R [pc]")
    ax.set_ylabel(r"$\Sigma$ [stars arcmin$^{-2}$]")
    ax.set_title("Draco Production Surface-Brightness Profile")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ABEL_INPUT_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    processed = build_profile()
    write_profile(processed)
    make_plots(processed)

    print(OUTPATH)
    print(f"Rows written: {len(processed)}")
    print(f"Adopted profile: {PREFERRED_PROFILE}")
    print(
        "Adopted S2 background: "
        f"{S2_BACKGROUND_ARCMIN2:.4f} +/- "
        f"{S2_BACKGROUND_ERR_ARCMIN2:.4f} stars arcmin^-2"
    )
    print(
        "Outermost corrected Sigma: "
        f"{processed[-1]['Sigma']:.6f} +/- "
        f"{processed[-1]['Sigma_err']:.6f} stars arcmin^-2"
    )
    print(f"light_frac sum: {sum(row['light_frac'] for row in processed):.12f}")
    print(CORRECTION_PLOT)
    print(ABEL_INPUT_PLOT)


if __name__ == "__main__":
    main()
