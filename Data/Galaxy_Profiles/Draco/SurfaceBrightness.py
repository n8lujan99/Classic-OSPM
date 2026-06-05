import csv
import math
from pathlib import Path


# ============================================================
# USER INPUT SECTION
# ============================================================

galaxy = "Draco"
source = "Odenkirchen_et_al_2001_Table_3"
radius_type = "elliptical_major_axis"

# Conversion and geometry
pc_per_arcmin = 221.0 / 10.0
ellipticity = 0.31
q_axis_ratio = 1.0 - ellipticity

# Which profile should become the generic Karl OSPM profile?
# Usually use "S2" if S2 is the cleaner CMD-filtered profile.
preferred_profile = "S2"

# Optional background values
backgrounds = {
    "S1_background_arcmin2": 0.1511,
    "S1_background_err_arcmin2": 0.0020,
    "S2_background_arcmin2": 0.0760,
    "S2_background_err_arcmin2": 0.0014,
}

note = (
    "Sigma values are background-subtracted stellar surface densities; "
    "S2 is usually the cleaner CMD-filtered profile."
)

# Rows must be:
# rin_arcmin, rout_arcmin, rm_arcmin, S1, S1_err, S2, S2_err
rows = [
    (0.0, 1.0, 0.71, 10.250, 2.137, 7.181, 1.795),
    (1.0, 2.0, 1.58, 8.467, 1.121, 5.087, 0.872),
    (2.0, 3.0, 2.55, 7.219, 0.802, 4.937, 0.666),
    (3.0, 4.0, 3.54, 7.448, 0.689, 4.937, 0.563),
    (4.0, 5.0, 4.53, 7.130, 0.594, 4.388, 0.468),
    (5.0, 6.0, 5.52, 4.740, 0.438, 3.142, 0.358),
    (6.0, 7.0, 6.52, 5.142, 0.420, 3.452, 0.345),
    (7.0, 8.0, 7.52, 3.654, 0.329, 2.723, 0.285),
    (8.0, 10.0, 9.06, 3.181, 0.198, 2.132, 0.163),
    (10.0, 12.0, 11.05, 2.380, 0.155, 1.642, 0.129),
    (12.0, 14.0, 13.04, 1.363, 0.108, 0.941, 0.090),
    (14.0, 16.0, 15.03, 1.099, 0.090, 0.778, 0.076),
    (16.0, 18.0, 17.03, 0.629, 0.064, 0.442, 0.054),
    (18.0, 22.0, 20.10, 0.485, 0.037, 0.278, 0.028),
    (22.0, 28.0, 25.18, 0.300, 0.021, 0.180, 0.016),
    (28.0, 34.0, 31.14, 0.193, 0.015, 0.110, 0.012),
    (34.0, 40.0, 37.12, 0.165, 0.013, 0.088, 0.009),
    (40.0, 60.0, 50.99, 0.150, 0.006, 0.078, 0.004),
]

outpath = Path("draco_oden_kirchen2001_surface_brightness_profile.csv")


# ============================================================
# PROCESSING SECTION
# ============================================================

if preferred_profile not in ("S1", "S2"):
    raise ValueError("preferred_profile must be 'S1' or 'S2'")

if not (0.0 <= ellipticity < 1.0):
    raise ValueError("ellipticity must satisfy 0 <= e < 1")

processed = []

for rin, rout, rmid, s1, e1, s2, e2 in rows:
    if rout <= rin:
        raise ValueError(f"Bad radial bin: rin={rin}, rout={rout}")

    area_arcmin2 = math.pi * q_axis_ratio * (rout**2 - rin**2)
    area_pc2 = area_arcmin2 * pc_per_arcmin**2

    s1_counts = s1 * area_arcmin2
    s1_counts_err = e1 * area_arcmin2

    s2_counts = s2 * area_arcmin2
    s2_counts_err = e2 * area_arcmin2

    row = {
        "galaxy": galaxy,
        "source": source,
        "radius_type": radius_type,

        "rin_arcmin": rin,
        "rout_arcmin": rout,
        "rm_arcmin": rmid,

        "rin_pc": rin * pc_per_arcmin,
        "rout_pc": rout * pc_per_arcmin,
        "rm_pc": rmid * pc_per_arcmin,

        "ellipticity": ellipticity,
        "q_axis_ratio": q_axis_ratio,

        "area_arcmin2": area_arcmin2,
        "area_pc2": area_pc2,

        "S1_sigma_arcmin2": s1,
        "S1_sigma_err_arcmin2": e1,
        "S1_counts_bin": s1_counts,
        "S1_counts_err_bin": s1_counts_err,

        "S2_sigma_arcmin2": s2,
        "S2_sigma_err_arcmin2": e2,
        "S2_counts_bin": s2_counts,
        "S2_counts_err_bin": s2_counts_err,
        "S1_sigma_pc2": s1 / pc_per_arcmin**2,
        "S1_sigma_err_pc2": e1 / pc_per_arcmin**2,
        "S2_sigma_pc2": s2 / pc_per_arcmin**2,
        "S2_sigma_err_pc2": e2 / pc_per_arcmin**2,
        "Sigma_units": "stars_per_arcmin2",
        "R_units": "pc",
        "area_model": "elliptical_annulus_pi_q_delta_a2",

        "pc_per_arcmin_assumed": pc_per_arcmin,
        "note": note,
    }

    row.update(backgrounds)
    processed.append(row)


s1_total = sum(r["S1_counts_bin"] for r in processed)
s2_total = sum(r["S2_counts_bin"] for r in processed)

if s1_total <= 0:
    raise ValueError("S1 total tracer-count proxy is not positive")

if s2_total <= 0:
    raise ValueError("S2 total tracer-count proxy is not positive")


for r in processed:
    r["S1_light_fraction"] = r["S1_counts_bin"] / s1_total
    r["S2_light_fraction"] = r["S2_counts_bin"] / s2_total

    r["S1_light_fraction_err_approx"] = r["S1_counts_err_bin"] / s1_total
    r["S2_light_fraction_err_approx"] = r["S2_counts_err_bin"] / s2_total

    # Generic columns for Karl-style OSPM.
    # These are what the surface_brightness_profile loader should read.
    r["preferred_profile"] = preferred_profile
    r["R_pc"] = r["rm_pc"]
    r["R_inner_pc"] = r["rin_pc"]
    r["R_outer_pc"] = r["rout_pc"]

    r["Sigma"] = r[f"{preferred_profile}_sigma_arcmin2"]
    r["Sigma_err"] = r[f"{preferred_profile}_sigma_err_arcmin2"]
    r["light_frac"] = r[f"{preferred_profile}_light_fraction"]
    r["light_frac_err_approx"] = r[f"{preferred_profile}_light_fraction_err_approx"]
    r["Sigma_pc2"] = r[f"{preferred_profile}_sigma_pc2"]
    r["Sigma_err_pc2"] = r[f"{preferred_profile}_sigma_err_pc2"]


fieldnames = [
    "galaxy",
    "source",
    "radius_type",
    "preferred_profile",

    "rin_arcmin",
    "rout_arcmin",
    "rm_arcmin",

    "rin_pc",
    "rout_pc",
    "rm_pc",

    "R_pc",
    "R_inner_pc",
    "R_outer_pc",

    "Sigma",
    "Sigma_err",
    "light_frac",
    "light_frac_err_approx",

    "ellipticity",
    "q_axis_ratio",

    "area_arcmin2",
    "area_pc2",

    "S1_sigma_arcmin2",
    "S1_sigma_err_arcmin2",
    "S1_counts_bin",
    "S1_counts_err_bin",
    "S1_light_fraction",
    "S1_light_fraction_err_approx",

    "S2_sigma_arcmin2",
    "S2_sigma_err_arcmin2",
    "S2_counts_bin",
    "S2_counts_err_bin",
    "S2_light_fraction",
    "S2_light_fraction_err_approx",

    "S1_background_arcmin2",
    "S1_background_err_arcmin2",
    "S2_background_arcmin2",
    "S2_background_err_arcmin2",

    "pc_per_arcmin_assumed",
    "note",
]


with outpath.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(processed)


print(outpath)
print(f"Rows written: {len(processed)}")
print(f"S1 total tracer counts proxy: {s1_total:.6f}")
print(f"S2 total tracer counts proxy: {s2_total:.6f}")
print(f"S1 light fraction sum: {sum(r['S1_light_fraction'] for r in processed):.12f}")
print(f"S2 light fraction sum: {sum(r['S2_light_fraction'] for r in processed):.12f}")
print(f"Karl preferred profile: {preferred_profile}")
print(f"Karl light fraction sum: {sum(r['light_frac'] for r in processed):.12f}")