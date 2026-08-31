# build_draco_losvd_apertures.py

import csv
import math
from pathlib import Path
from statistics import median


# ============================================================
# USER INPUT SECTION
# ============================================================

galaxy = "Draco"
source = "Walker_et_al_2023_velocity_sample"
surface_brightness_source = "Odenkirchen_et_al_2001_Table_3"

PROFILE_ROOT = Path(__file__).resolve().parent

star_csv = PROFILE_ROOT / "draco_walker2023.csv"
kinematic_bins_csv = PROFILE_ROOT / "draco_walker2023_kinematic_bins_20.csv"
surface_brightness_csv = PROFILE_ROOT / "draco_oden_kirchen2001_surface_brightness_on_walker_bins_20.csv"

outpath = PROFILE_ROOT / "draco_walker2023_losvd_apertures_20x21.csv"

# Star table columns
star_r_col = "r_pc"
star_v_col = "vlos"
star_verr_col = "vlos_err"

# Draco systemic velocity.
# Observed catalog velocities are heliocentric.
# OSPM orbit velocities are internal, centered near zero.
v_sys_kms = -291.68214888089926

# LOSVD binning
Nvbin = 21
velocity_pad_error_factor = 3.0

# Finite-count LOSVD uncertainty model
alpha_dirichlet = 0.5
sigma_floor = 1e-8

# Keep exact kinematic-bin edges by default.
# If we later decide the first bin should start at 0 pc, do that upstream in the bin file.
force_first_inner_zero = True

note = (
    "LOSVD aperture product. sumad is light_frac * p_losvd. "
    "sadfer is the finite-count uncertainty on sumad from the velocity sample only. "
    "sadfer_with_light also includes the approximate light_frac uncertainty."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ffloat(x, default=math.nan):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def iint(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gaussian_bin_probability(v_left, v_right, v0, sig):
    if not (math.isfinite(v_left) and math.isfinite(v_right) and math.isfinite(v0)):
        return 0.0

    if not (math.isfinite(sig) and sig > 0.0):
        return 1.0 if (v_left <= v0 < v_right) else 0.0

    z_left = (v_left - v0) / sig
    z_right = (v_right - v0) / sig

    p = normal_cdf(z_right) - normal_cdf(z_left)
    return max(p, 0.0)


def read_csv_dicts(path):
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def find_aperture_index(r_pc, bins):
    for i, b in enumerate(bins):
        rin = b["R_inner_pc"]
        rout = b["R_outer_pc"]

        if i == len(bins) - 1:
            if rin <= r_pc <= rout:
                return i
        else:
            if rin <= r_pc < rout:
                return i

    return None


def validate_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


# ============================================================
# LOAD INPUTS
# ============================================================

validate_file(star_csv, "star_csv")
validate_file(kinematic_bins_csv, "kinematic_bins_csv")
validate_file(surface_brightness_csv, "surface_brightness_csv")

star_rows = read_csv_dicts(star_csv)
bin_rows_raw = read_csv_dicts(kinematic_bins_csv)
sb_rows_raw = read_csv_dicts(surface_brightness_csv)

if not star_rows:
    raise ValueError(f"star_csv has no rows: {star_csv}")

if not bin_rows_raw:
    raise ValueError(f"kinematic_bins_csv has no rows: {kinematic_bins_csv}")

if not sb_rows_raw:
    raise ValueError(f"surface_brightness_csv has no rows: {surface_brightness_csv}")


# ============================================================
# PREPARE APERTURES
# ============================================================

bins = []

for j, r in enumerate(bin_rows_raw):
    bin_id = iint(r.get("bin_id"), j)
    rin = ffloat(r.get("R_inner_pc"))
    rout = ffloat(r.get("R_outer_pc"))
    rmid = ffloat(r.get("R_mid_pc"), 0.5 * (rin + rout))

    if not (math.isfinite(rin) and math.isfinite(rout) and rout > rin):
        raise ValueError(f"Bad kinematic bin row {j}: R_inner_pc={rin}, R_outer_pc={rout}")

    bins.append(
        {
            "bin_id": bin_id,
            "R_inner_pc": rin,
            "R_outer_pc": rout,
            "R_mid_pc": rmid,
        }
    )

bins.sort(key=lambda x: x["R_inner_pc"])

if force_first_inner_zero:
    bins[0]["R_inner_pc"] = 0.0


# ============================================================
# JOIN SURFACE-BRIGHTNESS LIGHT FRACTIONS
# ============================================================

sb_by_bin = {}

for j, r in enumerate(sb_rows_raw):
    bin_id = iint(r.get("bin_id"), j)

    sb_by_bin[bin_id] = {
        "light_frac": ffloat(r.get("light_frac")),
        "light_frac_err_approx": ffloat(r.get("light_frac_err_approx"), 0.0),
        "Sigma": ffloat(r.get("Sigma")),
        "Sigma_err": ffloat(r.get("Sigma_err")),
        "source_bin_index": j,
    }

for b in bins:
    bin_id = b["bin_id"]

    if bin_id not in sb_by_bin:
        raise KeyError(f"No surface-brightness row found for bin_id={bin_id}")

    sb = sb_by_bin[bin_id]
    L = sb["light_frac"]
    Lerr = sb["light_frac_err_approx"]

    if not (math.isfinite(L) and L >= 0.0):
        raise ValueError(f"Bad light_frac for bin_id={bin_id}: {L}")

    if not (math.isfinite(Lerr) and Lerr >= 0.0):
        Lerr = 0.0

    b.update(sb)
    b["light_frac_err_approx"] = Lerr


light_sum = sum(b["light_frac"] for b in bins)

if light_sum <= 0.0:
    raise ValueError("Total joined light_frac is not positive")

# Normalize gently in case the input file has roundoff.
for b in bins:
    b["light_frac"] /= light_sum
    b["light_frac_err_approx"] /= light_sum


# ============================================================
# PREPARE VALID STARS IN INTERNAL VELOCITY FRAME
# ============================================================

stars = []
dropped_bad = 0

for r in star_rows:
    R = ffloat(r.get(star_r_col))
    v_raw = ffloat(r.get(star_v_col))
    verr = ffloat(r.get(star_verr_col))

    if not (
        math.isfinite(R)
        and math.isfinite(v_raw)
        and math.isfinite(verr)
        and verr > 0.0
    ):
        dropped_bad += 1
        continue

    v_internal = v_raw - v_sys_kms

    stars.append(
        {
            "R_pc": R,
            "v_raw_kms": v_raw,
            "v_kms": v_internal,
            "verr_kms": verr,
        }
    )

if not stars:
    raise ValueError("No valid velocity stars after cleaning")


# ============================================================
# BUILD VELOCITY EDGES
# ============================================================

velocities = [s["v_kms"] for s in stars]
errors = [s["verr_kms"] for s in stars if math.isfinite(s["verr_kms"]) and s["verr_kms"] > 0.0]

pad = velocity_pad_error_factor * median(errors) if errors else max(1.0, 0.1 * (max(velocities) - min(velocities)))

vmin = min(velocities) - pad
vmax = max(velocities) + pad

if not (math.isfinite(vmin) and math.isfinite(vmax) and vmax > vmin):
    raise ValueError(f"Bad velocity range: vmin={vmin}, vmax={vmax}")

dv = (vmax - vmin) / Nvbin
velocity_edges = [vmin + k * dv for k in range(Nvbin + 1)]


# ============================================================
# ACCUMULATE EFFECTIVE LOSVD COUNTS
# ============================================================

Nspatial = len(bins)
Nlosvd = Nspatial * Nvbin

counts_eff = [0.0 for _ in range(Nlosvd)]
counts_by_spatial = [0 for _ in range(Nspatial)]

dropped_outside = 0

for s in stars:
    ib = find_aperture_index(s["R_pc"], bins)

    if ib is None:
        dropped_outside += 1
        continue

    counts_by_spatial[ib] += 1

    probs = [
        gaussian_bin_probability(
            velocity_edges[jb],
            velocity_edges[jb + 1],
            s["v_kms"],
            s["verr_kms"],
        )
        for jb in range(Nvbin)
    ]

    psum = sum(probs)

    if psum > 0.0:
        for jb, p in enumerate(probs):
            row = ib * Nvbin + jb
            counts_eff[row] += p / psum
    else:
        # Fallback for extremely narrow or out-of-grid cases.
        for jb in range(Nvbin):
            if velocity_edges[jb] <= s["v_kms"] < velocity_edges[jb + 1]:
                row = ib * Nvbin + jb
                counts_eff[row] += 1.0
                break


# ============================================================
# BUILD OUTPUT LOSVD APERTURE TABLE
# ============================================================

out_rows = []

for ib, b in enumerate(bins):
    N_i = counts_by_spatial[ib]
    L_i = b["light_frac"]
    Lerr_i = b["light_frac_err_approx"]

    if N_i > 0:
        a0 = N_i + Nvbin * alpha_dirichlet
    else:
        a0 = Nvbin * alpha_dirichlet

    for jb in range(Nvbin):
        row_index = ib * Nvbin + jb
        k_ij = max(counts_eff[row_index], 0.0)

        if N_i > 0:
            p_ij = k_ij / N_i
        else:
            p_ij = 0.0

        p_ij = min(max(p_ij, 0.0), 1.0)

        aj = k_ij + alpha_dirichlet
        var_p = aj * (a0 - aj) / (a0 * a0 * (a0 + 1.0)) if a0 > 0.0 else math.nan
        p_err = math.sqrt(max(var_p, 0.0)) if math.isfinite(var_p) else math.nan

        sumad = L_i * p_ij

        sadfer_kinematic = L_i * p_err if math.isfinite(p_err) else math.inf
        sadfer_light = abs(p_ij) * Lerr_i
        sadfer_with_light = math.sqrt(sadfer_kinematic**2 + sadfer_light**2)

        sadfer = max(sadfer_kinematic, sigma_floor)
        sadfer_with_light = max(sadfer_with_light, sigma_floor)

        v_left = velocity_edges[jb]
        v_right = velocity_edges[jb + 1]
        v_mid = 0.5 * (v_left + v_right)

        valid = N_i > 0 and L_i > 0.0 and math.isfinite(sadfer)

        out_rows.append(
            {
                "galaxy": galaxy,
                "source": source,
                "surface_brightness_source": surface_brightness_source,

                "aperture_id": ib,
                "bin_id": b["bin_id"],
                "vel_bin_id": jb,

                "R_inner_pc": b["R_inner_pc"],
                "R_outer_pc": b["R_outer_pc"],
                "R_mid_pc": b["R_mid_pc"],

                "v_left_kms": v_left,
                "v_right_kms": v_right,
                "v_mid_kms": v_mid,

                "N_vlos": N_i,
                "light_frac": L_i,
                "light_frac_err_approx": Lerr_i,

                "counts_eff": k_ij,
                "p_losvd": p_ij,
                "p_losvd_err_dirichlet": p_err,

                "sumad": sumad,
                "sadfer": sadfer,
                "sadfer_kinematic": sadfer_kinematic,
                "sadfer_light": sadfer_light,
                "sadfer_with_light": sadfer_with_light,

                "sadfer_model": "dirichlet_jeffreys_velocity_only",
                "alpha_dirichlet": alpha_dirichlet,
                "sigma_floor": sigma_floor,

                "v_sys_kms": v_sys_kms,
                "velocity_error_smearing": "gaussian_cdf_per_velocity_bin_then_renormalized",
                "velocity_units": "km_per_s",
                "sumad_units": "fraction_of_total_projected_light",
                "sadfer_units": "fraction_of_total_projected_light",

                "source_star_csv": str(star_csv),
                "source_kinematic_bins_csv": str(kinematic_bins_csv),
                "source_surface_brightness_csv": str(surface_brightness_csv),

                "valid": int(valid),
                "note": note,
            }
        )


# ============================================================
# WRITE OUTPUT
# ============================================================

fieldnames = [
    "galaxy",
    "source",
    "surface_brightness_source",

    "aperture_id",
    "bin_id",
    "vel_bin_id",

    "R_inner_pc",
    "R_outer_pc",
    "R_mid_pc",

    "v_left_kms",
    "v_right_kms",
    "v_mid_kms",

    "N_vlos",
    "light_frac",
    "light_frac_err_approx",

    "counts_eff",
    "p_losvd",
    "p_losvd_err_dirichlet",

    "sumad",
    "sadfer",
    "sadfer_kinematic",
    "sadfer_light",
    "sadfer_with_light",

    "sadfer_model",
    "alpha_dirichlet",
    "sigma_floor",

    "v_sys_kms",
    "velocity_error_smearing",
    "velocity_units",
    "sumad_units",
    "sadfer_units",

    "source_star_csv",
    "source_kinematic_bins_csv",
    "source_surface_brightness_csv",

    "valid",
    "note",
]

with outpath.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)


# ============================================================
# REPORT
# ============================================================

sum_light = sum(b["light_frac"] for b in bins)
sum_sumad = sum(r["sumad"] for r in out_rows)
sum_counts = sum(r["counts_eff"] for r in out_rows)

print(outpath)
print(f"Rows written: {len(out_rows)}")
print(f"Nspatial: {Nspatial}")
print(f"Nvbin: {Nvbin}")
print(f"Nlosvd: {Nlosvd}")
print(f"Valid stars read: {len(stars)}")
print(f"Dropped bad stars: {dropped_bad}")
print(f"Dropped outside apertures: {dropped_outside}")
print(f"Assigned velocity stars: {sum(counts_by_spatial)}")
print(f"Effective counts sum: {sum_counts:.12f}")
print(f"Light fraction sum: {sum_light:.12f}")
print(f"sumad sum: {sum_sumad:.12f}")
print(f"Velocity range internal km/s: {vmin:.6f} to {vmax:.6f}")
print(f"Velocity bin width km/s: {dv:.6f}")
print(f"sadfer min/max: {min(r['sadfer'] for r in out_rows):.12e} / {max(r['sadfer'] for r in out_rows):.12e}")
print("This file is a LOSVD aperture product. It should feed sumad/sadfer to Karl-style OSPM.")