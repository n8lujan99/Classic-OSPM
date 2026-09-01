import csv
import math
from pathlib import Path


# ============================================================
# SEGUE 1 SOURCE INPUT SECTION
# ============================================================

GALAXY = "Segue1"
SOURCE = "Niederste-Ostholt_et_al_2009_Fig7_digitized"
PREFERRED_PROFILE = "CMD_mask_number_counts"
RADIUS_TYPE = "projected_circular_radius"

DISTANCE_PC = 23000.0
PC_PER_ARCMIN = DISTANCE_PC * math.pi / (180.0 * 60.0)
ELLIPTICITY = 0.0
Q_AXIS_RATIO = 1.0

# Niederste-Ostholt et al. (2009) Fig. 7 uses fixed-width radial bins.
# The archived digitized x coordinates fall at the bin centers to within
# digitization/calibration error: approximately 0.0125, 0.0375, 0.0625, ... deg.
ANNULUS_WIDTH_DEG = 0.025
HALF_ANNULUS_WIDTH_DEG = 0.5 * ANNULUS_WIDTH_DEG

# No additional background subtraction is applied here. The archived source
# material contains only the digitized Fig. 7 profile and the previous Segue 1
# preparation did not apply a separate fitted-background correction.
BACKGROUND = 0.0
BACKGROUND_ERR = 0.0

NOTE = (
    "Digitized Niederste-Ostholt et al. 2009 Fig. 7 stellar number-count profile. "
    "Digitized x coordinates are treated as noisy measurements of the centers of "
    "fixed 0.025 deg annuli. Exact contiguous annulus edges are reconstructed from "
    "that published bin spacing. Digitized counts are rounded to integer tracer "
    "counts and assigned Poisson sqrt(N) uncertainties. No additional background "
    "subtraction is applied at this source-construction stage."
)

PROFILE_ROOT = Path(__file__).resolve().parent
RAW_INPUT = PROFILE_ROOT / "archive" / "Segue1_NO09_digitized_raw_points.csv"
OUTPATH = PROFILE_ROOT / "Segue1_surface_brightness_source.csv"


# ============================================================
# SOURCE CONSTRUCTION
# ============================================================

if not RAW_INPUT.is_file():
    raise FileNotFoundError(f"Missing Segue 1 digitized source points: {RAW_INPUT}")

with RAW_INPUT.open(newline="") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        raise ValueError(f"Source file has no CSV header: {RAW_INPUT}")

    required = {"rin_deg", "count_digitized"}
    missing = required - set(reader.fieldnames)
    if missing:
        raise KeyError(f"Digitized source file missing columns: {sorted(missing)}")

    raw_rows = []
    for row_number, row in enumerate(reader, start=2):
        try:
            digitized_center_deg = float(row["rin_deg"])
            count_digitized = float(row["count_digitized"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value in {RAW_INPUT} row {row_number}") from exc

        if not math.isfinite(digitized_center_deg) or digitized_center_deg <= 0.0:
            raise ValueError(f"Invalid digitized radius in row {row_number}: {digitized_center_deg}")
        if not math.isfinite(count_digitized) or count_digitized <= 0.0:
            raise ValueError(f"Invalid digitized count in row {row_number}: {count_digitized}")

        raw_rows.append((digitized_center_deg, count_digitized))

if not raw_rows:
    raise ValueError("Segue 1 digitized source file contains no data rows")

raw_rows.sort(key=lambda x: x[0])

mapped_rows = []
for digitized_center_deg, count_digitized in raw_rows:
    bin_id = int(round(digitized_center_deg / ANNULUS_WIDTH_DEG - 0.5))
    if bin_id < 0:
        raise ValueError(
            f"Digitized radius maps to a negative annulus index: "
            f"digitized_center={digitized_center_deg:.12g} deg, bin_id={bin_id}"
        )

    rmid_deg = (bin_id + 0.5) * ANNULUS_WIDTH_DEG
    center_offset_deg = digitized_center_deg - rmid_deg

    if abs(center_offset_deg) >= HALF_ANNULUS_WIDTH_DEG:
        raise ValueError(
            f"Digitized point cannot be assigned uniquely to a 0.025 deg annulus: "
            f"digitized_center={digitized_center_deg:.12g} deg, "
            f"nearest_center={rmid_deg:.12g} deg, "
            f"offset={center_offset_deg:.12g} deg"
        )

    mapped_rows.append((bin_id, digitized_center_deg, count_digitized, center_offset_deg))

bin_ids = [row[0] for row in mapped_rows]

if len(set(bin_ids)) != len(bin_ids):
    raise ValueError(f"Multiple digitized points map to the same annulus: {bin_ids}")

expected_bin_ids = list(range(bin_ids[0], bin_ids[-1] + 1))
if bin_ids != expected_bin_ids:
    raise ValueError(
        "Digitized points do not form one contiguous fixed-width annulus sequence: "
        f"mapped={bin_ids}, expected={expected_bin_ids}"
    )

if bin_ids[0] != 0:
    raise ValueError(
        f"Segue 1 source profile must begin with the central annulus; first mapped bin is {bin_ids[0]}"
    )

processed = []

for bin_id, digitized_center_deg, count_digitized, center_offset_deg in mapped_rows:
    rin_deg = bin_id * ANNULUS_WIDTH_DEG
    rout_deg = (bin_id + 1) * ANNULUS_WIDTH_DEG
    rmid_deg = (bin_id + 0.5) * ANNULUS_WIDTH_DEG

    rin_arcmin = 60.0 * rin_deg
    rout_arcmin = 60.0 * rout_deg
    rmid_arcmin = 60.0 * rmid_deg

    count = int(round(count_digitized))
    if count <= 0:
        raise ValueError(f"Rounded tracer count is not positive in bin {bin_id}: {count}")
    count_err = math.sqrt(count)

    area_arcmin2 = math.pi * Q_AXIS_RATIO * (rout_arcmin**2 - rin_arcmin**2)
    area_pc2 = area_arcmin2 * PC_PER_ARCMIN**2

    sigma = count / area_arcmin2
    sigma_err = count_err / area_arcmin2

    processed.append({
        "bin_id": bin_id,
        "galaxy": GALAXY,
        "source": SOURCE,
        "preferred_profile": PREFERRED_PROFILE,
        "radius_type": RADIUS_TYPE,

        "digitized_center_deg": digitized_center_deg,
        "digitized_center_offset_deg": center_offset_deg,
        "count_digitized": count_digitized,
        "count": count,
        "count_err": count_err,

        "rin_deg": rin_deg,
        "rout_deg": rout_deg,
        "rmid_deg": rmid_deg,
        "rin_arcmin": rin_arcmin,
        "rout_arcmin": rout_arcmin,
        "rm_arcmin": rmid_arcmin,

        "rin_pc": rin_arcmin * PC_PER_ARCMIN,
        "rout_pc": rout_arcmin * PC_PER_ARCMIN,
        "rm_pc": rmid_arcmin * PC_PER_ARCMIN,

        "Sigma": sigma,
        "Sigma_err": sigma_err,
        "Sigma_pc2": sigma / PC_PER_ARCMIN**2,
        "Sigma_err_pc2": sigma_err / PC_PER_ARCMIN**2,

        "background": BACKGROUND,
        "background_err": BACKGROUND_ERR,
        "ellipticity": ELLIPTICITY,
        "q_axis_ratio": Q_AXIS_RATIO,
        "area_arcmin2": area_arcmin2,
        "area_pc2": area_pc2,
        "Sigma_units": "stars_per_arcmin2",
        "R_units": "pc",
        "area_model": "circular_annulus_pi_delta_r2",
        "pc_per_arcmin_assumed": PC_PER_ARCMIN,
        "annulus_width_deg": ANNULUS_WIDTH_DEG,
        "note": NOTE,
    })


fieldnames = [
    "bin_id",
    "galaxy",
    "source",
    "preferred_profile",
    "radius_type",

    "digitized_center_deg",
    "digitized_center_offset_deg",
    "count_digitized",
    "count",
    "count_err",

    "rin_deg",
    "rout_deg",
    "rmid_deg",
    "rin_arcmin",
    "rout_arcmin",
    "rm_arcmin",

    "rin_pc",
    "rout_pc",
    "rm_pc",

    "Sigma",
    "Sigma_err",
    "Sigma_pc2",
    "Sigma_err_pc2",

    "background",
    "background_err",
    "ellipticity",
    "q_axis_ratio",
    "area_arcmin2",
    "area_pc2",
    "Sigma_units",
    "R_units",
    "area_model",
    "pc_per_arcmin_assumed",
    "annulus_width_deg",
    "note",
]

OUTPATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPATH.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(processed)

max_center_offset = max(abs(row["digitized_center_offset_deg"]) for row in processed)

print(OUTPATH)
print(f"Rows written: {len(processed)}")
print(f"Source: {SOURCE}")
print(f"Fixed annulus width: {ANNULUS_WIDTH_DEG:.6f} deg")
print(f"pc per arcmin: {PC_PER_ARCMIN:.12f}")
print(f"q_axis_ratio: {Q_AXIS_RATIO:.6f}")
print(f"Maximum digitized-center offset: {max_center_offset:.12g} deg")
print(f"First annulus: {processed[0]['rin_deg']:.6f} -> {processed[0]['rout_deg']:.6f} deg")
print(f"Last annulus: {processed[-1]['rin_deg']:.6f} -> {processed[-1]['rout_deg']:.6f} deg")
