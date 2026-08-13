surface_brightness_variants.py

Generate observationally allowed surface-brightness profile variants.

Two constraints are enforced:

1. Every changed bin stays within +/- MAX_BIN_SIGMA of its quoted uncertainty.
2. The area-weighted total galaxy brightness stays within +/- MAX_TOTAL_SIGMA
   of the nominal galaxy brightness uncertainty.

The second rule prevents a profile where many bins all move in the same
direction by an individually acceptable amount but collectively make the
galaxy implausibly bright or faint.

Positive sb_shift_sigma always means BRIGHTER.
For magnitude surface brightness this means the magnitude value decreases.

Examples
--------
Linear surface brightness:
    python surface_brightness_variants.py profile.csv variants \
        --radius-col R_pc \
        --brightness-col Sigma \
        --error-col Sigma_err \
        --units linear \
        --n-profiles 1000

Magnitude / arcsec^2:
    python surface_brightness_variants.py profile.csv variants \
        --radius-col R_arcmin \
        --brightness-col mu_V \
        --error-col mu_V_err \
        --units mag \
        --n-profiles 1000

If your file already contains annulus edges, use:
    --rin-col R_inner --rout-col R_outer

If a published uncertainty on the TOTAL galaxy brightness is known, it is
better to supply it as a fractional 1-sigma uncertainty:
    --total-frac-sigma 0.08

Otherwise the script estimates the total 1-sigma uncertainty from the
individual bin errors assuming independent bins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LN10_OVER_2P5 = np.log(10.0) / 2.5


def edges_from_centers(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct annulus edges from monotonically increasing bin centers."""
    radius = np.asarray(radius, dtype=float)

    if radius.ndim != 1 or len(radius) < 2:
        raise ValueError("Need at least two radial bin centers.")

    if np.any(~np.isfinite(radius)):
        raise ValueError("Radius contains non-finite values.")

    if np.any(np.diff(radius) <= 0):
        raise ValueError("Radius values must be strictly increasing.")

    edges = np.empty(len(radius) + 1, dtype=float)
    edges[1:-1] = 0.5 * (radius[:-1] + radius[1:])

    first_width = edges[1] - radius[0]
    edges[0] = max(0.0, radius[0] - first_width)

    last_width = radius[-1] - edges[-2]
    edges[-1] = radius[-1] + last_width

    return edges[:-1], edges[1:]


def annulus_areas(
    df: pd.DataFrame,
    radius_col: str,
    rin_col: str | None,
    rout_col: str | None,
) -> np.ndarray:
    """
    Return circular annulus areas.

    A constant flattening factor q would multiply every area by the same
    number. That common factor cancels in the standardized total-light
    displacement used here.
    """
    if rin_col is not None or rout_col is not None:
        if rin_col is None or rout_col is None:
            raise ValueError("Use both --rin-col and --rout-col together.")
        rin = df[rin_col].to_numpy(dtype=float)
        rout = df[rout_col].to_numpy(dtype=float)
    else:
        rin, rout = edges_from_centers(df[radius_col].to_numpy(dtype=float))

    if np.any(rin < 0):
        raise ValueError("Inner radii cannot be negative.")
    if np.any(rout <= rin):
        raise ValueError("Every outer radius must exceed its inner radius.")

    return np.pi * (rout**2 - rin**2)


def values_to_flux(
    values: np.ndarray,
    errors: np.ndarray,
    units: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the profile to linear relative flux and its 1-sigma uncertainty.

    Absolute photometric zero point is unnecessary because only relative
    integrated-light changes matter.
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(errors)):
        raise ValueError("Brightness values/errors contain non-finite values.")
    if np.any(errors < 0):
        raise ValueError("Brightness uncertainties cannot be negative.")

    if units == "linear":
        flux = values.copy()
        sigma_flux = errors.copy()

        if np.any(flux < 0):
            raise ValueError("Linear surface brightness cannot be negative.")

    elif units == "mag":
        # Arbitrary common zero point. It cancels from all relative tests.
        flux = 10.0 ** (-0.4 * values)

        # First-order propagation:
        # dF/F = -(ln10/2.5) dmu
        sigma_flux = LN10_OVER_2P5 * flux * errors

    else:
        raise ValueError(f"Unknown units: {units}")

    return flux, sigma_flux


def apply_sigma_shift(
    values: np.ndarray,
    errors: np.ndarray,
    z_bright: np.ndarray,
    units: str,
) -> np.ndarray:
    """
    Apply a requested bin shift.

    z_bright > 0 means the bin becomes brighter.
    z_bright < 0 means the bin becomes fainter.
    """
    if units == "linear":
        return values + z_bright * errors

    if units == "mag":
        # Smaller magnitude = brighter.
        return values - z_bright * errors

    raise ValueError(f"Unknown units: {units}")


def total_light_stats(
    nominal_flux: np.ndarray,
    sigma_flux: np.ndarray,
    candidate_flux: np.ndarray,
    area: np.ndarray,
    total_frac_sigma: float | None,
) -> tuple[float, float, float, float]:
    """
    Return:
        nominal_total,
        candidate_total,
        total_sigma,
        total_shift_in_sigma
    """
    nominal_total = float(np.sum(area * nominal_flux))
    candidate_total = float(np.sum(area * candidate_flux))

    if nominal_total <= 0:
        raise ValueError("Nominal integrated galaxy brightness must be positive.")

    if total_frac_sigma is None:
        # Independent-bin propagation.
        total_sigma = float(np.sqrt(np.sum((area * sigma_flux) ** 2)))
    else:
        if total_frac_sigma <= 0:
            raise ValueError("--total-frac-sigma must be positive.")
        total_sigma = float(total_frac_sigma * nominal_total)

    if total_sigma <= 0:
        raise ValueError("Total galaxy-brightness uncertainty is zero.")

    total_shift_sigma = (candidate_total - nominal_total) / total_sigma

    return nominal_total, candidate_total, total_sigma, total_shift_sigma


def generate_candidate(
    rng: np.random.Generator,
    nbin: int,
    max_bin_sigma: float,
    change_probability: float,
) -> np.ndarray:
    """Generate one random combination of changed and unchanged bins."""
    changed = rng.random(nbin) < change_probability

    if not np.any(changed):
        changed[rng.integers(0, nbin)] = True

    z = np.zeros(nbin, dtype=float)
    z[changed] = rng.uniform(-max_bin_sigma, max_bin_sigma, changed.sum())

    return z


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate surface-brightness variants with bin and total-light 2-sigma guards."
    )

    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)

    parser.add_argument("--radius-col", required=True)
    parser.add_argument("--brightness-col", required=True)
    parser.add_argument("--error-col", required=True)

    parser.add_argument(
        "--units",
        choices=("linear", "mag"),
        default="linear",
        help="Surface-brightness representation in the input file.",
    )

    parser.add_argument("--rin-col", default=None)
    parser.add_argument("--rout-col", default=None)

    parser.add_argument(
        "--n-profiles",
        type=int,
        default=100,
        help="Number of accepted perturbed profiles to create.",
    )

    parser.add_argument(
        "--max-bin-sigma",
        type=float,
        default=2.0,
        help="Maximum absolute shift allowed in any individual bin.",
    )

    parser.add_argument(
        "--max-total-sigma",
        type=float,
        default=2.0,
        help="Maximum absolute shift allowed in integrated galaxy brightness.",
    )

    parser.add_argument(
        "--total-frac-sigma",
        type=float,
        default=None,
        help=(
            "Optional fractional 1-sigma uncertainty in the total galaxy brightness. "
            "If omitted, it is propagated from the bin errors assuming independence."
        ),
    )

    parser.add_argument(
        "--change-probability",
        type=float,
        default=0.35,
        help="Probability that a given bin is changed in a proposed profile.",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1_000_000,
        help="Maximum proposals before aborting.",
    )

    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()

    if args.n_profiles < 1:
        raise ValueError("--n-profiles must be at least 1.")
    if args.max_bin_sigma <= 0:
        raise ValueError("--max-bin-sigma must be positive.")
    if args.max_total_sigma <= 0:
        raise ValueError("--max-total-sigma must be positive.")
    if not 0.0 < args.change_probability <= 1.0:
        raise ValueError("--change-probability must be in (0, 1].")

    df = pd.read_csv(args.input_csv)

    required = {args.radius_col, args.brightness_col, args.error_col}
    if args.rin_col is not None:
        required.add(args.rin_col)
    if args.rout_col is not None:
        required.add(args.rout_col)

    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    values = df[args.brightness_col].to_numpy(dtype=float)
    errors = df[args.error_col].to_numpy(dtype=float)

    area = annulus_areas(
        df=df,
        radius_col=args.radius_col,
        rin_col=args.rin_col,
        rout_col=args.rout_col,
    )

    nominal_flux, sigma_flux = values_to_flux(values, errors, args.units)

    nominal_total, _, total_sigma, _ = total_light_stats(
        nominal_flux=nominal_flux,
        sigma_flux=sigma_flux,
        candidate_flux=nominal_flux,
        area=area,
        total_frac_sigma=args.total_frac_sigma,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    # Save the untouched profile as profile 0000.
    nominal_df = df.copy()
    nominal_df[f"{args.brightness_col}_original"] = values
    nominal_df["sb_shift_sigma"] = np.zeros(len(df))
    nominal_name = f"{args.input_csv.stem}_sb_0000_nominal.csv"
    nominal_df.to_csv(args.output_dir / nominal_name, index=False)

    manifest_rows.append(
        {
            "profile_id": 0,
            "filename": nominal_name,
            "accepted": True,
            "total_light": nominal_total,
            "total_light_ratio": 1.0,
            "total_shift_sigma": 0.0,
            "max_abs_bin_shift_sigma": 0.0,
            "n_changed_bins": 0,
        }
    )

    rng = np.random.default_rng(args.seed)

    accepted = 0
    attempts = 0

    while accepted < args.n_profiles and attempts < args.max_attempts:
        attempts += 1

        z_bright = generate_candidate(
            rng=rng,
            nbin=len(df),
            max_bin_sigma=args.max_bin_sigma,
            change_probability=args.change_probability,
        )

        candidate_values = apply_sigma_shift(
            values=values,
            errors=errors,
            z_bright=z_bright,
            units=args.units,
        )

        # Linear profiles cannot cross below zero.
        if args.units == "linear" and np.any(candidate_values < 0):
            continue

        candidate_flux, _ = values_to_flux(
            values=candidate_values,
            errors=errors,
            units=args.units,
        )

        (
            _,
            candidate_total,
            _,
            total_shift_sigma,
        ) = total_light_stats(
            nominal_flux=nominal_flux,
            sigma_flux=sigma_flux,
            candidate_flux=candidate_flux,
            area=area,
            total_frac_sigma=args.total_frac_sigma,
        )

        # Collective galaxy-brightness guard.
        if abs(total_shift_sigma) > args.max_total_sigma:
            continue

        accepted += 1
        profile_id = accepted

        out = df.copy()
        out[f"{args.brightness_col}_original"] = values
        out[args.brightness_col] = candidate_values
        out["sb_shift_sigma"] = z_bright

        filename = f"{args.input_csv.stem}_sb_{profile_id:04d}.csv"
        out.to_csv(args.output_dir / filename, index=False)

        manifest_rows.append(
            {
                "profile_id": profile_id,
                "filename": filename,
                "accepted": True,
                "total_light": candidate_total,
                "total_light_ratio": candidate_total / nominal_total,
                "total_shift_sigma": total_shift_sigma,
                "max_abs_bin_shift_sigma": float(np.max(np.abs(z_bright))),
                "n_changed_bins": int(np.count_nonzero(z_bright)),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "surface_brightness_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print()
    print("Surface-brightness variant generation complete")
    print(f"Input:                  {args.input_csv}")
    print(f"Accepted variants:      {accepted}")
    print(f"Attempts:               {attempts}")
    print(f"Nominal total light:    {nominal_total:.8g}")
    print(f"1-sigma total error:    {total_sigma:.8g}")
    print(f"Allowed total range:    +/- {args.max_total_sigma:.3g} sigma")
    print(f"Allowed bin range:      +/- {args.max_bin_sigma:.3g} sigma")
    print(f"Manifest:               {manifest_path}")

    if accepted < args.n_profiles:
        raise RuntimeError(
            f"Only generated {accepted}/{args.n_profiles} accepted profiles "
            f"after {attempts} attempts. Increase --max-attempts or reduce "
            f"--change-probability."
        )


if __name__ == "__main__":
    main()



'''#!/usr/bin/env julia

using CSV
using DataFrames

"""
    SurfaceBrightnessPerturbations

Standalone helper for loading a compact surface-brightness perturbation deck
generated by surface_brightness_variants.py.

This file intentionally does NOT:
- launch orbits
- rebuild potentials
- rebuild A_light
- rebuild A_losvd
- contain OSPM solver internals

It only:
1. loads the perturbation CSV,
2. reconstructs each perturbed light target,
3. verifies the expected number of bins,
4. loops over the targets,
5. passes each target to a callback.

The callback is where OSPM can later reuse the already-built orbit library,
A_light, A_losvd, and nominal weights.
"""
module SurfaceBrightnessPerturbations

using CSV
using DataFrames

export SBPerturbation,
       load_sb_perturbations,
       run_sb_perturbations,
       get_light_target


struct SBPerturbation
    profile_id::Int
    light_target::Vector{Float64}
    delta_light::Vector{Float64}
    shift_sigma::Vector{Float64}

    total_light::Float64
    total_light_ratio::Float64
    total_shift_sigma::Float64
    max_abs_bin_shift_sigma::Float64
    n_changed_bins::Int
end


function _indexed_columns(
    names_list::Vector{String},
    prefix::String,
)
    matched = Tuple{Int,String}[]

    for name in names_list
        startswith(name, prefix) || continue

        suffix = name[(lastindex(prefix) + 1):end]

        isempty(suffix) && continue

        idx = try
            parse(Int, suffix)
        catch
            continue
        end

        push!(matched, (idx, name))
    end

    sort!(matched, by=first)

    return matched
end


function _require_columns(
    df::DataFrame,
    required::Vector{String},
)
    available = Set(String.(names(df)))

    missing = String[]

    for column in required
        column in available || push!(missing, column)
    end

    isempty(missing) ||
        error("Perturbation deck is missing required columns: $(join(missing, ", "))")

    return nothing
end


function load_sb_perturbations(
    perturbation_csv::AbstractString;
    expected_nbins::Union{Nothing,Int}=nothing,
)
    isfile(perturbation_csv) ||
        error("Surface-brightness perturbation file not found: $perturbation_csv")

    df = CSV.read(perturbation_csv, DataFrame)

    nrow(df) > 0 ||
        error("Surface-brightness perturbation deck is empty.")

    required = [
        "profile_id",
        "total_light",
        "total_light_ratio",
        "total_shift_sigma",
        "max_abs_bin_shift_sigma",
        "n_changed_bins",
    ]

    _require_columns(df, required)

    names_list = String.(names(df))

    brightness_cols = _indexed_columns(
        names_list,
        "brightness_",
    )

    delta_cols = _indexed_columns(
        names_list,
        "delta_brightness_",
    )

    sigma_cols = _indexed_columns(
        names_list,
        "shift_sigma_",
    )

    isempty(brightness_cols) &&
        error("No brightness_### columns were found in the perturbation deck.")

    nbins = length(brightness_cols)

    length(delta_cols) == nbins ||
        error(
            "Found $nbins brightness columns but $(length(delta_cols)) " *
            "delta_brightness columns."
        )

    length(sigma_cols) == nbins ||
        error(
            "Found $nbins brightness columns but $(length(sigma_cols)) " *
            "shift_sigma columns."
        )

    brightness_indices = first.(brightness_cols)
    delta_indices = first.(delta_cols)
    sigma_indices = first.(sigma_cols)

    brightness_indices == delta_indices ||
        error("brightness_### and delta_brightness_### bin indices do not match.")

    brightness_indices == sigma_indices ||
        error("brightness_### and shift_sigma_### bin indices do not match.")

    brightness_indices == collect(0:(nbins - 1)) ||
        error(
            "Surface-brightness columns must be contiguous and zero-indexed: " *
            "brightness_000, brightness_001, ..."
        )

    if expected_nbins !== nothing
        nbins == expected_nbins ||
            error(
                "Perturbation deck has $nbins light bins, but OSPM expected " *
                "$expected_nbins."
            )
    end

    brightness_names = last.(brightness_cols)
    delta_names = last.(delta_cols)
    sigma_names = last.(sigma_cols)

    perturbations = Vector{SBPerturbation}(undef, nrow(df))

    for row_index in 1:nrow(df)
        row = df[row_index, :]

        light_target = Float64[
            Float64(row[!, Symbol(name)])
            for name in brightness_names
        ]

        delta_light = Float64[
            Float64(row[!, Symbol(name)])
            for name in delta_names
        ]

        shift_sigma = Float64[
            Float64(row[!, Symbol(name)])
            for name in sigma_names
        ]

        perturbations[row_index] = SBPerturbation(
            Int(row.profile_id),
            light_target,
            delta_light,
            shift_sigma,
            Float64(row.total_light),
            Float64(row.total_light_ratio),
            Float64(row.total_shift_sigma),
            Float64(row.max_abs_bin_shift_sigma),
            Int(row.n_changed_bins),
        )
    end

    sort!(perturbations, by=p -> p.profile_id)

    return perturbations
end


function get_light_target(
    perturbation::SBPerturbation,
)
    return copy(perturbation.light_target)
end


"""
    run_sb_perturbations(
        perturbations,
        light_target_nominal,
        evaluate_target;
        include_nominal=false,
        verify_nominal=true,
        nominal_rtol=1e-10,
        nominal_atol=1e-12,
    )

Loop over already-generated surface-brightness targets.

Arguments
---------
perturbations:
    Output from `load_sb_perturbations`.

light_target_nominal:
    The nominal light target already used by OSPM for this theta.

evaluate_target:
    Callback with signature:

        evaluate_target(light_target_new, perturbation)

    This is intentionally generic. Later OSPM can place the existing
    weight-only solve here while reusing:
        A_light
        A_losvd
        orbit library
        nominal weights

Keywords
--------
include_nominal:
    If false, profile_id == 0 is skipped.

verify_nominal:
    If true, profile_id == 0 must match light_target_nominal.

Returns
-------
A vector containing whatever each callback invocation returns.
"""
function run_sb_perturbations(
    perturbations::Vector{SBPerturbation},
    light_target_nominal::AbstractVector{<:Real},
    evaluate_target::Function;
    include_nominal::Bool=false,
    verify_nominal::Bool=true,
    nominal_rtol::Float64=1e-10,
    nominal_atol::Float64=1e-12,
)
    isempty(perturbations) &&
        error("No surface-brightness perturbations were supplied.")

    nominal = Float64.(light_target_nominal)
    nbins = length(nominal)

    nbins > 0 ||
        error("Nominal light target is empty.")

    for perturbation in perturbations
        length(perturbation.light_target) == nbins ||
            error(
                "Profile $(perturbation.profile_id) has " *
                "$(length(perturbation.light_target)) bins; expected $nbins."
            )
    end

    if verify_nominal
        nominal_rows = filter(p -> p.profile_id == 0, perturbations)

        length(nominal_rows) == 1 ||
            error(
                "Expected exactly one nominal profile with profile_id == 0; " *
                "found $(length(nominal_rows))."
            )

        nominal_from_deck = nominal_rows[1].light_target

        isapprox(
            nominal_from_deck,
            nominal;
            rtol=nominal_rtol,
            atol=nominal_atol,
        ) || error(
            "Nominal light target in perturbation deck does not match " *
            "the nominal light target supplied by OSPM."
        )
    end

    results = Any[]

    for perturbation in perturbations
        if !include_nominal && perturbation.profile_id == 0
            continue
        end

        light_target_new = get_light_target(perturbation)

        result = evaluate_target(
            light_target_new,
            perturbation,
        )

        push!(results, result)
    end

    return results
end


end # module


# ---------------------------------------------------------------------------
# Standalone example
# ---------------------------------------------------------------------------
#
# Uncomment and adapt this later inside OSPM.
#
# using .SurfaceBrightnessPerturbations
#
# perturbations = load_sb_perturbations(
#     "surface_brightness_perturbations.csv";
#     expected_nbins=length(light_target),
# )
#
# results = run_sb_perturbations(
#     perturbations,
#     light_target,
# ) do light_target_new, perturbation
#
#     println(
#         "profile_id=$(perturbation.profile_id) " *
#         "total_shift_sigma=$(perturbation.total_shift_sigma) " *
#         "max_bin_sigma=$(perturbation.max_abs_bin_shift_sigma)"
#     )
#
#     # Later, this is where the existing weight solver goes.
#     #
#     # Example shape:
#     #
#     # w_trial = copy(w_nominal)
#     #
#     # solve_weights!(
#     #     w_trial,
#     #     A_light,
#     #     A_losvd,
#     #     light_target_new,
#     #     losvd_target,
#     # )
#     #
#     # return (
#     #     profile_id=perturbation.profile_id,
#     #     weights=w_trial,
#     # )
#
#     return (
#         profile_id=perturbation.profile_id,
#         light_target=light_target_new,
#     )
# end
'''    