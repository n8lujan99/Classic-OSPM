#!/usr/bin/env python3
"""
Generic surface-brightness / projected-density preparation for OSPM.

This utility converts a published radial projected-density profile into a
standardized OSPM surface-brightness CSV.

It does NOT repair suspicious data automatically.

Instead it preserves the prepared measurements, runs profile-quality checks,
records review flags in the output, writes a human-readable review report,
and prints a prominent warning whenever the profile should be inspected
before downstream use.

Galaxy-specific numbers do not belong in this file.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REVIEW_LOW_SN_SIGMA = 2.0
REVIEW_RISE_SIGMA = 2.0
REVIEW_OUTER_DETECTION_SIGMA = 3.0
RADIAL_EDGE_RTOL = 1e-10
RADIAL_EDGE_ATOL = 1e-12


def build_profile(source_df, *, galaxy, source, preferred_profile, radius_type, pc_per_arcmin, q_axis_ratio, background, background_err,
                  rin_col, rout_col, rmid_col, sigma_col, sigma_err_col, sigma_units, note=None):
    required = {rin_col, rout_col, rmid_col, sigma_col, sigma_err_col}
    missing = required - set(source_df.columns)
    if missing:
        raise KeyError(f"Source profile missing columns: {sorted(missing)}")

    if not math.isfinite(pc_per_arcmin) or pc_per_arcmin <= 0.0:
        raise ValueError("pc_per_arcmin must be positive and finite.")
    if not math.isfinite(q_axis_ratio) or q_axis_ratio <= 0.0:
        raise ValueError("q_axis_ratio must be positive and finite.")
    if not math.isfinite(background):
        raise ValueError("background must be finite.")
    if not math.isfinite(background_err) or background_err < 0.0:
        raise ValueError("background_err must be finite and non-negative.")

    ellipticity = 1.0 - q_axis_ratio
    rows = []

    for i, row in source_df.iterrows():
        rin = float(row[rin_col]); rout = float(row[rout_col]); rmid = float(row[rmid_col])
        sigma_source = float(row[sigma_col]); sigma_source_err = float(row[sigma_err_col])

        if not all(math.isfinite(x) for x in (rin, rout, rmid, sigma_source, sigma_source_err)):
            raise ValueError(f"Non-finite source value in row {i}.")
        if rin < 0.0 or rout <= rin or rmid <= 0.0:
            raise ValueError(f"Invalid radial bin in row {i}: rin={rin}, rout={rout}, rmid={rmid}")
        if sigma_source_err < 0.0:
            raise ValueError(f"Negative projected-density uncertainty in row {i}: {sigma_source_err}")

        sigma = sigma_source - background
        sigma_err = math.hypot(sigma_source_err, background_err)
        area_arcmin2 = math.pi * q_axis_ratio * (rout**2 - rin**2)
        area_pc2 = area_arcmin2 * pc_per_arcmin**2
        light_raw = sigma * area_arcmin2
        light_raw_err = sigma_err * area_arcmin2
        sigma_significance = sigma / sigma_err if sigma_err > 0.0 else np.inf

        rows.append({
            "galaxy": galaxy, "source": source, "preferred_profile": preferred_profile, "radius_type": radius_type,
            "rin_arcmin": rin, "rout_arcmin": rout, "rm_arcmin": rmid,
            "R_inner_pc": rin * pc_per_arcmin, "R_outer_pc": rout * pc_per_arcmin, "R_pc": rmid * pc_per_arcmin,
            "Sigma_source_arcmin2": sigma_source, "Sigma_source_err_arcmin2": sigma_source_err,
            "Sigma_background_arcmin2": background, "Sigma_background_err_arcmin2": background_err,
            "Sigma": sigma, "Sigma_err": sigma_err, "Sigma_pc2": sigma / pc_per_arcmin**2, "Sigma_err_pc2": sigma_err / pc_per_arcmin**2,
            "Sigma_significance": sigma_significance,
            "light_raw": light_raw, "light_raw_err": light_raw_err,
            "ellipticity": ellipticity, "q_axis_ratio": q_axis_ratio, "area_arcmin2": area_arcmin2, "area_pc2": area_pc2,
            "Sigma_units": sigma_units, "R_units": "pc", "area_model": "elliptical_annulus_pi_q_delta_a2",
            "pc_per_arcmin_assumed": pc_per_arcmin,
            "note": note or (
                f"Production projected-density profile for {galaxy}. A fitted background of {background:.12g} +/- "
                f"{background_err:.12g} in the source Sigma units was subtracted before normalization. Sigma_err combines "
                f"the source statistical uncertainty and fitted-background uncertainty in quadrature."
            ),
        })

    if not rows:
        raise ValueError("Source profile produced zero valid rows.")

    profile = pd.DataFrame(rows).sort_values("R_pc", kind="stable").reset_index(drop=True)
    return profile


def diagnose_profile(profile):
    flags = [[] for _ in range(len(profile))]
    summary = []

    R = profile["R_pc"].to_numpy(float)
    rin = profile["rin_arcmin"].to_numpy(float); rout = profile["rout_arcmin"].to_numpy(float)
    sigma = profile["Sigma"].to_numpy(float); sigma_err = profile["Sigma_err"].to_numpy(float)
    significance = profile["Sigma_significance"].to_numpy(float)

    nonpositive = np.where(sigma <= 0.0)[0]
    for i in nonpositive:
        flags[i].append("NONPOSITIVE_CORRECTED_SIGMA")
    if len(nonpositive):
        summary.append(
            f"{len(nonpositive)} bin(s) have Sigma <= 0 after background subtraction. "
            "These measurements are retained, but the profile is not ready for positive-density normalization or Abel input."
        )

    low_sn = np.where((sigma > 0.0) & (significance < REVIEW_LOW_SN_SIGMA))[0]
    for i in low_sn:
        flags[i].append("LOW_SIGNAL_TO_NOISE")
    if len(low_sn):
        summary.append(
            f"{len(low_sn)} positive bin(s) are less than {REVIEW_LOW_SN_SIGMA:.1f} sigma above the adopted background."
        )

    rise_significance = np.full(len(profile), np.nan)
    for i in range(1, len(profile)):
        denom = math.hypot(sigma_err[i - 1], sigma_err[i])
        rise_significance[i] = (sigma[i] - sigma[i - 1]) / denom if denom > 0.0 else np.nan
        if np.isfinite(rise_significance[i]) and rise_significance[i] >= REVIEW_RISE_SIGMA:
            flags[i].append("SIGNIFICANT_OUTWARD_RISE")
            summary.append(
                f"Outward rise between R={R[i - 1]:.6g} and {R[i]:.6g} pc is "
                f"{rise_significance[i]:.2f} sigma."
            )
        elif sigma[i] > sigma[i - 1]:
            flags[i].append("OUTWARD_RISE")

    gap_count = 0; overlap_count = 0
    for i in range(1, len(profile)):
        if np.isclose(rin[i], rout[i - 1], rtol=RADIAL_EDGE_RTOL, atol=RADIAL_EDGE_ATOL):
            continue
        if rin[i] > rout[i - 1]:
            gap_count += 1; flags[i].append("RADIAL_GAP_BEFORE_BIN")
        else:
            overlap_count += 1; flags[i].append("RADIAL_OVERLAP_BEFORE_BIN")

    if gap_count:
        summary.append(f"{gap_count} gap(s) exist between adjacent radial annuli.")
    if overlap_count:
        summary.append(f"{overlap_count} overlap(s) exist between adjacent radial annuli.")

    outer_significance = float(significance[-1])
    if np.isfinite(outer_significance) and outer_significance >= REVIEW_OUTER_DETECTION_SIGMA:
        flags[-1].append("OUTER_PROFILE_STILL_DETECTED")
        summary.append(
            f"The outermost bin remains {outer_significance:.2f} sigma above the adopted background. "
            "The measured profile may end before the tracer distribution has reached the background."
        )

    profile["outward_rise_significance"] = rise_significance
    profile["review_flags"] = [";".join(x) for x in flags]
    profile["bin_review_required"] = [bool(x) for x in flags]

    review_required = bool(summary)
    profile["profile_review_status"] = "REVIEW_REQUIRED" if review_required else "PASS"
    profile["profile_review_required"] = review_required

    return profile, summary


def normalize_light(profile):
    if np.any(profile["Sigma"].to_numpy(float) <= 0.0):
        profile["light_frac"] = np.nan
        profile["light_frac_err_approx"] = np.nan
        return profile, False

    light_total = float(profile["light_raw"].sum())
    if not math.isfinite(light_total) or light_total <= 0.0:
        raise ValueError("Corrected total light/tracer proxy is not positive.")

    profile["light_frac"] = profile["light_raw"] / light_total
    profile["light_frac_err_approx"] = profile["light_raw_err"] / light_total

    light_frac_sum = float(profile["light_frac"].sum())
    if not math.isclose(light_frac_sum, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"light_frac does not sum to unity: {light_frac_sum:.16g}")

    return profile, True


def write_review_report(profile, summary, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    status = str(profile["profile_review_status"].iloc[0])

    lines = [
        "SURFACE-BRIGHTNESS PREPARATION REVIEW",
        "=====================================",
        f"Galaxy: {profile['galaxy'].iloc[0]}",
        f"Source: {profile['source'].iloc[0]}",
        f"Profile: {profile['preferred_profile'].iloc[0]}",
        f"Status: {status}",
        "",
    ]

    if summary:
        lines.append("REVIEW REASONS")
        lines.append("--------------")
        for item in summary:
            lines.append(f"- {item}")
    else:
        lines.append("No automatic review conditions were triggered.")

    flagged = profile.loc[profile["bin_review_required"]]
    if len(flagged):
        lines.extend(["", "FLAGGED BINS", "------------"])
        for _, row in flagged.iterrows():
            lines.append(
                f"R={row['R_pc']:.6g} pc  Sigma={row['Sigma']:.6g} +/- {row['Sigma_err']:.6g}  "
                f"significance={row['Sigma_significance']:.3f}  flags={row['review_flags']}"
            )

    lines.extend([
        "",
        "No suspicious measurements were modified automatically.",
        "A REVIEW_REQUIRED status means the profile should be inspected before downstream modeling or deprojection.",
    ])

    report_path.write_text("\n".join(lines) + "\n")


def make_plots(profile, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    correction_plot = outdir / f"{stem}_background_correction.png"
    abel_input_plot = outdir / f"{stem}_abel_input.png"

    R_pc = profile["R_pc"].to_numpy(float)
    sigma_source = profile["Sigma_source_arcmin2"].to_numpy(float); sigma_source_err = profile["Sigma_source_err_arcmin2"].to_numpy(float)
    sigma = profile["Sigma"].to_numpy(float); sigma_err = profile["Sigma_err"].to_numpy(float)

    background = float(profile["Sigma_background_arcmin2"].iloc[0])
    galaxy = str(profile["galaxy"].iloc[0]); source = str(profile["source"].iloc[0])
    preferred_profile = str(profile["preferred_profile"].iloc[0]); sigma_units = str(profile["Sigma_units"].iloc[0])
    review_required = bool(profile["profile_review_required"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(R_pc, sigma_source, yerr=sigma_source_err, fmt="o", capsize=3, label=f"{source} {preferred_profile}".strip())
    ax.errorbar(R_pc, sigma, yerr=sigma_err, fmt="o", capsize=3, label="Background-subtracted profile")
    if background > 0.0:
        ax.axhline(background, linestyle="--", linewidth=1.2, label="Adopted fitted background")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Projected radius R [pc]"); ax.set_ylabel(f"Projected density [{sigma_units}]")
    ax.set_title(f"{galaxy} Surface-Brightness Background Correction"); ax.legend(); ax.grid(alpha=0.25)
    if review_required:
        ax.text(0.02, 0.02, "REVIEW REQUIRED", transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(correction_plot, dpi=220, bbox_inches="tight"); plt.close(fig)

    positive = sigma > 0.0
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(R_pc[positive], sigma[positive], yerr=sigma_err[positive], fmt="o", capsize=3, label="Positive Abel-input measurements")
    if np.any(~positive):
        ax.errorbar(R_pc[~positive], np.maximum(sigma_err[~positive], np.finfo(float).tiny), yerr=sigma_err[~positive],
                    fmt="x", capsize=3, label="Non-positive corrected measurement")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Projected radius R [pc]"); ax.set_ylabel(f"Projected density [{sigma_units}]")
    ax.set_title(f"{galaxy} Prepared Surface-Brightness Profile"); ax.legend(); ax.grid(alpha=0.25)
    if review_required:
        ax.text(0.02, 0.02, "REVIEW REQUIRED BEFORE ABEL USE", transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(abel_input_plot, dpi=220, bbox_inches="tight"); plt.close(fig)

    return correction_plot, abel_input_plot


def build_parser():
    p = argparse.ArgumentParser(description="Prepare and quality-check a published projected-density profile for OSPM.")

    p.add_argument("--input", required=True, help="Source radial-profile CSV.")
    p.add_argument("--out", required=True, help="Output prepared CSV.")
    p.add_argument("--galaxy", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--preferred-profile", default="")
    p.add_argument("--radius-type", default="elliptical_major_axis")
    p.add_argument("--pc-per-arcmin", type=float, required=True)

    shape = p.add_mutually_exclusive_group()
    shape.add_argument("--q-axis-ratio", type=float, default=None)
    shape.add_argument("--ellipticity", type=float, default=None)

    p.add_argument("--background", type=float, default=0.0)
    p.add_argument("--background-err", type=float, default=0.0)
    p.add_argument("--rin-col", default="rin_arcmin")
    p.add_argument("--rout-col", default="rout_arcmin")
    p.add_argument("--rmid-col", default="rm_arcmin")
    p.add_argument("--sigma-col", default="Sigma")
    p.add_argument("--sigma-err-col", default="Sigma_err")
    p.add_argument("--sigma-units", default="stars_per_arcmin2")
    p.add_argument("--plots-dir", default=None)
    p.add_argument("--plot-stem", default=None)
    p.add_argument("--review-report", default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--note", default=None)

    return p


def main():
    args = build_parser().parse_args()

    input_path = Path(args.input); out_path = Path(args.out)
    source_df = pd.read_csv(input_path)

    if args.q_axis_ratio is not None:
        q_axis_ratio = float(args.q_axis_ratio)
    elif args.ellipticity is not None:
        q_axis_ratio = 1.0 - float(args.ellipticity)
    else:
        q_axis_ratio = 1.0

    profile = build_profile(
        source_df, galaxy=args.galaxy, source=args.source, preferred_profile=args.preferred_profile, radius_type=args.radius_type,
        pc_per_arcmin=float(args.pc_per_arcmin), q_axis_ratio=q_axis_ratio, background=float(args.background),
        background_err=float(args.background_err), rin_col=args.rin_col, rout_col=args.rout_col, rmid_col=args.rmid_col,
        sigma_col=args.sigma_col, sigma_err_col=args.sigma_err_col, sigma_units=args.sigma_units, note=args.note,
    )

    profile, review_summary = diagnose_profile(profile)
    profile, normalized = normalize_light(profile)

    columns = [
        "galaxy", "source", "preferred_profile", "radius_type",
        "rin_arcmin", "rout_arcmin", "rm_arcmin", "R_inner_pc", "R_outer_pc", "R_pc",
        "Sigma_source_arcmin2", "Sigma_source_err_arcmin2", "Sigma_background_arcmin2", "Sigma_background_err_arcmin2",
        "Sigma", "Sigma_err", "Sigma_pc2", "Sigma_err_pc2", "Sigma_significance", "outward_rise_significance",
        "light_raw", "light_raw_err", "light_frac", "light_frac_err_approx",
        "ellipticity", "q_axis_ratio", "area_arcmin2", "area_pc2",
        "bin_review_required", "review_flags", "profile_review_required", "profile_review_status",
        "Sigma_units", "R_units", "area_model", "pc_per_arcmin_assumed", "note",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(out_path, columns=columns, index=False)

    report_path = Path(args.review_report) if args.review_report is not None else out_path.with_suffix(".review.txt")
    write_review_report(profile, review_summary, report_path)

    correction_plot = None; abel_input_plot = None
    if not args.no_plots:
        plots_dir = Path(args.plots_dir) if args.plots_dir is not None else out_path.parent / "plots"
        stem = args.plot_stem if args.plot_stem is not None else f"{args.galaxy}_surface_brightness"
        correction_plot, abel_input_plot = make_plots(profile, plots_dir, stem)

    print(out_path)
    print(f"Rows written: {len(profile)}")
    print(f"Galaxy: {args.galaxy}")
    print(f"Source: {args.source}")
    print(f"Preferred profile: {args.preferred_profile}")
    print(f"q_axis_ratio: {q_axis_ratio:.12g}")
    print(f"Adopted background: {float(args.background):.12g} +/- {float(args.background_err):.12g} [{args.sigma_units}]")
    print(f"Outermost corrected Sigma: {profile['Sigma'].iloc[-1]:.12g} +/- {profile['Sigma_err'].iloc[-1]:.12g}")
    print(f"Review report: {report_path}")

    print()
    print("=" * 72)
    print(f"PROFILE STATUS: {profile['profile_review_status'].iloc[0]}")
    print("=" * 72)

    if review_summary:
        for item in review_summary:
            print(f"WARNING: {item}")
        print()
        print("No suspicious measurements were modified automatically.")
        print("Review the profile and report before downstream modeling or Abel deprojection.")
    else:
        print("No automatic review conditions were triggered.")

    if normalized:
        print(f"light_frac sum: {profile['light_frac'].sum():.12f}")
    else:
        print("light_frac: NOT GENERATED because one or more corrected Sigma values are non-positive.")

    if correction_plot is not None:
        print(correction_plot)
    if abel_input_plot is not None:
        print(abel_input_plot)


if __name__ == "__main__":
    main()
