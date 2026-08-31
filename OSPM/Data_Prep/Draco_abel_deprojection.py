"""
Draco surface-density deprojection following the non-parametric Abel
procedure used by Gebhardt et al. (1996).
Active path:
    observed Sigma(R) +/- Sigma_err in physical pc units
        -> weighted smoothing spline in log R, log Sigma
        -> analytic spline derivative
        -> data-driven outer transition
        -> monotonic blend of d ln Sigma / d ln R
        -> outer power-law continuation
        -> inverse Abel transform
        -> nu(r) for r >= R_first
Adopted Draco choices from the validation tests:
    smoothing target s = 9
        Preserves the data-supported 40-100 pc flattening while avoiding
        the non-robust small 3D density rise produced by stronger smoothing.
    outer transition begins at the last high-significance measurement
        The default threshold is Sigma/Sigma_err >= 5. For Draco this selects
        R = 556.478 pc, after which the remaining measurements become
        progressively low-S/N.
    transition is applied to the logarithmic slope, not directly to Sigma
        d ln Sigma / d ln R moves smoothly and monotonically from the spline
        slope at the transition start to the fitted outer power-law slope at
        the final measured radius. This avoids both an abrupt slope step and
        the derivative overshoot produced by directly blending two profiles.
    outer power-law slope
        Fitted from the outer measurements with their uncertainties and used
        beyond the final measured radius so the Abel integral can extend to
        infinity.
No surface-density continuation is introduced interior to the first measured
radius. The inverse Abel transform at radius r only uses the projected profile
at R >= r.
No clipping or repair of the recovered density is performed. Non-finite or
non-positive Abel densities fail visibly.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.special import gamma


DEFAULT_SMOOTHING_TARGET = 9.0
DEFAULT_OUTER_TAIL_POINTS = 6
DEFAULT_OUTER_TRANSITION_SIGMA = 5.0
DEFAULT_N_GRID = 256
FLAT_REGION_PC = (40.0, 100.0)


def load_surface_brightness(path, radius_col="R_pc", sigma_col="Sigma_pc2", sigma_err_col="Sigma_err_pc2"):
    sb = pd.read_csv(path)
    required = {radius_col, sigma_col, sigma_err_col}
    missing = required - set(sb.columns)
    if missing:
        raise KeyError(f"surface-brightness CSV missing columns: {sorted(missing)}")
    R = sb[radius_col].to_numpy(float)
    Sigma = sb[sigma_col].to_numpy(float)
    Sigma_err = sb[sigma_err_col].to_numpy(float)
    good = np.isfinite(R) & np.isfinite(Sigma) & np.isfinite(Sigma_err) & (R > 0.0) & (Sigma > 0.0) & (Sigma_err > 0.0)
    R = R[good]; Sigma = Sigma[good]; Sigma_err = Sigma_err[good]
    if len(R) < 5:
        raise ValueError("Need at least five positive finite surface-brightness measurements.")
    order = np.argsort(R)
    return R[order], Sigma[order], Sigma_err[order]


def fit_weighted_log_spline(R, Sigma, Sigma_err, smoothing_target=DEFAULT_SMOOTHING_TARGET):
    x = np.log(np.asarray(R, float))
    y = np.log(np.asarray(Sigma, float))
    sigma_logSigma = np.asarray(Sigma_err, float) / np.asarray(Sigma, float)
    weights = 1.0 / sigma_logSigma
    spline = UnivariateSpline(x, y, w=weights, k=3, s=float(smoothing_target))
    return spline, sigma_logSigma, float(smoothing_target)


def evaluate_log_spline_profile(spline, R_eval):
    R_eval = np.asarray(R_eval, float)
    x = np.log(R_eval)
    logSigma = spline(x)
    Sigma = np.exp(logSigma)
    dlogSigma_dlogR = spline.derivative(1)(x)
    dSigma_dR = Sigma / R_eval * dlogSigma_dlogR
    return Sigma, dSigma_dR, dlogSigma_dlogR


def fit_outer_powerlaw_slope(R, Sigma, Sigma_err, n_tail=DEFAULT_OUTER_TAIL_POINTS):
    n_tail = int(n_tail)
    if n_tail < 3:
        raise ValueError("outer-tail fit requires at least three points")
    if n_tail > len(R):
        raise ValueError(f"outer-tail fit requested {n_tail} points but only {len(R)} are available")
    x = np.log(np.asarray(R[-n_tail:], float))
    y = np.log(np.asarray(Sigma[-n_tail:], float))
    sigma_y = np.asarray(Sigma_err[-n_tail:], float) / np.asarray(Sigma[-n_tail:], float)
    weights = 1.0 / sigma_y
    coeff, cov = np.polyfit(x, y, 1, w=weights, cov=True)
    slope = float(coeff[0])
    slope_err = float(np.sqrt(cov[0, 0]))
    if slope >= -1.0:
        raise ValueError(f"outer projected slope must be < -1 for Abel convergence; got {slope:.8f}")
    return slope, slope_err


def choose_outer_transition_start(R, Sigma, Sigma_err, significance_threshold=DEFAULT_OUTER_TRANSITION_SIGMA):
    significance = np.asarray(Sigma, float) / np.asarray(Sigma_err, float)
    secure = np.where(significance >= float(significance_threshold))[0]
    if len(secure) == 0:
        raise ValueError(f"No surface-brightness measurement reaches {significance_threshold:.3f} sigma.")
    start_index = int(secure[-1])
    if start_index >= len(R) - 1:
        raise ValueError("Outermost measurement still exceeds the outer-transition significance threshold; there is no low-S/N outer interval in which to apply the transition.")
    return float(R[start_index]), start_index, significance


def smootherstep(t):
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def integrated_smootherstep(t):
    return t**6 - 3.0 * t**5 + 2.5 * t**4


def build_outer_transition_evaluator(spline, R_start, R_last, outer_slope):
    x_start = np.log(float(R_start)); x_last = np.log(float(R_last))
    y_start = float(spline(x_start)); slope_start = float(spline.derivative(1)(x_start))
    delta_x = x_last - x_start; delta_slope = float(outer_slope) - slope_start
    if delta_x <= 0.0:
        raise ValueError("outer transition requires R_start < R_last")
    y_last = y_start + delta_x * (slope_start + 0.5 * delta_slope)
    Sigma_last = float(np.exp(y_last))
    def evaluate(R_eval):
        R_eval = np.asarray(R_eval, float)
        if np.any(R_eval <= 0.0):
            raise ValueError("profile evaluation radii must be positive")
        x = np.log(R_eval)
        y = np.empty_like(R_eval)
        slope = np.empty_like(R_eval)
        inner = x <= x_start
        outer = x >= x_last
        blend = ~(inner | outer)
        if np.any(inner):
            y[inner] = spline(x[inner])
            slope[inner] = spline.derivative(1)(x[inner])
        if np.any(blend):
            t = (x[blend] - x_start) / delta_x
            slope[blend] = slope_start + delta_slope * smootherstep(t)
            y[blend] = y_start + delta_x * (slope_start * t + delta_slope * integrated_smootherstep(t))
        if np.any(outer):
            y[outer] = y_last + float(outer_slope) * (x[outer] - x_last)
            slope[outer] = float(outer_slope)
        Sigma = np.exp(y)
        dSigma_dR = Sigma / R_eval * slope
        return Sigma, dSigma_dR, slope
    return evaluate, slope_start, Sigma_last


def abel_deproject_spherical(R_eval, evaluate_profile):
    R_eval = np.asarray(R_eval, float)
    if np.any(R_eval <= 0.0) or not np.all(np.diff(R_eval) > 0.0):
        raise ValueError("R_eval must be positive and strictly increasing")
    nu = np.full_like(R_eval, np.nan)
    quad_error = np.full_like(R_eval, np.nan)
    for i, r in enumerate(R_eval):
        def integrand(u):
            Rp = np.hypot(float(r), u)
            _, dSigma_dR, _ = evaluate_profile(np.array([Rp]))
            return -float(dSigma_dR[0]) / Rp / np.pi
        nu[i], quad_error[i] = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1e-8, limit=300)
    if np.any(~np.isfinite(nu)):
        raise ValueError(f"Abel deprojection produced non-finite density at radii: {R_eval[~np.isfinite(nu)]}")
    if np.any(nu <= 0.0):
        raise ValueError(f"Abel deprojection produced non-positive density at radii: {R_eval[nu <= 0.0]}")
    return nu, quad_error


def projected_powerlaw_tail_nu(r, R_last, Sigma_last, outer_slope):
    p = -float(outer_slope)
    if p <= 1.0:
        raise ValueError("projected power-law tail requires p > 1")
    coefficient = gamma((p + 1.0) / 2.0) / (np.sqrt(np.pi) * gamma(p / 2.0))
    nu_last = coefficient * float(Sigma_last) / float(R_last)
    return nu_last * (np.asarray(r, float) / float(R_last))**(-p - 1.0)


def load_unresolved_center_inputs(path):
    sb = pd.read_csv(path)
    required = {"R_inner_pc", "R_outer_pc", "light_raw", "q_axis_ratio"}
    missing = required - set(sb.columns)
    if missing:
        raise KeyError(f"surface-brightness CSV missing unresolved-center columns: {sorted(missing)}")
    if len(sb) == 0:
        raise ValueError("surface-brightness CSV is empty")
    first = sb.iloc[0]
    R_inner = float(first["R_inner_pc"])
    R_outer = float(first["R_outer_pc"])
    raw_light = float(first["light_raw"])
    q = float(first["q_axis_ratio"])
    if not np.isfinite(R_inner) or abs(R_inner) > 1e-12:
        raise ValueError(f"first observed annulus must begin at R=0 pc; got {R_inner}")
    if not np.isfinite(R_outer) or R_outer <= 0.0:
        raise ValueError("first observed annulus outer radius must be positive and finite")
    if not np.isfinite(raw_light) or raw_light <= 0.0:
        raise ValueError("first observed annulus raw light must be positive and finite")
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("q_axis_ratio must be positive and finite")
    return R_outer, raw_light, q


def unresolved_center_light_budget(R_nu, nu, R_last, Sigma_last, outer_slope, first_annulus_outer_pc, first_annulus_raw_light, q_axis_ratio):
    R_nu = np.asarray(R_nu, float)
    nu = np.asarray(nu, float)
    support_min_pc = float(R_nu[0])
    first_annulus_outer_pc = float(first_annulus_outer_pc)
    first_annulus_raw_light = float(first_annulus_raw_light)
    q_axis_ratio = float(q_axis_ratio)
    nu_spline = UnivariateSpline(np.log(R_nu), np.log(nu), k=3, s=0.0)
    def nu_function(r):
        if r <= R_last:
            return float(np.exp(nu_spline(np.log(r))))
        return float(projected_powerlaw_tail_nu(r, R_last, Sigma_last, outer_slope))
    def resolved_sigma(R_project):
        z_min = np.sqrt(max(support_min_pc**2 - float(R_project)**2, 0.0))
        def integrand(z):
            return 2.0 * nu_function(np.hypot(float(R_project), z))
        value, _ = quad(integrand, z_min, np.inf, epsabs=0.0, epsrel=1e-8, limit=300)
        return value
    def annulus_integrand(R_project):
        return 2.0 * np.pi * q_axis_ratio * float(R_project) * resolved_sigma(R_project)
    resolved_raw_light, _ = quad(annulus_integrand, 0.0, first_annulus_outer_pc, epsabs=0.0, epsrel=1e-8, limit=300)
    unresolved_raw_light = first_annulus_raw_light - resolved_raw_light
    if not np.isfinite(unresolved_raw_light) or unresolved_raw_light <= 0.0:
        raise ValueError(f"unresolved central raw light must be positive and finite; got {unresolved_raw_light}")
    boundary_nu_pc3 = float(nu[0])
    inner_cslope = 4.0 * np.pi * q_axis_ratio * boundary_nu_pc3 * support_min_pc**3 / unresolved_raw_light - 3.0
    if not np.isfinite(inner_cslope) or inner_cslope <= -3.0:
        raise ValueError(f"derived Karl-style inner cslope must be finite and > -3; got {inner_cslope}")
    reconstructed_raw_light = 4.0 * np.pi * q_axis_ratio * boundary_nu_pc3 * support_min_pc**3 / (3.0 + inner_cslope)
    reconstruction_residual = reconstructed_raw_light - unresolved_raw_light
    return {
        "abel_support_min_pc": support_min_pc,
        "unresolved_center_outer_pc": support_min_pc,
        "first_annulus_outer_pc": first_annulus_outer_pc,
        "first_annulus_raw_light": first_annulus_raw_light,
        "resolved_first_annulus_raw_light": float(resolved_raw_light),
        "unresolved_center_raw_light": float(unresolved_raw_light),
        "unresolved_center_first_annulus_fraction": float(unresolved_raw_light / first_annulus_raw_light),
        "inner_continuation_model": "karl_powerlaw",
        "inner_cslope": float(inner_cslope),
        "inner_boundary_nu_pc3": boundary_nu_pc3,
        "inner_continuation_raw_light": float(reconstructed_raw_light),
        "inner_continuation_light_residual": float(reconstruction_residual),
    }


def forward_project_density(R_project, R_nu, nu, R_last, Sigma_last, outer_slope):
    R_project = np.asarray(R_project, float)
    R_nu = np.asarray(R_nu, float)
    nu = np.asarray(nu, float)
    if np.any(R_project < R_nu[0]):
        raise ValueError("forward projection requested below the innermost deprojected radius")
    nu_spline = UnivariateSpline(np.log(R_nu), np.log(nu), k=3, s=0.0)
    def nu_function(r):
        if r <= R_last:
            return float(np.exp(nu_spline(np.log(r))))
        return float(projected_powerlaw_tail_nu(r, R_last, Sigma_last, outer_slope))
    Sigma_forward = np.full_like(R_project, np.nan)
    for i, R in enumerate(R_project):
        def integrand(z):
            return 2.0 * nu_function(np.hypot(float(R), z))
        Sigma_forward[i], _ = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1e-7, limit=300)
    return Sigma_forward

def plummer_dSigma_dR(R, L=1.0, a=200.0):
    R = np.asarray(R, float)
    return -4.0 * L * R / (np.pi * a**4) * (1.0 + R**2 / a**2)**(-3.0)

def plummer_nu(r, L=1.0, a=200.0):
    r = np.asarray(r, float)
    return 3.0 * L / (4.0 * np.pi * a**3) * (1.0 + r**2 / a**2)**(-2.5)

def validate_abel_integral_with_plummer():
    a = 200.0
    R = np.geomspace(0.05 * a, 10.0 * a, 128)
    nu_numeric = np.full_like(R, np.nan)
    for i, r in enumerate(R):
        def integrand(u):
            Rp = np.hypot(float(r), u)
            return -float(plummer_dSigma_dR(Rp, a=a)) / Rp / np.pi
        nu_numeric[i], _ = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1e-10, limit=300)
    nu_exact = plummer_nu(R, a=a)
    relative_error = np.abs((nu_numeric - nu_exact) / nu_exact)
    return float(np.median(relative_error)), float(np.max(relative_error))


def build_draco_deprojection(R, Sigma, Sigma_err, n_grid=DEFAULT_N_GRID, smoothing_target=DEFAULT_SMOOTHING_TARGET, outer_tail_points=DEFAULT_OUTER_TAIL_POINTS, outer_transition_sigma=DEFAULT_OUTER_TRANSITION_SIGMA):
    spline, sigma_logSigma, smoothing_target = fit_weighted_log_spline(R, Sigma, Sigma_err, smoothing_target=smoothing_target)
    outer_slope, outer_slope_err = fit_outer_powerlaw_slope(R, Sigma, Sigma_err, n_tail=outer_tail_points)
    R_transition, transition_index, significance = choose_outer_transition_start(R, Sigma, Sigma_err, significance_threshold=outer_transition_sigma)
    R_last = float(R[-1])
    evaluate_profile, transition_slope, Sigma_last = build_outer_transition_evaluator(spline, R_transition, R_last, outer_slope)
    R_grid = np.geomspace(R[0], R_last, int(n_grid))
    Sigma_grid, dSigma_dR, dlogSigma_dlogR = evaluate_profile(R_grid)
    Sigma_fit_obs, _, _ = evaluate_profile(R)
    residual_sigma = (Sigma - Sigma_fit_obs) / Sigma_err
    chi2_linear = float(np.sum(residual_sigma**2))
    log_residual_sigma = (np.log(Sigma) - np.log(Sigma_fit_obs)) / sigma_logSigma
    chi2_log = float(np.sum(log_residual_sigma**2))
    nu, quad_error = abel_deproject_spherical(R_grid, evaluate_profile)
    dlognu_dlogr = np.gradient(np.log(nu), np.log(R_grid))
    flat = (R_grid >= FLAT_REGION_PC[0]) & (R_grid <= FLAT_REGION_PC[1])

    return {
        "R_grid": R_grid, "Sigma_grid": Sigma_grid, "dSigma_dR": dSigma_dR, "dlogSigma_dlogR": dlogSigma_dlogR,
        "nu": nu, "dlognu_dlogr": dlognu_dlogr, "quad_error": quad_error, "spline": spline,
        "Sigma_fit_obs": Sigma_fit_obs, "residual_sigma": residual_sigma, "chi2_linear": chi2_linear, "chi2_log": chi2_log,
        "smoothing_target": smoothing_target, "outer_slope": outer_slope, "outer_slope_err": outer_slope_err,
        "outer_tail_points": int(outer_tail_points), "outer_transition_sigma": float(outer_transition_sigma),
        "R_transition": R_transition, "transition_index": transition_index, "transition_slope": transition_slope,
        "transition_significance": float(significance[transition_index]),
        "next_significance": float(significance[transition_index + 1]), "R_last": R_last, "Sigma_last": Sigma_last,
        "flat_median_slope": float(np.median(dlognu_dlogr[flat])),
        "flat_min_slope": float(np.min(dlognu_dlogr[flat])),
        "flat_max_slope": float(np.max(dlognu_dlogr[flat])),
        "flat_positive_points": int(np.count_nonzero(dlognu_dlogr[flat] > 0.0)),
        "flat_total_points": int(np.count_nonzero(flat)),
    }


def save_outputs(result, R, Sigma, Sigma_err, outdir, unresolved_center):
    outdir.mkdir(parents=True, exist_ok=True)
    resolved_profile = pd.DataFrame({
        "row_type": "resolved_abel",
        "r_pc": result["R_grid"],
        "Sigma_abel_input": result["Sigma_grid"],
        "dSigma_dR": result["dSigma_dR"],
        "dlnSigma_dlnR": result["dlogSigma_dlogR"],
        "nu_pc3": result["nu"],
        "dlnnu_dlnr": result["dlognu_dlogr"],
        "abel_quad_error": result["quad_error"],
    })

    center_row = {column: np.nan for column in resolved_profile.columns}
    center_row["row_type"] = "unresolved_center"
    for key, value in unresolved_center.items():
        center_row[key] = value
    profile = pd.concat([pd.DataFrame([center_row]), resolved_profile], ignore_index=True, sort=False)
    profile_path = outdir / "Draco_abel_deprojection.csv"
    profile.to_csv(profile_path, index=False)
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].errorbar(R, Sigma, yerr=Sigma_err, fmt="o", capsize=3, label="Odenkirchen et al. 2001")
    axes[0].plot(result["R_grid"], result["Sigma_grid"], linewidth=2.0, label="Adopted Abel input")
    axes[0].axvspan(result["R_transition"], result["R_last"], alpha=0.08, label=f"Outer transition: {result['R_transition']:.1f}-{result['R_last']:.1f} pc")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_ylabel(r"$\Sigma(R)$ [stars pc$^{-2}$]")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(result["R_grid"], result["nu"], linewidth=2.0)
    axes[1].axvspan(result["R_transition"], result["R_last"], alpha=0.08)
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("Radius [pc]")
    axes[1].set_ylabel(r"$\nu(r)$ [stars pc$^{-3}$]"); axes[1].grid(alpha=0.25)
    fig.suptitle("Draco 2D to 3D Abel Deprojection")
    fig.tight_layout()
    deprojection_path = outdir / "Draco_abel_deprojection.png"
    fig.savefig(deprojection_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(result["R_grid"], result["dlogSigma_dlogR"], linewidth=2.0)
    axes[0].axhline(0.0, linewidth=1.0); axes[0].axhline(result["outer_slope"], linestyle=":", linewidth=1.0)
    axes[0].axvspan(result["R_transition"], result["R_last"], alpha=0.08)
    axes[0].set_ylabel(r"$d\ln\Sigma/d\ln R$"); axes[0].grid(alpha=0.25)
    axes[1].plot(result["R_grid"], result["dlognu_dlogr"], linewidth=2.0)
    axes[1].axhline(0.0, linewidth=1.0); axes[1].axvspan(*FLAT_REGION_PC, alpha=0.08)
    axes[1].axvspan(result["R_transition"], result["R_last"], alpha=0.08)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Radius [pc]"); axes[1].set_ylabel(r"$d\ln\nu/d\ln r$")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Draco Abel Logarithmic Slopes")
    fig.tight_layout()
    derivative_path = outdir / "Draco_abel_spline_derivative.png"
    fig.savefig(derivative_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    Sigma_forward = forward_project_density(R, result["R_grid"], result["nu"], result["R_last"], result["Sigma_last"], result["outer_slope"])
    closure_relative = (Sigma_forward - result["Sigma_fit_obs"]) / result["Sigma_fit_obs"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(R, result["Sigma_fit_obs"], "o-", label="Adopted Abel input")
    axes[0].plot(R, Sigma_forward, "s--", label="Forward projection of Abel density")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_ylabel(r"$\Sigma(R)$ [stars pc$^{-2}$]")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(R, closure_relative, "o-"); axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Projected radius R [pc]"); axes[1].set_ylabel("Relative closure error")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Draco Abel Forward-Projection Closure")
    fig.tight_layout()
    closure_path = outdir / "Draco_abel_forward_closure.png"
    fig.savefig(closure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "profile": profile_path, "deprojection_figure": deprojection_path, "derivative_figure": derivative_path,
        "closure_figure": closure_path, "closure_median": float(np.median(np.abs(closure_relative))),
        "closure_max": float(np.max(np.abs(closure_relative))),
    }

def build_parser():
    p = argparse.ArgumentParser(description="Draco non-parametric Abel deprojection.")
    p.add_argument("--surface-brightness", default="Data/Galaxy_Profiles/Draco/Draco_surface_brightness.csv")
    p.add_argument("--outdir", default="Data/Galaxy_Profiles/Draco/plots")
    p.add_argument("--radius-col", default="R_pc")
    p.add_argument("--sigma-col", default="Sigma_pc2")
    p.add_argument("--sigma-err-col", default="Sigma_err_pc2")
    p.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    p.add_argument("--smoothing-target", type=float, default=DEFAULT_SMOOTHING_TARGET)
    p.add_argument("--outer-tail-points", type=int, default=DEFAULT_OUTER_TAIL_POINTS)
    p.add_argument("--outer-transition-sigma", type=float, default=DEFAULT_OUTER_TRANSITION_SIGMA)
    return p

def main():
    args = build_parser().parse_args()
    R, Sigma, Sigma_err = load_surface_brightness(args.surface_brightness, radius_col=args.radius_col, sigma_col=args.sigma_col, sigma_err_col=args.sigma_err_col)
    median_plummer_error, max_plummer_error = validate_abel_integral_with_plummer()
    result = build_draco_deprojection(R, Sigma, Sigma_err, n_grid=args.n_grid, smoothing_target=args.smoothing_target, outer_tail_points=args.outer_tail_points, outer_transition_sigma=args.outer_transition_sigma)
    first_annulus_outer_pc, first_annulus_raw_light, q_axis_ratio = load_unresolved_center_inputs(args.surface_brightness)
    unresolved_center = unresolved_center_light_budget(result["R_grid"], result["nu"], result["R_last"], result["Sigma_last"], result["outer_slope"], first_annulus_outer_pc, first_annulus_raw_light, q_axis_ratio)
    outputs = save_outputs(result, R, Sigma, Sigma_err, Path(args.outdir), unresolved_center)
    print("ABEL DEPROJECTION")
    print("-----------------")
    print(f"observed radius range [pc]: {R[0]:.6f} -> {R[-1]:.6f}")
    print(f"deprojected radius range:   {result['R_grid'][0]:.6f} -> {result['R_grid'][-1]:.6f}")
    print("Abel inner continuation:    none")
    print(f"force inner continuation:   {unresolved_center['inner_continuation_model']}")
    print(f"inner cslope:               {unresolved_center['inner_cslope']:.12f}")
    print(f"inner boundary nu [pc^-3]:  {unresolved_center['inner_boundary_nu_pc3']:.12e}")
    print(f"inner reconstructed light:  {unresolved_center['inner_continuation_raw_light']:.12f}")
    print(f"inner light residual:       {unresolved_center['inner_continuation_light_residual']:.6e}")
    print(f"smoothing target:           {result['smoothing_target']:.6f}")
    print(f"linear-space chi2:          {result['chi2_linear']:.6f}")
    print(f"log-space weighted chi2:    {result['chi2_log']:.6f}")
    print(f"RMS residual [sigma]:       {np.sqrt(np.mean(result['residual_sigma']**2)):.6f}")
    print(f"max |residual| [sigma]:     {np.max(np.abs(result['residual_sigma'])):.6f}")
    print(f"outer slope:                {result['outer_slope']:.6f} +/- {result['outer_slope_err']:.6f}")
    print(f"outer tail points:          {result['outer_tail_points']}")
    print(f"transition threshold:       {result['outer_transition_sigma']:.3f} sigma")
    print(f"transition start [pc]:      {result['R_transition']:.6f}")
    print(f"start-point significance:   {result['transition_significance']:.6f} sigma")
    print(f"next-point significance:    {result['next_significance']:.6f} sigma")
    print(f"spline slope at transition: {result['transition_slope']:.6f}")
    print(f"minimum nu [stars pc^-3]:   {np.min(result['nu']):.12e}")
    print(f"minimum dlnnu/dlnr:         {np.min(result['dlognu_dlogr']):.6f}")
    print(f"40-100 pc median slope:     {result['flat_median_slope']:.6f}")
    print(f"40-100 pc max slope:        {result['flat_max_slope']:.6f}")
    print(f"40-100 pc positive points:  {result['flat_positive_points']} / {result['flat_total_points']}")
    print(f"max Abel quadrature error:  {np.max(result['quad_error']):.6e}")
    print(f"Plummer median rel error:   {median_plummer_error:.6e}")
    print(f"Plummer max rel error:      {max_plummer_error:.6e}")
    print(f"closure median rel error:   {outputs['closure_median']:.6e}")
    print(f"closure max rel error:      {outputs['closure_max']:.6e}")
    print(f"Abel support minimum [pc]:  {unresolved_center['abel_support_min_pc']:.6f}")
    print(f"first annulus outer [pc]:   {unresolved_center['first_annulus_outer_pc']:.6f}")
    print(f"first annulus raw light:    {unresolved_center['first_annulus_raw_light']:.12f}")
    print(f"resolved annulus raw light: {unresolved_center['resolved_first_annulus_raw_light']:.12f}")
    print(f"unresolved center raw light:{unresolved_center['unresolved_center_raw_light']: .12f}")
    print(f"unresolved center fraction: {unresolved_center['unresolved_center_first_annulus_fraction']:.6f}")
    print()
    print("OUTER OBSERVED RESIDUALS")
    print("------------------------")
    for i in range(result["transition_index"], len(R)):
        print(f"R={R[i]:10.3f} pc   residual={result['residual_sigma'][i]: .6f} sigma")
    print()
    print("Saved:")
    for path in outputs.values():
        if isinstance(path, Path):
            print(path)

if __name__ == "__main__":
    main()
