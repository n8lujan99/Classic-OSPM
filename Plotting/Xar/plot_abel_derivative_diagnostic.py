from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
from scipy.special import gamma
from scipy.optimize import curve_fit

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from OSPM.Mapping.observable_mapping import smooth_log_profile

sb_path = Path("Data/Galaxy_Profiles/Draco/Draco_surface_brightness.csv")
outdir = Path("Data/Galaxy_Profiles/Draco/plots")

sb = pd.read_csv(sb_path)

R = sb["R_pc"].to_numpy(float)
Sigma = sb["Sigma"].to_numpy(float)

if "Sigma_err" not in sb.columns:
    raise KeyError("Draco surface-brightness file requires Sigma_err for weighted spline test.")

Sigma_err = sb["Sigma_err"].to_numpy(float)

good = (
    np.isfinite(R)
    & np.isfinite(Sigma)
    & np.isfinite(Sigma_err)
    & (R > 0.0)
    & (Sigma > 0.0)
    & (Sigma_err > 0.0)
)

R = R[good]
Sigma = Sigma[good]
Sigma_err = Sigma_err[good]

order = np.argsort(R)
R = R[order]
Sigma = Sigma[order]
Sigma_err = Sigma_err[order]


def observed_domain_profile(R, Sigma, n_grid=256):
    Rg = np.geomspace(R[0], R[-1], n_grid)
    logS = np.interp(np.log(Rg), np.log(R), np.log(Sigma))
    Sg = np.exp(logS)
    return Rg, Sg


def fit_outer_powerlaw_slope(R, Sigma, n_tail=6):
    n_tail = min(n_tail, len(R))
    slope = float(np.polyfit(np.log(R[-n_tail:]), np.log(Sigma[-n_tail:]), 1)[0])
    return slope


def raw_abel_finite_grid(R, Sigma):
    dSigma_dR = np.gradient(Sigma, R)
    nu = np.full_like(R, np.nan)

    for i, r in enumerate(R):
        Rp = R[i:]
        dS = dSigma_dR[i:]

        if len(Rp) < 2:
            continue

        u = np.sqrt(np.maximum(Rp**2 - r**2, 0.0))
        integrand = -dS / Rp
        nu[i] = np.trapezoid(integrand, u) / np.pi

    return nu


def raw_abel_powerlaw_tail(R, Sigma, outer_slope, tail_factor=100.0, points_per_decade=128):
    if outer_slope >= -1.0:
        raise ValueError(
            f"Outer slope must be < -1 for the Abel boundary term to vanish; got {outer_slope:.6f}"
        )

    decades = np.log10(tail_factor)
    n_tail = max(16, int(np.ceil(points_per_decade * decades)))

    R_tail = np.geomspace(R[-1], R[-1] * tail_factor, n_tail + 1)[1:]
    Sigma_tail = Sigma[-1] * (R_tail / R[-1])**outer_slope

    dSigma_dR = np.gradient(Sigma, R)
    dSigma_dR_tail = outer_slope * Sigma_tail / R_tail

    R_support = np.concatenate((R, R_tail))
    dSigma_dR_support = np.concatenate((dSigma_dR, dSigma_dR_tail))

    nu = np.full_like(R, np.nan)

    for i, r in enumerate(R):
        Rp = R_support[i:]
        dS = dSigma_dR_support[i:]

        u = np.sqrt(np.maximum(Rp**2 - r**2, 0.0))
        integrand = -dS / Rp
        nu[i] = np.trapezoid(integrand, u) / np.pi

    return nu


def report_raw_nu(label, Rg, nu):
    finite = np.isfinite(nu)
    negative = finite & (nu < 0.0)
    zero = finite & (nu == 0.0)

    print()
    print(label)
    print("-" * len(label))
    print(f"N total:      {len(nu)}")
    print(f"N finite:     {np.count_nonzero(finite)}")
    print(f"N negative:   {np.count_nonzero(negative)}")
    print(f"N zero:       {np.count_nonzero(zero)}")

    if np.any(finite):
        imin = np.nanargmin(nu)
        print(f"minimum nu:   {nu[imin]:.12e}")
        print(f"minimum R:    {Rg[imin]:.6f} pc")

    if np.any(negative):
        neg_idx = np.where(negative)[0]
        print("negative-density radii:")
        for i in neg_idx:
            print(f"    R={Rg[i]:10.3f} pc   nu={nu[i]: .12e}")


Rg_current, Sg_current = smooth_log_profile(R, Sigma, n_grid=256)

dSigma_dR_current = np.gradient(Sg_current, Rg_current)

positive = Sg_current > 0.0
dlnSigma_dlnR = np.full_like(Sg_current, np.nan)
dlnSigma_dlnR[positive] = np.gradient(
    np.log(Sg_current[positive]),
    np.log(Rg_current[positive]),
)

segment_slope = np.diff(np.log(Sigma)) / np.diff(np.log(R))
segment_mid = np.sqrt(R[:-1] * R[1:])

diag = pd.DataFrame({
    "R_pc": Rg_current,
    "Sigma": Sg_current,
    "dSigma_dR": dSigma_dR_current,
    "dlnSigma_dlnR": dlnSigma_dlnR,
})

outdir.mkdir(parents=True, exist_ok=True)
diag.to_csv(outdir / "Draco_abel_derivative_diagnostic.csv", index=False)

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(Rg_current, Sg_current, linewidth=2.0, label="Current Abel input")
axes[0].plot(R, Sigma, "o", label="Observed Draco profile")
axes[0].set_yscale("log")
axes[0].set_ylabel(r"$\Sigma(R)$")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(Rg_current, dSigma_dR_current)
axes[1].axhline(0.0, linewidth=1.0)
axes[1].set_ylabel(r"$d\Sigma/dR$")
axes[1].grid(alpha=0.25)

axes[2].plot(Rg_current, dlnSigma_dlnR, linewidth=1.5, label="Finite-difference slope")
axes[2].plot(segment_mid, segment_slope, "o", label="Exact log-log segment slope")
axes[2].axhline(0.0, linewidth=1.0)
axes[2].set_ylabel(r"$d\ln\Sigma/d\ln R$")
axes[2].set_xlabel("Projected radius R [pc]")
axes[2].legend()
axes[2].grid(alpha=0.25)

for ax in axes:
    ax.set_xscale("log")

fig.suptitle("Draco Abel Input and Derivative Diagnostic")
fig.tight_layout()
fig.savefig(
    outdir / "Draco_abel_derivative_diagnostic.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)


# ------------------------------------------------------------------
# Boundary and raw-Abel test
# ------------------------------------------------------------------

Rg_observed, Sg_observed = observed_domain_profile(R, Sigma, n_grid=256)

outer_slope = fit_outer_powerlaw_slope(R, Sigma, n_tail=6)

nu_current_raw = raw_abel_finite_grid(Rg_current, Sg_current)

nu_tail_10 = raw_abel_powerlaw_tail(
    Rg_observed,
    Sg_observed,
    outer_slope,
    tail_factor=10.0,
)

nu_tail_100 = raw_abel_powerlaw_tail(
    Rg_observed,
    Sg_observed,
    outer_slope,
    tail_factor=100.0,
)

nu_tail_1000 = raw_abel_powerlaw_tail(
    Rg_observed,
    Sg_observed,
    outer_slope,
    tail_factor=1000.0,
)

report_raw_nu("CURRENT PROFILE — RAW ABEL, NO CLEANUP", Rg_current, nu_current_raw)
report_raw_nu("OBSERVED DOMAIN + POWER-LAW TAIL x10", Rg_observed, nu_tail_10)
report_raw_nu("OBSERVED DOMAIN + POWER-LAW TAIL x100", Rg_observed, nu_tail_100)
report_raw_nu("OBSERVED DOMAIN + POWER-LAW TAIL x1000", Rg_observed, nu_tail_1000)

good_tail = (
    np.isfinite(nu_tail_10)
    & np.isfinite(nu_tail_100)
    & np.isfinite(nu_tail_1000)
    & (nu_tail_1000 != 0.0)
)

rel_10_100 = np.abs(
    (nu_tail_10[good_tail] - nu_tail_100[good_tail])
    / nu_tail_1000[good_tail]
)

rel_100_1000 = np.abs(
    (nu_tail_100[good_tail] - nu_tail_1000[good_tail])
    / nu_tail_1000[good_tail]
)

print()
print("OUTER-TAIL CONVERGENCE")
print("----------------------")
print(f"Fitted outer slope: {outer_slope:.8f}")
print(f"Observed R range:   {R[0]:.6f} -> {R[-1]:.6f} pc")
print(f"Current R range:    {Rg_current[0]:.6f} -> {Rg_current[-1]:.6f} pc")
print(f"Test R range:       {Rg_observed[0]:.6f} -> {Rg_observed[-1]:.6f} pc")
print(f"max |nu10-nu100|/|nu1000|:   {np.max(rel_10_100):.6e}")
print(f"max |nu100-nu1000|/|nu1000|: {np.max(rel_100_1000):.6e}")
print(f"median 10 vs 100:             {np.median(rel_10_100):.6e}")
print(f"median 100 vs 1000:           {np.median(rel_100_1000):.6e}")


current_in_observed = Rg_current >= R[0]

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

axes[0].plot(
    Rg_current,
    Sg_current,
    linewidth=1.8,
    label="Current: inner extension + taper",
)
axes[0].plot( Rg_observed, Sg_observed, linewidth=1.8, label="Observed-domain interpolation")
axes[0].plot(R, Sigma, "o", label="Observed Draco profile")
axes[0].axvline(R[0], linestyle="--", linewidth=1.0)
axes[0].axvline(R[-1], linestyle="--", linewidth=1.0)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_ylabel(r"$\Sigma(R)$")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot( Rg_current[current_in_observed], nu_current_raw[current_in_observed], label="Current boundaries, raw Abel")
axes[1].plot(Rg_observed, nu_tail_100, label="Power-law tail x100, raw Abel")
axes[1].axhline(0.0, linewidth=1.0)
axes[1].set_xscale("log")
axes[1].set_yscale("symlog", linthresh=1e-7)
axes[1].set_ylabel(r"raw $\nu(r)$")
axes[1].legend()
axes[1].grid(alpha=0.25)

axes[2].plot(Rg_observed, nu_tail_10, label="Tail to 10 Rlast")
axes[2].plot(Rg_observed, nu_tail_100, label="Tail to 100 Rlast")
axes[2].plot(Rg_observed, nu_tail_1000, label="Tail to 1000 Rlast")
axes[2].axhline(0.0, linewidth=1.0)
axes[2].set_xscale("log")
axes[2].set_yscale("symlog", linthresh=1e-7)
axes[2].set_xlabel("Radius r [pc]")
axes[2].set_ylabel(r"raw $\nu(r)$")
axes[2].legend()
axes[2].grid(alpha=0.25)

fig.suptitle("Draco Abel Boundary and Raw-Density Test")
fig.tight_layout()
fig.savefig(outdir / "Draco_abel_boundary_raw_nu_test.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print()
print("Saved:")
print(outdir / "Draco_abel_derivative_diagnostic.png")
print(outdir / "Draco_abel_derivative_diagnostic.csv")
print(outdir / "Draco_abel_boundary_raw_nu_test.png")

print()
print("Local log-log slopes between observed points:")
for i, slope in enumerate(segment_slope):
    print(
        f"{R[i]:10.3f} -> {R[i + 1]:10.3f} pc   "
        f"slope={slope: .6f}"
    )
# ------------------------------------------------------------------
# PURE ABEL-INTEGRAL TEST
#
# This deliberately removes:
#   - observational interpolation
#   - finite-difference derivatives
#   - inner extrapolation
#   - outer taper
#   - positivity cleanup
#
# We supply an analytic Plummer Sigma(R), its exact derivative,
# and compare the numerical Abel integral against the exact nu(r).
# ------------------------------------------------------------------

def plummer_sigma(R, L=1.0, a=200.0):
    R = np.asarray(R, float)
    return L / (np.pi * a**2) * (1.0 + R**2 / a**2)**(-2.0)


def plummer_dSigma_dR(R, L=1.0, a=200.0):
    R = np.asarray(R, float)
    return (
        -4.0
        * L
        * R
        / (np.pi * a**4)
        * (1.0 + R**2 / a**2)**(-3.0)
    )


def plummer_nu(r, L=1.0, a=200.0):
    r = np.asarray(r, float)
    return (
        3.0
        * L
        / (4.0 * np.pi * a**3)
        * (1.0 + r**2 / a**2)**(-2.5)
    )


def abel_integral_exact_derivative(R, dSigma_dR):
    R = np.asarray(R, float)
    dSigma_dR = np.asarray(dSigma_dR, float)

    nu = np.full_like(R, np.nan)

    for i, r in enumerate(R):
        Rp = R[i:]
        dS = dSigma_dR[i:]

        if len(Rp) < 2:
            continue

        u = np.sqrt(np.maximum(Rp**2 - r**2, 0.0))
        integrand = -dS / Rp

        nu[i] = np.trapezoid(integrand, u) / np.pi

    return nu


print()
print("PURE ABEL-INTEGRAL TEST — ANALYTIC PLUMMER")
print("------------------------------------------")

L_test = 1.0
a_test = 200.0

for n_grid_test in (128, 256, 512, 1024, 2048):
    R_test = np.geomspace(
        0.01 * a_test,
        100.0 * a_test,
        n_grid_test,
    )

    Sigma_test = plummer_sigma(
        R_test,
        L=L_test,
        a=a_test,
    )

    dSigma_exact = plummer_dSigma_dR(
        R_test,
        L=L_test,
        a=a_test,
    )

    nu_exact = plummer_nu(
        R_test,
        L=L_test,
        a=a_test,
    )

    nu_numeric = abel_integral_exact_derivative(
        R_test,
        dSigma_exact,
    )

    test_region = (
        np.isfinite(nu_numeric)
        & (R_test >= 0.05 * a_test)
        & (R_test <= 10.0 * a_test)
    )

    relative_error = np.abs(
        (nu_numeric[test_region] - nu_exact[test_region])
        / nu_exact[test_region]
    )

    print(
        f"N={n_grid_test:4d}   "
        f"median rel err={np.median(relative_error):.6e}   "
        f"max rel err={np.max(relative_error):.6e}"
    )


# Detailed 256-point comparison for plotting.
R_test = np.geomspace(
    0.01 * a_test,
    100.0 * a_test,
    256,
)

Sigma_test = plummer_sigma(
    R_test,
    L=L_test,
    a=a_test,
)

dSigma_exact = plummer_dSigma_dR(
    R_test,
    L=L_test,
    a=a_test,
)

nu_exact = plummer_nu(
    R_test,
    L=L_test,
    a=a_test,
)

nu_numeric = abel_integral_exact_derivative(
    R_test,
    dSigma_exact,
)

relative_error = np.full_like(R_test, np.nan)

valid = (
    np.isfinite(nu_numeric)
    & np.isfinite(nu_exact)
    & (nu_exact > 0.0)
)

relative_error[valid] = (
    nu_numeric[valid] - nu_exact[valid]
) / nu_exact[valid]


fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

axes[0].plot(
    R_test,
    nu_exact,
    linewidth=2.0,
    label="Analytic Plummer density",
)

axes[0].plot(
    R_test,
    nu_numeric,
    linestyle="--",
    linewidth=1.5,
    label="Numerical Abel integral",
)

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_ylabel(r"$\nu(r)$")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(
    R_test,
    relative_error,
)

axes[1].axhline(
    0.0,
    linewidth=1.0,
)

axes[1].set_xscale("log")
axes[1].set_xlabel("Radius r [pc]")
axes[1].set_ylabel(r"$(\nu_{\rm Abel}-\nu_{\rm exact})/\nu_{\rm exact}$")
axes[1].grid(alpha=0.25)

fig.suptitle("Pure Abel Integral Test — Analytic Plummer Profile")
fig.tight_layout()

fig.savefig(
    outdir / "Plummer_pure_abel_integral_test.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("Saved:")
print(outdir / "Plummer_pure_abel_integral_test.png")


# ------------------------------------------------------------------
# WEIGHTED SMOOTH-SPLINE ABEL CANDIDATE
#
# New candidate conditions:
#   - use observational Sigma_err
#   - fit smooth log(Sigma) as a function of log(R)
#   - obtain dSigma/dR analytically from the spline
#   - no inner extrapolation below first observed point
#   - explicit outer power-law continuation to infinity
#   - no clipping of negative nu
#   - no interpolation across failed nu
#   - forward-project recovered nu and test closure
# ------------------------------------------------------------------

def fit_weighted_log_spline(R, Sigma, Sigma_err, smoothing_target=None):
    x = np.log(R)
    y = np.log(Sigma)

    # First-order propagation:
    #
    #     sigma_logSigma ~= sigma_Sigma / Sigma
    #
    # Very uncertain low-Sigma points therefore receive correspondingly
    # small weight rather than controlling the derivative.
    sigma_logSigma = Sigma_err / Sigma
    weights = 1.0 / sigma_logSigma

    if smoothing_target is None:
        smoothing_target = float(len(R))

    spline = UnivariateSpline(
        x,
        y,
        w=weights,
        k=3,
        s=smoothing_target,
    )

    return spline, sigma_logSigma, smoothing_target


def evaluate_log_spline_profile(spline, R_eval):
    R_eval = np.asarray(R_eval, float)

    x = np.log(R_eval)

    logSigma = spline(x)
    Sigma_eval = np.exp(logSigma)

    dlogSigma_dlogR = spline.derivative(1)(x)

    dSigma_dR = (
        Sigma_eval
        / R_eval
        * dlogSigma_dlogR
    )

    return Sigma_eval, dSigma_dR, dlogSigma_dlogR


def abel_spline_powerlaw_infinity(
    R_eval,
    spline,
    R_last,
    Sigma_last,
    outer_slope,
):
    if outer_slope >= -1.0:
        raise ValueError(
            f"Outer slope must be < -1 for Abel convergence; got {outer_slope:.8f}"
        )

    spline_d1 = spline.derivative(1)

    nu = np.full_like(
        np.asarray(R_eval, float),
        np.nan,
    )

    quad_error = np.full_like(
        np.asarray(R_eval, float),
        np.nan,
    )

    def sigma_and_derivative(Rp):
        if Rp <= R_last:
            x = np.log(Rp)

            logS = float(spline(x))
            S = np.exp(logS)

            slope = float(spline_d1(x))

            dS = (
                S
                / Rp
                * slope
            )

            return S, dS

        S = (
            Sigma_last
            * (Rp / R_last)**outer_slope
        )

        dS = (
            outer_slope
            * S
            / Rp
        )

        return S, dS

    for i, r in enumerate(R_eval):
        r = float(r)

        # u = sqrt(R^2-r^2)
        #
        # R = sqrt(r^2+u^2)
        #
        # This directly evaluates the transformed Abel integral from
        # u=0 to infinity.
        def integrand(u):
            Rp = np.hypot(r, u)

            _, dS = sigma_and_derivative(Rp)

            return (
                -dS
                / Rp
                / np.pi
            )

        val, err = quad(
            integrand,
            0.0,
            np.inf,
            epsabs=1e-12,
            epsrel=1e-8,
            limit=300,
        )

        nu[i] = val
        quad_error[i] = err

    return nu, quad_error


def powerlaw_tail_nu(r, R_last, Sigma_last, outer_slope):
    # Sigma(R) proportional to R^m
    #
    # If p=-m, then
    #
    #     Sigma proportional to R^-p
    #     nu    proportional to r^-(p+1)
    #
    # This gives the exact 3D continuation corresponding to the
    # adopted projected power-law tail.

    m = float(outer_slope)
    p = -m

    if p <= 1.0:
        raise ValueError("Projected power-law tail requires p > 1.")

    coefficient = (
        gamma((p + 1.0) / 2.0)
        / (
            np.sqrt(np.pi)
            * gamma(p / 2.0)
        )
    )

    nu_last = (
        coefficient
        * Sigma_last
        / R_last
    )

    return (
        nu_last
        * (np.asarray(r, float) / R_last)**(m - 1.0)
    )


def forward_project_nu(
    R_project,
    R_nu,
    nu,
    R_last,
    Sigma_last,
    outer_slope,
):
    R_nu = np.asarray(R_nu, float)
    nu = np.asarray(nu, float)

    if np.any(~np.isfinite(nu)):
        raise ValueError("Cannot forward-project non-finite nu.")

    if np.any(nu <= 0.0):
        raise ValueError("Cannot use log interpolation for forward test because nu contains non-positive values.")

    nu_spline = UnivariateSpline(
        np.log(R_nu),
        np.log(nu),
        k=3,
        s=0.0,
    )

    nu_tail_at_last = float(
        powerlaw_tail_nu(
            R_last,
            R_last,
            Sigma_last,
            outer_slope,
        )
    )

    nu_numeric_at_last = float(nu[-1])

    print()
    print("POWER-LAW TAIL JOIN")
    print("-------------------")
    print(f"numeric nu(Rlast):  {nu_numeric_at_last:.12e}")
    print(f"analytic tail nu:   {nu_tail_at_last:.12e}")
    print(
        "relative mismatch: "
        f"{abs(nu_numeric_at_last - nu_tail_at_last) / nu_tail_at_last:.6e}"
    )

    def nu_function(r):
        if r <= R_last:
            return float(
                np.exp(
                    nu_spline(np.log(r))
                )
            )

        return float(
            powerlaw_tail_nu(
                r,
                R_last,
                Sigma_last,
                outer_slope,
            )
        )

    Sigma_forward = np.full_like(
        np.asarray(R_project, float),
        np.nan,
    )

    for i, Rp in enumerate(R_project):
        Rp = float(Rp)

        # Equivalent non-singular form of the forward projection:
        #
        # Sigma(R) = 2 integral_0^infinity
        #            nu(sqrt(R^2 + z^2)) dz
        def integrand(z):
            r = np.hypot(Rp, z)
            return 2.0 * nu_function(r)

        val, _ = quad(
            integrand,
            0.0,
            np.inf,
            epsabs=1e-10,
            epsrel=1e-7,
            limit=300,
        )

        Sigma_forward[i] = val

    return Sigma_forward


print()
print("WEIGHTED LOG-SPLINE ABEL TEST")
print("-----------------------------")

N_spline = len(R)

spline, sigma_logSigma, smoothing_target = fit_weighted_log_spline(
    R,
    Sigma,
    Sigma_err,
    smoothing_target=float(N_spline),
)

Rg_spline = np.geomspace(
    R[0],
    R[-1],
    256,
)

(
    Sigma_spline,
    dSigma_dR_spline,
    dlogSigma_dlogR_spline,
) = evaluate_log_spline_profile(
    spline,
    Rg_spline,
)

Sigma_fit_obs, _, slope_fit_obs = evaluate_log_spline_profile(
    spline,
    R,
)

residual_sigma = (
    Sigma
    - Sigma_fit_obs
) / Sigma_err

chi2_linear = float(
    np.sum(
        residual_sigma**2
    )
)

log_residual_sigma = (
    np.log(Sigma)
    - np.log(Sigma_fit_obs)
) / sigma_logSigma

chi2_log = float(
    np.sum(
        log_residual_sigma**2
    )
)

outer_slope_spline = float(
    spline.derivative(1)(
        np.log(R[-1])
    )
)

Sigma_last_spline = float(
    np.exp(
        spline(
            np.log(R[-1])
        )
    )
)

print(f"N observations:           {len(R)}")
print(f"smoothing target s:       {smoothing_target:.6f}")
print(f"linear-space chi2:        {chi2_linear:.6f}")
print(f"log-space weighted chi2:  {chi2_log:.6f}")
print(f"RMS residual [sigma]:     {np.sqrt(np.mean(residual_sigma**2)):.6f}")
print(f"max |residual| [sigma]:   {np.max(np.abs(residual_sigma)):.6f}")
print(f"minimum log slope:        {np.min(dlogSigma_dlogR_spline):.6f}")
print(f"maximum log slope:        {np.max(dlogSigma_dlogR_spline):.6f}")
print(
    "N grid points slope > 0: "
    f"{np.count_nonzero(dlogSigma_dlogR_spline > 0.0)} / {len(Rg_spline)}"
)
print(f"spline outer slope:       {outer_slope_spline:.8f}")
print(f"previous last-6 slope:    {outer_slope:.8f}")


nu_spline_raw, nu_spline_quad_error = abel_spline_powerlaw_infinity(
    Rg_spline,
    spline,
    R[-1],
    Sigma_last_spline,
    outer_slope_spline,
)

report_raw_nu(
    "WEIGHTED SPLINE + POWER-LAW TO INFINITY — RAW ABEL",
    Rg_spline,
    nu_spline_raw,
)

print()
print("ABEL QUADRATURE")
print("---------------")
print(
    "max reported quad error:    "
    f"{np.nanmax(nu_spline_quad_error):.6e}"
)
print(
    "median reported quad error: "
    f"{np.nanmedian(nu_spline_quad_error):.6e}"
)


if np.all(np.isfinite(nu_spline_raw)) and np.all(nu_spline_raw > 0.0):
    Sigma_forward = forward_project_nu(
        R,
        Rg_spline,
        nu_spline_raw,
        R[-1],
        Sigma_last_spline,
        outer_slope_spline,
    )

    Sigma_target = Sigma_fit_obs

    closure_relative = (
        Sigma_forward
        - Sigma_target
    ) / Sigma_target

    print()
    print("FORWARD-PROJECTION CLOSURE")
    print("--------------------------")
    print(
        "median |relative error|: "
        f"{np.median(np.abs(closure_relative)):.6e}"
    )
    print(
        "max |relative error|:    "
        f"{np.max(np.abs(closure_relative)):.6e}"
    )

else:
    Sigma_forward = None

    print()
    print("FORWARD-PROJECTION CLOSURE")
    print("--------------------------")
    print("SKIPPED: raw Abel density contains non-positive or non-finite values.")


# ------------------------------------------------------------------
# Candidate comparison plot
# ------------------------------------------------------------------

current_observed_slope = np.gradient(
    np.log(Sg_observed),
    np.log(Rg_observed),
)

fig, axes = plt.subplots(
    4,
    1,
    figsize=(10, 15),
    sharex=True,
)

axes[0].errorbar(
    R,
    Sigma,
    yerr=Sigma_err,
    fmt="o",
    capsize=3,
    label="Odenkirchen data",
)

axes[0].plot(
    Rg_observed,
    Sg_observed,
    linewidth=1.5,
    label="Current piecewise log-log",
)

axes[0].plot(
    Rg_spline,
    Sigma_spline,
    linewidth=2.0,
    label="Weighted smoothing spline",
)

axes[0].set_yscale("log")
axes[0].set_ylabel(r"$\Sigma(R)$")
axes[0].legend()
axes[0].grid(alpha=0.25)


axes[1].axhline(
    0.0,
    linewidth=1.0,
)

axes[1].errorbar(
    R,
    residual_sigma,
    yerr=np.ones_like(R),
    fmt="o",
    capsize=3,
)

axes[1].axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
)

axes[1].axhline(
    -1.0,
    linestyle="--",
    linewidth=1.0,
)

axes[1].axhline(
    2.0,
    linestyle=":",
    linewidth=1.0,
)

axes[1].axhline(
    -2.0,
    linestyle=":",
    linewidth=1.0,
)

axes[1].set_ylabel("Spline residual [sigma]")
axes[1].grid(alpha=0.25)


axes[2].plot(
    Rg_observed,
    current_observed_slope,
    linewidth=1.5,
    label="Current finite-difference slope",
)

axes[2].plot(
    Rg_spline,
    dlogSigma_dlogR_spline,
    linewidth=2.0,
    label="Analytic spline slope",
)

axes[2].axhline(
    0.0,
    linewidth=1.0,
)

axes[2].set_ylabel(r"$d\ln\Sigma/d\ln R$")
axes[2].legend()
axes[2].grid(alpha=0.25)


axes[3].plot(
    Rg_observed,
    nu_tail_100,
    linewidth=1.5,
    label="Current interpolation + power-law tail",
)

axes[3].plot(
    Rg_spline,
    nu_spline_raw,
    linewidth=2.0,
    label="Weighted spline + analytic derivative",
)

axes[3].axhline(
    0.0,
    linewidth=1.0,
)

axes[3].set_yscale(
    "symlog",
    linthresh=1e-7,
)

axes[3].set_ylabel(r"raw $\nu(r)$")
axes[3].set_xlabel("Radius [pc]")
axes[3].legend()
axes[3].grid(alpha=0.25)


for ax in axes:
    ax.set_xscale("log")


fig.suptitle(
    "Draco Weighted-Spline Abel Candidate"
)

fig.tight_layout()

fig.savefig(
    outdir / "Draco_weighted_spline_abel_candidate.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------------
# Forward-projection closure plot
# ------------------------------------------------------------------

if Sigma_forward is not None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    axes[0].plot(
        R,
        Sigma_target,
        "o-",
        label="Spline Sigma input",
    )

    axes[0].plot(
        R,
        Sigma_forward,
        "s--",
        label="Forward-projected Abel density",
    )

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\Sigma(R)$")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        R,
        closure_relative,
        "o-",
    )

    axes[1].axhline(
        0.0,
        linewidth=1.0,
    )

    axes[1].set_xlabel("Projected radius R [pc]")
    axes[1].set_ylabel("Forward closure relative error")
    axes[1].grid(alpha=0.25)

    for ax in axes:
        ax.set_xscale("log")

    fig.suptitle(
        "Draco Abel Forward-Projection Closure"
    )

    fig.tight_layout()

    fig.savefig(
        outdir / "Draco_weighted_spline_forward_closure.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


candidate = pd.DataFrame({
    "R_pc": Rg_spline,
    "Sigma_spline": Sigma_spline,
    "dSigma_dR_spline": dSigma_dR_spline,
    "dlnSigma_dlnR_spline": dlogSigma_dlogR_spline,
    "nu_raw": nu_spline_raw,
    "abel_quad_error": nu_spline_quad_error,
})

candidate.to_csv(
    outdir / "Draco_weighted_spline_abel_candidate.csv",
    index=False,
)

print()
print("Saved:")
print(outdir / "Draco_weighted_spline_abel_candidate.png")
print(outdir / "Draco_weighted_spline_abel_candidate.csv")

if Sigma_forward is not None:
    print(outdir / "Draco_weighted_spline_forward_closure.png")


# ------------------------------------------------------------------
# OUTER-TAIL DIAGNOSTIC
#
# Diagnose whether the steep spline endpoint slope is physically
# supported by the outer Odenkirchen measurements or is primarily
# spline boundary behavior.
# ------------------------------------------------------------------

print()
print("OUTER-TAIL DIAGNOSTIC")
print("---------------------")

x_all = np.log(R)
y_all = np.log(Sigma)
sigma_log_all = Sigma_err / Sigma

spline_slope_at_data = spline.derivative(1)(x_all)

incoming_segment_slope = np.full_like(R, np.nan)
incoming_segment_slope[1:] = np.diff(y_all) / np.diff(x_all)

outer_table = pd.DataFrame({
    "R_pc": R,
    "Sigma": Sigma,
    "Sigma_err": Sigma_err,
    "relative_error": Sigma_err / Sigma,
    "incoming_segment_slope": incoming_segment_slope,
    "spline_log_slope": spline_slope_at_data,
    "spline_residual_sigma": residual_sigma,
})

print()
print("OUTER DATA POINTS")
print("-----------------")
print(
    outer_table.tail(8).to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


tail_fits = []

for n_tail_fit in range(3, 9):
    x = x_all[-n_tail_fit:]
    y = y_all[-n_tail_fit:]
    sigma_y = sigma_log_all[-n_tail_fit:]

    weights = 1.0 / sigma_y

    coeff, cov = np.polyfit(
        x,
        y,
        1,
        w=weights,
        cov=True,
    )

    slope = float(coeff[0])
    intercept = float(coeff[1])
    slope_err = float(np.sqrt(cov[0, 0]))

    model = intercept + slope * x
    chi2 = float(np.sum(((y - model) / sigma_y)**2))
    dof = n_tail_fit - 2
    reduced_chi2 = chi2 / dof if dof > 0 else np.nan

    tail_fits.append({
        "N_tail": n_tail_fit,
        "slope": slope,
        "slope_err": slope_err,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
        "R_first_pc": R[-n_tail_fit],
        "R_last_pc": R[-1],
    })


tail_fit_df = pd.DataFrame(tail_fits)

print()
print("WEIGHTED LOG-LOG POWER-LAW FITS")
print("-------------------------------")
print(
    tail_fit_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

tail_fit_df.to_csv(
    outdir / "Draco_outer_tail_powerlaw_fits.csv",
    index=False,
)


# ------------------------------------------------------------------
# Compare spline derivative through the outer observed region.
# ------------------------------------------------------------------

R_outer_dense = np.geomspace(
    R[-8],
    R[-1],
    256,
)

Sigma_outer_dense, _, spline_outer_slope_dense = evaluate_log_spline_profile(
    spline,
    R_outer_dense,
)


# ------------------------------------------------------------------
# Build Abel solutions using different outer-tail slopes.
#
# The interior spline is IDENTICAL in every case.
# Only the continuation beyond R_last changes.
# ------------------------------------------------------------------

candidate_slopes = {
    "Spline endpoint": outer_slope_spline,
}

for n_tail_fit in (4, 5, 6, 7, 8):
    row = tail_fit_df.loc[
        tail_fit_df["N_tail"] == n_tail_fit
    ].iloc[0]

    candidate_slopes[f"Last {n_tail_fit} weighted fit"] = float(
        row["slope"]
    )


tail_nu = {}

for label, slope in candidate_slopes.items():
    if slope >= -1.0:
        print(
            f"SKIPPING {label}: slope={slope:.6f} "
            "does not satisfy Abel outer-boundary convergence."
        )
        continue

    nu_test, _ = abel_spline_powerlaw_infinity(
        Rg_spline,
        spline,
        R[-1],
        Sigma_last_spline,
        slope,
    )

    tail_nu[label] = nu_test


reference_label = "Last 6 weighted fit"
nu_reference = tail_nu[reference_label]


print()
print("TAIL SENSITIVITY OF ABEL DENSITY")
print("--------------------------------")

inner_region = Rg_spline <= 0.5 * R[-1]
outer_region = (
    (Rg_spline > 0.5 * R[-1])
    & (Rg_spline <= R[-1])
)

for label, nu_test in tail_nu.items():
    relative = np.abs(
        (nu_test - nu_reference)
        / nu_reference
    )

    print()
    print(label)
    print(
        f"  slope:                         "
        f"{candidate_slopes[label]:.8f}"
    )
    print(
        f"  max relative change R<0.5Rlast: "
        f"{np.max(relative[inner_region]):.6e}"
    )
    print(
        f"  median change R<0.5Rlast:       "
        f"{np.median(relative[inner_region]):.6e}"
    )
    print(
        f"  max relative change outer half: "
        f"{np.max(relative[outer_region]):.6e}"
    )
    print(
        f"  median change outer half:       "
        f"{np.median(relative[outer_region]):.6e}"
    )


# ------------------------------------------------------------------
# Plot outer data, spline slope, tail continuations, and Abel response.
# ------------------------------------------------------------------

R_tail_plot = np.geomspace(
    R[-1],
    10.0 * R[-1],
    256,
)

fig, axes = plt.subplots(
    4,
    1,
    figsize=(10, 15),
    sharex=False,
)


# Observed outer profile + spline.
axes[0].errorbar(
    R[-8:],
    Sigma[-8:],
    yerr=Sigma_err[-8:],
    fmt="o",
    capsize=3,
    label="Outer Odenkirchen points",
)

axes[0].plot(
    R_outer_dense,
    Sigma_outer_dense,
    linewidth=2.0,
    label="Weighted spline",
)

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_ylabel(r"$\Sigma(R)$")
axes[0].legend()
axes[0].grid(alpha=0.25)


# Spline derivative versus data-supported power-law slopes.
axes[1].plot(
    R_outer_dense,
    spline_outer_slope_dense,
    linewidth=2.0,
    label="Spline derivative",
)

for n_tail_fit in (4, 5, 6, 7, 8):
    row = tail_fit_df.loc[
        tail_fit_df["N_tail"] == n_tail_fit
    ].iloc[0]

    axes[1].axhline(
        float(row["slope"]),
        linestyle="--",
        linewidth=1.0,
        label=f"Last {n_tail_fit}: {row['slope']:.2f}",
    )

axes[1].axhline(
    outer_slope_spline,
    linestyle=":",
    linewidth=1.5,
    label=f"Spline endpoint: {outer_slope_spline:.2f}",
)

axes[1].set_xscale("log")
axes[1].set_ylabel(r"$d\ln\Sigma/d\ln R$")
axes[1].legend()
axes[1].grid(alpha=0.25)


# Actual extrapolated tails, all anchored to the same endpoint.
for label, slope in candidate_slopes.items():
    if slope >= -1.0:
        continue

    Sigma_tail_plot = (
        Sigma_last_spline
        * (R_tail_plot / R[-1])**slope
    )

    axes[2].plot(
        R_tail_plot,
        Sigma_tail_plot,
        label=f"{label}: {slope:.2f}",
    )

axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_ylabel(r"Extrapolated $\Sigma(R)$")
axes[2].set_xlabel("Projected radius R [pc]")
axes[2].legend()
axes[2].grid(alpha=0.25)


# Effect of each tail on the recovered density.
for label, nu_test in tail_nu.items():
    relative = (
        nu_test
        - nu_reference
    ) / nu_reference

    axes[3].plot(
        Rg_spline,
        relative,
        label=label,
    )

axes[3].axhline(
    0.0,
    linewidth=1.0,
)

axes[3].axvline(
    0.5 * R[-1],
    linestyle="--",
    linewidth=1.0,
)

axes[3].set_xscale("log")
axes[3].set_xlabel("Intrinsic radius r [pc]")
axes[3].set_ylabel(
    r"$(\nu-\nu_{\rm last6})/\nu_{\rm last6}$"
)
axes[3].legend()
axes[3].grid(alpha=0.25)


fig.suptitle(
    "Draco Outer-Tail Diagnostic"
)

fig.tight_layout()

fig.savefig(
    outdir / "Draco_abel_outer_tail_diagnostic.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("Saved:")
print(outdir / "Draco_outer_tail_powerlaw_fits.csv")
print(outdir / "Draco_abel_outer_tail_diagnostic.png")


# ------------------------------------------------------------------
# LINEAR-SPACE OUTER POWER-LAW FIT
#
# Fit the measured outer Sigma values directly:
#
#     Sigma(R) = A * (R / R0)^m
#
# using the actual Sigma_err values.
#
# No log transform of the measurements is used in the fit.
# ------------------------------------------------------------------

print()
print("LINEAR-SPACE OUTER POWER-LAW FIT")
print("--------------------------------")

linear_tail_fits = []

for n_tail_fit in range(4, 9):
    R_fit = R[-n_tail_fit:]
    Sigma_fit = Sigma[-n_tail_fit:]
    Sigma_err_fit = Sigma_err[-n_tail_fit:]

    R0 = float(np.exp(np.mean(np.log(R_fit))))

    def powerlaw_model(R_eval, A, slope):
        return A * (R_eval / R0)**slope

    slope_guess = float(
        np.polyfit(
            np.log(R_fit),
            np.log(Sigma_fit),
            1,
        )[0]
    )

    A_guess = float(
        np.median(Sigma_fit)
    )

    popt, pcov = curve_fit(
        powerlaw_model,
        R_fit,
        Sigma_fit,
        p0=(A_guess, slope_guess),
        sigma=Sigma_err_fit,
        absolute_sigma=True,
        bounds=(
            [0.0, -20.0],
            [np.inf, -1.000001],
        ),
        maxfev=100000,
    )

    A_fit = float(popt[0])
    slope_fit = float(popt[1])

    A_err = float(np.sqrt(pcov[0, 0]))
    slope_err = float(np.sqrt(pcov[1, 1]))

    model_fit = powerlaw_model(
        R_fit,
        A_fit,
        slope_fit,
    )

    residual_sigma_fit = (
        Sigma_fit
        - model_fit
    ) / Sigma_err_fit

    chi2 = float(
        np.sum(
            residual_sigma_fit**2
        )
    )

    dof = n_tail_fit - 2
    reduced_chi2 = (
        chi2 / dof
        if dof > 0
        else np.nan
    )

    linear_tail_fits.append({
        "N_tail": n_tail_fit,
        "R0_pc": R0,
        "A_at_R0": A_fit,
        "A_err": A_err,
        "slope": slope_fit,
        "slope_err": slope_err,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
        "R_first_pc": R_fit[0],
        "R_last_pc": R_fit[-1],
    })


linear_tail_df = pd.DataFrame(
    linear_tail_fits
)

print(
    linear_tail_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

linear_tail_df.to_csv(
    outdir / "Draco_outer_tail_linear_space_fits.csv",
    index=False,
)


# ------------------------------------------------------------------
# Compare linear-space slopes against previous log-space slopes.
# ------------------------------------------------------------------

print()
print("LINEAR VS LOG-SPACE OUTER SLOPES")
print("--------------------------------")

for n_tail_fit in range(4, 9):
    linear_row = linear_tail_df.loc[
        linear_tail_df["N_tail"] == n_tail_fit
    ].iloc[0]

    log_row = tail_fit_df.loc[
        tail_fit_df["N_tail"] == n_tail_fit
    ].iloc[0]

    print(
        f"last {n_tail_fit}:  "
        f"linear={linear_row['slope']: .6f} +/- {linear_row['slope_err']:.6f}   "
        f"log={log_row['slope']: .6f}"
    )


# ------------------------------------------------------------------
# Abel sensitivity using LINEAR-space tail slopes.
# ------------------------------------------------------------------

linear_tail_nu = {}

for n_tail_fit in range(4, 9):
    row = linear_tail_df.loc[
        linear_tail_df["N_tail"] == n_tail_fit
    ].iloc[0]

    slope = float(
        row["slope"]
    )

    nu_test, _ = abel_spline_powerlaw_infinity(
        Rg_spline,
        spline,
        R[-1],
        Sigma_last_spline,
        slope,
    )

    linear_tail_nu[
        f"Last {n_tail_fit}"
    ] = nu_test


reference_label = "Last 6"
nu_reference_linear = linear_tail_nu[
    reference_label
]


print()
print("LINEAR-SPACE TAIL SENSITIVITY")
print("-----------------------------")

inner_region = (
    Rg_spline
    <= 0.5 * R[-1]
)

outer_region = (
    (Rg_spline > 0.5 * R[-1])
    & (Rg_spline <= R[-1])
)

for label, nu_test in linear_tail_nu.items():
    relative = np.abs(
        (
            nu_test
            - nu_reference_linear
        )
        / nu_reference_linear
    )

    n_tail_fit = int(
        label.split()[-1]
    )

    slope = float(
        linear_tail_df.loc[
            linear_tail_df["N_tail"] == n_tail_fit,
            "slope",
        ].iloc[0]
    )

    print()
    print(label)
    print(
        f"  slope:                           "
        f"{slope:.8f}"
    )
    print(
        f"  max relative change R<0.5Rlast: "
        f"{np.max(relative[inner_region]):.6e}"
    )
    print(
        f"  median change R<0.5Rlast:       "
        f"{np.median(relative[inner_region]):.6e}"
    )
    print(
        f"  max relative change outer half: "
        f"{np.max(relative[outer_region]):.6e}"
    )
    print(
        f"  median change outer half:       "
        f"{np.median(relative[outer_region]):.6e}"
    )


# ------------------------------------------------------------------
# Plot linear-space fits.
# ------------------------------------------------------------------

R_tail_dense = np.geomspace(
    R[-8],
    10.0 * R[-1],
    512,
)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 12),
    sharex=False,
)


axes[0].errorbar(
    R[-8:],
    Sigma[-8:],
    yerr=Sigma_err[-8:],
    fmt="o",
    capsize=3,
    label="Odenkirchen outer data",
)

for n_tail_fit in range(4, 9):
    row = linear_tail_df.loc[
        linear_tail_df["N_tail"] == n_tail_fit
    ].iloc[0]

    R0 = float(
        row["R0_pc"]
    )

    A = float(
        row["A_at_R0"]
    )

    slope = float(
        row["slope"]
    )

    Sigma_model = (
        A
        * (R_tail_dense / R0)**slope
    )

    axes[0].plot(
        R_tail_dense,
        Sigma_model,
        label=f"Last {n_tail_fit}: {slope:.2f}",
    )

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_ylabel(r"$\Sigma(R)$")
axes[0].legend()
axes[0].grid(alpha=0.25)


axes[1].errorbar(
    linear_tail_df["N_tail"],
    linear_tail_df["slope"],
    yerr=linear_tail_df["slope_err"],
    fmt="o-",
    capsize=4,
    label="Linear-space fit",
)

axes[1].plot(
    tail_fit_df["N_tail"],
    tail_fit_df["slope"],
    "s--",
    label="Previous log-space fit",
)

axes[1].axhline(
    outer_slope_spline,
    linestyle=":",
    linewidth=1.5,
    label=f"Spline endpoint: {outer_slope_spline:.2f}",
)

axes[1].set_xlabel("Number of outer points")
axes[1].set_ylabel("Power-law slope")
axes[1].legend()
axes[1].grid(alpha=0.25)


for label, nu_test in linear_tail_nu.items():
    relative = (
        nu_test
        - nu_reference_linear
    ) / nu_reference_linear

    axes[2].plot(
        Rg_spline,
        relative,
        label=label,
    )

axes[2].axhline(
    0.0,
    linewidth=1.0,
)

axes[2].axvline(
    0.5 * R[-1],
    linestyle="--",
    linewidth=1.0,
)

axes[2].set_xscale("log")
axes[2].set_xlabel("Intrinsic radius r [pc]")
axes[2].set_ylabel(
    r"$(\nu-\nu_{\rm last6})/\nu_{\rm last6}$"
)
axes[2].legend()
axes[2].grid(alpha=0.25)


fig.suptitle(
    "Draco Linear-Space Outer-Tail Test"
)

fig.tight_layout()

fig.savefig(
    outdir / "Draco_abel_outer_tail_linear_space.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Saved:")
print(outdir / "Draco_outer_tail_linear_space_fits.csv")
print(outdir / "Draco_abel_outer_tail_linear_space.png")
