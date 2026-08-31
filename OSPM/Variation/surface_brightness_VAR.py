"""Surface-brightness response diagnostic for the current OSPM pipeline.

This replaces the old random surface-brightness variant generator.

The experiment is deliberately local to an already selected best theta:
1. build that theta once with the normal OSPM machinery,
2. keep its potential, orbit library, A matrices, and phase volumes fixed,
3. perturb one observed linear Sigma bin at a time by a requested number of
   quoted Sigma_err values,
4. rebuild the normalized light_frac and the current OSPM LOSVD/light targets,
5. rerun only the current expanded-CM weight solver,
6. report how the fit and orbit weights respond.

The default sweep is [-2, -1, +1, +2] sigma in every source photometric bin.
Use denser shifts only around bins that show leverage.

For tracer_constraint_mode="density_3d", this is intentionally a first-pass
sensitivity test. The current Abel-derived density target remains the nominal anchor.
The projected surface-brightness perturbation is mapped onto that target as a
fractional radial-shape change. A bin that shows real leverage should then be
followed by a full perturbed deprojection/tracer-grid rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import OSPM_Physics as physics
except ImportError:
    from OSPM.Physics import OSPM_Physics as physics


_HELPER_NAME = "OSPM_SurfaceBrightnessSensitivity.jl"


def _config_option(config, *names, default=None):
    cfg = dict(config or {})
    obs_cfg = cfg.get("OBSERVABLES", {}) or {}
    if not isinstance(obs_cfg, dict):
        raise TypeError("config['OBSERVABLES'] must be a dict")
    for source in (obs_cfg, cfg):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return default


def _to_numpy_column(value):
    try:
        return np.asarray(value)
    except Exception:
        return np.asarray(list(value))


def _jl_matrix_f64(Main, value, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    nrow, ncol = arr.shape
    Main._sb_sensitivity_matrix_flat = arr.ravel(order="F").tolist()
    Main._sb_sensitivity_matrix_nrow = int(nrow)
    Main._sb_sensitivity_matrix_ncol = int(ncol)
    return Main.seval("reshape(Float64[x for x in _sb_sensitivity_matrix_flat], _sb_sensitivity_matrix_nrow, _sb_sensitivity_matrix_ncol)")


def _jl_vector_f64(Main, value, name):
    arr = np.asarray(value, dtype=np.float64).ravel()
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    Main._sb_sensitivity_vector_f64 = arr.tolist()
    return Main.seval("Float64[x for x in _sb_sensitivity_vector_f64]")


def _jl_vector_bool(Main, value):
    arr = np.asarray(value, dtype=bool).ravel()
    Main._sb_sensitivity_vector_bool = arr.tolist()
    return Main.seval("Bool[x for x in _sb_sensitivity_vector_bool]")


def _jl_surface_brightness_profile(Main, profile):
    required = ("R_pc", "R_inner_pc", "R_outer_pc", "light_frac", "Sigma", "Sigma_err")
    missing = [key for key in required if key not in profile]
    if missing:
        raise KeyError("surface_brightness_profile is missing " + ", ".join(missing))
    Main._sb_sensitivity_R_pc = _jl_vector_f64(Main, profile["R_pc"], "surface_brightness_profile.R_pc")
    Main._sb_sensitivity_R_inner_pc = _jl_vector_f64(Main, profile["R_inner_pc"], "surface_brightness_profile.R_inner_pc")
    Main._sb_sensitivity_R_outer_pc = _jl_vector_f64(Main, profile["R_outer_pc"], "surface_brightness_profile.R_outer_pc")
    Main._sb_sensitivity_light_frac = _jl_vector_f64(Main, profile["light_frac"], "surface_brightness_profile.light_frac")
    Main._sb_sensitivity_Sigma = _jl_vector_f64(Main, profile["Sigma"], "surface_brightness_profile.Sigma")
    Main._sb_sensitivity_Sigma_err = _jl_vector_f64(Main, profile["Sigma_err"], "surface_brightness_profile.Sigma_err")
    return Main.seval("Dict{Symbol,Any}(:R_pc => _sb_sensitivity_R_pc, :R_inner_pc => _sb_sensitivity_R_inner_pc, :R_outer_pc => _sb_sensitivity_R_outer_pc, :light_frac => _sb_sensitivity_light_frac, :Sigma => _sb_sensitivity_Sigma, :Sigma_err => _sb_sensitivity_Sigma_err)")


def summarize_surface_brightness_sensitivity(df):
    """Rank source photometric bins by their largest fixed-theta response."""
    required = {"source_bin", "R_inner_pc", "R_outer_pc", "status", "delta_chi2_losvd", "delta_chi2_light", "weight_tv", "max_abs_tracer_shift_sigma"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Sensitivity table is missing columns: {missing}")
    work = df.loc[df["source_bin"] > 0].copy()
    if work.empty:
        return pd.DataFrame(columns=["source_bin", "R_inner_pc", "R_outer_pc", "max_abs_delta_chi2_losvd", "max_abs_delta_chi2_light", "max_weight_tv", "max_abs_tracer_shift_sigma", "solver_failures"])
    work["abs_delta_chi2_losvd"] = np.abs(pd.to_numeric(work["delta_chi2_losvd"], errors="coerce"))
    work["abs_delta_chi2_light"] = np.abs(pd.to_numeric(work["delta_chi2_light"], errors="coerce"))
    work["solver_failure"] = work["status"].ne("ok").astype(int)
    summary = work.groupby(["source_bin", "R_inner_pc", "R_outer_pc"], as_index=False).agg(
        max_abs_delta_chi2_losvd=("abs_delta_chi2_losvd", "max"),
        max_abs_delta_chi2_light=("abs_delta_chi2_light", "max"),
        max_weight_tv=("weight_tv", "max"),
        max_abs_tracer_shift_sigma=("max_abs_tracer_shift_sigma", "max"),
        solver_failures=("solver_failure", "sum"),
    )
    return summary.sort_values(["solver_failures", "max_abs_delta_chi2_losvd", "max_weight_tv"], ascending=[False, False, False], na_position="last").reset_index(drop=True)


def run_surface_brightness_sensitivity(obs, theta, *, halo_type, config=None, shifts_sigma=(-2.0, -1.0, 1.0, 2.0), output_csv=None, include_nominal=True, seed=0):
    """Run the fixed-theta surface-brightness response test.

    Parameters
    ----------
    obs:
        The already constructed OSPM stellar observable object.
    theta:
        Best-model theta in the normal external OSPM parameterization.
    halo_type:
        Current OSPM halo type.
    config:
        Current merged OSPM config.
    shifts_sigma:
        Per-source-bin Sigma/Sigma_err shifts to test.
    output_csv:
        Optional CSV path for the diagnostic table.
    include_nominal:
        Include the unperturbed best-model weight solve as row zero.
    seed:
        Common solver seed used for nominal and all perturbed solves.
    """
    cfg = dict(config or {})
    shifts = np.asarray(shifts_sigma, dtype=float).ravel()
    if shifts.size == 0:
        raise ValueError("shifts_sigma cannot be empty")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("shifts_sigma contains non-finite values")

    threads_per_model = int(_config_option(cfg, "THREADS_PER_MODEL", "CPUS_PER_MODEL", "threads_per_model", default=8))
    physics._jl_init(threads_per_model=threads_per_model)
    Main = physics._Main
    helper_path = Path(__file__).resolve().with_name(_HELPER_NAME)
    loaded = bool(Main.seval("isdefined(Main.OSPMPhysicsSpherical, :run_surface_brightness_sensitivity)"))
    if not loaded:
        if not helper_path.is_file():
            raise FileNotFoundError(f"Surface-brightness Julia helper not found: {helper_path}")
        helper_literal = json.dumps(str(helper_path))
        Main.seval(f"Base.include(Main.OSPMPhysicsSpherical, {helper_literal})")

    A, meta = physics.build_A_matrix_from_theta(obs, theta, halo_type=halo_type, diag=True, config=cfg)
    A = np.asarray(A, dtype=float)
    meta = dict(meta)

    Nlosvd = int(meta["Nlosvd"])
    Nlight = int(meta["Nlight"])
    Nvbin = int(meta["Nvbin"])
    if A.ndim != 2:
        raise RuntimeError("OSPM A matrix is not two-dimensional")
    if A.shape[0] != Nlosvd + Nlight:
        raise RuntimeError(f"A matrix rows {A.shape[0]} do not match Nlosvd+Nlight={Nlosvd + Nlight}")

    A_losvd = np.ascontiguousarray(A[:Nlosvd, :], dtype=float)
    A_light = np.ascontiguousarray(A[Nlosvd:, :], dtype=float)
    light_target = np.asarray(meta["light_target"], dtype=float).ravel()
    light_sigma = np.asarray(meta["light_sigma"], dtype=float).ravel()
    losvd_target = np.asarray(meta["losvd_target"], dtype=float).ravel()
    losvd_sigma = np.asarray(meta["losvd_sigma"], dtype=float).ravel()
    spatial_edges = np.asarray(meta["spatial_edges"], dtype=float).ravel()
    light_edges = np.asarray(meta["light_edges"], dtype=float).ravel()
    velocity_edges = np.asarray(meta["velocity_edges"], dtype=float).ravel()
    wphase = np.asarray(meta["wphase"], dtype=float).ravel()
    tracer_constraint_mode = str(meta["tracer_constraint_mode"])

    R, v, ve = physics._get_obs_arrays(obs)
    valid = physics._get_valid_vlos(obs, R, v, ve)
    surface_brightness_profile = physics._get_surface_brightness_profile(obs=obs, config=cfg)

    A_light_jl = _jl_matrix_f64(Main, A_light, "A_light")
    A_losvd_jl = _jl_matrix_f64(Main, A_losvd, "A_losvd")
    light_target_jl = _jl_vector_f64(Main, light_target, "light_target")
    light_sigma_jl = _jl_vector_f64(Main, light_sigma, "light_sigma")
    losvd_target_jl = _jl_vector_f64(Main, losvd_target, "losvd_target")
    losvd_sigma_jl = _jl_vector_f64(Main, losvd_sigma, "losvd_sigma")
    R_jl = _jl_vector_f64(Main, R, "R_star_m")
    valid_jl = _jl_vector_bool(Main, valid)
    v_jl = _jl_vector_f64(Main, v, "v_star_mps")
    ve_jl = _jl_vector_f64(Main, ve, "verr_star_mps")
    spatial_edges_jl = _jl_vector_f64(Main, spatial_edges, "spatial_edges")
    light_edges_jl = _jl_vector_f64(Main, light_edges, "light_edges")
    velocity_edges_jl = _jl_vector_f64(Main, velocity_edges, "velocity_edges")
    surface_brightness_profile_jl = _jl_surface_brightness_profile(Main, surface_brightness_profile)
    wphase_jl = _jl_vector_f64(Main, wphase, "wphase")
    shifts_jl = _jl_vector_f64(Main, shifts, "shifts_sigma")

    jl_result = Main.OSPMPhysicsSpherical.run_surface_brightness_sensitivity(
        A_light_jl,
        A_losvd_jl,
        light_target_jl,
        light_sigma_jl,
        losvd_target_jl,
        losvd_sigma_jl,
        R_jl,
        valid_jl,
        v_jl,
        ve_jl,
        spatial_edges_jl,
        light_edges_jl,
        velocity_edges_jl,
        surface_brightness_profile_jl,
        wphase_jl,
        Nvbin=Nvbin,
        tracer_constraint_mode=tracer_constraint_mode,
        shifts_sigma=shifts_jl,
        alphat=float(_config_option(cfg, "KARL_ALPHAT", "alphat", default=1.0)),
        light_rel_tol=float(_config_option(cfg, "KARL_LIGHT_REL_TOL", "light_rel_tol", default=0.01)),
        light_sigma_tol=float(_config_option(cfg, "KARL_LIGHT_SIGMA_TOL", "light_sigma_tol", default=2.0)),
        delta_chi2_iter_tol=float(_config_option(cfg, "KARL_DELTA_CHI2_ITER_TOL", "delta_chi2_iter_tol", default=0.3)),
        maxiter=int(_config_option(cfg, "KARL_MAXITER", "MAXITER", "maxiter", default=60)),
        entropy_floor=float(_config_option(cfg, "ENTROPY_FLOOR", "entropy_floor", default=1e-30)),
        apfac=float(_config_option(cfg, "KARL_APFAC", "apfac", default=0.01)),
        seed=int(seed),
        include_nominal=bool(include_nominal),
    )

    columns = dict(jl_result)
    data = {str(name): _to_numpy_column(values) for name, values in columns.items()}
    df = pd.DataFrame(data)

    theta_values = np.asarray(theta, dtype=float).ravel()
    parameter_names = list(cfg.get("PARAMETER_NAMES", []))
    if len(parameter_names) == theta_values.size:
        for name, value in zip(parameter_names, theta_values):
            df[name] = float(value)
    else:
        for index, value in enumerate(theta_values):
            df[f"theta_{index}"] = float(value)
    df["halo_type"] = str(halo_type)

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SB SENSITIVITY] wrote {len(df)} rows -> {output_path}")

    return df


__all__ = ["run_surface_brightness_sensitivity", "summarize_surface_brightness_sensitivity"]



# Surface-brightness sensitivity diagnostic for the current OSPM pipeline.
# Include this file into Main.OSPMPhysicsSpherical after OSPM_Physics_Spherical.jl loads.
#
# The diagnostic keeps the potential, orbit library, A matrices, and phase volumes fixed.
# It perturbs one observed linear Sigma bin at a time within its quoted Sigma_err,
# applies the corresponding fractional change to the authoritative light_frac, renormalizes it, regenerates the current OSPM targets,
# and reruns only the expanded-CM weight solve.

const DEFAULT_SB_SENSITIVITY_SHIFTS = Float64[-2.0, -1.0, 1.0, 2.0]

@inline function _sb_diag_value(diag, name::Symbol, default)
    diag === nothing && return default
    return hasproperty(diag, name) ? getproperty(diag, name) : default
end

function _sb_profile_arrays(surface_brightness_profile)
    p = normalize_surface_brightness_profile(surface_brightness_profile)
    p === nothing && error("surface_brightness_profile is required")
    for key in (:R_inner_pc, :R_outer_pc, :light_frac, :Sigma, :Sigma_err)
        haskey(p, key) || error("surface_brightness_profile is missing $(key)")
    end
    rin = Float64.(p[:R_inner_pc])
    rout = Float64.(p[:R_outer_pc])
    light_frac = Float64.(p[:light_frac])
    Sigma = Float64.(p[:Sigma])
    Sigma_err = Float64.(p[:Sigma_err])
    n = length(Sigma)
    n > 0 || error("surface_brightness_profile has no Sigma bins")
    length(rin) == n || error("R_inner_pc length does not match Sigma")
    length(rout) == n || error("R_outer_pc length does not match Sigma")
    length(light_frac) == n || error("light_frac length does not match Sigma")
    length(Sigma_err) == n || error("Sigma_err length does not match Sigma")
    all(isfinite, rin) || error("R_inner_pc contains nonfinite values")
    all(isfinite, rout) || error("R_outer_pc contains nonfinite values")
    all(isfinite, light_frac) || error("light_frac contains nonfinite values")
    all(isfinite, Sigma) || error("Sigma contains nonfinite values")
    all(isfinite, Sigma_err) || error("Sigma_err contains nonfinite values")
    all(rin .>= 0.0) || error("R_inner_pc contains negative values")
    all(rout .> rin) || error("Every R_outer_pc must exceed R_inner_pc")
    all(Sigma .> 0.0) || error("Sigma must be strictly positive for this sensitivity diagnostic")
    all(Sigma_err .>= 0.0) || error("Sigma_err cannot be negative")
    return p, rin, rout, light_frac, Sigma, Sigma_err
end

function _sb_normalize_light_frac(light_frac::Vector{Float64}, total_reference::Float64)
    all(isfinite, light_frac) || error("light_frac contains nonfinite values")
    all(light_frac .>= 0.0) || error("light_frac contains negative values")
    total = sum(light_frac)
    isfinite(total) && total > 0.0 || error("light_frac has nonpositive total")
    isfinite(total_reference) && total_reference > 0.0 || error("Reference light normalization is nonpositive")
    return light_frac .* (total_reference / total)
end

function _sb_candidate_profile(p::Dict{Symbol,Any}, light_frac::Vector{Float64}, Sigma::Vector{Float64}, Sigma_err::Vector{Float64}, source_bin::Int, shift_sigma::Float64)
    1 <= source_bin <= length(Sigma) || error("source_bin is outside the surface-brightness profile")
    Sigma_new = copy(Sigma)
    Sigma_new[source_bin] += shift_sigma * Sigma_err[source_bin]
    if !(isfinite(Sigma_new[source_bin]) && Sigma_new[source_bin] > 0.0)
        return nothing, NaN, Float64[]
    end
    total_reference = sum(light_frac)
    factor = Sigma_new[source_bin] / Sigma[source_bin]
    light_frac_new = copy(light_frac)
    light_frac_new[source_bin] *= factor
    light_frac_new = _sb_normalize_light_frac(light_frac_new, total_reference)
    candidate = copy(p)
    candidate[:Sigma] = Sigma_new
    candidate[:Sigma_err] = copy(Sigma_err)
    candidate[:light_frac] = light_frac_new
    return candidate, Sigma_new[source_bin], light_frac_new
end

function _sb_resolve_candidate_tracer(mode::Symbol, nominal_light_target::Vector{Float64}, projected_nominal::Vector{Float64}, projected_candidate::Vector{Float64}, projected_sigma_candidate::Vector{Float64})
    mode === :projected_light && return copy(projected_candidate), copy(projected_sigma_candidate)
    mode === :density_3d || error("Unsupported tracer_constraint_mode=$(mode)")
    length(nominal_light_target) == length(projected_nominal) == length(projected_candidate) == length(projected_sigma_candidate) || error("Projected and density tracer bins do not match")
    denom = max.(abs.(projected_nominal), 1.0e-12)
    ratio = projected_candidate ./ denom
    candidate = nominal_light_target .* ratio
    _normalize_nonnegative!(candidate)
    frac_sigma = max.(abs.(projected_sigma_candidate), 1.0e-12) ./ max.(abs.(projected_candidate), 1.0e-12)
    candidate_sigma = max.(abs.(candidate) .* frac_sigma, max.(abs.(projected_sigma_candidate), 1.0e-12))
    return candidate, candidate_sigma
end

function _sb_weight_metrics(w::Vector{Float64}, w_nominal::Union{Nothing,Vector{Float64}}=nothing)
    sw = sum(w)
    if !(isfinite(sw) && sw > 0.0 && all(isfinite, w))
        return 0, NaN, NaN, NaN
    end
    wn = w ./ sw
    n_nonzero = count(>(1.0e-12), wn)
    denom = sum(abs2, wn)
    effective_N = isfinite(denom) && denom > 0.0 ? 1.0 / denom : NaN
    max_fraction = isempty(wn) ? NaN : maximum(wn)
    weight_tv = NaN
    if w_nominal !== nothing && length(w_nominal) == length(w)
        sw0 = sum(w_nominal)
        if isfinite(sw0) && sw0 > 0.0 && all(isfinite, w_nominal)
            w0n = w_nominal ./ sw0
            weight_tv = 0.5 * sum(abs.(wn .- w0n))
        end
    end
    return n_nonzero, effective_N, max_fraction, weight_tv
end

function _sb_light_residual_metrics(A_light::Matrix{Float64}, w::Vector{Float64}, target::Vector{Float64}, sigma::Vector{Float64})
    if length(w) != size(A_light, 2) || !all(isfinite, w)
        return NaN, 0
    end
    residual = abs.(A_light * w .- target) ./ max.(abs.(sigma), 1.0e-12)
    isempty(residual) && return 0.0, 0
    value, idx = findmax(residual)
    return value, idx
end

function _sb_init_output()
    out = Dict{String,Any}()
    out["source_bin"] = Int[]
    out["R_inner_pc"] = Float64[]
    out["R_outer_pc"] = Float64[]
    out["requested_sigma"] = Float64[]
    out["Sigma_nominal"] = Float64[]
    out["Sigma_candidate"] = Float64[]
    out["Sigma_err"] = Float64[]
    out["source_light_frac_nominal"] = Float64[]
    out["source_light_frac_candidate"] = Float64[]
    out["max_abs_tracer_shift_sigma"] = Float64[]
    out["losvd_target_l1_fraction"] = Float64[]
    out["solver_ok"] = Bool[]
    out["chi2_losvd"] = Float64[]
    out["delta_chi2_losvd"] = Float64[]
    out["chi2_light"] = Float64[]
    out["delta_chi2_light"] = Float64[]
    out["max_light_sigma_residual"] = Float64[]
    out["worst_light_bin"] = Int[]
    out["weight_tv"] = Float64[]
    out["N_nonzero_weights"] = Int[]
    out["effective_N_orbits"] = Float64[]
    out["max_weight_fraction"] = Float64[]
    out["iterations"] = Int[]
    out["failure_reason"] = String[]
    out["tracer_constraint_mode"] = String[]
    out["status"] = String[]
    return out
end

function _sb_push_result!(out::Dict{String,Any}, source_bin::Int, rin::Float64, rout::Float64, requested_sigma::Float64, Sigma_nominal::Float64, Sigma_candidate::Float64, Sigma_err::Float64, frac_nominal::Float64, frac_candidate::Float64, max_abs_tracer_shift_sigma::Float64, losvd_target_l1_fraction::Float64, solver_ok::Bool, chi2_losvd::Float64, delta_chi2_losvd::Float64, chi2_light::Float64, delta_chi2_light::Float64, max_light_sigma_residual::Float64, worst_light_bin::Int, weight_tv::Float64, N_nonzero_weights::Int, effective_N_orbits::Float64, max_weight_fraction::Float64, iterations::Int, failure_reason::String, tracer_constraint_mode::String, status::String)
    push!(out["source_bin"], source_bin)
    push!(out["R_inner_pc"], rin)
    push!(out["R_outer_pc"], rout)
    push!(out["requested_sigma"], requested_sigma)
    push!(out["Sigma_nominal"], Sigma_nominal)
    push!(out["Sigma_candidate"], Sigma_candidate)
    push!(out["Sigma_err"], Sigma_err)
    push!(out["source_light_frac_nominal"], frac_nominal)
    push!(out["source_light_frac_candidate"], frac_candidate)
    push!(out["max_abs_tracer_shift_sigma"], max_abs_tracer_shift_sigma)
    push!(out["losvd_target_l1_fraction"], losvd_target_l1_fraction)
    push!(out["solver_ok"], solver_ok)
    push!(out["chi2_losvd"], chi2_losvd)
    push!(out["delta_chi2_losvd"], delta_chi2_losvd)
    push!(out["chi2_light"], chi2_light)
    push!(out["delta_chi2_light"], delta_chi2_light)
    push!(out["max_light_sigma_residual"], max_light_sigma_residual)
    push!(out["worst_light_bin"], worst_light_bin)
    push!(out["weight_tv"], weight_tv)
    push!(out["N_nonzero_weights"], N_nonzero_weights)
    push!(out["effective_N_orbits"], effective_N_orbits)
    push!(out["max_weight_fraction"], max_weight_fraction)
    push!(out["iterations"], iterations)
    push!(out["failure_reason"], failure_reason)
    push!(out["tracer_constraint_mode"], tracer_constraint_mode)
    push!(out["status"], status)
    return nothing
end

function run_surface_brightness_sensitivity(A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, light_target_reference::Vector{Float64}, light_sigma_reference::Vector{Float64}, losvd_target_reference::Vector{Float64}, losvd_sigma_reference::Vector{Float64}, R_star_m::Vector{Float64}, valid_vlos::Vector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, spatial_edges::Vector{Float64}, light_edges::Vector{Float64}, velocity_edges::Vector{Float64}, surface_brightness_profile, wphase::Vector{Float64}; Nvbin::Int, tracer_constraint_mode="projected_light", shifts_sigma::Vector{Float64}=DEFAULT_SB_SENSITIVITY_SHIFTS, alphat::Float64=DEFAULT_KARL_ALPHAT, light_rel_tol::Float64=DEFAULT_KARL_LIGHT_REL_TOL, light_sigma_tol::Float64=2.0, delta_chi2_iter_tol::Float64=DEFAULT_KARL_DELTA_CHI2_ITER_TOL, maxiter::Int=DEFAULT_KARL_MAXITER, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, apfac::Float64=DEFAULT_KARL_APFAC, seed::Integer=0, target_rtol::Float64=1.0e-8, target_atol::Float64=1.0e-12, include_nominal::Bool=true)
    Nspatial = length(spatial_edges) - 1
    Nlight = size(A_light, 1)
    Nlosvd = size(A_losvd, 1)
    Nspatial > 0 || error("spatial_edges has fewer than two edges")
    length(light_edges) == Nlight + 1 || error("light_edges length does not match A_light")
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match A_losvd")
    size(A_light, 2) == size(A_losvd, 2) || error("A_light and A_losvd orbit counts do not match")
    length(wphase) == size(A_light, 2) || error("wphase length does not match orbit columns")
    length(light_target_reference) == Nlight || error("light_target_reference length does not match A_light")
    length(light_sigma_reference) == Nlight || error("light_sigma_reference length does not match A_light")
    length(losvd_target_reference) == Nlosvd || error("losvd_target_reference length does not match A_losvd")
    length(losvd_sigma_reference) == Nlosvd || error("losvd_sigma_reference length does not match A_losvd")
    mode = _normalize_tracer_constraint_mode(tracer_constraint_mode)
    p, rin, rout, light_frac_input, Sigma, Sigma_err = _sb_profile_arrays(surface_brightness_profile)
    light_frac_total = sum(light_frac_input)
    isfinite(light_frac_total) && light_frac_total > 0.0 || error("Current light_frac has nonpositive total")
    light_frac_nominal = _sb_normalize_light_frac(copy(light_frac_input), light_frac_total)
    losvd_nominal_build, losvd_sigma_nominal_build, projected_nominal, projected_sigma_nominal, _ = observed_targets_karl(R_star_m, valid_vlos, v_star_mps, verr_star_mps, spatial_edges, velocity_edges; surface_brightness_profile=p, light_edges=light_edges)
    isapprox(losvd_nominal_build, losvd_target_reference; rtol=target_rtol, atol=target_atol) || error("Rebuilt nominal LOSVD target does not match the orbit-library diagnostic target")
    isapprox(losvd_sigma_nominal_build, losvd_sigma_reference; rtol=target_rtol, atol=target_atol) || error("Rebuilt nominal LOSVD sigma does not match the orbit-library diagnostic sigma")
    if mode === :projected_light
        isapprox(projected_nominal, light_target_reference; rtol=target_rtol, atol=target_atol) || error("Rebuilt nominal projected-light target does not match the orbit-library diagnostic target")
        isapprox(projected_sigma_nominal, light_sigma_reference; rtol=target_rtol, atol=target_atol) || error("Rebuilt nominal projected-light sigma does not match the orbit-library diagnostic sigma")
    end
    seed_use = UInt(seed)
    w_nominal, ok_nominal, diag_nominal = solve_weights_karl_expanded_cm(A_light, A_losvd, light_target_reference, light_sigma_reference, losvd_target_reference, losvd_sigma_reference; Nspatial=Nspatial, Nvbin=Nvbin, alphat=alphat, light_rel_tol=light_rel_tol, light_sigma_tol=light_sigma_tol, delta_chi2_iter_tol=delta_chi2_iter_tol, wphase=wphase, maxiter=maxiter, seed=seed_use, entropy_floor=entropy_floor, apfac=apfac, return_diag=true)
    ok_nominal || error("Nominal best-model weight solve failed inside the surface-brightness diagnostic: $(_sb_diag_value(diag_nominal, :failure_reason, :unknown))")
    chi2_losvd_nominal = chi2_block(A_losvd, w_nominal, losvd_target_reference, losvd_sigma_reference)
    chi2_light_nominal = chi2_block(A_light, w_nominal, light_target_reference, light_sigma_reference)
    nominal_light_sigma_residual, nominal_worst_light_bin = _sb_light_residual_metrics(A_light, w_nominal, light_target_reference, light_sigma_reference)
    nominal_nonzero, nominal_effective, nominal_max_fraction, _ = _sb_weight_metrics(w_nominal)
    out = _sb_init_output()
    if include_nominal
        _sb_push_result!(out, 0, NaN, NaN, 0.0, NaN, NaN, NaN, NaN, NaN, 0.0, 0.0, true, chi2_losvd_nominal, 0.0, chi2_light_nominal, 0.0, nominal_light_sigma_residual, nominal_worst_light_bin, 0.0, nominal_nonzero, nominal_effective, nominal_max_fraction, Int(_sb_diag_value(diag_nominal, :iterations, 0)), String(_sb_diag_value(diag_nominal, :failure_reason, :none)), String(mode), "nominal")
    end
    losvd_norm = max(sum(abs, losvd_target_reference), 1.0e-12)
    for source_bin in eachindex(Sigma)
        for requested_sigma in shifts_sigma
            shift = Float64(requested_sigma)
            isfinite(shift) || continue
            shift == 0.0 && continue
            candidate_profile, Sigma_candidate, light_frac_candidate = _sb_candidate_profile(p, light_frac_nominal, Sigma, Sigma_err, source_bin, shift)
            if candidate_profile === nothing
                _sb_push_result!(out, source_bin, rin[source_bin], rout[source_bin], shift, Sigma[source_bin], NaN, Sigma_err[source_bin], light_frac_nominal[source_bin], NaN, NaN, NaN, false, NaN, NaN, NaN, NaN, NaN, 0, NaN, 0, NaN, NaN, 0, "candidate_sigma_nonpositive", String(mode), "invalid_profile")
                continue
            end
            losvd_candidate, losvd_sigma_candidate, projected_candidate, projected_sigma_candidate, _ = observed_targets_karl(R_star_m, valid_vlos, v_star_mps, verr_star_mps, spatial_edges, velocity_edges; surface_brightness_profile=candidate_profile, light_edges=light_edges)
            light_target_candidate, light_sigma_candidate = _sb_resolve_candidate_tracer(mode, light_target_reference, projected_nominal, projected_candidate, projected_sigma_candidate)
            tracer_shift = maximum(abs.(light_target_candidate .- light_target_reference) ./ max.(abs.(light_sigma_reference), 1.0e-12))
            losvd_l1_fraction = sum(abs.(losvd_candidate .- losvd_target_reference)) / losvd_norm
            w, ok, diag = solve_weights_karl_expanded_cm(A_light, A_losvd, light_target_candidate, light_sigma_candidate, losvd_candidate, losvd_sigma_candidate; Nspatial=Nspatial, Nvbin=Nvbin, alphat=alphat, light_rel_tol=light_rel_tol, light_sigma_tol=light_sigma_tol, delta_chi2_iter_tol=delta_chi2_iter_tol, wphase=wphase, maxiter=maxiter, seed=seed_use, entropy_floor=entropy_floor, apfac=apfac, return_diag=true)
            failure_reason = String(_sb_diag_value(diag, :failure_reason, ok ? :none : :unknown))
            iterations = Int(_sb_diag_value(diag, :iterations, 0))
            if ok && length(w) == size(A_light, 2) && all(isfinite, w)
                chi2_losvd = chi2_block(A_losvd, w, losvd_candidate, losvd_sigma_candidate)
                chi2_light = chi2_block(A_light, w, light_target_candidate, light_sigma_candidate)
                max_light_sigma_residual, worst_light_bin = _sb_light_residual_metrics(A_light, w, light_target_candidate, light_sigma_candidate)
                N_nonzero_weights, effective_N_orbits, max_weight_fraction, weight_tv = _sb_weight_metrics(w, w_nominal)
                _sb_push_result!(out, source_bin, rin[source_bin], rout[source_bin], shift, Sigma[source_bin], Sigma_candidate, Sigma_err[source_bin], light_frac_nominal[source_bin], light_frac_candidate[source_bin], tracer_shift, losvd_l1_fraction, true, chi2_losvd, chi2_losvd - chi2_losvd_nominal, chi2_light, chi2_light - chi2_light_nominal, max_light_sigma_residual, worst_light_bin, weight_tv, N_nonzero_weights, effective_N_orbits, max_weight_fraction, iterations, failure_reason, String(mode), "ok")
            else
                max_light_sigma_residual, worst_light_bin = _sb_light_residual_metrics(A_light, w, light_target_candidate, light_sigma_candidate)
                N_nonzero_weights, effective_N_orbits, max_weight_fraction, weight_tv = _sb_weight_metrics(w, w_nominal)
                _sb_push_result!(out, source_bin, rin[source_bin], rout[source_bin], shift, Sigma[source_bin], Sigma_candidate, Sigma_err[source_bin], light_frac_nominal[source_bin], light_frac_candidate[source_bin], tracer_shift, losvd_l1_fraction, false, NaN, NaN, NaN, NaN, max_light_sigma_residual, worst_light_bin, weight_tv, N_nonzero_weights, effective_N_orbits, max_weight_fraction, iterations, failure_reason, String(mode), "solver_failed")
            end
        end
    end
    out["nominal_chi2_losvd"] = fill(chi2_losvd_nominal, length(out["source_bin"]))
    out["nominal_chi2_light"] = fill(chi2_light_nominal, length(out["source_bin"]))
    return out
end
