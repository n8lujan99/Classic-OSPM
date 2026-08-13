# ============================================================================
# OSPM ACTIVE-SET / SPEAR STANDALONE TEST HARNESS
# Purpose:
#   Exercise the current Karl-style expanded-Cm SPEAR machinery outside OSPM.
#   The active-set algorithm below intentionally keeps the CURRENT release rule
#   unchanged.  Extra diagnostics compare that rule with the full stationarity
#   expression, detect active-mask revisits, and allow inherited-bound A/B tests.
# Run: julia OSPM_active_set_standalone.jl
# Or include it and call run_state_tests(...) with real OSPM matrices/vectors.
# ============================================================================
using LinearAlgebra
using Random
using Printf

const DEFAULT_KARL_ALPHAT = 1.0
const DEFAULT_KARL_ENTROPY_FLOOR = 1.0e-30
const DEFAULT_KARL_APFAC = 0.01
const DEFAULT_KARL_STEP_SAFETY = 0.90
const DEFAULT_KARL_SPEAR_RCOND_WARN = 1.0e-12
const DEFAULT_KARL_INVALID_SIGMA_SENTINEL = -666.0

@inline function _safe_positive(x::Float64; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    return isfinite(x) && x > floor ? x : floor
end

@inline function _spear_safe_ddS(x::Float64; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    isfinite(x) || return -floor
    x < 0.0 || error("SPEAR requires a strictly negative entropy Hessian")
    return x > -floor ? -floor : x
end

# ============================================================================
# LOW-LEVEL SPEAR PRIMITIVES
# ============================================================================

function karl_spear_build_Am(Cm::Matrix{Float64}, ddS::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Narr, nvar = size(Cm)
    length(ddS) == nvar || error("ddS length must match Cm columns")
    invdd = Vector{Float64}(undef, nvar)
    @inbounds for j in 1:nvar
        invdd[j] = 1.0 / _spear_safe_ddS(ddS[j]; floor=floor)
    end
    return Cm * Diagonal(invdd) * transpose(Cm)
end

function karl_spear_rhs!(delY::Vector{Float64}, Cm::Matrix{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Narr, nvar = size(Cm)
    length(delY) == Narr || error("delY length must match Cm rows")
    length(dS) == nvar || error("dS length must match Cm columns")
    length(ddS) == nvar || error("ddS length must match Cm columns")
    tmp = similar(dS)
    @inbounds for j in 1:nvar
        tmp[j] = dS[j] / _spear_safe_ddS(ddS[j]; floor=floor)
    end
    delY .+= Cm * tmp
    return delY
end

function karl_spear_delta_w(Cm::Matrix{Float64}, lambda::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Narr, nvar = size(Cm)
    length(lambda) == Narr || error("lambda length must match Cm rows")
    length(dS) == nvar || error("dS length must match Cm columns")
    length(ddS) == nvar || error("ddS length must match Cm columns")
    dw = transpose(Cm) * lambda
    @inbounds for j in 1:nvar
        dw[j] = (dw[j] - dS[j]) / _spear_safe_ddS(ddS[j]; floor=floor)
    end
    return dw
end

function _solve_spear_system(Am::Matrix{Float64}, rhs::Vector{Float64})
    size(Am, 1) == size(Am, 2) || error("SPEAR Am must be square")
    size(Am, 1) == length(rhs) || error("SPEAR Am and rhs dimensions do not match")
    isempty(rhs) && return Float64[]
    all(isfinite, Am) || error("SPEAR Am contains nonfinite values")
    all(isfinite, rhs) || error("SPEAR rhs contains nonfinite values")

    F = bunchkaufman(Symmetric(Am); check=false)
    lambda = F \ rhs
    all(isfinite, lambda) || error("Karl-like SPEAR solve produced nonfinite multipliers")
    return Vector{Float64}(lambda)
end

# ============================================================================
# EXPANDED-CM / LOSVD HELPERS
# ============================================================================


function _expanded_light_implies_normalization(A_light::Matrix{Float64}, light_target::Vector{Float64}; tol::Float64=1e-12)
    size(A_light, 1) == length(light_target) || return false
    isempty(light_target) && return false
    orbit_light_sums = vec(sum(A_light, dims=1))
    return maximum(abs.(orbit_light_sums .- 1.0)) <= tol && abs(sum(light_target) - 1.0) <= tol
end

function build_expanded_Cm_with_losvd_slack(A_light::Matrix{Float64}, A_losvd::Matrix{Float64}; enforce_normalization::Bool=true)
    Nlight, Norbit = size(A_light)
    Nlosvd, Norbit2 = size(A_losvd)
    Norbit == Norbit2 || error("A_light and A_losvd must have the same number of orbit columns")
    Narr = Nlight + Nlosvd + (enforce_normalization ? 1 : 0)
    Nslack = Nlosvd
    Cm = zeros(Float64, Narr, Norbit + Nslack)
    Cm[1:Nlight, 1:Norbit] .= A_light
    losvd_rows = (Nlight + 1):(Nlight + Nlosvd)
    Cm[losvd_rows, 1:Norbit] .= A_losvd
    Cm[losvd_rows, (Norbit + 1):(Norbit + Nslack)] .= Matrix{Float64}(I, Nlosvd, Nlosvd)
    if enforce_normalization
        Cm[Narr, 1:Norbit] .= 1.0
    end
    return Cm
end

function build_expanded_target(light_target::Vector{Float64}, losvd_target::Vector{Float64}; enforce_normalization::Bool=true)
    return enforce_normalization ? vcat(light_target, losvd_target, 1.0) : vcat(light_target, losvd_target)
end

function build_expanded_weights_initial(w_orbit::Vector{Float64}, A_losvd::Matrix{Float64}, losvd_target::Vector{Float64})
    Nlosvd, Norbit = size(A_losvd)
    length(w_orbit) == Norbit || error("w_orbit length does not match A_losvd columns")
    length(losvd_target) == Nlosvd || error("losvd_target length does not match A_losvd rows")
    slack = losvd_target .- A_losvd * w_orbit
    return vcat(w_orbit, slack)
end

function karl_losvd_fracnew_state(A_losvd::Matrix{Float64}, w::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}, Nspatial::Int, Nvbin::Int; invalid_sigma_sentinel::Float64=DEFAULT_KARL_INVALID_SIGMA_SENTINEL)
    Nlosvd, Norbit = size(A_losvd)
    length(w) == Norbit || error("w length does not match A_losvd columns")
    length(losvd_target) == Nlosvd || error("losvd_target length does not match A_losvd rows")
    length(losvd_sigma) == Nlosvd || error("losvd_sigma length does not match A_losvd rows")
    Nspatial > 0 || error("Nspatial must be positive")
    Nvbin > 0 || error("Nvbin must be positive")
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match Nlosvd")

    model = A_losvd * w
    all(isfinite, model) || error("Karl LOSVD model contains nonfinite values")

    fracnew = ones(Float64, Nspatial)
    effective_target = similar(losvd_target)
    effective_sigma = similar(losvd_sigma)
    residual = similar(losvd_target)
    chi_by_spatial = zeros(Float64, Nspatial)

    @inbounds for ib in 1:Nspatial
        rows = ((ib - 1) * Nvbin + 1):(ib * Nvbin)
        sumt = 0.0
        sumt2 = 0.0
        for row in rows
            sumt += model[row]
            if losvd_sigma[row] != invalid_sigma_sentinel
                sumt2 += model[row]
            end
        end
        frac = abs(sumt) > eps(Float64) ? sumt2 / sumt : 1.0
        isfinite(frac) || error("Karl fracnew became nonfinite in spatial bin $ib")
        fracnew[ib] = frac
        for row in rows
            effective_target[row] = losvd_target[row] * frac
            effective_sigma[row] = losvd_sigma[row] == invalid_sigma_sentinel ? 1.0e6 : max(abs(losvd_sigma[row] * frac), 1.0e-12)
            residual[row] = effective_target[row] - model[row]
            rr = residual[row] / effective_sigma[row]
            chi_by_spatial[ib] += rr * rr
        end
    end

    return (model=model, fracnew=fracnew, effective_target=effective_target, effective_sigma=effective_sigma, residual=residual, chi_by_spatial=chi_by_spatial, chi_total=sum(chi_by_spatial))
end

function build_expanded_entropy_derivatives(w_all::Vector{Float64}, Norbit::Int, wphase_orbit::Vector{Float64}, losvd_residual::Vector{Float64}, losvd_sigma_effective::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Nvar = length(w_all)
    Nlosvd = Nvar - Norbit
    Nlosvd >= 0 || error("Norbit cannot exceed total variable count")
    length(wphase_orbit) == Norbit || error("wphase length must match Norbit")
    length(losvd_residual) == Nlosvd || error("LOSVD residual length must match slack count")
    length(losvd_sigma_effective) == Nlosvd || error("effective LOSVD sigma length must match slack count")
    dS = zeros(Float64, Nvar)
    ddS = zeros(Float64, Nvar)
    entropy = 0.0
    chi_slack = 0.0

    @inbounds for j in 1:Norbit
        wj = _safe_positive(w_all[j]; floor=entropy_floor)
        pj = wphase_orbit[j]
        isfinite(pj) && pj > 0.0 || error("invalid Karl inverse phase volume at orbit column $j")
        log_measure = log(wj) + log(pj)
        entropy -= wj * log_measure
        dS[j] = -1.0 - log_measure
        ddS[j] = -1.0 / wj
    end

    @inbounds for k in 1:Nlosvd
        idx = Norbit + k
        ww = losvd_residual[k]
        sig = max(abs(losvd_sigma_effective[k]), 1.0e-12)
        den = sig * sig
        entropy -= ww * ww * alphat / den
        chi_slack += ww * ww * alphat / den
        dS[idx] = -2.0 * ww * alphat / den
        ddS[idx] = -2.0 * alphat / den
        ddS[idx] == 0.0 && (ddS[idx] = -DEFAULT_KARL_ENTROPY_FLOOR)
    end

    return entropy, chi_slack, dS, ddS
end

# ============================================================================
# DIAGNOSTIC UTILITIES
# ============================================================================

@inline _relerr(x, scale) = x / max(scale, eps(Float64))

function _svd_rank_info(Am::Matrix{Float64})
    F = svd(Am)
    smax = isempty(F.S) ? 0.0 : maximum(F.S)
    rank_tol = max(size(Am)...) * eps(Float64) * max(smax, 1.0)
    rank_Am = count(s -> s > rank_tol, F.S)
    rcond_supported = begin
        supported = F.S[F.S .> rank_tol]
        isempty(supported) || smax == 0.0 ? 0.0 : minimum(supported) / smax
    end
    return F, rank_Am, rank_tol, rcond_supported
end

function raw_spear_state(w_all::Vector{Float64}, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}; entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    base_delY = target .- Cm * w_all
    rhs = copy(base_delY)
    Am = karl_spear_build_Am(Cm, ddS; floor=entropy_floor)
    karl_spear_rhs!(rhs, Cm, dS, ddS; floor=entropy_floor)
    lambda = _solve_spear_system(Am, rhs)
    dw = karl_spear_delta_w(Cm, lambda, dS, ddS; floor=entropy_floor)
    return (base_delY=base_delY, rhs=rhs, Am=Am, lambda=lambda, dw=dw)
end

function test_raw_spear_identities(state; label="RAW")
    constraint_rel = norm(state.Am * state.lambda - state.rhs) / max(norm(state.rhs), eps(Float64))
    println("[$label SPEAR SYSTEM] relative_residual=", constraint_rel)
    return constraint_rel
end

function test_primal_and_stationarity(Cm, base_delY, dS, ddS, lambda, dw; label="RAW")
    primal_rel = norm(Cm * dw - base_delY) / max(norm(base_delY), eps(Float64))
    stationarity = transpose(Cm) * lambda .- dS .- ddS .* dw
    stationarity_rel = norm(stationarity) / max(norm(dS), 1.0)
    println("[$label PRIMAL] relative_error=", primal_rel)
    println("[$label STATIONARITY] relative_error=", stationarity_rel)
    return primal_rel, stationarity_rel
end

function test_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, Nspatial, Nvbin)
    s = karl_losvd_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, Nspatial, Nvbin)
    residual_error = maximum(abs.(s.residual .- (s.effective_target .- s.model)))
    chi_check = sum((s.residual ./ s.effective_sigma).^2)
    println("[FRACNEW TEST] residual_error=", residual_error,
        " chi_difference=", s.chi_total - chi_check,
        " fracnew_min=", minimum(s.fracnew),
        " fracnew_max=", maximum(s.fracnew))
    return s
end

function test_entropy_derivatives(w_all, Norbit, wphase, losvd_state; alphat=DEFAULT_KARL_ALPHAT, entropy_floor=DEFAULT_KARL_ENTROPY_FLOOR)
    entropy, chi_slack, dS, ddS = build_expanded_entropy_derivatives(w_all, Norbit, wphase, losvd_state.residual, losvd_state.effective_sigma; alphat=alphat, entropy_floor=entropy_floor)
    max_dS_error = 0.0
    max_ddS_error = 0.0
    @inbounds for j in 1:Norbit
        wj = max(w_all[j], entropy_floor)
        expected_dS = -1.0 - log(wj * wphase[j])
        expected_ddS = -1.0 / wj
        max_dS_error = max(max_dS_error, abs(dS[j] - expected_dS))
        max_ddS_error = max(max_ddS_error, abs(ddS[j] - expected_ddS))
    end

    println("[ENTROPY TEST] max_orbit_dS_error=", max_dS_error,
        " max_orbit_ddS_error=", max_ddS_error,
        " entropy=", entropy,
        " chi_slack=", chi_slack)

    return entropy, chi_slack, dS, ddS
end

function first_boundary(w_all, dw, Norbit; apfac=DEFAULT_KARL_APFAC, entropy_floor=DEFAULT_KARL_ENTROPY_FLOOR, active=falses(Norbit))
    boundary = Inf
    idx = 0
    @inbounds for j in 1:Norbit
        active[j] && continue
        if dw[j] < 0.0
            candidate = (w_all[j] - entropy_floor) / (-dw[j])
            if isfinite(candidate) && candidate >= 0.0 && candidate < boundary
                boundary = candidate
                idx = j
            end
        end
    end
    return idx, boundary, isfinite(boundary) && boundary <= apfac * (1.0 + 100.0 * eps(Float64))
end

function test_boundary_state(w_all, dw, Norbit; apfac=DEFAULT_KARL_APFAC, entropy_floor=DEFAULT_KARL_ENTROPY_FLOOR, active=falses(Norbit), label="BOUNDARY")
    idx, candidate, hits = first_boundary(w_all, dw, Norbit; apfac=apfac, entropy_floor=entropy_floor, active=active)
    trial_value = idx == 0 ? NaN : w_all[idx] + candidate * dw[idx]
    println("[$label TEST] idx=", idx,
        " candidate=", candidate,
        " within_requested_step=", hits,
        " value_at_boundary=", trial_value,
        " floor=", entropy_floor)
    return idx, candidate, hits
end

function inspect_reduced_system(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}, active_bound::AbstractVector{Bool}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Narr, nvar = size(Cm)
    active = BitVector(active_bound)
    free_orbits = findall(!, active)
    free_vars = vcat(free_orbits, collect((Norbit + 1):nvar))
    fixed_dw = zeros(Float64, nvar)
    @inbounds for j in 1:Norbit
        active[j] && (fixed_dw[j] = (entropy_floor - w_all[j]) / apfac)
    end
    base_delY = target .- Cm * w_all
    rhs = copy(base_delY)
    @inbounds for j in 1:Norbit
        active[j] && fixed_dw[j] != 0.0 && (rhs .-= (@view Cm[:, j]) .* fixed_dw[j])
    end
    Cm_free = Matrix{Float64}(Cm[:, free_vars])
    dS_free = dS[free_vars]
    ddS_free = ddS[free_vars]
    Am = karl_spear_build_Am(Cm_free, ddS_free; floor=entropy_floor)
    karl_spear_rhs!(rhs, Cm_free, dS_free, ddS_free; floor=entropy_floor)
    Fsvd, rank_Am, rank_tol, supported_rcond = _svd_rank_info(Am)
    lambda = if rank_Am < Narr
        coeff = transpose(Fsvd.U) * rhs
        @inbounds for i in eachindex(coeff)
            coeff[i] = Fsvd.S[i] > rank_tol ? coeff[i] / Fsvd.S[i] : 0.0
        end
        Fsvd.V * coeff
    else
        _solve_spear_system(Am, rhs)
    end
    residual = norm(Am * lambda - rhs) / max(norm(rhs), eps(Float64))
    dw_free = karl_spear_delta_w(Cm_free, lambda, dS_free, ddS_free; floor=entropy_floor)
    dw = copy(fixed_dw)
    dw[free_vars] .= dw_free
    primal_rel = norm(Cm * dw - base_delY) / max(norm(base_delY), eps(Float64))
    bound_kkt = NamedTuple[]
    @inbounds for j in 1:Norbit
        active[j] || continue
        projected_gradient = dot(@view(Cm[:, j]), lambda)
        mu_current = projected_gradient - dS[j]
        mu_full = projected_gradient - dS[j] - ddS[j] * dw[j]
        push!(bound_kkt, (orbit=j, current=mu_current, full=mu_full, missing=mu_full-mu_current))
    end
    return (active=active, free_orbits=free_orbits, free_vars=free_vars, fixed_dw=fixed_dw, base_delY=base_delY, rhs=rhs, Am=Am, lambda=Vector{Float64}(lambda), dw=dw, rank=rank_Am, Narr=Narr, supported_rcond=supported_rcond, svd_relative_residual=residual, primal_relative_error=primal_rel, bound_kkt=bound_kkt)
end

function test_single_bound_reduced_system(w_all, Norbit, Cm, target, dS, ddS, boundary_idx; apfac=DEFAULT_KARL_APFAC, entropy_floor=DEFAULT_KARL_ENTROPY_FLOOR)
    boundary_idx > 0 || begin
        println("[SINGLE BOUND TEST] no raw boundary to force")
        return nothing
    end
    active = falses(Norbit)
    active[boundary_idx] = true
    s = inspect_reduced_system(w_all, Norbit, Cm, target, dS, ddS, active; apfac=apfac, entropy_floor=entropy_floor)
    fixed_final = w_all[boundary_idx] + apfac * s.dw[boundary_idx]
    println("[SINGLE BOUND TEST] orbit=", boundary_idx,
        " rank=", s.rank, "/", s.Narr,
        " svd_residual=", s.svd_relative_residual,
        " primal_rel=", s.primal_relative_error,
        " final_bound_weight=", fixed_final,
        " floor=", entropy_floor)
    if !isempty(s.bound_kkt)
        k = only(s.bound_kkt)
        println("[SINGLE BOUND KKT] current=", k.current,
            " full=", k.full,
            " missing_term=", k.missing)
    end
    return s
end

function probe_bound_releases(w_all, Norbit, Cm, target, dS, ddS, active_bound; apfac=DEFAULT_KARL_APFAC, entropy_floor=DEFAULT_KARL_ENTROPY_FLOOR, top::Int=10)
    active = BitVector(active_bound)
    candidates = NamedTuple[]
    for j in findall(active)
        trial_active = copy(active)
        trial_active[j] = false
        s = inspect_reduced_system(w_all, Norbit, Cm, target, dS, ddS, trial_active; apfac=apfac, entropy_floor=entropy_floor)
        push!(candidates, (orbit=j, rank=s.rank, residual=s.svd_relative_residual, primal=s.primal_relative_error))
    end
    sort!(candidates; by=x -> (x.residual, -x.rank))
    println("[RELEASE RECOVERY PROBE] tested=", length(candidates))
    for x in first(candidates, min(top, length(candidates)))
        println("  orbit=", x.orbit, " rank=", x.rank, " residual=", x.residual, " primal=", x.primal)
    end
    return candidates
end


function _best_active_set_recovery_release(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}, active::AbstractVector{Bool}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    best_orbit = 0
    best_rank = -1
    best_residual = Inf
    best_primal = Inf
    best_floor_violators = typemax(Int)
    best_rebind = true

    for j in findall(active)
        trial_active = BitVector(active)
        trial_active[j] = false

        local state
        try
            state = inspect_reduced_system(w_all, Norbit, Cm, target, dS, ddS, trial_active; apfac=apfac, entropy_floor=entropy_floor)
        catch
            continue
        end

        residual = isfinite(state.svd_relative_residual) ? state.svd_relative_residual : Inf
        primal = isfinite(state.primal_relative_error) ? state.primal_relative_error : Inf
        floor_violators = 0
        released_rebind = false

        @inbounds for k in state.free_orbits
            if state.dw[k] < 0.0
                candidate = (w_all[k] - entropy_floor) / (-state.dw[k])
                if isfinite(candidate) && candidate >= 0.0 && candidate <= apfac * (1.0 + 100.0 * eps(Float64))
                    floor_violators += 1
                    k == j && (released_rebind = true)
                end
            end
        end

        score = (floor_violators, released_rebind ? 1 : 0, -state.rank, residual, primal)
        best_score = (best_floor_violators, best_rebind ? 1 : 0, -best_rank, best_residual, best_primal)

        if score < best_score
            best_orbit = j
            best_rank = state.rank
            best_residual = residual
            best_primal = primal
            best_floor_violators = floor_violators
            best_rebind = released_rebind
        end
    end

    return (orbit=best_orbit, rank=best_rank, residual=best_residual, primal=best_primal, floor_violators=best_floor_violators, released_rebind=best_rebind)
end

# ============================================================================
# ACTIVE-SET SOLVER -- CURRENT BEHAVIOR + DIAGNOSTICS ONLY
# ============================================================================

function karl_spear_active_set_update(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}, active_bound::AbstractVector{Bool}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, compute_rcond::Bool=true, rcond_warn::Float64=DEFAULT_KARL_SPEAR_RCOND_WARN, max_active_passes::Int=512, bound_kkt_tol::Float64=1.0e-10, svd_residual_tol::Float64=1.0e-10, trace::Bool=true)
    Narr, nvar = size(Cm)
    length(w_all) == nvar || error("w_all length must match Cm columns")
    length(target) == Narr || error("target length must match Cm rows")
    length(dS) == nvar || error("dS length must match Cm columns")
    length(ddS) == nvar || error("ddS length must match Cm columns")
    length(active_bound) == Norbit || error("active_bound length must match Norbit")
    0 < Norbit <= nvar || error("Norbit must be between 1 and length(w_all)")
    isfinite(apfac) && apfac > 0.0 || error("apfac must be finite and positive")
    isfinite(entropy_floor) && entropy_floor > 0.0 || error("entropy_floor must be finite and positive")
    max_active_passes > 0 || error("max_active_passes must be positive")
    isfinite(bound_kkt_tol) && bound_kkt_tol >= 0.0 || error("bound_kkt_tol must be finite and nonnegative")
    isfinite(svd_residual_tol) && svd_residual_tol > 0.0 || error("svd_residual_tol must be finite and positive")
    all(x -> isfinite(x) && x >= entropy_floor, @view(w_all[1:Norbit])) || error("physical orbit weights must start finite and at or above the entropy floor")

    active = BitVector(active_bound)
    model = Cm * w_all
    base_delY = target .- model
    n_activated_total = 0
    released_total = 0
    active_set_failure_reason = :none
    active_passes = 0
    mask_first_seen = Dict{UInt,Int}()
    cycle_detected = false
    cycle_first_pass = 0
    cycle_repeat_pass = 0
    history = NamedTuple[]
    last_Am = zeros(Float64, 0, 0)
    last_rhs = Float64[]
    last_lambda = Float64[]
    last_dw = zeros(Float64, nvar)
    last_rank = 0
    last_svd_residual = NaN
    last_rcond = NaN

    for pass in 1:max_active_passes
        active_passes = pass

        mask_hash = hash(active)
        if haskey(mask_first_seen, mask_hash) && !cycle_detected
            cycle_detected = true
            cycle_first_pass = mask_first_seen[mask_hash]
            cycle_repeat_pass = pass
            trace && println("[ASET CYCLE] first_pass=", cycle_first_pass, " repeat_pass=", cycle_repeat_pass, " bound=", count(active))
        else
            mask_first_seen[mask_hash] = pass
        end

        free_orbits = findall(!, active)

        if isempty(free_orbits)
            recovery = _best_active_set_recovery_release(w_all, Norbit, Cm, target, dS, ddS, active; apfac=apfac, entropy_floor=entropy_floor)

            if recovery.orbit > 0
                active[recovery.orbit] = false
                released_total += 1
                trace && println("[ASET] pass=", pass, " bound=", count(active), " event=recovery_release cause=no_free_orbits orbit=", recovery.orbit, " trial_rank=", recovery.rank, "/", Narr, " trial_residual=", recovery.residual)
                push!(history, (pass=pass, event=:recovery_release, orbit=recovery.orbit, bound=count(active), rank=recovery.rank, svd_residual=recovery.residual, mask_hash=mask_hash, multiplier_current=NaN, multiplier_full=NaN))
                continue
            end

            active_set_failure_reason = :no_free_orbits
            push!(history, (pass=pass, event=:fail_no_free_orbits, orbit=0, bound=count(active), rank=0, svd_residual=NaN, mask_hash=mask_hash, multiplier_current=NaN, multiplier_full=NaN))
            break
        end

        free_vars = vcat(free_orbits, collect((Norbit + 1):nvar))

        if length(free_vars) < Narr
            recovery = _best_active_set_recovery_release(w_all, Norbit, Cm, target, dS, ddS, active; apfac=apfac, entropy_floor=entropy_floor)

            if recovery.orbit > 0
                active[recovery.orbit] = false
                released_total += 1
                trace && println("[ASET] pass=", pass, " bound=", count(active), " event=recovery_release cause=insufficient_free_variables orbit=", recovery.orbit, " trial_rank=", recovery.rank, "/", Narr, " trial_residual=", recovery.residual)
                push!(history, (pass=pass, event=:recovery_release, orbit=recovery.orbit, bound=count(active), rank=recovery.rank, svd_residual=recovery.residual, mask_hash=mask_hash, multiplier_current=NaN, multiplier_full=NaN))
                continue
            end

            active_set_failure_reason = :insufficient_free_variables
            push!(history, (pass=pass, event=:fail_insufficient_free_variables, orbit=0, bound=count(active), rank=0, svd_residual=NaN, mask_hash=mask_hash, multiplier_current=NaN, multiplier_full=NaN))
            break
        end

        fixed_dw = zeros(Float64, nvar)
        @inbounds for j in 1:Norbit
            active[j] && (fixed_dw[j] = (entropy_floor - w_all[j]) / apfac)
        end

        rhs = copy(base_delY)
        @inbounds for j in 1:Norbit
            active[j] && fixed_dw[j] != 0.0 && (rhs .-= (@view Cm[:, j]) .* fixed_dw[j])
        end

        Cm_free = Matrix{Float64}(Cm[:, free_vars])
        dS_free = dS[free_vars]
        ddS_free = ddS[free_vars]
        Am = karl_spear_build_Am(Cm_free, ddS_free; floor=entropy_floor)
        karl_spear_rhs!(rhs, Cm_free, dS_free, ddS_free; floor=entropy_floor)
        Fsvd, reduced_Am_rank, rank_tol, supported_rcond = _svd_rank_info(Am)
        rcond_est = isempty(Fsvd.S) || maximum(Fsvd.S) == 0.0 ? 0.0 : minimum(Fsvd.S) / maximum(Fsvd.S)
        last_rank = reduced_Am_rank
        last_rcond = compute_rcond ? rcond_est : supported_rcond
        lambda = Float64[]
        svd_relative_residual = NaN

        if reduced_Am_rank < Narr
            coeff = transpose(Fsvd.U) * rhs
            @inbounds for i in eachindex(coeff)
                coeff[i] = Fsvd.S[i] > rank_tol ? coeff[i] / Fsvd.S[i] : 0.0
            end
            lambda = Fsvd.V * coeff
            svd_relative_residual = norm(Am * lambda - rhs) / max(norm(rhs), eps(Float64))
            last_svd_residual = svd_relative_residual

            if !isfinite(svd_relative_residual) || svd_relative_residual > svd_residual_tol
                last_Am = Am
                last_rhs = rhs
                last_lambda = lambda

                recovery = _best_active_set_recovery_release(w_all, Norbit, Cm, target, dS, ddS, active; apfac=apfac, entropy_floor=entropy_floor)

                if recovery.orbit > 0
                    active[recovery.orbit] = false
                    released_total += 1
                    trace && println("[ASET] pass=", pass, " bound=", count(active), " rank=", reduced_Am_rank, "/", Narr, " event=recovery_release cause=reduced_spear_inconsistent orbit=", recovery.orbit, " trial_rank=", recovery.rank, "/", Narr, " old_residual=", svd_relative_residual, " trial_residual=", recovery.residual)
                    push!(history, (pass=pass, event=:recovery_release, orbit=recovery.orbit, bound=count(active), rank=recovery.rank, svd_residual=recovery.residual, mask_hash=mask_hash, multiplier_current=NaN, multiplier_full=NaN))
                    continue
                end

                active_set_failure_reason = :reduced_spear_inconsistent
                trace && println("[ASET] pass=", pass, " bound=", count(active), " rank=", reduced_Am_rank, "/", Narr, " event=inconsistent svd_residual=", svd_relative_residual, " rcond=", last_rcond)
                push!(history, (pass=pass, event=:inconsistent, orbit=0, bound=count(active), rank=reduced_Am_rank, svd_residual=svd_relative_residual, mask_hash=mask_hash, multiplier_current=NaN, multiplier_full=NaN))
                break
            end
        else
            lambda = try
                _solve_spear_system(Am, rhs)
            catch
                active_set_failure_reason = :reduced_spear_solve_failed
                push!(history, (pass=pass, event=:solve_failed, orbit=0, bound=count(active), rank=reduced_Am_rank, svd_residual=NaN, mask_hash=mask_hash, multiplier_current=NaN, multiplier_full=NaN))
                break
            end
        end
        svd_relative_residual = norm(Am * lambda - rhs) / max(norm(rhs), eps(Float64))
        last_svd_residual = svd_relative_residual

        dw_free = karl_spear_delta_w(Cm_free, lambda, dS_free, ddS_free; floor=entropy_floor)
        dw = copy(fixed_dw)
        dw[free_vars] .= dw_free
        trial = w_all .+ apfac .* dw
        @inbounds for j in 1:Norbit
            active[j] && (trial[j] = entropy_floor)
        end
        # Verify the reduced solve reconstructs the desired linearized constraint.
        primal_rel = norm(Cm * dw - base_delY) / max(norm(base_delY), eps(Float64))
        boundary_indices = Int[]
        boundary_candidates = Float64[]

        @inbounds for j in free_orbits
            if dw[j] < 0.0
                candidate = (w_all[j] - entropy_floor) / (-dw[j])
                if isfinite(candidate) && candidate >= 0.0 && candidate <= apfac * (1.0 + 100.0 * eps(Float64))
                    push!(boundary_indices, j)
                    push!(boundary_candidates, candidate)
                end
            end
        end

        if !isempty(boundary_indices)
            imin = argmin(boundary_candidates)
            boundary_idx = boundary_indices[imin]
            boundary_candidate = boundary_candidates[imin]
            nbind = length(boundary_indices)

            @inbounds for j in boundary_indices
                active[j] = true
            end
            n_activated_total += nbind

            trace && println("[ASET] pass=", pass, " bound=", count(active), " rank=", reduced_Am_rank, "/", Narr, " event=bind_all batch=", nbind, " first_orbit=", boundary_idx, " first_candidate=", boundary_candidate, " primal_rel=", primal_rel)

            push!(history, (pass=pass, event=:bind, orbit=boundary_idx, bound=count(active), rank=reduced_Am_rank, svd_residual=svd_relative_residual, mask_hash=mask_hash, multiplier_current=NaN, multiplier_full=NaN))
            last_Am = Am
            last_rhs = rhs
            last_lambda = lambda
            last_dw = dw
            continue
        end

        release_candidates = Int[]
        release_current = Float64[]
        release_full = Float64[]
        @inbounds for j in 1:Norbit
            if active[j]
                projected_gradient = dot(@view(Cm[:, j]), lambda)
                # CURRENT OSPM release test.  Do not change behavior here.
                multiplier_current = projected_gradient - dS[j]
                # Diagnostic only: full stationarity residual for the fixed dw.
                multiplier_full = projected_gradient - dS[j] - ddS[j] * dw[j]
                scale = max(1.0, abs(dS[j]), abs(projected_gradient))
                if multiplier_current < -bound_kkt_tol * scale
                    push!(release_candidates, j)
                    push!(release_current, multiplier_current)
                    push!(release_full, multiplier_full)
                end
            end
        end
        if !isempty(release_candidates)
            nrelease = length(release_candidates)
            imin = argmin(release_current)
            jrel = release_candidates[imin]
            mu_current = release_current[imin]
            mu_full = release_full[imin]

            @inbounds for j in release_candidates
                active[j] = false
            end
            released_total += nrelease

            trace && println("[ASET] pass=", pass, " bound=", count(active), " rank=", reduced_Am_rank, "/", Narr, " event=release batch=", nrelease, " most_negative_orbit=", jrel, " mu_current=", mu_current, " mu_full=", mu_full, " missing_term=", mu_full - mu_current, " primal_rel=", primal_rel)

            push!(history, (pass=pass, event=:release, orbit=jrel, bound=count(active), rank=reduced_Am_rank, svd_residual=svd_relative_residual, mask_hash=mask_hash, multiplier_current=mu_current, multiplier_full=mu_full))
            last_Am = Am
            last_rhs = rhs
            last_lambda = lambda
            last_dw = dw
            continue
        end

        trace && println("[ASET] pass=", pass,
            " bound=", count(active),
            " rank=", reduced_Am_rank, "/", Narr,
            " event=stable",
            " primal_rel=", primal_rel,
            " rcond=", last_rcond)
        push!(history, (pass=pass, event=:stable, orbit=0, bound=count(active), rank=reduced_Am_rank, svd_residual=svd_relative_residual, mask_hash=mask_hash, multiplier_current=NaN, multiplier_full=NaN))
        return (w=Vector{Float64}(trial), dw=Vector{Float64}(dw), lambda=Vector{Float64}(lambda), Am=Matrix{Float64}(Am), rhs=Vector{Float64}(rhs), active_bound=active, active_set_stabilized=true, active_set_failure_reason=:none, active_passes=pass, n_active_bound=count(active), n_activated_total=n_activated_total, released_total=released_total, cycle_detected=cycle_detected, cycle_first_pass=cycle_first_pass, cycle_repeat_pass=cycle_repeat_pass, reduced_Am_rank=reduced_Am_rank, svd_relative_residual=svd_relative_residual, rcond_est=last_rcond, history=history)
    end
    return (w=copy(w_all), dw=last_dw, lambda=last_lambda, Am=last_Am, rhs=last_rhs, active_bound=active, active_set_stabilized=false, active_set_failure_reason=active_set_failure_reason == :none ? :active_set_max_passes : active_set_failure_reason, active_passes=active_passes, n_active_bound=count(active), n_activated_total=n_activated_total, released_total=released_total, cycle_detected=cycle_detected, cycle_first_pass=cycle_first_pass, cycle_repeat_pass=cycle_repeat_pass, reduced_Am_rank=last_rank, svd_relative_residual=last_svd_residual, rcond_est=last_rcond, history=history)
end

function karl_spear_update_expanded(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, active_bound=nothing, max_active_passes::Int=512, bound_kkt_tol::Float64=1.0e-10, trace::Bool=true)
    active = active_bound === nothing ? falses(Norbit) : BitVector(active_bound)
    aset = karl_spear_active_set_update(w_all, Norbit, Cm, target, dS, ddS, active; apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, bound_kkt_tol=bound_kkt_tol, trace=trace)
    return aset
end

function karl_spear_step_light_losvd_all(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target_base::Vector{Float64}, A_losvd::Matrix{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}, wphase_orbit::Vector{Float64}; Nlight::Int, Nspatial::Int, Nvbin::Int, alphat::Float64=DEFAULT_KARL_ALPHAT, apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, active_bound=nothing, max_active_passes::Int=512, bound_kkt_tol::Float64=1.0e-10, trace::Bool=true)
    Nlosvd, Norbit2 = size(A_losvd)
    Norbit2 == Norbit || error("A_losvd orbit count does not match Norbit")
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match Nlosvd")
    active = active_bound === nothing ? falses(Norbit) : BitVector(active_bound)
    w_orbit = Vector{Float64}(@view w_all[1:Norbit])
    losvd_state = karl_losvd_fracnew_state(A_losvd, w_orbit, losvd_target, losvd_sigma, Nspatial, Nvbin)
    entropy, chi_slack, dS, ddS = build_expanded_entropy_derivatives(w_all, Norbit, wphase_orbit, losvd_state.residual, losvd_state.effective_sigma; alphat=alphat, entropy_floor=entropy_floor)
    target = copy(target_base)
    target[(Nlight + 1):(Nlight + Nlosvd)] .= losvd_state.effective_target
    update = karl_spear_update_expanded(w_all, Norbit, Cm, target, dS, ddS; apfac=apfac, entropy_floor=entropy_floor, active_bound=active, max_active_passes=max_active_passes, bound_kkt_tol=bound_kkt_tol, trace=trace)
    return (update=update, target=target, losvd_state=losvd_state, entropy=entropy, chi_slack=chi_slack, dS=dS, ddS=ddS)
end

# ============================================================================
# INHERITED-BOUND A/B TEST
# ============================================================================

function inherited_bound_ab_test(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}, inherited_active::AbstractVector{Bool}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, max_active_passes::Int=512, bound_kkt_tol::Float64=1.0e-10)
    println("\n=== INHERITED ACTIVE-SET A/B TEST ===")
    println("A: inherited bounds = ", count(inherited_active))
    A = karl_spear_active_set_update(w_all, Norbit, Cm, target, dS, ddS, inherited_active; apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, bound_kkt_tol=bound_kkt_tol, trace=true)
    println("\nB: reset bounds = 0")
    B = karl_spear_active_set_update(w_all, Norbit, Cm, target, dS, ddS, falses(Norbit); apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, bound_kkt_tol=bound_kkt_tol, trace=true)
    println("\n[AB SUMMARY] inherited_stable=", A.active_set_stabilized, " inherited_reason=", A.active_set_failure_reason, " inherited_passes=", A.active_passes, " reset_stable=", B.active_set_stabilized, " reset_reason=", B.active_set_failure_reason, " reset_passes=", B.active_passes)
    if A.active_set_stabilized && B.active_set_stabilized
        max_weight_difference = maximum(abs.(A.w .- B.w))
        mask_difference = count(A.active_bound .!= B.active_bound)
        println("[AB ENDPOINT] max_weight_difference=", max_weight_difference, " mask_difference=", mask_difference, " inherited_bound=", count(A.active_bound), " reset_bound=", count(B.active_bound))
    end
    return A, B
end

# ============================================================================
# FULL STATE TEST FOR ONE FROZEN OUTER ITERATION
# ============================================================================

function run_state_tests(A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, light_target::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}, w::Vector{Float64}, wphase::Vector{Float64}; Nspatial::Int, Nvbin::Int, apfac::Float64=DEFAULT_KARL_APFAC, alphat::Float64=DEFAULT_KARL_ALPHAT, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, active_bound=falses(length(w)), max_active_passes::Int=512, bound_kkt_tol::Float64=1.0e-10)
    Nlight, Norbit = size(A_light)
    Nlosvd, Norbit2 = size(A_losvd)
    Norbit == Norbit2 == length(w) == length(wphase) || error("orbit dimensions do not match")
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match Nlosvd")

    println("\n============================================================")
    println("OSPM SPEAR / ACTIVE-SET STATE TEST")
    println("Norbit=", Norbit, " Nlight=", Nlight, " Nlosvd=", Nlosvd,
        " apfac=", apfac, " inherited_bound=", count(active_bound))
    println("============================================================\n")

    enforce_normalization = !_expanded_light_implies_normalization(A_light, light_target)
    println("[EXPANDED CM] enforce_normalization=", enforce_normalization)
    Cm = build_expanded_Cm_with_losvd_slack(A_light, A_losvd; enforce_normalization=enforce_normalization)
    target_base = build_expanded_target(light_target, losvd_target; enforce_normalization=enforce_normalization)
    w_all = build_expanded_weights_initial(w, A_losvd, losvd_target)

    # 1-2: fracnew and target state
    losvd_state = test_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, Nspatial, Nvbin)

    # 3: entropy derivatives
    entropy, chi_slack, dS, ddS = test_entropy_derivatives(w_all, Norbit, wphase, losvd_state; alphat=alphat, entropy_floor=entropy_floor)

    target = copy(target_base)
    target[(Nlight + 1):(Nlight + Nlosvd)] .= losvd_state.effective_target

    # 1: raw SPEAR identities, before bounds
    raw = raw_spear_state(w_all, Cm, target, dS, ddS; entropy_floor=entropy_floor)
    test_raw_spear_identities(raw)
    test_primal_and_stationarity(Cm, raw.base_delY, dS, ddS, raw.lambda, raw.dw)

    # 4: first boundary
    boundary_idx, boundary_candidate, boundary_hits = test_boundary_state(w_all, raw.dw, Norbit; apfac=apfac, entropy_floor=entropy_floor)

    # 5: explicitly force only that first bound and inspect the reduced system.
    if boundary_hits
        test_single_bound_reduced_system(w_all, Norbit, Cm, target, dS, ddS, boundary_idx; apfac=apfac, entropy_floor=entropy_floor)
    end

    # 6-7: active-set reduced solves, KKT release diagnostics, cycle history
    println("\n=== ACTIVE-SET TRACE ===")
    aset = karl_spear_active_set_update(w_all, Norbit, Cm, target, dS, ddS, active_bound; apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, bound_kkt_tol=bound_kkt_tol, trace=true)

    println("\n[ACTIVE SET SUMMARY] stabilized=", aset.active_set_stabilized,
        " reason=", aset.active_set_failure_reason,
        " passes=", aset.active_passes,
        " bound=", aset.n_active_bound,
        " activated=", aset.n_activated_total,
        " released=", aset.released_total,
        " cycle_detected=", aset.cycle_detected,
        " cycle_first_pass=", aset.cycle_first_pass,
        " cycle_repeat_pass=", aset.cycle_repeat_pass)

    if !aset.active_set_stabilized && aset.active_set_failure_reason == :reduced_spear_inconsistent && any(aset.active_bound)
        println("\n=== ONE-BOUND RELEASE RECOVERY PROBE ===")
        probe_bound_releases(w_all, Norbit, Cm, target, dS, ddS, aset.active_bound; apfac=apfac, entropy_floor=entropy_floor)
    end

    # 8: inherited-bound A/B test using the supplied inherited active mask.
    if any(active_bound)
        inherited_bound_ab_test(w_all, Norbit, Cm, target, dS, ddS, active_bound; apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, bound_kkt_tol=bound_kkt_tol)
    end

    return (Cm=Cm, target=target, w_all=w_all, losvd_state=losvd_state, entropy=entropy, chi_slack=chi_slack, dS=dS, ddS=ddS, raw=raw, active_set=aset)
end

# ============================================================================
# DETERMINISTIC SYNTHETIC EXAMPLE
# ============================================================================

function synthetic_problem(; seed::Int=653)
    Random.seed!(seed)
    Norbit = 100
    Nspatial = 2
    Nvbin = 2
    Nlight = 3
    Nlosvd = Nspatial * Nvbin
    A_light = rand(Nlight, Norbit)
    A_light ./= sum(A_light, dims=1)
    A_losvd = rand(Nlosvd, Norbit)
    A_losvd ./= sum(A_losvd, dims=1)
    w = exp.(-range(0.0, 4.0; length=Norbit))
    w ./= sum(w)
    w_target = reverse(w)
    w_target ./= sum(w_target)
    light_target = A_light * w_target
    losvd_target = A_losvd * w_target
    losvd_sigma = fill(0.03, Nlosvd)
    wphase = exp.(range(log(0.2), log(5.0); length=Norbit))
    return (A_light=A_light, A_losvd=A_losvd, light_target=light_target, losvd_target=losvd_target, losvd_sigma=losvd_sigma, w=w, w_target=w_target, wphase=wphase, Nspatial=Nspatial, Nvbin=Nvbin)
end

function build_frozen_test_state(A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, light_target::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}, w::Vector{Float64}, wphase::Vector{Float64}; Nspatial::Int, Nvbin::Int, alphat::Float64=DEFAULT_KARL_ALPHAT, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Nlight, Norbit = size(A_light)
    Nlosvd, Norbit2 = size(A_losvd)
    Norbit == Norbit2 == length(w) == length(wphase) || error("orbit dimensions do not match")
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match Nlosvd")
    enforce_normalization = !_expanded_light_implies_normalization(A_light, light_target)
    Cm = build_expanded_Cm_with_losvd_slack(A_light, A_losvd; enforce_normalization=enforce_normalization)
    target = build_expanded_target(light_target, losvd_target; enforce_normalization=enforce_normalization)
    w_all = build_expanded_weights_initial(w, A_losvd, losvd_target)
    losvd_state = karl_losvd_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, Nspatial, Nvbin)
    target[(Nlight + 1):(Nlight + Nlosvd)] .= losvd_state.effective_target
    entropy, chi_slack, dS, ddS = build_expanded_entropy_derivatives(w_all, Norbit, wphase, losvd_state.residual, losvd_state.effective_sigma; alphat=alphat, entropy_floor=entropy_floor)
    return (Norbit=Norbit, Nlight=Nlight, Nlosvd=Nlosvd, Cm=Cm, target=target, w_all=w_all, losvd_state=losvd_state, entropy=entropy, chi_slack=chi_slack, dS=dS, ddS=ddS)
end

function make_floor_heavy_weights(w::Vector{Float64}, fraction::Float64; entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    0.0 <= fraction < 1.0 || error("fraction must satisfy 0 <= fraction < 1")
    Norbit = length(w)
    nfloor = clamp(round(Int, fraction * Norbit), 0, Norbit - 1)
    floor_mask = falses(Norbit)
    if nfloor > 0
        floor_indices = sortperm(w)[1:nfloor]
        floor_mask[floor_indices] .= true
    end
    w2 = copy(w)
    w2[floor_mask] .= entropy_floor
    free_mask = .!floor_mask
    free_sum = sum(w2[free_mask])
    free_sum > 0.0 || error("floor-heavy state has no positive free weight")
    target_free_sum = 1.0 - count(floor_mask) * entropy_floor
    w2[free_mask] .*= target_free_sum / free_sum
    return w2, floor_mask
end

function run_floor_heavy_tests(p; fractions=(0.20, 0.50, 0.80, 0.95), apfac::Float64=DEFAULT_KARL_APFAC, max_active_passes::Int=512, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    println("\n============================================================")
    println("FLOOR-HEAVY ACTIVE-SET TESTS")
    println("============================================================")
    results = NamedTuple[]
    for fraction in fractions
        w_floor, floor_mask = make_floor_heavy_weights(p.w, fraction; entropy_floor=entropy_floor)
        s = build_frozen_test_state(p.A_light, p.A_losvd, p.light_target, p.losvd_target, p.losvd_sigma, w_floor, p.wphase; Nspatial=p.Nspatial, Nvbin=p.Nvbin, entropy_floor=entropy_floor)
        result = karl_spear_active_set_update(s.w_all, s.Norbit, s.Cm, s.target, s.dS, s.ddS, falses(s.Norbit); apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, trace=false)
        initial_floor = count(floor_mask)
        final_floor = count(x -> x <= entropy_floor * (1.0 + 100.0 * eps(Float64)), @view(result.w[1:s.Norbit]))
        println("[FLOOR HEAVY] fraction=", fraction, " initial_floor=", initial_floor, " stable=", result.active_set_stabilized, " reason=", result.active_set_failure_reason, " passes=", result.active_passes, " activated=", result.n_activated_total, " released=", result.released_total, " active_final=", result.n_active_bound, " weight_floor_final=", final_floor, " cycle=", result.cycle_detected)
        push!(results, (fraction=fraction, initial_floor=initial_floor, result=result, weight_floor_final=final_floor))
    end
    return results
end

function run_overconstrained_recovery_test(p; apfac::Float64=DEFAULT_KARL_APFAC, max_active_passes::Int=512, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    println("\n============================================================")
    println("OVERCONSTRAINED ACTIVE-SET RECOVERY TEST")
    println("============================================================")
    s = build_frozen_test_state(p.A_light, p.A_losvd, p.light_target, p.losvd_target, p.losvd_sigma, p.w, p.wphase; Nspatial=p.Nspatial, Nvbin=p.Nvbin, entropy_floor=entropy_floor)
    active = trues(s.Norbit)
    active[1] = false
    active[2] = false
    free_vars = count(!, active) + (length(s.w_all) - s.Norbit)
    println("[OVERCONSTRAINED START] active=", count(active), " free_orbits=", count(!, active), " free_vars=", free_vars, " constraints=", size(s.Cm, 1))
    failed = karl_spear_active_set_update(s.w_all, s.Norbit, s.Cm, s.target, s.dS, s.ddS, active; apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, trace=true)
    println("[OVERCONSTRAINED RESULT] stable=", failed.active_set_stabilized, " reason=", failed.active_set_failure_reason, " passes=", failed.active_passes, " active=", failed.n_active_bound)
    candidates = probe_bound_releases(s.w_all, s.Norbit, s.Cm, s.target, s.dS, s.ddS, active; apfac=apfac, entropy_floor=entropy_floor, top=10)
    isempty(candidates) && return (failed=failed, candidates=candidates, repaired=nothing)
    best = first(candidates)
    repaired_active = copy(active)
    repaired_active[best.orbit] = false
    println("[OVERCONSTRAINED MANUAL RELEASE] orbit=", best.orbit, " rank=", best.rank, " residual=", best.residual, " primal=", best.primal)
    repaired = karl_spear_active_set_update(s.w_all, s.Norbit, s.Cm, s.target, s.dS, s.ddS, repaired_active; apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, trace=false)
    println("[OVERCONSTRAINED REPAIRED] stable=", repaired.active_set_stabilized, " reason=", repaired.active_set_failure_reason, " passes=", repaired.active_passes, " active_final=", repaired.n_active_bound, " released=", repaired.released_total)
    return (failed=failed, candidates=candidates, repaired=repaired)
end

function run_rank_deficient_recovery_test(p; apfac::Float64=DEFAULT_KARL_APFAC, max_active_passes::Int=512, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    println("\n============================================================")
    println("REDUCED-SPEAR INCONSISTENCY RECOVERY TEST")
    println("============================================================")
    A_light = copy(p.A_light)
    A_light[:, 3] .= A_light[:, 1]
    light_target = A_light * p.w_target
    s = build_frozen_test_state(A_light, p.A_losvd, light_target, p.losvd_target, p.losvd_sigma, p.w, p.wphase; Nspatial=p.Nspatial, Nvbin=p.Nvbin, entropy_floor=entropy_floor)
    active = trues(s.Norbit)
    active[1] = false
    active[2] = false
    active[3] = false
    println("[RANK-DEFICIENT START] active=", count(active), " free_orbits=", count(!, active), " free_vars=", count(!, active) + (length(s.w_all) - s.Norbit), " constraints=", size(s.Cm, 1))
    failed = karl_spear_active_set_update(s.w_all, s.Norbit, s.Cm, s.target, s.dS, s.ddS, active; apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, trace=true)
    println("[RANK-DEFICIENT RESULT] stable=", failed.active_set_stabilized, " reason=", failed.active_set_failure_reason, " passes=", failed.active_passes, " rank=", failed.reduced_Am_rank, " residual=", failed.svd_relative_residual)
    candidates = probe_bound_releases(s.w_all, s.Norbit, s.Cm, s.target, s.dS, s.ddS, active; apfac=apfac, entropy_floor=entropy_floor, top=10)
    isempty(candidates) && return (failed=failed, candidates=candidates, repaired=nothing)
    best = first(candidates)
    repaired_active = copy(active)
    repaired_active[best.orbit] = false
    println("[RANK-DEFICIENT MANUAL RELEASE] orbit=", best.orbit, " rank=", best.rank, " residual=", best.residual, " primal=", best.primal)
    repaired = karl_spear_active_set_update(s.w_all, s.Norbit, s.Cm, s.target, s.dS, s.ddS, repaired_active; apfac=apfac, entropy_floor=entropy_floor, max_active_passes=max_active_passes, trace=false)
    println("[RANK-DEFICIENT REPAIRED] stable=", repaired.active_set_stabilized, " reason=", repaired.active_set_failure_reason, " passes=", repaired.active_passes, " active_final=", repaired.n_active_bound, " released=", repaired.released_total)
    return (failed=failed, candidates=candidates, repaired=repaired)
end

function run_active_set_suite(; stress_apfac::Float64=2.0, realistic_apfac::Float64=DEFAULT_KARL_APFAC, max_active_passes::Int=512)
    p = synthetic_problem()

    println("\n################################################################")
    println("SCENARIO 1: FRESH MASK / NORMAL BINDING")
    println("################################################################")
    first_state = run_state_tests(p.A_light, p.A_losvd, p.light_target, p.losvd_target, p.losvd_sigma, p.w, p.wphase; Nspatial=p.Nspatial, Nvbin=p.Nvbin, apfac=stress_apfac, max_active_passes=max_active_passes)

    println("\n################################################################")
    println("SCENARIO 2: INHERITED MASK VS FRESH RECONSTRUCTION")
    println("################################################################")
    second_state = nothing
    if first_state.active_set.active_set_stabilized && any(first_state.active_set.active_bound)
        Norbit = length(p.w)
        w2 = Vector{Float64}(@view first_state.active_set.w[1:Norbit])
        second_state = run_state_tests(p.A_light, p.A_losvd, p.light_target, p.losvd_target, p.losvd_sigma, w2, p.wphase; Nspatial=p.Nspatial, Nvbin=p.Nvbin, apfac=stress_apfac, active_bound=first_state.active_set.active_bound, max_active_passes=max_active_passes)
    else
        println("[SCENARIO 2 SKIPPED] first state produced no stable nonempty active set")
    end

    println("\n################################################################")
    println("SCENARIO 3: FLOOR-HEAVY FRESH-MASK STATES AT REALISTIC APFAC")
    println("################################################################")
    floor_results = run_floor_heavy_tests(p; apfac=realistic_apfac, max_active_passes=max_active_passes)

    println("\n################################################################")
    println("SCENARIO 4A: DELIBERATELY OVERCONSTRAINED MASK")
    println("################################################################")
    overconstrained = run_overconstrained_recovery_test(p; apfac=realistic_apfac, max_active_passes=max_active_passes)

    println("\n################################################################")
    println("SCENARIO 4B: DELIBERATELY RANK-DEFICIENT REDUCED SYSTEM")
    println("################################################################")
    rank_deficient = run_rank_deficient_recovery_test(p; apfac=realistic_apfac, max_active_passes=max_active_passes)

    println("\n################################################################")
    println("ACTIVE-SET SUITE COMPLETE")
    println("################################################################")
    println("[SUITE SUMMARY] first_stable=", first_state.active_set.active_set_stabilized, " second_ran=", second_state !== nothing, " floor_tests=", length(floor_results), " overconstrained_reason=", overconstrained.failed.active_set_failure_reason, " rank_deficient_reason=", rank_deficient.failed.active_set_failure_reason)
    return (first_state=first_state, second_state=second_state, floor_results=floor_results, overconstrained=overconstrained, rank_deficient=rank_deficient)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_active_set_suite()
end
