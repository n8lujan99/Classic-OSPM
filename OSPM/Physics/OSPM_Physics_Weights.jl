# ========================================================================================================================
# OSPM_Physics_Weights.jl — Karl-style orbit weight machinery.
# Included by OSPM_Physics_Support.jl — do NOT load directly.
# Owns:
#   - strict Karl inverse-phase-volume handling
#   - Karl entropy type 2
#   - paired prograde/retrograde initial weights
#   - expanded-Cm SPEAR solver
#   - LOSVD slack variables
#   - standard LOSVD χ² block scoring
#   - xmu / M-L helper functions
# ========================================================================================================================
function _prepare_wphase(wphase, n::Int; require_paired::Bool=true, pair_rtol::Float64=1.0e-12,)
    n > 0 || error("Norbit must be positive before preparing Karl phase volumes")
    wphase === nothing && error("Karl inverse phase volumes are required. Build and compact wphase " * "with OSPM_Physics_PhaseVolume.jl before calling the weight solver.")
    wp = vec(Float64.(wphase))
    length(wp) == n ||
        error("wphase length $(length(wp)) does not match Norbit=$n")
    invalid = Int[]
    @inbounds for i in eachindex(wp)
        pi = wp[i]
        if !(isfinite(pi) && pi > 0.0)
            push!(invalid, i)
        end
    end
    if !isempty(invalid)
        preview = join(first(invalid, min(length(invalid), 20)), ",")
        suffix = length(invalid) > 20 ? ",..." : ""
        error("Karl wphase contains $(length(invalid)) nonfinite or non-positive " * "entry/entries at [$preview$suffix]. No unity or entropy-floor " * "fallback is allowed.")
    end
    if require_paired
        iseven(n) || error("Karl phase-volume mode requires paired prograde/retrograde orbit columns; " * "got Norbit=$n")
        isfinite(pair_rtol) && pair_rtol >= 0.0 ||
            error("pair_rtol must be finite and nonnegative")
        bad_pairs = Int[]
        @inbounds for ibase in 1:(n ÷ 2)
            ip = 2 * ibase - 1
            ir = 2 * ibase
            scale = max(abs(wp[ip]), abs(wp[ir]), floatmin(Float64))
            abs(wp[ip] - wp[ir]) <= pair_rtol * scale || push!(bad_pairs, ibase)
        end
        if !isempty(bad_pairs)
            preview = join(first(bad_pairs, min(length(bad_pairs), 20)), ",")
            suffix = length(bad_pairs) > 20 ? ",..." : ""
            error("Karl prograde/retrograde wphase values disagree for " * "$(length(bad_pairs)) base orbit pair(s): [$preview$suffix]. " * "The phase-volume vector is not aligned with the A-matrix columns.")
        end
    end
    return wp
end

function karl_wphase_diagnostics(wphase::Vector{Float64}; paired::Bool=true)
    isempty(wphase) && error("wphase diagnostics require at least one orbit")
    all(isfinite, wphase) || error("wphase diagnostics received nonfinite values")
    all(>(0.0), wphase) || error("wphase diagnostics received non-positive values")

    log_wp = log.(wphase)
    log_min, log_max = extrema(log_wp)
    log_dynamic_range = log_max - log_min
    dynamic_range = log_dynamic_range <= log(floatmax(Float64)) ?
        exp(log_dynamic_range) :
        Inf
    geometric_mean = exp(sum(log_wp) / length(log_wp))
    pair_max_relative_mismatch = 0.0
    if paired
        iseven(length(wphase)) || error("paired wphase diagnostics require an even vector length")
        @inbounds for ibase in 1:(length(wphase) ÷ 2)
            ip = 2 * ibase - 1
            ir = 2 * ibase
            scale = max(abs(wphase[ip]), abs(wphase[ir]), floatmin(Float64))
            pair_max_relative_mismatch = max(pair_max_relative_mismatch, abs(wphase[ip] - wphase[ir]) / scale)
        end
    end
    wphase_min = exp(log_min)
    wphase_max = exp(log_max)
    phase_volume_min = 1.0 / wphase_max
    phase_volume_max = 1.0 / wphase_min
    return (
        convention=:inverse_phase_volume,
        entropy_expression=Symbol("-sum(w*log(w*wphase))"),
        wphase_min=wphase_min,
        wphase_max=wphase_max,
        wphase_dynamic_range=dynamic_range,
        wphase_log_dynamic_range=log_dynamic_range,
        wphase_geometric_mean=geometric_mean,
        phase_volume_min=phase_volume_min,
        phase_volume_max=phase_volume_max,
        phase_volume_dynamic_range=dynamic_range,
        pair_max_relative_mismatch=pair_max_relative_mismatch,
    )
end

function karl_initial_weights_from_wphase(wphase::Vector{Float64}; paired::Bool=true, rotfrac::Float64=0.75, floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    n = length(wphase)
    n > 0 || return Float64[]
    all(isfinite, wphase) || error("Karl initial weights received nonfinite wphase")
    all(>(0.0), wphase) || error("Karl initial weights received non-positive wphase")
    isfinite(floor) && floor > 0.0 || error("initial-weight floor must be finite and positive")
    n * floor < 1.0 || error("initial-weight floor is too large for Norbit=$n")
    q = zeros(Float64, n)
    if paired
        iseven(n) || error("paired Karl initial weights require an even number of orbit columns",)
        isfinite(rotfrac) && 0.0 < rotfrac < 1.0 ||
            error("rotfrac must lie strictly between 0 and 1")
        log_volume = Vector{Float64}(undef, n ÷ 2)
        @inbounds for ibase in eachindex(log_volume)
            ip = 2 * ibase - 1
            ir = 2 * ibase
            scale = max(abs(wphase[ip]), abs(wphase[ir]), floatmin(Float64))
            abs(wphase[ip] - wphase[ir]) <= 1.0e-12 * scale ||
                error("paired wphase mismatch at base orbit $ibase")
            log_volume[ibase] = -0.5 * (log(wphase[ip]) + log(wphase[ir]))
        end
        log_volume_max = maximum(log_volume)
        @inbounds for ibase in eachindex(log_volume)
            pair_volume = exp(log_volume[ibase] - log_volume_max)
            ip = 2 * ibase - 1
            ir = 2 * ibase
            q[ip] = rotfrac * pair_volume
            q[ir] = (1.0 - rotfrac) * pair_volume
        end
    else
        log_volume = -log.(wphase)
        log_volume_max = maximum(log_volume)
        @inbounds for i in eachindex(q)
            q[i] = exp(log_volume[i] - log_volume_max)
        end
    end
    qsum = sum(q)
    isfinite(qsum) && qsum > 0.0 ||
        error("Karl phase-volume initial distribution has non-positive sum")
    q ./= qsum
    # Keep every physical orbit strictly inside the entropy domain while
    # preserving the phase-volume prior in the remaining probability mass.
    free_mass = 1.0 - n * floor
    w = similar(q)
    @inbounds for i in eachindex(q)
        w[i] = floor + free_mass * q[i]
    end
    abs(sum(w) - 1.0) <= 100.0 * eps(Float64) * max(n, 1) ||
        error("Karl initial orbit weights failed normalization")
    minimum(w) >= floor ||
        error("Karl initial orbit weights fell below the entropy floor")
    return w
end

@inline function _safe_positive(x::Float64; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    return (isfinite(x) && x > floor) ? x : floor
end

# ========================================================================================================================

function losvd_width_at_fraction(v::Vector{Float64}, f::Vector{Float64}, frac::Float64)
    length(v) == length(f) || error("velocity and LOSVD arrays must match")
    length(v) >= 2 || return NaN
    fmax = maximum(f)
    !(isfinite(fmax) && fmax > 0.0) && return NaN
    level = frac * fmax
    inds = findall(x -> isfinite(x) && x >= level, f)
    isempty(inds) && return NaN
    return maximum(v[inds]) - minimum(v[inds])
end

function karl_update_xmu_from_fwhm(xmu::Float64, model_losvd_by_bin::Vector{Vector{Float64}}, data_losvd_by_bin::Vector{Vector{Float64}}, velocity_centers_by_bin::Vector{Vector{Float64}}; apfacmu::Float64=1.0, fractions::NTuple{2,Float64}=(0.25, 0.50))
    length(model_losvd_by_bin) == length(data_losvd_by_bin) == length(velocity_centers_by_bin) ||
        error("LOSVD bin collections must have matching lengths")

    sxmu = 0.0
    nuse = 0
    for ib in eachindex(model_losvd_by_bin)
        model = model_losvd_by_bin[ib]
        data = data_losvd_by_bin[ib]
        vel = velocity_centers_by_bin[ib]
        for frac in fractions
            fwm = losvd_width_at_fraction(vel, model, frac)
            fwd = losvd_width_at_fraction(vel, data, frac)
            if isfinite(fwm) && isfinite(fwd) && fwd > 0.0
                sxmu += fwm / fwd
                nuse += 1
            end
        end
    end
    nuse > 0 || return xmu, NaN, NaN
    sxmu /= nuse
    pml_xmu = xmu + xmu * (sxmu - 1.0)
    xmu_new = xmu + apfacmu * xmu * (sxmu - 1.0)
    return xmu_new, sxmu, pml_xmu
end

@inline function karl_ml_from_xmu(xmu::Float64)
    return xmu > 0.0 ? 1.0 / (xmu * xmu) : Inf
end

function chi2_block_karl_fracnew(A_losvd::Matrix{Float64}, w::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}, Nspatial::Int, Nvbin::Int)
    state = karl_losvd_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, Nspatial, Nvbin)
    return state.chi_total, state.chi_by_spatial, state.fracnew
end

@inline function karl_entropy_value(w::Vector{Float64}, wphase::Vector{Float64}; entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    length(w) == length(wphase) ||
        error("weight and wphase lengths do not match")
    all(isfinite, wphase) || error("entropy received nonfinite wphase")
    all(>(0.0), wphase) || error("entropy received non-positive wphase")
    S = 0.0
    @inbounds for i in eachindex(w)
        wi = max(w[i], entropy_floor)
        S -= wi * (log(wi) + log(wphase[i]))
    end
    return S
end

function max_light_relative_residual(A_light::Matrix{Float64}, w::Vector{Float64}, light_target::Vector{Float64}; relative_floor::Float64=1e-12)
    size(A_light, 1) == length(light_target) || error("light_target length must match A_light rows")
    size(A_light, 2) == length(w) || error("w length must match A_light columns")
    isempty(light_target) && return Inf
    light_model = A_light * w
    denominator_floor = relative_floor * max(1.0, maximum(abs.(light_target)))
    max_relative_residual = 0.0
    @inbounds for i in eachindex(light_target)
        relative_residual = abs(light_model[i] - light_target[i]) / max(abs(light_target[i]), denominator_floor)
        max_relative_residual = max(max_relative_residual, relative_residual)
    end
    return max_relative_residual
end

##CHI##
@inline function chi2_block(A::Matrix{Float64}, w::Vector{Float64}, d::Vector{Float64}, sigma::Vector{Float64})
    p = A * w
    s = 0.0
    @inbounds for i in eachindex(d)
        si = max(sigma[i], 1e-12)
        rr = (p[i] - d[i]) / si
        s += rr * rr
    end
    return s
end

# ========================================================================================================================
# §3c  SHARED SPEAR LINEAR-ALGEBRA PRIMITIVES
# ========================================================================================================================
# Used by the expanded-Cm solver below.
@inline function _spear_safe_ddS(x::Float64; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    isfinite(x) || return -floor
    x < 0.0 || error("SPEAR requires a strictly negative entropy Hessian")
    return x > -floor ? -floor : x
end

# ========================================================================================================================
# §3d  KARL EXPANDED CM WITH LOSVD SLACK VARIABLES
# ========================================================================================================================
# This is the full Karl-style SPEAR shape:
#   rows    = light constraints followed by LOSVD constraints
#   columns = orbit weights followed by LOSVD slack variables
#
# The orbit solver returns only the orbit-weight part.  Slack variables are
# internal SPEAR variables that carry the LOSVD residual term the way Karl's
# entropy.f / spear.f system does.
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



# ========================================================================================================================

# ================================================================================================================================================================================================================================================

# ================================================================================================================================================================================================================================================



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

# ================================================================================================================================================================================================================================================

function solve_weights_karl_expanded_cm(A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, light_target::Vector{Float64}, light_sigma::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64};
    Nspatial::Int, Nvbin::Int, alphat::Float64=DEFAULT_KARL_ALPHAT, light_rel_tol::Float64=DEFAULT_KARL_LIGHT_REL_TOL, light_sigma_tol::Float64=2.0, delta_chi2_iter_tol::Float64=DEFAULT_KARL_DELTA_CHI2_ITER_TOL, wphase=nothing, 
    maxiter::Int=DEFAULT_KARL_MAXITER, seed::UInt=UInt(0), entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, apfac::Float64=DEFAULT_KARL_APFAC, return_diag::Bool=false, rcond_every::Int=250,
    rcond_warn::Float64=DEFAULT_KARL_SPEAR_RCOND_WARN, max_active_passes::Int=512, bound_kkt_tol::Float64=1.0e-10)

    Nlight, Norbit = size(A_light)
    Nlosvd, Norbit2 = size(A_losvd)
    fail_w = zeros(Float64, Norbit)
    Norbit == Norbit2 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    Nlight > 0 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match Nlosvd")
    length(light_target) == Nlight || error("light_target length does not match A_light")
    length(light_sigma) == Nlight || error("light_sigma length does not match A_light")
    length(losvd_target) == Nlosvd || error("losvd_target length does not match A_losvd")
    length(losvd_sigma) == Nlosvd || error("losvd_sigma length does not match A_losvd")
    iseven(Norbit) || error("Karl phase-volume solving requires paired orbit columns")
    light_sigma_use = max.(abs.(light_sigma), 1.0e-12)
    light_sigma_residual(w) = maximum(abs.(A_light * w .- light_target) ./ light_sigma_use)
    wp = _prepare_wphase(wphase, Norbit; require_paired=true, pair_rtol=1.0e-12)
    wphase_diag = karl_wphase_diagnostics(wp; paired=true)
    w = karl_initial_weights_from_wphase(wp; paired=true, rotfrac=0.75, floor=entropy_floor)
    enforce_normalization = !_expanded_light_implies_normalization(A_light, light_target)
    Cm = build_expanded_Cm_with_losvd_slack(A_light, A_losvd; enforce_normalization=enforce_normalization)
    target_base = build_expanded_target(light_target, losvd_target; enforce_normalization=enforce_normalization)
    w_all = build_expanded_weights_initial(w, A_losvd, losvd_target)
    initial_losvd = karl_losvd_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, Nspatial, Nvbin)
    previous_chi2_losvd = initial_losvd.chi_total
    delta_chi2_iteration = Inf
    delta_chi2_iteration_step_normalized = Inf
    max_light_relative_residual_value = max_light_relative_residual(A_light, w, light_target)
    max_light_sigma_residual_value = light_sigma_residual(w)
    light_constraint_ok = max_light_sigma_residual_value <= light_sigma_tol
    active_bound = falses(Norbit)
    last_diag = nothing
    last_rcond_est = NaN
    raw_spear_diag = nothing
    failure_reason = :none
    iterations = 0
    converged = false
    ok = true

    for iter in 1:maxiter
        iterations = iter
        compute_rcond = iter == 1 || iter == maxiter || iter % rcond_every == 0
        w_all_new, step_ok, sdiag = karl_spear_step_light_losvd_all(
            w_all, Norbit, Cm, target_base, A_losvd, losvd_target, losvd_sigma, wp;
            Nlight=Nlight, Nspatial=Nspatial, Nvbin=Nvbin, alphat=alphat, apfac=apfac,
            entropy_floor=entropy_floor, compute_rcond=compute_rcond, rcond_warn=rcond_warn,
            print_consistency=(iter == 1), active_bound=active_bound,
            max_active_passes=max_active_passes, bound_kkt_tol=bound_kkt_tol)
        last_diag = sdiag
        iter == 1 && (raw_spear_diag = sdiag.raw_spear_diag)
        isfinite(sdiag.rcond_est) && (last_rcond_est = sdiag.rcond_est)
        active_changed = any(sdiag.active_bound .!= active_bound)
        active_bound .= sdiag.active_bound
        if !step_ok
            failure_reason = sdiag.failure_reason
            ok = false
            break
        end
        state_scale = max(1.0, maximum(abs, w_all), maximum(abs, w_all_new))
        state_change = maximum(abs.(w_all_new .- w_all))
        state_tol = 100.0 * eps(Float64) * state_scale
        w_all .= w_all_new
        w_current = Vector{Float64}(@view w_all[1:Norbit])
        slack_current = Vector{Float64}(@view w_all[(Norbit + 1):end])
        losvd_state = karl_losvd_fracnew_state(A_losvd, w_current, losvd_target, losvd_sigma, Nspatial, Nvbin)
        chi2_losvd_current = losvd_state.chi_total
        delta_chi2_iteration = abs(chi2_losvd_current - previous_chi2_losvd)
        step_scale = isfinite(sdiag.stepfac) && sdiag.stepfac > 0.0 ? sdiag.stepfac : apfac
        delta_chi2_iteration_step_normalized = delta_chi2_iteration / step_scale
        max_light_relative_residual_value = max_light_relative_residual(A_light, w_current, light_target)
        max_light_sigma_residual_value = light_sigma_residual(w_current)
        light_constraint_ok = max_light_sigma_residual_value <= light_sigma_tol
        slack_residual_l2 = norm(slack_current .- losvd_state.residual)
        slack_scale = max(1.0, norm(slack_current), norm(losvd_state.residual))
        slack_consistent = slack_residual_l2 <= light_rel_tol * slack_scale
        normalized = abs(sum(w_current) - 1.0) <= light_rel_tol
        light_model = A_light * w_current
        light_sigma_progress = abs.(light_model .- light_target) ./ light_sigma_use
        worst_light_bin = argmax(light_sigma_progress)
        println("[WEIGHT PROGRESS] iteration=", iter,
            " active_passes=", sdiag.active_passes,
            " N_active_bound=", sdiag.n_active_bound,
            " active_changed=", active_changed,
            " stepfac=", sdiag.stepfac,
            " state_change=", state_change,
            " fracnew_min=", minimum(losvd_state.fracnew),
            " fracnew_max=", maximum(losvd_state.fracnew),
            " chi_losvd=", chi2_losvd_current,
            " max_light_sigma_residual=", light_sigma_progress[worst_light_bin],
            " worst_light_bin=", worst_light_bin,
            " delta_chi2=", delta_chi2_iteration,
            " delta_chi2_step_normalized=", delta_chi2_iteration_step_normalized)
        if light_constraint_ok && slack_consistent && normalized && delta_chi2_iteration_step_normalized <= delta_chi2_iter_tol
            converged = true
            break
        end
        chi_scale = max(1.0, abs(previous_chi2_losvd), abs(chi2_losvd_current))
        chi_stationary_tol = 100.0 * eps(Float64) * chi_scale
        stationary = !active_changed && state_change <= state_tol && delta_chi2_iteration <= chi_stationary_tol
        if stationary
            failure_reason = sdiag.active_set_stabilized ? :stationary_constraints_unsatisfied :
                (sdiag.active_set_failure_reason === :none ? :stationary_active_set : sdiag.active_set_failure_reason)
            println("[KARL STATIONARY] iteration=", iter,
                " chi_losvd=", chi2_losvd_current,
                " max_light_sigma_residual=", max_light_sigma_residual_value,
                " active_bound=", count(active_bound),
                " state_change=", state_change,
                " active_set_stabilized=", sdiag.active_set_stabilized,
                " active_set_reason=", sdiag.active_set_failure_reason,
                " failure_reason=", failure_reason)
            ok = false
            break
        end
        previous_chi2_losvd = chi2_losvd_current
    end
    w = Vector{Float64}(@view w_all[1:Norbit])
    slack = Vector{Float64}(@view w_all[(Norbit + 1):end])
    finite_state = all(isfinite, w) && all(isfinite, slack)
    !finite_state && (failure_reason = :nonfinite_final_state; ok = false)
    final_losvd = finite_state ? karl_losvd_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, Nspatial, Nvbin) : nothing
    chi2_losvd = final_losvd === nothing ? Inf : final_losvd.chi_total
    slack_residual_l2 = final_losvd === nothing ? Inf : norm(slack .- final_losvd.residual)
    slack_scale = final_losvd === nothing ? Inf : max(1.0, norm(slack), norm(final_losvd.residual))
    slack_consistent = finite_state && slack_residual_l2 <= light_rel_tol * slack_scale
    normalization_error = finite_state ? abs(sum(w) - 1.0) : Inf
    normalized = normalization_error <= light_rel_tol
    light_residual_l2 = finite_state ? norm(light_target .- A_light * w) : Inf
    max_light_relative_residual_value = finite_state ? max_light_relative_residual(A_light, w, light_target) : Inf
    max_light_sigma_residual_value = finite_state ? light_sigma_residual(w) : Inf
    light_constraint_ok = max_light_sigma_residual_value <= light_sigma_tol
    final_target = copy(target_base)
    final_losvd !== nothing && (final_target[(Nlight + 1):(Nlight + Nlosvd)] .= final_losvd.effective_target)
    constraint_l2 = finite_state ? norm(final_target .- Cm * w_all) : Inf
    constraint_ok = light_constraint_ok && slack_consistent && normalized
    delta_chi2_ok = isfinite(delta_chi2_iteration_step_normalized) && delta_chi2_iteration_step_normalized <= delta_chi2_iter_tol
    solver_converged = ok && converged && constraint_ok && delta_chi2_ok
    if ok && !light_constraint_ok
        failure_reason = :light_constraint_failed
    elseif ok && !slack_consistent
        failure_reason = :losvd_slack_inconsistent
    elseif ok && !normalized
        failure_reason = :normalization_failed
    elseif ok && !delta_chi2_ok
        failure_reason = :delta_chi2_not_converged
    elseif ok && !solver_converged
        failure_reason = :karl_not_converged
    end
    if return_diag
        ent = finite_state ? karl_entropy_value(w, wp; entropy_floor=entropy_floor) : -Inf
        losvd_penalty = alphat * chi2_losvd
        diag = (
            entropy=ent, chi=chi2_losvd, chi_losvd=chi2_losvd, profit=ent-losvd_penalty,
            alphat=alphat, losvd_penalty=losvd_penalty, chi_slack=losvd_penalty,
            slack_to_losvd=chi2_losvd > 0.0 ? losvd_penalty / chi2_losvd : NaN,
            fracnew=final_losvd === nothing ? Float64[] : final_losvd.fracnew,
            fracnew_min=final_losvd === nothing ? NaN : minimum(final_losvd.fracnew),
            fracnew_max=final_losvd === nothing ? NaN : maximum(final_losvd.fracnew),
            delta_chi2_iteration=delta_chi2_iteration,
            delta_chi2_iteration_step_normalized=delta_chi2_iteration_step_normalized,
            delta_chi2_iteration_ok=delta_chi2_ok,
            delta_chi2_iteration_tol=delta_chi2_iter_tol,
            max_light_relative_residual=max_light_relative_residual_value,
            max_light_sigma_residual=max_light_sigma_residual_value,
            light_constraint_ok=light_constraint_ok,
            light_rel_tol=light_rel_tol,
            light_sigma_tol=light_sigma_tol,
            solver_converged=solver_converged,
            failure_reason=failure_reason,
            rcond_est=last_rcond_est,
            max_abs_dw=last_diag === nothing ? NaN : last_diag.max_abs_dw,
            stepfac=last_diag === nothing ? NaN : last_diag.stepfac,
            iterations=iterations,
            constraint_ok=constraint_ok,
            slack_consistent=slack_consistent,
            normalized=normalized,
            normalization_error=normalization_error,
            light_residual_l2=light_residual_l2,
            constraint_l2=constraint_l2,
            slack_residual_l2=slack_residual_l2,
            slack_l2=finite_state ? sum(slack .^ 2) : Inf,
            slack_max_abs=finite_state && !isempty(slack) ? maximum(abs.(slack)) : 0.0,
            N_slack=Nlosvd,
            normalization_row_enforced=enforce_normalization,
            n_active_bound=count(active_bound),
            active_passes=last_diag === nothing ? 0 : last_diag.active_passes,
            wphase_required=true,
            wphase_convention=wphase_diag.convention,
            wphase_entropy_expression=wphase_diag.entropy_expression,
            wphase_min=wphase_diag.wphase_min,
            wphase_max=wphase_diag.wphase_max,
            wphase_dynamic_range=wphase_diag.wphase_dynamic_range,
            wphase_log_dynamic_range=wphase_diag.wphase_log_dynamic_range,
            wphase_geometric_mean=wphase_diag.wphase_geometric_mean,
            phase_volume_min=wphase_diag.phase_volume_min,
            phase_volume_max=wphase_diag.phase_volume_max,
            phase_volume_dynamic_range=wphase_diag.phase_volume_dynamic_range,
            wphase_pair_max_relative_mismatch=wphase_diag.pair_max_relative_mismatch,
            raw_spear_rank=raw_spear_diag === nothing ? 0 : raw_spear_diag.Am_rank,
            raw_spear_nullity=raw_spear_diag === nothing ? 0 : raw_spear_diag.Am_nullity,
            raw_spear_supported_rcond=raw_spear_diag === nothing ? NaN : raw_spear_diag.Am_supported_rcond,
            raw_spear_negative_orbits=raw_spear_diag === nothing ? 0 : raw_spear_diag.raw_negative_orbits,
            raw_spear_first_boundary_idx=raw_spear_diag === nothing ? 0 : raw_spear_diag.raw_first_boundary_idx,
            raw_spear_first_boundary_candidate=raw_spear_diag === nothing ? NaN : raw_spear_diag.raw_first_boundary_candidate,
            raw_spear_system_relative_residual=raw_spear_diag === nothing ? NaN : raw_spear_diag.spear_system_relative_residual,
        )

        return w, solver_converged, diag
    end

    return w, solver_converged
end

function karl_spear_step_light_losvd_all(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target_base::Vector{Float64}, A_losvd::Matrix{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}, wphase_orbit::Vector{Float64}; Nlight::Int, Nspatial::Int, Nvbin::Int, alphat::Float64=DEFAULT_KARL_ALPHAT, apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, compute_rcond::Bool=true, rcond_warn::Float64=DEFAULT_KARL_SPEAR_RCOND_WARN, print_consistency::Bool=false, active_bound=nothing, max_active_passes::Int=512, bound_kkt_tol::Float64=1.0e-10)
    Narr, nvar = size(Cm)
    Nlosvd, Norbit2 = size(A_losvd)

    Norbit2 == Norbit || error("A_losvd orbit count does not match Norbit")
    length(w_all) == nvar || error("w_all length does not match expanded Cm")
    length(target_base) == Narr || error("target length does not match expanded Cm rows")
    nvar - Norbit == Nlosvd || error("expanded slack count does not match LOSVD row count")
    length(wphase_orbit) == Norbit || error("wphase length does not match Norbit")
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match Nlosvd")

    active = active_bound === nothing ? falses(Norbit) : BitVector(active_bound)
    w_orbit = Vector{Float64}(@view w_all[1:Norbit])
    losvd_state = karl_losvd_fracnew_state(A_losvd, w_orbit, losvd_target, losvd_sigma, Nspatial, Nvbin)
    entropy, chi_slack, dS, ddS = build_expanded_entropy_derivatives(w_all, Norbit, wphase_orbit, losvd_state.residual, losvd_state.effective_sigma; alphat=alphat, entropy_floor=entropy_floor)

    target = copy(target_base)
    target[(Nlight + 1):(Nlight + Nlosvd)] .= losvd_state.effective_target

    raw_diag = print_consistency ? karl_raw_spear_consistency_diagnostic(w_all, Norbit, Cm, target, dS, ddS; apfac=apfac, entropy_floor=entropy_floor) : nothing
    update_diag = karl_spear_update_expanded(w_all, Norbit, Cm, target, dS, ddS; apfac=apfac, entropy_floor=entropy_floor, compute_rcond=compute_rcond, rcond_warn=rcond_warn, active_bound=active, max_active_passes=max_active_passes, bound_kkt_tol=bound_kkt_tol)

    wnew = Vector{Float64}(update_diag.w)
    finite_state = all(isfinite, wnew)
    negative_orbits = count(x -> x <= 0.0, @view wnew[1:Norbit])

    step_ok = finite_state

    failure_reason = finite_state ? :none : :nonfinite_updated_state
    step_warning = negative_orbits > 0 ? :negative_orbit_weights : :none

    diag = merge(update_diag, (
        entropy=entropy, chi_slack=chi_slack, fracnew=losvd_state.fracnew,
        effective_losvd_target=losvd_state.effective_target, effective_losvd_sigma=losvd_state.effective_sigma,
        losvd_residual=losvd_state.residual, chi_by_spatial=losvd_state.chi_by_spatial, chi_losvd_karl=losvd_state.chi_total,
        raw_spear_diag=raw_diag, slack=Vector{Float64}(wnew[(Norbit + 1):end]), w_all=wnew,
        failure_reason=failure_reason, step_warning=step_warning,
    ))

    return wnew, step_ok, diag
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
            effective_sigma[row] = losvd_sigma[row] == invalid_sigma_sentinel ?
                1.0e6 :
                max(abs(losvd_sigma[row] * frac), 1.0e-12)
            residual[row] = effective_target[row] - model[row]
            rr = residual[row] / effective_sigma[row]
            chi_by_spatial[ib] += rr * rr
        end
    end
    return ( model=model, fracnew=fracnew, effective_target=effective_target, effective_sigma=effective_sigma, residual=residual, chi_by_spatial=chi_by_spatial, chi_total=sum(chi_by_spatial))
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

function karl_spear_update_expanded(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, step_safety::Float64=DEFAULT_KARL_STEP_SAFETY, compute_rcond::Bool=true, rcond_warn::Float64=DEFAULT_KARL_SPEAR_RCOND_WARN, active_bound=nothing, max_active_passes::Int=512, bound_kkt_tol::Float64=1.0e-10)
    Narr, nvar = size(Cm)
    length(w_all) == nvar || error("w_all length must match Cm columns")
    0 < Norbit <= nvar || error("Norbit must be between 1 and length(w_all)")
    length(target) == Narr || error("target length must match Cm rows")
    length(dS) == nvar || error("dS length must match Cm columns")
    length(ddS) == nvar || error("ddS length must match Cm columns")

    model = Cm * w_all
    base_delY = target .- model
    rhs = copy(base_delY)

    Am = karl_spear_build_Am(Cm, ddS; floor=entropy_floor)
    karl_spear_rhs!(rhs, Cm, dS, ddS; floor=entropy_floor)

    lambda = _solve_spear_system(Am, rhs)
    dw = karl_spear_delta_w(Cm, lambda, dS, ddS; floor=entropy_floor)
    wnew = w_all .+ apfac .* dw

    spear_rhs_l2 = norm(rhs)
    spear_system_residual_l2 = norm(Am * lambda - rhs)
    spear_system_relative_residual = spear_system_residual_l2 / max(spear_rhs_l2, eps(Float64))

    rcond_est = NaN
    near_singular_spear_matrix = false
    if compute_rcond
        cond_est = try
            cond(Am)
        catch
            Inf
        end
        rcond_est = isfinite(cond_est) ? 1.0 / max(cond_est, 1.0) : 0.0
        near_singular_spear_matrix = !isfinite(rcond_est) || rcond_est <= rcond_warn
    end

    orbit_before = @view w_all[1:Norbit]
    orbit_after = @view wnew[1:Norbit]
    orbit_dw = @view dw[1:Norbit]

    min_orbit_weight = minimum(orbit_before)
    min_updated_orbit_weight, min_updated_orbit_idx = findmin(orbit_after)
    n_below_floor = count(x -> x < entropy_floor, orbit_after)
    n_nonpositive_orbits = count(x -> x <= 0.0, orbit_after)

    limiting_idx = 0
    limiting_candidate = Inf
    @inbounds for j in 1:Norbit
        if dw[j] < 0.0
            candidate = (w_all[j] - entropy_floor) / (-dw[j])
            if candidate < limiting_candidate
                limiting_candidate = candidate
                limiting_idx = j
            end
        end
    end

    limiting_weight = limiting_idx > 0 ? w_all[limiting_idx] : NaN
    limiting_dw = limiting_idx > 0 ? dw[limiting_idx] : NaN
    linearized_constraint_error_l2 = norm(Cm * (wnew .- w_all) .- apfac .* base_delY)
    post_step_constraint_l2 = norm(target .- Cm * wnew)

    println("[KARL RAW STEP] stepfac=", apfac, " negative_orbits=", n_nonpositive_orbits, " below_floor=", n_below_floor, " limiting_idx=", limiting_idx, " limiting_candidate=", limiting_candidate, " rcond_est=", rcond_est)

    return (w=Vector{Float64}(wnew), dw=Vector{Float64}(dw), lambda=Vector{Float64}(lambda), Am=Matrix{Float64}(Am), rhs=Vector{Float64}(rhs), active_bound=falses(Norbit), active_set_stabilized=true, active_set_failure_reason=:none, active_passes=1, recovery_locked_final=0, recovery_lock_peak=0, multi_release_recovery_total=0, n_active_bound=0, n_free_orbits_final=Norbit, n_activated_total=0, activated_events_total=0, boundary_events_total=0, max_batch_activated=0, last_batch_activated=0, last_boundary_batch_size=0, last_boundary_idx=0, last_boundary_candidate=Inf, n_release_candidates=0, max_release_candidates_seen=0, released_total=0, last_released_idx=0, last_released_multiplier=NaN, min_bound_multiplier=NaN, max_bound_multiplier=NaN, most_negative_bound_multiplier=0.0, bound_kkt_tolerance=bound_kkt_tol, weight_moved_to_floor=0.0, initial_min_trial_weight=min_updated_orbit_weight, initial_min_trial_idx=min_updated_orbit_idx, n_initial_violators=n_below_floor, initial_n_negative_dw=count(<(0.0), orbit_dw), limiting_idx=limiting_idx, limiting_candidate=limiting_candidate, rcond_est=rcond_est, rcond_min=rcond_est, rcond_warn=rcond_warn, near_singular_spear_matrix=near_singular_spear_matrix, near_singular_seen=near_singular_spear_matrix, reduced_Am_rank=Narr, minimum_reduced_Am_rank=Narr, weak_constraint_row=0, svd_fallback_used=false, svd_fallback_count=0, svd_relative_residual=spear_system_relative_residual, max_svd_relative_residual=spear_system_relative_residual, svd_residual_tol=NaN, model=model, delY=base_delY, max_abs_dw=maximum(abs, dw), spear_rhs_l2=spear_rhs_l2, spear_system_residual_l2=spear_system_residual_l2, spear_system_relative_residual=spear_system_relative_residual, requested_step=apfac, stepfac=apfac, step_safety=step_safety, step_limited=false, limiting_weight=limiting_weight, limiting_dw=limiting_dw, min_orbit_weight=min_orbit_weight, min_updated_orbit_weight=min_updated_orbit_weight, min_updated_orbit_idx=min_updated_orbit_idx, n_nonpositive_orbits=n_nonpositive_orbits, n_below_floor=n_below_floor, n_at_floor=count(x -> x <= entropy_floor, orbit_after), n_at_floor_before=count(x -> x <= entropy_floor, orbit_before), n_at_floor_after=count(x -> x <= entropy_floor, orbit_after), n_negative_dw=count(<(0.0), orbit_dw), strict_cycle_visits=0, cycle_break_release_total=0, locked_release_candidates=0, release_trial_count=0, release_locked_count=0, degenerate_rebinds_total=0, release_locks_total=0, linearized_constraint_error_l2=linearized_constraint_error_l2, post_step_constraint_l2=post_step_constraint_l2)
end



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

function _best_locked_active_set_recovery_release(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}, active::AbstractVector{Bool}, recovery_locked::AbstractVector{Bool}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    best_orbit = 0
    best_rank = -1
    best_residual = Inf
    best_primal = Inf
    best_floor_violators = typemax(Int)
    best_locked_violators = typemax(Int)
    best_locked_min_candidate = -Inf
    best_rebind = true
    limit = apfac * (1.0 + 100.0 * eps(Float64))

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
        locked_violators = 0
        locked_min_candidate = Inf
        released_rebind = false

        @inbounds for k in state.free_orbits
            if state.dw[k] < 0.0
                candidate = (w_all[k] - entropy_floor) / (-state.dw[k])

                if isfinite(candidate) && candidate >= 0.0 && candidate <= limit
                    floor_violators += 1
                    k == j && (released_rebind = true)

                    if recovery_locked[k]
                        locked_violators += 1
                        locked_min_candidate = min(locked_min_candidate, candidate)
                    end
                end
            end
        end

        score = (locked_violators, -state.rank, -locked_min_candidate, released_rebind ? 1 : 0, floor_violators, residual, primal)
        best_score = (best_locked_violators, -best_rank, -best_locked_min_candidate, best_rebind ? 1 : 0, best_floor_violators, best_residual, best_primal)

        if score < best_score
            best_orbit = j
            best_rank = state.rank
            best_residual = residual
            best_primal = primal
            best_floor_violators = floor_violators
            best_locked_violators = locked_violators
            best_locked_min_candidate = locked_min_candidate
            best_rebind = released_rebind
        end
    end

    return (orbit=best_orbit, rank=best_rank, residual=best_residual, primal=best_primal, floor_violators=best_floor_violators, locked_violators=best_locked_violators, locked_min_candidate=best_locked_min_candidate, released_rebind=best_rebind)
end

function _best_rank_deficiency_recovery_release(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}, active::AbstractVector{Bool}, Fsvd, current_rank::Int; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, topk::Int=96)
    Narr = size(Cm, 1)
    current_rank < Narr || return (orbit=0, rank=current_rank, residual=Inf, primal=Inf, floor_violators=typemax(Int), released_rebind=false, null_support=0.0)

    U_null = @view Fsvd.U[:, (current_rank + 1):Narr]
    support_candidates = NamedTuple[]

    for j in findall(active)
        col = @view Cm[:, j]
        projected = transpose(U_null) * col
        denom = abs(_spear_safe_ddS(ddS[j]; floor=entropy_floor))
        support = sum(abs2, projected) / denom

        if isfinite(support) && support > 0.0
            push!(support_candidates, (orbit=j, support=support))
        end
    end

    isempty(support_candidates) && return (orbit=0, rank=current_rank, residual=Inf, primal=Inf, floor_violators=typemax(Int), released_rebind=false, null_support=0.0)

    sort!(support_candidates; by=x -> -x.support)
    ntest = min(topk, length(support_candidates))

    best_orbit = 0
    best_rank = current_rank
    best_residual = Inf
    best_primal = Inf
    best_floor_violators = typemax(Int)
    best_rebind = false
    best_support = 0.0
    best_score = nothing

    for i in 1:ntest
        candidate = support_candidates[i]
        j = candidate.orbit
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
                boundary = (w_all[k] - entropy_floor) / (-state.dw[k])
                if isfinite(boundary) && boundary >= 0.0 && boundary <= apfac * (1.0 + 100.0 * eps(Float64))
                    floor_violators += 1
                    k == j && (released_rebind = true)
                end
            end
        end

        score = (-state.rank, -candidate.support, released_rebind ? 1 : 0, floor_violators, residual, primal)

        if best_score === nothing || score < best_score
            best_score = score
            best_orbit = j
            best_rank = state.rank
            best_residual = residual
            best_primal = primal
            best_floor_violators = floor_violators
            best_rebind = released_rebind
            best_support = candidate.support
        end
    end

    return (orbit=best_orbit, rank=best_rank, residual=best_residual, primal=best_primal, floor_violators=best_floor_violators, released_rebind=best_rebind, null_support=best_support)
end

function karl_spear_active_set_update(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}, active_bound::AbstractVector{Bool}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, compute_rcond::Bool=true, rcond_warn::Float64=DEFAULT_KARL_SPEAR_RCOND_WARN, max_active_passes::Int=512, bound_kkt_tol::Float64=1.0e-10, svd_residual_tol::Float64=1.0e-10)
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
    activated_events_total = 0
    released_total = 0

    max_batch_activated = 0
    last_batch_activated = 0
    last_boundary_idx = 0
    last_boundary_candidate = Inf
    last_boundary_batch_size = 0

    max_release_candidates_seen = 0
    last_release_candidates = 0
    last_released_idx = 0
    last_released_multiplier = NaN

    weight_moved_to_floor = 0.0

    rcond_est = NaN
    rcond_min = Inf
    near_singular_spear_matrix = false
    near_singular_seen = false

    reduced_Am_rank = 0
    minimum_reduced_Am_rank = Narr
    weak_constraint_row = 0

    svd_fallback_used = false
    svd_fallback_count = 0
    svd_relative_residual = NaN
    max_svd_relative_residual = 0.0

    initial_min_trial_weight = minimum(@view w_all[1:Norbit])
    initial_min_trial_idx = argmin(@view w_all[1:Norbit])
    initial_n_violators = 0
    initial_n_negative_dw = 0

    limiting_idx = 0
    limiting_candidate = Inf

    min_bound_multiplier = NaN
    max_bound_multiplier = NaN
    most_negative_bound_multiplier = 0.0

    active_set_failure_reason = :none
    active_passes = 0
    mask_first_seen = Dict{UInt,Int}()
    cycle_detected = false
    cycle_first_pass = 0
    cycle_repeat_pass = 0
    recovery_locked = falses(Norbit)
    recovery_lock_peak = 0
    multi_release_recovery_total = 0

    for pass in 1:max_active_passes
        active_passes = pass

        mask_hash = hash(recovery_locked, hash(active))
        if haskey(mask_first_seen, mask_hash) && !cycle_detected
            cycle_detected = true
            cycle_first_pass = mask_first_seen[mask_hash]
            cycle_repeat_pass = pass
            println("[KARL ACTIVE CYCLE] first_pass=", cycle_first_pass, " repeat_pass=", cycle_repeat_pass, " bound=", count(active))
        else
            mask_first_seen[mask_hash] = pass
        end

        free_orbits = findall(!, active)

        if isempty(free_orbits)
            recovery = _best_active_set_recovery_release(w_all, Norbit, Cm, target, dS, ddS, active; apfac=apfac, entropy_floor=entropy_floor)

            if recovery.orbit > 0
                active[recovery.orbit] = false
                recovery_locked[recovery.orbit] = recovery.released_rebind
                recovery_lock_peak = max(recovery_lock_peak, count(recovery_locked))
                released_total += 1
                last_released_idx = recovery.orbit
                last_released_multiplier = NaN
                println("[KARL ACTIVE RECOVERY] pass=", pass, " cause=no_free_orbits orbit=", recovery.orbit, " trial_rank=", recovery.rank, "/", Narr, " trial_residual=", recovery.residual, " floor_violators=", recovery.floor_violators, " released_rebind=", recovery.released_rebind, " remaining_bound=", count(active))
                continue
            end

            active_set_failure_reason = :no_free_orbits
            break
        end

        free_vars = vcat(free_orbits, collect((Norbit + 1):nvar))

        if length(free_vars) < Narr
            recovery = _best_active_set_recovery_release(w_all, Norbit, Cm, target, dS, ddS, active; apfac=apfac, entropy_floor=entropy_floor)

            if recovery.orbit > 0
                active[recovery.orbit] = false
                recovery_locked[recovery.orbit] = recovery.released_rebind
                recovery_lock_peak = max(recovery_lock_peak, count(recovery_locked))
                released_total += 1
                last_released_idx = recovery.orbit
                last_released_multiplier = NaN
                println("[KARL ACTIVE RECOVERY] pass=", pass, " cause=insufficient_free_variables orbit=", recovery.orbit, " trial_rank=", recovery.rank, "/", Narr, " trial_residual=", recovery.residual, " floor_violators=", recovery.floor_violators, " released_rebind=", recovery.released_rebind, " remaining_bound=", count(active))
                continue
            end

            active_set_failure_reason = :insufficient_free_variables
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

        Fsvd = svd(Am)
        smax = isempty(Fsvd.S) ? 0.0 : maximum(Fsvd.S)
        rank_tol = max(size(Am)...) * eps(Float64) * max(smax, 1.0)

        reduced_Am_rank = count(s -> s > rank_tol, Fsvd.S)
        minimum_reduced_Am_rank = min(minimum_reduced_Am_rank, reduced_Am_rank)
        weak_constraint_row = isempty(Fsvd.S) ? 0 : argmax(abs.(Fsvd.U[:, end]))

        if compute_rcond
            this_rcond = isempty(Fsvd.S) || smax == 0.0 ? 0.0 : minimum(Fsvd.S) / smax
            rcond_est = this_rcond
            rcond_min = min(rcond_min, this_rcond)
            near_singular_spear_matrix = !isfinite(this_rcond) || this_rcond <= rcond_warn
            near_singular_seen |= near_singular_spear_matrix
        end

        lambda = Float64[]

        if reduced_Am_rank < Narr
            coeff = transpose(Fsvd.U) * rhs

            @inbounds for i in eachindex(coeff)
                coeff[i] = Fsvd.S[i] > rank_tol ? coeff[i] / Fsvd.S[i] : 0.0
            end

            lambda = Fsvd.V * coeff

            residual_norm = norm(Am * lambda - rhs)
            rhs_norm = norm(rhs)
            svd_relative_residual = residual_norm / max(rhs_norm, eps(Float64))
            max_svd_relative_residual = max(max_svd_relative_residual, svd_relative_residual)
            svd_fallback_used = true
            svd_fallback_count += 1

            println("[KARL ACTIVE SVD] pass=", pass, " bound=", count(active), " free=", length(free_orbits), " Am_rank=", reduced_Am_rank, " Am_rcond=", rcond_est, " weak_constraint_row=", weak_constraint_row, " relative_residual=", svd_relative_residual, " tolerance=", svd_residual_tol)

            if !isfinite(svd_relative_residual) || svd_relative_residual > svd_residual_tol
                recovery = _best_rank_deficiency_recovery_release(w_all, Norbit, Cm, target, dS, ddS, active, Fsvd, reduced_Am_rank; apfac=apfac, entropy_floor=entropy_floor)

                if recovery.orbit > 0
                    active[recovery.orbit] = false
                    recovery_locked[recovery.orbit] = recovery.released_rebind
                    recovery_lock_peak = max(recovery_lock_peak, count(recovery_locked))
                    released_total += 1
                    last_released_idx = recovery.orbit
                    last_released_multiplier = NaN
                    println("[KARL ACTIVE RECOVERY] pass=", pass, " cause=reduced_spear_inconsistent orbit=", recovery.orbit, " old_rank=", reduced_Am_rank, "/", Narr, " old_residual=", svd_relative_residual, " trial_rank=", recovery.rank, "/", Narr, " trial_residual=", recovery.residual, " floor_violators=", recovery.floor_violators, " released_rebind=", recovery.released_rebind, " remaining_bound=", count(active))
                    continue
                end

                active_set_failure_reason = :reduced_spear_inconsistent
                break
            end
        else
            lambda = try
                _solve_spear_system(Am, rhs)
            catch
                active_set_failure_reason = :reduced_spear_solve_failed
                break
            end
        end

        svd_relative_residual = norm(Am * lambda - rhs) / max(norm(rhs), eps(Float64))
        max_svd_relative_residual = max(max_svd_relative_residual, svd_relative_residual)

        dw_free = karl_spear_delta_w(Cm_free, lambda, dS_free, ddS_free; floor=entropy_floor)

        dw = copy(fixed_dw)
        dw[free_vars] .= dw_free

        trial = w_all .+ apfac .* dw

        @inbounds for j in 1:Norbit
            active[j] && (trial[j] = entropy_floor)
        end

        if pass == 1
            initial_min_trial_weight, initial_min_trial_idx = findmin(@view trial[1:Norbit])
            initial_n_violators = count(x -> x < entropy_floor, @view trial[1:Norbit])
            initial_n_negative_dw = count(<(0.0), @view dw[1:Norbit])

            @inbounds for j in free_orbits
                if dw[j] < 0.0
                    candidate = (w_all[j] - entropy_floor) / (-dw[j])

                    if candidate < limiting_candidate
                        limiting_candidate = candidate
                        limiting_idx = j
                    end
                end
            end
        end

        boundary_indices = Int[]
        boundary_candidates = Float64[]
        locked_boundary_indices = Int[]
        locked_boundary_candidates = Float64[]

        @inbounds for j in free_orbits
            if dw[j] < 0.0
                candidate = (w_all[j] - entropy_floor) / (-dw[j])
                if isfinite(candidate) && candidate >= 0.0 && candidate <= apfac * (1.0 + 100.0 * eps(Float64))
                    if recovery_locked[j]
                        push!(locked_boundary_indices, j)
                        push!(locked_boundary_candidates, candidate)
                    else
                        push!(boundary_indices, j)
                        push!(boundary_candidates, candidate)
                    end
                elseif recovery_locked[j]
                    recovery_locked[j] = false
                end
            elseif recovery_locked[j]
                recovery_locked[j] = false
            end
        end

        boundary_candidate = isempty(boundary_candidates) ? Inf : minimum(boundary_candidates)
        boundary_idx = isempty(boundary_candidates) ? 0 : boundary_indices[argmin(boundary_candidates)]

        if compute_rcond
            println("[KARL ACTIVE PASS] pass=", pass, " bound=", count(active), " free=", length(free_orbits), " Am_rank=", reduced_Am_rank, " Am_rcond=", rcond_est, " weak_constraint_row=", weak_constraint_row, " boundary_idx=", boundary_idx, " boundary_candidate=", boundary_candidate, " boundary_batch=", length(boundary_indices), " recovery_locked_floor=", length(locked_boundary_indices), " near_singular=", near_singular_spear_matrix)
        end

        if !isempty(boundary_indices)
            activated_events_total += 1
            n_activated_total += length(boundary_indices)
            max_batch_activated = max(max_batch_activated, length(boundary_indices))
            last_batch_activated = length(boundary_indices)
            last_boundary_batch_size = length(boundary_indices)
            last_boundary_idx = boundary_idx
            last_boundary_candidate = boundary_candidate

            @inbounds for j in boundary_indices
                weight_moved_to_floor += max(w_all[j] - entropy_floor, 0.0)
                active[j] = true
            end

            println("[KARL ACTIVE BIND] pass=", pass, " batch=", length(boundary_indices), " first_orbit=", boundary_idx, " first_candidate=", boundary_candidate, " recovery_locked_floor=", length(locked_boundary_indices), " remaining_free=", Norbit-count(active))
            continue
        end

        if !isempty(locked_boundary_indices)
            recovery = _best_locked_active_set_recovery_release(w_all, Norbit, Cm, target, dS, ddS, active, recovery_locked; apfac=apfac, entropy_floor=entropy_floor)

            if recovery.orbit > 0
                active[recovery.orbit] = false
                recovery_locked[recovery.orbit] = recovery.released_rebind
                recovery_lock_peak = max(recovery_lock_peak, count(recovery_locked))
                released_total += 1
                multi_release_recovery_total += 1
                last_released_idx = recovery.orbit
                last_released_multiplier = NaN
                ilocked = argmin(locked_boundary_candidates)

                println("[KARL ACTIVE MULTIRELEASE] pass=", pass, " locked_floor=", length(locked_boundary_indices), " locked_first_orbit=", locked_boundary_indices[ilocked], " locked_first_candidate=", locked_boundary_candidates[ilocked], " released_orbit=", recovery.orbit, " trial_rank=", recovery.rank, "/", Narr, " trial_residual=", recovery.residual, " floor_violators=", recovery.floor_violators, " locked_violators=", recovery.locked_violators, " locked_trial_candidate=", recovery.locked_min_candidate, " released_rebind=", recovery.released_rebind, " total_locked=", count(recovery_locked), " remaining_bound=", count(active))
                continue
            end

            active_set_failure_reason = :active_set_recovery_floor_trap
            break
        end

        release_candidates = Int[]
        release_multipliers = Float64[]
        bound_values = Float64[]

        @inbounds for j in 1:Norbit
            if active[j]
                projected_gradient = dot(@view(Cm[:, j]), lambda)
                multiplier = projected_gradient - dS[j]

                push!(bound_values, multiplier)

                scale = max(1.0, abs(dS[j]), abs(projected_gradient))

                if multiplier < -bound_kkt_tol * scale
                    push!(release_candidates, j)
                    push!(release_multipliers, multiplier)
                end
            end
        end

        if isempty(bound_values)
            min_bound_multiplier = NaN
            max_bound_multiplier = NaN
            most_negative_bound_multiplier = 0.0
        else
            min_bound_multiplier = minimum(bound_values)
            max_bound_multiplier = maximum(bound_values)
            most_negative_bound_multiplier = min(0.0, min_bound_multiplier)
        end

        last_release_candidates = length(release_candidates)
        max_release_candidates_seen = max(max_release_candidates_seen, last_release_candidates)

        if !isempty(release_candidates)
            imin = argmin(release_multipliers)
            last_released_idx = release_candidates[imin]
            last_released_multiplier = release_multipliers[imin]

            @inbounds for j in release_candidates
                active[j] = false
            end
            released_total += length(release_candidates)

            println("[KARL ACTIVE RELEASE] pass=", pass, " batch=", length(release_candidates), " most_negative_orbit=", last_released_idx, " multiplier=", last_released_multiplier, " remaining_bound=", count(active))

            continue
        end

        return (
            w=Vector{Float64}(trial), dw=Vector{Float64}(dw), lambda=Vector{Float64}(lambda), Am=Matrix{Float64}(Am), rhs=Vector{Float64}(rhs), active_bound=active,
            active_set_stabilized=true, active_set_failure_reason=:none, active_passes=active_passes,
            cycle_detected=cycle_detected, cycle_first_pass=cycle_first_pass, cycle_repeat_pass=cycle_repeat_pass,
            recovery_locked_final=count(recovery_locked), recovery_lock_peak=recovery_lock_peak, multi_release_recovery_total=multi_release_recovery_total,
            n_active_bound=count(active), n_free_orbits_final=Norbit-count(active),
            n_activated_total=n_activated_total, activated_events_total=activated_events_total, boundary_events_total=activated_events_total,
            max_batch_activated=max_batch_activated, last_batch_activated=last_batch_activated, last_boundary_batch_size=last_boundary_batch_size,
            last_boundary_idx=last_boundary_idx, last_boundary_candidate=last_boundary_candidate,
            n_release_candidates=last_release_candidates, max_release_candidates_seen=max_release_candidates_seen, released_total=released_total,
            last_released_idx=last_released_idx, last_released_multiplier=last_released_multiplier,
            min_bound_multiplier=min_bound_multiplier, max_bound_multiplier=max_bound_multiplier, most_negative_bound_multiplier=most_negative_bound_multiplier,
            bound_kkt_tolerance=bound_kkt_tol, weight_moved_to_floor=weight_moved_to_floor,
            initial_min_trial_weight=initial_min_trial_weight, initial_min_trial_idx=initial_min_trial_idx,
            n_initial_violators=initial_n_violators, initial_n_negative_dw=initial_n_negative_dw,
            limiting_idx=limiting_idx, limiting_candidate=limiting_candidate,
            rcond_est=rcond_est, rcond_min=isfinite(rcond_min) ? rcond_min : NaN, rcond_warn=rcond_warn,
            near_singular_spear_matrix=near_singular_spear_matrix, near_singular_seen=near_singular_seen,
            reduced_Am_rank=reduced_Am_rank, minimum_reduced_Am_rank=minimum_reduced_Am_rank, weak_constraint_row=weak_constraint_row,
            svd_fallback_used=svd_fallback_used, svd_fallback_count=svd_fallback_count,
            svd_relative_residual=svd_relative_residual, max_svd_relative_residual=max_svd_relative_residual, svd_residual_tol=svd_residual_tol,
        )
    end

    return (
        w=copy(w_all), dw=zeros(Float64, nvar), lambda=Float64[], Am=zeros(Float64, 0, 0), rhs=Float64[], active_bound=active,
        active_set_stabilized=false, active_set_failure_reason=active_set_failure_reason == :none ? :active_set_max_passes : active_set_failure_reason,
        active_passes=active_passes,
        cycle_detected=cycle_detected, cycle_first_pass=cycle_first_pass, cycle_repeat_pass=cycle_repeat_pass,
        recovery_locked_final=count(recovery_locked), recovery_lock_peak=recovery_lock_peak, multi_release_recovery_total=multi_release_recovery_total,
        n_active_bound=count(active), n_free_orbits_final=Norbit-count(active),
        n_activated_total=n_activated_total, activated_events_total=activated_events_total, boundary_events_total=activated_events_total,
        max_batch_activated=max_batch_activated, last_batch_activated=last_batch_activated, last_boundary_batch_size=last_boundary_batch_size,
        last_boundary_idx=last_boundary_idx, last_boundary_candidate=last_boundary_candidate,
        n_release_candidates=last_release_candidates, max_release_candidates_seen=max_release_candidates_seen, released_total=released_total,
        last_released_idx=last_released_idx, last_released_multiplier=last_released_multiplier,
        min_bound_multiplier=min_bound_multiplier, max_bound_multiplier=max_bound_multiplier, most_negative_bound_multiplier=most_negative_bound_multiplier,
        bound_kkt_tolerance=bound_kkt_tol, weight_moved_to_floor=weight_moved_to_floor,
        initial_min_trial_weight=initial_min_trial_weight, initial_min_trial_idx=initial_min_trial_idx,
        n_initial_violators=initial_n_violators, initial_n_negative_dw=initial_n_negative_dw,
        limiting_idx=limiting_idx, limiting_candidate=limiting_candidate,
        rcond_est=rcond_est, rcond_min=isfinite(rcond_min) ? rcond_min : NaN, rcond_warn=rcond_warn,
        near_singular_spear_matrix=near_singular_spear_matrix, near_singular_seen=near_singular_seen,
        reduced_Am_rank=reduced_Am_rank, minimum_reduced_Am_rank=minimum_reduced_Am_rank, weak_constraint_row=weak_constraint_row,
        svd_fallback_used=svd_fallback_used, svd_fallback_count=svd_fallback_count,
        svd_relative_residual=svd_relative_residual, max_svd_relative_residual=max_svd_relative_residual, svd_residual_tol=svd_residual_tol,
    )
end




# ================================================================================================================================================================================================================================================

# ================================================================================================================================================================================================================================================
# CHI
# ========================================================================================================================

function karl_raw_spear_consistency_diagnostic(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    model = Cm * w_all
    rhs = target .- model
    Am = karl_spear_build_Am(Cm, ddS; floor=entropy_floor)
    karl_spear_rhs!(rhs, Cm, dS, ddS; floor=entropy_floor)

    F = svd(Am)
    smax = maximum(F.S)
    tol = max(size(Am)...) * eps(Float64) * smax
    rank_Am = count(s -> s > tol, F.S)
    nullity_Am = length(F.S) - rank_Am
    supported = F.S[F.S .> tol]
    supported_rcond = isempty(supported) || smax == 0.0 ? 0.0 : minimum(supported) / smax

    lambda = _solve_spear_system(Am, rhs)
    dw = karl_spear_delta_w(Cm, lambda, dS, ddS; floor=entropy_floor)
    trial = w_all .+ apfac .* dw
    orbit_trial = @view trial[1:Norbit]
    min_weight, min_idx = findmin(orbit_trial)

    first_boundary = Inf
    first_boundary_idx = 0
    @inbounds for j in 1:Norbit
        if dw[j] < 0.0
            candidate = (w_all[j] - entropy_floor) / (-dw[j])
            if candidate < first_boundary
                first_boundary = candidate
                first_boundary_idx = j
            end
        end
    end

    system_residual = norm(Am * lambda - rhs) / max(norm(rhs), eps(Float64))

    println("[KARL CONSISTENCY] solve=bunchkaufman Am_rank=", rank_Am, " Am_nullity=", nullity_Am, " Am_supported_rcond=", supported_rcond, " raw_min_orbit_weight=", min_weight, " raw_min_orbit_idx=", min_idx, " raw_negative_orbits=", count(<(0.0), orbit_trial), " raw_negative_dw=", count(<(0.0), @view(dw[1:Norbit])), " raw_first_boundary_idx=", first_boundary_idx, " raw_first_boundary_candidate=", first_boundary, " raw_max_abs_dw=", maximum(abs, dw), " spear_system_relative_residual=", system_residual)

    return (Am_rank=rank_Am, Am_nullity=nullity_Am, Am_supported_rcond=supported_rcond, raw_min_orbit_weight=min_weight, raw_min_orbit_idx=min_idx, raw_negative_orbits=count(<(0.0), orbit_trial), raw_negative_dw=count(<(0.0), @view(dw[1:Norbit])), raw_first_boundary_idx=first_boundary_idx, raw_first_boundary_candidate=first_boundary, raw_max_abs_dw=maximum(abs, dw), spear_system_relative_residual=system_residual)
end

function _project_expanded_weights!(w_all::Vector{Float64}, Norbit::Int; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    0 < Norbit <= length(w_all) || error("Norbit must be between 1 and length(w_all)")
    all(isfinite, w_all) || return false
    @inbounds for j in 1:Norbit
        w_all[j] >= floor || return false
    end
    return true
end

function karl_safe_step_factor(w::AbstractVector{Float64}, dw::AbstractVector{Float64}; requested_step::Float64=DEFAULT_KARL_APFAC, floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, safety::Float64=DEFAULT_KARL_STEP_SAFETY, active_bound=nothing)
    length(w) == length(dw) || error("w and dw lengths must match")
    isfinite(requested_step) && requested_step > 0.0 || error("requested_step must be finite and positive")
    isfinite(safety) && 0.0 < safety < 1.0 || error("safety must lie strictly between 0 and 1")
    isfinite(floor) && floor > 0.0 || error("floor must be finite and positive")
    active_bound !== nothing && length(active_bound) != length(w) && error("active_bound length must match w")

    boundary = Inf
    limiting_idx = 0

    @inbounds for j in eachindex(w)
        isfinite(w[j]) && isfinite(dw[j]) || error("nonfinite weight or Newton direction")
        w[j] >= floor || error("orbit weight $j is already below the entropy floor")
        active_bound !== nothing && active_bound[j] && continue

        if dw[j] < 0.0
            candidate = (w[j] - floor) / (-dw[j])
            if candidate < boundary
                boundary = candidate
                limiting_idx = j
            end
        end
    end

    if isfinite(boundary) && boundary <= 0.0
        return 0.0, limiting_idx, boundary
    end

    stepfac = isfinite(boundary) ? min(requested_step, safety * boundary) : requested_step
    return stepfac, limiting_idx, boundary
end



