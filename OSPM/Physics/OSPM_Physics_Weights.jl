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

@inline function _safe_positive(x::Float64; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    return (isfinite(x) && x > floor) ? x : floor
end

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

# ========================================================================================================================
# §3c  SHARED SPEAR LINEAR-ALGEBRA PRIMITIVES
# ========================================================================================================================
# Used by the expanded-Cm solver below.
@inline function _spear_safe_ddS(x::Float64; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    isfinite(x) || return -floor
    x < 0.0 || error("SPEAR requires a strictly negative entropy Hessian")
    return x > -floor ? -floor : x
end

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
# CHI
# ========================================================================================================================
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


# ========================================================================================================================
# Working AREA 
# ====================================================================================================================================================================================

function karl_safe_step_factor(w::AbstractVector{Float64}, dw::AbstractVector{Float64}; requested_step::Float64=DEFAULT_KARL_APFAC, floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, safety::Float64=DEFAULT_KARL_STEP_SAFETY)
    length(w) == length(dw) || error("w and dw lengths must match")
    isfinite(requested_step) && requested_step > 0.0 || error("requested_step must be finite and positive")
    isfinite(safety) && 0.0 < safety < 1.0 || error("safety must lie strictly between 0 and 1")
    isfinite(floor) && floor > 0.0 || error("floor must be finite and positive")

    boundary = Inf
    limiting_idx = 0

    @inbounds for j in eachindex(w)
        isfinite(w[j]) && isfinite(dw[j]) || error("nonfinite weight or Newton direction")
        w[j] >= floor || error("orbit weight $j is already below the entropy floor")

        if dw[j] < 0.0
            candidate = (w[j] - floor) / (-dw[j])

            if candidate < boundary
                boundary = candidate
                limiting_idx = j
            end
        end
    end

    if isfinite(boundary) && boundary <= 0.0
        error("Karl Newton direction cannot move without crossing the entropy floor; limiting orbit=$limiting_idx boundary=$boundary")
    end

    stepfac = isfinite(boundary) ? min(requested_step, safety * boundary) : requested_step
    isfinite(stepfac) && stepfac > 0.0 || error("Karl safe step factor is non-positive or nonfinite")

    return stepfac, limiting_idx, boundary
end


function karl_spear_update_expanded(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64};
    apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, step_safety::Float64=DEFAULT_KARL_STEP_SAFETY, compute_rcond::Bool=true)

    Narr, nvar = size(Cm)

    length(w_all) == nvar || error("w_all length must match Cm columns")
    0 < Norbit <= nvar || error("Norbit must be between 1 and length(w_all)")
    length(target) == Narr || error("target length must match Cm rows")
    length(dS) == nvar || error("dS length must match Cm columns")
    length(ddS) == nvar || error("ddS length must match Cm columns")
    isfinite(apfac) && apfac > 0.0 || error("apfac must be finite and positive")
    isfinite(step_safety) && 0.0 < step_safety < 1.0 || error("step_safety must lie strictly between 0 and 1")

    model = Cm * w_all
    base_delY = target .- model

    Am = karl_spear_build_Am(Cm, ddS; floor=entropy_floor)
    rhs = copy(base_delY)
    karl_spear_rhs!(rhs, Cm, dS, ddS; floor=entropy_floor)

    lambda = _solve_spear_system(Am, rhs)
    dw = karl_spear_delta_w(Cm, lambda, dS, ddS; floor=entropy_floor)

    spear_rhs_l2 = norm(rhs)
    spear_system_residual_l2 = norm(Am * lambda - rhs)
    spear_system_relative_residual = spear_system_residual_l2 / max(spear_rhs_l2, eps(Float64))

    full_trial = w_all .+ apfac .* dw
    full_trial_orbits = @view full_trial[1:Norbit]
    initial_min_trial_weight, initial_min_trial_idx = findmin(full_trial_orbits)
    n_initial_violators = count(x -> x < entropy_floor, full_trial_orbits)
    initial_n_negative_dw = count(x -> x < 0.0, @view(dw[1:Norbit]))

    stepfac, limiting_idx, limiting_candidate = karl_safe_step_factor(@view(w_all[1:Norbit]), @view(dw[1:Norbit]); requested_step=apfac, floor=entropy_floor, safety=step_safety)

    wnew = w_all .+ stepfac .* dw
    all(isfinite, wnew) || error("Karl damped SPEAR step produced nonfinite weights")

    orbit_before = @view w_all[1:Norbit]
    orbit_after = @view wnew[1:Norbit]
    orbit_dw = @view dw[1:Norbit]

    min_orbit_weight = minimum(orbit_before)
    min_updated_orbit_weight, min_updated_orbit_idx = findmin(orbit_after)
    n_at_floor_before = count(x -> x <= entropy_floor, orbit_before)
    n_at_floor_after = count(x -> x <= entropy_floor, orbit_after)
    n_below_floor = count(x -> x < entropy_floor, orbit_after)
    n_nonpositive_orbits = count(x -> x <= 0.0, orbit_after)

    n_below_floor == 0 || error("Karl damped step crossed the entropy floor")
    n_nonpositive_orbits == 0 || error("Karl damped step produced non-positive orbit weights")

    limiting_weight = limiting_idx > 0 ? w_all[limiting_idx] : NaN
    limiting_dw = limiting_idx > 0 ? dw[limiting_idx] : NaN
    step_limited = stepfac < apfac * (1.0 - 100.0 * eps(Float64))

    linearized_constraint_error_l2 = norm(stepfac .* (Cm * dw) .- base_delY)
    post_step_constraint_l2 = norm(target .- Cm * wnew)
    rcond_est = compute_rcond ? 1.0 / max(cond(Am), 1.0) : NaN
    max_abs_dw = maximum(abs, dw)

    first_row = @view Cm[1, :]
    first_orbit_row = @view Cm[1, 1:Norbit]
    first_constraint_n_contributors = 0
    first_constraint_weight_on_contributors_before = 0.0
    first_constraint_weight_on_contributors_after = 0.0
    first_constraint_max_orbit_coefficient = 0.0
    first_constraint_max_contribution_before = 0.0
    first_constraint_max_contribution_after = 0.0
    first_constraint_max_contribution_idx = 0

    @inbounds for j in 1:Norbit
        coefficient = first_orbit_row[j]

        if coefficient > 0.0
            first_constraint_n_contributors += 1
            first_constraint_weight_on_contributors_before += w_all[j]
            first_constraint_weight_on_contributors_after += wnew[j]
            first_constraint_max_orbit_coefficient = max(first_constraint_max_orbit_coefficient, coefficient)

            contribution_before = coefficient * w_all[j]
            contribution_after = coefficient * wnew[j]

            if contribution_before > first_constraint_max_contribution_before
                first_constraint_max_contribution_before = contribution_before
                first_constraint_max_contribution_idx = j
            end

            first_constraint_max_contribution_after = max(first_constraint_max_contribution_after, contribution_after)
        end
    end

    first_constraint_target = target[1]
    first_constraint_model_before = model[1]
    first_constraint_model_after = dot(first_row, wnew)
    first_constraint_denominator = max(abs(first_constraint_target), eps(Float64))
    first_constraint_relative_before = abs(first_constraint_model_before - first_constraint_target) / first_constraint_denominator
    first_constraint_relative_after = abs(first_constraint_model_after - first_constraint_target) / first_constraint_denominator

    println("[KARL DAMPED STEP] requested_step=", apfac,
        " stepfac=", stepfac,
        " safety=", step_safety,
        " limited=", step_limited,
        " limiting_orbit=", limiting_idx,
        " boundary=", limiting_candidate,
        " full_step_violators=", n_initial_violators,
        " full_step_min_weight=", initial_min_trial_weight,
        " updated_min_weight=", min_updated_orbit_weight,
        " rcond_est=", rcond_est)

    return (
        w=wnew,
        dw=dw,
        lambda=lambda,
        Am=Am,
        delY=base_delY,
        model=model,
        rcond_est=rcond_est,
        max_abs_dw=max_abs_dw,
        spear_rhs_l2=spear_rhs_l2,
        spear_system_residual_l2=spear_system_residual_l2,
        spear_system_relative_residual=spear_system_relative_residual,

        stepfac=stepfac,
        requested_step=apfac,
        step_safety=step_safety,
        step_limited=step_limited,
        limiting_idx=limiting_idx,
        limiting_weight=limiting_weight,
        limiting_dw=limiting_dw,
        limiting_candidate=limiting_candidate,

        min_orbit_weight=min_orbit_weight,
        min_updated_orbit_weight=min_updated_orbit_weight,
        min_updated_orbit_idx=min_updated_orbit_idx,
        initial_min_trial_weight=initial_min_trial_weight,
        initial_min_trial_idx=initial_min_trial_idx,
        n_initial_violators=n_initial_violators,
        initial_n_negative_dw=initial_n_negative_dw,
        n_nonpositive_orbits=n_nonpositive_orbits,
        n_below_floor=n_below_floor,
        n_at_floor=n_at_floor_after,
        n_at_floor_before=n_at_floor_before,
        n_at_floor_after=n_at_floor_after,
        n_negative_dw=count(x -> x < 0.0, orbit_dw),

        # Compatibility fields: there is no active-set solve anymore.
        n_active_bound=0,
        n_activated_total=0,
        activated_events_total=0,
        n_free_orbits_final=Norbit,
        active_passes=1,
        active_set_stabilized=true,
        boundary_events_total=step_limited ? 1 : 0,
        max_batch_activated=0,
        last_batch_activated=0,
        last_boundary_idx=limiting_idx,
        last_boundary_candidate=limiting_candidate,
        last_boundary_batch_size=0,
        n_release_candidates=0,
        max_release_candidates_seen=0,
        released_total=0,
        last_released_idx=0,
        last_released_multiplier=NaN,
        most_negative_bound_multiplier=0.0,
        min_bound_multiplier=NaN,
        max_bound_multiplier=NaN,
        bound_kkt_tolerance=NaN,
        strict_cycle_visits=0,
        cycle_break_release_total=0,
        locked_release_candidates=0,
        release_trial_count=0,
        release_locked_count=0,
        degenerate_rebinds_total=0,
        release_locks_total=0,
        weight_moved_to_floor=0.0,

        linearized_constraint_error_l2=linearized_constraint_error_l2,
        post_step_constraint_l2=post_step_constraint_l2,

        first_constraint_target=first_constraint_target,
        first_constraint_model_before=first_constraint_model_before,
        first_constraint_model_after=first_constraint_model_after,
        first_constraint_relative_before=first_constraint_relative_before,
        first_constraint_relative_after=first_constraint_relative_after,
        first_constraint_n_contributors=first_constraint_n_contributors,
        first_constraint_weight_on_contributors_before=first_constraint_weight_on_contributors_before,
        first_constraint_weight_on_contributors_after=first_constraint_weight_on_contributors_after,
        first_constraint_max_orbit_coefficient=first_constraint_max_orbit_coefficient,
        first_constraint_max_contribution_before=first_constraint_max_contribution_before,
        first_constraint_max_contribution_after=first_constraint_max_contribution_after,
        first_constraint_max_contribution_idx=first_constraint_max_contribution_idx,
    )
end


function solve_weights_karl_expanded_cm(A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, light_target::Vector{Float64}, light_sigma::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}; Nspatial::Int, Nvbin::Int, alphat::Float64=DEFAULT_KARL_ALPHAT, light_rel_tol::Float64=DEFAULT_KARL_LIGHT_REL_TOL, light_sigma_tol::Float64=2.0, delta_chi2_iter_tol::Float64=DEFAULT_KARL_DELTA_CHI2_ITER_TOL, wphase=nothing, maxiter::Int=DEFAULT_KARL_MAXITER, seed::UInt=UInt(0), entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, apfac::Float64=DEFAULT_KARL_APFAC, return_diag::Bool=false, rcond_every::Int=250)
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

    # Active projmass.f initialization.
    w = karl_initial_weights_from_wphase(wp; paired=true, rotfrac=0.75, floor=entropy_floor)

    enforce_normalization = !_expanded_light_implies_normalization(A_light, light_target)
    Cm = build_expanded_Cm_with_losvd_slack(A_light, A_losvd; enforce_normalization=enforce_normalization)
    target_base = build_expanded_target(light_target, losvd_target; enforce_normalization=enforce_normalization)

    # projmass.f initializes the auxiliary LOSVD variables with raw data-model residual.
    w_all = build_expanded_weights_initial(w, A_losvd, losvd_target)

    initial_losvd = karl_losvd_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, Nspatial, Nvbin)
    previous_chi2_losvd = initial_losvd.chi_total
    delta_chi2_iteration = Inf
    max_light_relative_residual_value = max_light_relative_residual(A_light, w, light_target)
    max_light_sigma_residual_value = light_sigma_residual(w)
    light_constraint_ok = max_light_sigma_residual_value <= light_sigma_tol

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

        w_all_new, step_ok, sdiag = karl_spear_step_light_losvd_all(w_all, Norbit, Cm, target_base, A_losvd, losvd_target, losvd_sigma, wp; Nlight=Nlight, Nspatial=Nspatial, Nvbin=Nvbin, alphat=alphat, apfac=apfac, entropy_floor=entropy_floor, compute_rcond=compute_rcond, print_consistency=(iter == 1))
        last_diag = sdiag
        iter == 1 && (raw_spear_diag = sdiag.raw_spear_diag)
        isfinite(sdiag.rcond_est) && (last_rcond_est = sdiag.rcond_est)

        if !step_ok
            failure_reason = sdiag.failure_reason
            ok = false
            break
        end

        w_all .= w_all_new
        w_current = Vector{Float64}(@view w_all[1:Norbit])
        slack_current = Vector{Float64}(@view w_all[(Norbit + 1):end])
        losvd_state = karl_losvd_fracnew_state(A_losvd, w_current, losvd_target, losvd_sigma, Nspatial, Nvbin)

        chi2_losvd_current = losvd_state.chi_total
        delta_chi2_iteration = abs(chi2_losvd_current - previous_chi2_losvd)
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

        println("[WEIGHT PROGRESS] iteration=", iter, " active_passes=", sdiag.active_passes, " N_active_bound=", sdiag.n_active_bound, " stepfac=", sdiag.stepfac, " fracnew_min=", minimum(losvd_state.fracnew), " fracnew_max=", maximum(losvd_state.fracnew), " chi_losvd=", chi2_losvd_current, " max_light_sigma_residual=", light_sigma_progress[worst_light_bin], " worst_light_bin=", worst_light_bin, " delta_chi2=", delta_chi2_iteration)

        if light_constraint_ok && slack_consistent && normalized && delta_chi2_iteration <= delta_chi2_iter_tol
            converged = true
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
    delta_chi2_ok = isfinite(delta_chi2_iteration) && delta_chi2_iteration <= delta_chi2_iter_tol
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
            entropy=ent,
            chi=chi2_losvd,
            chi_losvd=chi2_losvd,
            profit=ent - losvd_penalty,
            alphat=alphat,
            losvd_penalty=losvd_penalty,
            chi_slack=losvd_penalty,
            slack_to_losvd=chi2_losvd > 0.0 ? losvd_penalty / chi2_losvd : NaN,
            fracnew=final_losvd === nothing ? Float64[] : final_losvd.fracnew,
            fracnew_min=final_losvd === nothing ? NaN : minimum(final_losvd.fracnew),
            fracnew_max=final_losvd === nothing ? NaN : maximum(final_losvd.fracnew),
            delta_chi2_iteration=delta_chi2_iteration,
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
            n_active_bound=last_diag === nothing ? 0 : last_diag.n_active_bound,
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


function karl_spear_step_light_losvd_all(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target_base::Vector{Float64}, A_losvd::Matrix{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}, wphase_orbit::Vector{Float64}; Nlight::Int, Nspatial::Int, Nvbin::Int, alphat::Float64=DEFAULT_KARL_ALPHAT, apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, compute_rcond::Bool=true, print_consistency::Bool=false)
    Narr, nvar = size(Cm)
    Nlosvd, Norbit2 = size(A_losvd)

    Norbit2 == Norbit || error("A_losvd orbit count does not match Norbit")
    length(w_all) == nvar || error("w_all length does not match expanded Cm")
    length(target_base) == Narr || error("target length does not match expanded Cm rows")
    nvar - Norbit == Nlosvd || error("expanded slack count does not match LOSVD row count")
    length(wphase_orbit) == Norbit || error("wphase length does not match Norbit")
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match Nlosvd")

    w_orbit = Vector{Float64}(@view w_all[1:Norbit])
    losvd_state = karl_losvd_fracnew_state(A_losvd, w_orbit, losvd_target, losvd_sigma, Nspatial, Nvbin)
    entropy, chi_slack, dS, ddS = build_expanded_entropy_derivatives(w_all, Norbit, wphase_orbit, losvd_state.residual, losvd_state.effective_sigma; alphat=alphat, entropy_floor=entropy_floor)

    target = copy(target_base)
    target[(Nlight + 1):(Nlight + Nlosvd)] .= losvd_state.effective_target

    raw_diag = print_consistency ? karl_raw_spear_consistency_diagnostic(w_all, Norbit, Cm, target, dS, ddS; apfac=apfac, entropy_floor=entropy_floor) : nothing

    update_diag = karl_spear_update_expanded(w_all, Norbit, Cm, target, dS, ddS; apfac=apfac, entropy_floor=entropy_floor, compute_rcond=compute_rcond)
    wnew = Vector{Float64}(update_diag.w)

    finite_state = all(isfinite, wnew)
    orbit_floor_ok = finite_state && _project_expanded_weights!(wnew, Norbit; floor=entropy_floor)
    step_ok = finite_state && orbit_floor_ok && update_diag.active_set_stabilized
    failure_reason = !finite_state ? :nonfinite_updated_state : !orbit_floor_ok ? :orbit_floor_violation : !update_diag.active_set_stabilized ? :active_set_not_stabilized : :none

    diag = merge(update_diag, (entropy=entropy, chi_slack=chi_slack, fracnew=losvd_state.fracnew, effective_losvd_target=losvd_state.effective_target, effective_losvd_sigma=losvd_state.effective_sigma, losvd_residual=losvd_state.residual, chi_by_spatial=losvd_state.chi_by_spatial, chi_losvd_karl=losvd_state.chi_total, raw_spear_diag=raw_diag, slack=Vector{Float64}(wnew[(Norbit + 1):end]), w_all=wnew, failure_reason=failure_reason))
    return wnew, step_ok, diag
end


# ====================================================================================================================================================================================
# ========================================================================================================================

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
