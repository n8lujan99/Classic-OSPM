# ============================================================
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
# ============================================================

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
        iseven(length(wphase)) ||
            error("paired wphase diagnostics require an even vector length")
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

# ============================================================
# §3c  SHARED SPEAR LINEAR-ALGEBRA PRIMITIVES
# ============================================================
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
    if size(Am, 1) == 0
        return Float64[]
    end
    # Solve the original SPEAR system without perturbing Am.
    # Use an SVD pseudoinverse so numerically unsupported singular directions are discarded instead of being altered by diagonal ridge regularization.
    F = svd(Am)
    smax = maximum(F.S)
    tol = max(size(Am)...) * eps(Float64) * smax
    coeff = transpose(F.U) * rhs
    @inbounds for i in eachindex(coeff)
        coeff[i] = F.S[i] > tol ? coeff[i] / F.S[i] : 0.0
    end
    return F.V * coeff
end

# ============================================================
# §3d  KARL EXPANDED CM WITH LOSVD SLACK VARIABLES
# ============================================================
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

# ============================================================
# CHI
# ============================================================

function build_expanded_entropy_derivatives(w_all::Vector{Float64}, Norbit::Int, wphase_orbit::Vector{Float64}, losvd_sigma::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Nvar = length(w_all)
    Nlosvd = Nvar - Norbit
    Nlosvd >= 0 || error("Norbit cannot exceed total variable count")
    length(wphase_orbit) == Norbit || error("wphase length must match Norbit")
    length(losvd_sigma) == Nlosvd || error("losvd_sigma length must match LOSVD slack count")
    dS = zeros(Float64, Nvar)
    ddS = zeros(Float64, Nvar)
    entropy = 0.0
    chi_slack = 0.0
    @inbounds for j in 1:Norbit
        wj = _safe_positive(w_all[j]; floor=entropy_floor)
        pj = wphase_orbit[j]
        isfinite(pj) && pj > 0.0 ||
            error("invalid Karl inverse phase volume at orbit column $j")
        log_measure = log(wj) + log(pj)
        entropy -= wj * log_measure
        dS[j] = -1.0 - log_measure
        ddS[j] = -1.0 / wj
    end
    @inbounds for k in 1:Nlosvd
        idx = Norbit + k
        sig = max(abs(losvd_sigma[k]), 1e-12)
        den = sig * sig
        y = w_all[idx]
        entropy -= y * y * alphat / den
        chi_slack += y * y * alphat / den
        dS[idx] = -2.0 * y * alphat / den
        ddS[idx] = -2.0 * alphat / den
        if ddS[idx] == 0.0
            ddS[idx] = -DEFAULT_KARL_ENTROPY_FLOOR
        end
    end
    return entropy, chi_slack, dS, ddS
end

function _project_expanded_weights!(w_all::Vector{Float64}, Norbit::Int; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    0 < Norbit <= length(w_all) || error("Norbit must be between 1 and length(w_all)")
    all(isfinite, w_all) || return false
    @inbounds for j in 1:Norbit
        w_all[j] >= floor || return false
    end
    return true
end

function karl_spear_update_expanded(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64};
    apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, compute_rcond::Bool=true)

    Narr, nvar = size(Cm)
    length(w_all) == nvar || error("w_all length must match Cm columns")
    0 < Norbit <= nvar || error("Norbit must be between 1 and length(w_all)")
    length(target) == Narr || error("target length must match Cm rows")
    length(dS) == nvar || error("dS length must match Cm columns")
    length(ddS) == nvar || error("ddS length must match Cm columns")
    isfinite(apfac) && apfac > 0.0 || error("apfac must be finite and positive")

    model = Cm * w_all
    base_delY = target .- model
    active_bound = falses(Norbit)
    fixed_dw = zeros(Float64, nvar)
    dw = zeros(Float64, nvar)
    trial = similar(w_all)
    wnew = similar(w_all)
    lambda = zeros(Float64, Narr)
    Am = zeros(Float64, Narr, Narr)

    active_passes = 0
    active_set_stabilized = false
    n_initial_violators = 0
    initial_n_negative_dw = 0
    max_batch_activated = 0
    last_batch_activated = 0
    activated_events_total = 0
    boundary_events_total = 0

    initial_min_trial_weight = NaN
    initial_min_trial_idx = 0
    limiting_idx = 0
    limiting_weight = NaN
    limiting_dw = NaN
    limiting_candidate = apfac

    last_boundary_idx = 0
    last_boundary_candidate = NaN
    last_boundary_batch_size = 0

    spear_rhs_l2 = NaN
    spear_system_residual_l2 = NaN
    spear_system_relative_residual = NaN

    n_release_candidates = 0
    max_release_candidates_seen = 0
    released_total = 0
    last_released_idx = 0
    last_released_multiplier = NaN
    most_negative_bound_multiplier = 0.0
    min_bound_multiplier = NaN
    max_bound_multiplier = NaN
    bound_kkt_tolerance = NaN

    # A repeated active set means the exact same constrained SPEAR problem
    # has returned. The second visit forces strict one-orbit tie-breaking.
    # A third visit is a genuine cycle and stops immediately.
    active_set_visits = Dict{UInt, Int}()

    max_active_passes = 2 * Norbit + 1

    for pass in 1:max_active_passes
        active_passes = pass
        active_signature = UInt(0)

        @inbounds for j in 1:Norbit
            if active_bound[j]
                active_signature = hash(j, active_signature)
            end
        end

        active_signature = hash(count(active_bound), active_signature)

        active_visit = get(active_set_visits, active_signature, 0) + 1
        active_set_visits[active_signature] = active_visit

        if active_visit >= 3
            error("expanded SPEAR active set entered an exact cycle after $pass passes; active_bound=$(count(active_bound))")
        end

        strict_cycle_mode = active_visit == 2

        if strict_cycle_mode && compute_rcond
            println("[ACTIVE CYCLE DIAG]",
                " pass=", pass,
                " active_bound=", count(active_bound),
                " visit=", active_visit,
                " mode=strict_single_orbit")
        end

        active_idx = findall(active_bound)

        free_idx = Int[]
        sizehint!(free_idx, nvar - length(active_idx))

        @inbounds for j in 1:nvar
            if j > Norbit || !active_bound[j]
                push!(free_idx, j)
            end
        end

        isempty(free_idx) && error("expanded SPEAR active set removed every variable")

        Cm_free = Cm[:, free_idx]
        dS_free = dS[free_idx]
        ddS_free = ddS[free_idx]

        reduced_delY = copy(base_delY)

        if !isempty(active_idx)
            reduced_delY .-= Cm[:, active_idx] * fixed_dw[active_idx]
        end

        Am = karl_spear_build_Am(Cm_free, ddS_free; floor=entropy_floor)
        karl_spear_rhs!(reduced_delY, Cm_free, dS_free, ddS_free; floor=entropy_floor)

        lambda = _solve_spear_system(Am, reduced_delY)

        spear_rhs_l2 = norm(reduced_delY)
        spear_system_residual_l2 = norm(Am * lambda - reduced_delY)
        spear_system_relative_residual = spear_system_residual_l2 / max(spear_rhs_l2, eps(Float64))

        dw_free = karl_spear_delta_w(Cm_free, lambda, dS_free, ddS_free; floor=entropy_floor)

        fill!(dw, 0.0)

        @inbounds for j in active_idx
            dw[j] = fixed_dw[j]
        end

        @inbounds for k in eachindex(free_idx)
            dw[free_idx[k]] = dw_free[k]
        end

        @inbounds for j in eachindex(w_all)
            trial[j] = w_all[j] + apfac * dw[j]
        end

        if pass == 1
            initial_min_trial_weight, initial_min_trial_idx = findmin(@view trial[1:Norbit])
            initial_n_negative_dw = count(direction -> direction < 0.0, @view dw[1:Norbit])
            n_initial_violators = count(weight -> weight < entropy_floor, @view trial[1:Norbit])
        end

        # ------------------------------------------------------------
        # FRACTION-TO-BOUNDARY PRIMAL TEST
        #
        # Do NOT activate every orbit that the full proposed Newton step
        # would carry below the floor.
        #
        # Instead find the first free orbit encountered while travelling
        # along the current SPEAR direction:
        #
        #   alpha_j = (w_j - floor) / (-dw_j)
        #
        # Only the earliest blocker, or a numerically tied set of earliest
        # blockers, is allowed to become active. Then the entire coupled
        # SPEAR direction is rebuilt.
        # ------------------------------------------------------------

        boundary_candidate = Inf
        boundary_idx = 0

        @inbounds for j in 1:Norbit
            if !active_bound[j] && dw[j] < 0.0
                candidate = (w_all[j] - entropy_floor) / (-dw[j])

                if candidate < boundary_candidate
                    boundary_candidate = candidate
                    boundary_idx = j
                end
            end
        end

        if pass == 1 && boundary_idx != 0
            limiting_idx = boundary_idx
            limiting_weight = w_all[boundary_idx]
            limiting_dw = dw[boundary_idx]
            limiting_candidate = boundary_candidate
        end

        # candidate is measured in the same step-factor units as apfac.
        # If the first boundary lies before the requested step, the current
        # full step is not primal feasible.
        if boundary_idx != 0 && boundary_candidate < apfac
            boundary_tolerance = max(1.0e-12, 1.0e-8 * max(1.0, abs(boundary_candidate)))
            blocking_idx = Int[]

            if strict_cycle_mode
                push!(blocking_idx, boundary_idx)
            else
                @inbounds for j in 1:Norbit
                    if !active_bound[j] && dw[j] < 0.0
                        candidate = (w_all[j] - entropy_floor) / (-dw[j])

                        if candidate <= boundary_candidate + boundary_tolerance
                            push!(blocking_idx, j)
                        end
                    end
                end
            end

            isempty(blocking_idx) && push!(blocking_idx, boundary_idx)

            newly_active = length(blocking_idx)

            @inbounds for j in blocking_idx
                active_bound[j] = true
                fixed_dw[j] = (entropy_floor - w_all[j]) / apfac
            end

            activated_events_total += newly_active
            boundary_events_total += 1
            max_batch_activated = max(max_batch_activated, newly_active)

            last_batch_activated = newly_active
            last_boundary_idx = boundary_idx
            last_boundary_candidate = boundary_candidate
            last_boundary_batch_size = newly_active

            if compute_rcond && (boundary_events_total == 1 || boundary_events_total % 250 == 0 || strict_cycle_mode)
                println("[ACTIVE BOUNDARY STEP]",
                    " pass=", pass,
                    " active_before=", length(active_idx),
                    " boundary_orbit=", boundary_idx,
                    " boundary_candidate=", boundary_candidate,
                    " requested_step=", apfac,
                    " tied_blockers=", newly_active,
                    " strict_cycle_mode=", strict_cycle_mode,
                    " active_after=", count(active_bound),
                    " boundary_events=", boundary_events_total,
                    " activated_events_total=", activated_events_total)
            end

            # The working set changed. Rebuild the entire coupled SPEAR
            # solution before judging any other orbit.
            continue
        end

        # ------------------------------------------------------------
        # KKT RELEASE TEST
        #
        # We now know no currently free orbit crosses the floor.
        # Test whether any orbit already on the floor was incorrectly
        # constrained there.
        #
        # Release only the single most-negative multiplier, then rebuild
        # the whole coupled system. There is deliberately no mass release.
        # ------------------------------------------------------------

        active_idx = findall(active_bound)

        n_release_candidates = 0
        most_negative_bound_multiplier = 0.0
        min_bound_multiplier = NaN
        max_bound_multiplier = NaN
        bound_kkt_tolerance = NaN

        if !isempty(active_idx)
            bound_multipliers = Vector{Float64}(undef, length(active_idx))

            @inbounds for k in eachindex(active_idx)
                j = active_idx[k]
                ddSj = _spear_safe_ddS(ddS[j]; floor=entropy_floor)

                bound_multipliers[k] =
                    dot(@view(Cm[:, j]), lambda) -
                    dS[j] -
                    ddSj * fixed_dw[j]
            end

            all(isfinite, bound_multipliers) || error("expanded SPEAR active-set KKT multipliers became nonfinite")

            min_bound_multiplier, max_bound_multiplier = extrema(bound_multipliers)
            multiplier_scale = max(1.0, maximum(abs, bound_multipliers))
            bound_kkt_tolerance = 1.0e-10 * multiplier_scale

            n_release_candidates = count(mu -> mu < -bound_kkt_tolerance, bound_multipliers)
            max_release_candidates_seen = max(max_release_candidates_seen, n_release_candidates)

            if n_release_candidates > 0
                local_release_idx = argmin(bound_multipliers)
                release_idx = active_idx[local_release_idx]
                release_multiplier = bound_multipliers[local_release_idx]

                most_negative_bound_multiplier = release_multiplier
                released_total += 1
                last_released_idx = release_idx
                last_released_multiplier = release_multiplier

                if compute_rcond && (released_total == 1 || released_total % 250 == 0 || strict_cycle_mode)
                    println("[ACTIVE RELEASE STEP]",
                        " pass=", pass,
                        " active_bound=", length(active_idx),
                        " release_candidates=", n_release_candidates,
                        " releasing_orbit=", release_idx,
                        " multiplier=", release_multiplier,
                        " kkt_tolerance=", bound_kkt_tolerance,
                        " released_total=", released_total,
                        " strict_cycle_mode=", strict_cycle_mode)
                end

                active_bound[release_idx] = false
                fixed_dw[release_idx] = 0.0

                # Rebuild the whole coupled direction. The newly released
                # orbit is not automatically allowed to drag thousands of
                # other free orbits through the floor; the fraction-to-
                # boundary test above decides the next genuine blocker.
                continue
            end
        end

        # ------------------------------------------------------------
        # STABLE WORKING SET
        #
        # 1. No free orbit crosses the floor along the current step.
        # 2. No active orbit has a negative KKT multiplier.
        #
        # Only here do we accept the active set.
        # ------------------------------------------------------------

        if compute_rcond
            println("[ACTIVE RELEASE DIAG]",
                " active_bound=", length(active_idx),
                " release_candidates=", n_release_candidates,
                " released_total=", released_total,
                " max_release_candidates_seen=", max_release_candidates_seen,
                " boundary_events=", boundary_events_total,
                " activated_events_total=", activated_events_total,
                " last_boundary_idx=", last_boundary_idx,
                " last_boundary_candidate=", last_boundary_candidate,
                " last_boundary_batch_size=", last_boundary_batch_size,
                " min_bound_multiplier=", min_bound_multiplier,
                " max_bound_multiplier=", max_bound_multiplier,
                " kkt_tolerance=", bound_kkt_tolerance,
                " last_released_idx=", last_released_idx,
                " last_released_multiplier=", last_released_multiplier)
        end

        copyto!(wnew, trial)

        @inbounds for j in 1:Norbit
            active_bound[j] && (wnew[j] = entropy_floor)
        end

        active_set_stabilized = true
        break
    end

    active_set_stabilized || error("expanded SPEAR fraction-to-boundary active set did not stabilize after $max_active_passes passes")

    orbit_before = @view w_all[1:Norbit]
    orbit_after = @view wnew[1:Norbit]
    orbit_dw = @view dw[1:Norbit]

    min_orbit_weight = minimum(orbit_before)
    min_updated_orbit_weight, min_updated_orbit_idx = findmin(orbit_after)
    n_at_floor_before = count(weight -> weight <= entropy_floor, orbit_before)
    n_at_floor_after = count(weight -> weight <= entropy_floor, orbit_after)
    n_below_floor = count(weight -> weight < entropy_floor, orbit_after)
    n_nonpositive_orbits = count(weight -> weight <= 0.0, orbit_after)
    n_negative_dw = count(direction -> direction < 0.0, orbit_dw)
    n_active_bound = count(active_bound)
    n_free_orbits_final = Norbit - n_active_bound

    weight_moved_to_floor = 0.0

    @inbounds for j in 1:Norbit
        if active_bound[j]
            weight_moved_to_floor += w_all[j] - entropy_floor
        end
    end

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

    linearized_constraint_error_l2 = norm(apfac .* (Cm * dw) .- base_delY)
    post_step_constraint_l2 = norm(target .- Cm * wnew)
    rcond_est = compute_rcond ? 1.0 / max(cond(Am), 1.0) : NaN
    max_abs_dw = maximum(abs, dw)

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
        n_release_candidates=n_release_candidates,
        max_release_candidates_seen=max_release_candidates_seen,
        released_total=released_total,
        last_released_idx=last_released_idx,
        last_released_multiplier=last_released_multiplier,
        most_negative_bound_multiplier=most_negative_bound_multiplier,
        min_bound_multiplier=min_bound_multiplier,
        max_bound_multiplier=max_bound_multiplier,
        bound_kkt_tolerance=bound_kkt_tolerance,
        stepfac=apfac,
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
        n_negative_dw=n_negative_dw,
        n_active_bound=n_active_bound,
        n_activated_total=n_active_bound,
        activated_events_total=activated_events_total,
        n_free_orbits_final=n_free_orbits_final,
        active_passes=active_passes,
        max_batch_activated=max_batch_activated,
        last_batch_activated=last_batch_activated,
        active_set_stabilized=active_set_stabilized,
        boundary_events_total=boundary_events_total,
        last_boundary_idx=last_boundary_idx,
        last_boundary_candidate=last_boundary_candidate,
        last_boundary_batch_size=last_boundary_batch_size,
        weight_moved_to_floor=weight_moved_to_floor,
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

function solve_weights_karl_expanded_cm(
    A_light::Matrix{Float64},
    A_losvd::Matrix{Float64},
    light_target::Vector{Float64},
    light_sigma::Vector{Float64},
    losvd_target::Vector{Float64},
    losvd_sigma::Vector{Float64};
    alphat::Float64=DEFAULT_KARL_ALPHAT,
    light_rel_tol::Float64=DEFAULT_KARL_LIGHT_REL_TOL,
    light_sigma_tol::Float64=2.0,
    delta_chi2_iter_tol::Float64=DEFAULT_KARL_DELTA_CHI2_ITER_TOL,
    wphase=nothing,
    maxiter::Int=DEFAULT_KARL_MAXITER,
    seed::UInt=UInt(0),
    entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR,
    apfac::Float64=DEFAULT_KARL_APFAC,
    return_diag::Bool=false,
    rcond_every::Int=250)

    Nlight, Norbit = size(A_light)
    Nlosvd, Norbit2 = size(A_losvd)
    fail_w = zeros(Float64, Norbit)

    Norbit == Norbit2 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    Nlight > 0 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(light_target) == Nlight || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(light_sigma) == Nlight || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(losvd_target) == Nlosvd || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(losvd_sigma) == Nlosvd || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    Norbit > 0 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)

    light_rel_tol > 0.0 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    light_sigma_tol > 0.0 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    delta_chi2_iter_tol >= 0.0 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    rcond_every > 0 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)

    all(isfinite, A_light) || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    all(isfinite, A_losvd) || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    all(isfinite, light_target) || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    all(isfinite, light_sigma) || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    all(>(0.0), light_sigma) || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    all(isfinite, losvd_target) || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    all(isfinite, losvd_sigma) || return return_diag ? (fail_w, false, nothing) : (fail_w, false)

    iseven(Norbit) ||
        error("Karl phase-volume weight solving requires paired prograde/retrograde orbit columns; got Norbit=$Norbit")

    vsig = max.(abs.(Float64.(losvd_sigma)), 1e-12)
    light_sigma_use = max.(abs.(Float64.(light_sigma)), 1e-12)

    function light_sigma_residual(w_use::Vector{Float64})
        light_model = A_light * w_use
        max_residual = 0.0

        @inbounds for ib in 1:Nlight
            residual =
                abs(light_model[ib] - light_target[ib]) /
                light_sigma_use[ib]

            max_residual = max(max_residual, residual)
        end

        return max_residual
    end

    wp = _prepare_wphase(
        wphase,
        Norbit;
        require_paired=true,
        pair_rtol=1.0e-12,
    )

    wphase_diag = karl_wphase_diagnostics(wp; paired=true)

    w = karl_initial_weights_from_wphase(
        wp;
        paired=true,
        rotfrac=0.75,
        floor=entropy_floor,
    )

    enforce_normalization =
        !_expanded_light_implies_normalization(A_light, light_target)

    Cm = build_expanded_Cm_with_losvd_slack(
        A_light,
        A_losvd;
        enforce_normalization=enforce_normalization,
    )

    target = build_expanded_target(
        light_target,
        losvd_target;
        enforce_normalization=enforce_normalization,
    )

    w_all = build_expanded_weights_initial(
        w,
        A_losvd,
        losvd_target,
    )

    previous_chi2_losvd =
        chi2_block(A_losvd, w, losvd_target, vsig)

    delta_chi2_iteration = Inf

    max_light_relative_residual_value =
        max_light_relative_residual(A_light, w, light_target)

    max_light_sigma_residual_value =
        light_sigma_residual(w)

    light_constraint_ok =
        max_light_sigma_residual_value <= light_sigma_tol

    last_diag = nothing
    last_rcond_est = NaN
    failure_reason = :none
    iterations = 0
    converged = false
    ok = true

    w_current = similar(w)
    slack_current = Vector{Float64}(undef, Nlosvd)
    losvd_model_current = Vector{Float64}(undef, Nlosvd)
    losvd_residual_current = Vector{Float64}(undef, Nlosvd)

    for iter in 1:maxiter
        iterations = iter

        compute_rcond =
            iter == 1 ||
            iter == maxiter ||
            iter % rcond_every == 0

        w_all_new, step_ok, sdiag =
            karl_spear_step_light_losvd_all(
                w_all,
                Norbit,
                Cm,
                target,
                vsig,
                wp;
                alphat=alphat,
                apfac=apfac,
                entropy_floor=entropy_floor,
                compute_rcond=compute_rcond,
            )

        last_diag = sdiag

        if isfinite(sdiag.rcond_est)
            last_rcond_est = sdiag.rcond_est
        end

        if !step_ok
            rcond_report =
                isfinite(sdiag.rcond_est) ?
                sdiag.rcond_est :
                last_rcond_est

            println(
                "[EXPANDED CM ROOT]",
                " iteration=", iter,
                " reason=", sdiag.failure_reason,
                " stepfac=", sdiag.stepfac,
                " limiting_orbit=", sdiag.limiting_idx,
                " limiting_weight=", sdiag.limiting_weight,
                " entropy_floor=", entropy_floor,
                " limiting_dw=", sdiag.limiting_dw,
                " limiting_candidate=", sdiag.limiting_candidate,
                " min_orbit_weight=", sdiag.min_orbit_weight,
                " N_at_floor=", sdiag.n_at_floor,
                " N_negative_dw=", sdiag.n_negative_dw,
                " rcond_est=", rcond_report,
                " max_abs_dw=", sdiag.max_abs_dw,
            )

            failure_reason = sdiag.failure_reason
            ok = false
            break
        end

        w_all .= w_all_new

        copyto!(w_current, 1, w_all, 1, Norbit)
        copyto!(slack_current, 1, w_all, Norbit + 1, Nlosvd)

        mul!(losvd_model_current, A_losvd, w_current)

        chi2_losvd_current = 0.0
        slack_residual_sq = 0.0
        slack_norm_sq = 0.0
        losvd_residual_norm_sq = 0.0

        @inbounds for k in 1:Nlosvd
            residual =
                losvd_target[k] -
                losvd_model_current[k]

            losvd_residual_current[k] = residual

            scaled_residual =
                residual / vsig[k]

            chi2_losvd_current +=
                scaled_residual * scaled_residual

            slack_difference =
                slack_current[k] -
                residual

            slack_residual_sq +=
                slack_difference * slack_difference

            slack_norm_sq +=
                slack_current[k] * slack_current[k]

            losvd_residual_norm_sq +=
                residual * residual
        end

        delta_chi2_iteration =
            abs(
                chi2_losvd_current -
                previous_chi2_losvd,
            )

        max_light_relative_residual_value =
            max_light_relative_residual(
                A_light,
                w_current,
                light_target,
            )

        max_light_sigma_residual_value =
            light_sigma_residual(w_current)

        light_constraint_ok =
            max_light_sigma_residual_value <=
            light_sigma_tol

        slack_residual_l2_current =
            sqrt(slack_residual_sq)

        slack_scale_current =
            max(
                1.0,
                sqrt(slack_norm_sq),
                sqrt(losvd_residual_norm_sq),
            )

        slack_consistent_current =
            slack_residual_l2_current <=
            light_rel_tol * slack_scale_current

        normalized_current =
            abs(sum(w_current) - 1.0) <=
            light_rel_tol

        light_model_progress =
            A_light * w_current

        light_relative_progress =
            abs.(
                light_model_progress .-
                light_target
            ) ./
            max.(
                abs.(light_target),
                eps(Float64),
            )

        light_sigma_progress =
            abs.(
                light_model_progress .-
                light_target
            ) ./
            light_sigma_use

        worst_light_bin =
            argmax(light_sigma_progress)

        println(
            "[WEIGHT PROGRESS]",
            " iteration=", iter,
            " active_passes=", sdiag.active_passes,
            " N_active_bound=", sdiag.n_active_bound,
            " N_at_floor=", sdiag.n_at_floor,
            " stepfac=", sdiag.stepfac,
            " max_light_relative_residual=", maximum(light_relative_progress),
            " max_light_sigma_residual=", light_sigma_progress[worst_light_bin],
            " worst_light_bin=", worst_light_bin,
            " light_target=", light_target[worst_light_bin],
            " light_model=", light_model_progress[worst_light_bin],
            " light_sigma=", light_sigma_use[worst_light_bin],
            " delta_chi2=", delta_chi2_iteration,
        )

        if light_constraint_ok &&
           slack_consistent_current &&
           normalized_current &&
           delta_chi2_iteration <= delta_chi2_iter_tol

            converged = true
            break
        end

        previous_chi2_losvd =
            chi2_losvd_current
    end

    w =
        Vector{Float64}(
            w_all[1:Norbit],
        )

    slack =
        Vector{Float64}(
            w_all[(Norbit + 1):end],
        )

    finite_state =
        all(isfinite, w) &&
        all(isfinite, slack)

    if !finite_state
        failure_reason =
            :nonfinite_final_state

        ok = false
    end

    chi2_losvd =
        finite_state ?
        chi2_block(
            A_losvd,
            w,
            losvd_target,
            vsig,
        ) :
        Inf

    losvd_residual =
        finite_state ?
        losvd_target .-
        A_losvd * w :
        fill(Inf, Nlosvd)

    slack_residual_l2 =
        finite_state ?
        norm(slack .- losvd_residual) :
        Inf

    slack_scale =
        finite_state ?
        max(
            1.0,
            norm(slack),
            norm(losvd_residual),
        ) :
        Inf

    slack_consistent =
        finite_state &&
        slack_residual_l2 <=
        light_rel_tol * slack_scale

    normalization_error =
        finite_state ?
        abs(sum(w) - 1.0) :
        Inf

    normalized =
        normalization_error <=
        light_rel_tol

    light_residual_l2 =
        finite_state ?
        norm(
            light_target .-
            A_light * w,
        ) :
        Inf

    max_light_relative_residual_value =
        finite_state ?
        max_light_relative_residual(A_light, w, light_target) :
        Inf

    max_light_sigma_residual_value =
        finite_state ?
        light_sigma_residual(w) :
        Inf

    light_constraint_ok =
        max_light_sigma_residual_value <=
        light_sigma_tol

    constraint_l2 =
        finite_state ?
        norm(target .- Cm * w_all) :
        Inf

    constraint_ok =
        light_constraint_ok &&
        slack_consistent &&
        normalized

    delta_chi2_ok =
        isfinite(delta_chi2_iteration) &&
        delta_chi2_iteration <=
        delta_chi2_iter_tol

    solver_converged =
        ok &&
        converged &&
        constraint_ok &&
        delta_chi2_ok

    if ok && !light_constraint_ok
        failure_reason =
            :light_constraint_failed

    elseif ok && !slack_consistent
        failure_reason =
            :losvd_slack_inconsistent

    elseif ok && !normalized
        failure_reason =
            :normalization_failed

    elseif ok && !delta_chi2_ok
        failure_reason =
            :delta_chi2_not_converged

    elseif ok && !solver_converged
        failure_reason =
            :karl_not_converged
    end

    if return_diag
        chi2_slack =
            finite_state ?
            sum((slack ./ vsig) .^ 2) :
            Inf

        losvd_penalty =
            alphat * chi2_slack

        slack_to_losvd =
            chi2_losvd > 0.0 ?
            chi2_slack / chi2_losvd :
            NaN

        ent =
            finite_state ?
            karl_entropy_value(w, wp; entropy_floor=entropy_floor) :
            -Inf

        diag = (
            entropy=ent,
            chi=chi2_losvd,
            chi_losvd=chi2_losvd,
            profit=ent - losvd_penalty,
            alphat=alphat,
            losvd_penalty=losvd_penalty,
            chi_slack=chi2_slack,
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
            slack_to_losvd=slack_to_losvd,
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
        )

        return w, solver_converged, diag
    end

    return w, solver_converged
end
# ============================================================
# ============================================================


function karl_spear_step_light_losvd_all(w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, losvd_sigma::Vector{Float64},
    wphase_orbit::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, apfac::Float64=DEFAULT_KARL_APFAC,
    entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, compute_rcond::Bool=true)

    Narr, nvar = size(Cm)
    length(w_all) == nvar || error("w_all length does not match expanded Cm columns")
    length(target) == Narr || error("target length does not match expanded Cm rows")
    0 < Norbit < nvar || error("expanded Cm state must contain orbit weights followed by LOSVD slack variables")
    length(wphase_orbit) == Norbit || error("wphase_orbit length must match Norbit")

    Nslack = nvar - Norbit
    length(losvd_sigma) == Nslack || error("losvd_sigma length must match expanded slack count")
    isfinite(apfac) && apfac > 0.0 || error("apfac must be finite and positive")

    entropy, chi_slack, dS, ddS = build_expanded_entropy_derivatives(w_all, Norbit, wphase_orbit, losvd_sigma;
        alphat=alphat, entropy_floor=entropy_floor)

    model = Cm * w_all
    delY = target .- model

    # Local working set only. Nothing about individual orbit decisions is
    # persisted between outer weight iterations.
    active_bound = falses(Norbit)
    fixed_dw = zeros(Float64, nvar)
    trial = similar(w_all)


    dw = zeros(Float64, nvar)
    lambda = zeros(Float64, Narr)
    Am = zeros(Float64, Narr, Narr)

    active_passes = 0
    active_set_stabilized = false
    max_active_passes = Norbit + 1

    initial_min_trial_weight = NaN
    initial_min_trial_idx = 0
    initial_n_negative_dw = 0
    n_initial_violators = 0

    spear_rhs_l2 = NaN
    spear_system_residual_l2 = NaN
    spear_system_relative_residual = NaN
    rcond_est = NaN

    floor_state_tolerance = 100.0 * eps(Float64) * max(1.0, maximum(abs, @view w_all[1:Norbit]))

    # ------------------------------------------------------------
    # COUPLED BATCH ACTIVE-SET SOLVE
    #
    # Only weights ALREADY at the lower bound and pointing downward
    # are removed from the Newton system. They are removed as a set.
    # No orbit-by-orbit boundary walking occurs here.
    # ------------------------------------------------------------

    for pass in 1:max_active_passes
        active_passes = pass

        active_idx = findall(active_bound)

        free_idx = Int[]
        sizehint!(free_idx, nvar - length(active_idx))

        @inbounds for j in 1:nvar
            if j > Norbit || !active_bound[j]
                push!(free_idx, j)
            end
        end

        isempty(free_idx) && error("expanded SPEAR batch active set removed every variable")

        Cm_free = Cm[:, free_idx]
        dS_free = dS[free_idx]
        ddS_free = ddS[free_idx]

        rhs = copy(delY)

        if !isempty(active_idx)
            rhs .-= Cm[:, active_idx] * fixed_dw[active_idx]
        end

        Am = karl_spear_build_Am(Cm_free, ddS_free; floor=entropy_floor)
        karl_spear_rhs!(rhs, Cm_free, dS_free, ddS_free; floor=entropy_floor)

        lambda = _solve_spear_system(Am, rhs)
        dw_free = karl_spear_delta_w(Cm_free, lambda, dS_free, ddS_free; floor=entropy_floor)

        fill!(dw, 0.0)
        @inbounds for j in active_idx
            dw[j] = fixed_dw[j]
        end
        @inbounds for k in eachindex(free_idx)
            dw[free_idx[k]] = dw_free[k]
        end

        spear_rhs_l2 = norm(rhs)
        spear_system_residual_l2 = norm(Am * lambda - rhs)
        spear_system_relative_residual = spear_system_residual_l2 / max(spear_rhs_l2, eps(Float64))

        if compute_rcond
            rcond_est = 1.0 / max(cond(Am), 1.0)
        end

        @inbounds for j in eachindex(w_all)
            trial[j] = w_all[j] + apfac * dw[j]
        end

        if pass == 1
            initial_min_trial_weight, initial_min_trial_idx = findmin(@view trial[1:Norbit])
            initial_n_negative_dw = count(x -> x < 0.0, @view dw[1:Norbit])
            n_initial_violators = count(x -> x < entropy_floor, @view trial[1:Norbit])
        end

        newly_active = 0

        @inbounds for j in 1:Norbit
            if !active_bound[j] && trial[j] < entropy_floor
                active_bound[j] = true
                fixed_dw[j] = (entropy_floor - w_all[j]) / apfac
                newly_active += 1
            end
        end

        if newly_active > 0
            if compute_rcond
                println("[ACTIVE BATCH FLOOR]",
                    " pass=", pass,
                    " new_floor_blockers=", newly_active,
                    " active_bound=", count(active_bound))
            end

            continue
        end

        active_set_stabilized = true
        break

    end

    if !active_set_stabilized
        base_diag = (
            entropy=entropy,
            chi_slack=chi_slack,
            rcond_est=rcond_est,
            max_abs_dw=maximum(abs, dw),
            stepfac=0.0,
            limiting_idx=0,
            limiting_weight=NaN,
            limiting_dw=NaN,
            limiting_candidate=NaN,
            min_orbit_weight=minimum(@view w_all[1:Norbit]),
            n_at_floor=count(x -> x <= entropy_floor, @view w_all[1:Norbit]),
            n_negative_dw=count(x -> x < 0.0, @view dw[1:Norbit]),
            n_active_bound=count(active_bound),
            active_passes=active_passes,
            active_set_stabilized=false,
            n_projected_to_floor=0,
            projection_correction_l2=NaN,
            spear_rhs_l2=spear_rhs_l2,
            spear_system_residual_l2=spear_system_residual_l2,
            spear_system_relative_residual=spear_system_relative_residual,
        )

        return w_all, false, merge(base_diag,
            (slack=Float64[], w_all=w_all, failure_reason=:batch_active_set_not_stabilized))
    end

    stepfac = apfac
    wnew = w_all .+ stepfac .* dw

    floor_roundoff_tolerance = 100.0 * eps(Float64) * max(1.0, maximum(abs, @view wnew[1:Norbit]))

    @inbounds for j in 1:Norbit
        if wnew[j] < entropy_floor
            if entropy_floor - wnew[j] <= floor_roundoff_tolerance
                wnew[j] = entropy_floor
            else
                error("batch SPEAR solve produced a physical orbit weight below the entropy floor")
            end
        end
    end

    n_projected_to_floor = 0
    projection_correction_l2 = 0.0

    @inbounds for j in 1:Norbit
        if active_bound[j]
            wnew[j] = entropy_floor
        elseif wnew[j] < entropy_floor
            wnew[j] = entropy_floor
            n_projected_to_floor += 1
        end
    end

    # Measure how much the positivity projection altered the unconstrained
    # coupled Newton proposal.
    projection_correction_sq = 0.0

    @inbounds for j in 1:Norbit
        correction = wnew[j] - trial[j]
        projection_correction_sq += correction * correction
    end

    projection_correction_l2 = sqrt(projection_correction_sq)

    n_at_floor_before = count(x -> x <= entropy_floor, @view w_all[1:Norbit])
    n_at_floor_after = count(x -> x <= entropy_floor, @view wnew[1:Norbit])
    n_below_floor = count(x -> x < entropy_floor, @view wnew[1:Norbit])
    n_negative_dw = count(x -> x < 0.0, @view dw[1:Norbit])
    n_active_bound = count(active_bound)

    actual_step = wnew .- w_all

    max_abs_dw = maximum(abs, actual_step)
    post_step_constraint_l2 = norm(target .- Cm * wnew)

    # This now measures the ACTUAL projected update rather than the
    # unprojected Newton direction.
    linearized_constraint_error_l2 = norm(Cm * actual_step - delY)

    first_row = @view Cm[1, :]
    first_constraint_target = target[1]
    first_constraint_model_before = dot(first_row, w_all)
    first_constraint_model_after = dot(first_row, wnew)
    first_constraint_denominator = max(abs(first_constraint_target), eps(Float64))

    first_constraint_relative_before =
        abs(first_constraint_model_before - first_constraint_target) / first_constraint_denominator

    first_constraint_relative_after =
        abs(first_constraint_model_after - first_constraint_target) / first_constraint_denominator

    if compute_rcond
        println("[BATCH ACTIVE DIAG]",
            " active_passes=", active_passes,
            " active_bound=", n_active_bound,
            " floor_before=", n_at_floor_before,
            " floor_after=", n_at_floor_after,
            " projected_to_floor=", n_projected_to_floor,
            " stepfac=", stepfac,
            " initial_violators=", n_initial_violators,
            " projection_correction_l2=", projection_correction_l2,
            " first_relative_before=", first_constraint_relative_before,
            " first_relative_after=", first_constraint_relative_after,
            " spear_system_relative_residual=", spear_system_relative_residual,
            " post_step_constraint_l2=", post_step_constraint_l2)
    end

    base_diag = (
        entropy=entropy,
        chi_slack=chi_slack,
        rcond_est=rcond_est,
        max_abs_dw=max_abs_dw,
        stepfac=stepfac,
        # There is no individual limiting orbit anymore.
        limiting_idx=0,
        limiting_weight=NaN,
        limiting_dw=NaN,
        limiting_candidate=NaN,
        min_orbit_weight=minimum(@view w_all[1:Norbit]),
        min_updated_orbit_weight=minimum(@view wnew[1:Norbit]),
        initial_min_trial_weight=initial_min_trial_weight,
        initial_min_trial_idx=initial_min_trial_idx,
        n_initial_violators=n_initial_violators,
        initial_n_negative_dw=initial_n_negative_dw,
        n_at_floor=n_at_floor_after,
        n_at_floor_before=n_at_floor_before,
        n_at_floor_after=n_at_floor_after,
        n_below_floor=n_below_floor,
        n_negative_dw=n_negative_dw,
        n_active_bound=n_active_bound,
        active_passes=active_passes,
        active_set_stabilized=true,
        n_projected_to_floor=n_projected_to_floor,
        projection_correction_l2=projection_correction_l2,
        spear_rhs_l2=spear_rhs_l2,
        spear_system_residual_l2=spear_system_residual_l2,
        spear_system_relative_residual=spear_system_relative_residual,
        linearized_constraint_error_l2=linearized_constraint_error_l2,
        post_step_constraint_l2=post_step_constraint_l2,
        first_constraint_target=first_constraint_target,
        first_constraint_model_before=first_constraint_model_before,
        first_constraint_model_after=first_constraint_model_after,
        first_constraint_relative_before=first_constraint_relative_before,
        first_constraint_relative_after=first_constraint_relative_after,
    )

    return Vector{Float64}(wnew), true, merge(base_diag,
        (slack=Vector{Float64}(wnew[(Norbit + 1):end]),
         w_all=Vector{Float64}(wnew),
         failure_reason=:none))
end


# ============================================================
# ============================================================

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
