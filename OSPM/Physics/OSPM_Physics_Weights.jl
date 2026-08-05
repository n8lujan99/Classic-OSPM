# ============================================================
# OSPM_Physics_Weights.jl — Karl-style orbit weight machinery.
# Included by OSPM_Physics_Support.jl — do NOT load directly.
#
# Owns:
#   - wphase handling
#   - Karl entropy type 2
#   - paired prograde/retrograde initial weights
#   - expanded-Cm SPEAR solver
#   - LOSVD slack variables
#   - standard LOSVD χ² block scoring
#   - xmu / M-L helper functions
#
# Assumes OSPM_Physics_Support.jl has already defined:
#   f64, DEFAULT_KARL_ALPHAT, DEFAULT_KARL_MAXITER,
#   DEFAULT_KARL_ENTROPY_FLOOR, DEFAULT_KARL_APFAC
# ============================================================

function _prepare_wphase(wphase, n::Int; entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    if wphase === nothing
        return ones(Float64, n)
    end
    wp = Float64.(wphase)
    length(wp) == n || error("wphase length $(length(wp)) does not match Norbit=$n")
    @inbounds for i in eachindex(wp)
        (!isfinite(wp[i]) || wp[i] <= 0.0) && (wp[i] = entropy_floor)
    end
    return wp
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
    w = zeros(Float64, n)
    if paired
        iseven(n) || error("paired Karl initial weights require an even number of orbit columns")
        Norb = n ÷ 2
        @inbounds for i in 1:Norb
            ip = 2 * i - 1
            ir = 2 * i
            wp = _safe_positive(wphase[ip]; floor=floor)
            wr = _safe_positive(wphase[ir]; floor=floor)
            w[ip] = rotfrac / wp
            w[ir] = (1.0 - rotfrac) / wr
        end
    else
        @inbounds for i in 1:n
            wp = _safe_positive(wphase[i]; floor=floor)
            w[i] = 1.0 / wp
        end
    end
    s = sum(w)
    (!isfinite(s) || s <= 0.0) && error("Karl initial weights have non-positive sum")
    w ./= s
    return w
end

@inline function karl_entropy_value(w::Vector{Float64}, wphase::Vector{Float64}; entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    S = 0.0
    @inbounds for i in eachindex(w)
        wi = max(w[i], entropy_floor)
        pi = max(wphase[i], entropy_floor)
        S -= wi * log(wi * pi)
    end
    return S
end

# ============================================================
# §3c  SHARED SPEAR LINEAR-ALGEBRA PRIMITIVES
# ============================================================
# Used by the expanded-Cm solver below.

@inline function _spear_safe_ddS(x::Float64; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    if !isfinite(x)
        return -floor
    end
    if abs(x) < floor
        return x < 0.0 ? -floor : floor
    end
    return x
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
    # The SPEAR matrix can be nearly singular when the orbit library has repeated
    # columns.  Use a tiny diagonal floor only to make the Newton solve finite.
    scale = maximum(abs.(Am))
    ridge = max(scale, 1.0) * 1e-12
    Areg = copy(Am)
    @inbounds for i in 1:size(Areg, 1)
        Areg[i, i] += ridge
    end
    return Areg \ rhs
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

##CHI##
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
        pj = _safe_positive(wphase_orbit[j]; floor=entropy_floor)
        entropy -= wj * log(wj * pj)
        dS[j] = -1.0 - log(wj * pj)
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

function karl_spear_update_expanded(
    w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64};
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

    initial_min_trial_weight = NaN
    initial_min_trial_idx = 0

    limiting_idx = 0
    limiting_weight = NaN
    limiting_dw = NaN
    limiting_candidate = apfac

    for pass in 1:(Norbit + 1)
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

        # Active orbit directions are chosen so that
        # w + apfac*dw lands exactly on the entropy floor.
        reduced_delY = copy(base_delY)

        if !isempty(active_idx)
            reduced_delY .-= Cm[:, active_idx] * fixed_dw[active_idx]
        end

        Am = karl_spear_build_Am(Cm_free, ddS_free; floor=entropy_floor)
        karl_spear_rhs!(reduced_delY, Cm_free, dS_free, ddS_free; floor=entropy_floor)

        lambda = _solve_spear_system(Am, reduced_delY)
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

            if initial_min_trial_weight < entropy_floor
                limiting_idx = initial_min_trial_idx
                limiting_weight = w_all[limiting_idx]
                limiting_dw = dw[limiting_idx]
                limiting_candidate = limiting_dw < 0.0 ? (limiting_weight - entropy_floor) / (-limiting_dw) : apfac
            end
        end

        newly_active = 0

        @inbounds for j in 1:Norbit
            if !active_bound[j] && trial[j] < entropy_floor
                active_bound[j] = true
                fixed_dw[j] = (entropy_floor - w_all[j]) / apfac
                newly_active += 1
            end
        end

        pass == 1 && (n_initial_violators = newly_active)
        max_batch_activated = max(max_batch_activated, newly_active)
        last_batch_activated = newly_active

        if newly_active == 0
            copyto!(wnew, trial)

            @inbounds for j in 1:Norbit
                active_bound[j] && (wnew[j] = entropy_floor)
            end

            active_set_stabilized = true
            break
        end
    end

    active_set_stabilized || error("expanded SPEAR batch active set did not stabilize")

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
        active_bound[j] && (weight_moved_to_floor += w_all[j] - entropy_floor)
    end

    # The first expanded constraint is the innermost fitted light bin in
    # the current Draco construction. These diagnostics show how broadly
    # the starting weight is distributed across center-crossing orbits.
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

    linearized_constraint_error_l2 = norm(Cm * dw - base_delY)
    post_step_constraint_l2 = norm(target .- Cm * wnew)

    rcond_est = compute_rcond ? 1.0 / max(cond(Am), 1.0) : NaN
    max_abs_dw = maximum(abs, dw)

    return (
        w=wnew, dw=dw, lambda=lambda, Am=Am, delY=base_delY, model=model,
        rcond_est=rcond_est, max_abs_dw=max_abs_dw, stepfac=apfac,
        limiting_idx=limiting_idx, limiting_weight=limiting_weight, limiting_dw=limiting_dw, limiting_candidate=limiting_candidate,
        min_orbit_weight=min_orbit_weight, min_updated_orbit_weight=min_updated_orbit_weight, min_updated_orbit_idx=min_updated_orbit_idx,
        initial_min_trial_weight=initial_min_trial_weight, initial_min_trial_idx=initial_min_trial_idx,
        n_initial_violators=n_initial_violators, initial_n_negative_dw=initial_n_negative_dw,
        n_nonpositive_orbits=n_nonpositive_orbits, n_below_floor=n_below_floor,
        n_at_floor=n_at_floor_after, n_at_floor_before=n_at_floor_before, n_at_floor_after=n_at_floor_after,
        n_negative_dw=n_negative_dw, n_active_bound=n_active_bound, n_activated_total=n_active_bound,
        n_free_orbits_final=n_free_orbits_final, active_passes=active_passes,
        max_batch_activated=max_batch_activated, last_batch_activated=last_batch_activated,
        active_set_stabilized=active_set_stabilized, weight_moved_to_floor=weight_moved_to_floor,
        linearized_constraint_error_l2=linearized_constraint_error_l2, post_step_constraint_l2=post_step_constraint_l2,
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

function karl_spear_step_light_losvd_all(
    w_all::Vector{Float64}, Norbit::Int, Cm::Matrix{Float64}, target::Vector{Float64}, losvd_sigma::Vector{Float64},
    wphase_orbit::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, apfac::Float64=DEFAULT_KARL_APFAC,
    entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, compute_rcond::Bool=true)

    Narr, nvar = size(Cm)

    length(w_all) == nvar || error("w_all length does not match expanded Cm columns")
    length(target) == Narr || error("target length does not match expanded Cm rows")
    0 < Norbit < nvar || error("expanded Cm state must contain orbit weights followed by LOSVD slack variables")
    length(wphase_orbit) == Norbit || error("wphase_orbit length must match Norbit")

    Nslack = nvar - Norbit
    length(losvd_sigma) == Nslack || error("losvd_sigma length must match the number of expanded slack variables")

    entropy, chi_slack, dS, ddS = build_expanded_entropy_derivatives(
        w_all, Norbit, wphase_orbit, losvd_sigma; alphat=alphat, entropy_floor=entropy_floor)

    out = karl_spear_update_expanded(
        w_all, Norbit, Cm, target, dS, ddS;
        apfac=apfac, entropy_floor=entropy_floor, compute_rcond=compute_rcond)

    base_diag = (
        entropy=entropy, chi_slack=chi_slack, rcond_est=out.rcond_est, max_abs_dw=out.max_abs_dw, stepfac=out.stepfac,
        limiting_idx=out.limiting_idx, limiting_weight=out.limiting_weight, limiting_dw=out.limiting_dw, limiting_candidate=out.limiting_candidate,
        min_orbit_weight=out.min_orbit_weight, min_updated_orbit_weight=out.min_updated_orbit_weight,
        initial_min_trial_weight=out.initial_min_trial_weight, initial_min_trial_idx=out.initial_min_trial_idx,
        n_initial_violators=out.n_initial_violators, initial_n_negative_dw=out.initial_n_negative_dw,
        n_nonpositive_orbits=out.n_nonpositive_orbits, n_below_floor=out.n_below_floor,
        n_at_floor=out.n_at_floor, n_at_floor_before=out.n_at_floor_before, n_at_floor_after=out.n_at_floor_after,
        n_negative_dw=out.n_negative_dw, n_active_bound=out.n_active_bound, n_activated_total=out.n_activated_total,
        n_free_orbits_final=out.n_free_orbits_final, active_passes=out.active_passes,
        max_batch_activated=out.max_batch_activated, last_batch_activated=out.last_batch_activated,
        active_set_stabilized=out.active_set_stabilized, weight_moved_to_floor=out.weight_moved_to_floor,
        linearized_constraint_error_l2=out.linearized_constraint_error_l2, post_step_constraint_l2=out.post_step_constraint_l2,
        first_constraint_target=out.first_constraint_target,
        first_constraint_model_before=out.first_constraint_model_before,
        first_constraint_model_after=out.first_constraint_model_after,
        first_constraint_relative_before=out.first_constraint_relative_before,
        first_constraint_relative_after=out.first_constraint_relative_after,
        first_constraint_n_contributors=out.first_constraint_n_contributors,
        first_constraint_weight_on_contributors_before=out.first_constraint_weight_on_contributors_before,
        first_constraint_weight_on_contributors_after=out.first_constraint_weight_on_contributors_after,
        first_constraint_max_orbit_coefficient=out.first_constraint_max_orbit_coefficient,
        first_constraint_max_contribution_before=out.first_constraint_max_contribution_before,
        first_constraint_max_contribution_after=out.first_constraint_max_contribution_after,
        first_constraint_max_contribution_idx=out.first_constraint_max_contribution_idx,
    )

    # Printed on iteration 1, every rcond_every iteration, and maxiter because
    # the outer solver controls compute_rcond on that cadence.
    if compute_rcond
        println(
            "[BATCH ACTIVE DIAG]",
            " active_passes=", out.active_passes,
            " initial_violators=", out.n_initial_violators,
            " activated_total=", out.n_activated_total,
            " max_batch=", out.max_batch_activated,
            " free_orbits=", out.n_free_orbits_final,
            " floor_before=", out.n_at_floor_before,
            " floor_after=", out.n_at_floor_after,
            " initial_min_trial=", out.initial_min_trial_weight,
            " final_min_weight=", out.min_updated_orbit_weight,
            " weight_moved_to_floor=", out.weight_moved_to_floor,
            " first_target=", out.first_constraint_target,
            " first_model_before=", out.first_constraint_model_before,
            " first_model_after=", out.first_constraint_model_after,
            " first_relative_before=", out.first_constraint_relative_before,
            " first_relative_after=", out.first_constraint_relative_after,
            " first_contributors=", out.first_constraint_n_contributors,
            " first_contributor_weight_before=", out.first_constraint_weight_on_contributors_before,
            " first_contributor_weight_after=", out.first_constraint_weight_on_contributors_after,
            " first_max_coefficient=", out.first_constraint_max_orbit_coefficient,
            " first_max_contribution=", out.first_constraint_max_contribution_before,
            " first_max_contribution_idx=", out.first_constraint_max_contribution_idx,
            " linearized_constraint_error_l2=", out.linearized_constraint_error_l2,
            " post_step_constraint_l2=", out.post_step_constraint_l2,
        )
    end

    if !isfinite(out.stepfac) || out.stepfac <= 0.0
        failure_reason = !isfinite(out.stepfac) ? :nonfinite_stepfac : :nonpositive_stepfac
        return w_all, false, merge(base_diag, (slack=Float64[], w_all=w_all, failure_reason=failure_reason))
    end

    if !out.active_set_stabilized
        return w_all, false, merge(base_diag, (slack=Float64[], w_all=w_all, failure_reason=:active_set_not_stabilized))
    end

    w_all_new = Vector{Float64}(out.w)

    if !all(isfinite, w_all_new)
        return w_all, false, merge(base_diag, (max_abs_dw=Inf, slack=Float64[], w_all=w_all, failure_reason=:nonfinite_updated_state))
    end

    if out.n_below_floor > 0 || out.n_nonpositive_orbits > 0
        return w_all, false, merge(base_diag, (slack=Float64[], w_all=w_all, failure_reason=:orbit_weight_below_floor))
    end

    return w_all_new, true, merge(base_diag, (
        slack=Vector{Float64}(w_all_new[(Norbit + 1):end]),
        w_all=w_all_new,
    ))
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

function solve_weights_karl_expanded_cm( A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, light_target::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64};
    alphat::Float64=DEFAULT_KARL_ALPHAT, light_rel_tol::Float64=DEFAULT_KARL_LIGHT_REL_TOL, delta_chi2_iter_tol::Float64=DEFAULT_KARL_DELTA_CHI2_ITER_TOL, wphase=nothing,
    maxiter::Int=DEFAULT_KARL_MAXITER, seed::UInt=UInt(0), entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, apfac::Float64=DEFAULT_KARL_APFAC, return_diag::Bool=false,
    rcond_every::Int=250)
    Nlight, Norbit = size(A_light)
    Nlosvd, Norbit2 = size(A_losvd)
    fail_w = zeros(Float64, Norbit)

    Norbit == Norbit2 ||
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    Nlight > 0 ||
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(light_target) == Nlight ||
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(losvd_target) == Nlosvd ||
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(losvd_sigma) == Nlosvd ||
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    Norbit > 0 ||
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    light_rel_tol > 0.0 ||
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    delta_chi2_iter_tol >= 0.0 ||
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    rcond_every > 0 ||
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, A_light) &&
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, A_losvd) &&
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, light_target) &&
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, losvd_target) &&
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, losvd_sigma) &&
        return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    vsig = max.(abs.(Float64.(losvd_sigma)), 1e-12)
    wp = _prepare_wphase( wphase, Norbit; entropy_floor=entropy_floor)
    w = if iseven(Norbit)
        karl_initial_weights_from_wphase( wp; paired=true, rotfrac=0.75, floor=entropy_floor)
    else
        karl_initial_weights_from_wphase( wp; paired=false, floor=entropy_floor)
    end
    enforce_normalization = !_expanded_light_implies_normalization(A_light, light_target)
    # These objects are fixed for the entire weight solve.
    Cm = build_expanded_Cm_with_losvd_slack( A_light, A_losvd; enforce_normalization=enforce_normalization)
    target = build_expanded_target( light_target, losvd_target; enforce_normalization=enforce_normalization)
    w_all = build_expanded_weights_initial( w, A_losvd, losvd_target)
    previous_chi2_losvd = chi2_block(A_losvd, w, losvd_target, vsig)
    delta_chi2_iteration = Inf
    max_light_relative_residual_value = max_light_relative_residual(A_light, w, light_target)
    light_constraint_ok = max_light_relative_residual_value <= light_rel_tol
    last_diag = nothing
    last_rcond_est = NaN
    failure_reason = :none
    iterations = 0
    converged = false
    ok = true
    # Reused iteration work arrays.
    w_current = similar(w)
    slack_current = Vector{Float64}(undef, Nlosvd)
    losvd_model_current = Vector{Float64}(undef, Nlosvd)
    losvd_residual_current = Vector{Float64}(undef, Nlosvd)

    for iter in 1:maxiter
        iterations = iter
        compute_rcond = iter == 1 || iter == maxiter || iter % rcond_every == 0
        w_all_new, step_ok, sdiag = karl_spear_step_light_losvd_all( w_all, Norbit, Cm, target, vsig, wp; alphat=alphat, apfac=apfac, entropy_floor=entropy_floor, compute_rcond=compute_rcond)
        last_diag = sdiag
        if isfinite(sdiag.rcond_est)
            last_rcond_est = sdiag.rcond_est
        end
        if !step_ok
            rcond_report = isfinite(sdiag.rcond_est) ?
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
        copyto!( slack_current, 1, w_all, Norbit + 1, Nlosvd)
        # One LOSVD matrix product supplies both χ² and slack consistency.
        mul!(losvd_model_current, A_losvd, w_current)
        chi2_losvd_current = 0.0
        slack_residual_sq = 0.0
        slack_norm_sq = 0.0
        losvd_residual_norm_sq = 0.0
        @inbounds for k in 1:Nlosvd
            residual = losvd_target[k] - losvd_model_current[k]
            losvd_residual_current[k] = residual
            scaled_residual = residual / vsig[k]
            chi2_losvd_current += scaled_residual * scaled_residual
            slack_difference = slack_current[k] - residual
            slack_residual_sq += slack_difference * slack_difference
            slack_norm_sq += slack_current[k] * slack_current[k]
            losvd_residual_norm_sq += residual * residual
        end

        delta_chi2_iteration = abs(chi2_losvd_current - previous_chi2_losvd)
        max_light_relative_residual_value = max_light_relative_residual( A_light, w_current, light_target)
        light_constraint_ok = max_light_relative_residual_value <= light_rel_tol
        slack_residual_l2_current = sqrt(slack_residual_sq)
        slack_scale_current = max( 1.0, sqrt(slack_norm_sq), sqrt(losvd_residual_norm_sq))
        slack_consistent_current = slack_residual_l2_current <= light_rel_tol * slack_scale_current
        normalized_current = abs(sum(w_current) - 1.0) <= light_rel_tol
        if iter == 1 || iter % 250 == 0 || iter == maxiter
            light_model_progress = A_light * w_current
            light_relative_progress = abs.(light_model_progress .- light_target) ./ max.(abs.(light_target), eps(Float64))
            worst_light_bin = argmax(light_relative_progress)

            println(
                "[WEIGHT PROGRESS]",
                " iteration=", iter, " N_active_bound=", sdiag.n_active_bound, " N_at_floor=", sdiag.n_at_floor,
                " stepfac=", sdiag.stepfac, " max_light_relative_residual=", light_relative_progress[worst_light_bin],
                " worst_light_bin=", worst_light_bin, " light_target=", light_target[worst_light_bin],
                " light_model=", light_model_progress[worst_light_bin], " delta_chi2=", delta_chi2_iteration,
            )
        end
        if light_constraint_ok && slack_consistent_current && normalized_current &&
           delta_chi2_iteration <= delta_chi2_iter_tol
            converged = true
            break
        end
        previous_chi2_losvd = chi2_losvd_current
    end
    w = Vector{Float64}(w_all[1:Norbit])
    slack = Vector{Float64}(w_all[(Norbit + 1):end])
    finite_state = all(isfinite, w) && all(isfinite, slack)
    if !finite_state
        failure_reason = :nonfinite_final_state
        ok = false
    end
    chi2_losvd = finite_state ?
        chi2_block(A_losvd, w, losvd_target, vsig) :
        Inf
    losvd_residual = finite_state ?
        losvd_target .- A_losvd * w :
        fill(Inf, Nlosvd)
    slack_residual_l2 = finite_state ?
        norm(slack .- losvd_residual) :
        Inf
    slack_scale = finite_state ?
        max(1.0, norm(slack), norm(losvd_residual)) :
        Inf
    slack_consistent = finite_state && slack_residual_l2 <= light_rel_tol * slack_scale
    normalization_error = finite_state ?
        abs(sum(w) - 1.0) :
        Inf
    normalized = normalization_error <= light_rel_tol
    light_residual_l2 = finite_state ?
        norm(light_target .- A_light * w) :
        Inf
    max_light_relative_residual_value = finite_state ?
        max_light_relative_residual( A_light, w, light_target) :
        Inf
    light_constraint_ok = max_light_relative_residual_value <= light_rel_tol
    constraint_l2 = finite_state ?
        norm(target .- Cm * w_all) :
        Inf
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
        chi2_slack = finite_state ?
            sum((slack ./ vsig) .^ 2) :
            Inf
        losvd_penalty = alphat * chi2_slack
        slack_to_losvd = chi2_losvd > 0.0 ?
            chi2_slack / chi2_losvd :
            NaN
        ent = finite_state ?
            karl_entropy_value( w, wp; entropy_floor=entropy_floor) :
            -Inf

        diag = ( entropy=ent, chi=chi2_losvd, chi_losvd=chi2_losvd, profit=ent - losvd_penalty, alphat=alphat, losvd_penalty=losvd_penalty, chi_slack=chi2_slack,
            slack_to_losvd=slack_to_losvd, delta_chi2_iteration=delta_chi2_iteration, delta_chi2_iteration_ok=delta_chi2_ok, delta_chi2_iteration_tol=delta_chi2_iter_tol,
            max_light_relative_residual= max_light_relative_residual_value, light_constraint_ok=light_constraint_ok, light_rel_tol=light_rel_tol, solver_converged=solver_converged,
            failure_reason=failure_reason, rcond_est=last_rcond_est,
            max_abs_dw=
                last_diag === nothing ?
                NaN :
                last_diag.max_abs_dw,
            stepfac=
                last_diag === nothing ?
                NaN :
                last_diag.stepfac,
            iterations=iterations, constraint_ok=constraint_ok, slack_consistent=slack_consistent, normalized=normalized, normalization_error=normalization_error, light_residual_l2=light_residual_l2,
            constraint_l2=constraint_l2, slack_residual_l2=slack_residual_l2,
            slack_l2=
                finite_state ?
                sum(slack .^ 2) :
                Inf,
            slack_max_abs=
                finite_state && !isempty(slack) ?
                maximum(abs.(slack)) :
                0.0,
            N_slack=Nlosvd, normalization_row_enforced=enforce_normalization,
            n_active_bound=
                last_diag === nothing ?
                0 :
                last_diag.n_active_bound,
            active_passes=
                last_diag === nothing ?
                0 :
                last_diag.active_passes )
        return w, solver_converged, diag
    end
    return w, solver_converged
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
