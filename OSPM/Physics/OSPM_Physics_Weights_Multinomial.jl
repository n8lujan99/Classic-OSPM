# ========================================================================================================================
# OSPM_Physics_Weights_Multinomial.jl — optional multinomial LOSVD weight backend.
# Included by OSPM_Physics_Weights.jl — do NOT load directly.
# Retained for controlled comparison with the production chi-square/SPEAR fit.
# Contains only multinomial-specific likelihood, curvature, and solver machinery.
# ========================================================================================================================

function build_light_constraint_Cm(A_light::Matrix{Float64}; enforce_normalization::Bool=true)
    Nlight, Norbit = size(A_light)
    Narr = Nlight + (enforce_normalization ? 1 : 0)
    Cm = zeros(Float64, Narr, Norbit)
    Cm[1:Nlight, :] .= A_light
    enforce_normalization && (Cm[Narr, :] .= 1.0)
    return Cm
end

function build_light_constraint_target(light_target::Vector{Float64}; enforce_normalization::Bool=true)
    return enforce_normalization ? vcat(light_target, 1.0) : copy(light_target)
end

function _selected_aperture_matrix(A_losvd::Matrix{Float64}, Nspatial::Int, Nvbin::Int)
    Nlosvd, Norbit = size(A_losvd)
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match A_losvd rows")
    selected = zeros(Float64, Nspatial, Norbit)
    @inbounds for ib in 1:Nspatial
        rows = ((ib - 1) * Nvbin + 1):(ib * Nvbin)
        selected[ib, :] .= vec(sum(@view(A_losvd[rows, :]), dims=1))
    end
    return selected
end

function multinomial_losvd_state(A_losvd::Matrix{Float64}, A_kinematic::Matrix{Float64}, w::Vector{Float64}, losvd_counts::Vector{Int}, Nspatial::Int, Nvbin::Int; conditioning=:vlos_cut)
    Nlosvd, Norbit = size(A_losvd)
    size(A_kinematic, 1) == Nspatial || error("A_kinematic row count does not match Nspatial")
    size(A_kinematic, 2) == Norbit || error("A_kinematic orbit count does not match A_losvd")
    length(w) == Norbit || error("w length does not match A_losvd columns")
    length(losvd_counts) == Nlosvd || error("losvd_counts length does not match A_losvd rows")
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match Nlosvd")
    any(<(0), losvd_counts) && error("multinomial LOSVD counts must be nonnegative")
    conditioning_sym = _normalize_losvd_conditioning(conditioning)
    model = A_losvd * w
    projected_total = A_kinematic * w
    all(isfinite, model) || error("multinomial LOSVD model contains nonfinite values")
    all(isfinite, projected_total) || error("multinomial aperture totals contain nonfinite values")
    all(>=(0.0), model) || error("multinomial LOSVD model contains negative bin mass")
    all(>=(0.0), projected_total) || error("multinomial aperture totals contain negative mass")
    probabilities = zeros(Float64, Nlosvd)
    counts_by_spatial = zeros(Int, Nspatial)
    selected_total = zeros(Float64, Nspatial)
    normalization = zeros(Float64, Nspatial)
    outside_probability = zeros(Float64, Nspatial)
    loglikelihood_by_spatial = zeros(Float64, Nspatial)
    saturated_loglikelihood_by_spatial = zeros(Float64, Nspatial)
    deviance_by_spatial = zeros(Float64, Nspatial)
    @inbounds for ib in 1:Nspatial
        rows = ((ib - 1) * Nvbin + 1):(ib * Nvbin)
        Ni = sum(@view losvd_counts[rows])
        counts_by_spatial[ib] = Ni
        selected = sum(@view model[rows])
        total = projected_total[ib]
        selected_total[ib] = selected
        if conditioning_sym === :none
            scale = max(abs(selected), abs(total), 1.0)
            selected <= total + 1.0e-12 * scale || error("LOSVD in-window model mass exceeds full aperture mass in spatial bin $ib")
        end
        denom = conditioning_sym === :vlos_cut ? selected : total
        normalization[ib] = denom
        Ni == 0 && continue
        isfinite(denom) && denom > 0.0 || error("multinomial LOSVD normalization is non-positive in populated spatial bin $ib")
        outside_probability[ib] = conditioning_sym === :none ? max(total - selected, 0.0) / denom : 0.0
        log_i = 0.0
        sat_i = 0.0
        for row in rows
            nij = losvd_counts[row]
            qij = model[row] / denom
            qij >= 0.0 || error("multinomial LOSVD probability became negative in row $row")
            probabilities[row] = qij
            if nij > 0
                qij > 0.0 || error("observed LOSVD count has zero model probability in row $row")
                log_i += nij * log(qij)
                phat = nij / Ni
                sat_i += nij * log(phat)
            end
        end
        qsum = sum(@view probabilities[rows])
        if conditioning_sym === :vlos_cut
            abs(qsum - 1.0) <= 1.0e-10 || error("conditioned LOSVD probabilities do not sum to one in spatial bin $ib")
        else
            qsum <= 1.0 + 1.0e-10 || error("unconditioned LOSVD in-grid probabilities exceed one in spatial bin $ib")
        end
        loglikelihood_by_spatial[ib] = log_i
        saturated_loglikelihood_by_spatial[ib] = sat_i
        deviance_by_spatial[ib] = max(0.0, 2.0 * (sat_i - log_i))
    end
    return (model=model, projected_total=projected_total, selected_total=selected_total, normalization=normalization, probabilities=probabilities, outside_probability=outside_probability, counts_by_spatial=counts_by_spatial, loglikelihood_by_spatial=loglikelihood_by_spatial, loglikelihood=sum(loglikelihood_by_spatial), saturated_loglikelihood=sum(saturated_loglikelihood_by_spatial), deviance_by_spatial=deviance_by_spatial, deviance_total=sum(deviance_by_spatial), conditioning=conditioning_sym)
end

function build_multinomial_entropy_derivatives(w::Vector{Float64}, wphase::Vector{Float64}, A_losvd::Matrix{Float64}, A_kinematic::Matrix{Float64}, A_losvd_sq::Matrix{Float64}, denominator_matrix::Matrix{Float64}, denominator_matrix_sq::Matrix{Float64}, outside_matrix, outside_matrix_sq, losvd_counts::Vector{Int}, Nspatial::Int, Nvbin::Int; conditioning=:vlos_cut, alphat::Float64=DEFAULT_ALPHAT)
    state = multinomial_losvd_state(A_losvd, A_kinematic, w, losvd_counts, Nspatial, Nvbin; conditioning=conditioning)
    Norbit = length(w)
    Nlosvd = length(losvd_counts)
    length(wphase) == Norbit || error("wphase length does not match orbit weights")
    isfinite(alphat) && alphat >= 0.0 || error("multinomial alphat must be finite and nonnegative")
    size(A_losvd_sq) == size(A_losvd) || error("A_losvd_sq shape mismatch")
    size(denominator_matrix) == (Nspatial, Norbit) || error("multinomial denominator matrix shape mismatch")
    size(denominator_matrix_sq) == size(denominator_matrix) || error("multinomial denominator-square matrix shape mismatch")
    if state.conditioning === :none
        outside_matrix === nothing && error("unconditioned multinomial Fisher curvature requires outside_matrix")
        outside_matrix_sq === nothing && error("unconditioned multinomial Fisher curvature requires outside_matrix_sq")
        size(outside_matrix) == (Nspatial, Norbit) || error("multinomial outside matrix shape mismatch")
        size(outside_matrix_sq) == size(outside_matrix) || error("multinomial outside-square matrix shape mismatch")
    end
    row_gradient_coeff = zeros(Float64, Nlosvd)
    row_fisher_coeff = zeros(Float64, Nlosvd)
    aperture_gradient_coeff = zeros(Float64, Nspatial)
    aperture_fisher_coeff = zeros(Float64, Nspatial)
    outside_fisher_coeff = zeros(Float64, Nspatial)
    @inbounds for ib in 1:Nspatial
        rows = ((ib - 1) * Nvbin + 1):(ib * Nvbin)
        Ni = state.counts_by_spatial[ib]
        Ni == 0 && continue
        denom = state.normalization[ib]
        aperture_gradient_coeff[ib] = Ni / denom
        aperture_fisher_coeff[ib] = Ni / (denom * denom)
        for row in rows
            nij = losvd_counts[row]
            mij = state.model[row]
            if nij > 0
                mij > 0.0 || error("observed LOSVD count has zero model mass in row $row")
                row_gradient_coeff[row] = nij / mij
            end
            if mij > 0.0
                row_fisher_coeff[row] = Ni / (denom * mij)
            end
        end
        if state.conditioning === :none && outside_matrix !== nothing
            outside_mass = max(state.projected_total[ib] - state.selected_total[ib], 0.0)
            outside_mass > 0.0 && (outside_fisher_coeff[ib] = Ni / (denom * outside_mass))
        end
    end
    likelihood_gradient = transpose(A_losvd) * row_gradient_coeff - transpose(denominator_matrix) * aperture_gradient_coeff
    fisher_diag = transpose(A_losvd_sq) * row_fisher_coeff - transpose(denominator_matrix_sq) * aperture_fisher_coeff
    if state.conditioning === :none && outside_matrix !== nothing
        fisher_diag .+= transpose(outside_matrix_sq) * outside_fisher_coeff
    end
    all(isfinite, likelihood_gradient) || error("multinomial likelihood gradient contains nonfinite values")
    all(isfinite, fisher_diag) || error("multinomial Fisher diagonal contains nonfinite values")
    fisher_diag .= max.(fisher_diag, 0.0)
    dS = zeros(Float64, Norbit)
    ddS = zeros(Float64, Norbit)
    entropy = 0.0
    @inbounds for j in 1:Norbit
        wj = w[j]
        isfinite(wj) && wj > 0.0 || error("multinomial entropy received non-positive orbit weight at column $j: $wj")
        pj = wphase[j]
        isfinite(pj) && pj > 0.0 || error("multinomial entropy received invalid inverse phase volume at column $j")
        log_measure = log(wj) + log(pj)
        entropy -= wj * log_measure
        dS[j] = -1.0 - log_measure + alphat * likelihood_gradient[j]
        ddS[j] = -1.0 / wj - alphat * fisher_diag[j]
    end
    return entropy, state, dS, ddS, fisher_diag
end

function solve_weights_multinomial(A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, A_kinematic::Matrix{Float64}, light_target::Vector{Float64}, light_sigma::Vector{Float64}, losvd_counts::Vector{Int}; Nspatial::Int, Nvbin::Int, conditioning=:vlos_cut, alphat::Float64=DEFAULT_ALPHAT, light_rel_tol::Float64=DEFAULT_LIGHT_REL_TOL, light_sigma_tol::Float64=2.0, delta_statistic_iter_tol::Float64=DEFAULT_DELTA_CHI2_ITER_TOL, wphase=nothing, maxiter::Int=DEFAULT_MAXITER, seed::UInt=UInt(0), entropy_floor::Float64=DEFAULT_ENTROPY_FLOOR, apfac::Float64=DEFAULT_APFAC, return_diag::Bool=false, rcond_every::Int=250, rcond_warn::Float64=DEFAULT_SPEAR_RCOND_WARN)
    Nlight, Norbit = size(A_light)
    Nlosvd, Norbit2 = size(A_losvd)
    fail_w = zeros(Float64, Norbit)
    Norbit == Norbit2 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    size(A_kinematic) == (Nspatial, Norbit) || error("A_kinematic shape does not match multinomial solver")
    Nlight > 0 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    Nspatial * Nvbin == Nlosvd || error("Nspatial*Nvbin does not match Nlosvd")
    length(light_target) == Nlight || error("light_target length does not match A_light")
    length(light_sigma) == Nlight || error("light_sigma length does not match A_light")
    length(losvd_counts) == Nlosvd || error("losvd_counts length does not match A_losvd")
    any(<(0), losvd_counts) && error("losvd_counts must be nonnegative")
    isfinite(alphat) && alphat >= 0.0 || error("multinomial alphat must be finite and nonnegative")
    isfinite(delta_statistic_iter_tol) && delta_statistic_iter_tol >= 0.0 || error("delta_statistic_iter_tol must be finite and nonnegative")
    maxiter > 0 || error("multinomial maxiter must be positive")
    rcond_every > 0 || error("multinomial rcond_every must be positive")
    iseven(Norbit) || error("Phase-volume solving requires paired orbit columns")
    conditioning_sym = _normalize_losvd_conditioning(conditioning)
    light_sigma_diagnostics(w) = _tracer_sigma_diagnostics(A_light * w, light_target, light_sigma)
    wp = _prepare_wphase(wphase, Norbit; require_paired=true, pair_rtol=1.0e-12)
    wphase_diag = wphase_diagnostics(wp; paired=true)
    w = fill(1.0 / Norbit, Norbit)
    selected_matrix = _selected_aperture_matrix(A_losvd, Nspatial, Nvbin)
    min_outside_entry = minimum(A_kinematic .- selected_matrix)
    scale_outside = max(1.0, maximum(abs, A_kinematic), maximum(abs, selected_matrix))
    min_outside_entry >= -1.0e-12 * scale_outside || error("A_kinematic is smaller than summed A_losvd for at least one orbit/aperture cell")
    outside_matrix = max.(A_kinematic .- selected_matrix, 0.0)
    denominator_matrix = conditioning_sym === :vlos_cut ? selected_matrix : A_kinematic
    A_losvd_sq = A_losvd .* A_losvd
    denominator_matrix_sq = denominator_matrix .* denominator_matrix
    outside_matrix_sq = conditioning_sym === :none ? outside_matrix .* outside_matrix : nothing

    enforce_normalization = !_expanded_light_implies_normalization(A_light, light_target)
    Cm = build_light_constraint_Cm(A_light; enforce_normalization=enforce_normalization)
    target = build_light_constraint_target(light_target; enforce_normalization=enforce_normalization)

    zero_support_rows = Int[]
    @inbounds for row in eachindex(losvd_counts)
        if losvd_counts[row] > 0 && !any(>(0.0), @view(A_losvd[row, :]))
            push!(zero_support_rows, row)
        end
    end

    if !isempty(zero_support_rows)
        zero_support_apertures = [div(row - 1, Nvbin) + 1 for row in zero_support_rows]
        zero_support_velocity_bins = [mod(row - 1, Nvbin) + 1 for row in zero_support_rows]
        zero_support_counts = losvd_counts[zero_support_rows]
        @warn "Observed LOSVD bins have zero orbit-library support" rows=zero_support_rows apertures=zero_support_apertures velocity_bins=zero_support_velocity_bins observed_counts=zero_support_counts
    end

    initial_state = multinomial_losvd_state(A_losvd, A_kinematic, w, losvd_counts, Nspatial, Nvbin; conditioning=conditioning_sym)

    previous_statistic = initial_state.deviance_total
    delta_statistic_iteration = Inf
    max_light_relative_residual_value = max_light_relative_residual(A_light, w, light_target)
    light_sigma_diag = light_sigma_diagnostics(w)
    max_light_sigma_residual_value = light_sigma_diag.max_sigma_residual
    light_constraint_ok = max_light_relative_residual_value <= light_rel_tol
    light_sigma_constraint_ok = light_sigma_diag.worst_sigma_bin == 0 || max_light_sigma_residual_value <= light_sigma_tol
    last_diag = nothing
    last_rcond_est = NaN
    failure_reason = :none
    iterations = 0
    converged = false
    ok = true
    for iter in 1:maxiter
        iterations = iter
        compute_rcond = iter == 1 || iter == maxiter || iter % rcond_every == 0
        entropy_current, state_current, dS, ddS, fisher_diag = build_multinomial_entropy_derivatives(w, wp, A_losvd, A_kinematic, A_losvd_sq, denominator_matrix, denominator_matrix_sq, conditioning_sym === :none ? outside_matrix : nothing, outside_matrix_sq, losvd_counts, Nspatial, Nvbin; conditioning=conditioning_sym, alphat=alphat)
        sdiag = spear_update_expanded(w, Norbit, Cm, target, dS, ddS; apfac=apfac, entropy_floor=entropy_floor, compute_rcond=compute_rcond, rcond_warn=rcond_warn)
        wnew = Vector{Float64}(sdiag.w)
        finite_update = all(isfinite, wnew)
        positive_update = finite_update && all(>(0.0), wnew)
        step_ok = finite_update && positive_update && sdiag.active_set_stabilized
        last_diag = sdiag
        isfinite(sdiag.rcond_est) && (last_rcond_est = sdiag.rcond_est)
        if !step_ok
            failure_reason = !finite_update ? :nonfinite_updated_state : !positive_update ? :negative_orbit_weights : :active_set_failed
            ok = false
            break
        end
        state_scale = max(1.0, maximum(abs, w), maximum(abs, wnew))
        state_change = maximum(abs.(wnew .- w))
        state_tol = 100.0 * eps(Float64) * state_scale
        w .= wnew
        state_new = multinomial_losvd_state(A_losvd, A_kinematic, w, losvd_counts, Nspatial, Nvbin; conditioning=conditioning_sym)
        statistic_current = state_new.deviance_total
        delta_statistic_iteration = abs(statistic_current - previous_statistic)
        max_light_relative_residual_value = max_light_relative_residual(A_light, w, light_target)
        light_sigma_diag = light_sigma_diagnostics(w)
        max_light_sigma_residual_value = light_sigma_diag.max_sigma_residual
        light_constraint_ok = max_light_relative_residual_value <= light_rel_tol
        light_sigma_constraint_ok = light_sigma_diag.worst_sigma_bin == 0 || max_light_sigma_residual_value <= light_sigma_tol
        normalized = abs(sum(w) - 1.0) <= light_rel_tol
        println("[WEIGHT PROGRESS MULTINOMIAL] iteration=", iter, " conditioning=", conditioning_sym, " deviance=", statistic_current, " loglikelihood=", state_new.loglikelihood, " max_light_relative_residual=", max_light_relative_residual_value, " light_constraint_ok=", light_constraint_ok, " max_light_sigma_residual=", max_light_sigma_residual_value, " light_sigma_constraint_ok=", light_sigma_constraint_ok, " zero_target_floor_rows=", light_sigma_diag.zero_target_floor_rows, " max_zero_target_leakage_rel=", light_sigma_diag.max_zero_target_leakage_rel, " worst_zero_target_bin=", light_sigma_diag.worst_zero_target_bin, " delta_statistic=", delta_statistic_iteration, " fisher_diag_min=", minimum(fisher_diag), " fisher_diag_max=", maximum(fisher_diag))
        if light_constraint_ok && normalized && delta_statistic_iteration <= delta_statistic_iter_tol
            converged = true
            break
        end
        stat_scale = max(1.0, abs(previous_statistic), abs(statistic_current))
        stat_stationary_tol = 100.0 * eps(Float64) * stat_scale
        if state_change <= state_tol && delta_statistic_iteration <= stat_stationary_tol
            failure_reason = :stationary_constraints_unsatisfied
            ok = false
            break
        end
        previous_statistic = statistic_current
    end
    finite_state = all(isfinite, w)
    !finite_state && (failure_reason = :nonfinite_final_state; ok = false)
    final_state = finite_state ? multinomial_losvd_state(A_losvd, A_kinematic, w, losvd_counts, Nspatial, Nvbin; conditioning=conditioning_sym) : nothing
    statistic = final_state === nothing ? Inf : final_state.deviance_total
    normalization_error = finite_state ? abs(sum(w) - 1.0) : Inf
    normalized = normalization_error <= light_rel_tol
    max_light_relative_residual_value = finite_state ? max_light_relative_residual(A_light, w, light_target) : Inf
    light_sigma_diag = finite_state ? light_sigma_diagnostics(w) : nothing
    max_light_sigma_residual_value = light_sigma_diag === nothing ? Inf : light_sigma_diag.max_sigma_residual
    light_constraint_ok = max_light_relative_residual_value <= light_rel_tol
    light_sigma_constraint_ok = light_sigma_diag !== nothing && (light_sigma_diag.worst_sigma_bin == 0 || max_light_sigma_residual_value <= light_sigma_tol)
    constraint_l2 = finite_state ? norm(target .- Cm * w) : Inf
    light_residual_l2 = finite_state ? norm(light_target .- A_light * w) : Inf
    constraint_ok = light_constraint_ok && normalized
    delta_statistic_ok = isfinite(delta_statistic_iteration) && delta_statistic_iteration <= delta_statistic_iter_tol
    solver_converged = ok && converged && constraint_ok && delta_statistic_ok
    if ok && !light_constraint_ok
        failure_reason = :light_constraint_failed
    elseif ok && !normalized
        failure_reason = :normalization_failed
    elseif ok && !delta_statistic_ok
        failure_reason = :delta_statistic_not_converged
    elseif ok && !solver_converged
        failure_reason = :multinomial_not_converged
    end
    if return_diag
        ent = finite_state ? entropy_value(w, wp; entropy_floor=entropy_floor) : -Inf
        losvd_penalty = 0.5 * alphat * statistic
        diag = (entropy=ent, chi=statistic, chi_losvd=statistic, fit_statistic=:multinomial_deviance, conditioning=conditioning_sym, loglikelihood=final_state === nothing ? -Inf : final_state.loglikelihood, saturated_loglikelihood=final_state === nothing ? -Inf : final_state.saturated_loglikelihood, deviance=statistic, deviance_by_spatial=final_state === nothing ? fill(Inf, Nspatial) : final_state.deviance_by_spatial, probabilities=final_state === nothing ? Float64[] : final_state.probabilities, outside_probability=final_state === nothing ? Float64[] : final_state.outside_probability, profit=ent-losvd_penalty, alphat=alphat, losvd_penalty=losvd_penalty, chi_slack=NaN, slack_to_losvd=NaN, fracnew=Float64[], fracnew_min=NaN, fracnew_max=NaN, delta_fit_statistic_iteration=delta_statistic_iteration, delta_chi2_iteration=delta_statistic_iteration, delta_chi2_iteration_step_normalized=NaN, delta_chi2_iteration_ok=delta_statistic_ok, delta_chi2_iteration_tol=delta_statistic_iter_tol, max_light_relative_residual=max_light_relative_residual_value, max_light_sigma_residual=max_light_sigma_residual_value, zero_target_floor_rows=light_sigma_diag === nothing ? 0 : light_sigma_diag.zero_target_floor_rows, max_zero_target_leakage_rel=light_sigma_diag === nothing ? Inf : light_sigma_diag.max_zero_target_leakage_rel, worst_zero_target_bin=light_sigma_diag === nothing ? 0 : light_sigma_diag.worst_zero_target_bin, light_constraint_ok=light_constraint_ok, light_sigma_constraint_ok=light_sigma_constraint_ok, light_rel_tol=light_rel_tol, light_sigma_tol=light_sigma_tol, solver_converged=solver_converged, failure_reason=failure_reason, rcond_est=last_rcond_est, max_abs_dw=last_diag === nothing ? NaN : last_diag.max_abs_dw, stepfac=last_diag === nothing ? NaN : last_diag.stepfac, iterations=iterations, constraint_ok=constraint_ok, slack_consistent=true, normalized=normalized, normalization_error=normalization_error, light_residual_l2=light_residual_l2, constraint_l2=constraint_l2, slack_residual_l2=0.0, slack_l2=0.0, slack_max_abs=0.0, N_slack=0, normalization_row_enforced=enforce_normalization, n_active_bound=0, active_passes=last_diag === nothing ? 0 : last_diag.active_passes, wphase_required=true, wphase_convention=wphase_diag.convention, wphase_entropy_expression=wphase_diag.entropy_expression, wphase_min=wphase_diag.wphase_min, wphase_max=wphase_diag.wphase_max, wphase_dynamic_range=wphase_diag.wphase_dynamic_range, wphase_log_dynamic_range=wphase_diag.wphase_log_dynamic_range, wphase_geometric_mean=wphase_diag.wphase_geometric_mean, phase_volume_min=wphase_diag.phase_volume_min, phase_volume_max=wphase_diag.phase_volume_max, phase_volume_dynamic_range=wphase_diag.phase_volume_dynamic_range, wphase_pair_max_relative_mismatch=wphase_diag.pair_max_relative_mismatch, raw_spear_rank=0, raw_spear_nullity=0, raw_spear_supported_rcond=NaN, raw_spear_negative_orbits=0, raw_spear_first_boundary_idx=0, raw_spear_first_boundary_candidate=NaN, raw_spear_system_relative_residual=NaN)
        return w, solver_converged, diag
    end
    return w, solver_converged
end
