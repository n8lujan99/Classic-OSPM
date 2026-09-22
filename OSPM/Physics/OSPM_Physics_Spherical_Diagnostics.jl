# ========================================================================================================================
# OSPM_Physics_Spherical_Diagnostics.jl
# ========================================================================================================================
function _print_tracer_bin_diagnostics(A_constraint::Matrix{Float64}, w::Vector{Float64}, target::Vector{Float64}, sigma::Vector{Float64}, constraint_edges::Vector{Float64}; model_index::Int=1, print_bins::Bool=false, d3_constraints=nothing)
    size(A_constraint, 1) == length(target) || error("Tracer diagnostic target length mismatch")
    length(target) == length(sigma) || error("Tracer diagnostic sigma length mismatch")
    size(A_constraint, 2) == length(w) || error("Tracer diagnostic weight length mismatch")
    d3_constraints === nothing ? (length(constraint_edges) == length(target) + 1 || error("Tracer diagnostic edge length mismatch")) : (length(d3_constraints.target) == length(target) || error("Tracer diagnostic d3 row mismatch"))
    iseven(length(w)) || error("Tracer diagnostic requires prograde/retrograde orbit pairs")

    model = A_constraint * w
    sigma_diag = _tracer_sigma_diagnostics(model, target, sigma)
    Nbase = length(w) ÷ 2
    relative_residual = zeros(Float64, length(target))
    support_fraction = zeros(Float64, length(target))
    effective_support = zeros(Float64, length(target))
    zero_target_atol = sigma_diag.zero_target_atol
    zero_target_reference = sigma_diag.zero_target_reference

    @inbounds for ib in eachindex(target)
        pair_contrib = zeros(Float64, Nbase)
        n_support = 0

        for c in 1:Nbase
            jpro = 2 * c - 1
            jret = 2 * c
            apro = A_constraint[ib, jpro]
            aret = A_constraint[ib, jret]
            if apro > 0.0 || aret > 0.0
                n_support += 1
            end

            pair_contrib[c] = apro * w[jpro] + aret * w[jret]
        end

        total_contrib = sum(pair_contrib)
        if isfinite(total_contrib) && total_contrib > 0.0
            p = pair_contrib ./ total_contrib
            effective_support[ib] = 1.0 / sum(abs2, p)
        end

        relative_denominator = abs(target[ib]) > zero_target_atol ? abs(target[ib]) : zero_target_reference
        relative_residual[ib] = abs(model[ib] - target[ib]) / relative_denominator
        support_fraction[ib] = n_support / Nbase

        if print_bins
            if d3_constraints === nothing
                if sigma_diag.zero_target_floor[ib]
                    println("[TRACER BIN]", " i=", model_index, " bin=", ib, " R_inner_pc=", constraint_edges[ib] / pc, " R_outer_pc=", constraint_edges[ib + 1] / pc, " target=", target[ib], " model=", model[ib], " sigma_residual=not_applicable", " zero_target_leakage_rel=", sigma_diag.zero_target_leakage_rel[ib], " support_fraction=", support_fraction[ib], " effective_N_base_orbits=", effective_support[ib])
                else
                    println("[TRACER BIN]", " i=", model_index, " bin=", ib, " R_inner_pc=", constraint_edges[ib] / pc, " R_outer_pc=", constraint_edges[ib + 1] / pc, " target=", target[ib], " model=", model[ib], " sigma_residual=", sigma_diag.sigma_residual[ib], " zero_target_leakage_rel=not_applicable", " support_fraction=", support_fraction[ib], " effective_N_base_orbits=", effective_support[ib])
                end
            else
                ir = d3_constraints.row_radial[ib]
                iv = d3_constraints.row_angular[ib]
                vinner = iv == 0 ? d3_constraints.angular_edges[1] : d3_constraints.angular_edges[iv]
                vouter = iv == 0 ? d3_constraints.angular_edges[end] : d3_constraints.angular_edges[iv + 1]
                if sigma_diag.zero_target_floor[ib]
                    println("[TRACER BIN]", " i=", model_index, " bin=", ib, " radial_bin=", ir, " angular_bin=", iv, " R_inner_pc=", d3_constraints.radial_edges_m[ir] / pc, " R_outer_pc=", d3_constraints.radial_edges_m[ir + 1] / pc, " vcoord_inner=", vinner, " vcoord_outer=", vouter, " target=", target[ib], " model=", model[ib], " sigma_residual=not_applicable", " zero_target_leakage_rel=", sigma_diag.zero_target_leakage_rel[ib], " support_fraction=", support_fraction[ib], " effective_N_base_orbits=", effective_support[ib])
                else
                    println("[TRACER BIN]", " i=", model_index, " bin=", ib, " radial_bin=", ir, " angular_bin=", iv, " R_inner_pc=", d3_constraints.radial_edges_m[ir] / pc, " R_outer_pc=", d3_constraints.radial_edges_m[ir + 1] / pc, " vcoord_inner=", vinner, " vcoord_outer=", vouter, " target=", target[ib], " model=", model[ib], " sigma_residual=", sigma_diag.sigma_residual[ib], " zero_target_leakage_rel=not_applicable", " support_fraction=", support_fraction[ib], " effective_N_base_orbits=", effective_support[ib])
                end
            end
        end
    end

    max_relative_residual, worst_relative_bin = findmax(relative_residual)
    min_support_fraction, weakest_support_bin = findmin(support_fraction)
    min_effective_support, weakest_effective_bin = findmin(effective_support)

    println("[TRACER]",
        " i=", model_index,
        " bins=", length(target),
        " max_rel=", max_relative_residual,
        " worst_rel_bin=", worst_relative_bin,
        " max_sigma=", sigma_diag.max_sigma_residual,
        " worst_sigma_bin=", sigma_diag.worst_sigma_bin,
        " zero_target_floor_rows=", sigma_diag.zero_target_floor_rows,
        " max_zero_target_leakage_rel=", sigma_diag.max_zero_target_leakage_rel,
        " worst_zero_target_bin=", sigma_diag.worst_zero_target_bin,
        " min_support_fraction=", min_support_fraction,
        " weakest_support_bin=", weakest_support_bin,
        " min_effective_N=", min_effective_support,
        " weakest_effective_bin=", weakest_effective_bin)

    return nothing
end




function _print_losvd_window_diagnostics(A_losvd::Matrix{Float64}, A_kinematic::Matrix{Float64}, w::Vector{Float64}, aperture_ranges::Vector{Tuple{Float64,Float64}}, velocity_edges::Vector{Float64}, Nvbin::Int; model_index::Int=1, print_bins::Bool=true)
    Nspatial = size(A_kinematic, 1)
    size(A_kinematic, 2) == length(w) || error("LOSVD window diagnostic weight length mismatch")
    size(A_losvd, 1) == Nspatial * Nvbin || error("LOSVD window diagnostic row count mismatch")
    size(A_losvd, 2) == length(w) || error("LOSVD window diagnostic orbit count mismatch")
    length(aperture_ranges) == Nspatial || error("LOSVD window diagnostic aperture-range mismatch")

    model_losvd = A_losvd * w
    projected_total = A_kinematic * w
    inside = zeros(Float64, Nspatial)
    outside = zeros(Float64, Nspatial)
    outside_fraction = zeros(Float64, Nspatial)

    @inbounds for ib in 1:Nspatial
        rows = ((ib - 1) * Nvbin + 1):(ib * Nvbin)
        inside[ib] = sum(@view model_losvd[rows])
        outside[ib] = max(projected_total[ib] - inside[ib], 0.0)
        outside_fraction[ib] = projected_total[ib] > 0.0 ? outside[ib] / projected_total[ib] : 0.0

        if print_bins
            println("[LOSVD WINDOW DIAG]",
                " i=", model_index,
                " bin=", ib,
                " R_inner_pc=", aperture_ranges[ib][1] / pc,
                " R_outer_pc=", aperture_ranges[ib][2] / pc,
                " projected_total=", projected_total[ib],
                " inside_velocity_window=", inside[ib],
                " outside_velocity_window=", outside[ib],
                " outside_fraction=", outside_fraction[ib],
            )
        end
    end

    total_projected = sum(projected_total)
    total_inside = sum(inside)
    total_outside = sum(outside)
    global_fraction = total_projected > 0.0 ? total_outside / total_projected : 0.0
    max_fraction, worst_bin = findmax(outside_fraction)

    println("[LOSVD WINDOW]",
        " i=", model_index,
        " vmin_kms=", velocity_edges[1] / 1.0e3,
        " vmax_kms=", velocity_edges[end] / 1.0e3,
        " global_outside_fraction=", global_fraction,
        " max_aperture_outside_fraction=", max_fraction,
        " worst_aperture=", worst_bin)

    return (outside_fraction_by_spatial=outside_fraction, global_outside_fraction=global_fraction, max_outside_fraction=max_fraction, worst_bin=worst_bin)
end

function _print_orbit_failure_diagnostics(st::OrbitWorkState, model_index::Int)
    nlfrac = length(st.Lfrac)
    nthird = length(st.third_launches)

    total_stage_counts = Dict{Symbol,Int}()
    total_termination_counts = Dict{Symbol,Int}()
    total_abs_drift_max = Dict{Symbol,Float64}()
    total_rel_drift_max = Dict{Symbol,Float64}()

    cell_stage_counts = Dict{Tuple{Int,Int},Dict{Symbol,Int}}()
    cell_termination_counts = Dict{Tuple{Int,Int},Dict{Symbol,Int}}()
    cell_termination_abs_max = Dict{Tuple{Int,Int,Symbol},Float64}()
    cell_termination_rel_max = Dict{Tuple{Int,Int,Symbol},Float64}()

    cell_sos_min = Dict{Tuple{Int,Int},Int}()
    cell_sos_max = Dict{Tuple{Int,Int},Int}()
    cell_integration_min = Dict{Tuple{Int,Int},Int}()
    cell_integration_max = Dict{Tuple{Int,Int},Int}()
    cell_abs_drift_max = Dict{Tuple{Int,Int},Float64}()
    cell_rel_drift_max = Dict{Tuple{Int,Int},Float64}()

    @inbounds for c in st.launch_order
        _, lfrac_id, third_id = _orbit_grid_indices(c, st.Nshells, nlfrac, nthird)

        stage = st.failure_stage[c]
        stage_reason = stage === :launch_failed ? st.launch_failure_state[c] : stage
        termination = st.integration_termination[c]
        absolute_drift = st.max_energy_drift[c]
        relative_drift = st.max_relative_energy_drift[c]
        cell = (lfrac_id, third_id)

        total_stage_counts[stage_reason] = get(total_stage_counts, stage_reason, 0) + 1
        total_termination_counts[termination] = get(total_termination_counts, termination, 0) + 1

        stage_counts = get!(cell_stage_counts, cell, Dict{Symbol,Int}())
        termination_counts = get!(cell_termination_counts, cell, Dict{Symbol,Int}())

        stage_counts[stage_reason] = get(stage_counts, stage_reason, 0) + 1
        termination_counts[termination] = get(termination_counts, termination, 0) + 1

        if isfinite(absolute_drift)
            total_abs_drift_max[termination] = max(get(total_abs_drift_max, termination, 0.0), absolute_drift)
            cell_abs_drift_max[cell] = max(get(cell_abs_drift_max, cell, 0.0), absolute_drift)

            key = (lfrac_id, third_id, termination)
            cell_termination_abs_max[key] = max(get(cell_termination_abs_max, key, 0.0), absolute_drift)
        end

        if isfinite(relative_drift)
            total_rel_drift_max[termination] = max(get(total_rel_drift_max, termination, 0.0), relative_drift)
            cell_rel_drift_max[cell] = max(get(cell_rel_drift_max, cell, 0.0), relative_drift)

            key = (lfrac_id, third_id, termination)
            cell_termination_rel_max[key] = max(get(cell_termination_rel_max, key, 0.0), relative_drift)
        end

        nsos = st.sos_points[c]
        nintegration = st.integration_points[c]

        cell_sos_min[cell] = min(get(cell_sos_min, cell, typemax(Int)), nsos)
        cell_sos_max[cell] = max(get(cell_sos_max, cell, 0), nsos)
        cell_integration_min[cell] = min(get(cell_integration_min, cell, typemax(Int)), nintegration)
        cell_integration_max[cell] = max(get(cell_integration_max, cell, 0), nintegration)
    end

    println(
        "[ORBIT FAILURE SUMMARY]",
        " i=", model_index,
        " planned=", length(st.launch_order),
        " succeeded=", count(identity, st.success_flags),
    )

    for reason in sort!(collect(keys(total_stage_counts)); by=string)
        println(
            "[ORBIT FAILURE TOTAL]",
            " i=", model_index,
            " stage=", reason,
            " count=", total_stage_counts[reason],
        )
    end

    for reason in sort!(collect(keys(total_termination_counts)); by=string)
        println(
            "[ORBIT INTEGRATION TOTAL]",
            " i=", model_index,
            " termination=", reason,
            " count=", total_termination_counts[reason],
            " max_abs_energy_drift=", get(total_abs_drift_max, reason, NaN),
            " max_rel_energy_drift=", get(total_rel_drift_max, reason, NaN),
        )
    end

    for cell in sort!(collect(keys(cell_stage_counts)))
        stage_counts = cell_stage_counts[cell]
        termination_counts = cell_termination_counts[cell]

        stage_summary = join(
            ["$(reason):$(stage_counts[reason])" for reason in sort!(collect(keys(stage_counts)); by=string)],
            ",",
        )

        termination_summary = join(
            [
                string(reason) * ":" * string(termination_counts[reason]) *
                ":max_abs=" * string(get(cell_termination_abs_max, (cell[1], cell[2], reason), NaN)) *
                ":max_rel=" * string(get(cell_termination_rel_max, (cell[1], cell[2], reason), NaN))
                for reason in sort!(collect(keys(termination_counts)); by=string)
            ],
            ",",
        )

        third_id = cell[2]

        println(
            "[ORBIT FAILURE CELL]",
            " i=", model_index,
            " lfrac_id=", cell[1],
            " third_id=", third_id,
            " third_u=", st.third_launches[third_id],
            " stages=", stage_summary,
            " terminations=", termination_summary,
            " integration_min=", cell_integration_min[cell],
            " integration_max=", cell_integration_max[cell],
            " sos_min=", cell_sos_min[cell],
            " sos_max=", cell_sos_max[cell],
            " max_abs_energy_drift=", get(cell_abs_drift_max, cell, NaN),
            " max_rel_energy_drift=", get(cell_rel_drift_max, cell, NaN),
        )
    end

    return nothing
end

function _print_orbit_weight_family_diagnostics(st::OrbitWorkState, successful_columns::Vector{Int}, w::Vector{Float64}; model_index::Int=1, inner_radius_pc::Float64=30.0)
    length(successful_columns) == length(w) || error("Orbit-family diagnostic column/weight length mismatch")
    iseven(length(w)) || error("Orbit-family diagnostic requires prograde/retrograde orbit pairs")
    all(isfinite, w) || error("Orbit-family diagnostic requires finite weights")

    Npair = length(w) ÷ 2
    nlfrac = length(st.Lfrac)
    nthird = length(st.third_launches)
    pair_weights = zeros(Float64, Npair)
    shell_ids = zeros(Int, Npair)
    lfrac_ids = zeros(Int, Npair)
    third_ids = zeros(Int, Npair)
    theta0_values = zeros(Float64, Npair)
    launch_ltot_frac_values = zeros(Float64, Npair)
    min_r_pc_values = zeros(Float64, Npair)
    inner_flags = falses(Npair)

    @inbounds for j in 1:Npair
        jpro = 2 * j - 1
        jret = 2 * j
        col_pro = successful_columns[jpro]
        col_ret = successful_columns[jret]
        isodd(col_pro) || error("Orbit-family diagnostic expected prograde column first")
        col_ret == col_pro + 1 || error("Orbit-family diagnostic orbit pair is not contiguous")
        c = (col_pro + 1) ÷ 2
        shell_id, lfrac_id, third_id = _orbit_grid_indices(c, st.Nshells, nlfrac, nthird)
        theta0 = f64(st.launch_theta0[c])
        sintheta0 = sin(theta0)
        pair_weights[j] = w[jpro] + w[jret]
        shell_ids[j] = shell_id
        lfrac_ids[j] = lfrac_id
        third_ids[j] = third_id
        theta0_values[j] = theta0
        launch_ltot_frac_values[j] = isfinite(sintheta0) && abs(sintheta0) > EPS_SIN ? f64(st.Lfrac[lfrac_id]) / abs(sintheta0) : NaN
        min_r_pc_values[j] = f64(st.min_r_reached[c]) / pc
        inner_flags[j] = st.shells[shell_id] / pc < inner_radius_pc
    end

    all(isfinite, theta0_values) || error("Orbit-family diagnostic found nonfinite launch theta")
    all(isfinite, launch_ltot_frac_values) || error("Orbit-family diagnostic found nonfinite launch total-angular-momentum ratio")
    all(isfinite, min_r_pc_values) || error("Orbit-family diagnostic found nonfinite minimum radius")

    wsum = sum(pair_weights)
    isfinite(wsum) && wsum > 0.0 || error("Orbit-family diagnostic has nonpositive total weight")
    p = pair_weights ./ wsum
    lfrac_values = [f64(st.Lfrac[idx]) for idx in lfrac_ids]
    theta0_deg_values = theta0_values .* (180.0 / pi)
    inner_sum = sum(pair_weights[inner_flags])
    outer_flags = .!inner_flags
    outer_sum = sum(pair_weights[outer_flags])
    regular_flags = lfrac_ids .< nlfrac
    regular_sum = sum(pair_weights[regular_flags])
    low_lfrac_weight = sum(pair_weights[lfrac_values .<= 0.2])
    high_lfrac_weight = sum(pair_weights[lfrac_values .>= 0.7])
    circular_weight = sum(pair_weights[lfrac_ids .== nlfrac])
    inner_high_lfrac_weight = sum(pair_weights[inner_flags .& (lfrac_values .>= 0.7)])
    inner_high_ltot_weight = sum(pair_weights[inner_flags .& (launch_ltot_frac_values .>= 0.7)])
    inner_min_r_lt_5pc_weight = sum(pair_weights[inner_flags .& (min_r_pc_values .< 5.0)])
    inner_min_r_lt_10pc_weight = sum(pair_weights[inner_flags .& (min_r_pc_values .< 10.0)])
    inner_min_r_lt_15pc_weight = sum(pair_weights[inner_flags .& (min_r_pc_values .< 15.0)])
    weighted_mean_lfrac = sum(p .* lfrac_values)
    weighted_mean_launch_ltot_frac = sum(p .* launch_ltot_frac_values)
    weighted_mean_min_r_pc = sum(p .* min_r_pc_values)
    inner_mean_lfrac = inner_sum > 0.0 ? sum(pair_weights[inner_flags] .* lfrac_values[inner_flags]) / inner_sum : NaN
    inner_mean_launch_ltot_frac = inner_sum > 0.0 ? sum(pair_weights[inner_flags] .* launch_ltot_frac_values[inner_flags]) / inner_sum : NaN
    inner_mean_min_r_pc = inner_sum > 0.0 ? sum(pair_weights[inner_flags] .* min_r_pc_values[inner_flags]) / inner_sum : NaN
    outer_mean_lfrac = outer_sum > 0.0 ? sum(pair_weights[outer_flags] .* lfrac_values[outer_flags]) / outer_sum : NaN
    regular_mean_third = regular_sum > 0.0 ? sum(pair_weights[regular_flags] .* st.third_launches[third_ids[regular_flags]]) / regular_sum : NaN
    prograde_fraction = sum(@view w[1:2:end]) / sum(w)
    retrograde_fraction = sum(@view w[2:2:end]) / sum(w)

    println("[ORBIT WEIGHT FAMILY SUMMARY]",
        " i=", model_index,
        " paired_base_orbits=", Npair,
        " weight_sum=", wsum,
        " weighted_mean_lfrac=", weighted_mean_lfrac,
        " weighted_mean_launch_ltot_frac=", weighted_mean_launch_ltot_frac,
        " weighted_mean_min_r_pc=", weighted_mean_min_r_pc,
        " low_lfrac_le_0p2_fraction=", low_lfrac_weight / wsum,
        " high_lfrac_ge_0p7_fraction=", high_lfrac_weight / wsum,
        " circular_lfrac_1_fraction=", circular_weight / wsum,
        " inner_launch_radius_pc=", inner_radius_pc,
        " inner_launch_weight_fraction=", inner_sum / wsum,
        " inner_high_lfrac_ge_0p7_fraction=", inner_sum > 0.0 ? inner_high_lfrac_weight / inner_sum : NaN,
        " inner_weighted_mean_lfrac=", inner_mean_lfrac,
        " inner_weighted_mean_launch_ltot_frac=", inner_mean_launch_ltot_frac,
        " inner_launch_ltot_ge_0p7_fraction=", inner_sum > 0.0 ? inner_high_ltot_weight / inner_sum : NaN,
        " inner_weighted_mean_min_r_pc=", inner_mean_min_r_pc,
        " inner_fraction_min_r_lt_5pc=", inner_sum > 0.0 ? inner_min_r_lt_5pc_weight / inner_sum : NaN,
        " inner_fraction_min_r_lt_10pc=", inner_sum > 0.0 ? inner_min_r_lt_10pc_weight / inner_sum : NaN,
        " inner_fraction_min_r_lt_15pc=", inner_sum > 0.0 ? inner_min_r_lt_15pc_weight / inner_sum : NaN,
        " outer_weighted_mean_lfrac=", outer_mean_lfrac,
        " regular_family_weight_fraction=", regular_sum / wsum,
        " regular_weighted_mean_third_u=", regular_mean_third,
        " prograde_fraction=", prograde_fraction,
        " retrograde_fraction=", retrograde_fraction,
    )

    @inbounds for lfrac_id in 1:nlfrac
        mask = lfrac_ids .== lfrac_id
        group_weight = sum(pair_weights[mask])
        group_inner_weight = sum(pair_weights[mask .& inner_flags])
        group_outer_weight = sum(pair_weights[mask .& outer_flags])
        group_pairs = pair_weights[mask]
        effective_pairs = group_weight > 0.0 ? 1.0 / sum(abs2, group_pairs ./ group_weight) : 0.0
        println("[ORBIT WEIGHT LFRAC]",
            " i=", model_index,
            " lfrac_id=", lfrac_id,
            " lfrac=", f64(st.Lfrac[lfrac_id]),
            " weight_fraction=", group_weight / wsum,
            " inner_fraction=", inner_sum > 0.0 ? group_inner_weight / inner_sum : NaN,
            " outer_fraction=", outer_sum > 0.0 ? group_outer_weight / outer_sum : NaN,
            " N_base_orbits=", count(identity, mask),
            " effective_N_base_orbits=", effective_pairs,
        )
    end

    regular_inner_sum = sum(pair_weights[regular_flags .& inner_flags])
    @inbounds for third_id in 1:nthird
        mask = regular_flags .& (third_ids .== third_id)
        inner_mask = mask .& inner_flags
        group_weight = sum(pair_weights[mask])
        group_inner_weight = sum(pair_weights[inner_mask])
        group_pairs = pair_weights[mask]
        effective_pairs = group_weight > 0.0 ? 1.0 / sum(abs2, group_pairs ./ group_weight) : 0.0
        mean_theta0_deg = group_weight > 0.0 ? sum(pair_weights[mask] .* theta0_deg_values[mask]) / group_weight : NaN
        mean_launch_ltot_frac = group_weight > 0.0 ? sum(pair_weights[mask] .* launch_ltot_frac_values[mask]) / group_weight : NaN
        mean_min_r_pc = group_weight > 0.0 ? sum(pair_weights[mask] .* min_r_pc_values[mask]) / group_weight : NaN
        inner_mean_theta0_deg = group_inner_weight > 0.0 ? sum(pair_weights[inner_mask] .* theta0_deg_values[inner_mask]) / group_inner_weight : NaN
        inner_mean_launch_ltot_frac = group_inner_weight > 0.0 ? sum(pair_weights[inner_mask] .* launch_ltot_frac_values[inner_mask]) / group_inner_weight : NaN
        inner_mean_min_r_pc = group_inner_weight > 0.0 ? sum(pair_weights[inner_mask] .* min_r_pc_values[inner_mask]) / group_inner_weight : NaN
        inner_min_r_lt_10pc_weight = sum(pair_weights[inner_mask .& (min_r_pc_values .< 10.0)])
        println("[ORBIT WEIGHT THIRD]",
            " i=", model_index,
            " third_id=", third_id,
            " third_u=", st.third_launches[third_id],
            " regular_weight_fraction=", regular_sum > 0.0 ? group_weight / regular_sum : NaN,
            " inner_regular_fraction=", regular_inner_sum > 0.0 ? group_inner_weight / regular_inner_sum : NaN,
            " weighted_mean_theta0_deg=", mean_theta0_deg,
            " weighted_mean_launch_ltot_frac=", mean_launch_ltot_frac,
            " weighted_mean_min_r_pc=", mean_min_r_pc,
            " inner_weighted_mean_theta0_deg=", inner_mean_theta0_deg,
            " inner_weighted_mean_launch_ltot_frac=", inner_mean_launch_ltot_frac,
            " inner_weighted_mean_min_r_pc=", inner_mean_min_r_pc,
            " inner_fraction_min_r_lt_10pc=", group_inner_weight > 0.0 ? inner_min_r_lt_10pc_weight / group_inner_weight : NaN,
            " N_base_orbits=", count(identity, mask),
            " effective_N_base_orbits=", effective_pairs,
        )
    end

    nshell_bands = min(st.shell_band_count, st.Nshells)
    @inbounds for band in 1:nshell_bands
        mask = falses(Npair)
        shell_min = Inf
        shell_max = 0.0
        for j in 1:Npair
            shell_band = fld((shell_ids[j] - 1) * nshell_bands, st.Nshells) + 1
            shell_band == band || continue
            mask[j] = true
            shell_pc = st.shells[shell_ids[j]] / pc
            shell_min = min(shell_min, shell_pc)
            shell_max = max(shell_max, shell_pc)
        end
        group_weight = sum(pair_weights[mask])
        println("[ORBIT WEIGHT SHELL]",
            " i=", model_index,
            " shell_band=", band,
            " shell_min_pc=", isfinite(shell_min) ? shell_min : NaN,
            " shell_max_pc=", shell_max > 0.0 ? shell_max : NaN,
            " weight_fraction=", group_weight / wsum,
            " N_base_orbits=", count(identity, mask),
        )
    end

    return nothing
end

function _print_orbit_projection_diagnostics(st::OrbitWorkState, successful_columns::Vector{Int}, w::Vector{Float64}, A_losvd::Matrix{Float64}, A_kinematic::Matrix{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}; model_index::Int=1, inner_radius_pc::Float64=30.0, target_lfrac::Float64=0.2, target_third_u::Float64=0.5, target_aperture::Int=1)
    st.projection_diag_enabled || return nothing
    length(successful_columns) == length(w) || error("Orbit projection diagnostic column/weight length mismatch")
    size(A_losvd, 2) == length(w) || error("Orbit projection diagnostic LOSVD column mismatch")
    size(A_kinematic, 2) == length(w) || error("Orbit projection diagnostic kinematic column mismatch")
    size(A_losvd, 1) == st.Nspatial * st.Nvbin || error("Orbit projection diagnostic LOSVD row mismatch")
    size(A_kinematic, 1) == st.Nspatial || error("Orbit projection diagnostic aperture mismatch")
    length(losvd_target) == size(A_losvd, 1) || error("Orbit projection diagnostic LOSVD target length mismatch")
    length(losvd_sigma) == size(A_losvd, 1) || error("Orbit projection diagnostic LOSVD sigma length mismatch")
    1 <= target_aperture <= st.Nspatial || error("Orbit projection diagnostic target aperture is outside the spatial grid")
    iseven(length(w)) || error("Orbit projection diagnostic requires prograde/retrograde orbit pairs")

    nlfrac = length(st.Lfrac)
    nthird = length(st.third_launches)
    lfrac_id_target = argmin(abs.(Float64.(collect(st.Lfrac)) .- target_lfrac))
    third_id_target = argmin(abs.(st.third_launches .- target_third_u))
    lfrac_value = f64(st.Lfrac[lfrac_id_target])
    third_u_value = f64(st.third_launches[third_id_target])
    abs(lfrac_value - target_lfrac) <= 1.0e-12 || error("Orbit projection target Lfrac is not present in the launch grid")
    abs(third_u_value - target_third_u) <= 1.0e-12 || error("Orbit projection target third_u is not present in the launch grid")

    family_columns = Int[]
    inner_family_columns = Int[]
    family_base_orbits = Int[]
    inner_family_base_orbits = Int[]
    @inbounds for j in 1:2:length(successful_columns)
        col_pro = successful_columns[j]
        col_ret = successful_columns[j + 1]
        isodd(col_pro) || error("Orbit projection diagnostic expected prograde column first")
        col_ret == col_pro + 1 || error("Orbit projection diagnostic orbit pair is not contiguous")
        c = (col_pro + 1) ÷ 2
        shell_id, lfrac_id, third_id = _orbit_grid_indices(c, st.Nshells, nlfrac, nthird)
        lfrac_id == lfrac_id_target || continue
        third_id == third_id_target || continue
        push!(family_columns, j)
        push!(family_columns, j + 1)
        push!(family_base_orbits, c)
        if st.shells[shell_id] / pc < inner_radius_pc
            push!(inner_family_columns, j)
            push!(inner_family_columns, j + 1)
            push!(inner_family_base_orbits, c)
        end
    end

    isempty(family_columns) && error("Orbit projection diagnostic found no successful columns for the requested family")
    isempty(inner_family_columns) && error("Orbit projection diagnostic found no inner successful columns for the requested family")

    total_weight = sum(w)
    family_weight = sum(w[family_columns])
    inner_family_weight = sum(w[inner_family_columns])
    println("[ORBIT PROJECTION FAMILY SUMMARY]",
        " i=", model_index,
        " target_lfrac=", lfrac_value,
        " target_third_u=", third_u_value,
        " inner_launch_radius_pc=", inner_radius_pc,
        " N_family_base_orbits=", length(family_base_orbits),
        " N_inner_family_base_orbits=", length(inner_family_base_orbits),
        " family_weight_fraction=", family_weight / total_weight,
        " inner_family_weight_fraction=", inner_family_weight / total_weight,
        " inner_fraction_of_family_weight=", family_weight > 0.0 ? inner_family_weight / family_weight : NaN,
    )

    velocity_centers = 0.5 .* (st.velocity_edges[1:end-1] .+ st.velocity_edges[2:end])
    threshold_kms = (10.0, 20.0, 30.0, 50.0)
    losvd_state = losvd_chi2_state(A_losvd, w, losvd_target, losvd_sigma, st.Nspatial, st.Nvbin)

    @inbounds for ib in 1:st.Nspatial
        rows = ((ib - 1) * st.Nvbin + 1):(ib * st.Nvbin)
        model_projected = dot(@view(A_kinematic[ib, :]), w)
        family_projected = dot(@view(A_kinematic[ib, inner_family_columns]), @view(w[inner_family_columns]))
        family_hist = A_losvd[rows, inner_family_columns] * w[inner_family_columns]
        family_inside = sum(family_hist)
        family_outside = max(family_projected - family_inside, 0.0)
        family_outside_fraction = family_projected > 0.0 ? family_outside / family_projected : NaN
        family_fraction_of_aperture = model_projected > 0.0 ? family_projected / model_projected : NaN

        metric_weight = 0.0
        mean_v3d = 0.0
        rms_v3d_sq = 0.0
        mean_abs_vlos = 0.0
        rms_vlos_sq = 0.0
        mean_ratio = 0.0
        frac_ratio_lt_0p25 = 0.0
        frac_ratio_lt_0p5 = 0.0
        for compact_col in inner_family_columns
            full_col = successful_columns[compact_col]
            aperture_fraction = A_kinematic[ib, compact_col]
            contribution = w[compact_col] * aperture_fraction
            contribution > 0.0 || continue
            v3d_mean = st.projection_v3d_mean[ib, full_col]
            v3d_rms = st.projection_v3d_rms[ib, full_col]
            abs_vlos_mean = st.projection_abs_vlos_mean[ib, full_col]
            vlos_rms = st.projection_vlos_rms[ib, full_col]
            ratio_mean = st.projection_abs_vlos_over_v3d_mean[ib, full_col]
            ratio_lt_0p25 = st.projection_los_ratio_lt_0p25[ib, full_col]
            ratio_lt_0p5 = st.projection_los_ratio_lt_0p5[ib, full_col]
            if all(isfinite, (v3d_mean, v3d_rms, abs_vlos_mean, vlos_rms, ratio_mean, ratio_lt_0p25, ratio_lt_0p5))
                metric_weight += contribution
                mean_v3d += contribution * v3d_mean
                rms_v3d_sq += contribution * v3d_rms * v3d_rms
                mean_abs_vlos += contribution * abs_vlos_mean
                rms_vlos_sq += contribution * vlos_rms * vlos_rms
                mean_ratio += contribution * ratio_mean
                frac_ratio_lt_0p25 += contribution * ratio_lt_0p25
                frac_ratio_lt_0p5 += contribution * ratio_lt_0p5
            end
        end
        if metric_weight > 0.0
            mean_v3d /= metric_weight
            mean_abs_vlos /= metric_weight
            mean_ratio /= metric_weight
            frac_ratio_lt_0p25 /= metric_weight
            frac_ratio_lt_0p5 /= metric_weight
            rms_v3d_sq /= metric_weight
            rms_vlos_sq /= metric_weight
        else
            mean_v3d = NaN
            mean_abs_vlos = NaN
            mean_ratio = NaN
            frac_ratio_lt_0p25 = NaN
            frac_ratio_lt_0p5 = NaN
            rms_v3d_sq = NaN
            rms_vlos_sq = NaN
        end
        rms_v3d = isfinite(rms_v3d_sq) ? sqrt(max(rms_v3d_sq, 0.0)) : NaN
        rms_vlos = isfinite(rms_vlos_sq) ? sqrt(max(rms_vlos_sq, 0.0)) : NaN

        hist_sum = sum(family_hist)
        hist_abs_mean = hist_sum > 0.0 ? sum(family_hist .* abs.(velocity_centers)) / hist_sum : NaN
        hist_rms = hist_sum > 0.0 ? sqrt(sum(family_hist .* velocity_centers .* velocity_centers) / hist_sum) : NaN
        high_fractions = Float64[]
        for threshold in threshold_kms
            threshold_mps = threshold * 1.0e3
            high_fraction = hist_sum > 0.0 ? sum(family_hist[abs.(velocity_centers) .>= threshold_mps]) / hist_sum : NaN
            push!(high_fractions, high_fraction)
        end

        println("[ORBIT PROJECTION APERTURE]",
            " i=", model_index,
            " aperture=", ib,
            " R_inner_pc=", st.spatial_edges[ib] / pc,
            " R_outer_pc=", st.spatial_edges[ib + 1] / pc,
            " family_projected_weight=", family_projected,
            " model_projected_weight=", model_projected,
            " family_fraction_of_aperture=", family_fraction_of_aperture,
            " family_inside_velocity_window=", family_inside,
            " family_outside_velocity_window=", family_outside,
            " family_outside_fraction=", family_outside_fraction,
            " mean_v3d_kms=", mean_v3d / 1.0e3,
            " rms_v3d_kms=", rms_v3d / 1.0e3,
            " mean_abs_vlos_kms=", mean_abs_vlos / 1.0e3,
            " rms_vlos_kms=", rms_vlos / 1.0e3,
            " mean_abs_vlos_over_v3d=", mean_ratio,
            " fraction_abs_vlos_over_v3d_lt_0p25=", frac_ratio_lt_0p25,
            " fraction_abs_vlos_over_v3d_lt_0p5=", frac_ratio_lt_0p5,
            " losvd_hist_mean_abs_v_kms=", hist_abs_mean / 1.0e3,
            " losvd_hist_rms_v_kms=", hist_rms / 1.0e3,
            " losvd_frac_abs_v_ge_10kms=", high_fractions[1],
            " losvd_frac_abs_v_ge_20kms=", high_fractions[2],
            " losvd_frac_abs_v_ge_30kms=", high_fractions[3],
            " losvd_frac_abs_v_ge_50kms=", high_fractions[4],
        )
    end

    rows = ((target_aperture - 1) * st.Nvbin + 1):(target_aperture * st.Nvbin)
    family_hist = A_losvd[rows, inner_family_columns] * w[inner_family_columns]
    model_hist = losvd_state.model[rows]
    raw_target = losvd_target[rows]
    raw_sigma = losvd_sigma[rows]
    effective_target = losvd_state.effective_target[rows]
    effective_sigma = losvd_state.effective_sigma[rows]
    target_sum = sum(effective_target)
    model_sum = sum(model_hist)
    family_sum = sum(family_hist)
    chi_aperture = losvd_state.chi_by_spatial[target_aperture]
    chi_without_family = 0.0
    max_abs_sigma_residual = 0.0
    max_abs_sigma_bin = 0
    println("[ORBIT PROJECTION LOSVD SUMMARY] i=", model_index, " aperture=", target_aperture, " R_inner_pc=", st.spatial_edges[target_aperture] / pc, " R_outer_pc=", st.spatial_edges[target_aperture + 1] / pc, " fracnew=", losvd_state.fracnew[target_aperture], " target_sum=", target_sum, " model_sum=", model_sum, " inner_family_sum=", family_sum, " inner_family_fraction_of_model=", model_sum > 0.0 ? family_sum / model_sum : NaN, " chi2_aperture=", chi_aperture)
    @inbounds for local_bin in 1:st.Nvbin
        row = first(rows) + local_bin - 1
        sigma_eff = max(effective_sigma[local_bin], 1.0e-12)
        residual_sigma = (model_hist[local_bin] - effective_target[local_bin]) / sigma_eff
        chi_bin = residual_sigma * residual_sigma
        model_without_family = model_hist[local_bin] - family_hist[local_bin]
        residual_without_family_sigma = (model_without_family - effective_target[local_bin]) / sigma_eff
        chi_without_bin = residual_without_family_sigma * residual_without_family_sigma
        chi_without_family += chi_without_bin
        abs_residual_sigma = abs(residual_sigma)
        if abs_residual_sigma > max_abs_sigma_residual
            max_abs_sigma_residual = abs_residual_sigma
            max_abs_sigma_bin = local_bin
        end
        target_shape_fraction = target_sum > 0.0 ? effective_target[local_bin] / target_sum : NaN
        model_shape_fraction = model_sum > 0.0 ? model_hist[local_bin] / model_sum : NaN
        family_shape_fraction = family_sum > 0.0 ? family_hist[local_bin] / family_sum : NaN
        family_fraction_of_model_bin = model_hist[local_bin] > 0.0 ? family_hist[local_bin] / model_hist[local_bin] : NaN
        sigma_is_invalid = raw_sigma[local_bin] == DEFAULT_INVALID_SIGMA_SENTINEL
        println("[ORBIT PROJECTION LOSVD BIN] i=", model_index, " aperture=", target_aperture, " velocity_bin=", local_bin, " v_inner_kms=", st.velocity_edges[local_bin] / 1.0e3, " v_outer_kms=", st.velocity_edges[local_bin + 1] / 1.0e3, " v_center_kms=", velocity_centers[local_bin] / 1.0e3, " raw_target=", raw_target[local_bin], " raw_sigma=", raw_sigma[local_bin], " sigma_is_invalid=", sigma_is_invalid, " effective_target=", effective_target[local_bin], " effective_sigma=", effective_sigma[local_bin], " total_model=", model_hist[local_bin], " inner_family_model=", family_hist[local_bin], " model_without_inner_family=", model_without_family, " target_shape_fraction=", target_shape_fraction, " model_shape_fraction=", model_shape_fraction, " inner_family_shape_fraction=", family_shape_fraction, " inner_family_fraction_of_model_bin=", family_fraction_of_model_bin, " residual_sigma=", residual_sigma, " chi2_bin=", chi_bin, " chi2_without_inner_family=", chi_without_bin, " delta_chi2_inner_family=", chi_bin - chi_without_bin)
    end
    println("[ORBIT PROJECTION LOSVD CHI SUMMARY] i=", model_index, " aperture=", target_aperture, " chi2_with_inner_family=", chi_aperture, " chi2_without_inner_family_fixed_other_weights=", chi_without_family, " delta_chi2_inner_family=", chi_aperture - chi_without_family, " max_abs_sigma_residual=", max_abs_sigma_residual, " max_abs_sigma_bin=", max_abs_sigma_bin)

    return nothing
end

function _print_orbit_family_multinomial_diagnostics(
    st::OrbitWorkState,
    successful_columns::Vector{Int},
    w::Vector{Float64},
    A_losvd::Matrix{Float64},
    A_kinematic::Matrix{Float64},
    losvd_counts::Vector{Int};
    conditioning=:vlos_cut,
    model_index::Int=1,
    target_lfrac::Float64=0.2,
    target_third_u::Float64=0.5,
    target_aperture::Int=1,
)
    st.projection_diag_enabled || return nothing

    length(successful_columns) == length(w) ||
        error("Orbit family multinomial diagnostic column/weight length mismatch")
    size(A_losvd, 2) == length(w) ||
        error("Orbit family multinomial diagnostic LOSVD column mismatch")
    size(A_kinematic, 2) == length(w) ||
        error("Orbit family multinomial diagnostic kinematic column mismatch")
    length(losvd_counts) == size(A_losvd, 1) ||
        error("Orbit family multinomial diagnostic count length mismatch")
    1 <= target_aperture <= st.Nspatial ||
        error("Orbit family multinomial diagnostic target aperture is outside spatial grid")
    iseven(length(w)) ||
        error("Orbit family multinomial diagnostic requires prograde/retrograde pairs")

    nlfrac = length(st.Lfrac)
    nthird = length(st.third_launches)

    lfrac_id_target = argmin(abs.(Float64.(collect(st.Lfrac)) .- target_lfrac))
    third_id_target = argmin(abs.(st.third_launches .- target_third_u))
    lfrac_value = f64(st.Lfrac[lfrac_id_target])
    third_u_value = f64(st.third_launches[third_id_target])

    abs(lfrac_value - target_lfrac) <= 1.0e-12 ||
        error("Requested Lfrac is not present in launch grid")
    abs(third_u_value - target_third_u) <= 1.0e-12 ||
        error("Requested third_u is not present in launch grid")

    # Select the ENTIRE requested family across every successful launch shell.
    # This intentionally differs from _print_orbit_projection_diagnostics,
    # which also constructs an inner-launch subset for its legacy diagnostic.
    family_columns = Int[]
    family_base_orbits = Int[]

    @inbounds for j in 1:2:length(successful_columns)
        col_pro = successful_columns[j]
        col_ret = successful_columns[j + 1]

        isodd(col_pro) || error("Expected prograde column first")
        col_ret == col_pro + 1 || error("Orbit pair is not contiguous")

        c = (col_pro + 1) ÷ 2
        shell_id, lfrac_id, third_id =
            _orbit_grid_indices(c, st.Nshells, nlfrac, nthird)

        lfrac_id == lfrac_id_target || continue
        third_id == third_id_target || continue

        push!(family_columns, j)
        push!(family_columns, j + 1)
        push!(family_base_orbits, c)
    end

    isempty(family_columns) &&
        error("No successful columns found for requested orbit family")

    # Production statistic using the fitted orbit weights.
    state_with = multinomial_losvd_state(
        A_losvd,
        A_kinematic,
        w,
        losvd_counts,
        st.Nspatial,
        st.Nvbin;
        conditioning=conditioning,
    )

    # Diagnostic counterfactual only: remove this family while freezing all
    # other fitted weights. This is deliberately NOT a re-fit.
    w_without = copy(w)
    w_without[family_columns] .= 0.0

    state_without = try
        multinomial_losvd_state(
            A_losvd,
            A_kinematic,
            w_without,
            losvd_counts,
            st.Nspatial,
            st.Nvbin;
            conditioning=conditioning,
        )
    catch err
        println(
            "[ORBIT FAMILY MULTINOMIAL ERROR]",
            " i=", model_index,
            " target_lfrac=", lfrac_value,
            " target_third_u=", third_u_value,
            " error=", sprint(showerror, err),
        )
        return nothing
    end

    total_weight = sum(w)
    family_weight = sum(w[family_columns])

    println(
        "[ORBIT FAMILY MULTINOMIAL SUMMARY]",
        " i=", model_index,
        " target_lfrac=", lfrac_value,
        " target_third_u=", third_u_value,
        " N_family_base_orbits=", length(family_base_orbits),
        " family_weight_fraction=", family_weight / total_weight,
        " deviance_with=", state_with.deviance_total,
        " deviance_without_fixed_other_weights=", state_without.deviance_total,
        " delta_deviance_remove_family=", state_without.deviance_total - state_with.deviance_total,
    )

    # Positive delta => this family helps the fit.
    # Negative delta => removing this family improves the fit.
    @inbounds for ib in 1:st.Nspatial
        rows = ((ib - 1) * st.Nvbin + 1):(ib * st.Nvbin)

        family_projected = dot(
            @view(A_kinematic[ib, family_columns]),
            @view(w[family_columns]),
        )
        model_projected = dot(@view(A_kinematic[ib, :]), w)
        family_selected = sum(A_losvd[rows, family_columns] * w[family_columns])
        model_selected = state_with.selected_total[ib]

        family_fraction_projected =
            model_projected > 0.0 ? family_projected / model_projected : NaN
        family_fraction_selected =
            model_selected > 0.0 ? family_selected / model_selected : NaN

        d_with = state_with.deviance_by_spatial[ib]
        d_without = state_without.deviance_by_spatial[ib]

        println(
            "[ORBIT FAMILY MULTINOMIAL APERTURE]",
            " i=", model_index,
            " aperture=", ib,
            " R_inner_pc=", st.spatial_edges[ib] / pc,
            " R_outer_pc=", st.spatial_edges[ib + 1] / pc,
            " Nstars=", state_with.counts_by_spatial[ib],
            " family_fraction_projected=", family_fraction_projected,
            " family_fraction_selected=", family_fraction_selected,
            " deviance_with=", d_with,
            " deviance_without_fixed_other_weights=", d_without,
            " delta_deviance_remove_family=", d_without - d_with,
        )
    end

    # Detailed production-probability comparison for one selected aperture.
    rows = ((target_aperture - 1) * st.Nvbin + 1):(target_aperture * st.Nvbin)
    family_hist = A_losvd[rows, family_columns] * w[family_columns]

    println(
        "[ORBIT FAMILY MULTINOMIAL TARGET]",
        " i=", model_index,
        " aperture=", target_aperture,
        " R_inner_pc=", st.spatial_edges[target_aperture] / pc,
        " R_outer_pc=", st.spatial_edges[target_aperture + 1] / pc,
        " Nstars=", state_with.counts_by_spatial[target_aperture],
        " deviance_with=", state_with.deviance_by_spatial[target_aperture],
        " deviance_without_fixed_other_weights=", state_without.deviance_by_spatial[target_aperture],
        " delta_deviance_remove_family=", state_without.deviance_by_spatial[target_aperture] - state_with.deviance_by_spatial[target_aperture],
    )

    @inbounds for local_bin in 1:st.Nvbin
        row = first(rows) + local_bin - 1

        println(
            "[ORBIT FAMILY MULTINOMIAL BIN]",
            " i=", model_index,
            " aperture=", target_aperture,
            " velocity_bin=", local_bin,
            " v_inner_kms=", st.velocity_edges[local_bin] / 1.0e3,
            " v_outer_kms=", st.velocity_edges[local_bin + 1] / 1.0e3,
            " observed_count=", losvd_counts[row],
            " probability_with=", state_with.probabilities[row],
            " probability_without=", state_without.probabilities[row],
            " model_mass_with=", state_with.model[row],
            " model_mass_without=", state_without.model[row],
            " family_model_mass=", family_hist[local_bin],
        )
    end

    return nothing
end

function _print_orbit_phase_volume_diagnostics(st::OrbitWorkState, successful_columns::Vector{Int}, w::Vector{Float64}, wphase_use::Vector{Float64}, phase_result; model_index::Int=1, inner_radius_pc::Float64=30.0, topn::Int=10)
    length(successful_columns) == length(w) || error("Orbit phase-volume diagnostic column/weight length mismatch")
    length(wphase_use) == length(w) || error("Orbit phase-volume diagnostic wphase/weight length mismatch")
    iseven(length(w)) || error("Orbit phase-volume diagnostic requires prograde/retrograde orbit pairs")
    topn > 0 || error("Orbit phase-volume diagnostic topn must be positive")
    all(isfinite, w) || error("Orbit phase-volume diagnostic requires finite weights")
    all(>(0.0), w) || error("Orbit phase-volume diagnostic requires positive orbit weights")
    all(isfinite, wphase_use) || error("Orbit phase-volume diagnostic requires finite wphase")
    all(>(0.0), wphase_use) || error("Orbit phase-volume diagnostic requires positive wphase")

    Npair = length(w) ÷ 2
    nlfrac = length(st.Lfrac)
    nthird = length(st.third_launches)

    base_indices = zeros(Int, Npair)
    shell_ids = zeros(Int, Npair)
    lfrac_ids = zeros(Int, Npair)
    third_ids = zeros(Int, Npair)
    pair_weights = zeros(Float64, Npair)
    pair_phase_volume = zeros(Float64, Npair)
    theta0_deg_values = zeros(Float64, Npair)
    launch_ltot_frac_values = zeros(Float64, Npair)
    min_r_pc_values = zeros(Float64, Npair)
    inner_flags = falses(Npair)

    @inbounds for j in 1:Npair
        jpro = 2 * j - 1
        jret = 2 * j
        col_pro = successful_columns[jpro]
        col_ret = successful_columns[jret]
        isodd(col_pro) || error("Orbit phase-volume diagnostic expected prograde column first")
        col_ret == col_pro + 1 || error("Orbit phase-volume diagnostic orbit pair is not contiguous")
        c = (col_pro + 1) ÷ 2
        shell_id, lfrac_id, third_id = _orbit_grid_indices(c, st.Nshells, nlfrac, nthird)
        pv = f64(phase_result.phase_volume_base[c])
        isfinite(pv) && pv > 0.0 || error("Orbit phase-volume diagnostic found invalid phase volume at base orbit $c")
        theta0 = f64(st.launch_theta0[c])
        sintheta0 = sin(theta0)

        base_indices[j] = c
        shell_ids[j] = shell_id
        lfrac_ids[j] = lfrac_id
        third_ids[j] = third_id
        pair_weights[j] = w[jpro] + w[jret]
        pair_phase_volume[j] = pv
        theta0_deg_values[j] = theta0 * (180.0 / pi)
        launch_ltot_frac_values[j] = isfinite(sintheta0) && abs(sintheta0) > EPS_SIN ? f64(st.Lfrac[lfrac_id]) / abs(sintheta0) : NaN
        min_r_pc_values[j] = f64(st.min_r_reached[c]) / pc
        inner_flags[j] = st.shells[shell_id] / pc < inner_radius_pc
    end

    all(isfinite, theta0_deg_values) || error("Orbit phase-volume diagnostic found nonfinite launch theta")
    all(isfinite, launch_ltot_frac_values) || error("Orbit phase-volume diagnostic found nonfinite launch total-angular-momentum ratio")
    all(isfinite, min_r_pc_values) || error("Orbit phase-volume diagnostic found nonfinite minimum radius")

    phase_volume_columns = Float64.(phase_result.phase_volume_paired[successful_columns])
    all(isfinite, phase_volume_columns) || error("Orbit phase-volume diagnostic found nonfinite compact phase-volume columns")
    all(>(0.0), phase_volume_columns) || error("Orbit phase-volume diagnostic found non-positive compact phase-volume columns")

    reciprocal_error = maximum(abs.(phase_volume_columns .* wphase_use .- 1.0))
    wsum = sum(w)
    isfinite(wsum) && wsum > 0.0 || error("Orbit phase-volume diagnostic has nonpositive total fitted weight")
    phase_column_sum = sum(phase_volume_columns)
    isfinite(phase_column_sum) && phase_column_sum > 0.0 || error("Orbit phase-volume diagnostic has nonpositive total phase volume")

    wnorm = w ./ wsum
    phase_prior_columns = phase_volume_columns ./ phase_column_sum
    column_kl_terms = wnorm .* log.(wnorm ./ phase_prior_columns)
    column_kl = sum(column_kl_terms)
    entropy_direct = entropy_value(wnorm, wphase_use)
    entropy_reconstructed = log(phase_column_sum) - column_kl
    entropy_reconstruction_error = entropy_direct - entropy_reconstructed

    pair_weight_sum = sum(pair_weights)
    pair_phase_sum = sum(pair_phase_volume)
    pair_weight_share = pair_weights ./ pair_weight_sum
    pair_phase_share = pair_phase_volume ./ pair_phase_sum
    pair_enhancement = pair_weight_share ./ pair_phase_share
    pair_kl_contribution = pair_weight_share .* log.(pair_enhancement)
    pair_kl = sum(pair_kl_contribution)
    pair_weight_neff = 1.0 / sum(abs2, pair_weight_share)
    pair_phase_neff = 1.0 / sum(abs2, pair_phase_share)
    max_pair_enhancement, max_pair_enhancement_idx = findmax(pair_enhancement)

    println("[ORBIT PHASE ENTROPY SUMMARY]",
        " i=", model_index,
        " paired_base_orbits=", Npair,
        " fitted_weight_sum=", wsum,
        " compact_phase_volume_sum=", phase_column_sum,
        " reciprocal_alignment_max_abs=", reciprocal_error,
        " entropy_direct_normalized=", entropy_direct,
        " entropy_reconstructed=", entropy_reconstructed,
        " entropy_reconstruction_error=", entropy_reconstruction_error,
        " entropy_normalization_free=", -column_kl,
        " KL_column_to_phase_prior=", column_kl,
        " KL_pair_to_phase_prior=", pair_kl,
        " KL_rotation_excess=", column_kl - pair_kl,
        " fitted_effective_N_pairs=", pair_weight_neff,
        " phase_prior_effective_N_pairs=", pair_phase_neff,
        " max_pair_weight_to_phase_ratio=", max_pair_enhancement,
        " max_pair_weight_to_phase_base_orbit=", base_indices[max_pair_enhancement_idx],
    )

    inner_weight_sum = sum(pair_weights[inner_flags])
    inner_phase_sum = sum(pair_phase_volume[inner_flags])
    regular_flags = lfrac_ids .< nlfrac
    regular_weight_sum = sum(pair_weights[regular_flags])
    regular_phase_sum = sum(pair_phase_volume[regular_flags])
    regular_inner_flags = regular_flags .& inner_flags
    regular_inner_weight_sum = sum(pair_weights[regular_inner_flags])
    regular_inner_phase_sum = sum(pair_phase_volume[regular_inner_flags])

    @inbounds for lfrac_id in 1:nlfrac
        mask = lfrac_ids .== lfrac_id
        inner_mask = mask .& inner_flags
        group_weight = sum(pair_weights[mask])
        group_phase = sum(pair_phase_volume[mask])
        group_weight_fraction = group_weight / pair_weight_sum
        group_phase_fraction = group_phase / pair_phase_sum
        group_inner_weight = sum(pair_weights[inner_mask])
        group_inner_phase = sum(pair_phase_volume[inner_mask])
        inner_weight_fraction = inner_weight_sum > 0.0 ? group_inner_weight / inner_weight_sum : NaN
        inner_phase_fraction = inner_phase_sum > 0.0 ? group_inner_phase / inner_phase_sum : NaN
        println("[ORBIT PHASE LFRAC]",
            " i=", model_index,
            " lfrac_id=", lfrac_id,
            " lfrac=", f64(st.Lfrac[lfrac_id]),
            " weight_fraction=", group_weight_fraction,
            " phase_volume_fraction=", group_phase_fraction,
            " weight_to_phase_ratio=", group_phase_fraction > 0.0 ? group_weight_fraction / group_phase_fraction : NaN,
            " inner_weight_fraction=", inner_weight_fraction,
            " inner_phase_volume_fraction=", inner_phase_fraction,
            " inner_weight_to_phase_ratio=", isfinite(inner_phase_fraction) && inner_phase_fraction > 0.0 ? inner_weight_fraction / inner_phase_fraction : NaN,
            " N_base_orbits=", count(identity, mask),
        )
    end

    @inbounds for third_id in 1:nthird
        mask = regular_flags .& (third_ids .== third_id)
        inner_mask = mask .& inner_flags
        group_weight = sum(pair_weights[mask])
        group_phase = sum(pair_phase_volume[mask])
        group_inner_weight = sum(pair_weights[inner_mask])
        group_inner_phase = sum(pair_phase_volume[inner_mask])
        regular_weight_fraction = regular_weight_sum > 0.0 ? group_weight / regular_weight_sum : NaN
        regular_phase_fraction = regular_phase_sum > 0.0 ? group_phase / regular_phase_sum : NaN
        inner_regular_weight_fraction = regular_inner_weight_sum > 0.0 ? group_inner_weight / regular_inner_weight_sum : NaN
        inner_regular_phase_fraction = regular_inner_phase_sum > 0.0 ? group_inner_phase / regular_inner_phase_sum : NaN
        println("[ORBIT PHASE THIRD]",
            " i=", model_index,
            " third_id=", third_id,
            " third_u=", st.third_launches[third_id],
            " regular_weight_fraction=", regular_weight_fraction,
            " regular_phase_volume_fraction=", regular_phase_fraction,
            " regular_weight_to_phase_ratio=", isfinite(regular_phase_fraction) && regular_phase_fraction > 0.0 ? regular_weight_fraction / regular_phase_fraction : NaN,
            " inner_regular_weight_fraction=", inner_regular_weight_fraction,
            " inner_regular_phase_volume_fraction=", inner_regular_phase_fraction,
            " inner_regular_weight_to_phase_ratio=", isfinite(inner_regular_phase_fraction) && inner_regular_phase_fraction > 0.0 ? inner_regular_weight_fraction / inner_regular_phase_fraction : NaN,
            " N_base_orbits=", count(identity, mask),
        )
    end

    nshell_bands = min(st.shell_band_count, st.Nshells)
    @inbounds for band in 1:nshell_bands
        mask = falses(Npair)
        shell_min = Inf
        shell_max = 0.0
        for j in 1:Npair
            shell_band = fld((shell_ids[j] - 1) * nshell_bands, st.Nshells) + 1
            shell_band == band || continue
            mask[j] = true
            shell_pc = st.shells[shell_ids[j]] / pc
            shell_min = min(shell_min, shell_pc)
            shell_max = max(shell_max, shell_pc)
        end
        group_weight_fraction = sum(pair_weights[mask]) / pair_weight_sum
        group_phase_fraction = sum(pair_phase_volume[mask]) / pair_phase_sum
        println("[ORBIT PHASE SHELL]",
            " i=", model_index,
            " shell_band=", band,
            " shell_min_pc=", isfinite(shell_min) ? shell_min : NaN,
            " shell_max_pc=", shell_max > 0.0 ? shell_max : NaN,
            " weight_fraction=", group_weight_fraction,
            " phase_volume_fraction=", group_phase_fraction,
            " weight_to_phase_ratio=", group_phase_fraction > 0.0 ? group_weight_fraction / group_phase_fraction : NaN,
            " N_base_orbits=", count(identity, mask),
        )
    end

    function print_phase_orbit_row(tag::String, rank::Int, j::Int)
        c = base_indices[j]
        println(tag,
            " i=", model_index,
            " rank=", rank,
            " base_orbit=", c,
            " shell_id=", shell_ids[j],
            " shell_pc=", st.shells[shell_ids[j]] / pc,
            " lfrac_id=", lfrac_ids[j],
            " lfrac=", f64(st.Lfrac[lfrac_ids[j]]),
            " third_id=", third_ids[j],
            " third_u=", st.third_launches[third_ids[j]],
            " weight_fraction=", pair_weight_share[j],
            " phase_volume_fraction=", pair_phase_share[j],
            " weight_to_phase_ratio=", pair_enhancement[j],
            " KL_pair_contribution=", pair_kl_contribution[j],
            " normalized_phase_volume=", pair_phase_volume[j],
            " raw_phase_volume=", phase_result.raw_phase_volume_base[c],
            " sos_area=", phase_result.sos_area[c],
            " delta_sos_area=", phase_result.delta_sos_area[c],
            " dE=", phase_result.dE[c],
            " dLz=", phase_result.dLz[c],
            " energy=", st.phase_volume_state.energy[c],
            " lz_abs=", st.phase_volume_state.lz_abs[c],
            " launch_r0_pc=", st.launch_r0[c] / pc,
            " launch_theta0_deg=", theta0_deg_values[j],
            " launch_ltot_frac=", launch_ltot_frac_values[j],
            " min_r_pc=", min_r_pc_values[j],
            " sos_points=", st.sos_points[c],
        )
    end

    top_count = min(topn, Npair)
    top_kl_order = sortperm(pair_kl_contribution; rev=true)
    @inbounds for rank in 1:top_count
        print_phase_orbit_row("[ORBIT PHASE TOP KL]", rank, top_kl_order[rank])
    end

    u05_id = argmin(abs.(st.third_launches .- 0.5))
    if abs(st.third_launches[u05_id] - 0.5) <= 1.0e-12
        u05_indices = findall(regular_flags .& inner_flags .& (third_ids .== u05_id))
        sort!(u05_indices; by=j -> pair_weights[j], rev=true)
        u05_count = min(topn, length(u05_indices))
        @inbounds for rank in 1:u05_count
            print_phase_orbit_row("[ORBIT PHASE INNER U0P5]", rank, u05_indices[rank])
        end
    end

    return nothing
end

function _print_phase_volume_diagnostics!(i::Int, phase_diag)
    phase_diag === nothing && return nothing
    println("[PHASE VOLUME]",
        " i=", i,
        " convention=", phase_diag.convention,
        " normalization=", phase_diag.normalization,
        " valid_base_orbits=", phase_diag.valid_base_orbits,
        " raw_dynamic_range=", phase_diag.raw_phase_volume_dynamic_range,
        " normalized_min=", phase_diag.normalized_phase_volume_min,
        " normalized_max=", phase_diag.normalized_phase_volume_max)
    if get(ENV, "OSPM_DIAG_PHASE_VOLUME", "0") == "1"
        println("[PHASE VOLUME DETAIL]",
            " i=", i,
            " launches_recorded=", phase_diag.launches_recorded,
            " sos_recorded=", phase_diag.sos_recorded,
            " nested_groups=", phase_diag.nested_groups,
            " duplicate_area_clusters=", phase_diag.duplicate_area_clusters,
            " duplicate_area_orbits=", phase_diag.duplicate_area_orbits,
            " raw_min=", phase_diag.raw_phase_volume_min,
            " raw_max=", phase_diag.raw_phase_volume_max,
            " wphase_min=", phase_diag.wphase_min,
            " wphase_max=", phase_diag.wphase_max)
    end
    return nothing
end
