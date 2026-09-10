# ========================================================================================================================
# ========================================================================================================================
module OSPMPhysicsSpherical
# This file is becoming very large and may need to be split into support and weights
@info "OSPMPhysicsSpherical Karl-style loaded from" @__FILE__
using LinearAlgebra, StaticArrays, Statistics, Random, Base.Threads, Optim
export build_R_halo_physical, halo_from_theta, tables_spherical, make_potential_force_funcs, integrate_orbit_rk4, build_A_matrix_hybrid, mass_enclosed_two_radii, evaluate_batch_theta, NTHREADS, force_at_rtheta
include("OSPM_Physics_Support.jl")
include("OSPM_Physics_PhaseVolume.jl")
@info "OSPMPhysicsSpherical supports spherical frc(r,theta)->(fr,0) and axisymmetric frc(r,theta)->(fr,ftheta)"
const DEFAULT_ORBIT_FILL_PCT = 0.85
const DEFAULT_ORBIT_REGIONAL_FLOOR = 0.80
const DEFAULT_ORBIT_MAX_REGIONAL_GAP = 0.25 
const DEFAULT_ORBIT_SHELL_BANDS = 8
const DEFAULT_ORBIT_COVERAGE_CHECK_EVERY = 50
const DEFAULT_ORBIT_WARN_FILL_PCT = 0.95
const DEFAULT_ORBIT_WARN_SUCCESS_PCT = 0.99
const DEFAULT_ORBIT_WARN_REGIONAL_FLOOR = 0.80
const DEFAULT_ORBIT_WARN_MAX_REGIONAL_GAP = 0.15

@inline function _normalize_tracer_constraint_mode(mode)
    mode_sym = Symbol(lowercase(String(mode)))
    mode_sym in (:projected_light, :density_3d) || error("Unknown tracer_constraint_mode=$(mode). Use projected_light or density_3d")
    return mode_sym
end

@inline function _normalize_losvd_target_mode(mode)
    mode_sym = Symbol(lowercase(String(mode)))
    mode_sym in (:current, :karl_resolved_stars, :karl_mode0_observables) || error("Unknown losvd_target_mode=$(mode). Use current, karl_resolved_stars, or karl_mode0_observables")
    return mode_sym
end

function _resolve_karl_observables(mode::Symbol; observables_csv=nothing)
    mode === :karl_mode0_observables || return nothing
    observables_csv === nothing && error("karl_observables_csv is required for losvd_target_mode=:karl_mode0_observables")
    path = String(observables_csv)
    isfile(path) || error("Karl observables CSV not found: $path")
    return load_karl_observables(path)
end

function _observed_targets_for_mode(R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, st, surface_brightness_profile, losvd_target_mode::Symbol, karl_observables; karl_resolved_kde_grid::Int=DEFAULT_KARL_RESOLVED_KDE_GRID, karl_resolved_kde_width_bins::Float64=DEFAULT_KARL_RESOLVED_KDE_WIDTH_BINS, karl_resolved_vmin_kms::Float64=DEFAULT_KARL_RESOLVED_VMIN_KMS, karl_resolved_vmax_kms::Float64=DEFAULT_KARL_RESOLVED_VMAX_KMS, karl_resolved_bootstraps::Int=DEFAULT_KARL_RESOLVED_BOOTSTRAPS, karl_resolved_envelope_floor::Float64=DEFAULT_KARL_RESOLVED_ENVELOPE_FLOOR)
    if losvd_target_mode === :karl_mode0_observables
        karl_observables === nothing && error("Karl observables runtime state is missing")
        targets = karl_observables_targets(karl_observables)
        length(targets.losvd_target) == st.Nlosvd || error("Karl observables target length does not match the orbit LOSVD matrix")
        length(targets.losvd_sigma) == st.Nlosvd || error("Karl observables sigma length does not match the orbit LOSVD matrix")
        length(karl_observables.star_count) == st.Nspatial || error("Karl observables star-count length does not match the orbit aperture count")
        maximum(abs.(targets.velocity_edges_mps .- st.velocity_edges)) <= 1.0e-9 || error("Karl observables target and orbit velocity grids do not match")
        projected_light_target = light_target_from_surface_brightness(surface_brightness_profile, st.light_edges; normalize=true)
        projected_light_sigma = light_sigma_from_surface_brightness(surface_brightness_profile, st.light_edges; normalize=true, sigma_floor=1.0e-8)
        counts_by_spatial = Float64.(karl_observables.star_count)
        return copy(targets.losvd_target), copy(targets.losvd_sigma), projected_light_target, projected_light_sigma, counts_by_spatial, nothing
    end
    if losvd_target_mode === :karl_resolved_stars
        return observed_targets_karl(R_star_m, valid_vlos, v_star_mps, verr_star_mps, st.spatial_edges, st.velocity_edges; surface_brightness_profile=surface_brightness_profile, light_edges=st.light_edges, target_mode=losvd_target_mode, karl_resolved_kde_grid=karl_resolved_kde_grid, karl_resolved_kde_width_bins=karl_resolved_kde_width_bins, karl_resolved_vmin_kms=karl_resolved_vmin_kms, karl_resolved_vmax_kms=karl_resolved_vmax_kms, karl_resolved_bootstraps=karl_resolved_bootstraps, karl_resolved_envelope_floor=karl_resolved_envelope_floor, return_counts=true)
    end
    losvd_target, losvd_sigma, projected_light_target, projected_light_sigma, counts_by_spatial = observed_targets_karl(R_star_m, valid_vlos, v_star_mps, verr_star_mps, st.spatial_edges, st.velocity_edges; surface_brightness_profile=surface_brightness_profile, light_edges=st.light_edges, target_mode=losvd_target_mode, karl_resolved_kde_grid=karl_resolved_kde_grid, karl_resolved_kde_width_bins=karl_resolved_kde_width_bins, karl_resolved_vmin_kms=karl_resolved_vmin_kms, karl_resolved_vmax_kms=karl_resolved_vmax_kms, karl_resolved_bootstraps=karl_resolved_bootstraps, karl_resolved_envelope_floor=karl_resolved_envelope_floor)
    return losvd_target, losvd_sigma, projected_light_target, projected_light_sigma, counts_by_spatial, nothing
end

function _resolve_tracer_constraint_targets(mode::Symbol, projected_target::Vector{Float64}, projected_sigma::Vector{Float64}, d3_constraints)
    if mode === :projected_light
        return projected_target, projected_sigma
    end

    mode === :density_3d || error("Unsupported tracer constraint mode: $mode")
    d3_constraints === nothing && error("density_3d tracer constraints were not initialized")
    length(projected_target) == length(projected_sigma) || error("Projected tracer target and sigma lengths do not match")

    density_target = copy(d3_constraints.target)
    projected_sigma_safe = max.(abs.(projected_sigma), 1.0e-12)
    fractional_sigma = projected_sigma_safe ./ max.(abs.(projected_target), 1.0e-12)
    finite_fractional_sigma = fractional_sigma[isfinite.(fractional_sigma) .& (fractional_sigma .> 0.0)]
    fallback_fractional_sigma = isempty(finite_fractional_sigma) ? 1.0 : median(finite_fractional_sigma)

    density_sigma = similar(density_target)

    if length(projected_target) == d3_constraints.nradial
        @inbounds for row in eachindex(density_target)
            ir = d3_constraints.row_radial[row]
            frac_sigma = isfinite(fractional_sigma[ir]) && fractional_sigma[ir] > 0.0 ? fractional_sigma[ir] : fallback_fractional_sigma
            density_sigma[row] = max(abs(density_target[row]) * frac_sigma, 1.0e-12)
        end
    else
        @inbounds for row in eachindex(density_target)
            density_sigma[row] = max(abs(density_target[row]) * fallback_fractional_sigma, 1.0e-12)
        end
    end

    return density_target, density_sigma
end

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

# Work state
mutable struct OrbitWorkState
    Norbit::Int
    Nbase_orbit::Int
    Nstar::Int
    Nspatial::Int
    Nvbin::Int
    Nlosvd::Int
    Nlight::Int
    Nshells::Int
    nsteps::Int
    max_attempts_factor::Int

    third_launches::Vector{Float64}
    launch_r0::Vector{Float64}
    launch_theta0::Vector{Float64}
    launch_energy::Vector{Float64}
    launch_lz::Vector{Float64}

    sini::Float64
    cosi::Float64
    R_star_m::Vector{Float64}
    valid_vlos::Vector{Bool}
    v_star_mps::Vector{Float64}
    verr_star_mps::Vector{Float64}

    spatial_edges::Vector{Float64}
    light_edges::Vector{Float64}
    velocity_edges::Vector{Float64}
    shells::Vector{Float64}
    launch_order::Vector{Int}

    orbit_ctx
    pot
    frc
    Lfrac
    force_geometry::Symbol
    tracer_constraint_mode::Symbol
    d3_constraints
    losvd_target_mode::Symbol
    losvd_aperture_ranges::Vector{Tuple{Float64,Float64}}
    karl_observables

    dt_frac_orbit::Float64
    t_deadline::UInt64
    fill_pct::Float64
    regional_floor::Float64
    max_regional_gap::Float64
    shell_band_count::Int
    coverage_check_every::Int
    next_coverage_check::Threads.Atomic{Int}

    A_losvd::Matrix{Float64}
    A_kinematic::Matrix{Float64}
    A_light::Matrix{Float64}

    projection_diag_enabled::Bool
    projection_v3d_mean::Matrix{Float64}
    projection_v3d_rms::Matrix{Float64}
    projection_abs_vlos_mean::Matrix{Float64}
    projection_vlos_rms::Matrix{Float64}
    projection_abs_vlos_over_v3d_mean::Matrix{Float64}
    projection_los_ratio_lt_0p25::Matrix{Float64}
    projection_los_ratio_lt_0p5::Matrix{Float64}

    success_flags::Vector{Bool}
    attempts_used::Vector{Int}
    min_r_reached::Vector{Float64}
    rapo_list::Vector{Float64}
    failure_stage::Vector{Symbol}
    launch_failure_state::Vector{Symbol}
    sos_points::Vector{Int}
    integration_points::Vector{Int}
    integration_termination::Vector{Symbol}

    initial_energy_diag::Vector{Float64}
    final_energy_diag::Vector{Float64}
    max_energy_drift::Vector{Float64}
    max_relative_energy_drift::Vector{Float64}

    phase_volume_state::KarlPhaseVolumeState
    next_orbit::Threads.Atomic{Int}
    filled_atomic::Threads.Atomic{Int}
    phase::Threads.Atomic{Int}
    active_workers::Threads.Atomic{Int}
    worker_gate::ReentrantLock
end

@inline function _orbit_grid_indices(base_index::Int, Nshells::Int, nlfrac::Int, nthird::Int)
    base_index > 0 || error("base_index must be positive")
    Nshells > 0 || error("Nshells must be positive")
    nlfrac > 0 || error("nlfrac must be positive")
    nthird > 0 || error("nthird must be positive")

    regular_lfrac = nlfrac - 1
    regular_cells = Nshells * regular_lfrac * nthird
    planned_cells = regular_cells + Nshells
    base_index <= planned_cells || error("base_index=$base_index is outside the planned Karl phase grid " * "1:$planned_cells")

    if base_index <= regular_cells
        offset = base_index - 1
        shell_id = mod(offset, Nshells) + 1
        offset = fld(offset, Nshells)
        lfrac_id = mod(offset, regular_lfrac) + 1
        third_id = fld(offset, regular_lfrac) + 1
        return shell_id, lfrac_id, third_id
    end

    shell_id = base_index - regular_cells
    lfrac_id = nlfrac
    third_id = nthird
    return shell_id, lfrac_id, third_id
end

@inline function _orbit_grid_index(shell_id::Int, lfrac_id::Int, third_id::Int, Nshells::Int, nlfrac::Int, nthird::Int)
    1 <= shell_id <= Nshells || error("shell_id is outside the orbit grid")
    1 <= lfrac_id <= nlfrac || error("lfrac_id is outside the orbit grid")
    1 <= third_id <= nthird || error("third_id is outside the orbit grid")
    regular_lfrac = nlfrac - 1
    if lfrac_id == nlfrac
        third_id == nthird || error("The circular boundary family exists only at the final " * "normalized third-integral index")
        return Nshells * regular_lfrac * nthird + shell_id
    end
    return shell_id + Nshells * ((lfrac_id - 1) + regular_lfrac * (third_id - 1))
end

function _build_family_launch_grid(Nbase_orbit::Int, shells::Vector{Float64}, Lfrac, third_launches::Vector{Float64}, pot, frc, force_geometry::Symbol)
    Nshells = length(shells)
    nlfrac = length(Lfrac)
    nthird = length(third_launches)

    nlfrac >= 2 ||
        error("Karl phase grid requires at least one regular Lz family and one circular boundary family")

    nthird > 0 ||
        error("Karl phase grid requires a third-integral coordinate")

    full_phase_grid = Nshells * ((nlfrac - 1) * nthird + 1)

    Nbase_orbit >= full_phase_grid ||
        error(
            "Orbit library has $Nbase_orbit base slots for " *
            "$full_phase_grid normalized family cells"
        )

    launch_r0 = fill(NaN, Nbase_orbit)
    launch_theta0 = fill(NaN, Nbase_orbit)
    launch_energy = fill(NaN, Nbase_orbit)
    launch_lz = fill(NaN, Nbase_orbit)

    # _orbit_grid_index maps the normalized Karl grid uniquely onto the
    # contiguous range 1:full_phase_grid. The old implementation pushed
    # these values serially and then sorted/uniqued them. Constructing the
    # final range directly removes the only shared mutable object that
    # prevented safe shell-level parallelism.
    planned_indices = collect(1:full_phase_grid)

    axisymmetric = force_geometry === :axisymmetric_density_grid
    theta_equator = f64(pi / 2)
    ncurve = max(65, 8 * nthird)

    if axisymmetric
        abs(first(third_launches)) <= 1.0e-12 ||
            error("Normalized third-integral launches must begin at u=0")

        abs(last(third_launches) - 1.0) <= 1.0e-12 ||
            error("Normalized third-integral launches must end at u=1")
    elseif nthird != 1
        error("A spherical orbit family must use exactly one third-integral launch")
    end

    function build_shell!(shell_id::Int)
        rapo = f64(shells[shell_id])

        isfinite(rapo) && rapo > 0.0 ||
            error("Orbit shell $shell_id has an invalid radius")

        @inbounds for lfrac_id in 1:nlfrac
            lf = f64(Lfrac[lfrac_id])

            Lz_family, E_family, vc_family, family_state =
                karl_orbit_family_integrals(
                    rapo=rapo,
                    Lz_frac=lf,
                    pot=pot,
                    frc=frc,
                )

            family_state == :ok ||
                error(
                    "Unable to construct Karl orbit family at " *
                    "shell_id=$shell_id lfrac_id=$lfrac_id " *
                    "state=$family_state"
                )

            circular_boundary = lfrac_id == nlfrac

            if !axisymmetric
                rturn, turning_state =
                    _karl_outer_zero_velocity_radius(
                        energy=E_family,
                        lz=Lz_family,
                        theta0=theta_equator,
                        rapo_max=rapo,
                        pot=pot,
                    )

                turning_state == :ok ||
                    error(
                        "Unable to construct spherical ZVC launch at " *
                        "shell_id=$shell_id lfrac_id=$lfrac_id " *
                        "state=$turning_state"
                    )

                third_id = 1

                c = _orbit_grid_index(
                    shell_id,
                    lfrac_id,
                    third_id,
                    Nshells,
                    nlfrac,
                    nthird,
                )

                launch_r0[c] = rturn
                launch_theta0[c] = theta_equator
                launch_energy[c] = E_family
                launch_lz[c] = Lz_family

                continue
            end

            family_points =
                _karl_family_zvc_launches(
                    energy=E_family,
                    lz=Lz_family,
                    rapo=rapo,
                    pot=pot,
                    Nlaunch=circular_boundary ? 1 : nthird,
                    circular_boundary=circular_boundary,
                    ncurve=circular_boundary ? 0 : ncurve,
                )

            family_points.state == :ok ||
                error(
                    "Unable to construct normalized family ZVC at " *
                    "shell_id=$shell_id lfrac_id=$lfrac_id " *
                    "state=$(family_points.state)"
                )

            if circular_boundary
                third_id = nthird

                c = _orbit_grid_index(
                    shell_id,
                    lfrac_id,
                    third_id,
                    Nshells,
                    nlfrac,
                    nthird,
                )

                launch_r0[c] = only(family_points.r)
                launch_theta0[c] = only(family_points.theta)
                launch_energy[c] = E_family
                launch_lz[c] = Lz_family

                continue
            end

            length(family_points.r) == nthird ||
                error("Family ZVC radius count does not match nthird")

            length(family_points.theta) == nthird ||
                error("Family ZVC theta count does not match nthird")

            length(family_points.u) == nthird ||
                error("Family ZVC normalized-coordinate count does not match nthird")

            maximum(abs.(family_points.u .- third_launches)) <= 1.0e-12 ||
                error("Family ZVC normalized coordinates do not match the common grid")

            for third_id in 1:nthird
                c = _orbit_grid_index(
                    shell_id,
                    lfrac_id,
                    third_id,
                    Nshells,
                    nlfrac,
                    nthird,
                )

                launch_r0[c] = family_points.r[third_id]
                launch_theta0[c] = family_points.theta[third_id]
                launch_energy[c] = E_family
                launch_lz[c] = Lz_family
            end
        end

        return nothing
    end

    if Threads.nthreads() > 1 && Nshells > 1
        next_shell = Threads.Atomic{Int}(1)
        nworkers = min(Threads.nthreads(), Nshells)

        @sync for _ in 1:nworkers
            Threads.@spawn begin
                while true
                    shell_id = Threads.atomic_add!(next_shell, 1)
                    shell_id > Nshells && break
                    build_shell!(shell_id)
                end
            end
        end
    else
        @inbounds for shell_id in 1:Nshells
            build_shell!(shell_id)
        end
    end

    all(isfinite, @view launch_r0[planned_indices]) ||
        error("Normalized family grid contains nonfinite launch radii")

    all(isfinite, @view launch_theta0[planned_indices]) ||
        error("Normalized family grid contains nonfinite launch theta values")

    all(isfinite, @view launch_energy[planned_indices]) ||
        error("Normalized family grid contains nonfinite launch energies")

    all(isfinite, @view launch_lz[planned_indices]) ||
        error("Normalized family grid contains nonfinite launch angular momenta")

    return (
        planned_indices=planned_indices,
        launch_r0=launch_r0,
        launch_theta0=launch_theta0,
        launch_energy=launch_energy,
        launch_lz=launch_lz,
    )
end

function _balanced_launch_order(planned_indices::Vector{Int}, shells::Vector{Float64}, Lfrac, third_launches::Vector{Float64}, shell_band_count::Int,)
    Nshells = length(shells)
    nshell_bands = min(shell_band_count, Nshells)
    nlfrac = length(Lfrac)
    nthird = length(third_launches)

    orbit_cost = Dict{Int,Float64}()
    cells = Dict{NTuple{3,Int},Vector{Int}}()

    @inbounds for c in planned_indices
        shell_id, lfrac_id, third_id = _orbit_grid_indices(c, Nshells, nlfrac, nthird)
        shell_band = fld((shell_id - 1) * nshell_bands, Nshells) + 1
        rapo = shells[shell_id]
        orbit_cost[c] = f64(Lfrac[lfrac_id]) * rapo
        push!(get!(cells, (shell_band, lfrac_id, third_id), Int[]), c)
    end

    for members in values(cells)
        sort!(members; by=c -> (orbit_cost[c], c))

        if length(members) > 1
            radial_order = similar(members)
            lo = 1
            hi = length(members)
            k = 1

            while lo <= hi
                radial_order[k] = members[lo]
                k += 1
                lo += 1

                if lo <= hi
                    radial_order[k] = members[hi]
                    k += 1
                    hi -= 1
                end
            end

            members .= radial_order
        end
    end

    cell_keys = sort!(collect(keys(cells)))
    launch_order = Int[]
    sizehint!(launch_order, length(planned_indices))
    depth = 1

    while length(launch_order) < length(planned_indices)
        added = false

        @inbounds for cell in cell_keys
            members = cells[cell]

            if depth <= length(members)
                push!(launch_order, members[depth])
                added = true
            end
        end

        added || break
        depth += 1
    end

    length(launch_order) == length(planned_indices) ||
        error("Balanced launch ordering lost orbit cells: " * "ordered=$(length(launch_order)) " * "planned=$(length(planned_indices))")

    return launch_order
end

# editting the following 2 to work with Draco and not just segue1

function _build_orbit_shells(R_star_m::Vector{Float64}, light_edges::Vector{Float64}, spatial_edges::Vector{Float64}, max_shells::Int)
    max_shells > 0 || error("Orbit shell budget must be positive; got max_shells=$max_shells")

    shells = Float64[]
    sizehint!(shells, length(R_star_m) + length(light_edges))

    @inbounds for r in R_star_m
        if isfinite(r) && r > 0.0
            push!(shells, r)
        end
    end

    @inbounds for j in 1:(length(light_edges) - 1)
        rlo = light_edges[j]
        rhi = light_edges[j + 1]
        if isfinite(rlo) && isfinite(rhi) && rhi > max(rlo, 0.0)
            rmid = rlo > 0.0 ? sqrt(rlo * rhi) : 0.5 * rhi
            isfinite(rmid) && rmid > 0.0 && push!(shells, rmid)
        end
    end

    rlight_max = light_edges[end]
    isfinite(rlight_max) && rlight_max > 0.0 && push!(shells, rlight_max)

    sort!(shells)
    unique!(shells)

    isempty(shells) && return shells

    candidate_count = length(shells)

    # Preserve the existing sparse-galaxy behavior exactly whenever it fits.
    candidate_count <= max_shells && return shells

    rmin = shells[1]
    rmax = shells[end]

    adaptive_shells = Float64[]
    sizehint!(adaptive_shells, max_shells)

    # Preserve the radial locations of the actual observational constraints.
    for edges in (light_edges, spatial_edges)
        @inbounds for j in 1:(length(edges) - 1)
            rlo = edges[j]
            rhi = edges[j + 1]

            if isfinite(rlo) && isfinite(rhi) && rhi > max(rlo, 0.0)
                rmid = rlo > 0.0 ? sqrt(rlo * rhi) : 0.5 * rhi

                if isfinite(rmid) && rmid >= rmin && rmid <= rmax
                    push!(adaptive_shells, rmid)
                end
            end
        end

        redge_max = edges[end]

        if isfinite(redge_max) && redge_max >= rmin && redge_max <= rmax
            push!(adaptive_shells, redge_max)
        end
    end

    # Always preserve the radial extent of the original shell construction.
    push!(adaptive_shells, rmin)
    push!(adaptive_shells, rmax)

    sort!(adaptive_shells)
    unique!(adaptive_shells)

    length(adaptive_shells) <= max_shells || error("Orbit shell budget $max_shells is too small to preserve $(length(adaptive_shells)) observational radial anchors; increase Norbit")

    # Fill all remaining shell slots logarithmically so the inner galaxy keeps
    # finer absolute radial resolution without tying shell count to Nstar.
    nfill = max_shells - length(adaptive_shells)

    if nfill > 0 && rmax > rmin
        log_rmin = log(rmin)
        log_span = log(rmax) - log_rmin

        @inbounds for k in 1:nfill
            frac = k / (nfill + 1)
            push!(adaptive_shells, exp(log_rmin + frac * log_span))
        end
    end

    sort!(adaptive_shells)
    unique!(adaptive_shells)

    println("[ORBIT SHELL GRID]",
        " candidate=", candidate_count,
        " budget=", max_shells,
        " used=", length(adaptive_shells),
        " compressed=true",
        " N_light=", length(light_edges) - 1,
        " N_kin=", length(spatial_edges) - 1,
        " rmin_pc=", adaptive_shells[1] / pc,
        " rmax_pc=", adaptive_shells[end] / pc,
    )

    return adaptive_shells
end

function _init_orbit_work(
    Norbit::Int, R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool},
    v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, sini::Float64, ctx;
    nsteps::Int, Lfrac, dt_frac_orbit::Float64, max_attempts_factor::Int,
    t_deadline::UInt64, velocity_edges=nothing, light_bin_edges=nothing,
    kinematic_bin_edges=nothing, Nvbin::Int=21, Ntheta_launch::Int=5,
    fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT,
    regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR,
    max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP,
    shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS,
    coverage_check_every::Int=DEFAULT_ORBIT_COVERAGE_CHECK_EVERY,
    tracer_constraint_mode="projected_light",
    losvd_target_mode=:current,
    karl_observables=nothing,
    precomputed_d3_constraints=nothing,
    parallel_init::Bool=true,
)
    iseven(Norbit) || error(
        "Karl prograde/retrograde orbit pairing requires even Norbit because " *
        "Norbit is the final A-matrix column count"
    )

    max_attempts_factor > 0 || error("max_attempts_factor must be positive")

    Nbase_orbit = Norbit ÷ 2
    Nstar = length(R_star_m)

    valid_vec = collect(Bool, valid_vlos)
    vlos_idx = Int[]

    @inbounds for i in 1:Nstar
        valid_vec[i] && push!(vlos_idx, i)
    end

    losvd_mode = _normalize_losvd_target_mode(losvd_target_mode)

    if losvd_mode === :karl_mode0_observables
        karl_observables === nothing &&
            error("Karl observables runtime state is required by _init_orbit_work")

        losvd_aperture_ranges =
            karl_observables_aperture_ranges_m(karl_observables)

        Nspatial = length(karl_observables.aperture_ids)

        length(losvd_aperture_ranges) == Nspatial ||
            error("Karl observables aperture-range count does not match aperture count")

        spatial_edges = vcat(
            0.0,
            [range[2] for range in losvd_aperture_ranges],
        )

        any(diff(spatial_edges) .<= 0.0) &&
            error(
                "Karl observables aperture outer radii must be strictly increasing " *
                "for the orbit shell/diagnostic grid"
            )

        light_bin_edges === nothing &&
            error(
                "light_bin_edges is required for Karl observables mode so the " *
                "existing tracer constraints remain unchanged"
            )

        light_edges = resolve_karl_light_edges(light_bin_edges)
        velocity_edges_use = copy(karl_observables.velocity_edges_mps)

        if velocity_edges !== nothing
            supplied_velocity_edges = Float64.(velocity_edges)

            length(supplied_velocity_edges) == length(velocity_edges_use) ||
                error("Supplied velocity grid does not match the Karl observables CSV")

            maximum(abs.(supplied_velocity_edges .- velocity_edges_use)) <= 1.0e-9 ||
                error("Supplied velocity grid does not match the Karl observables CSV")
        end
    else
        spatial_edges = resolve_karl_spatial_edges(kinematic_bin_edges)

        light_edges =
            light_bin_edges === nothing ?
            spatial_edges :
            resolve_karl_light_edges(light_bin_edges)

        Nspatial = length(spatial_edges) - 1

        losvd_aperture_ranges = [
            (spatial_edges[ib], spatial_edges[ib + 1])
            for ib in 1:Nspatial
        ]

        if losvd_mode === :karl_resolved_stars
            velocity_edges === nothing &&
                error("Karl resolved LOSVD requires explicit velocity_edges defining the selected sample window")
            velocity_edges_use = Float64.(velocity_edges)
        else
            velocity_edges_use =
                velocity_edges === nothing ?
                build_velocity_edges_auto(
                    v_star_mps[vlos_idx],
                    verr_star_mps[vlos_idx];
                    Nvbin=Nvbin,
                ) :
                Float64.(velocity_edges)
        end
    end

    Nlight_projected = length(light_edges) - 1
    Nvbin_eff = length(velocity_edges_use) - 1

    if losvd_mode === :karl_resolved_stars
        any(.!isfinite.(velocity_edges_use)) && error("Karl resolved LOSVD selection edges contain nonfinite values")
        any(diff(velocity_edges_use) .<= 0.0) && error("Karl resolved LOSVD selection edges must be strictly increasing")
        Nvbin_eff == Nvbin || error("Karl resolved LOSVD selection grid must contain exactly Nvbin=$Nvbin bins; got $Nvbin_eff")
        @inbounds for i in vlos_idx
            _bin_index(velocity_edges_use, v_star_mps[i]) != 0 ||
                error("Karl resolved stellar velocity $(v_star_mps[i] / 1.0e3) km/s lies outside the selected sample window [$(velocity_edges_use[1] / 1.0e3), $(velocity_edges_use[end] / 1.0e3)) km/s")
        end
    end

    losvd_mode === :karl_mode0_observables &&
        Nvbin_eff != karl_observables.nvel &&
        error("Karl observables velocity-bin count does not match the CSV metadata")

    Nlosvd = Nspatial * Nvbin_eff

    force_geometry =
        haskey(ctx.halo, :stellar_model) ?
        stellar_model_geometry(ctx.halo[:stellar_model]) :
        :spherical_shell_grid

    tracer_mode = _normalize_tracer_constraint_mode(tracer_constraint_mode)

    stellar_model_state =
        haskey(ctx.halo, :stellar_model) ?
        normalize_stellar_model(ctx.halo[:stellar_model]) :
        nothing

    tracer_mode === :density_3d &&
        stellar_model_state === nothing &&
        error("density_3d tracer constraint requires a 3-D stellar model")

    tracer_mode === :density_3d &&
        force_geometry !== :axisymmetric_density_grid &&
        error("density_3d tracer constraint requires geometry=axisymmetric_density_grid")

    d3_radial_edges =
        losvd_mode === :karl_mode0_observables ?
        copy(karl_observables.radial_edges_m) :
        copy(light_edges)

    d3_angular_edges =
        losvd_mode === :karl_mode0_observables ?
        copy(karl_observables.angular_edges) :
        collect(range(0.0, 1.0; length=6))

    # ------------------------------------------------------------------------------------------------
    # Start D3 tracer construction early.
    #
    # This is independent of the model's orbit-family launch construction, so
    # let another Julia task work on it while this task continues preparing the
    # shell/family geometry.
    #
    # Eventually the normal batch path should pass precomputed_d3_constraints,
    # because these tracer constraints are observational and do not depend on
    # MBH / halo parameters.
    # ------------------------------------------------------------------------------------------------

    d3_task = nothing

    if tracer_mode === :density_3d &&
       precomputed_d3_constraints === nothing &&
       parallel_init &&
       Threads.nthreads() > 1

        d3_task = Threads.@spawn load_karl_d3_tracer_constraints(
            stellar_model_state,
            d3_radial_edges,
            d3_angular_edges,
        )
    end

    third_launches =
        force_geometry === :axisymmetric_density_grid ?
        collect(range(0.0, 1.0; length=max(3, Ntheta_launch))) :
        [1.0]

    families_per_shell =
        (length(Lfrac) - 1) * length(third_launches) + 1

    families_per_shell > 0 ||
        error("Karl family grid requires at least one family per radial shell")

    max_shells = Nbase_orbit ÷ families_per_shell

    max_shells > 0 ||
        error(
            "Orbit library has only $Nbase_orbit base slots but requires " *
            "$families_per_shell base slots per radial shell; increase Norbit"
        )

    shell_support_edges =
        tracer_mode === :density_3d ?
        d3_radial_edges :
        light_edges

    shells = _build_orbit_shells(
        R_star_m,
        shell_support_edges,
        spatial_edges,
        max_shells,
    )

    isempty(shells) &&
        error("Orbit shell grid has no finite positive radii")

    if tracer_mode === :density_3d
        shells[end] >= d3_radial_edges[end] ||
            error(
                "Orbit shell grid does not cover the D3 tracer grid: " *
                "shell_rmax=$(shells[end] / pc) pc " *
                "d3_rmax=$(d3_radial_edges[end] / pc) pc"
            )
    end

    ctx.R[end] > shells[end] ||
        error(
            "Force grid does not cover orbit shell grid: " *
            "force_rmax=$(ctx.R[end] / pc) pc " *
            "shell_rmax=$(shells[end] / pc) pc"
        )

    Nshells = length(shells)

    Nbase_orbit >= Nshells ||
        error(
            "Orbit library has $Nbase_orbit base slots for $Nshells " *
            "required radial shells; increase Norbit"
        )

    full_phase_grid = Nshells * families_per_shell

    Nbase_orbit >= full_phase_grid ||
        error(
            "Karl normalized family grid requires at least $full_phase_grid base orbits " *
            "for Nshells=$Nshells, NLfrac=$(length(Lfrac)), and " *
            "Nthird=$(length(third_launches)); got Nbase_orbit=$Nbase_orbit. " *
            "Increase Norbit to at least $(2 * full_phase_grid)."
        )

    # ------------------------------------------------------------------------------------------------
    # Launch family-grid construction as a separate Julia task.
    #
    # This lets it overlap D3 tracer loading. After _build_family_launch_grid
    # itself is threaded across shell_id, this task will fan out across the
    # available Julia worker pool.
    # ------------------------------------------------------------------------------------------------

    family_task = nothing

    if parallel_init && Threads.nthreads() > 1
        family_task = Threads.@spawn _build_family_launch_grid(
            Nbase_orbit,
            shells,
            Lfrac,
            third_launches,
            ctx.pot,
            ctx.frc,
            force_geometry,
        )
    end

    # ------------------------------------------------------------------------------------------------
    # Do cheap independent allocations while the expensive initialization
    # tasks are running.
    # ------------------------------------------------------------------------------------------------

    A_losvd = zeros(Float64, Nlosvd, Norbit)
    A_kinematic = zeros(Float64, Nspatial, Norbit)

    projection_diag_enabled =
        get(ENV, "OSPM_DIAG_ORBIT_FAMILIES", "0") == "1" &&
        losvd_mode !== :karl_mode0_observables

    projection_shape =
        projection_diag_enabled ?
        (Nspatial, Norbit) :
        (0, 0)

    projection_v3d_mean = fill(NaN, projection_shape...)
    projection_v3d_rms = fill(NaN, projection_shape...)
    projection_abs_vlos_mean = fill(NaN, projection_shape...)
    projection_vlos_rms = fill(NaN, projection_shape...)
    projection_abs_vlos_over_v3d_mean = fill(NaN, projection_shape...)
    projection_los_ratio_lt_0p25 = fill(NaN, projection_shape...)
    projection_los_ratio_lt_0p5 = fill(NaN, projection_shape...)

    success_flags = fill(false, Nbase_orbit)
    attempts_used = zeros(Int, Nbase_orbit)
    min_r_reached = fill(Inf, Nbase_orbit)
    rapo_list = fill(NaN, Nbase_orbit)

    failure_stage = fill(:not_attempted, Nbase_orbit)
    launch_failure_state = fill(:none, Nbase_orbit)
    sos_points = zeros(Int, Nbase_orbit)
    integration_points = zeros(Int, Nbase_orbit)
    integration_termination = fill(:not_run, Nbase_orbit)

    initial_energy_diag = fill(NaN, Nbase_orbit)
    final_energy_diag = fill(NaN, Nbase_orbit)
    max_energy_drift = fill(NaN, Nbase_orbit)
    max_relative_energy_drift = fill(NaN, Nbase_orbit)

    phase_volume_state = init_karl_phase_volume_state(Nbase_orbit)

    # ------------------------------------------------------------------------------------------------
    # Join the expensive initialization tasks.
    # ------------------------------------------------------------------------------------------------

    d3_constraints =
        tracer_mode !== :density_3d ?
        nothing :
        precomputed_d3_constraints !== nothing ?
        precomputed_d3_constraints :
        d3_task !== nothing ?
        fetch(d3_task) :
        load_karl_d3_tracer_constraints(
            stellar_model_state,
            d3_radial_edges,
            d3_angular_edges,
        )

    Nlight =
        d3_constraints === nothing ?
        Nlight_projected :
        length(d3_constraints.target)

    A_light = zeros(Float64, Nlight, Norbit)

    family_grid =
        family_task !== nothing ?
        fetch(family_task) :
        _build_family_launch_grid(
            Nbase_orbit,
            shells,
            Lfrac,
            third_launches,
            ctx.pot,
            ctx.frc,
            force_geometry,
        )

    launch_order = _balanced_launch_order(
        family_grid.planned_indices,
        shells,
        Lfrac,
        third_launches,
        shell_band_count,
    )

    first_coverage_check =
        max(1, ceil(Int, fill_pct * length(launch_order)))

    sini_use = clamp01(f64(sini))
    cosi_use = sqrt(max(0.0, 1.0 - sini_use * sini_use))

    orbit_ctx = (
        frc=ctx.frc,
        R_pos=ctx.R,
        halo=ctx.halo,
        force_geometry=force_geometry,
    )

    return OrbitWorkState(
        Norbit,
        Nbase_orbit,
        Nstar,
        Nspatial,
        Nvbin_eff,
        Nlosvd,
        Nlight,
        Nshells,
        nsteps,
        max_attempts_factor,

        third_launches,
        family_grid.launch_r0,
        family_grid.launch_theta0,
        family_grid.launch_energy,
        family_grid.launch_lz,

        sini_use,
        cosi_use,
        R_star_m,
        valid_vec,
        v_star_mps,
        verr_star_mps,

        spatial_edges,
        light_edges,
        velocity_edges_use,
        shells,
        launch_order,

        orbit_ctx,
        ctx.pot,
        ctx.frc,
        Lfrac,
        force_geometry,
        tracer_mode,
        d3_constraints,
        losvd_mode,
        losvd_aperture_ranges,

        losvd_mode === :karl_mode0_observables ?
        karl_observables :
        nothing,

        dt_frac_orbit,
        t_deadline,
        fill_pct,
        regional_floor,
        max_regional_gap,
        shell_band_count,
        max(1, coverage_check_every),
        Threads.Atomic{Int}(first_coverage_check),

        A_losvd,
        A_kinematic,
        A_light,

        projection_diag_enabled,
        projection_v3d_mean,
        projection_v3d_rms,
        projection_abs_vlos_mean,
        projection_vlos_rms,
        projection_abs_vlos_over_v3d_mean,
        projection_los_ratio_lt_0p25,
        projection_los_ratio_lt_0p5,

        success_flags,
        attempts_used,
        min_r_reached,
        rapo_list,
        failure_stage,
        launch_failure_state,
        sos_points,
        integration_points,
        integration_termination,

        initial_energy_diag,
        final_energy_diag,
        max_energy_drift,
        max_relative_energy_drift,

        phase_volume_state,
        Threads.Atomic{Int}(1),
        Threads.Atomic{Int}(0),
        Threads.Atomic{Int}(0),
        Threads.Atomic{Int}(0),
        ReentrantLock(),
    )
end

@inline function _project_axisym_sample_full(ri::Float64, vr::Float64, vtheta::Float64, vphi::Float64, theta::Float64, phi::Float64, sini::Float64, cosi::Float64)
    st, ct = _sincos_safe(theta)
    cp, sp = cos(phi), sin(phi)
    x = ri * st * cp
    y = ri * st * sp
    z = ri * ct
    vx = vr * st * cp + vtheta * ct * cp - vphi * sp
    vz = vr * ct - vtheta * st
    xsky = y
    ysky = cosi * x - sini * z
    Rproj = sqrt(xsky * xsky + ysky * ysky)
    vlos = sini * vx + cosi * vz
    return Rproj, vlos, xsky, ysky
end

@inline function _project_axisym_sample(ri::Float64, vr::Float64, vtheta::Float64, vphi::Float64, theta::Float64, phi::Float64, sini::Float64, cosi::Float64)
    Rproj, vlos, _, _ = _project_axisym_sample_full(ri, vr, vtheta, vphi, theta, phi, sini, cosi)
    return Rproj, vlos
end

@inline function _orbit_attempt_seed(c_claim::Int, attempt::Int)
    return UInt(0x5eed1234) + UInt(c_claim) * UInt(104729) + UInt(attempt) * UInt(13007)
end

function _coverage_axis_stats( labels::Vector{Int}, attempted::AbstractVector{Bool}, succeeded::AbstractVector{Bool}, nlabels::Int)
    stats = NamedTuple[]
    coverages = Float64[]
    @inbounds for label in 1:nlabels
        planned = 0
        attempted_count = 0
        succeeded_count = 0
        for c in eachindex(labels)
            labels[c] == label || continue
            planned += 1
            attempted[c] && (attempted_count += 1)
            succeeded[c] && (succeeded_count += 1)
        end
        planned == 0 && continue
        attempted_fraction = attempted_count / planned
        success_fraction = attempted_count == 0 ? 0.0 : succeeded_count / attempted_count
        coverage_fraction = succeeded_count / planned
        push!(coverages, coverage_fraction)
        push!(stats,( label=label, planned=planned, attempted=attempted_count, succeeded=succeeded_count, attempted_fraction=attempted_fraction, success_fraction=success_fraction, coverage_fraction=coverage_fraction))
    end

    minimum_coverage = isempty(coverages) ? 0.0 : minimum(coverages)
    coverage_gap = isempty(coverages) ? 0.0 : maximum(coverages) - minimum_coverage
    return stats, minimum_coverage, coverage_gap
end

function _assess_orbit_coverage(st::OrbitWorkState; fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT, regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR,
    max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP, shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS, verify_atomic::Bool=true)

    0.0 < fill_pct <= 1.0 || error("fill_pct must be in (0, 1]")
    0.0 < regional_floor <= 1.0 || error("regional_floor must be in (0, 1]")
    0.0 <= max_regional_gap <= 1.0 || error("max_regional_gap must be in [0, 1]")
    shell_band_count > 0 || error("shell_band_count must be positive")

    planned_indices = st.launch_order
    attempted = BitVector(st.attempts_used[planned_indices] .> 0)
    succeeded = BitVector(st.success_flags[planned_indices])

    any(succeeded .& .!attempted) &&
        error("Orbit coverage accounting is inconsistent: a launch succeeded without an attempt")

    planned = length(planned_indices)
    attempted_count = count(identity, attempted)
    succeeded_count = count(identity, succeeded)
    total_success_count = count(identity, st.success_flags)

    !verify_atomic ||
        st.filled_atomic[] == total_success_count ||
        error("Orbit coverage accounting is inconsistent: filled_atomic=$(st.filled_atomic[]) but success_flags=$total_success_count")

    required = ceil(Int, fill_pct * planned)
    total_coverage = succeeded_count / planned
    nshell_bands = min(shell_band_count, st.Nshells)
    nlfrac = length(st.Lfrac)
    nthird = length(st.third_launches)

    shell_band_labels = Vector{Int}(undef, planned)
    lfrac_labels = Vector{Int}(undef, planned)
    third_labels = Vector{Int}(undef, planned)

    @inbounds for j in 1:planned
        c = planned_indices[j]
        shell_id, lfrac_id, third_id = _orbit_grid_indices(c, st.Nshells, nlfrac, nthird)
        shell_band_labels[j] = fld((shell_id - 1) * nshell_bands, st.Nshells) + 1
        lfrac_labels[j] = lfrac_id
        third_labels[j] = third_id
    end

    shell_stats, shell_min, shell_gap = _coverage_axis_stats(shell_band_labels, attempted, succeeded, nshell_bands)
    lfrac_stats, lfrac_min, lfrac_gap = _coverage_axis_stats(lfrac_labels, attempted, succeeded, nlfrac)
    third_stats, third_min, third_gap = _coverage_axis_stats(third_labels, attempted, succeeded, nthird)

    joint_planned = Dict{NTuple{3,Int},Int}()
    joint_attempted = Dict{NTuple{3,Int},Int}()
    joint_succeeded = Dict{NTuple{3,Int},Int}()

    @inbounds for c in 1:planned
        cell = (shell_band_labels[c], lfrac_labels[c], third_labels[c])
        joint_planned[cell] = get(joint_planned, cell, 0) + 1
        attempted[c] && (joint_attempted[cell] = get(joint_attempted, cell, 0) + 1)
        succeeded[c] && (joint_succeeded[cell] = get(joint_succeeded, cell, 0) + 1)
    end

    joint_holes = NTuple{3,Int}[]
    joint_coverages = Float64[]
    joint_stats = NamedTuple[]

    for cell in sort!(collect(keys(joint_planned)))
        cell_planned = joint_planned[cell]
        cell_attempted = get(joint_attempted, cell, 0)
        cell_succeeded = get(joint_succeeded, cell, 0)

        attempted_fraction = cell_attempted / cell_planned
        success_fraction = cell_attempted == 0 ? 0.0 : cell_succeeded / cell_attempted
        coverage_fraction = cell_succeeded / cell_planned

        push!(joint_coverages, coverage_fraction)
        push!(joint_stats, (shell_band=cell[1], lfrac=cell[2], theta_launch=cell[3], planned=cell_planned, attempted=cell_attempted, succeeded=cell_succeeded,
            attempted_fraction=attempted_fraction, success_fraction=success_fraction, coverage_fraction=coverage_fraction))

        cell_succeeded == 0 && push!(joint_holes, cell)
    end

    sort!(joint_holes)

    joint_min = isempty(joint_coverages) ? 0.0 : minimum(joint_coverages)

    rejection_reasons = String[]

    attempted_count < required && push!(
        rejection_reasons,
        "only $attempted_count planned phase-grid orbit(s) were attempted; $required are required by fill_pct=$(fill_pct)",
    )

    total_coverage < fill_pct && push!(
        rejection_reasons,
        "total orbit coverage $(round(total_coverage; digits=3)) is below $(fill_pct)",
    )

    for (axis_name, axis_min, axis_gap) in (
        ("shell_band", shell_min, shell_gap),
        ("Lfrac", lfrac_min, lfrac_gap),
        ("third_integral", third_min, third_gap),
    )
        axis_min < regional_floor && push!(
            rejection_reasons,
            "$axis_name minimum coverage $(round(axis_min; digits=3)) is below $(regional_floor)",
        )

        axis_gap > max_regional_gap && push!(
            rejection_reasons,
            "$axis_name coverage gap $(round(axis_gap; digits=3)) exceeds $(max_regional_gap)",
        )
    end

    !isempty(joint_holes) && push!(
        rejection_reasons,
        "$(length(joint_holes)) planned shell-band/Lfrac/third-integral cells have no successful orbit",
    )

    successful_launches = planned_indices[findall(succeeded)]
    successful_columns = Vector{Int}(undef, 2 * length(successful_launches))

    @inbounds for (j, c) in pairs(successful_launches)
        successful_columns[2 * j - 1] = 2 * c - 1
        successful_columns[2 * j] = 2 * c
    end

    accepted =
        total_coverage >= fill_pct &&
        shell_min >= regional_floor &&
        lfrac_min >= regional_floor &&
        third_min >= regional_floor &&
        shell_gap <= max_regional_gap &&
        lfrac_gap <= max_regional_gap &&
        third_gap <= max_regional_gap &&
        isempty(joint_holes)

    return (
        accepted=accepted,
        planned=planned,
        attempted=attempted_count,
        succeeded=succeeded_count,
        required=required,
        attempted_fraction=attempted_count / planned,
        success_fraction=attempted_count == 0 ? 0.0 : succeeded_count / attempted_count,
        coverage_fraction=total_coverage,
        shell_bands=shell_stats,
        lfrac=lfrac_stats,
        theta_launches=third_stats,
        shell_minimum_coverage=shell_min,
        lfrac_minimum_coverage=lfrac_min,
        theta_minimum_coverage=third_min,
        shell_coverage_gap=shell_gap,
        lfrac_coverage_gap=lfrac_gap,
        theta_coverage_gap=third_gap,
        joint_cell_count=length(joint_planned),
        joint_cells=joint_stats,
        joint_minimum_coverage=joint_min,
        joint_holes=joint_holes,
        successful_launches=successful_launches,
        successful_columns=successful_columns,
        rejection_reasons=rejection_reasons,
    )
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

@inline function _shell_region(label::Int, nlabels::Int)
    x = (label - 0.5) / max(nlabels, 1)
    x <= 1 / 3 && return "inner"
    x >= 2 / 3 && return "outer"
    return "middle"
end

function _coverage_metadata( coverage; fill_pct::Float64, regional_floor::Float64, max_regional_gap::Float64, warn_fill_pct::Float64, warn_success_pct::Float64, warn_regional_floor::Float64, warn_max_regional_gap::Float64)
    strict_pass = coverage.accepted
    soft_pass =
        coverage.coverage_fraction >= warn_fill_pct &&
        coverage.success_fraction >= warn_success_pct &&
        coverage.shell_minimum_coverage >= warn_regional_floor &&
        coverage.lfrac_minimum_coverage >= warn_regional_floor &&
        coverage.theta_minimum_coverage >= warn_regional_floor &&
        coverage.shell_coverage_gap <= warn_max_regional_gap &&
        coverage.lfrac_coverage_gap <= warn_max_regional_gap &&
        coverage.theta_coverage_gap <= warn_max_regional_gap
    coverage_status = strict_pass ? "strict_pass" :
        (soft_pass ? "coverage_warn" : "severe_coverage_warn")
    issue_axes = String[]
    coverage.coverage_fraction < fill_pct && push!(issue_axes, "total")
    for (axis_name, axis_min, axis_gap) in (
        ("shell", coverage.shell_minimum_coverage, coverage.shell_coverage_gap),
        ("lfrac", coverage.lfrac_minimum_coverage, coverage.lfrac_coverage_gap),
        ("theta", coverage.theta_minimum_coverage, coverage.theta_coverage_gap))
        (axis_min < regional_floor || axis_gap > max_regional_gap) && push!(issue_axes, axis_name)
    end
    !isempty(coverage.joint_holes) && push!(issue_axes, "joint")
    unique!(issue_axes)
    issue_axis = isempty(issue_axes) ? "none" :
        (length(issue_axes) == 1 ? only(issue_axes) : "multiple")
    shell_coverages = [Float64(stat.coverage_fraction) for stat in coverage.shell_bands]
    max_shell_coverage = isempty(shell_coverages) ? 0.0 : maximum(shell_coverages)
    issue_shell_bands = Int[]
    for stat in coverage.shell_bands
        if stat.coverage_fraction < regional_floor || max_shell_coverage - stat.coverage_fraction > max_regional_gap
            push!(issue_shell_bands, Int(stat.label))
        end
    end
    for hole in coverage.joint_holes
        push!(issue_shell_bands, Int(hole[1]))
    end
    sort!(issue_shell_bands)
    unique!(issue_shell_bands)
    issue_region = "none"
    if !isempty(issue_shell_bands)
        n_shell_bands = length(coverage.shell_bands)
        regions = unique([_shell_region(label, n_shell_bands) for label in issue_shell_bands])
        issue_region = length(regions) == 1 ? only(regions) : "multiple"
    elseif coverage.coverage_fraction < fill_pct
        issue_region = "all"
    end
    return (status=coverage_status, issue_axis=issue_axis, issue_region=issue_region, issue_shell_bands=join(issue_shell_bands, ";"), reasons=join(coverage.rejection_reasons, " | "))
end

function _maybe_stop_orbit_phase_for_coverage!(st::OrbitWorkState)
    return false
end

function _orbit_library_usable(st::OrbitWorkState, successful_columns::Vector{Int})
    if isempty(successful_columns)
        println("[ORBIT LIBRARY USABILITY FAIL] reason=no_successful_columns")
        return false
    end
    bad_columns = Int[]
    @inbounds for col in successful_columns
        losvd_activity = sum(abs, @view(st.A_losvd[:, col]))
        light_activity = sum(abs, @view(st.A_light[:, col]))
        activity = losvd_activity + light_activity
        if !(isfinite(activity) && activity > 0.0)
            push!(bad_columns, col)
            println(
                "[ORBIT LIBRARY BAD COLUMN]",
                " col=", col,
                " losvd_activity=", losvd_activity,
                " light_activity=", light_activity,
                " total_activity=", activity,
            )
        end
    end
    bad_light_rows = Int[]
    @inbounds for row in 1:st.Nlight
        activity = sum(abs, @view(st.A_light[row, successful_columns]))
        if !(isfinite(activity) && activity > 0.0)
            push!(bad_light_rows, row)
            if st.d3_constraints === nothing
                println("[ORBIT LIBRARY BAD LIGHT ROW]", " row=", row, " constraint_inner_pc=", st.light_edges[row] / pc, " constraint_outer_pc=", st.light_edges[row + 1] / pc, " activity=", activity)
            else
                ir = st.d3_constraints.row_radial[row]
                iv = st.d3_constraints.row_angular[row]
                println("[ORBIT LIBRARY BAD LIGHT ROW]", " row=", row, " radial_bin=", ir, " angular_bin=", iv, " constraint_inner_pc=", st.d3_constraints.radial_edges_m[ir] / pc, " constraint_outer_pc=", st.d3_constraints.radial_edges_m[ir + 1] / pc, " activity=", activity)
            end
        end
    end
    if !isempty(bad_columns) || !isempty(bad_light_rows)
        println(
            "[ORBIT LIBRARY USABILITY FAIL]",
            " bad_columns=", length(bad_columns),
            " bad_light_rows=", length(bad_light_rows),
            " successful_columns=", length(successful_columns),
        )
        return false
    end
    return true
end

function _compact_orbit_matrices(st::OrbitWorkState, successful_columns::Vector{Int})
    return st.A_losvd[:, successful_columns], st.A_light[:, successful_columns], st.A_kinematic[:, successful_columns]
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
    losvd_state = karl_losvd_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, st.Nspatial, st.Nvbin)

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
        sigma_is_invalid = raw_sigma[local_bin] == DEFAULT_KARL_INVALID_SIGMA_SENTINEL
        println("[ORBIT PROJECTION LOSVD BIN] i=", model_index, " aperture=", target_aperture, " velocity_bin=", local_bin, " v_inner_kms=", st.velocity_edges[local_bin] / 1.0e3, " v_outer_kms=", st.velocity_edges[local_bin + 1] / 1.0e3, " v_center_kms=", velocity_centers[local_bin] / 1.0e3, " raw_target=", raw_target[local_bin], " raw_sigma=", raw_sigma[local_bin], " sigma_is_invalid=", sigma_is_invalid, " effective_target=", effective_target[local_bin], " effective_sigma=", effective_sigma[local_bin], " total_model=", model_hist[local_bin], " inner_family_model=", family_hist[local_bin], " model_without_inner_family=", model_without_family, " target_shape_fraction=", target_shape_fraction, " model_shape_fraction=", model_shape_fraction, " inner_family_shape_fraction=", family_shape_fraction, " inner_family_fraction_of_model_bin=", family_fraction_of_model_bin, " residual_sigma=", residual_sigma, " chi2_bin=", chi_bin, " chi2_without_inner_family=", chi_without_bin, " delta_chi2_inner_family=", chi_bin - chi_without_bin)
    end
    println("[ORBIT PROJECTION LOSVD CHI SUMMARY] i=", model_index, " aperture=", target_aperture, " chi2_with_inner_family=", chi_aperture, " chi2_without_inner_family_fixed_other_weights=", chi_without_family, " delta_chi2_inner_family=", chi_aperture - chi_without_family, " max_abs_sigma_residual=", max_abs_sigma_residual, " max_abs_sigma_bin=", max_abs_sigma_bin)

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
    entropy_direct = karl_entropy_value(wnorm, wphase_use)
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

function _build_compact_karl_wphase(st::OrbitWorkState, successful_columns::Vector{Int})
    phase_result = compute_karl_phase_volumes(st.phase_volume_state; normalization=:geometric_mean, strict=true)
    wphase_use = compact_karl_wphase(phase_result.wphase_paired, successful_columns, st.Norbit)
    length(wphase_use) == length(successful_columns) ||
        error("compacted Karl wphase length does not match successful orbit columns")
    return wphase_use, phase_result.diagnostics, phase_result
end

# This is the main worker function that runs in a thread to compute orbits and fill the A-matrix.
function _orbit_worker!(st::OrbitWorkState)

    col_losvd_pro = zeros(Float64, st.Nlosvd)
    col_losvd_ret = zeros(Float64, st.Nlosvd)
    col_kinematic = zeros(Float64, st.Nspatial)
    col_light = zeros(Float64, st.Nlight)
    projection_hits = zeros(Int, st.Nspatial)
    projection_v3d_sum = zeros(Float64, st.Nspatial)
    projection_v3d_sq_sum = zeros(Float64, st.Nspatial)
    projection_abs_vlos_pro_sum = zeros(Float64, st.Nspatial)
    projection_abs_vlos_ret_sum = zeros(Float64, st.Nspatial)
    projection_vlos_pro_sq_sum = zeros(Float64, st.Nspatial)
    projection_vlos_ret_sq_sum = zeros(Float64, st.Nspatial)
    projection_ratio_pro_sum = zeros(Float64, st.Nspatial)
    projection_ratio_ret_sum = zeros(Float64, st.Nspatial)
    projection_ratio_pro_lt_0p25 = zeros(Int, st.Nspatial)
    projection_ratio_ret_lt_0p25 = zeros(Int, st.Nspatial)
    projection_ratio_pro_lt_0p5 = zeros(Int, st.Nspatial)
    projection_ratio_ret_lt_0p5 = zeros(Int, st.Nspatial)
    s_arr = Vector{Float64}(undef, st.nsteps)
    vlos_pro_buf = Vector{Float64}(undef, st.nsteps)
    vlos_ret_buf = Vector{Float64}(undef, st.nsteps)
    xsky_arr = st.losvd_target_mode === :karl_mode0_observables ? Vector{Float64}(undef, st.nsteps) : Float64[]
    ysky_arr = st.losvd_target_mode === :karl_mode0_observables ? Vector{Float64}(undef, st.nsteps) : Float64[]
    karl_vlib_pro = st.losvd_target_mode === :karl_mode0_observables ? zeros(Float64, st.karl_observables.nvel, st.karl_observables.nrlib, st.karl_observables.nvlib) : zeros(Float64, 0, 0, 0)
    karl_spatial = st.losvd_target_mode === :karl_mode0_observables ? zeros(Float64, st.karl_observables.nrlib, st.karl_observables.nvlib) : zeros(Float64, 0, 0)
    energy_drift_tolerance = 1.0e-2
    dt_scales = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    continuation_step_safeties = (0.10, 0.075, 0.05, 0.025, 0.0125)
    sos_chunk_base_steps = 4_000
    max_sos_base_steps = max(st.nsteps, 40_000)
    max_plunging_sos_base_steps = max(st.nsteps, 120_000)

    function integrate_stably(ic_use, Lz_use, base_steps::Int; start_scale::Float64=1.0)
        last_result = nothing
        for dt_scale in dt_scales
            dt_scale <= start_scale * (1.0 + 10.0 * eps(Float64)) || continue
            time_ns() > st.t_deadline && break
            st.phase[] == 1 || break
            scaled_steps = ceil(Int, base_steps / dt_scale)
            scaled_energy_check_every = max(1, round(Int, 100 / dt_scale))
            last_result = integrate_orbit_rk4(ic=ic_use, xLz=Lz_use, orbit_ctx=st.orbit_ctx, nsteps=scaled_steps, pot=st.pot, return_diag=true,
                dt_scale=dt_scale, energy_check_every=scaled_energy_check_every, max_relative_energy_drift_allowed=energy_drift_tolerance)
            diag = last_result[5]
            if diag.termination_reason === :completed && diag.energy_valid
                return last_result
            end
        end

        return last_result
    end

    function continue_orbit_chunk(ic_use, Lz_use, base_steps::Int, previous_diag, reference_energy)
        time_ns() > st.t_deadline && return nothing
        dt_scale = f64(previous_diag.dt_scale)
        scaled_steps = ceil(Int, base_steps / dt_scale)
        scaled_energy_check_every = max(1, round(Int, 100 / dt_scale))
        last_result = nothing
        for local_step_safety in continuation_step_safeties
            time_ns() > st.t_deadline && return nothing
            st.phase[] == 1 || return nothing
            last_result = integrate_orbit_rk4( ic=ic_use, xLz=Lz_use, orbit_ctx=st.orbit_ctx, nsteps=scaled_steps, pot=st.pot, return_diag=true, dt_scale=dt_scale,
                energy_check_every=scaled_energy_check_every, max_relative_energy_drift_allowed=energy_drift_tolerance, local_step_safety=local_step_safety, max_substeps_per_step=1024,
                continuation_state=( previous_diag.final_R, previous_diag.final_z, previous_diag.final_vR, previous_diag.final_vz), reference_energy=reference_energy)
            diag = last_result[5]
            if diag.termination_reason === :completed && diag.energy_valid
                return last_result
            end
        end
        return last_result
    end

    function store_integration_diag!(c_claim, diag, total_points, reference_energy, max_abs_drift, max_rel_drift)
        st.integration_points[c_claim] = total_points
        st.integration_termination[c_claim] = diag.termination_reason
        st.initial_energy_diag[c_claim] = reference_energy
        st.final_energy_diag[c_claim] = diag.final_energy
        st.max_energy_drift[c_claim] = max_abs_drift
        st.max_relative_energy_drift[c_claim] = max_rel_drift
    end

    while true
        time_ns() > st.t_deadline && break
        st.phase[] != 1 && break
        slot_seq = Threads.atomic_add!(st.next_orbit, 1)
        slot_seq > length(st.launch_order) && break
        c_claim = st.launch_order[slot_seq]
        st.failure_stage[c_claim] = :claimed
        shell_id, lfrac_id, third_id = _orbit_grid_indices(c_claim, st.Nshells, length(st.Lfrac), length(st.third_launches))
        rapo = f64(st.shells[shell_id])
        st.rapo_list[c_claim] = rapo

        if !(isfinite(rapo) && rapo > 0.0)
            st.failure_stage[c_claim] = :invalid_shell_radius
            continue
        end

        lf = f64(st.Lfrac[lfrac_id])
        rturn = f64(st.launch_r0[c_claim])
        theta0 = f64(st.launch_theta0[c_claim])
        E_family = f64(st.launch_energy[c_claim])
        Lz_family = f64(st.launch_lz[c_claim])

        if !(isfinite(rturn) && rturn > 0.0 && isfinite(theta0) && isfinite(E_family) && isfinite(Lz_family))
            st.failure_stage[c_claim] = :invalid_family_geometry
            continue
        end

        circular_boundary = lfrac_id == length(st.Lfrac)
        equatorial_planar = !circular_boundary && abs(theta0 - DEFAULT_KARL_PHASE_SECTION_THETA) <= 1.0e-10

        if time_ns() > st.t_deadline
            st.failure_stage[c_claim] = :deadline_before_launch
            return nothing
        end

        if st.phase[] != 1
            st.failure_stage[c_claim] = :phase_closed_before_launch
            return nothing
        end

        st.attempts_used[c_claim] = 1
        ic, Lz0, E0, vc, launch_state = launch_orbit_apocenter(rapo=rapo, theta0=theta0, Lz_frac=lf, pot=st.pot, frc=st.frc, dt_frac=st.dt_frac_orbit, fixed_energy=E_family, fixed_lz=Lz_family, fixed_rturn=rturn)

        if launch_state != :ok
            st.failure_stage[c_claim] = :launch_failed
            st.launch_failure_state[c_claim] = launch_state
            continue
        end

        integration_result = integrate_stably(ic, Lz0, st.nsteps)

        if integration_result === nothing
            st.failure_stage[c_claim] = :deadline_during_integration
            return nothing
        end

        r, vr, theta, vtheta, integration_diag = integration_result
        reference_energy = integration_diag.initial_energy
        orbit_max_abs_drift = integration_diag.max_absolute_energy_drift
        orbit_max_rel_drift = integration_diag.max_relative_energy_drift

        store_integration_diag!(c_claim, integration_diag, length(r), reference_energy, orbit_max_abs_drift, orbit_max_rel_drift)
        if integration_diag.termination_reason === :hit_rmax
            println(
                "[ORBIT RMAX HIT]",
                " c=", c_claim,
                " shell_id=", shell_id,
                " lfrac_id=", lfrac_id,
                " third_id=", third_id,
                " third_u=", st.third_launches[third_id],
                " rapo_pc=", rapo / pc,
                " rturn_pc=", rturn / pc,
                " rmax_stop_pc=", integration_diag.rmax_stop / pc,
                " maximum_r_pc=", integration_diag.maximum_r / pc,
                " final_r_pc=", integration_diag.final_r / pc,
                " completed_steps=", integration_diag.completed_steps,
                " dt_scale=", integration_diag.dt_scale,
                " max_rel_energy_drift=", integration_diag.max_relative_energy_drift,
            )
        end

        if !integration_diag.energy_valid
            st.failure_stage[c_claim] = :energy_drift_exceeded
            continue
        elseif integration_diag.termination_reason !== :completed
            st.failure_stage[c_claim] = :integration_terminated
            continue
        elseif isempty(r)
            st.failure_stage[c_claim] = :empty_integration
            continue
        end

        sos_r, sos_vr_abs = if circular_boundary
            finite_index = findfirst(k -> isfinite(r[k]) && r[k] > 0.0, eachindex(r))
            finite_index === nothing ? (Float64[], Float64[]) : ([f64(r[finite_index])], [0.0])

        elseif equatorial_planar || st.force_geometry !== :axisymmetric_density_grid
            rr = Float64[]
            vv = Float64[]
            sizehint!(rr, length(r))
            sizehint!(vv, length(vr))

            @inbounds for k in eachindex(r)
                rk = f64(r[k])
                vk = abs(f64(vr[k]))

                if isfinite(rk) && rk > 0.0 && isfinite(vk)
                    push!(rr, rk)
                    push!(vv, vk)
                end
            end

            rr, vv
        else
            collect_karl_equatorial_sos(r, vr, theta; section_theta=DEFAULT_KARL_PHASE_SECTION_THETA, crossing_mode=:karl_step, direction=:up, skip_first=true)
        end
        if !circular_boundary && !equatorial_planar && st.force_geometry === :axisymmetric_density_grid &&
           length(sos_r) < DEFAULT_KARL_PHASE_MIN_SOS_POINTS
            base_steps_used = st.nsteps
            sos_base_step_limit = lfrac_id == 1 ? max_plunging_sos_base_steps : max_sos_base_steps
            extension_failure = nothing
            while length(sos_r) < DEFAULT_KARL_PHASE_MIN_SOS_POINTS && base_steps_used < sos_base_step_limit
                if time_ns() > st.t_deadline
                    st.failure_stage[c_claim] = :deadline_during_extended_integration
                    return nothing
                end
                chunk_base_steps = min(sos_chunk_base_steps, sos_base_step_limit - base_steps_used)
                continuation_result = continue_orbit_chunk(ic, Lz0, chunk_base_steps, integration_diag, reference_energy)
                if continuation_result === nothing
                    st.failure_stage[c_claim] = :deadline_during_extended_integration
                    return nothing
                end
                r_chunk, vr_chunk, theta_chunk, vtheta_chunk, chunk_diag = continuation_result
                orbit_max_abs_drift = max(orbit_max_abs_drift, chunk_diag.max_absolute_energy_drift)
                orbit_max_rel_drift = max(orbit_max_rel_drift, chunk_diag.max_relative_energy_drift)
                if !chunk_diag.energy_valid
                    store_integration_diag!(c_claim, chunk_diag, length(r) + length(r_chunk), reference_energy, orbit_max_abs_drift, orbit_max_rel_drift)
                    extension_failure = :extended_energy_drift_exceeded
                    break
                elseif chunk_diag.termination_reason !== :completed
                    store_integration_diag!(c_claim, chunk_diag, length(r) + length(r_chunk), reference_energy, orbit_max_abs_drift, orbit_max_rel_drift)
                    extension_failure = :extended_integration_terminated
                    break
                elseif isempty(r_chunk)
                    store_integration_diag!(c_claim, chunk_diag, length(r), reference_energy, orbit_max_abs_drift, orbit_max_rel_drift)
                    extension_failure = :empty_extended_integration
                    break
                end
                append!(r, r_chunk)
                append!(vr, vr_chunk)
                append!(theta, theta_chunk)
                append!(vtheta, vtheta_chunk)
                integration_diag = chunk_diag
                base_steps_used += chunk_base_steps
                store_integration_diag!(c_claim, integration_diag, length(r), reference_energy, orbit_max_abs_drift, orbit_max_rel_drift)
                sos_r, sos_vr_abs = collect_karl_equatorial_sos(r, vr, theta; section_theta=DEFAULT_KARL_PHASE_SECTION_THETA, crossing_mode=:karl_step, direction=:up, skip_first=true)
            end
            if extension_failure !== nothing
                st.failure_stage[c_claim] = extension_failure
                continue
            end
        end
        st.sos_points[c_claim] = length(sos_r)
        if circular_boundary
            if length(sos_r) != 1
                st.failure_stage[c_claim] = :invalid_circular_sos
                continue
            end
        elseif length(sos_r) < DEFAULT_KARL_PHASE_MIN_SOS_POINTS
            st.failure_stage[c_claim] = :insufficient_sos
            continue
        end
        st.min_r_reached[c_claim] = minimum(r)
        Nhits = length(r)
        dt_orb = integration_diag.dt
        resize!(s_arr, Nhits)
        resize!(vlos_pro_buf, Nhits)
        resize!(vlos_ret_buf, Nhits)
        st.losvd_target_mode === :karl_mode0_observables && resize!(xsky_arr, Nhits)
        st.losvd_target_mode === :karl_mode0_observables && resize!(ysky_arr, Nhits)
        phi = 0.0
        @inbounds for i in 1:Nhits
            ri = f64(r[i])
            thi = f64(theta[i])
            si = _ssin(thi)
            vphi_i = f64(Lz0) / max(ri * si, 1.0e-30)
            if st.losvd_target_mode === :karl_mode0_observables
                s_arr[i], vlos_pro_buf[i], xsky_arr[i], ysky_arr[i] = _project_axisym_sample_full(ri, f64(vr[i]), f64(vtheta[i]), vphi_i, thi, phi, st.sini, st.cosi)
                vlos_ret_buf[i] = -vlos_pro_buf[i]
            else
                s_arr[i], vlos_pro_buf[i] = _project_axisym_sample(ri, f64(vr[i]), f64(vtheta[i]), vphi_i, thi, phi, st.sini, st.cosi)
                _, vlos_ret_buf[i] = _project_axisym_sample(ri, f64(vr[i]), f64(vtheta[i]), -vphi_i, thi, phi, st.sini, st.cosi)
            end
            phi += f64(Lz0) / max(ri * ri * si * si, 1.0e-30) * dt_orb
        end

        fill!(col_losvd_pro, 0.0)
        fill!(col_losvd_ret, 0.0)
        fill!(col_kinematic, 0.0)
        fill!(col_light, 0.0)
        st.losvd_target_mode === :karl_mode0_observables && fill!(karl_vlib_pro, 0.0)
        st.losvd_target_mode === :karl_mode0_observables && fill!(karl_spatial, 0.0)
        fill!(projection_hits, 0)
        fill!(projection_v3d_sum, 0.0)
        fill!(projection_v3d_sq_sum, 0.0)
        fill!(projection_abs_vlos_pro_sum, 0.0)
        fill!(projection_abs_vlos_ret_sum, 0.0)
        fill!(projection_vlos_pro_sq_sum, 0.0)
        fill!(projection_vlos_ret_sq_sum, 0.0)
        fill!(projection_ratio_pro_sum, 0.0)
        fill!(projection_ratio_ret_sum, 0.0)
        fill!(projection_ratio_pro_lt_0p25, 0)
        fill!(projection_ratio_ret_lt_0p25, 0)
        fill!(projection_ratio_pro_lt_0p5, 0)
        fill!(projection_ratio_ret_lt_0p5, 0)

        @inbounds for k in 1:Nhits
            il = st.tracer_constraint_mode === :density_3d ? karl_d3_tracer_row(st.d3_constraints, f64(r[k]), f64(theta[k])) : _bin_index(st.light_edges, s_arr[k])
            il > 0 && (col_light[il] += 1.0)

            if st.losvd_target_mode === :karl_mode0_observables
                ir_karl, iv_karl = karl_observables_projected_cell(st.karl_observables, s_arr[k], ysky_arr[k])
                ir_karl == 0 && continue
                iv_karl == 0 && continue
                karl_spatial[ir_karl, iv_karl] += 1.0
                # Karl's raw v1lib source grid is the x>=0 half-plane. Fold x<0 samples back to that source grid and reverse their LOS velocity before the seeing convolution.
                vlos_folded = xsky_arr[k] < 0.0 ? -vlos_pro_buf[k] : vlos_pro_buf[k]
                jb_pro = _bin_index(st.velocity_edges, vlos_folded)
                jb_pro > 0 && (karl_vlib_pro[jb_pro, ir_karl, iv_karl] += 1.0)
                continue
            end

            ik = _bin_index(st.spatial_edges, s_arr[k])
            ik == 0 && continue
            col_kinematic[ik] += 1.0
            if st.projection_diag_enabled
                rk = f64(r[k])
                thk = f64(theta[k])
                sik = _ssin(thk)
                vrk = f64(vr[k])
                vtk = f64(vtheta[k])
                vphik = f64(Lz0) / max(rk * sik, 1.0e-30)
                v3d = sqrt(vrk * vrk + vtk * vtk + vphik * vphik)
                abs_vlos_pro = abs(vlos_pro_buf[k])
                abs_vlos_ret = abs(vlos_ret_buf[k])
                ratio_pro = v3d > 0.0 ? abs_vlos_pro / v3d : 0.0
                ratio_ret = v3d > 0.0 ? abs_vlos_ret / v3d : 0.0
                projection_hits[ik] += 1
                projection_v3d_sum[ik] += v3d
                projection_v3d_sq_sum[ik] += v3d * v3d
                projection_abs_vlos_pro_sum[ik] += abs_vlos_pro
                projection_abs_vlos_ret_sum[ik] += abs_vlos_ret
                projection_vlos_pro_sq_sum[ik] += vlos_pro_buf[k] * vlos_pro_buf[k]
                projection_vlos_ret_sq_sum[ik] += vlos_ret_buf[k] * vlos_ret_buf[k]
                projection_ratio_pro_sum[ik] += ratio_pro
                projection_ratio_ret_sum[ik] += ratio_ret
                ratio_pro < 0.25 && (projection_ratio_pro_lt_0p25[ik] += 1)
                ratio_ret < 0.25 && (projection_ratio_ret_lt_0p25[ik] += 1)
                ratio_pro < 0.5 && (projection_ratio_pro_lt_0p5[ik] += 1)
                ratio_ret < 0.5 && (projection_ratio_ret_lt_0p5[ik] += 1)
            end
            jb_pro = _bin_index(st.velocity_edges, vlos_pro_buf[k])
            if jb_pro > 0
                row_pro = (ik - 1) * st.Nvbin + jb_pro
                col_losvd_pro[row_pro] += 1.0
            end
            jb_ret = _bin_index(st.velocity_edges, vlos_ret_buf[k])
            if jb_ret > 0
                row_ret = (ik - 1) * st.Nvbin + jb_ret
                col_losvd_ret[row_ret] += 1.0
            end
        end

        col_light ./= Nhits
        if st.losvd_target_mode === :karl_mode0_observables
            karl_vlib_pro ./= Nhits
            karl_spatial ./= Nhits
            karl_response = karl_observables_seeing_response(karl_vlib_pro, karl_spatial, st.karl_observables)
            col_losvd_pro .= karl_response.losvd
            col_kinematic .= karl_response.kinematic
            @inbounds for ia in 1:st.Nspatial
                for jb in 1:st.Nvbin
                    row = (ia - 1) * st.Nvbin + jb
                    reverse_row = (ia - 1) * st.Nvbin + (st.Nvbin - jb + 1)
                    col_losvd_ret[row] = col_losvd_pro[reverse_row]
                end
            end
        else
            col_kinematic ./= Nhits
            col_losvd_pro ./= Nhits
            col_losvd_ret ./= Nhits
        end
        pro_activity = sum(abs, col_losvd_pro) + sum(abs, col_light)
        ret_activity = sum(abs, col_losvd_ret) + sum(abs, col_light)
        if !(isfinite(pro_activity) && pro_activity > 0.0 && isfinite(ret_activity) && ret_activity > 0.0)
            st.failure_stage[c_claim] = :zero_observable_support
            continue
        end
        register_karl_phase_launch!(st.phase_volume_state, c_claim; energy=E0, lz=Lz0, energy_index=shell_id, lz_index=lfrac_id, third_index=third_id)
        record_karl_phase_sos!(st.phase_volume_state, c_claim, sos_r, sos_vr_abs)
        col_pro = 2 * c_claim - 1
        col_ret = 2 * c_claim
        @inbounds st.A_losvd[:, col_pro] .= col_losvd_pro
        @inbounds st.A_losvd[:, col_ret] .= col_losvd_ret
        @inbounds st.A_kinematic[:, col_pro] .= col_kinematic
        @inbounds st.A_kinematic[:, col_ret] .= col_kinematic
        @inbounds st.A_light[:, col_pro] .= col_light
        @inbounds st.A_light[:, col_ret] .= col_light
        if st.projection_diag_enabled
            @inbounds for ik in 1:st.Nspatial
                nhit = projection_hits[ik]
                nhit > 0 || continue
                invhit = 1.0 / nhit
                v3d_mean = projection_v3d_sum[ik] * invhit
                v3d_rms = sqrt(max(projection_v3d_sq_sum[ik] * invhit, 0.0))
                st.projection_v3d_mean[ik, col_pro] = v3d_mean
                st.projection_v3d_mean[ik, col_ret] = v3d_mean
                st.projection_v3d_rms[ik, col_pro] = v3d_rms
                st.projection_v3d_rms[ik, col_ret] = v3d_rms
                st.projection_abs_vlos_mean[ik, col_pro] = projection_abs_vlos_pro_sum[ik] * invhit
                st.projection_abs_vlos_mean[ik, col_ret] = projection_abs_vlos_ret_sum[ik] * invhit
                st.projection_vlos_rms[ik, col_pro] = sqrt(max(projection_vlos_pro_sq_sum[ik] * invhit, 0.0))
                st.projection_vlos_rms[ik, col_ret] = sqrt(max(projection_vlos_ret_sq_sum[ik] * invhit, 0.0))
                st.projection_abs_vlos_over_v3d_mean[ik, col_pro] = projection_ratio_pro_sum[ik] * invhit
                st.projection_abs_vlos_over_v3d_mean[ik, col_ret] = projection_ratio_ret_sum[ik] * invhit
                st.projection_los_ratio_lt_0p25[ik, col_pro] = projection_ratio_pro_lt_0p25[ik] * invhit
                st.projection_los_ratio_lt_0p25[ik, col_ret] = projection_ratio_ret_lt_0p25[ik] * invhit
                st.projection_los_ratio_lt_0p5[ik, col_pro] = projection_ratio_pro_lt_0p5[ik] * invhit
                st.projection_los_ratio_lt_0p5[ik, col_ret] = projection_ratio_ret_lt_0p5[ik] * invhit
            end
        end
        st.failure_stage[c_claim] = :success
        st.success_flags[c_claim] = true
        Threads.atomic_add!(st.filled_atomic, 1)
        _maybe_stop_orbit_phase_for_coverage!(st)
    end

    return nothing
end

function _run_orbit_worker!(st::OrbitWorkState; scheduler_counters=nothing, helper::Bool=false)
    admitted = false
    lock(st.worker_gate)
    try
        if st.phase[] == 1
            Threads.atomic_add!(st.active_workers, 1)
            admitted = true
        end
    finally
        unlock(st.worker_gate)
    end
    admitted || return false
    if scheduler_counters !== nothing
        Threads.atomic_add!(scheduler_counters.orbit_workers, 1)
        helper && Threads.atomic_add!(scheduler_counters.helper_workers, 1)
    end
    did_work = false
    try
        while st.phase[] == 1 && time_ns() <= st.t_deadline && st.next_orbit[] <= length(st.launch_order)
            next_before = st.next_orbit[]
            try
                _orbit_worker!(st)
                did_work = true
                break
            catch e
                next_after = st.next_orbit[]
                if next_after <= next_before
                    rethrow()
                end
                did_work = true
                println("[ORBIT QUARANTINED] worker=", Threads.threadid(), " helper=", helper, " claimed_progress=", next_after - next_before, " error=", sprint(showerror, e))
                st.phase[] == 1 || break
                time_ns() <= st.t_deadline || break
                st.next_orbit[] <= length(st.launch_order) || break
            end
        end
    finally
        if scheduler_counters !== nothing
            helper && Threads.atomic_add!(scheduler_counters.helper_workers, -1)
            Threads.atomic_add!(scheduler_counters.orbit_workers, -1)
        end
        Threads.atomic_add!(st.active_workers, -1)
    end
    return did_work
end

function _close_orbit_phase!(st::OrbitWorkState; next_phase::Int=2)
    lock(st.worker_gate)
    try
        Threads.atomic_xchg!(st.phase, next_phase)
    finally
        unlock(st.worker_gate)
    end
    while st.active_workers[] > 0
        yield()
    end
    return nothing
end

# Main A-matrix builder: maps orbital weights → Karl observables.
function build_A_matrix_hybrid(Norbit::Int, R_star_m::Vector{Float64}, has_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64},
    verr_star_mps::Vector{Float64}, sini::Float64, rho_s::Float64, r_s::Float64, MBH::Float64, ML::Float64, halo_type::String; stellar_model=nothing,
    surface_brightness_profile=nothing, tracer_constraint_mode="projected_light", nsteps::Int=DEFAULT_NSTEPS, Lfrac::NTuple{5,Float64}=DEFAULT_LFRAC,
    dt_frac_orbit::Float64=DEFAULT_DT_FRAC, max_attempts_factor::Int=DEFAULT_MAX_ATTEMPTS, diag::Bool=false, threaded::Bool=true,
    fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT, regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR, max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP,
    shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS, t_deadline::UInt64=typemax(UInt64), velocity_edges=nothing, light_bin_edges=nothing, kinematic_bin_edges=nothing,
    Nvbin::Int=21, Ntheta_launch::Int=5, halo_q_axis_ratio::Float64=1.0, karl_halo_params=nothing, losvd_target_mode=:current, karl_resolved_kde_grid::Int=DEFAULT_KARL_RESOLVED_KDE_GRID,
    karl_resolved_kde_width_bins::Float64=DEFAULT_KARL_RESOLVED_KDE_WIDTH_BINS, karl_resolved_vmin_kms::Float64=DEFAULT_KARL_RESOLVED_VMIN_KMS, karl_resolved_vmax_kms::Float64=DEFAULT_KARL_RESOLVED_VMAX_KMS,
    karl_resolved_bootstraps::Int=DEFAULT_KARL_RESOLVED_BOOTSTRAPS, karl_resolved_envelope_floor::Float64=DEFAULT_KARL_RESOLVED_ENVELOPE_FLOOR, karl_observables_csv=nothing)
    Nstar = length(R_star_m)
    @assert length(has_vlos) == Nstar
    @assert length(v_star_mps) == Nstar
    @assert length(verr_star_mps) == Nstar
    surface_brightness_profile === nothing && error("surface_brightness_profile is required for Karl-style OSPM; no star-count fallback is allowed")
    Nstar == 0 && return zeros(Float64, 0, Norbit)
    stellar_model_jl = normalize_stellar_model(stellar_model)
    surface_brightness_profile_jl = normalize_surface_brightness_profile(surface_brightness_profile)
    tracer_constraint_mode_sym = _normalize_tracer_constraint_mode(tracer_constraint_mode)
    losvd_target_mode_sym = _normalize_losvd_target_mode(losvd_target_mode)
    karl_observables = _resolve_karl_observables(losvd_target_mode_sym; observables_csv=karl_observables_csv)
    prewarm_stellar_force_cache(stellar_model_jl)
    light_edges_force = light_bin_edges === nothing ? resolve_karl_spatial_edges(kinematic_bin_edges) : resolve_karl_light_edges(light_bin_edges)
    required_force_rmax_m = 1.5 * light_edges_force[end]
    ctx = get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=stellar_model_jl, required_rmax_m=required_force_rmax_m, halo_q_axis_ratio=halo_q_axis_ratio, karl_halo_params=karl_halo_params)
    sini = clamp01(f64(sini))
    Rmin = minimum(R_star_m)
    Rmax = maximum(R_star_m)
    if !(isfinite(Rmin) && isfinite(Rmax) && Rmax > Rmin)
        return zeros(Float64, 0, Norbit)
    end
    st = _init_orbit_work(Norbit, R_star_m, has_vlos, v_star_mps, verr_star_mps, sini, ctx; nsteps=nsteps, Lfrac=Lfrac, dt_frac_orbit=dt_frac_orbit, max_attempts_factor=max_attempts_factor,
        t_deadline=t_deadline, velocity_edges=velocity_edges, light_bin_edges=light_bin_edges, kinematic_bin_edges=kinematic_bin_edges, Nvbin=Nvbin, Ntheta_launch=Ntheta_launch,
        fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, shell_band_count=shell_band_count, tracer_constraint_mode=tracer_constraint_mode_sym, losvd_target_mode=losvd_target_mode_sym, karl_observables=karl_observables)
    Threads.atomic_xchg!(st.phase, 1)
    nworkers = threaded ? Threads.nthreads() : 1
    if threaded && nworkers > 1
        Threads.@threads for t in 1:nworkers
            _run_orbit_worker!(st)
        end
    else
        _run_orbit_worker!(st)
    end
    _close_orbit_phase!(st)
    filled = st.filled_atomic[]
    coverage = _assess_orbit_coverage(st; fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, shell_band_count=shell_band_count)

    if !coverage.accepted
        _print_orbit_failure_diagnostics(st, 0)
        println("[ORBIT COVERAGE WARNING] filled=", filled, " planned=", coverage.planned, " succeeded=", coverage.succeeded, " coverage_fraction=", coverage.coverage_fraction, " reasons=", isempty(coverage.rejection_reasons) ? "none" : join(coverage.rejection_reasons, " | "))
    end

    _orbit_library_usable(st, coverage.successful_columns) || error("Unusable orbit library: no usable compact orbit solution remains after failed orbit columns are removed")

    A_losvd, A_light, A_kinematic = _compact_orbit_matrices(st, coverage.successful_columns)
    wphase_use, phase_diag, _ = _build_compact_karl_wphase(st, coverage.successful_columns)
    A = vcat(A_losvd, A_light)
    if diag
        losvd_target, losvd_sigma, projected_light_target, projected_light_sigma, counts_by_spatial, losvd_counts = _observed_targets_for_mode(R_star_m, has_vlos, v_star_mps, verr_star_mps, st, surface_brightness_profile_jl, losvd_target_mode_sym, karl_observables; karl_resolved_kde_grid=karl_resolved_kde_grid, karl_resolved_kde_width_bins=karl_resolved_kde_width_bins, karl_resolved_vmin_kms=karl_resolved_vmin_kms, karl_resolved_vmax_kms=karl_resolved_vmax_kms, karl_resolved_bootstraps=karl_resolved_bootstraps, karl_resolved_envelope_floor=karl_resolved_envelope_floor)
        light_target, light_sigma = _resolve_tracer_constraint_targets(tracer_constraint_mode_sym, projected_light_target, projected_light_sigma, st.d3_constraints)
        return (
            A,
            Dict(
                "filled" => filled,
                "Nbase_orbit" => st.Nbase_orbit,
                "required" => coverage.required,
                "complete" => coverage.accepted,
                "coverage_fraction" => coverage.coverage_fraction,
                "attempted_fraction" => coverage.attempted_fraction,
                "success_fraction" => coverage.success_fraction,
                "shell_minimum_coverage" => coverage.shell_minimum_coverage,
                "lfrac_minimum_coverage" => coverage.lfrac_minimum_coverage,
                "theta_minimum_coverage" => coverage.theta_minimum_coverage,
                "shell_band_coverage" => coverage.shell_bands,
                "lfrac_coverage" => coverage.lfrac,
                "theta_launch_coverage" => coverage.theta_launches,
                "shell_coverage_gap" => coverage.shell_coverage_gap,
                "lfrac_coverage_gap" => coverage.lfrac_coverage_gap,
                "theta_coverage_gap" => coverage.theta_coverage_gap,
                "joint_cell_count" => coverage.joint_cell_count,
                "joint_cells" => coverage.joint_cells,
                "joint_minimum_coverage" => coverage.joint_minimum_coverage,
                "joint_holes" => coverage.joint_holes,
                "successful_orbit_columns" => length(coverage.successful_columns),
                "paired_orbit_columns" => true,
                "attempts" => sum(st.attempts_used),
                "Nspatial" => st.Nspatial,
                "Nvbin" => st.Nvbin,
                "Nlosvd" => st.Nlosvd,
                "Nlight" => st.Nlight,
                "Nshells" => st.Nshells,
                "shell_max_pc" => st.shells[end] / pc,
                "spatial_edges" => st.spatial_edges,
                "losvd_aperture_ranges" => st.losvd_aperture_ranges,
                "light_edges" => st.light_edges,
                "velocity_edges" => st.velocity_edges,
                "losvd_target" => losvd_target,
                "losvd_sigma" => losvd_sigma,
                "light_target" => light_target,
                "light_sigma" => light_sigma,
                "counts_by_spatial" => counts_by_spatial,
                "losvd_counts" => losvd_counts === nothing ? Int[] : losvd_counts,
                "force_geometry" => String(st.force_geometry),
                "tracer_constraint_mode" => String(tracer_constraint_mode_sym),
                "losvd_target_mode" => String(losvd_target_mode_sym),
                "karl_resolved_kde_grid" => karl_resolved_kde_grid,
                "karl_resolved_kde_width_bins" => karl_resolved_kde_width_bins,
                "karl_resolved_vmin_kms" => karl_resolved_vmin_kms,
                "karl_resolved_vmax_kms" => karl_resolved_vmax_kms,
                "karl_resolved_bootstraps" => karl_resolved_bootstraps,
                "karl_resolved_envelope_floor" => karl_resolved_envelope_floor,
                "wphase" => wphase_use,
                "phase_volume_convention" => string(phase_diag.convention),
                "phase_volume_normalization" => string(phase_diag.normalization),
                "phase_volume_launches_recorded" => phase_diag.launches_recorded,
                "phase_volume_sos_recorded" => phase_diag.sos_recorded,
                "phase_volume_valid_base_orbits" => phase_diag.valid_base_orbits,
                "phase_volume_nested_groups" => phase_diag.nested_groups,
                "phase_volume_duplicate_area_clusters" => phase_diag.duplicate_area_clusters,
                "phase_volume_duplicate_area_orbits" => phase_diag.duplicate_area_orbits,
                "raw_phase_volume_min" => phase_diag.raw_phase_volume_min,
                "raw_phase_volume_max" => phase_diag.raw_phase_volume_max,
                "raw_phase_volume_dynamic_range" => phase_diag.raw_phase_volume_dynamic_range,
                "normalized_phase_volume_min" => phase_diag.normalized_phase_volume_min,
                "normalized_phase_volume_max" => phase_diag.normalized_phase_volume_max,
                "wphase_min" => phase_diag.wphase_min,
                "wphase_max" => phase_diag.wphase_max,
            ),
        )
    end

    return A
end

# Batch evaluator: Karl-style binned LOSVD + selectable projected-light or 3-D tracer-density constraint.
# This is the Heart of the whole Pipeline
# and is where all the parallelism is implemented
function evaluate_batch_theta(thetas::AbstractMatrix{<:Real}, R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, sini::Float64, Norbit::Int, halo_type::String; stellar_model=nothing, surface_brightness_profile=nothing, tracer_constraint_mode="projected_light", alphat::Float64=DEFAULT_KARL_ALPHAT, apfac::Float64=DEFAULT_KARL_APFAC, light_rel_tol::Float64=DEFAULT_KARL_LIGHT_REL_TOL, light_sigma_tol::Float64=2.0, delta_chi2_iter_tol::Float64=DEFAULT_KARL_DELTA_CHI2_ITER_TOL, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, maxiter::Int=DEFAULT_KARL_MAXITER, timeout_s::Float64=120.0, fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT, regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR, max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP, shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS, coverage_check_every::Int=DEFAULT_ORBIT_COVERAGE_CHECK_EVERY, warn_fill_pct::Float64=DEFAULT_ORBIT_WARN_FILL_PCT, warn_success_pct::Float64=DEFAULT_ORBIT_WARN_SUCCESS_PCT, warn_regional_floor::Float64=DEFAULT_ORBIT_WARN_REGIONAL_FLOOR, warn_max_regional_gap::Float64=DEFAULT_ORBIT_WARN_MAX_REGIONAL_GAP, model_owner_limit::Int=0, threads_per_model::Int=2, R_inner_pc::Float64=30.0, velocity_edges=nothing, kinematic_bin_edges=nothing, light_bin_edges=nothing, Nvbin::Int=21, Ntheta_launch::Int=5, halo_q_axis_ratio::Float64=1.0, karl_halo_params=nothing, losvd_target_mode=:current, karl_resolved_kde_grid::Int=DEFAULT_KARL_RESOLVED_KDE_GRID, karl_resolved_kde_width_bins::Float64=DEFAULT_KARL_RESOLVED_KDE_WIDTH_BINS, karl_resolved_vmin_kms::Float64=DEFAULT_KARL_RESOLVED_VMIN_KMS, karl_resolved_vmax_kms::Float64=DEFAULT_KARL_RESOLVED_VMAX_KMS, karl_resolved_bootstraps::Int=DEFAULT_KARL_RESOLVED_BOOTSTRAPS, karl_resolved_envelope_floor::Float64=DEFAULT_KARL_RESOLVED_ENVELOPE_FLOOR, karl_observables_csv=nothing, losvd_conditioning=:vlos_cut, losvd_fit_statistic=nothing, delta_statistic_iter_tol::Float64=delta_chi2_iter_tol)
    nrow, nbatch = size(thetas)
    surface_brightness_profile === nothing && error("surface_brightness_profile is required for Karl-style OSPM; no star-count fallback is allowed")
    apfac > 0.0 || error("apfac must be positive")
    light_rel_tol > 0.0 || error("light_rel_tol must be positive")
    light_sigma_tol > 0.0 || error("light_sigma_tol must be positive")
    delta_chi2_iter_tol >= 0.0 || error("delta_chi2_iter_tol must be nonnegative")
    delta_statistic_iter_tol >= 0.0 || error("delta_statistic_iter_tol must be nonnegative")
    threads_per_model > 0 || error("threads_per_model must be positive")
    losvd_target_mode_sym = _normalize_losvd_target_mode(losvd_target_mode)
    losvd_conditioning_sym = _normalize_losvd_conditioning(losvd_conditioning)
    losvd_fit_statistic_sym = _normalize_losvd_fit_statistic(losvd_fit_statistic, losvd_target_mode_sym)
    karl_observables = _resolve_karl_observables(losvd_target_mode_sym; observables_csv=karl_observables_csv)
    if losvd_target_mode_sym === :karl_resolved_stars
        velocity_edges === nothing && error("Karl resolved LOSVD requires explicit velocity_edges defining the selected sample window")
        karl_resolved_kde_grid > 1 || error("karl_resolved_kde_grid must exceed one")
        isfinite(karl_resolved_kde_width_bins) && karl_resolved_kde_width_bins > 0.0 || error("karl_resolved_kde_width_bins must be finite and positive")
        isfinite(karl_resolved_vmin_kms) && isfinite(karl_resolved_vmax_kms) && karl_resolved_vmax_kms > karl_resolved_vmin_kms || error("Karl resolved LOSVD velocity bounds are invalid")
        karl_resolved_bootstraps > 1 || error("karl_resolved_bootstraps must exceed one")
        isfinite(karl_resolved_envelope_floor) && karl_resolved_envelope_floor >= 0.0 || error("karl_resolved_envelope_floor must be finite and nonnegative")
    end
    BLAS.set_num_threads(nbatch == 1 ? Threads.nthreads() : 1) # set to highest for a single model test change for multi-model runs
    allocated_threads = tryparse(Int, get(ENV, "SLURM_CPUS_PER_TASK", ""))
    if allocated_threads !== nothing &&
    Threads.nthreads() != allocated_threads
        error("Julia thread mismatch: " * "Threads.nthreads()=$(Threads.nthreads()) " * "but SLURM_CPUS_PER_TASK=$allocated_threads")
    end

    println(
        "[RUNTIME CONTRACT]",
        " host=", gethostname(),
        " julia_version=", VERSION,
        " julia_threads=", Threads.nthreads(),
        " blas_threads=", BLAS.get_num_threads(),
        " slurm_cpus=", get(ENV, "SLURM_CPUS_PER_TASK", "local"),
        " threads_per_model=", threads_per_model,
        " model_owner_limit_requested=", model_owner_limit,
        " losvd_target_mode=", losvd_target_mode_sym,
        " losvd_fit_statistic=", losvd_fit_statistic_sym,
        " losvd_conditioning=", losvd_conditioning_sym)
    stellar_model_jl = normalize_stellar_model(stellar_model)
    surface_brightness_profile_jl = normalize_surface_brightness_profile(surface_brightness_profile)
    tracer_constraint_mode_sym = _normalize_tracer_constraint_mode(tracer_constraint_mode)
    prewarm_stellar_force_cache(stellar_model_jl)
    light_edges_force = light_bin_edges === nothing ? resolve_karl_spatial_edges(kinematic_bin_edges) : resolve_karl_light_edges(light_bin_edges)
    required_force_rmax_m = 1.5 * light_edges_force[end]
    status = fill(4, nbatch)
    chi2_losvd = fill(Inf, nbatch)
    chi2_inner = fill(Inf, nbatch)
    chi2_outer = fill(Inf, nbatch)
    delta_chi2_iteration = fill(Inf, nbatch)
    max_light_relative_residual = fill(Inf, nbatch)
    max_light_sigma_residual = fill(Inf, nbatch)
    light_constraint_ok = fill(false, nbatch)
    solver_converged = fill(false, nbatch)
    solver_iterations = zeros(Int, nbatch)
    solver_failure_reason = fill("not_run", nbatch)
    N_inner = zeros(Int, nbatch)
    N_outer = zeros(Int, nbatch)
    N_nonzero_weights = zeros(Int, nbatch)
    effective_N_orbits = zeros(Float64, nbatch)
    max_weight_fraction = zeros(Float64, nbatch)
    coverage_status = fill("not_assessed", nbatch)
    coverage_issue_region = fill("none", nbatch)
    coverage_issue_axis = fill("none", nbatch)
    coverage_issue_shell_bands = fill("", nbatch)
    coverage_reasons = fill("", nbatch)
    coverage_fraction = zeros(Float64, nbatch)
    coverage_attempted_fraction = zeros(Float64, nbatch)
    coverage_success_fraction = zeros(Float64, nbatch)
    coverage_shell_min = zeros(Float64, nbatch)
    coverage_lfrac_min = zeros(Float64, nbatch)
    coverage_theta_min = zeros(Float64, nbatch)
    coverage_shell_gap = zeros(Float64, nbatch)
    coverage_lfrac_gap = zeros(Float64, nbatch)
    coverage_theta_gap = zeros(Float64, nbatch)
    coverage_joint_holes = zeros(Int, nbatch)
    coverage_deadline_hit = fill(false, nbatch)
    successful_base_orbits = zeros(Int, nbatch)
    planned_base_orbits = fill(Norbit ÷ 2, nbatch)
    phase_volume_valid = fill(false, nbatch)
    phase_volume_convention = fill("not_computed", nbatch)
    phase_volume_normalization = fill("not_computed", nbatch)
    phase_volume_launches_recorded = zeros(Int, nbatch)
    phase_volume_sos_recorded = zeros(Int, nbatch)
    phase_volume_valid_base_orbits = zeros(Int, nbatch)
    phase_volume_invalid_recorded_orbits = zeros(Int, nbatch)
    phase_volume_nested_groups = zeros(Int, nbatch)
    phase_volume_duplicate_area_clusters = zeros(Int, nbatch)
    phase_volume_duplicate_area_orbits = zeros(Int, nbatch)
    raw_phase_volume_min = fill(NaN, nbatch)
    raw_phase_volume_max = fill(NaN, nbatch)
    raw_phase_volume_dynamic_range = fill(NaN, nbatch)
    normalized_phase_volume_min = fill(NaN, nbatch)
    normalized_phase_volume_max = fill(NaN, nbatch)
    wphase_min = fill(NaN, nbatch)
    wphase_max = fill(NaN, nbatch)
    wphase_dynamic_range = fill(NaN, nbatch)
    wphase_pair_max_relative_mismatch = fill(NaN, nbatch)
    work_states = Vector{Union{Nothing, OrbitWorkState}}(undef, nbatch)
    fill!(work_states, nothing)
    work_states_lock = ReentrantLock()
    next_theta = Threads.Atomic{Int}(1)
    nthreads = Threads.nthreads()

    workers_per_model = min(threads_per_model, nthreads)
    max_parallel_models = max(1, fld(nthreads, workers_per_model))
    owner_limit = model_owner_limit > 0 ? min(model_owner_limit, nbatch, max_parallel_models) : min(nbatch, max_parallel_models)

    scheduler_counters = (orbit_models=Threads.Atomic{Int}(0), orbit_workers=Threads.Atomic{Int}(0), helper_workers=Threads.Atomic{Int}(0), weight_models=Threads.Atomic{Int}(0), active_model_owners=Threads.Atomic{Int}(0), completed_models=Threads.Atomic{Int}(0), stop_monitor=Threads.Atomic{Int}(0))
    scheduler_started_ns = time_ns()
    println("[SCHED] julia_threads=", nthreads, " workers_per_model=", workers_per_model, " max_parallel_models=", max_parallel_models, " active_model_limit=", owner_limit)
    if get(ENV, "OSPM_DIAG_JULIA_HANDOFF", "0") == "1"
        if losvd_target_mode_sym === :karl_resolved_stars
            println("[JULIA OBSERVABLES] mode=", losvd_target_mode_sym, " Nvbin=", Nvbin, " kde_grid=", karl_resolved_kde_grid, " kde_width_bins=", karl_resolved_kde_width_bins, " vmin_kms=", karl_resolved_vmin_kms, " vmax_kms=", karl_resolved_vmax_kms)
        elseif losvd_target_mode_sym === :karl_mode0_observables
            println("[JULIA OBSERVABLES] mode=", losvd_target_mode_sym, " Nvbin=", Nvbin, " csv=", karl_observables_csv)
        else
            println("[JULIA OBSERVABLES] mode=", losvd_target_mode_sym, " Nvbin=", Nvbin)
        end
    end

    function _print_scheduler_diagnostics!()
        get(ENV, "OSPM_DIAG_SCHEDULER", "0") == "1" || return nothing
        claimed = clamp(next_theta[] - 1, 0, nbatch)
        completed = scheduler_counters.completed_models[]
        orbit_models = scheduler_counters.orbit_models[]
        weight_models = scheduler_counters.weight_models[]
        other_models = max(0, claimed - completed - orbit_models - weight_models)
        elapsed_s = (time_ns() - scheduler_started_ns) / 1e9
        println("[SCHED DIAG] elapsed_s=", round(elapsed_s; digits=1), " claimed=", claimed, " queued=", nbatch - claimed, " completed=", completed, " orbit_models=", orbit_models, " orbit_workers=", scheduler_counters.orbit_workers[], " helper_workers=", scheduler_counters.helper_workers[], " weight_models=", weight_models, " active_model_owners=", scheduler_counters.active_model_owners[], " workers_per_model=", workers_per_model, " max_parallel_models=", max_parallel_models, " active_model_limit=", owner_limit, " other_models=", other_models, " julia_threads=", nthreads)
        return nothing
    end
    function _store_weight_diagnostics!(i::Int, w_best::Vector{Float64})
        wsum = sum(w_best)
        wmin = isempty(w_best) ? NaN : minimum(w_best)
        wmax = isempty(w_best) ? NaN : maximum(w_best)
        nneg = count(x -> x < 0.0, w_best)
        nbad = count(x -> !isfinite(x), w_best)
        if isfinite(wsum) && wsum > 0.0
            pwt = w_best ./ wsum
            pmin = isempty(pwt) ? NaN : minimum(pwt)
            pmax = isempty(pwt) ? NaN : maximum(pwt)
            N_nonzero_weights[i] = count(pwt .> 1e-12)
            effective_N_orbits[i] = 1.0 / sum(pwt .^ 2)
            max_weight_fraction[i] = maximum(pwt)
            if nneg > 0 || nbad > 0 || pmax > 1.0 || pmin < 0.0
                println("[WEIGHT DEBUG] i=", i, " wsum=", wsum, " wmin=", wmin, " wmax=", wmax, " nneg=", nneg, " nbad=", nbad, " pmin=", pmin, " pmax=", pmax, " N_nonzero=", N_nonzero_weights[i], " Neff=", effective_N_orbits[i])
            end
        else
            println("[WEIGHT DEBUG] i=", i, " BAD SUM wsum=", wsum, " wmin=", wmin, " wmax=", wmax, " nneg=", nneg, " nbad=", nbad)
        end
        return nothing
    end
    function _wdiag_value(wdiag, field::Symbol, default)
        wdiag === nothing && return default
        field in propertynames(wdiag) || return default
        return getproperty(wdiag, field)
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

    function _store_solver_diagnostics!(i::Int, wdiag)
        delta_chi2_iteration[i] = Float64(_wdiag_value(wdiag, :delta_chi2_iteration, Inf))
        max_light_relative_residual[i] = Float64(_wdiag_value(wdiag, :max_light_relative_residual, Inf))
        max_light_sigma_residual[i] = Float64(_wdiag_value(wdiag, :max_light_sigma_residual, Inf))
        light_constraint_ok[i] = Bool(_wdiag_value(wdiag, :light_constraint_ok, false))
        solver_converged[i] = Bool(_wdiag_value(wdiag, :solver_converged, false))
        solver_iterations[i] = Int(_wdiag_value(wdiag, :iterations, 0))
        solver_failure_reason[i] = string(_wdiag_value(wdiag, :failure_reason, :missing_diagnostics))
        return nothing
    end
    function _print_karl_diagnostics!(i::Int, tid::Int, wdiag, chi2_score::Float64)
        wdiag === nothing && return nothing
        println("[WEIGHT RESULT]",
            " i=", i,
            " tid=", tid,
            " converged=", _wdiag_value(wdiag, :solver_converged, false),
            " iterations=", _wdiag_value(wdiag, :iterations, 0),
            " chi2=", chi2_score,
            " fit_statistic=", _wdiag_value(wdiag, :fit_statistic, :legacy_chi2),
            " conditioning=", _wdiag_value(wdiag, :conditioning, :none),
            " delta_chi2=", _wdiag_value(wdiag, :delta_chi2_iteration, NaN),
            " light_rel=", _wdiag_value(wdiag, :max_light_relative_residual, NaN),
            " light_sigma=", _wdiag_value(wdiag, :max_light_sigma_residual, NaN),
            " zero_target_floor_rows=", _wdiag_value(wdiag, :zero_target_floor_rows, 0),
            " zero_target_leakage_rel=", _wdiag_value(wdiag, :max_zero_target_leakage_rel, 0.0),
            " worst_zero_target_bin=", _wdiag_value(wdiag, :worst_zero_target_bin, 0),
            " N_nonzero=", N_nonzero_weights[i],
            " Neff=", effective_N_orbits[i],
            " max_weight=", max_weight_fraction[i])
        if get(ENV, "OSPM_DIAG_SOLVER", "0") == "1"
            println("[WEIGHT RESULT DETAIL] i=", i, " tid=", tid, " chi_losvd_solver=", _wdiag_value(wdiag, :chi_losvd, NaN), " entropy=", _wdiag_value(wdiag, :entropy, NaN), " profit=", _wdiag_value(wdiag, :profit, NaN), " losvd_penalty=", _wdiag_value(wdiag, :losvd_penalty, NaN), " chi_slack=", _wdiag_value(wdiag, :chi_slack, NaN), " slack_to_losvd=", _wdiag_value(wdiag, :slack_to_losvd, NaN), " slack_l2=", _wdiag_value(wdiag, :slack_l2, NaN), " slack_max_abs=", _wdiag_value(wdiag, :slack_max_abs, NaN), " rcond_est=", _wdiag_value(wdiag, :rcond_est, NaN), " max_abs_dw=", _wdiag_value(wdiag, :max_abs_dw, NaN), " stepfac=", _wdiag_value(wdiag, :stepfac, NaN), " constraint_ok=", _wdiag_value(wdiag, :constraint_ok, false), " slack_consistent=", _wdiag_value(wdiag, :slack_consistent, false), " normalized=", _wdiag_value(wdiag, :normalized, false), " failure_reason=", _wdiag_value(wdiag, :failure_reason, :none), " wphase_convention=", _wdiag_value(wdiag, :wphase_convention, :missing), " wphase_dynamic_range=", _wdiag_value(wdiag, :wphase_dynamic_range, NaN), " phase_volume_dynamic_range=", _wdiag_value(wdiag, :phase_volume_dynamic_range, NaN), " wphase_pair_max_relative_mismatch=", _wdiag_value(wdiag, :wphase_pair_max_relative_mismatch, NaN))
        end
        return nothing
    end

    function _print_karl_failure!(i::Int, tid::Int, wdiag)
        println("[KARL SOLVER FAIL] i=", i, " tid=", tid, " failure_reason=", _wdiag_value(wdiag, :failure_reason, :missing_diagnostics), " delta_chi2_iteration=", _wdiag_value(wdiag, :delta_chi2_iteration, NaN), " max_light_relative_residual=", _wdiag_value(wdiag, :max_light_relative_residual, NaN), " max_light_sigma_residual=", _wdiag_value(wdiag, :max_light_sigma_residual, NaN), " light_constraint_ok=", _wdiag_value(wdiag, :light_constraint_ok, false), " solver_converged=", _wdiag_value(wdiag, :solver_converged, false), " iterations=", _wdiag_value(wdiag, :iterations, 0), " rcond_est=", _wdiag_value(wdiag, :rcond_est, NaN), " max_abs_dw=", _wdiag_value(wdiag, :max_abs_dw, NaN), " stepfac=", _wdiag_value(wdiag, :stepfac, NaN), " chi_slack=", _wdiag_value(wdiag, :chi_slack, NaN))
        return nothing
    end
    function _try_claim_theta!()
        next_theta[] > nbatch && return 0
        while true
            active = scheduler_counters.active_model_owners[]
            active >= owner_limit && return 0
            Threads.atomic_cas!(scheduler_counters.active_model_owners, active, active + 1) == active || continue
            i = Threads.atomic_add!(next_theta, 1)
            if i > nbatch
                Threads.atomic_add!(scheduler_counters.active_model_owners, -1)
                return 0
            end
            return i
        end
    end
    function _batch_worker!(tid::Int)
        while true
            i = _try_claim_theta!()
            if i > 0
                orbit_owner_slot_held = true
                theta_deadline = time_ns() + UInt64(round(timeout_s * 1e9))
                try
                    rho_s = Float64(thetas[1, i])
                    r_s = Float64(thetas[2, i])
                    MBH = nrow >= 3 ? Float64(thetas[3, i]) : 0.0
                    ML = nrow >= 4 ? Float64(thetas[4, i]) : 1.0
                    if isempty(R_star_m)
                        status[i] = 1
                        solver_failure_reason[i] = "empty_stellar_sample"
                        continue
                    end
                    Rmin_v = minimum(R_star_m)
                    Rmax_v = maximum(R_star_m)
                    if !(isfinite(Rmin_v) && isfinite(Rmax_v) && Rmax_v > Rmin_v)
                        status[i] = 1
                        solver_failure_reason[i] = "invalid_stellar_radius_range"
                        continue
                    end
                    ctx = get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=stellar_model_jl, required_rmax_m=required_force_rmax_m, halo_q_axis_ratio=halo_q_axis_ratio, karl_halo_params=karl_halo_params)
                    ws = _init_orbit_work(Norbit, R_star_m, valid_vlos, v_star_mps, verr_star_mps, sini, ctx; nsteps=DEFAULT_NSTEPS, Lfrac=DEFAULT_LFRAC, dt_frac_orbit=DEFAULT_DT_FRAC, max_attempts_factor=DEFAULT_MAX_ATTEMPTS, t_deadline=theta_deadline, velocity_edges=velocity_edges, light_bin_edges=light_bin_edges, kinematic_bin_edges=kinematic_bin_edges, Nvbin=Nvbin, Ntheta_launch=Ntheta_launch, fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, shell_band_count=shell_band_count, coverage_check_every=coverage_check_every, tracer_constraint_mode=tracer_constraint_mode_sym, losvd_target_mode=losvd_target_mode_sym, karl_observables=karl_observables)
                    lock(work_states_lock)
                    try
                        work_states[i] = ws
                    finally
                        unlock(work_states_lock)
                    end
                    planned_base_orbits[i] = length(ws.launch_order)
                    if i == 1
                        println("[MODEL GRID] tracer_bins=", ws.Nlight, " apertures=", ws.Nspatial, " Nvbin=", ws.Nvbin, " shells=", ws.Nshells, " constraints=", ws.Nlight + ws.Nspatial * ws.Nvbin, " R_tracer_max_pc=", (ws.d3_constraints === nothing ? ws.light_edges[end] : ws.d3_constraints.radial_edges_m[end]) / pc, " R_aperture_max_pc=", maximum(last.(ws.losvd_aperture_ranges)) / pc, " R_shell_max_pc=", ws.shells[end] / pc)
                    end

                    Threads.atomic_add!(scheduler_counters.orbit_models, 1)
                    try
                        Threads.atomic_xchg!(ws.phase, 1)

                        @sync for _ in 1:workers_per_model
                            Threads.@spawn _run_orbit_worker!(ws; scheduler_counters=scheduler_counters)
                        end

                        _close_orbit_phase!(ws)
                    finally
                        Threads.atomic_add!(scheduler_counters.orbit_models, -1)
                        if orbit_owner_slot_held
                            Threads.atomic_add!(scheduler_counters.active_model_owners, -1)
                            orbit_owner_slot_held = false
                        end
                    end

                    coverage = _assess_orbit_coverage(ws; fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, shell_band_count=shell_band_count)
                    coverage_deadline_hit[i] = time_ns() > theta_deadline
                    successful_base_orbits[i] = coverage.succeeded
                    coverage_fraction[i] = coverage.coverage_fraction
                    coverage_attempted_fraction[i] = coverage.attempted_fraction
                    coverage_success_fraction[i] = coverage.success_fraction
                    coverage_shell_min[i] = coverage.shell_minimum_coverage
                    coverage_lfrac_min[i] = coverage.lfrac_minimum_coverage
                    coverage_theta_min[i] = coverage.theta_minimum_coverage
                    coverage_shell_gap[i] = coverage.shell_coverage_gap
                    coverage_lfrac_gap[i] = coverage.lfrac_coverage_gap
                    coverage_theta_gap[i] = coverage.theta_coverage_gap
                    coverage_joint_holes[i] = length(coverage.joint_holes)
                    coverage_meta = _coverage_metadata(coverage; fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, warn_fill_pct=warn_fill_pct, warn_success_pct=warn_success_pct, warn_regional_floor=warn_regional_floor, warn_max_regional_gap=warn_max_regional_gap)
                    coverage_status[i] = coverage_meta.status
                    coverage_issue_region[i] = coverage_meta.issue_region
                    coverage_issue_axis[i] = coverage_meta.issue_axis
                    coverage_issue_shell_bands[i] = coverage_meta.issue_shell_bands
                    coverage_reasons[i] = coverage_meta.reasons
                    if !coverage.accepted
                        _print_orbit_failure_diagnostics(ws, i)

                        println("[ORBIT COVERAGE WARNING] i=", i, " region=", coverage_meta.issue_region, " axis=", coverage_meta.issue_axis, " shell_bands=", isempty(coverage_meta.issue_shell_bands) ? "none" : coverage_meta.issue_shell_bands,
                        " succeeded=", coverage.succeeded, " attempted=", coverage.attempted, " planned=", coverage.planned, " required=", coverage.required, " coverage_fraction=", coverage.coverage_fraction, " attempted_fraction=",
                        coverage.attempted_fraction, " success_fraction=", coverage.success_fraction, " shell_min=", coverage.shell_minimum_coverage, " lfrac_min=", coverage.lfrac_minimum_coverage, " theta_min=", coverage.theta_minimum_coverage,
                        " shell_gap=", coverage.shell_coverage_gap, " lfrac_gap=", coverage.lfrac_coverage_gap, " theta_gap=", coverage.theta_coverage_gap, " joint_holes=", length(coverage.joint_holes), " deadline_hit=", coverage_deadline_hit[i],
                        " reasons=", isempty(coverage_meta.reasons) ? "none" : coverage_meta.reasons)
                    end
                    if !_orbit_library_usable(ws, coverage.successful_columns)
                        status[i] = 1
                        solver_failure_reason[i] = "unusable_orbit_library"
                        println("[ORBIT LIBRARY REJECTED] i=", i, " successful_base_orbits=", coverage.succeeded, " successful_columns=", length(coverage.successful_columns), " reason=no_usable_compact_orbit_solution")
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end
                    A_losvd, A_light, A_kinematic = _compact_orbit_matrices(ws, coverage.successful_columns)
                    wphase_use = Float64[]
                    phase_diag = nothing
                    phase_result = nothing
                    try
                        wphase_use, phase_diag, phase_result = _build_compact_karl_wphase(ws, coverage.successful_columns,)
                    catch phase_error
                        status[i] = 1
                        solver_failure_reason[i] = "phase_volume_failed"
                        println("[KARL PHASE VOLUME FAIL]", " i=", i, " error=", sprint(showerror, phase_error))
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end
                    phase_volume_valid[i] = true
                    phase_volume_convention[i] = string(phase_diag.convention)
                    phase_volume_normalization[i] = string(phase_diag.normalization)
                    phase_volume_launches_recorded[i] = Int(phase_diag.launches_recorded)
                    phase_volume_sos_recorded[i] = Int(phase_diag.sos_recorded)
                    phase_volume_valid_base_orbits[i] = Int(phase_diag.valid_base_orbits)
                    phase_volume_invalid_recorded_orbits[i] = Int(_wdiag_value(phase_diag, :invalid_recorded_orbits, 0))
                    phase_volume_nested_groups[i] = Int(phase_diag.nested_groups)
                    phase_volume_duplicate_area_clusters[i] = Int(phase_diag.duplicate_area_clusters)
                    phase_volume_duplicate_area_orbits[i] = Int(phase_diag.duplicate_area_orbits)
                    raw_phase_volume_min[i] = Float64(phase_diag.raw_phase_volume_min)
                    raw_phase_volume_max[i] = Float64(phase_diag.raw_phase_volume_max)
                    raw_phase_volume_dynamic_range[i] = Float64(phase_diag.raw_phase_volume_dynamic_range)
                    normalized_phase_volume_min[i] = Float64(phase_diag.normalized_phase_volume_min)
                    normalized_phase_volume_max[i] = Float64(phase_diag.normalized_phase_volume_max)
                    wphase_min[i] = Float64(phase_diag.wphase_min)
                    wphase_max[i] = Float64(phase_diag.wphase_max)
                    wphase_dynamic_range[i] = isfinite(wphase_min[i]) && wphase_min[i] > 0.0 && isfinite(wphase_max[i]) ? wphase_max[i] / wphase_min[i] : NaN
                    pair_max_relative_mismatch = 0.0
                    @inbounds for j in 1:2:length(wphase_use)
                        wpro = wphase_use[j]
                        wret = wphase_use[j + 1]
                        pair_scale = max(abs(wpro), abs(wret), eps(Float64))
                        pair_max_relative_mismatch = max(pair_max_relative_mismatch, abs(wpro - wret) / pair_scale)
                    end
                    wphase_pair_max_relative_mismatch[i] = pair_max_relative_mismatch
                    i == 1 && _print_phase_volume_diagnostics!(i, phase_diag)
                    if size(A_losvd, 1) == 0 || size(A_losvd, 2) == 0 || !all(isfinite, A_losvd) || size(A_light, 1) == 0 || size(A_light, 2) == 0 || !all(isfinite, A_light)
                        status[i] = 1
                        solver_failure_reason[i] = "invalid_observable_matrix"
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end
                    losvd_target, losvd_sigma, projected_light_target, projected_light_sigma, counts_by_spatial, losvd_counts = _observed_targets_for_mode(R_star_m, valid_vlos, v_star_mps, verr_star_mps, ws,
                    surface_brightness_profile_jl, losvd_target_mode_sym, karl_observables; karl_resolved_kde_grid=karl_resolved_kde_grid, karl_resolved_kde_width_bins=karl_resolved_kde_width_bins,
                    karl_resolved_vmin_kms=karl_resolved_vmin_kms, karl_resolved_vmax_kms=karl_resolved_vmax_kms, karl_resolved_bootstraps=karl_resolved_bootstraps, karl_resolved_envelope_floor=karl_resolved_envelope_floor)
                    light_target, light_sigma = _resolve_tracer_constraint_targets(tracer_constraint_mode_sym, projected_light_target, projected_light_sigma, ws.d3_constraints)
                    light_fit_mask = trues(length(light_target))
                    any(light_fit_mask) || error("No tracer-constraint bins are available")
                    A_light_fit = A_light
                    light_target_fit = light_target
                    light_sigma_fit = light_sigma
                    if i == 1
                        tracer_rmax_m = ws.d3_constraints === nothing ? ws.light_edges[end] : ws.d3_constraints.radial_edges_m[end]
                        println("[TRACER CONFIG]",
                            " mode=", tracer_constraint_mode_sym,
                            " bins=", length(light_target_fit),
                            " Rmax_pc=", tracer_rmax_m / pc,
                            " target_sum=", sum(light_target_fit))
                        if get(ENV, "OSPM_DIAG_TRACER", "0") == "1"
                            if ws.d3_constraints === nothing
                                println("[TRACER CONFIG DETAIL]", " projected_target_l1=", sum(abs.(light_target_fit .- projected_light_target)), " sigma_min=", minimum(light_sigma_fit), " sigma_max=", maximum(light_sigma_fit), " R_kin_max_pc=", maximum(last.(ws.losvd_aperture_ranges)) / pc)
                            else
                                d3_radial_target = karl_d3_radial_target(ws.d3_constraints)
                                println("[TRACER CONFIG DETAIL]", " radial_target_l1_vs_projected=", sum(abs.(d3_radial_target .- projected_light_target)), " radial_bins=", ws.d3_constraints.nradial, " angular_bins=", ws.d3_constraints.nangular, " inner_theta_collapsed=true", " sigma_min=", minimum(light_sigma_fit), " sigma_max=", maximum(light_sigma_fit), " R_kin_max_pc=", maximum(last.(ws.losvd_aperture_ranges)) / pc)
                            end
                        end
                    end
                    Threads.atomic_add!(scheduler_counters.weight_models, 1)
                    w = Float64[]
                    ok = false
                    wdiag = nothing
                    try
                        if losvd_fit_statistic_sym === :multinomial
                            losvd_counts === nothing && error("multinomial LOSVD fit requires resolved-star integer counts")
                            w, ok, wdiag = solve_weights_karl_multinomial(A_light_fit, A_losvd, A_kinematic, light_target_fit, light_sigma_fit, losvd_counts; Nspatial=ws.Nspatial, Nvbin=ws.Nvbin, conditioning=losvd_conditioning_sym, alphat=alphat, light_rel_tol=light_rel_tol, light_sigma_tol=light_sigma_tol, delta_statistic_iter_tol=delta_statistic_iter_tol, wphase=wphase_use, maxiter=maxiter, seed=UInt(i), entropy_floor=entropy_floor, apfac=apfac, return_diag=true)
                        else
                            w, ok, wdiag = solve_weights_karl_expanded_cm(A_light_fit, A_losvd, light_target_fit, light_sigma_fit, losvd_target, losvd_sigma; Nspatial=ws.Nspatial, Nvbin=ws.Nvbin, alphat=alphat, light_rel_tol=light_rel_tol, light_sigma_tol=light_sigma_tol, delta_chi2_iter_tol=delta_chi2_iter_tol, wphase=wphase_use, maxiter=maxiter, seed=UInt(i), entropy_floor=entropy_floor, apfac=apfac, return_diag=true)
                        end
                    finally
                        Threads.atomic_add!(scheduler_counters.weight_models, -1)
                    end

                _store_solver_diagnostics!(i, wdiag)
                if length(w) == size(A_light_fit, 2) && all(isfinite, w)
                    print_tracer_bins = get(ENV, "OSPM_DIAG_TRACER", "0") == "1"
                    _print_tracer_bin_diagnostics(A_light_fit, w, light_target_fit, light_sigma_fit, ws.light_edges; model_index=i, print_bins=print_tracer_bins, d3_constraints=ws.d3_constraints)
                end
                finite_weight_solution = length(w) == size(A_losvd, 2) && all(isfinite, w)
                cl = Inf
                losvd_window_diag = nothing

                if finite_weight_solution
                    print_window_bins = get(ENV, "OSPM_DIAG_LOSVD_WINDOW", "0") == "1"
                    losvd_window_diag = _print_losvd_window_diagnostics(A_losvd, A_kinematic, w, ws.losvd_aperture_ranges, ws.velocity_edges, ws.Nvbin; model_index=i, print_bins=print_window_bins)

                    losvd_score_state = losvd_fit_statistic_sym === :multinomial ? karl_multinomial_losvd_state(A_losvd, A_kinematic, w, losvd_counts, ws.Nspatial, ws.Nvbin; conditioning=losvd_conditioning_sym) : karl_losvd_fracnew_state(A_losvd, w, losvd_target, losvd_sigma, ws.Nspatial, ws.Nvbin)
                    cl = losvd_fit_statistic_sym === :multinomial ? losvd_score_state.deviance_total : losvd_score_state.chi_total
                    score_by_spatial = losvd_fit_statistic_sym === :multinomial ? losvd_score_state.deviance_by_spatial : losvd_score_state.chi_by_spatial
                    chi2_losvd[i] = cl

                    R_inner_m = R_inner_pc * pc
                    inner_chi = 0.0
                    outer_chi = 0.0
                    ninner = 0
                    nouter = 0

                    @inbounds for ib in 1:ws.Nspatial
                        rmid = 0.5 * (ws.losvd_aperture_ranges[ib][1] + ws.losvd_aperture_ranges[ib][2])

                        if rmid < R_inner_m
                            inner_chi += score_by_spatial[ib]
                            ninner += Int(round(counts_by_spatial[ib]))
                        else
                            outer_chi += score_by_spatial[ib]
                            nouter += Int(round(counts_by_spatial[ib]))
                        end
                    end

                    chi2_inner[i] = inner_chi
                    chi2_outer[i] = outer_chi
                    N_inner[i] = ninner
                    N_outer[i] = nouter

                    _store_weight_diagnostics!(i, w)
                    if get(ENV, "OSPM_DIAG_ORBIT_FAMILIES", "0") == "1"
                        _print_orbit_weight_family_diagnostics(ws, coverage.successful_columns, w; model_index=i, inner_radius_pc=R_inner_pc)
                        _print_orbit_phase_volume_diagnostics(ws, coverage.successful_columns, w, wphase_use, phase_result; model_index=i, inner_radius_pc=R_inner_pc)
                        target_lfrac = parse(Float64, get(ENV, "OSPM_DIAG_PROJECTION_LFRAC", "0.2"))
                        target_third_u = parse(Float64, get(ENV, "OSPM_DIAG_PROJECTION_THIRD_U", "0.5"))
                        target_aperture = parse(Int, get(ENV, "OSPM_DIAG_PROJECTION_APERTURE", "1"))
                        _print_orbit_projection_diagnostics(ws, coverage.successful_columns, w, A_losvd, A_kinematic, losvd_target, losvd_sigma; model_index=i, inner_radius_pc=R_inner_pc, target_lfrac=target_lfrac, target_third_u=target_third_u, target_aperture=target_aperture)
                    end
                end
                if !ok
                    if length(w) == size(A_light_fit, 2) && all(isfinite, w)
                        light_model_fit = A_light_fit * w
                        light_relative_fit = abs.(light_model_fit .- light_target_fit) ./ max.(abs.(light_target_fit), 1e-12)
                        light_sigma_residual_fit = abs.(light_model_fit .- light_target_fit) ./ max.(light_sigma_fit, 1e-12)
                        jfit = argmax(light_sigma_residual_fit)
                        fitted_indices = findall(light_fit_mask)
                        jfull = fitted_indices[jfit]
                        if ws.d3_constraints === nothing
                            println("[KARL TRACER FAIL BIN]", " fit_idx=", jfit, " full_idx=", jfull, " constraint_inner_pc=", ws.light_edges[jfull] / pc, " constraint_outer_pc=", ws.light_edges[jfull + 1] / pc, " target=", light_target_fit[jfit], " model=", light_model_fit[jfit], " relative_error=", light_relative_fit[jfit], " normalization_error=", _wdiag_value(wdiag, :normalization_error, NaN), " N_active_bound=", _wdiag_value(wdiag, :n_active_bound, 0), " active_passes=", _wdiag_value(wdiag, :active_passes, 0), " sigma=", light_sigma_fit[jfit], " sigma_residual=", light_sigma_residual_fit[jfit], " chi2_losvd=", chi2_losvd[i])
                        else
                            ir = ws.d3_constraints.row_radial[jfull]
                            iv = ws.d3_constraints.row_angular[jfull]
                            println("[KARL TRACER FAIL BIN]", " fit_idx=", jfit, " full_idx=", jfull, " radial_bin=", ir, " angular_bin=", iv, " constraint_inner_pc=", ws.d3_constraints.radial_edges_m[ir] / pc, " constraint_outer_pc=", ws.d3_constraints.radial_edges_m[ir + 1] / pc, " target=", light_target_fit[jfit], " model=", light_model_fit[jfit], " relative_error=", light_relative_fit[jfit], " normalization_error=", _wdiag_value(wdiag, :normalization_error, NaN), " N_active_bound=", _wdiag_value(wdiag, :n_active_bound, 0), " active_passes=", _wdiag_value(wdiag, :active_passes, 0), " sigma=", light_sigma_fit[jfit], " sigma_residual=", light_sigma_residual_fit[jfit], " chi2_losvd=", chi2_losvd[i])
                        end
                    else
                        println(
                            "[KARL TRACER FAIL BIN] unavailable=true",
                            " weight_count=", length(w),
                            " expected_weight_count=", size(A_light_fit, 2),
                            " chi2_losvd=", chi2_losvd[i],
                        )
                    end
                    _print_karl_failure!(i, tid, wdiag)
                    status[i] = 2
                    Threads.atomic_xchg!(ws.phase, 3)
                    continue
                end
                _print_karl_diagnostics!(i, tid, wdiag, cl)
                if losvd_target_mode_sym === :karl_resolved_stars && losvd_window_diag !== nothing
                    if losvd_window_diag.max_outside_fraction > DEFAULT_KARL_RESOLVED_SELECTION_WARN_FRACTION
                        println("[LOSVD SELECTION NOTICE] i=", i, " worst_bin=", losvd_window_diag.worst_bin, " max_aperture_outside_fraction=", losvd_window_diag.max_outside_fraction, " warning_fraction=", DEFAULT_KARL_RESOLVED_SELECTION_WARN_FRACTION, " action=diagnostic_only", " chi2_losvd=", cl)
                    end
                elseif losvd_target_mode_sym !== :karl_mode0_observables && losvd_window_diag !== nothing && losvd_window_diag.max_outside_fraction > DEFAULT_LOSVD_MAX_OUTSIDE_FRACTION
                    solver_failure_reason[i] = "losvd_window_exceeded"
                    solver_converged[i] = false
                    println("[LOSVD WINDOW REJECT] i=", i, " worst_bin=", losvd_window_diag.worst_bin, " max_aperture_outside_fraction=", losvd_window_diag.max_outside_fraction, " allowed_max=", DEFAULT_LOSVD_MAX_OUTSIDE_FRACTION, " chi2_losvd=", cl)
                    status[i] = 2
                    Threads.atomic_xchg!(ws.phase, 3)
                    continue
                end

                    status[i] = 0
                    Threads.atomic_xchg!(ws.phase, 3)
                catch e
                    status[i] = 3
                    solver_failure_reason[i] = "exception"
                    ws_i = work_states[i]
                    if ws_i !== nothing
                        _close_orbit_phase!(ws_i; next_phase=3)
                    end
                    @warn "evaluate_batch_theta Karl exception on i=$i" exception=(e, catch_backtrace()) halo_type=halo_type
                finally
                    orbit_owner_slot_held && Threads.atomic_add!(scheduler_counters.active_model_owners, -1)
                    Threads.atomic_add!(scheduler_counters.completed_models, 1)
                end
                continue
            end

            scheduler_counters.completed_models[] >= nbatch && break
            sleep(0.001)
        end
    end

    monitor_task = Threads.@spawn begin
        last_report_ns = scheduler_started_ns
        while scheduler_counters.stop_monitor[] == 0
            sleep(1.0)
            now_ns = time_ns()
            if now_ns - last_report_ns >= UInt64(10_000_000_000)
                _print_scheduler_diagnostics!()
                last_report_ns = now_ns
            end
        end
    end
    try
        Threads.@threads :static for t in 1:nthreads
            _batch_worker!(t)
        end
    finally
        Threads.atomic_xchg!(scheduler_counters.stop_monitor, 1)
        wait(monitor_task)
    end
    _print_scheduler_diagnostics!()
    return (status, chi2_losvd, chi2_inner, chi2_outer, delta_chi2_iteration, max_light_relative_residual, max_light_sigma_residual, light_constraint_ok, solver_converged, solver_iterations,
    solver_failure_reason, N_inner, N_outer, N_nonzero_weights, effective_N_orbits, max_weight_fraction, coverage_status, coverage_issue_region, coverage_issue_axis,
    coverage_issue_shell_bands, coverage_reasons, coverage_fraction, coverage_attempted_fraction, coverage_success_fraction, coverage_shell_min, coverage_lfrac_min,
    coverage_theta_min, coverage_shell_gap, coverage_lfrac_gap, coverage_theta_gap, coverage_joint_holes, coverage_deadline_hit, successful_base_orbits, planned_base_orbits,
    phase_volume_valid, phase_volume_convention, phase_volume_normalization, phase_volume_launches_recorded, phase_volume_sos_recorded, phase_volume_valid_base_orbits,
    phase_volume_invalid_recorded_orbits, phase_volume_nested_groups, phase_volume_duplicate_area_clusters, phase_volume_duplicate_area_orbits, raw_phase_volume_min,
    raw_phase_volume_max, raw_phase_volume_dynamic_range, normalized_phase_volume_min, normalized_phase_volume_max, wphase_min, wphase_max, wphase_dynamic_range, wphase_pair_max_relative_mismatch)
end


end # module
