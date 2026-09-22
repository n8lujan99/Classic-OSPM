# ========================================================================================================================
# OSPM_Physics_Spherical.jl — spherical orbit-library orchestration and batch evaluation.
# Owns orbit-library state, family-grid construction, A-matrix assembly, coverage checks,
# model evaluation, and batch scheduling orchestration.
# Shared support, observables, force construction, phase volume, weights, low-level orbit integration,
# scheduler primitives, and diagnostics live in their dedicated included files.
# ========================================================================================================================

module OSPMPhysicsSpherical

@info "OSPMPhysicsSpherical OSPM-style loaded from" @__FILE__

using LinearAlgebra, StaticArrays, Statistics, Random, Base.Threads, Optim

export build_R_halo_physical, halo_from_theta, tables_spherical, make_potential_force_funcs, integrate_orbit_rk4, build_A_matrix_hybrid, mass_enclosed_two_radii, evaluate_batch_theta, NTHREADS, force_at_rtheta

include("OSPM_Physics_Support.jl")
include("OSPM_Physics_Observables.jl")
include("OSPM_Physics_Force.jl")
include("OSPM_Physics_PhaseVolume.jl")
include("OSPM_Physics_Weights.jl")
include("OSPM_Physics_Spherical_Orbits.jl")
include("OSPM_Physics_Spherical_Scheduler.jl")

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
    mode_sym in (:current, :resolved_stars, :mode0_observables) || error("Unknown losvd_target_mode=$(mode). Use current, resolved_stars, or mode0_observables")
    return mode_sym
end

function _resolve_observables(mode::Symbol; observables_csv=nothing)
    mode === :mode0_observables || return nothing
    observables_csv === nothing && error("observables_csv is required for losvd_target_mode=:mode0_observables")
    path = String(observables_csv)
    isfile(path) || error("OSPM observables CSV not found: $path")
    return load_observables(path)
end

function _observed_targets_for_mode(R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, st, surface_brightness_profile, losvd_target_mode::Symbol, observables; resolved_kde_grid::Int=DEFAULT_RESOLVED_KDE_GRID, resolved_kde_width_bins::Float64=DEFAULT_RESOLVED_KDE_WIDTH_BINS, resolved_vmin_kms::Float64=DEFAULT_RESOLVED_VMIN_KMS, resolved_vmax_kms::Float64=DEFAULT_RESOLVED_VMAX_KMS, resolved_bootstraps::Int=DEFAULT_RESOLVED_BOOTSTRAPS, resolved_envelope_floor::Float64=DEFAULT_RESOLVED_ENVELOPE_FLOOR)
    if losvd_target_mode === :mode0_observables
        observables === nothing && error("OSPM observables runtime state is missing")
        targets = observables_targets(observables)
        length(targets.losvd_target) == st.Nlosvd || error("OSPM observables target length does not match the orbit LOSVD matrix")
        length(targets.losvd_sigma) == st.Nlosvd || error("OSPM observables sigma length does not match the orbit LOSVD matrix")
        length(observables.star_count) == st.Nspatial || error("OSPM observables star-count length does not match the orbit aperture count")
        maximum(abs.(targets.velocity_edges_mps .- st.velocity_edges)) <= 1.0e-9 || error("OSPM observables target and orbit velocity grids do not match")
        projected_light_target = light_target_from_surface_brightness(surface_brightness_profile, st.light_edges; normalize=true)
        projected_light_sigma = light_sigma_from_surface_brightness(surface_brightness_profile, st.light_edges; normalize=true, sigma_floor=1.0e-8)
        counts_by_spatial = Float64.(observables.star_count)
        return copy(targets.losvd_target), copy(targets.losvd_sigma), projected_light_target, projected_light_sigma, counts_by_spatial, nothing
    end
    if losvd_target_mode === :resolved_stars
        return observed_targets(R_star_m, valid_vlos, v_star_mps, verr_star_mps, st.spatial_edges, st.velocity_edges; surface_brightness_profile=surface_brightness_profile, light_edges=st.light_edges, target_mode=losvd_target_mode, resolved_kde_grid=resolved_kde_grid, resolved_kde_width_bins=resolved_kde_width_bins, resolved_vmin_kms=resolved_vmin_kms, resolved_vmax_kms=resolved_vmax_kms, resolved_bootstraps=resolved_bootstraps, resolved_envelope_floor=resolved_envelope_floor, return_counts=true)
    end
    losvd_target, losvd_sigma, projected_light_target, projected_light_sigma, counts_by_spatial = observed_targets(R_star_m, valid_vlos, v_star_mps, verr_star_mps, st.spatial_edges, st.velocity_edges; surface_brightness_profile=surface_brightness_profile, light_edges=st.light_edges, target_mode=losvd_target_mode, resolved_kde_grid=resolved_kde_grid, resolved_kde_width_bins=resolved_kde_width_bins, resolved_vmin_kms=resolved_vmin_kms, resolved_vmax_kms=resolved_vmax_kms, resolved_bootstraps=resolved_bootstraps, resolved_envelope_floor=resolved_envelope_floor)
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
    observables

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

    phase_volume_state::PhaseVolumeState
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
    base_index <= planned_cells || error("base_index=$base_index is outside the planned OSPM phase grid " * "1:$planned_cells")

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

include("OSPM_Physics_Spherical_Diagnostics.jl")

function _build_family_launch_grid(Nbase_orbit::Int, shells::Vector{Float64}, Lfrac, third_launches::Vector{Float64}, pot, frc, force_geometry::Symbol; worker_limit::Int=Threads.nthreads(), worker_pool=nothing)
    Nshells = length(shells)
    nlfrac = length(Lfrac)
    nthird = length(third_launches)
    nlfrac >= 2 ||
        error("OSPM phase grid requires at least one regular Lz family and one circular boundary family")
    nthird > 0 ||
        error("OSPM phase grid requires a third-integral coordinate")
    full_phase_grid = Nshells * ((nlfrac - 1) * nthird + 1)
    Nbase_orbit >= full_phase_grid ||
        error("Orbit library has $Nbase_orbit base slots for " * "$full_phase_grid normalized family cells")
    launch_r0 = fill(NaN, Nbase_orbit)
    launch_theta0 = fill(NaN, Nbase_orbit)
    launch_energy = fill(NaN, Nbase_orbit)
    launch_lz = fill(NaN, Nbase_orbit)

    # _orbit_grid_index maps the normalized OSPM grid uniquely onto the
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

            Lz_family, E_family, vc_family, family_state = orbit_family_integrals( rapo=rapo, Lz_frac=lf, pot=pot, frc=frc)

            family_state == :ok ||
                error("Unable to construct OSPM orbit family at " * "shell_id=$shell_id lfrac_id=$lfrac_id " * "state=$family_state")

            circular_boundary = lfrac_id == nlfrac

            if !axisymmetric
                rturn, turning_state = _outer_zero_velocity_radius(energy=E_family, lz=Lz_family, theta0=theta_equator, rapo_max=rapo, pot=pot)
                turning_state == :ok ||
                    error("Unable to construct spherical ZVC launch at " * "shell_id=$shell_id lfrac_id=$lfrac_id " * "state=$turning_state")
                third_id = 1
                c = _orbit_grid_index(shell_id, lfrac_id, third_id, Nshells, nlfrac, nthird)
                launch_r0[c] = rturn
                launch_theta0[c] = theta_equator
                launch_energy[c] = E_family
                launch_lz[c] = Lz_family

                continue
            end

            family_points =
                _family_zvc_launches(
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
                c = _orbit_grid_index(shell_id, lfrac_id, third_id, Nshells, nlfrac, nthird)
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
                c = _orbit_grid_index(shell_id, lfrac_id, third_id, Nshells, nlfrac, nthird)
                launch_r0[c] = family_points.r[third_id]
                launch_theta0[c] = family_points.theta[third_id]
                launch_energy[c] = E_family
                launch_lz[c] = Lz_family
            end
        end
        return nothing
    end

    worker_limit = max(1, min(worker_limit, Threads.nthreads()))
    if worker_limit > 1 && Nshells > 1
        next_shell = Threads.Atomic{Int}(1)
        nworkers = min(worker_limit, Nshells)

        @sync for _ in 1:nworkers
            Threads.@spawn begin
                worker_lease_held = false
                if worker_pool !== nothing
                    _acquire_worker!(worker_pool)
                    worker_lease_held = true
                end
                try
                    while true
                        shell_id = Threads.atomic_add!(next_shell, 1)
                        shell_id > Nshells && break
                        build_shell!(shell_id)
                    end
                finally
                    worker_lease_held && _release_worker!(worker_pool)
                end
            end
        end
    else
        worker_lease_held = false
        if worker_pool !== nothing
            _acquire_worker!(worker_pool)
            worker_lease_held = true
        end
        try
            @inbounds for shell_id in 1:Nshells
                build_shell!(shell_id)
            end
        finally
            worker_lease_held && _release_worker!(worker_pool)
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

    return (planned_indices=planned_indices, launch_r0=launch_r0, launch_theta0=launch_theta0, launch_energy=launch_energy, launch_lz=launch_lz)
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
    observables=nothing,
    precomputed_d3_constraints=nothing,
    parallel_init::Bool=true,
    worker_limit::Int=Threads.nthreads(),
    worker_pool=nothing,
)
    iseven(Norbit) || error(
        "OSPM prograde/retrograde orbit pairing requires even Norbit because " *
        "Norbit is the final A-matrix column count"
    )

    max_attempts_factor > 0 || error("max_attempts_factor must be positive")
    worker_limit > 0 || error("worker_limit must be positive")
    worker_limit = min(worker_limit, Threads.nthreads())

    Nbase_orbit = Norbit ÷ 2
    Nstar = length(R_star_m)

    valid_vec = collect(Bool, valid_vlos)
    vlos_idx = Int[]

    @inbounds for i in 1:Nstar
        valid_vec[i] && push!(vlos_idx, i)
    end

    losvd_mode = _normalize_losvd_target_mode(losvd_target_mode)

    if losvd_mode === :mode0_observables
        observables === nothing &&
            error("OSPM observables runtime state is required by _init_orbit_work")

        losvd_aperture_ranges =
            observables_aperture_ranges_m(observables)

        Nspatial = length(observables.aperture_ids)

        length(losvd_aperture_ranges) == Nspatial ||
            error("OSPM observables aperture-range count does not match aperture count")

        spatial_edges = vcat(
            0.0,
            [range[2] for range in losvd_aperture_ranges],
        )

        any(diff(spatial_edges) .<= 0.0) &&
            error(
                "OSPM observables aperture outer radii must be strictly increasing " *
                "for the orbit shell/diagnostic grid"
            )

        light_bin_edges === nothing &&
            error(
                "light_bin_edges is required for OSPM observables mode so the " *
                "existing tracer constraints remain unchanged"
            )

        light_edges = resolve_light_edges(light_bin_edges)
        velocity_edges_use = copy(observables.velocity_edges_mps)

        if velocity_edges !== nothing
            supplied_velocity_edges = Float64.(velocity_edges)

            length(supplied_velocity_edges) == length(velocity_edges_use) ||
                error("Supplied velocity grid does not match the OSPM observables CSV")

            maximum(abs.(supplied_velocity_edges .- velocity_edges_use)) <= 1.0e-9 ||
                error("Supplied velocity grid does not match the OSPM observables CSV")
        end
    else
        spatial_edges = resolve_spatial_edges(kinematic_bin_edges)

        light_edges =
            light_bin_edges === nothing ?
            spatial_edges :
            resolve_light_edges(light_bin_edges)

        Nspatial = length(spatial_edges) - 1

        losvd_aperture_ranges = [
            (spatial_edges[ib], spatial_edges[ib + 1])
            for ib in 1:Nspatial
        ]

        if losvd_mode === :resolved_stars
            velocity_edges === nothing &&
                error("OSPM resolved LOSVD requires explicit velocity_edges defining the selected sample window")
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

    if losvd_mode === :resolved_stars
        any(.!isfinite.(velocity_edges_use)) && error("OSPM resolved LOSVD selection edges contain nonfinite values")
        any(diff(velocity_edges_use) .<= 0.0) && error("OSPM resolved LOSVD selection edges must be strictly increasing")
        Nvbin_eff == Nvbin || error("OSPM resolved LOSVD selection grid must contain exactly Nvbin=$Nvbin bins; got $Nvbin_eff")
        @inbounds for i in vlos_idx
            _bin_index(velocity_edges_use, v_star_mps[i]) != 0 ||
                error("OSPM resolved stellar velocity $(v_star_mps[i] / 1.0e3) km/s lies outside the selected sample window [$(velocity_edges_use[1] / 1.0e3), $(velocity_edges_use[end] / 1.0e3)) km/s")
        end
    end

    losvd_mode === :mode0_observables &&
        Nvbin_eff != observables.nvel &&
        error("OSPM observables velocity-bin count does not match the CSV metadata")

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
        losvd_mode === :mode0_observables ?
        copy(observables.radial_edges_m) :
        copy(light_edges)

    d3_angular_edges =
        losvd_mode === :mode0_observables ?
        copy(observables.angular_edges) :
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
       worker_limit > 1

        d3_task = Threads.@spawn begin
            worker_lease_held = false
            if worker_pool !== nothing
                _acquire_worker!(worker_pool)
                worker_lease_held = true
            end
            try
                load_d3_tracer_constraints(stellar_model_state, d3_radial_edges, d3_angular_edges)
            finally
                worker_lease_held && _release_worker!(worker_pool)
            end
        end
    end

    third_launches =
        force_geometry === :axisymmetric_density_grid ?
        collect(range(0.0, 1.0; length=max(3, Ntheta_launch))) :
        [1.0]

    families_per_shell =
        (length(Lfrac) - 1) * length(third_launches) + 1

    families_per_shell > 0 ||
        error("OSPM family grid requires at least one family per radial shell")

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
            "OSPM normalized family grid requires at least $full_phase_grid base orbits " *
            "for Nshells=$Nshells, NLfrac=$(length(Lfrac)), and " *
            "Nthird=$(length(third_launches)); got Nbase_orbit=$Nbase_orbit. " *
            "Increase Norbit to at least $(2 * full_phase_grid)."
        )

    # ------------------------------------------------------------------------------------------------
    # Launch family-grid construction as a separate Julia task.
    #
    # This lets it overlap D3 tracer loading. After _build_family_launch_grid
    # itself is threaded across shell_id, this task will fan out only across
    # the worker budget assigned to this model.
    # ------------------------------------------------------------------------------------------------

    family_task = nothing
    family_worker_limit = d3_task === nothing ? worker_limit : max(1, worker_limit - 1)

    if parallel_init && family_worker_limit > 1
        family_task = Threads.@spawn _build_family_launch_grid(
            Nbase_orbit,
            shells,
            Lfrac,
            third_launches,
            ctx.pot,
            ctx.frc,
            force_geometry;
            worker_limit=family_worker_limit,
            worker_pool=worker_pool,
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
        losvd_mode !== :mode0_observables

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

    phase_volume_state = init_phase_volume_state(Nbase_orbit)

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
        load_d3_tracer_constraints(
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
            force_geometry;
            worker_limit=family_worker_limit,
            worker_pool=worker_pool,
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

        losvd_mode === :mode0_observables ?
        observables :
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

function _build_compact_wphase(st::OrbitWorkState, successful_columns::Vector{Int})
    phase_result = compute_phase_volumes(st.phase_volume_state; normalization=:geometric_mean, strict=true)
    wphase_use = compact_wphase(phase_result.wphase_paired, successful_columns, st.Norbit)
    length(wphase_use) == length(successful_columns) ||
        error("compacted OSPM wphase length does not match successful orbit columns")
    return wphase_use, phase_result.diagnostics, phase_result
end

# This is the main worker function that runs in a thread to compute orbits and fill the A-matrix.
function _orbit_worker!(st::OrbitWorkState; max_claims::Int=typemax(Int))
    max_claims > 0 || error("max_claims must be positive")
    claims_processed = 0

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
    xsky_arr = st.losvd_target_mode === :mode0_observables ? Vector{Float64}(undef, st.nsteps) : Float64[]
    ysky_arr = st.losvd_target_mode === :mode0_observables ? Vector{Float64}(undef, st.nsteps) : Float64[]
    vlib_pro = st.losvd_target_mode === :mode0_observables ? zeros(Float64, st.observables.nvel, st.observables.nrlib, st.observables.nvlib) : zeros(Float64, 0, 0, 0)
    spatial = st.losvd_target_mode === :mode0_observables ? zeros(Float64, st.observables.nrlib, st.observables.nvlib) : zeros(Float64, 0, 0)
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
        claims_processed >= max_claims && break
        time_ns() > st.t_deadline && break
        st.phase[] != 1 && break
        slot_seq = Threads.atomic_add!(st.next_orbit, 1)
        slot_seq > length(st.launch_order) && break
        c_claim = st.launch_order[slot_seq]
        claims_processed += 1
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
        equatorial_planar = !circular_boundary && abs(theta0 - DEFAULT_PHASE_SECTION_THETA) <= 1.0e-10

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
            collect_equatorial_sos(r, vr, theta; section_theta=DEFAULT_PHASE_SECTION_THETA, crossing_mode=:step, direction=:up, skip_first=true)
        end
        if !circular_boundary && !equatorial_planar && st.force_geometry === :axisymmetric_density_grid &&
           length(sos_r) < DEFAULT_PHASE_MIN_SOS_POINTS
            base_steps_used = st.nsteps
            sos_base_step_limit = lfrac_id == 1 ? max_plunging_sos_base_steps : max_sos_base_steps
            extension_failure = nothing
            while length(sos_r) < DEFAULT_PHASE_MIN_SOS_POINTS && base_steps_used < sos_base_step_limit
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
                sos_r, sos_vr_abs = collect_equatorial_sos(r, vr, theta; section_theta=DEFAULT_PHASE_SECTION_THETA, crossing_mode=:step, direction=:up, skip_first=true)
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
        elseif length(sos_r) < DEFAULT_PHASE_MIN_SOS_POINTS
            st.failure_stage[c_claim] = :insufficient_sos
            continue
        end
        st.min_r_reached[c_claim] = minimum(r)
        Nhits = length(r)
        dt_orb = integration_diag.dt
        resize!(s_arr, Nhits)
        resize!(vlos_pro_buf, Nhits)
        resize!(vlos_ret_buf, Nhits)
        st.losvd_target_mode === :mode0_observables && resize!(xsky_arr, Nhits)
        st.losvd_target_mode === :mode0_observables && resize!(ysky_arr, Nhits)
        phi = 0.0
        @inbounds for i in 1:Nhits
            ri = f64(r[i])
            thi = f64(theta[i])
            si = _ssin(thi)
            vphi_i = f64(Lz0) / max(ri * si, 1.0e-30)
            if st.losvd_target_mode === :mode0_observables
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
        st.losvd_target_mode === :mode0_observables && fill!(vlib_pro, 0.0)
        st.losvd_target_mode === :mode0_observables && fill!(spatial, 0.0)
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
            il = st.tracer_constraint_mode === :density_3d ? d3_tracer_row(st.d3_constraints, f64(r[k]), f64(theta[k])) : _bin_index(st.light_edges, s_arr[k])
            il > 0 && (col_light[il] += 1.0)

            if st.losvd_target_mode === :mode0_observables
                ir_obs, iv_obs = observables_projected_cell(st.observables, s_arr[k], ysky_arr[k])
                ir_obs == 0 && continue
                iv_obs == 0 && continue
                spatial[ir_obs, iv_obs] += 1.0
                # OSPM's raw v1lib source grid is the x>=0 half-plane. Fold x<0 samples back to that source grid and reverse their LOS velocity before the seeing convolution.
                vlos_folded = xsky_arr[k] < 0.0 ? -vlos_pro_buf[k] : vlos_pro_buf[k]
                jb_pro = _bin_index(st.velocity_edges, vlos_folded)
                jb_pro > 0 && (vlib_pro[jb_pro, ir_obs, iv_obs] += 1.0)
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
        if st.losvd_target_mode === :mode0_observables
            vlib_pro ./= Nhits
            spatial ./= Nhits
            response = observables_seeing_response(vlib_pro, spatial, st.observables)
            col_losvd_pro .= response.losvd
            col_kinematic .= response.kinematic
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
        register_phase_launch!(st.phase_volume_state, c_claim; energy=E0, lz=Lz0, energy_index=shell_id, lz_index=lfrac_id, third_index=third_id)
        record_phase_sos!(st.phase_volume_state, c_claim, sos_r, sos_vr_abs)
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

function _run_orbit_worker!(st::OrbitWorkState; scheduler_counters=nothing, helper::Bool=false, max_claims::Int=typemax(Int))
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
                _orbit_worker!(st; max_claims=max_claims)
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

# Main A-matrix builder: maps orbital weights → OSPM observables.
function build_A_matrix_hybrid(Norbit::Int, R_star_m::Vector{Float64}, has_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64},
    verr_star_mps::Vector{Float64}, sini::Float64, rho_s::Float64, r_s::Float64, MBH::Float64, ML::Float64, halo_type::String; stellar_model=nothing,
    surface_brightness_profile=nothing, tracer_constraint_mode="projected_light", nsteps::Int=DEFAULT_NSTEPS, Lfrac::NTuple{5,Float64}=DEFAULT_LFRAC,
    dt_frac_orbit::Float64=DEFAULT_DT_FRAC, max_attempts_factor::Int=DEFAULT_MAX_ATTEMPTS, diag::Bool=false, threaded::Bool=true,
    fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT, regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR, max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP,
    shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS, t_deadline::UInt64=typemax(UInt64), velocity_edges=nothing, light_bin_edges=nothing, kinematic_bin_edges=nothing,
    Nvbin::Int=21, Ntheta_launch::Int=5, halo_q_axis_ratio::Float64=1.0, halo_params=nothing, losvd_target_mode=:current, resolved_kde_grid::Int=DEFAULT_RESOLVED_KDE_GRID,
    resolved_kde_width_bins::Float64=DEFAULT_RESOLVED_KDE_WIDTH_BINS, resolved_vmin_kms::Float64=DEFAULT_RESOLVED_VMIN_KMS, resolved_vmax_kms::Float64=DEFAULT_RESOLVED_VMAX_KMS,
    resolved_bootstraps::Int=DEFAULT_RESOLVED_BOOTSTRAPS, resolved_envelope_floor::Float64=DEFAULT_RESOLVED_ENVELOPE_FLOOR, observables_csv=nothing)
    Nstar = length(R_star_m)
    @assert length(has_vlos) == Nstar
    @assert length(v_star_mps) == Nstar
    @assert length(verr_star_mps) == Nstar
    surface_brightness_profile === nothing && error("surface_brightness_profile is required for OSPM; no star-count fallback is allowed")
    Nstar == 0 && return zeros(Float64, 0, Norbit)
    stellar_model_jl = normalize_stellar_model(stellar_model)
    surface_brightness_profile_jl = normalize_surface_brightness_profile(surface_brightness_profile)
    tracer_constraint_mode_sym = _normalize_tracer_constraint_mode(tracer_constraint_mode)
    losvd_target_mode_sym = _normalize_losvd_target_mode(losvd_target_mode)
    observables = _resolve_observables(losvd_target_mode_sym; observables_csv=observables_csv)
    prewarm_stellar_force_cache(stellar_model_jl)
    light_edges_force = light_bin_edges === nothing ? resolve_spatial_edges(kinematic_bin_edges) : resolve_light_edges(light_bin_edges)
    required_force_rmax_m = 1.5 * light_edges_force[end]
    ctx = get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=stellar_model_jl, required_rmax_m=required_force_rmax_m, halo_q_axis_ratio=halo_q_axis_ratio, halo_params=halo_params)
    sini = clamp01(f64(sini))
    Rmin = minimum(R_star_m)
    Rmax = maximum(R_star_m)
    if !(isfinite(Rmin) && isfinite(Rmax) && Rmax > Rmin)
        return zeros(Float64, 0, Norbit)
    end
    nworkers = threaded ? Threads.nthreads() : 1
    worker_pool = SharedWorkerPool(nworkers)
    st = _init_orbit_work(Norbit, R_star_m, has_vlos, v_star_mps, verr_star_mps, sini, ctx; nsteps=nsteps, Lfrac=Lfrac, dt_frac_orbit=dt_frac_orbit, max_attempts_factor=max_attempts_factor,
        t_deadline=t_deadline, velocity_edges=velocity_edges, light_bin_edges=light_bin_edges, kinematic_bin_edges=kinematic_bin_edges, Nvbin=Nvbin, Ntheta_launch=Ntheta_launch,
        fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, shell_band_count=shell_band_count, tracer_constraint_mode=tracer_constraint_mode_sym, losvd_target_mode=losvd_target_mode_sym, observables=observables, worker_limit=nworkers, worker_pool=worker_pool)
    Threads.atomic_xchg!(st.phase, 1)
    if threaded && nworkers > 1
        worker_tasks = [Threads.@spawn begin
            _acquire_worker!(worker_pool)
            try
                _run_orbit_worker!(st)
            finally
                _release_worker!(worker_pool)
            end
        end for _ in 1:nworkers]
        for task in worker_tasks
            wait(task)
        end
    else
        _acquire_worker!(worker_pool)
        try
            _run_orbit_worker!(st)
        finally
            _release_worker!(worker_pool)
        end
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
    wphase_use, phase_diag, _ = _build_compact_wphase(st, coverage.successful_columns)
    A = vcat(A_losvd, A_light)
    if diag
        losvd_target, losvd_sigma, projected_light_target, projected_light_sigma, counts_by_spatial, losvd_counts = _observed_targets_for_mode(R_star_m, has_vlos, v_star_mps, verr_star_mps, st, surface_brightness_profile_jl, losvd_target_mode_sym, observables; resolved_kde_grid=resolved_kde_grid, resolved_kde_width_bins=resolved_kde_width_bins, resolved_vmin_kms=resolved_vmin_kms, resolved_vmax_kms=resolved_vmax_kms, resolved_bootstraps=resolved_bootstraps, resolved_envelope_floor=resolved_envelope_floor)
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
                "resolved_kde_grid" => resolved_kde_grid,
                "resolved_kde_width_bins" => resolved_kde_width_bins,
                "resolved_vmin_kms" => resolved_vmin_kms,
                "resolved_vmax_kms" => resolved_vmax_kms,
                "resolved_bootstraps" => resolved_bootstraps,
                "resolved_envelope_floor" => resolved_envelope_floor,
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

# Batch evaluator: OSPM-style binned LOSVD + selectable projected-light or 3-D tracer-density constraint.
# This is the Heart of the whole Pipeline
# and is where all the parallelism is implemented

function evaluate_batch_theta(thetas::AbstractMatrix{<:Real}, R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, sini::Float64, Norbit::Int, halo_type::String; stellar_model=nothing, surface_brightness_profile=nothing, tracer_constraint_mode="projected_light", alphat::Float64=DEFAULT_ALPHAT, apfac::Float64=DEFAULT_APFAC, light_rel_tol::Float64=DEFAULT_LIGHT_REL_TOL, light_sigma_tol::Float64=2.0, delta_chi2_iter_tol::Float64=DEFAULT_DELTA_CHI2_ITER_TOL, entropy_floor::Float64=DEFAULT_ENTROPY_FLOOR, maxiter::Int=DEFAULT_MAXITER, timeout_s::Float64=120.0, fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT, regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR, max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP, shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS, coverage_check_every::Int=DEFAULT_ORBIT_COVERAGE_CHECK_EVERY, warn_fill_pct::Float64=DEFAULT_ORBIT_WARN_FILL_PCT, warn_success_pct::Float64=DEFAULT_ORBIT_WARN_SUCCESS_PCT, warn_regional_floor::Float64=DEFAULT_ORBIT_WARN_REGIONAL_FLOOR, warn_max_regional_gap::Float64=DEFAULT_ORBIT_WARN_MAX_REGIONAL_GAP, model_owner_limit::Int=0, initial_model_owners::Int=0, threads_per_model::Int=2, total_workers::Int=0, reserve_workers::Int=0, admission_worker_limit::Int=0, dynamic_admission::Bool=false, whole_model_admission::Bool=true, finishing_priority::Bool=true, R_inner_pc::Float64=30.0, velocity_edges=nothing, kinematic_bin_edges=nothing, light_bin_edges=nothing, Nvbin::Int=21, Ntheta_launch::Int=5, halo_q_axis_ratio::Float64=1.0, halo_params=nothing, losvd_target_mode=:current, resolved_kde_grid::Int=DEFAULT_RESOLVED_KDE_GRID, resolved_kde_width_bins::Float64=DEFAULT_RESOLVED_KDE_WIDTH_BINS, resolved_vmin_kms::Float64=DEFAULT_RESOLVED_VMIN_KMS, resolved_vmax_kms::Float64=DEFAULT_RESOLVED_VMAX_KMS, resolved_bootstraps::Int=DEFAULT_RESOLVED_BOOTSTRAPS, resolved_envelope_floor::Float64=DEFAULT_RESOLVED_ENVELOPE_FLOOR, observables_csv=nothing, losvd_conditioning=:vlos_cut, losvd_fit_statistic=nothing, delta_statistic_iter_tol::Float64=delta_chi2_iter_tol)
    nrow, nbatch = size(thetas)
    surface_brightness_profile === nothing && error("surface_brightness_profile is required for OSPM; no star-count fallback is allowed")
    apfac > 0.0 || error("apfac must be positive")
    light_rel_tol > 0.0 || error("light_rel_tol must be positive")
    light_sigma_tol > 0.0 || error("light_sigma_tol must be positive")
    delta_chi2_iter_tol >= 0.0 || error("delta_chi2_iter_tol must be nonnegative")
    delta_statistic_iter_tol >= 0.0 || error("delta_statistic_iter_tol must be nonnegative")
    model_owner_limit >= 0 || error("model_owner_limit must be nonnegative")
    initial_model_owners >= 0 || error("initial_model_owners must be nonnegative")
    threads_per_model > 0 || error("threads_per_model must be positive")
    total_workers >= 0 || error("total_workers must be nonnegative")
    reserve_workers >= 0 || error("reserve_workers must be nonnegative")
    admission_worker_limit >= 0 || error("admission_worker_limit must be nonnegative")
    dynamic_admission && !whole_model_admission && error("Dynamic Smart Run requires whole_model_admission=true")
    losvd_target_mode_sym = _normalize_losvd_target_mode(losvd_target_mode)
    losvd_conditioning_sym = _normalize_losvd_conditioning(losvd_conditioning)
    losvd_fit_statistic_sym = _normalize_losvd_fit_statistic(losvd_fit_statistic, losvd_target_mode_sym)
    observables = _resolve_observables(losvd_target_mode_sym; observables_csv=observables_csv)
    if losvd_target_mode_sym === :resolved_stars
        velocity_edges === nothing && error("OSPM resolved LOSVD requires explicit velocity_edges defining the selected sample window")
        resolved_kde_grid > 1 || error("resolved_kde_grid must exceed one")
        isfinite(resolved_kde_width_bins) && resolved_kde_width_bins > 0.0 || error("resolved_kde_width_bins must be finite and positive")
        isfinite(resolved_vmin_kms) && isfinite(resolved_vmax_kms) && resolved_vmax_kms > resolved_vmin_kms || error("OSPM resolved LOSVD velocity bounds are invalid")
        resolved_bootstraps > 1 || error("resolved_bootstraps must exceed one")
        isfinite(resolved_envelope_floor) && resolved_envelope_floor >= 0.0 || error("resolved_envelope_floor must be finite and nonnegative")
    end
    nthreads = Threads.nthreads()
    total_worker_pool = total_workers > 0 ? total_workers : nthreads
    total_worker_pool <= nthreads || error("total_workers=$total_worker_pool exceeds Threads.nthreads()=$nthreads")
    workers_per_model = min(threads_per_model, total_worker_pool)
    reserve_worker_count = dynamic_admission ? (reserve_workers > 0 ? reserve_workers : min(workers_per_model, max(0, total_worker_pool - workers_per_model))) : min(reserve_workers, max(0, total_worker_pool - 1))
    reserve_worker_count < total_worker_pool || error("reserve_workers must leave at least one runnable worker")
    admission_limit = dynamic_admission ? (admission_worker_limit > 0 ? admission_worker_limit : total_worker_pool - reserve_worker_count) : total_worker_pool
    admission_limit > 0 || error("admission_worker_limit must be positive after Smart Run normalization")
    admission_limit <= total_worker_pool - reserve_worker_count || error("admission_worker_limit=$admission_limit exceeds total_workers-reserve_workers=$(total_worker_pool - reserve_worker_count)")
    admission_limit >= min(workers_per_model, total_worker_pool) || error("admission_worker_limit=$admission_limit cannot admit one whole model with workers_per_model=$workers_per_model")
    max_parallel_models = max(1, fld(total_worker_pool, workers_per_model))
    hard_model_limit = model_owner_limit > 0 ? min(model_owner_limit, nbatch) : nbatch
    initial_owner_target = initial_model_owners > 0 ? min(initial_model_owners, hard_model_limit) : min(hard_model_limit, max_parallel_models)
    initial_owner_target = max(1, initial_owner_target)
    BLAS.set_num_threads(dynamic_admission || nbatch > 1 ? 1 : min(total_worker_pool, workers_per_model))
    allocated_threads = tryparse(Int, get(ENV, "SLURM_CPUS_PER_TASK", ""))
    if allocated_threads !== nothing &&
    nthreads != allocated_threads
        error("Julia thread mismatch: " * "Threads.nthreads()=$nthreads " * "but SLURM_CPUS_PER_TASK=$allocated_threads")
    end

    println(
        "[RUNTIME CONTRACT]",
        " host=", gethostname(),
        " julia_version=", VERSION,
        " julia_threads=", nthreads,
        " blas_threads=", BLAS.get_num_threads(),
        " slurm_cpus=", get(ENV, "SLURM_CPUS_PER_TASK", "local"),
        " threads_per_model=", workers_per_model,
        " model_owner_limit_requested=", model_owner_limit,
        " initial_model_owners=", initial_owner_target,
        " total_workers=", total_worker_pool,
        " reserve_workers=", reserve_worker_count,
        " admission_worker_limit=", admission_limit,
        " dynamic_admission=", dynamic_admission,
        " whole_model_admission=", whole_model_admission,
        " finishing_priority=", finishing_priority,
        " losvd_target_mode=", losvd_target_mode_sym,
        " losvd_fit_statistic=", losvd_fit_statistic_sym,
        " losvd_conditioning=", losvd_conditioning_sym)
    stellar_model_jl = normalize_stellar_model(stellar_model)
    surface_brightness_profile_jl = normalize_surface_brightness_profile(surface_brightness_profile)
    tracer_constraint_mode_sym = _normalize_tracer_constraint_mode(tracer_constraint_mode)
    prewarm_stellar_force_cache(stellar_model_jl)
    light_edges_force = light_bin_edges === nothing ? resolve_spatial_edges(kinematic_bin_edges) : resolve_light_edges(light_bin_edges)
    required_force_rmax_m = 1.5 * light_edges_force[end]
    batch_d3_constraints = nothing
    if tracer_constraint_mode_sym === :density_3d
        batch_d3_radial_edges = losvd_target_mode_sym === :mode0_observables ? copy(observables.radial_edges_m) : copy(light_edges_force)
        batch_d3_angular_edges = losvd_target_mode_sym === :mode0_observables ? copy(observables.angular_edges) : collect(range(0.0, 1.0; length=6))
        batch_d3_constraints = load_d3_tracer_constraints(stellar_model_jl, batch_d3_radial_edges, batch_d3_angular_edges)
    end
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
    model_worker_leases = [Threads.Atomic{Int}(0) for _ in 1:nbatch]
    worker_pool = SharedWorkerPool(total_worker_pool)
    next_theta = Threads.Atomic{Int}(1)
    admission_lock = ReentrantLock()
    scheduler_counters = (initial_claims=Threads.Atomic{Int}(0), dynamic_claims=Threads.Atomic{Int}(0), initializing_models=Threads.Atomic{Int}(0), orbit_models=Threads.Atomic{Int}(0), orbit_workers=Threads.Atomic{Int}(0), helper_workers=Threads.Atomic{Int}(0), finishing_waiters=Threads.Atomic{Int}(0), finishing_models=Threads.Atomic{Int}(0), weight_models=Threads.Atomic{Int}(0), active_model_owners=Threads.Atomic{Int}(0), completed_models=Threads.Atomic{Int}(0), stop_monitor=Threads.Atomic{Int}(0))
    julia_stage_timing = get(ENV, "OSPM_DIAG_JULIA_STAGE_TIMING", "0") == "1" || get(ENV, "OSPM_DIAG_PIPELINE_TIMING", "0") == "1"
    orbit_failure_diag = julia_stage_timing || get(ENV, "OSPM_DIAG_ORBIT_FAILURES", "0") == "1"

    function _julia_stage_begin(i::Int, stage::String, tid::Int)
        t0_ns = time_ns()
        julia_stage_timing && println("[JULIA STAGE TIMING] wall_s=", time(), " i=", i, " tid=", tid, " stage=", stage, " event=begin", " leased_workers=", worker_pool.leased[], " active_owners=", scheduler_counters.active_model_owners[], " orbit_workers=", scheduler_counters.orbit_workers[], " finishing_models=", scheduler_counters.finishing_models[], " weight_models=", scheduler_counters.weight_models[])
        return t0_ns
    end

    function _julia_stage_end(i::Int, stage::String, tid::Int, t0_ns::UInt64)
        julia_stage_timing && println("[JULIA STAGE TIMING] wall_s=", time(), " i=", i, " tid=", tid, " stage=", stage, " event=end", " elapsed_s=", (time_ns() - t0_ns) / 1.0e9, " leased_workers=", worker_pool.leased[], " active_owners=", scheduler_counters.active_model_owners[], " orbit_workers=", scheduler_counters.orbit_workers[], " finishing_models=", scheduler_counters.finishing_models[], " weight_models=", scheduler_counters.weight_models[])
        return nothing
    end

    println("[SCHED] julia_threads=", nthreads, " total_workers=", total_worker_pool, " workers_per_model=", workers_per_model, " max_parallel_models=", max_parallel_models, " initial_model_owners=", initial_owner_target, " reserve_workers=", reserve_worker_count, " admission_worker_limit=", admission_limit, " hard_model_limit=", hard_model_limit, " dynamic_admission=", dynamic_admission)
    if get(ENV, "OSPM_DIAG_JULIA_HANDOFF", "0") == "1"
        if losvd_target_mode_sym === :resolved_stars
            println("[JULIA OBSERVABLES] mode=", losvd_target_mode_sym, " Nvbin=", Nvbin, " kde_grid=", resolved_kde_grid, " kde_width_bins=", resolved_kde_width_bins, " vmin_kms=", resolved_vmin_kms, " vmax_kms=", resolved_vmax_kms)
        elseif losvd_target_mode_sym === :mode0_observables
            println("[JULIA OBSERVABLES] mode=", losvd_target_mode_sym, " Nvbin=", Nvbin, " csv=", observables_csv)
        else
            println("[JULIA OBSERVABLES] mode=", losvd_target_mode_sym, " Nvbin=", Nvbin)
        end
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
    function _print_diagnostics!(i::Int, tid::Int, wdiag, chi2_score::Float64)
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

    function _print_failure!(i::Int, tid::Int, wdiag)
        println("[OSPM SOLVER FAIL] i=", i, " tid=", tid, " failure_reason=", _wdiag_value(wdiag, :failure_reason, :missing_diagnostics), " delta_chi2_iteration=", _wdiag_value(wdiag, :delta_chi2_iteration, NaN), " max_light_relative_residual=", _wdiag_value(wdiag, :max_light_relative_residual, NaN), " max_light_sigma_residual=", _wdiag_value(wdiag, :max_light_sigma_residual, NaN), " light_constraint_ok=", _wdiag_value(wdiag, :light_constraint_ok, false), " solver_converged=", _wdiag_value(wdiag, :solver_converged, false), " iterations=", _wdiag_value(wdiag, :iterations, 0), " rcond_est=", _wdiag_value(wdiag, :rcond_est, NaN), " max_abs_dw=", _wdiag_value(wdiag, :max_abs_dw, NaN), " stepfac=", _wdiag_value(wdiag, :stepfac, NaN), " chi_slack=", _wdiag_value(wdiag, :chi_slack, NaN))
        return nothing
    end
    function _unleased_orbit_demand()
        demand = 0
        lock(work_states_lock)
        try
            @inbounds for i in 1:nbatch
                ws = work_states[i]
                ws === nothing && continue
                ws.phase[] == 1 || continue
                remaining = max(0, length(ws.launch_order) - (ws.next_orbit[] - 1))
                remaining == 0 && continue
                lease_count = model_worker_leases[i][]
                desired = min(workers_per_model, remaining)
                demand += max(0, desired - lease_count)
            end
        finally
            unlock(work_states_lock)
        end
        return demand
    end

    function _try_claim_theta!()
        next_theta[] > nbatch && return 0
        lock(admission_lock)
        try
            next_theta[] > nbatch && return 0
            active = scheduler_counters.active_model_owners[]
            active >= hard_model_limit && return 0
            claimed = clamp(next_theta[] - 1, 0, nbatch)
            dynamic_claim = false
            admission_load = NaN
            runnable_demand = NaN
            if claimed < initial_owner_target
                Threads.atomic_add!(scheduler_counters.initial_claims, 1)
            elseif dynamic_admission
                scheduler_counters.initializing_models[] > 0 && return 0
                finishing_priority && scheduler_counters.finishing_waiters[] > 0 && return 0
                admission_load = worker_pool.leased[]
                runnable_demand = admission_load + _unleased_orbit_demand() + scheduler_counters.finishing_waiters[]
                active >= initial_owner_target && runnable_demand >= admission_limit && return 0
                Threads.atomic_add!(scheduler_counters.dynamic_claims, 1)
                dynamic_claim = true
            else
                active >= initial_owner_target && return 0
            end
            Threads.atomic_add!(scheduler_counters.active_model_owners, 1)
            Threads.atomic_add!(scheduler_counters.initializing_models, 1)
            i = Threads.atomic_add!(next_theta, 1)
            if i > nbatch
                Threads.atomic_add!(scheduler_counters.initializing_models, -1)
                Threads.atomic_add!(scheduler_counters.active_model_owners, -1)
                return 0
            end
            if dynamic_claim
                println("[SCHED ADMIT] i=", i, " leased_workers_before_claim=", admission_load, " runnable_demand_before_claim=", runnable_demand, " workers_per_model=", workers_per_model, " admission_worker_limit=", admission_limit, " reserve_workers=", reserve_worker_count, " active_models_after_claim=", scheduler_counters.active_model_owners[])
            end
            return i
        finally
            unlock(admission_lock)
        end
    end

    function _dispatch_orbit_workers!()
        free_capacity = total_worker_pool - worker_pool.leased[]
        if finishing_priority && free_capacity > 0
            free_capacity -= min(free_capacity, scheduler_counters.finishing_waiters[])
        end
        free_capacity <= 0 && return 0
        dispatched = 0
        while free_capacity > 0
            candidates = NamedTuple[]
            lock(work_states_lock)
            try
                @inbounds for i in 1:nbatch
                    ws = work_states[i]
                    ws === nothing && continue
                    ws.phase[] == 1 || continue
                    ws.next_orbit[] <= length(ws.launch_order) || continue
                    lease_count = model_worker_leases[i][]
                    lease_count < workers_per_model || continue
                    remaining = max(0, length(ws.launch_order) - (ws.next_orbit[] - 1))
                    push!(candidates, (i=i, ws=ws, lease_count=lease_count, remaining=remaining))
                end
            finally
                unlock(work_states_lock)
            end
            isempty(candidates) && break
            sort!(candidates; by=item -> (item.lease_count == 0 ? 0 : 1, finishing_priority ? item.remaining : item.i, item.i))
            candidate = first(candidates)
            i = candidate.i
            ws = candidate.ws
            helper = candidate.lease_count > 0
            _try_acquire_worker!(worker_pool) || break
            Threads.atomic_add!(model_worker_leases[i], 1)
            let model_index=i, work_state=ws, helper_worker=helper
                Threads.@spawn begin
                    try
                        _run_orbit_worker!(work_state; scheduler_counters=scheduler_counters, helper=helper_worker, max_claims=4)
                    finally
                        Threads.atomic_add!(model_worker_leases[model_index], -1)
                        _release_worker!(worker_pool)
                    end
                end
            end
            dispatched += 1
            free_capacity -= 1
        end
        return dispatched
    end

    function _batch_worker!(tid::Int)
        while true
            i = _try_claim_theta!()
            if i > 0
                model_owner_slot_held = true
                initializing_phase_counted = true
                finishing_phase_counted = false
                finishing_worker_lease_held = false
                theta_deadline = time_ns() + UInt64(round(timeout_s * 1e9))
                model_t0 = _julia_stage_begin(i, "MODEL_TOTAL", tid)
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
                    model_init_t0 = _julia_stage_begin(i, "MODEL_INIT", tid)
                    ctx = get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=stellar_model_jl, required_rmax_m=required_force_rmax_m, halo_q_axis_ratio=halo_q_axis_ratio, halo_params=halo_params)
                    ws = _init_orbit_work(Norbit, R_star_m, valid_vlos, v_star_mps, verr_star_mps, sini, ctx; nsteps=DEFAULT_NSTEPS, Lfrac=DEFAULT_LFRAC, dt_frac_orbit=DEFAULT_DT_FRAC, max_attempts_factor=DEFAULT_MAX_ATTEMPTS, t_deadline=theta_deadline, velocity_edges=velocity_edges, light_bin_edges=light_bin_edges, kinematic_bin_edges=kinematic_bin_edges, Nvbin=Nvbin, Ntheta_launch=Ntheta_launch, fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, shell_band_count=shell_band_count, coverage_check_every=coverage_check_every, tracer_constraint_mode=tracer_constraint_mode_sym, losvd_target_mode=losvd_target_mode_sym, observables=observables, precomputed_d3_constraints=batch_d3_constraints, worker_limit=workers_per_model, worker_pool=worker_pool)
                    _julia_stage_end(i, "MODEL_INIT", tid, model_init_t0)
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

                    if initializing_phase_counted
                        lock(admission_lock)
                        try
                            Threads.atomic_add!(scheduler_counters.initializing_models, -1)
                            initializing_phase_counted = false
                        finally
                            unlock(admission_lock)
                        end
                    end
                    orbit_t0 = _julia_stage_begin(i, "ORBIT_INTEGRATION", tid)
                    Threads.atomic_add!(scheduler_counters.orbit_models, 1)
                    try
                        Threads.atomic_xchg!(ws.phase, 1)
                        while ws.phase[] == 1 && time_ns() <= theta_deadline
                            all_claimed = ws.next_orbit[] > length(ws.launch_order)
                            all_claimed && model_worker_leases[i][] == 0 && break
                            sleep(0.001)
                        end
                        _close_orbit_phase!(ws)
                    finally
                        Threads.atomic_add!(scheduler_counters.orbit_models, -1)
                    end
                    _julia_stage_end(i, "ORBIT_INTEGRATION", tid, orbit_t0)
                    finisher_wait_t0 = _julia_stage_begin(i, "FINISHER_WAIT", tid)
                    Threads.atomic_add!(scheduler_counters.finishing_waiters, 1)
                    try
                        _acquire_worker!(worker_pool; priority=finishing_priority)
                        finishing_worker_lease_held = true
                    finally
                        Threads.atomic_add!(scheduler_counters.finishing_waiters, -1)
                    end
                    _julia_stage_end(i, "FINISHER_WAIT", tid, finisher_wait_t0)
                    Threads.atomic_add!(scheduler_counters.finishing_models, 1)
                    finishing_phase_counted = true

                    coverage_t0 = _julia_stage_begin(i, "COVERAGE", tid)
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
                    _julia_stage_end(i, "COVERAGE", tid, coverage_t0)
                    if orbit_failure_diag && coverage.accepted && (coverage.succeeded < coverage.planned || coverage.attempted < coverage.planned)
                        println("[ORBIT FAILURE DIAG]", " i=", i, " planned=", coverage.planned, " attempted=", coverage.attempted, " succeeded=", coverage.succeeded, " failed_attempted=", coverage.attempted - coverage.succeeded, " unattempted=", coverage.planned - coverage.attempted, " deadline_hit=", coverage_deadline_hit[i])
                        _print_orbit_failure_diagnostics(ws, i)
                    end
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
                    phase_volume_t0 = _julia_stage_begin(i, "PHASE_VOLUME", tid)
                    A_losvd, A_light, A_kinematic = _compact_orbit_matrices(ws, coverage.successful_columns)
                    wphase_use = Float64[]
                    phase_diag = nothing
                    phase_result = nothing
                    try
                        wphase_use, phase_diag, phase_result = _build_compact_wphase(ws, coverage.successful_columns,)
                    catch phase_error
                        status[i] = 1
                        solver_failure_reason[i] = "phase_volume_failed"
                        println("[OSPM PHASE VOLUME FAIL]", " i=", i, " error=", sprint(showerror, phase_error))
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
                    _julia_stage_end(i, "PHASE_VOLUME", tid, phase_volume_t0)
                    if size(A_losvd, 1) == 0 || size(A_losvd, 2) == 0 || !all(isfinite, A_losvd) || size(A_light, 1) == 0 || size(A_light, 2) == 0 || !all(isfinite, A_light)
                        status[i] = 1
                        solver_failure_reason[i] = "invalid_observable_matrix"
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end
                    targets_t0 = _julia_stage_begin(i, "TARGETS", tid)
                    losvd_target, losvd_sigma, projected_light_target, projected_light_sigma, counts_by_spatial, losvd_counts = _observed_targets_for_mode(R_star_m, valid_vlos, v_star_mps, verr_star_mps, ws,
                    surface_brightness_profile_jl, losvd_target_mode_sym, observables; resolved_kde_grid=resolved_kde_grid, resolved_kde_width_bins=resolved_kde_width_bins,
                    resolved_vmin_kms=resolved_vmin_kms, resolved_vmax_kms=resolved_vmax_kms, resolved_bootstraps=resolved_bootstraps, resolved_envelope_floor=resolved_envelope_floor)
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
                                d3_radial_target = d3_radial_target(ws.d3_constraints)
                                println("[TRACER CONFIG DETAIL]", " radial_target_l1_vs_projected=", sum(abs.(d3_radial_target .- projected_light_target)), " radial_bins=", ws.d3_constraints.nradial, " angular_bins=", ws.d3_constraints.nangular, " inner_theta_collapsed=true", " sigma_min=", minimum(light_sigma_fit), " sigma_max=", maximum(light_sigma_fit), " R_kin_max_pc=", maximum(last.(ws.losvd_aperture_ranges)) / pc)
                            end
                        end
                    end
                    _julia_stage_end(i, "TARGETS", tid, targets_t0)
                    weight_solver_t0 = _julia_stage_begin(i, "WEIGHT_SOLVER", tid)
                    Threads.atomic_add!(scheduler_counters.weight_models, 1)
                    w = Float64[]
                    ok = false
                    wdiag = nothing
                    try
                        if losvd_fit_statistic_sym === :multinomial
                            losvd_counts === nothing && error("multinomial LOSVD fit requires resolved-star integer counts")
                            w, ok, wdiag = solve_weights_multinomial(A_light_fit, A_losvd, A_kinematic, light_target_fit, light_sigma_fit, losvd_counts; Nspatial=ws.Nspatial, Nvbin=ws.Nvbin, conditioning=losvd_conditioning_sym, alphat=alphat, light_rel_tol=light_rel_tol, light_sigma_tol=light_sigma_tol, delta_statistic_iter_tol=delta_statistic_iter_tol, wphase=wphase_use, maxiter=maxiter, seed=UInt(i), entropy_floor=entropy_floor, apfac=apfac, return_diag=true)
                        else
                            w, ok, wdiag = solve_weights(A_light_fit, A_losvd, light_target_fit, light_sigma_fit, losvd_target, losvd_sigma; Nspatial=ws.Nspatial, Nvbin=ws.Nvbin, alphat=alphat, light_rel_tol=light_rel_tol, light_sigma_tol=light_sigma_tol, delta_chi2_iter_tol=delta_chi2_iter_tol, wphase=wphase_use, maxiter=maxiter, seed=UInt(i), entropy_floor=entropy_floor, apfac=apfac, return_diag=true)
                        end
                    finally
                        Threads.atomic_add!(scheduler_counters.weight_models, -1)
                    end
                    _julia_stage_end(i, "WEIGHT_SOLVER", tid, weight_solver_t0)
                    post_solver_t0 = _julia_stage_begin(i, "POST_SOLVER", tid)

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

                    losvd_score_state = losvd_fit_statistic_sym === :multinomial ? multinomial_losvd_state(A_losvd, A_kinematic, w, losvd_counts, ws.Nspatial, ws.Nvbin; conditioning=losvd_conditioning_sym) : losvd_chi2_state(A_losvd, w, losvd_target, losvd_sigma, ws.Nspatial, ws.Nvbin)
                    cl = losvd_fit_statistic_sym === :multinomial ? losvd_score_state.deviance_total : losvd_score_state.chi_total
                    score_by_spatial = losvd_fit_statistic_sym === :multinomial ? losvd_score_state.deviance_by_spatial : losvd_score_state.chi_by_spatial
                    if get(ENV, "OSPM_DIAG_APERTURE_SCORE", "0") == "1"
                        @inbounds for ib in 1:ws.Nspatial
                            rlo_pc = ws.losvd_aperture_ranges[ib][1] / pc
                            rhi_pc = ws.losvd_aperture_ranges[ib][2] / pc
                            rmid_pc = 0.5 * (rlo_pc + rhi_pc)
                            println("[LOSVD APERTURE SCORE] model=", i, " aperture=", ib, " Rlo_pc=", rlo_pc, " Rhi_pc=", rhi_pc, " Rmid_pc=", rmid_pc, " Nstars=", Int(round(counts_by_spatial[ib])), " score=", score_by_spatial[ib])
                        end
                    end
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
                        if losvd_fit_statistic_sym === :multinomial
                            losvd_counts === nothing && error("multinomial orbit-family diagnostic requires resolved-star integer counts")
                            _print_orbit_family_multinomial_diagnostics(
                                ws,
                                coverage.successful_columns,
                                w,
                                A_losvd,
                                A_kinematic,
                                losvd_counts;
                                conditioning=losvd_conditioning_sym,
                                model_index=i,
                                target_lfrac=target_lfrac,
                                target_third_u=target_third_u,
                                target_aperture=target_aperture,
                            )
                        else
                            _print_orbit_projection_diagnostics(
                                ws,
                                coverage.successful_columns,
                                w,
                                A_losvd,
                                A_kinematic,
                                losvd_target,
                                losvd_sigma;
                                model_index=i,
                                inner_radius_pc=R_inner_pc,
                                target_lfrac=target_lfrac,
                                target_third_u=target_third_u,
                                target_aperture=target_aperture,
                            )
                        end
                    end
                end
                _julia_stage_end(i, "POST_SOLVER", tid, post_solver_t0)
                if !ok
                    if length(w) == size(A_light_fit, 2) && all(isfinite, w)
                        light_model_fit = A_light_fit * w
                        light_relative_fit = abs.(light_model_fit .- light_target_fit) ./ max.(abs.(light_target_fit), 1e-12)
                        light_sigma_residual_fit = abs.(light_model_fit .- light_target_fit) ./ max.(light_sigma_fit, 1e-12)
                        jfit = argmax(light_sigma_residual_fit)
                        fitted_indices = findall(light_fit_mask)
                        jfull = fitted_indices[jfit]
                        if ws.d3_constraints === nothing
                            println("[OSPM TRACER FAIL BIN]", " fit_idx=", jfit, " full_idx=", jfull, " constraint_inner_pc=", ws.light_edges[jfull] / pc, " constraint_outer_pc=", ws.light_edges[jfull + 1] / pc, " target=", light_target_fit[jfit], " model=", light_model_fit[jfit], " relative_error=", light_relative_fit[jfit], " normalization_error=", _wdiag_value(wdiag, :normalization_error, NaN), " N_active_bound=", _wdiag_value(wdiag, :n_active_bound, 0), " active_passes=", _wdiag_value(wdiag, :active_passes, 0), " sigma=", light_sigma_fit[jfit], " sigma_residual=", light_sigma_residual_fit[jfit], " chi2_losvd=", chi2_losvd[i])
                        else
                            ir = ws.d3_constraints.row_radial[jfull]
                            iv = ws.d3_constraints.row_angular[jfull]
                            println("[OSPM TRACER FAIL BIN]", " fit_idx=", jfit, " full_idx=", jfull, " radial_bin=", ir, " angular_bin=", iv, " constraint_inner_pc=", ws.d3_constraints.radial_edges_m[ir] / pc, " constraint_outer_pc=", ws.d3_constraints.radial_edges_m[ir + 1] / pc, " target=", light_target_fit[jfit], " model=", light_model_fit[jfit], " relative_error=", light_relative_fit[jfit], " normalization_error=", _wdiag_value(wdiag, :normalization_error, NaN), " N_active_bound=", _wdiag_value(wdiag, :n_active_bound, 0), " active_passes=", _wdiag_value(wdiag, :active_passes, 0), " sigma=", light_sigma_fit[jfit], " sigma_residual=", light_sigma_residual_fit[jfit], " chi2_losvd=", chi2_losvd[i])
                        end
                    else
                        println(
                            "[OSPM TRACER FAIL BIN] unavailable=true",
                            " weight_count=", length(w),
                            " expected_weight_count=", size(A_light_fit, 2),
                            " chi2_losvd=", chi2_losvd[i],
                        )
                    end
                    _print_failure!(i, tid, wdiag)
                    status[i] = 2
                    Threads.atomic_xchg!(ws.phase, 3)
                    continue
                end
                _print_diagnostics!(i, tid, wdiag, cl)
                if losvd_target_mode_sym === :resolved_stars && losvd_window_diag !== nothing
                    if losvd_window_diag.max_outside_fraction > DEFAULT_RESOLVED_SELECTION_WARN_FRACTION
                        println("[LOSVD SELECTION NOTICE] i=", i, " worst_bin=", losvd_window_diag.worst_bin, " max_aperture_outside_fraction=", losvd_window_diag.max_outside_fraction, " warning_fraction=", DEFAULT_RESOLVED_SELECTION_WARN_FRACTION, " action=diagnostic_only", " chi2_losvd=", cl)
                    end
                elseif losvd_target_mode_sym !== :mode0_observables && losvd_window_diag !== nothing && losvd_window_diag.max_outside_fraction > DEFAULT_LOSVD_MAX_OUTSIDE_FRACTION
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
                    @warn "evaluate_batch_theta OSPM exception on i=$i" exception=(e, catch_backtrace()) halo_type=halo_type
                finally
                    if initializing_phase_counted
                        lock(admission_lock)
                        try
                            Threads.atomic_add!(scheduler_counters.initializing_models, -1)
                        finally
                            unlock(admission_lock)
                        end
                    end
                    finishing_phase_counted && Threads.atomic_add!(scheduler_counters.finishing_models, -1)
                    if finishing_worker_lease_held
                        _release_worker!(worker_pool)
                        finishing_worker_lease_held = false
                    end
                    _julia_stage_end(i, "MODEL_TOTAL", tid, model_t0)
                    model_owner_slot_held && Threads.atomic_add!(scheduler_counters.active_model_owners, -1)
                    Threads.atomic_add!(scheduler_counters.completed_models, 1)
                end
                continue
            end

            scheduler_counters.completed_models[] >= nbatch && break
            sleep(0.001)
        end
    end

    dispatcher_task = Threads.@spawn begin
        while scheduler_counters.stop_monitor[] == 0
            _dispatch_orbit_workers!()
            sleep(0.001)
        end
    end

    owner_task_count = min(nbatch, hard_model_limit)
    owner_tasks = [Threads.@spawn _batch_worker!(t) for t in 1:owner_task_count]
    try
        for task in owner_tasks
            wait(task)
        end
    finally
        Threads.atomic_xchg!(scheduler_counters.stop_monitor, 1)
        wait(dispatcher_task)
    end
    return (status, chi2_losvd, chi2_inner, chi2_outer, delta_chi2_iteration, max_light_relative_residual, max_light_sigma_residual, light_constraint_ok, solver_converged, solver_iterations,
    solver_failure_reason, N_inner, N_outer, N_nonzero_weights, effective_N_orbits, max_weight_fraction, coverage_status, coverage_issue_region, coverage_issue_axis,
    coverage_issue_shell_bands, coverage_reasons, coverage_fraction, coverage_attempted_fraction, coverage_success_fraction, coverage_shell_min, coverage_lfrac_min,
    coverage_theta_min, coverage_shell_gap, coverage_lfrac_gap, coverage_theta_gap, coverage_joint_holes, coverage_deadline_hit, successful_base_orbits, planned_base_orbits,
    phase_volume_valid, phase_volume_convention, phase_volume_normalization, phase_volume_launches_recorded, phase_volume_sos_recorded, phase_volume_valid_base_orbits,
    phase_volume_invalid_recorded_orbits, phase_volume_nested_groups, phase_volume_duplicate_area_clusters, phase_volume_duplicate_area_orbits, raw_phase_volume_min,
    raw_phase_volume_max, raw_phase_volume_dynamic_range, normalized_phase_volume_min, normalized_phase_volume_max, wphase_min, wphase_max, wphase_dynamic_range, wphase_pair_max_relative_mismatch)
end

end # module
