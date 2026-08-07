module OSPMPhysicsSpherical
# This file is becoming very large and may need to be split into support and weights
@info "OSPMPhysicsSpherical Karl-style loaded from" @__FILE__
using LinearAlgebra, StaticArrays, Statistics, Random, Base.Threads, Optim
export build_R_halo_physical, halo_from_theta, tables_spherical, make_potential_force_funcs, integrate_orbit_rk4, build_A_matrix_hybrid, mass_enclosed_two_radii, evaluate_batch_theta, NTHREADS, force_at_rtheta
include("OSPM_Physics_Support.jl")
include("OSPM_Physics_PhaseVolume.jl")
@info "OSPMPhysicsSpherical supports spherical frc(r,theta)->(fr,0) and axisymmetric frc(r,theta)->(fr,ftheta)"
# HOT PATH — Karl-style OSPM A-matrix builder & batch evaluator.
# Applied new karl fixes on 04/06/26 @1600

const DEFAULT_ORBIT_FILL_PCT = 0.85
const DEFAULT_ORBIT_REGIONAL_FLOOR = 0.80
const DEFAULT_ORBIT_MAX_REGIONAL_GAP = 0.10 
const DEFAULT_ORBIT_SHELL_BANDS = 8
const DEFAULT_ORBIT_COVERAGE_CHECK_EVERY = 50
const DEFAULT_ORBIT_WARN_FILL_PCT = 0.95
const DEFAULT_ORBIT_WARN_SUCCESS_PCT = 0.99
const DEFAULT_ORBIT_WARN_REGIONAL_FLOOR = 0.80
const DEFAULT_ORBIT_WARN_MAX_REGIONAL_GAP = 0.15

# -----------------------------------------------------------------------------
# Work state
# -----------------------------------------------------------------------------
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

    dt_frac_orbit::Float64
    t_deadline::UInt64
    fill_pct::Float64
    regional_floor::Float64
    max_regional_gap::Float64
    shell_band_count::Int
    coverage_check_every::Int
    next_coverage_check::Threads.Atomic{Int}

    A_losvd::Matrix{Float64}
    A_light::Matrix{Float64}

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

function _build_family_launch_grid( Nbase_orbit::Int, shells::Vector{Float64}, Lfrac, third_launches::Vector{Float64}, pot, frc, force_geometry::Symbol)
    Nshells = length(shells)
    nlfrac = length(Lfrac)
    nthird = length(third_launches)
    nlfrac >= 2 || error("Karl phase grid requires at least one regular Lz family " * "and one circular boundary family" )
    nthird > 0 || error("Karl phase grid requires a third-integral coordinate")
    full_phase_grid = Nshells * ((nlfrac - 1) * nthird + 1)
    Nbase_orbit >= full_phase_grid || error("Orbit library has $Nbase_orbit base slots for " * "$full_phase_grid normalized family cells")

    launch_r0 = fill(NaN, Nbase_orbit)
    launch_theta0 = fill(NaN, Nbase_orbit)
    launch_energy = fill(NaN, Nbase_orbit)
    launch_lz = fill(NaN, Nbase_orbit)
    planned_indices = Int[]
    sizehint!(planned_indices, full_phase_grid)

    axisymmetric = force_geometry === :axisymmetric_density_grid
    theta_equator = f64(pi / 2)
    ncurve = max(65, 8 * nthird)

    if axisymmetric
        abs(first(third_launches)) <= 1.0e-12 ||
            error("Normalized third-integral launches must begin at u=0")
        abs(last(third_launches) - 1.0) <= 1.0e-12 ||
            error("Normalized third-integral launches must end at u=1")
    elseif nthird != 1
        error("A spherical orbit family must use exactly one " * "third-integral launch")
    end

    @inbounds for shell_id in 1:Nshells
        rapo = f64(shells[shell_id])
        isfinite(rapo) && rapo > 0.0 || error("Orbit shell $shell_id has an invalid radius")

        for lfrac_id in 1:nlfrac
            lf = f64(Lfrac[lfrac_id])
            Lz_family, E_family, vc_family, family_state =
                karl_orbit_family_integrals( rapo=rapo, Lz_frac=lf, pot=pot, frc=frc)
            family_state == :ok ||
                error("Unable to construct Karl orbit family at " * "shell_id=$shell_id lfrac_id=$lfrac_id " * "state=$family_state")
            circular_boundary = lfrac_id == nlfrac
            if !axisymmetric
                rturn, turning_state = _karl_outer_zero_velocity_radius(energy=E_family, lz=Lz_family, theta0=theta_equator, rapo_max=rapo, pot=pot)
                turning_state == :ok || error("Unable to construct spherical ZVC launch at " * "shell_id=$shell_id lfrac_id=$lfrac_id " * "state=$turning_state")
                third_id = 1
                c = _orbit_grid_index(shell_id, lfrac_id, third_id, Nshells, nlfrac, nthird)
                push!(planned_indices, c)
                launch_r0[c] = rturn
                launch_theta0[c] = theta_equator
                launch_energy[c] = E_family
                launch_lz[c] = Lz_family
                continue
            end
            family_points = _karl_family_zvc_launches(energy=E_family, lz=Lz_family, rapo=rapo, pot=pot, Nlaunch=circular_boundary ? 1 : nthird, circular_boundary=circular_boundary, ncurve=circular_boundary ? 0 : ncurve)
            family_points.state == :ok ||
                error("Unable to construct normalized family ZVC at " * "shell_id=$shell_id lfrac_id=$lfrac_id " *"state=$(family_points.state)")
            if circular_boundary
                third_id = nthird
                c = _orbit_grid_index(shell_id, lfrac_id, third_id, Nshells, nlfrac, nthird)
                push!(planned_indices, c)
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
                push!(planned_indices, c)
                launch_r0[c] = family_points.r[third_id]
                launch_theta0[c] = family_points.theta[third_id]
                launch_energy[c] = E_family
                launch_lz[c] = Lz_family
            end
        end
    end

    length(planned_indices) == full_phase_grid ||
        error("Normalized family grid built $(length(planned_indices)) cells " * "but expected $full_phase_grid")
    sort!(planned_indices)
    unique!(planned_indices)
    length(planned_indices) == full_phase_grid ||
        error("Normalized family grid contains duplicate orbit indices")

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

function _build_orbit_shells(R_star_m::Vector{Float64}, light_edges::Vector{Float64})
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
    return shells
end

function _init_orbit_work(
    Norbit::Int, R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool},
    v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, sini::Float64, ctx;
    nsteps::Int, Lfrac, dt_frac_orbit::Float64, max_attempts_factor::Int,
    t_deadline::UInt64, velocity_edges=nothing, light_bin_edges=nothing,
    kinematic_bin_edges=nothing, Nvbin::Int=21, Ntheta_launch::Int=9,
    fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT,
    regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR,
    max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP,
    shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS,
    coverage_check_every::Int=DEFAULT_ORBIT_COVERAGE_CHECK_EVERY)
    iseven(Norbit) || error(
        "Karl prograde/retrograde orbit pairing requires even Norbit because " *
        "Norbit is the final A-matrix column count",
    )

    max_attempts_factor > 0 || error("max_attempts_factor must be positive")

    Nbase_orbit = Norbit ÷ 2
    Nstar = length(R_star_m)
    valid_vec = collect(Bool, valid_vlos)
    vlos_idx = Int[]

    @inbounds for i in 1:Nstar
        valid_vec[i] && push!(vlos_idx, i)
    end

    spatial_edges = resolve_karl_spatial_edges(kinematic_bin_edges)
    light_edges = light_bin_edges === nothing ? spatial_edges : resolve_karl_light_edges(light_bin_edges)

    Nspatial = length(spatial_edges) - 1
    Nlight = length(light_edges) - 1

    velocity_edges_use = velocity_edges === nothing ?
        build_velocity_edges_auto(v_star_mps[vlos_idx], verr_star_mps[vlos_idx]; Nvbin=Nvbin) :
        Float64.(velocity_edges)

    Nvbin_eff = length(velocity_edges_use) - 1
    Nlosvd = Nspatial * Nvbin_eff

    shells = _build_orbit_shells(R_star_m, light_edges)
    isempty(shells) && error("Orbit shell grid has no finite positive radii")

    Nshells = length(shells)

    Nbase_orbit >= Nshells || error(
        "Orbit library has $Nbase_orbit base slots for $Nshells required radial shells; increase Norbit",
    )

    A_losvd = zeros(Float64, Nlosvd, Norbit)
    A_light = zeros(Float64, Nlight, Norbit)

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

    force_geometry = haskey(ctx.halo, :stellar_model) ?
        stellar_model_geometry(ctx.halo[:stellar_model]) :
        :spherical_shell_grid

    third_launches = force_geometry === :axisymmetric_density_grid ?
        collect(range(0.0, 1.0; length=max(3, Ntheta_launch))) :
        [1.0]

    full_phase_grid = Nshells * ((length(Lfrac) - 1) * length(third_launches) + 1)

    Nbase_orbit >= full_phase_grid || error(
        "Karl normalized family grid requires at least $full_phase_grid base orbits " *
        "for Nshells=$Nshells, NLfrac=$(length(Lfrac)), and " *
        "Nthird=$(length(third_launches)); got Nbase_orbit=$Nbase_orbit. " *
        "Increase Norbit to at least $(2 * full_phase_grid).",
    )

    family_grid = _build_family_launch_grid(
        Nbase_orbit, shells, Lfrac, third_launches,
        ctx.pot, ctx.frc, force_geometry,
    )

    launch_order = _balanced_launch_order(
        family_grid.planned_indices, shells, Lfrac,
        third_launches, shell_band_count,
    )

    first_coverage_check = max(1, ceil(Int, fill_pct * length(launch_order)))

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

        dt_frac_orbit,
        t_deadline,
        fill_pct,
        regional_floor,
        max_regional_gap,
        shell_band_count,
        max(1, coverage_check_every),
        Threads.Atomic{Int}(first_coverage_check),

        A_losvd,
        A_light,

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

@inline function _project_axisym_sample(ri::Float64, vr::Float64, vtheta::Float64, vphi::Float64, theta::Float64, phi::Float64, sini::Float64, cosi::Float64)
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

    attempted_count < planned && push!(
        rejection_reasons,
        "$(planned - attempted_count) planned phase-grid orbit(s) were not attempted",
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
        attempted_count == planned &&
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

function _coverage_metadata( coverage;
    fill_pct::Float64,
    regional_floor::Float64,
    max_regional_gap::Float64,
    warn_fill_pct::Float64,
    warn_success_pct::Float64,
    warn_regional_floor::Float64,
    warn_max_regional_gap::Float64)
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
        ("theta", coverage.theta_minimum_coverage, coverage.theta_coverage_gap),
    )
        (axis_min < regional_floor || axis_gap > max_regional_gap) &&
            push!(issue_axes, axis_name)
    end
    !isempty(coverage.joint_holes) && push!(issue_axes, "joint")
    unique!(issue_axes)
    issue_axis = isempty(issue_axes) ? "none" :
        (length(issue_axes) == 1 ? only(issue_axes) : "multiple")

    shell_coverages = [Float64(stat.coverage_fraction) for stat in coverage.shell_bands]
    max_shell_coverage = isempty(shell_coverages) ? 0.0 : maximum(shell_coverages)
    issue_shell_bands = Int[]
    for stat in coverage.shell_bands
        if stat.coverage_fraction < regional_floor ||
           max_shell_coverage - stat.coverage_fraction > max_regional_gap
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

    return (
        status=coverage_status,
        issue_axis=issue_axis,
        issue_region=issue_region,
        issue_shell_bands=join(issue_shell_bands, ";"),
        reasons=join(coverage.rejection_reasons, " | "),
    )
end

function _maybe_stop_orbit_phase_for_coverage!(st::OrbitWorkState)
    st.phase[] == 1 || return false
    filled = st.filled_atomic[]
    next_check = st.next_coverage_check[]
    filled >= next_check || return false
    Threads.atomic_cas!(st.next_coverage_check, next_check, next_check + st.coverage_check_every) == next_check || return false
    coverage = _assess_orbit_coverage(st; fill_pct=st.fill_pct, regional_floor=st.regional_floor, max_regional_gap=st.max_regional_gap, shell_band_count=st.shell_band_count, verify_atomic=false)
    coverage.accepted || return false

    if Threads.atomic_cas!(st.phase, 1, 2) == 1
        println(
            "[ORBIT COVERAGE TARGET] ",
            "filled=", coverage.succeeded,
            " planned=", coverage.planned,
            " coverage_fraction=", coverage.coverage_fraction,
            " shell_min=", coverage.shell_minimum_coverage,
            " lfrac_min=", coverage.lfrac_minimum_coverage,
            " theta_min=", coverage.theta_minimum_coverage,
            " shell_gap=", coverage.shell_coverage_gap,
            " lfrac_gap=", coverage.lfrac_coverage_gap,
            " theta_gap=", coverage.theta_coverage_gap,
        )
        return true
    end
    return false
end

function _orbit_library_usable(st::OrbitWorkState, successful_columns::Vector{Int})
    isempty(successful_columns) && return false
    @inbounds for col in successful_columns
        activity = sum(abs, @view(st.A_losvd[:, col])) + sum(abs, @view(st.A_light[:, col]))
        if !(isfinite(activity) && activity > 0.0)
            return false
        end
    end
    @inbounds for row in 1:st.Nlight
        activity = sum(abs, @view(st.A_light[row, successful_columns]))
        if !(isfinite(activity) && activity > 0.0)
            return false
        end
    end
    return true
end

function _compact_orbit_matrices(st::OrbitWorkState, successful_columns::Vector{Int})
    return st.A_losvd[:, successful_columns], st.A_light[:, successful_columns]
end

function _build_compact_karl_wphase(st::OrbitWorkState, successful_columns::Vector{Int})
    wphase_paired, phase_diag = build_karl_wphase(st.phase_volume_state; normalization=:geometric_mean, strict=true)
    wphase_use = compact_karl_wphase(wphase_paired, successful_columns, st.Norbit)
    length(wphase_use) == length(successful_columns) ||
        error("compacted Karl wphase length does not match successful orbit columns")
    return wphase_use, phase_diag
end
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# This is the main worker function that runs in a thread to compute orbits and fill the A-matrix.

function _orbit_worker!(st::OrbitWorkState)

    col_losvd_pro = zeros(Float64, st.Nlosvd)
    col_losvd_ret = zeros(Float64, st.Nlosvd)
    col_light = zeros(Float64, st.Nlight)
    s_arr = Vector{Float64}(undef, st.nsteps)
    vlos_pro_buf = Vector{Float64}(undef, st.nsteps)
    vlos_ret_buf = Vector{Float64}(undef, st.nsteps)
    energy_drift_tolerance = 1.0e-2
    dt_scales = (1.0, 0.5, 0.25, 0.125, 0.0625)
    sos_chunk_base_steps = 4_000
    max_sos_base_steps = 40_000

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

        for local_step_safety in (0.10, 0.075, 0.05)
            time_ns() > st.t_deadline && return nothing

            last_result = integrate_orbit_rk4(ic=ic_use, xLz=Lz_use, orbit_ctx=st.orbit_ctx, nsteps=scaled_steps, pot=st.pot, return_diag=true, dt_scale=dt_scale,
                energy_check_every=scaled_energy_check_every, max_relative_energy_drift_allowed=energy_drift_tolerance, local_step_safety=local_step_safety,
                continuation_state=(previous_diag.final_R, previous_diag.final_z, previous_diag.final_vR, previous_diag.final_vz), reference_energy=reference_energy)

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

        ic, Lz0, E0, vc, launch_state = launch_orbit_apocenter(rapo=rapo, theta0=theta0, Lz_frac=lf, pot=st.pot, frc=st.frc, dt_frac=st.dt_frac_orbit,
            fixed_energy=E_family, fixed_lz=Lz_family, fixed_rturn=rturn)

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
            extension_failure = nothing

            while length(sos_r) < DEFAULT_KARL_PHASE_MIN_SOS_POINTS && base_steps_used < max_sos_base_steps
                if time_ns() > st.t_deadline
                    st.failure_stage[c_claim] = :deadline_during_extended_integration
                    return nothing
                end

                chunk_base_steps = min(sos_chunk_base_steps, max_sos_base_steps - base_steps_used)
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

                sos_r, sos_vr_abs = collect_karl_equatorial_sos(r, vr, theta; section_theta=DEFAULT_KARL_PHASE_SECTION_THETA,
                    crossing_mode=:karl_step, direction=:up, skip_first=true)
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
        phi = 0.0
        @inbounds for i in 1:Nhits
            ri = f64(r[i])
            thi = f64(theta[i])
            si = _ssin(thi)
            vphi_i = f64(Lz0) / max(ri * si, 1.0e-30)
            s_arr[i], vlos_pro_buf[i] = _project_axisym_sample(ri, f64(vr[i]), f64(vtheta[i]), vphi_i, thi, phi, st.sini, st.cosi)
            _, vlos_ret_buf[i] = _project_axisym_sample(ri, f64(vr[i]), f64(vtheta[i]), -vphi_i, thi, phi, st.sini, st.cosi)
            phi += f64(Lz0) / max(ri * ri * si * si, 1.0e-30) * dt_orb
        end

        fill!(col_losvd_pro, 0.0)
        fill!(col_losvd_ret, 0.0)
        fill!(col_light, 0.0)

        @inbounds for k in 1:Nhits
            il = _bin_index(st.light_edges, s_arr[k])
            ik = _bin_index(st.spatial_edges, s_arr[k])

            il > 0 && (col_light[il] += 1.0)
            ik == 0 && continue

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
        col_losvd_pro ./= Nhits
        col_losvd_ret ./= Nhits

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
        @inbounds st.A_light[:, col_pro] .= col_light
        @inbounds st.A_light[:, col_ret] .= col_light

        st.failure_stage[c_claim] = :success
        st.success_flags[c_claim] = true

        Threads.atomic_add!(st.filled_atomic, 1)
        _maybe_stop_orbit_phase_for_coverage!(st)
    end

    return nothing
end
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

function _run_orbit_worker!(st::OrbitWorkState; scheduler_counters=nothing, helper::Bool=false)
    # Admit a worker and close the orbit phase through the same gate.  This
    # prevents a helper that observed phase=1 from entering after the owner has
    # already moved on to coverage assessment.
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
    try
        _orbit_worker!(st)
    finally
        if scheduler_counters !== nothing
            helper && Threads.atomic_add!(scheduler_counters.helper_workers, -1)
            Threads.atomic_add!(scheduler_counters.orbit_workers, -1)
        end
        Threads.atomic_add!(st.active_workers, -1)
    end
    return true
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

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main A-matrix builder: maps orbital weights → Karl observables.

function build_A_matrix_hybrid(Norbit::Int, R_star_m::Vector{Float64}, has_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, sini::Float64, rho_s::Float64, r_s::Float64, MBH::Float64, ML::Float64, halo_type::String; stellar_model=nothing, surface_brightness_profile=nothing, nsteps::Int=DEFAULT_NSTEPS, Lfrac::NTuple{5,Float64}=DEFAULT_LFRAC, dt_frac_orbit::Float64=DEFAULT_DT_FRAC, max_attempts_factor::Int=DEFAULT_MAX_ATTEMPTS, diag::Bool=false, threaded::Bool=true, fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT, regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR, max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP, shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS, t_deadline::UInt64=typemax(UInt64), velocity_edges=nothing, light_bin_edges=nothing, kinematic_bin_edges=nothing, Nvbin::Int=21, Ntheta_launch::Int=9, halo_q_axis_ratio::Float64=1.0, karl_halo_params=nothing)
    Nstar = length(R_star_m)
    @assert length(has_vlos) == Nstar
    @assert length(v_star_mps) == Nstar
    @assert length(verr_star_mps) == Nstar
    surface_brightness_profile === nothing && error("surface_brightness_profile is required for Karl-style OSPM; no star-count fallback is allowed")
    Nstar == 0 && return zeros(Float64, 0, Norbit)
    stellar_model_jl = normalize_stellar_model(stellar_model)
    surface_brightness_profile_jl = normalize_surface_brightness_profile(surface_brightness_profile)
    prewarm_stellar_force_cache(stellar_model_jl)
    ctx = get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=stellar_model_jl, halo_q_axis_ratio=halo_q_axis_ratio, karl_halo_params=karl_halo_params)
    sini = clamp01(f64(sini))
    Rmin = minimum(R_star_m)
    Rmax = maximum(R_star_m)
    if !(isfinite(Rmin) && isfinite(Rmax) && Rmax > Rmin)
        return zeros(Float64, 0, Norbit)
    end
    st = _init_orbit_work(Norbit, R_star_m, has_vlos, v_star_mps, verr_star_mps, sini, ctx; nsteps=nsteps, Lfrac=Lfrac, dt_frac_orbit=dt_frac_orbit, max_attempts_factor=max_attempts_factor,
        t_deadline=t_deadline, velocity_edges=velocity_edges, light_bin_edges=light_bin_edges, kinematic_bin_edges=kinematic_bin_edges, Nvbin=Nvbin, Ntheta_launch=Ntheta_launch,
        fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, shell_band_count=shell_band_count)
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
        error("Incomplete orbit coverage: filled $filled / " * "$(coverage.planned) base slots; " * join(coverage.rejection_reasons, " | "))
    end
    _orbit_library_usable(st, coverage.successful_columns) ||
        error("Unusable orbit library: a successful paired orbit column or projected-light row has no support")
    A_losvd, A_light = _compact_orbit_matrices(st, coverage.successful_columns)
    wphase_use, phase_diag = _build_compact_karl_wphase(st, coverage.successful_columns)
    A = vcat(A_losvd, A_light)
    if diag
        losvd_target, losvd_sigma, light_target, light_sigma, counts_by_spatial = observed_targets_karl(R_star_m, has_vlos, v_star_mps, verr_star_mps, st.spatial_edges, st.velocity_edges; surface_brightness_profile=surface_brightness_profile_jl, light_edges=st.light_edges)
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
                "light_edges" => st.light_edges,
                "velocity_edges" => st.velocity_edges,
                "losvd_target" => losvd_target,
                "losvd_sigma" => losvd_sigma,
                "light_target" => light_target,
                "light_sigma" => light_sigma,
                "counts_by_spatial" => counts_by_spatial,
                "force_geometry" => String(st.force_geometry),
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

# Batch evaluator: Karl-style binned LOSVD + projected-light fit.
# This is the Heart of the whole Pipeline 
# and is where all the parallelism is implemented
function evaluate_batch_theta(thetas::AbstractMatrix{<:Real}, R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, sini::Float64, Norbit::Int, halo_type::String; stellar_model=nothing, surface_brightness_profile=nothing, alphat::Float64=DEFAULT_KARL_ALPHAT, light_rel_tol::Float64=DEFAULT_KARL_LIGHT_REL_TOL, light_sigma_tol::Float64=2.0, delta_chi2_iter_tol::Float64=DEFAULT_KARL_DELTA_CHI2_ITER_TOL, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, maxiter::Int=DEFAULT_KARL_MAXITER, timeout_s::Float64=120.0, fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT, regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR, max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP, shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS, coverage_check_every::Int=DEFAULT_ORBIT_COVERAGE_CHECK_EVERY, warn_fill_pct::Float64=DEFAULT_ORBIT_WARN_FILL_PCT, warn_success_pct::Float64=DEFAULT_ORBIT_WARN_SUCCESS_PCT, warn_regional_floor::Float64=DEFAULT_ORBIT_WARN_REGIONAL_FLOOR, warn_max_regional_gap::Float64=DEFAULT_ORBIT_WARN_MAX_REGIONAL_GAP, model_owner_limit::Int=0, threads_per_model::Int=2, R_inner_pc::Float64=30.0, velocity_edges=nothing, kinematic_bin_edges=nothing, light_bin_edges=nothing, Nvbin::Int=21, Ntheta_launch::Int=9, halo_q_axis_ratio::Float64=1.0, karl_halo_params=nothing)
    nrow, nbatch = size(thetas)
    surface_brightness_profile === nothing && error("surface_brightness_profile is required for Karl-style OSPM; no star-count fallback is allowed")
    light_rel_tol > 0.0 || error("light_rel_tol must be positive")
    light_sigma_tol > 0.0 || error("light_sigma_tol must be positive")
    delta_chi2_iter_tol >= 0.0 || error("delta_chi2_iter_tol must be nonnegative")
    threads_per_model > 0 || error("threads_per_model must be positive")
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
    )
    stellar_model_jl = normalize_stellar_model(stellar_model)
    surface_brightness_profile_jl = normalize_surface_brightness_profile(surface_brightness_profile)
    prewarm_stellar_force_cache(stellar_model_jl)
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
    next_theta = Threads.Atomic{Int}(1)
    nthreads = Threads.nthreads()

    owner_limit = if model_owner_limit > 0
        clamp(model_owner_limit, 1, min(nthreads, nbatch))
    else
        min(nbatch, max(1, cld(nthreads, threads_per_model)))
    end

    group_base = fld(nthreads, owner_limit)
    group_remainder = rem(nthreads, owner_limit)
    group_sizes = [group_base + (igroup <= group_remainder ? 1 : 0) for igroup in 1:owner_limit]
    scheduler_counters = (orbit_models=Threads.Atomic{Int}(0), orbit_workers=Threads.Atomic{Int}(0), helper_workers=Threads.Atomic{Int}(0), weight_models=Threads.Atomic{Int}(0), active_model_owners=Threads.Atomic{Int}(0), completed_models=Threads.Atomic{Int}(0), stop_monitor=Threads.Atomic{Int}(0))
    scheduler_started_ns = time_ns()
    println("[SCHED GROUPS] julia_threads=", nthreads, " threads_per_model_target=", threads_per_model, " model_groups=", owner_limit, " group_sizes=", join(group_sizes, ","))

    function _print_scheduler_diagnostics!()
        claimed = clamp(next_theta[] - 1, 0, nbatch)
        completed = scheduler_counters.completed_models[]
        orbit_models = scheduler_counters.orbit_models[]
        weight_models = scheduler_counters.weight_models[]
        other_models = max(0, claimed - completed - orbit_models - weight_models)
        elapsed_s = (time_ns() - scheduler_started_ns) / 1e9
        println("[SCHED DIAG] elapsed_s=", round(elapsed_s; digits=1), " claimed=", claimed, " queued=", nbatch - claimed, " completed=", completed, " orbit_models=", orbit_models, " orbit_workers=", scheduler_counters.orbit_workers[], " helper_workers=", scheduler_counters.helper_workers[], " weight_models=", weight_models, " active_model_owners=", scheduler_counters.active_model_owners[], " model_owner_limit=", owner_limit, " threads_per_model_target=", threads_per_model, " group_sizes=", join(group_sizes, ","), " other_models=", other_models, " julia_threads=", nthreads)
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
        println(
            "[KARL PHASE VOLUME DIAG]",
            " i=", i,
            " convention=", phase_diag.convention,
            " normalization=", phase_diag.normalization,
            " launches_recorded=", phase_diag.launches_recorded,
            " sos_recorded=", phase_diag.sos_recorded,
            " valid_base_orbits=", phase_diag.valid_base_orbits,
            " nested_groups=", phase_diag.nested_groups,
            " duplicate_area_clusters=", phase_diag.duplicate_area_clusters,
            " duplicate_area_orbits=", phase_diag.duplicate_area_orbits,
            " raw_min=", phase_diag.raw_phase_volume_min,
            " raw_max=", phase_diag.raw_phase_volume_max,
            " raw_dynamic_range=", phase_diag.raw_phase_volume_dynamic_range,
            " normalized_min=", phase_diag.normalized_phase_volume_min,
            " normalized_max=", phase_diag.normalized_phase_volume_max,
            " wphase_min=", phase_diag.wphase_min,
            " wphase_max=", phase_diag.wphase_max,
        )
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
        println("[KARL SOLVER DIAG] i=", i, " tid=", tid, " chi_losvd_score=", chi2_score, " chi_losvd_solver=", _wdiag_value(wdiag, :chi_losvd, NaN), " delta_chi2_iteration=", _wdiag_value(wdiag, :delta_chi2_iteration, NaN), " delta_chi2_iteration_ok=", _wdiag_value(wdiag, :delta_chi2_iteration_ok, false), " max_light_relative_residual=", _wdiag_value(wdiag, :max_light_relative_residual, NaN), " max_light_sigma_residual=", _wdiag_value(wdiag, :max_light_sigma_residual, NaN), " light_constraint_ok=", _wdiag_value(wdiag, :light_constraint_ok, false), " entropy=", _wdiag_value(wdiag, :entropy, NaN), " profit=", _wdiag_value(wdiag, :profit, NaN), " losvd_penalty=", _wdiag_value(wdiag, :losvd_penalty, NaN), " chi_slack=", _wdiag_value(wdiag, :chi_slack, NaN), " slack_to_losvd=", _wdiag_value(wdiag, :slack_to_losvd, NaN), " slack_l2=", _wdiag_value(wdiag, :slack_l2, NaN), " slack_max_abs=", _wdiag_value(wdiag, :slack_max_abs, NaN), " rcond_est=", _wdiag_value(wdiag, :rcond_est, NaN), " max_abs_dw=", _wdiag_value(wdiag, :max_abs_dw, NaN), " stepfac=", _wdiag_value(wdiag, :stepfac, NaN), " iterations=", _wdiag_value(wdiag, :iterations, 0), " constraint_ok=", _wdiag_value(wdiag, :constraint_ok, false), " slack_consistent=", _wdiag_value(wdiag, :slack_consistent, false), " normalized=", _wdiag_value(wdiag, :normalized, false), " solver_converged=", _wdiag_value(wdiag, :solver_converged, false), " failure_reason=", _wdiag_value(wdiag, :failure_reason, :none), " wphase_convention=", _wdiag_value(wdiag, :wphase_convention, :missing), " wphase_min=", _wdiag_value(wdiag, :wphase_min, NaN), " wphase_max=", _wdiag_value(wdiag, :wphase_max, NaN), " wphase_dynamic_range=", _wdiag_value(wdiag, :wphase_dynamic_range, NaN), " phase_volume_min=", _wdiag_value(wdiag, :phase_volume_min, NaN), " phase_volume_max=", _wdiag_value(wdiag, :phase_volume_max, NaN), " phase_volume_dynamic_range=", _wdiag_value(wdiag, :phase_volume_dynamic_range, NaN), " wphase_pair_max_relative_mismatch=", _wdiag_value(wdiag, :wphase_pair_max_relative_mismatch, NaN), " N_nonzero=", N_nonzero_weights[i], " Neff=", effective_N_orbits[i], " max_weight_fraction=", max_weight_fraction[i])
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
                    ctx = get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=stellar_model_jl, halo_q_axis_ratio=halo_q_axis_ratio, karl_halo_params=karl_halo_params)
                    ws = _init_orbit_work(Norbit, R_star_m, valid_vlos, v_star_mps, verr_star_mps, sini, ctx; nsteps=DEFAULT_NSTEPS, Lfrac=DEFAULT_LFRAC, dt_frac_orbit=DEFAULT_DT_FRAC, max_attempts_factor=DEFAULT_MAX_ATTEMPTS, t_deadline=theta_deadline, velocity_edges=velocity_edges, light_bin_edges=light_bin_edges, kinematic_bin_edges=kinematic_bin_edges, Nvbin=Nvbin, Ntheta_launch=Ntheta_launch, fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, shell_band_count=shell_band_count, coverage_check_every=coverage_check_every)
                    work_states[i] = ws
                    planned_base_orbits[i] = length(ws.launch_order)
                    if i == 1
                        println("[KARL BIN DIAG] N_light=", ws.Nlight, " N_kin=", ws.Nspatial, " Nvbin=", ws.Nvbin, " A_light_rows=", size(ws.A_light, 1), " A_losvd_rows=", size(ws.A_losvd, 1), " R_light_max_pc=", ws.light_edges[end] / pc, " R_kin_max_pc=", ws.spatial_edges[end] / pc, " R_shell_max_pc=", ws.shells[end] / pc, " N_shells=", ws.Nshells, " N_constraints=", ws.Nlight + ws.Nspatial * ws.Nvbin)
                    end
                    Threads.atomic_add!(scheduler_counters.orbit_models, 1)
                    try
                        Threads.atomic_xchg!(ws.phase, 1)
                        _run_orbit_worker!(ws; scheduler_counters=scheduler_counters)
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

                        println(
                            "[ORBIT COVERAGE WARNING]",
                            " i=", i,
                            " region=", coverage_meta.issue_region,
                            " axis=", coverage_meta.issue_axis,
                            " shell_bands=", isempty(coverage_meta.issue_shell_bands) ?
                                "none" : coverage_meta.issue_shell_bands,
                            " succeeded=", coverage.succeeded,
                            " attempted=", coverage.attempted,
                            " planned=", coverage.planned,
                            " required=", coverage.required,
                            " coverage_fraction=", coverage.coverage_fraction,
                            " attempted_fraction=", coverage.attempted_fraction,
                            " success_fraction=", coverage.success_fraction,
                            " shell_min=", coverage.shell_minimum_coverage,
                            " lfrac_min=", coverage.lfrac_minimum_coverage,
                            " theta_min=", coverage.theta_minimum_coverage,
                            " shell_gap=", coverage.shell_coverage_gap,
                            " lfrac_gap=", coverage.lfrac_coverage_gap,
                            " theta_gap=", coverage.theta_coverage_gap,
                            " joint_holes=", length(coverage.joint_holes),
                            " deadline_hit=", coverage_deadline_hit[i],
                            " reasons=", isempty(coverage_meta.reasons) ?
                                "none" : coverage_meta.reasons,
                        )
                        status[i] = 1
                        solver_failure_reason[i] = "incomplete_orbit_coverage"
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end
                    if !_orbit_library_usable(ws, coverage.successful_columns)
                        status[i] = 1
                        solver_failure_reason[i] = "unusable_orbit_library"
                        println("[ORBIT LIBRARY REJECTED] i=", i, " reason=zero_support_successful_column_or_light_row")
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end
                    A_losvd, A_light = _compact_orbit_matrices(ws, coverage.successful_columns)
                    wphase_use = Float64[]
                    phase_diag = nothing
                    try
                        wphase_use, phase_diag = _build_compact_karl_wphase(ws, coverage.successful_columns,)
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
                    losvd_target, losvd_sigma, light_target, light_sigma, counts_by_spatial = observed_targets_karl(R_star_m, valid_vlos, v_star_mps, verr_star_mps, ws.spatial_edges, ws.velocity_edges; surface_brightness_profile=surface_brightness_profile_jl, light_edges=ws.light_edges)
                    light_fit_mask = trues(length(light_target))
                    any(light_fit_mask) || error("No projected-light bins lie completely inside the kinematic footprint")
                    A_light_fit = A_light
                    light_target_fit = light_target
                    light_sigma_fit = light_sigma
                    if i == 1
                        println(
                            "[KARL LIGHT FIT DIAG] N_light_full=", length(light_target),
                            " N_light_fit=", length(light_target_fit),
                            " R_light_full_max_pc=", ws.light_edges[end] / pc,
                            " R_light_fit_max_pc=", maximum(ws.light_edges[2:end][light_fit_mask]) / pc,
                            " R_kin_max_pc=", ws.spatial_edges[end] / pc,
                            " target_full_sum=", sum(light_target),
                            " target_fit_sum=", sum(light_target_fit),
                            " light_sigma_min=", minimum(light_sigma_fit),
                            " light_sigma_max=", maximum(light_sigma_fit),
                        )
                    end
                    Threads.atomic_add!(scheduler_counters.weight_models, 1)
                    w = Float64[]
                    ok = false
                    wdiag = nothing
                    try
                        w, ok, wdiag = solve_weights_karl_expanded_cm(A_light_fit, A_losvd, light_target_fit, light_sigma_fit, losvd_target, losvd_sigma; alphat=alphat, light_rel_tol=light_rel_tol, light_sigma_tol=light_sigma_tol, delta_chi2_iter_tol=delta_chi2_iter_tol, wphase=wphase_use, maxiter=maxiter, seed=UInt(i), entropy_floor=entropy_floor, apfac=DEFAULT_KARL_APFAC, return_diag=true)
                    finally
                        Threads.atomic_add!(scheduler_counters.weight_models, -1)
                    end
                    _store_solver_diagnostics!(i, wdiag)
                    if !ok
                        if length(w) == size(A_light_fit, 2)
                            light_model_fit = A_light_fit * w
                            light_relative_fit = abs.(light_model_fit .- light_target_fit) ./ max.(abs.(light_target_fit), 1e-12)
                            light_sigma_residual_fit = abs.(light_model_fit .- light_target_fit) ./ max.(light_sigma_fit, 1e-12)
                            jfit = argmax(light_sigma_residual_fit)
                            fitted_indices = findall(light_fit_mask)
                            jfull = fitted_indices[jfit]
                            println(
                                "[KARL LIGHT FAIL BIN]",
                                " fit_idx=", jfit,
                                " full_idx=", jfull,
                                " R_inner_pc=", ws.light_edges[jfull] / pc,
                                " R_outer_pc=", ws.light_edges[jfull + 1] / pc,
                                " target=", light_target_fit[jfit],
                                " model=", light_model_fit[jfit],
                                " relative_error=", light_relative_fit[jfit],
                                " normalization_error=", _wdiag_value(wdiag, :normalization_error, NaN),
                                " N_active_bound=", _wdiag_value(wdiag, :n_active_bound, 0),
                                " active_passes=", _wdiag_value(wdiag, :active_passes, 0),
                                " sigma=", light_sigma_fit[jfit],
                                " sigma_residual=", light_sigma_residual_fit[jfit],
                            )
                        else
                            println("[KARL LIGHT FAIL BIN] unavailable=true", " weight_count=", length(w),  " expected_weight_count=", size(A_light_fit, 2))
                        end
                        _print_karl_failure!(i, tid, wdiag)
                        status[i] = 2
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end
                    cl = chi2_block(A_losvd, w, losvd_target, losvd_sigma)
                    chi2_losvd[i] = cl
                    R_inner_m = R_inner_pc * pc
                    ninner = 0
                    nouter = 0
                    inner_rows = Int[]
                    outer_rows = Int[]
                    @inbounds for ib in 1:ws.Nspatial
                        rmid = 0.5 * (ws.spatial_edges[ib] + ws.spatial_edges[ib + 1])
                        rows = ((ib - 1) * ws.Nvbin + 1):(ib * ws.Nvbin)
                        if rmid < R_inner_m
                            append!(inner_rows, rows)
                            ninner += Int(round(counts_by_spatial[ib]))
                        else
                            append!(outer_rows, rows)
                            nouter += Int(round(counts_by_spatial[ib]))
                        end
                    end
                    N_inner[i] = ninner
                    N_outer[i] = nouter
                    !isempty(inner_rows) && (chi2_inner[i] = chi2_block(A_losvd[inner_rows, :], w, losvd_target[inner_rows], losvd_sigma[inner_rows]))
                    !isempty(outer_rows) && (chi2_outer[i] = chi2_block(A_losvd[outer_rows, :], w, losvd_target[outer_rows], losvd_sigma[outer_rows]))
                    _store_weight_diagnostics!(i, w)
                    _print_karl_diagnostics!(i, tid, wdiag, cl)
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
            helped = false
            scan_start = mod1(tid, nbatch)
            for offset in 0:(nbatch - 1)
                scan = mod1(scan_start + offset, nbatch)
                ws_scan = work_states[scan]
                ws_scan === nothing && continue
                ws_scan.phase[] != 1 && continue
                time_ns() > ws_scan.t_deadline && continue
                ws_scan.next_orbit[] > length(ws_scan.launch_order) && continue
                try
                    helped = _run_orbit_worker!(ws_scan; scheduler_counters=scheduler_counters, helper=true)
                catch e
                    @warn "tail orbit helper exception on i=$scan" exception=(e, catch_backtrace())
                end
                helped && break
            end
            if !helped
                scheduler_counters.completed_models[] >= nbatch && break
                sleep(0.001)
            end
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
end # module
