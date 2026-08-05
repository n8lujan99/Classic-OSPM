module OSPMPhysicsSpherical
@info "OSPMPhysicsSpherical Karl-style loaded from" @__FILE__
using LinearAlgebra, StaticArrays, Statistics, Random, Base.Threads, Optim
export build_R_halo_physical, halo_from_theta, tables_spherical, make_potential_force_funcs, integrate_orbit_rk4, build_A_matrix_hybrid, mass_enclosed_two_radii, evaluate_batch_theta, NTHREADS, force_at_rtheta
include("OSPM_Physics_Support.jl")
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
    theta_launches::Vector{Float64}
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
    next_orbit::Threads.Atomic{Int}
    filled_atomic::Threads.Atomic{Int}
    phase::Threads.Atomic{Int}
    active_workers::Threads.Atomic{Int}
    worker_gate::ReentrantLock
end

function _balanced_launch_order( Nbase_orbit::Int, shells::Vector{Float64}, Lfrac, theta_launches::Vector{Float64}, shell_band_count::Int)
    Nshells = length(shells)
    nshell_bands = min(shell_band_count, Nshells)
    nlfrac = length(Lfrac)
    ntheta = length(theta_launches)
    orbit_cost = Vector{Float64}(undef, Nbase_orbit)
    cells = Dict{NTuple{3,Int},Vector{Int}}()
    @inbounds for c in 1:Nbase_orbit
        shell_id = mod1(c, Nshells)
        shell_band = fld((shell_id - 1) * nshell_bands, Nshells) + 1
        lfrac_id = mod1(c, nlfrac)
        theta_id = mod1(c, ntheta)
        rapo = shells[shell_id]
        orbit_cost[c] = f64(Lfrac[lfrac_id]) * rapo
        push!(get!(cells, (shell_band, lfrac_id, theta_id), Int[]), c)
    end
    for members in values(cells)
        sort!(members; by=c -> (orbit_cost[c], c))
    end
    cell_keys = sort!(collect(keys(cells)))
    launch_order = Int[]
    sizehint!(launch_order, Nbase_orbit)
    depth = 1
    while length(launch_order) < Nbase_orbit
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
    length(launch_order) == Nbase_orbit ||
        error("Balanced launch ordering lost orbit slots: ordered=$(length(launch_order)) planned=$Nbase_orbit")
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

function _init_orbit_work(Norbit::Int, R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, sini::Float64, ctx; nsteps::Int, Lfrac, dt_frac_orbit::Float64, max_attempts_factor::Int, t_deadline::UInt64, velocity_edges=nothing, light_bin_edges=nothing, kinematic_bin_edges=nothing, Nvbin::Int=21, Ntheta_launch::Int=9, fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT, regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR, max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP, shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS, coverage_check_every::Int=DEFAULT_ORBIT_COVERAGE_CHECK_EVERY)
    iseven(Norbit) || error("Karl prograde/retrograde orbit pairing requires even Norbit because Norbit is the final A-matrix column count")
    max_attempts_factor > 0 || error("max_attempts_factor must be positive")
    Nbase_orbit = Norbit ÷ 2
    Nstar = length(R_star_m)
    valid_vec = collect(Bool, valid_vlos)
    vlos_idx = Int[]

    @inbounds for i in 1:Nstar
        valid_vec[i] && push!(vlos_idx, i)
    end

    spatial_edges = resolve_karl_spatial_edges(kinematic_bin_edges)
    Nspatial = length(spatial_edges) - 1
    light_edges = light_bin_edges === nothing ? spatial_edges : resolve_karl_light_edges(light_bin_edges)
    Nlight = length(light_edges) - 1
    velocity_edges_use = velocity_edges === nothing ? build_velocity_edges_auto(v_star_mps[vlos_idx], verr_star_mps[vlos_idx]; Nvbin=Nvbin) : Float64.(velocity_edges)
    Nvbin_eff = length(velocity_edges_use) - 1
    Nlosvd = Nspatial * Nvbin_eff
    shells = _build_orbit_shells(R_star_m, light_edges)
    isempty(shells) && error("Orbit shell grid has no finite positive radii")
    Nshells = length(shells)
    Nbase_orbit >= Nshells || error("Orbit library has $Nbase_orbit base slots for $Nshells required radial shells; increase Norbit")

    A_losvd = zeros(Float64, Nlosvd, Norbit)
    A_light = zeros(Float64, Nlight, Norbit)
    success_flags = fill(false, Nbase_orbit)
    attempts_used = zeros(Int, Nbase_orbit)
    min_r_reached = fill(Inf, Nbase_orbit)
    rapo_list = fill(NaN, Nbase_orbit)
    force_geometry = haskey(ctx.halo, :stellar_model) ? stellar_model_geometry(ctx.halo[:stellar_model]) : :spherical_shell_grid
    theta_launches = force_geometry === :axisymmetric_density_grid ? collect(range(0.15 * pi, 0.85 * pi; length=max(3, Ntheta_launch))) : [f64(pi / 2)]
    launch_order = _balanced_launch_order(Nbase_orbit, shells, Lfrac, theta_launches, shell_band_count)
    first_coverage_check = max(1, ceil(Int, fill_pct * Nbase_orbit))
    sini_use = clamp01(f64(sini))
    cosi_use = sqrt(max(0.0, 1.0 - sini_use * sini_use))
    orbit_ctx = (frc=ctx.frc, R_pos=ctx.R, halo=ctx.halo, force_geometry=force_geometry)

    return OrbitWorkState(Norbit, Nbase_orbit, Nstar, Nspatial, Nvbin_eff, Nlosvd, Nlight, Nshells, nsteps, max_attempts_factor, theta_launches, sini_use, cosi_use, R_star_m, valid_vec, v_star_mps, verr_star_mps, spatial_edges, light_edges, velocity_edges_use, shells, launch_order, orbit_ctx, ctx.pot, ctx.frc, Lfrac, force_geometry, dt_frac_orbit, t_deadline, fill_pct, regional_floor, max_regional_gap, shell_band_count, max(1, coverage_check_every), Threads.Atomic{Int}(first_coverage_check), A_losvd, A_light, success_flags, attempts_used, min_r_reached, rapo_list, Threads.Atomic{Int}(1), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), ReentrantLock())
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

function _assess_orbit_coverage( st::OrbitWorkState;
    fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT,
    regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR,
    max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP,
    shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS,
    verify_atomic::Bool=true)
    0.0 < fill_pct <= 1.0 || error("fill_pct must be in (0, 1]")
    0.0 < regional_floor <= 1.0 || error("regional_floor must be in (0, 1]")
    0.0 <= max_regional_gap <= 1.0 || error("max_regional_gap must be in [0, 1]")
    shell_band_count > 0 || error("shell_band_count must be positive")

    attempted = BitVector(st.attempts_used .> 0)
    succeeded = copy(st.success_flags)
    any(succeeded .& .!attempted) &&
        error("Orbit coverage accounting is inconsistent: a launch succeeded without an attempt")

    planned = st.Nbase_orbit
    attempted_count = count(identity, attempted)
    succeeded_count = count(identity, succeeded)
    !verify_atomic || st.filled_atomic[] == succeeded_count ||
        error("Orbit coverage accounting is inconsistent: filled_atomic=$(st.filled_atomic[]) but success_flags=$succeeded_count")
    required = max(1, ceil(Int, fill_pct * planned))
    total_coverage = succeeded_count / planned

    nshell_bands = min(shell_band_count, st.Nshells)
    nlfrac = length(st.Lfrac)
    ntheta = length(st.theta_launches)
    shell_band_labels = Vector{Int}(undef, planned)
    lfrac_labels = Vector{Int}(undef, planned)
    theta_labels = Vector{Int}(undef, planned)

    @inbounds for c in 1:planned
        shell_id = mod1(c, st.Nshells)
        shell_band_labels[c] = fld((shell_id - 1) * nshell_bands, st.Nshells) + 1
        lfrac_labels[c] = mod1(c, nlfrac)
        theta_labels[c] = mod1(c, ntheta)
    end

    shell_stats, shell_min, shell_gap =
        _coverage_axis_stats(shell_band_labels, attempted, succeeded, nshell_bands)
    lfrac_stats, lfrac_min, lfrac_gap =
        _coverage_axis_stats(lfrac_labels, attempted, succeeded, nlfrac)
    theta_stats, theta_min, theta_gap =
        _coverage_axis_stats(theta_labels, attempted, succeeded, ntheta)

    joint_planned = Dict{NTuple{3,Int},Int}()
    joint_attempted = Dict{NTuple{3,Int},Int}()
    joint_succeeded = Dict{NTuple{3,Int},Int}()
    @inbounds for c in 1:planned
        cell = (shell_band_labels[c], lfrac_labels[c], theta_labels[c])
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
        push!(
            joint_stats,
            (
                shell_band=cell[1],
                lfrac=cell[2],
                theta_launch=cell[3],
                planned=cell_planned,
                attempted=cell_attempted,
                succeeded=cell_succeeded,
                attempted_fraction=attempted_fraction,
                success_fraction=success_fraction,
                coverage_fraction=coverage_fraction,
            ),
        )
        cell_succeeded == 0 && push!(joint_holes, cell)
    end
    sort!(joint_holes)
    joint_min = isempty(joint_coverages) ? 0.0 : minimum(joint_coverages)

    rejection_reasons = String[]
    succeeded_count < required && push!(
        rejection_reasons,
        "total coverage $(round(total_coverage; digits=3)) is below $(fill_pct)",
    )

    for (axis_name, axis_min, axis_gap) in (
        ("shell_band", shell_min, shell_gap),
        ("Lfrac", lfrac_min, lfrac_gap),
        ("theta_launch", theta_min, theta_gap),
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
        "$(length(joint_holes)) planned shell-band/Lfrac/theta cells have no successful orbit",
    )

    successful_launches = findall(succeeded)
    successful_columns = Vector{Int}(undef, 2 * length(successful_launches))
    @inbounds for (j, c) in pairs(successful_launches)
        successful_columns[2 * j - 1] = 2 * c - 1
        successful_columns[2 * j] = 2 * c
    end

    return (
        accepted=isempty(rejection_reasons),
        planned=planned,
        attempted=attempted_count,
        succeeded=succeeded_count,
        required=required,
        attempted_fraction=attempted_count / planned,
        success_fraction=attempted_count == 0 ? 0.0 : succeeded_count / attempted_count,
        coverage_fraction=total_coverage,
        shell_bands=shell_stats,
        lfrac=lfrac_stats,
        theta_launches=theta_stats,
        shell_minimum_coverage=shell_min,
        lfrac_minimum_coverage=lfrac_min,
        theta_minimum_coverage=theta_min,
        shell_coverage_gap=shell_gap,
        lfrac_coverage_gap=lfrac_gap,
        theta_coverage_gap=theta_gap,
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
    Threads.atomic_cas!(
        st.next_coverage_check,
        next_check,
        next_check + st.coverage_check_every,
    ) == next_check || return false

    coverage = _assess_orbit_coverage(
        st;
        fill_pct=st.fill_pct,
        regional_floor=st.regional_floor,
        max_regional_gap=st.max_regional_gap,
        shell_band_count=st.shell_band_count,
        verify_atomic=false,
    )
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

function _compact_wphase(wphase, successful_columns::Vector{Int}, Norbit::Int)
    wphase === nothing && return nothing
    wphase_vec = Float64.(wphase)
    length(wphase_vec) == Norbit ||
        error("wphase length $(length(wphase_vec)) does not match the planned Norbit=$Norbit")
    return wphase_vec[successful_columns]
end

function _orbit_worker!(st::OrbitWorkState)
    col_losvd_pro = zeros(Float64, st.Nlosvd)
    col_losvd_ret = zeros(Float64, st.Nlosvd)
    col_light = zeros(Float64, st.Nlight)
    s_arr = Vector{Float64}(undef, st.nsteps)
    vlos_pro_buf = Vector{Float64}(undef, st.nsteps)
    vlos_ret_buf = Vector{Float64}(undef, st.nsteps)
    while true
        time_ns() > st.t_deadline && break
        st.phase[] != 1 && break
        slot_seq = Threads.atomic_add!(st.next_orbit, 1)
        slot_seq > st.Nbase_orbit && break
        c_claim = st.launch_order[slot_seq]
        idx_local = mod1(c_claim, st.Nshells)
        rapo = f64(st.shells[idx_local])
        st.rapo_list[c_claim] = rapo
        !(isfinite(rapo) && rapo > 0.0) && continue
        lf = st.Lfrac[1 + ((c_claim - 1) % length(st.Lfrac))]

        for attempt in 0:(st.max_attempts_factor - 1)
            time_ns() > st.t_deadline && return nothing
            st.phase[] != 1 && return nothing
            st.attempts_used[c_claim] = attempt + 1
            rng = MersenneTwister(_orbit_attempt_seed(c_claim, attempt))
            r0_frac = 0.95 - 0.05 * f64(attempt) / st.max_attempts_factor + 0.04 * rand(rng)
            theta0 = st.theta_launches[mod1(c_claim, length(st.theta_launches))]
            ic, Lz0, E0, vc, launch_state = launch_orbit_apocenter(rapo=rapo, theta0=theta0, Lz_frac=f64(lf), pot=st.pot, frc=st.frc, r0_frac=r0_frac, dt_frac=st.dt_frac_orbit)
            launch_state != :ok && continue
            r, vr, theta, vtheta = integrate_orbit_rk4( ic=ic, xLz=Lz0, orbit_ctx=st.orbit_ctx, nsteps=st.nsteps)
            isempty(r) && continue

            st.min_r_reached[c_claim] = minimum(r)
            Nhits = length(r)
            dt_orb = f64(ic[3])
            resize!(s_arr, Nhits)
            resize!(vlos_pro_buf, Nhits)
            resize!(vlos_ret_buf, Nhits)
            phi = 0.0
            @inbounds for i in 1:Nhits
                ri = f64(r[i])
                thi = f64(theta[i])
                si = _ssin(thi)
                vphi_i = f64(Lz0) / max(ri * si, 1e-30)
                s_arr[i], vlos_pro_buf[i] = _project_axisym_sample(ri, f64(vr[i]), f64(vtheta[i]), vphi_i, thi, phi, st.sini, st.cosi)
                _, vlos_ret_buf[i] = _project_axisym_sample(ri, f64(vr[i]), f64(vtheta[i]), -vphi_i, thi, phi, st.sini, st.cosi)
                phi += f64(Lz0) / max(ri * ri * si * si, 1e-30) * dt_orb
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
            if !(isfinite(pro_activity) && pro_activity > 0.0 &&
                 isfinite(ret_activity) && ret_activity > 0.0)
                continue
            end
            col_pro = 2 * c_claim - 1
            col_ret = 2 * c_claim
            @inbounds st.A_losvd[:, col_pro] .= col_losvd_pro
            @inbounds st.A_losvd[:, col_ret] .= col_losvd_ret
            @inbounds st.A_light[:, col_pro] .= col_light
            @inbounds st.A_light[:, col_ret] .= col_light
            st.success_flags[c_claim] = true
            Threads.atomic_add!(st.filled_atomic, 1)
            _maybe_stop_orbit_phase_for_coverage!(st)
            break
        end
    end
    return nothing
end

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

    st = _init_orbit_work(Norbit, R_star_m, has_vlos, v_star_mps, verr_star_mps, sini, ctx; nsteps=nsteps, Lfrac=Lfrac, dt_frac_orbit=dt_frac_orbit, max_attempts_factor=max_attempts_factor, t_deadline=t_deadline, velocity_edges=velocity_edges, light_bin_edges=light_bin_edges, kinematic_bin_edges=kinematic_bin_edges, Nvbin=Nvbin, Ntheta_launch=Ntheta_launch, fill_pct=fill_pct, regional_floor=regional_floor, max_regional_gap=max_regional_gap, shell_band_count=shell_band_count)

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
    coverage.accepted || error("Incomplete orbit coverage: filled $filled / $(st.Nbase_orbit) base slots; " * join(coverage.rejection_reasons, " | "))
    _orbit_library_usable(st, coverage.successful_columns) || error("Unusable orbit library: a successful paired orbit column or projected-light row has no support")

    A_losvd, A_light = _compact_orbit_matrices(st, coverage.successful_columns)
    A = vcat(A_losvd, A_light)

    if diag
        losvd_target, losvd_sigma, light_target, counts_by_spatial = observed_targets_karl(R_star_m, has_vlos, v_star_mps, verr_star_mps, st.spatial_edges, st.velocity_edges; surface_brightness_profile=surface_brightness_profile_jl, light_edges=st.light_edges)

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
                "counts_by_spatial" => counts_by_spatial,
                "force_geometry" => String(st.force_geometry),
            ),
        )
    end

    return A
end

# Batch evaluator: Karl-style binned LOSVD + projected-light fit.
# This is the Heart of the whole Pipeline 
# and is where all the parallelism is implemented
function evaluate_batch_theta(thetas::AbstractMatrix{<:Real}, R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, sini::Float64, Norbit::Int, halo_type::String; stellar_model=nothing, surface_brightness_profile=nothing, alphat::Float64=DEFAULT_KARL_ALPHAT, light_rel_tol::Float64=DEFAULT_KARL_LIGHT_REL_TOL, delta_chi2_iter_tol::Float64=DEFAULT_KARL_DELTA_CHI2_ITER_TOL, wphase=nothing, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, maxiter::Int=DEFAULT_KARL_MAXITER, timeout_s::Float64=120.0, fill_pct::Float64=DEFAULT_ORBIT_FILL_PCT, regional_floor::Float64=DEFAULT_ORBIT_REGIONAL_FLOOR, max_regional_gap::Float64=DEFAULT_ORBIT_MAX_REGIONAL_GAP, shell_band_count::Int=DEFAULT_ORBIT_SHELL_BANDS, coverage_check_every::Int=DEFAULT_ORBIT_COVERAGE_CHECK_EVERY, warn_fill_pct::Float64=DEFAULT_ORBIT_WARN_FILL_PCT, warn_success_pct::Float64=DEFAULT_ORBIT_WARN_SUCCESS_PCT, warn_regional_floor::Float64=DEFAULT_ORBIT_WARN_REGIONAL_FLOOR, warn_max_regional_gap::Float64=DEFAULT_ORBIT_WARN_MAX_REGIONAL_GAP, model_owner_limit::Int=0, threads_per_model::Int=8, R_inner_pc::Float64=30.0, velocity_edges=nothing, kinematic_bin_edges=nothing, light_bin_edges=nothing, Nvbin::Int=21, Ntheta_launch::Int=9, halo_q_axis_ratio::Float64=1.0, karl_halo_params=nothing)
    nrow, nbatch = size(thetas)
    surface_brightness_profile === nothing && error("surface_brightness_profile is required for Karl-style OSPM; no star-count fallback is allowed")
    light_rel_tol > 0.0 || error("light_rel_tol must be positive")
    delta_chi2_iter_tol >= 0.0 || error("delta_chi2_iter_tol must be nonnegative")
    threads_per_model > 0 || error("threads_per_model must be positive")
    stellar_model_jl = normalize_stellar_model(stellar_model)
    surface_brightness_profile_jl = normalize_surface_brightness_profile(surface_brightness_profile)
    prewarm_stellar_force_cache(stellar_model_jl)
    status = fill(4, nbatch)
    chi2_losvd = fill(Inf, nbatch)
    chi2_inner = fill(Inf, nbatch)
    chi2_outer = fill(Inf, nbatch)
    delta_chi2_iteration = fill(Inf, nbatch)
    max_light_relative_residual = fill(Inf, nbatch)
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
    work_states = Vector{Union{Nothing, OrbitWorkState}}(undef, nbatch)
    fill!(work_states, nothing)
    next_theta = Threads.Atomic{Int}(1)
    nthreads = Threads.nthreads()
    owner_limit = model_owner_limit > 0 ? clamp(model_owner_limit, 1, nthreads) : max(1, cld(nthreads, threads_per_model))
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
    function _store_solver_diagnostics!(i::Int, wdiag)
        delta_chi2_iteration[i] = Float64(_wdiag_value(wdiag, :delta_chi2_iteration, Inf))
        max_light_relative_residual[i] = Float64(_wdiag_value(wdiag, :max_light_relative_residual, Inf))
        light_constraint_ok[i] = Bool(_wdiag_value(wdiag, :light_constraint_ok, false))
        solver_converged[i] = Bool(_wdiag_value(wdiag, :solver_converged, false))
        solver_iterations[i] = Int(_wdiag_value(wdiag, :iterations, 0))
        solver_failure_reason[i] = string(_wdiag_value(wdiag, :failure_reason, :missing_diagnostics))
        return nothing
    end
    function _print_karl_diagnostics!(i::Int, tid::Int, wdiag, chi2_score::Float64)
        wdiag === nothing && return nothing
        println("[KARL SOLVER DIAG] i=", i, " tid=", tid, " chi_losvd_score=", chi2_score, " chi_losvd_solver=", _wdiag_value(wdiag, :chi_losvd, NaN), " delta_chi2_iteration=", _wdiag_value(wdiag, :delta_chi2_iteration, NaN), " delta_chi2_iteration_ok=", _wdiag_value(wdiag, :delta_chi2_iteration_ok, false), " max_light_relative_residual=", _wdiag_value(wdiag, :max_light_relative_residual, NaN), " light_constraint_ok=", _wdiag_value(wdiag, :light_constraint_ok, false), " entropy=", _wdiag_value(wdiag, :entropy, NaN), " profit=", _wdiag_value(wdiag, :profit, NaN), " losvd_penalty=", _wdiag_value(wdiag, :losvd_penalty, NaN), " chi_slack=", _wdiag_value(wdiag, :chi_slack, NaN), " slack_to_losvd=", _wdiag_value(wdiag, :slack_to_losvd, NaN), " slack_l2=", _wdiag_value(wdiag, :slack_l2, NaN), " slack_max_abs=", _wdiag_value(wdiag, :slack_max_abs, NaN), " rcond_est=", _wdiag_value(wdiag, :rcond_est, NaN), " max_abs_dw=", _wdiag_value(wdiag, :max_abs_dw, NaN), " stepfac=", _wdiag_value(wdiag, :stepfac, NaN), " iterations=", _wdiag_value(wdiag, :iterations, 0), " constraint_ok=", _wdiag_value(wdiag, :constraint_ok, false), " slack_consistent=", _wdiag_value(wdiag, :slack_consistent, false), " normalized=", _wdiag_value(wdiag, :normalized, false), " solver_converged=", _wdiag_value(wdiag, :solver_converged, false), " failure_reason=", _wdiag_value(wdiag, :failure_reason, :none), " N_nonzero=", N_nonzero_weights[i], " Neff=", effective_N_orbits[i], " max_weight_fraction=", max_weight_fraction[i])
        return nothing
    end
    function _print_karl_failure!(i::Int, tid::Int, wdiag)
        println("[KARL SOLVER FAIL] i=", i, " tid=", tid, " failure_reason=", _wdiag_value(wdiag, :failure_reason, :missing_diagnostics), " delta_chi2_iteration=", _wdiag_value(wdiag, :delta_chi2_iteration, NaN), " max_light_relative_residual=", _wdiag_value(wdiag, :max_light_relative_residual, NaN), " light_constraint_ok=", _wdiag_value(wdiag, :light_constraint_ok, false), " solver_converged=", _wdiag_value(wdiag, :solver_converged, false), " iterations=", _wdiag_value(wdiag, :iterations, 0), " rcond_est=", _wdiag_value(wdiag, :rcond_est, NaN), " max_abs_dw=", _wdiag_value(wdiag, :max_abs_dw, NaN), " stepfac=", _wdiag_value(wdiag, :stepfac, NaN), " chi_slack=", _wdiag_value(wdiag, :chi_slack, NaN))
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
                owner_slot_held = true
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
                        println("[ORBIT COVERAGE WARNING] i=", i, " coverage_status=", coverage_status[i], " issue_region=", coverage_issue_region[i], " issue_axis=", coverage_issue_axis[i], " issue_shell_bands=", coverage_issue_shell_bands[i], " filled=", coverage.succeeded, " required=", coverage.required, " planned=", coverage.planned, " attempted=", coverage.attempted, " attempted_fraction=", coverage.attempted_fraction, " success_fraction=", coverage.success_fraction, " coverage_fraction=", coverage.coverage_fraction, " shell_min=", coverage.shell_minimum_coverage, " lfrac_min=", coverage.lfrac_minimum_coverage, " theta_min=", coverage.theta_minimum_coverage, " shell_gap=", coverage.shell_coverage_gap, " lfrac_gap=", coverage.lfrac_coverage_gap, " theta_gap=", coverage.theta_coverage_gap, " joint_holes=", length(coverage.joint_holes), " reasons=", join(coverage.rejection_reasons, " | "), " timed_out=", coverage_deadline_hit[i])
                    end
                    if !_orbit_library_usable(ws, coverage.successful_columns)
                        status[i] = 1
                        solver_failure_reason[i] = "unusable_orbit_library"
                        println("[ORBIT LIBRARY REJECTED] i=", i, " reason=zero_support_successful_column_or_light_row")
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end
                    A_losvd, A_light = _compact_orbit_matrices(ws, coverage.successful_columns)
                    wphase_use = _compact_wphase(wphase, coverage.successful_columns, ws.Norbit)
                    if size(A_losvd, 1) == 0 || size(A_losvd, 2) == 0 || !all(isfinite, A_losvd) || size(A_light, 1) == 0 || size(A_light, 2) == 0 || !all(isfinite, A_light)
                        status[i] = 1
                        solver_failure_reason[i] = "invalid_observable_matrix"
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end
                    losvd_target, losvd_sigma, light_target, counts_by_spatial = observed_targets_karl(R_star_m, valid_vlos, v_star_mps, verr_star_mps, ws.spatial_edges, ws.velocity_edges; surface_brightness_profile=surface_brightness_profile_jl, light_edges=ws.light_edges)
                    light_fit_mask = ws.light_edges[2:end] .<= ws.spatial_edges[end]
                    any(light_fit_mask) || error("No projected-light bins lie completely inside the kinematic footprint")
                    A_light_fit = A_light[light_fit_mask, :]
                    light_target_fit = light_target[light_fit_mask]
                    if i == 1
                        println(
                            "[KARL LIGHT FIT DIAG] N_light_full=", length(light_target),
                            " N_light_fit=", length(light_target_fit),
                            " R_light_full_max_pc=", ws.light_edges[end] / pc,
                            " R_light_fit_max_pc=", maximum(ws.light_edges[2:end][light_fit_mask]) / pc,
                            " R_kin_max_pc=", ws.spatial_edges[end] / pc,
                            " target_full_sum=", sum(light_target),
                            " target_fit_sum=", sum(light_target_fit),
                        )
                    end
                    Threads.atomic_add!(scheduler_counters.weight_models, 1)
                    w = Float64[]
                    ok = false
                    wdiag = nothing
                    try
                        w, ok, wdiag = solve_weights_karl_expanded_cm(A_light_fit, A_losvd, light_target_fit, losvd_target, losvd_sigma; alphat=alphat, light_rel_tol=light_rel_tol, delta_chi2_iter_tol=delta_chi2_iter_tol, wphase=wphase_use, maxiter=maxiter, seed=UInt(i), entropy_floor=entropy_floor, apfac=DEFAULT_KARL_APFAC, return_diag=true)
                    finally
                        Threads.atomic_add!(scheduler_counters.weight_models, -1)
                    end
                    _store_solver_diagnostics!(i, wdiag)
                    if !ok
                        if length(w) == size(A_light_fit, 2)
                            light_model_fit = A_light_fit * w
                            light_relative_fit = abs.(light_model_fit .- light_target_fit) ./ max.(abs.(light_target_fit), 1e-12)
                            jfit = argmax(light_relative_fit)
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
                            )
                        else
                            println(
                                "[KARL LIGHT FAIL BIN] unavailable=true",
                                " weight_count=", length(w),
                                " expected_weight_count=", size(A_light_fit, 2),
                            )
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
                    owner_slot_held && Threads.atomic_add!(scheduler_counters.active_model_owners, -1)
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
                ws_scan.next_orbit[] > ws_scan.Nbase_orbit && continue
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
    return (status, chi2_losvd, chi2_inner, chi2_outer, delta_chi2_iteration, max_light_relative_residual, light_constraint_ok, solver_converged, solver_iterations, solver_failure_reason, N_inner, N_outer, N_nonzero_weights, effective_N_orbits, max_weight_fraction, coverage_status, coverage_issue_region, coverage_issue_axis, coverage_issue_shell_bands, coverage_reasons, coverage_fraction, coverage_attempted_fraction, coverage_success_fraction, coverage_shell_min, coverage_lfrac_min, coverage_theta_min, coverage_shell_gap, coverage_lfrac_gap, coverage_theta_gap, coverage_joint_holes, coverage_deadline_hit, successful_base_orbits, planned_base_orbits)
end
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
end # module
