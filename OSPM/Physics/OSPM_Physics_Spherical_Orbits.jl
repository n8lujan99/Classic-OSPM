# ========================================================================================================================
# OSPM_Physics_Spherical_Orbits.jl — spherical orbit-family construction and integration.
# Included by OSPM_Physics_Spherical.jl — do NOT load directly.
# Owns orbit equations of motion, zero-velocity-curve launch construction,
# orbit-family integrals, apocenter initial conditions, and RK4 orbit integration.
# Weight solving and force construction are owned by their dedicated physics files.
# ========================================================================================================================

@inline function derivs(s::SVector{4,Float64}, Lz::Float64, frc, R)
    invalid = SVector(NaN, NaN, NaN, NaN)
    Rcyl, z, vR, vz = s
    length(R) >= 2 || return invalid
    all(isfinite, s) && isfinite(Lz) || return invalid
    Rcyl >= 0.0 || return invalid
    r = hypot(Rcyl, z)
    isfinite(r) && r > 0.0 || return invalid
    rmin_force = f64(R[1])
    rmax_force = f64(R[end])
    isfinite(rmin_force) && isfinite(rmax_force) &&
        0.0 < rmin_force < rmax_force || return invalid
    rmin_force <= r <= rmax_force || return invalid
    if Lz != 0.0 && Rcyl <= 0.0
        return invalid
    end
    st = Rcyl / r
    ct = z / r
    theta = atan(Rcyl, z)
    fr, ftheta = frc(r, theta)
    isfinite(fr) && isfinite(ftheta) || return invalid
    FR = fr * st + ftheta * ct
    Fz = fr * ct - ftheta * st
    centrifugal = Lz == 0.0 ? 0.0 : (Lz^2) / (Rcyl^3)
    dR = vR
    dz = vz
    dvR = centrifugal + FR
    dvz = Fz
    result = SVector(dR, dz, dvR, dvz)
    return all(isfinite, result) ? result : invalid
end

function _outer_zero_velocity_radius(; energy::Float64, lz::Float64, theta0::Float64, rapo_max::Float64, pot, nscan::Int=256, max_expand::Int=32, expand_factor::Float64=1.5)
    ss = _ssin(theta0)
    if !(isfinite(ss) && abs(ss) > EPS_SIN)
        return NaN, :reject_sin
    end
    if !(isfinite(energy) && isfinite(lz) && isfinite(rapo_max) && rapo_max > 0.0)
        return NaN, :reject_family_integrals
    end
    nscan > 1 || error("nscan must exceed one")
    max_expand > 0 || error("max_expand must be positive")
    isfinite(expand_factor) && expand_factor > 1.0 ||
        error("expand_factor must exceed one")

    function radial_budget(r::Float64)
        P = pot(r, theta0)
        isfinite(P) || return NaN
        return 2.0 * (energy - P) - (lz^2) / (r^2 * ss^2)
    end

    r_seed = rapo_max
    budget_seed = radial_budget(r_seed)
    isfinite(budget_seed) || return NaN, :reject_pot

    P_seed = pot(r_seed, theta0)
    centrifugal_seed = (lz^2) / (r_seed^2 * ss^2)
    scale = max( abs(2.0 * energy), isfinite(P_seed) ? abs(2.0 * P_seed) : 0.0, abs(centrifugal_seed), abs(budget_seed), 1.0)
    budget_tol = 1.0e-12 * scale
    abs(budget_seed) <= budget_tol && return r_seed, :ok

    r_inner = NaN
    budget_inner = NaN
    r_outer = NaN
    budget_outer = NaN
    bracket_found = false

    if budget_seed > 0.0
        # The off-equatorial outer boundary can lie outside the equatorial
        # reference apocenter. Expand outward until the forbidden side of the
        # zero-velocity curve is reached.
        r_inner = r_seed
        budget_inner = budget_seed
        @inbounds for _ in 1:max_expand
            candidate = r_inner * expand_factor
            budget_candidate = radial_budget(candidate)
            if !isfinite(budget_candidate)
                r_inner = candidate
                continue
            end
            if budget_candidate <= 0.0
                r_outer = candidate
                budget_outer = budget_candidate
                bracket_found = true
                break
            end
            r_inner = candidate
            budget_inner = budget_candidate
        end
    else
        # The supplied radius is already outside the accessible region. Scan
        # inward until the allowed side of the outer zero-velocity boundary is
        # found.
        r_outer = r_seed
        budget_outer = budget_seed
        r_floor = max(rapo_max * 1.0e-10, 1.0e-12)
        log_ratio = log(r_floor / rapo_max)
        previous_r = r_outer
        previous_budget = budget_outer

        @inbounds for k in 1:nscan
            candidate = rapo_max * exp(log_ratio * k / nscan)
            budget_candidate = radial_budget(candidate)
            isfinite(budget_candidate) || continue

            if budget_candidate >= 0.0 && previous_budget <= 0.0
                r_inner = candidate
                budget_inner = budget_candidate
                r_outer = previous_r
                budget_outer = previous_budget
                bracket_found = true
                break
            end

            previous_r = candidate
            previous_budget = budget_candidate
        end
    end
    bracket_found || return NaN, :reject_zero_velocity_curve
    @inbounds for _ in 1:80
        r_mid = 0.5 * (r_inner + r_outer)
        budget_mid = radial_budget(r_mid)
        isfinite(budget_mid) || return NaN, :reject_pot
        if budget_mid >= 0.0
            r_inner = r_mid
            budget_inner = budget_mid
        else
            r_outer = r_mid
            budget_outer = budget_mid
        end
        abs(r_outer - r_inner) <=
            1.0e-12 * max(abs(r_mid), 1.0) && break
    end
    return 0.5 * (r_inner + r_outer), :ok
end

function _family_zvc_launches(; energy::Float64, lz::Float64, rapo::Float64, pot, Nlaunch::Int, circular_boundary::Bool=false, theta_floor::Float64=1.0e-4, ncurve::Int=0)
    Nlaunch > 0 || error("Nlaunch must be positive")
    isfinite(theta_floor) && 0.0 < theta_floor < pi / 2 || error("theta_floor must lie strictly between zero and pi/2")
    isfinite(energy) && isfinite(lz) && isfinite(rapo) && rapo > 0.0 || return (state=:reject_family_integrals, u=Float64[], r=Float64[], theta=Float64[], R=Float64[], z=Float64[], arc_length=NaN)
    theta_equator = f64(pi / 2)
    if circular_boundary
        r_equator, state = _outer_zero_velocity_radius(energy=energy, lz=lz, theta0=theta_equator, rapo_max=rapo, pot=pot)
        state == :ok || return (state=state, u=Float64[], r=Float64[], theta=Float64[], R=Float64[], z=Float64[], arc_length=NaN)
        return (state=:ok, u=[1.0], r=[r_equator], theta=[theta_equator], R=[r_equator], z=[0.0], arc_length=0.0)
    end
    ncurve_use = ncurve > 0 ? ncurve : max(257, 32 * Nlaunch)
    ncurve_use >= Nlaunch || error("ncurve must be at least as large as Nlaunch")
    theta_scan = collect(range(theta_floor, theta_equator; length=ncurve_use))
    radius_scan = fill(NaN, ncurve_use)
    valid_scan = falses(ncurve_use)
    @inbounds for i in eachindex(theta_scan)
        radius, state = _outer_zero_velocity_radius(energy=energy, lz=lz, theta0=theta_scan[i], rapo_max=rapo, pot=pot)
        if state == :ok && isfinite(radius) && radius > 0.0
            radius_scan[i] = radius
            valid_scan[i] = true
        end
    end
    valid_scan[end] || return (state=:reject_equatorial_zvc, u=Float64[], r=Float64[], theta=Float64[], R=Float64[], z=Float64[], arc_length=NaN)
    # Keep only the contiguous accessible branch that terminates at the
    # equatorial plane. Any isolated lower-theta roots are not part of this
    # one-sided family sequence.
    first_valid = length(valid_scan)
    while first_valid > 1 && valid_scan[first_valid - 1]
        first_valid -= 1
    end
    theta_curve = Float64[]
    radius_curve = Float64[]
    if first_valid > 1
        theta_invalid = theta_scan[first_valid - 1]
        theta_valid = theta_scan[first_valid]
        radius_valid = radius_scan[first_valid]
        @inbounds for _ in 1:60
            theta_mid = 0.5 * (theta_invalid + theta_valid)
            radius_mid, state_mid = _outer_zero_velocity_radius(energy=energy, lz=lz, theta0=theta_mid, rapo_max=rapo, pot=pot)
            if state_mid == :ok && isfinite(radius_mid) && radius_mid > 0.0
                theta_valid = theta_mid
                radius_valid = radius_mid
            else
                theta_invalid = theta_mid
            end
        end
        push!(theta_curve, theta_valid)
        push!(radius_curve, radius_valid)
        push!(theta_curve, theta_valid)
        push!(radius_curve, radius_valid)
        append!(theta_curve, theta_scan[first_valid:end])
        append!(radius_curve, radius_scan[first_valid:end])
    else
        append!(theta_curve, theta_scan)
        append!(radius_curve, radius_scan)
    end
    length(theta_curve) >= 2 || return (state=:reject_degenerate_zvc, u=Float64[], r=Float64[], theta=Float64[], R=Float64[], z=Float64[], arc_length=0.0)
    R_curve = similar(radius_curve)
    z_curve = similar(radius_curve)
    @inbounds for i in eachindex(radius_curve)
        st, ct = sincos(theta_curve[i])
        R_curve[i] = radius_curve[i] * st
        z_curve[i] = radius_curve[i] * ct
    end
    arc = zeros(Float64, length(radius_curve))
    @inbounds for i in 2:length(radius_curve)
        dR = R_curve[i] - R_curve[i - 1]
        dz = z_curve[i] - z_curve[i - 1]
        arc[i] = arc[i - 1] + hypot(dR, dz)
    end
    total_arc = arc[end]
    arc_scale = max(maximum(abs, R_curve), maximum(abs, z_curve), rapo, 1.0)
    if !(isfinite(total_arc) && total_arc > 1.0e-12 * arc_scale)
        return ( state=:reject_degenerate_zvc, u=Float64[], r=Float64[], theta=Float64[], R=Float64[], z=Float64[], arc_length=total_arc)
    end
    u_launch = collect(range(0.0, 1.0; length=Nlaunch))
    r_launch = Vector{Float64}(undef, Nlaunch)
    theta_launch = Vector{Float64}(undef, Nlaunch)
    R_launch = Vector{Float64}(undef, Nlaunch)
    z_launch = Vector{Float64}(undef, Nlaunch)
    @inbounds for i in eachindex(u_launch)
        target_arc = u_launch[i] * total_arc
        if i == 1
            theta_target = theta_curve[1]
        elseif i == Nlaunch
            theta_target = theta_curve[end]
        else
            j = clamp(searchsortedlast(arc, target_arc), 1, length(arc) - 1)
            ds = arc[j + 1] - arc[j]
            frac = ds > 0.0 ? (target_arc - arc[j]) / ds : 0.0
            theta_target =
                theta_curve[j] + frac * (theta_curve[j + 1] - theta_curve[j])
        end
        radius_target, state_target = _outer_zero_velocity_radius(energy=energy, lz=lz, theta0=theta_target, rapo_max=rapo, pot=pot)
        state_target == :ok || return (state=state_target, u=Float64[], r=Float64[], theta=Float64[], R=Float64[], z=Float64[], arc_length=total_arc)
        st, ct = sincos(theta_target)
        r_launch[i] = radius_target
        theta_launch[i] = theta_target
        R_launch[i] = radius_target * st
        z_launch[i] = radius_target * ct
    end
    return (state=:ok, u=u_launch, r=r_launch, theta=theta_launch, R=R_launch, z=z_launch, arc_length=total_arc)
end

function orbit_family_integrals(; rapo::Float64, Lz_frac::Float64, pot, frc, debug::Bool=true)
    theta_reference = f64(pi / 2)
    if !(isfinite(Lz_frac) && 0.0 <= Lz_frac <= 1.0)
        return (0.0, 0.0, 0.0, :reject_lfrac)
    end
    frs, _ = frc(rapo, theta_reference)
    if !(isfinite(frs) && isfinite(rapo) && rapo > 0.0)
        return (0.0, 0.0, 0.0, :reject_force)
    end
    r_in  = rapo * (1 - BRACKET_FRAC)
    r_out = rapo * (1 + BRACKET_FRAC)
    fr_in,  _ = frc(r_in,  theta_reference)
    fr_out, _ = frc(r_out, theta_reference)
    fr_scale = max(abs(frs), abs(fr_in), abs(fr_out), EPS_FORCE)
    fr_tol   = max(EPS_FORCE, REL_FORCE * fr_scale)
    if frs > fr_tol
        return debug ?
            (0.0, 0.0, 0.0, :reject_force) :
            (0.0, 0.0, 0.0, :reject_force)
    end
    vc2 = (-frs) * rapo
    if vc2 <= 0.0
        vc2 = fr_tol * rapo
    end
    vc = sqrt(vc2)
    if !(isfinite(vc) && vc > EPS_VEL)
        return (0.0, 0.0, 0.0, :reject_vc)
    end
    Lz = Lz_frac * rapo * vc
    Papo = pot(rapo, theta_reference)
    if !isfinite(Papo)
        return (0.0, 0.0, vc, :reject_pot)
    end
    E = Papo + (Lz^2) / (2 * rapo^2)
    return Lz, E, vc, :ok
end

function launch_orbit_apocenter(; rapo::Float64, theta0::Float64, Lz_frac::Float64, pot, frc, r0_frac::Float64=DEFAULT_R0_FRAC, dt_frac::Float64=DEFAULT_DT_FRAC,
    dt_floor::Float64=DEFAULT_DT_FLOOR, fixed_energy=nothing, fixed_lz=nothing, fixed_rturn=nothing, debug::Bool=true)
    ss = _ssin(theta0)
    if !(isfinite(ss) && abs(ss) > EPS_SIN)
        return (nothing, 0.0, 0.0, 0.0, :reject_sin)
    end
    if !(isfinite(Lz_frac) && 0.0 <= Lz_frac <= 1.0)
        return (nothing, 0.0, 0.0, 0.0, :reject_lfrac)
    end
    if (fixed_energy === nothing) != (fixed_lz === nothing)
        return (nothing, 0.0, 0.0, 0.0, :reject_family_integrals)
    end
    if fixed_rturn !== nothing && fixed_energy === nothing
        return (nothing, 0.0, 0.0, 0.0, :reject_fixed_rturn_without_family)
    end

    if fixed_energy !== nothing
        E = f64(fixed_energy)
        Lz = abs(f64(fixed_lz))

        rturn = if fixed_rturn === nothing
            radius, turning_state = _outer_zero_velocity_radius(energy=E, lz=Lz, theta0=theta0, rapo_max=rapo, pot=pot)
            turning_state != :ok &&
                return (nothing, Lz, E, 0.0, turning_state)
            radius
        else
            radius = f64(fixed_rturn)
            if !(isfinite(radius) && radius > 0.0)
                return (nothing, Lz, E, 0.0, :reject_fixed_rturn)
            end

            Pturn = pot(radius, theta0)
            isfinite(Pturn) || return (nothing, Lz, E, 0.0, :reject_pot0)

            centrifugal = (Lz^2) / (radius^2 * ss^2)
            budget = 2.0 * (E - Pturn) - centrifugal
            budget_scale = max(abs(2.0 * E), abs(2.0 * Pturn), abs(centrifugal), 1.0)
            abs(budget) <= 1.0e-8 * budget_scale ||
                return debug ?
                    ((radius, theta0, Lz, E, budget), Lz, E, 0.0, :reject_fixed_zvc) :
                    (nothing, Lz, E, 0.0, :reject_fixed_zvc)
            radius
        end

        frturn, _ = frc(rturn, theta0)
        isfinite(frturn) || return (nothing, Lz, E, 0.0, :reject_force)
        vc = sqrt(max(abs(frturn) * rturn, EPS_VEL^2))
        vphi = abs(Lz) / max(rturn * abs(ss), 1.0e-30)
        velocity_scale = max(vc, vphi, EPS_VEL)
        Om = velocity_scale / rturn
        dt = dt_frac / max(Om, dt_floor)

        return ((rturn, theta0, dt, 0.0, 0.0), Lz, E, vc, :ok)
    end

    frs, _ = frc(rapo, theta0)
    if !(isfinite(frs) && isfinite(rapo) && rapo > 0.0)
        return (nothing, 0.0, 0.0, 0.0, :reject_force)
    end
    r_in  = rapo * (1 - BRACKET_FRAC)
    r_out = rapo * (1 + BRACKET_FRAC)
    fr_in,  _ = frc(r_in,  theta0)
    fr_out, _ = frc(r_out, theta0)
    fr_scale = max(abs(frs), abs(fr_in), abs(fr_out), EPS_FORCE)
    fr_tol   = max(EPS_FORCE, REL_FORCE * fr_scale)
    if frs > fr_tol
        return debug ?
            ((rapo, theta0, ss, frs, fr_tol, fr_scale), 0.0, 0.0, 0.0, :reject_force) :
            (nothing, 0.0, 0.0, 0.0, :reject_force)
    end
    vc2 = (-frs) * rapo
    if vc2 <= 0.0
        vc2 = fr_tol * rapo
    end
    vc = sqrt(vc2)
    if !(isfinite(vc) && vc > EPS_VEL)
        return debug ?
            ((rapo, theta0, ss, frs, vc, EPS_VEL), 0.0, 0.0, 0.0, :reject_vc) :
            (nothing, 0.0, 0.0, 0.0, :reject_vc)
    end
    Lz = Lz_frac * rapo * abs(ss) * vc
    Papo = pot(rapo, theta0)
    if !isfinite(Papo)
        return (nothing, 0.0, 0.0, vc, :reject_pot)
    end
    E = Papo + (Lz^2) / (2 * rapo^2 * ss^2)
    if Lz_frac == 1.0
        Om = abs(vc / rapo)
        dt = dt_frac / max(Om, dt_floor)
        return ((rapo, theta0, dt, 0.0, 0.0), Lz, E, vc, :ok)
    end
    if !(isfinite(r0_frac) && 0.0 < r0_frac < 1.0)
        return (nothing, Lz, E, vc, :reject_r0)
    end
    r0 = r0_frac * rapo
    P0 = pot(r0, theta0)
    if !isfinite(P0)
        return (nothing, 0.0, E, vc, :reject_pot0)
    end
    arg = 2 * (E - P0) - (Lz^2) / (r0^2 * ss^2)
    if !(isfinite(arg) && arg > -EPS_ARG)
        return debug ?
            ((rapo, theta0, Lz, arg), Lz, E, vc, :reject_turning) :
            (nothing, Lz, E, vc, :reject_turning)
    end
    vr0 = -sqrt(max(arg, 0.0))
    Om  = abs(vc / r0)
    dt  = dt_frac / max(Om, dt_floor)
    return ((r0, theta0, dt, vr0, 0.0), Lz, E, vc, :ok)
end

function integrate_orbit_rk4(; ic, xLz, orbit_ctx, nsteps=DEFAULT_NSTEPS, stop_rmin_factor=DEFAULT_STOP_RMIN_FACTOR, return_diag::Bool=false, pot=nothing, energy_check_every::Int=100, dt_scale::Float64=1.0, max_relative_energy_drift_allowed::Float64=Inf, energy_drift_boundary_allowance::Float64=5.0e-4, local_step_safety::Float64=0.10, max_substeps_per_step::Int=256, continuation_state=nothing, reference_energy=nothing)
    isfinite(dt_scale) && dt_scale > 0.0 || error("dt_scale must be finite and positive")
    !isnan(max_relative_energy_drift_allowed) && max_relative_energy_drift_allowed > 0.0 || error("max_relative_energy_drift_allowed must be positive or Inf")
    isfinite(energy_drift_boundary_allowance) && energy_drift_boundary_allowance >= 0.0 || error("energy_drift_boundary_allowance must be finite and nonnegative")
    isfinite(local_step_safety) && local_step_safety > 0.0 || error("local_step_safety must be finite and positive")
    max_substeps_per_step > 0 || error("max_substeps_per_step must be positive")

    ns = Int(nsteps)
    ns > 0 || error("nsteps must be positive")
    length(orbit_ctx.R_pos) >= 2 || error("orbit force-radius grid must contain at least two points")

    return_diag && pot === nothing && error("integrate_orbit_rk4 requires pot when return_diag=true")
    return_diag && energy_check_every <= 0 && error("energy_check_every must be positive")

    energy_drift_limit = max_relative_energy_drift_allowed + energy_drift_boundary_allowance

    halo = orbit_ctx.halo
    force_rmin = f64(orbit_ctx.R_pos[1])
    force_rmax = f64(orbit_ctx.R_pos[end])
    halo_rmin_stop = stop_rmin_factor * f64(halo[:rmin])
    rmin_stop = max(force_rmin, halo_rmin_stop)
    rmax_stop = force_rmax

    isfinite(rmin_stop) && isfinite(rmax_stop) && 0.0 < rmin_stop < rmax_stop || error("invalid orbit integration radius domain: [$rmin_stop, $rmax_stop]")

    r0 = f64(ic[1])
    theta0 = f64(ic[2])
    base_dt = f64(ic[3])
    dt = base_dt * dt_scale
    vr0 = length(ic) >= 4 ? f64(ic[4]) : 0.0
    vtheta0 = length(ic) >= 5 ? f64(ic[5]) : 0.0

    state = if continuation_state === nothing
        st0, ct0 = sincos(theta0)
        R0 = r0 * st0
        z0 = r0 * ct0
        vR0 = vr0 * st0 + vtheta0 * ct0
        vz0 = vr0 * ct0 - vtheta0 * st0
        SVector(R0, z0, vR0, vz0)
    else
        length(continuation_state) == 4 || error("continuation_state must contain (R, z, vR, vz)")
        SVector(f64(continuation_state[1]), f64(continuation_state[2]), f64(continuation_state[3]), f64(continuation_state[4]))
    end

    r = Vector{Float64}(undef, ns)
    vr = Vector{Float64}(undef, ns)
    theta = Vector{Float64}(undef, ns)
    vtheta = Vector{Float64}(undef, ns)

    actual = 0
    termination_reason = :completed

    initial_energy = NaN
    final_energy = NaN
    max_absolute_energy_drift = NaN
    max_relative_energy_drift = NaN

    total_substeps = 0
    max_substeps_used = 0
    minimum_substep = Inf
    maximum_substep = 0.0
    completed_duration = 0.0

    function state_exit_reason(s)
        all(isfinite, s) || return :nonfinite_state
        Rcyl, z = s[1], s[2]
        Rcyl < 0.0 && return :crossed_cylindrical_axis

        rr = hypot(Rcyl, z)
        isfinite(rr) || return :nonfinite_radius
        rr < rmin_stop && return :hit_rmin
        rr > rmax_stop && return :hit_rmax

        if xLz != 0.0 && Rcyl <= 0.0
            return :hit_cylindrical_axis
        end

        return :ok
    end

    function spherical_state(s)
        Rcyl, z, vR, vz = s
        rr = hypot(Rcyl, z)

        isfinite(rr) && rr > 0.0 || return (NaN, NaN, NaN, NaN)

        st = Rcyl / rr
        ct = z / rr
        tr = atan(Rcyl, z)
        vrr = vR * st + vz * ct
        vtt = vR * ct - vz * st

        return rr, tr, vrr, vtt
    end

    function orbit_energy(s)
        Rcyl, z, vR, vz = s

        all(isfinite, s) || return NaN
        Rcyl >= 0.0 || return NaN

        rr = hypot(Rcyl, z)
        rmin_stop <= rr <= rmax_stop || return NaN

        tr = atan(Rcyl, z)
        potential = f64(pot(rr, tr))
        isfinite(potential) || return NaN

        if xLz != 0.0 && Rcyl <= 0.0
            return NaN
        end

        vphi = xLz == 0.0 ? 0.0 : f64(xLz) / Rcyl

        return potential + 0.5 * (vR^2 + vz^2 + vphi^2)
    end

    function local_step_limit(s, k1)
        Rcyl, z, vR, vz = s
        rr = hypot(Rcyl, z)

        isfinite(rr) && rr > 0.0 || return NaN

        accel = hypot(k1[3], k1[4])
        omega_dyn = isfinite(accel) && accel > 0.0 ? sqrt(accel / max(rr, rmin_stop)) : 0.0
        omega_rad = hypot(vR, vz) / max(rr, rmin_stop)
        omega_R = Rcyl > 0.0 ? abs(vR) / Rcyl : 0.0
        omega_phi = xLz != 0.0 && Rcyl > 0.0 ? abs(f64(xLz)) / (Rcyl^2) : 0.0
        omega_local = max(omega_dyn, omega_rad, omega_R, omega_phi)

        if !isfinite(omega_local)
            return NaN
        elseif omega_local <= 0.0
            return abs(dt)
        end

        return local_step_safety / omega_local
    end

    function rk4_substep(s, h, k1)
        state2 = s + 0.5 * h * k1
        reason2 = state_exit_reason(state2)
        reason2 !== :ok && return s, reason2

        k2 = derivs(state2, xLz, orbit_ctx.frc, orbit_ctx.R_pos)
        !all(isfinite, k2) && return s, :invalid_k2

        state3 = s + 0.5 * h * k2
        reason3 = state_exit_reason(state3)
        reason3 !== :ok && return s, reason3

        k3 = derivs(state3, xLz, orbit_ctx.frc, orbit_ctx.R_pos)
        !all(isfinite, k3) && return s, :invalid_k3

        state4 = s + h * k3
        reason4 = state_exit_reason(state4)
        reason4 !== :ok && return s, reason4

        k4 = derivs(state4, xLz, orbit_ctx.frc, orbit_ctx.R_pos)
        !all(isfinite, k4) && return s, :invalid_k4

        next_state = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        all(isfinite, next_state) || return s, :nonfinite_updated_state

        next_reason = state_exit_reason(next_state)
        next_reason !== :ok && return s, next_reason

        return next_state, :ok
    end

    if !(isfinite(dt) && dt != 0.0)
        termination_reason = :invalid_initial_timestep
    end

    if termination_reason === :completed
        initial_state_reason = state_exit_reason(state)
        initial_state_reason !== :ok && (termination_reason = initial_state_reason)
    end

    if return_diag && termination_reason === :completed
        current_start_energy = orbit_energy(state)

        if !isfinite(current_start_energy)
            termination_reason = :nonfinite_initial_energy
        else
            initial_energy = reference_energy === nothing ? current_start_energy : f64(reference_energy)

            if !isfinite(initial_energy)
                termination_reason = :nonfinite_reference_energy
            else
                final_energy = current_start_energy
                max_absolute_energy_drift = abs(current_start_energy - initial_energy)
                max_relative_energy_drift = max_absolute_energy_drift / max(abs(initial_energy), 1.0)

                if max_relative_energy_drift > energy_drift_limit
                    termination_reason = :energy_drift_exceeded
                end
            end
        end
    end

    if termination_reason === :completed
        @inbounds for step in 1:ns
            current_reason = state_exit_reason(state)

            if current_reason !== :ok
                termination_reason = current_reason
                break
            end

            rr, tr, vrr, vtt = spherical_state(state)

            if !(isfinite(rr) && isfinite(tr) && isfinite(vrr) && isfinite(vtt))
                termination_reason = :nonfinite_spherical_conversion
                break
            end

            actual += 1
            r[actual] = rr
            theta[actual] = tr
            vr[actual] = vrr
            vtheta[actual] = vtt

            if return_diag && (step == 1 || step % energy_check_every == 0)
                current_energy = orbit_energy(state)

                if !isfinite(current_energy)
                    termination_reason = :nonfinite_energy
                    break
                end

                absolute_drift = abs(current_energy - initial_energy)
                relative_drift = absolute_drift / max(abs(initial_energy), 1.0)

                final_energy = current_energy
                max_absolute_energy_drift = max(max_absolute_energy_drift, absolute_drift)
                max_relative_energy_drift = max(max_relative_energy_drift, relative_drift)

                if relative_drift > energy_drift_limit
                    termination_reason = :energy_drift_exceeded
                    break
                end
            end

            remaining = abs(dt)
            dt_sign = sign(dt)
            substeps_this_step = 0

            while remaining > 0.0
                substeps_this_step += 1

                if substeps_this_step > max_substeps_per_step
                    termination_reason = :adaptive_substep_limit
                    break
                end

                k1 = derivs(state, xLz, orbit_ctx.frc, orbit_ctx.R_pos)

                if !all(isfinite, k1)
                    termination_reason = :invalid_k1
                    break
                end

                h_limit = local_step_limit(state, k1)

                if !(isfinite(h_limit) && h_limit > 0.0)
                    termination_reason = :invalid_local_timestep
                    break
                end

                h_abs = min(remaining, h_limit)

                if !(isfinite(h_abs) && h_abs > 0.0)
                    termination_reason = :adaptive_timestep_underflow
                    break
                end

                h = dt_sign * h_abs
                next_state, substep_reason = rk4_substep(state, h, k1)

                if substep_reason === :crossed_cylindrical_axis || substep_reason === :hit_cylindrical_axis
                    h_abs *= 0.5

                    if h_abs <= abs(dt) / max_substeps_per_step
                        termination_reason = :adaptive_axis_resolution_failed
                        break
                    end

                    h = dt_sign * h_abs
                    next_state, substep_reason = rk4_substep(state, h, k1)
                end

                if substep_reason !== :ok
                    termination_reason = substep_reason
                    break
                end

                state = next_state
                remaining = max(0.0, remaining - h_abs)
                completed_duration += h_abs
                total_substeps += 1
                minimum_substep = min(minimum_substep, h_abs)
                maximum_substep = max(maximum_substep, h_abs)
            end

            max_substeps_used = max(max_substeps_used, substeps_this_step)

            termination_reason !== :completed && break
        end
    end

    resize!(r, actual)
    resize!(vr, actual)
    resize!(theta, actual)
    resize!(vtheta, actual)

    if termination_reason === :completed
        final_state_reason = state_exit_reason(state)
        final_state_reason !== :ok && (termination_reason = final_state_reason)
    end

    if return_diag
        if termination_reason === :completed && state_exit_reason(state) === :ok
            checked_final_energy = orbit_energy(state)

            if isfinite(checked_final_energy)
                absolute_drift = abs(checked_final_energy - initial_energy)
                relative_drift = absolute_drift / max(abs(initial_energy), 1.0)

                final_energy = checked_final_energy
                max_absolute_energy_drift = max(max_absolute_energy_drift, absolute_drift)
                max_relative_energy_drift = max(max_relative_energy_drift, relative_drift)

                if relative_drift > energy_drift_limit
                    termination_reason = :energy_drift_exceeded
                end
            else
                termination_reason = :nonfinite_final_energy
            end
        end

        final_R = all(isfinite, state) ? f64(state[1]) : NaN
        final_z = all(isfinite, state) ? f64(state[2]) : NaN
        final_vR = all(isfinite, state) ? f64(state[3]) : NaN
        final_vz = all(isfinite, state) ? f64(state[4]) : NaN
        final_r = isfinite(final_R) && isfinite(final_z) ? hypot(final_R, final_z) : NaN
        final_theta = isfinite(final_r) && final_r > 0.0 && isfinite(final_R) && final_R >= 0.0 ? atan(final_R, final_z) : NaN

        energy_valid = isfinite(max_relative_energy_drift) && max_relative_energy_drift <= energy_drift_limit

        diag = (
            termination_reason=termination_reason,
            requested_steps=ns,
            completed_steps=actual,
            base_dt=base_dt,
            dt=dt,
            dt_scale=dt_scale,
            requested_duration=isfinite(dt) ? ns * abs(dt) : NaN,
            completed_duration=completed_duration,
            force_rmin=force_rmin,
            force_rmax=force_rmax,
            rmin_stop=rmin_stop,
            rmax_stop=rmax_stop,
            minimum_r=isempty(r) ? NaN : minimum(r),
            maximum_r=isempty(r) ? NaN : maximum(r),
            final_r=final_r,
            final_theta=final_theta,
            final_R=final_R,
            final_z=final_z,
            final_vR=final_vR,
            final_vz=final_vz,
            initial_energy=initial_energy,
            final_energy=final_energy,
            max_absolute_energy_drift=max_absolute_energy_drift,
            max_relative_energy_drift=max_relative_energy_drift,
            max_relative_energy_drift_allowed=max_relative_energy_drift_allowed,
            energy_drift_boundary_allowance=energy_drift_boundary_allowance,
            energy_drift_limit=energy_drift_limit,
            energy_valid=energy_valid,
            energy_check_every=energy_check_every,
            local_step_safety=local_step_safety,
            total_substeps=total_substeps,
            max_substeps_used=max_substeps_used,
            minimum_substep=isfinite(minimum_substep) ? minimum_substep : NaN,
            maximum_substep=maximum_substep,
        )

        return r, vr, theta, vtheta, diag
    end

    return r, vr, theta, vtheta
end
