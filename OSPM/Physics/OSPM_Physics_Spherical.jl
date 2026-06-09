# HOT PATH — Karl-style OSPM A-matrix builder & batch evaluator.

# Applied new karl fixes on 04/06/26 @1600

module OSPMPhysicsSpherical
@info "OSPMPhysicsSpherical Karl-style loaded from" @__FILE__

using LinearAlgebra, StaticArrays, Statistics, Random, Base.Threads, Optim

export build_R_halo_physical, rho_interp, halo_from_theta, tables_spherical,
    make_potential_force_funcs, integrate_orbit_rk4, build_A_matrix_hybrid,
    mass_enclosed_two_radii, evaluate_batch_theta, NTHREADS, force_at_rtheta

include("OSPM_Physics_Support.jl")
@info "OSPMPhysicsSpherical supports spherical frc(r,theta)->(fr,0) and axisymmetric frc(r,theta)->(fr,ftheta)"

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
    fill_target::Int
    orbit_budget::Int
    return_light::Bool
    theta_launches::Vector{Float64}
    sini::Float64
    cosi::Float64
    R_star_m::Vector{Float64}
    valid_vlos::Vector{Bool}
    v_star_mps::Vector{Float64}
    verr_star_mps::Vector{Float64}
    spatial_edges::Vector{Float64}
    velocity_edges::Vector{Float64}
    shells::Vector{Float64}
    cost_order::Vector{Int}
    orbit_ctx
    pot
    frc
    Lfrac
    force_geometry::Symbol
    dt_frac_orbit::Float64
    t_deadline::UInt64
    A_losvd::Matrix{Float64}
    A_light::Matrix{Float64}
    success_flags::BitVector
    min_r_reached::Vector{Float64}
    rapo_list::Vector{Float64}
    next_orbit::Threads.Atomic{Int}
    filled_atomic::Threads.Atomic{Int}
    phase::Threads.Atomic{Int}
end

function _init_orbit_work( Norbit::Int, R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64},
    sini::Float64, ctx; nsteps::Int, Lfrac, dt_frac_orbit::Float64, Nbins_occ::Int, return_occ::Bool, max_attempts_factor::Int, fill_pct::Float64, t_deadline::UInt64,
    velocity_edges=nothing, kinematic_bin_edges=nothing, min_stars_per_bin::Int=20, Nvbin::Int=21, Ntheta_launch::Int=9)

    iseven(Norbit) || error("Karl prograde/retrograde orbit pairing requires even Norbit because Norbit is the final A-matrix column count")
    Nbase_orbit = Norbit ÷ 2

    Nstar = length(R_star_m)
    valid_vec = collect(Bool, valid_vlos)
    vlos_idx = Int[]

    @inbounds for i in 1:Nstar
        valid_vec[i] && push!(vlos_idx, i)
    end

    spatial_edges = resolve_karl_spatial_edges(kinematic_bin_edges)
    Nspatial = length(spatial_edges) - 1

    velocity_edges_use =
        velocity_edges === nothing ?
        build_velocity_edges_auto(v_star_mps[vlos_idx], verr_star_mps[vlos_idx]; Nvbin=Nvbin) :
        Float64.(velocity_edges)

    Nvbin_eff = length(velocity_edges_use) - 1
    Nlosvd = Nspatial * Nvbin_eff
    Nlight = Nspatial

    shells = sort(copy(R_star_m[isfinite.(R_star_m)]))
    isempty(shells) && (shells = [spatial_edges[1], spatial_edges[end]])

    Nshells = length(shells)
    A_losvd = zeros(Float64, Nlosvd, Norbit)
    A_light = zeros(Float64, Nlight, Norbit)
    success_flags = falses(Nbase_orbit)
    min_r_reached = fill(Inf, Nbase_orbit)
    rapo_list = fill(NaN, Nbase_orbit)
    _orbit_cost = Vector{Float64}(undef, Nbase_orbit)

    @inbounds for c in 1:Nbase_orbit
        lf = Lfrac[1 + ((c - 1) % length(Lfrac))]
        rapo = shells[mod1(c, Nshells)]
        _orbit_cost[c] = lf * rapo
    end

    cost_order = sortperm(_orbit_cost)
    fill_target = max(1, round(Int, clamp(fill_pct, 0.0, 1.0) * Nbase_orbit))
    orbit_budget = max_attempts_factor * Nbase_orbit
    force_geometry = haskey(ctx.halo, :stellar_model) ? stellar_model_geometry(ctx.halo[:stellar_model]) : :spherical_shell_grid
    theta_launches = force_geometry === :axisymmetric_density_grid ?
        collect(range(0.15 * pi, 0.85 * pi; length=max(3, Ntheta_launch))) :
        [f64(pi / 2)]
    sini_use = clamp01(f64(sini))
    cosi_use = sqrt(max(0.0, 1.0 - sini_use * sini_use))

    orbit_ctx = ( frc=ctx.frc, R_pos=ctx.R, halo=ctx.halo, force_geometry=force_geometry)

    return OrbitWorkState( Norbit, Nbase_orbit, Nstar, Nspatial, Nvbin_eff, Nlosvd, Nlight, Nshells, nsteps, max_attempts_factor, fill_target, orbit_budget, return_occ, theta_launches,
        sini_use, cosi_use, R_star_m, valid_vec, v_star_mps, verr_star_mps, spatial_edges, velocity_edges_use, shells, cost_order, orbit_ctx, ctx.pot, ctx.frc, Lfrac,
        force_geometry, dt_frac_orbit, t_deadline, A_losvd, A_light, success_flags, min_r_reached, rapo_list, Threads.Atomic{Int}(1), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))
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

function _orbit_worker!(st::OrbitWorkState, rng)
    col_losvd_pro = zeros(Float64, st.Nlosvd)
    col_losvd_ret = zeros(Float64, st.Nlosvd)
    col_light = zeros(Float64, st.Nlight)

    s_arr = Vector{Float64}(undef, st.nsteps)
    vlos_pro_buf = Vector{Float64}(undef, st.nsteps)
    vlos_ret_buf = Vector{Float64}(undef, st.nsteps)

    while true
        time_ns() > st.t_deadline && break
        st.filled_atomic[] >= st.fill_target && break
        st.phase[] != 1 && break

        c_seq = Threads.atomic_add!(st.next_orbit, 1)
        c_seq > st.orbit_budget && break

        attempt = (c_seq - 1) ÷ st.Nbase_orbit
        c_claim = st.cost_order[mod1(c_seq, st.Nbase_orbit)]

        attempt > 0 && st.success_flags[c_claim] && continue

        idx_local = mod1(c_claim, st.Nshells)
        rapo = f64(st.shells[idx_local])
        st.rapo_list[c_claim] = rapo

        !(isfinite(rapo) && rapo > 0.0) && continue

        lf = st.Lfrac[1 + ((c_claim - 1) % length(st.Lfrac))]

        r0_frac = 0.95 - 0.05 * f64(attempt) / st.max_attempts_factor + 0.04 * rand(rng)

        theta0 = st.theta_launches[mod1(c_claim, length(st.theta_launches))]
        ic, Lz0, E0, vc, launch_state = launch_orbit_apocenter(rapo=rapo, theta0=theta0, Lz_frac=f64(lf), pot=st.pot, frc=st.frc, r0_frac=r0_frac, dt_frac=st.dt_frac_orbit)

        launch_state != :ok && continue

        r, vr, theta, vtheta = integrate_orbit_rk4( ic=ic, xLz=Lz0, orbit_ctx=st.orbit_ctx, nsteps=st.nsteps)

        isempty(r) && continue

        st.success_flags[c_claim] = true
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
            ib = _bin_index(st.spatial_edges, s_arr[k])
            ib == 0 && continue

            col_light[ib] += 1.0

            jb_pro = _bin_index(st.velocity_edges, vlos_pro_buf[k])
            if jb_pro > 0
                row_pro = (ib - 1) * st.Nvbin + jb_pro
                col_losvd_pro[row_pro] += 1.0
            end

            jb_ret = _bin_index(st.velocity_edges, vlos_ret_buf[k])
            if jb_ret > 0
                row_ret = (ib - 1) * st.Nvbin + jb_ret
                col_losvd_ret[row_ret] += 1.0
            end
        end

        if Nhits > 0
            col_light ./= Nhits
            col_losvd_pro ./= Nhits
            col_losvd_ret ./= Nhits
        end

        col_pro = 2 * c_claim - 1
        col_ret = 2 * c_claim

        @inbounds st.A_losvd[:, col_pro] .= col_losvd_pro
        @inbounds st.A_losvd[:, col_ret] .= col_losvd_ret
        @inbounds st.A_light[:, col_pro] .= col_light
        @inbounds st.A_light[:, col_ret] .= col_light

        Threads.atomic_add!(st.filled_atomic, 1)
    end
end

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Main A-matrix builder: maps orbital weights → Karl observables.
function build_A_matrix_hybrid(Norbit::Int, R_star_m::Vector{Float64}, has_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, sini::Float64,
        rho_s::Float64, r_s::Float64, MBH::Float64, ML::Float64, halo_type::String; stellar_model=nothing, surface_brightness_profile=nothing,
        nsteps::Int=DEFAULT_NSTEPS, Lfrac::NTuple{5,Float64}=DEFAULT_LFRAC, dt_frac_orbit::Float64=DEFAULT_DT_FRAC,
        dR_frac::Float64=DEFAULT_DR_FRAC, Nbins_occ::Int=DEFAULT_NBINS_OCC, return_occ::Bool=true, max_attempts_factor::Int=DEFAULT_MAX_ATTEMPTS,
        diag::Bool=false, threaded::Bool=true, fill_pct::Float64=0.80, t_deadline::UInt64=typemax(UInt64), velocity_edges=nothing,
        kinematic_bin_edges=nothing, min_stars_per_bin::Int=20, Nvbin::Int=21, Ntheta_launch::Int=9, halo_q_axis_ratio::Float64=1.0,
        karl_halo_params=nothing)

    Nstar = length(R_star_m)
    @assert length(has_vlos) == Nstar
    @assert length(v_star_mps) == Nstar
    @assert length(verr_star_mps) == Nstar
    surface_brightness_profile === nothing && error("surface_brightness_profile is required for Karl-style OSPM; no star-count fallback is allowed")
    Nstar == 0 && return zeros(Float64, 0, Norbit)

    stellar_model_jl = normalize_stellar_model(stellar_model)
    surface_brightness_profile_jl = normalize_surface_brightness_profile(surface_brightness_profile)
    ctx = get_halo_context(
        rho_s,
        r_s,
        MBH,
        ML,
        halo_type;
        stellar_model=stellar_model_jl,
        halo_q_axis_ratio=halo_q_axis_ratio,
        karl_halo_params=karl_halo_params,
    )
    sini = clamp01(f64(sini))
    Rmin = minimum(R_star_m)
    Rmax = maximum(R_star_m)

    if !(isfinite(Rmin) && isfinite(Rmax) && Rmax > Rmin)
        return zeros(Float64, 0, Norbit)
    end

    st = _init_orbit_work(Norbit, R_star_m, has_vlos, v_star_mps, verr_star_mps, sini, ctx;
        nsteps=nsteps, Lfrac=Lfrac, dt_frac_orbit=dt_frac_orbit, Nbins_occ=Nbins_occ,
        return_occ=return_occ, max_attempts_factor=max_attempts_factor, fill_pct=fill_pct,
        t_deadline=t_deadline, velocity_edges=velocity_edges, kinematic_bin_edges=kinematic_bin_edges, min_stars_per_bin=min_stars_per_bin, Nvbin=Nvbin, Ntheta_launch=Ntheta_launch)

    Threads.atomic_xchg!(st.phase, 1)
    nworkers = threaded ? Threads.nthreads() : 1
    rngs = [MersenneTwister(0x5eed1234 + UInt(t)) for t in 1:nworkers]

    if threaded && nworkers > 1
        Threads.@threads for t in 1:nworkers
            _orbit_worker!(st, rngs[t])
        end
    else
        _orbit_worker!(st, rngs[1])
    end

    filled = st.filled_atomic[]
    if filled < st.fill_target
        println("WARNING: build_A_matrix_hybrid filled ", filled, " / ", st.fill_target,
            " | missing ", round(100 * (st.fill_target - filled) / st.fill_target, digits=1), "%")
    end

    A = return_occ ? vcat(st.A_losvd, st.A_light) : st.A_losvd

    if diag
        losvd_target, losvd_sigma, light_target, light_sigma, counts_by_spatial = observed_targets_karl( R_star_m, has_vlos, v_star_mps,
            verr_star_mps, st.spatial_edges, st.velocity_edges; surface_brightness_profile=surface_brightness_profile_jl)

        return (
            A,
            Dict(
                "filled" => filled,
                "Nbase_orbit" => st.Nbase_orbit,
                "paired_orbit_columns" => true,
                "attempts" => Norbit,
                "Nspatial" => st.Nspatial,
                "Nvbin" => st.Nvbin,
                "Nlosvd" => st.Nlosvd,
                "Nlight" => st.Nlight,
                "spatial_edges" => st.spatial_edges,
                "velocity_edges" => st.velocity_edges,
                "losvd_target" => losvd_target,
                "light_target" => light_target,
                "counts_by_spatial" => counts_by_spatial,
                "force_geometry" => String(st.force_geometry),
            ),
        )
    end

    return A
end

# Batch evaluator: Karl-style binned LOSVD + projected-light fit.
function evaluate_batch_theta(thetas::AbstractMatrix{<:Real}, R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64},
    verr_star_mps::Vector{Float64}, sini::Float64, Norbit::Int, halo_type::String; stellar_model=nothing, surface_brightness_profile=nothing,
    Nocc::Int=0, lambda_occ::Float64=1.0, alpha::Float64=DEFAULT_KARL_ALPHA, alphat::Float64=DEFAULT_KARL_ALPHAT,
    weight_mode=:entropy, weight_solver_mode=:orbit_only, wphase=nothing, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, losvd_score_mode=:karl_fracnew,
    maxiter::Int=DEFAULT_KARL_MAXITER, max_refine::Int=0, timeout_s::Float64=120.0,
    R_inner_pc::Float64=30.0, use_radial_vlos_weights::Bool=false, use_weighted_score::Bool=false, R_weight_pc::Float64=-1.0,
    radial_weight_gamma::Float64=2.0, radial_weight_floor::Float64=0.3, velocity_edges=nothing, kinematic_bin_edges=nothing,
    min_stars_per_bin::Int=20, Nvbin::Int=21, Ntheta_launch::Int=9, halo_q_axis_ratio::Float64=1.0,
    karl_halo_params=nothing)

    nrow, nbatch = size(thetas)
    surface_brightness_profile === nothing && error("surface_brightness_profile is required for Karl-style OSPM; no star-count fallback is allowed")
    stellar_model_jl = normalize_stellar_model(stellar_model)
    surface_brightness_profile_jl = normalize_surface_brightness_profile(surface_brightness_profile)

    losvd_score_sym = Symbol(lowercase(String(losvd_score_mode)))
    weight_mode_sym = Symbol(lowercase(String(weight_mode)))
    weight_solver_sym = Symbol(lowercase(String(weight_solver_mode)))
    if !(losvd_score_sym === :standard || losvd_score_sym === :karl_fracnew)
        error("Unknown losvd_score_mode=$(losvd_score_sym). Use :standard or :karl_fracnew")
    end
    if !(weight_solver_sym === :orbit_only || weight_solver_sym === :expanded_cm)
        error("Unknown weight_solver_mode=$(weight_solver_sym). Use :orbit_only or :expanded_cm")
    end

    status = fill(4, nbatch)
    refine_used = zeros(Int, nbatch)
    chi2_losvd = fill(Inf, nbatch)
    chi2_total = fill(Inf, nbatch)
    chi2_inner = fill(Inf, nbatch)
    chi2_outer = fill(Inf, nbatch)
    chi2_light = fill(Inf, nbatch)
    N_inner = zeros(Int, nbatch)
    N_outer = zeros(Int, nbatch)
    N_nonzero_weights = zeros(Int, nbatch)
    effective_N_orbits = zeros(Float64, nbatch)
    max_weight_fraction = zeros(Float64, nbatch)

    work_states = Vector{Union{Nothing, OrbitWorkState}}(undef, nbatch)
    fill!(work_states, nothing)
    next_theta = Threads.Atomic{Int}(1)
    nthreads = Threads.nthreads()
    helper_rngs = [MersenneTwister(0x0E100000 + UInt(t) + UInt(nbatch)) for t in 1:nthreads]

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
                println(
                    "[WEIGHT DEBUG] ",
                    "i=", i,
                    " solver=", weight_solver_sym,
                    " wsum=", wsum,
                    " wmin=", wmin,
                    " wmax=", wmax,
                    " nneg=", nneg,
                    " nbad=", nbad,
                    " pmin=", pmin,
                    " pmax=", pmax,
                    " N_nonzero=", N_nonzero_weights[i],
                    " Neff=", effective_N_orbits[i],
                )
            end
        else
            println(
                "[WEIGHT DEBUG] ",
                "i=", i,
                " solver=", weight_solver_sym,
                " BAD SUM",
                " wsum=", wsum,
                " wmin=", wmin,
                " wmax=", wmax,
                " nneg=", nneg,
                " nbad=", nbad,
            )
        end

        return nothing
    end

    function _batch_worker!(tid::Int)
        rng_own = MersenneTwister(0x5eed1234 + UInt(tid))
        rng_help = helper_rngs[tid]

        while true
            i = Threads.atomic_add!(next_theta, 1)
            if i <= nbatch
                theta_deadline = time_ns() + UInt64(round(timeout_s * 1e9))
                try
                    rho_s = Float64(thetas[1, i])
                    r_s = Float64(thetas[2, i])
                    MBH = nrow >= 3 ? Float64(thetas[3, i]) : 0.0
                    ML = nrow >= 4 ? Float64(thetas[4, i]) : 1.0

                    Rmin_v = minimum(R_star_m)
                    Rmax_v = maximum(R_star_m)
                    if !(isfinite(Rmin_v) && isfinite(Rmax_v) && Rmax_v > Rmin_v) || length(R_star_m) == 0
                        status[i] = 1
                        continue
                    end

                    ctx = get_halo_context( rho_s, r_s, MBH, ML, halo_type; stellar_model=stellar_model_jl, halo_q_axis_ratio=halo_q_axis_ratio, karl_halo_params=karl_halo_params )
                    force_geometry = haskey(ctx.halo, :stellar_model) ? stellar_model_geometry(ctx.halo[:stellar_model]) : :spherical_shell_grid
                    ws = _init_orbit_work(Norbit, R_star_m, valid_vlos, v_star_mps, verr_star_mps, sini, ctx;
                        nsteps=DEFAULT_NSTEPS, Lfrac=DEFAULT_LFRAC, dt_frac_orbit=DEFAULT_DT_FRAC,
                        Nbins_occ=Nocc, return_occ=true, max_attempts_factor=DEFAULT_MAX_ATTEMPTS,
                        fill_pct=0.80, t_deadline=theta_deadline, velocity_edges=velocity_edges,
                        kinematic_bin_edges=kinematic_bin_edges, min_stars_per_bin=min_stars_per_bin, Nvbin=Nvbin, Ntheta_launch=Ntheta_launch)
                    work_states[i] = ws

                    Threads.atomic_xchg!(ws.phase, 1)
                    _orbit_worker!(ws, rng_own)
                    Threads.atomic_xchg!(ws.phase, 2)

                    A_losvd = ws.A_losvd
                    A_light = ws.A_light

                    if size(A_losvd, 1) == 0 || size(A_losvd, 2) == 0 || !all(isfinite, A_losvd) ||
                       size(A_light, 1) == 0 || size(A_light, 2) == 0 || !all(isfinite, A_light)
                        status[i] = 1
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end

                    losvd_target, losvd_sigma, light_target, light_sigma, counts_by_spatial = observed_targets_karl(
                        R_star_m, valid_vlos, v_star_mps, verr_star_mps, ws.spatial_edges, ws.velocity_edges;
                        surface_brightness_profile=surface_brightness_profile_jl)

                    if weight_solver_sym === :expanded_cm
                        w, ok, wdiag = solve_weights_karl_expanded_cm( A_light, A_losvd, light_target, light_sigma, losvd_target, losvd_sigma;
                            alphat=alphat, lambda_light=lambda_occ, wphase=wphase, maxiter=maxiter, seed=UInt(i), entropy_floor=entropy_floor, apfac=DEFAULT_KARL_APFAC, return_diag=true )
                    else
                        A = vcat(A_losvd, A_light)
                        d = vcat(losvd_target, light_target)
                        light_sigma_eff = light_sigma ./ sqrt(max(lambda_occ, 1e-12))
                        sigma = vcat(losvd_sigma, light_sigma_eff)

                        w, ok, wdiag = solve_weights_karl_jl(
                            A,
                            d,
                            sigma;
                            alpha=alpha,
                            alphat=alphat,
                            weight_mode=weight_mode_sym,
                            wphase=wphase,
                            maxiter=maxiter,
                            seed=UInt(i),
                            entropy_floor=entropy_floor,
                            return_diag=true,
                        )
                    end
                    if !ok
                        status[i] = 2
                        Threads.atomic_xchg!(ws.phase, 3)
                        continue
                    end

                    cl = losvd_score_sym === :karl_fracnew ?
                        chi2_block_karl_fracnew(A_losvd, w, losvd_target, losvd_sigma, ws.Nspatial, ws.Nvbin) :
                        chi2_block(A_losvd, w, losvd_target, losvd_sigma)

                    cb = chi2_block(A_light, w, light_target, light_sigma)

                    # Karl-style convention:
                    # Store raw χ² sums, not reduced χ².
                    chi2_losvd[i] = cl
                    chi2_light[i] = cb
                    chi2_total[i] = chi2_losvd[i] + lambda_occ * chi2_light[i]
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
                    !isempty(inner_rows) && (
                        chi2_inner[i] = chi2_block(
                            A_losvd[inner_rows, :],
                            w,
                            losvd_target[inner_rows],
                            losvd_sigma[inner_rows],
                        )
                    )

                    !isempty(outer_rows) && (
                        chi2_outer[i] = chi2_block(
                            A_losvd[outer_rows, :],
                            w,
                            losvd_target[outer_rows],
                            losvd_sigma[outer_rows],
                        )
                    )

                    _store_weight_diagnostics!(i, w)
                    status[i] = 0
                    Threads.atomic_xchg!(ws.phase, 3)
                catch e
                    status[i] = 3
                    ws_i = work_states[i]
                    ws_i !== nothing && Threads.atomic_xchg!(ws_i.phase, 3)
                    @warn "evaluate_batch_theta Karl exception on i=$i" exception=(e, catch_backtrace()) halo_type=halo_type
                end
                continue
            end

            helped = false
            for scan in 1:nbatch
                ws_scan = work_states[scan]
                ws_scan === nothing && continue
                ws_scan.phase[] != 1 && continue
                time_ns() > ws_scan.t_deadline && continue
                _orbit_worker!(ws_scan, rng_help)
                helped = true
                break
            end
            helped || break
        end
    end

    Threads.@threads :static for t in 1:nthreads
        _batch_worker!(t)
    end

    # Return shape is kept close to the previous hot path so the Python side can
    # be updated gradually.  Here chi2 is Karl-style raw χ²_LOSVD and the old chi2_occ slot is raw χ²_light.
    return (status, chi2_losvd, refine_used, chi2_inner, chi2_outer, chi2_light,
            N_inner, N_outer, N_nonzero_weights, effective_N_orbits, max_weight_fraction)
end
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

end # module

# WHAT THIS VERSION DOES
# Orbit-superposition physics engine for Karl-style spherical or axisymmetric force modelling.
# Given halo parameters θ = (ρ_s, r_s, M_BH, M/L), this file builds a gravitational
# potential, launches a library of stellar orbits, then projects each orbit into
# Karl-style observables:
#
#   (1) projected-light / surface-brightness radial bins
#   (2) binned LOSVD rows: spatial bin × line-of-sight velocity bin
#
# This is different from the previous star-level likelihood version.  The old
# hot path made one velocity row per observed star.  This version makes one
# velocity-distribution row per radial aperture and velocity bin.
#
# The main reported χ² is raw χ²_LOSVD from the hard orbit-weight LOSVD
# residual A_losvd*w - target.  Light rows remain an extra diagnostic and can
# optionally help constrain weights through lambda_light.
#
# Solver modes:
#   orbit_only  = production/debug steering mode; no LOSVD slack can hide residuals
#   expanded_cm = Karl-style experimental mode with LOSVD slack variables
#
# REQUIRED PIPELINE RULE
# A real observed surface_brightness_profile is required.  There is no star-count
# fallback in the Karl-style copy.  If the profile is missing, malformed, or does
# not match the spatial bins, the run must fail before scoring.
#
# KARL ORBIT-COLUMN RULE
# Norbit is the final number of A-matrix columns.  The live path integrates
# Norbit/2 base orbits and writes two columns per base orbit:
#   2i-1 -> prograde / +vphi
#   2i   -> retrograde / -vphi
# This is not optional in this Karl-style copy.
#
# KARL HALO PARAMETER RULE
# evaluate_batch_theta and build_A_matrix_hybrid accept karl_halo_params and pass
# them into get_halo_context.  This lets old-Karl halo dictionaries reach the
# Force layer instead of being silently ignored.