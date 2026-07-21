# ============================================================
# OSPM_Physics_Support.jl — Karl-style support layer.
# Included by OSPM_Physics_Spherical.jl — do NOT load directly.
#
# Contains only the shared support shell and Karl-style observable machinery:
# constants, halo context construction, radial/velocity binning,
# surface-brightness targets, binned LOSVD targets, and includes for the
# weight/SPEAR and force machinery.

# Applied new karl fixes on 04/06/26 @1600

# Legacy star-level likelihood code and old back-compat sigma2 paths removed.
# ============================================================

# §1  CONSTANTS
const NTHREADS = Threads.nthreads()
const G    = 6.67430e-11
const c    = 2.99792458e8
const pc   = 3.0856775814913673e16
const Msun = 1.98847e30
# machine floors
const EPS_FORCE = 1e-14
const EPS_VEL   = 1e-14
const EPS_ARG   = 1e-14
# physical geometry gate
const EPS_SIN = 1e-6
# scale-aware force gate
const REL_FORCE    = 1e-10   # loosen to 1e-9 if needed
const BRACKET_FRAC = 1e-6    # MUST be >> eps(Float64)

# TUNABLE KNOBS — adjust these to control resolution, accuracy, and parallelism.
# -- Halo potential grid --
const DEFAULT_NR              = 256       # radial grid points for potential table
const DEFAULT_RMAX_FACTOR     = 300.0     # max radius in units of r_s
# -- Orbit integration --
const DEFAULT_NSTEPS          = 4000      # RK4 steps per orbit
const DEFAULT_STOP_RMIN_FACTOR = 1.001    # orbit stops when r < factor * rmin
const DEFAULT_DT_FRAC         = 0.01      # timestep = dt_frac / orbital_frequency
const DEFAULT_DT_FLOOR        = 1e-30     # floor on orbital-frequency denominator
const DEFAULT_R0_FRAC         = 0.98      # starting radius as fraction of apocenter
# -- A-matrix / orbit library --
const DEFAULT_LFRAC           = (0.05, 0.2, 0.4, 0.7, 1.0)  # angular momentum fractions
const DEFAULT_DR_FRAC         = 0.05      # radial matching tolerance (fraction of R)
const DEFAULT_NBINS_OCC       = 6         # occupancy histogram bins
const DEFAULT_MAX_ATTEMPTS    = 60        # max orbit-launch attempts multiplier
const DEFAULT_DR_FLOOR_FRAC   = 0.01      # floor on dR (fraction)
const DEFAULT_DR_FLOOR_PC     = 0.0       # floor on dR (parsecs)
# -- Karl-style binned LOSVD / projected-light fit --
const DEFAULT_MIN_STARS_PER_BIN = 20       # minimum stars per projected radial bin
const DEFAULT_NVBIN             = 21       # LOSVD velocity bins per radial aperture
const DEFAULT_KARL_ALPHA        = 1e-4     # legacy value retained only for call compatibility
const DEFAULT_KARL_ALPHAT       = 1.0      # Karl-style data-mismatch multiplier in entropy mode
const DEFAULT_KARL_MAXITER      = 60       # Karl SPEAR/Newton iteration cap
const DEFAULT_KARL_ENTROPY_FLOOR = 1e-12   # floor for log(w_i*wphase_i) entropy
const DEFAULT_KARL_APFAC         = 1.0      # Karl SPEAR step factor

# ============================================================
# §2  TYPES, CACHES, INLINE HELPERS
# ============================================================
@inline f64(x)=Float64(x)
@inline safe_sign(x)=x>0 ? 1.0 : (x<0 ? -1.0 : 0.0)
@inline _ssin(theta::Float64)=begin s=sin(theta); abs(s)>1e-12 ? s : safe_sign(s)*1e-12 end
@inline function _sincos_safe(theta::Float64); s,cc=sincos(theta); abs(s)>1e-12 ? (s,cc) : (safe_sign(s)*1e-12,cc) end
@inline clamp01(x::Float64)=x<0 ? 0.0 : (x>1 ? 1.0 : x)

struct HaloContext
    halo::Dict{Symbol,Any}
    R::Vector{Float64}
    tabv::Vector{Float64}
    tabfr::Vector{Float64}
    Menc::Vector{Float64}
    pot::Function
    frc::Function
end

const _HALO_CTX_CACHE = Dict{Tuple{Float64,Float64,Float64,Float64,UInt64,Symbol,Float64,Int,Float64},HaloContext}()
const _HALO_LOCK = ReentrantLock()

# ============================================================
# §3  SMALL UTILITIES
# ============================================================
@inline function normalize_halo(halo)
    h=Dict{Symbol,Any}()
    for (k,v) in halo
        h[k isa Symbol ? k : Symbol(String(k))]=v
    end
    if haskey(h,:type) && !(h[:type] isa Symbol)
        h[:type]=Symbol(lowercase(String(h[:type])))
    end
    h
end
logspace10(a,b,n)=n==1 ? [10.0^a] : (da=(b-a)/(n-1); [10.0^(a+(i-1)*da) for i in 1:n])
build_R_halo_physical(n; rmin=1e-3, rmax=300.0)=logspace10(log10(rmin), log10(rmax), n)
@inline function _quant(x::Float64; digits::Int=10)
    return round(x, digits=digits)
end

# ============================================================
# §3b  KARL-STYLE OBSERVABLE HELPERS
# ============================================================
# These helpers support the copied Karl-style OSPM branch.
# They build projected radial bins, LOSVD velocity bins, observed target vectors,
# surface-brightness light targets, and WLS/NNLS-style orbit weights.
# No star-count fallback is allowed for the projected-light target.

function build_min_count_radial_edges(R_star_m::Vector{Float64}, valid_idx::Vector{Int}; min_stars_per_bin::Int=DEFAULT_MIN_STARS_PER_BIN)
    R_use = isempty(valid_idx) ? copy(R_star_m) : R_star_m[valid_idx]
    R_use = sort(R_use[isfinite.(R_use)])
    n = length(R_use)

    n == 0 && return [0.0, 1.0]
    if n == 1
        r0 = R_use[1]
        return [max(0.0, 0.9 * r0), 1.1 * r0]
    end

    if n <= min_stars_per_bin
        lo = R_use[1]
        hi = R_use[end]
        hi <= lo && (hi = lo + max(abs(lo), 1.0))
        return [lo, hi]
    end

    edges = Float64[R_use[1]]
    i = 1

    while i + min_stars_per_bin <= n
        j = i + min_stars_per_bin - 1
        if j < n
            push!(edges, 0.5 * (R_use[j] + R_use[j + 1]))
        end
        i = j + 1
    end

    edges[end] < R_use[end] && push!(edges, R_use[end])
    edges = sort(unique(edges))
    if length(edges) < 2
        edges = [R_use[1], R_use[end] + max(abs(R_use[end]), 1.0)]
    end
    return edges
end

function resolve_karl_spatial_edges(kinematic_bin_edges)
    kinematic_bin_edges === nothing &&
        error("kinematic_bin_edges is required; no adaptive radial-bin fallback is allowed")
    edges = Float64.(kinematic_bin_edges)
    length(edges) >= 2 || error("kinematic_bin_edges must contain at least two edges")
    any(.!isfinite.(edges)) && error("kinematic_bin_edges contains non-finite values")
    # Projected radius cannot be negative, and the first aperture should include
    # the galaxy center. Some kinematic-bin products start at the innermost
    # observed star radius instead of 0 pc, which drops central stars from the
    # LOSVD target builder.
    edges[1] = 0.0
    any(diff(edges) .<= 0.0) && error("kinematic_bin_edges must be strictly increasing after forcing first edge to 0 pc")
    return edges
end

function resolve_karl_light_edges(light_bin_edges)
    light_bin_edges === nothing &&
        error("light_bin_edges is required for Karl-style light constraints")
    edges = Float64.(light_bin_edges)
    length(edges) >= 2 || error("light_bin_edges must contain at least two edges")
    any(.!isfinite.(edges)) && error("light_bin_edges contains non-finite values")
    edges[1] = 0.0
    any(diff(edges) .<= 0.0) && error("light_bin_edges must be strictly increasing after forcing first edge to 0 pc")
    return edges
end

function build_velocity_edges_auto(v_mps::Vector{Float64}, verr_mps::Vector{Float64}; Nvbin::Int=DEFAULT_NVBIN)
    vv = v_mps[isfinite.(v_mps)]
    if isempty(vv)
        return collect(range(-1.0, 1.0; length=Nvbin + 1))
    end
    sig = verr_mps[isfinite.(verr_mps) .& (verr_mps .> 0.0)]
    pad = isempty(sig) ? max(std(vv), 1.0) : 3.0 * median(sig)
    vmin = minimum(vv) - pad
    vmax = maximum(vv) + pad
    vmax <= vmin && (vmax = vmin + 2.0 * max(abs(vmin), 1.0))

    return collect(range(vmin, vmax; length=Nvbin + 1))
end

@inline function _bin_index(edges::Vector{Float64}, x::Float64)
    j = searchsortedlast(edges, x)
    if j < 1 || j >= length(edges)
        return 0
    end
    return j
end

# Fast dependency-free normal CDF approximation.
# We avoid SpecialFunctions.erf here so the hot Julia path does not need an
# extra package just to smear observed LOSVD targets by measurement error.
@inline function _normal_cdf_unit(x::Float64)
    if !isfinite(x)
        return x > 0.0 ? 1.0 : 0.0
    end
    # Abramowitz-Stegun / Hart-style logistic-polynomial approximation.
    # Accuracy is more than enough for bin-probability deposition.
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    pdf = 0.3989422804014327 * exp(-0.5 * x * x)
    cdf_pos = 1.0 - pdf * poly
    return x >= 0.0 ? cdf_pos : 1.0 - cdf_pos
end

@inline function _gaussian_bin_probability(vlo::Float64, vhi::Float64, v0::Float64, sig::Float64)
    if !(isfinite(v0) && isfinite(sig) && sig > 0.0 && isfinite(vlo) && isfinite(vhi) && vhi > vlo)
        return 0.0
    end
    return max(0.0, _normal_cdf_unit((vhi - v0) / sig) - _normal_cdf_unit((vlo - v0) / sig))
end

function _normalize_nonnegative!(x::Vector{Float64})
    @inbounds for i in eachindex(x)
        (!isfinite(x[i]) || x[i] < 0.0) && (x[i] = 0.0)
    end
    s = sum(x)
    if isfinite(s) && s > 0.0
        x ./= s
    end
    return x
end

@inline function normalize_surface_brightness_profile(profile)
    profile === nothing && return nothing
    out = Dict{Symbol,Any}()
    for (k, v) in profile
        ks = k isa Symbol ? k : Symbol(String(k))
        if ks in (:R_pc, :R_inner_pc, :R_outer_pc, :light_frac, :Sigma, :Sigma_err)
            out[ks] = Float64[x for x in v]
        else
            out[ks] = v
        end
    end
    return out
end

function light_target_from_surface_brightness(profile, spatial_edges_m::Vector{Float64}; normalize::Bool=true)
    profile === nothing && error("surface_brightness_profile is required for Karl-style OSPM; no star-count fallback is allowed")
    p = normalize_surface_brightness_profile(profile)
    Nspatial = length(spatial_edges_m) - 1
    Nspatial > 0 || error("surface_brightness_profile cannot be binned because spatial_edges has fewer than two edges")
    # Already binned light fractions.  This is the preferred input because it
    # makes the Python data product the authority on the observed light profile.
    if haskey(p, :light_frac)
        t = Float64.(p[:light_frac])
        if length(t) == Nspatial
            out = copy(t)
        elseif haskey(p, :R_inner_pc) && haskey(p, :R_outer_pc)
            rin = Float64.(p[:R_inner_pc]) .* pc
            rout = Float64.(p[:R_outer_pc]) .* pc
            length(rin) == length(rout) == length(t) || error("surface_brightness_profile binned radius arrays do not match light_frac length")
            out = zeros(Float64, Nspatial)
            @inbounds for k in eachindex(t)
                lk = t[k]
                if !(isfinite(lk) && lk >= 0.0 && isfinite(rin[k]) && isfinite(rout[k]) && rout[k] > rin[k])
                    continue
                end
                src_area = rout[k]^2 - rin[k]^2
                src_area <= 0.0 && continue
                for ib in 1:Nspatial
                    lo = max(rin[k], spatial_edges_m[ib])
                    hi = min(rout[k], spatial_edges_m[ib + 1])
                    if hi > lo
                        out[ib] += lk * (hi^2 - lo^2) / src_area
                    end
                end
            end
        else
            error("surface_brightness_profile light_frac length $(length(t)) does not match Nspatial=$Nspatial and no binned radii are available for rebinning")
        end
        normalize && _normalize_nonnegative!(out)
        sum(out) > 0.0 || error("surface_brightness_profile light_frac sums to zero after cleanup")
        return out
    end
    # Unbinned projected profile sampled at R_pc.  Values are accumulated into
    # the model spatial bins and normalized to unit light.
    if !(haskey(p, :R_pc) && haskey(p, :Sigma))
        error("surface_brightness_profile must include light_frac or R_pc + Sigma")
    end
    R_m = Float64.(p[:R_pc]) .* pc
    Sigma = Float64.(p[:Sigma])
    length(R_m) == length(Sigma) || error("surface_brightness_profile R_pc and Sigma lengths do not match")
    target = zeros(Float64, Nspatial)
    @inbounds for k in eachindex(R_m)
        ib = _bin_index(spatial_edges_m, R_m[k])
        if ib > 0 && isfinite(Sigma[k]) && Sigma[k] >= 0.0
            target[ib] += Sigma[k]
        end
    end
    _normalize_nonnegative!(target)
    sum(target) > 0.0 || error("surface_brightness_profile produced zero light in the model spatial bins")
    return target
end

function observed_targets_karl( R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, kinematic_edges::Vector{Float64}, velocity_edges::Vector{Float64}; surface_brightness_profile=nothing, light_edges=nothing, sigma_floor::Float64=1e-8)
    kinematic_edges = resolve_karl_spatial_edges(kinematic_edges)
    light_edges_use = light_edges === nothing ? kinematic_edges : resolve_karl_light_edges(light_edges)
    velocity_edges = Float64.(velocity_edges)
    vlos_idx = Int[]
    @inbounds for i in eachindex(valid_vlos)
        valid_vlos[i] &&
            isfinite(R_star_m[i]) &&
            isfinite(v_star_mps[i]) &&
            isfinite(verr_star_mps[i]) &&
            verr_star_mps[i] > 0.0 &&
            push!(vlos_idx, i)
    end
    Nspatial = length(kinematic_edges) - 1
    Nvbin = length(velocity_edges) - 1
    Nlosvd = Nspatial * Nvbin
    counts_losvd = zeros(Float64, Nlosvd)
    counts_by_spatial = zeros(Float64, Nspatial)
    @inbounds for idx in vlos_idx
        ib = _bin_index(kinematic_edges, R_star_m[idx])
        ib == 0 && continue
        counts_by_spatial[ib] += 1.0
        v0 = f64(v_star_mps[idx])
        sig = f64(verr_star_mps[idx])
        psum = 0.0
        for jb in 1:Nvbin
            psum += _gaussian_bin_probability( velocity_edges[jb], velocity_edges[jb + 1], v0, sig)
        end
        if psum > 0.0
            for jb in 1:Nvbin
                row = (ib - 1) * Nvbin + jb
                p = _gaussian_bin_probability( velocity_edges[jb], velocity_edges[jb + 1], v0, sig ) / psum
                counts_losvd[row] += p
            end
        else
            jb = _bin_index(velocity_edges, v0)
            if jb > 0
                row = (ib - 1) * Nvbin + jb
                counts_losvd[row] += 1.0
            end
        end
    end
    light_target = light_target_from_surface_brightness( surface_brightness_profile, light_edges_use; normalize=true)
    losvd_light_target = light_target_from_surface_brightness( surface_brightness_profile, kinematic_edges; normalize=false)
    losvd_target = zeros(Float64, Nlosvd)
    @inbounds for ib in 1:Nspatial
        nbin = counts_by_spatial[ib]
        nbin <= 0.0 && continue
        for jb in 1:Nvbin
            row = (ib - 1) * Nvbin + jb
            losvd_target[row] = losvd_light_target[ib] * counts_losvd[row] / nbin
        end
    end

    # ================================================================================================================================================================================
    # CURRENT POI FOR KARL-STYLE OSPM
    #
    # The LOSVD target in each row is built as:
    #
    #     y_ij = L_i * p_ij
    #
    # where:
    #
    #     i    = spatial aperture / projected radial bin
    #     j    = velocity bin
    #     L_i  = projected-light fraction from the surface-brightness profile
    #     p_ij = observed LOSVD probability in that aperture after velocity-error smearing
    #
    # counts_losvd[row] is the effective observed count k_ij after each star is
    # smeared through the velocity bins by its measured velocity error.
    #
    # counts_by_spatial[ib] is the number of observed velocity stars in aperture i.
    #
    # The LOSVD sigma below is currently the main scale-setting piece for the
    # kinematic chi-square.  If chi is too small or too large, start here.
    #
    # Current diagnostic model:
    #
    #     Use a finite-count Dirichlet / Jeffreys-style uncertainty for p_ij,
    #     then propagate it through y_ij = L_i * p_ij.
    #
    # This keeps empty velocity bins from becoming hard zero-probability walls.
    # With only about 20 stars per aperture, an empty observed velocity bin means
    # no star landed there.  It does not mean the true LOSVD probability is known
    # to be exactly zero.
    #
    # The light target is still set by the surface-brightness profile integrated
    # over the same spatial bins.  The light_sigma block below is kept separate
    # from the LOSVD sigma while debugging, so the light term does not hide the
    # kinematic behavior.
    # ================================================================================================================================================================================

    losvd_sigma = similar(losvd_target)

    # Finite-count LOSVD uncertainty.
    #
    # The observable is:
    #     y_ij = L_i * p_ij
    #
    # counts_losvd[row] is the effective count k_ij after velocity-error
    # Gaussian deposition. With only N_i stars in an aperture, an empty
    # velocity bin is not known perfectly. A Jeffreys/Dirichlet pseudo-count
    # keeps zero-count bins from becoming hard walls.
    alpha_dirichlet = 0.5

    @inbounds for ib in 1:Nspatial
        nbin = max(counts_by_spatial[ib], 1.0)
        Li = max(losvd_light_target[ib], 0.0)
        a0 = nbin + Nvbin * alpha_dirichlet
        for jb in 1:Nvbin
            row = (ib - 1) * Nvbin + jb
            kij = max(counts_losvd[row], 0.0)
            aj = kij + alpha_dirichlet
            var_pij = aj * (a0 - aj) / (a0 * a0 * (a0 + 1.0))
            losvd_sigma[row] = max( Li * sqrt(max(var_pij, 0.0)), sigma_floor)
        end
    end

    # ================================================================================================================================================================================
    # Light uncertainty block.
    # Keep separate from LOSVD sigma while debugging.  Do not let this hide LOSVD behavior.
    # ================================================================================================================================================================================
    light_sigma = similar(light_target)
    ntot = max(sum(counts_by_spatial), 1.0)
    @inbounds for ib in eachindex(light_target)
        light_sigma[ib] = max(sqrt(max(light_target[ib], sigma_floor)) / sqrt(ntot), sigma_floor)
    end
    return losvd_target, losvd_sigma, light_target, light_sigma, counts_by_spatial
end

# ============================================================
# §4  KARL WEIGHT / SPEAR SOLVER
# ============================================================
# The entropy, wphase, expanded Cm, LOSVD slack-variable SPEAR solve,
# xmu helpers, and χ² scoring live in OSPM_Physics_Weights.jl.
include("OSPM_Physics_Weights.jl")
include("OSPM_Physics_Force.jl")

# ============================================================
# §5  ORBIT INTEGRATION
# ============================================================

@inline function derivs(s::SVector{4,Float64}, Lz::Float64, frc, R)
    r, theta, vr, vtheta = s
    !(isfinite(r) && isfinite(theta) && isfinite(vr) && isfinite(vtheta)) &&
        return SVector(0.0, 0.0, 0.0, 0.0)
    r_safe = max(abs(r), 1e-12)
    st, ct = _sincos_safe(theta)
    r_tab = clamp(r_safe, R[1], R[end])
    fr, ftheta = frc(r_tab, theta)
    !(isfinite(fr) && isfinite(ftheta)) &&
        return SVector(0.0, 0.0, 0.0, 0.0)
    dr = vr
    dtheta = vtheta / r_safe
    dvr = (vtheta * vtheta) / r_safe + (Lz * Lz) / (r_safe^3 * st * st) + fr
    dvtheta = (Lz * Lz) * ct / (r_safe^3 * st^3) - (vr * vtheta) / r_safe + ftheta
    return SVector(dr, dtheta, dvr, dvtheta)
end

function launch_orbit_apocenter(; rapo::Float64, theta0::Float64, Lz_frac::Float64, pot, frc, r0_frac::Float64=DEFAULT_R0_FRAC, dt_frac::Float64=DEFAULT_DT_FRAC, dt_floor::Float64=DEFAULT_DT_FLOOR, debug::Bool=true)
    ss = _ssin(theta0)
    if !(isfinite(ss) && abs(ss) > EPS_SIN)
        return (nothing, 0.0, 0.0, 0.0, :reject_sin)
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
    Lz = Lz_frac * rapo * vc
    Papo = pot(rapo, theta0)
    if !isfinite(Papo)
        return (nothing, 0.0, 0.0, vc, :reject_pot)
    end
    E = Papo + (Lz^2) / (2 * rapo^2 * ss^2)
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

function integrate_orbit_rk4(; ic, xLz, orbit_ctx, nsteps=DEFAULT_NSTEPS, stop_rmin_factor=DEFAULT_STOP_RMIN_FACTOR)
    halo = orbit_ctx.halo
    rmin_stop = stop_rmin_factor * f64(halo[:rmin])
    r0      = f64(ic[1])
    theta0  = f64(ic[2])
    dt      = f64(ic[3])
    vr0     = length(ic)>=4 ? f64(ic[4]) : 0.0
    vtheta0 = length(ic)>=5 ? f64(ic[5]) : 0.0
    state = SVector(r0,theta0,vr0,vtheta0)
    ns = Int(nsteps)
    r      = Vector{Float64}(undef, ns)
    vr     = Vector{Float64}(undef, ns)
    theta  = Vector{Float64}(undef, ns)
    vtheta = Vector{Float64}(undef, ns)
    rmax_stop = 10.0 * f64(orbit_ctx.R_pos[end])
    actual = 0
    @inbounds for step in 1:ns
        !all(isfinite,state) && break
        rr = state[1]
        tr = state[2]
        (rr<=rmin_stop || rr>=rmax_stop || abs(tr)>1e6) && break
        actual += 1
        r[actual]      = rr
        vr[actual]     = state[3]
        theta[actual]  = tr
        vtheta[actual] = state[4]
        k1 = derivs(state, xLz, orbit_ctx.frc, orbit_ctx.R_pos)
        k2 = derivs(state + 0.5*dt*k1, xLz, orbit_ctx.frc, orbit_ctx.R_pos)
        k3 = derivs(state + 0.5*dt*k2, xLz, orbit_ctx.frc, orbit_ctx.R_pos)
        k4 = derivs(state + dt*k3, xLz, orbit_ctx.frc, orbit_ctx.R_pos)
        state += (dt/6.0)*(k1 + 2k2 + 2k3 + k4)
        state = SVector(state[1], clamp(state[2],1e-6,pi-1e-6), state[3], state[4])
    end
    resize!(r, actual); resize!(vr, actual); resize!(theta, actual); resize!(vtheta, actual)
    return r, vr, theta, vtheta
end

