# ========================================================================================================================
# OSPM_Physics_Support.jl — Karl-style support layer.
# Included by OSPM_Physics_Spherical.jl — do NOT load directly.
# Contains only the shared support shell and Karl-style observable machinery:
# constants, halo context construction, radial/velocity binning,
# surface-brightness targets, binned LOSVD targets, and includes for the
# weight/SPEAR and force machinery.
# Applied new karl fixes on 04/06/26 @1600
# Legacy star-level likelihood code and old back-compat sigma2 paths removed.
# ========================================================================================================================
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
const DEFAULT_DR_FRAC         = 0.01      # radial matching tolerance (fraction of R)
const DEFAULT_NBINS_OCC       = 6         # occupancy histogram bins
const DEFAULT_MAX_ATTEMPTS    = 6        # max orbit-launch attempts multiplier
const DEFAULT_DR_FLOOR_FRAC   = 0.01      # floor on dR (fraction)
const DEFAULT_DR_FLOOR_PC     = 0.0       # floor on dR (parsecs)

# -- Karl-style binned LOSVD / projected-light fit --
const DEFAULT_MIN_STARS_PER_BIN = 20       # minimum stars per projected radial bin
const DEFAULT_NVBIN             = 21       # requested LOSVD resolution; auto support may add bins at the same width
const DEFAULT_LOSVD_MIN_HALF_WIDTH_KMS = 100.0
const DEFAULT_LOSVD_MAX_OUTSIDE_FRACTION = 0.01
const DEFAULT_KARL_RESOLVED_KDE_GRID = 17
const DEFAULT_KARL_RESOLVED_KDE_WIDTH_BINS = 3.0
const DEFAULT_KARL_RESOLVED_VMIN_KMS = -25.0
const DEFAULT_KARL_RESOLVED_VMAX_KMS = 25.0
const DEFAULT_KARL_RESOLVED_BOOTSTRAPS = 300
const DEFAULT_KARL_RESOLVED_TRANSVD_SAMPLES = 1000
const DEFAULT_KARL_RESOLVED_ENVELOPE_FLOOR = 0.003

# -- Consolidated Karl observables CSV runtime --
const DEFAULT_KARL_OBSERVABLES_SCHEMA_VERSION = 1
const DEFAULT_KARL_ALPHA        = 1e-4     # legacy value retained only for call compatibility
const DEFAULT_KARL_ALPHAT       = 1.0      # Karl-style data-mismatch multiplier in entropy mode

const DEFAULT_KARL_MAXITER       = 250       # Karl SPEAR/Newton iteration cap
const DEFAULT_KARL_ENTROPY_FLOOR = 1e-30    # initialization/numerical floor only; not the positivity boundary
const DEFAULT_KARL_FILTER_TINY   = 1e-37    # Karl filter.f physical orbit-weight floor after each SPEAR step
const DEFAULT_KARL_APFAC         = 0.01      # maximum requested SPEAR step
const DEFAULT_KARL_LIGHT_REL_TOL = 0.01
const DEFAULT_KARL_DELTA_CHI2_ITER_TOL = 0.3
const DEFAULT_KARL_INVALID_SIGMA_SENTINEL = -666.0
const DEFAULT_KARL_STEP_SAFETY = 0.90        # take 90% of zero-weight boundary when step-limited
const DEFAULT_KARL_SPEAR_RCOND_WARN = 1.0e-12

# ========================================================================================================================
# §2  TYPES, CACHES, INLINE HELPERS
# ========================================================================================================================
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

struct KarlObservables
    path::String
    schema_version::Int
    galaxy::String
    distance_pc::Float64
    axis_ratio_q::Float64
    seeing_arcsec::Float64
    nrdat::Int
    nvdat::Int
    nrlib::Int
    nvlib::Int
    nvel::Int
    irrat::Int
    ivrat::Int
    radial_edges_arcsec::Vector{Float64}
    radial_edges_m::Vector{Float64}
    angular_edges::Vector{Float64}
    light_raw::Matrix{Float64}
    light_seen::Matrix{Float64}
    sumb::Matrix{Float64}
    sumbn::Matrix{Float64}
    spatial::Matrix{Float64}
    aperture_ids::Vector{Int}
    aperture_names::Vector{String}
    aperture_binsets::Vector{NTuple{4,Int}}
    aperture_light::Vector{Float64}
    star_count::Vector{Int}
    velocity_edges_mps::Vector{Float64}
    velocity_centers_mps::Vector{Float64}
    losvd_target::Vector{Float64}
    losvd_sigma::Vector{Float64}
    losvd_supported::BitVector
    losvd_center_kms::Float64
    losvd_shape::String
    losvd_sigma_extent::Float64
    mkherm_nsim::Int
    mkherm_econt_frac::Float64
    mkherm_rng::String
end

struct KarlD3TracerConstraints
    path::String
    nradial::Int
    nangular::Int
    radial_edges_m::Vector{Float64}
    angular_edges::Vector{Float64}
    row_index::Matrix{Int}
    row_radial::Vector{Int}
    row_angular::Vector{Int}
    target::Vector{Float64}
end

const _KARL_OBSERVABLES_CACHE = Dict{String,KarlObservables}()
const _KARL_OBSERVABLES_LOCK = ReentrantLock()
const _KARL_D3_TRACER_CACHE = Dict{Tuple{String,UInt64,UInt64},KarlD3TracerConstraints}()
const _KARL_D3_TRACER_LOCK = ReentrantLock()

const _HALO_CTX_CACHE = Dict{Tuple{Float64,Float64,Float64,Float64,UInt64,Symbol,Float64,Int,Float64,Float64},HaloContext}()
const _HALO_LOCK = ReentrantLock()
# ========================================================================================================================
# §3  SMALL UTILITIES
# ========================================================================================================================
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

# ========================================================================================================================
# §3b  KARL-STYLE OBSERVABLE HELPERS
# ========================================================================================================================
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

function build_velocity_edges_auto(v_mps::Vector{Float64}, verr_mps::Vector{Float64}; Nvbin::Int=DEFAULT_NVBIN, min_half_width_kms::Float64=DEFAULT_LOSVD_MIN_HALF_WIDTH_KMS)
    Nvbin > 0 || error("Nvbin must be positive")
    isfinite(min_half_width_kms) && min_half_width_kms > 0.0 || error("min_half_width_kms must be finite and positive")
    vv = v_mps[isfinite.(v_mps)]
    if isempty(vv)
        half_width = min_half_width_kms * 1.0e3
        return collect(range(-half_width, half_width; length=Nvbin + 1))
    end
    sig = verr_mps[isfinite.(verr_mps) .& (verr_mps .> 0.0)]
    pad = isempty(sig) ? max(std(vv), 1.0) : 3.0 * median(sig)
    vmin_observed = minimum(vv) - pad
    vmax_observed = maximum(vv) + pad
    vmax_observed <= vmin_observed && (vmax_observed = vmin_observed + 2.0 * max(abs(vmin_observed), 1.0))

    center = 0.5 * (vmin_observed + vmax_observed)
    observed_width = vmax_observed - vmin_observed
    bin_width_target = observed_width / Nvbin
    half_width = max(0.5 * observed_width, min_half_width_kms * 1.0e3)
    Nvbin_use = max(Nvbin, ceil(Int, 2.0 * half_width / max(bin_width_target, 1.0e-12)))
    iseven(Nvbin_use) && (Nvbin_use += 1)
    velocity_edges = collect(range(center - half_width, center + half_width; length=Nvbin_use + 1))
    actual_bin_width = (velocity_edges[end] - velocity_edges[1]) / Nvbin_use

    println("[LOSVD VELOCITY GRID]",
        " requested_Nvbin=", Nvbin,
        " used_Nvbin=", Nvbin_use,
        " observed_vmin_kms=", vmin_observed / 1.0e3,
        " observed_vmax_kms=", vmax_observed / 1.0e3,
        " grid_vmin_kms=", velocity_edges[1] / 1.0e3,
        " grid_vmax_kms=", velocity_edges[end] / 1.0e3,
        " target_bin_width_kms=", bin_width_target / 1.0e3,
        " actual_bin_width_kms=", actual_bin_width / 1.0e3,
    )

    return velocity_edges
end

@inline function _bin_index(edges::Vector{Float64}, x::Float64)
    j = searchsortedlast(edges, x)
    if j < 1 || j >= length(edges)
        return 0
    end
    return j
end


@inline function _karl_d3_stellar_model_value(stellar_model, key::Symbol)
    stellar_model === nothing && return nothing
    haskey(stellar_model, key) && return stellar_model[key]
    skey = String(key)
    haskey(stellar_model, skey) && return stellar_model[skey]
    return nothing
end

@inline function _karl_d3_homeoid_mu_from_karl_v(v::Float64, q::Float64)
    0.0 <= v <= 1.0 || error("Karl d3 angular coordinate must lie in [0,1]")
    isfinite(q) && q > 0.0 || error("Karl d3 source axis ratio must be finite and positive")
    denom2 = q^2 + (1.0 - q^2) * v^2
    denom2 > 0.0 || error("Karl d3 homeoid angular transform is singular")
    return clamp(v / sqrt(denom2), 0.0, 1.0)
end

function _karl_d3_homeoid_radial_measure(source_m_inner::Float64, source_m_outer::Float64, mu_lo::Float64, mu_hi::Float64, target_r_inner::Float64, target_r_outer::Float64, q::Float64; nquad::Int=32)
    mu_hi > mu_lo || return 0.0
    nquad > 0 || error("Karl d3 homeoid overlap quadrature must be positive")
    dmu = (mu_hi - mu_lo) / nquad
    overlap = 0.0
    q2 = q^2
    @inbounds for k in 1:nquad
        mu = mu_lo + (k - 0.5) * dmu
        radial_scale2 = 1.0 - (1.0 - q2) * mu^2
        radial_scale2 > 0.0 || continue
        radial_scale = sqrt(radial_scale2)
        overlap_m_inner = max(source_m_inner, target_r_inner / radial_scale)
        overlap_m_outer = min(source_m_outer, target_r_outer / radial_scale)
        overlap_m_outer > overlap_m_inner && (overlap += overlap_m_outer^3 - overlap_m_inner^3)
    end
    return overlap * dmu
end

function _load_karl_d3_tracer_constraints_uncached(stellar_model, radial_edges_m::Vector{Float64}, angular_edges::Vector{Float64})
    tracer_path_value = _karl_d3_stellar_model_value(stellar_model, :tracer_grid_csv)
    tracer_path_value === nothing && error("density_3d tracer constraint requires STELLAR_MODEL.tracer_grid_csv")
    path = abspath(String(tracer_path_value))
    isfile(path) || error("density_3d tracer grid CSV not found: $path")
    length(radial_edges_m) >= 2 || error("Karl d3 radial grid requires at least two edges")
    length(angular_edges) >= 2 || error("Karl d3 angular grid requires at least two edges")
    any(.!isfinite.(radial_edges_m)) && error("Karl d3 radial edges contain nonfinite values")
    any(.!isfinite.(angular_edges)) && error("Karl d3 angular edges contain nonfinite values")
    any(diff(radial_edges_m) .<= 0.0) && error("Karl d3 radial edges must be strictly increasing")
    any(diff(angular_edges) .<= 0.0) && error("Karl d3 angular edges must be strictly increasing")
    abs(angular_edges[1]) <= 1.0e-12 || error("Karl d3 angular grid must begin at v=0")
    abs(angular_edges[end] - 1.0) <= 1.0e-12 || error("Karl d3 angular grid must end at v=1")

    raw_lines = readlines(path)
    isempty(raw_lines) && error("density_3d tracer grid CSV is empty: $path")
    header = String.(strip.(split(chomp(raw_lines[1]), ","; keepempty=true)))
    columns = Dict{String,Int}(name => i for (i, name) in pairs(header))
    required = ("R_inner_pc", "R_outer_pc", "theta_inner_rad", "theta_outer_rad", "cell_luminosity_Lsun", "q_axis_ratio", "flattened_geometry", "density_coordinate")
    for name in required
        haskey(columns, name) || error("density_3d tracer grid $path is missing required column $name")
    end

    nradial = length(radial_edges_m) - 1
    nangular = length(angular_edges) - 1
    Nconstraint = 1 + (nradial - 1) * nangular
    row_index = zeros(Int, nradial, nangular)
    row_radial = Vector{Int}(undef, Nconstraint)
    row_angular = Vector{Int}(undef, Nconstraint)
    target = zeros(Float64, Nconstraint)

    @inbounds for iv in 1:nangular
        row_index[1, iv] = 1
    end
    row_radial[1] = 1
    row_angular[1] = 0

    row = 1
    @inbounds for ir in 2:nradial
        for iv in 1:nangular
            row += 1
            row_index[ir, iv] = row
            row_radial[row] = ir
            row_angular[row] = iv
        end
    end
    row == Nconstraint || error("Karl d3 row construction produced $row constraints; expected $Nconstraint")

    radial_edges_pc = radial_edges_m ./ pc
    total_luminosity = 0.0
    used_luminosity = 0.0
    q_reference = NaN
    max_required_column = maximum(columns[name] for name in required)

    @inbounds for iline in 2:length(raw_lines)
        stripped = strip(raw_lines[iline])
        isempty(stripped) && continue

        fields = String.(strip.(split(chomp(raw_lines[iline]), ","; keepempty=true)))
        length(fields) >= max_required_column || error("density_3d tracer grid $path line $iline does not contain the required columns")

        source_m_inner = parse(Float64, fields[columns["R_inner_pc"]])
        source_m_outer = parse(Float64, fields[columns["R_outer_pc"]])
        theta_inner = parse(Float64, fields[columns["theta_inner_rad"]])
        theta_outer = parse(Float64, fields[columns["theta_outer_rad"]])
        luminosity = parse(Float64, fields[columns["cell_luminosity_Lsun"]])
        q = parse(Float64, fields[columns["q_axis_ratio"]])
        flattened_geometry = strip(fields[columns["flattened_geometry"]])
        density_coordinate = strip(fields[columns["density_coordinate"]])

        isfinite(source_m_inner) && isfinite(source_m_outer) && isfinite(theta_inner) && isfinite(theta_outer) && isfinite(luminosity) && isfinite(q) ||
            error("density_3d tracer grid $path contains nonfinite required values at line $iline")
        source_m_inner >= 0.0 || error("density_3d tracer grid $path has negative R_inner_pc at line $iline")
        source_m_outer > source_m_inner || error("density_3d tracer grid $path has invalid radial cell at line $iline")
        0.0 <= theta_inner < theta_outer <= pi || error("density_3d tracer grid $path has invalid theta cell at line $iline")
        luminosity >= 0.0 || error("density_3d tracer grid $path contains negative cell luminosity at line $iline")
        q > 0.0 || error("density_3d tracer grid $path has non-positive q_axis_ratio at line $iline")
        flattened_geometry == "oblate_homeoid" || error("density_3d tracer grid $path requires flattened_geometry=oblate_homeoid; got $flattened_geometry at line $iline")
        density_coordinate == "m_pc" || error("density_3d tracer grid $path requires density_coordinate=m_pc; got $density_coordinate at line $iline")

        if isnan(q_reference)
            q_reference = q
        else
            scale = max(abs(q_reference), abs(q), 1.0)
            abs(q - q_reference) <= 1.0e-12 * scale || error("density_3d tracer grid $path changes q_axis_ratio between rows")
        end

        total_luminosity += luminosity
        luminosity == 0.0 && continue

        source_radial_measure = source_m_outer^3 - source_m_inner^3
        source_radial_measure > 0.0 || continue

        mu_a = cos(theta_inner)
        mu_b = cos(theta_outer)
        source_mu_lo = min(mu_a, mu_b)
        source_mu_hi = max(mu_a, mu_b)
        source_angular_measure = source_mu_hi - source_mu_lo
        source_angular_measure > 0.0 || continue

        source_measure = source_radial_measure * source_angular_measure
        source_measure > 0.0 || continue

        for ir in 1:nradial
            target_r_inner = radial_edges_pc[ir]
            target_r_outer = radial_edges_pc[ir + 1]

            if ir == 1
                overlap_measure = _karl_d3_homeoid_radial_measure(source_m_inner, source_m_outer, source_mu_lo, source_mu_hi, target_r_inner, target_r_outer, q)
                overlap_measure > 0.0 || continue
                contribution = luminosity * overlap_measure / source_measure
                target[1] += contribution
                used_luminosity += contribution
                continue
            end

            for iv in 1:nangular
                v_inner = angular_edges[iv]
                v_outer = angular_edges[iv + 1]
                homeoid_mu_inner = _karl_d3_homeoid_mu_from_karl_v(v_inner, q)
                homeoid_mu_outer = _karl_d3_homeoid_mu_from_karl_v(v_outer, q)

                positive_lo = max(source_mu_lo, homeoid_mu_inner)
                positive_hi = min(source_mu_hi, homeoid_mu_outer)
                negative_lo = max(source_mu_lo, -homeoid_mu_outer)
                negative_hi = min(source_mu_hi, -homeoid_mu_inner)

                overlap_measure = 0.0
                positive_hi > positive_lo && (overlap_measure += _karl_d3_homeoid_radial_measure(source_m_inner, source_m_outer, positive_lo, positive_hi, target_r_inner, target_r_outer, q))
                negative_hi > negative_lo && (overlap_measure += _karl_d3_homeoid_radial_measure(source_m_inner, source_m_outer, negative_lo, negative_hi, target_r_inner, target_r_outer, q))
                overlap_measure > 0.0 || continue

                contribution = luminosity * overlap_measure / source_measure
                target[row_index[ir, iv]] += contribution
                used_luminosity += contribution
            end
        end
    end

    isfinite(total_luminosity) && total_luminosity > 0.0 || error("density_3d tracer grid $path has non-positive total luminosity")
    isfinite(used_luminosity) && used_luminosity > 0.0 || error("density_3d tracer target has no luminosity inside the Karl d3 grid")

    target ./= used_luminosity
    isapprox(sum(target), 1.0; rtol=1.0e-12, atol=1.0e-12) || error("Karl d3 tracer target does not normalize to one")

    retained_fraction = used_luminosity / total_luminosity
    zero_target_rows = count(==(0.0), target)

    println("[KARL D3 TRACER LOAD]",
        " path=", path,
        " radial_bins=", nradial,
        " angular_bins=", nangular,
        " constraints=", Nconstraint,
        " inner_theta_collapsed=true",
        " source_coordinate=oblate_homeoid_m",
        " target_coordinate=karl_spherical_r_v",
        " overlap_rebinned=true",
        " overlap_quadrature=32",
        " q_axis_ratio=", q_reference,
        " zero_target_rows=", zero_target_rows,
        " retained_light_fraction=", retained_fraction,
        " Rmax_pc=", radial_edges_m[end] / pc)

    return KarlD3TracerConstraints(path, nradial, nangular, copy(radial_edges_m), copy(angular_edges), row_index, row_radial, row_angular, target)
end

function load_karl_d3_tracer_constraints(stellar_model, radial_edges_m::Vector{Float64}, angular_edges::Vector{Float64})
    tracer_path_value = _karl_d3_stellar_model_value(stellar_model, :tracer_grid_csv)
    tracer_path_value === nothing && error("density_3d tracer constraint requires STELLAR_MODEL.tracer_grid_csv")
    path = abspath(String(tracer_path_value))
    key = (path, UInt64(hash(Tuple(radial_edges_m))), UInt64(hash(Tuple(angular_edges))))
    lock(_KARL_D3_TRACER_LOCK)
    try
        haskey(_KARL_D3_TRACER_CACHE, key) && return _KARL_D3_TRACER_CACHE[key]
    finally
        unlock(_KARL_D3_TRACER_LOCK)
    end
    constraints = _load_karl_d3_tracer_constraints_uncached(stellar_model, radial_edges_m, angular_edges)
    lock(_KARL_D3_TRACER_LOCK)
    try
        return get!(_KARL_D3_TRACER_CACHE, key, constraints)
    finally
        unlock(_KARL_D3_TRACER_LOCK)
    end
end

@inline function karl_d3_tracer_row(constraints::KarlD3TracerConstraints, r_m::Float64, theta::Float64)
    isfinite(r_m) && r_m >= 0.0 || return 0
    isfinite(theta) || return 0
    ir = _bin_index(constraints.radial_edges_m, r_m)
    ir == 0 && return 0
    vcoord = clamp(abs(cos(theta)), 0.0, 1.0)
    iv = searchsortedlast(constraints.angular_edges, vcoord)
    iv < 1 && return 0
    iv >= length(constraints.angular_edges) && (iv = constraints.nangular)
    return constraints.row_index[ir, iv]
end

function karl_d3_radial_target(constraints::KarlD3TracerConstraints)
    radial = zeros(Float64, constraints.nradial)
    @inbounds for row in eachindex(constraints.target)
        radial[constraints.row_radial[row]] += constraints.target[row]
    end
    return radial
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

@inline function _karl_nint(x::Float64)
    return x >= 0.0 ? floor(Int, x + 0.5) : ceil(Int, x - 0.5)
end

mutable struct _KarlRan2State
    idum::Int64
    idum2::Int64
    iv::Vector{Int64}
    iy::Int64
end

_KarlRan2State() = _KarlRan2State(-1, 123456789, zeros(Int64, 32), 0)

function _karl_ran2!(state::_KarlRan2State)
    IM1 = 2147483563
    IM2 = 2147483399
    AM = 1.0 / IM1
    IMM1 = IM1 - 1
    IA1 = 40014
    IA2 = 40692
    IQ1 = 53668
    IQ2 = 52774
    IR1 = 12211
    IR2 = 3791
    NTAB = 32
    NDIV = 1 + IMM1 ÷ NTAB
    EPS = 1.2e-7
    RNMX = 1.0 - EPS

    if state.idum <= 0
        state.idum = max(-state.idum, 1)
        state.idum2 = state.idum
        @inbounds for j in (NTAB + 8):-1:1
            k = state.idum ÷ IQ1
            state.idum = IA1 * (state.idum - k * IQ1) - k * IR1
            state.idum < 0 && (state.idum += IM1)
            j <= NTAB && (state.iv[j] = state.idum)
        end
        state.iy = state.iv[1]
    end

    k = state.idum ÷ IQ1
    state.idum = IA1 * (state.idum - k * IQ1) - k * IR1
    state.idum < 0 && (state.idum += IM1)

    k = state.idum2 ÷ IQ2
    state.idum2 = IA2 * (state.idum2 - k * IQ2) - k * IR2
    state.idum2 < 0 && (state.idum2 += IM2)

    j = 1 + state.iy ÷ NDIV
    state.iy = state.iv[j] - state.idum2
    state.iv[j] = state.idum
    state.iy < 1 && (state.iy += IMM1)

    return min(AM * state.iy, RNMX)
end

function _karl_accumulate_kernel!(ek::Vector{Float64}, xscaled::Vector{Float64}, widths::Vector{Float64})
    length(xscaled) == length(widths) || error("Karl adaptive-kernel widths do not match sample length")
    ngrid = length(ek)
    fill!(ek, 0.0)

    @inbounds for i in eachindex(xscaled)
        xi = xscaled[i]
        h = widths[i]
        isfinite(h) && h > 0.0 || error("Karl adaptive-kernel width must be finite and positive")
        jlo = max(1, _karl_nint(xi - h))
        jhi = min(ngrid, _karl_nint(xi + h))
        jhi < jlo && continue
        xp = xi / h

        for j in jlo:jhi
            x1 = f64(j) - 0.5
            x2 = x1 + 1.0
            abs(xi - x1) > h && (x1 = xi - h)
            abs(xi - x2) > h && (x2 = xi + h)
            x1 /= h
            x2 /= h
            xint = x2 - x1 + ((xp - x2)^3 - (xp - x1)^3) / 3.0
            ek[j] += xint
        end
    end

    return ek
end

function _karl_adaptive_kde_1d(v_kms::Vector{Float64}; grid_size::Int=DEFAULT_KARL_RESOLVED_KDE_GRID, width_bins::Float64=DEFAULT_KARL_RESOLVED_KDE_WIDTH_BINS, vmin_kms::Float64=DEFAULT_KARL_RESOLVED_VMIN_KMS, vmax_kms::Float64=DEFAULT_KARL_RESOLVED_VMAX_KMS, alpha::Float64=0.5)
    n = length(v_kms)
    n > 0 || error("Karl adaptive KDE requires at least one velocity")
    grid_size > 1 || error("Karl adaptive KDE grid_size must exceed one")
    isfinite(width_bins) && width_bins > 0.0 || error("Karl adaptive KDE width_bins must be finite and positive")
    isfinite(vmin_kms) && isfinite(vmax_kms) && vmax_kms > vmin_kms || error("Karl adaptive KDE velocity bounds are invalid")
    isfinite(alpha) && alpha > 0.0 || error("Karl adaptive KDE alpha must be finite and positive")

    scalx = f64(grid_size) / (vmax_kms - vmin_kms)
    xscaled = Vector{Float64}(undef, n)
    @inbounds for i in eachindex(v_kms)
        vi = v_kms[i]
        isfinite(vi) || error("Karl adaptive KDE received a nonfinite velocity")
        xi = scalx * (vi - vmin_kms) + 0.5
        xi <= 0.5 && (xi = 0.5001)
        xi >= grid_size + 0.5 && (xi = grid_size + 0.4990)
        xscaled[i] = xi
    end

    pilot = zeros(Float64, grid_size)
    _karl_accumulate_kernel!(pilot, xscaled, fill(width_bins, n))

    log_g_sum = 0.0
    @inbounds for i in eachindex(xscaled)
        ix = clamp(_karl_nint(xscaled[i]), 1, grid_size)
        log_g_sum += log(max(pilot[ix], 1.0e-20))
    end
    g = exp(log_g_sum / n)
    isfinite(g) && g > 0.0 || error("Karl adaptive KDE geometric-mean pilot density is invalid")

    widths = Vector{Float64}(undef, n)
    @inbounds for i in eachindex(xscaled)
        ix = clamp(_karl_nint(xscaled[i]), 1, grid_size)
        clam = (pilot[ix] / g)^(-alpha)
        widths[i] = width_bins * clam
    end

    ek = zeros(Float64, grid_size)
    _karl_accumulate_kernel!(ek, xscaled, widths)
    con = 3.0 / (4.0 * n)
    ek .*= con * scalx

    xl = Vector{Float64}(undef, grid_size)
    @inbounds for j in 1:grid_size
        xl[j] = (f64(j) - 0.5) / scalx + vmin_kms
    end

    return xl, ek
end

function _karl_bootstrap_kde_envelope(v_kms::Vector{Float64}; grid_size::Int=DEFAULT_KARL_RESOLVED_KDE_GRID, width_bins::Float64=DEFAULT_KARL_RESOLVED_KDE_WIDTH_BINS, vmin_kms::Float64=DEFAULT_KARL_RESOLVED_VMIN_KMS, vmax_kms::Float64=DEFAULT_KARL_RESOLVED_VMAX_KMS, bootstraps::Int=DEFAULT_KARL_RESOLVED_BOOTSTRAPS)
    n = length(v_kms)
    n > 0 || error("Karl bootstrap KDE requires at least one velocity")
    bootstraps > 1 || error("Karl bootstrap KDE requires at least two bootstrap realizations")

    xl, central = _karl_adaptive_kde_1d(v_kms; grid_size=grid_size, width_bins=width_bins, vmin_kms=vmin_kms, vmax_kms=vmax_kms, alpha=0.5)
    xsim = Matrix{Float64}(undef, bootstraps, grid_size)
    sample = Vector{Float64}(undef, n)
    rng = _KarlRan2State()

    @inbounds for isim in 1:bootstraps
        for i in 1:n
            iran = _karl_nint(_karl_ran2!(rng) * (n - 1)) + 1
            sample[i] = v_kms[iran]
        end
        _, sim = _karl_adaptive_kde_1d(sample; grid_size=grid_size, width_bins=width_bins, vmin_kms=vmin_kms, vmax_kms=vmax_kms, alpha=0.5)
        xsim[isim, :] .= sim
    end

    # ncont1.f indexes the sorted bootstrap array after its DO variable has
    # advanced to bootstraps+1. Preserve that historical 16/84 behavior.
    i16 = clamp(_karl_nint((bootstraps + 1) * 0.16), 1, bootstraps)
    i84 = clamp(_karl_nint((bootstraps + 1) * 0.84), 1, bootstraps)
    lower = Vector{Float64}(undef, grid_size)
    upper = Vector{Float64}(undef, grid_size)

    @inbounds for j in 1:grid_size
        column = sort!(collect(view(xsim, :, j)))
        lower[j] = column[i16]
        upper[j] = column[i84]
    end

    return xl, central, lower, upper
end

@inline function _karl_linear_interp(x::Vector{Float64}, y::Vector{Float64}, xp::Float64)
    length(x) == length(y) || error("Karl interpolation arrays have different lengths")
    length(x) >= 2 || error("Karl interpolation requires at least two samples")
    xp <= x[1] && return y[1]
    xp >= x[end] && return y[end]
    j = clamp(searchsortedlast(x, xp), 1, length(x) - 1)
    dx = x[j + 1] - x[j]
    dx > 0.0 || error("Karl interpolation grid is not strictly increasing")
    return y[j] + (y[j + 1] - y[j]) * (xp - x[j]) / dx
end

function _karl_transvd_profile(v_kms::Vector{Float64}, central::Vector{Float64}, lower::Vector{Float64}, upper::Vector{Float64}; ntot::Int=DEFAULT_KARL_RESOLVED_TRANSVD_SAMPLES, envelope_floor::Float64=DEFAULT_KARL_RESOLVED_ENVELOPE_FLOOR)
    length(v_kms) == length(central) == length(lower) == length(upper) || error("Karl transvd arrays have inconsistent lengths")
    length(v_kms) >= 2 || error("Karl transvd requires at least two LOSVD samples")
    ntot > 1 || error("Karl transvd ntot must exceed one")
    isfinite(envelope_floor) && envelope_floor >= 0.0 || error("Karl transvd envelope floor must be finite and nonnegative")

    sum_central = sum(central)
    isfinite(sum_central) && sum_central > 0.0 || error("Karl transvd central LOSVD has non-positive normalization")

    y = central ./ sum_central
    yl = (central .- lower) ./ sum_central
    yh = (upper .- central) ./ sum_central

    vfine = Vector{Float64}(undef, ntot)
    yfine = Vector{Float64}(undef, ntot)
    ylow = Vector{Float64}(undef, ntot)
    yhigh = Vector{Float64}(undef, ntot)
    vfirst = v_kms[1]

    @inbounds for i in 1:ntot
        vp = vfirst - 2.0 * vfirst / (ntot - 1) * (i - 1)
        yp = _karl_linear_interp(v_kms, y, vp)
        ylp = _karl_linear_interp(v_kms, yl, vp)
        yhp = _karl_linear_interp(v_kms, yh, vp)
        yn = max(0.0, yp)
        ynl = max(0.0, yn - abs(ylp))
        ynh = yn + abs(yhp)
        if ynh < envelope_floor
            ynl = 0.0
            ynh = envelope_floor
        end
        vfine[i] = vp
        yfine[i] = yn
        ylow[i] = ynl
        yhigh[i] = ynh
    end

    return vfine, yfine, ylow, yhigh
end

@inline function _karl_observables_required_field(fields::Vector{String}, columns::Dict{String,Int}, name::AbstractString, path::AbstractString, lineno::Int)
    haskey(columns, String(name)) || error("Karl observables CSV $path is missing column $name")
    value = strip(fields[columns[String(name)]])
    isempty(value) && error("Karl observables CSV $path has an empty $name value at line $lineno")
    return value
end

@inline function _karl_observables_parse_float(fields::Vector{String}, columns::Dict{String,Int}, name::AbstractString, path::AbstractString, lineno::Int)
    value = parse(Float64, _karl_observables_required_field(fields, columns, name, path, lineno))
    isfinite(value) || error("Karl observables CSV $path has nonfinite $name at line $lineno")
    return value
end

@inline function _karl_observables_parse_int(fields::Vector{String}, columns::Dict{String,Int}, name::AbstractString, path::AbstractString, lineno::Int)
    raw = _karl_observables_required_field(fields, columns, name, path, lineno)
    try
        return parse(Int, raw)
    catch
        value = parse(Float64, raw)
        isfinite(value) && isinteger(value) || error("Karl observables CSV $path has non-integer $name=$raw at line $lineno")
        return Int(value)
    end
end

function _karl_observables_set_edge!(edges::Vector{Float64}, index::Int, value::Float64, label::AbstractString, path::AbstractString)
    1 <= index <= length(edges) || error("Karl observables $label edge index $index is outside 1:$(length(edges)) in $path")
    isfinite(value) || error("Karl observables $label edge $index is nonfinite in $path")
    if isnan(edges[index])
        edges[index] = value
    else
        scale = max(abs(edges[index]), abs(value), 1.0)
        abs(edges[index] - value) <= 1.0e-10 * scale || error("Karl observables $label edge $index is inconsistent in $path")
    end
    return nothing
end

function _load_karl_observables_uncached(path::AbstractString)
    isfile(path) || error("Karl observables CSV not found: $path")
    raw_lines = readlines(path)
    isempty(raw_lines) && error("Karl observables CSV is empty: $path")
    header = String.(strip.(split(chomp(raw_lines[1]), ","; keepempty=true)))
    columns = Dict{String,Int}(name => i for (i, name) in pairs(header))
    required = ("schema_version", "row_type", "galaxy", "distance_pc", "axis_ratio_q", "seeing_arcsec", "nrdat", "nvdat", "nrlib", "nvlib", "nvel", "irrat", "ivrat", "ir", "iv", "r_inner_arcsec", "r_outer_arcsec", "r_inner_pc", "r_outer_pc", "vcoord_inner", "vcoord_outer", "light_raw", "light_seen", "source_ir", "source_iv", "target_ir", "target_iv", "sumb", "sumbn", "aperture_id", "aperture_name", "ir_start", "ir_end", "iv_start", "iv_end", "aperture_light", "star_count", "losvd_center_kms", "losvd_shape", "losvd_sigma_extent", "mkherm_nsim", "mkherm_econt_frac", "mkherm_rng", "velocity_bin", "velocity_low_kms", "velocity_center_kms", "velocity_high_kms", "losvd_target", "losvd_sigma", "losvd_supported")
    for name in required
        haskey(columns, name) || error("Karl observables CSV $path is missing required column $name")
    end

    parsed_rows = Tuple{Int,Vector{String}}[]
    metadata_rows = Tuple{Int,Vector{String}}[]
    @inbounds for iline in 2:length(raw_lines)
        stripped = strip(raw_lines[iline])
        isempty(stripped) && continue
        fields = String.(strip.(split(chomp(raw_lines[iline]), ","; keepempty=true)))
        length(fields) == length(header) || error("Karl observables CSV $path line $iline has $(length(fields)) columns; expected $(length(header))")
        push!(parsed_rows, (iline, fields))
        strip(fields[columns["row_type"]]) == "metadata" && push!(metadata_rows, (iline, fields))
    end
    length(metadata_rows) == 1 || error("Karl observables CSV $path must contain exactly one metadata row; found $(length(metadata_rows))")

    metadata_lineno, metadata = metadata_rows[1]
    schema_version = _karl_observables_parse_int(metadata, columns, "schema_version", path, metadata_lineno)
    schema_version == DEFAULT_KARL_OBSERVABLES_SCHEMA_VERSION || error("Karl observables CSV $path schema_version=$schema_version; expected $DEFAULT_KARL_OBSERVABLES_SCHEMA_VERSION")
    galaxy = _karl_observables_required_field(metadata, columns, "galaxy", path, metadata_lineno)
    distance_pc = _karl_observables_parse_float(metadata, columns, "distance_pc", path, metadata_lineno)
    axis_ratio_q = _karl_observables_parse_float(metadata, columns, "axis_ratio_q", path, metadata_lineno)
    seeing_arcsec = _karl_observables_parse_float(metadata, columns, "seeing_arcsec", path, metadata_lineno)
    nrdat = _karl_observables_parse_int(metadata, columns, "nrdat", path, metadata_lineno)
    nvdat = _karl_observables_parse_int(metadata, columns, "nvdat", path, metadata_lineno)
    nrlib = _karl_observables_parse_int(metadata, columns, "nrlib", path, metadata_lineno)
    nvlib = _karl_observables_parse_int(metadata, columns, "nvlib", path, metadata_lineno)
    nvel = _karl_observables_parse_int(metadata, columns, "nvel", path, metadata_lineno)
    irrat = _karl_observables_parse_int(metadata, columns, "irrat", path, metadata_lineno)
    ivrat = _karl_observables_parse_int(metadata, columns, "ivrat", path, metadata_lineno)
    losvd_center_kms = _karl_observables_parse_float(metadata, columns, "losvd_center_kms", path, metadata_lineno)
    losvd_shape = lowercase(_karl_observables_required_field(metadata, columns, "losvd_shape", path, metadata_lineno))
    losvd_sigma_extent = _karl_observables_parse_float(metadata, columns, "losvd_sigma_extent", path, metadata_lineno)
    mkherm_nsim = _karl_observables_parse_int(metadata, columns, "mkherm_nsim", path, metadata_lineno)
    mkherm_econt_frac = _karl_observables_parse_float(metadata, columns, "mkherm_econt_frac", path, metadata_lineno)
    mkherm_rng = _karl_observables_required_field(metadata, columns, "mkherm_rng", path, metadata_lineno)

    distance_pc > 0.0 || error("Karl observables distance_pc must be positive in $path")
    axis_ratio_q > 0.0 || error("Karl observables axis_ratio_q must be positive in $path")
    seeing_arcsec >= 0.0 || error("Karl observables seeing_arcsec must be nonnegative in $path")
    nrdat > 0 && nvdat > 0 && nrlib > 0 && nvlib > 0 && nvel > 1 || error("Karl observables grid dimensions are invalid in $path")
    nrdat % nrlib == 0 || error("Karl observables nrdat=$nrdat is not divisible by nrlib=$nrlib in $path")
    nvdat % nvlib == 0 || error("Karl observables nvdat=$nvdat is not divisible by nvlib=$nvlib in $path")
    irrat == nrdat ÷ nrlib || error("Karl observables irrat=$irrat does not match nrdat/nrlib=$(nrdat ÷ nrlib) in $path")
    ivrat == nvdat ÷ nvlib || error("Karl observables ivrat=$ivrat does not match nvdat/nvlib=$(nvdat ÷ nvlib) in $path")
    losvd_shape == "gaussian" || error("Karl observables losvd_shape=$losvd_shape is unsupported; expected gaussian")
    isfinite(losvd_sigma_extent) && losvd_sigma_extent > 0.0 || error("Karl observables losvd_sigma_extent must be positive in $path")
    mkherm_nsim > 0 || error("Karl observables mkherm_nsim must be positive in $path")
    mkherm_econt_frac >= 0.0 || error("Karl observables mkherm_econt_frac must be nonnegative in $path")

    radial_edges_arcsec = fill(NaN, nrlib + 1)
    radial_edges_pc = fill(NaN, nrlib + 1)
    angular_edges = fill(NaN, nvlib + 1)
    light_raw = fill(NaN, nrlib, nvlib)
    light_seen = fill(NaN, nrlib, nvlib)
    spatial_seen = falses(nrlib, nvlib)

    Ncell = nrlib * nvlib
    sumb = zeros(Float64, Ncell, Ncell)
    sumbn = zeros(Float64, Ncell, Ncell)
    transfer_seen = falses(Ncell, Ncell)

    aperture_ids_raw = Int[]
    aperture_names_raw = String[]
    aperture_binsets_raw = NTuple{4,Int}[]
    aperture_light_raw = Float64[]
    star_count_raw = Int[]

    losvd_rows = Tuple{Int,Int,Float64,Float64,Float64,Float64,Float64,Bool,Int}[]
    spatial_count = 0
    transfer_count = 0

    @inbounds for (lineno, fields) in parsed_rows
        row_type = strip(fields[columns["row_type"]])
        if row_type == "metadata"
            continue
        elseif row_type == "spatial_cell"
            ir = _karl_observables_parse_int(fields, columns, "ir", path, lineno)
            iv = _karl_observables_parse_int(fields, columns, "iv", path, lineno)
            1 <= ir <= nrlib || error("Karl observables ir=$ir is outside 1:$nrlib at line $lineno in $path")
            1 <= iv <= nvlib || error("Karl observables iv=$iv is outside 1:$nvlib at line $lineno in $path")
            spatial_seen[ir, iv] && error("Karl observables repeats spatial cell ($ir,$iv) in $path")
            spatial_seen[ir, iv] = true
            _karl_observables_set_edge!(radial_edges_arcsec, ir, _karl_observables_parse_float(fields, columns, "r_inner_arcsec", path, lineno), "radial arcsec", path)
            _karl_observables_set_edge!(radial_edges_arcsec, ir + 1, _karl_observables_parse_float(fields, columns, "r_outer_arcsec", path, lineno), "radial arcsec", path)
            _karl_observables_set_edge!(radial_edges_pc, ir, _karl_observables_parse_float(fields, columns, "r_inner_pc", path, lineno), "radial pc", path)
            _karl_observables_set_edge!(radial_edges_pc, ir + 1, _karl_observables_parse_float(fields, columns, "r_outer_pc", path, lineno), "radial pc", path)
            _karl_observables_set_edge!(angular_edges, iv, _karl_observables_parse_float(fields, columns, "vcoord_inner", path, lineno), "angular", path)
            _karl_observables_set_edge!(angular_edges, iv + 1, _karl_observables_parse_float(fields, columns, "vcoord_outer", path, lineno), "angular", path)
            light_raw[ir, iv] = _karl_observables_parse_float(fields, columns, "light_raw", path, lineno)
            light_seen[ir, iv] = _karl_observables_parse_float(fields, columns, "light_seen", path, lineno)
            light_raw[ir, iv] >= 0.0 || error("Karl observables light_raw is negative at line $lineno in $path")
            light_seen[ir, iv] >= 0.0 || error("Karl observables light_seen is negative at line $lineno in $path")
            spatial_count += 1
        elseif row_type == "seeing_transfer"
            source_ir = _karl_observables_parse_int(fields, columns, "source_ir", path, lineno)
            source_iv = _karl_observables_parse_int(fields, columns, "source_iv", path, lineno)
            target_ir = _karl_observables_parse_int(fields, columns, "target_ir", path, lineno)
            target_iv = _karl_observables_parse_int(fields, columns, "target_iv", path, lineno)
            1 <= source_ir <= nrlib || error("Karl observables source_ir=$source_ir is outside 1:$nrlib at line $lineno in $path")
            1 <= target_ir <= nrlib || error("Karl observables target_ir=$target_ir is outside 1:$nrlib at line $lineno in $path")
            1 <= source_iv <= nvlib || error("Karl observables source_iv=$source_iv is outside 1:$nvlib at line $lineno in $path")
            1 <= target_iv <= nvlib || error("Karl observables target_iv=$target_iv is outside 1:$nvlib at line $lineno in $path")
            source_cell = (source_ir - 1) * nvlib + source_iv
            target_cell = (target_ir - 1) * nvlib + target_iv
            transfer_seen[source_cell, target_cell] && error("Karl observables repeats seeing transfer ($source_ir,$source_iv)->($target_ir,$target_iv) in $path")
            transfer_seen[source_cell, target_cell] = true
            positive = _karl_observables_parse_float(fields, columns, "sumb", path, lineno)
            negative = _karl_observables_parse_float(fields, columns, "sumbn", path, lineno)
            positive >= 0.0 || error("Karl observables sumb is negative at line $lineno in $path")
            negative >= 0.0 || error("Karl observables sumbn is negative at line $lineno in $path")
            sumb[source_cell, target_cell] = positive
            sumbn[source_cell, target_cell] = negative
            transfer_count += 1
        elseif row_type == "aperture"
            aperture_id = _karl_observables_parse_int(fields, columns, "aperture_id", path, lineno)
            aperture_id > 0 || error("Karl observables aperture_id must be positive at line $lineno in $path")
            aperture_id in aperture_ids_raw && error("Karl observables repeats aperture_id=$aperture_id in $path")
            aperture_name = _karl_observables_required_field(fields, columns, "aperture_name", path, lineno)
            ir_start = _karl_observables_parse_int(fields, columns, "ir_start", path, lineno)
            ir_end = _karl_observables_parse_int(fields, columns, "ir_end", path, lineno)
            iv_start = _karl_observables_parse_int(fields, columns, "iv_start", path, lineno)
            iv_end = _karl_observables_parse_int(fields, columns, "iv_end", path, lineno)
            1 <= ir_start <= ir_end <= nrlib || error("Karl observables aperture $aperture_id has invalid radial bin range $ir_start:$ir_end in $path")
            1 <= iv_start <= iv_end <= nvlib || error("Karl observables aperture $aperture_id has invalid angular bin range $iv_start:$iv_end in $path")
            aperture_light = _karl_observables_parse_float(fields, columns, "aperture_light", path, lineno)
            aperture_light > 0.0 || error("Karl observables aperture $aperture_id has non-positive light in $path")
            star_count = _karl_observables_parse_int(fields, columns, "star_count", path, lineno)
            star_count >= 0 || error("Karl observables aperture $aperture_id has negative star_count in $path")
            push!(aperture_ids_raw, aperture_id)
            push!(aperture_names_raw, aperture_name)
            push!(aperture_binsets_raw, (ir_start, ir_end, iv_start, iv_end))
            push!(aperture_light_raw, aperture_light)
            push!(star_count_raw, star_count)
        elseif row_type == "losvd_bin"
            aperture_id = _karl_observables_parse_int(fields, columns, "aperture_id", path, lineno)
            velocity_bin = _karl_observables_parse_int(fields, columns, "velocity_bin", path, lineno)
            vlo = _karl_observables_parse_float(fields, columns, "velocity_low_kms", path, lineno)
            vcenter = _karl_observables_parse_float(fields, columns, "velocity_center_kms", path, lineno)
            vhi = _karl_observables_parse_float(fields, columns, "velocity_high_kms", path, lineno)
            target = _karl_observables_parse_float(fields, columns, "losvd_target", path, lineno)
            sigma = _karl_observables_parse_float(fields, columns, "losvd_sigma", path, lineno)
            supported_int = _karl_observables_parse_int(fields, columns, "losvd_supported", path, lineno)
            supported_int in (0, 1) || error("Karl observables losvd_supported=$supported_int is not 0 or 1 at line $lineno in $path")
            target >= 0.0 || error("Karl observables LOSVD target is negative at line $lineno in $path")
            supported = supported_int == 1
            if supported
                sigma >= 0.0 || error("Karl observables supported LOSVD bin has negative sigma=$sigma at line $lineno in $path")
            else
                sigma == DEFAULT_KARL_INVALID_SIGMA_SENTINEL || error("Karl observables unsupported LOSVD bin has sigma=$sigma rather than $DEFAULT_KARL_INVALID_SIGMA_SENTINEL at line $lineno in $path")
            end
            push!(losvd_rows, (aperture_id, velocity_bin, vlo, vcenter, vhi, target, sigma, supported, lineno))
        else
            error("Karl observables CSV $path has unknown row_type=$row_type at line $lineno")
        end
    end

    spatial_count == Ncell || error("Karl observables CSV $path has $spatial_count spatial_cell rows; expected $Ncell")
    all(spatial_seen) || error("Karl observables CSV $path does not contain every spatial cell")
    transfer_count == Ncell * Ncell || error("Karl observables CSV $path has $transfer_count seeing_transfer rows; expected $(Ncell * Ncell)")
    all(transfer_seen) || error("Karl observables CSV $path does not contain every seeing source/target pair")
    all(isfinite, radial_edges_arcsec) || error("Karl observables radial arcsec edges are incomplete in $path")
    all(isfinite, radial_edges_pc) || error("Karl observables radial pc edges are incomplete in $path")
    all(isfinite, angular_edges) || error("Karl observables angular edges are incomplete in $path")
    any(diff(radial_edges_arcsec) .<= 0.0) && error("Karl observables radial arcsec edges are not strictly increasing in $path")
    any(diff(radial_edges_pc) .<= 0.0) && error("Karl observables radial pc edges are not strictly increasing in $path")
    any(diff(angular_edges) .<= 0.0) && error("Karl observables angular edges are not strictly increasing in $path")

    raw_light_sum = sum(light_raw)
    seen_light_sum = sum(light_seen)
    isapprox(raw_light_sum, 1.0; rtol=1.0e-8, atol=1.0e-10) || error("Karl observables light_raw sum=$raw_light_sum rather than 1 in $path")
    isapprox(seen_light_sum, 1.0; rtol=1.0e-8, atol=1.0e-10) || error("Karl observables light_seen sum=$seen_light_sum rather than 1 in $path")

    spatial = sumb + sumbn
    source_retained = vec(sum(spatial, dims=2))
    all(x -> isfinite(x) && x > 0.0 && x <= 1.0 + 1.0e-10, source_retained) || error("Karl observables seeing-transfer source retention fractions are invalid in $path")

    Naperture = length(aperture_ids_raw)
    Naperture > 0 || error("Karl observables CSV $path contains no aperture rows")
    aperture_order = sortperm(aperture_ids_raw)
    aperture_ids = aperture_ids_raw[aperture_order]
    aperture_names = aperture_names_raw[aperture_order]
    aperture_binsets = aperture_binsets_raw[aperture_order]
    aperture_light = aperture_light_raw[aperture_order]
    star_count = star_count_raw[aperture_order]
    aperture_ids == collect(1:Naperture) || error("Karl observables aperture IDs must be contiguous 1:$Naperture in $path; got $aperture_ids")
    @inbounds for ia in 1:Naperture
        ir_start, ir_end, iv_start, iv_end = aperture_binsets[ia]
        expected_light = sum(@view light_seen[ir_start:ir_end, iv_start:iv_end])
        scale = max(abs(expected_light), abs(aperture_light[ia]), 1.0)
        abs(expected_light - aperture_light[ia]) <= 1.0e-10 * scale || error("Karl observables aperture $(aperture_ids[ia]) light=$(aperture_light[ia]) does not match summed light_seen=$expected_light in $path")
    end

    expected_losvd_rows = Naperture * nvel
    length(losvd_rows) == expected_losvd_rows || error("Karl observables CSV $path has $(length(losvd_rows)) losvd_bin rows; expected $expected_losvd_rows")
    sort!(losvd_rows; by=row -> (row[1], row[2]))
    velocity_edges_kms = fill(NaN, nvel + 1)
    velocity_centers_kms = fill(NaN, nvel)
    losvd_target = zeros(Float64, expected_losvd_rows)
    losvd_sigma = fill(DEFAULT_KARL_INVALID_SIGMA_SENTINEL, expected_losvd_rows)
    losvd_supported = falses(expected_losvd_rows)
    seen_losvd = falses(Naperture, nvel)

    @inbounds for row_data in losvd_rows
        aperture_id, velocity_bin, vlo, vcenter, vhi, target, sigma, supported, lineno = row_data
        1 <= aperture_id <= Naperture || error("Karl observables losvd_bin aperture_id=$aperture_id is outside 1:$Naperture at line $lineno in $path")
        1 <= velocity_bin <= nvel || error("Karl observables velocity_bin=$velocity_bin is outside 1:$nvel at line $lineno in $path")
        seen_losvd[aperture_id, velocity_bin] && error("Karl observables repeats LOSVD bin aperture=$aperture_id velocity_bin=$velocity_bin in $path")
        seen_losvd[aperture_id, velocity_bin] = true
        _karl_observables_set_edge!(velocity_edges_kms, velocity_bin, vlo, "velocity", path)
        _karl_observables_set_edge!(velocity_edges_kms, velocity_bin + 1, vhi, "velocity", path)
        if isnan(velocity_centers_kms[velocity_bin])
            velocity_centers_kms[velocity_bin] = vcenter
        else
            scale = max(abs(velocity_centers_kms[velocity_bin]), abs(vcenter), 1.0)
            abs(velocity_centers_kms[velocity_bin] - vcenter) <= 1.0e-10 * scale || error("Karl observables velocity center for bin $velocity_bin changes between apertures in $path")
        end
        row_index = (aperture_id - 1) * nvel + velocity_bin
        losvd_target[row_index] = target
        losvd_sigma[row_index] = sigma
        losvd_supported[row_index] = supported
    end
    all(seen_losvd) || error("Karl observables CSV $path does not contain every aperture/velocity LOSVD bin")
    any(diff(velocity_edges_kms) .<= 0.0) && error("Karl observables velocity edges are not strictly increasing in $path")
    @inbounds for j in 1:nvel
        expected_center = 0.5 * (velocity_edges_kms[j] + velocity_edges_kms[j + 1])
        scale = max(abs(expected_center), abs(velocity_centers_kms[j]), 1.0)
        abs(expected_center - velocity_centers_kms[j]) <= 1.0e-10 * scale || error("Karl observables velocity_center_kms is inconsistent with edges for bin $j in $path")
    end
    @inbounds for ia in 1:Naperture
        rows = ((ia - 1) * nvel + 1):(ia * nvel)
        supported_count = count(@view losvd_supported[rows])
        supported_count > 0 || error("Karl observables aperture $(aperture_ids[ia]) has no supported LOSVD bins in $path")
        target_sum = sum(@view losvd_target[rows])
        target_sum <= aperture_light[ia] + 1.0e-10 * max(aperture_light[ia], 1.0) || error("Karl observables aperture $(aperture_ids[ia]) LOSVD target sum=$target_sum exceeds aperture light=$(aperture_light[ia]) in $path")
    end

    radial_edges_m = radial_edges_pc .* pc
    velocity_edges_mps = velocity_edges_kms .* 1.0e3
    velocity_centers_mps = velocity_centers_kms .* 1.0e3
    observables = KarlObservables(String(path), schema_version, galaxy, distance_pc, axis_ratio_q, seeing_arcsec, nrdat, nvdat, nrlib, nvlib, nvel, irrat, ivrat, radial_edges_arcsec, radial_edges_m, angular_edges, light_raw, light_seen, sumb, sumbn, spatial, aperture_ids, aperture_names, aperture_binsets, aperture_light, star_count, velocity_edges_mps, velocity_centers_mps, losvd_target, losvd_sigma, losvd_supported, losvd_center_kms, losvd_shape, losvd_sigma_extent, mkherm_nsim, mkherm_econt_frac, mkherm_rng)
    supported_by_aperture = [count(@view losvd_supported[((ia - 1) * nvel + 1):(ia * nvel)]) for ia in 1:Naperture]
    println("[KARL OBSERVABLES LOAD]", " path=", path, " galaxy=", galaxy, " Nrlib=", nrlib, " Nvlib=", nvlib, " Naperture=", Naperture, " Nvel=", nvel, " seeing_arcsec=", seeing_arcsec, " supported_by_aperture=", supported_by_aperture, " retained_min=", minimum(source_retained), " retained_max=", maximum(source_retained))
    return observables
end

function load_karl_observables(path::AbstractString)
    key = abspath(String(path))
    lock(_KARL_OBSERVABLES_LOCK)
    try
        haskey(_KARL_OBSERVABLES_CACHE, key) && return _KARL_OBSERVABLES_CACHE[key]
    finally
        unlock(_KARL_OBSERVABLES_LOCK)
    end
    observables = _load_karl_observables_uncached(key)
    lock(_KARL_OBSERVABLES_LOCK)
    try
        return get!(_KARL_OBSERVABLES_CACHE, key, observables)
    finally
        unlock(_KARL_OBSERVABLES_LOCK)
    end
end

function karl_observables_aperture_ranges_m(observables::KarlObservables)
    ranges = Vector{Tuple{Float64,Float64}}(undef, length(observables.aperture_binsets))
    @inbounds for ia in eachindex(observables.aperture_binsets)
        ir_start, ir_end, _, _ = observables.aperture_binsets[ia]
        ranges[ia] = (observables.radial_edges_m[ir_start], observables.radial_edges_m[ir_end + 1])
    end
    return ranges
end

function karl_observables_targets(observables::KarlObservables)
    Naperture = length(observables.aperture_ids)
    supported_by_aperture = [count(@view observables.losvd_supported[((ia - 1) * observables.nvel + 1):(ia * observables.nvel)]) for ia in 1:Naperture]
    return (losvd_target=copy(observables.losvd_target), losvd_sigma=copy(observables.losvd_sigma), velocity_edges_mps=copy(observables.velocity_edges_mps), velocity_centers_mps=copy(observables.velocity_centers_mps), aperture_light=copy(observables.aperture_light), supported_by_aperture=supported_by_aperture)
end

@inline function karl_observables_projected_cell(observables::KarlObservables, Rproj_m::Float64, yproj_m::Float64)
    isfinite(Rproj_m) && Rproj_m >= 0.0 || return 0, 0
    isfinite(yproj_m) || return 0, 0
    Rproj_m <= observables.radial_edges_m[end] || return 0, 0
    ir = searchsortedlast(observables.radial_edges_m, Rproj_m)
    ir < 1 && return 0, 0
    ir >= length(observables.radial_edges_m) && (ir = observables.nrlib)
    spatial_v = Rproj_m > 0.0 ? clamp(abs(yproj_m) / Rproj_m, 0.0, 1.0) : 0.0
    iv = searchsortedlast(observables.angular_edges, spatial_v)
    iv < 1 && return 0, 0
    iv >= length(observables.angular_edges) && (iv = observables.nvlib)
    return ir, iv
end

function karl_observables_convolve_losvd(raw_losvd::Array{Float64,3}, observables::KarlObservables)
    size(raw_losvd) == (observables.nvel, observables.nrlib, observables.nvlib) || error("Karl observables raw LOSVD cube must have size ($(observables.nvel),$(observables.nrlib),$(observables.nvlib)); got $(size(raw_losvd))")
    all(isfinite, raw_losvd) || error("Karl observables raw LOSVD cube contains nonfinite values")
    all(x -> x >= 0.0, raw_losvd) || error("Karl observables raw LOSVD cube contains negative values")
    Ncell = observables.nrlib * observables.nvlib
    raw_flat = zeros(Float64, observables.nvel, Ncell)
    @inbounds for ir in 1:observables.nrlib
        for iv in 1:observables.nvlib
            cell = (ir - 1) * observables.nvlib + iv
            raw_flat[:, cell] .= @view raw_losvd[:, ir, iv]
        end
    end
    convolved_flat = raw_flat * observables.sumb + reverse(raw_flat; dims=1) * observables.sumbn
    convolved = zeros(Float64, observables.nvel, observables.nrlib, observables.nvlib)
    @inbounds for ir in 1:observables.nrlib
        for iv in 1:observables.nvlib
            cell = (ir - 1) * observables.nvlib + iv
            convolved[:, ir, iv] .= @view convolved_flat[:, cell]
        end
    end
    return convolved
end

function karl_observables_convolve_spatial(raw_spatial::Matrix{Float64}, observables::KarlObservables)
    size(raw_spatial) == (observables.nrlib, observables.nvlib) || error("Karl observables raw spatial grid must have size ($(observables.nrlib),$(observables.nvlib)); got $(size(raw_spatial))")
    all(isfinite, raw_spatial) || error("Karl observables raw spatial grid contains nonfinite values")
    all(x -> x >= 0.0, raw_spatial) || error("Karl observables raw spatial grid contains negative values")
    Ncell = observables.nrlib * observables.nvlib
    raw_flat = zeros(Float64, Ncell)
    @inbounds for ir in 1:observables.nrlib
        for iv in 1:observables.nvlib
            raw_flat[(ir - 1) * observables.nvlib + iv] = raw_spatial[ir, iv]
        end
    end
    convolved_flat = transpose(observables.spatial) * raw_flat
    convolved = zeros(Float64, observables.nrlib, observables.nvlib)
    @inbounds for ir in 1:observables.nrlib
        for iv in 1:observables.nvlib
            convolved[ir, iv] = convolved_flat[(ir - 1) * observables.nvlib + iv]
        end
    end
    return convolved
end

function karl_observables_collapse_losvd_apertures(convolved_losvd::Array{Float64,3}, observables::KarlObservables)
    size(convolved_losvd) == (observables.nvel, observables.nrlib, observables.nvlib) || error("Karl observables convolved LOSVD cube has unexpected dimensions")
    Naperture = length(observables.aperture_binsets)
    out = zeros(Float64, Naperture * observables.nvel)
    @inbounds for ia in 1:Naperture
        ir_start, ir_end, iv_start, iv_end = observables.aperture_binsets[ia]
        for ir in ir_start:ir_end
            for iv in iv_start:iv_end
                for ivel in 1:observables.nvel
                    out[(ia - 1) * observables.nvel + ivel] += convolved_losvd[ivel, ir, iv]
                end
            end
        end
    end
    return out
end

function karl_observables_collapse_spatial_apertures(convolved_spatial::Matrix{Float64}, observables::KarlObservables)
    size(convolved_spatial) == (observables.nrlib, observables.nvlib) || error("Karl observables convolved spatial grid has unexpected dimensions")
    aperture_total = zeros(Float64, length(observables.aperture_binsets))
    @inbounds for ia in eachindex(observables.aperture_binsets)
        ir_start, ir_end, iv_start, iv_end = observables.aperture_binsets[ia]
        for ir in ir_start:ir_end
            for iv in iv_start:iv_end
                aperture_total[ia] += convolved_spatial[ir, iv]
            end
        end
    end
    return aperture_total
end

function karl_observables_seeing_response(raw_losvd::Array{Float64,3}, raw_spatial::Matrix{Float64}, observables::KarlObservables)
    convolved_losvd = karl_observables_convolve_losvd(raw_losvd, observables)
    convolved_spatial = karl_observables_convolve_spatial(raw_spatial, observables)
    losvd = karl_observables_collapse_losvd_apertures(convolved_losvd, observables)
    kinematic = karl_observables_collapse_spatial_apertures(convolved_spatial, observables)
    return (losvd=losvd, kinematic=kinematic)
end

function _karl_rebin_transvd_to_model(vfine_kms::Vector{Float64}, central::Vector{Float64}, lower::Vector{Float64}, upper::Vector{Float64}, velocity_edges_mps::Vector{Float64}, aperture_light::Float64; invalid_sigma_sentinel::Float64=DEFAULT_KARL_INVALID_SIGMA_SENTINEL)
    length(vfine_kms) == length(central) == length(lower) == length(upper) || error("Karl model rebin arrays have inconsistent lengths")
    ndata = length(vfine_kms)
    ndata >= 2 || error("Karl model rebin requires at least two fine LOSVD samples")
    Nvbin = length(velocity_edges_mps) - 1
    Nvbin > 0 || error("Karl model rebin velocity_edges must contain at least two values")
    any(diff(velocity_edges_mps) .<= 0.0) && error("Karl model rebin velocity_edges must be strictly increasing")
    isfinite(aperture_light) && aperture_light >= 0.0 || error("Karl model rebin aperture light must be finite and nonnegative")

    suma = sum(central)
    isfinite(suma) && suma > 0.0 || error("Karl vdataread-equivalent LOSVD normalization is non-positive")
    ad = central .* (aperture_light / suma)
    adfer = Vector{Float64}(undef, ndata)
    @inbounds for k in 1:ndata
        adfer[k] = max(upper[k] - central[k], (upper[k] - lower[k]) / 2.0) * aperture_light / suma
    end

    vdata = vfine_kms .* 1.0e3
    v1 = Vector{Float64}(undef, ndata)
    v2 = Vector{Float64}(undef, ndata)
    v1[1] = vdata[1] - (vdata[2] - vdata[1]) / 2.0
    v2[1] = vdata[1] + (vdata[2] - vdata[1]) / 2.0
    v1[end] = vdata[end] - (vdata[end] - vdata[end - 1]) / 2.0
    v2[end] = vdata[end] + (vdata[end] - vdata[end - 1]) / 2.0
    @inbounds for k in 2:(ndata - 1)
        v1[k] = vdata[k] - (vdata[k] - vdata[k - 1]) / 2.0
        v2[k] = vdata[k] + (vdata[k + 1] - vdata[k]) / 2.0
    end

    target = zeros(Float64, Nvbin)
    sigma = fill(invalid_sigma_sentinel, Nvbin)

    @inbounds for j in 1:Nvbin
        vlo = velocity_edges_mps[j]
        vhi = velocity_edges_mps[j + 1]
        vcenter = 0.5 * (vlo + vhi)
        if vcenter < v1[1] || vcenter > v2[end]
            target[j] = 0.0
            sigma[j] = invalid_sigma_sentinel
            continue
        end

        sum_target = 0.0
        sum_sigma = 0.0
        for k in 1:ndata
            data_width = v2[k] - v1[k]
            data_width > 0.0 || continue
            overlap = min(vhi, v2[k]) - max(vlo, v1[k])
            if overlap > 0.0
                frac = min(1.0, overlap / data_width)
                sum_target += frac * ad[k]
                sum_sigma += frac * adfer[k]
            end
        end
        target[j] = sum_target
        sigma[j] = sum_sigma
    end

    return target, sigma
end

@inline function _karl_poisson_upper_sigma(n::Int)
    n >= 0 || error("Karl Poisson count must be nonnegative")
    return 1.0 + sqrt(f64(n) + 0.75)
end

function _observed_targets_karl_resolved(R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, kinematic_edges::Vector{Float64}, velocity_edges::Vector{Float64}; surface_brightness_profile=nothing, light_edges=nothing, sigma_floor::Float64=1e-8, kde_grid::Int=DEFAULT_KARL_RESOLVED_KDE_GRID, kde_width_bins::Float64=DEFAULT_KARL_RESOLVED_KDE_WIDTH_BINS, kde_vmin_kms::Float64=DEFAULT_KARL_RESOLVED_VMIN_KMS, kde_vmax_kms::Float64=DEFAULT_KARL_RESOLVED_VMAX_KMS, bootstraps::Int=DEFAULT_KARL_RESOLVED_BOOTSTRAPS, envelope_floor::Float64=DEFAULT_KARL_RESOLVED_ENVELOPE_FLOOR)
    kinematic_edges = resolve_karl_spatial_edges(kinematic_edges)
    light_edges_use = light_edges === nothing ? kinematic_edges : resolve_karl_light_edges(light_edges)
    velocity_edges = Float64.(velocity_edges)
    any(diff(velocity_edges) .<= 0.0) && error("Karl resolved velocity_edges must be strictly increasing")
    Nspatial = length(kinematic_edges) - 1
    Nvbin = length(velocity_edges) - 1
    Nlosvd = Nspatial * Nvbin
    counts_losvd = zeros(Int, Nspatial, Nvbin)
    counts_by_spatial = zeros(Float64, Nspatial)

    @inbounds for i in eachindex(valid_vlos)
        valid_vlos[i] || continue
        isfinite(R_star_m[i]) || continue
        isfinite(v_star_mps[i]) || continue
        ib = _bin_index(kinematic_edges, R_star_m[i])
        ib == 0 && continue
        jb = _bin_index(velocity_edges, v_star_mps[i])
        jb == 0 && error("Karl resolved stellar velocity $(v_star_mps[i] / 1.0e3) km/s in radial aperture $ib lies outside configured LOSVD velocity support [$(velocity_edges[1] / 1.0e3), $(velocity_edges[end] / 1.0e3)] km/s")
        counts_losvd[ib, jb] += 1
        counts_by_spatial[ib] += 1.0
    end

    light_target = light_target_from_surface_brightness(surface_brightness_profile, light_edges_use; normalize=true)
    light_sigma = light_sigma_from_surface_brightness(surface_brightness_profile, light_edges_use; normalize=true, sigma_floor=sigma_floor)
    length(light_sigma) == length(light_target) || error("light_sigma length does not match light_target")
    losvd_light_target = light_target_from_surface_brightness(surface_brightness_profile, kinematic_edges; normalize=false)
    losvd_target = zeros(Float64, Nlosvd)
    losvd_sigma = fill(DEFAULT_KARL_INVALID_SIGMA_SENTINEL, Nlosvd)

    @inbounds for ib in 1:Nspatial
        nbin = Int(round(counts_by_spatial[ib]))
        nbin == 0 && continue
        counted = sum(@view counts_losvd[ib, :])
        counted == nbin || error("Karl resolved LOSVD count mismatch in radial aperture $ib: velocity bins contain $counted stars but radial aperture contains $nbin")
        Li = max(losvd_light_target[ib], 0.0)

        for jb in 1:Nvbin
            row = (ib - 1) * Nvbin + jb
            nij = counts_losvd[ib, jb]
            losvd_target[row] = Li * nij / nbin
            losvd_sigma[row] = max(Li * _karl_poisson_upper_sigma(nij) / nbin, sigma_floor)
        end

        rows = ((ib - 1) * Nvbin + 1):(ib * Nvbin)
        target_sum = sum(@view losvd_target[rows])
        scale = max(abs(target_sum), abs(Li), 1.0)
        abs(target_sum - Li) <= 1.0e-12 * scale || error("Karl resolved LOSVD target does not recover aperture light in radial aperture $ib")
    end

    println("[KARL RESOLVED LOSVD TARGET]",
        " Nspatial=", Nspatial,
        " Nvbin=", Nvbin,
        " profile=hard_velocity_counts",
        " uncertainty=gehrels_upper_1sigma",
        " zero_count_sigma_counts=", _karl_poisson_upper_sigma(0),
        " zero_velocity_bins=", count(==(0), counts_losvd),
        " low_count_velocity_bins=", count(n -> 0 < n < 5, counts_losvd),
        " valid_losvd_bins=", count(!=(DEFAULT_KARL_INVALID_SIGMA_SENTINEL), losvd_sigma),
    )

    return losvd_target, losvd_sigma, light_target, light_sigma, counts_by_spatial
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

function light_sigma_from_surface_brightness(profile, spatial_edges_m::Vector{Float64}; normalize::Bool=true, sigma_floor::Float64=1e-12)
    profile === nothing && error("surface_brightness_profile is required for Karl-style light uncertainties")

    p = normalize_surface_brightness_profile(profile)
    Nspatial = length(spatial_edges_m) - 1
    Nspatial > 0 || error("surface_brightness_profile cannot be binned because spatial_edges has fewer than two edges")

    haskey(p, :Sigma) || error("surface_brightness_profile must include Sigma for light uncertainties")
    haskey(p, :Sigma_err) || error("surface_brightness_profile must include Sigma_err for light uncertainties")

    Sigma = Float64.(p[:Sigma])
    Sigma_err = Float64.(p[:Sigma_err])

    length(Sigma) == length(Sigma_err) ||
        error("surface_brightness_profile Sigma and Sigma_err lengths do not match")

    all(isfinite, Sigma) || error("surface_brightness_profile Sigma contains nonfinite values")
    all(isfinite, Sigma_err) || error("surface_brightness_profile Sigma_err contains nonfinite values")
    all(x -> x >= 0.0, Sigma_err) || error("surface_brightness_profile Sigma_err contains negative values")

    # Preferred current OSPM path: light_frac is already the authoritative
    # binned light product. Carry the measured fractional surface-density
    # uncertainty onto that same light fraction:
    #
    #     sigma(L_i) / L_i = sigma(Sigma_i) / Sigma_i
    #
    # Rebin independent source-bin uncertainties in quadrature.
    if haskey(p, :light_frac)
        t = Float64.(p[:light_frac])

        length(t) == length(Sigma) ||
            error("surface_brightness_profile light_frac and Sigma lengths do not match")

        source_sigma = zeros(Float64, length(t))

        @inbounds for k in eachindex(t)
            lk = t[k]

            isfinite(lk) && lk >= 0.0 ||
                error("surface_brightness_profile contains invalid light_frac at index $k")

            if lk == 0.0
                source_sigma[k] = 0.0
            else
                Sigma[k] > 0.0 ||
                    error("cannot propagate light uncertainty from non-positive Sigma at index $k")

                source_sigma[k] = lk * Sigma_err[k] / Sigma[k]
            end
        end

        out_light = zeros(Float64, Nspatial)
        out_var = zeros(Float64, Nspatial)

        if length(t) == Nspatial
            copyto!(out_light, t)

            @inbounds for ib in 1:Nspatial
                out_var[ib] = source_sigma[ib]^2
            end

        elseif haskey(p, :R_inner_pc) && haskey(p, :R_outer_pc)
            rin = Float64.(p[:R_inner_pc]) .* pc
            rout = Float64.(p[:R_outer_pc]) .* pc

            length(rin) == length(rout) == length(t) ||
                error("surface_brightness_profile binned radius arrays do not match light_frac length")

            @inbounds for k in eachindex(t)
                if !(isfinite(rin[k]) && isfinite(rout[k]) && rout[k] > rin[k])
                    continue
                end

                src_area = rout[k]^2 - rin[k]^2
                src_area > 0.0 || continue

                for ib in 1:Nspatial
                    lo = max(rin[k], spatial_edges_m[ib])
                    hi = min(rout[k], spatial_edges_m[ib + 1])

                    if hi > lo
                        frac = (hi^2 - lo^2) / src_area

                        out_light[ib] += t[k] * frac
                        out_var[ib] += (source_sigma[k] * frac)^2
                    end
                end
            end

        else
            error("surface_brightness_profile light_frac length $(length(t)) does not match Nspatial=$Nspatial and no binned radii are available for rebinning")
        end

        out_sigma = sqrt.(out_var)

        if normalize
            light_sum = sum(out_light)
            isfinite(light_sum) && light_sum > 0.0 ||
                error("surface_brightness_profile light uncertainty normalization has non-positive total light")

            out_sigma ./= light_sum
        end

        @inbounds for ib in eachindex(out_sigma)
            out_sigma[ib] = max(out_sigma[ib], sigma_floor)
        end

        return out_sigma
    end

    # Fallback matching the existing unbinned R_pc + Sigma target path.
    haskey(p, :R_pc) || error("surface_brightness_profile must include light_frac or R_pc + Sigma")
    R_m = Float64.(p[:R_pc]) .* pc
    length(R_m) == length(Sigma) || error("surface_brightness_profile R_pc and Sigma lengths do not match")
    raw_light = zeros(Float64, Nspatial)
    raw_var = zeros(Float64, Nspatial)
    @inbounds for k in eachindex(R_m)
        ib = _bin_index(spatial_edges_m, R_m[k])
        if ib > 0 && isfinite(Sigma[k]) && Sigma[k] >= 0.0
            raw_light[ib] += Sigma[k]
            raw_var[ib] += Sigma_err[k]^2
        end
    end
    out_sigma = sqrt.(raw_var)
    if normalize
        light_sum = sum(raw_light)
        isfinite(light_sum) && light_sum > 0.0 || error("surface_brightness_profile produced zero light while normalizing uncertainties")
        out_sigma ./= light_sum
    end
    @inbounds for ib in eachindex(out_sigma)
        out_sigma[ib] = max(out_sigma[ib], sigma_floor)
    end
    return out_sigma
end

function observed_targets_karl(R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, kinematic_edges::Vector{Float64}, velocity_edges::Vector{Float64}; surface_brightness_profile=nothing, light_edges=nothing, sigma_floor::Float64=1e-8, target_mode=:current, karl_resolved_kde_grid::Int=DEFAULT_KARL_RESOLVED_KDE_GRID, karl_resolved_kde_width_bins::Float64=DEFAULT_KARL_RESOLVED_KDE_WIDTH_BINS, karl_resolved_vmin_kms::Float64=DEFAULT_KARL_RESOLVED_VMIN_KMS, karl_resolved_vmax_kms::Float64=DEFAULT_KARL_RESOLVED_VMAX_KMS, karl_resolved_bootstraps::Int=DEFAULT_KARL_RESOLVED_BOOTSTRAPS, karl_resolved_envelope_floor::Float64=DEFAULT_KARL_RESOLVED_ENVELOPE_FLOOR)
    mode = target_mode isa Symbol ? target_mode : Symbol(lowercase(String(target_mode)))
    if mode === :karl_resolved_stars
        return _observed_targets_karl_resolved(R_star_m, valid_vlos, v_star_mps, kinematic_edges, velocity_edges; surface_brightness_profile=surface_brightness_profile, light_edges=light_edges, sigma_floor=sigma_floor, kde_grid=karl_resolved_kde_grid, kde_width_bins=karl_resolved_kde_width_bins, kde_vmin_kms=karl_resolved_vmin_kms, kde_vmax_kms=karl_resolved_vmax_kms, bootstraps=karl_resolved_bootstraps, envelope_floor=karl_resolved_envelope_floor)
    elseif mode !== :current
        error("Unsupported LOSVD target_mode=$target_mode; expected :current or :karl_resolved_stars")
    end
    kinematic_edges = resolve_karl_spatial_edges(kinematic_edges)
    light_edges_use = light_edges === nothing ? kinematic_edges : resolve_karl_light_edges(light_edges)
    velocity_edges = Float64.(velocity_edges)
    vlos_idx = Int[]
    @inbounds for i in eachindex(valid_vlos)
        valid_vlos[i] && isfinite(R_star_m[i]) && isfinite(v_star_mps[i]) && isfinite(verr_star_mps[i]) && verr_star_mps[i] > 0.0 && push!(vlos_idx, i)
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
            psum += _gaussian_bin_probability(velocity_edges[jb], velocity_edges[jb + 1], v0, sig)
        end
        if psum > 0.0
            for jb in 1:Nvbin
                row = (ib - 1) * Nvbin + jb
                p = _gaussian_bin_probability(velocity_edges[jb], velocity_edges[jb + 1], v0, sig) / psum
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
    light_target = light_target_from_surface_brightness(surface_brightness_profile, light_edges_use; normalize=true)
    light_sigma = light_sigma_from_surface_brightness(surface_brightness_profile, light_edges_use; normalize=true, sigma_floor=sigma_floor)
    length(light_sigma) == length(light_target) || error("light_sigma length does not match light_target")
    losvd_light_target = light_target_from_surface_brightness(surface_brightness_profile, kinematic_edges; normalize=false)
    losvd_target = zeros(Float64, Nlosvd)
    @inbounds for ib in 1:Nspatial
        nbin = counts_by_spatial[ib]
        nbin <= 0.0 && continue

        for jb in 1:Nvbin
            row = (ib - 1) * Nvbin + jb
            losvd_target[row] = losvd_light_target[ib] * counts_losvd[row] / nbin
        end
    end

    # The LOSVD target in each row is y_ij = L_i * p_ij. The light profile fixes
    # L_i structurally. Only the finite kinematic sample sets a statistical sigma.
    losvd_sigma = similar(losvd_target)
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
            losvd_sigma[row] = max(Li * sqrt(max(var_pij, 0.0)), sigma_floor)
        end
    end

    return losvd_target, losvd_sigma, light_target, light_sigma, counts_by_spatial
end

# ========================================================================================================================
# §4  KARL WEIGHT / SPEAR SOLVER
# ========================================================================================================================
# The entropy, wphase, expanded Cm, LOSVD slack-variable SPEAR solve,
# xmu helpers, and χ² scoring live in OSPM_Physics_Weights.jl.
include("OSPM_Physics_Weights.jl")
include("OSPM_Physics_Force.jl")

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

function _karl_outer_zero_velocity_radius(; energy::Float64, lz::Float64, theta0::Float64, rapo_max::Float64, pot, nscan::Int=256, max_expand::Int=32, expand_factor::Float64=1.5)
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

function _karl_family_zvc_launches(; energy::Float64, lz::Float64, rapo::Float64, pot, Nlaunch::Int, circular_boundary::Bool=false, theta_floor::Float64=1.0e-4, ncurve::Int=0)
    Nlaunch > 0 || error("Nlaunch must be positive")
    isfinite(theta_floor) && 0.0 < theta_floor < pi / 2 || error("theta_floor must lie strictly between zero and pi/2")
    isfinite(energy) && isfinite(lz) && isfinite(rapo) && rapo > 0.0 || return (state=:reject_family_integrals, u=Float64[], r=Float64[], theta=Float64[], R=Float64[], z=Float64[], arc_length=NaN)
    theta_equator = f64(pi / 2)
    if circular_boundary
        r_equator, state = _karl_outer_zero_velocity_radius(energy=energy, lz=lz, theta0=theta_equator, rapo_max=rapo, pot=pot)
        state == :ok || return (state=state, u=Float64[], r=Float64[], theta=Float64[], R=Float64[], z=Float64[], arc_length=NaN)
        return (state=:ok, u=[1.0], r=[r_equator], theta=[theta_equator], R=[r_equator], z=[0.0], arc_length=0.0)
    end
    ncurve_use = ncurve > 0 ? ncurve : max(257, 32 * Nlaunch)
    ncurve_use >= Nlaunch || error("ncurve must be at least as large as Nlaunch")
    theta_scan = collect(range(theta_floor, theta_equator; length=ncurve_use))
    radius_scan = fill(NaN, ncurve_use)
    valid_scan = falses(ncurve_use)
    @inbounds for i in eachindex(theta_scan)
        radius, state = _karl_outer_zero_velocity_radius(energy=energy, lz=lz, theta0=theta_scan[i], rapo_max=rapo, pot=pot)
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
            radius_mid, state_mid = _karl_outer_zero_velocity_radius(energy=energy, lz=lz, theta0=theta_mid, rapo_max=rapo, pot=pot)
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
        radius_target, state_target = _karl_outer_zero_velocity_radius(energy=energy, lz=lz, theta0=theta_target, rapo_max=rapo, pot=pot)
        state_target == :ok || return (state=state_target, u=Float64[], r=Float64[], theta=Float64[], R=Float64[], z=Float64[], arc_length=total_arc)
        st, ct = sincos(theta_target)
        r_launch[i] = radius_target
        theta_launch[i] = theta_target
        R_launch[i] = radius_target * st
        z_launch[i] = radius_target * ct
    end
    return (state=:ok, u=u_launch, r=r_launch, theta=theta_launch, R=R_launch, z=z_launch, arc_length=total_arc)
end

function karl_orbit_family_integrals(; rapo::Float64, Lz_frac::Float64, pot, frc, debug::Bool=true)
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
            radius, turning_state = _karl_outer_zero_velocity_radius(energy=E, lz=Lz, theta0=theta0, rapo_max=rapo, pot=pot)
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
