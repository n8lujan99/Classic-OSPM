# ========================================================================================================================
# OSPM_Physics_Support.jl — shared physics support definitions.
# Included by OSPM_Physics_Spherical.jl — do NOT load directly.
# Owns shared constants, data contracts, caches, and small helpers used across
# the force, observable, weight, phase-volume, and spherical-orbit machinery.
# Contains no orbit integration, observable construction, force solver, or weight solver.
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

# -- OSPM-style binned LOSVD / projected-light fit --
const DEFAULT_MIN_STARS_PER_BIN = 20       # minimum stars per projected radial bin
const DEFAULT_NVBIN             = 21       # requested LOSVD resolution; auto support may add bins at the same width
const DEFAULT_LOSVD_MIN_HALF_WIDTH_KMS = 100.0
const DEFAULT_LOSVD_MAX_OUTSIDE_FRACTION = 0.01       # legacy :current-mode hard-rejection threshold
const DEFAULT_RESOLVED_SELECTION_WARN_FRACTION = 0.01  # diagnostic only for selected hard-count LOSVDs
const DEFAULT_LOSVD_CONDITIONING = :vlos_cut
const DEFAULT_LOSVD_FIT_STATISTIC = :legacy_chi2
const DEFAULT_RESOLVED_KDE_GRID = 17
const DEFAULT_RESOLVED_KDE_WIDTH_BINS = 3.0
const DEFAULT_RESOLVED_VMIN_KMS = -25.0
const DEFAULT_RESOLVED_VMAX_KMS = 25.0
const DEFAULT_RESOLVED_BOOTSTRAPS = 300
const DEFAULT_RESOLVED_TRANSVD_SAMPLES = 1000
const DEFAULT_RESOLVED_ENVELOPE_FLOOR = 0.003

# -- Consolidated OSPM observables CSV runtime --
const DEFAULT_OBSERVABLES_SCHEMA_VERSION = 1
const DEFAULT_ALPHA        = 1e-4     # legacy value retained only for call compatibility
const DEFAULT_ALPHAT       = 1.0      # OSPM-style data-mismatch multiplier in entropy mode

const DEFAULT_MAXITER       = 250       # OSPM SPEAR/Newton iteration cap
const DEFAULT_ENTROPY_FLOOR = 1e-30    # initialization/numerical floor only; not the positivity boundary
const DEFAULT_FILTER_TINY   = 1e-37    # OSPM filter.f physical orbit-weight floor after each SPEAR step
const DEFAULT_APFAC         = 0.01      # maximum requested SPEAR step
const DEFAULT_LIGHT_REL_TOL = 0.01
const DEFAULT_DELTA_CHI2_ITER_TOL = 0.3
const DEFAULT_INVALID_SIGMA_SENTINEL = -666.0
const DEFAULT_STEP_SAFETY = 0.90        # take 90% of zero-weight boundary when step-limited
const DEFAULT_SPEAR_RCOND_WARN = 1.0e-12

# ========================================================================================================================
# §2  TYPES, CACHES, INLINE HELPERS
# ========================================================================================================================
@inline f64(x)=Float64(x)
@inline safe_sign(x)=x>0 ? 1.0 : (x<0 ? -1.0 : 0.0)
@inline _ssin(theta::Float64)=begin s=sin(theta); abs(s)>1e-12 ? s : safe_sign(s)*1e-12 end
@inline function _sincos_safe(theta::Float64); s,cc=sincos(theta); abs(s)>1e-12 ? (s,cc) : (safe_sign(s)*1e-12,cc) end
@inline clamp01(x::Float64)=x<0 ? 0.0 : (x>1 ? 1.0 : x)

@inline function _normalize_losvd_conditioning(mode)
    mode_sym = mode === nothing ? DEFAULT_LOSVD_CONDITIONING : Symbol(lowercase(String(mode)))
    mode_sym in (:vlos_cut, :none) || error("Unknown losvd_conditioning=$(mode). Use vlos_cut or none")
    return mode_sym
end

@inline function _normalize_losvd_fit_statistic(mode, target_mode::Symbol)
    mode_sym = mode === nothing ? (target_mode === :resolved_stars ? DEFAULT_LOSVD_FIT_STATISTIC : :legacy_chi2) : Symbol(lowercase(String(mode)))
    mode_sym in (:multinomial, :legacy_chi2) || error("Unknown losvd_fit_statistic=$(mode). Use multinomial or legacy_chi2")
    mode_sym === :multinomial && target_mode !== :resolved_stars && error("losvd_fit_statistic=:multinomial requires losvd_target_mode=:resolved_stars")
    return mode_sym
end

struct HaloContext
    halo::Dict{Symbol,Any}
    R::Vector{Float64}
    tabv::Vector{Float64}
    tabfr::Vector{Float64}
    Menc::Vector{Float64}
    pot::Function
    frc::Function
end

struct Observables
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

struct D3TracerConstraints
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

const _OBSERVABLES_CACHE = Dict{String,Observables}()
const _OBSERVABLES_LOCK = ReentrantLock()
const _D3_TRACER_CACHE = Dict{Tuple{String,UInt64,UInt64},D3TracerConstraints}()
const _D3_TRACER_LOCK = ReentrantLock()

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
