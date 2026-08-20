# ========================================================================================================================
# OSPM_Physics_Support.jl — Karl-style support layer.
# Included by OSPM_Physics_Spherical.jl — do NOT load directly.
# Contains only the shared support shell and Karl-style observable machinery:
# constants, halo context construction, radial/velocity binning,
# surface-brightness targets, binned LOSVD targets, and includes for the
# weight/SPEAR and force machinery.
# Applied new karl fixes on 04/06/26 @1600
# Legacy star-level likelihood code and old back-compat sigma2 paths removed.
# NOTES ON THIS FILE:
# This is the shared Julia support layer used by OSPM_Physics_Spherical.jl.
#
# It has four main jobs:
#   1. define physical constants and shared numerical defaults,
#   2. turn the observed light and stellar velocities into model targets,
#   3. provide generic force/orbit helpers,
#   4. launch and integrate orbit families.
#
# The actual weight/SPEAR solver lives in OSPM_Physics_Weights.jl.
# The halo and stellar-force construction lives in OSPM_Physics_Force.jl.
# ========================================================================================================================
# §1  CONSTANTS
# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ---------------------------------------------------------------------------
#
# NTHREADS:
#   Number of Julia threads available to this process.
#
# G:
#   Newton's gravitational constant in SI units.
#
# c:
#   Speed of light in m/s. Kept here as a shared physical constant even though
#   the functions in this file do not directly use it.
#
# pc and Msun:
#   Conversion factors from parsecs and solar masses into SI units.
#
# IMPORTANT:
# Much of the outer configuration is expressed in pc, km/s, and Msun. The Julia
# force/orbit machinery commonly converts those values into meters, m/s, and kg.
const NTHREADS = Threads.nthreads()
const G    = 6.67430e-11
const c    = 2.99792458e8
const pc   = 3.0856775814913673e16
const Msun = 1.98847e30

# machine floors
# ---------------------------------------------------------------------------
# MACHINE / GEOMETRY SAFETY FLOORS
# ---------------------------------------------------------------------------
#
# These are numerical guards. They are not physical parameters of a galaxy.
#
# EPS_FORCE:
#   Tiny force scale used to avoid classifying roundoff-level values as a
#   meaningful outward force.
#
# EPS_VEL:
#   Tiny positive velocity scale used when checking circular-speed calculations.
#
# EPS_ARG:
#   Tolerance for expressions that should physically be non-negative before
#   taking a square root.
#
# EPS_SIN:
#   Rejects launch angles too close to the symmetry axis when expressions contain
#   1/sin(theta). This avoids singular cylindrical/angular-momentum terms.
const EPS_FORCE = 1e-14
const EPS_VEL   = 1e-14
const EPS_ARG   = 1e-14

# physical geometry gate
const EPS_SIN = 1e-6

# scale-aware force gate
const REL_FORCE    = 1e-10   # loosen to 1e-9 if needed
const BRACKET_FRAC = 1e-6    # MUST be >> eps(Float64)
# ---------------------------------------------------------------------------
# FORCE-SCALE / ROOT-BRACKETING CONTROLS
# ---------------------------------------------------------------------------
#
# REL_FORCE:
#   Scale-aware tolerance used when deciding whether the local force has the
#   expected inward sign.
#
# BRACKET_FRAC:
#   Small fractional displacement around a reference radius when sampling the
#   force on both sides of that point.
#
# IMPORTANT:
# BRACKET_FRAC must remain much larger than floating-point epsilon so the two
# sample radii are numerically distinct.

# TUNABLE KNOBS — adjust these to control resolution, accuracy, and parallelism.
# -- Halo potential grid --
const DEFAULT_NR              = 256       # radial grid points for potential table
const DEFAULT_RMAX_FACTOR     = 300.0     # max radius in units of r_s
# ---------------------------------------------------------------------------
# SHARED TUNABLE DEFAULTS
# ---------------------------------------------------------------------------
#
# These are generic defaults. Galaxy configs or higher-level calls may override
# some of them.
#
# Halo grid:
#   DEFAULT_NR          = number of radial points in the spherical force/potential table.
#   DEFAULT_RMAX_FACTOR = how far the default halo table extends in units of its scale radius.
#
# Orbit integration:
#   DEFAULT_NSTEPS           = requested top-level RK4 steps.
#   DEFAULT_STOP_RMIN_FACTOR = safety factor outside the innermost halo radius.
#   DEFAULT_DT_FRAC          = converts a local orbital frequency into a timestep.
#   DEFAULT_DT_FLOOR         = denominator floor for that timestep calculation.
#   DEFAULT_R0_FRAC          = non-circular launches begin just inside the turning radius.
#
# Orbit-library/A-matrix:
#   DEFAULT_LFRAC          = default fractions of circular angular momentum.
#   DEFAULT_DR_FRAC        = radial matching tolerance.
#   DEFAULT_NBINS_OCC      = occupancy-histogram resolution.
#   DEFAULT_MAX_ATTEMPTS   = launch-attempt multiplier.
#   DEFAULT_DR_FLOOR_FRAC  = fractional radial matching floor.
#   DEFAULT_DR_FLOOR_PC    = absolute parsec floor on radial matching.
#
# Karl observable/weight defaults:
#   DEFAULT_MIN_STARS_PER_BIN            = adaptive-bin helper target only.
#   DEFAULT_NVBIN                        = requested LOSVD velocity resolution.
#   DEFAULT_LOSVD_MIN_HALF_WIDTH_KMS     = minimum LOSVD half-window.
#   DEFAULT_LOSVD_MAX_OUTSIDE_FRACTION   = accepted outside-window fraction used elsewhere.
#   DEFAULT_KARL_ALPHAT                  = LOSVD mismatch strength in the entropy objective.
#   DEFAULT_KARL_MAXITER                 = weight-solver iteration cap.
#   DEFAULT_KARL_ENTROPY_FLOOR           = tiny positive numerical weight floor.
#   DEFAULT_KARL_APFAC                   = maximum requested SPEAR correction step.
#   DEFAULT_KARL_LIGHT_REL_TOL           = hard relative tracer/light tolerance.
#   DEFAULT_KARL_DELTA_CHI2_ITER_TOL     = LOSVD chi2 iteration-stability tolerance.
#   DEFAULT_KARL_INVALID_SIGMA_SENTINEL  = marks LOSVD cells that should not carry normal chi2 weight.
#   DEFAULT_KARL_STEP_SAFETY             = safety fraction for a boundary-limited step helper.
#   DEFAULT_KARL_SPEAR_RCOND_WARN        = warns when the SPEAR system is nearly singular.
#
# NOTE:
# DEFAULT_KARL_ALPHA is retained only so older call sites still parse. The active
# entropy-mode mismatch multiplier is DEFAULT_KARL_ALPHAT.

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
const DEFAULT_KARL_ALPHA        = 1e-4     # legacy value retained only for call compatibility
const DEFAULT_KARL_ALPHAT       = 1.0      # Karl-style data-mismatch multiplier in entropy mode

const DEFAULT_KARL_MAXITER       = 250       # Karl SPEAR/Newton iteration cap
const DEFAULT_KARL_ENTROPY_FLOOR = 1e-30    # initialization/numerical floor only; not the positivity boundary
const DEFAULT_KARL_APFAC         = 0.01      # maximum requested SPEAR step
const DEFAULT_KARL_LIGHT_REL_TOL = 0.01
const DEFAULT_KARL_DELTA_CHI2_ITER_TOL = 0.3
const DEFAULT_KARL_INVALID_SIGMA_SENTINEL = -666.0
const DEFAULT_KARL_STEP_SAFETY = 0.90        # take 90% of zero-weight boundary when step-limited
const DEFAULT_KARL_SPEAR_RCOND_WARN = 1.0e-12



# ========================================================================================================================
# §2  TYPES, CACHES, INLINE HELPERS
# ========================================================================================================================
# ---------------------------------------------------------------------------
# VERY SMALL INLINE HELPERS
# ---------------------------------------------------------------------------
#
# f64:
#   Force a value into Float64.
#
# safe_sign:
#   Return +1, -1, or 0 without relying on a branch elsewhere.
#
# _ssin:
#   Safe sin(theta) used when angular-momentum formulas would divide by sin(theta).
#
# _sincos_safe:
#   Return sin/cos together while preventing an exactly tiny sin(theta) from entering
#   a later denominator.
#
# clamp01:
#   Clamp a Float64 into the physical fraction interval [0,1].
@inline f64(x)=Float64(x)
@inline safe_sign(x)=x>0 ? 1.0 : (x<0 ? -1.0 : 0.0)
@inline _ssin(theta::Float64)=begin s=sin(theta); abs(s)>1e-12 ? s : safe_sign(s)*1e-12 end
# WHAT:
# Returns sin(theta) and cos(theta) together, with a tiny floor on sin(theta).
#
# WHY:
# Cylindrical/spherical conversions and Lz terms can contain division by sin(theta).
# Near the symmetry axis, an exactly tiny sine would make those expressions unstable.
#
# NOTE:
# Only the sine is protected. The cosine is returned unchanged.
@inline function _sincos_safe(theta::Float64); s,cc=sincos(theta); abs(s)>1e-12 ? (s,cc) : (safe_sign(s)*1e-12,cc) end
@inline clamp01(x::Float64)=x<0 ? 0.0 : (x>1 ? 1.0 : x)

# WHAT:
# Compact container for one fully constructed halo/potential context.
#
# CONTENTS:
#   halo  = normalized parameter dictionary
#   R     = radial force/potential grid
#   tabv  = spherical halo potential table
#   tabfr = spherical halo radial-force table
#   Menc  = spherical enclosed-mass table
#   pot   = callable total-potential function
#   frc   = callable total-force function
#
# WHY:
# Orbit integration needs the same potential and force repeatedly. Keeping the
# precomputed tables and closures together avoids rebuilding them for each force call.
struct HaloContext
    halo::Dict{Symbol,Any}
    R::Vector{Float64}
    tabv::Vector{Float64}
    tabfr::Vector{Float64}
    Menc::Vector{Float64}
    pot::Function
    frc::Function
end
# HALO CONTEXT CACHE:
# Stores already-built HaloContext objects keyed by the physical model parameters,
# geometry signature, and numerical grid settings.
#
# _HALO_LOCK protects this shared cache when several Julia threads request contexts.

const _HALO_CTX_CACHE = Dict{Tuple{Float64,Float64,Float64,Float64,UInt64,Symbol,Float64,Int,Float64,Float64},HaloContext}()
const _HALO_LOCK = ReentrantLock()
# ========================================================================================================================
# §3  SMALL UTILITIES
# ========================================================================================================================
# WHAT:
# Converts a halo dictionary into the Symbol-keyed form expected by the Julia physics code.
#
# HOW:
# String-like keys become Symbols. The :type value is also normalized to a lowercase Symbol.
#
# WHY:
# Config data can arrive from Python or Julia with slightly different key/value types.
# Normalizing once prevents every downstream force function from handling both forms.
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
# WHAT:
# Builds base-10 logarithmically spaced values between 10^a and 10^b.
#
# WHAT build_R_halo_physical DOES:
# Uses logspace10 to construct the physical radial grid for halo tables.
#
# WHY LOG SPACING:
# The force can change rapidly near the center while still needing to extend far into
# the halo. Log spacing gives much finer relative resolution at small radius.

logspace10(a,b,n)=n==1 ? [10.0^a] : (da=(b-a)/(n-1); [10.0^(a+(i-1)*da) for i in 1:n])
build_R_halo_physical(n; rmin=1e-3, rmax=300.0)=logspace10(log10(rmin), log10(rmax), n)
# WHAT:
# Rounds a Float64 to a fixed number of decimal digits.
#
# WHY:
# Used when constructing cache keys so tiny floating-point noise does not create many
# duplicate HaloContext entries for what is effectively the same model.
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
# WHAT:
# Builds adaptive projected radial-bin edges so each bin contains roughly a requested
# minimum number of stars.
#
# HOW:
# Sort the usable projected radii. Walk outward in groups of min_stars_per_bin.
# Place boundaries halfway between the last star of one group and the first star of
# the next.
#
# EDGE CASES:
# Empty data returns a harmless two-edge placeholder. One star gets a narrow bracket.
# If the total sample is smaller than the requested count, everything becomes one bin.
#
# IMPORTANT CURRENT-PIPELINE NOTE:
# resolve_karl_spatial_edges below explicitly forbids an adaptive fallback for the
# active Karl LOSVD path. This helper remains available, but supplied kinematic bins
# are the authority there.
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

# WHAT:
# Validates the supplied LOSVD projected-radius bin edges.
#
# IMPORTANT:
# The first edge is forcibly set to 0 pc so the central aperture really includes the
# galaxy center. Some prebuilt bin files otherwise begin at the radius of the first
# observed star, which would silently drop stars interior to that value.
#
# WHY:
# The kinematic-bin CSV is authoritative. There is deliberately no automatic
# re-binning fallback in this path.
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

# WHAT:
# Validates the projected-radius edges used for the light/tracer constraint.
#
# HOW:
# Require finite, strictly increasing edges. Force the first edge to 0 pc.
#
# WHY:
# The projected light target and orbit light matrix must describe exactly the same
# radial apertures.
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

# WHAT:
# Builds the LOSVD velocity grid from the observed stellar velocities and errors.
#
# MAIN GOAL:
# Make the velocity window wide enough that high-velocity model stars cannot simply
# fall outside the scored grid while preserving approximately the originally
# requested velocity resolution.
#
# HOW:
#   1. Find the observed velocity range.
#   2. Pad it using the measurement uncertainties.
#   3. Measure the bin width implied by requested Nvbin.
#   4. Enforce at least the configured minimum half-width.
#   5. If widening the window requires more bins, add bins rather than making each
#      velocity bin much broader.
#   6. Force the final number of velocity bins to be odd.
#
# WHY THIS MATTERS:
# A finite LOSVD window is part of the likelihood geometry. If the window is too
# narrow, fast model stars can disappear from the comparison rather than worsening
# chi2.
#
# UNITS:
# Inputs and returned edges are in m/s. The diagnostic print converts to km/s.
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

# WHAT:
# Finds which interval of an ordered edge vector contains x.
#
# RETURNS:
# A 1-based Julia bin index, or 0 when x lies outside the usable interval.
#
# NOTE:
# A value on or beyond the final edge is treated as outside because there is no bin
# beginning at the final edge.
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
# WHAT:
# Fast approximation to the standard normal cumulative distribution function.
#
# WHY:
# Observed stellar velocity errors are deposited across LOSVD velocity bins using a
# Gaussian. The bin probabilities require normal-CDF differences.
#
# IMPORTANT:
# This avoids adding SpecialFunctions.erf to the hot Julia path. It is a numerical
# approximation used for bin deposition, not a new statistical model.
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

# WHAT:
# Computes the probability that a Gaussian measurement centered on v0 with standard
# deviation sig falls between velocity edges vlo and vhi.
#
# HOW:
# Subtract the standard-normal CDF evaluated at the two normalized edges.
#
# WHY:
# One observed star contributes probability across several LOSVD bins according to
# its measurement uncertainty rather than being deposited as a delta function.
@inline function _gaussian_bin_probability(vlo::Float64, vhi::Float64, v0::Float64, sig::Float64)
    if !(isfinite(v0) && isfinite(sig) && sig > 0.0 && isfinite(vlo) && isfinite(vhi) && vhi > vlo)
        return 0.0
    end
    return max(0.0, _normal_cdf_unit((vhi - v0) / sig) - _normal_cdf_unit((vlo - v0) / sig))
end

# WHAT:
# Cleans a vector that represents probabilities or non-negative fractions, then
# normalizes it to unit sum when possible.
#
# HOW:
# Nonfinite and negative values become zero. Positive finite values are divided by
# their total.
#
# NOTE:
# The exclamation mark is meaningful here: this function modifies x in place.
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

# WHAT:
# Normalizes the key and numeric-array types of a surface-brightness/tracer profile.
#
# HOW:
# Keys become Symbols. Known numerical profile columns are converted to Float64 arrays.
# Other metadata is copied without reinterpretation.
#
# WHY:
# The target-building functions can then use one consistent representation regardless
# of whether the profile originated in Python or Julia.
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

# WHAT:
# Converts the observed projected stellar profile into the normalized light/tracer
# target vector used by the orbit-weight solver.
#
# PREFERRED PATH:
# If the data product already contains light_frac, those fractions are authoritative.
# If their original radial bins differ from the solver bins, the function redistributes
# each source bin by overlapping annular area.
#
# AREA REBINNING:
# For circular projected annuli, area is proportional to R_outer^2-R_inner^2. The
# source light is split according to the fraction of that annular area overlapping
# each destination bin.
#
# FALLBACK PATH:
# If no light_frac exists, sampled R_pc + Sigma values are accumulated into the
# requested spatial bins.
#
# IMPORTANT:
# There is no fallback to the kinematic star counts. The observed light profile is
# the authority for the tracer constraint.
#
# normalize=true:
# The resulting target is rescaled to sum to one, matching orbit weights interpreted
# as fractions of the total tracer light.
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

# WHAT:
# Propagates the observed surface-density uncertainties into uncertainties on the
# light/tracer target bins.
#
# PREFERRED light_frac PATH:
# Preserve the measured fractional uncertainty from the source surface-density bin:
#
#   sigma(L_i)/L_i = sigma(Sigma_i)/Sigma_i
#
# This converts each source light fraction into a source light-fraction uncertainty.
#
# REBINNING:
# When source and destination radial bins differ, each source contribution is split
# by annular overlap. Independent source-bin variances are then added in quadrature.
#
# NORMALIZATION:
# If the light target is normalized, the propagated sigma is divided by the same
# total-light normalization.
#
# FALLBACK:
# For unbinned R_pc + Sigma data, Sigma_err values falling into one destination bin
# are added in quadrature.
#
# sigma_floor:
# Prevents a nominally exact zero uncertainty from creating an infinite statistical
# weight later.
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

# WHAT:
# Builds all observed Karl-style targets needed by the orbit-weight solve:
#
#   losvd_target
#   losvd_sigma
#   light_target
#   light_sigma
#   counts_by_spatial
#
# LOSVD CONSTRUCTION:
# Each valid observed star is assigned to a projected radial bin. Its measured vlos
# is not placed into one hard velocity cell. Instead its Gaussian measurement error
# is integrated across all LOSVD velocity bins.
#
# The per-star probabilities are renormalized inside the finite velocity grid before
# being accumulated. This keeps each accepted star contributing total probability 1
# to its spatial LOSVD histogram.
#
# LIGHT COUPLING:
# The LOSVD in spatial bin i is scaled by the observed projected tracer light L_i:
#
#   y_ij = L_i * p_ij
#
# This means the LOSVD matrix and light profile share the same spatial normalization.
#
# LOSVD UNCERTAINTY:
# The finite stellar sample supplies uncertainty on the velocity-bin fractions. The
# code uses a Jeffreys-like Dirichlet pseudocount alpha=0.5 to avoid zero-variance
# histogram cells. The light L_i is treated structurally here and scales that
# probability uncertainty.
#
# IMPORTANT:
# light_target/light_sigma may use their own light_edges. LOSVD normalization always
# uses the kinematic_edges because its rows are organized by the kinematic apertures.
function observed_targets_karl(R_star_m::Vector{Float64}, valid_vlos::AbstractVector{Bool}, v_star_mps::Vector{Float64}, verr_star_mps::Vector{Float64}, kinematic_edges::Vector{Float64}, velocity_edges::Vector{Float64}; surface_brightness_profile=nothing, light_edges=nothing, sigma_floor::Float64=1e-8)
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
    # STEP: Build the independent projected tracer/light constraint.
    light_target = light_target_from_surface_brightness(surface_brightness_profile, light_edges_use; normalize=true)
    light_sigma = light_sigma_from_surface_brightness(surface_brightness_profile, light_edges_use; normalize=true, sigma_floor=sigma_floor)
    length(light_sigma) == length(light_target) || error("light_sigma length does not match light_target")
    # STEP: Put the LOSVD histogram onto the observed light normalization of each kinematic aperture.
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
    # STEP: Convert the finite stellar counts into statistical LOSVD-bin uncertainties.
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

# WHAT:
# Pulls in the two lower-level modules used by the support layer.
#
# OSPM_Physics_Weights.jl:
#   entropy, phase-volume weights, LOSVD slack variables, SPEAR weight solver.
#
# OSPM_Physics_Force.jl:
#   halo/stellar density, potential and force construction, force caches.
#
# WHY HERE:
# OSPM_Physics_Spherical.jl includes this support file once, then receives all three
# layers through one include chain.
# ========================================================================================================================
# §4  KARL WEIGHT / SPEAR SOLVER
# ========================================================================================================================
# The entropy, wphase, expanded Cm, LOSVD slack-variable SPEAR solve,
# xmu helpers, and χ² scoring live in OSPM_Physics_Weights.jl.
include("OSPM_Physics_Weights.jl")
include("OSPM_Physics_Force.jl")

# WHAT:
# Returns the time derivatives for one orbit state in cylindrical coordinates.
#
# STATE:
#   s = (R_cyl, z, vR, vz)
#
# CONSERVED INPUT:
#   Lz is the azimuthal angular momentum. vphi is not stored as a state variable;
#   its centrifugal effect appears as Lz^2/R^3.
#
# HOW:
# Convert cylindrical position to spherical r,theta so the shared frc closure can
# return radial and polar force components. Convert those forces back to cylindrical
# FR,Fz. Add the centrifugal term to radial acceleration.
#
# RETURNS:
#   (dR/dt, dz/dt, dvR/dt, dvz/dt)
#
# SAFETY:
# States outside the precomputed force domain, nonfinite states, negative R, or an
# Lz!=0 orbit reaching the cylindrical axis return an all-NaN derivative. The
# integrator interprets that as an invalid step rather than extrapolating the force.
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

# WHAT:
# Finds the outer zero-velocity radius for a fixed orbit family (E,Lz) at one theta.
#
# PHYSICAL PICTURE:
# At a turning point in the meridional plane, radial/polar kinetic energy vanishes.
# The remaining energy budget satisfies:
#
#   2(E-Phi) - Lz^2/(r^2 sin^2(theta)) = 0
#
# This function finds the outer root of that equation.
#
# HOW:
# Start from rapo_max. If that point is still inside the allowed region, expand
# outward until the forbidden side is found. If it is already outside, scan inward
# until the allowed side is found. Once a sign-changing bracket exists, bisect it.
#
# WHY:
# All members of one fixed-(E,|Lz|) orbit family must launch on the same zero-velocity
# curve. This helper supplies the radius of that curve at each theta.
#
# RETURN STATE:
# Returns the radius plus a Symbol describing success or the exact rejection reason.
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

    # WHAT:
    # Local helper for the zero-velocity-curve root search.
    #
    # Positive radial_budget means the chosen (r,theta) lies in the accessible region
    # for this E,Lz family. Zero is the turning curve. Negative is forbidden.
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

# WHAT:
# Builds a sequence of launch points along the outer zero-velocity curve for one
# fixed-(E,|Lz|) orbit family.
#
# PHYSICAL PICTURE:
# E and |Lz| define a family. Different third-integral orbits are launched from
# different positions along that family's meridional zero-velocity boundary.
#
# HOW:
#   1. Sample theta from near the axis to the equatorial plane.
#   2. Find the outer ZVC radius at every sampled theta.
#   3. Keep only the contiguous valid branch connected to the equator.
#   4. Convert the curve into cylindrical R,z.
#   5. Measure arc length along the curve.
#   6. Place Nlaunch launch points at equal fractions of that arc length.
#   7. Re-solve the exact ZVC radius at each selected theta.
#
# WHY ARC LENGTH:
# Equal spacing in theta would crowd launches where the ZVC bends sharply or stretch
# them where it is flat. Arc-length spacing distributes the third-integral launch
# sequence more uniformly along the actual physical boundary.
#
# CIRCULAR BOUNDARY:
# The circular member is a special one-point family at the equatorial ZVC and has
# zero arc length.
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

# WHAT:
# Defines the conserved E and Lz for one orbit family using an equatorial reference
# apocenter and a fraction of the local circular angular momentum.
#
# HOW:
# Evaluate the inward force at rapo. From circular balance:
#
#   vc^2 = -F_r * rapo
#
# Then:
#
#   Lz = Lz_frac * rapo * vc
#   E  = Phi(rapo) + Lz^2/(2 rapo^2)
#
# WHY:
# rapo selects the energy shell. Lz_frac selects how much angular momentum that
# family has relative to the circular orbit at that shell.
#
# IMPORTANT:
# This family definition is made at theta=pi/2. The off-equatorial launches later
# reuse exactly this E and |Lz| rather than recomputing a new family at each theta.
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

# WHAT:
# Converts an orbit-family choice into the actual initial condition used by the
# integrator.
#
# TWO MODES:
#
# 1. FREE FAMILY CONSTRUCTION:
#    Given rapo, theta0, and Lz_frac, compute E and Lz locally. Circular launches
#    begin at the turning point. Non-circular launches begin slightly inside it at
#    r0_frac*rapo with inward vr chosen from the energy equation.
#
# 2. FIXED-FAMILY LAUNCH:
#    If fixed_energy and fixed_lz are supplied, preserve those exact family
#    integrals. Find or accept the corresponding zero-velocity turning radius at
#    theta0. This is the path needed for the fixed-(E,|Lz|) third-integral sequence.
#
# TIMESTEP:
# Estimate a local orbital frequency from the velocity scale and set:
#
#   dt = dt_frac / Omega
#
# WHY:
# Faster inner orbits receive shorter physical timesteps than slower outer orbits.
#
# SAFETY:
# The function rejects invalid geometry, bad forces/potentials, inconsistent fixed
# family arguments, or a launch point that does not satisfy the energy budget.
#
# RETURN:
# Initial-condition tuple, Lz, E, circular-speed scale, and a status Symbol.
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

# WHAT:
# Numerically integrates one orbit through the already-built gravitational potential.
#
# STATE REPRESENTATION:
# Internally the RK4 state is cylindrical:
#
#   (R, z, vR, vz)
#
# Lz is held separately as a conserved azimuthal angular momentum. Stored output is
# converted back to:
#
#   r, vr, theta, vtheta
#
# METHOD:
# Classical fourth-order Runge-Kutta, but each requested top-level dt can be split
# into smaller local substeps when the orbit is moving through a fast region.
#
# LOCAL STEP CONTROL:
# Estimate several local frequencies from acceleration, meridional motion, radial
# cylindrical motion, and azimuthal motion. The largest frequency sets a maximum
# safe substep:
#
#   h_max = local_step_safety / omega_local
#
# WHY:
# A fixed timestep that is fine near apocenter can be far too coarse during a rapid
# central passage. The local substeps let one orbit keep the same top-level time
# sampling while resolving those fast portions more carefully.
#
# DOMAIN PROTECTION:
# The orbit is stopped rather than extrapolated when it leaves the precomputed force
# table, reaches the inner stop radius, crosses an invalid cylindrical-axis state,
# or develops a nonfinite RK4 stage.
#
# ENERGY DIAGNOSTIC:
# With return_diag=true, the code periodically recomputes
#
#   E = Phi + 1/2(vR^2 + vz^2 + vphi^2)
#
# and tracks the largest relative drift from the reference energy. Excess drift is a
# failed integration rather than an accepted orbit.
#
# CONTINUATION:
# continuation_state lets a later integration segment resume from a previous final
# cylindrical state. reference_energy keeps all segments judged against one energy.
#
# IMPORTANT:
# This is adaptive substepping inside a classical RK4 step. It is not an embedded
# RK method with a local truncation-error estimate.
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

    # STEP: Build the initial cylindrical phase-space state or resume a prior segment.
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

    # WHAT:
    # Local gate that decides whether a cylindrical orbit state is still inside the
    # valid integration domain.
    #
    # WHY:
    # The force closure should never be asked to extrapolate beyond its radial table.
    # Lz!=0 orbits also cannot pass through R=0 because vphi=Lz/R would diverge.
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

    # WHAT:
    # Converts the current cylindrical state back into the spherical variables stored
    # by the orbit library.
    #
    # RETURNS:
    #   r, theta, vr, vtheta
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

    # WHAT:
    # Evaluates the total specific orbital energy of the current cylindrical state.
    #
    # HOW:
    # Recover vphi=Lz/R, add the meridional kinetic energy, then add the total
    # gravitational potential from pot(r,theta).
    #
    # WHY:
    # Used only when orbit diagnostics request energy-conservation checks.
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

    # WHAT:
    # Estimates the maximum safe RK4 substep from the fastest local dynamical rate.
    #
    # FREQUENCY ESTIMATES:
    #   omega_dyn  from local acceleration
    #   omega_rad  from total meridional speed / radius
    #   omega_R    from cylindrical radial crossing rate
    #   omega_phi  from Lz/R^2
    #
    # The largest one controls the local resolution requirement.
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

    # WHAT:
    # Executes one classical RK4 substep of size h.
    #
    # SAFETY:
    # Every intermediate RK stage is checked before its derivative is used. If an
    # intermediate state leaves the valid domain, the function returns the original
    # state plus the specific failure reason instead of completing a bad RK4 step.
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

    # STEP: Integrate only while the current orbit remains numerically valid.
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

    # STEP: Integrate only while the current orbit remains numerically valid.
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

            # STEP: One requested dt may be split into several locally safe RK4 substeps.
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

    # STEP: Integrate only while the current orbit remains numerically valid.
    if termination_reason === :completed
        final_state_reason = state_exit_reason(state)
        final_state_reason !== :ok && (termination_reason = final_state_reason)
    end

    # STEP: Package the integration-quality information used by orbit diagnostics.
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