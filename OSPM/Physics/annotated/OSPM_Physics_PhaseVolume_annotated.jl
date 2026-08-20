# ========================================================================================================================
# OSPM_Physics_PhaseVolume.jl — Karl-style orbit phase-volume machinery.
# Included by OSPM_Physics_Support.jl — do NOT load directly.
# Owns:
#   - Karl equatorial surface-of-section extraction
#   - enclosed SOS-area calculation
#   - fixed-(E, |Lz|) third-integral cell construction
#   - energy and angular-momentum phase-cell widths
#   - circular-boundary phase-cell handling
#   - prograde/retrograde phase-volume pairing
#   - inverse-phase-volume wphase construction
#   - deterministic phase-volume self-checks
# ========================================================================================================================

# NOTES ON THE CONSTANTS BELOW:
# These set the default rules used when turning an integrated orbit into a
# phase-volume measurement.
#
# MIN_SOS_POINTS:
#   Minimum number of distinct surface-of-section radii needed before an
#   ordinary orbit is trusted enough to measure an enclosed area.
#
# DUPLICATE_RTOL:
#   Relative tolerance used to decide when two measured SOS areas are
#   effectively the same third-integral boundary.
#
# RADIUS_RTOL:
#   Relative tolerance used when two SOS points land at essentially the same
#   radius. Those points are merged before measuring area.
#
# SECTION_THETA:
#   The equatorial plane used for the surface of section. pi/2 is the
#   equatorial crossing in the current spherical-coordinate convention.
# ========================================================================================================================
# §1  CONSTANTS
# ========================================================================================================================

const DEFAULT_KARL_PHASE_MIN_SOS_POINTS = 8
const DEFAULT_KARL_PHASE_DUPLICATE_RTOL = 1.0e-10
const DEFAULT_KARL_PHASE_RADIUS_RTOL = 1.0e-10
const DEFAULT_KARL_PHASE_SECTION_THETA = pi / 2

# ========================================================================================================================
# §2  PHASE-VOLUME STATE
# ========================================================================================================================
# WHAT:
# Stores everything needed to compute Karl-style phase volumes for the base
# orbit library.
#
# PHYSICAL PICTURE:
# Each base orbit occupies a little cell in integral space. The cell is
# described by its energy, |Lz|, and the area between neighboring orbital
# curves on an equatorial surface of section.
#
# IMPORTANT:
# energy_index, lz_index, and third_index are launch-grid labels. In the
# current spherical library these correspond naturally to shell_id,
# lfrac_id, and theta_id.
#
# Only the positive-vr half of the SOS is stored. The full enclosed area is
# recovered later by doubling that half.
#
# third_index is retained as the launch-grid identity. The present phase-volume
# calculation finds the third-integral cell width from nested SOS areas rather
# than multiplying directly by third_index.
mutable struct KarlPhaseVolumeState
    Nbase_orbit::Int
    # Launch-grid coordinates.  These are the physical coordinates used by
    # phasvol.f after the orbit library has defined its E and Lz cells.
    energy::Vector{Float64}
    lz_abs::Vector{Float64}
    # Integer launch-grid labels.  For the current OSPM spherical file these
    # map naturally to shell_id, lfrac_id, and theta_id.
    energy_index::Vector{Int}
    lz_index::Vector{Int}
    third_index::Vector{Int}
    # Karl SOS points.  Only the positive-radial-velocity half is stored,
    # because the Fortran orbit path writes (r, abs(vr)).
    sos_r::Vector{Vector{Float64}}
    sos_vr_abs::Vector{Vector{Float64}}
    launch_recorded::Vector{Bool}
    sos_recorded::Vector{Bool}
end

# WHAT:
# Creates an empty KarlPhaseVolumeState large enough for every planned base
# orbit.
#
# HOW:
# Physical values start as NaN, integer labels start at 0, SOS point lists
# start empty, and the two recorded flags start false.
#
# WHY:
# Orbit integration happens later and fills this state one base orbit at a
# time. The fixed array size keeps the phase-volume bookkeeping tied to the
# original planned orbit numbering.
function init_karl_phase_volume_state(Nbase_orbit::Int)
    Nbase_orbit > 0 || error("Nbase_orbit must be positive")

    return KarlPhaseVolumeState( Nbase_orbit, fill(NaN, Nbase_orbit), fill(NaN, Nbase_orbit), zeros(Int, Nbase_orbit), zeros(Int, Nbase_orbit),
        zeros(Int, Nbase_orbit), [Float64[] for _ in 1:Nbase_orbit], [Float64[] for _ in 1:Nbase_orbit], fill(false, Nbase_orbit), fill(false, Nbase_orbit))
end

# WHAT:
# Checks that a requested base-orbit index actually exists in the phase-volume
# state.
#
# WHY:
# A bad index here would attach launch or SOS information to the wrong orbit.
# Failing immediately is safer than silently corrupting the orbit library.
@inline function _karl_phase_check_index(st::KarlPhaseVolumeState, base_index::Int)
    1 <= base_index <= st.Nbase_orbit ||
        error("base_index=$base_index is outside 1:$(st.Nbase_orbit)")
    return nothing
end

# WHAT:
# Records the integral-space launch coordinates for one base orbit.
#
# HOW:
# Stores the orbit energy, absolute angular momentum |Lz|, and the three
# launch-grid labels.
#
# WHY:
# The later phase-volume calculation needs to know the spacing of neighboring
# energy and angular-momentum cells.
#
# IMPORTANT:
# The sign of Lz is intentionally removed here. Prograde and retrograde copies
# of one base orbit are paired later and receive the same phase-volume factor.
function register_karl_phase_launch!( st::KarlPhaseVolumeState, base_index::Int; energy::Real, lz::Real, energy_index::Int, lz_index::Int, third_index::Int)
    _karl_phase_check_index(st, base_index)
    E = Float64(energy)
    L = abs(Float64(lz))
    isfinite(E) || error("nonfinite launch energy for base orbit $base_index")
    isfinite(L) || error("nonfinite launch angular momentum for base orbit $base_index")
    energy_index > 0 || error("energy_index must be positive")
    lz_index > 0 || error("lz_index must be positive")
    third_index > 0 || error("third_index must be positive")
    st.energy[base_index] = E
    st.lz_abs[base_index] = L
    st.energy_index[base_index] = energy_index
    st.lz_index[base_index] = lz_index
    st.third_index[base_index] = third_index
    st.launch_recorded[base_index] = true
    return nothing
end

# ========================================================================================================================
# §3  KARL EQUATORIAL SURFACE OF SECTION
# ========================================================================================================================
# WHAT:
# Extracts an equatorial surface of section from a fully integrated orbit.
#
# PHYSICAL PICTURE:
# Imagine watching the orbit only when it crosses the equatorial plane. At
# each accepted crossing we keep its radius and |vr|. Repeated crossings trace
# the orbital curve in the (r, vr) surface-of-section plane.
#
# HOW:
#   1. Find steps that cross theta = section_theta.
#   2. Keep up, down, or both crossing directions.
#   3. Optionally discard the first crossing.
#   4. Either use Karl's step endpoint or linearly interpolate the crossing.
#   5. Store radius and abs(vr).
#
# IMPORTANT:
# :karl_step intentionally uses the integrated endpoint r1,v1, matching the
# discrete Karl-style crossing convention. :linear estimates the actual plane
# crossing between the two integration steps.
#
# skip_first avoids treating the launch/initial crossing as an independent
# sampled return to the section.
function collect_karl_equatorial_sos( r::AbstractVector{<:Real}, vr::AbstractVector{<:Real}, theta::AbstractVector{<:Real}; section_theta::Float64=DEFAULT_KARL_PHASE_SECTION_THETA, crossing_mode::Symbol=:karl_step,
    direction::Symbol=:up, skip_first::Bool=true)
    n = length(r)
    length(vr) == n || error("r and vr lengths do not match")
    length(theta) == n || error("r and theta lengths do not match")
    n >= 2 || return Float64[], Float64[]
    crossing_mode in (:karl_step, :linear) ||
        error("crossing_mode must be :karl_step or :linear")
    direction in (:up, :down, :both) ||
        error("direction must be :up, :down, or :both")
    rsos = Float64[]
    vsos = Float64[]
    first_crossing = true
    @inbounds for k in 2:n
        r0 = Float64(r[k - 1])
        r1 = Float64(r[k])
        v0 = Float64(vr[k - 1])
        v1 = Float64(vr[k])
        q0 = Float64(theta[k - 1]) - section_theta
        q1 = Float64(theta[k]) - section_theta
        if !(isfinite(r0) && isfinite(r1) && isfinite(v0) && isfinite(v1) &&
             isfinite(q0) && isfinite(q1))
            continue
        end
        up_cross = q0 <= 0.0 && q1 >= 0.0 && (q0 < 0.0 || q1 > 0.0)
        down_cross = q0 >= 0.0 && q1 <= 0.0 && (q0 > 0.0 || q1 < 0.0)
        crossed = direction === :up ? up_cross :
                  direction === :down ? down_cross :
                  (up_cross || down_cross)
        crossed || continue
        if skip_first && first_crossing
            first_crossing = false
            continue
        end
        first_crossing = false
        rcross = r1
        vcross = v1
        if crossing_mode === :linear
            den = q1 - q0
            if isfinite(den) && abs(den) > eps(Float64)
                t = clamp(-q0 / den, 0.0, 1.0)
                rcross = muladd(t, r1 - r0, r0)
                vcross = muladd(t, v1 - v0, v0)
            end
        end
        if isfinite(rcross) && rcross > 0.0 && isfinite(vcross)
            push!(rsos, rcross)
            push!(vsos, abs(vcross))
        end
    end
    return rsos, vsos
end

# WHAT:
# Cleans and stores an already-computed set of SOS points for one base orbit.
#
# HOW:
# Nonfinite points and non-positive radii are discarded. Radial velocity is
# stored as an absolute value.
#
# NOTE:
# sos_recorded becomes true only when at least one valid point survives.
function record_karl_phase_sos!(st::KarlPhaseVolumeState, base_index::Int, sos_r::AbstractVector{<:Real}, sos_vr_abs::AbstractVector{<:Real})
    _karl_phase_check_index(st, base_index)
    length(sos_r) == length(sos_vr_abs) ||
        error("SOS radius and radial-velocity lengths do not match")
    rr = Float64[]
    vv = Float64[]
    sizehint!(rr, length(sos_r))
    sizehint!(vv, length(sos_vr_abs))
    @inbounds for k in eachindex(sos_r)
        rk = Float64(sos_r[k])
        vk = abs(Float64(sos_vr_abs[k]))
        if isfinite(rk) && rk > 0.0 && isfinite(vk)
            push!(rr, rk)
            push!(vv, vk)
        end
    end
    st.sos_r[base_index] = rr
    st.sos_vr_abs[base_index] = vv
    st.sos_recorded[base_index] = !isempty(rr)
    return nothing
end

# WHAT:
# Convenience path that takes an integrated orbit, extracts its equatorial SOS,
# and records that SOS into the phase-volume state.
#
# WHY:
# This keeps the crossing convention in one place rather than making every
# orbit-integration caller reproduce the same section-selection logic.
#
# IMPORTANT:
# The launch must already have been registered. Phase volume needs the orbit's
# E and |Lz| coordinates as well as its SOS geometry.
function record_karl_phase_orbit!(st::KarlPhaseVolumeState, base_index::Int, r::AbstractVector{<:Real}, vr::AbstractVector{<:Real}, theta::AbstractVector{<:Real};
    section_theta::Float64=DEFAULT_KARL_PHASE_SECTION_THETA, crossing_mode::Symbol=:karl_step, direction::Symbol=:up, skip_first::Bool=true)
    _karl_phase_check_index(st, base_index)
    st.launch_recorded[base_index] ||
        error("register the launch before recording the SOS for base orbit $base_index")
    rsos, vsos = collect_karl_equatorial_sos( r, vr, theta; section_theta=section_theta, crossing_mode=crossing_mode, direction=direction, skip_first=skip_first)
    record_karl_phase_sos!(st, base_index, rsos, vsos)
    return length(rsos)
end

# ========================================================================================================================
# §4  ENCLOSED SOS AREA
# ========================================================================================================================
# WHAT:
# Converts a cloud of SOS crossings into one ordered upper boundary v(r).
#
# PHYSICAL PICTURE:
# An orbit can cross the section many times at nearly the same radius. For the
# enclosed area we want the outer edge of the positive-vr half, not every
# repeated point inside that edge.
#
# HOW:
# Sort by radius. Points at nearly identical radii are grouped. Only the
# largest |vr| in each radius group is kept.
#
# WHY:
# This produces a clean curve that can be integrated for the SOS area.
function _karl_phase_upper_envelope(sos_r::AbstractVector{<:Real}, sos_vr_abs::AbstractVector{<:Real}; radius_rtol::Float64=DEFAULT_KARL_PHASE_RADIUS_RTOL)
    length(sos_r) == length(sos_vr_abs) ||
        error("SOS radius and radial-velocity lengths do not match")
    points = Tuple{Float64,Float64}[]
    sizehint!(points, length(sos_r))
    @inbounds for k in eachindex(sos_r)
        r = Float64(sos_r[k])
        v = abs(Float64(sos_vr_abs[k]))
        if isfinite(r) && r > 0.0 && isfinite(v)
            push!(points, (r, v))
        end
    end
    isempty(points) && return Float64[], Float64[]
    sort!(points; by=first)
    rr = Float64[]
    vv = Float64[]
    sizehint!(rr, length(points))
    sizehint!(vv, length(points))
    i = 1
    while i <= length(points)
        rref = points[i][1]
        vmax = points[i][2]
        j = i + 1
        while j <= length(points)
            rj = points[j][1]
            tol = radius_rtol * max(abs(rref), abs(rj), 1.0)
            abs(rj - rref) <= tol || break
            vmax = max(vmax, points[j][2])
            j += 1
        end
        push!(rr, rref)
        push!(vv, vmax)
        i = j
    end
    return rr, vv
end

# WHAT:
# Identifies the special circular-orbit boundary of an (E,|Lz|) family.
#
# PHYSICAL PICTURE:
# A circular orbit has no radial oscillation, so its SOS collapses to one point
# with vr = 0 rather than enclosing a finite area.
#
# NOTE:
# The exact representation expected here is one recorded SOS point with
# positive radius and exactly zero radial velocity.
@inline function _karl_phase_is_circular_boundary(st::KarlPhaseVolumeState, base_index::Int)
    _karl_phase_check_index(st, base_index)
    st.launch_recorded[base_index] || return false
    st.sos_recorded[base_index] || return false
    length(st.sos_r[base_index]) == 1 || return false
    length(st.sos_vr_abs[base_index]) == 1 || return false
    r = st.sos_r[base_index][1]
    v = st.sos_vr_abs[base_index][1]
    return isfinite(r) && r > 0.0 && isfinite(v) && v == 0.0
end

# WHAT:
# Measures the full area enclosed by one orbit on the equatorial surface of
# section.
#
# HOW:
# First build the positive-vr upper envelope. Integrate under that curve with
# the trapezoid rule. Double the result to restore the negative-vr half.
#
# PHYSICAL MEANING:
# This area is the third-integral part of the orbit's phase-space cell.
# Neighboring nested orbit areas will later be differenced to get the actual
# cell area assigned to each orbit.
#
# IMPORTANT:
# A circular boundary is allowed to have exactly zero enclosed area. Ordinary
# orbits with too few SOS points return NaN rather than inventing an area.
function karl_sos_enclosed_area(sos_r::AbstractVector{<:Real}, sos_vr_abs::AbstractVector{<:Real}; min_points::Int=DEFAULT_KARL_PHASE_MIN_SOS_POINTS, radius_rtol::Float64=DEFAULT_KARL_PHASE_RADIUS_RTOL)
    min_points >= 2 || error("min_points must be at least 2")
    rr, vv = _karl_phase_upper_envelope(sos_r, sos_vr_abs; radius_rtol=radius_rtol)
    if length(rr) == 1 && length(vv) == 1 && vv[1] == 0.0
        return 0.0
    end
    length(rr) >= min_points || return NaN
    positive_half_area = 0.0
    @inbounds for k in 2:length(rr)
        dr = rr[k] - rr[k - 1]
        if isfinite(dr) && dr > 0.0
            positive_half_area += 0.5 * (vv[k] + vv[k - 1]) * dr
        end
    end
    area = 2.0 * positive_half_area
    return isfinite(area) && area > 0.0 ? area : NaN
end

# ========================================================================================================================
# §5  PHASE-GRID CELL WIDTHS
# ========================================================================================================================
# WHAT:
# Returns the median of a Float64 vector.
#
# WHY HERE:
# Multiple base orbits can share one launch-grid label. Their measured E or
# |Lz| values should represent one grid center. The median gives a stable
# center without letting one odd value move the whole cell.
@inline function _karl_phase_median(values::Vector{Float64})
    isempty(values) && return NaN
    work = sort(copy(values))
    n = length(work)
    isodd(n) && return work[(n + 1) >>> 1]
    return 0.5 * (work[n >>> 1] + work[(n >>> 1) + 1])
end

# WHAT:
# Finds one representative physical coordinate for each integer launch-grid
# label.
#
# HOW:
# Gather the valid values belonging to each label, then use their median.
#
# EXAMPLE:
# All orbits carrying energy_index=3 contribute to the estimated physical
# energy coordinate of energy cell 3.
function _karl_phase_centers_by_label( labels::Vector{Int}, values::Vector{Float64}, use_mask::AbstractVector{Bool})
    length(labels) == length(values) == length(use_mask) ||
        error("label, value, and mask lengths do not match")
    gathered = Dict{Int,Vector{Float64}}()
    @inbounds for i in eachindex(labels)
        use_mask[i] || continue
        label = labels[i]
        value = values[i]
        label > 0 || continue
        isfinite(value) || continue
        push!(get!(gathered, label, Float64[]), value)
    end
    centers = Dict{Int,Float64}()
    for (label, samples) in gathered
        center = _karl_phase_median(samples)
        isfinite(center) && (centers[label] = center)
    end
    return centers
end

# WHAT:
# Turns a set of grid-cell centers into physical cell widths.
#
# HOW:
# Neighboring centers define midpoint boundaries. Each width is the distance
# between the two boundaries surrounding that center.
#
# EDGE CELLS:
# The first and last cells are extended outward by half of their nearest
# center spacing.
#
# NOTE:
# If only one center exists, there is no neighboring spacing to infer. The
# caller-supplied singleton_width is used instead.
function _karl_phase_widths_from_centers( centers::Dict{Int,Float64}; singleton_width::Float64=1.0)
    isfinite(singleton_width) && singleton_width > 0.0 ||
        error("singleton_width must be finite and positive")
    isempty(centers) && return Dict{Int,Float64}()
    ordered = sort(collect(centers); by=last)
    widths = Dict{Int,Float64}()
    if length(ordered) == 1
        widths[ordered[1][1]] = singleton_width
        return widths
    end
    x = last.(ordered)
    n = length(x)
    boundaries = Vector{Float64}(undef, n + 1)
    @inbounds for k in 2:n
        boundaries[k] = 0.5 * (x[k - 1] + x[k])
    end
    boundaries[1] = x[1] - 0.5 * (x[2] - x[1])
    boundaries[end] = x[end] + 0.5 * (x[end] - x[end - 1])
    @inbounds for k in 1:n
        width = abs(boundaries[k + 1] - boundaries[k])
        if !(isfinite(width) && width > 0.0)
            error("non-positive launch-grid cell width at label $(ordered[k][1])")
        end
        widths[ordered[k][1]] = width
    end
    return widths
end

# WHAT:
# Computes dE, the energy-cell width assigned to every base orbit.
#
# HOW:
# Determine one physical energy center per energy_index. Build widths between
# those centers. Copy the width for each label back onto every orbit carrying
# that label.
#
# PHYSICAL ROLE:
# dE supplies the energy dimension of the orbit's phase-space volume.
function _karl_phase_energy_widths( st::KarlPhaseVolumeState; singleton_width::Float64=1.0)
    centers = _karl_phase_centers_by_label(st.energy_index, st.energy, st.launch_recorded)
    widths_by_label = _karl_phase_widths_from_centers( centers; singleton_width=singleton_width)
    dE = fill(NaN, st.Nbase_orbit)
    @inbounds for i in 1:st.Nbase_orbit
        label = st.energy_index[i]
        haskey(widths_by_label, label) && (dE[i] = widths_by_label[label])
    end
    return dE, centers, widths_by_label
end

# WHAT:
# Computes dLz, the |Lz|-cell width assigned to every base orbit.
#
# IMPORTANT:
# Lz spacing is calculated separately inside each energy shell. This matters
# because the allowed angular-momentum range changes with energy.
#
# PHYSICAL ROLE:
# dLz supplies the angular-momentum dimension of the orbit's phase-space cell.
function _karl_phase_lz_widths( st::KarlPhaseVolumeState; singleton_width::Float64=1.0)
    dLz = fill(NaN, st.Nbase_orbit)
    centers_by_energy = Dict{Int,Dict{Int,Float64}}()
    widths_by_energy = Dict{Int,Dict{Int,Float64}}()
    energy_labels = sort(unique(filter(>(0), st.energy_index[st.launch_recorded])))
    for energy_label in energy_labels
        mask = st.launch_recorded .& (st.energy_index .== energy_label)
        centers = _karl_phase_centers_by_label(st.lz_index, st.lz_abs, mask)
        widths = _karl_phase_widths_from_centers( centers; singleton_width=singleton_width,)
        centers_by_energy[energy_label] = centers
        widths_by_energy[energy_label] = widths
        @inbounds for i in 1:st.Nbase_orbit
            st.energy_index[i] == energy_label || continue
            label = st.lz_index[i]
            haskey(widths, label) && (dLz[i] = widths[label])
        end
    end
    return dLz, centers_by_energy, widths_by_energy
end

# WHAT:
# Converts nested enclosed SOS areas into the actual third-integral cell area
# assigned to each orbit at fixed (E,|Lz|).
#
# PHYSICAL PICTURE:
# Suppose several orbital curves are nested like rings. The first orbit owns
# the area from zero to its curve. The next orbit owns only the annulus between
# its curve and the previous one. That annular area is delta_area.
#
# HOW:
# Orbits are grouped by (energy_index,lz_index), then sorted by enclosed SOS
# area. Consecutive areas are differenced.
#
# DUPLICATES:
# Nearly equal areas are treated as one cluster. The available annular area is
# divided evenly among the duplicate orbits so none receives zero phase volume.
#
# NOTE:
# This is where the third-integral spacing enters the current implementation.
# The stored third_index labels are not directly multiplied into the volume.
function _karl_phase_nested_area_differences!(delta_area::Vector{Float64}, sos_area::Vector{Float64}, valid_mask::AbstractVector{Bool}, energy_index::Vector{Int}, lz_index::Vector{Int}; duplicate_rtol::Float64=DEFAULT_KARL_PHASE_DUPLICATE_RTOL)
    n = length(sos_area)
    length(delta_area) == n || error("delta_area and sos_area lengths do not match")
    length(valid_mask) == n || error("valid_mask length does not match sos_area")
    length(energy_index) == n || error("energy_index length does not match sos_area")
    length(lz_index) == n || error("lz_index length does not match sos_area")
    groups = Dict{Tuple{Int,Int},Vector{Int}}()
    @inbounds for i in 1:n
        valid_mask[i] || continue
        key = (energy_index[i], lz_index[i])
        key[1] > 0 && key[2] > 0 || continue
        push!(get!(groups, key, Int[]), i)
    end
    duplicate_clusters = 0
    duplicate_orbits = 0
    for members in values(groups)
        sort!(members; by=i -> (sos_area[i], i))
        previous_area = 0.0
        k = 1
        while k <= length(members)
            first_member = members[k]
            cluster_area = sos_area[first_member]
            j = k + 1
            while j <= length(members)
                next_area = sos_area[members[j]]
                tol = duplicate_rtol * max(abs(cluster_area), abs(next_area), 1.0)
                abs(next_area - cluster_area) <= tol || break
                cluster_area = max(cluster_area, next_area)
                j += 1
            end
            cluster_count = j - k
            annular_area = cluster_area - previous_area
            if !(isfinite(annular_area) && annular_area > 0.0)
                scale = max(abs(cluster_area), abs(previous_area), 1.0)
                annular_area = max(duplicate_rtol * scale, eps(Float64) * scale)
            end
            per_orbit_area = annular_area / cluster_count
            @inbounds for q in k:(j - 1)
                delta_area[members[q]] = per_orbit_area
            end
            if cluster_count > 1
                duplicate_clusters += 1
                duplicate_orbits += cluster_count
            end
            previous_area = max(previous_area, cluster_area)
            k = j
        end
    end
    return duplicate_clusters, duplicate_orbits, length(groups)
end

# ========================================================================================================================
# §6  COMPLETE KARL PHASE-VOLUME CALCULATION
# ========================================================================================================================
# WHAT:
# Duplicates every base-orbit value into a prograde/retrograde pair.
#
# EXAMPLE:
# [A,B] becomes [A,A,B,B].
#
# WHY:
# One base orbit supplies two signed-Lz columns in the paired orbit library.
# Both signs occupy the same phase volume, so they must receive the same
# phase-volume factor.
function _karl_phase_repeat_pairs(values::Vector{Float64})
    paired = Vector{Float64}(undef, 2 * length(values))
    @inbounds for i in eachindex(values)
        paired[2 * i - 1] = values[i]
        paired[2 * i] = values[i]
    end
    return paired
end

# WHAT:
# Optionally rescales raw phase volumes without changing their relative ratios.
#
# :none:
#   Leave every valid phase volume in its raw physical scale.
#
# :geometric_mean:
#   Divide all valid volumes by their geometric mean. The typical normalized
#   phase volume is then near 1.
#
# WHY:
# The entropy machinery mainly needs the relative phase-volume factors.
# Removing a huge common scale improves numerical conditioning.
#
# RETURN:
# The normalized volumes plus the mean log-volume used for the normalization.
function _karl_phase_normalize(raw_phase_volume::Vector{Float64}, valid_mask::AbstractVector{Bool}, mode::Symbol)
    mode in (:none, :geometric_mean) ||
        error("normalization must be :none or :geometric_mean")
    normalized = fill(NaN, length(raw_phase_volume))
    valid_indices = Int[]
    @inbounds for i in eachindex(raw_phase_volume)
        if valid_mask[i] && isfinite(raw_phase_volume[i]) && raw_phase_volume[i] > 0.0
            push!(valid_indices, i)
        end
    end
    isempty(valid_indices) && return normalized, NaN
    if mode === :none
        @inbounds for i in valid_indices
            normalized[i] = raw_phase_volume[i]
        end
        return normalized, 0.0
    end
    mean_log_volume = 0.0
    @inbounds for i in valid_indices
        mean_log_volume += log(raw_phase_volume[i])
    end
    mean_log_volume /= length(valid_indices)
    @inbounds for i in valid_indices
        normalized[i] = exp(log(raw_phase_volume[i]) - mean_log_volume)
    end
    return normalized, mean_log_volume
end

# WHAT:
# Gives the circular-orbit boundary a finite third-integral cell width even
# though the circular SOS itself has zero enclosed area.
#
# PHYSICAL PICTURE:
# The circular orbit is the edge of the allowed SOS family. Its cell should
# represent roughly half of the neighboring interval, just like a boundary
# bin on an ordinary grid.
#
# HOW:
# Find the adjacent interior Lz family at the same energy. Use half of its
# nearest positive SOS area as the circular boundary's delta_area.
#
# IMPORTANT:
# Without this special boundary rule the circular orbit would receive zero
# phase volume and an unusable inverse phase-volume weight.
function _karl_phase_assign_circular_boundary_widths!(delta_area::Vector{Float64}, sos_area::Vector{Float64}, boundary_mask::AbstractVector{Bool}, valid_mask::AbstractVector{Bool}, energy_index::Vector{Int}, lz_index::Vector{Int})
    n = length(sos_area)
    length(delta_area) == n || error("delta_area and sos_area lengths do not match")
    length(boundary_mask) == n || error("boundary_mask length does not match sos_area")
    length(valid_mask) == n || error("valid_mask length does not match sos_area")
    length(energy_index) == n || error("energy_index length does not match sos_area")
    length(lz_index) == n || error("lz_index length does not match sos_area")
    assigned = 0
    @inbounds for i in 1:n
        boundary_mask[i] && valid_mask[i] || continue
        energy_label = energy_index[i]
        boundary_lz_label = lz_index[i]
        adjacent_lz_label = 0
        for j in 1:n
            valid_mask[j] || continue
            boundary_mask[j] && continue
            energy_index[j] == energy_label || continue
            lz_index[j] < boundary_lz_label || continue
            adjacent_lz_label = max(adjacent_lz_label, lz_index[j])
        end
        adjacent_lz_label > 0 || continue
        nearest_area = Inf
        for j in 1:n
            valid_mask[j] || continue
            boundary_mask[j] && continue
            energy_index[j] == energy_label || continue
            lz_index[j] == adjacent_lz_label || continue
            area = sos_area[j]
            isfinite(area) && area > 0.0 || continue
            nearest_area = min(nearest_area, area)
        end
        if isfinite(nearest_area) && nearest_area > 0.0
            delta_area[i] = 0.5 * nearest_area
            assigned += 1
        end
    end
    return assigned
end

# WHAT:
# Performs the complete Karl-style phase-volume calculation for the base orbit
# library.
#
# CORE PHYSICAL CONSTRUCTION:
#
#   phase volume ~ delta(SOS area) * dE * d|Lz|
#
# The three factors correspond to the third-integral cell, energy-cell width,
# and angular-momentum-cell width.
#
# MAIN STEPS:
#   1. Measure each recorded orbit's enclosed SOS area.
#   2. Compute dE from the launch energy grid.
#   3. Compute dLz separately inside each energy shell.
#   4. Difference nested SOS areas at fixed (E,|Lz|).
#   5. Give circular boundaries a half-cell SOS width.
#   6. Multiply delta_area*dE*dLz.
#   7. Normalize the valid phase volumes if requested.
#   8. Set wphase = 1/phase_volume.
#   9. Duplicate each value for prograde/retrograde orbit columns.
#
# WHY WPHASE IS THE INVERSE:
# A small phase-space cell represents less available phase space. Its inverse
# is larger. The entropy expression uses this factor to measure orbital
# occupation relative to the available phase-space volume rather than treating
# every discrete library column as an equally sized cell.
#
# STRICT MODE:
# A recorded SOS that cannot produce a valid phase-volume product causes an
# error instead of silently entering the entropy calculation.
#
# IMPORTANT:
# required_mask is based on sos_recorded. A launch with no recorded SOS is not
# counted as a required phase-volume entry here. Orbit-library coverage logic
# elsewhere must decide whether such a missing orbit is acceptable.
function compute_karl_phase_volumes(st::KarlPhaseVolumeState; normalization::Symbol=:geometric_mean, min_sos_points::Int=DEFAULT_KARL_PHASE_MIN_SOS_POINTS, duplicate_rtol::Float64=DEFAULT_KARL_PHASE_DUPLICATE_RTOL, radius_rtol::Float64=DEFAULT_KARL_PHASE_RADIUS_RTOL, singleton_energy_width::Float64=1.0, singleton_lz_width::Float64=1.0, strict::Bool=true)
    n = st.Nbase_orbit
    sos_area = fill(NaN, n)
    circular_boundary_mask = fill(false, n)
    @inbounds for i in 1:n
        st.sos_recorded[i] || continue
        circular_boundary_mask[i] = _karl_phase_is_circular_boundary(st, i)
        sos_area[i] = karl_sos_enclosed_area(st.sos_r[i], st.sos_vr_abs[i]; min_points=min_sos_points, radius_rtol=radius_rtol)
    end
    # STEP: Measure the physical widths of the E and |Lz| launch cells.
    dE, energy_centers, energy_widths = _karl_phase_energy_widths(st; singleton_width=singleton_energy_width)
    dLz, lz_centers, lz_widths = _karl_phase_lz_widths(st; singleton_width=singleton_lz_width)
    # STEP: Decide which recorded SOS entries are complete enough to participate.
    required_mask = copy(st.sos_recorded)
    valid_mask = fill(false, n)
    @inbounds for i in 1:n
        valid_mask[i] =
            required_mask[i] &&
            st.launch_recorded[i] &&
            isfinite(st.energy[i]) &&
            isfinite(st.lz_abs[i]) &&
            isfinite(sos_area[i]) && sos_area[i] >= 0.0 &&
            isfinite(dE[i]) && dE[i] > 0.0 &&
            isfinite(dLz[i]) && dLz[i] > 0.0
    end

    invalid_required = findall(required_mask .& .!valid_mask)

    if strict && !isempty(invalid_required)
        preview = join(first(invalid_required, min(length(invalid_required), 20)), ",")
        suffix = length(invalid_required) > 20 ? ",..." : ""
        error("Karl phase-volume calculation failed for $(length(invalid_required)) recorded base orbit(s): [$preview$suffix]")
    end

    # STEP: Turn nested enclosed SOS curves into non-overlapping cell areas.
    delta_sos_area = fill(NaN, n)
    interior_mask = valid_mask .& .!circular_boundary_mask
    duplicate_clusters, duplicate_orbits, nested_groups = _karl_phase_nested_area_differences!(delta_sos_area, sos_area, interior_mask, st.energy_index, st.lz_index; duplicate_rtol=duplicate_rtol)
    circular_boundary_widths_assigned = _karl_phase_assign_circular_boundary_widths!(delta_sos_area, sos_area, circular_boundary_mask, valid_mask, st.energy_index, st.lz_index)
    # STEP: Combine the three cell dimensions into the raw phase volume.
    raw_phase_volume = fill(NaN, n)

    @inbounds for i in 1:n
        valid_mask[i] || continue
        volume = abs(delta_sos_area[i] * dE[i] * dLz[i])
        if isfinite(volume) && volume > 0.0
            raw_phase_volume[i] = volume
        else
            valid_mask[i] = false
        end
    end
    invalid_after_product = findall(required_mask .& .!valid_mask)
    if strict && !isempty(invalid_after_product)
        preview = join(first(invalid_after_product, min(length(invalid_after_product), 20)), ",")
        suffix = length(invalid_after_product) > 20 ? ",..." : ""
        error("Karl phase-volume product is invalid for $(length(invalid_after_product)) recorded base orbit(s): [$preview$suffix]")
    end
    # STEP: Remove an optional common scale, then form the entropy phase factor.
    phase_volume, mean_log_normalization = _karl_phase_normalize(raw_phase_volume, valid_mask, normalization)
    wphase = fill(NaN, n)
    @inbounds for i in 1:n
        if valid_mask[i]
            wphase[i] = 1.0 / phase_volume[i]
        end
    end

    # STEP: Expand each base-orbit result into its +Lz and -Lz solver columns.
    raw_phase_volume_paired = _karl_phase_repeat_pairs(raw_phase_volume)
    phase_volume_paired = _karl_phase_repeat_pairs(phase_volume)
    wphase_paired = _karl_phase_repeat_pairs(wphase)
    valid_paired = _karl_phase_repeat_pairs(Float64.(valid_mask)) .== 1.0
    valid_indices = findall(valid_mask)
    raw_min = isempty(valid_indices) ? NaN : minimum(raw_phase_volume[valid_indices])
    raw_max = isempty(valid_indices) ? NaN : maximum(raw_phase_volume[valid_indices])
    norm_min = isempty(valid_indices) ? NaN : minimum(phase_volume[valid_indices])
    norm_max = isempty(valid_indices) ? NaN : maximum(phase_volume[valid_indices])
    wphase_min = isempty(valid_indices) ? NaN : minimum(wphase[valid_indices])
    wphase_max = isempty(valid_indices) ? NaN : maximum(wphase[valid_indices])

    diagnostics = (convention=:inverse_phase_volume, entropy_expression=Symbol("-sum(w*log(w*wphase))"), normalization=normalization, mean_log_normalization=mean_log_normalization,
        planned_base_orbits=n, planned_paired_columns=2 * n, launches_recorded=count(identity, st.launch_recorded), sos_recorded=count(identity, st.sos_recorded), valid_base_orbits=length(valid_indices),
        invalid_recorded_orbits=count(identity, required_mask .& .!valid_mask), nested_groups=nested_groups, duplicate_area_clusters=duplicate_clusters, duplicate_area_orbits=duplicate_orbits,
        circular_boundary_orbits=count(identity, circular_boundary_mask), circular_boundary_widths_assigned=circular_boundary_widths_assigned, raw_phase_volume_min=raw_min,
        raw_phase_volume_max=raw_max, raw_phase_volume_dynamic_range=(isfinite(raw_min) && raw_min > 0.0) ? raw_max / raw_min : NaN, normalized_phase_volume_min=norm_min, normalized_phase_volume_max=norm_max,
        wphase_min=wphase_min, wphase_max=wphase_max, energy_centers=energy_centers, energy_widths=energy_widths, lz_centers=lz_centers, lz_widths=lz_widths)

    return (raw_phase_volume_base=raw_phase_volume, phase_volume_base=phase_volume, wphase_base=wphase, raw_phase_volume_paired=raw_phase_volume_paired,
        phase_volume_paired=phase_volume_paired, wphase_paired=wphase_paired, valid_base=valid_mask, valid_paired=valid_paired, sos_area=sos_area,
        delta_sos_area=delta_sos_area, dE=dE, dLz=dLz, diagnostics=diagnostics)
end

# WHAT:
# Small public wrapper that computes the full phase-volume result and returns
# only the paired inverse-phase-volume vector plus diagnostics.
#
# WHY:
# The weight solver usually needs wphase, not every intermediate SOS and cell
# width produced by compute_karl_phase_volumes.
function build_karl_wphase(st::KarlPhaseVolumeState; kwargs...)
    result = compute_karl_phase_volumes(st; kwargs...)
    return result.wphase_paired, result.diagnostics
end


# WHAT:
# Removes phase-volume entries belonging to orbit columns that failed before
# the final solver matrix was built.
#
# HOW:
# successful_columns indexes the planned paired wphase vector. Only those
# entries are copied into the compact solver-sized vector.
#
# SAFETY:
# Every surviving wphase must be finite and strictly positive. Entropy cannot
# safely use a missing, zero, or negative phase-volume factor.
function compact_karl_wphase( wphase_paired::AbstractVector{<:Real}, successful_columns::AbstractVector{<:Integer}, planned_norbit::Int)
    length(wphase_paired) == planned_norbit ||
        error("wphase length $(length(wphase_paired)) does not match planned Norbit=$planned_norbit")
    compacted = Float64.(wphase_paired[successful_columns])
    all(isfinite, compacted) || error("compacted Karl wphase contains nonfinite values")
    all(>(0.0), compacted) || error("compacted Karl wphase contains non-positive values")
    return compacted
end

# ========================================================================================================================
# §7  DETERMINISTIC SELF-CHECK
# ========================================================================================================================
# WHAT:
# Deterministic self-test for the phase-volume machinery.
#
# HOW:
# Build two synthetic elliptical SOS curves whose exact areas are known:
#
#   area = pi*a*b
#
# Check that the numerical SOS integration recovers those areas. Then place
# the two curves in one fixed-(E,|Lz|) family and verify that the inner cell
# gets area1 while the outer cell gets area2-area1.
#
# It also checks that each base-orbit phase factor is copied identically into
# its prograde and retrograde columns.
#
# WHY:
# This catches changes that would break the geometry of the phase-volume
# calculation even if the rest of OSPM still runs.
function karl_phase_volume_selftest(; rtol::Float64=2.0e-2)
    npoint = 2048
    angle = range(0.0, 2.0 * pi; length=npoint + 1)[1:end-1]
    r0 = 10.0
    a1 = 2.0
    b1 = 3.0
    a2 = 3.0
    b2 = 4.0
    r1 = r0 .+ a1 .* cos.(angle)
    v1 = abs.(b1 .* sin.(angle))
    r2 = r0 .+ a2 .* cos.(angle)
    v2 = abs.(b2 .* sin.(angle))
    area1 = karl_sos_enclosed_area(r1, v1; min_points=8)
    area2 = karl_sos_enclosed_area(r2, v2; min_points=8)
    expected1 = pi * a1 * b1
    expected2 = pi * a2 * b2
    isapprox(area1, expected1; rtol=rtol) ||
        error("Karl SOS area selftest failed for orbit 1: got $area1 expected $expected1")
    isapprox(area2, expected2; rtol=rtol) ||
        error("Karl SOS area selftest failed for orbit 2: got $area2 expected $expected2")
    st = init_karl_phase_volume_state(2)
    register_karl_phase_launch!(st, 1; energy=-10.0, lz=2.0, energy_index=1, lz_index=1, third_index=1)
    register_karl_phase_launch!(st, 2; energy=-10.0, lz=2.0, energy_index=1, lz_index=1, third_index=2)
    record_karl_phase_sos!(st, 1, r1, v1)
    record_karl_phase_sos!(st, 2, r2, v2)
    result = compute_karl_phase_volumes( st; normalization=:none, singleton_energy_width=1.0, singleton_lz_width=1.0, strict=true)
    isapprox(result.delta_sos_area[1], expected1; rtol=rtol) ||
        error("nested-area selftest failed for inner orbit")
    isapprox(result.delta_sos_area[2], expected2 - expected1; rtol=rtol) ||
        error("nested-area selftest failed for outer orbit")
    result.wphase_paired[1] == result.wphase_paired[2] ||
        error("prograde/retrograde wphase duplication failed")
    result.wphase_paired[3] == result.wphase_paired[4] ||
        error("prograde/retrograde wphase duplication failed")
    return ( passed=true, measured_area_1=area1, expected_area_1=expected1, measured_area_2=area2, expected_area_2=expected2, result=result)
end
