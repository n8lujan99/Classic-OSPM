# ============================================================
# OSPM_Physics_PhaseVolume.jl — Karl-style orbit phase-volume machinery.
# Included by OSPM_Physics_Support.jl — do NOT load directly.
# Reproduces the Karl area.f / phasvol.f calculation in the form used by
# the live Julia orbit library:
#
#   1. Record one equatorial surface-of-section (SOS) using (r, |v_r|).
#   2. Integrate the enclosed SOS area for every base orbit.
#   3. At fixed launch-grid (E, |L_z|), difference nested SOS areas to
#      obtain the third-integral cell area.
#   4. Multiply by the local energy and angular-momentum cell widths:
#          phase_volume_i = ΔA_SOS,i * ΔE_i * Δ|L_z|_i
#   5. Return wphase_i = 1 / phase_volume_i because Karl entropy type 2 is
#          S = -Σ w_i log(w_i * wphase_i).

# A common normalization of every phase volume does not change the fitted
# weights.  The raw Karl-comparable values are retained in the result.
# ============================================================

const DEFAULT_KARL_PHASE_MIN_SOS_POINTS = 8
const DEFAULT_KARL_PHASE_DUPLICATE_RTOL = 1.0e-10
const DEFAULT_KARL_PHASE_RADIUS_RTOL = 1.0e-10
const DEFAULT_KARL_PHASE_SECTION_THETA = pi / 2

# -----------------------------------------------------------------------------
# Per-library phase-volume state
# -----------------------------------------------------------------------------

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

function init_karl_phase_volume_state(Nbase_orbit::Int)
    Nbase_orbit > 0 || error("Nbase_orbit must be positive")

    return KarlPhaseVolumeState( Nbase_orbit, fill(NaN, Nbase_orbit), fill(NaN, Nbase_orbit), zeros(Int, Nbase_orbit), zeros(Int, Nbase_orbit),
        zeros(Int, Nbase_orbit), [Float64[] for _ in 1:Nbase_orbit], [Float64[] for _ in 1:Nbase_orbit], fill(false, Nbase_orbit), fill(false, Nbase_orbit))
end

@inline function _karl_phase_check_index(st::KarlPhaseVolumeState, base_index::Int)
    1 <= base_index <= st.Nbase_orbit ||
        error("base_index=$base_index is outside 1:$(st.Nbase_orbit)")
    return nothing
end

#register_karl_phase_launch!(st, base_index; energy, lz, energy_index, lz_index, third_index)
#Record the physical launch coordinates and their discrete Karl-grid labels.
#Call this after `launch_orbit_apocenter` succeeds.  It may be called again for
#a retry; the final successful launch should be the one left in the state.

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

# -----------------------------------------------------------------------------
# Karl equatorial surface of section
# -----------------------------------------------------------------------------
#collect_karl_equatorial_sos(...)
#Extract Karl's surface of section from an integrated Julia orbit.
#The original Fortran orbit code stores a point when the latitude crosses from
#negative to positive, writing the current-step values `(r, abs(vr))`.  Julia's
#orbit coordinate is polar angle, so the same equatorial section is
#theta - pi/2 = 0`.
#crossing_mode=:karl_step` reproduces the Fortran sampling.  The optional
#:linear` mode interpolates to the crossing and is intended only as a numerical
#diagnostic.

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

#record_karl_phase_sos!(st, base_index, sos_r, sos_vr_abs)
#Store an already extracted Karl SOS for one base orbit.
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

#record_karl_phase_orbit!(st, base_index, r, vr, theta; ...)
#Convenience hook for the orbit worker.  The launch must already have been
#registered with `register_karl_phase_launch!`.

function record_karl_phase_orbit!(st::KarlPhaseVolumeState, base_index::Int, r::AbstractVector{<:Real}, vr::AbstractVector{<:Real}, theta::AbstractVector{<:Real};
    section_theta::Float64=DEFAULT_KARL_PHASE_SECTION_THETA, crossing_mode::Symbol=:karl_step, direction::Symbol=:up, skip_first::Bool=true)
    _karl_phase_check_index(st, base_index)
    st.launch_recorded[base_index] ||
        error("register the launch before recording the SOS for base orbit $base_index")
    rsos, vsos = collect_karl_equatorial_sos( r, vr, theta; section_theta=section_theta, crossing_mode=crossing_mode, direction=direction, skip_first=skip_first)
    record_karl_phase_sos!(st, base_index, rsos, vsos)
    return length(rsos)
end

# -----------------------------------------------------------------------------
# area.f equivalent: enclosed area in the (r, v_r) surface of section
# -----------------------------------------------------------------------------

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

#karl_sos_enclosed_area(sos_r, sos_vr_abs; min_points=8)
#Trapezoidally integrate the positive-|v_r| SOS branch and reflect it across
#`v_r=0`.  This is the Julia equivalent of Karl's `area.f` calculation for the
#stored `(r, abs(vr))` section.

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

# -----------------------------------------------------------------------------
# phasvol.f helpers: launch-grid widths and nested-area differences
# -----------------------------------------------------------------------------

@inline function _karl_phase_median(values::Vector{Float64})
    isempty(values) && return NaN
    work = sort(copy(values))
    n = length(work)
    isodd(n) && return work[(n + 1) >>> 1]
    return 0.5 * (work[n >>> 1] + work[(n >>> 1) + 1])
end

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

function _karl_phase_nested_area_differences!( delta_area::Vector{Float64}, sos_area::Vector{Float64}, valid_mask::AbstractVector{Bool}, energy_index::Vector{Int}, lz_index::Vector{Int}; duplicate_rtol::Float64=DEFAULT_KARL_PHASE_DUPLICATE_RTOL)
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

# -----------------------------------------------------------------------------
# Complete Karl phase-volume calculation
# -----------------------------------------------------------------------------

function _karl_phase_repeat_pairs(values::Vector{Float64})
    paired = Vector{Float64}(undef, 2 * length(values))
    @inbounds for i in eachindex(values)
        paired[2 * i - 1] = values[i]
        paired[2 * i] = values[i]
    end
    return paired
end

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

function compute_karl_phase_volumes(st::KarlPhaseVolumeState; normalization::Symbol=:geometric_mean, min_sos_points::Int=DEFAULT_KARL_PHASE_MIN_SOS_POINTS, duplicate_rtol::Float64=DEFAULT_KARL_PHASE_DUPLICATE_RTOL,
    radius_rtol::Float64=DEFAULT_KARL_PHASE_RADIUS_RTOL, singleton_energy_width::Float64=1.0, singleton_lz_width::Float64=1.0, strict::Bool=true)
    n = st.Nbase_orbit
    sos_area = fill(NaN, n)
    circular_boundary_mask = fill(false, n)
    @inbounds for i in 1:n
        st.sos_recorded[i] || continue
        circular_boundary_mask[i] = _karl_phase_is_circular_boundary(st, i)
        sos_area[i] = karl_sos_enclosed_area(st.sos_r[i], st.sos_vr_abs[i]; min_points=min_sos_points, radius_rtol=radius_rtol)
    end
    dE, energy_centers, energy_widths = _karl_phase_energy_widths(st; singleton_width=singleton_energy_width)
    dLz, lz_centers, lz_widths = _karl_phase_lz_widths(st; singleton_width=singleton_lz_width)
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
        error("Karl phase-volume calculation failed for $(length(invalid_required)) " * "recorded base orbit(s): [$preview$suffix]")
    end
    delta_sos_area = fill(NaN, n)
    interior_mask = valid_mask .& .!circular_boundary_mask
    duplicate_clusters, duplicate_orbits, nested_groups = _karl_phase_nested_area_differences!(delta_sos_area, sos_area, interior_mask, st.energy_index, st.lz_index; duplicate_rtol=duplicate_rtol)
    circular_boundary_widths_assigned = _karl_phase_assign_circular_boundary_widths!(delta_sos_area, sos_area, circular_boundary_mask, valid_mask, st.energy_index, st.lz_index)
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
        error( "Karl phase-volume product is invalid for $(length(invalid_after_product)) " * "recorded base orbit(s): [$preview$suffix]")
    end
    phase_volume, mean_log_normalization = _karl_phase_normalize(raw_phase_volume, valid_mask, normalization)
    wphase = fill(NaN, n)
    @inbounds for i in 1:n
        if valid_mask[i]
            wphase[i] = 1.0 / phase_volume[i]
        end
    end
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
    return (raw_phase_volume_base=raw_phase_volume, phase_volume_base=phase_volume, wphase_base=wphase, raw_phase_volume_paired=raw_phase_volume_paired, phase_volume_paired=phase_volume_paired,
        wphase_paired=wphase_paired, valid_base=valid_mask, valid_paired=valid_paired, sos_area=sos_area, delta_sos_area=delta_sos_area, dE=dE, dLz=dLz, diagnostics=diagnostics)
end

#build_karl_wphase(st; kwargs...)
#Return only the planned prograde/retrograde inverse phase-volume vector and its
#diagnostics.  This is the direct handoff to `solve_weights_karl_expanded_cm`.

function build_karl_wphase(st::KarlPhaseVolumeState; kwargs...)
    result = compute_karl_phase_volumes(st; kwargs...)
    return result.wphase_paired, result.diagnostics
end

#compact_karl_wphase(wphase_paired, successful_columns, planned_norbit)
#Apply the exact successful-column mask used for `A_losvd` and `A_light`.

function compact_karl_wphase( wphase_paired::AbstractVector{<:Real}, successful_columns::AbstractVector{<:Integer}, planned_norbit::Int)
    length(wphase_paired) == planned_norbit ||
        error("wphase length $(length(wphase_paired)) does not match planned Norbit=$planned_norbit")
    compacted = Float64.(wphase_paired[successful_columns])
    all(isfinite, compacted) || error("compacted Karl wphase contains nonfinite values")
    all(>(0.0), compacted) || error("compacted Karl wphase contains non-positive values")
    return compacted
end

# -----------------------------------------------------------------------------
# Deterministic numerical self-check
# -----------------------------------------------------------------------------

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
