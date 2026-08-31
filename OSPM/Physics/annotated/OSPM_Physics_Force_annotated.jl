# ============================================================
# OSPM_Physics_Force.jl — force and stellar/halo model machinery.
# Included by OSPM_Physics_Support.jl — do NOT load directly.
#
# Contains the halo density tables, stellar-light model readers,
# spherical and axisymmetric stellar-force construction, potential/force
# closures, halo-context caching, and direct force/mass diagnostics.
# Function names and public contracts are preserved.
# ============================================================
# ============================================================
# §4  HALO PHYSICS
# ============================================================

# WHAT: Convert spherical position (r, theta) into the halo's ellipsoidal radius m.
# HOW: Stretch or compress the direction perpendicular to the symmetry axis by qdm.
# WHY: A flattened halo should have density surfaces labeled by ellipsoids rather than spheres.
# NOTE: qdm is forced away from zero so the coordinate conversion cannot divide by zero.
@inline function karl_m_ellipsoidal(r::Float64, theta::Float64, qdm::Float64)
    q = max(abs(qdm), 1e-6)
    cth = cos(theta)
    sth = sin(theta)
    return r * sqrt(cth * cth + (sth * sth) / (q * q))
end

# WHAT: Return Karl's Dehnen/Plummer halo density at one position.
# HOW: First convert the point to ellipsoidal radius m. gamma!=0 uses the Dehnen form; gamma=0 uses the Plummer form.
# WHY: This keeps the older Karl halo families available behind one density interface.
# NOTE: The returned number is treated downstream as Msun/pc^3.
@inline function karl_halo_density_dehnen_plummer(r::Float64, theta::Float64; qdm::Float64, xmgamma::Float64, rsgamma_pc::Float64, gamma::Float64)
    m = max(karl_m_ellipsoidal(r, theta, qdm), 1e-30)
    a = max(rsgamma_pc, 1e-30)
    if gamma != 0.0
        return xmgamma / (4.0 * pi) * (3.0 - gamma) * a / (m^gamma * (a + m)^(4.0 - gamma))
    else
        return 3.0 * xmgamma / (4.0 * pi * a^3) * (1.0 + (m / a)^2)^(-2.5)
    end
end

# WHAT: Return Karl's concentration-parameterized NFW density at one position.
# HOW: Convert concentration into the usual overdensity factor, then evaluate the NFW 1/[x(1+x)^2] profile.
# WHY: This reproduces the Karl-style NFW parameterization using concentration and scale radius.
# NOTE: The returned number is treated downstream as Msun/pc^3.
@inline function karl_halo_density_nfw_concentration(r::Float64, theta::Float64; qdm::Float64, cnfw::Float64, rsnfw_pc::Float64, hparam::Float64=70.0)
    m = max(karl_m_ellipsoidal(r, theta, qdm), 1e-30)
    rs = max(rsnfw_pc, 1e-30)
    xhparam = hparam / 100.0
    rhocrit = 2.7754996776e-7 * xhparam^2
    c = max(cnfw, 1e-30)
    xd = 200.0 / 3.0 * c^3 / (log(1.0 + c) - c / (1.0 + c))
    x = m / rs
    return rhocrit * xd / (x * (1.0 + x)^2 + 1e-30)
end

# WHAT: Return Karl's nonsingular isothermal spheroid density at one position.
# HOW: Reproduce the coordinate and normalization convention used by Karl halodens.f.
# WHY: This is the direct compatibility form for Karl's flattened isothermal halo density.
# NOTE: xR=r*cos(theta) and xZ=r*sin(theta) here are Karl's convention, not the later spherical-coordinate convention.
@inline function karl_halo_density_isothermal_spheroid(r::Float64, theta::Float64; qdm::Float64, v0::Float64, rc_pc::Float64, dis::Float64)
    q = max(abs(qdm), 1e-6)
    rc = max(rc_pc, 1e-30)
    # Karl halodens.f convention:
    # xR = r*cos(theta), xZ = r*sin(theta)
    xR = r * cos(theta)
    xZ = r * sin(theta)
    xrho = 0.78722918 / (dis * dis)
    xrho *= v0 * v0 / (q * q)
    num = (2.0 * q * q + 1.0) * rc * rc + xR * xR + 2.0 * (1.0 - 0.5 / (q * q)) * xZ * xZ
    den = (rc * rc + xR * xR + xZ * xZ / (q * q))^2
    return xrho * num / max(den, 1e-30)
end

# WHAT: Evaluate the active nonsingular-isothermal halo density in cylindrical coordinates.
# HOW: Use R, z, v0, core radius, and halo flattening q in the analytic density formula.
# WHY: The force-table machinery needs a density value at arbitrary physical positions.
# NOTE: This path uses SI quantities: R,z,rc are meters, v0 is m/s, and the result is kg/m^3.
@inline function nonsingular_isothermal_density_cylindrical( R::Float64, z::Float64, halo)
    q = halo_q_axis_ratio(halo)
    v0 = f64(halo[:v0_ms])
    rc = max(f64(halo[:rc_m]), 1e-30)
    q2 = q * q
    R2 = R * R
    z2 = z * z
    rc2 = rc * rc
    numerator = (2.0 * q2 + 1.0) * rc2 + R2 + (2.0 - 1.0 / q2) * z2
    denominator = (rc2 + R2 + z2 / q2)^2
    return (v0 * v0 / (4.0 * pi * G * q2)) *
           numerator / max(denominator, 1e-30)
end

# WHAT: Pack Karl-style halo parameters into the dictionary format used by this file.
# HOW: Store the selected halo family and every Karl parameter under Symbol keys.
# WHY: Later density, cache, and compatibility functions expect one common halo object.
# NOTE: This only packages parameters. It does not build a force table or evaluate a density.
function karl_halo_from_params(; ihalo::Int=4, qdm::Float64=1.0, dis::Float64=1.0, v0::Float64=0.0, rc_pc::Float64=1.0, xmgamma::Float64=0.0, rsgamma_pc::Float64=1.0, gamma::Float64=1.0, cnfw::Float64=1.0, rsnfw_pc::Float64=1.0, gdennorm::Float64=1.0)
    return Dict{Symbol,Any}( :type => :karl_halo, :ihalo => ihalo, :qdm => qdm, :dis => dis, :v0 => v0, :rc_pc => rc_pc, :xmgamma => xmgamma, :rsgamma_pc => rsgamma_pc, :gamma => gamma, :cnfw => cnfw, :rsnfw_pc => rsnfw_pc, :gdennorm => gdennorm)
end

# WHAT: Build a hash signature describing a Karl halo and its force-table settings.
# HOW: Normalize the halo dictionary, collect all fields that can change its physics/numerics, then hash them together.
# WHY: Cached halo products must only be reused when the underlying halo definition is truly the same.
# NOTE: Any parameter that affects the halo but is missing from this hash could cause an incorrect cache hit.
@inline function karl_halo_sig(halo)
    h = normalize_halo(halo)
    return hash((get(h, :type, nothing), get(h, :ihalo, nothing), get(h, :qdm, nothing), get(h, :dis, nothing), get(h, :v0, nothing), get(h, :rc_pc, get(h, :rc, nothing)), get(h, :xmgamma, nothing), get(h, :rsgamma_pc, get(h, :rsgamma, nothing)), get(h, :gamma, nothing), get(h, :cnfw, nothing), get(h, :rsnfw_pc, get(h, :rsnfw, nothing)), get(h, :gdennorm, nothing), get(h, :halo_force_nR, nothing), get(h, :halo_force_nZ, nothing), get(h, :halo_force_nphi, nothing), get(h, :halo_force_nm, nothing), get(h, :halo_force_ntheta, nothing), get(h, :halo_force_softening_pc, nothing)))
end

# WHAT: Convert cylindrical R,z into the angle convention expected by Karl's halo-density routines.
# HOW: Compute atan(|z|,|R|), so theta=0 lies in the cylindrical R direction.
# WHY: Karl halodens.f defines xR=r*cos(theta) and xZ=r*sin(theta).
# NOTE: This angle convention differs from the standard spherical theta used elsewhere in this file.
@inline function _theta_from_cylindrical_Rz(R::Float64, z::Float64)
    # Karl's halodens.f uses xR = r*cos(theta), xZ = r*sin(theta).
    # This helper therefore maps cylindrical R,z to that convention:
    #   xR -> cylindrical R
    #   xZ -> vertical z
    return atan(abs(z), max(abs(R), 1e-300))
end

# WHAT: Evaluate a Karl halo density from cylindrical coordinates.
# HOW: Convert R,z to Karl's (r,theta) convention, then call the Karl density dispatcher.
# WHY: Axisymmetric grid builders work naturally in cylindrical coordinates.
# NOTE: The unit conversion to SI happens inside rho_interp_karl_halo.
@inline function rho_karl_halo_cylindrical(R::Float64, z::Float64, halo)
    r = sqrt(R * R + z * z)
    theta = _theta_from_cylindrical_Rz(R, z)
    return rho_interp_karl_halo((r, theta), halo)
end

# WHAT: Dispatch to the requested Karl halo-density formula.
# HOW: Read ihalo, evaluate Dehnen/Plummer, NFW, isothermal, or no-halo density, then convert Msun/pc^3 to SI.
# WHY: The rest of the force machinery needs one density function regardless of which Karl halo family is selected.
# NOTE: ihalo=4 means zero halo density. Unknown values stop with an error.
function rho_interp_karl_halo(rv, halo)
    r = abs(f64(rv[1]))
    theta = length(rv) >= 2 ? f64(rv[2]) : pi / 2
    ihalo = Int(get(halo, :ihalo, 4))
    qdm = f64(get(halo, :qdm, get(halo, :halo_q_axis_ratio, 1.0)))
    if ihalo == 1
        rho_msun_pc3 = karl_halo_density_dehnen_plummer(r, theta; qdm=qdm, xmgamma=f64(get(halo, :xmgamma, 0.0)), rsgamma_pc=f64(get(halo, :rsgamma_pc, get(halo, :rsgamma, 1.0))), gamma=f64(get(halo, :gamma, 1.0)))
    elseif ihalo == 2
        rho_msun_pc3 = karl_halo_density_nfw_concentration(r, theta; qdm=qdm, cnfw=f64(get(halo, :cnfw, 1.0)), rsnfw_pc=f64(get(halo, :rsnfw_pc, get(halo, :rsnfw, 1.0))))
    elseif ihalo == 3
        rho_msun_pc3 = karl_halo_density_isothermal_spheroid(r, theta; qdm=qdm, v0=f64(get(halo, :v0, 0.0)), rc_pc=f64(get(halo, :rc_pc, get(halo, :rc, 1.0))), dis=f64(get(halo, :dis, 1.0)))
    elseif ihalo == 4
        rho_msun_pc3 = 0.0
    else
        error("Unknown Karl ihalo value: $ihalo")
    end
    return rho_msun_pc3 * Msun / pc^3
end

# WHAT: General halo-density dispatcher for all supported halo families.
# HOW: Read halo[:type], evaluate that profile at radius r, and return the density in the units expected by the force integrator.
# WHY: tables_spherical and other grid builders should not need separate code for every halo model.
# NOTE: For nonsingular_isothermal this calls the cylindrical formula on the equatorial plane when used spherically.
function rho_interp(rv, halo)
    halo[:type] === :karl_halo &&
        return rho_interp_karl_halo(rv, halo)
    r    = abs(rv[1])
    rhos = halo[:rho_s]
    rs   = halo[:r_s]
    x    = r / max(rs, 1e-30)
    halo[:type] === :none &&
        return 0.0
    halo[:type] === :nfw &&
        return rhos / (x * (1 + x)^2 + 1e-30)
    halo[:type] === :cored &&
        return rhos / ((1 + x) * (1 + x^2) + 1e-30)
    halo[:type] === :nonsingular_isothermal &&
        return nonsingular_isothermal_density_cylindrical(r, 0.0, halo)
    halo[:type] === :einasto && begin
        α = halo[:alpha]          # curvature parameter
        return rhos * exp(-2/α * (x^α - 1))
    end
    error("Unknown halo type: $(halo[:type])")
end

# WHAT: Return the halo axis ratio q in a safe, normalized form.
# HOW: Read qdm for Karl halos or halo_q_axis_ratio for normal halos, then enforce |q|>=1e-6.
# WHY: Many axisymmetric formulas divide by q, so q cannot be allowed to reach zero.
# NOTE: q=1 is spherical. Values different from 1 describe a flattened or elongated halo.
@inline function halo_q_axis_ratio(halo)
    if get(halo, :type, :none) === :karl_halo
        q = haskey(halo, :qdm) ? f64(halo[:qdm]) : (haskey(halo, :halo_q_axis_ratio) ? f64(halo[:halo_q_axis_ratio]) : 1.0)
        return max(abs(q), 1e-6)
    end
    q = haskey(halo, :halo_q_axis_ratio) ? f64(halo[:halo_q_axis_ratio]) : 1.0
    return max(abs(q), 1e-6)
end

# WHAT: Compute ellipsoidal radius for an axisymmetric halo.
# HOW: Use m=sqrt(R^2+(z/q)^2).
# WHY: Spherical density laws can be generalized onto spheroidal density surfaces through m.
# NOTE: This is a geometry helper only; it does not evaluate density or force.
@inline function halo_m_axisym(R::Float64, z::Float64, halo)
    q = halo_q_axis_ratio(halo)
    return sqrt(R * R + (z / q) * (z / q))
end

# WHAT: Evaluate halo density at an axisymmetric cylindrical position.
# HOW: Use the dedicated Karl or nonsingular-isothermal formula when needed; otherwise evaluate the selected profile at ellipsoidal radius m.
# WHY: Axisymmetric mass-grid construction needs one density interface for every supported halo family.
# NOTE: The active production path currently forbids flattened halos later in halo_from_theta until force and launch potential are paired consistently.
@inline function rho_halo_axisym(R::Float64, z::Float64, halo)
    if get(halo, :type, :none) === :karl_halo
        return rho_karl_halo_cylindrical(R, z, halo)
    end
    if get(halo, :type, :none) === :nonsingular_isothermal
        return nonsingular_isothermal_density_cylindrical(R, z, halo)
    end
    m = halo_m_axisym(R, z, halo)
    return rho_interp((m, 0.0), halo)
end

# WHAT: Normalize a stellar-model dictionary so all keys are Julia Symbols.
# HOW: Copy every key/value pair into a new Dict{Symbol,Any}.
# WHY: Config data may arrive with String keys while the Julia physics code expects Symbol keys.
# NOTE: Values are not changed here; only the key representation is normalized.
@inline function normalize_stellar_model(stellar_model)
    stellar_model === nothing && return nothing
    out = Dict{Symbol,Any}()
    for (k, v) in stellar_model
        ks = k isa Symbol ? k : Symbol(String(k))
        out[ks] = v
    end
    return out
end

# WHAT: Read the header of a Karl-style stellar-grid CSV and provide a helper for loading numeric columns.
# HOW: Read all lines, map column names to positions, then return that map plus the nested colfloat reader.
# WHY: Several stellar-grid builders need flexible column access without depending on a heavier CSV package.
# NOTE: Missing/unparseable numeric entries become NaN so later validation can reject them.
function _read_karl_light_grid(path::String)
    lines = readlines(path)
    isempty(lines) && error("empty karl_light_grid CSV: $path")
    header = split(strip(lines[1]), ",")
    idx = Dict(Symbol(strip(h)) => i for (i, h) in enumerate(header))

    # WHAT: Read one named numeric column from the already-loaded stellar-grid CSV.
    # HOW: Find the column index, parse each row as Float64, and insert NaN for values that cannot be parsed.
    # WHY: The outer grid reader can request only the physical columns it needs.
    # NOTE: A completely missing column is an error; a bad individual cell becomes NaN.
    function colfloat(name::Symbol)
        haskey(idx, name) || error("karl_light_grid missing column: $(name)")
        out = Float64[]
        j = idx[name]
        for line in lines[2:end]
            isempty(strip(line)) && continue
            vals = split(line, ",")
            raw = j <= length(vals) ? strip(vals[j]) : ""
            x = tryparse(Float64, raw)
            push!(out, x === nothing ? NaN : x)
        end
        return out
    end
    return idx, colfloat
end

# WHAT: Build a cache signature for the stellar model.
# HOW: Hash the model type, geometry, source grid, normalization, column choices, flattening, and force-table settings.
# WHY: Unit-M/L stellar force tables can be safely reused only when every relevant stellar-model setting matches.
# NOTE: The signature is about cache identity, not a physical observable.
@inline function stellar_model_sig(stellar_model)
    stellar_model === nothing && return UInt(0)
    sm = normalize_stellar_model(stellar_model)
    stype = Symbol(lowercase(String(get(sm, :type, :none))))
    geom  = Symbol(lowercase(String(get(sm, :geometry, :spherical_shell_grid))))
    if stype === :plummer
        return hash((stype, geom, get(sm, :Ltot, nothing), get(sm, :a_pc, nothing)))
    elseif stype === :karl_light_grid
        return hash((stype, geom, get(sm, :grid_csv, nothing), get(sm, :Ltot, nothing), get(sm, :radius_col, nothing),
        get(sm, :theta_col, nothing), get(sm, :nu_col, nothing), get(sm, :lenc_frac_col, nothing), get(sm, :R_cyl_col, nothing),
        get(sm, :z_col, nothing), get(sm, :volume_col, nothing), get(sm, :luminosity_col, nothing), get(sm, :q_axis_ratio, nothing),
        get(sm, :force_softening_pc, nothing), get(sm, :force_nR, nothing), get(sm, :force_nZ, nothing), get(sm, :force_nphi, nothing)))
    end
    return hash((stype, geom, get(sm, :Ltot, nothing)))
end

# WHAT: Linearly interpolate a value on a one-dimensional grid.
# HOW: Clamp outside the grid to the nearest endpoint, then blend between the two surrounding samples.
# WHY: Enclosed-light fractions are tabulated at finite radii while the orbit code asks for arbitrary radii.
# NOTE: This is endpoint clamping, not extrapolation beyond the supplied grid.
@inline function _interp_linear_grid(xs::Vector{Float64}, ys::Vector{Float64}, x::Float64)
    n = length(xs)
    n == length(ys) || error("grid interpolation arrays have different lengths")
    n == 0 && return 0.0
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    j = searchsortedlast(xs, x)
    j = clamp(j, 1, n - 1)
    t = (x - xs[j]) / max(xs[j + 1] - xs[j], 1e-30)
    return (1.0 - t) * ys[j] + t * ys[j + 1]
end

# WHAT: Return stellar mass enclosed within spherical radius r for a spherical Karl light grid.
# HOW: Interpolate the enclosed-light fraction, multiply by total luminosity and M/L, then convert to kilograms.
# WHY: A spherical stellar force can be obtained directly from enclosed mass.
# NOTE: This helper assumes spherical geometry. Axisymmetric grids use force tables instead.
@inline function stellar_Menc_karl_light_grid(r::Float64, ML::Float64, grid)
    rr = max(r, 1e-30)
    f = _interp_linear_grid(grid.R_m, grid.Lenc_frac, rr)
    return ML * grid.Ltot * f * Msun
end

# WHAT: Compute the spherical gravitational potential of a tabulated stellar light profile.
# HOW: Combine mass already enclosed inside r with the potential contribution from all outer shells.
# WHY: Orbit launch energies need the stellar potential, not only the stellar force.
# NOTE: The calculation assumes spherical shells and piecewise-linear enclosed-light fraction; it must not be used for axisymmetric_density_grid.
@inline function stellar_Phi_karl_light_grid( r::Float64, ML::Float64, grid)
    rr = max(r, 1e-30)
    radii = grid.R_m
    fractions = grid.Lenc_frac
    n = length(radii)
    n == length(fractions) ||
        error("stellar light-grid arrays have different lengths")
    n >= 2 ||
        error("stellar light grid requires at least two radial points")
    enclosed_fraction = _interp_linear_grid( radii, fractions, rr)
    outer_shell_integral = 0.0
    if rr <= radii[1]
        @inbounds for i in 1:(n - 1)
            slope = ( fractions[i + 1] - fractions[i]) / ( radii[i + 1] - radii[i])
            outer_shell_integral += slope * log( radii[i + 1] / radii[i])
        end
    elseif rr < radii[end]
        i = clamp( searchsortedlast(radii, rr), 1, n - 1)
        slope = (fractions[i + 1] - fractions[i]) / (radii[i + 1] - radii[i])
        outer_shell_integral += slope * log( radii[i + 1] / rr)
        @inbounds for j in (i + 1):(n - 1)
            slope = (fractions[j + 1] - fractions[j]) / ( radii[j + 1] - radii[j])
            outer_shell_integral += slope * log( radii[j + 1] / radii[j] )
        end
    end
    stellar_mass_scale = ML * grid.Ltot * Msun
    return -G * stellar_mass_scale * ( enclosed_fraction / rr + outer_shell_integral)
end

# ============================================================
# §4a  STELLAR MODEL GEOMETRY HELPERS
# ============================================================

# WHAT: Return the declared stellar geometry as a normalized Symbol.
# HOW: Normalize the model dictionary and read :geometry, defaulting to spherical_shell_grid.
# WHY: Force construction must choose between spherical enclosed-mass logic and axisymmetric force tables.
# NOTE: A missing stellar model returns :none.
@inline function stellar_model_geometry(stellar_model)
    stellar_model === nothing && return :none
    sm = normalize_stellar_model(stellar_model)
    return Symbol(lowercase(String(get(sm, :geometry, :spherical_shell_grid))))
end

# WHAT: Return the declared stellar-model type as a normalized Symbol.
# HOW: Normalize the model dictionary and read :type.
# WHY: Later code must choose between Plummer and Karl light-grid machinery.
# NOTE: A missing stellar model returns :none.
@inline function stellar_model_type(stellar_model)
    stellar_model === nothing && return :none
    sm = normalize_stellar_model(stellar_model)
    return Symbol(lowercase(String(get(sm, :type, :none))))
end

# WHAT: Answer whether the stellar model uses the axisymmetric density-grid geometry.
# HOW: Compare stellar_model_geometry(...) with :axisymmetric_density_grid.
# WHY: Diagnostics and force helpers need a simple guard against applying spherical formulas to a flattened model.
# NOTE: This is only a geometry test.
@inline function is_axisymmetric_stellar_model(stellar_model)
    geom = stellar_model_geometry(stellar_model)
    return geom === :axisymmetric_density_grid
end

# WHAT: Protect spherical stellar-force code from receiving an axisymmetric density grid.
# HOW: Check the declared geometry and throw an error if it is axisymmetric_density_grid.
# WHY: Using spherical enclosed-light formulas on a flattened density field would give the wrong gravity.
# NOTE: Successful return means only that this particular forbidden geometry was not declared.
@inline function require_spherical_stellar_geometry(stellar_model)
    geom = stellar_model_geometry(stellar_model)
    if geom === :axisymmetric_density_grid
        error(
            "axisymmetric_density_grid was declared, but this path uses the spherical enclosed-light force. " *
            "Use the axisymmetric force-table path instead."
        )
    end
    return nothing
end

# WHAT: Read a config value as Float64 or return a supplied default.
# HOW: Try parsing the value through String first, then fall back to the shared f64 conversion.
# WHY: Config values can arrive as numeric objects or text.
# NOTE: This is a convenience helper; it does not validate the physical range of the value.
@inline function _get_float_or_default(sm, key::Symbol, default::Float64)
    haskey(sm, key) || return default
    x = tryparse(Float64, String(sm[key]))
    x === nothing ? f64(sm[key]) : x
end

# ============================================================
# §4b  AXISYMMETRIC STELLAR GRID + FORCE TABLE
# ============================================================

# WHAT: Turn the stellar CSV into the normalized axisymmetric luminous-cell model used for force calculations.
# HOW: Read R,z and cell luminosities, reconstruct luminosity from density*volume if needed, reject bad cells, then renormalize all cells to Ltot.
# WHY: The stellar gravity should follow the observed/deprojected stellar shape while keeping the config's total luminosity normalization.
# NOTE: Returned positions are converted from pc to meters. L_cell stays in Lsun until M/L is applied later.
function build_axisymmetric_light_grid_model(stellar_model)
    sm = normalize_stellar_model(stellar_model)
    geom = stellar_model_geometry(sm)
    geom === :axisymmetric_density_grid || error("build_axisymmetric_light_grid_model requires geometry='axisymmetric_density_grid'")
    path = String(sm[:grid_csv])
    _, colfloat = _read_karl_light_grid(path)
    Rcol = Symbol(String(get(sm, :R_cyl_col, "R_cyl_pc")))
    zcol = Symbol(String(get(sm, :z_col, "z_pc")))
    ncol = Symbol(String(get(sm, :nu_col, "nu_Lsun_pc3")))
    vcol = Symbol(String(get(sm, :volume_col, "cell_volume_pc3")))
    lcol = Symbol(String(get(sm, :luminosity_col, "cell_luminosity_Lsun")))
    R_pc = colfloat(Rcol)
    z_pc = colfloat(zcol)
    has_luminosity = true
    L_cell = Float64[]
    try
        L_cell = colfloat(lcol)
    catch
        has_luminosity = false
    end
    if !has_luminosity
        nu = colfloat(ncol)
        vol = colfloat(vcol)
        length(nu) == length(vol) || error("axisymmetric grid nu and volume lengths do not match")
        L_cell = nu .* vol
    end
    length(R_pc) == length(z_pc) == length(L_cell) || error("axisymmetric grid R, z, and luminosity lengths do not match")
    good = isfinite.(R_pc) .& isfinite.(z_pc) .& isfinite.(L_cell) .& (L_cell .>= 0.0)
    R_pc = R_pc[good]
    z_pc = z_pc[good]
    L_cell = L_cell[good]
    length(R_pc) > 0 || error("axisymmetric grid contains no valid luminous cells")
    Ltot = f64(sm[:Ltot])
    Lsum = sum(L_cell)
    (!isfinite(Lsum) || Lsum <= 0.0) && error("axisymmetric grid luminosity sum is non-positive")
    L_cell .*= Ltot / Lsum
    q = haskey(sm, :q_axis_ratio) ? f64(sm[:q_axis_ratio]) : 1.0
    soft_pc = haskey(sm, :force_softening_pc) ? f64(sm[:force_softening_pc]) : 0.5
    return (R_m=Float64.(R_pc) .* pc, z_m=Float64.(z_pc) .* pc, L_cell=Float64.(L_cell), q=q, soft_m=soft_pc * pc, Ltot=Ltot)
end

# WHAT: Directly sum the gravitational force from an axisymmetric set of mass cells at one field point.
# HOW: Treat each (R,z) cell as a ring, split that ring into nphi azimuthal pieces, then sum their softened Newtonian gravity.
# WHY: This supplies the reference force/potential used to build fast interpolation tables for flattened stellar or halo components.
# NOTE: soft_m prevents a numerical singularity when a field point lies extremely close to one discretized mass element.
function _axisym_force_from_mass_cells(Rf::Float64, zf::Float64, R_cells::Vector{Float64}, z_cells::Vector{Float64}, M_cells::Vector{Float64}, soft_m::Float64; nphi::Int=32, return_potential::Bool=false)
    FR = 0.0
    FZ = 0.0
    Phi = 0.0
    soft2 = soft_m * soft_m
    dphi = 2.0 * pi / nphi
    @inbounds for i in eachindex(R_cells)
        Rs = R_cells[i]
        zs = z_cells[i]
        dm = M_cells[i]
        if !(isfinite(dm) && dm > 0.0)
            continue
        end
        for k in 1:nphi
            phi = (k - 0.5) * dphi
            cp = cos(phi)
            dxR = Rf - Rs * cp
            dz = zf - zs
            d2 = Rf * Rf + Rs * Rs - 2.0 * Rf * Rs * cp + dz * dz + soft2
            invd = 1.0 / sqrt(d2)
            invd3 = invd / d2
            mass_fraction = dm / nphi
            Phi += -G * mass_fraction * invd
            FR += -G * mass_fraction * dxR * invd3
            FZ += -G * mass_fraction * dz * invd3
        end
    end
    if return_potential
        return FR, FZ, Phi
    end
    return FR, FZ
end

# WHAT: Convenience wrapper that turns luminous stellar cells into mass cells and evaluates their axisymmetric force.
# HOW: Multiply every cell luminosity by M/L and Msun, then call _axisym_force_from_mass_cells.
# WHY: Stellar grids are stored in light units while gravity requires mass.
# NOTE: This computes force directly and does not use the cached interpolation table.
function _axisym_force_from_cells(Rf::Float64, zf::Float64, ML::Float64, grid; nphi::Int=32)
    M_cells = Float64.(ML .* grid.L_cell .* Msun)
    return _axisym_force_from_mass_cells(Rf, zf, grid.R_m, grid.z_m, M_cells, grid.soft_m; nphi=nphi)
end

# WHAT: Build a logarithmically spaced radial axis for a force table.
# HOW: Enforce a positive lower bound and a sensible upper/lower separation, then call logspace10.
# WHY: Gravity changes rapidly near the center, so logarithmic spacing gives more resolution there.
# NOTE: The returned axis never includes exactly zero.
function _make_force_axis(minval::Float64, maxval::Float64, n::Int)
    lo = max(minval, 1e-8)
    hi = max(maxval, 10.0 * lo)
    return logspace10(log10(lo), log10(hi), n)
end

# WHAT: Precompute stellar FR, FZ, and potential on a 2D (R,z) grid.
# HOW: Build table axes, convert luminous cells to mass, directly sum each table point, and parallelize independent R rows across Julia threads.
# WHY: Orbit integration needs millions of force evaluations; interpolating a table is far cheaper than resumming every stellar cell each time.
# NOTE: This is the expensive one-time calculation later cached at unit M/L because stellar gravity and potential scale linearly with M/L.
function build_axisymmetric_force_table( grid, ML::Float64; nR::Int=96, nZ::Int=96, nphi::Int=32)
    Rmax = maximum(grid.R_m)
    zmax = maximum(abs.(grid.z_m))
    rmax = max(Rmax, zmax, 1.0 * pc)
    R_axis = _make_force_axis( 1e-4 * pc, 2.0 * rmax, nR,)
    z_axis = collect( range(0.0, 2.0 * rmax; length=nZ))
    FR = zeros(Float64, nR, nZ)
    FZ = zeros(Float64, nR, nZ)
    Phi = zeros(Float64, nR, nZ)
    M_cells = Float64.( ML .* grid.L_cell .* Msun)
    # Every table cell is independent.  This table is now built only once for
    # a given stellar model, so use the full Julia pool for that one-time cost.
    Threads.@threads :dynamic for i in 1:nR
        Rf = R_axis[i]
        @inbounds for j in 1:nZ
            zf = z_axis[j]
            fr, fz, phi =
                _axisym_force_from_mass_cells( Rf, zf, grid.R_m, grid.z_m, M_cells, grid.soft_m; nphi=nphi, return_potential=true)
            FR[i, j] = fr
            FZ[i, j] = fz
            Phi[i, j] = phi
        end
    end
    return ( R_axis=R_axis, z_axis=z_axis, FR=FR, FZ=FZ, Phi=Phi, nR=nR, nZ=nZ, nphi=nphi)
end

# WHAT: Discretize an axisymmetric halo density field into finite mass cells.
# HOW: Divide ellipsoidal radius and polar angle into cells, evaluate density at each cell center, multiply by cell volume, and retain positive finite masses.
# WHY: The same ring-summation machinery used for stellar gravity can then be applied to a flattened halo.
# NOTE: This path exists for axisymmetric halos, but flattened halo use is currently blocked in halo_from_theta for force/potential consistency.
function build_axisymmetric_halo_mass_grid(halo; n_m::Int=128, ntheta::Int=64, rmax_factor::Float64=DEFAULT_RMAX_FACTOR, softening_pc::Float64=0.5)
    q = halo_q_axis_ratio(halo)
    mmin = max(f64(halo[:rmin]), 1e-4 * pc)
    mmax = max(rmax_factor * f64(halo[:rs]), 10.0 * mmin)
    m_edges = logspace10(log10(mmin), log10(mmax), n_m + 1)
    theta_edges = collect(range(0.0, pi; length=ntheta + 1))
    R_cells = Float64[]
    z_cells = Float64[]
    M_cells = Float64[]
    @inbounds for im in 1:n_m
        m0 = m_edges[im]
        m1 = m_edges[im + 1]
        m  = 0.5 * (m0 + m1)
        shell_volume = (4.0 * pi / 3.0) * q * (m1^3 - m0^3)
        for it in 1:ntheta
            th0 = theta_edges[it]
            th1 = theta_edges[it + 1]
            th = 0.5 * (th0 + th1)
            theta_fraction = abs(cos(th0) - cos(th1)) / 2.0
            cell_volume = shell_volume * theta_fraction
            Rcell = m * sin(th)
            zcell = q * m * cos(th)
            rho = rho_halo_axisym(Rcell, zcell, halo)
            dm = rho * cell_volume
            if isfinite(dm) && dm > 0.0
                push!(R_cells, Rcell)
                push!(z_cells, zcell)
                push!(M_cells, dm)
            end
        end
    end
    length(M_cells) > 0 || error("axisymmetric halo mass grid contains no valid mass cells")
    return (R_m=R_cells, z_m=z_cells, M_cell=M_cells, q=q, soft_m=softening_pc * pc, rmax_m=mmax)
end

# WHAT: Precompute the cylindrical force of a discretized axisymmetric halo on an (R,z) table.
# HOW: Build halo mass cells, create force-table axes, then directly sum FR and FZ at every table point.
# WHY: A flattened halo cannot be represented by a purely spherical enclosed-mass force.
# NOTE: This table currently stores FR/FZ but not Phi. Flattened halos are disabled upstream until the force and launch-potential treatments are made consistent.
function build_axisymmetric_halo_force_table(halo; nR::Int=96, nZ::Int=96, nphi::Int=32, n_m::Int=128, ntheta::Int=64, softening_pc::Float64=0.5, rmax_factor::Float64=DEFAULT_RMAX_FACTOR)
    grid = build_axisymmetric_halo_mass_grid(halo; n_m=n_m, ntheta=ntheta, rmax_factor=rmax_factor, softening_pc=softening_pc)
    Rmax = maximum(grid.R_m)
    zmax = maximum(abs.(grid.z_m))
    rmax = max(Rmax, zmax, grid.rmax_m, 1.0 * pc)
    R_axis = _make_force_axis(1e-4 * pc, 2.0 * rmax, nR)
    z_axis = collect(range(0.0, 2.0 * rmax; length=nZ))
    FR = zeros(Float64, nR, nZ)
    FZ = zeros(Float64, nR, nZ)
    @inbounds for i in 1:nR
        Rf = R_axis[i]
        for j in 1:nZ
            zf = z_axis[j]
            fr, fz = _axisym_force_from_mass_cells(Rf, zf, grid.R_m, grid.z_m, grid.M_cell, grid.soft_m; nphi=nphi)
            FR[i, j] = fr
            FZ[i, j] = fz
        end
    end
    return ( R_axis = R_axis, z_axis = z_axis, FR = FR, FZ = FZ, nR = nR, nZ = nZ, nphi = nphi, q = grid.q )
end

# WHAT: Bilinearly interpolate potential from an axisymmetric (R,z) table.
# HOW: Clamp the requested point to the table domain and blend the four surrounding Phi samples.
# WHY: Orbit launch energy needs potential at arbitrary positions without recomputing the full mass-cell sum.
# NOTE: The stellar axisymmetric table contains Phi. The current axisymmetric halo table does not, which is one reason flattened halo use remains disabled.
@inline function _interp_axisym_potential(table, Rf::Float64, zf::Float64)
    R = clamp(abs(Rf), table.R_axis[1], table.R_axis[end])
    z = clamp(abs(zf), table.z_axis[1], table.z_axis[end])
    i = clamp(searchsortedlast(table.R_axis, R), 1, table.nR - 1)
    j = clamp(searchsortedlast(table.z_axis, z), 1, table.nZ - 1)
    R1 = table.R_axis[i]
    R2 = table.R_axis[i + 1]
    z1 = table.z_axis[j]
    z2 = table.z_axis[j + 1]
    t = (R - R1) / max(R2 - R1, 1e-30)
    u = (z - z1) / max(z2 - z1, 1e-30)
    Phi11 = table.Phi[i, j]
    Phi21 = table.Phi[i + 1, j]
    Phi12 = table.Phi[i, j + 1]
    Phi22 = table.Phi[i + 1, j + 1]
    return (1.0 - t) * (1.0 - u) * Phi11 + t * (1.0 - u) * Phi21 + (1.0 - t) * u * Phi12 + t * u * Phi22
end

# WHAT: Bilinearly interpolate cylindrical force components FR and FZ from a force table.
# HOW: Reflect to |z|, clamp to the tabulated domain, blend the four neighboring cells, then restore the sign of FZ.
# WHY: This makes axisymmetric force evaluation fast enough for orbit integration.
# NOTE: Requests outside the table are clamped to its edge rather than extrapolated.
@inline function _interp_axisym_force(table, Rf::Float64, zf::Float64)
    R = max(abs(Rf), table.R_axis[1])
    zsign = zf < 0.0 ? -1.0 : 1.0
    z = abs(zf)
    R = min(R, table.R_axis[end])
    z = min(z, table.z_axis[end])
    i = searchsortedlast(table.R_axis, R)
    j = searchsortedlast(table.z_axis, z)
    i = clamp(i, 1, table.nR - 1)
    j = clamp(j, 1, table.nZ - 1)
    R1 = table.R_axis[i]
    R2 = table.R_axis[i + 1]
    z1 = table.z_axis[j]
    z2 = table.z_axis[j + 1]
    t = (R - R1) / max(R2 - R1, 1e-30)
    u = (z - z1) / max(z2 - z1, 1e-30)
    FR11 = table.FR[i, j]
    FR21 = table.FR[i + 1, j]
    FR12 = table.FR[i, j + 1]
    FR22 = table.FR[i + 1, j + 1]
    FZ11 = table.FZ[i, j]
    FZ21 = table.FZ[i + 1, j]
    FZ12 = table.FZ[i, j + 1]
    FZ22 = table.FZ[i + 1, j + 1]
    FRv = (1.0 - t) * (1.0 - u) * FR11 + t * (1.0 - u) * FR21 + (1.0 - t) * u * FR12 + t * u * FR22
    FZv = (1.0 - t) * (1.0 - u) * FZ11 + t * (1.0 - u) * FZ21 + (1.0 - t) * u * FZ12 + t * u * FZ22
    return FRv, zsign * FZv
end

# WHAT: Rotate cylindrical force components into spherical radial and polar components.
# HOW: Project FR and FZ onto the local r and theta directions.
# WHY: The orbit integrator works with spherical-coordinate force components even when gravity was computed in cylindrical coordinates.
# NOTE: r is not used in the algebra; it is retained in the helper signature for the force-coordinate interface.
@inline function _cyl_force_to_spherical_force(r::Float64, theta::Float64, FR::Float64, FZ::Float64)
    st, ct = _sincos_safe(theta)
    fr = FR * st + FZ * ct
    ftheta = FR * ct - FZ * st
    return fr, ftheta
end

# WHAT: Return the axisymmetric stellar force at spherical position (r,theta).
# HOW: Convert the point to cylindrical R,z, interpolate FR/FZ from the stellar table, then rotate back to fr/ftheta.
# WHY: This bridges the axisymmetric stellar-force table to the spherical-coordinate orbit equations.
# NOTE: The returned force is the table's normalization; the caller applies M/L scaling when the table was cached at unit M/L.
@inline function stellar_force_axisymmetric_spherical(r::Float64, theta::Float64, table)
    rr = max(abs(r), 1e-30)
    st, ct = _sincos_safe(theta)
    Rf = rr * st
    zf = rr * ct
    FR, FZ = _interp_axisym_force(table, Rf, zf)
    return _cyl_force_to_spherical_force(rr, theta, FR, FZ)
end

# WHAT: Return the axisymmetric halo force at spherical position (r,theta).
# HOW: Convert to R,z, interpolate cylindrical force, then rotate to spherical components.
# WHY: This is the force-side bridge needed for a flattened halo.
# NOTE: Flattened halo models are currently disabled upstream until their potential is made consistent with this force.
@inline function halo_force_axisymmetric_spherical(r::Float64, theta::Float64, table)
    rr = max(abs(r), 1e-30)
    st, ct = _sincos_safe(theta)
    Rf = rr * st
    zf = rr * ct
    FR, FZ = _interp_axisym_force(table, Rf, zf)
    return _cyl_force_to_spherical_force(rr, theta, FR, FZ)
end

# WHAT: Return stellar mass enclosed within r for a Plummer sphere.
# HOW: Use the analytic Plummer enclosed-mass expression with total mass=(M/L)*Ltot.
# WHY: Plummer models do not need a numerical stellar force table.
# NOTE: Output is in kilograms because Msun is applied here.
@inline function stellar_Menc_plummer(r::Float64, ML::Float64, Ltot::Float64, a::Float64)
    rr = max(r, 1e-30)
    Mtot = ML * Ltot * Msun
    return Mtot * rr^3 / (rr^2 + a^2)^(3/2)
end

# WHAT: Return the analytic gravitational potential of a Plummer stellar sphere.
# HOW: Evaluate -G*Mtot/sqrt(r^2+a^2).
# WHY: This supplies stellar potential directly for orbit energies when a Plummer model is selected.
# NOTE: r and a are expected in meters.
@inline function stellar_Phi_plummer(r::Float64, ML::Float64, Ltot::Float64, a::Float64)
    rr = max(r, 1e-30)
    Mtot = ML * Ltot * Msun
    return -G * Mtot / sqrt(rr^2 + a^2)
end

# WHAT: Read a simple all-numeric CSV into a dictionary of Float64 columns.
# HOW: Create one vector per header, parse every cell, and store NaN where parsing fails.
# WHY: Some profile-reading paths only need a lightweight numeric table reader.
# NOTE: This reader does not handle quoted commas or general CSV syntax; it expects the simple files produced by this pipeline.
function _read_simple_csv_table(path::String)
    lines = readlines(path)
    isempty(lines) && error("empty stellar profile CSV: $path")
    header = split(strip(lines[1]), ",")
    cols = Dict{Symbol,Vector{Float64}}()
    for h in header
        cols[Symbol(strip(h))] = Float64[]
    end
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ",")

        for (j, h) in enumerate(header)
            key = Symbol(strip(h))
            raw = j <= length(vals) ? strip(vals[j]) : ""
            x = tryparse(Float64, raw)
            push!(cols[key], x === nothing ? NaN : x)
        end
    end
    return cols
end

# WHAT: Perform basic one-dimensional linear interpolation.
# HOW: Clamp x to endpoint values outside the grid and linearly blend between neighboring samples inside it.
# WHY: Several tabulated physical quantities need values at arbitrary radii.
# NOTE: This duplicates the behavior of _interp_linear_grid under a second historical helper name.
function _linear_interp(xs::Vector{Float64}, ys::Vector{Float64}, x::Float64)
    n = length(xs)
    n == length(ys) || error("interp arrays have different lengths")
    n == 0 && return 0.0
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    j = searchsortedlast(xs, x)
    j = clamp(j, 1, n - 1)
    t = (x - xs[j]) / max(xs[j + 1] - xs[j], 1e-30)
    return (1.0 - t) * ys[j] + t * ys[j + 1]
end

# WHAT: Convert one OSPM parameter point into the normalized halo dictionary used by the Julia force code.
# HOW: Interpret theta according to halo_type, convert pc/Msun/km/s inputs into SI fields, attach MBH, M/L, flattening, and stellar-model metadata.
# WHY: The rest of the force machinery needs one unit-consistent internal representation regardless of how the search parameters were named.
# NOTE: For nonsingular_isothermal, rho_s is intentionally interpreted as v0 [km/s] and r_s as core radius [pc], not literal density and scale radius.
# NOTE: karl_halo and q!=1 halo force modes are deliberately stopped here because their current unit or force-potential contracts are not validated.
# NOTE: The later ht===:karl_halo setup block is therefore unreachable unless the earlier safety error is removed in a future repair.
function halo_from_theta(rho_s, r_s, MBH, ML; halo_type="nfw", alpha=nothing, stellar_model=nothing, halo_q_axis_ratio=1.0, karl_halo_params=nothing)
    ht = Symbol(lowercase(String(halo_type)))
    qh = max(abs(f64(halo_q_axis_ratio)), 1e-6)
    if ht === :karl_halo
        error(
            "halo_type='karl_halo' is disabled: its density functions use parsec-valued " *
            "radii, while the current halo-table path supplies radii in meters. Repair and " *
            "validate that unit contract before enabling this mode."
        )
    end
    if abs(qh - 1.0) > 1e-8
        error(
            "Flattened halo forces are disabled: the axisymmetric halo force table is not " *
            "paired with the same axisymmetric potential used for orbit launch energy. " *
            "Use halo_q_axis_ratio=1.0 until that force-potential pair is repaired."
        )
    end
    rs_pc = f64(r_s)
    if ht === :nonsingular_isothermal
        rc_pc = max(rs_pc, 1e-12)
        v0_kms = f64(rho_s)
        h = Dict(
            :rho_s => 0.0,
            :r_s   => rc_pc * pc,
            :rs    => rc_pc * pc,
            :v0_ms => v0_kms * 1.0e3,
            :rc_m  => rc_pc * pc,
            :v0_kms => v0_kms,
            :rc_pc => rc_pc,
            :MBH   => f64(MBH) * Msun,
            :ML    => f64(ML),
            :type  => ht,
            :rmin  => 1e-6 * rc_pc * pc,
            :halo_q_axis_ratio => qh,
        )
    else
        h = Dict(
            :rho_s => f64(rho_s) * Msun / pc^3,
            :r_s   => rs_pc * pc,
            :rs    => rs_pc * pc,
            :MBH   => f64(MBH) * Msun,
            :ML    => f64(ML),
            :type  => ht,
            :rmin  => 1e-6 * rs_pc * pc,
            :halo_q_axis_ratio => qh,
        )
    end
    if ht === :karl_halo
        # Karl halo mode is a real force path.  The theta r_s value supplies the
        # default scale radius in pc.  Specific Karl fields may override through
        # karl_halo_params, but every such value is included in the cache key.
        h[:ihalo] = 2
        h[:qdm] = qh
        h[:cnfw] = max(f64(rho_s), 1e-12)
        h[:rsnfw_pc] = max(rs_pc, 1e-12)
        h[:dis] = 1.0
        h[:v0] = 0.0
        h[:rc_pc] = max(rs_pc, 1e-12)
        h[:xmgamma] = max(f64(rho_s), 0.0)
        h[:rsgamma_pc] = max(rs_pc, 1e-12)
        h[:gamma] = 1.0
        h[:gdennorm] = 1.0
        if karl_halo_params !== nothing
            for (k, v) in normalize_halo(karl_halo_params)
                h[k] = v
            end
            h[:type] = :karl_halo
        end
        !haskey(h, :qdm) && (h[:qdm] = qh)
        h[:halo_q_axis_ratio] = max(abs(f64(h[:qdm])), 1e-6)
        if haskey(h, :rsnfw)
            h[:rsnfw_pc] = f64(h[:rsnfw])
        end
        if haskey(h, :rsgamma)
            h[:rsgamma_pc] = f64(h[:rsgamma])
        end
        if haskey(h, :rc)
            h[:rc_pc] = f64(h[:rc])
        end
    end
    stellar_model !== nothing && (h[:stellar_model] = normalize_stellar_model(stellar_model))
    if ht === :einasto
        h[:alpha] = isnothing(alpha) ? 0.18 : f64(alpha)
    end
    return h
end

# WHAT: Build spherical halo density, enclosed-mass, potential, and radial-force tables on radius grid R.
# HOW: Sample rho(R), integrate inward for M(<r), integrate outward for the outer-shell potential term, then compute Phi and fr.
# WHY: A spherical halo can be reduced to fast one-dimensional tables instead of evaluating a 3D gravitational integral during every orbit step.
# NOTE: R and rho are expected in SI units, so Menc is kg, tabv is m^2/s^2, and tabfr is m/s^2.
# NOTE: nlegup is retained in the public contract but is not used inside this implementation.
function tables_spherical(R, nlegup, halo, rhofn)
    halo=normalize_halo(halo); n=length(R)
    rho=similar(R); tabv=zeros(n); tabfr=zeros(n); Menc=zeros(n)
    @inbounds for i in eachindex(R)
        v=rhofn((R[i],0.0),halo)
        rho[i]=isfinite(v) ? v : 0.0
    end
    @inbounds for i in 2:n
        dr=R[i]-R[i-1]
        Menc[i]=Menc[i-1]+0.5*dr*(R[i]^2*rho[i]+R[i-1]^2*rho[i-1])
    end
    Menc .*= 4*pi
    J=zeros(n)
    @inbounds for i in (n-1):-1:1
        dr=R[i+1]-R[i]
        J[i]=J[i+1]+0.5*dr*(R[i+1]*rho[i+1] + R[i]*rho[i])
    end
    J .*= 4*pi
    @inbounds for i in eachindex(R)
        r=max(R[i],1e-30)
        tabv[i]  = -G*(Menc[i]/r + J[i])
        tabfr[i] = -(G*Menc[i])/(r*r)
    end
    tabv, tabfr, Menc
end

# WHAT: Build the spherical Karl light-grid representation used by enclosed-mass and potential helpers.
# HOW: Read radius and enclosed-light fraction, remove invalid points, merge duplicate radii, force the fraction to be nondecreasing, then renormalize the outermost value to 1.
# WHY: Spherical stellar gravity needs a clean cumulative light profile that can be interpolated at any radius.
# NOTE: require_spherical_stellar_geometry prevents this path from silently consuming an axisymmetric density grid.
function build_karl_light_grid_model(stellar_model)
    sm = normalize_stellar_model(stellar_model)
    require_spherical_stellar_geometry(sm)
    path = String(sm[:grid_csv])
    rcol = Symbol(String(get(sm, :radius_col, "r_pc")))
    lcol = Symbol(String(get(sm, :lenc_frac_col, "Lenc_frac")))
    _, colfloat = _read_karl_light_grid(path)
    r_all = colfloat(rcol)
    l_all = colfloat(lcol)
    length(r_all) == length(l_all) || error("karl_light_grid radius and Lenc_frac lengths do not match")
    tmp = Dict{Float64,Float64}()
    @inbounds for i in eachindex(r_all)
        r = r_all[i]
        f = l_all[i]
        if isfinite(r) && isfinite(f) && r > 0.0
            if !haskey(tmp, r)
                tmp[r] = clamp01(f)
            else
                tmp[r] = max(tmp[r], clamp01(f))
            end
        end
    end
    length(tmp) >= 2 || error("karl_light_grid needs at least two valid radial points")
    rs = sort(collect(keys(tmp)))
    fs = [tmp[r] for r in rs]
    @inbounds for i in 2:length(fs)
        fs[i] = max(fs[i], fs[i - 1])
    end
    fmax = fs[end]
    (!isfinite(fmax) || fmax <= 0.0) && error("karl_light_grid Lenc_frac has non-positive maximum")
    fs ./= fmax
    return ( R_m = Float64.(rs) .* pc, Lenc_frac = Float64.(fs), Ltot = f64(sm[:Ltot]) )
end

# The stellar geometry is fixed throughout a daemon run.  Its force and
# potential are exactly linear in M/L, so cache a unit-M/L component instead
# of rebuilding the same 96×96×32 table for every proposed parameter point.
const _STELLAR_COMPONENT_CACHE = Dict{UInt64,Any}()
const _STELLAR_COMPONENT_LOCK = ReentrantLock()

# WHAT: Build the stellar gravity component once at M/L=1.
# HOW: For axisymmetric grids, construct the luminous grid and a full unit-M/L force/potential table; for spherical grids, build the cumulative light model.
# WHY: Stellar gravity is linear in M/L, so later model evaluations can scale this component instead of rebuilding it.
# NOTE: This cache optimization applies to karl_light_grid models only.
function _build_unit_stellar_component(stellar_model)
    sm = normalize_stellar_model(stellar_model)
    stype = stellar_model_type(sm)
    geom = stellar_model_geometry(sm)

    stype === :karl_light_grid ||
        error("Unit stellar-component caching is only used for karl_light_grid models")

    if geom === :axisymmetric_density_grid
        grid = build_axisymmetric_light_grid_model(sm)
        nR = haskey(sm, :force_nR) ? Int(f64(sm[:force_nR])) : 96
        nZ = haskey(sm, :force_nZ) ? Int(f64(sm[:force_nZ])) : 96
        nphi = haskey(sm, :force_nphi) ? Int(f64(sm[:force_nphi])) : 32
        table = build_axisymmetric_force_table(grid, 1.0; nR=nR, nZ=nZ, nphi=nphi)
        return (grid=grid, axis_table=table, geometry=geom)
    end

    grid = build_karl_light_grid_model(sm)
    return (grid=grid, axis_table=nothing, geometry=geom)
end

# WHAT: Fetch the cached unit-M/L stellar component or build it if it does not exist.
# HOW: Hash the stellar model, lock the shared cache, reuse a match, otherwise build and store the component while printing timing diagnostics.
# WHY: The expensive 96x96x32 stellar-force calculation should happen once per stellar geometry, not once per theta proposal.
# NOTE: The lock makes cache creation safe when multiple Julia tasks/threads reach this code at the same time.
function _get_unit_stellar_component(stellar_model)
    sig = stellar_model_sig(stellar_model)

    lock(_STELLAR_COMPONENT_LOCK)
    try
        cached = get(_STELLAR_COMPONENT_CACHE, sig, nothing)
        cached !== nothing && return cached

        started_ns = time_ns()
        println(
            "[STELLAR FORCE CACHE] building unit-ML component",
            " geometry=", stellar_model_geometry(stellar_model),
            " julia_threads=", Threads.nthreads(),
        )
        flush(stdout)

        component = _build_unit_stellar_component(stellar_model)
        _STELLAR_COMPONENT_CACHE[sig] = component

        println(
            "[STELLAR FORCE CACHE] ready",
            " elapsed_s=", round((time_ns() - started_ns) / 1e9; digits=2),
        )
        flush(stdout)
        return component
    finally
        unlock(_STELLAR_COMPONENT_LOCK)
    end
end

# WHAT: Build the stellar force cache before model evaluations begin.
# HOW: Ignore missing/non-Karl models; otherwise request the unit stellar component once.
# WHY: Paying the expensive stellar-table cost up front avoids the first evaluated model unexpectedly carrying that setup delay.
# NOTE: This changes runtime behavior only, not the physical model.
function prewarm_stellar_force_cache(stellar_model)
    stellar_model === nothing && return nothing
    sm = normalize_stellar_model(stellar_model)
    stellar_model_type(sm) === :karl_light_grid || return nothing
    _get_unit_stellar_component(sm)
    return nothing
end

# WHAT: Assemble the total gravitational potential and force functions used by the orbit integrator.
# HOW: Combine the halo table, central point-mass BH, and selected stellar model; use cached axisymmetric stellar tables when needed.
# WHY: Orbit integration needs one pot(r,theta) and one frc(r,theta) interface representing the complete trial galaxy.
# NOTE: The returned force is the sum of halo + BH + stars. The returned potential is the matching quantity used for orbit launch energies.
# NOTE: nlegup and Menc remain in the function contract, but this implementation mainly uses tabv/tabfr plus component-specific helpers.
function make_potential_force_funcs(halo, R, nlegup, tabv, tabfr, Menc)
    halo = normalize_halo(halo)
    MBH  = f64(halo[:MBH])
    ML   = haskey(halo, :ML) ? f64(halo[:ML]) : 0.0
    rmin = f64(halo[:rmin])
    stellar_model = get(halo, :stellar_model, nothing)
    has_stars = stellar_model !== nothing && ML > 0.0
    stellar_grid = nothing
    stellar_axis_table = nothing
    stellar_geom = stellar_model_geometry(stellar_model)
    halo_q = halo_q_axis_ratio(halo)
    use_axisym_halo = halo[:type] !== :none && abs(halo_q - 1.0) > 1e-8
    halo_axis_table = nothing
    if has_stars
        stype0 = stellar_model_type(stellar_model)
        if stype0 === :karl_light_grid
            component = _get_unit_stellar_component(stellar_model)
            stellar_grid = component.grid
            stellar_axis_table = component.axis_table
        elseif stype0 === :plummer
            stellar_grid = nothing
        else
            error("Unknown stellar model type: $(stellar_model[:type])")
        end
    end
    if use_axisym_halo
        nR_h = haskey(halo, :halo_force_nR) ? Int(f64(halo[:halo_force_nR])) : 96
        nZ_h = haskey(halo, :halo_force_nZ) ? Int(f64(halo[:halo_force_nZ])) : 96
        nphi_h = haskey(halo, :halo_force_nphi) ? Int(f64(halo[:halo_force_nphi])) : 32
        nm_h = haskey(halo, :halo_force_nm) ? Int(f64(halo[:halo_force_nm])) : 128
        nth_h = haskey(halo, :halo_force_ntheta) ? Int(f64(halo[:halo_force_ntheta])) : 64
        soft_h = haskey(halo, :halo_force_softening_pc) ? f64(halo[:halo_force_softening_pc]) : 0.5
        halo_axis_table = build_axisymmetric_halo_force_table(halo; nR=nR_h, nZ=nZ_h, nphi=nphi_h, n_m=nm_h, ntheta=nth_h, softening_pc=soft_h, rmax_factor=DEFAULT_RMAX_FACTOR)
    end
    rlgmin = log10(f64(R[1]))
    rlgmax = log10(f64(R[end]))
    np = length(R)
    rlgmax > rlgmin || error("Degenerate R grid")

    # WHAT: Interpolate one spherical halo table on the logarithmic radius grid used by make_potential_force_funcs.
    # HOW: Convert radius to its fractional log-grid index, clamp to the table range, then linearly blend neighboring samples.
    # WHY: The orbit integrator asks for halo potential/force at radii between the precomputed grid points.
    # NOTE: Radii below rmin or beyond the table are clamped rather than extrapolated.
    @inline function interp(arr, rr)
        r = max(f64(rr), rmin)
        lr = log10(r)
        x = (lr - rlgmin) * (np - 1) / (rlgmax - rlgmin)
        x = clamp(x, 0.0, np - 1.0)
        i0 = Int(floor(x)) + 1
        i1 = min(i0 + 1, np)
        t = x - (i0 - 1)
        return (1.0 - t) * arr[i0] + t * arr[i1]
    end

    # WHAT: Return enclosed stellar mass for stellar models where a spherical enclosed mass is physically defined.
    # HOW: Use the analytic Plummer expression or the spherical Karl cumulative-light grid.
    # WHY: Spherical stellar forces can be computed from M(<r).
    # NOTE: Calling this for axisymmetric_density_grid is explicitly rejected because one number M(<r) cannot represent its directional force correctly.
    @inline function Mstar_enc(rr)
        if !has_stars
            return 0.0
        end
        stype = stellar_model_type(stellar_model)
        if stype === :plummer
            Ltot = f64(stellar_model[:Ltot])
            a    = f64(stellar_model[:a_pc]) * pc
            return stellar_Menc_plummer(rr, ML, Ltot, a)
        elseif stype === :karl_light_grid
            if stellar_geom === :axisymmetric_density_grid
                error("Mstar_enc is not physically defined for axisymmetric_density_grid. Use force diagnostics instead.")
            end
            return stellar_Menc_karl_light_grid(rr, ML, stellar_grid)
        else
            error("Unknown stellar model type: $(stellar_model[:type])")
        end
    end

    # WHAT: Return the stellar gravitational potential at one orbit position.
    # HOW: Use the analytic Plummer potential, the interpolated axisymmetric unit-M/L potential scaled by ML, or the spherical Karl light-grid potential.
    # WHY: The stellar component must participate in the same total potential used to set and conserve orbit energy.
    # NOTE: For axisymmetric grids the theta dependence is real; the potential is not reduced to a spherical radius-only function.
    @inline function Phistar(rr, theta)
        if !has_stars
            return 0.0
        end
        stype = stellar_model_type(stellar_model)
        if stype === :plummer
            Ltot = f64(stellar_model[:Ltot])
            a = f64(stellar_model[:a_pc]) * pc
            return stellar_Phi_plummer(rr, ML, Ltot, a)
        elseif stype === :karl_light_grid
            if stellar_geom === :axisymmetric_density_grid
                st, ct = _sincos_safe(theta)
                Rf = rr * st
                zf = rr * ct
                return ML * _interp_axisym_potential(stellar_axis_table, Rf, zf)
            end
            return stellar_Phi_karl_light_grid(rr, ML, stellar_grid)
        else
            error("Unknown stellar model type: $(stellar_model[:type])")
        end
    end

    # WHAT: Return the total gravitational potential at one orbit position.
    # HOW: Add the spherical halo potential, central point-mass BH potential, and stellar potential.
    # WHY: Orbit launch conditions and energy bookkeeping need the complete potential of the trial galaxy.
    # NOTE: The BH term is -G*MBH/r. The stellar term can depend on theta when the stellar model is axisymmetric.
    pot(r, theta=pi / 2) = begin
        rr = max(abs(f64(r)), rmin)
        Ph  = interp(tabv, rr)
        Pbh = MBH > 0.0 ? (-G * MBH / rr) : 0.0
        Pst = has_stars ? Phistar(rr, f64(theta)) : 0.0
        return Ph + Pbh + Pst
    end

    # WHAT: Return the total gravitational force components at one orbit position.
    # HOW: Add halo, BH, and stellar contributions separately in radial/polar form, using axisymmetric tables where required.
    # WHY: These are the accelerations that actually bend each integrated stellar orbit through the trial potential.
    # NOTE: The BH contributes only radial force. Axisymmetric stars or halos can contribute a nonzero theta force.
    frc(r, theta=pi / 2) = begin
        rr = max(abs(f64(r)), rmin)
        frh = 0.0
        fth_h = 0.0
        if use_axisym_halo
            frh, fth_h = halo_force_axisymmetric_spherical(rr, f64(theta), halo_axis_table)
        else
            frh = interp(tabfr, rr)
        end
        frbh = MBH > 0.0 ? (-G * MBH / (rr * rr)) : 0.0
        fth_bh = 0.0
        frst = 0.0
        fth_st = 0.0
        if has_stars
            stype = stellar_model_type(stellar_model)
            if stype === :plummer
                Ltot = f64(stellar_model[:Ltot])
                a    = f64(stellar_model[:a_pc]) * pc
                Mst  = stellar_Menc_plummer(rr, ML, Ltot, a)
                frst = -G * Mst / (rr * rr)
            elseif stype === :karl_light_grid
                if stellar_geom === :axisymmetric_density_grid
                    fr_unit, fth_unit =
                        stellar_force_axisymmetric_spherical(rr, f64(theta), stellar_axis_table)
                    frst = ML * fr_unit
                    fth_st = ML * fth_unit
                else
                    Mst = stellar_Menc_karl_light_grid(rr, ML, stellar_grid)
                    frst = -G * Mst / (rr * rr)
                end
            else
                error("Unknown stellar model type: $(stellar_model[:type])")
            end
        end
        return frh + frbh + frst, fth_h + fth_bh + fth_st
    end
    return pot, frc, R
end

# WHAT: Build all force/potential products needed for one trial galaxy model.
# HOW: Convert theta to a halo dictionary, choose a large enough radial grid, build spherical halo tables, then assemble the total pot/frc closures.
# WHY: Keeping these pieces together in HaloContext gives the orbit code one self-contained gravitational model.
# NOTE: required_rmax_m can enlarge the table beyond the normal halo-scaled radius so the orbit library is never asked to integrate outside its prepared domain.
function build_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=nothing, nR=DEFAULT_NR, rmax_factor=DEFAULT_RMAX_FACTOR, required_rmax_m::Float64=0.0, halo_q_axis_ratio=1.0, karl_halo_params=nothing)
    isfinite(required_rmax_m) && required_rmax_m >= 0.0 || error("required_rmax_m must be finite and nonnegative")
    halo = halo_from_theta(rho_s, r_s, MBH, ML; halo_type=halo_type, stellar_model=stellar_model, halo_q_axis_ratio=halo_q_axis_ratio, karl_halo_params=karl_halo_params)
    halo_scaled_rmax = rmax_factor * halo[:rs]
    rmax_use = max(halo_scaled_rmax, required_rmax_m)
    R = build_R_halo_physical(nR; rmin=halo[:rmin], rmax=rmax_use)
    tabv, tabfr, Menc = tables_spherical(R, 1, halo, rho_interp)
    pot, frc, _ = make_potential_force_funcs(halo, R, 1, tabv, tabfr, Menc)
    HaloContext(halo, f64.(R), tabv, tabfr, Menc, pot, frc)
end

# WHAT: Return a cached HaloContext for a trial parameter point, building it only when necessary.
# HOW: Construct a cache key from the physical parameters, stellar/halo signatures, resolution, flattening, and radial extent; reuse a match or build/store a new context.
# WHY: Repeated or duplicate theta evaluations should not rebuild identical halo tables and force closures.
# NOTE: The cache is locked for thread safety and capped at 256 entries; when full, one existing entry is removed before inserting a new one.
function get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=nothing, nR=DEFAULT_NR, rmax_factor=DEFAULT_RMAX_FACTOR, required_rmax_m::Float64=0.0, halo_q_axis_ratio=1.0, karl_halo_params=nothing)
    isfinite(required_rmax_m) && required_rmax_m >= 0.0 || error("required_rmax_m must be finite and nonnegative")
    ht = Symbol(lowercase(String(halo_type)))
    sig = stellar_model_sig(stellar_model)
    qh = max(abs(f64(halo_q_axis_ratio)), 1e-6)
    halo_for_sig = halo_from_theta(rho_s, r_s, MBH, ML; halo_type=ht, stellar_model=nothing, halo_q_axis_ratio=qh, karl_halo_params=karl_halo_params)
    ksig = ht === :karl_halo ? karl_halo_sig(halo_for_sig) : UInt(0)
    combined_sig = hash((sig, ksig))
    key = (_quant(f64(rho_s)), _quant(f64(r_s)), _quant(f64(MBH)), _quant(f64(ML)), combined_sig, ht, _quant(qh), nR, _quant(f64(rmax_factor)), _quant(required_rmax_m / pc))
    lock(_HALO_LOCK)
    ctx = get(_HALO_CTX_CACHE, key, nothing)
    unlock(_HALO_LOCK)
    ctx !== nothing && return ctx
    newctx = build_halo_context(rho_s, r_s, MBH, ML, ht; stellar_model=stellar_model, nR=nR, rmax_factor=rmax_factor, required_rmax_m=required_rmax_m, halo_q_axis_ratio=qh, karl_halo_params=karl_halo_params)
    lock(_HALO_LOCK)
    ctx = get(_HALO_CTX_CACHE, key, nothing)
    if ctx === nothing
        if length(_HALO_CTX_CACHE) >= 256
            delete!(_HALO_CTX_CACHE, first(keys(_HALO_CTX_CACHE)))
        end
        _HALO_CTX_CACHE[key] = newctx
        ctx = newctx
    end
    unlock(_HALO_LOCK)
    return ctx
end

# ============================================================
# §6  MASS / DIAGNOSTIC HELPERS
# ============================================================

# WHAT: Report the effective enclosed mass at two radii for a spherical total force model.
# HOW: Evaluate radial acceleration and invert the spherical relation M(<r)=-r^2*fr/G at rin and rout.
# WHY: This gives a simple diagnostic of how much gravitating mass the fitted potential implies at useful physical scales.
# NOTE: The relation is only valid for spherical forces, so flattened halos and axisymmetric stellar models are explicitly rejected.
mass_enclosed_two_radii(rin, rout, rho_s, r_s, MBH, ML, halo_type; stellar_model=nothing, halo_q_axis_ratio=1.0, karl_halo_params=nothing) = begin
    if abs(max(abs(f64(halo_q_axis_ratio)), 1e-6) - 1.0) > 1e-8
        error("mass_enclosed_two_radii is only physically meaningful for spherical halo models. " * "For flattened halo_q_axis_ratio, use a force diagnostic instead.")
    end
    if is_axisymmetric_stellar_model(stellar_model)
        error( "mass_enclosed_two_radii is only physically meaningful for spherical force models. " * "For axisymmetric_density_grid, use a force diagnostic instead.")
    end
    ctx = get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=stellar_model, halo_q_axis_ratio=halo_q_axis_ratio, karl_halo_params=karl_halo_params)
    r1 = max(rin, ctx.halo[:rmin])
    r2 = max(rout, 1.001 * r1)
    fr1, _ = ctx.frc(r1, pi / 2)
    fr2, _ = ctx.frc(r2, pi / 2)
    return (-r1 * r1 * fr1 / G, -r2 * r2 * fr2 / G)
end

# WHAT: Diagnostic helper returning the total force at one physical position.
# HOW: Fetch the model context, evaluate spherical fr/ftheta, then rotate those components into cylindrical FR/FZ.
# WHY: This makes it easy to inspect and compare the actual gravity produced by a trial model without running a full orbit library.
# NOTE: The return order is (fr, ftheta, FR, FZ).
function force_at_rtheta(r, theta, rho_s, r_s, MBH, ML, halo_type; stellar_model=nothing, halo_q_axis_ratio=1.0, karl_halo_params=nothing)
    ctx = get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=stellar_model, halo_q_axis_ratio=halo_q_axis_ratio, karl_halo_params=karl_halo_params)
    rr = max(f64(r), ctx.halo[:rmin])
    th = f64(theta)
    fr, ftheta = ctx.frc(rr, th)
    st, ct = _sincos_safe(th)
    FR = fr * st + ftheta * ct
    FZ = fr * ct - ftheta * st
    return fr, ftheta, FR, FZ
end
