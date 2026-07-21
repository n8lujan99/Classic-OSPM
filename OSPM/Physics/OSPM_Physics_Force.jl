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


@inline function karl_m_ellipsoidal(r::Float64, theta::Float64, qdm::Float64)
    q = max(abs(qdm), 1e-6)
    cth = cos(theta)
    sth = sin(theta)
    return r * sqrt(cth * cth + (sth * sth) / (q * q))
end

@inline function karl_halo_density_dehnen_plummer(r::Float64, theta::Float64; qdm::Float64, xmgamma::Float64, rsgamma_pc::Float64, gamma::Float64)
    m = max(karl_m_ellipsoidal(r, theta, qdm), 1e-30)
    a = max(rsgamma_pc, 1e-30)
    if gamma != 0.0
        return xmgamma / (4.0 * pi) * (3.0 - gamma) * a / (m^gamma * (a + m)^(4.0 - gamma))
    else
        return 3.0 * xmgamma / (4.0 * pi * a^3) * (1.0 + (m / a)^2)^(-2.5)
    end
end

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

function karl_halo_from_params(; ihalo::Int=4, qdm::Float64=1.0, dis::Float64=1.0, v0::Float64=0.0, rc_pc::Float64=1.0, xmgamma::Float64=0.0, rsgamma_pc::Float64=1.0, gamma::Float64=1.0, cnfw::Float64=1.0, rsnfw_pc::Float64=1.0, gdennorm::Float64=1.0)
    return Dict{Symbol,Any}( :type => :karl_halo, :ihalo => ihalo, :qdm => qdm, :dis => dis, :v0 => v0, :rc_pc => rc_pc, :xmgamma => xmgamma, :rsgamma_pc => rsgamma_pc, :gamma => gamma, :cnfw => cnfw, :rsnfw_pc => rsnfw_pc, :gdennorm => gdennorm)
end


@inline function karl_halo_sig(halo)
    h = normalize_halo(halo)
    return hash((get(h, :type, nothing), get(h, :ihalo, nothing), get(h, :qdm, nothing), get(h, :dis, nothing), get(h, :v0, nothing), get(h, :rc_pc, get(h, :rc, nothing)), get(h, :xmgamma, nothing), get(h, :rsgamma_pc, get(h, :rsgamma, nothing)), get(h, :gamma, nothing), get(h, :cnfw, nothing), get(h, :rsnfw_pc, get(h, :rsnfw, nothing)), get(h, :gdennorm, nothing), get(h, :halo_force_nR, nothing), get(h, :halo_force_nZ, nothing), get(h, :halo_force_nphi, nothing), get(h, :halo_force_nm, nothing), get(h, :halo_force_ntheta, nothing), get(h, :halo_force_softening_pc, nothing)))
end

@inline function _theta_from_cylindrical_Rz(R::Float64, z::Float64)
    # Karl's halodens.f uses xR = r*cos(theta), xZ = r*sin(theta).
    # This helper therefore maps cylindrical R,z to that convention:
    #   xR -> cylindrical R
    #   xZ -> vertical z
    return atan(abs(z), max(abs(R), 1e-300))
end

@inline function rho_karl_halo_cylindrical(R::Float64, z::Float64, halo)
    r = sqrt(R * R + z * z)
    theta = _theta_from_cylindrical_Rz(R, z)
    return rho_interp_karl_halo((r, theta), halo)
end

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
    halo[:type] === :einasto && begin
        α = halo[:alpha]          # curvature parameter
        return rhos * exp(-2/α * (x^α - 1))
    end
    error("Unknown halo type: $(halo[:type])")
end

@inline function halo_q_axis_ratio(halo)
    if get(halo, :type, :none) === :karl_halo
        q = haskey(halo, :qdm) ? f64(halo[:qdm]) : (haskey(halo, :halo_q_axis_ratio) ? f64(halo[:halo_q_axis_ratio]) : 1.0)
        return max(abs(q), 1e-6)
    end
    q = haskey(halo, :halo_q_axis_ratio) ? f64(halo[:halo_q_axis_ratio]) : 1.0
    return max(abs(q), 1e-6)
end

@inline function halo_m_axisym(R::Float64, z::Float64, halo)
    q = halo_q_axis_ratio(halo)
    return sqrt(R * R + (z / q) * (z / q))
end

@inline function rho_halo_axisym(R::Float64, z::Float64, halo)
    if get(halo, :type, :none) === :karl_halo
        return rho_karl_halo_cylindrical(R, z, halo)
    end
    m = halo_m_axisym(R, z, halo)
    return rho_interp((m, 0.0), halo)
end

@inline function normalize_stellar_model(stellar_model)
    stellar_model === nothing && return nothing
    out = Dict{Symbol,Any}()
    for (k, v) in stellar_model
        ks = k isa Symbol ? k : Symbol(String(k))
        out[ks] = v
    end
    return out
end

function _read_karl_light_grid(path::String)
    lines = readlines(path)
    isempty(lines) && error("empty karl_light_grid CSV: $path")
    header = split(strip(lines[1]), ",")
    idx = Dict(Symbol(strip(h)) => i for (i, h) in enumerate(header))
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

@inline function stellar_Menc_karl_light_grid(r::Float64, ML::Float64, grid)
    rr = max(r, 1e-30)
    f = _interp_linear_grid(grid.R_m, grid.Lenc_frac, rr)
    return ML * grid.Ltot * f * Msun
end

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

@inline function stellar_model_geometry(stellar_model)
    stellar_model === nothing && return :none
    sm = normalize_stellar_model(stellar_model)
    return Symbol(lowercase(String(get(sm, :geometry, :spherical_shell_grid))))
end

@inline function stellar_model_type(stellar_model)
    stellar_model === nothing && return :none
    sm = normalize_stellar_model(stellar_model)
    return Symbol(lowercase(String(get(sm, :type, :none))))
end

@inline function is_axisymmetric_stellar_model(stellar_model)
    geom = stellar_model_geometry(stellar_model)
    return geom === :axisymmetric_density_grid
end

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

@inline function _get_float_or_default(sm, key::Symbol, default::Float64)
    haskey(sm, key) || return default
    x = tryparse(Float64, String(sm[key]))
    x === nothing ? f64(sm[key]) : x
end

# ============================================================
# §4b  AXISYMMETRIC STELLAR GRID + FORCE TABLE
# ============================================================

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

function _axisym_force_from_cells(Rf::Float64, zf::Float64, ML::Float64, grid; nphi::Int=32)
    M_cells = Float64.(ML .* grid.L_cell .* Msun)
    return _axisym_force_from_mass_cells(Rf, zf, grid.R_m, grid.z_m, M_cells, grid.soft_m; nphi=nphi)
end

function _make_force_axis(minval::Float64, maxval::Float64, n::Int)
    lo = max(minval, 1e-8)
    hi = max(maxval, 10.0 * lo)
    return logspace10(log10(lo), log10(hi), n)
end

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
    @inbounds for i in 1:nR
        Rf = R_axis[i]
        for j in 1:nZ
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

@inline function _cyl_force_to_spherical_force(r::Float64, theta::Float64, FR::Float64, FZ::Float64)
    st, ct = _sincos_safe(theta)
    fr = FR * st + FZ * ct
    ftheta = FR * ct - FZ * st
    return fr, ftheta
end

@inline function stellar_force_axisymmetric_spherical(r::Float64, theta::Float64, table)
    rr = max(abs(r), 1e-30)
    st, ct = _sincos_safe(theta)
    Rf = rr * st
    zf = rr * ct
    FR, FZ = _interp_axisym_force(table, Rf, zf)
    return _cyl_force_to_spherical_force(rr, theta, FR, FZ)
end

@inline function halo_force_axisymmetric_spherical(r::Float64, theta::Float64, table)
    rr = max(abs(r), 1e-30)
    st, ct = _sincos_safe(theta)
    Rf = rr * st
    zf = rr * ct
    FR, FZ = _interp_axisym_force(table, Rf, zf)
    return _cyl_force_to_spherical_force(rr, theta, FR, FZ)
end

@inline function stellar_Menc_plummer(r::Float64, ML::Float64, Ltot::Float64, a::Float64)
    rr = max(r, 1e-30)
    Mtot = ML * Ltot * Msun
    return Mtot * rr^3 / (rr^2 + a^2)^(3/2)
end

@inline function stellar_Phi_plummer(r::Float64, ML::Float64, Ltot::Float64, a::Float64)
    rr = max(r, 1e-30)
    Mtot = ML * Ltot * Msun
    return -G * Mtot / sqrt(rr^2 + a^2)
end

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

function halo_from_theta(rho_s, r_s, MBH, ML; halo_type="nfw", alpha=nothing, stellar_model=nothing, halo_q_axis_ratio=1.0, karl_halo_params=nothing)
    ht = Symbol(lowercase(String(halo_type)))
    qh = max(abs(f64(halo_q_axis_ratio)), 1e-6)
    rs_pc = f64(r_s)
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
            if stellar_geom === :axisymmetric_density_grid
                sm = normalize_stellar_model(stellar_model)
                stellar_grid = build_axisymmetric_light_grid_model(sm)
                nR = haskey(sm, :force_nR) ? Int(f64(sm[:force_nR])) : 96
                nZ = haskey(sm, :force_nZ) ? Int(f64(sm[:force_nZ])) : 96
                nphi = haskey(sm, :force_nphi) ? Int(f64(sm[:force_nphi])) : 32
                stellar_axis_table = build_axisymmetric_force_table( stellar_grid, ML; nR=nR, nZ=nZ, nphi=nphi )
            else
                stellar_grid = build_karl_light_grid_model(stellar_model)
            end
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
                return _interp_axisym_potential(stellar_axis_table, Rf, zf)
            end
            return stellar_Phi_karl_light_grid(rr, ML, stellar_grid)
        else
            error("Unknown stellar model type: $(stellar_model[:type])")
        end
    end

    pot(r, theta=pi / 2) = begin
        rr = max(abs(f64(r)), rmin)
        Ph  = interp(tabv, rr)
        Pbh = MBH > 0.0 ? (-G * MBH / rr) : 0.0
        Pst = has_stars ? Phistar(rr, f64(theta)) : 0.0
        return Ph + Pbh + Pst
    end

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
                    frst, fth_st = stellar_force_axisymmetric_spherical(rr, f64(theta), stellar_axis_table)
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

function build_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=nothing, nR=DEFAULT_NR, rmax_factor=DEFAULT_RMAX_FACTOR, halo_q_axis_ratio=1.0, karl_halo_params=nothing)
    halo = halo_from_theta(rho_s, r_s, MBH, ML; halo_type=halo_type, stellar_model=stellar_model, halo_q_axis_ratio=halo_q_axis_ratio, karl_halo_params=karl_halo_params)
    R = build_R_halo_physical(nR; rmin=halo[:rmin], rmax=rmax_factor * halo[:rs])
    tabv, tabfr, Menc = tables_spherical(R, 1, halo, rho_interp)
    pot, frc, _ = make_potential_force_funcs(halo, R, 1, tabv, tabfr, Menc)
    HaloContext(halo, f64.(R), tabv, tabfr, Menc, pot, frc)
end

function get_halo_context(rho_s, r_s, MBH, ML, halo_type; stellar_model=nothing, nR=DEFAULT_NR, rmax_factor=DEFAULT_RMAX_FACTOR, halo_q_axis_ratio=1.0, karl_halo_params=nothing)
    ht = Symbol(lowercase(String(halo_type)))
    sig = stellar_model_sig(stellar_model)
    qh = max(abs(f64(halo_q_axis_ratio)), 1e-6)
    halo_for_sig = halo_from_theta(rho_s, r_s, MBH, ML; halo_type=ht, stellar_model=nothing, halo_q_axis_ratio=qh, karl_halo_params=karl_halo_params)
    ksig = ht === :karl_halo ? karl_halo_sig(halo_for_sig) : UInt(0)
    combined_sig = hash((sig, ksig))
    key = (_quant(f64(rho_s)), _quant(f64(r_s)), _quant(f64(MBH)), _quant(f64(ML)), combined_sig, ht, _quant(qh), nR, _quant(f64(rmax_factor)))
    lock(_HALO_LOCK)
    ctx = get(_HALO_CTX_CACHE, key, nothing)
    unlock(_HALO_LOCK)
    ctx !== nothing && return ctx
    newctx = build_halo_context(rho_s, r_s, MBH, ML, ht; stellar_model=stellar_model, nR=nR, rmax_factor=rmax_factor, halo_q_axis_ratio=qh, karl_halo_params=karl_halo_params)
    lock(_HALO_LOCK)
    ctx = get(_HALO_CTX_CACHE, key, nothing)
    if ctx === nothing
        _HALO_CTX_CACHE[key] = newctx
        ctx = newctx
    end
    unlock(_HALO_LOCK)
    return ctx
end

# ============================================================
# §6  MASS / DIAGNOSTIC HELPERS
# ============================================================
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
