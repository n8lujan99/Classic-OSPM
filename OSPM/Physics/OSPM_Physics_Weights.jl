# ============================================================
# OSPM_Physics_Weights.jl — Karl-style orbit weight machinery.
# Included by OSPM_Physics_Support.jl — do NOT load directly.
#
# Owns:
#   - wphase handling
#   - Karl entropy type 2
#   - paired prograde/retrograde initial weights
#   - SPEAR / expanded Cm solver
#   - LOSVD slack variables
#   - Karl fracnew and standard χ² block scoring
#   - xmu / M-L helper functions
#
# Assumes OSPM_Physics_Support.jl has already defined:
#   f64, DEFAULT_KARL_ALPHAT, DEFAULT_KARL_MAXITER,
#   DEFAULT_KARL_ENTROPY_FLOOR, DEFAULT_KARL_APFAC,
#   DEFAULT_KARL_ALPHA
# ============================================================

@inline function _karl_mode_symbol(mode)
    return Symbol(lowercase(String(mode)))
end

function _prepare_wphase(wphase, n::Int; entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    if wphase === nothing
        return ones(Float64, n)
    end

    wp = Float64.(wphase)
    length(wp) == n || error("wphase length $(length(wp)) does not match Norbit=$n")

    @inbounds for i in eachindex(wp)
        (!isfinite(wp[i]) || wp[i] <= 0.0) && (wp[i] = entropy_floor)
    end

    return wp
end


@inline function _safe_positive(x::Float64; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    return (isfinite(x) && x > floor) ? x : floor
end

function normalize_wphase(wphase, n::Int; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    if wphase === nothing
        return ones(Float64, n)
    end

    wp = Float64.(wphase)
    length(wp) == n || error("wphase length $(length(wp)) does not match number of orbit weights $n")

    @inbounds for i in eachindex(wp)
        wp[i] = _safe_positive(wp[i]; floor=floor)
    end

    return wp
end

function karl_entropy_type2(w::Vector{Float64}, wphase::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    length(w) == length(wphase) || error("w and wphase lengths do not match")

    S = 0.0

    @inbounds for i in eachindex(w)
        wi = _safe_positive(w[i]; floor=floor)
        pi = _safe_positive(wphase[i]; floor=floor)
        S -= wi * log(wi * pi)
    end

    return S
end

function karl_entropy_type2_grad!(g::Vector{Float64}, w::Vector{Float64}, wphase::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    length(g) == length(w) == length(wphase) ||
        error("gradient, w, and wphase lengths do not match")

    @inbounds for i in eachindex(w)
        wi = _safe_positive(w[i]; floor=floor)
        pi = _safe_positive(wphase[i]; floor=floor)
        g[i] = -1.0 - log(wi * pi)
    end

    return g
end

function karl_entropy_type2_hessian_diag!(h::Vector{Float64}, w::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    length(h) == length(w) || error("hessian diagonal and w lengths do not match")

    @inbounds for i in eachindex(w)
        wi = _safe_positive(w[i]; floor=floor)
        h[i] = -1.0 / wi
    end

    return h
end

##CHI##
function karl_entropy_loss_and_grad!(g::Vector{Float64}, A::Matrix{Float64}, d::Vector{Float64}, sigma::Vector{Float64}, w::Vector{Float64}, wphase::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    m, n = size(A)
    length(d) == m || error("d length does not match A rows")
    length(sigma) == m || error("sigma length does not match A rows")
    length(w) == n || error("w length does not match A columns")
    length(wphase) == n || error("wphase length does not match A columns")
    length(g) == n || error("gradient length does not match A columns")

    sig = max.(Float64.(sigma), 1e-12)
    pred = A * w
    resid = (pred .- d) ./ sig

    chi = sum(resid .^ 2)
    entropy = karl_entropy_type2(w, wphase; floor=entropy_floor)

    fill!(g, 0.0)

    @inbounds for j in 1:n
        acc = 0.0

        for i in 1:m
            acc += A[i, j] * (pred[i] - d[i]) / (sig[i] * sig[i])
        end

        wi = _safe_positive(w[j]; floor=entropy_floor)
        pi = _safe_positive(wphase[j]; floor=entropy_floor)
        dS = -1.0 - log(wi * pi)

        # Minimization form of Karl profit:
        # loss = alphat * chi - entropy
        g[j] = 2.0 * alphat * acc - dS
    end

    loss = alphat * chi - entropy
    profit = entropy - alphat * chi

    return loss, chi, entropy, profit
end

function losvd_width_at_fraction(v::Vector{Float64}, f::Vector{Float64}, frac::Float64)
    length(v) == length(f) || error("velocity and LOSVD arrays must match")
    length(v) >= 2 || return NaN

    fmax = maximum(f)
    !(isfinite(fmax) && fmax > 0.0) && return NaN

    level = frac * fmax
    inds = findall(x -> isfinite(x) && x >= level, f)
    isempty(inds) && return NaN

    return maximum(v[inds]) - minimum(v[inds])
end

function karl_update_xmu_from_fwhm(xmu::Float64, model_losvd_by_bin::Vector{Vector{Float64}}, data_losvd_by_bin::Vector{Vector{Float64}}, velocity_centers_by_bin::Vector{Vector{Float64}}; apfacmu::Float64=1.0, fractions::NTuple{2,Float64}=(0.25, 0.50))
    length(model_losvd_by_bin) == length(data_losvd_by_bin) == length(velocity_centers_by_bin) ||
        error("LOSVD bin collections must have matching lengths")

    sxmu = 0.0
    nuse = 0

    for ib in eachindex(model_losvd_by_bin)
        model = model_losvd_by_bin[ib]
        data = data_losvd_by_bin[ib]
        vel = velocity_centers_by_bin[ib]

        for frac in fractions
            fwm = losvd_width_at_fraction(vel, model, frac)
            fwd = losvd_width_at_fraction(vel, data, frac)

            if isfinite(fwm) && isfinite(fwd) && fwd > 0.0
                sxmu += fwm / fwd
                nuse += 1
            end
        end
    end

    nuse > 0 || return xmu, NaN, NaN

    sxmu /= nuse
    pml_xmu = xmu + xmu * (sxmu - 1.0)
    xmu_new = xmu + apfacmu * xmu * (sxmu - 1.0)

    return xmu_new, sxmu, pml_xmu
end

@inline function karl_ml_from_xmu(xmu::Float64)
    return xmu > 0.0 ? 1.0 / (xmu * xmu) : Inf
end

function karl_initial_weights_from_wphase(wphase::Vector{Float64}; paired::Bool=true, rotfrac::Float64=0.75, floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    n = length(wphase)
    n > 0 || return Float64[]

    w = zeros(Float64, n)

    if paired
        iseven(n) || error("paired Karl initial weights require an even number of orbit columns")

        Norb = n ÷ 2

        @inbounds for i in 1:Norb
            ip = 2 * i - 1
            ir = 2 * i

            wp = _safe_positive(wphase[ip]; floor=floor)
            wr = _safe_positive(wphase[ir]; floor=floor)

            w[ip] = rotfrac / wp
            w[ir] = (1.0 - rotfrac) / wr
        end
    else
        @inbounds for i in 1:n
            wp = _safe_positive(wphase[i]; floor=floor)
            w[i] = 1.0 / wp
        end
    end

    s = sum(w)
    (!isfinite(s) || s <= 0.0) && error("Karl initial weights have non-positive sum")

    w ./= s
    return w
end

@inline function karl_entropy_value(w::Vector{Float64}, wphase::Vector{Float64}; entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    S = 0.0

    @inbounds for i in eachindex(w)
        wi = max(w[i], entropy_floor)
        pi = max(wphase[i], entropy_floor)
        S -= wi * log(wi * pi)
    end

    return S
end

function _karl_entropy_loss_gradient!(g::Vector{Float64}, Aw::Matrix{Float64}, w::Vector{Float64}, dw::Vector{Float64}, wphase::Vector{Float64};
    alphat::Float64=DEFAULT_KARL_ALPHAT, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)

    r = Aw * w
    r .-= dw
    mul!(g, transpose(Aw), r)
    g .*= 2.0 * alphat

    @inbounds for i in eachindex(w)
        wi = max(w[i], entropy_floor)
        pi = max(wphase[i], entropy_floor)
        # loss = alphat * χ² - S, with S = -Σ w_i log(w_i*wphase_i)
        g[i] += 1.0 + log(wi * pi)
    end

    return g
end

function _project_positive_simplex!(w::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    @inbounds for i in eachindex(w)
        (!isfinite(w[i]) || w[i] < floor) && (w[i] = floor)
    end

    s = sum(w)
    (!isfinite(s) || s <= 0.0) && return false
    w ./= s
    return true
end

##CHI##
function karl_weight_diagnostics(A::Matrix{Float64}, w::Vector{Float64}, d::Vector{Float64}, sigma::Vector{Float64}; wphase=nothing, alphat::Float64=DEFAULT_KARL_ALPHAT, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, normalization_mode=:probability, Nspatial::Int=0, Nvbin::Int=0)

    n = length(w)
    wp = _prepare_wphase(wphase, n; entropy_floor=entropy_floor)
    chi = if _karl_mode_symbol(normalization_mode) === :karl_fracnew && Nspatial > 0 && Nvbin > 0
        chi2_block_karl_fracnew(A, w, d, sigma, Nspatial, Nvbin)
    else
        chi2_block(A, w, d, sigma)
    end
    ent = karl_entropy_value(w, wp; entropy_floor=entropy_floor)
    profit = ent - alphat * chi

    return (entropy=ent, chi=chi, profit=profit, alphat=alphat)
end


# ============================================================
# §3c  KARL SPEAR / MAXIMUM-ENTROPY WEIGHT SOLVER
# ============================================================
# Karl's Fortran model uses entropy derivatives plus the SPEAR correction.
# This is now the live Karl weight path.  It updates orbit weights by solving
# the constrained Newton/Lagrange system for the target rows.

@inline function _spear_safe_ddS(x::Float64; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    if !isfinite(x)
        return -floor
    end
    if abs(x) < floor
        return x < 0.0 ? -floor : floor
    end
    return x
end

function karl_entropy_type2_derivatives(w::Vector{Float64}, wphase::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    length(w) == length(wphase) || error("w and wphase lengths do not match")
    n = length(w)
    S = 0.0
    dS = zeros(Float64, n)
    ddS = zeros(Float64, n)
    @inbounds for i in 1:n
        wi = _safe_positive(w[i]; floor=floor)
        pi = _safe_positive(wphase[i]; floor=floor)
        S -= wi * log(wi * pi)
        dS[i] = -1.0 - log(wi * pi)
        ddS[i] = -1.0 / wi
    end
    return S, dS, ddS
end

function karl_spear_build_Am(Cm::Matrix{Float64}, ddS::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Narr, nvar = size(Cm)
    length(ddS) == nvar || error("ddS length must match Cm columns")

    invdd = Vector{Float64}(undef, nvar)
    @inbounds for j in 1:nvar
        invdd[j] = 1.0 / _spear_safe_ddS(ddS[j]; floor=floor)
    end

    return Cm * Diagonal(invdd) * transpose(Cm)
end

function karl_spear_rhs!(delY::Vector{Float64}, Cm::Matrix{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Narr, nvar = size(Cm)
    length(delY) == Narr || error("delY length must match Cm rows")
    length(dS) == nvar || error("dS length must match Cm columns")
    length(ddS) == nvar || error("ddS length must match Cm columns")

    tmp = similar(dS)
    @inbounds for j in 1:nvar
        tmp[j] = dS[j] / _spear_safe_ddS(ddS[j]; floor=floor)
    end

    delY .+= Cm * tmp
    return delY
end

function karl_spear_delta_w(Cm::Matrix{Float64}, lambda::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Narr, nvar = size(Cm)
    length(lambda) == Narr || error("lambda length must match Cm rows")
    length(dS) == nvar || error("dS length must match Cm columns")
    length(ddS) == nvar || error("ddS length must match Cm columns")

    dw = transpose(Cm) * lambda
    @inbounds for j in 1:nvar
        dw[j] = (dw[j] - dS[j]) / _spear_safe_ddS(ddS[j]; floor=floor)
    end

    return dw
end

function _solve_spear_system(Am::Matrix{Float64}, rhs::Vector{Float64})
    if size(Am, 1) == 0
        return Float64[]
    end
    # The SPEAR matrix can be nearly singular when the orbit library has repeated
    # columns.  Use a tiny diagonal floor only to make the Newton solve finite.
    scale = maximum(abs.(Am))
    ridge = max(scale, 1.0) * 1e-12
    Areg = copy(Am)
    @inbounds for i in 1:size(Areg, 1)
        Areg[i, i] += ridge
    end
    return Areg \ rhs
end

function karl_spear_update_orbit_weights(w::Vector{Float64}, A::Matrix{Float64}, target::Vector{Float64}, wphase::Vector{Float64}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    m, n = size(A)
    length(w) == n || error("w length must match A columns")
    length(target) == m || error("target length must match A rows")
    length(wphase) == n || error("wphase length must match A columns")

    entropy, dS, ddS = karl_entropy_type2_derivatives(w, wphase; floor=entropy_floor)

    model = A * w
    delY = target .- model

    Am = karl_spear_build_Am(A, ddS; floor=entropy_floor)
    karl_spear_rhs!(delY, A, dS, ddS; floor=entropy_floor)

    lambda = _solve_spear_system(Am, delY)
    dw = karl_spear_delta_w(A, lambda, dS, ddS; floor=entropy_floor)

    wnew = similar(w)
    @inbounds for j in eachindex(w)
        wnew[j] = w[j] + apfac * dw[j]
    end

    _project_positive_simplex!(wnew; floor=entropy_floor) || return w, false, (entropy=entropy, rcond_est=0.0, max_abs_dw=Inf)

    rcond_est = 1.0 / max(cond(Am), 1.0)
    max_abs_dw = maximum(abs.(dw))

    return wnew, true, (entropy=entropy, rcond_est=rcond_est, max_abs_dw=max_abs_dw)
end

function solve_weights_karl_spear(A::Matrix{Float64}, d::Vector{Float64}, sigma::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, wphase=nothing, maxiter::Int=DEFAULT_KARL_MAXITER, seed::UInt=UInt(0), entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, apfac::Float64=DEFAULT_KARL_APFAC, return_diag::Bool=false)
    m, n = size(A)
    fail_w = zeros(Float64, n)

    (m == length(d) == length(sigma)) || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    n <= 0 && return return_diag ? (zeros(Float64, 0), false, nothing) : (zeros(Float64, 0), false)
    !all(isfinite, A) && return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, d) && return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, sigma) && return return_diag ? (fail_w, false, nothing) : (fail_w, false)

    sig = max.(Float64.(sigma), 1e-12)

    # Karl's entropy routine subtracts alphat * residual^2 / den.  Whitening by
    # sigma and multiplying rows by sqrt(alphat) makes the constrained residual
    # target carry the same data-pressure scale.
    row_scale = sqrt(max(alphat, 0.0)) ./ sig
    Aw = A .* row_scale
    dw = d .* row_scale

    wp = _prepare_wphase(wphase, n; entropy_floor=entropy_floor)

    # The Spherical path now writes prograde/retrograde columns in pairs.  Use
    # Karl's paired initial-weight shape by default when the column count is even.
    if iseven(n)
        w = karl_initial_weights_from_wphase(wp; paired=true, rotfrac=0.75, floor=entropy_floor)
    else
        w = karl_initial_weights_from_wphase(wp; paired=false, floor=entropy_floor)
    end

    last_diag = nothing
    ok = true

    for _ in 1:maxiter
        wnew, step_ok, sdiag = karl_spear_update_orbit_weights(w, Aw, dw, wp; apfac=apfac, entropy_floor=entropy_floor)

        last_diag = sdiag

        if !step_ok
            ok = false
            break
        end

        if norm(wnew .- w) <= 1e-8 * max(1.0, norm(w))
            w .= wnew
            break
        end

        w .= wnew
    end

    s = sum(w)
    (!isfinite(s) || s <= 0.0 || !ok) && return return_diag ? (fail_w, false, last_diag) : (fail_w, false)
    w ./= s

    if return_diag
        diag0 = karl_weight_diagnostics(A, w, d, sigma; wphase=wp, alphat=alphat, entropy_floor=entropy_floor)
        diag = (entropy=diag0.entropy, chi=diag0.chi, profit=diag0.profit, alphat=diag0.alphat, rcond_est=last_diag === nothing ? NaN : last_diag.rcond_est, max_abs_dw=last_diag === nothing ? NaN : last_diag.max_abs_dw)
        return (w, true, diag)
    end

    return (w, true)
end


# ============================================================
# §3d  KARL EXPANDED CM WITH LOSVD SLACK VARIABLES
# ============================================================
# This is the full Karl-style SPEAR shape:
#   rows    = light constraints followed by LOSVD constraints
#   columns = orbit weights followed by LOSVD slack variables
#
# The orbit solver returns only the orbit-weight part.  Slack variables are
# internal SPEAR variables that carry the LOSVD residual term the way Karl's
# entropy.f / spear.f system does.

function build_expanded_Cm_with_losvd_slack(A_light::Matrix{Float64}, A_losvd::Matrix{Float64})
    Nlight, Norbit = size(A_light)
    Nlosvd, Norbit2 = size(A_losvd)
    Norbit == Norbit2 || error("A_light and A_losvd must have the same number of orbit columns")

    Narr = Nlight + Nlosvd
    Nslack = Nlosvd

    Cm = zeros(Float64, Narr, Norbit + Nslack)

    Cm[1:Nlight, 1:Norbit] .= A_light
    Cm[(Nlight + 1):Narr, 1:Norbit] .= A_losvd
    Cm[(Nlight + 1):Narr, (Norbit + 1):(Norbit + Nslack)] .= Matrix{Float64}(I, Nlosvd, Nlosvd)

    return Cm
end

function build_expanded_target(light_target::Vector{Float64}, losvd_target::Vector{Float64})
    return vcat(light_target, losvd_target)
end

function build_expanded_weights_initial(w_orbit::Vector{Float64}, A_losvd::Matrix{Float64}, losvd_target::Vector{Float64})
    Nlosvd, Norbit = size(A_losvd)
    length(w_orbit) == Norbit || error("w_orbit length does not match A_losvd columns")
    length(losvd_target) == Nlosvd || error("losvd_target length does not match A_losvd rows")

    slack = losvd_target .- A_losvd * w_orbit
    return vcat(w_orbit, slack)
end

##CHI##
function build_expanded_entropy_derivatives(w_all::Vector{Float64}, Norbit::Int, wphase_orbit::Vector{Float64}, losvd_sigma::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Nvar = length(w_all)
    Nlosvd = Nvar - Norbit

    Nlosvd >= 0 || error("Norbit cannot exceed total variable count")
    length(wphase_orbit) == Norbit || error("wphase length must match Norbit")
    length(losvd_sigma) == Nlosvd || error("losvd_sigma length must match LOSVD slack count")

    dS = zeros(Float64, Nvar)
    ddS = zeros(Float64, Nvar)

    entropy = 0.0
    chi_slack = 0.0

    @inbounds for j in 1:Norbit
        wj = _safe_positive(w_all[j]; floor=entropy_floor)
        pj = _safe_positive(wphase_orbit[j]; floor=entropy_floor)

        entropy -= wj * log(wj * pj)
        dS[j] = -1.0 - log(wj * pj)
        ddS[j] = -1.0 / wj
    end

    @inbounds for k in 1:Nlosvd
        idx = Norbit + k
        sig = max(abs(losvd_sigma[k]), 1e-12)
        den = sig * sig
        y = w_all[idx]

        entropy -= y * y * alphat / den
        chi_slack += y * y * alphat / den

        dS[idx] = -2.0 * y * alphat / den
        ddS[idx] = -2.0 * alphat / den

        if ddS[idx] == 0.0
            ddS[idx] = -DEFAULT_KARL_ENTROPY_FLOOR
        end
    end

    return entropy, chi_slack, dS, ddS
end

function karl_spear_update_expanded(w_all::Vector{Float64}, Cm::Matrix{Float64}, target::Vector{Float64}, dS::Vector{Float64}, ddS::Vector{Float64}; apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Narr, nvar = size(Cm)

    length(w_all) == nvar || error("w_all length must match Cm columns")
    length(target) == Narr || error("target length must match Cm rows")
    length(dS) == nvar || error("dS length must match Cm columns")
    length(ddS) == nvar || error("ddS length must match Cm columns")

    model = Cm * w_all
    delY = target .- model

    Am = karl_spear_build_Am(Cm, ddS; floor=entropy_floor)
    karl_spear_rhs!(delY, Cm, dS, ddS; floor=entropy_floor)

    lambda = _solve_spear_system(Am, delY)
    dw = karl_spear_delta_w(Cm, lambda, dS, ddS; floor=entropy_floor)

    wnew = similar(w_all)
    @inbounds for j in eachindex(w_all)
        wnew[j] = w_all[j] + apfac * dw[j]
    end

    rcond_est = 1.0 / max(cond(Am), 1.0)
    max_abs_dw = maximum(abs.(dw))

    return (w=wnew, dw=dw, lambda=lambda, Am=Am, delY=delY, model=model, rcond_est=rcond_est, max_abs_dw=max_abs_dw)
end

function _project_expanded_weights!(w_all::Vector{Float64}, Norbit::Int; floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    Norbit <= length(w_all) || error("Norbit cannot exceed length(w_all)")

    worb = @view w_all[1:Norbit]

    @inbounds for i in eachindex(worb)
        (!isfinite(worb[i]) || worb[i] < floor) && (worb[i] = floor)
    end

    s = sum(worb)
    (!isfinite(s) || s <= 0.0) && return false
    worb ./= s

    @inbounds for i in (Norbit + 1):length(w_all)
        !isfinite(w_all[i]) && (w_all[i] = 0.0)
    end

    return true
end

function karl_spear_step_light_losvd_all(w_all::Vector{Float64}, Norbit::Int, A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, light_target::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}, wphase_orbit::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    # Karl SPEAR state rule:
    #   w_all = orbit weights followed by LOSVD slack weights.
    # The slack variables are state variables.  They must not be rebuilt from
    # target - A*w at each Newton/SPEAR step, or they hide the LOSVD residual.
    Cm = build_expanded_Cm_with_losvd_slack(A_light, A_losvd)
    target = build_expanded_target(light_target, losvd_target)
    length(w_all) == size(Cm, 2) || error("w_all length does not match expanded Cm columns")
    Norbit < length(w_all) || error("expanded Cm state must contain LOSVD slack variables after orbit weights")
    entropy, chi_slack, dS, ddS = build_expanded_entropy_derivatives(w_all, Norbit, wphase_orbit, losvd_sigma; alphat=alphat, entropy_floor=entropy_floor)
    out = karl_spear_update_expanded(w_all, Cm, target, dS, ddS; apfac=apfac, entropy_floor=entropy_floor)
    w_all_new = Vector{Float64}(out.w)

    # Expanded-Cm has two kinds of variables:
    #
    #   1:Norbit       = physical orbit weights
    #   Norbit+1:end   = LOSVD slack / residual variables
    #
    # Slack variables may be signed.  Orbit weights are luminosity/mass weights,
    # so they must remain finite, non-negative, and normalized.
    if !all(isfinite, w_all_new)
        return w_all, false, ( entropy=entropy, chi_slack=chi_slack, rcond_est=0.0, max_abs_dw=Inf, slack=Float64[], w_all=w_all)
    end
    if !_project_expanded_weights!(w_all_new, Norbit; floor=entropy_floor)
        return w_all, false, (entropy=entropy, chi_slack=chi_slack, rcond_est=0.0, max_abs_dw=Inf, slack=Float64[], w_all=w_all)
    end
    return w_all_new, true, ( entropy=entropy, chi_slack=chi_slack, rcond_est=out.rcond_est, max_abs_dw=out.max_abs_dw, slack=Vector{Float64}(w_all_new[(Norbit + 1):end]), w_all=w_all_new)
end

function karl_spear_step_light_losvd(w_orbit::Vector{Float64}, A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, light_target::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}, wphase_orbit::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, apfac::Float64=DEFAULT_KARL_APFAC, entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR)
    # Compatibility wrapper for old callers.  New production expanded-Cm solve
    # builds w_all once in solve_weights_karl_expanded_cm and carries it through
    # all SPEAR steps.
    w_all = build_expanded_weights_initial(w_orbit, A_losvd, losvd_target)
    w_all_new, ok, diag = karl_spear_step_light_losvd_all(w_all, length(w_orbit), A_light, A_losvd, light_target, losvd_target, losvd_sigma, wphase_orbit; alphat=alphat, apfac=apfac, entropy_floor=entropy_floor)

    return Vector{Float64}(w_all_new[1:length(w_orbit)]), ok, diag
end

##CHI##
function solve_weights_karl_expanded_cm(A_light::Matrix{Float64}, A_losvd::Matrix{Float64}, light_target::Vector{Float64}, light_sigma::Vector{Float64}, losvd_target::Vector{Float64}, losvd_sigma::Vector{Float64}; alphat::Float64=DEFAULT_KARL_ALPHAT, lambda_light::Float64=1.0, wphase=nothing, maxiter::Int=DEFAULT_KARL_MAXITER, seed::UInt=UInt(0), entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, apfac::Float64=DEFAULT_KARL_APFAC, return_diag::Bool=false)
    Nlight, Norbit = size(A_light)
    Nlosvd, Norbit2 = size(A_losvd)
    fail_w = zeros(Float64, Norbit)
    Norbit == Norbit2 || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(light_target) == Nlight || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(light_sigma) == Nlight || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(losvd_target) == Nlosvd || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    length(losvd_sigma) == Nlosvd || return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    Norbit <= 0 && return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, A_light) && return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, A_losvd) && return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, light_target) && return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    !all(isfinite, losvd_target) && return return_diag ? (fail_w, false, nothing) : (fail_w, false)
    lsig = max.(Float64.(light_sigma), 1e-12)
    vsig = max.(Float64.(losvd_sigma), 1e-12)
    # Put light rows into the same constrained system, but scale their pressure
    # with lambda_light so the external daemon weight retains its old meaning.
    lscale = sqrt(max(lambda_light, 0.0)) ./ lsig
    vscale = ones(Float64, Nlosvd)
    A_light_w = A_light .* lscale
    light_target_w = light_target .* lscale
    A_losvd_w = A_losvd .* vscale
    losvd_target_w = losvd_target .* vscale
    losvd_sigma_w = vsig
    wp = _prepare_wphase(wphase, Norbit; entropy_floor=entropy_floor)

    if iseven(Norbit)
        w = karl_initial_weights_from_wphase(wp; paired=true, rotfrac=0.75, floor=entropy_floor)
    else
        w = karl_initial_weights_from_wphase(wp; paired=false, floor=entropy_floor)
    end

    # Build the full Karl SPEAR state once.  Orbit weights occupy columns
    # 1:Norbit.  LOSVD slack weights occupy the remaining columns.  The slack
    # block is then carried forward as state, matching Karl's spear.f behavior.
    w_all = build_expanded_weights_initial(w, A_losvd_w, losvd_target_w)
    last_diag = nothing
    ok = true

    for _ in 1:maxiter
        w_all_new, step_ok, sdiag = karl_spear_step_light_losvd_all(w_all, Norbit, A_light_w, A_losvd_w, light_target_w, losvd_target_w, losvd_sigma_w, wp; alphat=alphat, apfac=apfac, entropy_floor=entropy_floor)
        last_diag = sdiag
        if !step_ok
            ok = false
            break
        end
        if norm(w_all_new .- w_all) <= 1e-8 * max(1.0, norm(w_all))
            w_all .= w_all_new
            break
        end
        w_all .= w_all_new
    end

    w = Vector{Float64}(w_all[1:Norbit])
    slack = Vector{Float64}(w_all[(Norbit + 1):end])
    (!ok || !all(isfinite, w) || !all(isfinite, slack)) && return return_diag ? (fail_w, false, last_diag) : (fail_w, false)

    if return_diag
        chi_losvd = chi2_block(A_losvd, w, losvd_target, vsig)
        chi_light = chi2_block(A_light, w, light_target, lsig)
        ent = karl_entropy_value(w, wp; entropy_floor=entropy_floor)
        diag = (entropy=ent, chi=chi_losvd + lambda_light * chi_light, chi_losvd=chi_losvd, chi_light=chi_light, profit=ent - alphat * chi_losvd - lambda_light * chi_light, alphat=alphat, rcond_est=last_diag === nothing ? NaN : last_diag.rcond_est, max_abs_dw=last_diag === nothing ? NaN : last_diag.max_abs_dw, chi_slack=last_diag === nothing ? NaN : last_diag.chi_slack, slack_l2=sum(slack .^ 2), slack_max_abs=isempty(slack) ? 0.0 : maximum(abs.(slack)), N_slack=Nlosvd)
        return (w, true, diag)
    end
    return (w, true)
end



function solve_weights_karl_jl(A::Matrix{Float64}, d::Vector{Float64}, sigma::Vector{Float64}; alpha::Float64=DEFAULT_KARL_ALPHA, alphat::Float64=DEFAULT_KARL_ALPHAT, weight_mode=:entropy, wphase=nothing, maxiter::Int=DEFAULT_KARL_MAXITER, seed::UInt=UInt(0), entropy_floor::Float64=DEFAULT_KARL_ENTROPY_FLOOR, return_diag::Bool=false)

    # Live Karl path.
    # weight_mode is accepted for the existing call contract, but the solver now
    # uses Karl entropy + SPEAR updates as the actual production behavior.
    return solve_weights_karl_spear( A, d, sigma; alphat=alphat, wphase=wphase, maxiter=maxiter, seed=seed, entropy_floor=entropy_floor, apfac=DEFAULT_KARL_APFAC, return_diag=return_diag )
end

##CHI##
@inline function chi2_block(A::Matrix{Float64}, w::Vector{Float64}, d::Vector{Float64}, sigma::Vector{Float64})
    p = A * w
    s = 0.0

    @inbounds for i in eachindex(d)
        si = max(sigma[i], 1e-12)
        rr = (p[i] - d[i]) / si
        s += rr * rr
    end

    return s
end


##CHI##
function chi2_block_karl_fracnew(A::Matrix{Float64}, w::Vector{Float64}, d::Vector{Float64}, sigma::Vector{Float64}, Nspatial::Int, Nvbin::Int; valid_mask=nothing, invalid_sigma_sentinel::Float64=-666.0, huge_den::Float64=(1e6)^2)

    p = A * w
    length(p) == length(d) == length(sigma) || error("chi2_block_karl_fracnew length mismatch")
    Nspatial * Nvbin <= length(d) || error("Nspatial*Nvbin exceeds target length")

    mask = if valid_mask === nothing
        [sigma[i] != invalid_sigma_sentinel for i in eachindex(sigma)]
    else
        collect(Bool, valid_mask)
    end
    length(mask) == length(d) || error("valid_mask length mismatch")

    s = 0.0

    @inbounds for ib in 1:Nspatial
        rows = ((ib - 1) * Nvbin + 1):(ib * Nvbin)
        sumt = 0.0
        sumt2 = 0.0

        for row in rows
            sumt += p[row]
            mask[row] && (sumt2 += p[row])
        end

        fracnew = (isfinite(sumt) && abs(sumt) > 1e-300) ? (sumt2 / sumt) : 1.0
        (!isfinite(fracnew) || fracnew <= 0.0) && (fracnew = 1.0)

        for row in rows
            if !mask[row] || sigma[row] == invalid_sigma_sentinel
                res = d[row] * fracnew - p[row]
                s += (res * res) / huge_den
            else
                si = max(abs(sigma[row] * fracnew), 1e-12)
                rr = (d[row] * fracnew - p[row]) / si
                s += rr * rr
            end
        end
    end

    if Nspatial * Nvbin < length(d)
        @inbounds for row in (Nspatial * Nvbin + 1):length(d)
            si = max(sigma[row], 1e-12)
            rr = (p[row] - d[row]) / si
            s += rr * rr
        end
    end

    return s
end
