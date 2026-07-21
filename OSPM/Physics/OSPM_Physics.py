# OSPM/Physics/OSPM_Physics.py
# Python-to-Julia physics bridge
# Python owns contracts, config plumbing, and conversion.
# Julia owns orbit integration, binned LOSVD/light A-matrix construction, weights, and chi2.

import os
import sys
import numpy as np

# --- PythonCall / JuliaCall must see these BEFORE importing juliacall ---
os.environ["PYTHON"] = sys.executable

# repo root owns Project.toml / Manifest.toml
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---- Force JuliaCall to use repo project deterministically ----
os.environ.setdefault("PYTHON_JULIACALL_PROJECT", _REPO_ROOT)
os.environ.setdefault("PYTHON_JULIACALL_EXE", os.path.expanduser("~/.juliaup/bin/julia"))

# ---- Embedded stability ----
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("JULIA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

USE_JULIA = os.environ.get("OSPM_USE_JULIA", "0").strip().lower() in ("1", "true", "yes")

_JL_READY = False
_Main = None
_LAST_SIG = None

print("[PY] OSPM_Physics Karl bridge imported from:", __file__)

pc = 3.085677581e16
kms = 1.0e3
Msun = 1.98847e30
G = 6.67430e-11
c = 2.99792458e8
_NFW_VCIRC_DENOM = 4.0 * np.pi * G * (np.log(2.0) - 0.5)

def _jl_init():
    global _JL_READY, _Main

    if _JL_READY:
        return

    if not USE_JULIA:
        raise RuntimeError("OSPM_USE_JULIA is not enabled")

    if "torch" in sys.modules:
        print("[WARN] torch imported before juliacall. This can be unstable on some systems.")

    os.environ.setdefault("JULIA_PROJECT", _REPO_ROOT)

    from juliacall import Main as _Main

    here = os.path.dirname(os.path.abspath(__file__))
    jl_path = os.path.join(here, "OSPM_Physics_Spherical.jl")

    if not os.path.exists(jl_path):
        raise FileNotFoundError(f"Julia backend file not found: {jl_path}")

    if not hasattr(_Main, "OSPMPhysicsSpherical"):
        # Safer HPC bridge include:
        # pass the path as a plain Julia string inside seval instead of
        # routing the Python string through _Main.include(jl_path).
        import json
        jl_path_literal = json.dumps(str(jl_path))

        _Main.seval(f"""
        try
            jl_path = {jl_path_literal}
            println("[JLINIT] Including Julia physics file: ", jl_path)
            Base.include(Main, jl_path)
            println("[JLINIT] Julia physics include OK")
        catch err
            println("[JLINIT ERROR TYPE]")
            println(typeof(err))

            println("[JLINIT ERROR]")
            showerror(stdout, err)
            println()

            println("[JLINIT STACKTRACE]")
            for frame in stacktrace(catch_backtrace())
                println(frame)
            end

            rethrow(err)
        end
        """)

    if not hasattr(_Main, "OSPMPhysicsSpherical"):
        raise RuntimeError("OSPMPhysicsSpherical failed to load into Main")

    _JL_READY = True

def _normalize_halo_parameterization(halo_parameterization=None):
    hp = "rho_rs" if halo_parameterization is None else str(halo_parameterization).strip().lower()
    if hp in ("", "default"):
        hp = "rho_rs"
    if hp not in ("rho_rs", "vcirc_rs"):
        raise ValueError(f"Unknown HALO_PARAMETERIZATION: {halo_parameterization!r}")
    return hp

def normalize_halo_parameterization(halo_parameterization=None):
    return _normalize_halo_parameterization(halo_parameterization)

def nfw_vcirc_rs_to_rho_s(vcirc_kms, r_s_pc):
    vcirc_mps = float(vcirc_kms) * kms
    r_s_m = float(r_s_pc) * pc
    if not np.isfinite(vcirc_mps) or not np.isfinite(r_s_m):
        raise ValueError("vcirc and r_s must be finite for vcirc_rs conversion")
    if r_s_m <= 0.0:
        raise ValueError("r_s must be positive for vcirc_rs conversion")
    rho_kg_m3 = (vcirc_mps * vcirc_mps) / (_NFW_VCIRC_DENOM * r_s_m * r_s_m)
    return rho_kg_m3 * pc**3 / Msun

def canonicalize_theta(theta, *, halo_type, halo_parameterization=None, bounds=None, require_mbh=True, require_ml=True):
    t = np.asarray(theta, float).ravel()

    if t.size < 2:
        raise ValueError("theta too short")

    hp = _normalize_halo_parameterization(halo_parameterization)
    first_name = "vcirc" if hp == "vcirc_rs" else "rho_s"

    if require_mbh and t.size < 3:
        raise ValueError(f"theta missing MBH; expects [{first_name}, r_s, MBH, ML]")

    if require_ml and t.size < 4:
        raise ValueError(f"theta missing ML; expects [{first_name}, r_s, MBH, ML]")

    ncheck = 4 if require_ml else 3

    if not np.all(np.isfinite(t[:ncheck])):
        raise ValueError("theta has non-finite values")

    first = float(t[0])
    r_s = float(t[1])
    MBH = float(t[2]) if t.size >= 3 else 0.0
    ML = float(t[3]) if t.size >= 4 else 1.0
    ht = str(halo_type).strip().lower()

    if bounds is not None:
        b = np.asarray(bounds, float)
        need = 4 if require_ml else 3

        if b.shape[0] < need:
            raise ValueError(f"bounds must cover at least first {need} parameters")

        vals = (first, r_s, MBH, ML)[:need]

        for i, x in enumerate(vals):
            lo, hi = float(b[i, 0]), float(b[i, 1])

            if not (lo <= x <= hi):
                raise ValueError(f"theta out of bounds at i={i}: {x} not in [{lo}, {hi}]")

    rho_s = nfw_vcirc_rs_to_rho_s(first, r_s) if hp == "vcirc_rs" else first

    return (rho_s, r_s, MBH, ML, ht)

def _theta_sig(theta, halo_type, halo_parameterization=None):
    return canonicalize_theta(theta, halo_type=halo_type, halo_parameterization=halo_parameterization)
def assert_theta_contract(theta, *, halo_type, bounds=None, require_mbh=True, require_ml=True, halo_parameterization=None):
    return canonicalize_theta( theta, halo_type=halo_type, halo_parameterization=halo_parameterization, bounds=bounds, require_mbh=require_mbh, require_ml=require_ml )
def _halo_parameterization_from_config(config=None):
    cfg = config or {}
    return _normalize_halo_parameterization(cfg.get("HALO_PARAMETERIZATION", "rho_rs"))

def canonicalize_theta_matrix(thetas, *, halo_type, halo_parameterization=None, bounds=None):
    theta_arr = np.asarray(thetas, dtype=float)
    if theta_arr.ndim != 2:
        raise RuntimeError("thetas must be a 2D array with shape (nparam, nbatch)")
    if theta_arr.shape[0] < 4:
        hp = _normalize_halo_parameterization(halo_parameterization)
        first_name = "vcirc" if hp == "vcirc_rs" else "rho_s"
        raise RuntimeError(f"thetas must have shape (4, nbatch): [{first_name}, r_s, MBH, ML]")
    out = theta_arr.copy()
    for i in range(theta_arr.shape[1]):
        out[:4, i] = assert_theta_contract( theta_arr[:4, i], halo_type=halo_type, halo_parameterization=halo_parameterization,
            bounds=bounds, require_mbh=True, require_ml=True)[:4]
    return out

def _as_float_vec(x, name):
    a = np.asarray(x, float).ravel()
    if a.size == 0:
        raise RuntimeError(f"{name} is empty")
    if not np.any(np.isfinite(a)):
        raise RuntimeError(f"{name} has no finite values")
    return a

def _get_obs_arrays(obs):
    R = getattr(obs, "R_star_m", None)
    v = getattr(obs, "v_star_mps", None)
    ve = getattr(obs, "verr_star_mps", None)
    if R is None:
        R = getattr(obs, "R_m", None)
    if v is None:
        v = getattr(obs, "v_mps", None)
    if ve is None:
        ve = getattr(obs, "verr_mps", None)
    if R is None or v is None or ve is None:
        raise AttributeError("obs must expose R_star_m, v_star_mps, and verr_star_mps")
    R = _as_float_vec(R, "R_star_m")
    v = _as_float_vec(v, "v_star_mps")
    ve = _as_float_vec(ve, "verr_star_mps")
    if not (R.size == v.size == ve.size):
        raise RuntimeError("R_star_m, v_star_mps, and verr_star_mps must have matching lengths")
    return R, v, ve

def _get_valid_vlos(obs, R, v, ve):
    valid = getattr(obs, "valid_vlos", None)
    if valid is None:
        valid = getattr(obs, "has_vlos", None)
    if valid is None:
        valid = np.ones(R.size, dtype=bool)
    else:
        valid = np.asarray(valid, bool).ravel()
        if valid.size != R.size:
            raise RuntimeError("valid_vlos/has_vlos length does not match star arrays")
    valid = valid & np.isfinite(R) & np.isfinite(v) & np.isfinite(ve) & (ve > 0)
    if not np.any(valid):
        raise RuntimeError("No valid line-of-sight velocity stars")
    return valid

def _get_surface_brightness_profile(obs=None, ctx=None, config=None):
    cfg = config or {}
    for key in ("surface_brightness_profile", "SURFACE_BRIGHTNESS_PROFILE"):
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    if isinstance(ctx, dict):
        for key in ("surface_brightness_profile", "SURFACE_BRIGHTNESS_PROFILE"):
            if key in ctx and ctx[key] is not None:
                return ctx[key]
    if obs is not None:
        for name in ("surface_brightness_profile", "SurfaceBrightnessProfile", "SURFACE_BRIGHTNESS_PROFILE"):
            if hasattr(obs, name):
                value = getattr(obs, name)
                if value is not None:
                    return value
    raise RuntimeError(
        "surface_brightness_profile is required for Karl-style OSPM; "
        "no star-count fallback is allowed"
    )

def _get_karl_options(obs=None, config=None):
    cfg = config or {}

    def grab(name, default):
        if name in cfg:
            return cfg[name]
        upper = name.upper()
        if upper in cfg:
            return cfg[upper]
        if obs is not None and hasattr(obs, name):
            return getattr(obs, name)
        if obs is not None and hasattr(obs, upper):
            return getattr(obs, upper)
        return default
    velocity_edges = grab("velocity_edges", None)
    light_bin_edges_pc = grab("light_bin_edges_pc", None)
    kinematic_bin_edges_pc = grab("kinematic_bin_edges_pc", None)
    return {
        "min_stars_per_bin": int(grab("min_stars_per_bin", 20)),
        "Nvbin": int(grab("Nvbin", 21)),
        "Ntheta_launch": int(grab("Ntheta_launch", 9)),
        "velocity_edges": velocity_edges,
        "light_bin_edges_pc": light_bin_edges_pc,
        "kinematic_bin_edges_pc": kinematic_bin_edges_pc,
    }

def maybe_reset_orbit_cache(theta, halo_type):
    # The Julia force/context cache is keyed by theta, halo type, and stellar-model
    # signature.  The old explicit orbit-cache reset path belonged to the removed
    # star-level likelihood backend and can make the Karl bridge fail if that
    # legacy Julia function is not exported.
    return

def mass_enclosed_two_radii_julia(*, r_in_m, r_out_m, theta, halo_type, stellar_model=None, halo_parameterization=None):
    if not USE_JULIA:
        raise RuntimeError("Julia required")
    _jl_init()
    rho_s, r_s, MBH, ML, ht = assert_theta_contract(theta, halo_type=halo_type, halo_parameterization=halo_parameterization, require_mbh=True, require_ml=True)
    Min, Mout = _Main.OSPMPhysicsSpherical.mass_enclosed_two_radii( float(r_in_m), float(r_out_m), float(rho_s), float(r_s), float(MBH), float(ML), str(ht), stellar_model=stellar_model)
    return float(Min), float(Mout)

def make_inclination(inclination_deg: float):
    inc = np.radians(float(inclination_deg))
    sini = float(np.sin(inc))
    cosi = float(np.cos(inc))
    edge_on = float(inclination_deg) >= 85.0
    return sini, cosi, edge_on

def halo_from_theta_astro(theta, halo_type="nfw", halo_parameterization=None):
    rho_s, r_s, MBH, ML, ht = assert_theta_contract( theta, halo_type=halo_type, halo_parameterization=halo_parameterization, require_mbh=True, require_ml=True,)
    return {"rho_s": rho_s, "r_s": r_s, "MBH": MBH, "ML": ML, "type": ht}

def build_dynamics_context(*, theta, halo_type, stellar_model=None, surface_brightness_profile=None, halo_parameterization=None, **_ignored):
    if not USE_JULIA:
        raise RuntimeError("Python backend is disabled. Set OSPM_USE_JULIA=1.")
    ctx = { "halo": halo_from_theta_astro(theta, halo_type=halo_type, halo_parameterization=halo_parameterization), "stellar_model": stellar_model, }
    if surface_brightness_profile is not None:
        ctx["surface_brightness_profile"] = surface_brightness_profile
    return ctx

def halo_kwargs_from_ctx(ctx):
    halo = ctx["halo"] if isinstance(ctx, dict) else ctx.halo
    for k in ("rho_s", "r_s", "MBH", "ML", "type"):
        if k not in halo:
            raise KeyError(f"halo missing required key '{k}'")
    return { "rho_s": float(halo["rho_s"]), "r_s": float(halo["r_s"]), "MBH": float(halo["MBH"]), "ML": float(halo["ML"]), "halo_type": str(halo["type"]),}

def build_A_matrix_karl_julia(*, R_star_m, valid_vlos, v_star_mps, verr_star_mps, sini, Norbit, theta, halo_type, stellar_model=None, surface_brightness_profile=None, halo_parameterization=None,
    return_occ=True, Nbins_occ=0, diag=False, velocity_edges=None, light_bin_edges_pc=None, kinematic_bin_edges_pc=None, min_stars_per_bin=20, Nvbin=21, Ntheta_launch=9):
    if not USE_JULIA:
        raise RuntimeError("Karl A-matrix mode requires Julia")
    if surface_brightness_profile is None:
        raise RuntimeError(
            "surface_brightness_profile is required for Karl-style OSPM;"
            "no star-count fallback is allowed"
        )

    _jl_init()
    rho_s, r_s, MBH, ML, ht = assert_theta_contract( theta, halo_type=halo_type, halo_parameterization=halo_parameterization, require_mbh=True, require_ml=True)
    maybe_reset_orbit_cache((rho_s, r_s, MBH, ML), ht)
    PC = _Main.PythonCall
    VecF = _Main.Vector[_Main.Float64]
    VecB = _Main.Vector[_Main.Bool]
    R_py = np.asarray(R_star_m, dtype=float).ravel()
    valid_py = np.asarray(valid_vlos, dtype=bool).ravel()
    v_py = np.asarray(v_star_mps, dtype=float).ravel()
    ve_py = np.asarray(verr_star_mps, dtype=float).ravel()
    if not (R_py.size == valid_py.size == v_py.size == ve_py.size):
        raise RuntimeError("R_star_m, valid_vlos, v_star_mps, and verr_star_mps must match")
    Rj = PC.pyconvert(VecF, R_py)
    validj = PC.pyconvert(VecB, valid_py)
    vj = PC.pyconvert(VecF, v_py)
    vej = PC.pyconvert(VecF, ve_py)
    kwargs = dict( stellar_model=stellar_model, surface_brightness_profile=surface_brightness_profile, return_occ=bool(return_occ), Nbins_occ=int(Nbins_occ), diag=bool(diag),
        min_stars_per_bin=int(min_stars_per_bin), Nvbin=int(Nvbin), Ntheta_launch=int(Ntheta_launch))
    if velocity_edges is not None:
        kwargs["velocity_edges"] = PC.pyconvert(VecF, np.asarray(velocity_edges, dtype=float).ravel())
    if light_bin_edges_pc is not None:
        kwargs["light_bin_edges"] = PC.pyconvert( VecF, np.asarray(light_bin_edges_pc, dtype=float).ravel() * pc )
    if kinematic_bin_edges_pc is not None:
        kwargs["kinematic_bin_edges"] = PC.pyconvert( VecF, np.asarray(kinematic_bin_edges_pc, dtype=float).ravel() * pc )
    out = _Main.OSPMPhysicsSpherical.build_A_matrix_hybrid( int(Norbit), Rj, validj, vj, vej, float(sini), float(rho_s), float(r_s), float(MBH), float(ML), str(ht), **kwargs)
    if diag:
        A, meta = out
        return np.asarray(A, float), dict(meta)
    return np.asarray(out, float)

def build_A_matrix(obs, ctx, *, return_occ=True, Nbins_occ=0, diag=False, config=None):
    mode = str(getattr(obs, "mode", "stellar")).strip().lower()
    if mode not in ("stellar", "karl", "losvd"):
        raise RuntimeError("build_A_matrix supports obs.mode in {'stellar', 'karl', 'losvd'} for Karl OSPM")
    hk = halo_kwargs_from_ctx(ctx)
    theta = [hk["rho_s"], hk["r_s"], hk["MBH"], hk["ML"]]
    halo_type = hk["halo_type"]
    stellar_model = (
        ctx.get("stellar_model", getattr(obs, "stellar_model", None))
        if isinstance(ctx, dict)
        else getattr(ctx, "stellar_model", getattr(obs, "stellar_model", None))
    )
    surface_brightness_profile = _get_surface_brightness_profile(obs=obs, ctx=ctx, config=config)
    R, v, ve = _get_obs_arrays(obs)
    valid = _get_valid_vlos(obs, R, v, ve)
    opts = _get_karl_options(obs=obs, config=config)
    return build_A_matrix_karl_julia( R_star_m=R, valid_vlos=valid, v_star_mps=v, verr_star_mps=ve, sini=float(obs.sini), Norbit=int(obs.Norbit), theta=theta, halo_type=halo_type,
        stellar_model=stellar_model, surface_brightness_profile=surface_brightness_profile, return_occ=bool(return_occ), Nbins_occ=int(Nbins_occ), diag=bool(diag),
        velocity_edges=opts["velocity_edges"], light_bin_edges_pc=opts["light_bin_edges_pc"], kinematic_bin_edges_pc=opts["kinematic_bin_edges_pc"], min_stars_per_bin=opts["min_stars_per_bin"], Nvbin=opts["Nvbin"], Ntheta_launch=opts["Ntheta_launch"])

def build_A_matrix_from_theta(obs, theta, *, halo_type="nfw", return_occ=True, Nbins_occ=0, diag=False, config=None):
    surface_brightness_profile = _get_surface_brightness_profile(obs=obs, config=config)
    halo_parameterization = _halo_parameterization_from_config(config)
    ctx = build_dynamics_context( theta=theta, halo_type=halo_type, halo_parameterization=halo_parameterization, stellar_model=getattr(obs, "stellar_model", None), surface_brightness_profile=surface_brightness_profile,)
    return build_A_matrix( obs, ctx, return_occ=bool(return_occ), Nbins_occ=int(Nbins_occ), diag=bool(diag), config=config,)

def evaluate_batch_theta_julia(
    *,
    thetas,
    obs,
    halo_type,
    stellar_model=None,
    surface_brightness_profile=None,
    Norbit=None,
    config=None,
):
    """
    Sole Python -> Julia batch-evaluation boundary.

    External theta:
        [vcirc, r_s, MBH, ML] when HALO_PARAMETERIZATION="vcirc_rs"

    Julia theta:
        [rho_s, r_s, MBH, ML]

    This function owns:
        - theta canonicalization
        - observational arrays
        - stellar model
        - full-light profile
        - light and kinematic bins
        - velocity bins
        - halo options
        - weight-solver options
        - scoring and timeout options
    """
    if not USE_JULIA:
        raise RuntimeError("Karl batch mode requires Julia")

    import json

    cfg = dict(config or {})
    observable_cfg = cfg.get("OBSERVABLES", {}) or {}

    if not isinstance(observable_cfg, dict):
        raise TypeError("config['OBSERVABLES'] must be a dict")

    def opt(*names, default=None):
        for source in (observable_cfg, cfg):
            for name in names:
                if name in source and source[name] is not None:
                    return source[name]

        for name in names:
            if hasattr(obs, name):
                value = getattr(obs, name)
                if value is not None:
                    return value

        return default

    if stellar_model is None:
        stellar_model = opt(
            "STELLAR_MODEL",
            "stellar_model",
            default=getattr(obs, "stellar_model", None),
        )

    if surface_brightness_profile is None:
        surface_brightness_profile = opt(
            "SURFACE_BRIGHTNESS_PROFILE",
            "surface_brightness_profile",
            default=None,
        )

    if surface_brightness_profile is None:
        surface_brightness_profile = _get_surface_brightness_profile(
            obs=obs,
            config=cfg,
        )

    light_bin_edges_pc = opt(
        "LIGHT_BIN_EDGES_PC",
        "light_bin_edges_pc",
        default=None,
    )
    kinematic_bin_edges_pc = opt(
        "KINEMATIC_BIN_EDGES_PC",
        "kinematic_bin_edges_pc",
        default=None,
    )
    velocity_edges = opt(
        "VELOCITY_EDGES",
        "velocity_edges",
        default=None,
    )

    if light_bin_edges_pc is None:
        raise RuntimeError("light_bin_edges_pc is required")

    if kinematic_bin_edges_pc is None:
        raise RuntimeError("kinematic_bin_edges_pc is required")

    min_stars_per_bin = int(
        opt("MIN_STARS_PER_BIN", "min_stars_per_bin", default=20)
    )
    Nvbin = int(
        opt("NVBIN", "Nvbin", "nvbin", default=21)
    )
    Ntheta_launch = int(
        opt("NTHETA_LAUNCH", "Ntheta_launch", "ntheta_launch", default=9)
    )

    Nocc = int(
        opt("NBINS_OCC", "Nocc", "nbins_occ", default=0)
    )
    lambda_light = float(
        opt(
            "LAMBDA_LIGHT",
            "LAMBDA_OCC",
            "lambda_light",
            "lambda_occ",
            default=1.0,
        )
    )

    alpha = float(
        opt("KARL_ALPHA", "ALPHA", "alpha", default=1e-4)
    )
    alphat = float(
        opt("KARL_ALPHAT", "ALPHAT", "alphat", default=1.0)
    )
    maxiter = int(
        opt("KARL_MAXITER", "MAXITER", "maxiter", default=60)
    )

    weight_mode = str(
        opt("WEIGHT_MODE", "weight_mode", default="entropy")
    ).strip().lower()

    weight_solver_mode = str(
        opt(
            "WEIGHT_SOLVER",
            "WEIGHT_SOLVER_MODE",
            "weight_solver_mode",
            default="orbit_only",
        )
    ).strip().lower()

    losvd_score_mode = str(
        opt(
            "LOSVD_SCORE_MODE",
            "losvd_score_mode",
            default="karl_fracnew",
        )
    ).strip().lower()

    entropy_floor = float(opt("ENTROPY_FLOOR", "entropy_floor", default=1e-12))
    max_refine = int(opt("MAX_REFINE", "max_refine", default=0))
    timeout_s = float(opt( "EVAL_TIMEOUT_S", "EVAL_TIMEOUT", "timeout_s", default=120.0, ))
    R_inner_pc = float(opt("R_INNER_DIAG_PC", "R_inner_pc", default=30.0))
    halo_q_axis_ratio = float( opt( "HALO_Q_AXIS_RATIO", "halo_q_axis_ratio", default=1.0, ))
    karl_halo_params = opt("KARL_HALO_PARAMS", "karl_halo_params", default=None)
    use_radial_vlos_weights = bool( opt("USE_RADIAL_VLOS_WEIGHTS", "use_radial_vlos_weights", default=False))
    use_weighted_score = bool( opt( "USE_WEIGHTED_SCORE", "use_weighted_score", default=False))
    R_weight_pc = float(opt("R_WEIGHT_PC", "R_weight_pc", default=-1.0))
    radial_weight_gamma = float(opt( "RADIAL_WEIGHT_GAMMA", "radial_weight_gamma", default=2.0, ))
    radial_weight_floor = float(opt( "RADIAL_WEIGHT_FLOOR", "radial_weight_floor", default=0.3,))
    R, v, ve = _get_obs_arrays(obs)
    valid = _get_valid_vlos(obs, R, v, ve)
    if Norbit is None:
        Norbit = int(getattr(obs, "Norbit"))
    if Norbit % 2 != 0:
        raise RuntimeError(
            "Karl paired-orbit mode requires an even Norbit; "
            f"got Norbit={Norbit}"
        )

    _jl_init()
    halo_parameterization = _halo_parameterization_from_config(cfg)
    theta_arr = canonicalize_theta_matrix( thetas, halo_type=halo_type, halo_parameterization=halo_parameterization, bounds=cfg.get("THETA_BOUNDS"))
    def jl_matrix_f64(value, name):
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
        nrow, ncol = arr.shape
        _Main._ospm_matrix_flat = arr.ravel(order="F").tolist()
        _Main._ospm_matrix_nrow = int(nrow)
        _Main._ospm_matrix_ncol = int(ncol)
        return _Main.seval("""reshape( Float64[x for x in _ospm_matrix_flat], _ospm_matrix_nrow, _ospm_matrix_ncol)""")
    def jl_vector_f64(value, name):
        arr = np.asarray(value, dtype=np.float64).ravel()
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
        _Main._ospm_vector_f64 = arr.tolist()
        return _Main.seval( "Float64[x for x in _ospm_vector_f64]")
    def jl_vector_bool(value):
        arr = np.asarray(value, dtype=bool).ravel()
        _Main._ospm_vector_bool = arr.tolist()
        return _Main.seval( "Bool[x for x in _ospm_vector_bool]" )
    def julia_literal(value, name):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            return "nothing"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError(f"{name} contains a non-finite value")
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value)
        raise TypeError(
            f"{name} contains unsupported value type "
            f"{type(value).__name__}"
        )
    def jl_primitive_dict(value, name):
        if value is None:
            return _Main.seval("nothing")
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a dict or None")
        entries = []
        for key, item in value.items():
            key_literal = json.dumps(str(key))
            value_literal = julia_literal(item, name)
            entries.append(f"{key_literal} => {value_literal}")
        expression = "Dict{String,Any}(" + ", ".join(entries) + ")"
        return _Main.seval(expression)
    def jl_surface_brightness_profile(profile):
        required = ("R_pc", "R_inner_pc", "R_outer_pc", "light_frac", "Sigma", "Sigma_err")
        missing = [key for key in required if key not in profile]
        if missing:
            raise KeyError( "surface_brightness_profile is missing " + ", ".join(missing))
        _Main._ospm_sb_R_pc = jl_vector_f64( profile["R_pc"], "surface_brightness_profile.R_pc",)
        _Main._ospm_sb_R_inner_pc = jl_vector_f64( profile["R_inner_pc"], "surface_brightness_profile.R_inner_pc" )
        _Main._ospm_sb_R_outer_pc = jl_vector_f64( profile["R_outer_pc"], "surface_brightness_profile.R_outer_pc")
        _Main._ospm_sb_light_frac = jl_vector_f64( profile["light_frac"], "surface_brightness_profile.light_frac",)
        _Main._ospm_sb_Sigma = jl_vector_f64( profile["Sigma"], "surface_brightness_profile.Sigma")
        _Main._ospm_sb_Sigma_err = jl_vector_f64( profile["Sigma_err"], "surface_brightness_profile.Sigma_err")
        return _Main.seval("""
            Dict{Symbol,Any}(
                :R_pc => _ospm_sb_R_pc,
                :R_inner_pc => _ospm_sb_R_inner_pc,
                :R_outer_pc => _ospm_sb_R_outer_pc,
                :light_frac => _ospm_sb_light_frac,
                :Sigma => _ospm_sb_Sigma,
                :Sigma_err => _ospm_sb_Sigma_err,
            )
        """)

    _Main._ospm_theta = jl_matrix_f64(theta_arr, "thetas")
    _Main._ospm_R = jl_vector_f64(R, "R_star_m")
    _Main._ospm_valid = jl_vector_bool(valid)
    _Main._ospm_v = jl_vector_f64(v, "v_star_mps")
    _Main._ospm_ve = jl_vector_f64(ve, "verr_star_mps")
    _Main._ospm_light_edges = jl_vector_f64( np.asarray(light_bin_edges_pc, dtype=float) * pc, "light_bin_edges_pc")
    _Main._ospm_kinematic_edges = jl_vector_f64( np.asarray(kinematic_bin_edges_pc, dtype=float) * pc, "kinematic_bin_edges_pc")
    if velocity_edges is None:
        _Main._ospm_velocity_edges = _Main.seval("nothing")
    else:
        _Main._ospm_velocity_edges = jl_vector_f64(velocity_edges, "velocity_edges")
    _Main._ospm_stellar_model = jl_primitive_dict( stellar_model, "stellar_model")
    _Main._ospm_karl_halo_params = jl_primitive_dict( karl_halo_params, "karl_halo_params")
    _Main._ospm_sb_profile = jl_surface_brightness_profile( surface_brightness_profile )
    _Main.seval(f"_ospm_sini = {float(obs.sini)!r}")
    _Main.seval(f"_ospm_Norbit = {int(Norbit)}")
    _Main.seval("_ospm_halo_type = " + json.dumps(str(halo_type)))
    _Main.seval(f"_ospm_Nocc = {Nocc}")
    _Main.seval(f"_ospm_lambda_light = {lambda_light!r}")
    _Main.seval(f"_ospm_alpha = {alpha!r}")
    _Main.seval(f"_ospm_alphat = {alphat!r}")
    _Main.seval("_ospm_weight_mode = " + json.dumps(weight_mode))
    _Main.seval("_ospm_weight_solver_mode = " + json.dumps(weight_solver_mode))
    _Main.seval("_ospm_losvd_score_mode = " + json.dumps(losvd_score_mode))
    _Main.seval(f"_ospm_entropy_floor = {entropy_floor!r}")
    _Main.seval(f"_ospm_maxiter = {maxiter}")
    _Main.seval(f"_ospm_max_refine = {max_refine}")
    _Main.seval(f"_ospm_timeout_s = {timeout_s!r}")
    _Main.seval(f"_ospm_R_inner_pc = {R_inner_pc!r}")
    _Main.seval(f"_ospm_min_stars_per_bin = {min_stars_per_bin}")
    _Main.seval(f"_ospm_Nvbin = {Nvbin}")
    _Main.seval(f"_ospm_Ntheta_launch = {Ntheta_launch}")
    _Main.seval( f"_ospm_halo_q_axis_ratio = {halo_q_axis_ratio!r}")
    _Main.seval("_ospm_use_radial_vlos_weights = " + ("true" if use_radial_vlos_weights else "false"))
    _Main.seval( "_ospm_use_weighted_score = " + ("true" if use_weighted_score else "false"))
    _Main.seval(f"_ospm_R_weight_pc = {R_weight_pc!r}")
    _Main.seval( f"_ospm_radial_weight_gamma = {radial_weight_gamma!r}")
    _Main.seval(f"_ospm_radial_weight_floor = {radial_weight_floor!r}")
    
    out = _Main.seval("""
        OSPMPhysicsSpherical.evaluate_batch_theta(
            _ospm_theta,
            _ospm_R,
            _ospm_valid,
            _ospm_v,
            _ospm_ve,
            _ospm_sini,
            _ospm_Norbit,
            _ospm_halo_type;
            stellar_model=_ospm_stellar_model,
            surface_brightness_profile=_ospm_sb_profile,
            Nocc=_ospm_Nocc,
            lambda_occ=_ospm_lambda_light,
            alpha=_ospm_alpha,
            alphat=_ospm_alphat,
            weight_mode=_ospm_weight_mode,
            weight_solver_mode=_ospm_weight_solver_mode,
            entropy_floor=_ospm_entropy_floor,
            losvd_score_mode=_ospm_losvd_score_mode,
            maxiter=_ospm_maxiter,
            max_refine=_ospm_max_refine,
            timeout_s=_ospm_timeout_s,
            R_inner_pc=_ospm_R_inner_pc,
            use_radial_vlos_weights=_ospm_use_radial_vlos_weights,
            use_weighted_score=_ospm_use_weighted_score,
            R_weight_pc=_ospm_R_weight_pc,
            radial_weight_gamma=_ospm_radial_weight_gamma,
            radial_weight_floor=_ospm_radial_weight_floor,
            velocity_edges=_ospm_velocity_edges,
            light_bin_edges=_ospm_light_edges,
            kinematic_bin_edges=_ospm_kinematic_edges,
            min_stars_per_bin=_ospm_min_stars_per_bin,
            Nvbin=_ospm_Nvbin,
            Ntheta_launch=_ospm_Ntheta_launch,
            halo_q_axis_ratio=_ospm_halo_q_axis_ratio,
            karl_halo_params=_ospm_karl_halo_params,
        )
    """)
    return tuple(np.asarray(value) for value in out)

def force_at_rtheta_julia(*, r_m, theta_rad, theta, halo_type, stellar_model=None, halo_parameterization=None):
    if not USE_JULIA:
        raise RuntimeError("Julia required")
    _jl_init()
    rho_s, r_s, MBH, ML, ht = assert_theta_contract( theta, halo_type=halo_type, halo_parameterization=halo_parameterization, require_mbh=True, require_ml=True)
    fr, ftheta, FR, FZ = _Main.OSPMPhysicsSpherical.force_at_rtheta( float(r_m), float(theta_rad), float(rho_s), float(r_s), float(MBH), float(ML), str(ht), stellar_model=stellar_model)
    return { "fr": float(fr), "ftheta": float(ftheta), "FR": float(FR), "FZ": float(FZ)}

def rho_interp(*args, **kwargs):
    raise RuntimeError("rho_interp is legacy-only. It should never be called in Karl Julia mode.")
