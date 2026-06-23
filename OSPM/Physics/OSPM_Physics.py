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

def _theta_sig(theta, halo_type):
    t = np.asarray(theta, float).ravel()

    if t.size < 2:
        raise ValueError("theta must have at least [rho_s, r_s]")

    rho_s = float(t[0])
    r_s = float(t[1])
    MBH = float(t[2]) if t.size >= 3 else 0.0
    ML = float(t[3]) if t.size >= 4 else 1.0
    ht = str(halo_type).strip().lower()

    return (rho_s, r_s, MBH, ML, ht)

def assert_theta_contract(theta, *, halo_type, bounds=None, require_mbh=True, require_ml=True):
    t = np.asarray(theta, float).ravel()

    if t.size < 2:
        raise ValueError("theta too short")

    if require_mbh and t.size < 3:
        raise ValueError("theta missing MBH; expects [rho_s, r_s, MBH, ML]")

    if require_ml and t.size < 4:
        raise ValueError("theta missing ML; expects [rho_s, r_s, MBH, ML]")

    ncheck = 4 if require_ml else 3

    if not np.all(np.isfinite(t[:ncheck])):
        raise ValueError("theta has non-finite values")

    rho_s = float(t[0])
    r_s = float(t[1])
    MBH = float(t[2]) if t.size >= 3 else 0.0
    ML = float(t[3]) if t.size >= 4 else 1.0
    ht = str(halo_type).strip().lower()

    if bounds is not None:
        b = np.asarray(bounds, float)
        need = 4 if require_ml else 3

        if b.shape[0] < need:
            raise ValueError(f"bounds must cover at least first {need} parameters")

        vals = (rho_s, r_s, MBH, ML)[:need]

        for i, x in enumerate(vals):
            lo, hi = float(b[i, 0]), float(b[i, 1])

            if not (lo <= x <= hi):
                raise ValueError(f"theta out of bounds at i={i}: {x} not in [{lo}, {hi}]")

    return (rho_s, r_s, MBH, ML, ht)

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
    kinematic_bin_edges_pc = grab("kinematic_bin_edges_pc", None)

    return {
        "min_stars_per_bin": int(grab("min_stars_per_bin", 20)),
        "Nvbin": int(grab("Nvbin", 21)),
        "Ntheta_launch": int(grab("Ntheta_launch", 9)),
        "velocity_edges": velocity_edges,
        "kinematic_bin_edges_pc": kinematic_bin_edges_pc,
    }

def maybe_reset_orbit_cache(theta, halo_type):
    # The Julia force/context cache is keyed by theta, halo type, and stellar-model
    # signature.  The old explicit orbit-cache reset path belonged to the removed
    # star-level likelihood backend and can make the Karl bridge fail if that
    # legacy Julia function is not exported.
    return

def mass_enclosed_two_radii_julia(*, r_in_m, r_out_m, theta, halo_type, stellar_model=None):
    if not USE_JULIA:
        raise RuntimeError("Julia required")

    _jl_init()

    rho_s, r_s, MBH, ML, ht = assert_theta_contract(theta, halo_type=halo_type, require_mbh=True, require_ml=True)
    Min, Mout = _Main.OSPMPhysicsSpherical.mass_enclosed_two_radii( float(r_in_m), float(r_out_m), float(rho_s), float(r_s), float(MBH), float(ML), str(ht), stellar_model=stellar_model)
    return float(Min), float(Mout)

def make_inclination(inclination_deg: float):
    inc = np.radians(float(inclination_deg))
    sini = float(np.sin(inc))
    cosi = float(np.cos(inc))
    edge_on = float(inclination_deg) >= 85.0
    return sini, cosi, edge_on

def halo_from_theta_astro(theta, halo_type="nfw"):
    rho_s, r_s, MBH, ML, ht = assert_theta_contract( theta, halo_type=halo_type, require_mbh=True, require_ml=True,)
    return {"rho_s": rho_s, "r_s": r_s, "MBH": MBH, "ML": ML, "type": ht}

def build_dynamics_context(*, theta, halo_type, stellar_model=None, surface_brightness_profile=None, **_ignored):
    if not USE_JULIA:
        raise RuntimeError("Python backend is disabled. Set OSPM_USE_JULIA=1.")
    ctx = { "halo": halo_from_theta_astro(theta, halo_type=halo_type), "stellar_model": stellar_model, }
    if surface_brightness_profile is not None:
        ctx["surface_brightness_profile"] = surface_brightness_profile
    return ctx

def halo_kwargs_from_ctx(ctx):
    halo = ctx["halo"] if isinstance(ctx, dict) else ctx.halo
    for k in ("rho_s", "r_s", "MBH", "ML", "type"):
        if k not in halo:
            raise KeyError(f"halo missing required key '{k}'")
    return { "rho_s": float(halo["rho_s"]), "r_s": float(halo["r_s"]), "MBH": float(halo["MBH"]), "ML": float(halo["ML"]), "halo_type": str(halo["type"]),}

def build_A_matrix_karl_julia(*, R_star_m, valid_vlos, v_star_mps, verr_star_mps, sini, Norbit, theta, halo_type, stellar_model=None, surface_brightness_profile=None,
    return_occ=True, Nbins_occ=0, diag=False, velocity_edges=None, kinematic_bin_edges_pc=None, min_stars_per_bin=20, Nvbin=21, Ntheta_launch=9):
    if not USE_JULIA:
        raise RuntimeError("Karl A-matrix mode requires Julia")

    if surface_brightness_profile is None:
        raise RuntimeError(
            "surface_brightness_profile is required for Karl-style OSPM; "
            "no star-count fallback is allowed"
        )

    _jl_init()
    rho_s, r_s, MBH, ML, ht = assert_theta_contract( theta, halo_type=halo_type, require_mbh=True, require_ml=True)
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
    kwargs = dict(
        stellar_model=stellar_model,
        surface_brightness_profile=surface_brightness_profile,
        return_occ=bool(return_occ),
        Nbins_occ=int(Nbins_occ),
        diag=bool(diag),
        min_stars_per_bin=int(min_stars_per_bin),
        Nvbin=int(Nvbin),
        Ntheta_launch=int(Ntheta_launch),
    )
    if velocity_edges is not None:
        kwargs["velocity_edges"] = PC.pyconvert(VecF, np.asarray(velocity_edges, dtype=float).ravel())
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
        velocity_edges=opts["velocity_edges"], kinematic_bin_edges_pc=opts["kinematic_bin_edges_pc"], min_stars_per_bin=opts["min_stars_per_bin"], Nvbin=opts["Nvbin"], Ntheta_launch=opts["Ntheta_launch"])

def build_A_matrix_from_theta(obs, theta, *, halo_type="nfw", return_occ=True, Nbins_occ=0, diag=False, config=None):
    surface_brightness_profile = _get_surface_brightness_profile(obs=obs, config=config)
    ctx = build_dynamics_context( theta=theta, halo_type=halo_type, stellar_model=getattr(obs, "stellar_model", None), surface_brightness_profile=surface_brightness_profile,)
    return build_A_matrix( obs, ctx, return_occ=bool(return_occ), Nbins_occ=int(Nbins_occ), diag=bool(diag), config=config,)

def evaluate_batch_theta_julia( *, thetas, obs, halo_type, stellar_model=None, surface_brightness_profile=None, Norbit=None, config=None,):
    if not USE_JULIA:
        raise RuntimeError("Karl batch mode requires Julia")

    surface_brightness_profile = surface_brightness_profile or _get_surface_brightness_profile(obs=obs, config=config)
    opts = _get_karl_options(obs=obs, config=config)

    R, v, ve = _get_obs_arrays(obs)
    valid = _get_valid_vlos(obs, R, v, ve)

    if Norbit is None:
        Norbit = int(getattr(obs, "Norbit"))

    _jl_init()

    PC = _Main.PythonCall
    VecF = _Main.Vector[_Main.Float64]
    VecB = _Main.Vector[_Main.Bool]

    theta_arr = np.asarray(thetas, dtype=float)

    if theta_arr.ndim != 2:
        raise RuntimeError("thetas must be a 2D array with shape (nparam, nbatch)")

    if theta_arr.shape[0] < 4:
        raise RuntimeError("thetas must have shape (4, nbatch): [rho_s, r_s, MBH, ML]")

    Rj = PC.pyconvert(VecF, R)
    validj = PC.pyconvert(VecB, valid)
    vj = PC.pyconvert(VecF, v)
    vej = PC.pyconvert(VecF, ve)

    kwargs = dict(
        stellar_model=stellar_model if stellar_model is not None else getattr(obs, "stellar_model", None),
        surface_brightness_profile=surface_brightness_profile,
        min_stars_per_bin=opts["min_stars_per_bin"],
        Nvbin=opts["Nvbin"],
        Ntheta_launch=opts["Ntheta_launch"],
    )

    if opts["velocity_edges"] is not None:
        kwargs["velocity_edges"] = PC.pyconvert(VecF, np.asarray(opts["velocity_edges"], dtype=float).ravel())
    if opts["kinematic_bin_edges_pc"] is not None:
        kwargs["kinematic_bin_edges"] = PC.pyconvert( VecF, np.asarray(opts["kinematic_bin_edges_pc"], dtype=float).ravel() * pc, )
    out = _Main.OSPMPhysicsSpherical.evaluate_batch_theta(theta_arr, Rj, validj, vj, vej, float(obs.sini), int(Norbit), str(halo_type), **kwargs,)
    return tuple(np.asarray(x) for x in out)


def force_at_rtheta_julia(*, r_m, theta_rad, theta, halo_type, stellar_model=None):
    if not USE_JULIA:
        raise RuntimeError("Julia required")

    _jl_init()

    rho_s, r_s, MBH, ML, ht = assert_theta_contract(
        theta,
        halo_type=halo_type,
        require_mbh=True,
        require_ml=True,
    )

    fr, ftheta, FR, FZ = _Main.OSPMPhysicsSpherical.force_at_rtheta(
        float(r_m),
        float(theta_rad),
        float(rho_s),
        float(r_s),
        float(MBH),
        float(ML),
        str(ht),
        stellar_model=stellar_model,
    )

    return {
        "fr": float(fr),
        "ftheta": float(ftheta),
        "FR": float(FR),
        "FZ": float(FZ),
    }

def rho_interp(*args, **kwargs):
    raise RuntimeError("rho_interp is legacy-only. It should never be called in Karl Julia mode.")
