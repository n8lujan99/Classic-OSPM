# OSPM/Physics/OSPM_PhysicsEngine.py
# Python safety wrapper + metadata packer.
#
# This wrapper does not score an A-matrix in Python.
# Karl-style OSPM scoring is done in Julia from binned LOSVD rows plus
# projected-light / surface-brightness rows.  The Python layer only validates
# observed arrays, wraps a scalar-returning engine, and attaches obs/config
# metadata so the daemon can call evaluate_batch_theta directly.

import numpy as np

_PRINT_EVERY = 200
_print_counter = 0


def _as_array_or_none(x):
    if x is None:
        return None
    return np.asarray(x, float).ravel()


def _cfg_get(cfg, upper_name, lower_name=None, default=None):
    lower_name = lower_name or upper_name.lower()

    if upper_name in cfg:
        return cfg[upper_name]

    if lower_name in cfg:
        return cfg[lower_name]

    return default


def _get_required_obs_array(obs, primary_name, fallback_name=None):
    value = getattr(obs, primary_name, None)

    if value is None and fallback_name is not None:
        value = getattr(obs, fallback_name, None)

    if value is None:
        names = primary_name if fallback_name is None else f"{primary_name} or {fallback_name}"
        raise AttributeError(f"obs must expose {names}")

    arr = np.asarray(value, float).ravel()

    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"obs.{primary_name} must be a non-empty 1D array")

    if not np.any(np.isfinite(arr)):
        raise ValueError(f"obs.{primary_name} has no finite values")

    return arr


def _get_surface_brightness_profile(obs, cfg):
    profile = _cfg_get(cfg, "SURFACE_BRIGHTNESS_PROFILE", "surface_brightness_profile", None)

    if profile is None:
        profile = getattr(obs, "surface_brightness_profile", None)

    if profile is None:
        profile = getattr(obs, "SurfaceBrightnessProfile", None)

    if profile is None:
        raise ValueError(
            "surface_brightness_profile is required for Karl-style OSPM; "
            "no star-count fallback is allowed"
        )

    return profile


def _get_kinematic_bin_edges_pc(obs, cfg):
    edges = _cfg_get(cfg, "KINEMATIC_BIN_EDGES_PC", "kinematic_bin_edges_pc", None)

    if edges is None:
        edges = getattr(obs, "kinematic_bin_edges_pc", None)

    if edges is None:
        return None

    edges = np.asarray(edges, float).ravel()

    if edges.size < 2:
        raise ValueError("kinematic_bin_edges_pc must contain at least two edges")

    if not np.all(np.isfinite(edges)):
        raise ValueError("kinematic_bin_edges_pc contains non-finite values")

    if not np.all(np.diff(edges) > 0):
        raise ValueError("kinematic_bin_edges_pc must be strictly increasing")

    return edges


def _get_velocity_edges(cfg):
    edges = _cfg_get(cfg, "VELOCITY_EDGES", "velocity_edges", None)

    if edges is None:
        return None

    edges = np.asarray(edges, float).ravel()

    if edges.size < 2:
        raise ValueError("velocity_edges must contain at least two edges")

    if not np.all(np.isfinite(edges)):
        raise ValueError("velocity_edges contains non-finite values")

    if not np.all(np.diff(edges) > 0):
        raise ValueError("velocity_edges must be strictly increasing")

    return edges


def _get_valid_vlos(obs, R_star_m, v_star_mps, verr_star_mps):
    valid_vlos = getattr(obs, "valid_vlos", None)

    if valid_vlos is None:
        valid_vlos = getattr(obs, "has_vlos", None)

    if valid_vlos is None:
        valid_vlos = np.ones(R_star_m.size, dtype=bool)
    else:
        valid_vlos = np.asarray(valid_vlos, bool).ravel()

    if valid_vlos.size != R_star_m.size:
        raise ValueError("obs.valid_vlos/has_vlos must match the length of the stellar arrays")

    valid_vlos = (
        valid_vlos
        & np.isfinite(R_star_m)
        & np.isfinite(v_star_mps)
        & np.isfinite(verr_star_mps)
        & (verr_star_mps > 0.0)
    )

    if np.count_nonzero(valid_vlos) == 0:
        raise ValueError("Karl-style OSPM needs at least one valid line-of-sight velocity")

    return valid_vlos


def wrap_physics_engine(base_engine, *, obs, halo_type, config=None):
    cfg = dict(config or {})

    print_every = int(_cfg_get(cfg, "PRINT_EVERY", "print_every", _PRINT_EVERY))

    R_star_m = _get_required_obs_array(obs, "R_star_m", "R_m")
    v_star_mps = _get_required_obs_array(obs, "v_star_mps", "v_mps")

    verr_star_mps = _as_array_or_none(getattr(obs, "verr_star_mps", None))
    if verr_star_mps is None:
        verr_star_mps = _as_array_or_none(getattr(obs, "verr_mps", None))

    if verr_star_mps is None:
        raise AttributeError("obs must expose verr_star_mps or verr_mps")

    if not (R_star_m.size == v_star_mps.size == verr_star_mps.size):
        raise ValueError(
            "obs arrays must have matching lengths: "
            f"R={R_star_m.size}, v={v_star_mps.size}, verr={verr_star_mps.size}"
        )

    valid_vlos = _get_valid_vlos(obs, R_star_m, v_star_mps, verr_star_mps)
    surface_brightness_profile = _get_surface_brightness_profile(obs, cfg)
    kinematic_bin_edges_pc = _get_kinematic_bin_edges_pc(obs, cfg)
    velocity_edges = _get_velocity_edges(cfg)

    min_stars_per_bin = int(_cfg_get(cfg, "MIN_STARS_PER_BIN", "min_stars_per_bin", 20))
    Nvbin = int(_cfg_get(cfg, "NVBIN", "Nvbin", 21))
    Ntheta_launch = int(_cfg_get(cfg, "NTHETA_LAUNCH", "Ntheta_launch", 9))
    lambda_light = float(
        cfg.get(
            "LAMBDA_LIGHT",
            cfg.get("lambda_light", cfg.get("LAMBDA_OCC", cfg.get("lambda_occ", 1.0))),
        )
    )

    if min_stars_per_bin <= 0:
        raise ValueError("MIN_STARS_PER_BIN/min_stars_per_bin must be positive")

    if Nvbin <= 0:
        raise ValueError("NVBIN/Nvbin must be positive")

    if Ntheta_launch <= 0:
        raise ValueError("NTHETA_LAUNCH/Ntheta_launch must be positive")

    def engine(theta):
        global _print_counter

        _print_counter += 1
        out = base_engine(theta)

        if isinstance(out, (float, int, np.floating, np.integer)):
            chi2 = float(out)
        elif (
            isinstance(out, (tuple, list))
            and len(out) > 0
            and isinstance(out[0], (float, int, np.floating, np.integer))
        ):
            chi2 = float(out[0])
        else:
            raise TypeError(
                "Karl-style OSPM wrapper expects base_engine(theta) to return a scalar chi2. "
                "Python-side A-matrix scoring has been removed."
            )

        if not np.isfinite(chi2):
            return float("inf")

        if print_every > 0 and (_print_counter % print_every == 0):
            MBH = float(theta[2]) if len(theta) > 2 else 0.0
            print(f"[PHYS] chi2_losvd={chi2:10.4f} MBH={MBH:9.3e}")

        return chi2

    # Attach obs and config so the daemon can bypass this scalar wrapper and call
    # Julia evaluate_batch_theta in batch mode.
    engine.__wrapped_obs__ = obs
    engine.__halo_type__ = str(halo_type).strip().lower()
    engine.__surface_brightness_profile__ = surface_brightness_profile
    engine.__valid_vlos__ = valid_vlos
    engine.__R_star_m__ = R_star_m
    engine.__v_star_mps__ = v_star_mps
    engine.__verr_star_mps__ = verr_star_mps
    engine.__kinematic_bin_edges_pc__ = kinematic_bin_edges_pc
    engine.__velocity_edges__ = velocity_edges
    engine.__karl_config__ = {
        "surface_brightness_profile": surface_brightness_profile,
        "kinematic_bin_edges_pc": kinematic_bin_edges_pc,
        "velocity_edges": velocity_edges,
        "min_stars_per_bin": min_stars_per_bin,
        "Nvbin": Nvbin,
        "Ntheta_launch": Ntheta_launch,
        "lambda_light": lambda_light,
    }

    return engine
