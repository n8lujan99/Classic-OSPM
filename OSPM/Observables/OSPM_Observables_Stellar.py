"""
OSPM_Observables_Stellar / Karl-style observable container.
Loads observed stars, surface-brightness profile, and optional precomputed
kinematic radial bins. This file does not do gravity or solve weights.
"""
import numpy as np
import pandas as pd
from ..Physics.OSPM_Physics import pc, kms, make_inclination

def _as_bool_array(x):
    return np.array([str(v).strip().lower() in {"1", "true", "yes", "y", "t"} for v in x], dtype=bool)

def _load_surface_brightness_profile(path):
    sb = pd.read_csv(path)
    required = {"R_inner_pc", "R_outer_pc", "R_pc", "light_frac"}
    missing = required - set(sb.columns)
    if missing:
        raise KeyError(f"Surface brightness profile missing columns: {sorted(missing)}")
    out = {
        "R_pc": sb["R_pc"].to_numpy(float),
        "R_inner_pc": sb["R_inner_pc"].to_numpy(float),
        "R_outer_pc": sb["R_outer_pc"].to_numpy(float),
        "light_frac": sb["light_frac"].to_numpy(float),
        "Sigma": sb["Sigma"].to_numpy(float) if "Sigma" in sb.columns else None,
        "Sigma_err": sb["Sigma_err"].to_numpy(float) if "Sigma_err" in sb.columns else None,
    }
    return out

def _load_kinematic_bins(path):
    kb = pd.read_csv(path)
    required = {"bin_id", "R_inner_pc", "R_outer_pc", "R_mid_pc", "N_vlos"}
    missing = required - set(kb.columns)
    if missing:
        raise KeyError(f"Kinematic bin file missing columns: {sorted(missing)}")
    edges = np.r_[kb["R_inner_pc"].to_numpy(float)[0], kb["R_outer_pc"].to_numpy(float)]
    return {
        "bin_id": kb["bin_id"].to_numpy(int),
        "R_inner_pc": kb["R_inner_pc"].to_numpy(float),
        "R_outer_pc": kb["R_outer_pc"].to_numpy(float),
        "R_mid_pc": kb["R_mid_pc"].to_numpy(float),
        "N_vlos": kb["N_vlos"].to_numpy(int),
        "edges_pc": edges,
    }

def _validate_surface_brightness_against_kinematic_bins(surface_brightness_profile, kinematic_bins):
    if surface_brightness_profile is None:
        raise KeyError("Karl-style observables require a surface_brightness_profile")
    if "light_frac" not in surface_brightness_profile:
        raise KeyError("surface_brightness_profile must include light_frac")
    light_frac = np.asarray(surface_brightness_profile["light_frac"], float)
    if light_frac.ndim != 1:
        raise ValueError("surface_brightness_profile light_frac must be one-dimensional")
    if light_frac.size == 0:
        raise ValueError("surface_brightness_profile light_frac is empty")
    if not np.all(np.isfinite(light_frac)):
        raise ValueError("surface_brightness_profile light_frac contains non-finite values")
    if np.any(light_frac < 0.0):
        raise ValueError("surface_brightness_profile light_frac contains negative values")
    if not np.isfinite(light_frac.sum()) or light_frac.sum() <= 0.0:
        raise ValueError("surface_brightness_profile light_frac must sum to a positive finite value")
    if kinematic_bins is None:
        raise KeyError("Karl-style observables require KINEMATIC_BINS_CSV; no adaptive radial-bin fallback is allowed")
    if "R_mid_pc" not in kinematic_bins:
        raise KeyError("kinematic_bins must include R_mid_pc")
    n_light = int(light_frac.size)
    n_kin = int(len(kinematic_bins["R_mid_pc"]))
    if n_light != n_kin:
        raise ValueError(
            "Surface-brightness light_frac rows must match kinematic bins: "
            f"got {n_light} light rows and {n_kin} kinematic bins. "
            "Run OSPM/Mapping/observable_mapping.py rebin-sb first."
        )
    for key in ("R_inner_pc", "R_outer_pc"):
        if key not in surface_brightness_profile:
            raise KeyError(f"surface_brightness_profile must include {key}")
    for key in ("R_inner_pc", "R_outer_pc"):
        if key not in kinematic_bins:
            raise KeyError(f"kinematic_bins must include {key}")
    sb_inner = np.asarray(surface_brightness_profile["R_inner_pc"], float)
    sb_outer = np.asarray(surface_brightness_profile["R_outer_pc"], float)
    kb_inner = np.asarray(kinematic_bins["R_inner_pc"], float)
    kb_outer = np.asarray(kinematic_bins["R_outer_pc"], float)
    if not (len(sb_inner) == len(sb_outer) == len(kb_inner) == len(kb_outer) == n_kin):
        raise ValueError("Surface-brightness and kinematic-bin radius arrays must have matching lengths")
    if not np.all(np.isfinite(sb_inner)) or not np.all(np.isfinite(sb_outer)):
        raise ValueError("Surface-brightness radial bin edges contain non-finite values")
    if not np.all(np.isfinite(kb_inner)) or not np.all(np.isfinite(kb_outer)):
        raise ValueError("Kinematic radial bin edges contain non-finite values")
    if np.any(sb_outer <= sb_inner):
        raise ValueError("Surface-brightness bins must have R_outer_pc > R_inner_pc")
    if np.any(kb_outer <= kb_inner):
        raise ValueError("Kinematic bins must have R_outer_pc > R_inner_pc")
    rtol = 1e-7
    atol = 1e-6
    if not np.allclose(sb_inner, kb_inner, rtol=rtol, atol=atol):
        raise ValueError(
            "Surface-brightness R_inner_pc values do not match kinematic bins. "
            "Run OSPM/Mapping/observable_mapping.py rebin-sb first."
        )
    if not np.allclose(sb_outer, kb_outer, rtol=rtol, atol=atol):
        raise ValueError(
            "Surface-brightness R_outer_pc values do not match kinematic bins. "
            "Run OSPM/Mapping/observable_mapping.py rebin-sb first."
        )

    return None

def _stellar_geometry_group(geometry):
    geom = str(geometry).strip().lower()
    spherical_geometries = { "spherical_shell_grid", "spherical_abel_grid", "spherical_enclosed_light_grid", "spherical_force_flattened_grid_metadata" }
    if geom in spherical_geometries:
        return "spherical"
    if geom == "axisymmetric_density_grid":
        return "axisymmetric"
    raise ValueError(f"Unknown karl_light_grid geometry: {geom}")

def _validate_stellar_model_geometry(stellar_model):
    if stellar_model is None:
        return None
    if "type" not in stellar_model:
        raise KeyError("STELLAR_MODEL must include type")
    stype = str(stellar_model["type"]).strip().lower()
    geometry = str(stellar_model.get("geometry", "spherical_shell_grid")).strip().lower()
    if stype == "karl_light_grid":
        group = _stellar_geometry_group(geometry)
        if group == "spherical":
            required = {"grid_csv", "Ltot", "radius_col", "theta_col", "nu_col", "lenc_frac_col"}
        elif group == "axisymmetric":
            required = { "grid_csv", "Ltot", "R_cyl_col", "z_col", "nu_col", "volume_col", "luminosity_col", "q_axis_ratio"}
        else:
            raise ValueError(f"Unknown karl_light_grid geometry group: {group}")
        missing = required - set(stellar_model.keys())
        if missing:
            raise KeyError(f"karl_light_grid STELLAR_MODEL missing keys for geometry={geometry}: {sorted(missing)}")
    return None

def _config_first(config, *keys, default=None):
    if config is None:
        return default
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default

def _apply_motion_model(df, *, v_col, config=None):
    """Return velocities in the model frame.
    Karl-style orbit velocities are internal velocities centered near zero.
    Observed catalog vlos values are usually heliocentric.  For pressure-supported
    systems such as Draco, subtracting V_SYS_KMS is a frame centering operation,
    not a rotating/streaming motion model.
    """
    v_raw = np.asarray(df[v_col].values, float)
    v_model = np.zeros_like(v_raw, dtype=float)
    if config is None:
        return v_raw, v_raw, v_model
    # Preferred simple contract for dwarf pressure-supported systems.
    # Example:
    #     "V_SYS_KMS": -291.68214888089926
    v_sys = _config_first(config, "V_SYS_KMS", "SYSTEMIC_VELOCITY_KMS", "VLOS_SYSTEMIC_KMS", default=None)
    if v_sys is not None:
        v_model = np.full_like(v_raw, float(v_sys), dtype=float)
        return v_raw - v_model, v_raw, v_model
    dyn_mode = config.get("DYNAMICAL_MODE", "pressure_supported")
    if dyn_mode in ("pressure_supported", "spherical_pressure_supported"):
        return v_raw, v_raw, v_model
    if dyn_mode != "non_pressure_supported":
        raise ValueError(f"Unknown DYNAMICAL_MODE: {dyn_mode}")
    motion = config.get("MOTION_MODEL", None)
    if motion is None:
        raise KeyError("DYNAMICAL_MODE='non_pressure_supported' requires MOTION_MODEL")
    if motion.get("mode") == "systemic":
        raw_v_col = motion.get("raw_v_col", v_col)
        v_raw = np.asarray(df[raw_v_col].values, float)
        v_model = np.full_like(v_raw, float(motion["v_sys_kms"]), dtype=float)
        return v_raw - v_model, v_raw, v_model
    raise ValueError(f"Unknown MOTION_MODEL mode: {motion.get('mode')}")

class OSPMObservablesStellar:
    def __init__(self, *, R_star_pc, v_star_kms, verr_star_kms, has_vlos, inclination_deg, Norbit, stellar_model=None, surface_brightness_profile=None, kinematic_bins=None, dynamical_mode=None, motion_model=None, v_star_raw_kms=None, v_motion_model_kms=None):
        self.mode = "karl"
        self.dynamical_mode = dynamical_mode
        self.motion_model = motion_model
        self.stellar_model = stellar_model
        self.surface_brightness_profile = surface_brightness_profile
        self.kinematic_bins = kinematic_bins
        self.kinematic_bin_edges_pc = None if kinematic_bins is None else kinematic_bins["edges_pc"]
        _validate_stellar_model_geometry(self.stellar_model)
        _validate_surface_brightness_against_kinematic_bins(self.surface_brightness_profile, self.kinematic_bins)
        R = np.asarray(R_star_pc, float)
        v = np.asarray(v_star_kms, float)
        ve = np.asarray(verr_star_kms, float)
        hv = np.asarray(has_vlos, bool)
        if not (len(R) == len(v) == len(ve) == len(hv)):
            raise ValueError("Star arrays must have equal length")
        geom = np.isfinite(R) & (R > 0)
        if not np.any(geom):
            raise RuntimeError("No valid stars after geometric filtering")
        self.R_star_pc = R[geom]
        self.v_star_kms = v[geom]
        self.verr_star_kms = ve[geom]
        self.has_vlos = hv[geom]
        self.valid_vlos = self.has_vlos & np.isfinite(self.v_star_kms) & np.isfinite(self.verr_star_kms) & (self.verr_star_kms > 0)
        if not np.any(self.valid_vlos):
            raise RuntimeError("No valid vlos stars after filtering")
        self.R_star_m = self.R_star_pc * pc
        self.v_star_mps = self.v_star_kms * kms
        self.verr_star_mps = self.verr_star_kms * kms
        self.v_star_raw_kms = None if v_star_raw_kms is None else np.asarray(v_star_raw_kms, float)[geom]
        self.v_motion_model_kms = None if v_motion_model_kms is None else np.asarray(v_motion_model_kms, float)[geom]
        good_v = self.has_vlos & np.isfinite(self.v_star_kms)
        self.v_star_kms_median = float(np.nanmedian(self.v_star_kms[good_v])) if np.any(good_v) else np.nan
        self.v_star_kms_mean = float(np.nanmean(self.v_star_kms[good_v])) if np.any(good_v) else np.nan
        self.v_star_kms_std = float(np.nanstd(self.v_star_kms[good_v])) if np.any(good_v) else np.nan
        self.sini, self.cosi, self.edge_on = make_inclination(inclination_deg)
        self.Norbit = int(Norbit)
        self.Nstar = len(self.R_star_m)
        self.Nstar_vlos = int(self.valid_vlos.sum())
        self.Nocc = 0
        self.lambda_occ = 1.0

    @classmethod
    def from_star_table(cls, csv_path, *, r_col="r_pc", v_col="vlos", verr_col="vlos_err", has_vlos_col="has_vlos", inclination_deg, Norbit, stellar_model=None, surface_brightness_path=None, kinematic_bins_path=None, config=None):
        df = pd.read_csv(csv_path)
        needed = [r_col, v_col, verr_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in star table: {missing}")
        if has_vlos_col in df.columns:
            has_vlos = _as_bool_array(df[has_vlos_col].values)
        else:
            has_vlos = np.isfinite(df[v_col].to_numpy(float)) & np.isfinite(df[verr_col].to_numpy(float))
        if surface_brightness_path is None and config is not None:
            surface_brightness_path = config.get("SURFACE_BRIGHTNESS_CSV")
        if surface_brightness_path is None:
            raise KeyError("Karl-style observables require SURFACE_BRIGHTNESS_CSV")
        if kinematic_bins_path is None and config is not None:
            kinematic_bins_path = config.get("KINEMATIC_BINS_CSV")
        if kinematic_bins_path is None:
            raise KeyError("Karl-style observables require KINEMATIC_BINS_CSV")
        surface_brightness_profile = _load_surface_brightness_profile(surface_brightness_path)
        kinematic_bins = _load_kinematic_bins(kinematic_bins_path)
        _validate_surface_brightness_against_kinematic_bins(surface_brightness_profile, kinematic_bins)
        _validate_stellar_model_geometry(stellar_model)
        v_used, v_raw, v_model = _apply_motion_model(df, v_col=v_col, config=config)
        return cls(R_star_pc=df[r_col].values, v_star_kms=v_used, verr_star_kms=df[verr_col].values, has_vlos=has_vlos, inclination_deg=inclination_deg, Norbit=Norbit, stellar_model=stellar_model, surface_brightness_profile=surface_brightness_profile, kinematic_bins=kinematic_bins, dynamical_mode=None if config is None else config.get("DYNAMICAL_MODE"), motion_model=None if config is None else config.get("MOTION_MODEL"), v_star_raw_kms=v_raw, v_motion_model_kms=v_model)