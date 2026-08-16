# OSPM_Daemon.py — STAYS IN PYTHON FOREVER.  Parallelism lives in Julia, not here.
# ========================================================================================================================
# WHAT THIS DOES
# Drives an RL-guided search over dark-matter halo parameters θ.  The first
# halo parameters are config-dependent: (rho_s,r_s), (vcirc,r_s), or (v0,r_c).
# Each iteration proposes a batch of candidate θ, ships them to the Julia orbit-
# superposition engine (OSPM_Physics) for χ² evaluation, records results in a CSV
# ledger, and trains a small surrogate network + policy agent so future proposals
# concentrate near low-χ² regions. A gatekeeper ("Fixer") keeps proposals purely
# random until enough data exists, then hands control to the RL agent. Convergence
# detectors (flatness + posterior tightness) stop the loop once the fit stabilises.
# All heavy physics stays in Julia; this file only orchestrates proposals, bookkeeping,
# and learning.
#
# LAYOUT
# ========================================================================================================================
# clamp(x,lo,hi)          scalar,scalar,scalar → scalar          — bound a number
# random_theta(bounds)     bounds → [float]                      — uniform random point in box
# min_dist(theta,arr)      point,points → float                  — nearest-neighbor distance
# IdentityScaler           X → X                                 — no-op stand-in for StandardScaler
#
# Model(dim)               θ tensor → predicted reward           — surrogate chi² landscape
# Agent(dim)               state tensor → action in [-1,1]^d     — RL policy that proposes moves
#   .act(x,noise)          state,σ → noisy clamped action
#
# Deck(config)             config dict → persistent CSV log      — append-buffered ledger of all evals
#   .add(theta,chi2,…)     point,metrics → row in CSV            — buffer a result row
#   .save()                (side-effect) → CSV on disk           — flush & write
#   .is_forbidden(theta)   point → bool                          — was this point marked forbidden?
#   .nearest_distance(…)   point,tol → float                     — closest existing point
#
# Fixer(cfg)               config → AI gatekeeper               — unlocks AI after enough passes
#   .unlock(deck,runner)   deck,runner → (side-effect)           — flips runner.ai=True when ready
#   .reward(status,chi2)   str,float → float                     — maps eval outcome to scalar reward
#
# FlatDetector(w,eps,p)    window,eps,patience → detector        — fires when chi² stops changing
#   .push(x) / .flat()     float → () / () → bool
#
# ConvergenceDetector(…)   cfg,bounds,cols → detector            — fires when posterior is tight
#   .check(deck,runner,n)  deck,runner,int → bool
#
# Runner(cfg)              config → proposal engine              — RL agent + surrogate + exploration
#   .propose(deck)         deck → [(theta,pid)]                  — next batch of candidate points
#   .train(deck)           deck → (side-effect)                  — one gradient step on surrogate
#   .detect_basin(deck)    deck → bool                           — is the posterior concentrated?
#
# run_daemon(config,engine)  config,engine → None                — outer loop: propose→eval→record→train
# ========================================================================================================================

import os, time, traceback, json
import numpy as np, pandas as pd
import torch, torch.nn as nn
from collections import deque
from OSPM.Physics.OSPM_Physics import (canonicalize_theta_matrix, nfw_vcirc_rs_to_rho_s, normalize_halo_parameterization,)
torch.backends.cudnn.benchmark = False
try: from sklearn.preprocessing import StandardScaler
except Exception: StandardScaler = None

PHASE_VOLUME_DIAG_COLUMNS = ["phase_volume_valid", "phase_volume_convention", "phase_volume_normalization", "phase_volume_launches_recorded", "phase_volume_sos_recorded",
    "phase_volume_valid_base_orbits", "phase_volume_invalid_recorded_orbits", "phase_volume_nested_groups", "phase_volume_duplicate_area_clusters", "phase_volume_duplicate_area_orbits",
    "raw_phase_volume_min", "raw_phase_volume_max", "raw_phase_volume_dynamic_range", "normalized_phase_volume_min", "normalized_phase_volume_max", "wphase_min", "wphase_max",
    "wphase_dynamic_range", "wphase_pair_max_relative_mismatch"]
# ========================================================================================================================
# ========================================================================================================================

def clamp(x, lo, hi): return max(lo, min(hi, x))

def _search_mode(name):
    name = str(name).strip().lower()
    if name in {"r_c", "r_s"}:
        return "log"
    if name == "mbh":
        return "logzero"
    return "linear"

def _to_search_value(name, x, mbh_floor=1.0e3):
    mode = _search_mode(name)
    x = float(x)
    if mode == "log":
        if x <= 0.0:
            raise ValueError(f"{name} must be > 0 for logarithmic search")
        return np.log10(x)
    if mode == "logzero":
        return np.log10(max(x, 0.0) + mbh_floor)
    return x

def _from_search_value(name, z, mbh_floor=1.0e3):
    mode = _search_mode(name)
    z = float(z)
    if mode == "log":
        return 10.0 ** z
    if mode == "logzero":
        return max(0.0, 10.0 ** z - mbh_floor)
    return z

def _search_bounds(bounds, names, mbh_floor=1.0e3):
    return [(_to_search_value(name, lo, mbh_floor),  _to_search_value(name, hi, mbh_floor) ) for name, (lo, hi) in zip(names, bounds)]

def _theta_to_search(theta, names, mbh_floor=1.0e3):
    return np.asarray([ _to_search_value(name, x, mbh_floor) for name, x in zip(names, theta)], dtype=float)

def _theta_from_search(z, names, bounds, mbh_floor=1.0e3):
    theta = []
    for name, zi, (lo, hi) in zip(names, z, bounds):
        x = _from_search_value(name, zi, mbh_floor)
        theta.append(clamp(x, float(lo), float(hi)))
    return theta

def random_theta(bounds, names, mbh_floor=1.0e3, mbh_zero_fraction=0.10):
    sbounds = _search_bounds(bounds, names, mbh_floor)
    theta = []
    for name, (lo, hi), (slo, shi) in zip(names, bounds, sbounds):
        if (str(name).strip().lower() == "mbh" and float(lo) <= 0.0 and np.random.rand() < mbh_zero_fraction):
            theta.append(0.0)
            continue
        z = np.random.uniform(slo, shi)
        x = _from_search_value(name, z, mbh_floor)
        theta.append(clamp(x, float(lo), float(hi)))
    return theta

def min_dist(theta, arr):
    if len(arr) == 0: return np.inf
    return np.linalg.norm(np.asarray(arr) - np.asarray(theta), axis=1).min()


def is_real_pass(status):
    return str(status) == "pass_full"

def family_pass_rows(df, family):
    if "status" not in df.columns:
        return df.iloc[0:0]
    label = f"pass_{str(family).strip().lower()}"
    return df[df["status"].astype(str) == label]

def real_pass_rows(df):
    return family_pass_rows(df, "full")

def finite_family_rows(df, family):
    if "status" not in df.columns or "chi2" not in df.columns:
        return df.iloc[0:0]
    suffix = f"_{str(family).strip().lower()}"
    status = df["status"].astype(str)
    chi2 = pd.to_numeric(df["chi2"], errors="coerce")
    return df[status.str.endswith(suffix) & np.isfinite(chi2) & (chi2 > 1e-12)]

def batch_result_status(code, chi2):
    code = int(code)
    chi2 = float(chi2)
    finite_chi2 = np.isfinite(chi2) and chi2 > 1e-12
    if code == 0:
        return ("pass" if finite_chi2 else "numeric_fail"), (chi2 if finite_chi2 else np.inf)
    status = {1: "orbit_fail", 2: "solver_failed", 3: "physics_exception", 4: "timeout"}.get(code, "unknown_fail")
    return status, (chi2 if finite_chi2 else np.inf)

def _fixed_theta_from_config(config):
    fixed = config.get("FIXED_THETA", None)
    if fixed is None:
        return None
    theta = [float(x) for x in fixed]
    names = list(config["PARAMETER_NAMES"])
    bounds = list(config["THETA_BOUNDS"])
    if len(theta) != len(names):
        raise ValueError( f"FIXED_THETA must contain {len(names)} values for {names}, got {len(theta)}" )
    for name, x, (lo, hi) in zip(names, theta, bounds):
        if not (np.isfinite(x) and float(lo) <= x <= float(hi)):
            raise ValueError( f"FIXED_THETA value {name}={x} is outside [{float(lo)}, {float(hi)}]")
    return theta

def _selected_variants(config, variant_map):
    selected = config.get("EVAL_VARIANTS", None)
    if selected is None:
        return list(variant_map)
    if isinstance(selected, str):
        selected = [selected]
    selected = [str(label).strip().lower() for label in selected]
    if not selected:
        raise ValueError("EVAL_VARIANTS must contain at least one variant label")
    unknown = [label for label in selected if label not in variant_map]
    if unknown:
        raise ValueError( f"Unknown EVAL_VARIANTS={unknown}; choose from {list(variant_map)}" )
    return selected

def _jl_matrix_f64(mat, Main, juliacall=None, name="mat"):
    arr = np.asarray(mat, dtype=np.float64)
    print(f"[JL MATRIX DEBUG] {name}: shape={arr.shape}, dtype={arr.dtype}", flush=True)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        bad = arr[~np.isfinite(arr)]
        raise ValueError(f"{name} has non-finite values: {bad[:10]}")
    nrow, ncol = arr.shape
    Main._tmp_matrix_flat = arr.ravel(order="F").tolist()
    Main._tmp_matrix_nrow = int(nrow)
    Main._tmp_matrix_ncol = int(ncol)
    return Main.seval( "reshape(Float64[y for y in _tmp_matrix_flat], _tmp_matrix_nrow, _tmp_matrix_ncol)")

def _jl_vector_f64(vec, Main, name="vec"):
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if not np.isfinite(arr).all():
        bad = arr[~np.isfinite(arr)]
        raise ValueError(f"{name} has non-finite values: {bad[:10]}")
    Main._tmp_vector_f64 = arr.tolist()
    return Main.seval("Float64[y for y in _tmp_vector_f64]")

def _jl_vector_bool(vec, Main, name="vec"):
    arr = np.asarray(vec, dtype=bool).ravel()
    Main._tmp_vector_bool = arr.tolist()
    return Main.seval("Bool[y for y in _tmp_vector_bool]")

def _julia_literal(value, name):
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
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")

def _jl_primitive_dict(value, Main, name):
    if value is None:
        return Main.seval("nothing")
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict or None")
    entries = []
    for key, item in value.items():
        key_literal = json.dumps(str(key))
        value_literal = _julia_literal(item, name)
        entries.append(f"{key_literal} => {value_literal}")
    expression = "Dict{String,Any}(" + ", ".join(entries) + ")"
    return Main.seval(expression)

def _jl_surface_brightness_profile(profile, Main):
    if profile is None:
        Main._sb_profile_jl = Main.seval("nothing")
        return Main._sb_profile_jl
    Main._sb_R_pc = np.asarray(profile["R_pc"], dtype=np.float64).ravel().tolist()
    Main._sb_R_inner_pc = np.asarray(profile["R_inner_pc"], dtype=np.float64).ravel().tolist()
    Main._sb_R_outer_pc = np.asarray(profile["R_outer_pc"], dtype=np.float64).ravel().tolist()
    Main._sb_light_frac = np.asarray(profile["light_frac"], dtype=np.float64).ravel().tolist()
    Main._sb_Sigma = np.asarray(profile["Sigma"], dtype=np.float64).ravel().tolist()
    Main._sb_Sigma_err = np.asarray(profile["Sigma_err"], dtype=np.float64).ravel().tolist()
    Main._sb_profile_jl = Main.seval("""
    Dict{Symbol,Any}(
    :R_pc => Float64[x for x in _sb_R_pc],
    :R_inner_pc => Float64[x for x in _sb_R_inner_pc],
    :R_outer_pc => Float64[x for x in _sb_R_outer_pc],
    :light_frac => Float64[x for x in _sb_light_frac],
    :Sigma => Float64[x for x in _sb_Sigma],
    :Sigma_err => Float64[x for x in _sb_Sigma_err],
    )""")
    return Main._sb_profile_jl

def _clean_stellar_model(model):
    if model is None:
        return None
    if "type" not in model:
        raise KeyError("STELLAR_MODEL must include 'type'")
    out = {}
    for key, value in model.items():
        k = str(key)
        if isinstance(value, bool):
            out[k] = bool(value)
        elif isinstance(value, int):
            out[k] = int(value)
        elif isinstance(value, float):
            out[k] = float(value)
        elif isinstance(value, str):
            out[k] = value
        else:
            out[k] = value
    out["type"] = str(out["type"]).strip().lower()
    if out["type"] == "plummer":
        for req in ("Ltot", "a_pc"):
            if req not in out:
                raise KeyError(f"Plummer STELLAR_MODEL requires '{req}'")
        out["Ltot"] = float(out["Ltot"])
        out["a_pc"] = float(out["a_pc"])
    elif out["type"] == "karl_light_grid":
        geom = str(out.get("geometry", "spherical_shell_grid")).strip().lower()
        out["geometry"] = geom
        for req in ("grid_csv", "Ltot"):
            if req not in out:
                raise KeyError(f"karl_light_grid STELLAR_MODEL requires '{req}'")
        if geom == "axisymmetric_density_grid":
            # Axisymmetric grids are force-cell products.  They do not require
            # the spherical enclosed-light columns.
            defaults = { "R_cyl_col": "R_cyl_pc", "z_col": "z_pc", "nu_col": "nu_Lsun_pc3", "volume_col": "cell_volume_pc3", "luminosity_col": "cell_luminosity_Lsun" }
            for key, value in defaults.items():
                out.setdefault(key, value)
        else:
            # Current Draco production path: theta rows are allowed as metadata,
            # but the force remains spherical through Lenc_frac(r).
            out.setdefault("geometry", "spherical_force_flattened_grid_metadata")
            for req in ("radius_col", "theta_col", "nu_col", "lenc_frac_col"):
                if req not in out:
                    raise KeyError(f"karl_light_grid STELLAR_MODEL requires '{req}' for spherical geometry")
        out["Ltot"] = float(out["Ltot"])
        for key in ("q_axis_ratio", "force_softening_pc"):
            if key in out and out[key] is not None:
                out[key] = float(out[key])
        for key in ("force_nR", "force_nZ", "force_nphi"):
            if key in out and out[key] is not None:
                out[key] = int(out[key])
    else:
        raise ValueError(f"Unknown STELLAR_MODEL type: {out['type']}")
    return out

def _clean_karl_halo_params(params):
    if params is None:
        return None
    if not isinstance(params, dict):
        raise TypeError("KARL_HALO_PARAMS must be a dict when provided")
    out = {}
    for key, value in params.items():
        k = str(key)
        if isinstance(value, bool):
            out[k] = bool(value)
        elif isinstance(value, int):
            out[k] = int(value)
        elif isinstance(value, float):
            out[k] = float(value)
        elif isinstance(value, str):
            out[k] = value
        elif value is None:
            out[k] = None
        else:
            out[k] = value
    for key in ( "qdm", "dis", "v0", "rc", "rc_pc", "xmgamma", "xmgamma_msun", "rsgamma", "rsgamma_pc", "gamma", "cnfw", "rsnfw", "rsnfw_pc", "gdennorm", "halo_force_softening_pc" ):
        if key in out and out[key] is not None:
            out[key] = float(out[key])
    for key in ("ihalo", "halo_force_nR", "halo_force_nZ", "halo_force_nphi", "halo_force_nm", "halo_force_ntheta"):
        if key in out and out[key] is not None:
            out[key] = int(out[key])
    return out

def _get_surface_brightness_profile(config, physics_engine, obs):
    candidates = [
        getattr(physics_engine, "__surface_brightness_profile__", None),
        getattr(obs, "surface_brightness_profile", None),
        config.get("SURFACE_BRIGHTNESS_PROFILE"),
        config.get("surface_brightness_profile"),]
    obs_cfg = config.get("OBSERVABLES", {})
    if isinstance(obs_cfg, dict):
        candidates.extend([
            obs_cfg.get("SURFACE_BRIGHTNESS_PROFILE"),
            obs_cfg.get("surface_brightness_profile"),])
    for profile in candidates:
        if profile is not None:
            return profile
    raise RuntimeError(
        "surface_brightness_profile is required for Karl-style OSPM; "
        "no star-count fallback is allowed")

def _observable_config(config):
    obs_cfg = config.get("OBSERVABLES", {})
    if obs_cfg is None:
        obs_cfg = {}
    if not isinstance(obs_cfg, dict):
        raise TypeError("config['OBSERVABLES'] must be a dict when provided")
    return obs_cfg
# ========================================================================================================================
# ========================================================================================================================

class IdentityScaler:
    def fit(self, X): return self
    def transform(self, X): return X

class Model(nn.Module):
    def __init__(self, dim, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
    def forward(self, x): return self.net(x)

class Agent(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dim), nn.Tanh())
    def forward(self, x): return self.net(x)
    @torch.no_grad()
    def act(self, x, noise):
        return torch.clamp(self.forward(x) + noise * torch.randn_like(x), -1.0, 1.0)

class Deck:
    def __init__(self, config):
        required_columns = list(config["REQUIRE_COLUMNS"])
        self.config = config
        self.path = config["CSV_PATH"]
        self.cols = list(dict.fromkeys(required_columns + PHASE_VOLUME_DIAG_COLUMNS))
        self.params, self.flush = config["PARAMETER_NAMES"], int(config.get("CSV_FLUSH_INTERVAL", 10))
        self._dirty = 0; self._buf = []; self._pbuf = []; self._sbuf = []
        self._load()
    def _load(self):
        d = os.path.dirname(self.path)
        if d: os.makedirs(d, exist_ok=True)
        if os.path.exists(self.path):
            df = pd.read_csv(self.path)
        else:
            row = {k: np.nan for k in self.cols}
            for i, k in enumerate(self.params): row[k] = self.config["INITIAL_THETA"][i]
            row["status"] = "todo"; df = pd.DataFrame([row]); df.to_csv(self.path, index=False)
        missing = [c for c in self.cols if c not in df.columns]
        if missing: raise KeyError(f"Deck missing required columns: {missing}")
        self.df = df[self.cols].copy()
        print("DECK LOAD DEBUG")
        print("path:", self.path)
        print("params:", self.params)
        print("columns:", self.df.columns.tolist())
        print("head:")
        print(self.df.head().to_string())
        self._params_arr = self.df[self.params].values.astype(float)
        self._status_arr = self.df["status"].values.astype(str)
    def _flush_buf(self):
        if not self._buf: return
        self.df = pd.concat([self.df, pd.DataFrame(self._buf, columns=self.cols)], ignore_index=True)
        self._params_arr = self.df[self.params].values.astype(float)
        self._status_arr = self.df["status"].values.astype(str)
        self._buf.clear(); self._pbuf.clear(); self._sbuf.clear()
    def save(self):
        self._flush_buf(); self.df.to_csv(self.path, index=False)
        print(f"[Deck] saved {len(self.df)} rows → {self.path}", flush=True)
    def _all_params(self):  return np.vstack([self._params_arr, np.array(self._pbuf)]) if self._pbuf else self._params_arr
    def _all_status(self):  return np.concatenate([self._status_arr, np.array(self._sbuf)]) if self._sbuf else self._status_arr
    def is_forbidden(self, theta, ndp=12):
        A, t = np.round(self._all_params(), ndp), np.round(theta, ndp)
        m = (A == t).all(axis=1)
        return (self._all_status()[m] == "forbidden").any() if m.any() else False
    def nearest_distance(self, theta, tol):
        A = self._all_params(); m = np.all(np.abs(A - theta) < tol, axis=1)
        return np.linalg.norm(A[m] - theta, axis=1).min() if m.any() else np.inf
    def add(self, theta, chi2, reward, pid, status, diag=None):
        row_dict = {k: theta[i] for i, k in enumerate(self.params)}
        row_dict |= dict(chi2=chi2, reward=reward, status=status, proposal_id=pid)
        if diag is not None:
            row_dict.update(diag)
        self._buf.append([row_dict.get(k) for k in self.cols])
        self._pbuf.append([theta[i] for i in range(len(self.params))])
        self._sbuf.append(status)
        self._dirty += 1
        if self._dirty >= self.flush:
            self._flush_buf()
            self.save()
            self._dirty = 0


class Fixer:
    def __init__(self, cfg):
        self.model_start = int(cfg.get("AI_MODEL_START_AFTER", 32))
        self.ai_start = int(cfg.get("AI_START_AFTER", 50))
        self.strong_start = int(cfg.get("AI_STRONG_AFTER", 150))
        self.model_started = False
        self.unlocked = False
        self.strong = False

    def unlock(self, deck, runner):
        npass = len(real_pass_rows(deck.df))
        if not self.model_started and npass >= self.model_start:
            runner.enable_model()
            self.model_started = True
            print(f"[AI] surrogate enabled at {npass} full passes", flush=True)
        if not self.unlocked and npass >= self.ai_start:
            if not self.model_started:
                runner.enable_model()
                self.model_started = True
            runner.enable_ai(npass)
            self.unlocked = True
            print(f"[AI] proposal agent enabled at {npass} full passes", flush=True)
        if not self.strong and npass >= self.strong_start:
            runner.enable_strong()
            self.strong = True
            print(f"[AI] strong proposal mode enabled at {npass} full passes", flush=True)

    def reward(self, status, chi2):
        return -1e6 if status != "pass" else -float(chi2)

class FlatDetector:
    def __init__(self, w, eps, p):
        self.w, self.eps, self.p = w, eps, p; self.buf = deque(maxlen=w); self.cnt = 0
    def push(self, x):
        if not np.isfinite(x): return
        self.buf.append(x)
        if len(self.buf) < self.w: self.cnt = 0; return
        self.cnt = self.cnt + 1 if np.std(self.buf) < self.eps and np.isfinite(x) else 0
    def flat(self): return self.cnt >= self.p


class ConvergenceDetector:
    def __init__(self, cfg, bounds, cols):
        self.rel_thr = float(cfg.get("CONVERGE_REL_SPREAD", 0.05))
        self.chi_thr = float(cfg.get("CONVERGE_CHI_STD", 0.5))
        self.n_top = int(cfg.get("CONVERGE_N_TOP", 200))
        self.n_min = int(cfg.get("CONVERGE_MIN_PASS", 500))
        self.patience = int(cfg.get("CONVERGE_PATIENCE", 3))
        self.every = int(cfg.get("CONVERGE_CHECK_EVERY", 500))
        self.bounds, self.cols, self.cnt = bounds, cols, 0

    def check(self, deck, runner, runs):
        if not runner.fill_mode:
            return False
        if runs % self.every != 0:
            return False
        good = real_pass_rows(deck.df)
        if len(good) < self.n_min:
            self.cnt = 0
            return False
        top = good.nsmallest(min(len(good), self.n_top), "chi2")
        X_unit = runner._unit_search_matrix(top[self.cols].values)
        if len(X_unit) == 0:
            self.cnt = 0
            return False
        chi_std = float(top["chi2"].std())
        rel_spread = float(np.mean(np.std(X_unit, axis=0)))
        self.cnt = self.cnt + 1 if chi_std < self.chi_thr and rel_spread < self.rel_thr else 0
        converged = self.cnt >= self.patience
        if converged:
            print(f"[Converge] posterior converged at run {runs}: rel_spread={rel_spread:.4f} chi_std={chi_std:.4f}", flush=True)
        return converged

# ========================================================================================================================
# 
# ========================================================================================================================


class Runner:
    def __init__(self, cfg):
        self.cfg, self.bounds, self.cols = cfg, cfg["THETA_BOUNDS"], cfg["PARAMETER_NAMES"]
        self.dim, self.batch = len(self.cols), int(cfg["BATCH_SIZE"])
        self.min_d = float(cfg.get("SEARCH_MIN_DISTANCE", cfg["MIN_DISTANCE"]))
        self.fill_min_d = float(cfg.get("FILL_MIN_DISTANCE", max(0.25 * self.min_d, 1.0e-6)))
        self.model_ready, self.ai, self.strong_ai = False, False, False
        self.model = self.agent = self.opt_m = self.opt_a = None
        self.no_bh_model = self.no_halo_model = None
        self.opt_no_bh = self.opt_no_halo = None
        self.noise0 = float(cfg.get("AI_NOISE_INIT", 0.30))
        self.noise1 = float(cfg.get("AI_NOISE_MIN", 0.02))
        self.tau = float(cfg.get("AI_NOISE_TAU", 300))
        self.ai_start_pass = 0
        self.good_full_count = 0
        self.step = 0
        self.recent = deque(maxlen=5000)
        self.fill_mode = self.fill_triggered = False
        self.explore_frac = float(cfg.get("EXPLORE_FRACTION", 0.10))
        self.early_explore_frac = float(cfg.get("AI_EARLY_EXPLORE_FRACTION", max(0.35, self.explore_frac)))
        self.train_window = int(cfg.get("TRAIN_WINDOW", 2000))
        self.train_recent_fraction = float(cfg.get("TRAIN_RECENT_FRACTION", 0.25))
        self.train_best_fraction = float(cfg.get("TRAIN_BEST_FRACTION", 0.50))
        self.train_global_fraction = float(cfg.get("TRAIN_GLOBAL_FRACTION", 0.25))
        self.min_train_points = int(cfg.get("MIN_TRAIN_POINTS", cfg.get("AI_MODEL_START_AFTER", 32)))
        self.alt_model_min_pairs = int(cfg.get("ALT_MODEL_MIN_PAIRS", 8))
        self.mbh_floor = float(cfg.get("MBH_LOG_FLOOR", 1.0e3))
        self.sbounds = _search_bounds(self.bounds, self.cols, self.mbh_floor)
        self.search_lo = np.asarray([lo for lo, hi in self.sbounds], dtype=float)
        self.search_hi = np.asarray([hi for lo, hi in self.sbounds], dtype=float)
        self.search_span = np.maximum(self.search_hi - self.search_lo, 1.0e-12)

    def enable_model(self):
        if self.model_ready:
            return
        self.model = Model(self.dim)
        self.no_bh_model = Model(self.dim)
        self.no_halo_model = Model(self.dim)
        self.opt_m = torch.optim.Adam(self.model.parameters(), 1e-3)
        self.opt_no_bh = torch.optim.Adam(self.no_bh_model.parameters(), 1e-3)
        self.opt_no_halo = torch.optim.Adam(self.no_halo_model.parameters(), 1e-3)
        self.model_ready = True

    def enable_ai(self, full_pass_count=0):
        if not self.model_ready:
            self.enable_model()
        if self.ai:
            return
        self.agent = Agent(self.dim)
        self.opt_a = torch.optim.Adam(self.agent.parameters(), 1e-3)
        self.ai_start_pass = int(full_pass_count)
        self.good_full_count = max(self.good_full_count, int(full_pass_count))
        self.ai = True

    def enable_strong(self):
        self.strong_ai = True

    def _noise(self):
        if not self.ai:
            return self.noise0
        age = max(0, self.good_full_count - self.ai_start_pass)
        return max(self.noise1, self.noise0 * np.exp(-age / max(self.tau, 1.0)))

    def _unit_search(self, theta):
        z = _theta_to_search(theta, self.cols, self.mbh_floor)
        return np.clip((z - self.search_lo) / self.search_span, 0.0, 1.0)

    def _theta_from_unit(self, u):
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        z = self.search_lo + u * self.search_span
        return _theta_from_search(z, self.cols, self.bounds, self.mbh_floor)

    def _unit_search_matrix(self, theta_matrix):
        A = np.asarray(theta_matrix, dtype=float)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if A.size == 0:
            return np.empty((0, self.dim), dtype=float)
        finite = np.all(np.isfinite(A), axis=1)
        if not np.any(finite):
            return np.empty((0, self.dim), dtype=float)
        return np.asarray([self._unit_search(row) for row in A[finite]], dtype=float)

    def _base(self, deck):
        good = real_pass_rows(deck.df)
        if len(good) >= 10:
            if np.random.rand() < 0.15:
                return good[self.cols].sample(1).values[0]
            return good.nsmallest(min(len(good), 500), "chi2")[self.cols].sample(1).values[0]
        candidates = deck.df[self.cols].dropna()
        if len(candidates):
            return candidates.sample(1).values[0]
        return random_theta(self.bounds, self.cols, mbh_floor=self.mbh_floor, mbh_zero_fraction=float(self.cfg.get("MBH_ZERO_FRACTION", 0.10)))

    def detect_basin(self, deck):
        good = real_pass_rows(deck.df)
        min_pass = int(self.cfg.get("BASIN_MIN_PASS", self.cfg.get("AI_STRONG_AFTER", 150)))
        if len(good) < min_pass:
            return False
        top = good.nsmallest(min(len(good), int(self.cfg.get("BASIN_TOP_N", 100))), "chi2")
        X_unit = self._unit_search_matrix(top[self.cols].values)
        if len(X_unit) == 0:
            return False
        chi_std = float(top["chi2"].std())
        rel_spread = float(np.mean(np.std(X_unit, axis=0)))
        return chi_std < float(self.cfg.get("BASIN_CHI_STD", 1.0)) and rel_spread < float(self.cfg.get("BASIN_REL_SPREAD", 0.15))

    def step_scale(self, deck):
        if not self.ai:
            return 0.20
        if not self.fill_mode:
            return float(self.cfg.get("AI_GLOBAL_STEP_SCALE", 0.20))
        good = real_pass_rows(deck.df)
        if len(good) == 0:
            return 0.05
        top = good.nsmallest(min(len(good), 200), "chi2")
        X_unit = self._unit_search_matrix(top[self.cols].values)
        if len(X_unit) == 0:
            return 0.05
        rel_spread = float(np.mean(np.std(X_unit, axis=0)))
        return clamp(0.01 + 0.20 * rel_spread, 0.01, 0.05)

    def propose(self, deck, n=None):
        target = self.batch if n is None else max(1, min(int(n), self.batch))
        out = []
        mbh_zero_fraction = float(self.cfg.get("MBH_ZERO_FRACTION", 0.10))
        distance_floor = self.fill_min_d if self.fill_mode else self.min_d
        max_attempts = int(self.cfg.get("PROPOSAL_MAX_ATTEMPTS", max(2000, 500 * target)))
        status = deck.df["status"].astype(str) if "status" in deck.df.columns else pd.Series([], dtype=str)
        full_rows = deck.df[status.str.endswith("_full")] if len(status) else deck.df.iloc[0:0]
        deck_unit = self._unit_search_matrix(full_rows[self.cols].values) if len(full_rows) else np.empty((0, self.dim), dtype=float)
        attempts = 0

        while len(out) < target:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Unable to generate {target} distinct proposals after {max_attempts} attempts; accepted={len(out)} distance_floor={distance_floor:g}")

            explore_fraction = self.explore_frac if self.strong_ai else self.early_explore_frac
            use_ai = self.ai and not (explore_fraction > 0.0 and np.random.rand() < explore_fraction)

            if use_ai:
                if self.fill_mode:
                    good = real_pass_rows(deck.df)
                    base = good.nsmallest(min(len(good), 100), "chi2")[self.cols].sample(1).values[0]
                else:
                    base = self._base(deck)
                base_unit = self._unit_search(base)
                action = self.agent.act(torch.tensor(base_unit.reshape(1, -1), dtype=torch.float32), self._noise()).numpy().squeeze()
                scale = self.step_scale(deck)
                theta = self._theta_from_unit(base_unit + scale * action)
            else:
                theta = random_theta(self.bounds, self.cols, mbh_floor=self.mbh_floor, mbh_zero_fraction=mbh_zero_fraction)

            if deck.is_forbidden(theta):
                continue

            u = self._unit_search(theta)
            if self.recent:
                recent_unit = np.asarray(self.recent, dtype=float)
                if np.linalg.norm(recent_unit - u, axis=1).min() < distance_floor:
                    continue
            if len(deck_unit) and np.linalg.norm(deck_unit - u, axis=1).min() < distance_floor:
                continue

            self.recent.append(u)
            self.step += 1
            out.append((theta, self.step))

        return out

    def _training_sample(self, df):
        if len(df) <= self.train_window:
            return df.copy()
        total = self.train_window
        fractions = np.asarray([self.train_recent_fraction, self.train_best_fraction, self.train_global_fraction], dtype=float)
        if not np.isfinite(fractions).all() or fractions.sum() <= 0.0:
            fractions = np.asarray([0.25, 0.50, 0.25], dtype=float)
        fractions /= fractions.sum()
        n_recent = max(1, int(round(total * fractions[0])))
        n_best = max(1, int(round(total * fractions[1])))
        n_global = max(1, total - n_recent - n_best)
        recent = df.tail(min(n_recent, len(df)))
        best = df.nsmallest(min(n_best, len(df)), "chi2")
        global_sample = df.sample(min(n_global, len(df)))
        sample = pd.concat([recent, best, global_sample]).loc[lambda x: ~x.index.duplicated(keep="last")]
        if len(sample) < total:
            remaining = df.loc[~df.index.isin(sample.index)]
            if len(remaining):
                sample = pd.concat([sample, remaining.sample(min(total - len(sample), len(remaining)))])
        return sample

    def _train_surrogate(self, model, optimizer, X, y):
        if len(X) == 0:
            return
        Xt = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
        pred = model(Xt)
        loss = ((pred - yt) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _matched_delta_training(self, deck, family):
        full = family_pass_rows(deck.df, "full")
        alt = family_pass_rows(deck.df, family)
        if len(full) == 0 or len(alt) == 0:
            return np.empty((0, self.dim), dtype=float), np.empty(0, dtype=float)
        full = full.copy()
        alt = alt.copy()
        full["proposal_id"] = pd.to_numeric(full["proposal_id"], errors="coerce")
        alt["proposal_id"] = pd.to_numeric(alt["proposal_id"], errors="coerce")
        full["chi2"] = pd.to_numeric(full["chi2"], errors="coerce")
        alt["chi2"] = pd.to_numeric(alt["chi2"], errors="coerce")
        full = full[np.isfinite(full["proposal_id"]) & np.isfinite(full["chi2"])].drop_duplicates("proposal_id", keep="last")
        alt = alt[np.isfinite(alt["proposal_id"]) & np.isfinite(alt["chi2"])].drop_duplicates("proposal_id", keep="last")
        merged = full[["proposal_id", "chi2"] + self.cols].merge(alt[["proposal_id", "chi2"]], on="proposal_id", suffixes=("_full", "_alt"))
        if len(merged) == 0:
            return np.empty((0, self.dim), dtype=float), np.empty(0, dtype=float)
        X = self._unit_search_matrix(merged[self.cols].values)
        y = merged["chi2_alt"].to_numpy(dtype=float) - merged["chi2_full"].to_numpy(dtype=float)
        finite = np.isfinite(y)
        return X[finite], y[finite]

    def predict_alt_delta(self, theta, family):
        model = self.no_bh_model if family == "no_bh" else self.no_halo_model if family == "no_halo" else None
        if model is None:
            return np.nan
        X, y = self._matched_delta_training_cache.get(family, (None, None)) if hasattr(self, "_matched_delta_training_cache") else (None, None)
        if X is None or len(X) < self.alt_model_min_pairs:
            return np.nan
        with torch.no_grad():
            value = model(torch.tensor(self._unit_search(theta).reshape(1, -1), dtype=torch.float32)).item()
        return float(value)

    def train(self, deck):
        if not self.model_ready:
            return

        df = real_pass_rows(deck.df)
        df = df[np.isfinite(pd.to_numeric(df["reward"], errors="coerce")) & np.isfinite(pd.to_numeric(df["chi2"], errors="coerce"))]
        self.good_full_count = len(df)
        if len(df) < self.min_train_points:
            return

        train_df = self._training_sample(df)
        X = self._unit_search_matrix(train_df[self.cols].values)
        y = train_df["reward"].to_numpy(dtype=float)
        self._train_surrogate(self.model, self.opt_m, X, y)

        self._matched_delta_training_cache = {}
        for family, model, optimizer in (("no_bh", self.no_bh_model, self.opt_no_bh), ("no_halo", self.no_halo_model, self.opt_no_halo)):
            X_alt, y_alt = self._matched_delta_training(deck, family)
            self._matched_delta_training_cache[family] = (X_alt, y_alt)
            if len(X_alt) >= self.alt_model_min_pairs:
                self._train_surrogate(model, optimizer, X_alt, y_alt)

        if not self.ai:
            return

        Xt = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        action = self.agent(Xt)
        action_scale = float(self.step_scale(deck))
        candidate_unit = torch.clamp(Xt + action_scale * action, 0.0, 1.0)

        for p in self.model.parameters():
            p.requires_grad_(False)

        predicted_reward = self.model(candidate_unit)
        agent_loss = -predicted_reward.mean() + 1.0e-4 * action.pow(2).mean()
        self.opt_a.zero_grad()
        agent_loss.backward()
        self.opt_a.step()

        for p in self.model.parameters():
            p.requires_grad_(True)

# ========================================================================================================================
#
# ========================================================================================================================

def run_daemon(config, physics_engine):
    from collections import defaultdict
    deck, runner, fixer = Deck(config), Runner(config), Fixer(config)
    halo_parameterization = normalize_halo_parameterization(config.get("HALO_PARAMETERIZATION", "rho_rs"))
    fixed_theta = _fixed_theta_from_config(config)
    if len(config["PARAMETER_NAMES"]) < 4: raise ValueError("PARAMETER_NAMES must cover four theta entries")
    expected_halo_names = {"rho_rs": ("rho_s", "r_s"), "vcirc_rs": ("vcirc", "r_s"), "v0_rc": ("v0", "r_c")}[halo_parameterization]
    actual_halo_names = tuple(config["PARAMETER_NAMES"][:2])
    if actual_halo_names != expected_halo_names: raise ValueError(f"HALO_PARAMETERIZATION={halo_parameterization!r} expects PARAMETER_NAMES to begin with {list(expected_halo_names)!r}, got {list(actual_halo_names)!r}")
    if len(config["THETA_BOUNDS"]) < len(config["PARAMETER_NAMES"]): raise ValueError("THETA_BOUNDS must cover PARAMETER_NAMES")
    print("CONFIG HALO_PARAMETERIZATION:", halo_parameterization)
    print("CONFIG PARAMETER_NAMES:", config["PARAMETER_NAMES"])
    print("CONFIG THETA_BOUNDS:", config["THETA_BOUNDS"])
    if fixed_theta is not None:
        print("CONFIG FIXED_THETA:", fixed_theta)
    if halo_parameterization == "vcirc_rs":
        vcirc0, rs0 = float(config["INITIAL_THETA"][0]), float(config["INITIAL_THETA"][1])
        rho0 = nfw_vcirc_rs_to_rho_s(vcirc0, rs0)
        print(f"[HALO CONVERSION] vcirc={vcirc0:g} km/s, r_s={rs0:g} pc -> rho_s={rho0:.8g} Msun/pc^3")
    print("runner.bounds:", runner.bounds)
    flat = FlatDetector(config.get("FLAT_WINDOW", 200), config.get("FLAT_THRESHOLD", 1e-6), config.get("FLAT_PATIENCE", 3))
    converge = ConvergenceDetector(config, config["THETA_BOUNDS"], config["PARAMETER_NAMES"])
    proposal_count, eval_count, full_count = 0, 0, 0
    existing_full = real_pass_rows(deck.df)
    best = float(existing_full["chi2"].min()) if len(existing_full) else np.inf
    existing_pid = pd.to_numeric(deck.df.get("proposal_id", pd.Series(dtype=float)), errors="coerce")
    if np.isfinite(existing_pid).any():
        runner.step = int(np.nanmax(existing_pid))
    t_acc, t_cnt = defaultdict(float), defaultdict(int); PROF_EVERY = int(config.get("PROF_EVERY", 25))
    obs = getattr(physics_engine, "__wrapped_obs__", None)
    if obs is None:
        raise RuntimeError("The Karl daemon requires a wrapped spherical observable engine")

    base_halo_type = str(getattr(physics_engine, "__halo_type__", config.get("HALO_TYPE", "nfw"))).strip().lower()
    from juliacall import Main; import juliacall
    surface_brightness_profile = _get_surface_brightness_profile(config, physics_engine, obs)
    obs_cfg = _observable_config(config)
    engine_cfg = getattr(physics_engine, "__karl_config__", {}) or {}
    if not isinstance(engine_cfg, dict):
        raise TypeError("physics_engine.__karl_config__ must be a dict when provided")

    def opt(*names, default=None):
        for name in names:
            if name in obs_cfg and obs_cfg[name] is not None: return obs_cfg[name]
            if name in engine_cfg and engine_cfg[name] is not None: return engine_cfg[name]
            if name in config and config[name] is not None:
                return config[name]
        for name in names:
            if hasattr(obs, name):
                value = getattr(obs, name)
                if value is not None:
                    return value
        return default

    stellar_model = _clean_stellar_model(opt("STELLAR_MODEL", "stellar_model", default=getattr(obs, "stellar_model", None)))
    if stellar_model is None:
        raise RuntimeError("stellar_model is required for the Karl daemon path; M/L cannot enter the force without it")
    nvbin = int(opt("NVBIN", "Nvbin", "nvbin", default=21))
    ntheta_launch = int(opt("NTHETA_LAUNCH", "Ntheta_launch", "ntheta_launch", default=9))
    velocity_edges = opt("VELOCITY_EDGES", "velocity_edges", default=None)
    tracer_constraint_mode = str(opt("TRACER_CONSTRAINT_MODE", "tracer_constraint_mode", default="projected_light")).strip().lower()
    if tracer_constraint_mode not in ("projected_light", "density_3d"):
        raise ValueError("TRACER_CONSTRAINT_MODE must be 'projected_light' or 'density_3d'")
    alphat = float(opt("KARL_ALPHAT", "alphat", default=config.get("ALPHAT", 1.0)))
    apfac = float(opt("KARL_APFAC", "apfac", default=0.01))
    light_rel_tol = float(opt("KARL_LIGHT_REL_TOL", "light_rel_tol", default=0.01))
    light_sigma_tol = float(opt("KARL_LIGHT_SIGMA_TOL", "light_sigma_tol", default=2.0))
    delta_chi2_iter_tol = float(opt("KARL_DELTA_CHI2_ITER_TOL", "delta_chi2_iter_tol", default=0.3))
    maxiter = int(opt("KARL_MAXITER", "maxiter", default=config.get("MAXITER", 60)))
    entropy_floor = float(opt("ENTROPY_FLOOR", "entropy_floor", default=config.get("ENTROPY_FLOOR", 1e-30)))
    halo_q_axis_ratio = float(opt("HALO_Q_AXIS_RATIO", "halo_q_axis_ratio", default=config.get("HALO_Q_AXIS_RATIO", 1.0)))
    karl_halo_params = _clean_karl_halo_params(opt("KARL_HALO_PARAMS", "karl_halo_params", default=config.get("KARL_HALO_PARAMS", None)))
    orbit_fill_pct = float(opt("ORBIT_FILL_PCT", default=0.85))
    orbit_regional_floor = float(opt("ORBIT_REGIONAL_FLOOR", default=0.80))
    orbit_max_regional_gap = float(opt("ORBIT_MAX_REGIONAL_GAP", default=0.10))
    orbit_shell_bands = int(opt("ORBIT_SHELL_BANDS", default=8))
    orbit_coverage_check_every = int(opt("ORBIT_COVERAGE_CHECK_EVERY", default=50))
    orbit_warn_fill_pct = float(opt("ORBIT_WARN_FILL_PCT", default=0.95))
    orbit_warn_success_pct = float(opt("ORBIT_WARN_SUCCESS_PCT", default=0.99))
    orbit_warn_regional_floor = float(opt("ORBIT_WARN_REGIONAL_FLOOR", default=0.80))
    orbit_warn_max_regional_gap = float(opt("ORBIT_WARN_MAX_REGIONAL_GAP", default=0.15))
    model_owner_limit = int(opt("MODEL_OWNER_LIMIT", default=0))
    threads_per_model = int(opt("THREADS_PER_MODEL", "threads_per_model", default=8))

    
    if base_halo_type == "karl_halo":
        raise RuntimeError(
            "HALO_TYPE='karl_halo' is disabled: its density functions use parsec-valued "
            "radii, while the current halo-table path supplies radii in meters"
        )
    if abs(halo_q_axis_ratio - 1.0) > 1e-8:
        raise RuntimeError(
            "Flattened halo forces are disabled because their axisymmetric force table "
            "does not share the potential used for orbit launch energy; set "
            "HALO_Q_AXIS_RATIO=1.0"
        )
    R_star_m = getattr(physics_engine, "__R_star_m__", getattr(obs, "R_star_m", None))
    valid_vlos = getattr(physics_engine, "__valid_vlos__", getattr(obs, "valid_vlos", None))
    v_star_mps = getattr(physics_engine, "__v_star_mps__", getattr(obs, "v_star_mps", None))
    verr_star_mps = getattr(physics_engine, "__verr_star_mps__", getattr(obs, "verr_star_mps", None))
    if R_star_m is None or valid_vlos is None or v_star_mps is None or verr_star_mps is None:
        raise RuntimeError("wrapped physics engine must expose R_star_m, valid_vlos, v_star_mps, and verr_star_mps")
    R_star_m = np.asarray(R_star_m, float).ravel()
    valid_vlos = np.asarray(valid_vlos, bool).ravel()
    v_star_mps = np.asarray(v_star_mps, float).ravel()
    verr_star_mps = np.asarray(verr_star_mps, float).ravel()
    if not (R_star_m.size == valid_vlos.size == v_star_mps.size == verr_star_mps.size):
        raise RuntimeError("wrapped physics arrays must match lengths: " f"R={R_star_m.size}, valid={valid_vlos.size}, v={v_star_mps.size}, verr={verr_star_mps.size}")
    kinematic_bin_edges = getattr(physics_engine, "__kinematic_bin_edges_pc__", None)
    if kinematic_bin_edges is None:
        kinematic_bin_edges = engine_cfg.get("kinematic_bin_edges_pc", None)
    if kinematic_bin_edges is None:
        kinematic_bin_edges = getattr(obs, "kinematic_bin_edges_pc", None)
    if kinematic_bin_edges is None:
        raise RuntimeError("kinematic_bin_edges_pc is required; no adaptive radial-bin fallback is allowed")
    light_bin_edges = getattr(physics_engine, "__light_bin_edges_pc__", None)
    if light_bin_edges is None:
        light_bin_edges = engine_cfg.get("light_bin_edges_pc", None)
    if light_bin_edges is None:
        light_bin_edges = getattr(obs, "light_bin_edges_pc", None)
    if light_bin_edges is None:
        raise RuntimeError("light_bin_edges_pc is required for Karl-style light constraints")
    kinematic_bin_edges = np.asarray(kinematic_bin_edges, float).ravel() * 3.0856775814913673e16
    light_bin_edges = np.asarray(light_bin_edges, float).ravel() * 3.0856775814913673e16
    n_light = max(0, len(light_bin_edges) - 1)
    n_kin = max(0, len(kinematic_bin_edges) - 1)
    r_light_max_pc = float(light_bin_edges[-1] / 3.0856775814913673e16) if len(light_bin_edges) else float("nan")
    r_kin_max_pc = float(kinematic_bin_edges[-1] / 3.0856775814913673e16) if len(kinematic_bin_edges) else float("nan")
    Main._stellar_model_jl = _jl_primitive_dict(stellar_model, Main, "stellar_model")
    Main._karl_halo_params_jl = _jl_primitive_dict(karl_halo_params, Main, "karl_halo_params")
    _jl_surface_brightness_profile(surface_brightness_profile, Main)
    if velocity_edges is None:
        Main._velocity_edges_jl = Main.seval("nothing")
    else:
        Main._velocity_edges_jl = _jl_vector_f64(velocity_edges, Main, name="velocity_edges")
    Main.seval("""
        _stellar_model_jl isa AbstractDict ||
            error("stellar_model handoff must be a Julia dictionary")
        (_karl_halo_params_jl === nothing || _karl_halo_params_jl isa AbstractDict) ||
            error("karl_halo_params handoff must be nothing or a Julia dictionary")
        (_velocity_edges_jl === nothing || _velocity_edges_jl isa AbstractVector{<:Real}) ||
            error("velocity_edges handoff must be nothing or a real Julia vector")
        println(
            "[Daemon] Julia handoff — stellar_model=", typeof(_stellar_model_jl),
            ", karl_halo_params=", typeof(_karl_halo_params_jl),
            ", velocity_edges=", typeof(_velocity_edges_jl),
        )
    """)

    sini = float(obs.sini)
    Norbit = int(obs.Norbit)
    if Norbit % 2 != 0:
        raise RuntimeError(f"Karl paired-orbit Spherical path requires even Norbit because Norbit is the final column count; got Norbit={Norbit}")
    nstar_vlos = int(np.count_nonzero(valid_vlos))
    print(
        f"[Daemon] Karl batch mode ON — Norbit={Norbit}, Nbase_orbit={Norbit // 2}, Nstar_vlos={nstar_vlos}, "
        f"Nvbin={nvbin}, Ntheta_launch={ntheta_launch}, tracer_constraint_mode={tracer_constraint_mode}, alphat={alphat}, apfac={apfac}, "
        f"light_rel_tol={light_rel_tol}, light_sigma_tol={light_sigma_tol}, delta_chi2_iter_tol={delta_chi2_iter_tol}, "
        f"halo_q={halo_q_axis_ratio}, karl_halo_params_active={karl_halo_params is not None}",
        flush=True,
    )
    print(
        f"[Daemon] Karl bin contract — N_light={n_light}, N_kin={n_kin}, "
        f"R_light_max_pc={r_light_max_pc:.6g}, R_kin_max_pc={r_kin_max_pc:.6g}, "
        f"N_constraints={n_light + n_kin * nvbin}",
        flush=True,
    )
    print(
        f"[Daemon] Orbit coverage — strict={orbit_fill_pct:.3f}/{orbit_regional_floor:.3f}/{orbit_max_regional_gap:.3f}, "
        f"warning={orbit_warn_fill_pct:.3f}/{orbit_warn_success_pct:.3f}/"
        f"{orbit_warn_regional_floor:.3f}/{orbit_warn_max_regional_gap:.3f}, "
        f"shell_bands={orbit_shell_bands}, check_every={orbit_coverage_check_every}, "
        f"model_owner_limit={model_owner_limit or 'auto'}",
        flush=True,
    )

    _jnt = os.environ.get("JULIA_NUM_THREADS", "1")
    _nthreads = (os.cpu_count() or 1) if _jnt == "auto" else int(_jnt)
    owner_capacity = model_owner_limit if model_owner_limit > 0 else max(1, _nthreads // max(threads_per_model, 1))
    feedback_cfg = int(config.get("FEEDBACK_BATCH_SIZE", 0))
    feedback_batch = owner_capacity if feedback_cfg <= 0 else min(feedback_cfg, runner.batch)
    feedback_batch = max(1, min(feedback_batch, runner.batch))
    CHUNK = max(1, int(config.get("CHUNK_SIZE", feedback_batch)))
    max_evals = int(config.get("MAX_EVALS", 0))
    alt_trigger_delta = float(config.get("ALT_TRIGGER_DELTA_CHI2", 100.0))
    alt_density_radius = float(config.get("ALT_DENSITY_RADIUS", 0.05))
    alt_sensitivity_delta = float(config.get("ALT_SENSITIVITY_DELTA_CHI2", 100.0))

    print(f"[SEARCH CADENCE] configured_batch={runner.batch} feedback_batch={feedback_batch} owner_capacity={owner_capacity} chunk={CHUNK}", flush=True)
    print(f"[SCIENCE BRANCH] full_delta<={alt_trigger_delta:g} density_radius={alt_density_radius:g} sensitivity_delta<={alt_sensitivity_delta:g}", flush=True)

    def _variant(theta, label):
        halo_param, halo_scale, MBH, ML = [float(x) for x in theta]
        variants = {
            "full": ([halo_param, halo_scale, MBH, ML], base_halo_type),
            "no_bh": ([halo_param, halo_scale, 0.0, ML], base_halo_type),
            "no_halo": ([0.0, halo_scale, MBH, ML], "none"),
            "no_bh_halo_up": ([halo_param * 2.0, halo_scale, 0.0, ML], base_halo_type),
            "no_bh_halo_down": ([halo_param * 0.5, halo_scale, 0.0, ML], base_halo_type),
            "no_bh_halo_scale_up": ([halo_param, halo_scale * 2.0, 0.0, ML], base_halo_type),
            "no_bh_halo_scale_down": ([halo_param, halo_scale * 0.5, 0.0, ML], base_halo_type),
            "no_bh_ml_up": ([halo_param, halo_scale, 0.0, ML * 2.0], base_halo_type),
            "no_bh_ml_down": ([halo_param, halo_scale, 0.0, ML * 0.5], base_halo_type),
            "no_halo_bh_up": ([0.0, halo_scale, MBH * 2.0, ML], "none"),
            "no_halo_bh_down": ([0.0, halo_scale, MBH * 0.5, ML], "none"),
            "no_halo_ml_up": ([0.0, halo_scale, MBH, ML * 2.0], "none"),
            "no_halo_ml_down": ([0.0, halo_scale, MBH, ML * 0.5], "none"),
        }
        aliases = {"halo_only": "no_bh", "bh_only": "no_halo"}
        label = aliases.get(str(label).strip().lower(), str(label).strip().lower())
        if label not in variants:
            raise ValueError(f"Unknown evaluation variant {label!r}; choose from {list(variants)}")
        return label, *variants[label]

    def _bounded_prop(theta, pid, label):
        label, tvar, halo_type_variant = _variant(theta, label)
        tfix = []
        for k, x in enumerate(tvar):
            lo, hi = config["THETA_BOUNDS"][k]
            tfix.append(clamp(float(x), float(lo), float(hi)))
        return tfix, pid, label, halo_type_variant

    def _dedupe_props(props):
        out, seen = [], set()
        for theta, pid, label, halo_type_variant in props:
            key = (str(halo_type_variant).strip().lower(), tuple(round(float(x), 12) for x in theta))
            if key in seen:
                continue
            seen.add(key)
            out.append((theta, pid, label, halo_type_variant))
        return out

    def _family_dims(family):
        if family == "no_bh":
            return (0, 1, 3)
        if family == "no_halo":
            return (2, 3)
        raise ValueError(f"Unknown restricted family {family!r}")

    def _family_region_sparse(theta, family):
        rows = family_pass_rows(deck.df, family)
        if len(rows) == 0:
            return True
        dims = _family_dims(family)
        target = runner._unit_search(theta)[list(dims)]
        X = runner._unit_search_matrix(rows[runner.cols].values)
        if len(X) == 0:
            return True
        distance = np.linalg.norm(X[:, list(dims)] - target, axis=1)
        return not np.any(distance <= alt_density_radius)

    def _record(theta, pid, label, status, chi2, diag=None):
        nonlocal best, eval_count, full_count
        finite_chi2 = np.isfinite(chi2) and chi2 > 1e-12
        if status == "pass" and not finite_chi2:
            status = "numeric_fail"
        if not finite_chi2:
            chi2 = np.inf
        if diag is not None:
            diag = dict(diag)
            diag["chi2_losvd"] = chi2

        reward = fixer.reward(status, chi2)
        final_status = f"{status}_{label}"
        t_add = time.perf_counter()
        deck.add(theta, chi2, reward, pid, final_status, diag=diag)
        t_acc["add"] += time.perf_counter() - t_add
        t_cnt["add"] += 1

        eval_count += 1
        if label == "full":
            full_count += 1

        valid_real_pass = final_status == "pass_full" and finite_chi2
        valid_family_pass = status == "pass" and finite_chi2
        new_best = False

        if label == "full":
            if valid_real_pass:
                flat.push(chi2)
                if chi2 < best:
                    best = chi2
                    new_best = True
            else:
                flat.push(np.inf)

        stop = False
        if fixed_theta is None and label == "full" and full_count >= int(config["MAX_RUNS"]):
            stop = True
            print(f"[Daemon] Reached MAX_RUNS={config['MAX_RUNS']} full models", flush=True)
        if max_evals > 0 and eval_count >= max_evals:
            stop = True
            print(f"[Daemon] Reached MAX_EVALS={max_evals}", flush=True)

        if label == "full" and full_count % PROF_EVERY == 0:
            avg = lambda k: (t_acc[k] / t_cnt[k]) if t_cnt[k] else 0.0
            t_eval = t_acc["eval"]
            per_wave = t_eval / max(t_cnt["propose"], 1)
            per_theta = t_eval / max(t_cnt["eval"], 1)
            print(f"[PROF] full={full_count} evals={eval_count} proposals={proposal_count} best={best:.4f} propose={avg('propose'):.4f}s eval/wave={per_wave:.4f}s eval/theta={per_theta:.4f}s add={avg('add'):.4f}s", flush=True)
            t_acc.clear()
            t_cnt.clear()

        return stop, dict(theta=list(theta), pid=pid, label=label, status=status, final_status=final_status, chi2=chi2, finite_chi2=finite_chi2, valid_real_pass=valid_real_pass, valid_family_pass=valid_family_pass, new_best=new_best, full_index=full_count if label == "full" else None)

    def _evaluate_props(props):
        records = []
        stop = False
        grouped_props = defaultdict(list)
        for theta, pid, label, halo_type_variant in props:
            grouped_props[str(halo_type_variant).strip().lower()].append((theta, pid, label, halo_type_variant))
        for halo_type_chunk, props_for_halo in grouped_props.items():
            thetas = [theta for theta, pid, label, halo_type_variant in props_for_halo]
            for i in range(0, len(thetas), CHUNK):
                chunk_props, chunk_thetas = props_for_halo[i:i+CHUNK], thetas[i:i+CHUNK]
                theta_mat_external = np.array(chunk_thetas, dtype=float).T
                theta_mat = canonicalize_theta_matrix( theta_mat_external, halo_type=halo_type_chunk, halo_parameterization=halo_parameterization, bounds=config["THETA_BOUNDS"] )
                chunk_t0 = time.perf_counter()
                try:
                    print("[JL KWARG DEBUG]", flush=True)
                    print("stellar_model type:", type(stellar_model), flush=True)
                    print("stellar_model value:", stellar_model, flush=True)
                    print("surface_brightness_profile type:", type(surface_brightness_profile), flush=True)
                    print("surface_brightness_profile value:", surface_brightness_profile, flush=True)
                    print("karl_halo_params type:", type(karl_halo_params), flush=True)
                    print("karl_halo_params value:", karl_halo_params, flush=True)
                    print("velocity_edges type:", type(velocity_edges), flush=True)
                    print("velocity_edges value:", velocity_edges, flush=True)
                    Main._theta_mat_jl = _jl_matrix_f64(theta_mat, Main, juliacall, name="theta_mat")
                    Main._R_star_jl = _jl_vector_f64(R_star_m, Main, name="R_star_m")
                    Main._valid_vlos_jl = _jl_vector_bool(valid_vlos, Main, name="valid_vlos")
                    Main._v_star_jl = _jl_vector_f64(v_star_mps, Main, name="v_star_mps")
                    Main._verr_star_jl = _jl_vector_f64(verr_star_mps, Main, name="verr_star_mps")
                    Main._light_bins_jl = _jl_vector_f64(light_bin_edges, Main, name="light_bin_edges")
                    Main._kin_bins_jl = _jl_vector_f64(kinematic_bin_edges, Main, name="kinematic_bin_edges")
                    # PythonCall direct Julia function invocation is broken on the cluster.
                    # Put scalars/strings into Julia globals through seval, then call the
                    # batch evaluator entirely from Julia.
                    Main.seval(f"_sini_jl = {float(sini)!r}")
                    Main.seval(f"_Norbit_jl = {int(Norbit)}")
                    Main.seval("_halo_type_jl = " + json.dumps(str(halo_type_chunk)))
                    Main.seval("_tracer_constraint_mode_jl = " + json.dumps(tracer_constraint_mode))
                    Main.seval(f"_alphat_jl = {float(alphat)!r}")
                    Main.seval(f"_apfac_jl = {float(apfac)!r}")
                    Main.seval(f"_light_rel_tol_jl = {float(light_rel_tol)!r}")
                    Main.seval(f"_light_sigma_tol_jl = {float(light_sigma_tol)!r}")
                    Main.seval(f"_delta_chi2_iter_tol_jl = {float(delta_chi2_iter_tol)!r}")
                    Main.seval(f"_entropy_floor_jl = {float(entropy_floor)!r}")
                    Main.seval(f"_maxiter_jl = {int(maxiter)}")
                    Main.seval(f"_timeout_s_jl = {float(config.get('EVAL_TIMEOUT_S', 120.0))!r}")
                    Main.seval(f"_R_inner_pc_jl = {float(config.get('R_INNER_DIAG_PC', 30.0))!r}")
                    Main.seval(f"_Nvbin_jl = {int(nvbin)}")
                    Main.seval(f"_Ntheta_launch_jl = {int(ntheta_launch)}")
                    Main.seval(f"_halo_q_axis_ratio_jl = {float(halo_q_axis_ratio)!r}")
                    Main.seval(f"_orbit_fill_pct_jl = {float(orbit_fill_pct)!r}")
                    Main.seval(f"_orbit_regional_floor_jl = {float(orbit_regional_floor)!r}")
                    Main.seval(f"_orbit_max_regional_gap_jl = {float(orbit_max_regional_gap)!r}")
                    Main.seval(f"_orbit_shell_bands_jl = {int(orbit_shell_bands)}")
                    Main.seval(f"_orbit_coverage_check_every_jl = {int(orbit_coverage_check_every)}")
                    Main.seval(f"_orbit_warn_fill_pct_jl = {float(orbit_warn_fill_pct)!r}")
                    Main.seval(f"_orbit_warn_success_pct_jl = {float(orbit_warn_success_pct)!r}")
                    Main.seval(f"_orbit_warn_regional_floor_jl = {float(orbit_warn_regional_floor)!r}")
                    Main.seval(f"_orbit_warn_max_regional_gap_jl = {float(orbit_warn_max_regional_gap)!r}")
                    Main.seval(f"_model_owner_limit_jl = {int(model_owner_limit)}")
                    Main.seval(f"_threads_per_model_jl = {int(threads_per_model)}")
                    batch_result = Main.seval("""
OSPMPhysicsSpherical.evaluate_batch_theta(
_theta_mat_jl,
_R_star_jl,
_valid_vlos_jl,
_v_star_jl,
_verr_star_jl,
_sini_jl,
_Norbit_jl,
_halo_type_jl;
stellar_model=_stellar_model_jl,
surface_brightness_profile=_sb_profile_jl,
tracer_constraint_mode=_tracer_constraint_mode_jl,
alphat=_alphat_jl,
apfac=_apfac_jl,
light_rel_tol=_light_rel_tol_jl,
light_sigma_tol=_light_sigma_tol_jl,
delta_chi2_iter_tol=_delta_chi2_iter_tol_jl,
entropy_floor=_entropy_floor_jl,
maxiter=_maxiter_jl,
timeout_s=_timeout_s_jl,
R_inner_pc=_R_inner_pc_jl,
Nvbin=_Nvbin_jl,
Ntheta_launch=_Ntheta_launch_jl,
fill_pct=_orbit_fill_pct_jl,
regional_floor=_orbit_regional_floor_jl,
max_regional_gap=_orbit_max_regional_gap_jl,
shell_band_count=_orbit_shell_bands_jl,
coverage_check_every=_orbit_coverage_check_every_jl,
warn_fill_pct=_orbit_warn_fill_pct_jl,
warn_success_pct=_orbit_warn_success_pct_jl,
warn_regional_floor=_orbit_warn_regional_floor_jl,
warn_max_regional_gap=_orbit_warn_max_regional_gap_jl,
model_owner_limit=_model_owner_limit_jl,
threads_per_model=_threads_per_model_jl,
halo_q_axis_ratio=_halo_q_axis_ratio_jl,
karl_halo_params=_karl_halo_params_jl,
velocity_edges=_velocity_edges_jl,
light_bin_edges=_light_bins_jl,
kinematic_bin_edges=_kin_bins_jl
)
""")

                    (
                        status_code_vec,
                        chi2_vec,
                        chi2_inner_vec,
                        chi2_outer_vec,
                        delta_chi2_iteration_vec,
                        max_light_relative_residual_vec,
                        max_light_sigma_residual_vec,
                        light_constraint_ok_vec,
                        solver_converged_vec,
                        solver_iterations_vec,
                        solver_failure_reason_vec,
                        N_inner_vec,
                        N_outer_vec,
                        N_nonzero_weights_vec,
                        effective_N_orbits_vec,
                        max_weight_fraction_vec,
                        coverage_status_vec,
                        coverage_issue_region_vec,
                        coverage_issue_axis_vec,
                        coverage_issue_shell_bands_vec,
                        coverage_reasons_vec,
                        coverage_fraction_vec,
                        coverage_attempted_fraction_vec,
                        coverage_success_fraction_vec,
                        coverage_shell_min_vec,
                        coverage_lfrac_min_vec,
                        coverage_theta_min_vec,
                        coverage_shell_gap_vec,
                        coverage_lfrac_gap_vec,
                        coverage_theta_gap_vec,
                        coverage_joint_holes_vec,
                        coverage_deadline_hit_vec,
                        successful_base_orbits_vec,
                        planned_base_orbits_vec,
                        phase_volume_valid_vec,
                        phase_volume_convention_vec,
                        phase_volume_normalization_vec,
                        phase_volume_launches_recorded_vec,
                        phase_volume_sos_recorded_vec,
                        phase_volume_valid_base_orbits_vec,
                        phase_volume_invalid_recorded_orbits_vec,
                        phase_volume_nested_groups_vec,
                        phase_volume_duplicate_area_clusters_vec,
                        phase_volume_duplicate_area_orbits_vec,
                        raw_phase_volume_min_vec,
                        raw_phase_volume_max_vec,
                        raw_phase_volume_dynamic_range_vec,
                        normalized_phase_volume_min_vec,
                        normalized_phase_volume_max_vec,
                        wphase_min_vec,
                        wphase_max_vec,
                        wphase_dynamic_range_vec,
                        wphase_pair_max_relative_mismatch_vec,
                    ) = batch_result

                    t_acc["eval"] += time.perf_counter() - chunk_t0
                    t_cnt["eval"] += len(chunk_thetas)

                    for j, (theta, pid, label, halo_type_variant) in enumerate(chunk_props):
                        code = int(status_code_vec[j])
                        raw_chi2 = float(chi2_vec[j])
                        finite_chi2 = np.isfinite(raw_chi2) and raw_chi2 > 1e-12
                        if code == 0:
                            status = "pass" if finite_chi2 else "numeric_fail"
                        else:
                            status = {1: "orbit_fail", 2: "solver_failed", 3: "physics_exception", 4: "timeout"}.get(code, "unknown_fail")
                        chi2 = raw_chi2 if finite_chi2 else np.inf
                        coverage_status = str(coverage_status_vec[j])
                        coverage_issue_region = str(coverage_issue_region_vec[j])
                        coverage_issue_axis = str(coverage_issue_axis_vec[j])
                        diag = dict(
                            chi2_losvd=chi2,
                            delta_chi2_iteration=float(delta_chi2_iteration_vec[j]),
                            max_light_relative_residual=float(max_light_relative_residual_vec[j]),
                            max_light_sigma_residual=float(max_light_sigma_residual_vec[j]),
                            light_constraint_ok=bool(light_constraint_ok_vec[j]),
                            solver_converged=bool(solver_converged_vec[j]),
                            solver_iterations=int(solver_iterations_vec[j]),
                            solver_failure_reason=str(solver_failure_reason_vec[j]),
                            julia_status_code=code,
                            chi2_inner=float(chi2_inner_vec[j]),
                            chi2_outer=float(chi2_outer_vec[j]),
                            N_inner=int(N_inner_vec[j]),
                            N_outer=int(N_outer_vec[j]),
                            N_nonzero_weights=int(N_nonzero_weights_vec[j]),
                            effective_N_orbits=float(effective_N_orbits_vec[j]),
                            max_weight_fraction=float(max_weight_fraction_vec[j]),
                            halo_type=str(halo_type_variant),
                            alphat=alphat,
                            light_rel_tol=light_rel_tol,
                            light_sigma_tol=light_sigma_tol,
                            delta_chi2_iter_tol=delta_chi2_iter_tol,
                            halo_q_axis_ratio=halo_q_axis_ratio,
                            karl_halo_params_active=bool(karl_halo_params),
                            coverage_status=coverage_status,
                            coverage_strict=(coverage_status == "strict_pass"),
                            coverage_issue_region=coverage_issue_region,
                            coverage_issue_axis=coverage_issue_axis,
                            coverage_issue_shell_bands=str(coverage_issue_shell_bands_vec[j]),
                            coverage_reasons=str(coverage_reasons_vec[j]),
                            coverage_fraction=float(coverage_fraction_vec[j]),
                            coverage_attempted_fraction=float(coverage_attempted_fraction_vec[j]),
                            coverage_success_fraction=float(coverage_success_fraction_vec[j]),
                            coverage_shell_min=float(coverage_shell_min_vec[j]),
                            coverage_lfrac_min=float(coverage_lfrac_min_vec[j]),
                            coverage_theta_min=float(coverage_theta_min_vec[j]),
                            coverage_shell_gap=float(coverage_shell_gap_vec[j]),
                            coverage_lfrac_gap=float(coverage_lfrac_gap_vec[j]),
                            coverage_theta_gap=float(coverage_theta_gap_vec[j]),
                            coverage_joint_holes=int(coverage_joint_holes_vec[j]),
                            coverage_deadline_hit=bool(coverage_deadline_hit_vec[j]),
                            successful_base_orbits=int(successful_base_orbits_vec[j]),
                            planned_base_orbits=int(planned_base_orbits_vec[j]),
                            phase_volume_valid=bool(phase_volume_valid_vec[j]),
                            phase_volume_convention=str(phase_volume_convention_vec[j]),
                            phase_volume_normalization=str(phase_volume_normalization_vec[j]),
                            phase_volume_launches_recorded=int(phase_volume_launches_recorded_vec[j]),
                            phase_volume_sos_recorded=int(phase_volume_sos_recorded_vec[j]),
                            phase_volume_valid_base_orbits=int(phase_volume_valid_base_orbits_vec[j]),
                            phase_volume_invalid_recorded_orbits=int(phase_volume_invalid_recorded_orbits_vec[j]),
                            phase_volume_nested_groups=int(phase_volume_nested_groups_vec[j]),
                            phase_volume_duplicate_area_clusters=int(phase_volume_duplicate_area_clusters_vec[j]),
                            phase_volume_duplicate_area_orbits=int(phase_volume_duplicate_area_orbits_vec[j]),
                            raw_phase_volume_min=float(raw_phase_volume_min_vec[j]),
                            raw_phase_volume_max=float(raw_phase_volume_max_vec[j]),
                            raw_phase_volume_dynamic_range=float(raw_phase_volume_dynamic_range_vec[j]),
                            normalized_phase_volume_min=float(normalized_phase_volume_min_vec[j]),
                            normalized_phase_volume_max=float(normalized_phase_volume_max_vec[j]),
                            wphase_min=float(wphase_min_vec[j]),
                            wphase_max=float(wphase_max_vec[j]),
                            wphase_dynamic_range=float(wphase_dynamic_range_vec[j]),
                            wphase_pair_max_relative_mismatch=float(wphase_pair_max_relative_mismatch_vec[j]),
                        )
                        stop_now, record = _record(theta, pid, label, status, chi2, diag=diag)
                        records.append(record)
                        if stop_now:
                            stop = True
                            break
                except Exception as e:
                    t_acc["eval"] += time.perf_counter() - chunk_t0
                    t_cnt["eval"] += len(chunk_thetas)
                    print("\n===== JULIA EXCEPTION =====", flush=True)
                    print("type:", type(e), flush=True)
                    print("repr:", repr(e), flush=True)
                    try:
                        print("str:", str(e), flush=True)
                    except Exception:
                        print("str(): failed", flush=True)
                    if hasattr(e, "exception"):
                        try:
                            print("julia exception:", repr(e.exception), flush=True)
                        except Exception:
                            print("could not print e.exception", flush=True)
                    print("[Daemon] Python traceback for failed chunk:", flush=True)
                    traceback.print_exc()
                    raise
                if stop:
                    break
            if stop:
                break
        return stop, records
    fixer.unlock(deck, runner)
    runner.train(deck)
    while full_count < int(config["MAX_RUNS"]):
        print(f"[Daemon] loop full={full_count} evals={eval_count} proposals={proposal_count}", flush=True)
        t0 = time.perf_counter()
        deck._flush_buf()
        if fixed_theta is not None:
            selected = config.get("EVAL_VARIANTS", ["full", "no_bh", "no_halo"])
            if isinstance(selected, str):
                selected = [selected]
            props = _dedupe_props([_bounded_prop(list(fixed_theta), runner.step + 1, label) for label in selected])
            runner.step += 1
            proposal_count += 1
            t_acc["propose"] += time.perf_counter() - t0
            t_cnt["propose"] += 1
            print(f"[Daemon] fixed-theta evaluation: {len(props)} unique variant(s)", flush=True)
            _evaluate_props(props)
            deck.save()
            return
        remaining = int(config["MAX_RUNS"]) - full_count
        wave_size = min(feedback_batch, remaining)
        base_props = runner.propose(deck, n=wave_size)
        proposal_count += len(base_props)
        print("base_props[:3] =", base_props[:3])
        full_props = _dedupe_props([_bounded_prop(theta, pid, "full") for theta, pid in base_props])
        t_acc["propose"] += time.perf_counter() - t0
        t_cnt["propose"] += 1
        print(f"[Daemon] proposing {len(full_props)} full models, starting eval...", flush=True)
        stop, full_records = _evaluate_props(full_props)
        deck._flush_buf()
        if stop:
            deck.save()
            return
        fixer.unlock(deck, runner)
        t0 = time.perf_counter()
        runner.train(deck)
        t_acc["train"] += time.perf_counter() - t0
        t_cnt["train"] += 1
        if not runner.fill_triggered and runner.detect_basin(deck):
            runner.fill_mode = runner.fill_triggered = True
            print(f"[Daemon] Basin detected at full_count={full_count} — switching to fill mode", flush=True)
        if flat.flat():
            deck.save()
            print(f"[Daemon] Flat region detected after {full_count} full models", flush=True)
            return
        if converge.check(deck, runner, full_count):
            deck.save()
            return
        relevant_full = [r for r in full_records if r["valid_real_pass"] and np.isfinite(best) and (r["chi2"] - best <= alt_trigger_delta)]
        alt_props = []
        for record in relevant_full:
            theta, pid = record["theta"], record["pid"]
            if _family_region_sparse(theta, "no_bh"):
                alt_props.append(_bounded_prop(theta, pid, "no_bh"))
            if _family_region_sparse(theta, "no_halo"):
                alt_props.append(_bounded_prop(theta, pid, "no_halo"))
        alt_props = _dedupe_props(alt_props)
        if alt_props:
            print(f"[SCIENCE BRANCH] evaluating {len(alt_props)} restricted-model probe(s) from {len(relevant_full)} relevant full fit(s)", flush=True)
            stop, alt_records = _evaluate_props(alt_props)
            deck._flush_buf()
            if stop:
                deck.save()
                return
            fixer.unlock(deck, runner)
            runner.train(deck)
            full_by_pid = {r["pid"]: r for r in full_records if r["valid_real_pass"]}
            sensitivity_props = []
            for record in alt_records:
                if not record["valid_family_pass"]:
                    continue
                matched_full = full_by_pid.get(record["pid"])
                if matched_full is None:
                    continue
                delta = record["chi2"] - matched_full["chi2"]
                if not np.isfinite(delta) or delta > alt_sensitivity_delta:
                    continue
                if record["label"] == "no_bh":
                    labels = ["no_bh_halo_up", "no_bh_halo_down", "no_bh_halo_scale_up", "no_bh_halo_scale_down", "no_bh_ml_up", "no_bh_ml_down"]
                elif record["label"] == "no_halo":
                    labels = ["no_halo_bh_up", "no_halo_bh_down", "no_halo_ml_up", "no_halo_ml_down"]
                else:
                    continue
                print(f"[SCIENCE BRANCH] {record['label']} competitive at proposal={record['pid']} delta_chi2={delta:.6g}; expanding local sensitivity", flush=True)
                sensitivity_props.extend(_bounded_prop(record["theta"], record["pid"], label) for label in labels)
            sensitivity_props = _dedupe_props(sensitivity_props)
            if sensitivity_props:
                stop, _ = _evaluate_props(sensitivity_props)
                deck._flush_buf()
                runner.train(deck)
                if stop:
                    deck.save()
                    return

    deck.save()