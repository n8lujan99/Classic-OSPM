# OSPM_Daemon.py — STAYS IN PYTHON FOREVER.  Parallelism lives in Julia, not here.
#
# WHAT THIS DOES
# Drives an RL-guided search over dark-matter halo parameters θ = (ρ_s, r_s, M_BH, M/L).
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
# ──────────────────────────────────────────────────────────────────────────────
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
# Corpo(engine)            physics_engine → wrapper              — calls Julia physics, returns status+chi²
#   .eval(theta)           [float] → (status_str, chi2)
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
# ──────────────────────────────────────────────────────────────────────────────

import os, time, sys
import numpy as np, pandas as pd
import torch, torch.nn as nn
from collections import deque

torch.backends.cudnn.benchmark = False
try: from sklearn.preprocessing import StandardScaler
except Exception: StandardScaler = None
def clamp(x, lo, hi): return max(lo, min(hi, x))
def random_theta(bounds): return [np.random.uniform(lo, hi) for lo, hi in bounds]
def min_dist(theta, arr):
    if len(arr) == 0: return np.inf
    return np.linalg.norm(np.asarray(arr) - np.asarray(theta), axis=1).min()

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
            defaults = {
                "R_cyl_col": "R_cyl_pc",
                "z_col": "z_pc",
                "nu_col": "nu_Lsun_pc3",
                "volume_col": "cell_volume_pc3",
                "luminosity_col": "cell_luminosity_Lsun",
            }
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

    for key in (
        "qdm", "dis", "v0", "rc", "rc_pc", "xmgamma", "xmgamma_msun",
        "rsgamma", "rsgamma_pc", "gamma", "cnfw", "rsnfw", "rsnfw_pc",
        "gdennorm", "halo_force_softening_pc",
    ):
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
        config.get("surface_brightness_profile"),
    ]

    obs_cfg = config.get("OBSERVABLES", {})
    if isinstance(obs_cfg, dict):
        candidates.extend([
            obs_cfg.get("SURFACE_BRIGHTNESS_PROFILE"),
            obs_cfg.get("surface_brightness_profile"),
        ])

    for profile in candidates:
        if profile is not None:
            return profile

    raise RuntimeError(
        "surface_brightness_profile is required for Karl-style OSPM; "
        "no star-count fallback is allowed"
    )

def _observable_config(config):
    obs_cfg = config.get("OBSERVABLES", {})
    if obs_cfg is None:
        obs_cfg = {}
    if not isinstance(obs_cfg, dict):
        raise TypeError("config['OBSERVABLES'] must be a dict when provided")
    return obs_cfg

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
        self.config, self.path, self.cols = config, config["CSV_PATH"], config["REQUIRE_COLUMNS"]
        self.params, self.flush = config["PARAMETER_NAMES"], int(config.get("CSV_FLUSH_INTERVAL", 50))
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

    def add(self, theta, chi2, reward, pid, status, refine_passes=None, diag=None):
        row_dict = {k: theta[i] for i, k in enumerate(self.params)}
        row_dict |= dict( chi2=chi2, reward=reward, status=status, proposal_id=pid, refine_passes=refine_passes)
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

        
class Corpo:
    def __init__(self, engine): self.engine = engine
    def eval(self, theta):
        try:
            chi2 = float(self.engine(theta))
            return ("pass", chi2) if np.isfinite(chi2) else ("numeric_fail", np.inf)
        except FloatingPointError: return "numeric_fail", np.inf
        except RuntimeError as e:
            import traceback; print("[Corpo] RuntimeError:", repr(e)); traceback.print_exc(); raise
        except Exception: raise
class Fixer:
    def __init__(self, cfg): self.warmup = int(cfg.get("AI_START_AFTER", 500)); self.unlocked = False
    def unlock(self, deck, runner):
        if self.unlocked: return
        if deck.df.status.str.startswith("pass").sum() >= self.warmup:
            runner.enable_ai(); self.unlocked = True; print("[AI] unlocked", flush=True)
    def reward(self, status, chi2): return -1e6 if status != "pass" else -float(chi2)

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
        self.rel_thr = float(cfg.get("CONVERGE_REL_SPREAD", 0.05)); self.chi_thr = float(cfg.get("CONVERGE_CHI_STD", 0.5))
        self.n_top = int(cfg.get("CONVERGE_N_TOP", 200)); self.n_min = int(cfg.get("CONVERGE_MIN_PASS", 500))
        self.patience = int(cfg.get("CONVERGE_PATIENCE", 3)); self.every = int(cfg.get("CONVERGE_CHECK_EVERY", 500))
        self.bounds, self.cols, self.cnt = bounds, cols, 0
    def check(self, deck, runner, runs):
        if not runner.fill_mode: return False
        if runs % self.every != 0: return False
        good = deck.df[deck.df.status.str.startswith("pass")]
        if len(good) < self.n_min: self.cnt = 0; return False
        top = good.nsmallest(min(len(good), self.n_top), "chi2")
        chi_std = top["chi2"].std(); spread = np.std(top[self.cols].values, axis=0)
        span = np.array([hi - lo for lo, hi in self.bounds]); rel_spread = np.mean(spread / span)
        if chi_std < self.chi_thr and rel_spread < self.rel_thr: self.cnt += 1
        else: self.cnt = 0
        converged = self.cnt >= self.patience
        if converged: print(f"[Converge] posterior converged at run {runs}: rel_spread={rel_spread:.4f} chi_std={chi_std:.4f}", flush=True)
        return converged
class Runner:
    def __init__(self, cfg):
        self.cfg, self.bounds, self.cols = cfg, cfg["THETA_BOUNDS"], cfg["PARAMETER_NAMES"]
        self.dim, self.batch, self.min_d = len(self.cols), int(cfg["BATCH_SIZE"]), float(cfg["MIN_DISTANCE"])
        self.ai, self.model, self.agent, self.opt_m, self.opt_a = False, None, None, None, None
        self.noise0, self.noise1, self.tau = float(cfg.get("AI_NOISE_INIT", 0.3)), float(cfg.get("AI_NOISE_MIN", 0.02)), float(cfg.get("AI_NOISE_TAU", 5000))
        self.step = 0; self.recent = deque(maxlen=5000)
        self.scaler = IdentityScaler() if StandardScaler is None else StandardScaler()
        self.scaled, self.fill_mode, self.fill_triggered = False, False, False
        self.explore_frac = float(cfg.get("EXPLORE_FRACTION", 0.0))

    def enable_ai(self):
        self.model = Model(self.dim); self.agent = Agent(self.dim)
        self.opt_m = torch.optim.Adam(self.model.parameters(), 1e-3)
        self.opt_a = torch.optim.Adam(self.agent.parameters(), 1e-3); self.ai = True

    def _noise(self): return self.noise0 if not self.ai else max(self.noise1, self.noise0 * np.exp(-self.step / self.tau))

    def _base(self, deck):
        good = deck.df[deck.df.status.str.startswith("pass")]
        if len(good) >= 10:
            if np.random.rand() < 0.15:
                return good[self.cols].sample(1).values[0]
            return good.nsmallest(min(len(good), 500), "chi2")[self.cols].sample(1).values[0]
        return deck.df[self.cols].dropna().sample(1).values[0]

    def detect_basin(self, deck):
        good = deck.df[deck.df.status.str.startswith("pass")]
        if len(good) < 500: return False
        top = good.nsmallest(min(len(good), 200), "chi2")
        chi_std = top["chi2"].std(); spread = np.std(top[self.cols].values, axis=0)
        span = np.array([hi - lo for lo, hi in self.bounds]); rel_spread = np.mean(spread / span)
        return (chi_std < 1.0) and (rel_spread < 0.15) and ("refine_passes" in good.columns and good["refine_passes"].fillna(0).median() >= self.cfg.get("MAX_REFINE", 0))

    def step_scale(self, deck):
        if not self.ai or not self.fill_mode: return 0.2
        good = deck.df[deck.df.status.str.startswith("pass")]
        top = good.nsmallest(min(len(good), 200), "chi2")
        spread = np.std(top[self.cols].values, axis=0); span = np.array([hi - lo for lo, hi in self.bounds])
        return clamp(0.01 + 0.2 * np.mean(spread / span), 0.01, 0.05)

    def propose(self, deck):
        out = []
        while len(out) < self.batch:
            if self.ai and not (self.explore_frac > 0 and np.random.rand() < self.explore_frac):
                if self.fill_mode:
                    good = deck.df[deck.df.status.str.startswith("pass")]
                    base = good.nsmallest(100, "chi2")[self.cols].sample(1).values[0]
                else:
                    base = self._base(deck)
                xb = self.scaler.transform(base.reshape(1, -1)) if self.scaled else base.reshape(1, -1)
                a = self.agent.act(torch.tensor(xb, dtype=torch.float32), self._noise()).numpy().squeeze()
                s = self.step_scale(deck)
                if self.fill_mode and self.step % 200 == 0:
                    print(f"[FillMode] step_scale={s:.4f}", flush=True)
                theta = [clamp(base[i] + s * (hi - lo) * a[i], lo, hi)
                         for i, (lo, hi) in enumerate(self.bounds)]
            else:
                theta = random_theta(self.bounds)

            if deck.is_forbidden(theta):
                continue
            if not self.fill_mode:
                if min_dist(theta, self.recent) < self.min_d:
                    continue
                if deck.nearest_distance(theta, self.min_d) < self.min_d:
                    continue

            self.recent.append(theta)
            self.step += 1
            out.append((theta, self.step))
        return out

    def train(self, deck):
        if not self.ai: return
        df = deck.df[deck.df.status.str.startswith("pass") & np.isfinite(deck.df.reward)]
        if len(df) < 200: return
        if len(df) > 5000: df = df.tail(5000)
        X, y = df[self.cols].values, df.reward.values.reshape(-1, 1)
        if not self.scaled: self.scaler.fit(X); self.scaled = True
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        loss = ((self.model(Xt) - yt) ** 2).mean()
        self.opt_m.zero_grad(); loss.backward(); self.opt_m.step()
        
def run_daemon(config, physics_engine):
    from collections import defaultdict
    deck, runner, corpo, fixer = Deck(config), Runner(config), Corpo(physics_engine), Fixer(config)
    print("CONFIG PARAMETER_NAMES:", config["PARAMETER_NAMES"])
    print("CONFIG THETA_BOUNDS:", config["THETA_BOUNDS"])
    print("runner.bounds:", runner.bounds)
    flat = FlatDetector(config.get("FLAT_WINDOW", 200), config.get("FLAT_THRESHOLD", 1e-6), config.get("FLAT_PATIENCE", 3))
    converge = ConvergenceDetector(config, config["THETA_BOUNDS"], config["PARAMETER_NAMES"])
    runs, best = 0, np.inf
    t_acc, t_cnt = defaultdict(float), defaultdict(int); PROF_EVERY = int(config.get("PROF_EVERY", 25))
    obs = getattr(physics_engine, "__wrapped_obs__", None)
    base_halo_type = str(getattr(physics_engine, "__halo_type__", config.get("HALO_TYPE", "nfw"))).strip().lower()
    halo_type = base_halo_type
    use_batch = obs is not None

    if use_batch:
        from juliacall import Main; import juliacall

        stellar_model = _clean_stellar_model(getattr(obs, "stellar_model", None))
        surface_brightness_profile = _get_surface_brightness_profile(config, physics_engine, obs)

        obs_cfg = _observable_config(config)
        engine_cfg = getattr(physics_engine, "__karl_config__", {}) or {}
        if not isinstance(engine_cfg, dict):
            raise TypeError("physics_engine.__karl_config__ must be a dict when provided")

        def opt(*names, default=None):
            for name in names:
                if name in obs_cfg and obs_cfg[name] is not None:
                    return obs_cfg[name]
                if name in engine_cfg and engine_cfg[name] is not None:
                    return engine_cfg[name]
                if name in config and config[name] is not None:
                    return config[name]
            return default

        lambda_light = float(opt("LAMBDA_LIGHT", "LAMBDA_OCC", "lambda_light", "lambda_occ", default=1.0))
        min_stars_per_bin = int(opt("MIN_STARS_PER_BIN", "min_stars_per_bin", default=20))
        nvbin = int(opt("NVBIN", "Nvbin", "nvbin", default=21))
        ntheta_launch = int(opt("NTHETA_LAUNCH", "Ntheta_launch", "ntheta_launch", default=9))
        velocity_edges = opt("VELOCITY_EDGES", "velocity_edges", default=None)
        alpha = float(opt("KARL_ALPHA", "alpha", default=config.get("ALPHA", 1e-4)))
        alphat = float(opt("KARL_ALPHAT", "alphat", default=config.get("ALPHAT", 1.0)))
        maxiter = int(opt("KARL_MAXITER", "maxiter", default=config.get("MAXITER", 60)))
        weight_mode = str(opt("WEIGHT_MODE", "weight_mode", default="entropy")).strip().lower()
        weight_solver_mode = str(opt("WEIGHT_SOLVER", "weight_solver_mode", default="orbit_only")).strip().lower()
        losvd_score_mode = str(opt("LOSVD_SCORE_MODE", "losvd_score_mode", default="karl_fracnew")).strip().lower()
        entropy_floor = float(opt("ENTROPY_FLOOR", "entropy_floor", default=config.get("ENTROPY_FLOOR", 1e-12)))
        halo_q_axis_ratio = float(opt("HALO_Q_AXIS_RATIO", "halo_q_axis_ratio", default=config.get("HALO_Q_AXIS_RATIO", 1.0)))
        karl_halo_params = _clean_karl_halo_params(
            opt("KARL_HALO_PARAMS", "karl_halo_params", default=config.get("KARL_HALO_PARAMS", None))
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
            raise RuntimeError(
                "wrapped physics arrays must match lengths: "
                f"R={R_star_m.size}, valid={valid_vlos.size}, v={v_star_mps.size}, verr={verr_star_mps.size}"
            )

        kinematic_bin_edges = getattr(physics_engine, "__kinematic_bin_edges_pc__", None)
        if kinematic_bin_edges is None:
            kinematic_bin_edges = engine_cfg.get("kinematic_bin_edges_pc", None)
        if kinematic_bin_edges is None:
            kinematic_bin_edges = getattr(obs, "kinematic_bin_edges_pc", None)
        if kinematic_bin_edges is None:
            raise RuntimeError("kinematic_bin_edges_pc is required; no adaptive radial-bin fallback is allowed")

        kinematic_bin_edges = np.asarray(kinematic_bin_edges, float).ravel() * 3.0856775814913673e16
        nocc_compat = int(opt("NBINS_OCC", "nbins_occ", default=0))

        jl_batch = Main.OSPMPhysicsSpherical.evaluate_batch_theta
        sini = float(obs.sini)
        Norbit = int(obs.Norbit)
        if Norbit % 2 != 0:
            raise RuntimeError(
                f"Karl paired-orbit Spherical path requires even Norbit because Norbit is the final column count; got Norbit={Norbit}"
            )
        nstar_vlos = int(np.count_nonzero(valid_vlos))
        print(
            f"[Daemon] Karl batch mode ON — Norbit={Norbit}, Nbase_orbit={Norbit // 2}, Nstar_vlos={nstar_vlos}, "
            f"min_stars_per_bin={min_stars_per_bin}, Nvbin={nvbin}, Ntheta_launch={ntheta_launch}, "
            f"weight_mode={weight_mode}, weight_solver_mode={weight_solver_mode}, "
            f"losvd_score_mode={losvd_score_mode}, halo_q={halo_q_axis_ratio}, "
            f"karl_halo_params_active={karl_halo_params is not None}",
            flush=True,
        )
    else: print("[Daemon] batch mode OFF — falling back to serial corpo.eval", flush=True)

    while runs < config["MAX_RUNS"]:
        print(f"[Daemon] loop iter runs={runs}", flush=True); t0 = time.perf_counter()
        deck._flush_buf()
        base_props = runner.propose(deck)
        print("base_props[:3] =", base_props[:3])
        props = []
        for theta, pid in base_props:
            rho_s, r_s, MBH, ML = theta
            variants = [
                ("full",       [rho_s,        r_s, MBH,       ML], base_halo_type),
                # isolate major gravitating components
                ("bh_only",    [0.0,          r_s, MBH,       ML], "none"),  # stars + BH, no halo
                ("halo_only",  [rho_s,        r_s, 0.0,       ML], base_halo_type),
                # BH perturbations
                ("bh_up",      [rho_s,        r_s, MBH * 2.0, ML], base_halo_type),
                ("bh_down",    [rho_s,        r_s, MBH * 0.5, ML], base_halo_type),
                # halo perturbations
                ("halo_up",    [rho_s * 2.0,  r_s, MBH,       ML], base_halo_type),
                ("halo_down",  [rho_s * 0.5,  r_s, MBH,       ML], base_halo_type),
                # stellar M/L perturbations
                ("ml_up",      [rho_s,        r_s, MBH,       ML * 2.0], base_halo_type),
                ("ml_down",    [rho_s,        r_s, MBH,       ML * 0.5], base_halo_type),
            ]
            # keep each perturbed theta inside bounds
            bounded_variants = []
            for label, tvar, halo_type_variant in variants:
                tfix = []
                for k, x in enumerate(tvar):
                    lo, hi = config["THETA_BOUNDS"][k]
                    tfix.append(clamp(float(x), float(lo), float(hi)))
                bounded_variants.append((label, tfix, halo_type_variant))

            for label, tvar, halo_type_variant in bounded_variants:
                props.append((tvar, pid, label, halo_type_variant))

        t_acc["propose"] += time.perf_counter() - t0; t_cnt["propose"] += 1
        print(f"[Daemon] proposing {len(props)} variants, starting eval...", flush=True)
        _jnt = os.environ.get("JULIA_NUM_THREADS", "1")
        _nthreads = (os.cpu_count() or 1) if _jnt == "auto" else int(_jnt)
        CHUNK = int(config.get("CHUNK_SIZE", max(3 * _nthreads, len(props))))

        def _record(theta, pid, label, status, chi2, refine_passes, diag=None):
            nonlocal best, runs
            base_reward = fixer.reward(status, chi2)
            reward = base_reward + 0.5 * (1.0 - refine_passes / max(1, config.get("MAX_REFINE", 1))) if status == "pass" else base_reward
            t_add = time.perf_counter()
            deck.add(theta, chi2, reward, pid, f"{status}_{label}", refine_passes=refine_passes, diag=diag)
            t_acc["add"] += time.perf_counter() - t_add; t_cnt["add"] += 1
            valid_pass = (status == "pass") and np.isfinite(chi2) and (chi2 > 1e-12)
            if valid_pass:
                flat.push(chi2 if refine_passes >= config.get("MAX_REFINE", 0) else np.inf)
                fixer.unlock(deck, runner)
                if chi2 < best: best = chi2
            else: flat.push(np.inf)
            runs += 1
            if not runner.fill_triggered and runner.detect_basin(deck):
                runner.fill_mode = runner.fill_triggered = True
                print(f"[Daemon] Basin detected at run {runs} — switching to fill mode", flush=True)
            if runs % PROF_EVERY == 0:
                avg = lambda k: (t_acc[k] / t_cnt[k]) if t_cnt[k] else 0.0
                t_eval = t_acc["eval"]; per_batch = t_eval / max(t_cnt["propose"], 1); per_theta = t_eval / max(t_cnt["eval"], 1)
                print(f"[PROF] runs={runs} best={best:.4f} propose={avg('propose'):.4f}s eval/batch={per_batch:.4f}s eval/theta={per_theta:.4f}s add={avg('add'):.4f}s", flush=True)
                t_acc.clear(); t_cnt.clear()
            if flat.flat(): deck.save(); print(f"[Daemon] Flat region detected after {runs} runs", flush=True); return True
            if converge.check(deck, runner, runs): deck.save(); return True
            return False
        stop = False
        if use_batch:
            grouped_props = defaultdict(list)
            for theta, pid, label, halo_type_variant in props:
                grouped_props[str(halo_type_variant).strip().lower()].append((theta, pid, label, halo_type_variant))

            for halo_type_chunk, props_for_halo in grouped_props.items():
                thetas = [theta for theta, pid, label, halo_type_variant in props_for_halo]

                for i in range(0, len(thetas), CHUNK):
                    chunk_props, chunk_thetas = props_for_halo[i:i+CHUNK], thetas[i:i+CHUNK]
                    theta_mat = np.array(chunk_thetas, dtype=float).T
                    chunk_t0 = time.perf_counter()

                    try:
                        (
                            status_code_vec,
                            chi2_vec,
                            refine_vec,
                            chi2_inner_vec,
                            chi2_outer_vec,
                            chi2_light_vec,
                            N_inner_vec,
                            N_outer_vec,
                            N_nonzero_weights_vec,
                            effective_N_orbits_vec,
                            max_weight_fraction_vec,
                        ) = jl_batch(
                            juliacall.convert(Main.Matrix[Main.Float64], theta_mat),
                            juliacall.convert(Main.Vector[Main.Float64], R_star_m),
                            juliacall.convert(Main.Vector[Main.Bool], valid_vlos),
                            juliacall.convert(Main.Vector[Main.Float64], v_star_mps),
                            juliacall.convert(Main.Vector[Main.Float64], verr_star_mps),
                            sini,
                            Norbit,
                            halo_type_chunk,
                            stellar_model=stellar_model,
                            surface_brightness_profile=surface_brightness_profile,
                            Nocc=nocc_compat,
                            lambda_occ=lambda_light,
                            alpha=alpha,
                            alphat=alphat,
                            weight_mode=weight_mode,
                            weight_solver_mode=weight_solver_mode,
                            entropy_floor=entropy_floor,
                            losvd_score_mode=losvd_score_mode,
                            maxiter=maxiter,
                            max_refine=config.get("MAX_REFINE", 0),
                            timeout_s=float(config.get("EVAL_TIMEOUT_S", 120.0)),
                            R_inner_pc=float(config.get("R_INNER_DIAG_PC", 30.0)),
                            min_stars_per_bin=min_stars_per_bin,
                            Nvbin=nvbin,
                            Ntheta_launch=ntheta_launch,
                            halo_q_axis_ratio=halo_q_axis_ratio,
                            karl_halo_params=karl_halo_params,
                            velocity_edges=velocity_edges,
                            kinematic_bin_edges=juliacall.convert(Main.Vector[Main.Float64], kinematic_bin_edges),
                        )

                        t_acc["eval"] += time.perf_counter() - chunk_t0
                        t_cnt["eval"] += len(chunk_thetas)

                        for j, (theta, pid, label, halo_type_variant) in enumerate(chunk_props):
                            chi2 = float(chi2_vec[j])
                            code = int(status_code_vec[j])
                            refine_passes = int(refine_vec[j])
                            valid_pass = (code == 0) and np.isfinite(chi2) and (chi2 > 1e-12)
                            status = "pass" if valid_pass else {1: "orbit_fail", 2: "numeric_fail", 4: "timeout"}.get(code, "unknown_fail")
                            chi2_light = float(chi2_light_vec[j])
                            diag = dict(
                                chi2_losvd=chi2,
                                chi2_light=chi2_light,
                                chi2_total=chi2 + lambda_light * chi2_light if np.isfinite(chi2) and np.isfinite(chi2_light) else np.inf,
                                chi2_inner=float(chi2_inner_vec[j]),
                                chi2_outer=float(chi2_outer_vec[j]),
                                N_inner=int(N_inner_vec[j]),
                                N_outer=int(N_outer_vec[j]),
                                N_nonzero_weights=int(N_nonzero_weights_vec[j]),
                                effective_N_orbits=float(effective_N_orbits_vec[j]),
                                max_weight_fraction=float(max_weight_fraction_vec[j]),
                                halo_type=str(halo_type_variant),
                                weight_mode=weight_mode,
                                weight_solver_mode=weight_solver_mode,
                                losvd_score_mode=losvd_score_mode,
                                alphat=alphat,
                                halo_q_axis_ratio=halo_q_axis_ratio,
                                karl_halo_params_active=bool(karl_halo_params),
                            )

                            if not valid_pass:
                                chi2 = np.inf
                            if _record(theta, pid, label, status, chi2, refine_passes, diag=diag):
                                stop = True
                                break

                    except Exception as e:
                        t_acc["eval"] += time.perf_counter() - chunk_t0
                        t_cnt["eval"] += len(chunk_thetas)
                        print(f"[Daemon] chunk failed halo_type={halo_type_chunk} (size={len(chunk_thetas)}): {e}", flush=True)
                        for theta, pid, label, halo_type_variant in chunk_props:
                            if _record(theta, pid, label, "numeric_fail", np.inf, 0, diag={"halo_type": str(halo_type_variant)}):
                                stop = True
                                break

                    if stop:
                        break

                if stop:
                    break

        else:
            for theta, pid, label, halo_type_variant in props:
                chunk_t0 = time.perf_counter(); status, chi2 = corpo.eval(theta)
                t_acc["eval"] += time.perf_counter() - chunk_t0; t_cnt["eval"] += 1
                if _record(theta, pid, label, status, chi2, 0, diag={"halo_type": str(halo_type_variant)}): stop = True; break
        if stop: return
        t0 = time.perf_counter(); deck._flush_buf(); runner.train(deck)
        t_acc["train"] += time.perf_counter() - t0; t_cnt["train"] += 1
    deck.save()