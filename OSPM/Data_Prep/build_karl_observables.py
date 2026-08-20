"""
# ========================================================================================================================

Build one galaxy-agnostic Karl-style observables CSV from an OSPM galaxy config.

The script contains the reusable Karl algorithms only. Galaxy-specific values
(distance, projected axis ratio, radial domain, seeing, aperture definitions,
velocity grid, source paths, and LOSVD settings) live in the galaxy config.

Required existing inputs are read from CONFIG:
    DATA_CSV
    SURFACE_BRIGHTNESS_CSV
    DISTANCE_PC
    AXIS_RATIO_Q
    STAR_R_COL
    STAR_V_COL
    STAR_VERR_COL
    V_SYS_KMS
    KARL_OBSERVABLES

KARL_OBSERVABLES chooses the reusable construction strategy. Supported radial
grid modes are:
    karl_log        - Karl's logarithmic projected grid, used for historical
                      reconstructions such as Segue 1.
    kinematic_bins  - use the existing KINEMATIC_BINS_CSV radial edges directly.

Supported aperture sources are:
    explicit        - use KARL_OBSERVABLES['apertures'].
    kinematic_bins  - one aperture per existing kinematic radial bin.

Supported velocity-grid modes are:
    fixed_centers   - configured model_vmin_kms/model_vmax_kms are the outer
                      velocity-bin centers.
    data_range      - derive those outer centers from the finite systemic-
                      centered stellar velocities in DATA_CSV.

Seeing is explicit. seeing_arcsec=0 means identity/no seeing; positive values
use the Karl getsum-style Gaussian transfer.

The generated CSV contains long-format rows with row_type:
    metadata
    spatial_cell
    seeing_transfer
    aperture
    losvd_bin

No historical ab.dat, galaxy.params, SB.new, SBsc2.dat, Hermite file, or
velbin file is required. Their useful information is generated directly from
the current galaxy data and the per-galaxy config.
# ========================================================================================================================

"""

from __future__ import annotations
import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

CVAL = 1.0
INVALID_SIGMA_SENTINEL = -666.0
DEFAULT_PSF_SAMPLES = 100
DEFAULT_LIGHT_QUADRATURE = 8
DEFAULT_TRANSVD_SAMPLES = 1000
KARL_LOSVD_SIGMA_EXTENT = 4.5
KARL_MKHERM_NSIM = 100
KARL_MKHERM_ECONT_FRAC = 1.0 / 60.0

def _fortran_nint(x: float) -> int:
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)

def _fortran_int_div(numerator: int, denominator: int) -> int:
    return math.trunc(numerator / denominator)

def _intrinsic_dispersion_mle(v_kms: np.ndarray, verr_kms: np.ndarray) -> tuple[float, float, float, float]:
    v = np.asarray(v_kms, dtype=np.float64)
    e = np.asarray(verr_kms, dtype=np.float64)
    good = np.isfinite(v) & np.isfinite(e) & (e > 0.0)
    v = v[good]
    e = e[good]
    if len(v) < 2:
        raise ValueError("Intrinsic-dispersion fit requires at least two finite velocities with positive errors")

    def profile_nll(sigma: float) -> tuple[float, float]:
        sigma = max(float(sigma), 0.0)
        var = sigma * sigma + e * e
        w = 1.0 / var
        mu = float(np.sum(w * v) / np.sum(w))
        nll = 0.5 * float(np.sum(np.log(var) + (v - mu) ** 2 / var))
        return nll, mu

    sample_scale = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    upper = max(1.0, 5.0 * sample_scale, 5.0 * float(np.median(e)), float(np.max(np.abs(v - np.median(v)))))
    grid = np.r_[0.0, np.geomspace(1.0e-6, upper, 512)]
    nll = np.asarray([profile_nll(s)[0] for s in grid], dtype=np.float64)
    i = int(np.argmin(nll))

    if i == 0:
        sigma_best = 0.0
    else:
        lo = float(grid[max(i - 1, 0)])
        hi = float(grid[min(i + 1, len(grid) - 1)])
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        c = hi - (hi - lo) / phi
        d = lo + (hi - lo) / phi
        fc = profile_nll(c)[0]
        fd = profile_nll(d)[0]
        for _ in range(120):
            if hi - lo <= 1.0e-10 * max(1.0, hi):
                break
            if fc <= fd:
                hi = d
                d = c
                fd = fc
                c = hi - (hi - lo) / phi
                fc = profile_nll(c)[0]
            else:
                lo = c
                c = d
                fc = fd
                d = lo + (hi - lo) / phi
                fd = profile_nll(d)[0]
        sigma_best = 0.5 * (lo + hi)

    best_nll, mu_best = profile_nll(sigma_best)
    if not math.isfinite(sigma_best) or sigma_best <= 0.0:
        raise ValueError(f"Intrinsic-dispersion fit returned non-positive sigma={sigma_best}")

    target_nll = best_nll + 0.5

    def bisect_crossing(lo: float, hi: float) -> float:
        flo = profile_nll(lo)[0] - target_nll
        fhi = profile_nll(hi)[0] - target_nll
        if flo == 0.0:
            return lo
        if fhi == 0.0:
            return hi
        if flo * fhi > 0.0:
            raise RuntimeError(f"Could not bracket 68% dispersion interval between {lo} and {hi}")
        for _ in range(160):
            mid = 0.5 * (lo + hi)
            fm = profile_nll(mid)[0] - target_nll
            if hi - lo <= 1.0e-10 * max(1.0, hi):
                return mid
            if flo * fm <= 0.0:
                hi = mid
                fhi = fm
            else:
                lo = mid
                flo = fm
        return 0.5 * (lo + hi)

    nll_zero = profile_nll(0.0)[0]
    sigma_lo = 0.0 if nll_zero <= target_nll else bisect_crossing(0.0, sigma_best)

    sigma_hi_bracket = max(1.0, sigma_best * 2.0)
    while profile_nll(sigma_hi_bracket)[0] < target_nll:
        sigma_hi_bracket *= 2.0
        if sigma_hi_bracket > 1.0e6:
            raise RuntimeError("Could not bracket upper 68% dispersion interval")
    sigma_hi = bisect_crossing(sigma_best, sigma_hi_bracket)

    return mu_best, sigma_best, sigma_lo, sigma_hi

def _require(mapping: dict, key: str):
    if key not in mapping:
        raise KeyError(f"Missing required config value: {key}")
    return mapping[key]

def _load_config(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    cwd = str(Path.cwd().resolve())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    if len(path.parents) >= 4:
        repo_candidate = str(path.parents[3])
        if repo_candidate not in sys.path:
            sys.path.insert(0, repo_candidate)

    spec = importlib.util.spec_from_file_location("_ospm_karl_observables_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "CONFIG", None)
    if not isinstance(cfg, dict):
        raise TypeError(f"{path} must define CONFIG as a dict")
    return cfg

def _read_csv_numeric(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        fields = [str(x) for x in reader.fieldnames]
        rows = list(reader)

    data: dict[str, np.ndarray] = {}
    for field in fields:
        values = []
        numeric = True
        for row in rows:
            text = (row.get(field) or "").strip()
            if text == "":
                values.append(float("nan"))
                continue
            try:
                values.append(float(text))
            except ValueError:
                numeric = False
                break
        if numeric:
            data[field] = np.asarray(values, dtype=np.float64)
    return fields, data

def _require_numeric_column(fields: list[str], data: dict[str, np.ndarray], name: str, label: str) -> np.ndarray:
    if name not in fields:
        raise KeyError(f"{label} column {name!r} is not present. Available columns: {fields}")
    if name not in data:
        raise TypeError(f"{label} column {name!r} is not numeric")
    return np.asarray(data[name], dtype=np.float64)

def _read_kinematic_bins(path: Path) -> tuple[np.ndarray, list[dict]]:
    fields, data = _read_csv_numeric(path)
    rin = _require_numeric_column(fields, data, "R_inner_pc", "kinematic-bin inner-radius")
    rout = _require_numeric_column(fields, data, "R_outer_pc", "kinematic-bin outer-radius")
    if len(rin) == 0 or len(rin) != len(rout):
        raise ValueError(f"{path} must contain matching non-empty R_inner_pc/R_outer_pc columns")
    if not np.all(np.isfinite(rin)) or not np.all(np.isfinite(rout)):
        raise ValueError(f"{path} contains non-finite kinematic-bin edges")
    if np.any(rout <= rin):
        raise ValueError(f"{path} requires R_outer_pc > R_inner_pc for every kinematic bin")
    scale = max(1.0, float(np.max(np.abs(np.r_[rin, rout]))))
    tol = 1.0e-10 * scale
    if len(rin) > 1 and np.any(np.abs(rin[1:] - rout[:-1]) > tol):
        raise ValueError(f"{path} kinematic bins must be contiguous to define one radial grid")
    edges = np.concatenate(([float(rin[0])], rout.astype(np.float64)))
    if edges[0] < 0.0 or np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{path} produces an invalid radial edge sequence")

    names = []
    if "bin_id" in data:
        raw_ids = data["bin_id"]
        for i, value in enumerate(raw_ids):
            names.append(f"kinematic_bin_{int(value)}" if math.isfinite(float(value)) else f"kinematic_bin_{i}")
    else:
        names = [f"kinematic_bin_{i}" for i in range(len(rin))]

    apertures = [
        {"id": i + 1, "name": names[i], "ir_start": i + 1, "ir_end": i + 1}
        for i in range(len(rin))
    ]
    return edges, apertures

def _solve_karl_a(rmin_over_rmax: float, nrdat: int) -> float:
    if not (math.isfinite(rmin_over_rmax) and 0.0 < rmin_over_rmax < 1.0):
        raise ValueError(f"radial_rmin/radial_rmax must lie in (0,1); got {rmin_over_rmax}")
    if nrdat <= 1:
        raise ValueError("nrdat must exceed one")

    def f(x: float) -> float:
        return rmin_over_rmax * math.exp(x * nrdat) - rmin_over_rmax - math.exp(x) + CVAL

    lo = 1.0e-4
    hi = 0.9
    flo = f(lo)
    fhi = f(hi)

    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(
            "Karl radial-grid root is not bracketed on [1e-4, 0.9]; "
            f"f(lo)={flo}, f(hi)={fhi}, rmin/rmax={rmin_over_rmax}, nrdat={nrdat}"
        )

    while hi - lo > 1.0e-5:
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0:
            return mid
        if flo * fm <= 0.0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)

@dataclass(frozen=True)
class KarlGridSpec:
    nrdat: int
    nvdat: int
    nrlib: int
    nvlib: int
    radial_rmin_arcsec: float
    radial_rmax_arcsec: float

    @property
    def irrat(self) -> int:
        return self.nrdat // self.nrlib

    @property
    def ivrat(self) -> int:
        return self.nvdat // self.nvlib

class KarlProjectedGrid:
    def __init__(self, spec: KarlGridSpec):
        if spec.nrdat <= 0 or spec.nvdat <= 0 or spec.nrlib <= 0 or spec.nvlib <= 0:
            raise ValueError("Karl grid dimensions must be positive")
        if spec.nrdat % spec.nrlib != 0:
            raise ValueError("nrdat must be divisible by nrlib")
        if spec.nvdat % spec.nvlib != 0:
            raise ValueError("nvdat must be divisible by nvlib")
        if not (0.0 < spec.radial_rmin_arcsec < spec.radial_rmax_arcsec):
            raise ValueError("Require 0 < radial_rmin_arcsec < radial_rmax_arcsec")

        self.spec = spec
        rmin_norm = spec.radial_rmin_arcsec / spec.radial_rmax_arcsec
        self.a = _solve_karl_a(rmin_norm, spec.nrdat)
        self.b = self.a * rmin_norm / (math.exp(self.a) - CVAL)

    def r_nee_ir(self, ir: int) -> float:
        return self.b / self.a * (math.exp(self.a * float(ir)) - CVAL)

    def ir_nee_r(self, r: float) -> int:
        argument = self.a * r / self.b + CVAL
        if not math.isfinite(argument) or argument <= 0.0:
            return 0
        return int(math.log(argument) / self.a + 0.5)

    def v_nee_iv(self, iv: int) -> float:
        return (float(iv - 1) + 0.5) / float(self.spec.nvdat)

    def iv_nee_v(self, v: float) -> int:
        sign = 1 if v >= 0.0 else -1
        fine = min(_fortran_nint(float(self.spec.nvdat) * abs(v) + 0.5), self.spec.nvdat)
        return sign * fine

    def irc_nee_r(self, r: float) -> int:
        return _fortran_int_div(self.ir_nee_r(r) - 1, self.spec.irrat) + 1

    def ivc_nee_v(self, v: float) -> int:
        return _fortran_int_div(self.iv_nee_v(v) - 1, self.spec.ivrat) + 1

    def source_r_center(self, ir: int) -> float:
        return self.r_nee_ir(ir * self.spec.irrat - 1)

    def source_v_center(self, iv: int) -> float:
        return self.v_nee_iv(iv * self.spec.ivrat - 1)

    def radial_edges_internal(self) -> np.ndarray:
        edges = np.zeros(self.spec.nrlib + 1, dtype=np.float64)
        edges[0] = 0.0
        for coarse in range(1, self.spec.nrlib):
            fine_boundary = coarse * self.spec.irrat + 0.5
            edges[coarse] = self.b / self.a * (math.exp(self.a * fine_boundary) - CVAL)
        fine_boundary = self.spec.nrlib * self.spec.irrat + 0.5
        edges[self.spec.nrlib] = self.b / self.a * (math.exp(self.a * fine_boundary) - CVAL)
        return edges

    def angular_edges(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.spec.nvlib + 1, dtype=np.float64)

class DirectProjectedGrid:
    def __init__(self, spec: KarlGridSpec, radial_edges_arcsec: np.ndarray):
        edges = np.asarray(radial_edges_arcsec, dtype=np.float64)
        if spec.nrdat <= 0 or spec.nvdat <= 0 or spec.nrlib <= 0 or spec.nvlib <= 0:
            raise ValueError("Projected grid dimensions must be positive")
        if spec.nrdat % spec.nrlib != 0:
            raise ValueError("nrdat must be divisible by nrlib")
        if spec.nvdat % spec.nvlib != 0:
            raise ValueError("nvdat must be divisible by nvlib")
        if len(edges) != spec.nrlib + 1:
            raise ValueError("Direct projected radial edge count must equal nrlib+1")
        if not np.all(np.isfinite(edges)) or edges[0] < 0.0 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("Direct projected radial edges must be finite and strictly increasing")
        if edges[-1] <= 0.0:
            raise ValueError("Direct projected radial grid must have a positive outer edge")

        self.spec = spec
        self._radial_edges_arcsec = edges
        self._radial_edges_internal = edges / float(edges[-1])
        self.a = math.nan
        self.b = math.nan

    def source_r_center(self, ir: int) -> float:
        lo = float(self._radial_edges_internal[ir - 1])
        hi = float(self._radial_edges_internal[ir])
        return 0.5 * (lo + hi)

    def source_v_center(self, iv: int) -> float:
        edges = self.angular_edges()
        return 0.5 * (float(edges[iv - 1]) + float(edges[iv]))

    def irc_nee_r(self, r: float) -> int:
        if not math.isfinite(r) or r < self._radial_edges_internal[0] or r > self._radial_edges_internal[-1]:
            return 0
        if r == self._radial_edges_internal[-1]:
            return self.spec.nrlib
        idx = int(np.searchsorted(self._radial_edges_internal, r, side="right") - 1)
        return idx + 1 if 0 <= idx < self.spec.nrlib else 0

    def ivc_nee_v(self, v: float) -> int:
        value = abs(float(v))
        edges = self.angular_edges()
        if not math.isfinite(value) or value < edges[0] or value > edges[-1]:
            return 0
        if value == edges[-1]:
            return self.spec.nvlib
        idx = int(np.searchsorted(edges, value, side="right") - 1)
        return idx + 1 if 0 <= idx < self.spec.nvlib else 0

    def radial_edges_internal(self) -> np.ndarray:
        return self._radial_edges_internal.copy()

    def angular_edges(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.spec.nvlib + 1, dtype=np.float64)

def _arcsec_to_pc(arcsec: np.ndarray | float, distance_pc: float) -> np.ndarray | float:
    return distance_pc * np.tan(np.asarray(arcsec) * math.pi / (180.0 * 3600.0))

def _pc_to_arcsec(pc_value: np.ndarray | float, distance_pc: float) -> np.ndarray | float:
    return np.arctan(np.asarray(pc_value) / distance_pc) * (180.0 * 3600.0) / math.pi

def _build_seeing_transfer(grid, seeing_arcsec: float, psf_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if not (math.isfinite(seeing_arcsec) and seeing_arcsec >= 0.0):
        raise ValueError("seeing_arcsec must be finite and nonnegative")
    if psf_samples < 2:
        raise ValueError("psf_samples must be at least two")

    spec = grid.spec
    if seeing_arcsec == 0.0:
        ncell = spec.nrlib * spec.nvlib
        identity = np.eye(ncell, dtype=np.float64).reshape(spec.nrlib, spec.nvlib, spec.nrlib, spec.nvlib)
        return identity, np.zeros_like(identity)

    seeing_internal = seeing_arcsec / spec.radial_rmax_arcsec
    rmaxs = 10.0 * seeing_internal / 2.35
    den = 2.0 * seeing_internal * seeing_internal / (2.35 * 2.35)
    step = 2.0 * rmaxs / float(psf_samples - 1)

    sumb = np.zeros((spec.nrlib, spec.nvlib, spec.nrlib, spec.nvlib), dtype=np.float64)
    sumbn = np.zeros_like(sumb)

    for ir in range(1, spec.nrlib + 1):
        r = grid.source_r_center(ir)
        for iv in range(1, spec.nvlib + 1):
            v = grid.source_v_center(iv)
            x = r * math.sqrt(max(0.0, 1.0 - v * v))
            y = r * v
            xmin = x - rmaxs
            ymin = y - rmaxs
            positive = np.zeros((spec.nrlib, spec.nvlib), dtype=np.float64)
            negative = np.zeros_like(positive)
            sumg = 0.0

            for ix in range(psf_samples):
                xp = xmin + ix * step
                xd = x - xp
                for iy in range(psf_samples):
                    yp = ymin + iy * step
                    yd = y - yp
                    rp = math.hypot(xp, yp)
                    vp = 0.0 if rp == 0.0 else abs(yp / rp)
                    irp = grid.irc_nee_r(rp)
                    ivp = grid.ivc_nee_v(vp)
                    psf = math.exp(-(xd * xd + yd * yd) / den)
                    sumg += psf

                    if 1 <= irp <= spec.nrlib and 1 <= ivp <= spec.nvlib:
                        if xp > 0.0:
                            positive[irp - 1, ivp - 1] += psf
                        else:
                            negative[irp - 1, ivp - 1] += psf

            if not (math.isfinite(sumg) and sumg > 0.0):
                raise RuntimeError(f"Karl getsum normalization failed at ir={ir}, iv={iv}")
            sumb[ir - 1, iv - 1, :, :] = positive / sumg
            sumbn[ir - 1, iv - 1, :, :] = negative / sumg

    return sumb, sumbn

class SurfaceBrightnessProfile:
    def __init__(self, path: Path, radius_col: str, sigma_col: str):
        fields, data = _read_csv_numeric(path)
        R = _require_numeric_column(fields, data, radius_col, "surface-brightness radius")
        S = _require_numeric_column(fields, data, sigma_col, "surface-brightness Sigma")
        good = np.isfinite(R) & np.isfinite(S) & (R > 0.0) & (S > 0.0)
        R = R[good]
        S = S[good]
        if len(R) < 4:
            raise ValueError(f"{path} needs at least four finite positive surface-brightness points")

        order = np.argsort(R)
        self.R = R[order]
        self.S = S[order]
        self.path = path

        dS_dR2 = (self.S[1] - self.S[0]) / (self.R[1] ** 2 - self.R[0] ** 2)
        self.inner_dS_dR2 = min(float(dS_dR2), 0.0)
        self.S_center = float(self.S[0] - self.inner_dS_dR2 * self.R[0] ** 2)

        n_tail = min(6, len(self.R))
        slope = np.polyfit(np.log(self.R[-n_tail:]), np.log(self.S[-n_tail:]), 1)[0]
        self.outer_slope = min(float(slope), 0.0)
        self.outer_zero_pc = 2.0 * float(self.R[-1])

    def __call__(self, radius_pc: np.ndarray) -> np.ndarray:
        Rq = np.asarray(radius_pc, dtype=np.float64)
        out = np.zeros_like(Rq)

        inner = Rq < self.R[0]
        middle = (Rq >= self.R[0]) & (Rq <= self.R[-1])
        outer = (Rq > self.R[-1]) & (Rq < self.outer_zero_pc)

        out[inner] = self.S_center + self.inner_dS_dR2 * Rq[inner] ** 2
        out[inner] = np.maximum(out[inner], 0.0)

        if np.any(middle):
            out[middle] = np.exp(
                np.interp(np.log(Rq[middle]), np.log(self.R), np.log(self.S))
            )

        if np.any(outer):
            rr = Rq[outer]
            t = np.clip((rr - self.R[-1]) / (self.outer_zero_pc - self.R[-1]), 0.0, 1.0)
            taper = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
            powerlaw = self.S[-1] * (rr / self.R[-1]) ** self.outer_slope
            out[outer] = np.maximum(powerlaw * taper, 0.0)

        return out

def _integrate_light_grid(profile: SurfaceBrightnessProfile, radial_edges_pc: np.ndarray, angular_edges: np.ndarray, q: float, quadrature_n: int) -> np.ndarray:
    if not (math.isfinite(q) and q > 0.0):
        raise ValueError("AXIS_RATIO_Q must be finite and positive")
    if quadrature_n < 2:
        raise ValueError("light_quadrature must be at least two")

    nrlib = len(radial_edges_pc) - 1
    nvlib = len(angular_edges) - 1
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_n)
    out = np.zeros((nrlib, nvlib), dtype=np.float64)

    for ir in range(nrlib):
        rlo = float(radial_edges_pc[ir])
        rhi = float(radial_edges_pc[ir + 1])
        rnodes = 0.5 * (rhi - rlo) * nodes + 0.5 * (rhi + rlo)
        rw = 0.5 * (rhi - rlo) * weights

        for iv in range(nvlib):
            tlo = math.asin(float(angular_edges[iv]))
            thi = math.asin(float(angular_edges[iv + 1]))
            tnodes = 0.5 * (thi - tlo) * nodes + 0.5 * (thi + tlo)
            tw = 0.5 * (thi - tlo) * weights

            rr, tt = np.meshgrid(rnodes, tnodes, indexing="ij")
            Rell = rr * np.sqrt(np.cos(tt) ** 2 + (np.sin(tt) / q) ** 2)
            sigma = profile(Rell)
            integrand = sigma * rr
            out[ir, iv] = float(np.sum(integrand * rw[:, None] * tw[None, :]))

    total = float(np.sum(out))
    if not (math.isfinite(total) and total > 0.0):
        raise RuntimeError("Projected Karl light grid has non-positive total")
    out /= total
    return out

def _convolve_spatial(raw: np.ndarray, sumb: np.ndarray, sumbn: np.ndarray) -> np.ndarray:
    transfer = sumb + sumbn
    out = np.zeros_like(raw)
    nrlib, nvlib = raw.shape
    for irb in range(nrlib):
        for ivb in range(nvlib):
            out += transfer[irb, ivb, :, :] * raw[irb, ivb]
    total = float(np.sum(out))
    if not (math.isfinite(total) and total > 0.0):
        raise RuntimeError("Seeing-convolved projected light has non-positive retained total")
    out /= total
    return out

class KarlRan1Gasdev:
    IA = 16807
    IM = 2147483647
    IQ = 127773
    IR = 2836
    NTAB = 32
    NDIV = 1 + (IM - 1) // NTAB
    AM = 1.0 / IM
    EPS = 1.2e-7
    RNMX = 1.0 - EPS

    def __init__(self, idum: int=-1):
        self.idum = int(idum)
        self.iv = [0] * self.NTAB
        self.iy = 0
        self.iset = 0
        self.gset = 0.0

    def ran1(self) -> float:
        if self.idum <= 0 or self.iy == 0:
            self.idum = max(-self.idum, 1)
            for j in range(self.NTAB + 8, 0, -1):
                k = self.idum // self.IQ
                self.idum = self.IA * (self.idum - k * self.IQ) - self.IR * k
                if self.idum < 0:
                    self.idum += self.IM
                if j <= self.NTAB:
                    self.iv[j - 1] = self.idum
            self.iy = self.iv[0]

        k = self.idum // self.IQ
        self.idum = self.IA * (self.idum - k * self.IQ) - self.IR * k
        if self.idum < 0:
            self.idum += self.IM

        j = 1 + self.iy // self.NDIV
        self.iy = self.iv[j - 1]
        self.iv[j - 1] = self.idum
        return min(self.AM * self.iy, self.RNMX)

    def gasdev(self) -> float:
        if self.iset == 0:
            while True:
                v1 = 2.0 * self.ran1() - 1.0
                v2 = 2.0 * self.ran1() - 1.0
                rsq = v1 * v1 + v2 * v2
                if rsq < 1.0 and rsq != 0.0:
                    break
            fac = math.sqrt(-2.0 * math.log(rsq) / rsq)
            self.gset = v1 * fac
            self.iset = 1
            return v2 * fac

        self.iset = 0
        return self.gset

def _karl_medmad(values: np.ndarray) -> tuple[float, float]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = len(x)

    if n < 1:
        return -666.0, -2.0
    if n == 1:
        return float(x[0]), -1.0

    n2 = n // 2
    if n % 2 == 0:
        xmed = 0.5 * (x[n2 - 1] + x[n2])
    else:
        xmed = float(x[n2])

    deviations = np.sort(np.abs(x - xmed))
    if n % 2 == 0:
        xmad = 0.5 * (deviations[n2 - 1] + deviations[n2])
    else:
        xmad = float(deviations[n2])

    return float(xmed), float(xmad)

def _karl_biwgt(values: np.ndarray) -> tuple[float, float]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = len(x)

    if n < 1:
        return -666.0, -2.0
    if n == 1:
        return float(x[0]), -1.0
    if n == 2:
        xbl = 0.5 * float(x[0] + x[1])
        return xbl, abs(float(x[0]) - xbl)

    xmed, xmad = _karl_medmad(x)

    if xmad < 1.0e-6 * max(1.0, abs(xmed)):
        return xmed, xmad

    xbl = xmed
    delta = max(2.0e-5, abs(xmed))
    cmad = 6.0 * xmad
    cmadsq = cmad * cmad
    icnt = 0

    while abs(delta) >= 1.0e-4 * xmad and icnt < 20:
        icnt += 1
        sum1 = 0.0
        sum2 = 0.0

        for value in x:
            t0 = float(value) - xbl
            if abs(t0) < cmad:
                t1 = cmadsq - t0 * t0
                t1 = t1 * t1
                sum1 += t0 * t1
                sum2 += t1

        if sum2 == 0.0:
            break
        delta = sum1 / sum2
        xbl += delta

    sum1 = 0.0
    sum2 = 0.0
    cmad = 9.0 * xmad
    cmadsq = cmad * cmad

    for value in x:
        t0 = float(value) - xbl
        if abs(t0) < cmad:
            t0 = t0 * t0
            t1 = cmadsq - t0
            sum1 += t0 * t1 * t1 * t1 * t1
            sum2 += t1 * (cmadsq - 5.0 * t0)

    if sum2 == 0.0:
        return xbl, 0.0

    xbs = n * math.sqrt(sum1 / (n - 1.0)) / abs(sum2)
    return xbl, xbs

def _mkherm_gaussian_target(sigma_kms: float, sigma_err_kms: float, ntot: int, rng: KarlRan1Gasdev) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (math.isfinite(sigma_kms) and sigma_kms > 0.0):
        raise ValueError("mkherm sigma must be finite and positive")
    if not (math.isfinite(sigma_err_kms) and sigma_err_kms >= 0.0):
        raise ValueError("mkherm sigma error must be finite and nonnegative")
    if ntot < 2:
        raise ValueError("mkherm requires at least two velocity samples")

    vmin = -KARL_LOSVD_SIGMA_EXTENT * sigma_kms
    vmax = KARL_LOSVD_SIGMA_EXTENT * sigma_kms
    vfine = np.linspace(vmin, vmax, ntot, dtype=np.float64)

    gmax = 1.0 / math.sqrt(2.0 * math.pi) / sigma_kms
    econt = KARL_MKHERM_ECONT_FRAC * gmax
    central = np.exp(-0.5 * (vfine / sigma_kms) ** 2) / (math.sqrt(2.0 * math.pi) * sigma_kms)
    lower = np.empty(ntot, dtype=np.float64)
    upper = np.empty(ntot, dtype=np.float64)

    for i, velocity in enumerate(vfine):
        sims = np.empty(KARL_MKHERM_NSIM, dtype=np.float64)
        for isim in range(KARL_MKHERM_NSIM):
            vel_draw = 0.0 + 0.0 * rng.gasdev()
            sigma_draw = sigma_kms + sigma_err_kms * rng.gasdev()
            amp_draw = 1.0 + 0.0 * rng.gasdev()
            h3_draw = 0.0 + 0.0 * rng.gasdev()
            h4_draw = 0.0 + 0.0 * rng.gasdev()
            continuum = econt * rng.gasdev()
            if sigma_draw == 0.0:
                sims[isim] = continuum
            else:
                w = (velocity - vel_draw) / sigma_draw
                sims[isim] = amp_draw * math.exp(-0.5 * w * w) / (math.sqrt(2.0 * math.pi) * sigma_draw) * (1.0 + h3_draw * 0.0 + h4_draw * 0.0) + continuum
        _, xs = _karl_biwgt(sims)
        y = max(0.0, float(central[i]))
        central[i] = y
        upper[i] = y + xs
        lower[i] = max(0.0, y - xs)

    return vfine, central, lower, upper

def _velocity_grid(nvel: int, vmin_center_kms: float, vmax_center_kms: float) -> tuple[np.ndarray, np.ndarray]:
    if nvel <= 1 or not (vmax_center_kms > vmin_center_kms):
        raise ValueError("Invalid model velocity-grid settings")
    centers = np.linspace(vmin_center_kms, vmax_center_kms, nvel, dtype=np.float64)
    width = float(centers[1] - centers[0])
    edges = np.concatenate(([centers[0] - 0.5 * width], centers + 0.5 * width))
    return centers, edges

def _rebin_to_model(vfine: np.ndarray, central: np.ndarray, lower: np.ndarray, upper: np.ndarray, velocity_edges: np.ndarray, aperture_light: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    suma = float(np.sum(central))
    if not (math.isfinite(suma) and suma > 0.0):
        raise RuntimeError("Karl LOSVD normalization is non-positive")

    ad = central * (aperture_light / suma)
    adfer = np.maximum(upper - central, (upper - lower) / 2.0) * (aperture_light / suma)

    ndata = len(vfine)
    v1 = np.empty(ndata, dtype=np.float64)
    v2 = np.empty(ndata, dtype=np.float64)
    v1[0] = vfine[0] - (vfine[1] - vfine[0]) / 2.0
    v2[0] = vfine[0] + (vfine[1] - vfine[0]) / 2.0
    v1[-1] = vfine[-1] - (vfine[-1] - vfine[-2]) / 2.0
    v2[-1] = vfine[-1] + (vfine[-1] - vfine[-2]) / 2.0
    for k in range(1, ndata - 1):
        v1[k] = vfine[k] - (vfine[k] - vfine[k - 1]) / 2.0
        v2[k] = vfine[k] + (vfine[k + 1] - vfine[k]) / 2.0

    nvbin = len(velocity_edges) - 1
    target = np.zeros(nvbin, dtype=np.float64)
    sigma = np.full(nvbin, INVALID_SIGMA_SENTINEL, dtype=np.float64)
    supported = np.zeros(nvbin, dtype=bool)

    for j in range(nvbin):
        vlo = float(velocity_edges[j])
        vhi = float(velocity_edges[j + 1])
        vcenter = 0.5 * (vlo + vhi)
        if vcenter < v1[0] or vcenter > v2[-1]:
            continue

        sum_target = 0.0
        sum_sigma = 0.0
        for k in range(ndata):
            data_width = float(v2[k] - v1[k])
            if data_width <= 0.0:
                continue
            overlap = min(vhi, float(v2[k])) - max(vlo, float(v1[k]))
            if overlap > 0.0:
                frac = min(1.0, overlap / data_width)
                sum_target += frac * float(ad[k])
                sum_sigma += frac * float(adfer[k])
        target[j] = sum_target
        sigma[j] = sum_sigma
        supported[j] = True

    return target, sigma, supported

def _aperture_specs(raw: list, nrlib: int, nvlib: int) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("KARL_OBSERVABLES['apertures'] must be a non-empty list")
    out = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"aperture {idx} must be a dict")
        ap = {
            "id": int(item.get("id", idx)),
            "name": str(item.get("name", f"aperture_{idx}")),
            "ir_start": int(_require(item, "ir_start")),
            "ir_end": int(_require(item, "ir_end")),
            "iv_start": int(item.get("iv_start", 1)),
            "iv_end": int(item.get("iv_end", nvlib)),
        }
        if not (1 <= ap["ir_start"] <= ap["ir_end"] <= nrlib):
            raise ValueError(f"Invalid radial bin range for aperture {idx}: {ap}")
        if not (1 <= ap["iv_start"] <= ap["iv_end"] <= nvlib):
            raise ValueError(f"Invalid angular bin range for aperture {idx}: {ap}")
        out.append(ap)
    return out

def _read_stars(cfg: dict, ko: dict) -> dict[str, np.ndarray]:
    path = Path(_require(cfg, "DATA_CSV"))
    fields, data = _read_csv_numeric(path)
    r_col = str(_require(cfg, "STAR_R_COL"))
    v_col = str(_require(cfg, "STAR_V_COL"))
    verr_col = str(_require(cfg, "STAR_VERR_COL"))
    r = _require_numeric_column(fields, data, r_col, "stellar projected-radius")
    v = _require_numeric_column(fields, data, v_col, "stellar LOS velocity")
    verr = _require_numeric_column(fields, data, verr_col, "stellar LOS velocity-error")

    velocity_frame = str(ko.get("star_velocity_frame", "subtract_systemic")).strip().lower()
    if velocity_frame == "subtract_systemic":
        vsys = float(_require(cfg, "V_SYS_KMS"))
        vrel = v - vsys
    elif velocity_frame == "already_relative":
        vrel = v.copy()
    else:
        raise ValueError("KARL_OBSERVABLES['star_velocity_frame'] must be 'subtract_systemic' or 'already_relative'")

    return {
        "r_pc": r,
        "vrel_kms": vrel,
        "verr_kms": verr,
        "r_pc": data[r_col],
        "vrel_kms": vrel,
        "verr_kms": data[verr_col],
        "path": np.asarray([str(path)], dtype=object),
    }

def _stars_in_aperture(stars: dict[str, np.ndarray], ap: dict, radial_edges_pc: np.ndarray, nvlib: int) -> tuple[np.ndarray, np.ndarray]:
    if ap["iv_start"] != 1 or ap["iv_end"] != nvlib:
        raise ValueError(
            f"Aperture {ap['name']!r} restricts angular bins, but the current generator "
            "only has the universally available projected radius for star membership. "
            "Add projected x/y support before using angularly restricted stellar apertures."
        )

    rlo = float(radial_edges_pc[ap["ir_start"] - 1])
    rhi = float(radial_edges_pc[ap["ir_end"]])
    r = stars["r_pc"]
    v = stars["vrel_kms"]
    e = stars["verr_kms"]
    good = np.isfinite(r) & np.isfinite(v) & np.isfinite(e) & (e > 0.0)
    if ap["ir_end"] == len(radial_edges_pc) - 1:
        good &= (r >= rlo) & (r <= rhi)
    else:
        good &= (r >= rlo) & (r < rhi)
    return v[good], e[good]

CSV_FIELDS = [
    "schema_version", "row_type", "galaxy",
    "distance_pc", "axis_ratio_q", "seeing_arcsec",
    "nrdat", "nvdat", "nrlib", "nvlib", "nvel", "irrat", "ivrat",
    "radial_rmin_arcsec", "radial_rmax_arcsec", "a", "b",
    "radial_grid_mode", "aperture_source", "velocity_grid_mode", "star_velocity_frame", "seeing_mode",
    "source_data_csv", "source_surface_brightness_csv", "source_kinematic_bins_csv",
    "ir", "iv", "r_inner_arcsec", "r_outer_arcsec", "r_inner_pc", "r_outer_pc",
    "vcoord_inner", "vcoord_outer", "light_raw", "light_seen",
    "source_ir", "source_iv", "target_ir", "target_iv", "sumb", "sumbn",
    "aperture_id", "aperture_name", "ir_start", "ir_end", "iv_start", "iv_end",
    "aperture_light", "star_count",
    "aperture_mean_velocity_kms", "aperture_sigma_kms", "aperture_sigma_lo_kms", "aperture_sigma_hi_kms", "aperture_sigma_err_kms",
    "losvd_center_kms", "losvd_shape", "losvd_vmin_kms", "losvd_vmax_kms", "losvd_sigma_extent", "mkherm_nsim", "mkherm_econt_frac", "mkherm_rng",
    "velocity_bin", "velocity_low_kms", "velocity_center_kms", "velocity_high_kms",
    "losvd_target", "losvd_sigma", "losvd_supported",
]

def _blank_row() -> dict:
    return {name: "" for name in CSV_FIELDS}

def build_karl_observables(config_path: Path, output_override: Path | None = None) -> Path:
    cfg = _load_config(config_path)
    ko = _require(cfg, "KARL_OBSERVABLES")
    if not isinstance(ko, dict):
        raise TypeError("CONFIG['KARL_OBSERVABLES'] must be a dict")

    galaxy = str(cfg.get("GALAXY", config_path.resolve().parent.name))
    distance_pc = float(_require(cfg, "DISTANCE_PC"))
    q = float(_require(cfg, "AXIS_RATIO_Q"))
    if not (math.isfinite(distance_pc) and distance_pc > 0.0):
        raise ValueError("DISTANCE_PC must be finite and positive")
    if not (math.isfinite(q) and q > 0.0):
        raise ValueError("AXIS_RATIO_Q must be finite and positive")

    losvd_center_mode = str(_require(ko, "losvd_center_mode")).strip().lower()
    losvd_shape = str(_require(ko, "losvd_shape")).strip().lower()
    if losvd_center_mode != "systemic":
        raise ValueError("Current Karl reconstruction supports losvd_center_mode='systemic'")
    if losvd_shape != "gaussian":
        raise ValueError("Current Karl reconstruction supports losvd_shape='gaussian'")

    sb_path = Path(_require(cfg, "SURFACE_BRIGHTNESS_CSV"))
    star_path = Path(_require(cfg, "DATA_CSV"))
    sb_radius_col = str(_require(ko, "surface_brightness_radius_col"))
    sb_sigma_col = str(_require(ko, "surface_brightness_sigma_col"))
    profile = SurfaceBrightnessProfile(sb_path, sb_radius_col, sb_sigma_col)
    stars = _read_stars(cfg, ko)

    nvdat = int(_require(ko, "nvdat"))
    nvlib = int(_require(ko, "nvlib"))
    if nvdat <= 0 or nvlib <= 0 or nvdat % nvlib != 0:
        raise ValueError("KARL_OBSERVABLES nvdat/nvlib must be positive with nvdat divisible by nvlib")

    radial_grid_mode = str(ko.get("radial_grid_mode", "karl_log")).strip().lower()
    aperture_source = str(ko.get("aperture_source", "explicit")).strip().lower()
    kinematic_bins_path = Path(_require(cfg, "KINEMATIC_BINS_CSV")) if radial_grid_mode == "kinematic_bins" or aperture_source == "kinematic_bins" else None
    kinematic_edges_pc = None
    kinematic_apertures = None
    if kinematic_bins_path is not None:
        kinematic_edges_pc, kinematic_apertures = _read_kinematic_bins(kinematic_bins_path)

    if radial_grid_mode == "karl_log":
        spec = KarlGridSpec(
            nrdat=int(_require(ko, "nrdat")),
            nvdat=nvdat,
            nrlib=int(_require(ko, "nrlib")),
            nvlib=nvlib,
            radial_rmin_arcsec=float(_require(ko, "radial_rmin_arcsec")),
            radial_rmax_arcsec=float(_require(ko, "radial_rmax_arcsec")),
        )
        grid = KarlProjectedGrid(spec)
        radial_edges_internal = grid.radial_edges_internal()
        radial_edges_arcsec = radial_edges_internal * spec.radial_rmax_arcsec
        radial_edges_pc = np.asarray(_arcsec_to_pc(radial_edges_arcsec, distance_pc), dtype=np.float64)
    elif radial_grid_mode == "kinematic_bins":
        if kinematic_edges_pc is None:
            raise RuntimeError("kinematic-bin radial grid was not loaded")
        nrlib = len(kinematic_edges_pc) - 1
        radial_subsample_ratio = int(ko.get("radial_subsample_ratio", 4))
        if radial_subsample_ratio <= 0:
            raise ValueError("radial_subsample_ratio must be positive")
        radial_edges_pc = np.asarray(kinematic_edges_pc, dtype=np.float64)
        radial_edges_arcsec = np.asarray(_pc_to_arcsec(radial_edges_pc, distance_pc), dtype=np.float64)
        spec = KarlGridSpec(
            nrdat=nrlib * radial_subsample_ratio,
            nvdat=nvdat,
            nrlib=nrlib,
            nvlib=nvlib,
            radial_rmin_arcsec=float(radial_edges_arcsec[0]),
            radial_rmax_arcsec=float(radial_edges_arcsec[-1]),
        )
        grid = DirectProjectedGrid(spec, radial_edges_arcsec)
        radial_edges_internal = grid.radial_edges_internal()
    else:
        raise ValueError("KARL_OBSERVABLES['radial_grid_mode'] must be 'karl_log' or 'kinematic_bins'")

    if aperture_source == "explicit":
        apertures = _aperture_specs(_require(ko, "apertures"), spec.nrlib, spec.nvlib)
    elif aperture_source == "kinematic_bins":
        if kinematic_apertures is None:
            raise RuntimeError("kinematic-bin apertures were not loaded")
        if radial_grid_mode != "kinematic_bins":
            raise ValueError("aperture_source='kinematic_bins' requires radial_grid_mode='kinematic_bins'")
        apertures = _aperture_specs(kinematic_apertures, spec.nrlib, spec.nvlib)
    else:
        raise ValueError("KARL_OBSERVABLES['aperture_source'] must be 'explicit' or 'kinematic_bins'")

    nvel = int(_require(ko, "nvel"))
    velocity_grid_mode = str(ko.get("velocity_grid_mode", "fixed_centers")).strip().lower()
    if velocity_grid_mode == "fixed_centers":
        vmin_center_kms = float(_require(ko, "model_vmin_kms"))
        vmax_center_kms = float(_require(ko, "model_vmax_kms"))
    elif velocity_grid_mode == "data_range":
        vrel = np.asarray(stars["vrel_kms"], dtype=np.float64)
        verr = np.asarray(stars["verr_kms"], dtype=np.float64)
        good_velocity = np.isfinite(vrel) & np.isfinite(verr) & (verr > 0.0)
        if np.count_nonzero(good_velocity) < 2:
            raise ValueError("velocity_grid_mode='data_range' needs at least two finite LOS velocities")
        vmin_center_kms = float(np.min(vrel[good_velocity]))
        vmax_center_kms = float(np.max(vrel[good_velocity]))
    else:
        raise ValueError("KARL_OBSERVABLES['velocity_grid_mode'] must be 'fixed_centers' or 'data_range'")
    velocity_centers, velocity_edges = _velocity_grid(nvel, vmin_center_kms, vmax_center_kms)

    seeing_arcsec = float(_require(ko, "seeing_arcsec"))
    if not (math.isfinite(seeing_arcsec) and seeing_arcsec >= 0.0):
        raise ValueError("seeing_arcsec must be finite and nonnegative")
    seeing_mode = "none" if seeing_arcsec == 0.0 else "gaussian_karl"

    psf_samples = int(ko.get("psf_samples", DEFAULT_PSF_SAMPLES))
    light_quadrature = int(ko.get("light_quadrature", DEFAULT_LIGHT_QUADRATURE))
    transvd_samples = int(ko.get("transvd_samples", DEFAULT_TRANSVD_SAMPLES))

    angular_edges = grid.angular_edges()

    raw_light = _integrate_light_grid(profile, radial_edges_pc, angular_edges, q, light_quadrature)
    sumb, sumbn = _build_seeing_transfer(grid, seeing_arcsec, psf_samples)
    seen_light = _convolve_spatial(raw_light, sumb, sumbn)

    aperture_light = []
    aperture_star_count = []
    aperture_velocities = []
    aperture_velocity_errors = []
    aperture_mean_velocity = []
    aperture_sigma = []
    aperture_sigma_lo = []
    aperture_sigma_hi = []
    aperture_sigma_err = []
    aperture_half_width = []
    for ap in apertures:
        light = float(np.sum(
            seen_light[
                ap["ir_start"] - 1:ap["ir_end"],
                ap["iv_start"] - 1:ap["iv_end"],
            ]
        ))
        velocities, velocity_errors = _stars_in_aperture(stars, ap, radial_edges_pc, spec.nvlib)
        if len(velocities) < 2:
            raise ValueError(f"Aperture {ap['name']!r} contains fewer than two valid LOSVD stars")
        mu_ap, sigma_ap, sigma_lo_ap, sigma_hi_ap = _intrinsic_dispersion_mle(velocities, velocity_errors)
        sigma_err_ap = 0.5 * (sigma_hi_ap - sigma_lo_ap)
        half_width = KARL_LOSVD_SIGMA_EXTENT * sigma_ap
        aperture_light.append(light)
        aperture_star_count.append(len(velocities))
        aperture_velocities.append(velocities)
        aperture_velocity_errors.append(velocity_errors)
        aperture_mean_velocity.append(mu_ap)
        aperture_sigma.append(sigma_ap)
        aperture_sigma_lo.append(sigma_lo_ap)
        aperture_sigma_hi.append(sigma_hi_ap)
        aperture_sigma_err.append(sigma_err_ap)
        aperture_half_width.append(half_width)

    target_rows = []
    supported_counts = []
    mkherm_rng = KarlRan1Gasdev(idum=-1)
    for ia, velocities in enumerate(aperture_velocities):
        vfine, central, lower, upper = _mkherm_gaussian_target(
            aperture_sigma[ia], aperture_sigma_err[ia], transvd_samples, mkherm_rng
        )
        target, sigma, supported = _rebin_to_model(
            vfine, central, lower, upper, velocity_edges, aperture_light[ia]
        )
        target_rows.append((target, sigma, supported))
        supported_counts.append(int(np.count_nonzero(supported)))

    common = {
        "schema_version": 1,
        "galaxy": galaxy,
        "distance_pc": f"{distance_pc:.17g}",
        "axis_ratio_q": f"{q:.17g}",
        "seeing_arcsec": f"{seeing_arcsec:.17g}",
        "nrdat": spec.nrdat,
        "nvdat": spec.nvdat,
        "nrlib": spec.nrlib,
        "nvlib": spec.nvlib,
        "nvel": nvel,
        "irrat": spec.irrat,
        "ivrat": spec.ivrat,
        "radial_rmin_arcsec": f"{spec.radial_rmin_arcsec:.17g}",
        "radial_rmax_arcsec": f"{spec.radial_rmax_arcsec:.17g}",
        "a": "" if not math.isfinite(grid.a) else f"{grid.a:.17g}",
        "b": "" if not math.isfinite(grid.b) else f"{grid.b:.17g}",
        "radial_grid_mode": radial_grid_mode,
        "aperture_source": aperture_source,
        "velocity_grid_mode": velocity_grid_mode,
        "star_velocity_frame": str(ko.get("star_velocity_frame", "subtract_systemic")).strip().lower(),
        "seeing_mode": seeing_mode,
        "losvd_center_kms": "0",
        "losvd_shape": losvd_shape,
        "losvd_sigma_extent": f"{KARL_LOSVD_SIGMA_EXTENT:.17g}",
        "mkherm_nsim": KARL_MKHERM_NSIM,
        "mkherm_econt_frac": f"{KARL_MKHERM_ECONT_FRAC:.17g}",
        "mkherm_rng": "ran1+gasdev_cached",
        "source_data_csv": str(star_path),
        "source_surface_brightness_csv": str(sb_path),
        "source_kinematic_bins_csv": "" if kinematic_bins_path is None else str(kinematic_bins_path),
    }

    rows: list[dict] = []

    row = _blank_row()
    row.update(common)
    row["row_type"] = "metadata"
    rows.append(row)

    for ir in range(1, spec.nrlib + 1):
        for iv in range(1, spec.nvlib + 1):
            row = _blank_row()
            row.update(common)
            row.update({
                "row_type": "spatial_cell",
                "ir": ir,
                "iv": iv,
                "r_inner_arcsec": f"{radial_edges_arcsec[ir - 1]:.17g}",
                "r_outer_arcsec": f"{radial_edges_arcsec[ir]:.17g}",
                "r_inner_pc": f"{radial_edges_pc[ir - 1]:.17g}",
                "r_outer_pc": f"{radial_edges_pc[ir]:.17g}",
                "vcoord_inner": f"{angular_edges[iv - 1]:.17g}",
                "vcoord_outer": f"{angular_edges[iv]:.17g}",
                "light_raw": f"{raw_light[ir - 1, iv - 1]:.17g}",
                "light_seen": f"{seen_light[ir - 1, iv - 1]:.17g}",
            })
            rows.append(row)

    for irb in range(1, spec.nrlib + 1):
        for ivb in range(1, spec.nvlib + 1):
            for ir in range(1, spec.nrlib + 1):
                for iv in range(1, spec.nvlib + 1):
                    row = _blank_row()
                    row.update(common)
                    row.update({
                        "row_type": "seeing_transfer",
                        "source_ir": irb,
                        "source_iv": ivb,
                        "target_ir": ir,
                        "target_iv": iv,
                        "sumb": f"{sumb[irb - 1, ivb - 1, ir - 1, iv - 1]:.17g}",
                        "sumbn": f"{sumbn[irb - 1, ivb - 1, ir - 1, iv - 1]:.17g}",
                    })
                    rows.append(row)

    for ia, ap in enumerate(apertures):
        row = _blank_row()
        row.update(common)
        row.update({
            "row_type": "aperture",
            "aperture_id": ap["id"],
            "aperture_name": ap["name"],
            "ir_start": ap["ir_start"],
            "ir_end": ap["ir_end"],
            "iv_start": ap["iv_start"],
            "iv_end": ap["iv_end"],
            "aperture_light": f"{aperture_light[ia]:.17g}",
            "star_count": aperture_star_count[ia],
            "aperture_mean_velocity_kms": f"{aperture_mean_velocity[ia]:.17g}",
            "aperture_sigma_kms": f"{aperture_sigma[ia]:.17g}",
            "aperture_sigma_lo_kms": f"{aperture_sigma_lo[ia]:.17g}",
            "aperture_sigma_hi_kms": f"{aperture_sigma_hi[ia]:.17g}",
            "aperture_sigma_err_kms": f"{aperture_sigma_err[ia]:.17g}",
            "losvd_vmin_kms": f"{-aperture_half_width[ia]:.17g}",
            "losvd_vmax_kms": f"{aperture_half_width[ia]:.17g}",
        })
        rows.append(row)

    for ia, ap in enumerate(apertures):
        target, sigma, supported = target_rows[ia]
        for j in range(nvel):
            row = _blank_row()
            row.update(common)
            row.update({
                "row_type": "losvd_bin",
                "aperture_id": ap["id"],
                "aperture_name": ap["name"],
                "aperture_light": f"{aperture_light[ia]:.17g}",
                "star_count": aperture_star_count[ia],
                "aperture_mean_velocity_kms": f"{aperture_mean_velocity[ia]:.17g}",
                "aperture_sigma_kms": f"{aperture_sigma[ia]:.17g}",
                "aperture_sigma_lo_kms": f"{aperture_sigma_lo[ia]:.17g}",
                "aperture_sigma_hi_kms": f"{aperture_sigma_hi[ia]:.17g}",
                "aperture_sigma_err_kms": f"{aperture_sigma_err[ia]:.17g}",
                "losvd_vmin_kms": f"{-aperture_half_width[ia]:.17g}",
                "losvd_vmax_kms": f"{aperture_half_width[ia]:.17g}",
                "velocity_bin": j + 1,
                "velocity_low_kms": f"{velocity_edges[j]:.17g}",
                "velocity_center_kms": f"{velocity_centers[j]:.17g}",
                "velocity_high_kms": f"{velocity_edges[j + 1]:.17g}",
                "losvd_target": f"{target[j]:.17g}",
                "losvd_sigma": f"{sigma[j]:.17g}",
                "losvd_supported": int(bool(supported[j])),
            })
            rows.append(row)

    output_path = output_override if output_override is not None else Path(_require(ko, "output_csv"))
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    transfer_sum = np.sum(sumb + sumbn, axis=(2, 3))
    negative_sum = np.sum(sumbn, axis=(2, 3))

    print("[KARL OBSERVABLES] generated one CSV")
    print(f"[KARL OBSERVABLES] galaxy={galaxy}")
    print(f"[KARL OBSERVABLES] output={output_path}")
    print(f"[KARL OBSERVABLES] rows={len(rows)}")
    print(f"[KARL OBSERVABLES] spatial_cells={spec.nrlib * spec.nvlib}")
    print(f"[KARL OBSERVABLES] transfer_rows={(spec.nrlib * spec.nvlib) ** 2}")
    print(f"[KARL OBSERVABLES] apertures={len(apertures)}")
    print(f"[KARL OBSERVABLES] losvd_rows={len(apertures) * nvel}")
    print(f"[KARL OBSERVABLES] radial_grid_mode={radial_grid_mode} aperture_source={aperture_source}")
    if math.isfinite(grid.a) and math.isfinite(grid.b):
        print(f"[KARL OBSERVABLES] a={grid.a:.12g} b={grid.b:.12g}")
    print(f"[KARL OBSERVABLES] velocity_grid_mode={velocity_grid_mode} model_centers_kms=[{vmin_center_kms:.12g},{vmax_center_kms:.12g}]")
    print(f"[KARL OBSERVABLES] seeing_mode={seeing_mode} seeing_arcsec={seeing_arcsec:.12g}")
    print(f"[KARL OBSERVABLES] raw_light_sum={raw_light.sum():.16g}")
    print(f"[KARL OBSERVABLES] seen_light_sum={seen_light.sum():.16g}")
    print(f"[KARL OBSERVABLES] transfer_sum_min={transfer_sum.min():.16g}")
    print(f"[KARL OBSERVABLES] transfer_sum_max={transfer_sum.max():.16g}")
    print(f"[KARL OBSERVABLES] negative_transfer_sum_max={negative_sum.max():.16g}")
    print("[KARL OBSERVABLES] aperture_light=" + ",".join(f"{x:.12g}" for x in aperture_light))
    print("[KARL OBSERVABLES] aperture_star_count=" + ",".join(str(x) for x in aperture_star_count))
    print("[KARL OBSERVABLES] aperture_mean_velocity_kms=" + ",".join(f"{x:.12g}" for x in aperture_mean_velocity))
    print("[KARL OBSERVABLES] aperture_sigma_kms=" + ",".join(f"{x:.12g}" for x in aperture_sigma))
    print("[KARL OBSERVABLES] aperture_sigma_lo_kms=" + ",".join(f"{x:.12g}" for x in aperture_sigma_lo))
    print("[KARL OBSERVABLES] aperture_sigma_hi_kms=" + ",".join(f"{x:.12g}" for x in aperture_sigma_hi))
    print("[KARL OBSERVABLES] aperture_sigma_err_kms=" + ",".join(f"{x:.12g}" for x in aperture_sigma_err))
    print("[KARL OBSERVABLES] aperture_losvd_half_width_kms=" + ",".join(f"{x:.12g}" for x in aperture_half_width))
    print(f"[KARL OBSERVABLES] losvd_shape={losvd_shape} losvd_center_mode={losvd_center_mode} mkherm_nsim={KARL_MKHERM_NSIM} mkherm_rng=ran1+gasdev_cached")
    print("[KARL OBSERVABLES] supported_velocity_bins=" + ",".join(str(x) for x in supported_counts))
    print("[KARL OBSERVABLES] radial_edges_arcsec=" + ",".join(f"{x:.6f}" for x in radial_edges_arcsec))
    return output_path

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one galaxy-agnostic Karl observables CSV from an OSPM galaxy config")
    parser.add_argument("--config", required=True, type=Path, help="Galaxy OSPM config defining CONFIG and KARL_OBSERVABLES")
    parser.add_argument("--out", type=Path, default=None, help="Optional output CSV override")
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    build_karl_observables(args.config, args.out)

if __name__ == "__main__":
    main()