import os
import sys

# Must be set before importing the OSPM/Julia bridge.
os.environ["JULIA_NUM_THREADS"] = "auto"
os.environ.setdefault("PYTHON_JULIACALL_PROJECT", os.getcwd())
os.environ.setdefault("PYTHON_JULIACALL_EXE", os.path.expanduser("~/.juliaup/bin/julia"))
os.environ.setdefault("OSPM_USE_JULIA", "1")

import datetime
from OSPM.load_config import load_config
from ..Observables.OSPM_Observables_Stellar import OSPMObservablesStellar
from ..Physics.OSPM_PhysicsEngine import wrap_physics_engine

def build_runtime(ctrl):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    wid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    rt = dict(ctrl)
    rt["RUN_ID"] = f"{ts}_pid{pid}" + (f"_w{wid}" if wid else "")
    rt["WORKER_ID"] = wid
    rt["RANDOM_SEED_EFFECTIVE"] = rt.get("RANDOM_SEED", 123456789) + wid
    return rt

def build_observables(config):
    return OSPMObservablesStellar.from_star_table(config["DATA_CSV"], inclination_deg=config["INCLINATION_DEG"], Norbit=config["NORBIT"], 
        stellar_model=config.get("STELLAR_MODEL", None), surface_brightness_path=config.get("SURFACE_BRIGHTNESS_CSV", None), 
        kinematic_bins_path=config.get("KINEMATIC_BINS_CSV", None), config=config)

def build_physics_engine(config):
    obs = build_observables(config)
    print("torch imported?", "torch" in sys.modules)
    print("obs type:", type(obs))
    print("has R_star_m:", hasattr(obs, "R_star_m"))
    print("has v_star_mps:", hasattr(obs, "v_star_mps"))
    print("has verr_star_mps:", hasattr(obs, "verr_star_mps"))
    print("has surface_brightness_profile:", hasattr(obs, "surface_brightness_profile"))
    print("has kinematic_bin_edges_pc:", hasattr(obs, "kinematic_bin_edges_pc"))
    def base_engine(theta, *, return_A=False, **_ignored):
        raise RuntimeError("Serial base_engine is disabled in the Karl-style branch." "Use daemon batch mode through Julia evaluate_batch_theta.")
    return wrap_physics_engine( base_engine, obs=obs, halo_type=config["HALO_TYPE"], config=config)

def main():
    config = load_config()
    runtime = build_runtime(config)
    from ..Physics import OSPM_Physics as P
    P._jl_init()
    from juliacall import Main
    print("Julia threads seen by module:", Main.OSPMPhysicsSpherical.NTHREADS)
    physics_engine = build_physics_engine(runtime)
    print("torch imported after build?", "torch" in sys.modules)
    # Preserve the old OSPM_API import/initialization order:
    # numpy -> daemon -> random seeding / torch.
    import numpy as np
    from ..AI.OSPM_Daemon import run_daemon
    seed = runtime.get("RANDOM_SEED_EFFECTIVE", runtime.get("RANDOM_SEED", None))
    if seed is not None:
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
    if physics_engine is None:
        raise RuntimeError("Physics engine not set.")
    result = run_daemon(runtime, physics_engine)
    print(result)
    
if __name__ == "__main__":
    main()
