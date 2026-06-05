import os
import sys

os.environ["JULIA_NUM_THREADS"] = "auto"  # Must be first, before ANY other imports
os.environ.setdefault("PYTHON_JULIACALL_PROJECT", os.getcwd())
os.environ.setdefault("PYTHON_JULIACALL_EXE", os.path.expanduser("~/.juliaup/bin/julia"))
os.environ.setdefault("OSPM_USE_JULIA", "1")

from OSPM.load_config import load_config
from .OSPM_Control import build_runtime
from .OSPM_MASTER import build_observables
from ..Physics.OSPM_PhysicsEngine import wrap_physics_engine

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
        raise RuntimeError(
            "Serial base_engine is disabled in the Karl-style branch. "
            "Use daemon batch mode through Julia evaluate_batch_theta."
        )

    return wrap_physics_engine(
        base_engine,
        obs=obs,
        halo_type=config["HALO_TYPE"],
        config=config,
    )


def main():
    config = load_config()
    runtime = build_runtime(config)

    from ..Physics import OSPM_Physics as P

    P._jl_init()

    from juliacall import Main

    print("Julia threads seen by module:", Main.OSPMPhysicsSpherical.NTHREADS)

    physics_engine = build_physics_engine(runtime)

    print("torch imported after build?", "torch" in sys.modules)

    from .OSPM_API import OSPM_API

    api = OSPM_API(runtime)
    api.set_physics_engine(physics_engine)

    result = api.run()
    print(result)


if __name__ == "__main__":
    main()