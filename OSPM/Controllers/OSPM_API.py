# OSPM_API.py
# STAYS in PYTHON
# NO Variables defined here

import numpy as np
from ..AI.OSPM_Daemon import run_daemon

class OSPM_API:
    def __init__(self, config):
        self.config = config
        self.physics_engine = None

        seed = config.get(
            "RANDOM_SEED_EFFECTIVE",
            config.get("RANDOM_SEED", None)
        )

        if seed is not None:
            np.random.seed(seed)
            import torch
            torch.manual_seed(seed)

    def set_physics_engine(self, physics_engine):
        self.physics_engine = physics_engine

    def run(self):
        if self.physics_engine is None:
            raise RuntimeError("Physics engine not set.")
        return run_daemon(self.config, self.physics_engine)
