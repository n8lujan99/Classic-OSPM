"""
OSPM_Master

Dispatch layer for OSPM.

Karl-style branch:
    - Observables are still built from the stellar catalog.
    - Surface brightness is required and passed into the observable object.
    - Kinematic bin edges may be loaded from a precomputed CSV.
    - The main fit is handled by the Julia batch backend through the daemon.
    - The old stellar solver is not imported here.
"""

from ..Observables.OSPM_Observables_Stellar import OSPMObservablesStellar


def build_observables(config):
    return OSPMObservablesStellar.from_star_table(
        config["DATA_CSV"],
        inclination_deg=config["INCLINATION_DEG"],
        Norbit=config["NORBIT"],
        stellar_model=config.get("STELLAR_MODEL", None),
        surface_brightness_path=config.get("SURFACE_BRIGHTNESS_CSV", None),
        kinematic_bins_path=config.get("KINEMATIC_BINS_CSV", None),
        config=config,
    )


def solve_ospm_theta(theta, obs, *, halo_type="nfw"):
    raise RuntimeError(
        "solve_ospm_theta is disabled in the Karl-style branch. "
        "Use the Julia batch path through OSPM_Daemon.evaluate_batch_theta."
    )