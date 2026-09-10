# OSPM_Config_Center — Draco
# Galaxy-specific configuration only.
# Shared solver, orbit-library, AI, deck, and runtime defaults come from OSPM/load_config.py.

from pathlib import Path
LOCAL_DEBUG = False
PROFILE_ROOT = Path(__file__).resolve().parent
if not PROFILE_ROOT.exists():
    raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")

KARL_OBSERVABLES_CSV = PROFILE_ROOT / "Draco_karl_observables.csv"

INITIAL_THETA = [100.0, 1800.0, 9.0e5, 1.0]
FIXED_THETA = INITIAL_THETA.copy() if LOCAL_DEBUG else None

V_SYS_KMS = -291.68214888089926

# Hard heliocentric velocity selection applied when constructing the Draco sample.
LOSVD_SELECTION_VMIN_HELIO_KMS = -330.0
LOSVD_SELECTION_VMAX_HELIO_KMS = -250.0

# OSPM LOSVD velocities are systemic-centered, so transform the actual
# observational selection boundaries into the model velocity frame.
LOSVD_VMIN_KMS = LOSVD_SELECTION_VMIN_HELIO_KMS - V_SYS_KMS
LOSVD_VMAX_KMS = LOSVD_SELECTION_VMAX_HELIO_KMS - V_SYS_KMS
LOSVD_NVBIN = 21
VELOCITY_EDGES_MPS = [
    1.0e3 * (LOSVD_VMIN_KMS + i * (LOSVD_VMAX_KMS - LOSVD_VMIN_KMS) / LOSVD_NVBIN)
    for i in range(LOSVD_NVBIN + 1)
]

CONFIG = {
    "LOCAL_DEBUG": LOCAL_DEBUG,
    "FIXED_THETA": FIXED_THETA,

    # Halo and search space
    "HALO_TYPE": "nonsingular_isothermal",
    "HALO_PARAMETERIZATION": "v0_rc",
    "PARAMETER_NAMES": ["v0", "r_c", "MBH", "ML"],
    "INITIAL_THETA": INITIAL_THETA,
    "THETA_BOUNDS": [
        (0.0, 200.0),           # v0 [km/s] — dark-halo velocity scale
        (1.0, 5000.0),          # r_c [pc] — dark-halo core radius
        (0.0, 5.0e6),           # MBH [Msun] — central black-hole mass
        (0.2, 20.0),            # M/L — stellar mass-to-light ratio
    ],

    # Galaxy geometry
    "RA0_DEG": 260.0517,
    "DEC0_DEG": 57.9153,
    "DISTANCE_PC": 76000.0,
    "PA_DEG": 90.0,
    "AXIS_RATIO_Q": 0.69,
    "R_HALF_LIGHT_PC": 221.0,
    "R_MAX_STARS_PC": 1500.0,
    "INCLINATION_DEG": 78.0,
    "V_SYS_KMS": V_SYS_KMS,

    # Intrinsic tracer constraint.
    # Keep the current force/tracer grids fixed while changing only the LOSVD representation.
    "TRACER_CONSTRAINT_MODE": "density_3d",
    "STELLAR_MODEL": {
        "type": "karl_light_grid",
        "grid_csv": str(PROFILE_ROOT / "Draco_stellar_force_grid.csv"),
        "tracer_grid_csv": str(PROFILE_ROOT / "Draco_tracer_density_3d.csv"),
        "Ltot": 2.7e5,
        "geometry": "axisymmetric_density_grid",
        "q_axis_ratio": 0.69,
        "R_cyl_col": "R_cyl_pc",
        "z_col": "z_pc",
        "nu_col": "nu_Lsun_pc3",
        "volume_col": "cell_volume_pc3",
        "luminosity_col": "cell_luminosity_Lsun",
        "force_softening_pc": 0.2,
        "force_nR": 96,
        "force_nZ": 96,
        "force_nphi": 32,
        "source": "Odenkirchen2001",
    },

    "DATA_PREP": {
        "surface_brightness": {
            "input_csv": str(PROFILE_ROOT / "Draco_surface_brightness_source.csv"),
            "input_type": "surface_density_annuli",
            "source": "Odenkirchen_et_al_2001_Table_3",
            "preferred_profile": "S2",
            "radius_type": "elliptical_major_axis",
            "background": 0.0760,
            "background_err": 0.0014,
            "radius_units": "arcmin",
            "rin_col": "rin_arcmin",
            "rout_col": "rout_arcmin",
            "rmid_col": "rm_arcmin",
            "sigma_col": "Sigma_table3_arcmin2",
            "sigma_err_col": "Sigma_table3_err_arcmin2",
        },
        "losvd_bins": {
            "mode": "min_count",
            "min_stars": 20,
            "drop_partial": False,
        },
        "abel_smoothing_target": 9.0,
        "abel_outer_transition_sigma": 5.0,
        "abel_outer_tail_points": 6,
    },

    # Draco data contract
    "STAR_R_COL": "r_pc",
    "STAR_V_COL": "vlos",
    "STAR_VERR_COL": "vlos_err",
    "RA_COL": "ra",
    "DEC_COL": "dec",
    "VLOS_COL": "vlos",

    # Observed products
    "SURFACE_BRIGHTNESS_CSV": str(PROFILE_ROOT / "Draco_surface_brightness.csv"),
    "KINEMATIC_BINS_CSV": str(PROFILE_ROOT / "Draco_losvd_bins.csv"),
    "DATA_CSV": str(PROFILE_ROOT / "Draco_stars.csv"),

    # Active resolved-star LOSVD representation.
    #
    # Patch 2 retains Karl-style radial x velocity LOSVD bins, but interprets
    # the observed hard counts with a multinomial likelihood. The Draco stellar
    # sample was constructed with an explicit heliocentric velocity cut of
    # [-330,-250] km/s, so the model likelihood is conditioned on passing that
    # selection rather than treating probability outside the cut as observed zero counts.
    "LOSVD_TARGET_MODE": "karl_resolved_stars",
    "LOSVD_FIT_STATISTIC": "multinomial",
    "LOSVD_CONDITIONING": "vlos_cut",
    "KARL_DELTA_STATISTIC_ITER_TOL": 0.3,
    "KARL_OBSERVABLES_CSV": str(KARL_OBSERVABLES_CSV),

    # The 21 Karl velocity bins span the actual sample-selection interval after
    # transforming the original heliocentric [-330,-250] km/s cut into the
    # systemic-centered velocity frame used internally by OSPM.
    "NVBIN": LOSVD_NVBIN,
    "VELOCITY_EDGES": VELOCITY_EDGES_MPS,

    # Retained for compatibility/legacy diagnostics. KDE/bootstrap smoothing
    # is not used by the active hard-count multinomial resolved-star likelihood.
    "KARL_RESOLVED_KDE_GRID": 17,
    "KARL_RESOLVED_KDE_WIDTH_BINS": 3.0,
    "KARL_RESOLVED_VMIN_KMS": LOSVD_VMIN_KMS,
    "KARL_RESOLVED_VMAX_KMS": LOSVD_VMAX_KMS,
    "KARL_RESOLVED_BOOTSTRAPS": 300,
    "KARL_RESOLVED_ENVELOPE_FLOOR": 0.003,

    # Galaxy-specific inputs for OSPM/Data_Prep/build_karl_observables.py.
    # Existing Draco kinematic bins are the radial grid and aperture authority.
    # Resolved-star LOS velocities use no seeing convolution.
    # The active Patch 2 velocity grid spans Draco's known selection interval.
    # The consolidated Gaussian CSV remains a legacy/diagnostic product.
    "KARL_OBSERVABLES": {
        "output_csv": str(KARL_OBSERVABLES_CSV),
        "surface_brightness_radius_col": "R_pc",
        "surface_brightness_sigma_col": "Sigma",
        "star_velocity_frame": "subtract_systemic",
        "radial_grid_mode": "kinematic_bins",
        "aperture_source": "kinematic_bins",
        "radial_subsample_ratio": 4,
        "nvdat": 20,
        "nvlib": 5,
        "seeing_arcsec": 0.0,
        "nvel": LOSVD_NVBIN,
        "velocity_grid_mode": "data_range",
        "losvd_center_mode": "systemic",
        "losvd_shape": "gaussian",
    },

    # Draco needs the longer weight solve.
    "OBSERVABLES": {"KARL_MAXITER": 4000},

    # Draco numerical domain
    "MIN_DISTANCE": 1e-6,
    "MAX_DISTANCE": 5e3,
    "POTENTIAL_EXTENT": 10.0,

    # Draco search behavior
    "MBH_LOG_FLOOR": 1.0e3,
    "MBH_ZERO_FRACTION": 0.10,

    # Draco-specific runtime overrides
    "CHUNK_SIZE": 40,
    "CSV_FLUSH_INTERVAL": 10,
    "EVAL_TIMEOUT_S": 1200.0,
    "PEN_SPHERE_STRENGTH": 200,
    "EVAL_VARIANTS": ["full"],

    # Run identity
    "CSV_PATH": str(PROFILE_ROOT / "default" / "draco_patch2_multinomial_integration_test.csv"),
}
