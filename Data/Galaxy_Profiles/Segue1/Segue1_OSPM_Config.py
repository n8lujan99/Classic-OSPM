# OSPM_Config_Center — Segue1
# Galaxy-specific configuration only.
# Shared solver, orbit-library, AI, deck, and runtime defaults come from OSPM/load_config.py.

from pathlib import Path
LOCAL_DEBUG = False
PROFILE_ROOT = Path(__file__).resolve().parent
if not PROFILE_ROOT.exists():
    raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")

KARL_OBSERVABLES_CSV = PROFILE_ROOT / "Segue1_karl_observables.csv"

INITIAL_THETA = [13.320825619743976, 27.64513174995341, 370858.67274967243, 1.433594982833482] # Default

#INITIAL_THETA = [18.68078541842375, 304.0490742108145, 3932645.9209644324, 3.093423297240227] # now fails as it should
#INITIAL_THETA = [13.320825619743976, 27.64513174995341, 1000000.0, 1.433594982833482]

FIXED_THETA = INITIAL_THETA.copy() if LOCAL_DEBUG else None

LOSVD_VMIN_KMS = -42.0
LOSVD_VMAX_KMS = 47.0
LOSVD_NVBIN = 21
VELOCITY_EDGES_MPS = [
    1.0e3 * (LOSVD_VMIN_KMS + i * (LOSVD_VMAX_KMS - LOSVD_VMIN_KMS) / LOSVD_NVBIN)
    for i in range(LOSVD_NVBIN + 1)
]

CONFIG = {
    "LOCAL_DEBUG": LOCAL_DEBUG,
    "FIXED_THETA": FIXED_THETA,

    "HALO_TYPE": "nonsingular_isothermal",
    "HALO_PARAMETERIZATION": "v0_rc",
    "PARAMETER_NAMES": ["v0", "r_c", "MBH", "ML"],
    "INITIAL_THETA": INITIAL_THETA,
    "THETA_BOUNDS": [
        (0.0, 25.0),                    # v0 [km/s] — dark-halo velocity scale
        (1.0, 500),                     # r_c [pc] — dark-halo core radius
        (0.0, 4e6),                     # MBH [Msun] — central black-hole mass
        (0.2, 5.0),                     # M/L — stellar mass-to-light ratio
    ],

    "RA0_DEG": 151.7667,
    "DEC0_DEG": 16.0819,
    "DISTANCE_PC": 23000.0,
    "PA_DEG": 90.0,
    "AXIS_RATIO_Q": 1.0,
    "R_HALF_LIGHT_PC": 29.4,
    "R_MAX_STARS_PC": 120.0,
    "INCLINATION_DEG": 90.0,
    "V_SYS_KMS": 208.5667270167265,

    "TRACER_CONSTRAINT_MODE": "density_3d",
    "STELLAR_MODEL": {
        "type": "karl_light_grid",
        "grid_csv": str(PROFILE_ROOT / "Segue1_stellar_force_grid.csv"),
        "tracer_grid_csv": str(PROFILE_ROOT / "Segue1_tracer_density_3d.csv"),
        "Ltot": 340.0,
        "geometry": "axisymmetric_density_grid",
        "q_axis_ratio": 1.0,
        "R_cyl_col": "R_cyl_pc",
        "z_col": "z_pc",
        "nu_col": "nu_Lsun_pc3",
        "volume_col": "cell_volume_pc3",
        "luminosity_col": "cell_luminosity_Lsun",
        "force_softening_pc": 0.2,
        "force_nR": 96,                 # radial/cylindrical sampling
        "force_nZ": 96,                 # vertical sampling
        "force_nphi": 32,               # azimuthal sampling around each ring
        "source": "Niederste-Ostholt2009_Fig7_digitized",
    },

    "DATA_PREP": {
        "surface_brightness": {
            "input_csv": str(PROFILE_ROOT / "archive" / "Segue1_NO09_digitized_raw_points.csv"),
            "input_type": "fixed_annulus_counts",
            "source": "Niederste-Ostholt_et_al_2009_Fig7_digitized",
            "preferred_profile": "CMD_mask_number_counts",
            "radius_type": "projected_circular_radius",
            "background": 0.0,
            "background_err": 0.0,
            "center_col": "rin_deg",
            "center_units": "deg",
            "count_col": "count_digitized",
            "annulus_width": 0.025,
            "annulus_width_units": "deg",
            "round_counts": True,
            "count_error_model": "poisson",
        },
        "losvd_bins": {
            "mode": "equal_count",
            "n_bins": 4,
        },
        "abel_smoothing_target": 9.0,
        "abel_outer_transition_sigma": 2.0,
        "abel_outer_tail_points": 35,
    },

    "STAR_R_COL": "r_pc",               # projected stellar radius [pc]
    "STAR_V_COL": "vlos",               # observed line-of-sight velocity [km/s]
    "STAR_VERR_COL": "vlos_err",        # velocity measurement uncertainty [km/s]
    "RA_COL": "ra_deg",
    "DEC_COL": "dec_deg",
    "VLOS_COL": "vlos",
    "RADIUS_DEG": 0.6,
    "RUWE_MAX": 1.4,
    "PAR_SNR_MIN": 5.0,

    "SURFACE_BRIGHTNESS_CSV": str(PROFILE_ROOT / "Segue1_surface_brightness.csv"),
    "KINEMATIC_BINS_CSV": str(PROFILE_ROOT / "Segue1_losvd_bins.csv"),

    # Active resolved-star LOSVD representation.
    "LOSVD_TARGET_MODE": "karl_resolved_stars",
    "LOSVD_FIT_STATISTIC": "multinomial",
    "LOSVD_CONDITIONING": "vlos_cut",
    "KARL_OBSERVABLES_CSV": str(KARL_OBSERVABLES_CSV),
    "KARL_DELTA_STATISTIC_ITER_TOL": 0.3,

    # Temporary Segue 1 resolved-star velocity window for Patch 2 integration testing.
    # Retain the existing 21-bin [-42,+47] km/s window so this test changes only
    # the statistical treatment. This is NOT yet adopted as the scientifically
    # verified Segue 1 membership-selection boundary.
    "NVBIN": LOSVD_NVBIN,
    "VELOCITY_EDGES": VELOCITY_EDGES_MPS,

    # Retained for compatibility; KDE/bootstrap settings are not used by the
    # active hard-count multinomial karl_resolved_stars likelihood.
    "KARL_RESOLVED_KDE_GRID": 17,
    "KARL_RESOLVED_KDE_WIDTH_BINS": 3.0,
    "KARL_RESOLVED_VMIN_KMS": LOSVD_VMIN_KMS,
    "KARL_RESOLVED_VMAX_KMS": LOSVD_VMAX_KMS,
    "KARL_RESOLVED_BOOTSTRAPS": 300,
    "KARL_RESOLVED_ENVELOPE_FLOOR": 0.003,

    # Galaxy-specific inputs for the generic Karl-observables generator.
    # This consolidated Gaussian CSV remains available as a legacy/diagnostic
    # product; the active OSPM target is karl_resolved_stars above.
    "KARL_OBSERVABLES": {
        "output_csv": str(KARL_OBSERVABLES_CSV),
        "surface_brightness_radius_col": "R_pc",
        "surface_brightness_sigma_col": "Sigma",
        "nrdat": 80,
        "nvdat": 20,
        "nrlib": 20,
        "nvlib": 5,
        "radial_rmin_arcsec": 1.0,
        "radial_rmax_arcsec": 11386.65,
        "seeing_arcsec": 1.5,
        "nvel": LOSVD_NVBIN,
        "model_vmin_kms": LOSVD_VMIN_KMS,
        "model_vmax_kms": LOSVD_VMAX_KMS,
        "losvd_center_mode": "systemic",
        "losvd_shape": "gaussian",
        "apertures": [
            {"name": "aperture_1", "ir_start": 1, "ir_end": 6, "iv_start": 1, "iv_end": 5},
            {"name": "aperture_2", "ir_start": 7, "ir_end": 8, "iv_start": 1, "iv_end": 5},
            {"name": "aperture_3", "ir_start": 9, "ir_end": 9, "iv_start": 1, "iv_end": 5},
            {"name": "aperture_4", "ir_start": 10, "ir_end": 11, "iv_start": 1, "iv_end": 5},
            {"name": "aperture_5", "ir_start": 11, "ir_end": 12, "iv_start": 1, "iv_end": 5},
        ],
    },

    "MAX_DISTANCE": 2e3,
    "MBH_LOG_FLOOR": 1.0e3,
    "MBH_ZERO_FRACTION": 0.10,
    "DATA_CSV": str(PROFILE_ROOT / "Segue1_stars.csv"),

    # Keep Patch 2 integration output isolated from all legacy chi-square decks.
    "CSV_PATH": str(PROFILE_ROOT / "default" / "segue1_patch2_multinomial_integration_test.csv"),
}
