# OSPM_Config_Center — Segue1
#
# PURPOSE:
# Defines everything that is specifically true for Segue 1.
#
# This file tells OSPM:
#   - which observational files belong to Segue 1
#   - what galaxy geometry to use
#   - which potential parameters are allowed
#   - how the stellar light/tracer distribution is represented
#   - where results for this run should be written
#
# Generic settings such as orbit count, entropy strength, solver tolerances,
# runtime behavior, and search machinery belong in OSPM/load_config.py.
#
# OBSERVATIONAL INPUTS:
#   1. Niederste-Ostholt et al. 2009 Fig. 7 digitized number-count profile
#   2. Simon stellar line-of-sight velocity sample
#
# IMPORTANT:
# The 3D tracer-density grid is derived from the observed 2D number-count
# profile. It is another representation of the same data, not an independent
# observational constraint.

from pathlib import Path
from Data.Data_Prep.Data_Paths import build_data_paths


# ---------------------------------------------------------------------------
# DEBUG / PRODUCTION MODE
# ---------------------------------------------------------------------------

# True:
#   Run the explicitly supplied FIXED_THETA model. Useful for testing one
#   potential and inspecting its orbit structure without performing a search.
#
# False:
#   Allow the normal OSPM parameter-search machinery to explore THETA_BOUNDS.
LOCAL_DEBUG = True


# ---------------------------------------------------------------------------
# SEGUE 1 DIRECTORY
# ---------------------------------------------------------------------------

# Find the directory containing this config file.
#
# All Segue 1 input/output paths below are built relative to this location.
# This avoids hard-coding a machine-specific absolute path.
PROFILE_ROOT = Path(__file__).resolve().parent

# Defensive check that the expected galaxy directory actually exists.
if not PROFILE_ROOT.exists(): raise FileNotFoundError(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")


# ---------------------------------------------------------------------------
# MODEL PARAMETERS
# ---------------------------------------------------------------------------

# Theta always has the ordering:
#
#   [v0, r_c, MBH, ML]
#
# v0  = dark-halo velocity scale [km/s]
# r_c = dark-halo core radius [pc]
# MBH = central black-hole mass [Msun]
# ML  = stellar mass-to-light ratio
#
# This model is the expected-BH comparison model.

INITIAL_THETA = [13.320825619743976, 27.64513174995341, 370858.67274967243, 1.433594982833482]

# Current diagnostic model.
# This is the very-large-BH comparison model used to inspect what orbit
# structure allows a roughly 3.93e6 Msun black hole to fit Segue 1.

#INITIAL_THETA = [18.68078541842375, 304.0490742108145, 3932645.9209644324, 3.093423297240227]

# In LOCAL_DEBUG mode, force OSPM to evaluate this exact model.
#
# In production mode this becomes None so the search is free to move through
# parameter space.
#
# .copy() prevents later changes to FIXED_THETA from modifying INITIAL_THETA.
FIXED_THETA = INITIAL_THETA.copy() if LOCAL_DEBUG else None


# ---------------------------------------------------------------------------
# MAIN SEGUE 1 CONFIGURATION
# ---------------------------------------------------------------------------

CONFIG = {
    # -----------------------------------------------------------------------
    # LOCAL DEBUG CONTROL
    # -----------------------------------------------------------------------

    # Passed into the rest of OSPM so the runtime knows whether this is a
    # fixed-model diagnostic run or a normal parameter search.
    "LOCAL_DEBUG": LOCAL_DEBUG,
    "FIXED_THETA": FIXED_THETA,


    # -----------------------------------------------------------------------
    # DARK-HALO MODEL AND MODEL PARAMETER ORDER
    # -----------------------------------------------------------------------

    # Use the nonsingular isothermal halo.
    #
    # "nonsingular" means the halo has a finite-density core rather than a
    # central density cusp/singularity.
    "HALO_TYPE": "nonsingular_isothermal",

    # Describe this halo using its velocity scale v0 and core radius r_c.
    "HALO_PARAMETERIZATION": "v0_rc",

    # Defines the meaning and ordering of every element of theta.
    #
    # IMPORTANT:
    # Python indexes this list starting at 0. Julia indexes arrays starting
    # at 1, but the physical ordering must remain identical everywhere.
    "PARAMETER_NAMES": ["v0", "r_c", "MBH", "ML"],

    # Starting model for the search, or the exact model used during
    # LOCAL_DEBUG through FIXED_THETA.
    "INITIAL_THETA": INITIAL_THETA,


    # -----------------------------------------------------------------------
    # PARAMETER SEARCH BOUNDS
    # -----------------------------------------------------------------------

    # These limits control which gravitational potentials OSPM is allowed
    # to test. They do not directly control the orbit weights.
    #
    # The MBH upper bound was deliberately extended above the 2025 paper
    # range because the newer implementation continued finding lower chi2
    # as MBH approached the old boundary. The larger bound lets us determine
    # whether the fit eventually turns over instead of stopping at the wall.
    #
    # The wide-core experiments previously allowed nearly halo-free,
    # black-hole-dominated solutions, so the halo range is now restricted
    # back toward the 2025 Segue 1 search domain.
    "THETA_BOUNDS": [
        (0.0, 25.0),        # v0 [km/s] — dark-halo velocity scale
        (1.0, 500),         # r_c [pc] — dark-halo core radius
        (0.0, 4e6),         # MBH [Msun] — central black-hole mass
        (0.2, 5.0),         # M/L — stellar mass-to-light ratio
    ],


    # -----------------------------------------------------------------------
    # GALAXY POSITION AND GEOMETRY
    # -----------------------------------------------------------------------

    # Adopted center of Segue 1 on the sky.
    # Stellar sky coordinates can be converted into projected radius from
    # this point.
    "RA0_DEG": 151.7667,
    "DEC0_DEG": 16.0819,

    # Adopted distance to Segue 1.
    # Needed to turn angular separation on the sky into physical projected
    # radius in parsecs.
    "DISTANCE_PC": 23000.0,

    # Position angle of the projected galaxy coordinate system.
    # For q=1 Segue 1 is spherical, so rotating the system on the sky should
    # not change the physical model. This remains here because the same
    # geometry machinery also handles flattened galaxies.
    "PA_DEG": 90.0,

    # Projected/stellar axis ratio.
    # q=1 means spherical.
    "AXIS_RATIO_Q": 1.0,

    # Characteristic observed half-light radius of Segue 1.
    # This is a useful physical scale for diagnostics and any code that needs
    # a characteristic galaxy radius. Trace the consuming function before
    # assuming it directly enters a particular force or fitting calculation.
    "R_HALF_LIGHT_PC": 29.4,

    # Adopted maximum stellar radius used by parts of the data/model setup.
    # IMPORTANT:
    # This does not automatically mean every OSPM calculation ends at 120 pc.
    # Its exact effect depends on the function that reads this setting.
    "R_MAX_STARS_PC": 120.0,

    # Inclination used by the generic axisymmetric geometry machinery.
    # For a spherical q=1 system the physical appearance is independent of
    # inclination, but the parameter remains part of the common machinery.
    "INCLINATION_DEG": 90.0,

    # Bulk line-of-sight velocity of Segue 1.
    # Internal stellar motions must be measured relative to the galaxy's
    # systemic motion rather than relative to 0 km/s.
    # Source: rebuilt Simon+2011 Table 3 stellar sample.
    "V_SYS_KMS": 208.5667270167265,


    # -----------------------------------------------------------------------
    # STELLAR TRACER CONSTRAINT
    # -----------------------------------------------------------------------

    # Select what spatial quantity the orbit weights are required to
    # reproduce.
    #
    # "density_3d":
    #   constrain the orbit mixture using the deprojected 3D tracer density.
    #
    # "projected_light":
    #   constrain the orbit mixture directly against the projected tracer
    #   profile on the sky.
    #
    # IMPORTANT:
    # density_3d is derived from the same observed surface-density profile.
    # It is not an additional independent observation.
    "TRACER_CONSTRAINT_MODE": "density_3d",  # Use "projected_light" to restore the current baseline.


    # -----------------------------------------------------------------------
    # STELLAR LIGHT / MASS MODEL
    # -----------------------------------------------------------------------

    "STELLAR_MODEL": {
        # Use the grid-based Karl-style stellar representation rather than a
        # simple analytic profile.
        "type": "karl_light_grid",

        # Grid used when computing the gravitational force produced by stars.
        #
        # The light distribution in this file is converted into stellar mass
        # through the model M/L parameter.
        "grid_csv": str(PROFILE_ROOT / "Segue1_stellar_force_grid.csv"),

        # 3D tracer-density grid used to tell the orbit-weight solver where
        # the stellar population is supposed to live.
        #
        # This grid constrains the orbital population. It is separate from
        # the job of calculating stellar gravity even though both originate
        # from the same observed stellar profile.
        "tracer_grid_csv": str(PROFILE_ROOT / "Segue1_tracer_density_3d.csv"),

        # Adopted total luminosity used to normalize the stellar light model.
        #
        # Stellar mass is set by approximately:
        #
        #   Mstar = (M/L) * L
        #
        # with the exact distribution supplied by the stellar grid.
        "Ltot": 340.0,

        # Interpret the stellar density grid using cylindrical axisymmetric
        # coordinates (R,z).
        #
        # Segue 1 becomes the spherical special case because q=1.
        "geometry": "axisymmetric_density_grid",

        # Intrinsic stellar axis ratio used by the density-grid machinery.
        #
        # q=1 gives a spherical stellar distribution.
        "q_axis_ratio": 1.0,


        # -------------------------------------------------------------------
        # COLUMN NAMES EXPECTED IN THE STELLAR GRID CSV
        # -------------------------------------------------------------------

        # Cylindrical radius from the symmetry axis [pc].
        "R_cyl_col": "R_cyl_pc",

        # Height above/below the equatorial plane [pc].
        "z_col": "z_pc",

        # Local 3D luminosity density [Lsun / pc^3].
        "nu_col": "nu_Lsun_pc3",

        # Physical volume represented by the grid cell [pc^3].
        "volume_col": "cell_volume_pc3",

        # Total luminosity contained in that grid cell [Lsun].
        #
        # Conceptually this is density times cell volume.
        "luminosity_col": "cell_luminosity_Lsun",


        # -------------------------------------------------------------------
        # STELLAR-FORCE NUMERICAL SETTINGS
        # -------------------------------------------------------------------

        # Small-scale gravitational softening used while evaluating the
        # stellar force.
        #
        # This prevents the discretized grid from creating artificial force
        # spikes when an orbit passes extremely close to a grid element.
        #
        # Because this modifies the force on very small scales, its exact
        # implementation should be checked in the stellar-force function.
        "force_softening_pc": 0.2,

        # Resolution used to construct/evaluate the stellar force grid.
        "force_nR": 96,     # radial/cylindrical sampling
        "force_nZ": 96,     # vertical sampling
        "force_nphi": 32,   # azimuthal sampling around each ring

        # Provenance of the stellar tracer profile used to build these grids.
        "source": "Niederste-Ostholt2009_Fig7_digitized",
    },


    # -----------------------------------------------------------------------
    # INDIVIDUAL STAR CSV COLUMN CONTRACT
    # -----------------------------------------------------------------------

    # These names tell the generic OSPM code which columns in Segue1_stars.csv
    # contain the quantities needed for the stellar kinematic calculation.
    "STAR_R_COL": "r_pc",          # projected stellar radius [pc]
    "STAR_V_COL": "vlos",          # observed line-of-sight velocity [km/s]
    "STAR_VERR_COL": "vlos_err",   # velocity measurement uncertainty [km/s]

    # Additional coordinate/velocity column names used by generic data-prep
    # or harvesting code.
    "RA_COL": "ra_deg",
    "DEC_COL": "dec_deg",
    "VLOS_COL": "vlos",


    # -----------------------------------------------------------------------
    # DATA HARVESTING / QUALITY SETTINGS
    # -----------------------------------------------------------------------

    # Angular search/preprocessing radius around the Segue 1 center.
    "RADIUS_DEG": 0.6,

    # Maximum Gaia RUWE accepted by preprocessing that uses astrometric
    # quality information.
    "RUWE_MAX": 1.4,

    # Minimum parallax signal-to-noise used by the relevant preprocessing
    # or contamination checks.
    "PAR_SNR_MIN": 5.0,


    # -----------------------------------------------------------------------
    # OBSERVED DATA PRODUCTS
    # -----------------------------------------------------------------------

    # Observed projected stellar number-count / surface-density profile.
    #
    # The historical variable name says "surface brightness", but for Segue 1
    # this comes from the Niederste-Ostholt number-count tracer profile.
    "SURFACE_BRIGHTNESS_CSV": str(PROFILE_ROOT / "Segue1_surface_brightness.csv"),

    # Defines the radial bins used to build/compare the LOSVD constraints.
    "KINEMATIC_BINS_CSV": str(PROFILE_ROOT / "Segue1_losvd_bins.csv"),


    # -----------------------------------------------------------------------
    # SEGUE 1 RESOLVED-STAR LOSVD CONSTRUCTION
    # -----------------------------------------------------------------------

    # Reproduce Karl's historical resolved-star Segue 1 LOSVD estimator
    # instead of the current per-star Gaussian-smearing + Dirichlet-error
    # construction.
    #
    # IMPORTANT:
    # This first controlled test keeps the CURRENT Segue 1 radial/kinematic
    # bins above. Only the LOSVD estimator changes. Karl's exact historical
    # s1...s6 spatial grouping can be tested separately afterward.
    "LOSVD_TARGET_MODE": "karl_resolved_stars",

    # ncont1 / adker1 settings recovered from Karl's Segue 1 rnc workflow.
    "KARL_RESOLVED_KDE_GRID": 17,
    "KARL_RESOLVED_KDE_WIDTH_BINS": 3.0,
    "KARL_RESOLVED_VMIN_KMS": -25.0,
    "KARL_RESOLVED_VMAX_KMS": 25.0,
    "KARL_RESOLVED_BOOTSTRAPS": 300,

    # Historical rnc post-processing:
    #
    # if the upper transvd LOSVD envelope falls below 0.003,
    # replace the lower/upper envelope with [0, 0.003].
    "KARL_RESOLVED_ENVELOPE_FLOOR": 0.003,

    # -----------------------------------------------------------------------
    # GALAXY-SCALE NUMERICAL DOMAIN
    # -----------------------------------------------------------------------

    # Maximum physical distance used by the relevant orbit/model-domain code.
    #
    # This is a numerical/modeling boundary rather than the observed stellar
    # radius. Trace the consuming function for its exact role.
    "MAX_DISTANCE": 2e3,


    # -----------------------------------------------------------------------
    # BLACK-HOLE PARAMETER SEARCH SUPPORT
    # -----------------------------------------------------------------------

    # A logarithmic sampler cannot represent MBH=0 because log(0) is
    # undefined.
    #
    # Positive black-hole proposals therefore use this as the finite lower
    # logarithmic scale.
    "MBH_LOG_FLOOR": 1.0e3,

    # Reserve a fraction of black-hole proposals for the exact no-BH case.
    #
    # This allows MBH=0 to remain explicitly testable while positive MBH
    # values can otherwise be sampled logarithmically.
    "MBH_ZERO_FRACTION": 0.10,


    # -----------------------------------------------------------------------
    # STANDARD OSPM PATHS
    # -----------------------------------------------------------------------

    # build_data_paths returns the standard directory/path entries expected
    # by the rest of OSPM. ** inserts those returned key/value pairs directly
    # into this CONFIG dictionary.
    **build_data_paths(PROFILE_ROOT),


    # Individual Segue 1 stars containing projected radius, vlos, and vlos
    # measurement uncertainty.
    "DATA_CSV": str(PROFILE_ROOT / "Segue1_stars.csv"),

    # Output table for this particular run.
    #
    # Each evaluated theta/model is written here along with chi2, solver
    # status, coverage information, and other run diagnostics.
    "CSV_PATH": str(PROFILE_ROOT/"default"/"Segue1-try1-density3d-abel-karl-resolved-losvd.csv"),
}