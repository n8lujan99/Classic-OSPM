================================================================================

OSPM_V2 REPOSITORY INDEX
PURPOSE
================================================================================

This file is a human-readable and machine-indexable map of the OSPM_V2
repository.

It is intended to answer questions such as:

Where does a particular part of the pipeline live?
Which files contain galaxy-specific data?
Which files control runs?
Where does the Julia physics happen?
Where are model outputs stored?
Which files are active pipeline code versus references or archives?
Where should a particular function or physical operation be searched for?

This top-level index stays intentionally broad.

Large source files and directories should receive their own detailed indexes
rather than making this file excessively long.

================================================================================
REPOSITORY ROOT
================================================================================

PATH: ~/research/OSPM_V2/

ROOT CONTENTS: Data | Lit | OSPM | logs | plots | utils | which_galaxy |
Manifest.toml | Project.toml | OSPM_FORTRAN_CODE.txt

Data/
Galaxy data, preprocessing machinery, generated galaxy products, and
cluster-related data support.

Lit/
Literature and reference material.

OSPM/
Main modeling pipeline.

logs/
Runtime and diagnostic logs.

plots/
User-facing plotting and analysis scripts.

utils/
Shell scripts and other utilities used to launch, inspect, or manage OSPM.

which_galaxy
Selects or records the active galaxy.

Project.toml
Julia project environment definition.

Manifest.toml
Locked Julia dependency environment.

OSPM_FORTRAN_CODE.txt
Additional Fortran reference material.

# DATA

PATH: Data/

CONTENTS: Data_Prep | Galaxy_Profiles | slurm

ROLE: All galaxy-specific observational inputs, prepared model inputs,
derived stellar products, data-preparation machinery, and cluster-related
data support.



                         ┌─────────────────────────────┐
Segue1_stars.csv ───────►│                             │
Segue1_surface_brightness│ Karl observables generator │
Segue1 config ──────────►│                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         Segue1_karl_observables.csv
                                        │
                                        │ LOSVD targets
                                        │ seeing
                                        │ apertures
                                        │ velocity support
                                        │
Segue1_stellar_force_grid.csv ──► stellar gravity
Segue1_tracer_density_3d.csv ───► tracer constraint
                                        │
θ=[v0,r_c,MBH,ML] ─────────────────────┤
                                        ▼
                                  TOTAL POTENTIAL
                                        │
                                        ▼
                                   ORBIT LIBRARY
                                      /      \
                                     /        \
                                    ▼          ▼
                              phase volume   A matrices
                                  │          /       \
                                  ▼         ▼         ▼
                               wphase    A_light   A_losvd
                                  \         |         /
                                   \        |        /
                                    └──── WEIGHTS ──┘
                                            │
                                            ▼
                                      model LOSVD
                                      model tracer
                                      entropy
                                      χ²
                                      diagnostics
                                            │
                                            ▼
                              Segue1-try1-....csv
                                            │
                                            ▼
                                   SCIENCE ANALYSIS


================================================================================
DATA ORGANIZATION
================================================================================


The most important directory is:

Data/Galaxy_Profiles/<galaxy>/

Each galaxy directory contains the information needed to connect real
observational data to the shared OSPM modeling code.

The same basic data roles are used for each galaxy.

CORE GALAXY FILE TYPES:

CONFIG                     Galaxy-specific OSPM setup and file paths
STARS                      Individual stellar kinematic measurements
LOSVD BINS                 Radial organization of the kinematic constraints
SURFACE BRIGHTNESS         Observed 2D projected stellar distribution
3D TRACER DENSITY          Intrinsic 3D stellar tracer distribution
STELLAR FORCE GRID         Stellar gravitational contribution
CENTER                     Adopted galaxy center
DEFAULT                    Primary run outputs
PLOTS                      Galaxy-specific diagnostic plots
ARCHIVE                    Old or inactive material

================================================================================
GALAXY DIRECTORY
================================================================================


GENERAL PATH:

Data/Galaxy_Profiles/<galaxy>/

STANDARDIZED FILE PATTERN:

<Galaxy>_OSPM_Config.py <Galaxy>_stars.csv <Galaxy>_losvd_bins.csv <Galaxy>_surface_brightness.csv <Galaxy>_tracer_density_3d.csv <Galaxy>_stellar_force_grid.csv
center.txt
default/
plots/
archive/

KNOWN GALAXIES:

Carina | Draco | LMC | Segue1

================================================================================
GALAXY CONFIG
================================================================================


GENERAL FILE:

<Galaxy>_OSPM_Config.py

EXAMPLES:

Data/Galaxy_Profiles/Segue1/Segue1_OSPM_Config.py
Data/Galaxy_Profiles/Draco/Draco_OSPM_Config.py

TYPE: CONFIGURATION
STATUS: ACTIVE

## PURPOSE

The galaxy config tells the shared OSPM code what data belongs to the
selected galaxy and how that galaxy should be interpreted.

The config does not contain the full shared OSPM solver configuration.

Shared solver, orbit-library, runtime, and AI defaults are primarily supplied
through:

OSPM/load_config.py
OSPM/AI_defaults.py

The galaxy config supplies galaxy-specific information on top of those
shared defaults.

## CONFIG LOADING

The active galaxy is selected through:

OSPM_GALAXY

or:

which_galaxy

OSPM/load_config.py then identifies:

Data/Galaxy_Profiles/<galaxy>/<Galaxy>_OSPM_Config.py

and combines the galaxy configuration with the shared OSPM configuration.

## IMPORTANT DATA PATH CONFIG VALUES

DATA_CSV
Individual stellar measurements.

Typical target:

<Galaxy>_stars.csv

SURFACE_BRIGHTNESS_CSV
Observed projected stellar-density / surface-brightness profile.

Typical target:

<Galaxy>_surface_brightness.csv

KINEMATIC_BINS_CSV
Definition of the radial bins used for the LOSVD / stellar kinematic
constraints.

Typical target:

<Galaxy>_losvd_bins.csv

STELLAR_MODEL.grid_csv
Stellar mass / force-grid source used to represent the gravitational
contribution of the stars.

Typical standardized target:

<Galaxy>_stellar_force_grid.csv

STELLAR_MODEL.tracer_grid_csv
Intrinsic 3D tracer-density grid used to constrain the stellar orbit
population when the tracer constraint operates in 3D.

Typical target:

<Galaxy>_tracer_density_3d.csv

CSV_PATH
Destination for the main model-result CSV.

Typical location:

Data/Galaxy_Profiles/<galaxy>/default/

## IMPORTANT DISTINCTION

STELLAR_MODEL.grid_csv and STELLAR_MODEL.tracer_grid_csv do NOT represent
the same physical thing.

grid_csv:
Used for the stellar gravitational contribution.

tracer_grid_csv:
Used to describe where the observed stellar tracer population exists in
three dimensions.

In physical terms:

STELLAR FORCE GRID = how the stars pull

3D TRACER GRID = where the stars being modeled are distributed

These can originate from the same observed light distribution but they
serve separate roles inside OSPM.

## TRACER CONSTRAINT MODE

Known active mode:

TRACER_CONSTRAINT_MODE = density_3d

In this mode the intrinsic stellar-density constraint comes from:

STELLAR_MODEL.tracer_grid_csv

The stellar-force information remains separately associated with:

STELLAR_MODEL.grid_csv

## OTHER IMPORTANT GALAXY CONFIG INFORMATION

Galaxy configs can also define information such as:

stellar catalog column names
distance
geometry
axis ratio
stellar mass-to-light treatment
halo geometry
model parameter bounds
initial parameter guesses
tracer mode
data paths
output paths
galaxy-specific physical constants

Known stellar-column configuration concepts include:

STAR_R_COL
STAR_V_COL
STAR_VERR_COL

These identify the columns corresponding to:

projected stellar radius
line-of-sight velocity
line-of-sight velocity uncertainty

SEARCH TERMS: config | galaxy config | DATA_CSV | SURFACE_BRIGHTNESS_CSV |
KINEMATIC_BINS_CSV | CSV_PATH | STELLAR_MODEL | grid_csv |
tracer_grid_csv | TRACER_CONSTRAINT_MODE

================================================================================
STARS CSV
================================================================================


GENERAL FILE:

<Galaxy>_stars.csv

EXAMPLE:

Data/Galaxy_Profiles/Segue1/Segue1_stars.csv

TYPE: OBSERVATIONAL / PREPARED STELLAR DATA

## PHYSICAL MEANING

One row corresponds approximately to one observed star in the prepared
stellar sample.

This is the star-by-star kinematic information from which the LOSVD
constraints are constructed.

## IMPORTANT INFORMATION

The relevant columns normally include quantities representing:

projected radius
line-of-sight velocity
velocity measurement error

Depending on the galaxy and data source the file may also contain:

RA
Dec
projected x and y coordinates
proper motion information
parallax information
quality flags
source identifiers
membership-related quantities
preprocessing diagnostics

## CORE OSPM QUANTITIES

Conceptually OSPM needs:

R
Projected distance of the star from the galaxy center.

vlos
Measured line-of-sight velocity.

vlos_err
Measurement uncertainty on that velocity.

## PHYSICAL ROLE

These stars carry the observed kinematic information.

They tell OSPM how the stellar velocities are distributed as a function of
projected position.

## RELATION TO LOSVD

The individual stars are organized into radial kinematic bins.

Within each radial region their measured velocities and uncertainties define
the observed LOSVD constraint.

The star CSV is therefore the underlying stellar kinematic sample.

The LOSVD-bin CSV describes how that sample is divided spatially.

## NOT THE SAME AS

The star CSV is NOT the surface-brightness profile.

It is NOT the 3D tracer-density grid.

It is NOT the stellar gravitational-force grid.

SEARCH TERMS: stars | stellar catalog | vlos | velocity | velocity error |
projected radius | kinematics | individual stars

================================================================================
LOSVD BINS CSV
================================================================================


GENERAL FILE:

<Galaxy>_losvd_bins.csv

EXAMPLE:

Data/Galaxy_Profiles/Segue1/Segue1_losvd_bins.csv

TYPE: GENERATED / PREPARED KINEMATIC DATA PRODUCT

## LOSVD

LOSVD = Line-Of-Sight Velocity Distribution

## PHYSICAL MEANING

The galaxy is divided into projected radial regions.

For each radial region, the velocities of the observed stars define a
line-of-sight velocity distribution.

The model does the same thing with its orbit library.

Observed LOSVD:

real stellar velocities in a radial region

Model LOSVD:

weighted orbit velocities projected into the same radial region

The comparison between those distributions provides the primary kinematic
constraint on the orbit weights.

## WHAT THE LOSVD-BIN FILE DEFINES

Typical information includes:

bin identifier
inner projected radius
outer projected radius
representative or midpoint radius
number of stars in the bin

Conceptually:

bin_id
R_inner_pc
R_outer_pc
R_mid_pc
N_vlos

The exact columns should be recorded when each galaxy file is indexed.

## IMPORTANT DISTINCTION

This CSV primarily defines the RADIAL KINEMATIC BINS.

It should not be confused with the internal velocity bins used to represent
the shape of each LOSVD.

There are two different binning operations:

1. RADIAL BINNING
   Which projected region of the galaxy a star belongs to.

2. VELOCITY BINNING
   Where a velocity falls inside the LOSVD for that radial region.

The galaxy LOSVD-bin CSV describes the first.

## PHYSICAL ROLE

The LOSVD constraints tell the model about stellar motion.

This is what strongly constrains the gravitational potential through the
stellar dynamics.

SEARCH TERMS: LOSVD | kinematic bins | velocity distribution |
radial bins | R_inner_pc | R_outer_pc | N_vlos | velocity constraints

================================================================================
SURFACE BRIGHTNESS CSV
================================================================================


GENERAL FILE:

<Galaxy>_surface_brightness.csv

EXAMPLE:

Data/Galaxy_Profiles/Segue1/Segue1_surface_brightness.csv

TYPE: OBSERVATIONAL / PREPARED TRACER DATA

## PHYSICAL MEANING

Describes the projected distribution of the galaxy's observed stellar
tracer population on the sky.

This answers:

How much stellar tracer density is seen at projected radius R?

Depending on the source data, the measured quantity may literally be a
surface brightness or may instead be a stellar number surface density.

OSPM uses it as the projected tracer distribution.

## TYPICAL INFORMATION

projected radius
radial-bin boundaries
surface density or surface brightness
measurement uncertainty
fraction of total tracer light or counts

Relevant concepts include:

R
Sigma
Sigma_error
light_frac

## PHYSICAL ROLE

The surface-brightness profile describes where the observed stars are
located in projection.

It constrains the spatial distribution of the model stellar population.

## IT DOES NOT DIRECTLY PROVIDE

stellar velocities

dark matter gravity

black-hole gravity

orbit weights

stellar gravitational force

## RELATION TO THE 3D TRACER GRID

The surface-brightness profile is a 2D projected observable.

Conceptually:

observed sky
-> projected surface brightness Sigma(R)

The intrinsic tracer-density product converts that information into a
three-dimensional stellar distribution:

projected Sigma
-> deprojection / geometry
-> intrinsic tracer density

The resulting file is:

<Galaxy>_tracer_density_3d.csv

SEARCH TERMS: surface brightness | surface density | Sigma | projected |
stellar profile | light profile | tracer profile | light fraction

================================================================================
3D TRACER DENSITY CSV
================================================================================


GENERAL FILE:

<Galaxy>_tracer_density_3d.csv

EXAMPLE:

Data/Galaxy_Profiles/Segue1/Segue1_tracer_density_3d.csv

TYPE: GENERATED PHYSICAL DATA PRODUCT

CONFIG CONNECTION:

STELLAR_MODEL.tracer_grid_csv

KNOWN MODE:

TRACER_CONSTRAINT_MODE = density_3d

## PHYSICAL MEANING

Represents the intrinsic three-dimensional distribution of the stellar
tracer population.

The surface-brightness profile tells us what the stars look like after
projection onto the sky.

The tracer-density grid represents the corresponding stellar distribution
inside the galaxy.

## TYPICAL COORDINATES

Depending on geometry, quantities can include:

shell_id
theta_id
R_cyl_pc
z_pc
r_pc
theta_rad

## TYPICAL TRACER QUANTITIES

nu
Intrinsic stellar tracer density.

cell_volume
Volume represented by a grid cell.

cell_luminosity or equivalent tracer count
Amount of tracer population represented by the cell.

light_frac
Fraction of the total tracer distribution represented by that cell or
region.

Lenc_frac
Enclosed tracer fraction when present.

## AXISYMMETRIC EXAMPLE

For an axisymmetric tracer model the density is represented over coordinates
such as:

R_cyl
z

or equivalently:

r
theta

This allows OSPM to constrain the orbit population in three-dimensional
space rather than only matching a projected radial profile.

## PHYSICAL ROLE

The orbit weights must combine to reproduce this intrinsic tracer
distribution.

This controls WHERE the stellar population represented by the orbit library
is allowed to live.

## VERY IMPORTANT

The 3D tracer-density file does NOT determine the gravitational force from
the stars.

That is the job of the stellar force grid.

SEARCH TERMS: tracer density | density_3d | nu | intrinsic density |
3D stellar distribution | R_cyl | z | shell | theta | light fraction

================================================================================
STELLAR FORCE GRID CSV
================================================================================


GENERAL FILE:

<Galaxy>_stellar_force_grid.csv

EXAMPLE:

Data/Galaxy_Profiles/Segue1/Segue1_stellar_force_grid.csv

TYPE: GENERATED PHYSICAL DATA PRODUCT

CONFIG CONNECTION:

STELLAR_MODEL.grid_csv

## PHYSICAL MEANING

Represents the gravitational field generated by the stellar mass
distribution.

The observed stellar light distribution is assigned a stellar mass through
the model mass-to-light ratio.

That stellar mass contributes to the gravitational acceleration experienced
by every integrated orbit.

## CONCEPTUAL FLOW

observed stellar distribution
-> intrinsic stellar distribution
-> stellar mass model
-> gravitational field / force grid
-> orbit acceleration

## MODEL CONNECTION

The stellar contribution depends on the stellar mass-to-light parameter:

ML

Changing ML changes the strength of the stellar gravitational contribution.

## USED BY

The Julia force / orbit machinery ultimately uses this information while
calculating stellar gravity.

Relevant Julia file:

OSPM/Physics/OSPM_Physics_Force.jl

## PHYSICAL ROLE

The force grid answers:

How strongly do the stars gravitationally pull at this position?

The tracer grid answers a different question:

How much of the observed stellar population should exist at this position?

Those two concepts must remain separate.

SEARCH TERMS: stellar force | force grid | stellar gravity | acceleration |
ML | mass-to-light | grid_csv

================================================================================
CENTER.TXT
================================================================================


GENERAL FILE:

center.txt

TYPE: GALAXY REFERENCE / CONFIGURATION DATA

## PHYSICAL MEANING

Stores the adopted galaxy center used when constructing projected stellar
coordinates and radii.

## WHY IT MATTERS

Observed sky coordinates must be converted into coordinates relative to a
chosen galaxy center.

That center affects:

projected x
projected y
projected radius R

Those quantities then affect:

stellar radial positions
kinematic-bin membership
surface-brightness coordinates
derived tracer products

The center is fundamentally a data-preparation reference.

It should not be confused with a fitted OSPM dynamical parameter.

SEARCH TERMS: center | RA center | Dec center | galaxy center |
projected coordinates | radial position

================================================================================
DEFAULT DIRECTORY
================================================================================


GENERAL PATH:

Data/Galaxy_Profiles/<galaxy>/default/

TYPE: OUTPUT

## ROLE

Primary location for normal OSPM results associated with the galaxy.

Typical products include model-search CSV files containing parameters,
scores, solver states, coverage diagnostics, and other model information.

CONFIG CONNECTION:

CSV_PATH commonly points into this directory.

## THIS IS NOT INPUT DATA

Files under default/ are generally outputs from OSPM runs rather than the
observational data used to define the galaxy.

SEARCH TERMS: default | results | model CSV | output | chi2 | run results

================================================================================
PLOTS DIRECTORY
================================================================================


GENERAL PATH:

Data/Galaxy_Profiles/<galaxy>/plots/

TYPE: OUTPUT

## ROLE

Galaxy-specific plots and diagnostic figures.

These are derived visualization products.

They should not normally be treated as model input.

================================================================================
ARCHIVE DIRECTORY
================================================================================


GENERAL PATH:

Data/Galaxy_Profiles/<galaxy>/archive/

TYPE: REFERENCE / LEGACY

## ROLE

Stores older data products, configurations, experiments, or retired files.

## IMPORTANT

Files in archive/ should not be treated as active pipeline inputs unless
explicitly restored or referenced by the active galaxy config.

Current data products should be generated from the current inputs rather
than silently pulling equivalent files from older archived versions.

================================================================================
SEGUE 1 ACTIVE DATA MAP
=======================

PATH:

Data/Galaxy_Profiles/Segue1/

CONFIG:

Segue1_OSPM_Config.py
Tells OSPM which Segue 1 files and galaxy-specific settings to use.

STARS:

Segue1_stars.csv
Individual stellar kinematic measurements.

LOSVD:

Segue1_losvd_bins.csv
Projected radial bins used to organize the stellar velocity constraints.

SURFACE BRIGHTNESS:

Segue1_surface_brightness.csv
Observed projected Segue 1 stellar tracer profile.

3D TRACER:

Segue1_tracer_density_3d.csv
Intrinsic three-dimensional Segue 1 tracer-density representation.

STELLAR GRAVITY:

Segue1_stellar_force_grid.csv
Precomputed stellar gravitational-field information.

CENTER:

center.txt
Adopted Segue 1 center used by coordinate preparation.

OUTPUT:

default/
Primary Segue 1 OSPM model results.

DIAGNOSTICS:

plots/
Segue 1 specific visualizations.

OLD MATERIAL:

archive/
Historical or inactive Segue 1 material.

REFERENCE:

Segue1_OSPM_Config_annotated.py
Annotated configuration reference.

karl_mode0_seeing.py
Karl / mode-0 seeing-related comparison or development support.

================================================================================
DATA FLOW BY PHYSICAL INFORMATION
================================================================================


## KINEMATICS

Observed stars:

<Galaxy>_stars.csv

are grouped spatially using:

<Galaxy>_losvd_bins.csv

and become the observed:

LOSVD constraints

These constrain stellar motion.

## STELLAR SPATIAL DISTRIBUTION

Observed projected profile:

<Galaxy>_surface_brightness.csv

is converted into / associated with:

<Galaxy>_tracer_density_3d.csv

These constrain where the modeled stellar tracer population exists.

## STELLAR GRAVITY

The stellar mass model is represented for gravitational calculations by:

<Galaxy>_stellar_force_grid.csv

Its gravitational strength is connected to:

ML

This contributes to the potential in which the orbit library is integrated.

## OTHER GRAVITY

The stellar force grid is only one part of the total gravitational model.

Other major components are supplied dynamically from the model parameters:

MBH
Central black-hole mass.

v0
Halo velocity-scale parameter.

r_c
Halo core-radius parameter.

These are model parameters rather than observational Data CSV files.

================================================================================
CONFIG TO FILE MAP
================================================================================


DATA_CSV
-> <Galaxy>_stars.csv
-> individual stellar kinematics

KINEMATIC_BINS_CSV
-> <Galaxy>_losvd_bins.csv
-> projected radial organization of kinematic constraints

SURFACE_BRIGHTNESS_CSV
-> <Galaxy>_surface_brightness.csv
-> observed 2D stellar tracer profile

STELLAR_MODEL.tracer_grid_csv
-> <Galaxy>_tracer_density_3d.csv
-> intrinsic 3D tracer constraint

STELLAR_MODEL.grid_csv
-> <Galaxy>_stellar_force_grid.csv
-> stellar gravitational contribution

CSV_PATH
-> usually <galaxy>/default/
-> model-result output destination

================================================================================
QUICK PHYSICAL LOOKUP
================================================================================


Need individual observed velocities? <Galaxy>_stars.csv

Need to know how stars are divided radially for kinematics? <Galaxy>_losvd_bins.csv

Need the observed projected stellar profile? <Galaxy>_surface_brightness.csv

Need the intrinsic 3D stellar tracer distribution? <Galaxy>_tracer_density_3d.csv

Need the gravitational pull produced by the stellar component? <Galaxy>_stellar_force_grid.csv

Need to know which of those files OSPM actually uses? <Galaxy>_OSPM_Config.py

Need shared solver and runtime configuration?
OSPM/load_config.py and OSPM/AI_defaults.py

Need normal model-run results? <galaxy>/default/

Need older material? <galaxy>/archive/

================================================================================
DATA_PREP
================================================================================


PATH: Data/Data_Prep/

## ROLE

Creates or prepares the data products used inside Galaxy_Profiles.

## HIGH-LEVEL RESPONSIBILITIES

Possible active responsibilities include:

stellar catalog preparation
galaxy-center coordinate conversion
projected radius calculation
kinematic bin construction
LOSVD preparation
surface-brightness preparation
surface-brightness remapping
2D to 3D tracer conversion
stellar density-grid construction
stellar force-grid construction
path management

## IMPORTANT BOUNDARY

Data_Prep builds the representation of the observed galaxy.

OSPM/Physics evaluates dynamical models using those prepared products.

Detailed Data_Prep scripts and functions should receive a separate index
later rather than being expanded here.

================================================================================
DATA/SLURM
================================================================================


PATH: Data/slurm/

## ROLE

Cluster / SLURM support associated with data or model execution.

This is operational support rather than a physical observable.

Detailed contents can be indexed separately if needed.

================================================================================
CORE DATA DISTINCTIONS
================================================================================


STARS:
Individual measurements.

LOSVD:
Velocity-distribution constraint built from those stellar measurements.

SURFACE BRIGHTNESS:
Observed two-dimensional spatial distribution of the tracer stars.

3D TRACER DENSITY:
Intrinsic three-dimensional spatial distribution the orbit population must
reproduce.

STELLAR FORCE GRID:
Gravitational acceleration produced by stellar mass.

CONFIG:
Tells the shared pipeline which of these products belongs to the active
galaxy and how to interpret them.

RESULT CSV:
Output from tested dynamical models.

================================================================================
SEARCH TERMS
================================================================================


Data | Galaxy_Profiles | galaxy data | config | stars | stellar catalog |
vlos | LOSVD | kinematic bins | surface brightness | Sigma | tracer |
density_3d | 3D density | stellar force | force grid | ML | galaxy center |
DATA_CSV | KINEMATIC_BINS_CSV | SURFACE_BRIGHTNESS_CSV | grid_csv |
tracer_grid_csv | CSV_PATH | Data_Prep

================================================================================
END OF DATA INDEX
================================================================================

================================================================================
PATH: OSPM/
================================================================================

CONTENTS:

# OSPM/AI

PATH: OSPM/AI/

CONTENTS:

OSPM_Daemon.py
OSPM_Daemon.py.pre_karl_resolved
**pycache**/

## ROLE

Parameter-space search and model-proposal layer.

The AI system decides which model parameters OSPM should test next.

It does NOT calculate the galaxy dynamics.

It does NOT integrate stellar orbits.

It does NOT solve the orbit weights.

It does NOT alter the observed galaxy data.

It does NOT decide directly whether a black hole, halo, or stellar model is
physically correct.

Those calculations are performed by the OSPM physics layer.

================================================================================
WHAT THE AI ACTUALLY DOES
================================================================================


The AI searches the allowed model parameter space.

For the current four-parameter model, a proposed point can be written as:

theta = [v0, r_c, MBH, ML]

where:

v0
Dark-matter halo velocity-scale parameter.

r_c
Dark-matter halo core-radius parameter.

MBH
Central black-hole mass.

ML
Stellar mass-to-light ratio.

The AI proposes combinations of these values.

Each proposed combination is handed to the physical model.

The physics layer returns the model result.

The AI uses the accumulated results to decide which parameter combinations
are useful to evaluate next.

================================================================================
AI VERSUS PHYSICS
================================================================================


AI QUESTION:

What model should OSPM evaluate next?

PHYSICS QUESTION:

Given these parameters, what stellar dynamical model results?

The separation is important.

The AI proposes:

v0 | r_c | MBH | ML

The physics code determines:

gravitational field
orbit library
orbit coverage
projected light
projected LOSVD
orbit weights
chi-square
solver state
model diagnostics

The AI cannot make a bad physical model become a good physical model.

It can only choose which physical models are evaluated.

================================================================================
OSPM_DAEMON.PY
================================================================================


FILE: OSPM/AI/OSPM_Daemon.py
LANGUAGE: Python
STATUS: ACTIVE
ROLE: Main parameter-search daemon.

## PRIMARY PURPOSE

Manages the ongoing search through OSPM parameter space.

It keeps track of previously evaluated models and generates additional
parameter proposals.

## HIGH-LEVEL RESPONSIBILITIES

Known responsibilities include:

parameter proposal
initial parameter launching
parameter bounds
batch management
search-state management
result ingestion
proposal identification
communication with the run / controller layer
passing numerical solver settings toward the physics layer
continued exploration of parameter space

The exact internal proposal functions should be indexed directly from the
source later.

================================================================================
BASIC SEARCH CYCLE
================================================================================


1. Define the allowed parameter bounds.

2. Establish initial parameter points.

3. Generate a batch of proposed models.

4. Send each proposal into the OSPM execution pipeline.

5. The physics layer evaluates each model independently.

6. Receive model scores, statuses, and diagnostics.

7. Add those results to the accumulated model history.

8. Use the growing result set to choose additional parameter proposals.

9. Continue until the requested run limit or stopping condition is reached.

================================================================================
MODEL PARAMETERS
================================================================================


CURRENT PRIMARY PARAMETER VECTOR:

theta = [v0, r_c, MBH, ML]

## PARAMETER: v0

TYPE: SCIENTIFIC

ROLE:
Sets the velocity scale of the active dark-matter halo model.

CURRENT HALO FAMILY:
v0_rc

## PARAMETER: r_c

TYPE: SCIENTIFIC

ROLE:
Sets the core radius of the active dark-matter halo model.

## PARAMETER: MBH

TYPE: SCIENTIFIC

ROLE:
Central black-hole mass.

## PARAMETER: ML

TYPE: SCIENTIFIC

ROLE:
Stellar mass-to-light ratio.

Controls the conversion between the tracer-light distribution and the
stellar gravitational mass contribution.

================================================================================
PARAMETER BOUNDS
================================================================================


The AI is not free to propose arbitrary values.

Each physical parameter has configured lower and upper bounds.

Conceptually:

THETA_BOUNDS = [
v0 bounds,
r_c bounds,
MBH bounds,
ML bounds
]

These bounds define the physical search volume accessible to the AI.

## GALAXY DEPENDENCE

Parameter bounds can differ between galaxies.

The search region appropriate for Segue 1 does not need to be identical to
the search region appropriate for Draco.

## CONFIG CONNECTION

Galaxy-specific values originate from the active galaxy configuration.

Shared search behavior is supplied by common OSPM configuration and AI
defaults.

SEARCH TERMS: THETA_BOUNDS | bounds | parameter range | v0 | r_c |
MBH | ML

================================================================================
INITIAL THETA
================================================================================


KNOWN CONFIG CONCEPT:

INITIAL_THETA

## ROLE

Provides an initial physical parameter point associated with the galaxy or
run configuration.

EXAMPLE STRUCTURE:

INITIAL_THETA = [v0, r_c, MBH, ML]

## PURPOSE

Gives the search a defined physical starting location.

It does not force the final result to remain near that location.

## IMPORTANT

INITIAL_THETA is a search initialization setting.

It is not an observational measurement.

It is not automatically the best-fit model.

It is not a physical prior unless additional search logic explicitly treats
it as one.

SEARCH TERMS: INITIAL_THETA | initial model | starting theta |
starting parameters

================================================================================
INITIAL SEARCH LAUNCH
================================================================================


KNOWN SETTING:

NTHETA_LAUNCH

TYPE: SEARCH / NUMERICAL

## ROLE

Controls the initial collection of parameter points launched when beginning
the search.

KNOWN CURRENT EXAMPLE:

NTHETA_LAUNCH = 9

The exact construction of these starting proposals should be documented from
OSPM_Daemon.py during the function-level AI expansion.

================================================================================
BATCH EXECUTION
================================================================================


KNOWN SETTINGS:

BATCH_SIZE
CHUNK_SIZE

## ROLE

Control how model proposals are grouped for execution.

KNOWN CURRENT EXAMPLES:

BATCH_SIZE = 120
CHUNK_SIZE = 60

TYPE: PERFORMANCE / SEARCH CONTROL

## IMPORTANT

Changing batch organization can alter execution behavior and throughput.

It should not by itself alter the equations used to evaluate one fixed
physical model.

SEARCH TERMS: batch | chunk | BATCH_SIZE | CHUNK_SIZE | parallel models

================================================================================
MAXIMUM SEARCH LENGTH
================================================================================


KNOWN SETTING:

MAX_RUNS

## ROLE

Limits the number of model evaluations allowed during a search.

KNOWN CURRENT EXAMPLE:

MAX_RUNS = 300000

TYPE: SEARCH / RUNTIME

## IMPORTANT

MAX_RUNS limits how long the parameter search may continue.

It does not change the physical result of an individual parameter point.

================================================================================
PROPOSAL ID
================================================================================


KNOWN OUTPUT FIELD:

proposal_id

## ROLE

Identifies or tracks a model proposal generated during the search.

## PURPOSE

Allows a model result to be associated with the proposal that generated it.

Useful for:

debugging
search-history reconstruction
batch tracking
proposal diagnostics

SEARCH TERMS: proposal_id | proposal | model proposal | search history

================================================================================
WHAT RETURNS TO THE AI
================================================================================


After a proposed theta point is evaluated, the broader pipeline returns
information describing what happened to that model.

KNOWN RESULT FIELDS INCLUDE:

v0
r_c
MBH
ML
chi2
reward
status
proposal_id

chi2_losvd
delta_chi2_iteration
max_light_relative_residual
light_constraint_ok

solver_converged
solver_iterations
solver_failure_reason
julia_status_code

chi2_inner
chi2_outer
N_inner
N_outer

N_nonzero_weights
effective_N_orbits
max_weight_fraction

coverage_status
coverage_strict
coverage_fraction
coverage_success_fraction

successful_base_orbits
planned_base_orbits

Not every field necessarily drives AI proposal generation.

Many are retained for physical diagnostics, solver diagnostics, plotting,
or later analysis.

The exact fields consumed by proposal logic should be identified during
source-level expansion.

================================================================================
CHI-SQUARE
================================================================================


KNOWN PRIMARY RESULT:

chi2

## ROLE

Measures model disagreement with the observational constraints according to
the active OSPM objective.

## IMPORTANT

The AI does not calculate chi-square itself.

The physical model produces the score.

The AI receives that score as information about the proposed point.

This distinction prevents a common misunderstanding:

AI does not decide that a model fits.

The dynamical calculation determines how well the model fits.

The AI uses that result to decide where to search next.

================================================================================
REWARD
================================================================================


KNOWN RESULT FIELD:

reward

## ROLE

Search-facing quantity associated with a completed proposal.

## IMPORTANT

reward is part of the parameter-search machinery.

chi2 is the physically interpretable model-fit quantity.

The exact transformation between model results and reward should be indexed
from the active OSPM_Daemon.py source rather than inferred here.

SEARCH TERMS: reward | chi2 | objective | search score

================================================================================
STATUS
================================================================================


KNOWN RESULT FIELD:

status

## ROLE

Records how a proposed model evaluation ended.

Possible outcomes can distinguish successful models from failures or
diagnostic variants.

Known examples from current result files include statuses such as:

pass_full
pass_no_bh
pass_no_bh_halo_up
pass_no_bh_halo_scale_down
pass_no_bh_ml_up
pass_no_bh_ml_down
pass_no_bh_halo_scale_up

Additional failure states originate from the physical and solver layers.

## IMPORTANT

A status is not simply an AI opinion.

It records what happened when the proposed physical model was evaluated.

================================================================================
AI DEFAULTS
================================================================================


RELATED FILE:

OSPM/AI_defaults.py

PATH:

OSPM/AI_defaults.py

## ROLE

Contains shared defaults associated with search behavior.

## RELATIONSHIP

OSPM_Daemon.py
Implements the active search machinery.

OSPM/AI_defaults.py
Provides common default values used by that machinery.

<Galaxy>_OSPM_Config.py
Provides galaxy-specific values or overrides when required.

OSPM/load_config.py
Combines the active configuration.

Exact ownership of every setting should be recorded when these files are
expanded.

================================================================================
PHYSICS SETTINGS PASSED THROUGH THE DAEMON
================================================================================


OSPM_Daemon.py also participates in passing settings toward the Julia
physics layer.

Known examples include:

KARL_ALPHAT
KARL_APFAC
KARL_LIGHT_REL_TOL
KARL_DELTA_CHI2_ITER_TOL

Examples of current values:

KARL_ALPHAT = 1.0
KARL_APFAC = 0.01
KARL_LIGHT_REL_TOL = 0.01
KARL_DELTA_CHI2_ITER_TOL = 0.3

## IMPORTANT DISTINCTION

The daemon may TRANSPORT these values.

That does not mean their physical or numerical implementation lives in the
AI.

Their actual effect occurs inside the Julia weight solver.

Example:

KARL_APFAC
OSPM_Daemon.py reads / passes the setting.

OSPM_Physics_Weights.jl uses the setting during the weight-solver
calculation.

This distinction should be preserved throughout the index.

================================================================================
AI CONFIGURATION TYPES
================================================================================


AI-related settings can be separated into several categories.

## SCIENTIFIC

Changes the physical parameter space being tested.

Examples:

THETA_BOUNDS
INITIAL_THETA when used to define the starting physical model

## SEARCH

Changes how parameter space is explored.

Examples:

proposal behavior
initial search launch
search-state rules

## PERFORMANCE

Changes how work is grouped or executed without intentionally changing the
fixed-theta physical calculation.

Examples:

BATCH_SIZE
CHUNK_SIZE

## RUNTIME

Controls how long the search operates.

Example:

MAX_RUNS

## PHYSICS PASS-THROUGH

Values read or transported by Python but applied inside the Julia physics
calculation.

Examples:

KARL_ALPHAT
KARL_APFAC
KARL_LIGHT_REL_TOL
KARL_DELTA_CHI2_ITER_TOL

================================================================================
WHAT THE AI DOES NOT SEE DIRECTLY
================================================================================


At the proposal level, the AI does not need to understand a stellar orbit in
the way the Julia model does.

It does not independently calculate:

orbital energy
angular momentum
third integral
SOS structure
orbit trajectory
stellar acceleration
phase volume
LOSVD projection
light projection
entropy derivatives
SPEAR steps
active weight boundaries

Those belong to the physics calculation.

The AI operates one level above them:

propose physical parameters
-> receive model result
-> choose another proposal

================================================================================
AI AND BLACK-HOLE SEARCHES
================================================================================


MBH is one coordinate of the parameter space.

The AI can propose:

MBH = 0

or:

MBH > 0

The presence of MBH as an AI parameter does not force the fit toward a black
hole.

Models with different MBH values are evaluated through the same physical
machinery.

A preferred nonzero MBH can only emerge if those physical models produce
better accepted fits within the explored parameter space.

The AI's job is to make sure that relevant portions of that parameter space
are actually tested.

================================================================================
AI AND GALAXY DIFFERENCES
================================================================================


The same search machinery can be used for different galaxies.

The galaxy-specific differences come primarily from:

observational data
tracer structure
geometry
parameter bounds
initial theta
other galaxy configuration

The AI should not need a fundamentally separate algorithm for Segue 1,
Draco, Carina, or another galaxy unless an explicit scientific reason is
introduced.

This is part of the goal of keeping OSPM galaxy agnostic.

================================================================================
OSPM_DAEMON.PY.PRE_KARL_RESOLVED
================================================================================


FILE:

OSPM/AI/OSPM_Daemon.py.pre_karl_resolved

STATUS: REFERENCE / BACKUP

## ROLE

Historical snapshot of the daemon from before later Karl-fidelity changes
were resolved.

## IMPORTANT

This file should not be assumed to execute.

It exists as development history or comparison material.

Use:

OSPM/AI/OSPM_Daemon.py

when determining current search behavior.

================================================================================
AI DATA FLOW
================================================================================


1. Active galaxy configuration is loaded.

2. Parameter bounds are established.

3. Initial theta information is established.

4. OSPM_Daemon.py generates model proposals.

5. Proposed theta values enter the controller / physics pipeline.

6. Julia evaluates the physical model.

7. The model returns chi-square, status, and diagnostics.

8. The result is written into the model-history CSV.

9. The daemon uses accumulated search information to generate additional
   proposals.

10. The cycle continues until the search ends.

================================================================================
RELATION TO OTHER OSPM COMPONENTS
================================================================================


OSPM/AI/OSPM_Daemon.py
Chooses what physical parameters should be tested.

OSPM/Controllers/
Coordinates execution of those tests.

OSPM/Physics/OSPM_Physics.py
Provides the Python-side physics interface.

OSPM/Physics/OSPM_PhysicsEngine.py
Connects Python execution to Julia.

OSPM/Physics/OSPM_Physics_Spherical.jl
Evaluates the orbit model.

OSPM/Physics/OSPM_Physics_Weights.jl
Determines the stellar orbit weights.

Data/Galaxy_Profiles/<galaxy>/
Provides the observational information against which the model is tested.

<galaxy>/default/*.csv
Stores the accumulated model-search results.

================================================================================
COMMON AI MISUNDERSTANDINGS
================================================================================


QUESTION:
Does AI generate the physical solution?

ANSWER:
No. It generates parameter proposals. The physical solution is calculated
by the OSPM physics code.

QUESTION:
Does AI choose the orbit weights?

ANSWER:
No. The Julia weight solver determines the orbit weights.

QUESTION:
Does AI decide the chi-square?

ANSWER:
No. Chi-square comes from comparison between the physical model and the
observational constraints.

QUESTION:
Does AI decide that a black hole exists?

ANSWER:
No. MBH is one parameter being searched. Models with different MBH values
must survive the same physical evaluation.

QUESTION:
Can AI make an unphysical model fit?

ANSWER:
It cannot bypass the physical calculation. A proposed model still has to
produce a valid orbit library, satisfy the required constraints, and return
an acceptable model score.

QUESTION:
Why use AI at all?

ANSWER:
The physical parameter space is large enough that evaluating every possible
combination on a fine regular grid would be expensive.

The search layer attempts to spend model evaluations in useful parts of the
parameter space while retaining exploration.

QUESTION:
Is the AI the scientific model?

ANSWER:
No.

The scientific model is the gravitational potential, orbit library,
observable projection, orbit-weight solution, and comparison with galaxy
data.

The AI is the search strategy wrapped around that model.

================================================================================
QUICK LOOKUP
================================================================================


Need to know what parameter point is proposed next?
OSPM/AI/OSPM_Daemon.py

Need shared AI/search defaults?
OSPM/AI_defaults.py

Need galaxy-specific parameter bounds?
Data/Galaxy_Profiles/<galaxy>/<Galaxy>_OSPM_Config.py

Need the physical result for a proposed theta?
OSPM/Physics/

Need the actual orbit-weight solution?
OSPM/Physics/OSPM_Physics_Weights.jl

Need accumulated AI/model-search history?
Data/Galaxy_Profiles/<galaxy>/default/

Need historical daemon behavior?
OSPM/AI/OSPM_Daemon.py.pre_karl_resolved



================================================================================
SEARCH TERMS
================================================================================


AI | daemon | OSPM_Daemon | parameter search | proposal | proposal_id |
theta | INITIAL_THETA | THETA_BOUNDS | v0 | r_c | MBH | ML | reward |
chi2 | model search | optimization | batch | chunk | MAX_RUNS |
NTHETA_LAUNCH | AI_defaults | exploration | model proposal |
parameter space | black hole search

================================================================================
END OF OSPM AI INDEX
================================================================================



# HIGH-LEVEL RESPONSIBILITIES

Controllers/  Top-level run orchestration and execution control.
Physics/      Builds and evaluates dynamical models.
Observables/  Defines and prepares observational constraints used by OSPM.
Mapping/      Maps model or data quantities between observational representations.
Plotting/     Internal plotting, diagnostics, and model-analysis support.
Fortran/      Karl's original Fortran implementation retained for reference and fidelity checks.

================================================================================
OSPM/CONTROLLERS
================================================================================


PATH: OSPM/Controllers/

CONTENTS: OSPM_API.py | OSPM_Control.py | OSPM_MASTER.py | OSPM_RUN.py

ROLE: Top-level execution and coordination layer.

The Controllers directory manages how an OSPM run is started, organized,
executed, and connected to the rest of the pipeline.

It does not contain the main dynamical calculations.

The actual galaxy physics is handled by OSPM/Physics/.

## OSPM_MASTER.py

STATUS: ACTIVE
ROLE: High-level orchestration.

Coordinates the major pieces of an OSPM run.

This is part of the upper execution layer that connects configuration,
search machinery, model execution, and run management.

SEARCH TERMS: master | orchestration | top-level run | pipeline

## OSPM_RUN.py

STATUS: ACTIVE
ROLE: Model-run execution machinery.

Handles execution of requested OSPM model evaluations and associated
runtime behavior.

Acts between the higher-level controller machinery and the physics
evaluation layer.

SEARCH TERMS: run | execute | model run | runtime | evaluation

## OSPM_Control.py

STATUS: ACTIVE
ROLE: Pipeline control logic.

Contains control behavior used to manage OSPM execution.

This file belongs to the organizational layer of the pipeline rather than
the physical model itself.

SEARCH TERMS: control | execution control | run control | pipeline

## OSPM_API.py

STATUS: ACTIVE
ROLE: Internal interface layer.

Provides an interface between major OSPM components.

Used to keep higher-level run machinery separated from lower-level model
implementation details.

SEARCH TERMS: API | interface | communication | controller | physics

## CONTROLLER FLOW

1. OSPM receives a request to run or evaluate models.

2. Controller machinery establishes the run.

3. Parameter proposals are supplied by the search layer when applicable.

4. Individual model evaluations are passed toward OSPM/Physics/.

5. Physics results return to the controlling Python layer.

6. Results are recorded and the broader run continues.

## CONTROLLER BOUNDARY

Controllers decide HOW and WHEN model evaluations are executed.

Controllers do not determine:

gravitational forces | orbit trajectories | orbit weights | LOSVD |
surface-brightness agreement | physical chi-square

Those belong to the physics and observable layers.

## QUICK LOOKUP

Need high-level run coordination?
OSPM_MASTER.py

Need individual run execution behavior?
OSPM_RUN.py

Need execution-control logic?
OSPM_Control.py

Need the interface between OSPM components?
OSPM_API.py

SEARCH TERMS: Controllers | OSPM_MASTER | OSPM_RUN | OSPM_Control |
OSPM_API | orchestration | execution | runtime | interface

##################################################################

PATH: OSPM/Physics/

CONTENTS:

OSPM_Physics.py
OSPM_PhysicsEngine.py
OSPM_Physics_Support.jl
OSPM_Physics_Force.jl
OSPM_Physics_PhaseVolume.jl
OSPM_Physics_Weights.jl
OSPM_Physics_Spherical.jl
observable_mapping.py
annotated/
pycache/

# OSPM/PHYSICS - PYTHON LAYER

ROLE: Python-side interface between the broader OSPM pipeline and the Julia
physics implementation.

The Python physics layer prepares model inputs, manages configuration needed
for evaluation, calls the Julia model engine, and returns the resulting
model information to the rest of OSPM.

It does not contain the main orbit integration or orbit-weight solver.

## OSPM_Physics.py

PATH: OSPM/Physics/OSPM_Physics.py
LANGUAGE: Python
STATUS: ACTIVE
ROLE: Physics interface and configuration layer.

Connects the controller / search side of OSPM to the physics machinery.

Responsibilities include preparing physics-related inputs and settings
needed for a model evaluation.

Acts as part of the boundary between the general Python pipeline and the
lower-level Julia implementation.

SEARCH TERMS: physics interface | configuration | model parameters |
Python physics | Julia handoff

## OSPM_PhysicsEngine.py

PATH: OSPM/Physics/OSPM_PhysicsEngine.py
LANGUAGE: Python
STATUS: ACTIVE
ROLE: Python-to-Julia execution bridge.

Handles communication between Python and the Julia model-evaluation code.

A proposed physical model reaches this layer from the Python pipeline.

The engine invokes the Julia physics machinery and receives the resulting
model score, solver state, and diagnostics.

Julia calculations reached through this layer include:

orbit-library construction | orbit integration | observable projection |
orbit-weight solving | chi-square | model diagnostics

The engine does not independently perform those calculations.

SEARCH TERMS: PhysicsEngine | PythonCall | Julia bridge | Julia execution |
evaluate model | physics engine | model result

## PYTHON PHYSICS FLOW

1. A proposed model parameter set reaches the physics layer.

2. OSPM_Physics.py prepares the physics-side model information.

3. OSPM_PhysicsEngine.py communicates with Julia.

4. Julia evaluates the dynamical model.

5. Julia returns model results and diagnostics.

6. The Python layer returns those results to the controller / search
   machinery.

## BOUNDARY

Python physics layer:
model setup | configuration | communication | execution interface

Julia physics layer:
forces | orbit integration | phase volume | projection | orbit weights |
chi-square | detailed solver diagnostics

## QUICK LOOKUP

Need Python-side physics setup?
OSPM_Physics.py

Need the Python-to-Julia bridge?
OSPM_PhysicsEngine.py

SEARCH TERMS: Physics | OSPM_Physics | OSPM_PhysicsEngine | Python |
Julia | bridge | interface | model evaluation


JULIA PHYSICS LAYER

The Julia layer currently consists of:

OSPM_Physics_Support.jl
OSPM_Physics_Force.jl
OSPM_Physics_PhaseVolume.jl
OSPM_Physics_Weights.jl
OSPM_Physics_Spherical.jl

OSPM_Physics_Support.jl

ROLE: Shared Julia support code.

Contains common constants, defaults, structures, helpers, and numerical
support used by other Julia physics files.

Known example:

DEFAULT_KARL_ENTROPY_FLOOR = 1e-30

SEARCH TERMS: support | helpers | constants | defaults | structures |
entropy floor

OSPM_Physics_Force.jl

ROLE: Gravitational force calculations.

Represents the acceleration experienced by an orbit due to model mass
components.

Relevant physical components include:

central black hole | stars | dark matter halo

Relevant model parameters include:

v0 | r_c | MBH | ML

SEARCH TERMS: gravity | force | acceleration | potential | halo |
black hole | stellar force | v0 | rc | MBH | ML

OSPM_Physics_PhaseVolume.jl

ROLE: Orbit phase-volume calculations.

Handles the phase-space volume represented by the sampled orbit library.

Relevant orbit coordinates include:

energy_index = shell_id
lz_index = lfrac_id
third_index = theta_id

Phase volume has also been relevant to historical orbit-weight
initialization behavior.

SEARCH TERMS: phase volume | phase space | energy | shell | lfrac |
theta | third integral | orbit grid

OSPM_Physics_Weights.jl

ROLE: Orbit-weight optimization.

Determines the population assigned to each successfully integrated orbit.

Major concepts include:

weight initialization | entropy | light constraints | LOSVD constraints |
expanded constraint matrix | SPEAR | nonnegative weights | active boundary |
convergence | chi-square | solver diagnostics

Known important functions include:

solve_weights_karl_expanded_cm
karl_spear_step_light_losvd_all
karl_safe_step_factor
build_expanded_entropy_derivatives
karl_raw_spear_consistency_diagnostic
karl_losvd_fracnew_state
losvd_width_at_fraction

Known settings include:

KARL_ALPHAT | KARL_APFAC | KARL_LIGHT_REL_TOL |
KARL_DELTA_CHI2_ITER_TOL | entropy_floor

Karl reference files include:

spear.f.txt | entropy.f.txt | model.f.txt

STATUS: ACTIVE
DETAILED FUNCTION INDEX: TO BE EXPANDED NEXT

SEARCH TERMS: weights | orbit weights | entropy | SPEAR | Karl | LOSVD |
light constraint | active set | chi-square | expanded CM

OSPM_Physics_Spherical.jl

ROLE: Main Julia model-evaluation engine.

Coordinates the complete orbit-model evaluation for proposed physical
parameters.

Major responsibilities include:

model setup | orbit launch | orbit integration | SOS | orbit continuation |
coverage | projection | light constraints | LOSVD constraints |
weight-solver handoff | chi-square | diagnostics | result construction

Known central function:

evaluate_batch_theta

Known orbit-library indexing:

energy_index = shell_id
lz_index = lfrac_id
third_index = theta_id

Known diagnostic areas include:

coverage
inner versus outer chi-square
orbit-weight statistics
orbit-family diagnostics
solver status
light residuals

STATUS: ACTIVE
DETAILED FUNCTION INDEX: TO BE EXPANDED LATER

SEARCH TERMS: spherical | evaluate_batch_theta | orbit library |
integration | coverage | projection | LOSVD | light | weights | chi-square

observable_mapping.py

LANGUAGE: Python
ROLE: Physics-side observable mapping support.

NOTE:

There is also:

OSPM/Mapping/observable_mapping.py

The relationship between these two files should be documented during the
Mapping audit.

Possible classifications include:

ACTIVE SEPARATE PURPOSE | DUPLICATED | LEGACY | REFERENCE

annotated/

ROLE: Annotated or explanatory physics-code copies.

Files here should not be assumed to participate directly in execution.

The Julia files are intentionally NOT expanded function-by-function in this
version of the repository index.

Planned detailed indexing order:

OSPM_Physics_Support.jl
OSPM_Physics_Force.jl
OSPM_Physics_PhaseVolume.jl
OSPM_Physics_Weights.jl
OSPM_Physics_Spherical.jl

Each detailed Julia index should identify:

FUNCTION | SIGNATURE | ROLE | PHYSICAL PURPOSE | NUMERICAL PURPOSE | INPUTS |
OUTPUTS | IMPORTANT VARIABLES | UNITS | CALLS | CALLED BY | CONFIGURATION |
DIAGNOSTICS | FAILURE MODES | KARL EQUIVALENT | FIDELITY | SEARCH TERMS

FIDELITY LABELS: VERIFIED_KARL_MATCH | INTENTIONAL_JULIA_EXTENSION |
KNOWN_MISMATCH | MISMATCH_FIXED | NOT_YET_AUDITED | NO_KARL_EQUIVALENT

CONTROL TYPES: SCIENTIFIC | NUMERICAL | PERFORMANCE | DIAGNOSTIC_ONLY |
LEGACY

PATH: OSPM/Fortran/

CONTENTS:

entropy.f.txt
gden.f.txt
halodens.f.txt
model.f.txt
projmass.f.txt
spear.f.txt
fortran codes of interest.txt
local_next_time_note.txt

ROLE

Reference implementation from Karl's OSPM.

Used to determine intended algorithms, initialization, numerical procedures,
and physical behavior when auditing the Julia implementation.

entropy.f.txt

ROLE: Karl entropy-related implementation.

gden.f.txt

ROLE: Karl stellar-density-related implementation.

halodens.f.txt

ROLE: Karl halo-density implementation.

model.f.txt

ROLE: Karl model-level implementation and execution logic.

projmass.f.txt

ROLE: Karl projected-mass / projected-light related implementation.

spear.f.txt

ROLE: Karl orbit-weight adjustment and SPEAR solver implementation.

fortran codes of interest.txt

ROLE: Reference notes or collected Fortran material of particular interest.

local_next_time_note.txt

ROLE: Development note / reference.

Not assumed to participate in execution.

SEARCH TERMS: Karl | Fortran | spear | entropy | model | halo density |
projected mass | reference implementation | fidelity

PATH: OSPM/Mapping/

CONTENTS:

gen_fig.py
gen_fig_standard.py
observable_mapping.py
pycache/

ROLE

Maps model-space quantities into representations associated with the
observational data.

Exact responsibilities should be expanded separately.

observable_mapping.py

ROLE: Observable mapping.

NOTE:

A second file with the same name exists at:

OSPM/Physics/observable_mapping.py

Their relationship needs explicit classification.

gen_fig.py

ROLE: Figure or mapping-generation support.

gen_fig_standard.py

ROLE: Standardized figure or mapping-generation support.

SEARCH TERMS: mapping | observable mapping | projection | generated figure

PATH: OSPM/Observables/

ROLE

Contains machinery associated with observational quantities consumed by or
compared against the dynamical model.

This directory should receive its own index after its contents are
inspected.

SEARCH TERMS: observables | LOSVD | surface brightness | tracer |
kinematics | data constraints

PATH: OSPM/Plotting/

CONTENTS:

plot_analysis.py
map_sb_to_losvd_bins/
segue1_surface_brightness/
archive/
pycache/

ROLE

Internal OSPM plotting and analysis support.

plot_analysis.py

ROLE: Model-result plotting and analysis.

map_sb_to_losvd_bins/

ROLE: Surface-brightness to LOSVD-bin mapping tools or related diagnostics.

segue1_surface_brightness/

ROLE: Segue 1 surface-brightness plotting or analysis support.

archive/

ROLE: Retired plotting code or historical analysis material.

SEARCH TERMS: plotting | analysis | surface brightness | LOSVD mapping |
diagnostics

OSPM/load_config.py

STATUS: ACTIVE
ROLE: Shared configuration loader.

Combines shared OSPM defaults with the active galaxy-specific configuration.

SEARCH TERMS: config | defaults | galaxy config | load configuration

OSPM/Gal_Registry.py

ROLE: Galaxy registration and galaxy-selection support.

SEARCH TERMS: galaxy registry | active galaxy | profile lookup

OSPM/AI_defaults.py

ROLE: Shared defaults associated with AI / search behavior.

SEARCH TERMS: AI defaults | daemon defaults | search settings

OSPM/check_galaxy_model.py

ROLE: Galaxy/model validation and consistency checks.

SEARCH TERMS: check galaxy | validate model | configuration validation

OSPM/surface_brightness_variants.py

ROLE: Surface-brightness variation and diagnostic machinery.

Relevant to testing changes in the tracer profile or observational
uncertainty while holding other model information fixed.

SEARCH TERMS: surface brightness | variants | uncertainty | perturbation

OSPM/OSPM_active_set_standalone.jl

LANGUAGE: Julia
STATUS: NEEDS CLASSIFICATION

Possible role:

Standalone active-set weight-solver development or reference implementation.

Questions to resolve:

Is it imported by the active pipeline?
Is it development-only?
Is its active implementation now inside OSPM_Physics_Weights.jl?
Does it correspond directly to Karl solver machinery?

CLASSIFICATION OPTIONS: ACTIVE | DEVELOPMENT TOOL | REFERENCE | LEGACY |
UNUSED

PATH: plots/

ROLE

User-facing plotting scripts and generated scientific visualization support.

This is separate from OSPM/Plotting/, which contains plotting machinery
inside the OSPM package structure.

The distinction between the two should remain explicit.

PATH: utils/

ROLE

Command-line and shell utilities used to operate the repository.

Known uses include:

starting OSPM
running plotting tools
debugging the environment
cluster or local execution support

Examples used during development include:

bash utils/start
bash utils/plot ...

PATH: logs/

ROLE

Runtime output, debugging information, solver logs, and other execution
records.

Files here are outputs rather than primary scientific input.

PATH: Lit/

ROLE

Literature, papers, notes, and reference material associated with OSPM and
the galaxy models.

Select the active galaxy.
Load shared OSPM configuration.
Load galaxy-specific configuration.
Locate prepared observational and tracer data.
Start the controller / run machinery.
Generate or receive a proposed model parameter set.
Send the model parameters to the physics layer.
Enter the Julia model-evaluation layer.
Construct the gravitational model.
Build and integrate the orbit library.
Project successful orbits into observable constraints.
Solve the orbit weights.
Evaluate light and kinematic agreement.
Return model diagnostics and score.
Record the result.
Use the result to guide subsequent model proposals.
Analyze or plot completed model grids.

ACTIVE
Participates directly in the current production pipeline.

GENERATED DATA PRODUCT
Produced from source data for later use by OSPM.

SOURCE DATA
Observed or prepared observational information.

CONFIGURATION
Defines galaxy-specific or shared model behavior.

OUTPUT
Produced by runs, diagnostics, or plotting.

REFERENCE
Retained for comparison, explanation, or historical context.

DEVELOPMENT TOOL
Used during code development but not necessarily production execution.

LEGACY
Older implementation retained for history or comparison.

UNUSED
Confirmed not to participate in the current pipeline.

This documentation uses plain text intended for simple editors such as
Notepad.