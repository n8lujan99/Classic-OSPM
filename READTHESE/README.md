Gal_Dynamics/
│
├── Data/
│   │
│   ├── Data_Prep/
│   │   │
│   │   ├── Data_Config.py
│   │   │      ├─ defines GLOBAL constants
│   │   │      └─ imported by:
│   │   │           Data_Preprocess.py
│   │   │           Data_Asmbl.py
│   │   │           OSPM_Control.py
│   │   │
│   │   ├── Data_Web_Srcs.py
│   │   │      └─ fetches external catalogs
│   │   │
│   │   ├── Data_Sources.py
│   │   │      ├─ uses Data_Web_Srcs.py
│   │   │      └─ outputs raw source tables
│   │   │
│   │   ├── Data_Load_Center.py
│   │   │      ├─ reads galaxy center
│   │   │      └─ pulls from Galaxy_OSPM_Config.py
│   │   │
│   │   ├── Data_Geometry.py
│   │   │      ├─ uses center + distance
│   │   │      ├─ computes:
│   │   │      │     x_pc, y_pc
│   │   │      │     r_pc
│   │   │      │     r_ell_pc
│   │   │      └─ no knowledge of OSPM
│   │   │
│   │   ├── Data_Preprocess.py
│   │   │      ├─ uses Data_Config.py
│   │   │      ├─ uses Data_Geometry.py output
│   │   │      └─ applies cuts / masks only
│   │   │
│   │   ├── Data_Asmbl.py
│   │   │      ├─ merges all prepared columns
│   │   │      └─ writes final CSV
│   │   │
│   │   └── Data_Paths.py
│   │          └─ resolves paths into Profiles/
│   │
│   └── Profiles/
│       │
│       ├── <Galaxy>/
│       │   │
│       │   ├── Profiles/
│       │   │   │
│       │   │   ├── <Galaxy>_OSPM_Config.py
│       │   │   │      ├─ galaxy constants only
│       │   │   │      │   RA, DEC, distance
│       │   │   │      │   PA, ellipticity
│       │   │   │      │   R_max
│       │   │   │      └─ imported by:
│       │   │   │           Data_Load_Center.py
│       │   │   │           OSPM_RUN.py
│       │   │   │
│       │   │   └── <Galaxy>_Stellar_Profile.csv
│       │   │
│       │   ├── data/
│       │   │   └── ospm_input.csv
│       │   │
│       │   ├── logs/
│       │   ├── checkpoint/
│       │   └── Dump/
│       │
│       └── (each galaxy is self-contained)
│
├── OSPM/
│   │
│   ├── Controllers/
│   │   ├── OSPM_Control.py
│   │   │      ├─ reads Data_Config.py
│   │   │      └─ loads Galaxy_OSPM_Config.py
│   │   │
│   │   ├── OSPM_MASTER.py
│   │   │      └─ orchestrates solver + physics
│   │   │
│   │   └── OSPM_RUN.py
│   │          └─ entry point
│   │
│   ├── Observables/
│   │   └── OSPM_Observables_Stellar.py
│   │          └─ consumes ospm_input.csv
│   │
│   ├── Physics/
│   │   ├── OSPM_PhysicsEngine.py
│   │   │      └─ central physics interface
│   │   │
│   │   ├── OSPM_Physics.py
│   │   │      └─ shared constants + helpers
│   │   │
│   │   └── OSPM_Physics_Spherical.jl
│   │          └─ heavy orbit integration
│   │
│   ├── Solvers/
│   │   └── OSPM_Solver_stellar.py
│   │          └─ NNLS / weight solve
│   │
│   ├── AI/
│   │   └── OSPM_Daemon.py
│   │          └─ proposes theta, reads chi²
│   │
│   └── Plotting/
│       └── plot_analysis.py
│
├── Utils/
│   └── shell helpers only
│
└── _Setup/
    └── packaging / requirements
