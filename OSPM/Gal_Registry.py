"""
Galaxy registry and data loader.
No hard-coded galaxy imports.
CSV-driven capability detection.


from pathlib import Path
import importlib
import pandas as pd

GAL_ROOT = Path("Data/Gal_Profiles")

REQUIRED_STAR_COLS = {"r_pc"}
VLOS_COL = "vlos"
VLOS_ERR_COL = "vlos_err"
HAS_VLOS_COL = "has_vlos"


def iter_galaxies():
    for d in sorted(GAL_ROOT.iterdir()):
        if d.is_dir():
            yield d.name, d


def load_config(galaxy):
    modname = f"Data.Gal_Profiles.{galaxy}.{galaxy}_OSPM_Config"
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        raise RuntimeError(f"Failed to load config for {galaxy}") from e
    return mod.CONFIG, mod.PROFILE_ROOT


def load_csv(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def inspect_star_table(df):
    if df is None:
        return dict(has_geometry=False, has_vlos=False)

    cols = set(df.columns)

    has_geometry = REQUIRED_STAR_COLS.issubset(cols)

    has_vlos = (
        (VLOS_COL in cols)
        and (
            (VLOS_ERR_COL in cols)
            or (HAS_VLOS_COL in cols)
        )
    )

    return dict(
        has_geometry=has_geometry,
        has_vlos=has_vlos,
    )


def load_galaxy(galaxy):
    cfg, root = load_config(galaxy)

    default = root / "default"
    data_df = load_csv(default / "data.csv")
    star_df = load_csv(default / "star.csv")

    caps = inspect_star_table(star_df)

    return dict(
        galaxy=galaxy,
        config=cfg,
        profile_root=root,
        data=data_df,
        stars=star_df,
        **caps,
    )


def load_all_galaxies(skip=()):
    out = {}
    for gal, _ in iter_galaxies():
        if gal in skip:
            continue
        try:
            out[gal] = load_galaxy(gal)
        except Exception as e:
            out[gal] = dict(galaxy=gal, error=str(e))
    return out
"""