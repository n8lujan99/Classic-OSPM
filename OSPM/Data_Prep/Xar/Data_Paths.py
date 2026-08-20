# Data/Data_Prep/Data_Paths.py
from pathlib import Path
def build_data_paths(PROFILE_ROOT: Path, *, run_label: str = "default"):
    base = PROFILE_ROOT / run_label
    return {
        "PROFILE_ROOT":     PROFILE_ROOT,
        "DATA_CSV":         base / "data.csv",
        "STAR_CSV":         base / "star.csv",
        "OUT_CSV":          base / "ospm_input.csv",
        "PROFILE_CSV":      base / "stellar_profile.csv",
        "SPEC_PATH":        base / "spec.csv",
        "CSV_PATH":         base / "daemon_deck.csv",
        "LOG_PATH":         base / "ospm.log",
        "CHECKPOINT_DIR":   base / "checkpoints",
    }
def ensure_data_dirs(paths):
    for p in paths.values():
        if isinstance(p, Path):
            if p.suffix:
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir(parents=True, exist_ok=True)
