#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MATCHED = Path(
    "Data/Galaxy_Profiles/Segue1/default/hpc_compare/matched_bins/"
    "daemon_deck_karl_segue1_matched_bins_vcirc.csv"
)
DEFAULT_FULL = Path(
    "Data/Galaxy_Profiles/Segue1/default/hpc_compare/full_light/"
    "daemon_deck_karl_segue1_full_light_vcirc.csv"
)

METRICS = [
    "chi2_losvd",
    "chi2_light",
    "chi2_total",
    "chi2_inner",
    "chi2_outer",
    "vcirc",
    "r_s",
    "MBH",
    "ML",
    "N_nonzero_weights",
    "effective_N_orbits",
    "max_weight_fraction",
]


def _load_real_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "status" not in df.columns:
        raise KeyError(f"{path} is missing status")
    real = df[df["status"].astype(str).eq("pass_full")].copy()
    for col in METRICS + ["chi2"]:
        if col in real.columns:
            real[col] = pd.to_numeric(real[col], errors="coerce")
    return df, real


def _best_row(real):
    if real.empty:
        return None
    score_col = "chi2_total" if "chi2_total" in real.columns else "chi2"
    good = real[np.isfinite(real[score_col])]
    if good.empty:
        return None
    return good.loc[good[score_col].idxmin()]


def _print_case(label, path):
    df, real = _load_real_rows(path)
    print(f"\n== {label} ==")
    print(f"path: {path}")
    print(f"rows: {len(df)}")
    print("status counts:")
    print(df["status"].astype(str).value_counts(dropna=False).to_string())
    print(f"pass_full rows: {len(real)}")

    best = _best_row(real)
    if best is None:
        print("best pass_full: none")
        return None

    print("best pass_full:")
    for col in METRICS:
        if col in best.index:
            print(f"  {col}: {best[col]}")
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Compare Segue1 matched-bin and full-light HPC diagnostic decks."
    )
    parser.add_argument("--matched", default=str(DEFAULT_MATCHED))
    parser.add_argument("--full", default=str(DEFAULT_FULL))
    args = parser.parse_args()

    matched = _print_case("matched_bins", args.matched)
    full = _print_case("full_light", args.full)

    if matched is None or full is None:
        return

    print("\n== full_light minus matched_bins, best pass_full ==")
    for col in METRICS:
        if col in matched.index and col in full.index:
            try:
                delta = float(full[col]) - float(matched[col])
            except Exception:
                continue
            print(f"{col}: {delta}")


if __name__ == "__main__":
    main()
