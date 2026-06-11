import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def weighted_mean(v, verr):
    w = 1.0 / np.square(verr)
    return np.sum(w * v) / np.sum(w)


def main():
    parser = argparse.ArgumentParser(
        description="Build Segue 1 LOSVD radial bins from the Simon star catalog."
    )

    parser.add_argument(
        "--stars",
        default="Data/Galaxy_Profiles/Segue1/segue1_simon_data.csv",
        help="Input Simon star catalog.",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=4,
        help="Number of equal-count LOSVD radial bins.",
    )

    parser.add_argument(
        "--outdir",
        default="Data/Galaxy_Profiles/Segue1",
        help="Output directory.",
    )

    args = parser.parse_args()

    stars_path = Path(args.stars)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(stars_path)

    # Rename into V2-friendly names.
    df = df.rename(
        columns={
            "ra": "ra_deg",
            "dec": "dec_deg",
            "vlos": "vlos_kms",
            "vlos_err": "verr_kms",
            "src": "source",
            "r_separation_pc": "R_pc",
        }
    )

    needed = ["ra_deg", "dec_deg", "vlos_kms", "verr_kms", "source", "R_pc"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in ["ra_deg", "dec_deg", "vlos_kms", "verr_kms", "R_pc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ra_deg", "dec_deg", "vlos_kms", "verr_kms", "R_pc"])
    df = df[df["verr_kms"] > 0].copy()

    # Sort stars by projected radius.
    df = df.sort_values("R_pc").reset_index(drop=True)

    v_sys = weighted_mean(df["vlos_kms"].to_numpy(), df["verr_kms"].to_numpy())
    df["v_sys_kms"] = v_sys
    df["vlos_rel_kms"] = df["vlos_kms"] - v_sys

    groups = np.array_split(df.index.to_numpy(), args.bins)

    df["kin_bin"] = -1
    rows = []

    # Bin edges are halfway between neighboring stars at bin boundaries.
    edges = [0.0]

    for left_group, right_group in zip(groups[:-1], groups[1:]):
        r_left = df.loc[left_group[-1], "R_pc"]
        r_right = df.loc[right_group[0], "R_pc"]
        edges.append(0.5 * (r_left + r_right))

    edges.append(float(df["R_pc"].max()) + 1.0e-6)

    for i, idx in enumerate(groups, start=1):
        b = df.loc[idx].copy()
        df.loc[idx, "kin_bin"] = i

        rin_edge = edges[i - 1]
        rout_edge = edges[i]

        rows.append(
            {
                "kin_bin": i,
                "n_stars": len(b),
                "rin_pc_edge": rin_edge,
                "rout_pc_edge": rout_edge,
                "rmid_pc_edge": 0.5 * (rin_edge + rout_edge),
                "rin_pc_data_min": float(b["R_pc"].min()),
                "rout_pc_data_max": float(b["R_pc"].max()),
                "rmid_pc_data": 0.5 * float(b["R_pc"].min() + b["R_pc"].max()),
                "vlos_weighted_mean_kms": weighted_mean(
                    b["vlos_kms"].to_numpy(),
                    b["verr_kms"].to_numpy(),
                ),
                "vlos_rel_weighted_mean_kms": weighted_mean(
                    b["vlos_rel_kms"].to_numpy(),
                    b["verr_kms"].to_numpy(),
                ),
                "vlos_raw_std_kms": float(np.std(b["vlos_kms"], ddof=1)),
                "verr_median_kms": float(b["verr_kms"].median()),
            }
        )

    bins_df = pd.DataFrame(rows)

    stars_out = outdir / "Segue1_Simon_stars_v2.csv"
    bins_out = outdir / "Segue1_Simon_kinematic_bins_v2.csv"

    df.to_csv(stars_out, index=False)
    bins_df.to_csv(bins_out, index=False)

    print(f"Loaded {len(df)} stars")
    print(f"Weighted systemic velocity = {v_sys:.3f} km/s")
    print()
    print(f"Saved {stars_out}")
    print(f"Saved {bins_out}")
    print()
    print(bins_df.to_string(index=False))


if __name__ == "__main__":
    main()
