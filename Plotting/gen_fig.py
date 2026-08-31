#!/usr/bin/env python3
"""
gen_fig.py

Paper-style projected star figure.

Expected star CSV columns:
    x_pc, y_pc, r_pc, has_vlos

Expected surface-brightness CSV columns:
    R_inner_pc, R_outer_pc

Optional surface-brightness CSV column:
    q_axis_ratio
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MAROON = "#7A0019"


def as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_index_list(text):
    if text is None or str(text).strip() == "":
        return []
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if item == "":
            continue
        out.append(int(item))
    return out


def make_min_count_bins(r_pc, min_per_bin=20, drop_partial=False):
    r = np.sort(np.asarray(r_pc, float))
    r = r[np.isfinite(r)]

    if len(r) == 0:
        raise ValueError("No finite radii available for kinematic bins.")

    edges = [r[0]]
    rows = []
    i = 0
    bin_id = 0

    while i < len(r):
        j = min(i + min_per_bin, len(r))
        nbin = j - i

        if drop_partial and nbin < min_per_bin:
            break

        if nbin <= 0:
            break

        if j < len(r):
            edge_out = 0.5 * (r[j - 1] + r[j])
        else:
            edge_out = r[j - 1]

        if edge_out <= edges[-1]:
            edge_out = edges[-1] + max(abs(edges[-1]), 1.0) * 1e-9

        edges.append(edge_out)

        rows.append({
            "bin_id": bin_id,
            "R_inner_pc": edges[-2],
            "R_outer_pc": edges[-1],
            "R_mid_pc": 0.5 * (edges[-2] + edges[-1]),
            "N_vlos": nbin,
        })

        i = j
        bin_id += 1

    return np.asarray(edges, float), pd.DataFrame(rows)


def finite_edges_from_surface_brightness(sb):
    required = {"R_inner_pc", "R_outer_pc"}
    missing = required - set(sb.columns)

    if missing:
        raise KeyError(f"Surface-brightness CSV missing columns: {sorted(missing)}")

    edges = np.unique(
        np.r_[
            sb["R_inner_pc"].to_numpy(float),
            sb["R_outer_pc"].to_numpy(float),
        ]
    )

    edges = edges[np.isfinite(edges)]
    edges = np.sort(edges)

    if len(edges) < 2:
        raise ValueError("Surface-brightness file does not contain enough finite radial edges.")

    return edges


def draw_curve(ax, radius_pc, q=1.0, use_ellipse=False, **kwargs):
    t = np.linspace(0.0, 2.0 * np.pi, 800)

    if use_ellipse:
        x = radius_pc * np.cos(t)
        y = q * radius_pc * np.sin(t)
    else:
        x = radius_pc * np.cos(t)
        y = radius_pc * np.sin(t)

    ax.plot(x, y, **kwargs)


def label_curve(ax, radius_pc, text, angle_deg=35.0, q=1.0, use_ellipse=False,
                color="black", fontsize=9, rotation_mode="tangent", **kwargs):
    ang = np.deg2rad(angle_deg)

    if use_ellipse:
        x = radius_pc * np.cos(ang)
        y = q * radius_pc * np.sin(ang)
    else:
        x = radius_pc * np.cos(ang)
        y = radius_pc * np.sin(ang)

    if rotation_mode == "tangent":
        rotation = angle_deg + 90.0
    else:
        rotation = 0.0

    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=fontsize,
        ha="center",
        va="center",
        rotation=rotation,
        rotation_mode="anchor",
        bbox=dict(
            boxstyle="round,pad=0.10",
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
        ),
        zorder=6,
    )

def set_limits(ax, x, y, radius_edges, pad_fraction=0.08, xlim=None, ylim=None):
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None and ylim is not None:
        return

    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    finite_r = radius_edges[np.isfinite(radius_edges)]

    candidates = []
    if finite_x.size:
        candidates.extend(np.abs(finite_x))
    if finite_y.size:
        candidates.extend(np.abs(finite_y))
    if finite_r.size:
        candidates.extend(np.abs(finite_r))

    limit = float(np.nanmax(candidates)) if candidates else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    limit *= (1.0 + pad_fraction)

    if xlim is None:
        ax.set_xlim(-limit, limit)
    if ylim is None:
        ax.set_ylim(-limit, limit)


def choose_indices(n, step=1, include_last=True):
    if n <= 0:
        return []
    idx = list(range(0, n, max(1, step)))
    if include_last and (n - 1) not in idx:
        idx.append(n - 1)
    idx = sorted(set(i for i in idx if 0 <= i < n))
    return idx


def gen_fig(args):
    sb_path = Path(args.surface_brightness)
    stars_path = Path(args.stars)
    out_path = Path(args.out)

    if not sb_path.exists():
        raise FileNotFoundError(f"Surface-brightness file not found: {sb_path}")
    if not stars_path.exists():
        raise FileNotFoundError(f"Star file not found: {stars_path}")

    sb = pd.read_csv(sb_path)
    stars = pd.read_csv(stars_path)

    required_stars = {"x_pc", "y_pc", "r_pc", "has_vlos"}
    missing_stars = required_stars - set(stars.columns)
    if missing_stars:
        raise KeyError(f"Star file missing columns: {sorted(missing_stars)}")

    q = 1.0
    if "q_axis_ratio" in sb.columns:
        qvals = pd.to_numeric(sb["q_axis_ratio"], errors="coerce")
        qvals = qvals[np.isfinite(qvals)]
        if len(qvals) > 0:
            q = float(qvals.iloc[0])

    has_vlos = stars["has_vlos"].map(as_bool).to_numpy(bool)
    vstars = stars.loc[has_vlos].copy()
    if len(vstars) == 0:
        raise ValueError("No stars with has_vlos=True.")

    x = vstars["x_pc"].to_numpy(float)
    y = vstars["y_pc"].to_numpy(float)
    r = vstars["r_pc"].to_numpy(float)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
    x = x[good]
    y = y[good]
    r = r[good]

    if len(r) == 0:
        raise ValueError("No finite projected star coordinates after filtering.")

    kin_edges, kin_bins = make_min_count_bins(
        r,
        min_per_bin=args.min_stars,
        drop_partial=args.drop_partial_bins,
    )

    r_keep_max = kin_edges[-1]
    keep = r <= r_keep_max

    x_plot = x[keep]
    y_plot = y[keep]
    r_plot = r[keep]

    sb_edges = finite_edges_from_surface_brightness(sb)

    if args.bins_out is not None:
        bins_out = Path(args.bins_out)
        bins_out.parent.mkdir(parents=True, exist_ok=True)
        kin_bins.to_csv(bins_out, index=False)
        print(f"Saved bins: {bins_out}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # stars on top
    ax.scatter(
        x_plot,
        y_plot,
        s=args.star_size,
        c=args.star_color,
        alpha=args.star_alpha,
        edgecolors="none",
        zorder=4,
    )

    # choose which rings to draw
    kin_draw_idx = choose_indices(len(kin_edges), step=args.kinematic_step, include_last=True)
    sb_draw_idx = choose_indices(len(sb_edges), step=args.surface_brightness_step, include_last=True)

    # draw kinematic rings
    if args.show_kinematic_bins:
        for i in kin_draw_idx:
            rr = kin_edges[i]
            draw_curve(
                ax,
                rr,
                q=q,
                use_ellipse=args.ellipses,
                color=args.kinematic_color,
                linewidth=args.kinematic_lw,
                linestyle="-",
                alpha=args.kinematic_alpha,
                zorder=1,
            )

    # draw surface-brightness rings
    if args.show_surface_brightness_bins:
        for i in sb_draw_idx:
            rr = sb_edges[i]
            draw_curve(
                ax,
                rr,
                q=q,
                use_ellipse=args.ellipses,
                color=args.surface_brightness_color,
                linewidth=args.surface_brightness_lw,
                linestyle="--",
                alpha=args.surface_brightness_alpha,
                zorder=2,
            )

    # label selected kinematic rings
    kin_label_idx = parse_index_list(args.label_kinematic)
    for i in kin_label_idx:
        if 0 <= i < len(kin_edges):
            rr = kin_edges[i]
            label_curve(
                ax,
                rr,
                f"{rr:.0f} pc",
                angle_deg=args.kinematic_label_angle,
                q=q,
                use_ellipse=args.ellipses,
                color=args.kinematic_color,
                fontsize=args.label_fontsize,
                ha="left",
                va="bottom",
            )

    # label selected surface-brightness rings
    sb_label_idx = parse_index_list(args.label_surface_brightness)
    for i in sb_label_idx:
        if 0 <= i < len(sb_edges):
            rr = sb_edges[i]
            label_curve(
                ax,
                rr,
                f"{rr:.0f} pc",
                angle_deg=args.surface_brightness_label_angle,
                q=q,
                use_ellipse=args.ellipses,
                color=args.surface_brightness_color,
                fontsize=args.label_fontsize,
                ha="right",
                va="bottom",
            )

    if args.show_core:
        ax.scatter(
            [0.0], [0.0],
            s=args.core_size,
            c=args.core_color,
            marker="+",
            linewidths=args.core_lw,
            zorder=5,
        )

    ax.set_aspect("equal", adjustable="box")

    xlim = tuple(args.xlim) if args.xlim is not None else None
    ylim = tuple(args.ylim) if args.ylim is not None else None

    set_limits(
        ax,
        x_plot,
        y_plot,
        np.r_[kin_edges, sb_edges],
        pad_fraction=args.pad_fraction,
        xlim=xlim,
        ylim=ylim,
    )

    # no grid
    ax.grid(False)

    if args.no_axes:
        ax.set_axis_off()
    else:
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        if args.title is not None:
            ax.set_title(args.title)
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(args.spine_lw)

    if args.legend:
        handles = []
        labels = []
        if args.show_kinematic_bins:
            handles.append(plt.Line2D([0], [0], color=args.kinematic_color, lw=args.kinematic_lw, ls="-"))
            labels.append("kinematic bins")
        if args.show_surface_brightness_bins:
            handles.append(plt.Line2D([0], [0], color=args.surface_brightness_color, lw=args.surface_brightness_lw, ls="--"))
            labels.append("surface-brightness bins")
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=args.star_color, markersize=6, lw=0))
        labels.append("stars")
        ax.legend(handles, labels, frameon=False, loc=args.legend_loc)

    fig.tight_layout(pad=args.tight_pad)
    fig.savefig(out_path, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"N vlos stars total: {len(vstars)}")
    print(f"N vlos stars plotted: {len(r_plot)}")
    print(f"N kinematic bins total: {len(kin_bins)}")
    print(f"N surface-brightness bins total: {len(sb_edges) - 1}")
    print(f"Kinematic rings drawn: {kin_draw_idx}")
    print(f"Surface-brightness rings drawn: {sb_draw_idx}")
    print(f"q_axis_ratio read from profile: {q}")


def build_parser():
    p = argparse.ArgumentParser(description="Generate paper-style projected galaxy star maps.")

    p.add_argument("--surface-brightness", required=True)
    p.add_argument("--stars", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--min-stars", type=int, default=20)
    p.add_argument("--drop-partial-bins", action="store_true")
    p.add_argument("--bins-out", default=None)

    p.add_argument("--figsize", type=float, default=5.0)
    p.add_argument("--dpi", type=int, default=400)

    p.add_argument("--star-color", default=MAROON)
    p.add_argument("--star-size", type=float, default=18.0)
    p.add_argument("--star-alpha", type=float, default=0.90)

    p.add_argument("--show-core", action="store_true")
    p.add_argument("--core-color", default="black")
    p.add_argument("--core-size", type=float, default=80.0)
    p.add_argument("--core-lw", type=float, default=1.5)

    p.add_argument("--show-kinematic-bins", action="store_true")
    p.add_argument("--kinematic-color", default="black")
    p.add_argument("--kinematic-lw", type=float, default=1.0)
    p.add_argument("--kinematic-alpha", type=float, default=0.55)
    p.add_argument("--kinematic-step", type=int, default=2)
    p.add_argument("--label-kinematic", default="0,2,4")
    p.add_argument("--kinematic-label-angle", type=float, default=28.0)

    p.add_argument("--show-surface-brightness-bins", action="store_true")
    p.add_argument("--surface-brightness-color", default="0.45")
    p.add_argument("--surface-brightness-lw", type=float, default=0.9)
    p.add_argument("--surface-brightness-alpha", type=float, default=0.50)
    p.add_argument("--surface-brightness-step", type=int, default=4)
    p.add_argument("--label-surface-brightness", default="0,4,8")
    p.add_argument("--surface-brightness-label-angle", type=float, default=62.0)

    p.add_argument("--label-fontsize", type=float, default=9.0)

    p.add_argument("--ellipses", action="store_true",
                   help="Use q_axis_ratio from the surface-brightness file. Default is circular rings.")

    p.add_argument("--xlim", nargs=2, type=float, default=None)
    p.add_argument("--ylim", nargs=2, type=float, default=None)
    p.add_argument("--pad-fraction", type=float, default=0.08)

    p.add_argument("--title", default=None)
    p.add_argument("--xlabel", default=r"$x\ \mathrm{[pc]}$")
    p.add_argument("--ylabel", default=r"$y\ \mathrm{[pc]}$")
    p.add_argument("--no-axes", action="store_true")
    p.add_argument("--legend", action="store_true")
    p.add_argument("--legend-loc", default="best")
    p.add_argument("--spine-lw", type=float, default=1.0)
    p.add_argument("--tight-pad", type=float, default=0.05)

    return p


def main():
    args = build_parser().parse_args()
    gen_fig(args)


if __name__ == "__main__":
    main()