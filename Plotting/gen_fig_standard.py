#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

MAROON = "#A30323"
INSET_BG = "#f0f0f0"

# fixed physical scale colors
DEFAULT_SCALE_COLORS = {
    1.0:    "#6820B9",   # purple
    10.0:   "#0A99F1",   # blue
    100.0:  "#1A9850",   # green
    250.0:  "#F1963B",   # orange
    500.0:  "#E01515",   # red
    1000.0: "#D8D7D7",   # dark gray
}

# SMBH SOI colors
DEFAULT_SOI_COLORS = {
    1.0e5: "#269E1B",
    5.0e5: "#D9A002",
    1.0e6: "#D62A2A",
}

# G in pc (km/s)^2 / Msun
G_PC_KMS2_PER_MSUN = 4.30091e-3


def as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_float_list(text):
    if text is None:
        return []
    s = str(text).strip()
    if s == "":
        return []
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def sphere_of_influence_pc(mbh_msun, sigma_kms):
    sigma_kms = float(sigma_kms)
    if sigma_kms <= 0:
        raise ValueError("sigma_kms must be positive")
    return G_PC_KMS2_PER_MSUN * float(mbh_msun) / (sigma_kms ** 2)


def load_stars(stars_csv, require_vlos=True):
    df = pd.read_csv(stars_csv)
    required = {"x_pc", "y_pc"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Star file missing columns: {sorted(missing)}")
    if require_vlos and "has_vlos" in df.columns:
        mask = df["has_vlos"].map(as_bool).to_numpy(bool)
        df = df.loc[mask].copy()
    x = pd.to_numeric(df["x_pc"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["y_pc"], errors="coerce").to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) == 0:
        raise ValueError("No finite projected star positions found.")
    r = np.sqrt(x * x + y * y)
    return x, y, r


def get_limit(x, y, extra_radii=None, pad_fraction=0.06):
    vals = [np.nanmax(np.abs(x)), np.nanmax(np.abs(y))]
    if extra_radii is not None and len(extra_radii) > 0:
        vals.append(np.nanmax(np.asarray(extra_radii, float)))
    lim = np.nanmax(vals)
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return lim * (1.0 + pad_fraction)


def draw_circle(ax, radius_pc, color="black", lw=1.2, ls="-", alpha=0.8, zorder=1):
    t = np.linspace(0.0, 2.0 * np.pi, 800)
    x = radius_pc * np.cos(t)
    y = radius_pc * np.sin(t)
    ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)


def add_scale_circles(ax, radii, color_map, lw=1.4, alpha=0.85, zorder=1):
    for rr in radii:
        color = color_map.get(rr, "0.4")
        draw_circle(ax, rr, color=color, lw=lw, ls="-", alpha=alpha, zorder=zorder)


def add_soi_circles(ax, masses, sigma_kms, color_map, lw=1.2, alpha=0.95, zorder=3):
    soi_info = []
    for mbh in masses:
        rr = sphere_of_influence_pc(mbh, sigma_kms)
        color = color_map.get(mbh, "black")
        draw_circle(ax, rr, color=color, lw=lw, ls="--", alpha=alpha, zorder=zorder)
        soi_info.append((mbh, rr, color))
    return soi_info


def make_scale_legend_handles(radii, color_map):
    handles = []
    for rr in radii:
        handles.append(Line2D([0], [0], color=color_map.get(rr, "0.4"), lw=2.0, ls="-", label=f"{rr:g} pc"))
    return handles


def make_soi_legend_handles(soi_info):
    handles = []
    for mbh, rr, color in soi_info:
        label = rf"$M_{{\rm BH}}={mbh:.0e}\,M_\odot$ : $r_{{\rm soi}}={rr:.1f}\,$pc"
        handles.append(Line2D([0], [0], color=color, lw=1.8, ls="--", label=label))
    return handles


def style_axis(ax, show_axes=True):
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    if show_axes:
        ax.tick_params(direction="in", top=True, right=True, length=4, width=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
    else:
        ax.set_axis_off()


def plot_main_and_inset(x, y, out_path, galaxy_name="Galaxy", scale_radii=None, zoom_half_size=100.0, star_size=28.0, star_alpha=0.82, require_vlos=True, show_main_axes=False, show_inset_axes=False, show_soi=True, sigma_kms=10.0, soi_masses=None, title_on_main=False):
    if scale_radii is None:
        scale_radii = [1.0, 10.0, 100.0, 250.0, 500.0, 1000.0]
    if soi_masses is None:
        soi_masses = [1.0e5, 5.0e5, 1.0e6]

    scale_radii = sorted(set(float(v) for v in scale_radii))
    soi_masses = sorted(set(float(v) for v in soi_masses))

    r_star_max = np.nanmax(np.sqrt(x * x + y * y))
    main_candidate_max = max(r_star_max, max(scale_radii))
    main_radii = [rr for rr in scale_radii if rr <= main_candidate_max * 1.15]
    main_lim = get_limit(x, y, extra_radii=main_radii, pad_fraction=0.08)
    main_shift_x = 140.0
    main_shift_y = -110.0

    fig = plt.figure(figsize=(8.0, 8.0), facecolor="white")
    ax = fig.add_axes([0.07, 0.10, 0.83, 0.83])
    ax.set_facecolor("white")

    ax.scatter(x, y, s=star_size, c=MAROON, alpha=star_alpha, edgecolors="none", zorder=2)
    add_scale_circles(ax, main_radii, DEFAULT_SCALE_COLORS, lw=1.6, alpha=0.85, zorder=1)

    zoom_radius = float(zoom_half_size)
    zoom_color = DEFAULT_SCALE_COLORS.get(100.0, "#1A9850")
    draw_circle(ax, zoom_radius, color=zoom_color, lw=2.2, ls="-", alpha=1.0, zorder=4)

    ax.set_xlim(-main_lim + main_shift_x, main_lim + main_shift_x)
    ax.set_ylim(-main_lim + main_shift_y, main_lim + main_shift_y)

    if show_main_axes:
        ax.set_xlabel(r"$x\ [{\rm pc}]$")
        ax.set_ylabel(r"$y\ [{\rm pc}]$")
        if title_on_main:
            ax.set_title(galaxy_name, fontsize=18)
    else:
        ax.text(0.02, 0.98, galaxy_name, transform=ax.transAxes, ha="left", va="top", fontsize=16)

    style_axis(ax, show_axes=show_main_axes)

    axins = ax.inset_axes([0.61, 0.02, 0.38, 0.38], transform=ax.transAxes, zorder=20)
    axins.patch.set_alpha(0.0)
    for spine in axins.spines.values():
        spine.set_visible(False)

    inset_bg_circle = Circle((0.0, 0.0), zoom_half_size, facecolor=INSET_BG, edgecolor=DEFAULT_SCALE_COLORS.get(100.0, "#1A9850"), linewidth=2.0, alpha=1.0, zorder=0)
    axins.add_patch(inset_bg_circle)

    inset_mask = np.sqrt(x * x + y * y) <= zoom_half_size

    axins.scatter(x[inset_mask], y[inset_mask], s=max(14.0, 0.85 * star_size), c=MAROON, alpha=0.90, edgecolors="none", zorder=3)

    inset_radii = [rr for rr in scale_radii if rr < zoom_half_size]
    add_scale_circles(axins, inset_radii, DEFAULT_SCALE_COLORS, lw=1.5, alpha=0.90, zorder=1)

    soi_info = []
    if show_soi:
        soi_info = add_soi_circles(axins, masses=soi_masses, sigma_kms=sigma_kms, color_map=DEFAULT_SOI_COLORS, lw=1.3, alpha=0.95, zorder=2)

    axins.set_xlim(-zoom_half_size, zoom_half_size)
    axins.set_ylim(-zoom_half_size, zoom_half_size)
    style_axis(axins, show_axes=False)

    green_r = float(zoom_half_size)

    # exact 0 and 180 would overlap into one line, so keep a tiny offset
    eps = np.deg2rad(12.0)

    theta_main_top = 0.0 + eps
    theta_main_bot = 0.0 - eps
    theta_inset_top = np.pi - eps
    theta_inset_bot = -np.pi + eps

    xA1 = green_r * np.cos(theta_main_top)
    yA1 = green_r * np.sin(theta_main_top)
    xA2 = green_r * np.cos(theta_main_bot)
    yA2 = green_r * np.sin(theta_main_bot)
    xB1 = green_r * np.cos(theta_inset_top)
    yB1 = green_r * np.sin(theta_inset_top)
    xB2 = green_r * np.cos(theta_inset_bot)
    yB2 = green_r * np.sin(theta_inset_bot)

    con1 = ConnectionPatch(xyA=(xA1, yA1), coordsA=ax.transData, xyB=(xB1, yB1), coordsB=axins.transData, color="0.55", lw=1.4, ls="--", zorder=3)
    con2 = ConnectionPatch(xyA=(xA2, yA2), coordsA=ax.transData, xyB=(xB2, yB2), coordsB=axins.transData, color="0.55", lw=1.4, ls="--", zorder=3)

    fig.add_artist(con1)
    fig.add_artist(con2)

    scale_handles = make_scale_legend_handles(main_radii, DEFAULT_SCALE_COLORS)
    leg1 = ax.legend(handles=scale_handles, title="Scale circles", loc="lower left", bbox_to_anchor=(0.02, 0.03), frameon=False, fontsize=9, title_fontsize=10)
    ax.add_artist(leg1)

    if show_soi and len(soi_info) > 0:
        soi_handles = make_soi_legend_handles(soi_info)
        leg2 = ax.legend(handles=soi_handles, title=rf"SMBH SOI, $\sigma={sigma_kms:g}\,$km s$^{{-1}}$", loc="lower left", bbox_to_anchor=(0.28, 0.005), frameon=False, fontsize=7.5, title_fontsize=8.2)
        ax.add_artist(leg2)
    fig.savefig(out_path, dpi=400, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"N plotted stars: {len(x)}")
    print(f"N inset stars: {int(np.count_nonzero(inset_mask))}")
    print(f"Main scale radii drawn [pc]: {main_radii}")
    print(f"Inset scale radii drawn [pc]: {inset_radii}")
    if show_soi:
        print("SMBH SOI radii [pc]:")
        for mbh, rr, _ in soi_info:
            print(f"  MBH={mbh:.0e} Msun -> r_soi={rr:.4f} pc")


def build_parser():
    p = argparse.ArgumentParser(description="Paper-style galaxy figure with fixed scale circles and zoom inset.")
    p.add_argument("--stars", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--galaxy-name", default="Galaxy")
    p.add_argument("--scale-radii", default="1,10,100,250,500,1000", help="Comma-separated physical scale radii in pc.")
    p.add_argument("--zoom-half-size", type=float, default=100.0, help="Radius of circular zoom inset in pc. 100 means the inset shows r <= 100 pc.")
    p.add_argument("--star-size", type=float, default=28.0)
    p.add_argument("--star-alpha", type=float, default=0.82)
    p.add_argument("--all-stars", action="store_true", help="If set, do not filter on has_vlos.")
    p.add_argument("--show-main-axes", action="store_true")
    p.add_argument("--show-inset-axes", action="store_true")
    p.add_argument("--title-on-main", action="store_true")
    p.add_argument("--show-soi", action="store_true")
    p.add_argument("--sigma-kms", type=float, default=10.0, help="Velocity dispersion used in r_soi = G M_BH / sigma^2.")
    p.add_argument("--soi-masses", default="1e5,5e5,1e6", help="Comma-separated BH masses in Msun.")
    return p


def main():
    args = build_parser().parse_args()
    stars_path = Path(args.stars)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x, y, r = load_stars(stars_path, require_vlos=not args.all_stars)
    scale_radii = parse_float_list(args.scale_radii)
    soi_masses = parse_float_list(args.soi_masses)
    plot_main_and_inset(x=x, y=y, out_path=out_path, galaxy_name=args.galaxy_name, scale_radii=scale_radii, zoom_half_size=args.zoom_half_size, star_size=args.star_size, star_alpha=args.star_alpha, require_vlos=not args.all_stars, show_main_axes=args.show_main_axes, show_inset_axes=args.show_inset_axes, show_soi=args.show_soi, sigma_kms=args.sigma_kms, soi_masses=soi_masses, title_on_main=args.title_on_main)


if __name__ == "__main__":
    main()
