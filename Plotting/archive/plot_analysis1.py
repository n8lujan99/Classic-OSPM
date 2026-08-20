# plot_analysis1.py
# simple script to plot daemon_deck results for OSPM parameter sweeps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

DECK_ROOT = "/home/n8lujan99/research/github/Nates-OSPM/Data/Gal_Profiles/Segue1/default"
DECK_NAME = "daemon_deck.csv"
#DECK_NAME = "daemon_deck_nohalo"
#DECK_NAME = "daemon_deck_smallerbhandlargerr_0"
#DECK_NAME = "Draco_daemon_deck_TESTING"

plt.style.use("dark_background")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--include-diagnostics", action="store_true")
    return p.parse_args()

def resolve_daemon_deck(root: str, stem: str) -> str:
    if not stem.endswith(".csv"):
        stem += ".csv"
    path = os.path.join(root, stem)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Daemon deck not found:\n  {path}")
    return path

# ------------------------------------------------------------
# Resolve deck
# ------------------------------------------------------------
args = parse_args()
csv_file = resolve_daemon_deck(DECK_ROOT, DECK_NAME)
print(f"[PLOT] using {csv_file}")


# ------------------------------------------------------------
# Load + clean
# ------------------------------------------------------------
df = pd.read_csv(csv_file)

df["chi2"] = pd.to_numeric(df["chi2"], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["chi2"])

if not args.include_diagnostics and "status" in df.columns:
    df = df[df["status"].astype(str) == "pass_full"].copy()

for c in ("MBH", "rho_s", "r_s"):
    if c not in df.columns:
        raise KeyError(f"Missing column: {c}")

# ------------------------------------------------------------
# Split populations
# ------------------------------------------------------------
bh0  = df["MBH"] == 0
blue = df[bh0]
teal = df[~bh0]

best = df["chi2"].min()
zoom = df[df["chi2"] < best * 1.2]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(18, 10))

params = [
    ("MBH",   r"$M_{\rm BH}$"),
    ("rho_s", r"$\rho_s$"),
    ("r_s",   r"$r_s$")
]

for j, (x, label) in enumerate(params):
    ax[0, j].scatter(teal[x], teal["chi2"], s=10, c="#39EB33")
    ax[0, j].scatter(blue[x], blue["chi2"], s=16, c="#4FA3FF")
    ax[0, j].set_xscale("log")
    ax[0, j].set_title(rf"$\chi^2$ vs {label}")

    ax[1, j].scatter(zoom[x], zoom["chi2"], s=20, c="#39EB33")
    ax[1, j].set_xscale("log")
    ax[1, j].set_title(f"Zoomed {label}")

y = df["chi2"].values
ypad = 0.05 * (y.max() - y.min())
for a in ax[0]:
    a.set_ylim(y.min() - ypad, y.max() + ypad)

z = zoom["chi2"].values
zpad = 0.05 * (z.max() - z.min())
for a in ax[1]:
    a.set_ylim(z.min() - zpad, z.max() + zpad)

plt.tight_layout()
plt.show()


