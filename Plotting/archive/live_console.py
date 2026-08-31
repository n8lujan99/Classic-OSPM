import sys
import time
import re
import glob
import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt

import pyqtgraph as pg

pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", (10, 10, 15))
pg.setConfigOption("foreground", "w")


# --------------------------------------------------
# Repo context
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
WHICH = REPO_ROOT / "which_galaxy"
if not WHICH.exists():
    raise FileNotFoundError("which_galaxy not found at repo root")

GALAXY = WHICH.read_text().strip()
if not GALAXY:
    raise ValueError("which_galaxy is empty")

PROFILE_DIR_DEFAULT = REPO_ROOT / "Data" / "Gal_Profiles" / GALAXY / "default"


# --------------------------------------------------
# CSV Finder
# --------------------------------------------------

def _timestamp_from_filename(path: str):
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(path))
    return int(m.group(1) + m.group(2)) if m else None


def find_latest_daemon_deck(profile_dir: Path, pattern="*daemon_deck*") -> str:
    c = glob.glob(str(profile_dir / pattern))
    if not c:
        raise FileNotFoundError(f"No daemon_deck found in {profile_dir}")

    ts = [(_timestamp_from_filename(p), p) for p in c]
    ts = [t for t in ts if t[0] is not None]
    return max(ts, key=lambda x: x[0])[1] if ts else max(c, key=os.path.getmtime)


# --------------------------------------------------
# Widgets
# --------------------------------------------------

class AcceptanceGauge(QtWidgets.QWidget):
    """
    Semicircle gauge.
    value in [0,1].
    Colors are diagnostic, not decorative.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = np.nan
        self.setMinimumHeight(220)

    def set_value(self, v):
        self.value = v
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fill background to match app
        painter.fillRect(self.rect(), QColor(10, 10, 15))

        w = self.width()
        h = self.height()

        cx = w * 0.5
        cy = h * 0.82
        r = min(w, h) * 0.38

        # background arc
        pen = QPen(QColor(70, 70, 80))
        pen.setWidth(10)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        rect = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)
        painter.drawArc(rect, 180 * 16, 180 * 16)

        if self.value is None or np.isnan(self.value):
            self._draw_text(painter, cx, cy, "NA", "Acceptance (100)")
            return

        v = float(np.clip(self.value, 0.0, 1.0))

        # Diagnostic coloring: acceptance is a control parameter.
        # Sweet zone around 0.2–0.5 for many proposal schemes.
        if v < 0.15:
            col = QColor(255, 60, 60)
        elif v <= 0.55:
            col = QColor(0, 220, 140)
        else:
            col = QColor(255, 170, 40)

        pen = QPen(col)
        pen.setWidth(10)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, 180 * 16, int(180 * v * 16))

        self._draw_text(painter, cx, cy, f"{v:.2f}", "Acceptance (100)")

    def _draw_text(self, painter, cx, cy, main, sub):
        painter.setPen(QColor(240, 240, 240))
        f = QFont()
        f.setPointSize(18)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(int(cx - 60), int(cy - 25), 120, 30, Qt.AlignmentFlag.AlignCenter, main)

        painter.setPen(QColor(170, 170, 180))
        f2 = QFont()
        f2.setPointSize(10)
        painter.setFont(f2)
        painter.drawText(int(cx - 90), int(cy + 5), 180, 20, Qt.AlignmentFlag.AlignCenter, sub)


class StatCard(QtWidgets.QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame { background-color: rgb(14,14,20); border: 1px solid rgb(40,40,55); border-radius: 10px; }"
        )
        self.v = QtWidgets.QVBoxLayout(self)
        self.v.setContentsMargins(12, 10, 12, 10)
        self.v.setSpacing(6)

        self.t = QtWidgets.QLabel(title)
        self.t.setStyleSheet("color: rgb(180,180,200); font-size: 11px;")
        self.v.addWidget(self.t)

        self.val = QtWidgets.QLabel("—")
        self.val.setStyleSheet("color: white; font-size: 18px; font-weight: 600;")
        self.v.addWidget(self.val)

    def set_value(self, s: str):
        self.val.setText(s)


# --------------------------------------------------
# Main Dashboard
# --------------------------------------------------

class LiveConsole(QtWidgets.QWidget):
    def __init__(self, profile_dir=None, csv_override=None, pattern="*daemon_deck*", refresh=2.0, auto_switch=True):
        super().__init__()

        self.setStyleSheet("background-color: rgb(10,10,15);")
        self.setWindowTitle("OSPM Live Convergence Console")
        self.resize(1700, 950)

        self.profile_dir = Path(profile_dir) if profile_dir else PROFILE_DIR_DEFAULT
        self.pattern = pattern
        self.auto_switch = bool(auto_switch)

        self.refresh_ms = int(max(0.2, float(refresh)) * 1000)

        self.csv_file = str(csv_override) if csv_override else find_latest_daemon_deck(self.profile_dir, self.pattern)
        self.csv_mtime = os.path.getmtime(self.csv_file) if os.path.exists(self.csv_file) else 0.0

        # incremental read state
        self.last_row = 0
        self.df = pd.DataFrame()
        self.best_running = None

        # ---------- layout ----------
        root = QtWidgets.QGridLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        self.setLayout(root)

        root.setColumnStretch(0, 4)
        root.setColumnStretch(1, 1)

        # plots left
        self.plot_chi = pg.PlotWidget(title="χ² vs Run")
        self.plot_chi.getPlotItem().setContentsMargins(10, 10, 10, 10)

        self.plot_land = pg.PlotWidget(title="χ² vs MBH")
        self.plot_land.getPlotItem().setContentsMargins(10, 10, 10, 10)

        root.addWidget(self.plot_chi, 0, 0, 2, 1)
        root.addWidget(self.plot_land, 2, 0, 1, 1)

        # instrumentation right
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        self.card_gal = StatCard("Galaxy")
        self.card_src = StatCard("Source")
        self.card_runs = StatCard("Total runs")
        self.card_best = StatCard("Best χ²")
        self.card_stag = StatCard("Runs since improvement")

        self.card_gal.set_value(GALAXY)
        self.card_src.set_value(Path(self.csv_file).name)

        self.gauge_acc = AcceptanceGauge()

        # progress bar for stagnation
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            "QProgressBar { background-color: rgb(14,14,20); border: 1px solid rgb(40,40,55); border-radius: 8px; }"
            "QProgressBar::chunk { background-color: rgb(0,200,255); border-radius: 8px; }"
        )
        self.progress.setFixedHeight(16)
        self.progress_max = 2000  # default, auto-expanded
        self.progress.setMaximum(self.progress_max)

        # pack right column
        right.addWidget(self.card_gal)
        right.addWidget(self.card_src)
        right.addWidget(self.gauge_acc)

        right.addWidget(self.card_runs)
        right.addWidget(self.card_best)
        right.addWidget(self.card_stag)
        right.addWidget(self.progress)

        right.addStretch(1)

        right_wrap = QtWidgets.QWidget()
        right_wrap.setLayout(right)
        root.addWidget(right_wrap, 0, 1, 3, 1)

        # timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(self.refresh_ms)

    # --------------------------------------------------

    def _maybe_switch_source(self):
        if not self.auto_switch:
            return

        try:
            latest = find_latest_daemon_deck(self.profile_dir, self.pattern)
            if latest != self.csv_file:
                self.csv_file = latest
                self.csv_mtime = os.path.getmtime(self.csv_file)
                self.last_row = 0
                self.df = pd.DataFrame()
                self.best_running = None
                self.card_src.set_value(Path(self.csv_file).name)
        except Exception:
            pass

    def _append_new_rows(self):
        # If file changed but same name, just keep appending.
        df_full = pd.read_csv(self.csv_file)
        if len(df_full) <= self.last_row:
            return

        new_rows = df_full.iloc[self.last_row:].copy()
        self.last_row = len(df_full)

        if self.df.empty:
            self.df = new_rows
        else:
            self.df = pd.concat([self.df, new_rows], ignore_index=True)

    def update_dashboard(self):
        try:
            if not os.path.exists(self.csv_file):
                self._maybe_switch_source()
                return

            # rotate if a newer deck appears
            self._maybe_switch_source()

            # append only new lines
            self._append_new_rows()

            if self.df.empty:
                return

            # sanitize
            self.df["chi2"] = pd.to_numeric(self.df["chi2"], errors="coerce")
            self.df = self.df.replace([np.inf, -np.inf], np.nan)
            self.df = self.df.dropna(subset=["chi2"])

            if len(self.df) < 10:
                return

            if "status" in self.df.columns:
                model_df = self.df[self.df["status"].astype(str) == "pass_full"].copy()
            else:
                model_df = self.df.copy()

            if len(model_df) < 1:
                return

            chi = model_df["chi2"].to_numpy()
            runs = np.arange(len(chi))

            self.best_running = np.minimum.accumulate(chi)
            best_val = float(self.best_running[-1])

            best_idx = int(np.where(self.best_running == best_val)[0][0])
            runs_since_improve = int(len(chi) - best_idx)

            if "status" in self.df.columns:
                tail = self.df["status"].tail(100)
                acc_rate = float(np.mean(tail == "pass_full")) if len(tail) else np.nan
            else:
                acc_rate = np.nan

            # ---------- plots ----------
            self.plot_chi.clear()
            self.plot_chi.plot(runs, chi, pen=(120, 140, 255, 45))
            self.plot_chi.plot(runs, self.best_running, pen=pg.mkPen((0, 220, 200), width=2))

            self.plot_land.clear()
            if "MBH" in model_df.columns:
                x = model_df["MBH"].to_numpy()
                self.plot_land.plot(
                    x, chi,
                    pen=None,
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=(220, 210, 90, 120),
                    symbolPen=None,
                )

            # ---------- cards ----------
            self.card_runs.set_value(str(len(self.df)))
            self.card_best.set_value(f"{best_val:.4f}")
            self.card_stag.set_value(str(runs_since_improve))

            # ---------- gauges ----------
            self.gauge_acc.set_value(acc_rate)

            # progress scale grows with you
            if runs_since_improve > self.progress_max:
                self.progress_max = int(runs_since_improve * 1.3)
                self.progress.setMaximum(self.progress_max)
            self.progress.setValue(min(runs_since_improve, self.progress_max))

        except Exception as e:
            print("Dashboard error:", e)


# --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile-dir", default=str(PROFILE_DIR_DEFAULT))
    p.add_argument("--csv", default=None)
    p.add_argument("--pattern", default="*daemon_deck*")
    p.add_argument("--refresh", type=float, default=2.0)
    p.add_argument("--no-auto-switch", action="store_true")
    return p.parse_args()


def run():
    args = parse_args()

    app = QtWidgets.QApplication(sys.argv)

    win = LiveConsole(
        profile_dir=args.profile_dir,
        csv_override=args.csv,
        pattern=args.pattern,
        refresh=args.refresh,
        auto_switch=(not args.no_auto_switch),
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
