"""Shared helpers for experiment scripts: paths, plot style, default scenario."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indop_ma.topology import load_config, load_nodes, node_xy, to_local_xy  # noqa: E402
from sim.synthetic import make_scenario  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
RESULTS_DIR = ROOT / "results"
PAPER_DIR = ROOT / "data" / "paper_results"

# Categorical palette (fixed order, validated for CVD separation on adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "indop_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SYNTH_NOTE = "SYNTHETIC — simulated scenario, not POWDER measurements"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 160,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def save(fig, name: str, synthetic: bool = False) -> Path:
    if synthetic:
        fig.text(0.995, 0.005, SYNTH_NOTE, ha="right", va="bottom", fontsize=7,
                 color=MUTED, style="italic")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def campaign_geometry():
    """Nodes, (n, 2) metre coordinates, origin and the synthetic transmitter position."""
    nodes = load_nodes()
    xy, lat0, lon0 = node_xy(nodes)
    tx = load_config()["synthetic_transmitter"]
    txy = np.array(to_local_xy(tx["lat"], tx["lon"], lat0, lon0))
    return nodes, xy, (lat0, lon0), txy


def default_scenario(seed: int = 1, **kw):
    nodes, xy, _, txy = campaign_geometry()
    kw.setdefault("shadow_sigma_db", 1.5)
    return make_scenario([n.name for n in nodes], xy, txy, seed=seed, **kw)


def read_csv(name: str) -> np.ndarray:
    return np.genfromtxt(PAPER_DIR / name, delimiter=",", names=True, dtype=None, encoding="utf-8")
