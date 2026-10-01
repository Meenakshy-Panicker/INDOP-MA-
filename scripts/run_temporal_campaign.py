"""Extension 1: 24-hour multi-session campaign - can INDOP see the 4-7 am carrier idle window?

Simulates one 20-minute session per hour on all 7 agents (the proposed
morning / afternoon / evening / night campaign, at hourly granularity) under
a diurnal carrier load with a cooperating-carrier idle window 04:00-07:00
(as reported by OpenZMS / POWDER-RDZ). Each session is processed exactly as
on the node: anchor GMM + HMM, no calibration.

An hour is declared idle network-wide when every agent's HMM occupancy is
below IDLE_PCT. The figure shows the node x hour occupancy heat-map and the
detected idle window against the truth.

SYNTHETIC. On real data, run scripts/run_multinode_campaign.py on each node
per session and point --npz-dir at the collected results; the same heat-map
is produced from the NPZ files (file names <node>_<HH>.npz).

Usage:  python scripts/run_temporal_campaign.py [--minutes 20] [--npz-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import (AXIS, GRID, INK2, MUTED, RESULTS_DIR, SEQ_CMAP, SERIES, default_scenario,
                    save, setup_style)
import matplotlib.pyplot as plt

from indop import run_indop
from sim.synthetic import carrier_schedule, simulate_power

IDLE_PCT = 3.0
HOURS = np.arange(24)


def simulate(minutes: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    sc = default_scenario(seed=11, unit_spread_db=0.3)
    est = np.zeros((len(sc.node_names), 24))
    tru = np.zeros_like(est)
    for h in HOURS:
        sim = simulate_power(sc, minutes=minutes, duty=carrier_schedule, start_hour=float(h),
                             seed=500 + int(h))
        for i, n in enumerate(sc.node_names):
            est[i, h] = 100 * run_indop(sim["power"][i], n).occupancy_hmm_hard
        tru[:, h] = 100 * sim["truth"].mean(axis=1)
        print(f"hour {h:02d}: INDOP max {est[:, h].max():5.1f}%  truth max {tru[:, h].max():5.1f}%")
    return sc.node_names, est, tru


def from_npz(npz_dir: Path) -> tuple[list[str], np.ndarray, None]:
    from indop.pipeline import load_result_npz
    from indop_ma.topology import load_nodes

    names = [n.name for n in load_nodes()]
    est = np.full((len(names), 24), np.nan)
    for f in npz_dir.glob("*_[0-2][0-9].npz"):
        node, hh = f.stem.rsplit("_", 1)
        if node in names:
            s = load_result_npz(str(f))
            est[names.index(node), int(hh)] = 100 * (s["occupancy_hmm_hard"] or s["occupancy_gmm"])
    return names, est, None


def plot(names, est, tru, minutes: float = 20) -> None:
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={"height_ratios": [1.5, 1]},
                               sharex=True)
    im = a.imshow(est, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=max(40, np.nanmax(est)),
                  extent=[-0.5, 23.5, len(names) - 0.5, -0.5])
    a.set_yticks(range(len(names)), names)
    a.set_title(f"INDOP occupancy per node and hour ({minutes:g}-min session each hour)", loc="left")
    a.grid(False)
    cb = fig.colorbar(im, ax=a, pad=0.01, shrink=0.9)
    cb.set_label("Occupancy (%)", color=INK2)
    cb.outline.set_visible(False)
    idle = np.nanmax(est, axis=0) < IDLE_PCT
    for h in np.where(idle)[0]:
        a.add_patch(plt.Rectangle((h - 0.5, -0.5), 1, len(names), fill=False, ec=SERIES[1], lw=2))

    mean_est = np.nanmean(est, axis=0)
    b.axvspan(3.5, 6.5, color=GRID, alpha=0.8, lw=0)
    b.text(5, 6, "true carrier\nidle 04–07", ha="center", fontsize=8, color=INK2)
    if tru is not None:
        b.plot(HOURS, tru.mean(0), color=AXIS, lw=2, ls=(0, (4, 2)), label="Truth (node mean)")
    b.plot(HOURS, mean_est, color=SERIES[0], marker="o", ms=5, mec="white", mew=1,
           label="INDOP (node mean)")
    b.scatter(HOURS[idle], mean_est[idle], s=90, facecolor="none", edgecolor=SERIES[1], lw=2,
              zorder=4, label=f"Detected idle (all nodes < {IDLE_PCT:g}%)")
    b.set_xticks(HOURS, [f"{h:02d}" for h in HOURS])
    b.set_xlabel("Hour of day")
    b.set_ylabel("Occupancy (%)")
    b.legend(loc="upper left", fontsize=8, ncols=3)
    b.set_ylim(bottom=-1)
    fig.text(0.01, -0.02, "Outlined columns = hours INDOP declares idle at every node "
             "simultaneously; that is when OpenZMS could issue secondary grants.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "fig15_temporal_idle_window", synthetic=tru is not None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--npz-dir", type=Path, default=None)
    a = ap.parse_args()
    setup_style()
    names, est, tru = from_npz(a.npz_dir) if a.npz_dir else simulate(a.minutes)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / ("temporal_campaign.csv" if a.npz_dir else
                             "temporal_campaign_synthetic.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node"] + [f"h{h:02d}" for h in HOURS])
        for n, row in zip(names, est):
            w.writerow([n] + [f"{v:.2f}" for v in row])
    plot(names, est, tru, a.minutes)
