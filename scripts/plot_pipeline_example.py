"""Walk-through figure of the single-node INDOP pipeline on one simulated session.

Panels: (a) 100 ms power timeline with anchor window, noise floor and
threshold; (b) HMM posterior P(occupied) vs truth; (c) anchor-window
histogram with the fitted GMM.

Usage:  python scripts/plot_pipeline_example.py [--power results/npz/Web_power.npy]
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import norm

from common import AXIS, CRITICAL, GRID, INK2, MUTED, SERIES, default_scenario, save, setup_style
import matplotlib.pyplot as plt

from indop import run_indop
from sim.synthetic import simulate_power


def main(power_file: str | None) -> None:
    truth = None
    if power_file:
        power = np.load(power_file).astype(float)
    else:
        sc = default_scenario(seed=5)
        sim = simulate_power(sc, minutes=5, duty=0.3, seed=5)
        i = sc.node_names.index("Moran")
        power, truth = sim["power"][i], sim["truth"][i]
    res = run_indop(power, "example")
    g = res.gmm
    t = np.arange(len(power)) * 0.1

    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.35, wspace=0.28)
    a = fig.add_subplot(gs[0, :2])
    a.axvspan(0, g.anchor_chunks * 0.1, color=GRID, alpha=0.8, lw=0)
    a.text(g.anchor_chunks * 0.05, power.max(), f"anchor window\n{g.anchor_chunks * 0.1:.0f} s",
           ha="center", va="top", fontsize=8, color=INK2)
    a.plot(t, power, color=SERIES[0], lw=0.6)
    a.axhline(g.mu_noise, color=AXIS, lw=1.2, ls=(0, (4, 2)))
    a.axhline(g.threshold, color=CRITICAL, lw=1.4)
    a.text(t[-1], g.threshold, "  T", va="center", color=CRITICAL, fontsize=9)
    a.text(t[-1], g.mu_noise, "  μ_noise", va="center", color=MUTED, fontsize=8)
    a.set_xlim(0, t[-1])
    a.set_ylabel("P[n] (dBX)")
    a.set_title("(a) 100 ms power timeline — threshold learned from the anchor window",
                loc="left", fontsize=11)

    h = fig.add_subplot(gs[0, 2])
    anchor = power[:g.anchor_chunks]
    h.hist(anchor, bins=80, density=True, color=SERIES[0], alpha=0.35, orientation="horizontal")
    y = np.linspace(anchor.min(), anchor.max(), 400)
    for k in range(g.k):
        h.plot(g.weights[k] * norm.pdf(y, g.means[k], g.stds[k]), y, color=SERIES[k % 8 + 1]
               if k else SERIES[0], lw=1.5, label="noise" if k == 0 else f"signal-{k}")
    h.axhline(g.threshold, color=CRITICAL, lw=1.4)
    h.set_ylim(a.get_ylim())
    h.set_xlabel("density")
    h.set_title(f"(c) anchor GMM, K* = {g.k} (BIC)", loc="left", fontsize=11)
    h.legend(fontsize=7, loc="upper right")

    b = fig.add_subplot(gs[1, :])
    z0, z1 = 100.0, 160.0  # zoom: 60 s are readable at 100 ms resolution
    m = (t >= z0) & (t < z1)
    a.axvspan(z0, z1, ymax=0.04, color=SERIES[1], lw=0)
    if truth is not None:
        b.fill_between(t[m], 0, truth[m].astype(float), step="mid", color=SERIES[0], alpha=0.25,
                       lw=0, label="truth (sim)")
    b.step(t[m], res.hmm.posterior[m], where="mid", color=SERIES[1], lw=1.2, label="HMM P(occupied)")
    b.set_xlim(z0, z1)
    b.set_ylim(-0.05, 1.05)
    b.set_xlabel("Time (s)")
    b.set_ylabel("P(occupied)")
    b.set_title(f"(b) HMM refinement, zoom {z0:.0f}–{z1:.0f} s — session soft {100*res.hmm.soft_occupancy:.1f}%, "
                f"GMM {100*res.occupancy_gmm:.1f}%"
                + (f", truth {100*truth.mean():.1f}%" if truth is not None else ""),
                loc="left", fontsize=11)
    b.legend(loc="upper right", fontsize=8, ncols=2, frameon=True, facecolor="white",
             edgecolor="none")
    save(fig, "fig07_pipeline_walkthrough", synthetic=truth is not None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", default=None)
    setup_style()
    main(ap.parse_args().power)
