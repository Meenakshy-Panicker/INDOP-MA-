"""Figures built from the published (real-measurement) results in data/paper_results/.

Usage:  python scripts/plot_paper_figures.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from common import (AXIS, CRITICAL, GRID, INK, INK2, MUTED, SEQ_CMAP, SERIES,
                    campaign_geometry, read_csv, save, setup_style)
import matplotlib.pyplot as plt

TARGET_PP = 2.5


def fig_multinode() -> None:
    t = read_csv("table3_multinode.csv")
    order = np.argsort(t["gt_pct"])
    names, indop, gt, gap = (t["node"][order], t["indop_pct"][order],
                             t["gt_pct"][order], t["gap_pp"][order])
    exact = t["exact_calibration"][order] == "yes"
    y = np.arange(len(names))

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for i in y:
        a.plot([gt[i], indop[i]], [i, i], color=AXIS, lw=2, zorder=1)
    a.scatter(gt, y, s=70, color=SERIES[0], zorder=3, label="Physics GT", edgecolor="white", lw=1.5)
    a.scatter(indop, y, s=70, color=SERIES[1], zorder=3, label="INDOP (no calibration)",
              edgecolor="white", lw=1.5, marker="D")
    a.set_yticks(y, [f"{n}{' †' if e else ''}" for n, e in zip(names, exact)])
    a.set_xlabel("Occupancy (%)")
    a.set_title("Occupancy per node, CBRS 3595 MHz", loc="left")
    a.legend(loc="lower right")
    a.grid(axis="y", visible=False)

    b.axvspan(-TARGET_PP, TARGET_PP, color=GRID, alpha=0.7, lw=0)
    b.axvline(0, color=AXIS, lw=1)
    ok = np.abs(gap) <= TARGET_PP + 1e-9
    b.barh(y, gap, height=0.55, color=[SERIES[0] if o else SERIES[1] for o in ok])
    for i, g in enumerate(gap):
        b.text(g + (0.25 if g >= 0 else -0.25), i, f"{g:+.1f}", va="center",
               ha="left" if g >= 0 else "right", color=INK2, fontsize=9)
    b.set_yticks(y, names)
    b.set_xlim(-4.5, 7)
    b.set_xlabel("INDOP − GT (pp)")
    b.set_title(f"Gap; shaded = ±{TARGET_PP} pp target", loc="left")
    b.grid(axis="y", visible=False)
    fig.text(0.01, -0.04, "† exact per-radio calibration; others approximate. "
             "Source: paper Table 3 (July 2026). Law73 and EBC: validation pending.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "fig01_multinode_occupancy_gap")


def fig_anchor_sweep() -> None:
    t = read_csv("table1_anchor_sweep.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhspan(-TARGET_PP, TARGET_PP, color=GRID, alpha=0.7, lw=0)
    ax.axvspan(72, 90, color=SERIES[1], alpha=0.08, lw=0)
    ax.text(81, 18, "B210 noise-floor\ndrift onset", ha="center", fontsize=8, color=INK2)
    ax.plot(t["anchor_s"], t["gap_pp"], color=SERIES[0], marker="o", ms=7, mec="white", mew=1.5)
    for x, g in zip(t["anchor_s"], t["gap_pp"]):
        ax.annotate(f"{g:+.1f}", (x, g), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(t["anchor_s"])
    ax.set_xlabel("Anchor window (s)")
    ax.set_ylabel("Gap vs physics GT (pp)")
    ax.set_title("Anchor window size: 72 s is the last stationary window", loc="left")
    ax.set_ylim(-8, 28)
    fig.text(0.01, -0.03, "Source: paper Table 1, Sagepoint, CBRS 3595 MHz.", fontsize=8, color=MUTED)
    save(fig, "fig02_anchor_window_sweep")


def fig_methods() -> None:
    t = read_csv("table2_methods.csv")
    m = t[t["method"] != "Physics GT"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(m))[::-1]
    indop = np.array([s.startswith("INDOP") for s in m["method"]])
    ax.axvspan(-TARGET_PP, TARGET_PP, color=GRID, alpha=0.7, lw=0)
    ax.barh(y, m["gap_pp"], height=0.55, color=[SERIES[0] if i else MUTED for i in indop])
    ax.axvline(0, color=AXIS, lw=1)
    for yi, g, occ in zip(y, m["gap_pp"], m["occupancy_pct"]):
        ax.text(g + (0.4 if g >= 0 else -0.4), yi, f"{g:+.1f} pp  ({occ:.1f}%)", va="center",
                ha="left" if g >= 0 else "right", fontsize=8.5, color=INK2)
    ax.set_yticks(y, m["method"])
    ax.set_xlim(-27, 12)
    ax.set_xlabel("Gap vs physics GT = 19.0 % (pp)")
    ax.set_title("Single-node method comparison (none use calibration)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.04, "Source: paper Table 2, 20-min capture at 100 ms period.",
             fontsize=8, color=MUTED)
    save(fig, "fig03_method_comparison")


def fig_gmm() -> None:
    t = read_csv("table4_gmm_components.csv")
    x = np.linspace(-117.6, -114.0, 2000)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    total = np.zeros_like(x)
    for i, r in enumerate(t):
        s = max(r["std_db"], 0.02)  # Signal-3 std 0.001 dB: widen for visibility
        pdf = r["weight"] * norm.pdf(x, r["mean_dbx"], s)
        total += pdf
        ax.fill_between(x, pdf, color=SERIES[i], alpha=0.35, lw=0)
        ax.plot(x, pdf, color=SERIES[i], lw=1.5, label=f"{r['label']} (w={r['weight']:.3f})")
    ax.plot(x, total, color=INK, lw=1, ls="--", label="Mixture")
    thr = 0.5 * (t["mean_dbx"][0] + t["mean_dbx"][1])
    ax.axvline(thr, color=CRITICAL, lw=1.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 30)
    ax.text(thr + 0.04, 8, f"T = {thr:.3f} dBX\n(midpoint rule)", color=INK2, fontsize=8)
    ax.set_xlabel("Chunk power (dBX)")
    ax.set_ylabel("Weighted density")
    ax.set_title("Anchor-window GMM: noise cluster found without calibration", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.01, -0.03, "Source: paper Table 4 (72 s anchor, Sagepoint nuc1). "
             "Signal-3 width enlarged for visibility.", fontsize=8, color=MUTED)
    save(fig, "fig04_gmm_components")


def fig_node_map() -> None:
    nodes, xy, _, _ = campaign_geometry()
    t = read_csv("table3_multinode.csv")
    occ = {n: v for n, v in zip(t["node"], t["indop_pct"])}
    gap = {n: v for n, v in zip(t["node"], t["gap_pp"])}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    vals = [occ.get(n.name) for n in nodes]
    have = np.array([v is not None for v in vals])
    sc = ax.scatter(xy[have, 0], xy[have, 1], c=[v for v in vals if v is not None], cmap=SEQ_CMAP,
                    vmin=10, vmax=40, s=260, edgecolor="white", lw=2, zorder=3)
    ax.scatter(xy[~have, 0], xy[~have, 1], s=200, facecolor="none", edgecolor=MUTED, lw=1.5,
               ls="--", zorder=3)
    left = {"Web"}  # avoid collision with the nearby Sagepoint label
    for n, (x, y), v in zip(nodes, xy, vals):
        lab = f"{n.name}\n{v:.1f}% ({gap[n.name]:+.1f} pp)" if v is not None else f"{n.name}\npending"
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(-14, -12) if n.name in left else (12, -4),
                    ha="right" if n.name in left else "left", fontsize=8,
                    color=INK2 if v is not None else MUTED)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("INDOP occupancy (%)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Seven-node CBRS 3595 MHz deployment", loc="left")
    ax.margins(0.25)
    fig.text(0.01, -0.02, "Occupancy and gap from paper Table 3. Node positions approximate "
             "(config/nodes.yaml).", fontsize=8, color=MUTED)
    save(fig, "fig05_node_map")


def fig_scf_peaks() -> None:
    t = read_csv("cyclostationary_peaks.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar([f"α = {int(a)} Hz\n{f}" for a, f in zip(t["alpha_hz"], t["feature"])],
           t["peak_db_above_noise"], width=0.5, color=SERIES[0])
    for i, v in enumerate(t["peak_db_above_noise"]):
        ax.text(i, v + 0.3, f"+{v:.1f} dB", ha="center", color=INK2, fontsize=9)
    ax.set_ylabel("Peak above noise (dB)")
    ax.set_ylim(0, 14)
    ax.set_title("LTE cyclic features in raw IQ", loc="left")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.06, "Source: paper Sec. 5.4 (nfft = 8192, 122 Hz resolution).",
             fontsize=8, color=MUTED)
    save(fig, "fig06_scf_lte_peaks")


if __name__ == "__main__":
    setup_style()
    fig_multinode()
    fig_anchor_sweep()
    fig_methods()
    fig_gmm()
    fig_node_map()
    fig_scf_peaks()
