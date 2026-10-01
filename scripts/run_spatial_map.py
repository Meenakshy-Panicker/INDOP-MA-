"""Extension 3: spatial occupancy map from 7-node simultaneous observations.

Compares three maps of the same 20-minute window against the true occupancy
field (known only in simulation):

  1. INDOP-MA: per-node calibration-free INDOP at 100 ms, GP-interpolated
  2. Baseline: periodic 3 s sweeps with one fixed, manually set threshold
     shared by every monitor (the current OpenZMS DST approach)
  3. Baseline map error vs INDOP-MA map error, per proxel

SYNTHETIC. For the real comparison with OpenZMS Figure 7, export the DST
proxel occupancy for the same session window as CSV (lat,lon,occupancy) and
pass it with --dst-csv; the script then plots INDOP-MA next to the DST map
at the node locations.

Usage:  python scripts/run_spatial_map.py [--seed 7] [--dst-csv path]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import (INK2, MUTED, RESULTS_DIR, SEQ_CMAP, campaign_geometry, default_scenario,
                    save, setup_style)
import matplotlib.pyplot as plt

from indop import run_indop
from indop_ma.spatial_map import gp_map, map_error
from sim.synthetic import simulate_power

SWEEP_CHUNKS = 30  # 3 s periodic sweep = 30 x 100 ms averaged


def truth_field(sim: dict, sc, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gx, gy = np.meshgrid(x, y)
    snr = sc.snr_db(np.column_stack([gx.ravel(), gy.ravel()]))
    on, u = sim["active"], sim["load"]
    # occupied when 10log10(1 + u snr) >= margin  <=>  u >= (10^(m/10) - 1) / snr_lin
    need = (10 ** (sc.gt_margin_db / 10) - 1) / 10 ** (snr / 10)
    ua = np.sort(u[on])
    frac_on = on.mean()
    occ = frac_on * (1 - np.searchsorted(ua, need, side="left") / max(len(ua), 1))
    return occ.reshape(gx.shape)


def fixed_threshold_sweep(power: np.ndarray, threshold_dbx: float) -> float:
    n = (power.shape[-1] // SWEEP_CHUNKS) * SWEEP_CHUNKS
    lin = 10 ** (power[:n] / 10)
    sweep = 10 * np.log10(lin.reshape(-1, SWEEP_CHUNKS).mean(1))
    return float(np.mean(sweep > threshold_dbx))


def main(seed: int) -> None:
    nodes, xy, _, txy = campaign_geometry()
    sc = default_scenario(seed=seed, unit_spread_db=0.3)
    sim = simulate_power(sc, minutes=20, duty=0.4, seed=seed)
    indop = np.array([run_indop(sim["power"][i], n.name).occupancy_hmm_soft
                      for i, n in enumerate(nodes)])
    # Fixed threshold tuned on the nominal (datasheet) noise floor + 0.3 dB, same for all.
    t_fixed = sc.hw[0].mu_noise_dbx + 0.3
    fixed = np.array([fixed_threshold_sweep(sim["power"][i], t_fixed) for i in range(len(nodes))])

    m_indop = gp_map(xy, indop)
    m_fixed = gp_map(xy, fixed)
    truth = truth_field(sim, sc, m_indop.x, m_indop.y)
    e_indop, e_fixed = map_error(m_indop, truth), map_error(m_fixed, truth)
    node_truth = sim["truth"].mean(1)
    summary = {"seed": seed, "indop_map": e_indop, "fixed_sweep_map": e_fixed,
               "node_mae_pp": {"indop": float(100 * np.mean(np.abs(indop - node_truth))),
                               "fixed_sweep": float(100 * np.mean(np.abs(fixed - node_truth)))}}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "spatial_map_synthetic.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True, layout="constrained")
    ext = [m_indop.x[0], m_indop.x[-1], m_indop.y[0], m_indop.y[-1]]
    panels = [("Truth (simulator)\n ", truth, None),
              (f"INDOP-MA, 100 ms, GP\nmap MAE {e_indop['mae_pp']:.1f} pp", m_indop.value, indop),
              (f"Fixed threshold, 3 s sweeps\nmap MAE {e_fixed['mae_pp']:.1f} pp", m_fixed.value, fixed)]
    vmax = max(0.05, float(truth.max()))
    for ax, (title, val, obs) in zip(axes, panels):
        im = ax.imshow(val, origin="lower", extent=ext, cmap=SEQ_CMAP, vmin=0, vmax=vmax,
                       aspect="equal")
        ax.scatter(xy[:, 0], xy[:, 1], s=40, facecolor="white", edgecolor="#0b0b0b", lw=1, zorder=3)
        ax.scatter(*txy, marker="*", s=180, color="#eb6834", edgecolor="white", lw=1, zorder=4)
        for n, (x, y) in zip(nodes, xy):
            ax.annotate(n.name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7,
                        color="#0b0b0b")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("East (m)")
        ax.grid(False)
    axes[0].set_ylabel("North (m)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Occupancy", color=INK2)
    cb.outline.set_visible(False)
    fig.text(0.01, -0.04, "★ synthetic transmitter. White dots: the 7 POWDER agents "
             "(approximate positions). Maps share one colour scale.", fontsize=8, color=MUTED)
    save(fig, "fig12_spatial_occupancy_maps", synthetic=True)

    # Uncertainty map: where more sensors would help.
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(100 * m_indop.std, origin="lower", extent=ext, cmap=SEQ_CMAP, aspect="equal")
    ax.scatter(xy[:, 0], xy[:, 1], s=40, facecolor="white", edgecolor="#0b0b0b", lw=1, zorder=3)
    ax.set_title("INDOP-MA map uncertainty (GP σ)", loc="left")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("σ (pp)", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "fig13_spatial_map_uncertainty", synthetic=True)


def compare_dst(dst_csv: str, npz_dir: str) -> None:
    """Real-data path: INDOP-MA node occupancies vs DST proxel values at node sites."""
    from pathlib import Path

    from indop.pipeline import load_result_npz
    from indop_ma.spatial_map import load_dst_proxels_csv
    from indop_ma.topology import to_local_xy

    nodes, xy, (lat0, lon0), _ = campaign_geometry()
    occ = {}
    for f in Path(npz_dir).glob("*.npz"):
        s = load_result_npz(str(f))
        occ[s["node"]] = s["occupancy_hmm_soft"] or s["occupancy_gmm"]
    ll, dst = load_dst_proxels_csv(dst_csv)
    dx, dy = to_local_xy(ll[:, 0], ll[:, 1], lat0, lon0)
    dxy = np.column_stack([dx, dy])
    for n, p in zip(nodes, xy):
        j = np.argmin(((dxy - p) ** 2).sum(1))
        print(f"{n.name:<11} INDOP-MA {100*occ.get(n.name, np.nan):5.1f}%   DST {100*dst[j]:5.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dst-csv", default=None)
    ap.add_argument("--npz-dir", default="results/npz")
    a = ap.parse_args()
    setup_style()
    if a.dst_csv:
        compare_dst(a.dst_csv, a.npz_dir)
    else:
        main(a.seed)
