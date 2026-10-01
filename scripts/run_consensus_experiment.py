"""Extension 2: does noise-floor consensus remove a Guesthouse-style +5 pp gap?

Two hypothesised mechanisms for an over-estimating node are simulated:

  A. Unlucky anchor window - the node's noise floor sits lower by b dB during
     the 72 s anchor than for the rest of the session, so its threshold is
     too low and it over-reports occupancy.
  B. Low-SNR cluster overlap - the node is far from the transmitter, its
     Signal-1 cluster nearly overlaps the noise cluster and the midpoint
     threshold lands about 1 sigma above the noise mean.

For each, the faulty node's gap is measured with no consensus, absolute
(mu_noise) consensus and offset (mu_anchor - mu_ref) consensus, over
Monte-Carlo seeds, with small and large inter-unit hardware spread.

SYNTHETIC. Run the same comparison on the real July-2026 NPZ results with
``--npz-dir`` to answer the question for the real Guesthouse node.

Usage:  python scripts/run_consensus_experiment.py [--seeds 20] [--npz-dir results/npz]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import (AXIS, GRID, INK2, MUTED, RESULTS_DIR, SERIES, default_scenario, save,
                    setup_style)
import matplotlib.pyplot as plt

from indop import INDOPConfig, run_indop
from indop.anchor_gmm import occupancy
from indop_ma.consensus import NoiseReport, noise_floor_consensus
from sim.synthetic import simulate_power

FAULTY = "Guesthouse"
# (mode, inter-unit spread sigma_u in dB, label)
METHODS = [("none", 0.3, "No consensus"),
           ("absolute", 0.02, "Absolute μ consensus, σ_u = 0.02 dB"),
           ("absolute", 0.3, "Absolute μ consensus, σ_u = 0.3 dB"),
           ("offset", 0.3, "Offset consensus, σ_u = 0.3 dB"),
           ("guard", 0.3, "No consensus + 2.5σ separation guard")]
GUARD_SIGMA = 2.5
BIASES = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
SNRS = (-3.0, -6.0, -9.0, -12.0, -15.0)


def gaps(seed: int, spread: float, bias: float = 0.0, snr: float | None = None) -> dict:
    """Faulty-node gap (pp) for each consensus mode in one simulated session."""
    sc = default_scenario(seed=seed, unit_spread_db=spread,
                          hw_overrides={FAULTY: {"anchor_bias_db": -bias}})
    fi = sc.node_names.index(FAULTY)
    if snr is not None:
        sc.shadow_db[fi] += snr - sc.snr_db()[fi]
    sim = simulate_power(sc, minutes=20, duty=0.4, seed=seed)
    res = [run_indop(sim["power"][i], n, INDOPConfig(run_hmm=False))
           for i, n in enumerate(sc.node_names)]
    reports = [NoiseReport.from_fit(r.node, r.gmm, r.meta["mu_ref_dbx"]) for r in res]
    truth = sim["truth"][fi].mean()
    out = {}
    for mode in ("none", "absolute", "offset"):
        reps = reports if mode == "none" else noise_floor_consensus(reports, mode=mode).corrected
        out[mode] = 100 * (occupancy(sim["power"][fi], reps[fi].threshold) - truth)
    r = reports[fi]
    t_guard = r.threshold if res[fi].gmm.k == 1 else max(
        r.threshold, r.mu_noise + GUARD_SIGMA * r.sigma_noise)
    out["guard"] = 100 * (occupancy(sim["power"][fi], t_guard) - truth)
    return out


def synthetic_study(seeds: int) -> list[dict]:
    rows = []
    for study, xs in (("anchor_bias", BIASES), ("low_snr", SNRS)):
        for x in xs:
            kw = {"bias": x} if study == "anchor_bias" else {"snr": x}
            runs = {sp: [gaps(1000 + s, sp, **kw) for s in range(seeds)]
                    for sp in sorted({m[1] for m in METHODS})}
            for mode, spread, _ in METHODS:
                vals = [g[mode] for g in runs[spread]]
                rows.append({"study": study, "x": x, "mode": mode, "unit_spread_db": spread,
                             "gap_pp_median": float(np.median(vals)),
                             "gap_pp_q25": float(np.percentile(vals, 25)),
                             "gap_pp_q75": float(np.percentile(vals, 75))})
            print(f"{study} x={x} done")
    return rows


def plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True)
    cfg = {"anchor_bias": ("A · unlucky anchor: floor low by b during anchor",
                           "Anchor bias b (dB)"),
           "low_snr": ("B · far node: noise/signal cluster overlap",
                       f"{FAULTY} received SNR (dB)")}
    for ax, study in zip(axes, cfg):
        ax.axhspan(-2.5, 2.5, color=GRID, alpha=0.7, lw=0)
        ax.axhline(0, color=AXIS, lw=1)
        for k, (mode, spread, label) in enumerate(METHODS):
            r = [x for x in rows if x["study"] == study and x["mode"] == mode
                 and x["unit_spread_db"] == spread]
            xs = np.array([x["x"] for x in r])
            med = [x["gap_pp_median"] for x in r]
            if mode == "none":  # reference: grey, dashed, on top
                ax.plot(xs, med, color=MUTED, lw=2.2, ls=(0, (5, 2)), marker="s", ms=5,
                        label=label, zorder=5)
                continue
            color = SERIES[k - 1]
            if mode == "offset":
                ax.fill_between(xs, [x["gap_pp_q25"] for x in r], [x["gap_pp_q75"] for x in r],
                                color=color, alpha=0.15, lw=0)
            ax.plot(xs, med, marker="o", ms=6, mec="white", mew=1.2, color=color, label=label)
        if study == "low_snr":
            ax.annotate("signal below detectability:\ntruth ≈ 0, nothing to over-count",
                        (-15, 0), xytext=(-12.6, -7), textcoords="data", fontsize=7.5,
                        color=INK2, arrowprops={"arrowstyle": "-", "color": AXIS})
        ax.set_title(cfg[study][0], loc="left", fontsize=11)
        ax.set_xlabel(cfg[study][1])
    axes[1].invert_xaxis()
    axes[0].set_ylabel(f"{FAULTY} gap vs truth (pp)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.text(0.01, -0.03, "Median over Monte-Carlo seeds (offset consensus: with inter-quartile band). "
             "Shaded band = ±2.5 pp target. σ_u = inter-unit noise-floor spread.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "fig10_consensus_guesthouse", synthetic=True)


def real_npz(npz_dir: Path) -> None:
    """Apply consensus to real per-node result NPZs (no ground truth needed to see the shift)."""
    from indop.anchor_gmm import GMMFit
    from indop.pipeline import load_result_npz

    reps = []
    for f in sorted(npz_dir.glob("*.npz")):
        s = load_result_npz(str(f))
        g = s["gmm"]
        fit = GMMFit(np.array(g["means"]), np.array(g["stds"]), np.array(g["weights"]),
                     {}, g["threshold"], g["anchor_chunks"])
        reps.append(NoiseReport.from_fit(s["node"], fit, s.get("mu_ref_dbx")))
    for mode in ("absolute", "offset"):
        o = noise_floor_consensus(reps, mode=mode)
        print(f"[{mode}] outliers: {o.outliers}")
        for r0, r1 in zip(reps, o.corrected):
            print(f"  {r0.node:<11} T {r0.threshold:.3f} -> {r1.threshold:.3f} dBX  "
                  f"z={o.robust_z[r0.node]:+.2f}")
    print("Re-run scripts/run_multinode_campaign.py --consensus to recompute occupancy "
          "with corrected thresholds on-node.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--npz-dir", type=Path, default=None)
    a = ap.parse_args()
    if a.npz_dir:
        real_npz(a.npz_dir)
    else:
        setup_style()
        rows = synthetic_study(a.seeds)
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / "consensus_synthetic.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        plot(rows)
