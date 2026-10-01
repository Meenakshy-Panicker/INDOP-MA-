"""Extension 4: which joint occupancy fusion rule minimises false grants and missed detections?

For every chunk, each of the 7 agents produces an INDOP-HMM decision and
posterior. A secondary consumer at location c asks for a grant; the fused
decision "idle" grants it. We evaluate, over many consumer locations:

  false-grant rate     P(grant | band occupied at the consumer's location)
  missed detection     P(grant | incumbent active and detectable at >=1 sensor)
  false-denial rate    P(deny  | band idle at the consumer's location)
  grant utilisation    P(grant | band idle at the consumer's location)

Rules: OR, majority, AND, and location-aware weighted average (IDW to the
consumer, tau = 0.5 and 0.3).

SYNTHETIC: needs per-location truth, which only a simulator provides. On real
data, run the same rules on the posterior arrays in the result NPZs and use
the physics-GT nodes as held-out "consumers" (leave-one-node-out).

Usage:  python scripts/run_fusion_experiment.py [--seeds 6] [--consumers 120]
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from common import AXIS, INK2, MUTED, RESULTS_DIR, SERIES, default_scenario, save, setup_style
import matplotlib.pyplot as plt

from indop import run_indop
from indop_ma.fusion import fuse, grant_metrics, idw_weights
from sim.synthetic import simulate_power

RULES = [("or", {}, "OR (any node)"),
         ("majority", {}, "Majority (4 of 7)"),
         ("and", {}, "AND (all nodes)"),
         ("weighted", {"tau": 0.5}, "Weighted IDW, τ = 0.5"),
         ("weighted", {"tau": 0.3}, "Weighted IDW, τ = 0.3")]


def consumer_truth(sim: dict, sc, cxy: np.ndarray) -> np.ndarray:
    """(m, n_chunks) occupancy truth at consumer points, same physics as the nodes."""
    snr = sc.snr_db(cxy)
    rise = 10 * np.log10(1 + sim["active"] * sim["load"] * 10 ** (snr[:, None] / 10))
    return sim["active"][None, :] & (rise >= sc.gt_margin_db)


def run(seeds: int, n_consumers: int) -> list[dict]:
    acc = {label: [] for *_, label in RULES}
    for s in range(seeds):
        sc = default_scenario(seed=2000 + s, unit_spread_db=0.3)
        sim = simulate_power(sc, minutes=10, duty=0.35, seed=2000 + s)
        res = [run_indop(sim["power"][i], n) for i, n in enumerate(sc.node_names)]
        probs = np.vstack([r.hmm.posterior for r in res])
        dec = probs > 0.5
        incumbent = sim["truth"].any(axis=0)
        rng = np.random.default_rng(s)
        lo, hi = sc.node_xy.min(0) - 200, sc.node_xy.max(0) + 200
        cxy = rng.uniform(lo, hi, size=(n_consumers, 2))
        ct = consumer_truth(sim, sc, cxy)
        for rule, kw, label in RULES:
            for c in range(n_consumers):
                w = idw_weights(sc.node_xy, cxy[c])
                fused = fuse(rule, dec, probs=probs, weights=w, **kw)
                acc[label].append(grant_metrics(fused, ct[c], incumbent).as_dict())
        print(f"seed {s} done")
    rows = []
    for _, _, label in RULES:
        m = {k: float(np.mean([d[k] for d in acc[label]])) for k in acc[label][0]}
        rows.append({"rule": label, **m})
    return rows


def plot(rows: list[dict]) -> None:
    metrics = [("false_grant_rate", "False-grant rate"), ("missed_detection", "Missed detection"),
               ("false_denial_rate", "False-denial rate"), ("grant_utilisation", "Grant utilisation")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    h = 0.2
    for k, (key, lab) in enumerate(metrics[:3]):
        vals = [100 * r[key] for r in rows]
        ax.barh(y + (1 - k) * h, vals, height=h * 0.9, color=SERIES[k], label=lab)
        for yi, v in zip(y, vals):
            ax.text(v + 0.6, yi + (1 - k) * h, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [r["rule"] for r in rows])
    ax.set_xlabel("Rate (%) — lower is better")
    ax.set_title("Error rates by fusion rule", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    pts = np.array([[100 * r["false_grant_rate"], 100 * r["grant_utilisation"]] for r in rows])
    ax.scatter(pts[:, 0], pts[:, 1], s=80, color=SERIES[0], edgecolor="white", lw=1.5, zorder=3)
    # Label placement: neighbours (majority / weighted) sit close together, so spread the
    # labels vertically in utilisation order and connect them with leader lines.
    order = np.argsort(-pts[:, 1])
    for rank, k in enumerate(order):
        x, yv = pts[k]
        ax.annotate(rows[k]["rule"], (x, yv), xytext=(x + 8, pts[:, 1].max() - rank * 2.4),
                    textcoords="data", fontsize=8, color=INK2, va="center",
                    arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
    ax.set_xlabel("False-grant rate (%)")
    ax.set_ylabel("Grant utilisation (%)")
    ax.set_title("Trade-off (ideal: top-left)", loc="left")
    ax.set_xlim(-3, max(60, pts[:, 0].max() + 25))
    fig.text(0.01, -0.03, "Averaged over random consumer locations and Monte-Carlo seeds; "
             "7 agents, INDOP-HMM decisions at 100 ms.", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "fig11_fusion_rules", synthetic=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--consumers", type=int, default=120)
    a = ap.parse_args()
    setup_style()
    rows = run(a.seeds, a.consumers)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "fusion_synthetic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['rule']:<24} FG {100*r['false_grant_rate']:5.1f}  MD {100*r['missed_detection']:5.1f}  "
              f"FD {100*r['false_denial_rate']:5.1f}  UT {100*r['grant_utilisation']:5.1f}")
    plot(rows)
