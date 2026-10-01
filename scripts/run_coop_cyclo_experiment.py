"""Cooperative cyclostationary detection: can 7 averaged SCF profiles see what one cannot?

Each node computes the SCF alpha-profile of 0.25 s of its own IQ and shares
only the profile. The LTE statistic is the larger of the alpha = 1 kHz and
2 kHz peaks; thresholds are set for a 5 % false-alarm rate from noise-only
trials of the same configuration (single node / 3 nodes / 7 nodes).

SYNTHETIC (LTE-like frame structure in AWGN).

Usage:  python scripts/run_coop_cyclo_experiment.py [--trials 60]
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from common import AXIS, GRID, INK2, MUTED, RESULTS_DIR, SERIES, save, setup_style
import matplotlib.pyplot as plt

from indop.cyclo import LTE_ALPHAS_HZ, alpha_profile, average_profiles
from sim.synthetic import lte_like_iq

FS = 1e6
SECONDS = 0.25
GROUPS = (1, 3, 7)
SNRS = np.arange(-24, -9, 2.0)
PFA = 0.05


def stat(profile) -> float:
    return max(profile.peak_db(a) for a in LTE_ALPHAS_HZ)


def trial(n_nodes: int, snr_db: float | None, rng) -> tuple[float, object]:
    profs = []
    for _ in range(n_nodes):
        if snr_db is None:
            n = int(SECONDS * FS)
            iq = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
        else:
            iq = lte_like_iq(SECONDS, snr_db, FS, rng)
        profs.append(alpha_profile(iq, FS))
    p = average_profiles(profs) if n_nodes > 1 else profs[0]
    return stat(p), p


def main(trials: int) -> None:
    rng = np.random.default_rng(42)
    thr = {g: float(np.quantile([trial(g, None, rng)[0] for _ in range(trials * 2)], 1 - PFA))
           for g in GROUPS}
    print("thresholds (dB):", {g: round(v, 2) for g, v in thr.items()})
    rows = []
    for snr in SNRS:
        for g in GROUPS:
            pd = np.mean([trial(g, snr, rng)[0] >= thr[g] for _ in range(trials)])
            rows.append({"snr_db": float(snr), "nodes": g, "p_detect": float(pd),
                         "threshold_db": thr[g]})
        print(f"SNR {snr:+.0f} dB: " + ", ".join(
            f"{r['nodes']}n={r['p_detect']:.2f}" for r in rows[-len(GROUPS):]))
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "coop_cyclo_synthetic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for k, g in enumerate(GROUPS):
        r = [x for x in rows if x["nodes"] == g]
        a.plot([x["snr_db"] for x in r], [x["p_detect"] for x in r], marker="o", ms=6,
               mec="white", mew=1.2, color=SERIES[k],
               label=f"{g} node{'s' if g > 1 else ''}" + (" (single INDOP)" if g == 1 else " averaged"))
    a.axhline(0.9, color=AXIS, lw=1, ls=(0, (4, 3)))
    a.text(SNRS[0], 0.92, "P_d = 0.9", fontsize=8, color=MUTED)
    a.set_xlabel("Per-node SNR (dB)")
    a.set_ylabel(f"Detection probability (P_fa = {PFA:.0%})")
    a.set_ylim(-0.03, 1.05)
    a.set_title("Cooperative SCF detection of weak LTE", loc="left")
    a.legend(loc="lower right")

    snr_demo = -15.0
    _, p1 = trial(1, snr_demo, rng)
    _, p7 = trial(7, snr_demo, rng)
    b.plot(p1.alphas, p1.profile_db, color=SERIES[1], lw=1, label="Single node")
    b.plot(p7.alphas, p7.profile_db, color=SERIES[0], lw=1.6, label="7-node average")
    for alp in LTE_ALPHAS_HZ:
        b.axvline(alp, color=GRID, lw=3, zorder=0)
    b.set_xlim(0, 3500)
    b.set_xlabel("Cyclic frequency α (Hz)")
    b.set_ylabel("|SCF| profile (dB re. median)")
    b.set_title(f"α-profile at {snr_demo:.0f} dB per-node SNR", loc="left")
    b.legend(loc="upper right")
    ymax = b.get_ylim()[1]
    for alp, lab in zip(LTE_ALPHAS_HZ, ("1 ms subframe", "0.5 ms slot")):
        b.text(alp + 40, ymax * 0.92, lab, fontsize=8, color=INK2)
    fig.text(0.01, -0.03, f"{SECONDS:g} s of IQ per node at 1 MS/s, nfft = 8192. "
             "Thresholds calibrated on noise-only trials of each configuration.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "fig14_cooperative_scf", synthetic=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    setup_style()
    main(ap.parse_args().trials)
