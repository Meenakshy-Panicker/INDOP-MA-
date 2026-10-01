"""Run INDOP on each node's capture and the INDOP-MA coordination step.

Two inputs are supported:

  --iq FILE [--node NAME]      process a raw fc32/sc16 IQ file recorded on a node
  --synthetic                  simulate all 7 nodes (no radio needed)

Then, with all per-node result NPZs in --out, it runs noise-floor consensus
and prints a per-node table plus the ZeAL observation payloads that would be
posted to OpenZMS.

Usage:
  python scripts/run_multinode_campaign.py --synthetic
  python scripts/run_multinode_campaign.py --iq /data/web.fc32 --node Web --out results/npz
  python scripts/run_multinode_campaign.py --coordinate --out results/npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from common import ROOT, default_scenario  # noqa: F401  (sets sys.path)

from indop import INDOPConfig, run_indop
from indop.anchor_gmm import GMMFit
from indop.ground_truth import physics_threshold_for, self_calibrated_threshold
from indop.pipeline import load_result_npz
from indop.power import power_timeline_streaming, read_iq_file
from indop_ma.consensus import NoiseReport, noise_floor_consensus
from indop_ma.topology import load_nodes
from indop_ma.zeal import build_observation, to_json
from sim.synthetic import simulate_power


def process_iq(path: str, node: str, out: Path, dtype: str) -> None:
    power = np.fromiter(power_timeline_streaming(read_iq_file(path, dtype)), float)
    snippet = next(iter(read_iq_file(path, dtype, chunk_samples=2_000_000)))
    res = run_indop(power, node, INDOPConfig(), iq_for_scf=snippet)
    out.mkdir(parents=True, exist_ok=True)
    res.save_npz(str(out / f"{node}.npz"))
    np.save(out / f"{node}_power.npy", power.astype(np.float32))
    print(json.dumps(res.summary(), indent=2, default=float))


def synthetic(out: Path, minutes: float, seed: int) -> None:
    sc = default_scenario(seed=seed, unit_spread_db=0.3)
    sim = simulate_power(sc, minutes=minutes, duty=0.35, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(sc.node_names):
        res = run_indop(sim["power"][i], n, meta={"synthetic_truth": float(sim["truth"][i].mean())})
        res.save_npz(str(out / f"{n}.npz"))
        np.save(out / f"{n}_power.npy", sim["power"][i].astype(np.float32))
    print(f"wrote {len(sc.node_names)} synthetic node results to {out}")


def coordinate(out: Path) -> None:
    nodes = {n.name: n for n in load_nodes()}
    summaries = []
    for f in sorted(out.glob("*.npz")):
        s = load_result_npz(str(f))
        if s["node"] in nodes:
            summaries.append(s)
    if not summaries:
        raise SystemExit(f"no node NPZs in {out}")
    reports = []
    for s in summaries:
        g = s["gmm"]
        fit = GMMFit(np.array(g["means"]), np.array(g["stds"]), np.array(g["weights"]), {},
                     g["threshold"], g["anchor_chunks"])
        reports.append(NoiseReport.from_fit(s["node"], fit, s.get("mu_ref_dbx")))
    cons = noise_floor_consensus(reports, mode="offset")
    print(f"\nconsensus outliers: {cons.outliers or 'none'}\n")
    hdr = f"{'node':<11}{'K':>3}{'mu_noise':>11}{'T':>10}{'T_cons':>10}{'occ_GMM':>9}{'occ_HMM':>9}{'GT':>8}"
    print(hdr)
    print("-" * len(hdr))
    for s, r0, r1 in zip(summaries, reports, cons.corrected):
        n = nodes[s["node"]]
        power_f = out / f"{s['node']}_power.npy"
        t_gt = physics_threshold_for(n.serial or "", 3595, 60) if n.serial else \
            self_calibrated_threshold(r0.mu_noise, r0.sigma_noise)
        gt = float(np.mean(np.load(power_f) > t_gt)) if power_f.exists() else float("nan")
        gt = s.get("synthetic_truth", gt)
        print(f"{s['node']:<11}{s['k']:>3}{r0.mu_noise:>11.3f}{r0.threshold:>10.3f}{r1.threshold:>10.3f}"
              f"{100*s['occupancy_gmm']:>8.1f}%{100*(s['occupancy_hmm_soft'] or 0):>8.1f}%{100*gt:>7.1f}%")
    s, n = summaries[0], nodes[summaries[0]["node"]]
    obs = build_observation(
        s, monitor_id=f"indop-{n.name.lower()}", freq_start_hz=3594.5e6, freq_end_hz=3595.5e6,
        lat=n.lat, lon=n.lon, session_start=dt.datetime.now(dt.timezone.utc),
        session_seconds=len(s["posterior"]) * 0.1,
        ma_block={"consensus_mode": "offset", "outlier": n.name in cons.outliers,
                  "threshold_dbx_consensus": cons.corrected[0].threshold})
    print("\nexample ZeAL observation (first node):")
    print(to_json(obs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq")
    ap.add_argument("--node")
    ap.add_argument("--dtype", default="complex64", choices=["complex64", "int16"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--coordinate", action="store_true")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "npz")
    a = ap.parse_args()
    if a.iq:
        process_iq(a.iq, a.node or Path(a.iq).stem, a.out, a.dtype)
    if a.synthetic:
        synthetic(a.out, a.minutes, a.seed)
    if a.coordinate or a.synthetic:
        coordinate(a.out)
