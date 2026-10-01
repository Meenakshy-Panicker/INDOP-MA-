# Experiments: extending the multi-node contribution

Each extension below has a hypothesis, a protocol on POWDER, the metric that decides it, the script that computes the metric, and the synthetic pre-study that already exists in this repository.

---

## E1. Temporal campaign: is the 4–7 am carrier idle window visible?

**Hypothesis.** The OpenZMS / POWDER-RDZ paper reports a three-hour window (04:00–07:00) in which the cooperating carrier idles its cells. INDOP should detect this window passively and simultaneously on all seven nodes, at 100 ms resolution, without UEs and without carrier cooperation.

**Protocol.**

* Run 20-minute sessions at night (01, 03 h), during the idle window (04, 05, 06 h), in the morning (08, 10 h), in the afternoon (13, 15 h) and in the evening (18, 21 h).
* Repeat on at least 3 weekdays and 1 weekend day.
* All seven nodes capture in sync (`capture_uhd.py --session HH`).

**Metrics.**

* Per node-hour occupancy.
* Network idle detection, defined as all nodes below 3 %.
* Precision and recall of detected idle hours against the carrier schedule.
* Idle-window boundary error, in minutes, measured from the HMM posterior at 100 ms.

**Script.** `scripts/run_temporal_campaign.py --npz-dir results/npz`

**Synthetic pre-study.** With a diurnal load and a 04–07 idle window, INDOP flags exactly hours 04, 05 and 06 as idle on every node. The highest occupancy any node reports in those hours is 0.06 %. Outside the window the node mean tracks the truth. Isolated node-hours over-estimate by up to about 12 pp. ([fig15](figures/fig15_temporal_idle_window.png))

**Follow-ups.**

* Use the HMM posterior to locate the idle-window edges to within one 100 ms chunk, rather than to the hour.
* Compare against OpenZMS's RSRP-based detection with its −99 dBm threshold.

---

## E2. Noise-floor consensus and the Guesthouse +5.1 pp anomaly

**Hypotheses.** There are three candidate explanations for the Guesthouse gap:

* **H-A (unlucky anchor).** Guesthouse's noise floor during its 72 s anchor differed from the rest of the session. Offset consensus would then flag Guesthouse and pull its threshold back toward the majority, which should close the gap.
* **H-B (cluster overlap).** Guesthouse receives weak signal. Its Signal-1 cluster overlaps the noise cluster, so the midpoint threshold sits too close to the noise. Consensus would not flag it, but the separation guard would close the gap.
* **H-C (GT approximation, the paper's hypothesis).** INDOP is right and the approximate physics GT is off. Neither consensus nor the guard would change much, and exact calibration of the Guesthouse B210 would close the gap.

**Protocol.**

1. Run `run_consensus_experiment.py --npz-dir` on the July 2026 NPZs and record whether Guesthouse is flagged and how far its threshold shifts.
2. Recompute occupancy with the corrected threshold.
3. Record the GMM separation $(\mu_1-\mu_0)/\sigma_0$ for each node.
4. If possible, measure (K, NF) for the Guesthouse B210 in the lab and recompute its physics GT.

**Decision table.**

| Flagged by offset consensus | Separation < 2.5σ | Gap closes with | Supports |
|---|---|---|---|
| yes | — | consensus | H-A |
| no | yes | guard | H-B |
| no | no | exact calibration | H-C |

**Synthetic pre-study** ([fig10](figures/fig10_consensus_guesthouse.png)). A 0.06 dB anchor bias produces a +5.0 pp gap, which offset consensus reduces to −0.2 pp. Absolute consensus with 0.3 dB unit spread does nothing. Cluster overlap at −9 to −12 dB SNR produces +4.5 to +19.6 pp, which consensus cannot fix and the guard can.

---

## E3. Spatial occupancy map vs OpenZMS Figure 7

**Claim to test.** INDOP-MA maps are more accurate, because they are calibration-free and adapt per node, and they have finer temporal resolution (100 ms instead of periodic sweeps) than the DST maps in OpenZMS Figure 7.

**Protocol.**

* Use the same session window for INDOP-MA (seven nodes, GP map) and for the DST export.
* **Accuracy:**
  * Leave-one-node-out: predict each held-out node from the other six and compare with its physics GT.
  * At nodes with exact calibration, compare the DST value with the physics GT.
* **Temporal resolution:** recompute the INDOP map in 1 s, 10 s and 60 s windows and show the transients that the periodic DST sweep misses.

**Metrics.** Map MAE and RMSE in pp, leave-one-out error, and GP σ to show where an eighth sensor would help most.

**Script.** `scripts/run_spatial_map.py` (synthetic) or `scripts/run_spatial_map.py --dst-csv` (real).

**Synthetic pre-study** ([fig12](figures/fig12_spatial_occupancy_maps.png), [fig13](figures/fig13_spatial_map_uncertainty.png)). The INDOP-MA map MAE is 2.5 pp. A single fixed threshold on 3 s sweeps gives 7.4 pp, with a node-level MAE of 1.9 pp vs 27.0 pp.

---

## E4. Joint occupancy fusion rules

**Question.** Which rule minimises false grants and missed detections, and at what cost in spectrum utilisation?

**Protocol.**

* For each chunk, fuse the seven HMM decisions using OR, majority, AND and weighted IDW with τ ∈ {0.5, 0.3}.
* Ground truth at the consumer location comes from simulation. On POWDER, use leave-one-node-out: treat each validated node as the consumer and use its own physics-GT decisions as truth.

**Metrics.** False-grant rate, missed detection, false-denial rate and grant utilisation (definitions in [METHODOLOGY.md](METHODOLOGY.md)).

**Script.** `scripts/run_fusion_experiment.py`

**Synthetic pre-study** ([fig11](figures/fig11_fusion_rules.png)).

* **OR** gives 0.0 % false grants and 0.2 % missed detection at 89 % utilisation. Use it for incumbent protection.
* **Weighted IDW at τ = 0.3** gives 1.6 % false grants at 94.6 % utilisation. Use it for DSA scheduling.
* **AND** gives 45.6 % false grants. Never use it.

---

## Also implemented: cooperative cyclostationary detection

**Synthetic study** ([fig14](figures/fig14_cooperative_scf.png)). At $P_\text{fa}=5\%$ the single-node SCF statistic reaches $P_d\ge0.9$ at −14 dB per-node SNR. Seven averaged profiles reach $P_d=1.0$ at −16 dB, a gain of about 2–3 dB from sharing a few hundred bytes per node.

**POWDER test.** Pick the node with the weakest LTE cyclic peak. Check whether the peers' averaged profile confirms α = 1 kHz where its own profile does not.
