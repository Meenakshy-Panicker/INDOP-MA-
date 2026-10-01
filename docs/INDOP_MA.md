# INDOP-MA: multi-agent INDOP

Single-node occupancy tells you **what one point sees**. Multi-node occupancy tells the Zone Management Controller something different: **whether the band is idle at the location where a consumer wants to transmit**. INDOP-MA lets the seven INDOP agents cooperate before they report to OpenZMS. That cooperation is what makes location-aware grants possible; the current OpenZMS DST, with one fixed threshold per monitor, cannot issue them.

INDOP-MA has three communication forms. Each one adds to the existing ZeAL Monitor → DST interface without changing it.

## Form 1: noise-floor consensus

**Goal.** Make each agent robust to an unlucky anchor window, and test whether this explains the **Guesthouse +5.1 pp** anomaly in paper Table 3.

**Protocol.** After its local GMM fit, each agent broadcasts a `NoiseReport`. Each agent then computes a robust consensus over its peers and corrects itself only if it is an outlier (majority rule).

### Why offset consensus

Three feature spaces were tried in the synthetic study (`scripts/run_consensus_experiment.py`):

| Feature | Idea | Outcome |
|---|---|---|
| $\mu_\text{noise}$ (absolute) | Pull the outlier's noise floor toward the peers. | Works only if the inter-unit spread $\sigma_u$ is far below the anchor error. At $\sigma_u = 0.02$ dB it helps once the bias reaches 0.08 dB or more. At $\sigma_u = 0.3$ dB, which is realistic for different B210 units, it never flags the faulty node, because hardware differences hide the fault. |
| $(T-\mu)/\sigma$ (margin) | Use a hardware-independent detection margin. | **Rejected.** The margin grows with received SNR (Sagepoint 3.9σ vs Guesthouse 1.0σ in one draw). Consensus therefore flagged the healthy high-SNR nodes and "corrected" them into large errors. |
| $\mu_\text{noise} - \mu_\text{ref}$ (offset) | Each node compares its anchor noise mean with its own re-fit just after the anchor. The unit offset cancels. | **Adopted.** It keeps the faulty node within ±2.5 pp for anchor biases up to 0.10 dB, even with $\sigma_u = 0.3$ dB, and it leaves healthy nodes untouched. |

### What consensus cannot fix

Panel B of `docs/figures/fig10_consensus_guesthouse.png` shows a different failure. A node far from the transmitter, at about −9 to −12 dB SNR, has a correct noise floor, but its weak Signal-1 cluster overlaps the noise cluster. The midpoint threshold then lands about 1σ above the noise mean, and the node over-counts by up to about 20 pp. Noise-floor consensus has nothing to correct here. A **separation guard** ($T \ge \mu + 2.5\sigma$) fixes it at low SNR but costs 4–6 pp of under-estimation at moderate SNR. So the guard should be enabled only when the GMM separation $(\mu_1-\mu_0)/\sigma_0$ is small. Spatial fusion can also cover this case, because nearby high-SNR peers see the same transmitter clearly.

**Implication for Guesthouse.** In simulation, an anchor bias of about 0.06 dB reproduces a +5 pp gap, and offset consensus removes it. Overlap at roughly −9 dB SNR gives a similar magnitude, and consensus does not fix that. Running `scripts/run_consensus_experiment.py --npz-dir <real NPZs>` on the July 2026 results shows whether Guesthouse is flagged:

* **Flagged:** the anchor hypothesis is supported.
* **Not flagged, small GMM separation:** the overlap hypothesis is supported.
* **Neither:** the paper's own explanation (approximate calibration in the GT) is the likely cause.

## Form 2: spatial occupancy fusion

### Joint occupancy fusion rules

Fusion runs per 100 ms chunk across agents. The rules and metrics are defined in [METHODOLOGY.md](METHODOLOGY.md#2-joint-occupancy-fusion-indop_mafusionpy). The synthetic study is `scripts/run_fusion_experiment.py` and its output is `docs/figures/fig11_fusion_rules.png`.

In the synthetic study (6 seeds × 120 random consumer locations, 7 agents):

| Rule | False grant | Missed detection | False denial | Grant utilisation |
|---|---|---|---|---|
| OR (any node) | 0.0 % | 0.2 % | 10.9 % | 89.1 % |
| Majority (4 of 7) | 3.9 % | 10.1 % | 4.1 % | 95.9 % |
| AND (all nodes) | 45.6 % | 54.1 % | 0.3 % | 99.7 % |
| Weighted IDW, τ = 0.5 | 5.2 % | 13.2 % | 3.0 % | 97.0 % |
| **Weighted IDW, τ = 0.3** | **1.6 %** | **6.4 %** | 5.4 % | 94.6 % |

* **OR rule:** essentially no false grants or missed detections. The price is about 11 % of idle time, because the most sensitive node vetoes everyone. This is the right rule for **incumbent protection**.
* **AND rule:** grants into nearly half of the occupied periods. This rule is unsafe.
* **Location-aware weighted rule:** at τ = 0.3 it gives the best trade-off. Compared with majority voting it has fewer false grants and missed detections, at about the same utilisation. τ is the single knob the ZMC tunes per band or incumbent class.

Recommended policy: **OR for incumbent-protection decisions, and location-weighted fusion with a per-band τ for secondary-to-secondary scheduling.**

### Spatial maps

`indop_ma/spatial_map.py` interpolates the seven simultaneous per-node estimates onto DST proxels using GP regression or IDW. The GP's σ map shows where a few extra sensors would reduce map error the most. For the comparison with OpenZMS Figure 7, export the DST proxel occupancy for the same session window and run:

```bash
python scripts/run_spatial_map.py --dst-csv dst_export.csv --npz-dir results/npz
```

## Form 3: cooperative cyclostationary detection

Each agent shares its SCF α-profile, which is a few hundred bytes. Averaging profiles suppresses noise that is uncorrelated between nodes, while the LTE cyclic peaks at 1 kHz and 2 kHz stay in place. In the synthetic study (`docs/figures/fig14_cooperative_scf.png`), averaging across seven nodes lowers the SNR needed for reliable detection by several dB at a fixed 5 % false-alarm rate. That turns energy the nodes see but cannot attribute into confirmed LTE.

## Integration with OpenZMS

```json
{
  "monitor_id": "indop-guesthouse",
  "kind": "occupancy",
  "freq_start": 3594500000, "freq_end": 3595500000,
  "occupancy": 0.224, "resolution_ms": 100,
  "method": "INDOP anchor-GMM + HMM (calibration-free)",
  "detail": { "mu_noise_dbx": -117.17, "threshold_dbx": -117.00, "...": "..." },
  "indop_ma": {
    "consensus_mode": "offset",
    "outlier": true,
    "threshold_dbx_consensus": -116.94,
    "lte_confirmed_cooperatively": true
  }
}
```

The field names are this project's illustrative schema (`indop_ma/zeal.py`). Map them onto the deployment's ZeAL client.
