# Methodology

## Notation

| Symbol | Meaning |
|---|---|
| $x[\cdot]$ | complex baseband IQ samples, 1 MS/s |
| $N = 10^5$ | samples per 100 ms chunk |
| $P[n]$ | power of chunk $n$ in uncalibrated dBX/Hz |
| $W$ | anchor window length in chunks (default 720 = 72 s) |
| $\mu_k, \sigma_k, w_k$ | mean, std and weight of GMM component $k$, sorted by mean |
| $T$ | detection threshold (dBX) |
| $i \in \{1..7\}$ | agent (POWDER node) index |

## Single-node INDOP

These are the equations from paper Sec. 3, as implemented in `indop/`.

**Power timeline (Eq. 1).**

$$P[n] = 10\log_{10}\Big(\frac{1}{N}\sum_{k=0}^{N-1}|x[nN+k]|^2\Big) - 10\log_{10}(\mathrm{BW})$$

**Anchor GMM (Eq. 2).** Fit a GMM to $\{P[n]\}_{n<W}$ and choose the number of components by BIC:

$$K^* = \arg\min_K\; -2\log\hat L + K_p\log W$$

The lowest-mean component is the noise cluster.

**Threshold (Eq. 3).** $T = (\mu_\text{noise}+\mu_\text{signal1})/2$, or $T=\mu_\text{noise}+3\sigma_\text{noise}$ when $K^*=1$.

*Optional separation guard (this repo, off by default):* $T \leftarrow \max(T,\ \mu_\text{noise}+c\,\sigma_\text{noise})$. This is for nodes so far from the transmitter that the Signal-1 cluster overlaps the noise cluster. See the consensus experiment, panel B.

**Anchor-window selection.** Start at 300 chunks and grow in steps of 100 chunks (10 s). Stop when $|\Delta\mu_\text{noise}| > 0.05$ dB or $\sigma_\text{noise}$ grows by more than 50 %. Keep the last window before drift starts. The detector catches a drop in the noise floor, which is what the B210 does in paper Table 1. An upward step is absorbed as a new "signal" component and is not detected; this is a known limitation that `tests/test_indop.py` documents.

**HMM.** Two states, idle and occupied. The idle emission is $\mathcal N(\mu_\text{noise},\sigma^2_\text{noise})$. The occupied emission is the weighted mixture of the signal components. The transition matrix is estimated from the anchor labels $\mathbb 1[P[n]>T]$ with Laplace smoothing. The scaled forward filter gives $P(X_n=1\mid P[0..n])$. Soft occupancy is the mean posterior; hard occupancy thresholds the posterior at 0.5.

**Cyclostationary validation (Eq. 4).** The $\alpha$-profile $y(\alpha)=\max_f|\mathrm{SCF}(f,\alpha)|$ is computed with $N_\text{fft}=8192$ (122 Hz resolution). The cyclic features of interest (1 kHz and 2 kHz, i.e. the LTE subframe and slot) are far below the bandwidth. The implementation therefore evaluates the cyclic spectrum of the instantaneous power envelope, averaged over $L$ segments. This exposes the same frame periodicity and is cheap enough to run on the NUC. The profile is reported in dB relative to its median.

**Physics ground truth (Eq. 5).** $T_\text{dBm} = kTB + \mathrm{NF}(f,g) + 3$ dB. Converting to the dBX domain gives $T_\text{dBX} = T_\text{dBm/Hz} - K$. For serial 31722B7 at 3595 MHz and 60 dB ($K=-37.79$ dB, $\mathrm{NF}=16.19$ dB) this is **−116.995 dBX**, which matches the anchor-GMM threshold in paper Table 1 (−116.996 dBX). `tests/test_indop.py::test_physics_gt_matches_paper_threshold` checks this. Radios without calibration tables use $T_\text{self}=\mu_\text{noise}+3\sigma_\text{noise}$.

## INDOP-MA

### 1. Noise-floor consensus (`indop_ma/consensus.py`)

Agent $i$ broadcasts a `NoiseReport` $(\mu_i,\sigma_i,T_i,W_i,w_{0,i},\mu^\text{ref}_i)$. Here $\mu^\text{ref}_i$ is the noise-cluster mean re-fitted on the window $[W, 2W)$ just after the anchor.

**Feature.**

* `absolute`: $f_i=\mu_i$. This assumes identical hardware.
* `offset` (default): $f_i=\mu_i-\mu^\text{ref}_i$. The unit's gain and noise-figure offset appears in both terms, so it cancels.

**Outlier test.** A modified z-score over $\{f_i\}\cup\{f_j\}_{j\in\mathcal N(i)}$:

$$z_i=0.6745\,\frac{f_i-\operatorname{med}(f)}{\operatorname{MAD}(f)}$$

Agent $i$ is an outlier if $|z_i|>3.5$ and $|f_i-c_i|\ge 0.03$ dB.

**Consensus value.** $c_i$ is the median of the peers' features, weighted by precision $W_j w_{0,j}/\sigma_j^2$.

**Correction.** $T_i' = T_i + \beta\,(c_i-f_i)$ with blend $\beta\in(0,1]$ (default 1). Agents that are not outliers keep their own estimate.

**Leaderless variant.** `gossip_average` iterates $x_i\leftarrow x_i+\eta\cdot\operatorname{mean}_{j\in\mathcal N(i)}(x_j-x_i)$.

### 2. Joint occupancy fusion (`indop_ma/fusion.py`)

There is one decision per 100 ms chunk. Let $d_i[n]\in\{0,1\}$ be the HMM hard decision of agent $i$ and $p_i[n]$ its posterior.

| Rule | Occupied if | Intended use |
|---|---|---|
| OR | $\exists i: d_i[n]=1$ | incumbent protection |
| AND | $\forall i: d_i[n]=1$ | upper bound on reuse |
| k-of-n / majority | $\sum_i d_i[n]\ge k$ | robustness to one bad sensor |
| Weighted | $\sum_i \omega_i(c)\,p_i[n] \ge \tau$ | location-aware DSA scheduling |

The weights are $\omega_i(c)\propto (\lVert x_i-c\rVert+\epsilon)^{-2}$, i.e. inverse-distance weighting to the consumer location $c$.

**Grant metrics.** A grant is issued when the fused decision is "idle".

* False-grant rate $=P(\text{grant}\mid\text{occupied at }c)$
* Missed detection $=P(\text{grant}\mid\text{incumbent detectable at}\ge1\text{ sensor})$
* False-denial rate $=P(\text{deny}\mid\text{idle at }c)$
* Grant utilisation $=P(\text{grant}\mid\text{idle at }c)$

### 3. Spatial occupancy map (`indop_ma/spatial_map.py`)

Per-node occupancies are interpolated onto a regular grid of proxels, 20 m by default.

* **IDW** uses power 2.
* **GP regression** uses a constant × RBF + white-noise kernel with `normalize_y`. It also returns $\sigma$ per proxel, which shows where an extra sensor would help most.

### 4. Cooperative cyclostationary detection (`indop_ma/coop_cyclo.py`)

Each agent shares its $\alpha$-profile. The profiles are averaged in the linear domain. The statistic is $\max(y(1\,\text{kHz}), y(2\,\text{kHz}))$ in dB above the median. The threshold for each group size is set from noise-only trials at $P_\text{fa}=5\%$.

## Synthetic scenario (`sim/synthetic.py`)

The simulator exists only so INDOP-MA can be evaluated against known ground truth. Its parameters resemble the published Sagepoint fit (Table 4).

* **Noise floor:** −117.17 dBX, with jitter $\sigma=0.07$ dB per 100 ms chunk.
* **Hardware:** an inter-unit offset $\sigma_u$. The B210 drift starts at 70–95 s and reaches ±0.03–0.08 dB over the session. An optional anchor bias models an unlucky anchor window.
* **Transmitter:** two-state Markov on/off with mean burst length 2 s. The duty cycle follows a schedule, including a diurnal schedule with a 04–07 idle window. The per-chunk LTE load $u\sim$ Beta mixture.
* **Received SNR:** log-distance path loss (6 dB at 100 m, exponent 1.2) plus 1.5 dB shadowing.
* **Truth:** a chunk is occupied when the transmitter is on and $10\log_{10}(1+u\cdot\mathrm{SNR})\ge0.15$ dB. The paper's physics threshold sits about 0.17 dB above the noise mean, so the two definitions are close.

Every figure made from the simulator is stamped *SYNTHETIC* in its corner.
