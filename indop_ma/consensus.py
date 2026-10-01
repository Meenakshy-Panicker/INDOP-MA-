"""Noise-floor consensus (INDOP-MA communication form 1).

Each agent shares its anchor-GMM noise-cluster estimate with proximate peers.
A robust, precision-weighted consensus is computed and agents whose estimate
is an outlier relative to the majority are corrected.

Two feature spaces are supported:

* ``"absolute"`` - consensus on mu_noise (dBX). Only meaningful when peers run
  the same radio model at the same gain AND the unit-to-unit offset is small
  compared with the anchor error being corrected. The corrected threshold
  keeps the node's own detection margin: T_i' = T_i + (mu' - mu_i).
* ``"offset"`` - consensus on the *anchor offset* o_i = mu_i - mu_ref_i, where
  mu_ref_i is the node's own noise-cluster mean re-estimated on the window
  right after the anchor. Unit-to-unit gain and noise-figure differences
  cancel in o_i, so this works across heterogeneous radios. An outlier node
  shifts its threshold by the excess offset: T_i' = T_i + (o' - o_i).

A third candidate, consensus on the detection margin (T - mu) / sigma, was
evaluated and rejected: the margin grows with received SNR, so it flags the
healthy high-SNR nodes instead of the faulty one (see docs/INDOP_MA.md).

Whether consensus fixes the Guesthouse +5.1 pp gap is the empirical question
``scripts/run_consensus_experiment.py`` is set up to answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

MAD_SCALE = 0.6745  # makes MAD consistent with sigma for Gaussian data


@dataclass(frozen=True)
class NoiseReport:
    """What an agent broadcasts to its peers (a few dozen bytes)."""
    node: str
    mu_noise: float          # dBX, anchor-window noise mean
    sigma_noise: float       # dB
    threshold: float         # dBX
    anchor_chunks: int = 720
    weight_noise: float = 1.0      # GMM weight of the noise component
    mu_ref: float | None = None    # noise mean re-estimated after the anchor (dBX)

    @property
    def anchor_offset(self) -> float:
        if self.mu_ref is None:
            raise ValueError(f"{self.node}: 'offset' consensus needs mu_ref")
        return self.mu_noise - self.mu_ref

    @property
    def precision(self) -> float:
        """Inverse variance of the mu_noise estimate: n_noise / sigma^2."""
        n_eff = max(self.anchor_chunks * self.weight_noise, 1.0)
        return n_eff / max(self.sigma_noise, 1e-9) ** 2

    @classmethod
    def from_fit(cls, node: str, fit, mu_ref: float | None = None) -> "NoiseReport":
        return cls(node=node, mu_noise=fit.mu_noise, sigma_noise=fit.sigma_noise,
                   threshold=fit.threshold, anchor_chunks=fit.anchor_chunks,
                   weight_noise=float(fit.weights[0]), mu_ref=mu_ref)


@dataclass
class ConsensusOutcome:
    corrected: list[NoiseReport]
    outliers: list[str]
    consensus_value: dict[str, float]
    robust_z: dict[str, float] = field(default_factory=dict)


def robust_z(values: np.ndarray) -> np.ndarray:
    """Modified z-score (Iglewicz & Hoaglin): 0.6745 (x - median) / MAD."""
    v = np.asarray(values, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad < 1e-12:
        return np.zeros_like(v)
    return MAD_SCALE * (v - med) / mad


def weighted_consensus(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: robust to a minority of bad reports."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cw, 0.5 * cw[-1])])


def noise_floor_consensus(reports: list[NoiseReport], mode: str = "offset",
                          z_thresh: float = 3.5, min_abs_dev: float = 0.03,
                          peers: list[list[int]] | None = None,
                          blend: float = 1.0) -> ConsensusOutcome:
    """Detect and correct outlier noise-floor estimates by majority.

    Parameters
    ----------
    reports:     one report per agent.
    mode:        ``"absolute"`` or ``"offset"`` (see module docstring).
    z_thresh:    modified z-score above which an agent is an outlier.
    min_abs_dev: an agent is only an outlier if it also deviates from the
                 consensus by at least this many dB, so a very tight
                 majority does not flag harmless jitter.
    peers:       optional neighbourhood per agent (indices); default = all others.
    blend:       1.0 moves the outlier fully onto the consensus;
                 values in (0, 1) pull it part of the way.
    """
    if mode not in ("absolute", "offset"):
        raise ValueError("mode must be 'absolute' or 'offset'")
    n = len(reports)
    feat = np.array([r.mu_noise if mode == "absolute" else r.anchor_offset for r in reports])
    prec = np.array([r.precision for r in reports])
    corrected, outliers, zs, cons = [], [], {}, {}
    for i, r in enumerate(reports):
        group = list(peers[i]) if peers is not None else [j for j in range(n) if j != i]
        z = robust_z(feat[[i] + group])[0]
        c = weighted_consensus(feat[group], prec[group]) if group else feat[i]
        zs[r.node], cons[r.node] = float(z), c
        is_out = len(group) >= 2 and abs(z) > z_thresh and abs(feat[i] - c) >= min_abs_dev
        if not is_out:
            corrected.append(r)
            continue
        outliers.append(r.node)
        shift = blend * (c - feat[i])  # dB the noise estimate moves
        corrected.append(replace(r, mu_noise=r.mu_noise + shift, threshold=r.threshold + shift))
    return ConsensusOutcome(corrected=corrected, outliers=outliers,
                            consensus_value=cons, robust_z=zs)


def gossip_average(values: np.ndarray, peers: list[list[int]], rounds: int = 20,
                   step: float = 0.5) -> np.ndarray:
    """Fully distributed average consensus (no coordinator).

    x_i <- x_i + step * mean_{j in N(i)} (x_j - x_i). Converges to the network
    average on a connected peer graph; used when no node may act as leader.
    """
    x = np.asarray(values, float).copy()
    for _ in range(rounds):
        x = x + step * np.array([np.mean(x[p] - x[i]) if p else 0.0
                                 for i, p in enumerate(peers)])
    return x
