"""Two-state HMM temporal refinement (paper Sec. 3.4).

Hidden state X_n in {0 = idle, 1 = occupied}.
Emissions: idle  -> N(mu_noise, sigma_noise^2)
           busy  -> sum_k w_k N(mu_k, sigma_k^2) over the signal components.
Transitions are estimated from anchor-window hard labels 1[P > T].
The forward algorithm gives the filtered posterior P(X_n = 1 | P[0..n]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .anchor_gmm import GMMFit


@dataclass
class HMMResult:
    posterior: np.ndarray
    transition: np.ndarray

    @property
    def soft_occupancy(self) -> float:
        return float(np.mean(self.posterior))

    @property
    def hard_occupancy(self) -> float:
        return float(np.mean(self.posterior > 0.5))


def estimate_transition(labels: np.ndarray, prior: float = 1.0) -> np.ndarray:
    """2x2 transition matrix from a 0/1 label sequence with Laplace smoothing."""
    labels = np.asarray(labels, dtype=int)
    counts = np.full((2, 2), prior)
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def emission_loglik(power: np.ndarray, fit: GMMFit) -> np.ndarray:
    """Return (n, 2) log-likelihoods for idle / occupied."""
    p = np.asarray(power, dtype=float)
    idle = norm.logpdf(p, fit.means[0], max(fit.stds[0], 1e-6))
    if fit.k == 1:
        # No explicit signal cluster: a broad component above threshold.
        busy = norm.logpdf(p, fit.threshold + 3 * fit.stds[0], 10 * fit.stds[0])
    else:
        w = fit.weights[1:] / fit.weights[1:].sum()
        comps = [np.log(wk) + norm.logpdf(p, m, max(s, 1e-6))
                 for wk, m, s in zip(w, fit.means[1:], fit.stds[1:])]
        busy = np.logaddexp.reduce(np.vstack(comps), axis=0)
    return np.column_stack([idle, busy])


def forward_filter(power: np.ndarray, fit: GMMFit,
                   transition: np.ndarray | None = None) -> HMMResult:
    """Run the scaled forward algorithm and return filtered posteriors."""
    p = np.asarray(power, dtype=float)
    if transition is None:
        anchor = p[:fit.anchor_chunks]
        transition = estimate_transition((anchor > fit.threshold).astype(int))
    ll = emission_loglik(p, fit)
    lik = np.exp(ll - ll.max(axis=1, keepdims=True))
    pi = np.array([fit.weights[0], 1.0 - fit.weights[0]])
    post = np.empty(len(p))
    alpha = pi * lik[0]
    alpha /= alpha.sum()
    post[0] = alpha[1]
    for n in range(1, len(p)):
        alpha = (alpha @ transition) * lik[n]
        s = alpha.sum()
        alpha = alpha / s if s > 0 else np.array([0.5, 0.5])
        post[n] = alpha[1]
    return HMMResult(posterior=post, transition=transition)
