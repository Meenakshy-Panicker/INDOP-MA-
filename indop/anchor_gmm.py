"""Anchor-window GMM (paper Sec. 3.3, Eqs. 2-3).

A Gaussian Mixture Model is fitted on the first W chunks of the power
timeline (the *anchor window*). The number of components K is chosen by
BIC; the lowest-mean component is the noise cluster. The detection
threshold is the midpoint between the noise cluster and the first signal
cluster, or mu_noise + 3 sigma_noise when only one component is found.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GMMFit:
    means: np.ndarray            # sorted ascending (dBX)
    stds: np.ndarray             # dB
    weights: np.ndarray
    bic: dict[int, float]
    threshold: float             # dBX
    anchor_chunks: int

    @property
    def k(self) -> int:
        return len(self.means)

    @property
    def mu_noise(self) -> float:
        return float(self.means[0])

    @property
    def sigma_noise(self) -> float:
        return float(self.stds[0])

    @property
    def margin(self) -> float:
        """Threshold offset above the noise mean (dB). Hardware-independent."""
        return float(self.threshold - self.means[0])

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("means", "stds", "weights"):
            d[key] = np.asarray(d[key]).tolist()
        return d


def threshold_from_components(means: np.ndarray, stds: np.ndarray,
                              min_margin_sigma: float | None = None) -> float:
    """Paper Eq. 3: T = (mu_noise + mu_signal1) / 2, or mu + 3 sigma if K = 1.

    ``min_margin_sigma`` is an optional *separation guard* (not in the paper):
    when the first signal cluster overlaps the noise cluster (weak, distant
    transmitter), the midpoint can fall inside the noise distribution. The
    guard enforces T >= mu_noise + c * sigma_noise.
    """
    if len(means) == 1:
        return float(means[0] + 3.0 * stds[0])
    t = float(0.5 * (means[0] + means[1]))
    if min_margin_sigma is not None:
        t = max(t, float(means[0] + min_margin_sigma * stds[0]))
    return t


def fit_anchor_gmm(power: np.ndarray, anchor_chunks: int | None = None,
                   k_max: int = 5, seed: int = 0,
                   min_margin_sigma: float | None = None) -> GMMFit:
    """Fit a BIC-selected GMM on the anchor window of a power timeline."""
    x = np.asarray(power, dtype=float)
    if anchor_chunks is not None:
        x = x[:anchor_chunks]
    x = x.reshape(-1, 1)
    best, bics = None, {}
    for k in range(1, k_max + 1):
        if len(x) < 5 * k:
            break
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=seed, n_init=2, reg_covar=1e-6).fit(x)
        bics[k] = float(gm.bic(x))
        if best is None or bics[k] < bics[best.n_components]:
            best = gm
    order = np.argsort(best.means_.ravel())
    means = best.means_.ravel()[order]
    stds = np.sqrt(best.covariances_.reshape(-1))[order]
    weights = best.weights_[order]
    return GMMFit(means=means, stds=stds, weights=weights, bic=bics,
                  threshold=threshold_from_components(means, stds, min_margin_sigma),
                  anchor_chunks=len(x))


@dataclass
class AnchorSelection:
    anchor_chunks: int
    history: list[dict] = field(default_factory=list)


def select_anchor_window(power: np.ndarray, start_chunks: int = 300,
                         step_chunks: int = 100, max_chunks: int = 1800,
                         mean_tol_db: float = 0.05,
                         std_tol_frac: float = 0.5) -> AnchorSelection:
    """Drift-detection anchor-window selection (paper Sec. 3.3).

    The window is extended by ``step_chunks`` (10 s) until the noise-cluster
    mean moves by more than ``mean_tol_db`` or its std grows by more than
    ``std_tol_frac`` relative to the initial estimate; the last window before
    drift onset is returned.
    """
    power = np.asarray(power, dtype=float)
    max_chunks = min(max_chunks, len(power))
    ref = fit_anchor_gmm(power, min(start_chunks, max_chunks))
    sel = AnchorSelection(anchor_chunks=ref.anchor_chunks,
                          history=[{"chunks": ref.anchor_chunks,
                                    "mu_noise": ref.mu_noise,
                                    "sigma_noise": ref.sigma_noise}])
    w = start_chunks + step_chunks
    while w <= max_chunks:
        fit = fit_anchor_gmm(power, w)
        sel.history.append({"chunks": w, "mu_noise": fit.mu_noise,
                            "sigma_noise": fit.sigma_noise})
        drift = abs(fit.mu_noise - ref.mu_noise) > mean_tol_db
        spread = fit.sigma_noise > (1.0 + std_tol_frac) * ref.sigma_noise
        if drift or spread:
            break
        sel.anchor_chunks = w
        w += step_chunks
    return sel


def occupancy(power: np.ndarray, threshold: float) -> float:
    """Hard occupancy: fraction of chunks above threshold."""
    return float(np.mean(np.asarray(power) > threshold))
