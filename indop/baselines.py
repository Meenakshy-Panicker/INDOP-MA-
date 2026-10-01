"""Baselines compared against INDOP (paper Table 2)."""

from __future__ import annotations

import numpy as np


def dyspan_percentile_threshold(power: np.ndarray, pct: float = 5.0,
                                margin_db: float = 3.0) -> float:
    """Raouf et al. (DySPAN '26): 5th percentile of the power distribution + 3 dB."""
    return float(np.percentile(power, pct) + margin_db)


def full_capture_gmm_3sigma(power: np.ndarray) -> float:
    """Standard GMM on the whole capture, threshold mu_noise + 3 sigma_noise.

    Suffers from noise-floor drift splitting the noise distribution.
    """
    from .anchor_gmm import fit_anchor_gmm
    fit = fit_anchor_gmm(power, anchor_chunks=None)
    return float(fit.means[0] + 3 * fit.stds[0])


def fixed_threshold(power: np.ndarray, threshold_dbx: float) -> float:
    """OpenZMS DST style fixed, manually set threshold. Returns occupancy."""
    return float(np.mean(np.asarray(power) > threshold_dbx))
