"""Cooperative cyclostationary detection (INDOP-MA form 3).

Each agent computes the SCF alpha-profile of its own IQ locally and shares only
the profile (a few KB). Averaging profiles across agents suppresses noise that
is uncorrelated between nodes while the LTE cyclic features (alpha = 1 kHz,
2 kHz), present at every node that hears the same transmitter, add coherently
in the averaged statistic. A weak signal that no single node can confirm can
then be confirmed jointly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from indop.cyclo import LTE_ALPHAS_HZ, AlphaProfile, average_profiles


@dataclass
class CycloDecision:
    detected: bool
    peaks_db: dict[float, float]
    threshold_db: float


def detect_lte(profile: AlphaProfile, threshold_db: float = 3.0,
               alphas: tuple[float, ...] = LTE_ALPHAS_HZ, require: int = 1) -> CycloDecision:
    """Declare LTE present if at least ``require`` LTE cyclic peaks exceed the threshold."""
    peaks = {a: profile.peak_db(a) for a in alphas}
    hits = sum(v >= threshold_db for v in peaks.values())
    return CycloDecision(detected=hits >= require, peaks_db=peaks, threshold_db=threshold_db)


def cooperative_detect(profiles: list[AlphaProfile], threshold_db: float = 3.0,
                       require: int = 1) -> CycloDecision:
    return detect_lte(average_profiles(profiles), threshold_db, require=require)


def peak_snr_null(profile: AlphaProfile, exclude: tuple[float, ...] = LTE_ALPHAS_HZ,
                  guard_hz: float = 300.0) -> np.ndarray:
    """Profile values away from known cyclic frequencies (for setting false-alarm thresholds)."""
    m = np.ones_like(profile.alphas, bool)
    for a in exclude:
        for h in range(1, 6):
            m &= np.abs(profile.alphas - h * a) > guard_hz
    return profile.profile_db[m]
